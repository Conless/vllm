# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from functools import partial
from typing import Callable, Optional

import torch

from vllm.config import CompilationConfig
from vllm.nanoinfer.example import NanoFlowScheduler, NanoFlowSchedulerConfig
from vllm.nanoinfer.interface import InputInfo, OpInfo, SplitConfig
from vllm.nanoinfer.utils import tag_graph


class NanoInferManager:
    """
    NanoInfer integration manager.

    Extracts modules from FX graph and executes them with software-defined
    scheduling.
    """

    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        compilation_config: CompilationConfig,
        local_cache_dir: Optional[str] = None,
    ):
        self.graph_module = graph_module
        self.cached_config: Optional[SplitConfig] = None
        self.hook: Optional[Callable[
            [OpInfo], contextlib.AbstractContextManager[None]]] = None
        tag_graph(
            self.graph_module,
            {
                "vllm.unified_attention": "memory",
                "vllm.unified_attention_with_output": "memory",
                "vllm.all_reduce": "network",
            },
        )
        self.scheduler = NanoFlowScheduler(
            NanoFlowSchedulerConfig(
                min_nano_split_tokens=compilation_config.min_nano_split_tokens,
                max_num_nano_batches=compilation_config.max_num_nano_batches,
            ))

    def prepare(
        self,
        batch_size: int,
        num_tokens: list[int],
    ) -> SplitConfig:
        """Prepare split configuration and scheduler."""
        self.cached_config = self.scheduler.get_split_config(
            InputInfo(batch_size, num_tokens, sum(num_tokens)))
        return self.cached_config

    def set_hooks(self, op_hook: Callable):
        """Set user-defined hook."""
        self.hook = op_hook

    def get_callable(self) -> Callable:
        """Get callable that executes modules with software-defined
        scheduling."""

        def _forward(*args, **kwargs):
            if (self.cached_config is None
                    or self.cached_config.num_nano_batches == 1):
                # No splitting - just execute graph normally
                return self.graph_module(*args, **kwargs)

            num_nano_batches = self.cached_config.num_nano_batches
            assert self.scheduler is not None
            results = {}
            env = {idx: {} for idx in range(num_nano_batches)}
            op_infos = {idx: [] for idx in range(num_nano_batches)}
            op_mapping = {}

            for batch_idx in range(num_nano_batches):
                # Get batch slice indices for this nano-batch
                token_start = self.cached_config.split_indices[batch_idx]
                token_end = self.cached_config.split_indices[batch_idx + 1]
                num_tokens_in_batch = self.cached_config.num_tokens[batch_idx]
                total_num_tokens = sum(self.cached_config.num_tokens)
                placeholder_nodes = [
                    n for n in self.graph_module.graph.nodes
                    if n.op == "placeholder"
                ]

                for node in self.graph_module.graph.nodes:
                    if node.op == "placeholder":
                        placeholder_idx = placeholder_nodes.index(node)
                        example_value = getattr(node, "meta",
                                                {}).get("example_value", None)
                        if isinstance(example_value, torch.Tensor):
                            if isinstance(example_value.shape[0],
                                          torch.SymInt):
                                env[batch_idx][node] = args[placeholder_idx][
                                    token_start:token_end]
                            else:
                                assert all(
                                    isinstance(x, int)
                                    for x in example_value.shape)
                                env[batch_idx][node] = args[placeholder_idx]
                        elif isinstance(example_value, torch.SymInt):
                            assert args[placeholder_idx] == total_num_tokens
                            env[batch_idx][node] = num_tokens_in_batch
                        else:
                            raise ValueError("Invalid example value type: "
                                             f"{type(example_value)}")
                    elif node.op == "call_module":

                        def _call_module(node: torch.fx.Node, batch_idx: int):
                            assert isinstance(node.target, str)
                            module = getattr(self.graph_module, node.target)
                            node_args = [
                                env[batch_idx][arg] for arg in node.args
                                if isinstance(arg, torch.fx.Node)
                            ]
                            node_kwargs = {
                                k:
                                env[batch_idx][v] if isinstance(
                                    v, torch.fx.Node) else v
                                for k, v in node.kwargs.items()
                            }
                            env[batch_idx][node] = module(
                                *node_args, **node_kwargs)

                        assert isinstance(node.target, str)
                        op_info = OpInfo(
                            node.target,
                            getattr(
                                getattr(self.graph_module, node.target),
                                "tag",
                                "",
                            ),
                            batch_idx,
                        )
                        op_infos[batch_idx].append(op_info)
                        op_mapping[op_info] = partial(_call_module, node,
                                                      batch_idx)
                    elif node.op == "call_function":

                        def _call_function(node: torch.fx.Node,
                                           batch_idx: int):
                            assert isinstance(node.target, Callable)
                            node_args = [
                                env[batch_idx][arg] if isinstance(
                                    arg, torch.fx.Node) else arg
                                for arg in node.args
                            ]
                            node_kwargs = {
                                k:
                                env[batch_idx][v] if isinstance(
                                    v, torch.fx.Node) else v
                                for k, v in node.kwargs.items()
                            }
                            env[batch_idx][node] = node.target(
                                *node_args, **node_kwargs)

                        op_info = OpInfo("", getattr(node, "tag", ""),
                                         batch_idx)
                        op_infos[batch_idx].append(op_info)
                        op_mapping[op_info] = partial(_call_function, node,
                                                      batch_idx)
                    elif node.op == "call_method":

                        def _call_method(node: torch.fx.Node, batch_idx: int):
                            assert isinstance(node.target, str)
                            self_obj = env[batch_idx][node.args[0]]
                            method_args = [
                                env[batch_idx][arg] if isinstance(
                                    arg, torch.fx.Node) else arg
                                for arg in node.args[1:]
                            ]
                            node_kwargs = {
                                k:
                                env[batch_idx][v] if isinstance(
                                    v, torch.fx.Node) else v
                                for k, v in node.kwargs.items()
                            }
                            env[batch_idx][node] = getattr(
                                self_obj, node.target)(*method_args,
                                                       **node_kwargs)

                        op_info = OpInfo("", getattr(node, "tag", ""),
                                         batch_idx)
                        op_infos[batch_idx].append(op_info)
                        op_mapping[op_info] = partial(_call_method, node,
                                                      batch_idx)
                    elif node.op == "output":

                        def _output(node: torch.fx.Node, batch_idx: int):
                            if isinstance(node.args[0], torch.fx.Node):
                                output_val = env[batch_idx][node.args[0]]
                            elif isinstance(node.args[0], (tuple, list)):
                                output_val = tuple(
                                    env[batch_idx][arg] if isinstance(
                                        arg, torch.fx.Node) else arg
                                    for arg in node.args[0])
                            else:
                                output_val = node.args[0]
                            results[batch_idx] = output_val

                        op_info = OpInfo("", getattr(node, "tag", ""),
                                         batch_idx)
                        op_infos[batch_idx].append(op_info)
                        op_mapping[op_info] = partial(_output, node, batch_idx)

            def executor(op_info: OpInfo):
                if self.hook is not None:
                    with (
                            self.hook(op_info),
                            torch.cuda.nvtx.range(
                                f"op_{op_info.submod_name}_{op_info.tag}_{op_info.idx}"
                            ),
                    ):
                        op_mapping[op_info]()
                else:
                    op_mapping[op_info]()

            self.scheduler.schedule(self.cached_config, op_infos, executor)

            assert all(
                isinstance(e, type(results[0])) for e in
                results.values()), f"Results have different types: {results}"
            if isinstance(results[0], torch.Tensor):
                return torch.cat(
                    [results[idx] for idx in range(num_nano_batches)], dim=0)
            elif isinstance(results[0], tuple):
                num_elements = len(results[0])
                assert all(len(r) == num_elements for r in results.values()), (
                    f"Results have different number of elements: {results}")
                concatenated = []
                for i in range(num_elements):
                    elements = [
                        results[idx][i] for idx in range(num_nano_batches)
                    ]
                    assert all(
                        isinstance(e, type(elements[0])) for e in
                        elements), f"Elements have different types: {elements}"
                    concatenated.append(torch.cat(elements, dim=0))
                return tuple(concatenated)
            else:
                return results[0]

        return _forward


_manager = None


def get_callable(
    graph_module: torch.fx.GraphModule,
    compilation_config: CompilationConfig,
    local_cache_dir: Optional[str] = None,
) -> Callable:
    global _manager
    if _manager is None:
        _manager = NanoInferManager(graph_module, compilation_config,
                                    local_cache_dir)
    return _manager.get_callable()


def prepare_nano_split(
    batch_size: int,
    num_tokens: list[int],
) -> SplitConfig:
    global _manager
    if _manager is None:
        raise ValueError("Manager not initialized")
    return _manager.prepare(batch_size, num_tokens)


def set_op_hook(op_hook: Callable):
    global _manager
    if _manager is None:
        raise ValueError("Manager not initialized")
    _manager.set_hooks(op_hook)
