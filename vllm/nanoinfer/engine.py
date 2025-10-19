# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import contextlib
from collections import deque
from typing import Any, Callable, Optional

import torch

from vllm.nanoinfer.config import NanoInferConfig
from vllm.nanoinfer.context import NanoInferContext, set_forward_context
from vllm.nanoinfer.interface import OperatorHandle, SplitConfig


class NanoInferEngine:
    """Engine that executes FX graph with software-defined operator scheduling.

    The engine maintains two concurrent tasks:
    - Producer: Pushes ready module operators to op_queue
    - Consumer: Processes execution requests from execute_queue

    Non-module operations (call_function, call_method, output) are executed
    automatically in the background without scheduler involvement.
    """

    def __init__(
        self, graph_module: torch.fx.GraphModule, config: NanoInferConfig
    ):
        self.graph_module = graph_module
        self.config = config
        self.input_buffers = [{}] * config.scheduler_config.max_num_nano_batches

        self.module_name_to_node: dict[str, torch.fx.Node] = {}
        for node in self.graph_module.graph.nodes:
            if node.op == "call_module":
                assert isinstance(node.target, str)
                self.module_name_to_node[node.target] = node

        self.placeholder_nodes: list[torch.fx.Node] = [
            n for n in self.graph_module.graph.nodes if n.op == "placeholder"
        ]
        self.placeholder_node_to_idx: dict[torch.fx.Node, int] = {
            node: idx for idx, node in enumerate(self.placeholder_nodes)
        }
        self.get_attr_cache: dict[str, Any] = {}
        for node in self.graph_module.graph.nodes:
            if node.op == "get_attr":
                assert isinstance(node.target, str)
                self.get_attr_cache[node.target] = getattr(
                    self.graph_module, node.target
                )

    def dryrun(
        self,
        args: tuple,
        kwargs: dict,
        split_config: SplitConfig,
        hook: Optional[
            Callable[
                [tuple[OperatorHandle]], contextlib.AbstractContextManager[None]
            ]
        ] = None,
    ) -> tuple[dict[int, Any], list[torch.cuda.Event]]:
        assert split_config.num_nano_batches == 1 and split_config.is_dryrun
        use_fake_nano_batches = split_config.use_cudagraph
        num_nano_batches = (
            self.config.scheduler_config.max_num_nano_batches
            if use_fake_nano_batches
            else 1
        )
        env: dict[int, dict[torch.fx.Node, Any]] = {
            i: {} for i in range(num_nano_batches)
        }
        results: dict[int, Any] = {}
        num_tokens = split_config.num_tokens_padded[0]
        last_events = [torch.cuda.Event() for _ in range(num_nano_batches)]
        node_queue: dict[int, deque[torch.fx.Node]] = {
            i: deque(self.graph_module.graph.nodes)
            for i in range(num_nano_batches)
        }

        for batch_idx in range(num_nano_batches):
            while node_queue[batch_idx]:
                node = node_queue[batch_idx][0]
                if node.op == "placeholder":
                    placeholder_idx = self.placeholder_node_to_idx[node]
                    example_value = node.meta.get("example_value", None)
                    if isinstance(example_value, torch.Tensor):
                        if isinstance(example_value.shape[0], torch.SymInt):
                            if split_config.use_cudagraph:
                                assert self.input_buffers is not None
                                if (
                                    self.input_buffers[batch_idx].get(
                                        placeholder_idx
                                    )
                                    is None
                                ):
                                    self.input_buffers[batch_idx][
                                        placeholder_idx
                                    ] = torch.zeros(
                                        (16384,) + example_value.shape[1:],
                                        dtype=example_value.dtype,
                                        device=example_value.device,
                                    )
                                self.input_buffers[batch_idx][placeholder_idx][
                                    :num_tokens
                                ].copy_(args[placeholder_idx][:num_tokens])
                                env[batch_idx][node] = self.input_buffers[
                                    batch_idx
                                ][placeholder_idx][:num_tokens]
                            else:
                                env[batch_idx][node] = args[placeholder_idx][
                                    :num_tokens
                                ]
                        else:
                            env[batch_idx][node] = args[placeholder_idx]
                    elif isinstance(example_value, torch.SymInt):
                        assert args[placeholder_idx] == num_tokens
                        env[batch_idx][node] = num_tokens
                    else:
                        raise ValueError(
                            f"Invalid example value type: {type(example_value)}"
                        )
                elif node.op == "call_module":
                    assert isinstance(node.target, str)
                    module = getattr(self.graph_module, node.target)
                    op_handle = OperatorHandle(
                        module_name=node.target,
                        nano_batch_idx=batch_idx,
                        debug_info={"tag": getattr(module, "tag", "")},
                    )
                    node_args = [
                        env[batch_idx][arg]
                        if isinstance(arg, torch.fx.Node)
                        else arg
                        for arg in node.args
                    ]
                    node_kwargs = {
                        k: env[batch_idx][v]
                        if isinstance(v, torch.fx.Node)
                        else v
                        for k, v in node.kwargs.items()
                    }
                    with (
                        (
                            hook((op_handle,))
                            if hook is not None
                            else contextlib.nullcontext()
                        ),
                        torch.cuda.nvtx.range(f"op_{node.target}_{batch_idx}"),
                        set_forward_context(
                            NanoInferContext(
                                nano_batch_idx=(batch_idx,),
                                num_tokens_padded=(num_tokens,),
                                is_dryrun=split_config.is_dryrun,
                                use_cudagraph=split_config.use_cudagraph,
                            )
                        ),
                    ):
                        env[batch_idx][node] = module(*node_args, **node_kwargs)
                elif node.op == "output":
                    if isinstance(node.args[0], torch.fx.Node):
                        results[batch_idx] = env[batch_idx][node.args[0]]
                    elif isinstance(node.args[0], (tuple, list)):
                        results[batch_idx] = tuple(
                            env[batch_idx][arg]
                            if isinstance(arg, torch.fx.Node)
                            else arg
                            for arg in node.args[0]
                        )
                    else:
                        results[batch_idx] = node.args[0]
                else:
                    self._execute_non_module(
                        node, batch_idx, args, env, split_config
                    )
                node_queue[batch_idx].popleft()
            last_events[batch_idx].record()

        return {0: results[0]}, last_events

    async def execute(
        self,
        args: tuple,
        kwargs: dict,
        op_queue: dict[int, asyncio.Queue[OperatorHandle | None]],
        execute_queue: asyncio.Queue[
            tuple[
                tuple[OperatorHandle],
                Callable | None,
                asyncio.Event,
            ]
        ],
        split_config: SplitConfig,
        hook: Optional[
            Callable[
                [tuple[OperatorHandle]], contextlib.AbstractContextManager[None]
            ]
        ] = None,
    ) -> tuple[dict[int, Any], list[torch.cuda.Event]]:
        """Execute the model with simplified single-loop pattern.

        Args:
            args: Input arguments
            kwargs: Input keyword arguments
            op_queue: Queues to push ready operators (one per nano-batch)
            execute_queue: Queue to receive execution requests from scheduler
            split_config: Configuration for nano-batch splitting
            hook: Optional hook for operator execution

        Returns:
            Dictionary mapping nano-batch index to results
        """
        num_nano_batches = split_config.num_nano_batches

        env: dict[int, dict[torch.fx.Node, Any]] = {
            i: {} for i in range(num_nano_batches)
        }

        results: dict[int, Any] = {}
        last_events = [torch.cuda.Event() for _ in range(num_nano_batches)]
        node_queue: dict[int, deque[torch.fx.Node]] = {
            i: deque(self.graph_module.graph.nodes)
            for i in range(num_nano_batches)
        }

        for batch_idx in range(num_nano_batches):
            while node_queue[batch_idx]:
                node = node_queue[batch_idx][0]
                if node.op == "call_module" or node.op == "output":
                    break
                self._execute_non_module(
                    node, batch_idx, args, env, split_config
                )
                node_queue[batch_idx].popleft()
            last_events[batch_idx].record()

        pushed_operators: dict[int, list[OperatorHandle]] = {
            i: [] for i in range(num_nano_batches)
        }

        while any(node_queue.values()):
            for batch_idx in range(num_nano_batches):
                while node_queue[batch_idx]:
                    if op_queue[batch_idx].full():
                        break
                    node = node_queue[batch_idx][0]
                    if node.op == "call_module":
                        assert isinstance(node.target, str)
                        module = getattr(self.graph_module, node.target)
                        op_handle = OperatorHandle(
                            module_name=node.target,
                            nano_batch_idx=batch_idx,
                            debug_info={"tag": getattr(module, "tag", "")},
                        )
                        await op_queue[batch_idx].put(op_handle)
                        pushed_operators[batch_idx].append(op_handle)
                        node_queue[batch_idx].popleft()
                    else:
                        break

            item = await execute_queue.get()
            operators, func, done_event = item
            node_args = []
            node_kwargs = []
            for op in operators:
                batch_idx = op.nano_batch_idx
                node = self.module_name_to_node[op.module_name]
                last_events[batch_idx].wait()
                last_events[batch_idx] = torch.cuda.Event()
                node_args.append(
                    [
                        env[batch_idx][arg]
                        for arg in node.args
                        if isinstance(arg, torch.fx.Node)
                    ]
                )
                node_kwargs.append(
                    {
                        k: env[batch_idx][v]
                        if isinstance(v, torch.fx.Node)
                        else v
                        for k, v in node.kwargs.items()
                    }
                )
            node_args = tuple(node_args)
            node_kwargs = tuple(node_kwargs)
            exec_results = []
            if func is not None:
                with (
                    (
                        hook(operators)
                        if hook is not None
                        else contextlib.nullcontext()
                    ),
                    set_forward_context(
                        NanoInferContext(
                            nano_batch_idx=tuple(
                                op.nano_batch_idx for op in operators
                            ),
                            num_tokens_padded=tuple(
                                split_config.num_tokens_padded[
                                    op.nano_batch_idx
                                ]
                                for op in operators
                            ),
                            is_dryrun=split_config.is_dryrun,
                            use_cudagraph=split_config.use_cudagraph,
                        )
                    ),
                ):
                    exec_results = func(node_args, node_kwargs)
            else:
                for idx, op in enumerate(operators):
                    with (
                        (
                            hook((op,))
                            if hook is not None
                            else contextlib.nullcontext()
                        ),
                        torch.cuda.nvtx.range(
                            f"op_{op.module_name}_{op.nano_batch_idx}"
                        ),
                        set_forward_context(
                            NanoInferContext(
                                nano_batch_idx=(op.nano_batch_idx,),
                                num_tokens_padded=(
                                    split_config.num_tokens_padded[
                                        op.nano_batch_idx
                                    ],
                                ),
                                is_dryrun=split_config.is_dryrun,
                                use_cudagraph=split_config.use_cudagraph,
                            )
                        ),
                    ):
                        module = getattr(self.graph_module, op.module_name)
                        exec_results.append(
                            module(*node_args[idx], **node_kwargs[idx])
                        )

            for op, result in zip(operators, exec_results):
                batch_idx = op.nano_batch_idx
                node = self.module_name_to_node[op.module_name]
                last_events[batch_idx].record()
                env[batch_idx][node] = result
                assert op == pushed_operators[batch_idx][0]
                pushed_operators[batch_idx].pop(0)

            for batch_idx in range(num_nano_batches):
                if pushed_operators[batch_idx]:
                    continue
                while node_queue[batch_idx]:
                    node = node_queue[batch_idx][0]
                    if node.op == "call_module":
                        break
                    elif node.op == "output":
                        if isinstance(node.args[0], torch.fx.Node):
                            results[batch_idx] = env[batch_idx][node.args[0]]
                        elif isinstance(node.args[0], (tuple, list)):
                            results[batch_idx] = tuple(
                                env[batch_idx][arg]
                                if isinstance(arg, torch.fx.Node)
                                else arg
                                for arg in node.args[0]
                            )
                        else:
                            results[batch_idx] = node.args[0]
                        node_queue[batch_idx].popleft()
                        await op_queue[batch_idx].put(None)
                        break
                    else:
                        self._execute_non_module(
                            node,
                            batch_idx,
                            args,
                            env,
                            split_config,
                        )
                        node_queue[batch_idx].popleft()
            done_event.set()

        return results, last_events

    def _execute_non_module(
        self,
        node: torch.fx.Node,
        batch_idx: int,
        args: tuple,
        env: dict[int, dict[torch.fx.Node, Any]],
        split_config: SplitConfig,
    ) -> None:
        """Execute non-module operations (placeholder, get_attr, call_function,
        call_method)."""
        if node.op == "placeholder":
            placeholder_idx = self.placeholder_node_to_idx[node]

            example_value = node.meta.get("example_value", None)

            token_start = split_config.split_indices[batch_idx]
            token_end = split_config.split_indices[batch_idx + 1]
            num_tokens = split_config.num_tokens[batch_idx]
            num_tokens_padded = split_config.num_tokens_padded[batch_idx]
            total_num_tokens_padded = sum(split_config.num_tokens_padded)

            if isinstance(example_value, torch.Tensor):
                if isinstance(example_value.shape[0], torch.SymInt):
                    if split_config.use_cudagraph:
                        assert self.input_buffers is not None
                        assert (
                            self.input_buffers[batch_idx].get(placeholder_idx)
                            is not None
                        )
                        self.input_buffers[batch_idx][placeholder_idx][
                            :num_tokens
                        ].copy_(args[placeholder_idx][token_start:token_end])
                        env[batch_idx][node] = self.input_buffers[batch_idx][
                            placeholder_idx
                        ][:num_tokens_padded]
                    else:
                        env[batch_idx][node] = args[placeholder_idx][
                            token_start:token_end
                        ]
                else:
                    env[batch_idx][node] = args[placeholder_idx]
            elif isinstance(example_value, torch.SymInt):
                assert args[placeholder_idx] == total_num_tokens_padded
                env[batch_idx][node] = num_tokens_padded
            else:
                raise ValueError(
                    f"Invalid example value type: {type(example_value)}"
                )
        elif node.op == "call_function":
            target = node.target
            assert callable(target)
            node_args = [
                env[batch_idx][arg] if isinstance(arg, torch.fx.Node) else arg
                for arg in node.args
            ]
            node_kwargs = {
                k: env[batch_idx][v] if isinstance(v, torch.fx.Node) else v
                for k, v in node.kwargs.items()
            }
            env[batch_idx][node] = target(*node_args, **node_kwargs)
        elif node.op == "call_method":
            assert isinstance(node.target, str)
            self_obj = self.env[batch_idx][node.args[0]]  # type: ignore
            target = getattr(self_obj, node.target)
            assert callable(target)
            method_args = [
                env[batch_idx][arg] if isinstance(arg, torch.fx.Node) else arg
                for arg in node.args[1:]
            ]
            node_kwargs = {
                k: env[batch_idx][v] if isinstance(v, torch.fx.Node) else v
                for k, v in node.kwargs.items()
            }
            env[batch_idx][node] = target(*method_args, **node_kwargs)
        elif node.op == "get_attr":
            assert isinstance(node.target, str)
            env[batch_idx][node] = self.get_attr_cache[node.target]
        else:
            raise ValueError(f"Invalid node operation: {node.op}")
