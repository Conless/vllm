# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import contextlib
from typing import Callable, Optional

import torch

from vllm.config import CompilationConfig
from vllm.nanoinfer.engine import NanoInferEngine
from vllm.nanoinfer.example import NanoFlowScheduler, NanoFlowSchedulerConfig
from vllm.nanoinfer.interface import (
    ExecutionContext,
    InputInfo,
    OperatorHandle,
    SplitConfig,
)
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
        self.hook: Optional[
            Callable[[tuple[OperatorHandle]], contextlib.AbstractContextManager[None]]
        ] = None
        tag_graph(
            self.graph_module,
            {
                # "vllm.unified_attention": "memory",
                # "vllm.unified_attention_with_output": "memory",
                "vllm.all_reduce": "network",
                "vllm.moe_forward_dispatch": "network",
                "vllm.moe_forward_combine": "network",
                "vllm.moe_forward_combine_with_shared": "network"
            },
        )
        self.scheduler = NanoFlowScheduler(
            NanoFlowSchedulerConfig(
                min_nano_split_tokens=compilation_config.min_nano_split_tokens,
                max_num_nano_batches=compilation_config.max_num_nano_batches,
            )
        )
        self.engine = NanoInferEngine(self.graph_module)

    def prepare(
        self,
        batch_size: int,
        num_tokens: list[int],
    ) -> SplitConfig:
        """Prepare split configuration and scheduler."""
        self.cached_config = self.scheduler.get_split_config(
            InputInfo(batch_size, num_tokens, sum(num_tokens))
        )
        return self.cached_config

    def disable_nano_split(self):
        self.cached_config = None

    def set_hooks(
        self,
        op_hook: Callable[
            [tuple[OperatorHandle]], contextlib.AbstractContextManager[None]
        ],
    ):
        """Set user-defined hook."""
        self.hook = op_hook

    def get_callable(self) -> Callable:
        """Get callable that executes modules with software-defined
        scheduling."""

        def _forward(*args, **kwargs):
            if self.cached_config is None or self.cached_config.num_nano_batches == 1:
                return self.graph_module(*args, **kwargs)

            return asyncio.run(self._forward_async(args, kwargs))

        return _forward

    async def _forward_async(self, args: tuple, kwargs: dict):
        """Async forward execution with engine and scheduler."""
        assert self.cached_config is not None
        num_nano_batches = self.cached_config.num_nano_batches
        op_queue = {i: asyncio.Queue() for i in range(num_nano_batches)}
        execute_queue = asyncio.Queue()
        context = ExecutionContext(self.cached_config, op_queue, execute_queue)

        results_dict, _ = await asyncio.gather(
            self.engine.execute(
                args, kwargs, op_queue, execute_queue, self.cached_config, self.hook
            ),
            self.scheduler.schedule(context),
        )
        assert all(
            isinstance(e, type(results_dict[0])) for e in results_dict.values()
        ), f"Results have different types: {results_dict}"
        if isinstance(results_dict[0], torch.Tensor):
            return torch.cat(
                [results_dict[idx] for idx in range(num_nano_batches)], dim=0
            )
        elif isinstance(results_dict[0], tuple):
            num_elements = len(results_dict[0])
            assert all(len(r) == num_elements for r in results_dict.values()), (
                f"Results have different number of elements: {results_dict}"
            )
            concatenated = []
            for i in range(num_elements):
                elements = [results_dict[idx][i] for idx in range(num_nano_batches)]
                assert all(isinstance(e, type(elements[0])) for e in elements), (
                    f"Elements have different types: {elements}"
                )
                concatenated.append(torch.cat(elements, dim=0))
            return tuple(concatenated)
        else:
            return results_dict[0]


_manager = None


def get_callable(
    graph_module: torch.fx.GraphModule,
    compilation_config: CompilationConfig,
    local_cache_dir: Optional[str] = None,
) -> Callable:
    global _manager
    if _manager is None:
        _manager = NanoInferManager(graph_module, compilation_config, local_cache_dir)
    return _manager.get_callable()


def prepare_nano_split(
    batch_size: int,
    num_tokens: list[int],
) -> SplitConfig:
    global _manager
    if _manager is None:
        raise ValueError("Manager not initialized")
    return _manager.prepare(batch_size, num_tokens)


def disable_nano_split():
    global _manager
    if _manager is None:
        raise ValueError("Manager not initialized")
    _manager.disable_nano_split()


def set_op_hook(op_hook: Callable):
    global _manager
    if _manager is None:
        raise ValueError("Manager not initialized")
    _manager.set_hooks(op_hook)
