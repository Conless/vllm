# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import contextlib
from typing import Any, Callable

import torch

from vllm.nanoinfer.engine import NanoInferEngine
from vllm.nanoinfer.example.nanoflow import (
    NanoFlowScheduler,
    NanoFlowSchedulerConfig,
)
from vllm.nanoinfer.interface import (
    ExecutionContext,
    InputInfo,
    OpSchedulerBase,
    OperatorHandle,
    SplitConfig,
)
from vllm.nanoinfer.utils import compile_subgraphs
from vllm.nanoinfer.config import NanoInferConfig


class NanoInferManager:
    """
    NanoInfer integration manager.

    Extracts modules from FX graph and executes them with software-defined
    scheduling.
    """

    def __init__(self):
        self.initialized = False
        self.config: NanoInferConfig | None = None
        self.graph_module: torch.fx.GraphModule | None = None
        self.cached_config: SplitConfig | None = None
        self.hook: (
            Callable[
                [tuple[OperatorHandle]], contextlib.AbstractContextManager[None]
            ]
            | None
        ) = None
        self.scheduler: OpSchedulerBase | None = None
        self.engine: NanoInferEngine | None = None

    def initialize(
        self,
        graph_module: torch.fx.GraphModule,
        config: NanoInferConfig,
        example_inputs: list[Any],
    ) -> None:
        self.initialized = True
        self.config = config
        self.graph_module = compile_subgraphs(
            graph_module,
            self.config,
            example_inputs,
        )
        self.engine = NanoInferEngine(
            self.graph_module,
            self.config,
        )
        if isinstance(self.config.scheduler_config, NanoFlowSchedulerConfig):
            self.scheduler = NanoFlowScheduler(self.config.scheduler_config)
        else:
            raise ValueError(
                f"Unsupported scheduler config: {self.config.scheduler_config}"
            )

    def prepare(
        self,
        batch_size: int,
        num_tokens: list[int],
        is_dryrun: bool = False,
        use_cudagraph: bool = False,
    ) -> SplitConfig:
        """Prepare split configuration and scheduler."""
        if not self.initialized or is_dryrun:
            self.cached_config = SplitConfig(
                num_nano_batches=1,
                batch_sizes=[batch_size],
                batch_indices=[0, batch_size],
                num_tokens=[sum(num_tokens)],
                num_tokens_padded=[sum(num_tokens)],
                split_indices=[0, sum(num_tokens)],
                is_dryrun=is_dryrun,
                use_cudagraph=use_cudagraph,
            )
        else:
            assert self.scheduler is not None
            self.cached_config = self.scheduler.get_split_config(
                InputInfo(batch_size, num_tokens, sum(num_tokens)),
                use_cudagraph=use_cudagraph,
            )
        return self.cached_config

    def override_split_config(self, split_config: SplitConfig):
        self.cached_config = split_config

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
        assert self.initialized

        def _forward(*args, **kwargs) -> Any:
            assert self.cached_config is not None
            result = asyncio.run(self._forward_async(args, kwargs))
            self.cached_config = None
            return result

        return _forward

    async def _forward_async(self, args: tuple, kwargs: dict):
        """Async forward execution with engine and scheduler."""
        assert (
            self.initialized
            and self.engine is not None
            and self.scheduler is not None
            and self.cached_config is not None
        )
        num_nano_batches = self.cached_config.num_nano_batches
        op_queue = {i: asyncio.Queue() for i in range(num_nano_batches)}
        execute_queue = asyncio.Queue()
        context = ExecutionContext(self.cached_config, op_queue, execute_queue)

        if self.cached_config.is_dryrun:
            results_dict, events = self.engine.dryrun(
                args,
                kwargs,
                self.cached_config,
                self.hook,
            )
        else:
            (results_dict, events), _ = await asyncio.gather(
                self.engine.execute(
                    args,
                    kwargs,
                    op_queue,
                    execute_queue,
                    self.cached_config,
                    self.hook,
                ),
                self.scheduler.schedule(context),
            )
        for event in events:
            event.wait()
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
                elements = [
                    results_dict[idx][i] for idx in range(num_nano_batches)
                ]
                assert all(
                    isinstance(e, type(elements[0])) for e in elements
                ), f"Elements have different types: {elements}"
                concatenated.append(torch.cat(elements, dim=0))
            return tuple(concatenated)
        else:
            return results_dict[0]


_manager = NanoInferManager()


def get_callable(
    graph_module: torch.fx.GraphModule,
    config: NanoInferConfig,
    example_inputs: list[Any],
) -> Callable:
    global _manager
    assert not _manager.initialized
    _manager.initialize(graph_module, config, example_inputs)
    return _manager.get_callable()


def prepare_nano_split(
    batch_size: int,
    num_tokens: list[int],
    is_dryrun: bool = False,
    use_cudagraph: bool = False,
) -> SplitConfig:
    global _manager
    return _manager.prepare(batch_size, num_tokens, is_dryrun, use_cudagraph)


def override_split_config(split_config: SplitConfig) -> None:
    global _manager
    _manager.override_split_config(split_config)


def set_op_hook(op_hook: Callable):
    global _manager
    _manager.set_hooks(op_hook)
