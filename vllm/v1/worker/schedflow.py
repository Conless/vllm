# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Optional

import numpy as np
import torch

from schedflow.config import SchedFlowConfig
from schedflow.example.vllm.flux import FluxScheduler, FluxSchedulerConfig
from schedflow.example.vllm.tokenweave import (
    TokenWeaveScheduler,
    TokenWeaveSchedulerConfig,
)
from vllm.distributed.parallel_state import get_dp_group
from vllm.forward_context import DPMetadata
from schedflow.manager import SchedFlowManager
from schedflow.example.vllm.nanoflow import (
    NanoFlowScheduler,
    NanoFlowSchedulerConfig,
)
from schedflow.example.vllm.dbo import DBOScheduler, DBOSchedulerConfig
from schedflow.interface import OpSchedulerBase, SplitConfig
from vllm.v1.worker.ubatch_utils import UBatchSlice, UBatchSlices


_manager = SchedFlowManager()
_scheduler: (
    NanoFlowScheduler
    | DBOScheduler
    | TokenWeaveScheduler
    | FluxScheduler
    | None
) = None


def get_scheduler(
    config: NanoFlowSchedulerConfig
    | FluxSchedulerConfig
    | DBOSchedulerConfig
    | TokenWeaveSchedulerConfig,
) -> NanoFlowScheduler | DBOScheduler | TokenWeaveScheduler | FluxScheduler:
    global _scheduler
    if isinstance(config, TokenWeaveSchedulerConfig):
        _scheduler = TokenWeaveScheduler(config)
    elif isinstance(config, NanoFlowSchedulerConfig):
        _scheduler = NanoFlowScheduler(config)
    elif isinstance(config, DBOSchedulerConfig):
        _scheduler = DBOScheduler(config)
    elif isinstance(config, FluxSchedulerConfig):
        _scheduler = FluxScheduler(config)
    else:
        raise ValueError(f"Invalid scheduler config: {config}")
    return _scheduler


def get_manager(
    graph_module: torch.fx.GraphModule,
    config: SchedFlowConfig,
    scheduler: OpSchedulerBase,
    example_inputs: list[Any],
) -> SchedFlowManager:
    global _manager
    _manager.initialize(graph_module, config, scheduler, example_inputs)
    return _manager


def nano_ubatch_split(
    num_scheduled_tokens_per_request: np.ndarray,
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    is_dummy_run: bool = False,
    use_cudagraph: bool = False,
) -> tuple[UBatchSlices, torch.Tensor]:
    """
    Prepare two UBatch-compatible nano-batch slices.

    - Uses nano_manager.prepare_nano_split to decide if splitting is beneficial
      (i.e., num_nano_batches > 1).
    - Computes a single token split point using custom logic to remain
      compatible with UBatch execution.
    """
    batch_size = int(len(num_scheduled_tokens_per_request))
    tokens_list = num_scheduled_tokens_per_request.tolist()
    split_config = _manager.prepare(
        batch_size,
        tokens_list,
        is_dryrun=is_dummy_run,
        use_cudagraph=use_cudagraph,
    )
    dp_group = get_dp_group()
    dp_size, dp_rank = dp_group.world_size, dp_group.rank
    if dp_size == 1:
        dp_metadatas = None
        total_num_tokens_across_dp = [sum(split_config.num_tokens_padded)]
    else:
        dp_nano_split_config: list[Optional[SplitConfig]] = [
            None for _ in range(dp_size)
        ]
        dp_nano_split_config[dp_rank] = split_config
        disable_nano_split = False
        for i in range(dp_size):
            dp_nano_split_config[i] = dp_group.broadcast_object(
                dp_nano_split_config[i], src=i
            )
            remote_config = dp_nano_split_config[i]
            assert remote_config is not None
            if remote_config.num_nano_batches == 1:
                disable_nano_split = True
        if disable_nano_split:
            split_config = SplitConfig(
                num_nano_batches=1,
                batch_sizes=[batch_size],
                batch_indices=[0, batch_size],
                num_tokens=[num_tokens_unpadded],
                num_tokens_padded=[num_tokens_padded],
                split_indices=[0, num_tokens_unpadded],
                is_dryrun=is_dummy_run,
                use_cudagraph=use_cudagraph,
            )
            _manager.override_split_config(split_config)
            dp_nano_split_config[dp_rank] = split_config
            for i in range(dp_size):
                dp_nano_split_config[i] = dp_group.broadcast_object(
                    dp_nano_split_config[i], src=i
                )
            assert all(
                config is not None and config.num_nano_batches == 1
                for config in dp_nano_split_config
            )

        num_tokens_across_dp = [
            [
                config.num_tokens_padded[i]
                for config in dp_nano_split_config
                if config is not None
            ]
            for i in range(split_config.num_nano_batches)
        ]
        cu_num_tokens_across_dp = [
            [sum(tokens[: i + 1]) for i in range(len(tokens))]
            for tokens in num_tokens_across_dp
        ]
        total_num_tokens_across_dp = [
            sum(tokens[i] for tokens in num_tokens_across_dp)
            for i in range(dp_size)
        ]
        max_tokens_across_dp = [max(tokens) for tokens in num_tokens_across_dp]
        dp_metadatas = [
            DPMetadata(
                max_tokens_across_dp_cpu=torch.tensor(
                    max_tokens_across_dp[i],
                    device="cpu",
                    dtype=torch.int32,
                ),
                cu_tokens_across_dp_cpu=torch.tensor(
                    cu_num_tokens_across_dp[i],
                    device="cpu",
                    dtype=torch.int32,
                ),
                local_sizes=[
                    config.num_tokens_padded[i]
                    for config in dp_nano_split_config
                    if config is not None
                ],
            )
            for i in range(split_config.num_nano_batches)
        ]

    if dp_size > 1 and _scheduler is not None:
        assert isinstance(_scheduler, DBOScheduler)
        _scheduler.set_dp_metadata(dp_metadatas)

    return (
        [
            UBatchSlice(
                slice(
                    split_config.batch_indices[i],
                    split_config.batch_indices[i + 1],
                ),
                slice(
                    split_config.split_indices[i],
                    split_config.split_indices[i + 1],
                ),
            )
            for i in range(split_config.num_nano_batches)
        ],
        # The padding will be handled by the schedflow engine.
        torch.tensor(
            total_num_tokens_across_dp, device="cpu", dtype=torch.int32
        ),
    )
