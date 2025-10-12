# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch

from vllm.distributed.parallel_state import get_dp_group
from vllm.forward_context import DPMetadata, get_forward_context
from vllm.nanoinfer import manager as nano_manager
from vllm.nanoinfer.interface import OperatorHandle, SplitConfig
from vllm.v1.worker.ubatch_utils import UBatchSlice, UBatchSlices


def nano_ubatch_split(
    num_scheduled_tokens_per_request: np.ndarray,
    num_tokens_unpadded: int,
    num_tokens_padded: int,
) -> tuple[Optional[UBatchSlices], Optional[torch.Tensor]]:
    """
    Prepare two UBatch-compatible nano-batch slices.

    - Uses nano_manager.prepare_nano_split to decide if splitting is beneficial
      (i.e., num_nano_batches > 1).
    - Computes a single token split point using custom logic to remain
      compatible with UBatch execution.
    """
    assert num_tokens_unpadded == num_tokens_padded
    batch_size = int(len(num_scheduled_tokens_per_request))
    tokens_list = num_scheduled_tokens_per_request.tolist()
    split_config = nano_manager.prepare_nano_split(batch_size, tokens_list)
    dp_group = get_dp_group()
    dp_size, dp_rank = dp_group.world_size, dp_group.rank
    if dp_size == 1:
        if getattr(split_config, "num_nano_batches", 1) <= 1:
            return (None, None)
        assert split_config.num_nano_batches == 2
        dp_metadatas = None
        total_num_tokens_across_dp = [num_tokens_padded]
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
            nano_manager.disable_nano_split()
            return (None, None)
        num_tokens_across_dp = [
            [
                config.num_tokens[i]
                for config in dp_nano_split_config
                if config is not None
            ]
            for i in range(split_config.num_nano_batches)
        ]
        total_num_tokens_across_dp = [
            sum(tokens[i] for tokens in num_tokens_across_dp) for i in range(dp_size)
        ]
        max_tokens_across_dp = [max(tokens) for tokens in num_tokens_across_dp]
        dp_metadatas = [
            DPMetadata(
                max_tokens_across_dp_cpu=torch.tensor(
                    max_tokens_across_dp[i],
                    device="cpu",
                    dtype=torch.int32,
                ),
                num_tokens_across_dp_cpu=torch.tensor(
                    num_tokens_across_dp[i],
                    device="cpu",
                    dtype=torch.int32,
                ),
                local_sizes=[
                    config.num_tokens[i]
                    for config in dp_nano_split_config
                    if config is not None
                ],
            )
            for i in range(split_config.num_nano_batches)
        ]

    first_slice = UBatchSlice(
        slice(0, split_config.batch_indices[1]),
        slice(0, split_config.split_indices[1]),
    )
    second_slice = UBatchSlice(
        slice(split_config.batch_indices[1], batch_size),
        slice(split_config.split_indices[1], split_config.split_indices[2]),
    )

    @contextmanager
    def op_hook(op_info: tuple[OperatorHandle]):
        assert len(op_info) == 1
        ctx = get_forward_context()
        attn_metadata_list = ctx.attn_metadata
        if attn_metadata_list is not None:
            assert isinstance(attn_metadata_list, list)
            ctx.attn_metadata = attn_metadata_list[op_info[0].nano_batch_idx]
        previous_dp_metadata = None
        if dp_metadatas is not None:
            previous_dp_metadata = ctx.dp_metadata
            ctx.dp_metadata = dp_metadatas[op_info[0].nano_batch_idx]

        try:
            yield
        finally:
            ctx.attn_metadata = attn_metadata_list
            if previous_dp_metadata is not None:
                ctx.dp_metadata = previous_dp_metadata
            pass

    nano_manager.set_op_hook(op_hook)

    return (
        [first_slice, second_slice],
        torch.tensor(total_num_tokens_across_dp, device="cpu", dtype=torch.int32),
    )


def disable_nano_split():
    nano_manager.disable_nano_split()
