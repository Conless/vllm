# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from dataclasses import dataclass

import torch
from typing_extensions import override

from vllm.nanoinfer.interface import InputInfo, OpSchedulerBase, SplitConfig


@dataclass
class NanoFlowSchedulerConfig:
    min_nano_split_tokens: int
    max_num_nano_batches: int


class NanoFlowScheduler(OpSchedulerBase):

    def __init__(self, config: NanoFlowSchedulerConfig):
        super().__init__("nanoflow")
        self.config = config
        self.comm_stream = torch.cuda.Stream()
        self.comp_stream = torch.cuda.Stream()

    @override
    def get_split_config(self, input_info: InputInfo) -> SplitConfig:
        prefix_sum = [0] + list(itertools.accumulate(input_info.num_tokens))
        mid = min(
            range(len(prefix_sum)),
            key=lambda i: abs(prefix_sum[i] -
                              (prefix_sum[-1] - prefix_sum[i])),
        )

        if (prefix_sum[mid] < self.config.min_nano_split_tokens
                or (prefix_sum[-1] - prefix_sum[mid])
                < self.config.min_nano_split_tokens):
            return SplitConfig(
                num_nano_batches=1,
                batch_sizes=[input_info.batch_size],
                batch_indices=[0, input_info.batch_size],
                num_tokens=[prefix_sum[-1]],
                split_indices=[0, prefix_sum[-1]],
            )
        else:
            return SplitConfig(
                num_nano_batches=2,
                batch_sizes=[mid, input_info.batch_size - mid],
                batch_indices=[0, mid, input_info.batch_size],
                num_tokens=[prefix_sum[mid], prefix_sum[-1] - prefix_sum[mid]],
                split_indices=[0, prefix_sum[mid], prefix_sum[-1]],
            )

    @override
    async def schedule(self, context) -> None:
        """Schedule operators with stream overlap using async interface.

        This scheduler:
        - Pops operators from all nano-batches in lockstep
        - Assigns network ops to comm_stream, others to comp_stream
        - Engine handles cross-stream synchronization via events
        """
        from vllm.nanoinfer.interface import ExecutionContext
        assert isinstance(context, ExecutionContext)

        num_batches = context.split_config.num_nano_batches
        batch_indices = list(range(num_batches))

        while batch_indices:
            ops = []
            for batch_idx in batch_indices:
                op = await context.pop(batch_idx)
                if op is None:
                    batch_indices.remove(batch_idx)
                    continue
                ops.append((batch_idx, op))

            for batch_idx, op in ops:
                tag = op.debug_info.get("tag", "")
                if tag == "network":
                    stream = self.comm_stream
                else:
                    stream = self.comp_stream
                with torch.cuda.stream(stream):
                    await context.execute((op, ))

        stream_events = [torch.cuda.Event(), torch.cuda.Event()]
        with torch.cuda.stream(self.comp_stream):
            stream_events[0].record()
        with torch.cuda.stream(self.comm_stream):
            stream_events[1].record()
        for event in stream_events:
            event.wait()
