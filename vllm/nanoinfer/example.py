# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from typing_extensions import override

from vllm.nanoinfer.interface import (InputInfo, OpInfo, OpSchedulerBase,
                                      SplitConfig)


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
        self.comm_finished: list[Optional[torch.cuda.Event]] = [
            None for _ in range(config.max_num_nano_batches)
        ]
        self.comp_finished: list[Optional[torch.cuda.Event]] = [
            None for _ in range(config.max_num_nano_batches)
        ]

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
    def schedule(
        self,
        split_config: SplitConfig,
        op_infos: dict[int, list[OpInfo]],
        executor: Callable,
    ) -> None:
        for op_info_0, op_info_1 in zip(op_infos[0], op_infos[1]):
            op_info_list = [op_info_0, op_info_1]
            for batch_idx, op_info in enumerate(op_info_list):
                if op_info.submod_name == "":
                    executor(op_info)
                    continue
                tag = op_info.tag
                if tag == "network":
                    torch.cuda.set_stream(self.comm_stream)
                    self.comm_finished[batch_idx] = torch.cuda.Event()
                    if self.comp_finished[batch_idx] is not None:
                        comp_finished_event = self.comp_finished[batch_idx]
                        assert comp_finished_event is not None
                        comp_finished_event.wait()
                        self.comp_finished[batch_idx] = None
                else:
                    torch.cuda.set_stream(self.comp_stream)
                    self.comp_finished[batch_idx] = torch.cuda.Event()
                    if self.comm_finished[batch_idx] is not None:
                        comm_finished_event = self.comm_finished[batch_idx]
                        assert comm_finished_event is not None
                        comm_finished_event.wait()
                        self.comm_finished[batch_idx] = None
                executor(op_info)
                if tag == "network":
                    comm_finished_event = self.comm_finished[batch_idx]
                    assert comm_finished_event is not None
                    comm_finished_event.record()
                else:
                    comp_finished_event = self.comp_finished[batch_idx]
                    assert comp_finished_event is not None
                    comp_finished_event.record()
