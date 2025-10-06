# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class InputInfo:
    batch_size: int
    num_tokens: list[int]
    total_num_tokens: int


@dataclass
class SplitConfig:
    """Configuration for nano-batch splitting."""

    num_nano_batches: int
    batch_sizes: list[int]
    batch_indices: list[int]
    num_tokens: list[int]
    split_indices: list[int]


@dataclass(eq=False, frozen=True)
class OpInfo:
    """Information about an operator execution."""

    submod_name: str
    tag: str
    idx: int  # nano-batch index


class OpSchedulerBase(ABC):

    def __init__(self, policy_name: str):
        self.policy_name = policy_name

    @abstractmethod
    def get_split_config(self, input_info: InputInfo) -> SplitConfig:
        pass

    @abstractmethod
    def schedule(self, split_config: SplitConfig,
                 operators: dict[int, list[OpInfo]]) -> None:
        pass
