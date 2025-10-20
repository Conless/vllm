from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Generic, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from vllm.nanoinfer.example.nanoflow import NanoFlowSchedulerConfig
else:
    NanoFlowSchedulerConfig = Any


SupportedSchedulerConfig = TypeVar(
    "SupportedSchedulerConfig",
    bound=NanoFlowSchedulerConfig | None,
)


@dataclass
class CUDAGraphConfig:
    enabled: bool
    capture_sizes: list[int]
    weak_ref_output: bool = True
    check_ptr_consistency: bool = False


@dataclass
class InductorConfig:
    enabled: bool
    compile_sizes: set[int] | None = None
    options: dict | None = None
    disable_remote_cache: bool = True
    disable_autograd_cache: bool = True


@dataclass
class NanoInferConfig(Generic[SupportedSchedulerConfig]):
    splitting_ops: list[str]
    special_ops: dict[str, set[str]]
    scheduler_config: SupportedSchedulerConfig
    inductor_config: InductorConfig
    cudagraph_config: CUDAGraphConfig
