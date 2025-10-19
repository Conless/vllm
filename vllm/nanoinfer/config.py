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
    enabled: bool = False
    weak_ref_output: bool = True
    capture_sizes: list[int] = field(default_factory=list)


@dataclass
class InductorConfig:
    compile_sizes: set[int] | None = None
    options: dict | None = None
    disable_remote_cache: bool = True
    disable_autograd_cache: bool = True


@dataclass
class NanoInferConfig(Generic[SupportedSchedulerConfig]):
    splitting_ops: list[str]
    special_ops: dict[str, str]
    scheduler_config: SupportedSchedulerConfig
    inductor_config: InductorConfig = InductorConfig()
    cudagraph_config: CUDAGraphConfig = CUDAGraphConfig()
