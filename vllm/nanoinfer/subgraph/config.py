# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Optional


@dataclass
class SubgraphConfig:
    """
    Controls subgraph splitting and runtime compilation.

    - splitting_ops: Operator names to split on; if None, treat top-level
      call_module boundaries only.
    - compile_sizes: If provided, only compile per-size variants for sizes in
      this set; if None, compile on first encounter of any size.
    - inductor_options: Dict forwarded to compile_fx via config_patches.
      Defaults include: {"fx_graph_cache": True, "fx_graph_remote_cache": False}.
    - enable_cudagraph: If True, wrap compiled callables with CUDA graph
      capture/replay using CUDAGraphOptions.
    """
    splitting_ops: Optional[list[str]] = None
    compile_sizes: Optional[set[int]] = None
    inductor_options: Optional[dict] = None
    enable_cudagraph: bool = False


