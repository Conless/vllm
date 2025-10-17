# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Callable, Any, List, Optional

import copy
import torch.fx as fx

from .patching import (
    inductor_adaptor_patching_context,
    set_inductor_config,
)


def inductor_compile_fx_adaptor_style(
    sub_gm: fx.GraphModule,
    example_inputs: List[Any],
    *,
    inductor_options: Optional[dict],
    runtime_shape: Optional[int],
) -> Callable[..., Any]:
    """
    Compile an FX subgraph with Inductor (compile_fx) under InductorAdaptor-style patching.
    """
    from torch._inductor.compile_fx import compile_fx

    patches = {"fx_graph_cache": True, "fx_graph_remote_cache": False}
    if inductor_options:
        patches.update(inductor_options)

    # apply inductor config (tuning knobs) based on runtime_shape
    set_inductor_config(patches, runtime_shape)

    # protect original graph from in-place modification
    sub_gm_copied = copy.deepcopy(sub_gm)

    with inductor_adaptor_patching_context(runtime_shape=runtime_shape):
        compiled_graph = compile_fx(sub_gm_copied, list(example_inputs), config_patches=patches)
        return compiled_graph # type: ignore[return-value]
