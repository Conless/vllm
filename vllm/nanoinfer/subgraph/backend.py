# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Callable, Any, Optional, Dict, List

import torch
import torch.fx as fx

from .config import SubgraphConfig
from .inductor import inductor_compile_fx_adaptor_style
from .cudagraph import CUDAGraphWrapper, CUDAGraphOptions


class SubgraphBackend:
    gm: fx.GraphModule
    cfg: SubgraphConfig
    sym_shape_indices: List[int]
    general_callable: Optional[Callable[..., Any]]
    callables_per_size: Dict[int, Callable[..., Any]]

    def __init__(self, subgraph_gm: fx.GraphModule, sym_shape_indices: List[int], cfg: SubgraphConfig) -> None:
        self.gm = subgraph_gm
        self.cfg = cfg
        self.sym_shape_indices = sym_shape_indices
        self.general_callable = None
        self.callables_per_size = {}

    def _derive_runtime_size(self, args: tuple) -> Optional[int]:
        for i in self.sym_shape_indices:
            if i < len(args):
                x = args[i]
                if isinstance(x, torch.Tensor) and x.ndim >= 1:
                    try:
                        return int(x.shape[0])
                    except Exception:
                        return None
        return None

    def _maybe_wrap_cudagraph(self, fn: Callable[..., Any], is_first_graph: bool) -> Callable[..., Any]:
        if not self.cfg.enable_cudagraph:
            return fn
        options = CUDAGraphOptions(
            debug_log_enable=is_first_graph,
            gc_disable=not is_first_graph,
            weak_ref_output=True,
        )
        return CUDAGraphWrapper(fn, options)

    def _compile_general_dynamic(self, example_inputs: List[Any]) -> Callable[..., Any]:
        fn = inductor_compile_fx_adaptor_style(
            self.gm,
            example_inputs,
            inductor_options=self.cfg.inductor_options,
            runtime_shape=None,
        )
        return self._maybe_wrap_cudagraph(fn, is_first_graph=True)

    def _compile_for_size(self, example_inputs: List[Any], size: int) -> Callable[..., Any]:
        fn = inductor_compile_fx_adaptor_style(
            self.gm,
            example_inputs,
            inductor_options=self.cfg.inductor_options,
            runtime_shape=size,
        )
        return self._maybe_wrap_cudagraph(fn, is_first_graph=False)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # general dynamic
        if self.general_callable is None:
            self.general_callable = self._compile_general_dynamic(list(args))
            return self.general_callable(*args, **kwargs)

        size = self._derive_runtime_size(args)
        key = -1 if size is None else size
        if key not in self.callables_per_size:
            if (self.cfg.compile_sizes is not None and size in self.cfg.compile_sizes):
                self.callables_per_size[key] = self._compile_for_size(list(args), key)
            else:
                if self.general_callable is None:
                    self.general_callable = self._compile_general_dynamic(list(args))
                return self.general_callable(*args, **kwargs)

        return self.callables_per_size[key](*args, **kwargs)


