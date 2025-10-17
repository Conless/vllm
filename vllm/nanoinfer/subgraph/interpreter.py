# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import List, Any, Optional
from typing_extensions import override

import torch
import torch.fx as fx

from .backend import SubgraphBackend
from .config import SubgraphConfig

class SubgraphCompileInterpreter(fx.Interpreter):
    cfg: SubgraphConfig
    compile_submods: set[str]
    sym_shape_indices: List[int]

    def __init__(self, module: fx.GraphModule, compile_submods: List[str], cfg: SubgraphConfig) -> None:
        super().__init__(module)
        self.cfg = cfg
        self.compile_submods = set(compile_submods)
        self.sym_shape_indices = []
        from torch._guards import detect_fake_mode
        self.fake_mode = detect_fake_mode()

    @override
    def run(self, *args: Any, initial_env: Optional[dict[fx.Node, Any]] = None, enable_io_processing: bool = True) -> Any:
        assert self.fake_mode is not None
        fake_args = [
            self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in args
        ]
        from torch._dispatch.python import enable_python_dispatcher
        with self.fake_mode, enable_python_dispatcher():
            return super().run(*fake_args)

    @override
    def call_module(self, target, args: tuple, kwargs: dict) -> Any:
        assert isinstance(target, str)
        out = super().call_module(target, args, kwargs)
        if target in self.compile_submods:
            submod = self.fetch_attr(target)
            if not self.sym_shape_indices:
                self.sym_shape_indices = [i for i, x in enumerate(args) if isinstance(x, torch.SymInt)]
            assert isinstance(submod, fx.GraphModule)
            backend = SubgraphBackend(submod, self.sym_shape_indices, self.cfg)
            backend.general_callable = backend._compile_general_dynamic(list(args))
            # Replace submodule with backend
            self.module.__dict__[target] = backend
        return out
