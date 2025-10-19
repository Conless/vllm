# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Callable
from typing_extensions import override

import torch
import torch.fx as fx

from vllm.nanoinfer.backend.cudagraph import CUDAGraphWrapper
from vllm.nanoinfer.backend.inductor import inductor_compile_fx_adaptor_style
from vllm.nanoinfer.config import NanoInferConfig
from vllm.nanoinfer.context import get_forward_context


class SubgraphBackend:
    def __init__(
        self,
        subgraph_gm: fx.GraphModule,
        shape_arg_index: int,
        config: NanoInferConfig,
    ) -> None:
        self.gm = subgraph_gm
        self.config = config
        self.shape_arg_index = shape_arg_index
        self.callable_dynamic: Callable[..., Any] | None = None
        self.callables_per_size: dict[int, Callable[..., Any]] = {}

    def compile(
        self, example_inputs: list[Any], runtime_shape: int | None = None
    ) -> Callable[..., Any]:
        shape = example_inputs[self.shape_arg_index]
        assert (
            (isinstance(shape, torch.SymInt) and runtime_shape is None)
            or (int(shape) == runtime_shape)
        )
        fn = inductor_compile_fx_adaptor_style(
            self.gm,
            example_inputs,
            inductor_cfg=self.config.inductor_config,
            runtime_shape=runtime_shape,
        )
        if not self.config.cudagraph_config.enabled:
            return fn
        return CUDAGraphWrapper(fn, self.config.cudagraph_config)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        assert self.shape_arg_index < len(args)
        size = int(args[self.shape_arg_index])
        if (
            size is not None
            and self.config.inductor_config.compile_sizes is not None
            and size in self.config.inductor_config.compile_sizes
        ):
            if size not in self.callables_per_size:
                assert get_forward_context().is_dryrun
                self.callables_per_size[size] = self.compile(list(args), size)
            return self.callables_per_size[size](*args, **kwargs)
        assert self.callable_dynamic is not None
        return self.callable_dynamic(*args, **kwargs)


class SubgraphCompileInterpreter(fx.Interpreter):
    def __init__(
        self,
        module: fx.GraphModule,
        compile_submods: list[str],
        config: NanoInferConfig,
    ) -> None:
        super().__init__(module)
        self.config = config
        self.compile_submods = set(compile_submods)
        from torch._guards import detect_fake_mode

        self.fake_mode = detect_fake_mode()

    @override
    def run(
        self,
        *args: Any,
        initial_env: dict[fx.Node, Any] | None = None,
        enable_io_processing: bool = True,
    ) -> Any:
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
            sym_shape_indices = [
                i for i, x in enumerate(args) if isinstance(x, torch.SymInt)
            ]
            assert len(sym_shape_indices) > 0
            primary_shape_index: int = sym_shape_indices[0]
            assert isinstance(submod, fx.GraphModule)
            backend = SubgraphBackend(
                submod, primary_shape_index, self.config
            )
            backend.callable_dynamic = backend.compile(list(args))
            self.module.__dict__[target] = backend
        return out
