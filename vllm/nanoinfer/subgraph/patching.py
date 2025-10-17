# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import AbstractContextManager, ExitStack
from typing import Optional


class AlwaysHitShapeEnv:
    """
    Provides a shape env whose guards always succeed.
    Matches vllm.compilation.compiler_interface.AlwaysHitShapeEnv semantics.
    """

    def __init__(self) -> None:
        self.guards = []

    def evaluate_guards_expression(self, *args, **kwargs):
        return True

    def get_pruned_guards(self, *args, **kwargs):
        return []

    def produce_guards_expression(self, *args, **kwargs):
        return ""


def get_metrics_context() -> AbstractContextManager:
    """
    Returns a (possibly null) dynamo metrics context.
    Mirrors compiler_interface.InductorAdaptor.metrics_context().
    """
    import torch._dynamo.utils  # type: ignore
    return torch._dynamo.utils.get_metrics_context()  # type: ignore[attr-defined]


def inductor_adaptor_patching_context(
    *,
    runtime_shape: Optional[int],
    base_cache_dir: Optional[str] = None,
    disable_remote_cache: bool = True,
    enable_autograd_cache: bool = False,
) -> AbstractContextManager:
    """
    Context that applies InductorAdaptor-style patching around compile_fx:
    - Patch torch._inductor.codecache.FxGraphCache._get_shape_env -> AlwaysHitShapeEnv
    - Patch torch._inductor.codecache._check_can_cache -> no-op
    - Disable remote cache (torch._inductor.config.patch)
    - Optionally disable AOTAutograd cache (torch._functorch.config patches)
    - Apply a metrics context (re-entrant)
    - (Optional) We do not hook compile_fx_inner here; can be added if needed.
    """
    import torch
    from unittest.mock import patch

    class _Ctx(AbstractContextManager):
        def __enter__(self):
            self.stack = ExitStack()
            # metrics context
            self.stack.enter_context(get_metrics_context())
            # remote cache disable
            from torch._inductor import config as inductor_config  # type: ignore
            if disable_remote_cache and hasattr(inductor_config, "patch"):
                self.stack.enter_context(inductor_config.patch(
                    fx_graph_remote_cache=False))  # type: ignore
            # autograd cache toggles
            from torch._functorch import config as functorch_config  # type: ignore
            if not enable_autograd_cache and hasattr(functorch_config, "patch"):
                self.stack.enter_context(functorch_config.patch(
                    enable_autograd_cache=False))  # type: ignore
                self.stack.enter_context(functorch_config.patch(
                    enable_remote_autograd_cache=False))  # type: ignore
            # shape env + can_cache patches
            self.stack.enter_context(
                patch("torch._inductor.codecache.FxGraphCache._get_shape_env",
                      lambda *a, **k: AlwaysHitShapeEnv()))
            self.stack.enter_context(
                patch("torch._inductor.codecache.FxGraphCache._check_can_cache",
                      lambda *a, **k: None))
            return self

        def __exit__(self, exc_type, exc, tb):
            return self.stack.__exit__(exc_type, exc, tb)

    return _Ctx()


# Local replacements for vLLM utilities we cannot import

def set_inductor_config(config: dict, runtime_shape: Optional[int]) -> None:
    """
    Mirror of vllm.compilation.compiler_interface.set_inductor_config without imports.
    When runtime_shape is a specific int, enable tuning knobs based on env vars.
    """
    if isinstance(runtime_shape, int):
        config["max_autotune"] = True
        config["coordinate_descent_tuning"] = True
