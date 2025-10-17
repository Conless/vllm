# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, Callable, Optional, Dict, Tuple, Union

import torch

@dataclass
class CUDAGraphOptions:
    debug_log_enable: bool = True
    gc_disable: bool = False
    weak_ref_output: bool = True


def _weak_ref_tensor(tensor: Any) -> Any:
    """Create a weak reference to a tensor via torch custom op if available."""
    if isinstance(tensor, torch.Tensor):
        return torch.ops._C.weak_ref_tensor(tensor)  # type: ignore[attr-defined]
    return tensor


def _weak_ref_tensors(
    tensors: Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor], Any]
) -> Union[torch.Tensor, list[Any], tuple[Any], Any]:
    """Apply weak refs to tensors, lists, tuples, and duck-typed IntermediateTensors."""
    if isinstance(tensors, torch.Tensor):
        return _weak_ref_tensor(tensors)
    if isinstance(tensors, list):
        return [_weak_ref_tensor(t) for t in tensors]
    if isinstance(tensors, tuple):
        return tuple(_weak_ref_tensor(t) for t in tensors)
    # Duck-typing for IntermediateTensors without importing vLLM
    if hasattr(tensors, "tensors") and tensors.__class__.__name__ == "IntermediateTensors":
        inner = getattr(tensors, "tensors")
        if isinstance(inner, dict):
            new_inner = {k: _weak_ref_tensor(v) for k, v in inner.items()}
            try:
                return tensors.__class__(new_inner)
            except Exception:
                pass
    return tensors


class CUDAGraphWrapper:
    def __init__(self, runnable: Callable[..., Any], options: Optional[CUDAGraphOptions] = None):
        self.runnable = runnable
        self.options = options or CUDAGraphOptions()
        self.graph_pool = None
        self._entries: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    def derive_key(self, *args: Any, **kwargs: Any) -> Tuple[str, int]:
        for a in args:
            if isinstance(a, torch.Tensor) and a.ndim >= 1:
                return ("shape0", int(a.shape[0]))
        return ("default", 0)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        key = self.derive_key(*args, **kwargs)
        entry = self._entries.get(key)
        if entry is None:
            cudagraph = torch.cuda.CUDAGraph()
            # No platform pool integration to avoid external imports

            # optional gc suppression / logging / input address record
            input_addresses = [
                a.data_ptr() for a in args if isinstance(a, torch.Tensor)
            ]
            with torch.cuda.graph(cudagraph):
                out = self.runnable(*args, **kwargs)
                if self.options.weak_ref_output:
                    out = _weak_ref_tensors(out)

            self._entries[key] = {"graph": cudagraph, "out": out, "inputs": input_addresses}
            # return non-weak output on first capture
            return out

        # Input address validation (debug-like parity)
        new_addrs = [a.data_ptr() for a in args if isinstance(a, torch.Tensor)]
        if new_addrs != entry.get("inputs", new_addrs):
            raise RuntimeError("CUDAGraph input addresses changed between capture and replay")
        entry["graph"].replay()
        return entry["out"]


