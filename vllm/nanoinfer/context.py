# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Generator


@dataclass
class NanoInferContext:
    nano_batch_idx: tuple[int, ...]
    """The index of the nano-batch."""
    num_tokens_padded: tuple[int, ...]
    """The number of tokens in the padded nano-batch."""
    is_dryrun: bool
    """Whether this is a dry run."""
    use_cudagraph: bool
    """Whether to use CUDA graph."""


_forward_context: NanoInferContext | None = None


def get_forward_context() -> NanoInferContext:
    assert _forward_context is not None
    return _forward_context


@contextmanager
def set_forward_context(
    context: NanoInferContext,
) -> Generator[None, None, None]:
    global _forward_context
    prev_context = _forward_context
    _forward_context = context
    try:
        yield
    finally:
        _forward_context = prev_context
