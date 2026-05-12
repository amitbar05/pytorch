"""N.1.b / N.1.b-fast — ``aten.searchsorted`` + ``aten.repeat_interleave.Tensor``.

N.1.b (CPU roundtrip):
    Python-level PrivateUse1 overrides that perform CPU-side computation
    and ship the result back to Vulkan.  This unblocks the use-case
    immediately; a real GPU kernel is the N.1.b-fast follow-up.

N.1.b-fast (GPU-native binary search):
    A parallel-binary-search Slang compute shader (in ``bucket.slang``)
    replaces the CPU roundtrip.  Each thread handles one output element;
    log₂(seq_len) iterations, no shared memory, no barriers.

The ``aten.repeat_interleave.Tensor`` CPU roundtrip remains (it
decomposes through ``cumsum + arange + searchsorted + index``, and
with searchsorted on GPU, that chain is largely GPU-native except for
the final gather which falls to the pointwise path).
"""

from __future__ import annotations

import math
import os
import struct
from pathlib import Path

import torch

_registered = False
_searchsorted_src: str | None = None
_searchsorted_cache: dict[str, str] = {}


def _get_searchsorted_src() -> str:
    """Load the bucket.slang source and extract the searchsorted kernel."""
    global _searchsorted_src
    if _searchsorted_src is not None:
        return _searchsorted_src

    shader_dir = Path(__file__).parents[4] / "shaders" / "lib"
    bucket_path = shader_dir / "bucket.slang"
    if not bucket_path.exists():
        raise RuntimeError(f"N.1.b-fast: bucket.slang not found at {bucket_path}")
    _searchsorted_src = bucket_path.read_text(encoding="utf-8")
    return _searchsorted_src


def _dispatch_searchsorted_gpu(
    sorted_sequence: torch.Tensor,
    self: torch.Tensor,
    *,
    out_int32: bool = False,
    right: bool = False,
) -> torch.Tensor:
    """GPU-native parallel binary search for searchsorted.

    Dispatches the ``searchsorted`` compute shader from ``bucket.slang``.

    Args:
        sorted_sequence: Must be 1-D, sorted ascending.
        self: Values to search for (any shape).
        out_int32: If True, output dtype is int32; otherwise int64.
        right: If True, return right-most insertion index.

    Returns:
        Tensor of insertion indices, same shape as ``self``.
    """
    from ..runtime import compile_and_dispatch

    # sorted_sequence must be 1-D.
    if sorted_sequence.dim() != 1:
        raise ValueError(
            f"searchsorted: sorted_sequence must be 1-D, got {sorted_sequence.dim()}-D"
        )

    # Flatten values for dispatch; reshape output afterward.
    values_flat = self.reshape(-1).float()
    seq_len = sorted_sequence.shape[0]
    num_values = values_flat.shape[0]

    out_dtype = torch.int32 if out_int32 else torch.int64
    output_flat = torch.empty(num_values, dtype=out_dtype, device=self.device)

    # Ensure sorted_sequence is float32.
    sorted_seq_f32 = sorted_sequence.float()
    if not sorted_seq_f32.is_contiguous():
        sorted_seq_f32 = sorted_seq_f32.contiguous()
    if not values_flat.is_contiguous():
        values_flat = values_flat.contiguous()

    src = _get_searchsorted_src()

    # Push constants: seq_len, num_values, right
    pc = struct.pack("3I", seq_len, num_values, 1 if right else 0)

    # Buffers: [sorted_seq, values, output]
    buffers: list[torch.Tensor] = [sorted_seq_f32, values_flat, output_flat]

    # Workgroup size is 256; grid_x = ceil(num_values / 256)
    grid_x = (num_values + 255) // 256

    # Cache key based on source hash (the source is static).
    cache_key = "slang_searchsorted_bucket_v1"

    compile_and_dispatch(
        src,
        buffers,
        grid_x,
        1,
        1,
        push_constants=pc,
        num_outputs=1,
        entry="searchsorted",
        cache_key=cache_key,
    )

    # Reshape output to match input shape; cast to int64 if needed.
    result = output_flat.reshape(self.shape)
    if not out_int32 and result.dtype != torch.int64:
        result = result.to(torch.int64)
    return result


def _register_bucketize() -> None:
    """Register PrivateUse1 overrides for ``aten::bucketize`` (Tensor + Scalar).

    GPU-native: reuses the same ``searchsorted`` compute shader from
    ``bucket.slang``.  ``aten.bucketize`` is semantically identical to
    ``aten.searchsorted`` — both find insertion indices of values in a
    sorted sequence.  The only difference is in the Python-level kwarg
    names (``boundaries`` vs ``sorted_sequence``); the shader doesn't care.
    """
    import torch
    from torch.library import Library

    _lib = Library("aten", "IMPL")

    def _vulkan_bucketize_tensor(
        self,
        boundaries,
        *,
        out_int32=False,
        right=False,
    ):
        return _dispatch_searchsorted_gpu(
            boundaries,
            self,
            out_int32=bool(out_int32),
            right=bool(right),
        )

    def _vulkan_bucketize_scalar(
        self,
        boundaries,
        *,
        out_int32=False,
        right=False,
    ):
        values_t = torch.tensor(
            [self], device=boundaries.device, dtype=boundaries.dtype
        )
        result_t = _vulkan_bucketize_tensor(
            values_t,
            boundaries,
            out_int32=bool(out_int32),
            right=bool(right),
        )
        return result_t.item()

    def _try_impl(name, fn):
        try:
            _lib.impl(name, fn, "PrivateUse1", allow_override=True)
        except RuntimeError as exc:
            import logging

            logging.getLogger(__name__).warning(
                "bucketize: failed to register %s override (already installed?): %s",
                name,
                exc,
            )

    _try_impl("bucketize.Tensor", _vulkan_bucketize_tensor)
    _try_impl("bucketize.Scalar", _vulkan_bucketize_scalar)

    import sys

    sys.modules[__name__]._bucketize_lib = _lib  # type: ignore[attr-defined]


def _register_searchsorted_and_repeat_interleave_tensor() -> None:
    """Idempotently install PrivateUse1 overrides for ``aten::searchsorted``
    (Tensor + Scalar overloads), ``aten::bucketize`` (Tensor + Scalar),
    and ``aten::repeat_interleave.Tensor``.

    N.1.b-fast: ``searchsorted.Tensor`` and ``bucketize.Tensor`` use the GPU
    binary-search shader from ``bucket.slang`` instead of the CPU roundtrip.
    The Scalar overloads and ``repeat_interleave.Tensor`` still use CPU
    roundtrip (scalar is trivial; repeat_interleave decomposes through
    searchsorted which is now on GPU).

    All register with ``with_keyset=False`` (no parent kernel to
    forward to — the Vulkan path is unconditional) and
    ``allow_override=True`` so re-import in tests is a no-op rather
    than a hard error.
    """
    global _registered
    if _registered:
        return

    import torch
    from torch.library import Library

    _lib = Library("aten", "IMPL")

    def _vulkan_searchsorted_tensor(
        sorted_sequence,
        self,
        *,
        out_int32=False,
        right=False,
        side=None,
        sorter=None,
    ):
        # sorter is not supported on GPU path — fall back to CPU.
        if sorter is not None:
            cpu_seq = sorted_sequence.cpu()
            cpu_values = self.cpu()
            cpu_sorter = sorter.cpu()
            cpu_result = torch.searchsorted(
                cpu_seq,
                cpu_values,
                out_int32=bool(out_int32),
                right=bool(right),
                side=side,
                sorter=cpu_sorter,
            )
            return cpu_result.to(self.device)

        # GPU-native path: parallel binary search via bucket.slang.
        return _dispatch_searchsorted_gpu(
            sorted_sequence,
            self,
            out_int32=bool(out_int32),
            right=bool(right),
        )

    def _vulkan_searchsorted_scalar(
        sorted_sequence,
        self,
        *,
        out_int32=False,
        right=False,
        side=None,
        sorter=None,
    ):
        # Scalar case is trivial — wrap in a tensor and use GPU path.
        values_t = torch.tensor(
            [self], device=sorted_sequence.device, dtype=sorted_sequence.dtype
        )
        result_t = _vulkan_searchsorted_tensor(
            sorted_sequence,
            values_t,
            out_int32=bool(out_int32),
            right=bool(right),
            side=side,
            sorter=sorter,
        )
        return result_t.item()

    def _vulkan_repeat_interleave_tensor(repeats, *, output_size=None):
        cpu_repeats = repeats.cpu()
        cpu_result = torch.repeat_interleave(cpu_repeats, output_size=output_size)
        return cpu_result.to(repeats.device)

    def _try_impl(name, fn):
        try:
            _lib.impl(name, fn, "PrivateUse1", allow_override=True)
        except RuntimeError as exc:
            import logging

            logging.getLogger(__name__).warning(
                "N.1.b: failed to register %s override (already installed?): %s",
                name,
                exc,
            )

    _try_impl("searchsorted.Tensor", _vulkan_searchsorted_tensor)
    _try_impl("searchsorted.Scalar", _vulkan_searchsorted_scalar)
    _try_impl("repeat_interleave.Tensor", _vulkan_repeat_interleave_tensor)

    _registered = True

    # Hold a reference to the library so it isn't garbage-collected
    # (which would unregister our impls).  Mirrors ``bool_mask.py``.
    import sys

    sys.modules[__name__]._searchsorted_lib = _lib  # type: ignore[attr-defined]

    # Also register bucketize — same GPU shader, different ATen op name.
    _register_bucketize()
