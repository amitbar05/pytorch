"""Generic pointwise dispatch via Slang IPointwise/IPointwiseBinary interfaces.

CG.M12 — closes anti-goal #6 for pointwise by replacing embedded Slang
source strings with a single ``pointwise_generic.slang`` module.  Instead
of rendering a different Slang source per op, this module reads
``shaders/lib/pointwise_generic.slang`` (one file, four generic entry
points) and resolves the concrete op struct at SPIR-V compile time via
slangc's generic specialization — the same pattern the mm template uses
for its epilogue.

The source file lives in the shader lib directory so it participates in
the precompiled-module cache (``precompile_shader_libs()``) and benefits
from link-time IR re-use.

Usage (eager path — replaces ``ops::vulkan_abs`` etc.)::

    from .generic_pointwise_dispatch import dispatch_unary_pointwise

    output = dispatch_unary_pointwise("aten.abs", input_tensor)
"""

from __future__ import annotations

import hashlib
import os
import struct
from typing import Optional

import torch

from .generic_dispatch_table import (
    COMPLEX_POINTWISE_TABLE,
    POINTWISE_TABLE,
    PointwiseEntry,
)

_THREADGROUP_SIZE = 256

# ═══════════════════════════════════════════════════════════════════════════
# CG.M12: Single generic Slang source file — ONE file, ONE set of bindings,
# FOUR generic entry points, ALL pointwise ops.  The concrete op struct is
# supplied via the ``entry`` parameter to ``compile_slang_to_spirv``
# (e.g. ``entry="computeMain<OpAbs>"``).  Slang resolves the correct
# overload via the interface constraint.
# Anti-goal #6: no string-based template parameters — Slang generics only.
# ═══════════════════════════════════════════════════════════════════════════

# Path to the generic kernel source file, resolved relative to this module.
# (inductor/ → torch_vulkan/ → python/ → backends/vulkan_slang/ → shaders/lib/)
_POINTWISE_GENERIC_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "shaders",
        "lib",
        "pointwise_generic.slang",
    )
)

# Cached source content — read once, shared by all dispatches.
_pointwise_generic_src_cache: Optional[str] = None


def _get_pointwise_generic_src() -> str:
    """Return the contents of ``pointwise_generic.slang`` (cached)."""
    global _pointwise_generic_src_cache
    if _pointwise_generic_src_cache is None:
        with open(_POINTWISE_GENERIC_PATH) as f:
            _pointwise_generic_src_cache = f.read()
    return _pointwise_generic_src_cache


# ── Entry-point name ───────────────────────────────────────────────────
# CG.M12: All four entry points share the name ``computeMain`` — slangc
# resolves the correct overload via the interface constraint in the entry
# parameter (e.g. ``computeMain<OpAbs>`` matches the ``IPointwise``
# overload, ``computeMain<OpAdd>`` matches ``IPointwiseBinary``).


def _entry_name(entry: PointwiseEntry) -> str:
    """Return the slangc entry-point name for a specialized generic.

    E.g. ``"computeMain<OpAbs>"`` for aten.abs,
    ``"computeMain<OpAdd>"`` for aten.add.
    """
    return f"computeMain<{entry.op_struct}>"


def _make_cache_key(entry: PointwiseEntry, numel: int, complex_valued: bool) -> str:
    """Return a stable cache key for the (op, arity, numel) combination."""
    return hashlib.sha256(
        f"cgm12_{entry.op_struct}_{entry.arity}_{complex_valued}_{numel}".encode()
    ).hexdigest()[:12]


# ── Public dispatch API ──────────────────────────────────────────────────


def dispatch_unary_pointwise(
    aten_op: str,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    """Dispatch a unary pointwise op via the generic Slang template.

    Args:
        aten_op: The ATen op name (e.g. ``"aten.abs"``).
        input_tensor: Vulkan tensor, any supported dtype.

    Returns:
        Output Vulkan tensor with same shape and dtype as input.
    """
    entry = POINTWISE_TABLE.get(aten_op)
    if entry is None:
        raise RuntimeError(
            f"Generic pointwise dispatch: {aten_op} not in POINTWISE_TABLE"
        )
    if entry.arity != 1:
        raise RuntimeError(f"dispatch_unary_pointwise called for binary op {aten_op}")
    return _dispatch_pointwise_impl(entry, args=(input_tensor,))


def dispatch_binary_pointwise(
    aten_op: str,
    input_a: torch.Tensor,
    input_b: torch.Tensor,
) -> torch.Tensor:
    """Dispatch a binary pointwise op via the generic Slang template.

    Args:
        aten_op: The ATen op name (e.g. ``"aten.add"``).
        input_a, input_b: Vulkan tensors, same shape and dtype.

    Returns:
        Output Vulkan tensor.
    """
    entry = POINTWISE_TABLE.get(aten_op)
    if entry is None:
        raise RuntimeError(
            f"Generic pointwise dispatch: {aten_op} not in POINTWISE_TABLE"
        )
    if entry.arity != 2:
        raise RuntimeError(f"dispatch_binary_pointwise called for unary op {aten_op}")
    return _dispatch_pointwise_impl(entry, args=(input_a, input_b))


def dispatch_complex_pointwise(
    aten_op: str,
    *args: torch.Tensor,
) -> torch.Tensor:
    """Dispatch a complex-valued pointwise op via the generic Slang template.

    Supports both unary (float2 -> float2, e.g. conj) and binary
    (float2 x float2 -> float2, e.g. add/mul/div).  Input tensors must
    be complex (complex64 or complex128) on Vulkan.

    Args:
        aten_op: ATen op name (e.g. "aten.add").
        *args: One tensor for unary, two for binary.

    Returns:
        Complex-valued output Vulkan tensor.
    """
    entry = COMPLEX_POINTWISE_TABLE.get(aten_op)
    if entry is None:
        raise RuntimeError(
            f"Complex pointwise dispatch: {aten_op} not in COMPLEX_POINTWISE_TABLE"
        )
    if len(args) != entry.arity:
        raise RuntimeError(
            f"Complex pointwise dispatch: {aten_op} expects"
            f" {entry.arity} argument(s), got {len(args)}"
        )
    return _dispatch_pointwise_impl(entry, args=args, complex_valued=True)


# ── Implementation ───────────────────────────────────────────────────────


def _dispatch_pointwise_impl(
    entry: PointwiseEntry,
    args: tuple[torch.Tensor, ...],
    complex_valued: bool = False,
) -> torch.Tensor:
    """Compile and dispatch a pointwise kernel via Slang generics.

    CG.M12: Reads the single ``pointwise_generic.slang`` source file
    and compiles it with the op-specific entry point
    (e.g. ``computeMain<OpAbs>``).  Slang resolves the correct overload
    from the interface constraint.  The source is identical for every
    op — only the ``entry`` parameter changes.  The precompiled-module
    cache ensures slangc re-uses cached IR for the shared source across ops.

    No Jinja2 templating.  No per-op source variation.
    """
    from torch_vulkan.inductor.runtime import compile_and_dispatch

    tensor = args[0]
    numel = tensor.numel()

    src = _get_pointwise_generic_src()
    cache_key = _make_cache_key(entry, numel, complex_valued)

    wg_x = (numel + _THREADGROUP_SIZE - 1) // _THREADGROUP_SIZE
    pc_bytes = struct.pack("<I", numel)

    # CG.M12: ByteAddressBuffer uses fixed 3-buffer layout:
    #   binding 0: buf_in0  (first input, or only input for unary)
    #   binding 1: buf_in1  (second input for binary; dummy for unary)
    #   binding 2: buf_out  (output)
    output = torch.empty_like(tensor)
    if entry.arity == 1:
        # Unary: buf_in0 = input, buf_in1 = dummy (reuse input), buf_out = output
        bufs = [args[0], args[0], output]
    else:
        # Binary: buf_in0 = input_a, buf_in1 = input_b, buf_out = output
        bufs = [args[0], args[1], output]

    compile_and_dispatch(
        src=src,
        tensors=bufs,
        wg_x=wg_x,
        wg_y=1,
        wg_z=1,
        push_constants=pc_bytes,
        num_outputs=1,
        entry=_entry_name(entry),
        cache_key=cache_key,
    )
    return output


def can_dispatch(aten_op: str) -> Optional[PointwiseEntry]:
    """Check if an aten op is covered by generic pointwise dispatch."""
    return POINTWISE_TABLE.get(aten_op)


def can_dispatch_complex(aten_op: str) -> Optional[PointwiseEntry]:
    """Check if a complex aten op is covered by complex pointwise dispatch."""
    return COMPLEX_POINTWISE_TABLE.get(aten_op)
