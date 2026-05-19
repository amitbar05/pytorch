"""PF.6.b — unary bwd_diff dispatch."""

from __future__ import annotations

import struct

import torch
from torch_vulkan.inductor.bwd_diff.emit_helpers import (
    _DEFAULT_NUMTHREADS,
    _cache_key,
    _check_float,
    _check_vulkan,
    _ensure_f32,
    _entry,
    _narrow_from_f32,
    _slang_dtype_str,
    resolve_backward_kind,
)
from torch_vulkan.inductor.bwd_diff_table import emit_bwd_diff_kernel
from torch_vulkan.inductor.runtime import compile_and_dispatch


def dispatch_unary_bwd(
    aten_op: str,
    x: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    no_diff_kwargs: dict[str, float] | None = None,
    out: torch.Tensor | None = None,
    numthreads: int = _DEFAULT_NUMTHREADS,
) -> torch.Tensor:
    """Dispatch the autodiff-emitted backward for a unary entry.

    Returns ``grad_in`` with the same shape/dtype/device as ``x``.

    ``no_diff_kwargs`` supplies values for any ``no_diff_params`` declared
    on the entry (e.g. ``leaky_relu.negative_slope``); ``KeyError`` if a
    required key is missing or an unexpected key is provided. Mirrors
    ``dispatch_binary_bwd``.

    f16/bf16 inputs are widened to f32 for the compute kernel (T3.1).

    T4.2: Validates op routing against ``BWD_TEMPLATE_REGISTRY``.  If the
    op is registered as ``TEMPLATE_JINJA`` or ``BACKWARD_DERIVATIVE``
    instead of ``BWD_DIFF``, raises a clear error directing the caller to
    the correct dispatch path (``dispatch_template_bwd()``).
    """
    # T4.2: Check BWD_TEMPLATE_REGISTRY before dispatching.  If this op
    # is registered with a non-BWD_DIFF kind, the caller must use
    # dispatch_template_bwd() (for TEMPLATE_JINJA) or the fast backward
    # annotation path (for BACKWARD_DERIVATIVE).
    resolved = resolve_backward_kind(aten_op)
    if resolved is not None and not resolved.is_bwd_diff:
        raise ValueError(
            f"T4.2: dispatch_unary_bwd called for {aten_op!r}, but "
            f"BWD_TEMPLATE_REGISTRY lists kind={resolved.kind}. "
            f"Use dispatch_template_bwd() for TEMPLATE_JINJA entries."
        )

    entry = _entry(aten_op, expected_arity=1)
    orig_dtype = x.dtype
    _check_float(x, grad_out)
    _check_vulkan(x, grad_out)
    if grad_out.shape != x.shape:
        raise ValueError(
            f"PF.6.b: grad_out shape {tuple(grad_out.shape)} does not "
            f"match x shape {tuple(x.shape)} (unary backward)"
        )
    x_f32 = _ensure_f32(x)
    go_f32 = _ensure_f32(grad_out)
    grad_in_f32 = out.float() if out is not None else torch.empty_like(x_f32)
    if grad_in_f32.shape != x.shape:
        raise ValueError(
            f"PF.6.b: out tensor shape {tuple(grad_in_f32.shape)} "
            f"does not match x {tuple(x.shape)}"
        )
    _check_vulkan(grad_in_f32)
    # B.5.C: validate / pack no_diff scalars (e.g. leaky_relu negative_slope).
    no_diff_kwargs = dict(no_diff_kwargs or {})
    missing = [k for k in entry.no_diff_params if k not in no_diff_kwargs]
    if missing:
        raise KeyError(
            f"PF.6.b: aten op {aten_op!r} requires no_diff_kwargs "
            f"{missing}; got keys {list(no_diff_kwargs)}"
        )
    extra = [k for k in no_diff_kwargs if k not in entry.no_diff_params]
    if extra:
        raise KeyError(
            f"PF.6.b: aten op {aten_op!r} has no_diff_params "
            f"{list(entry.no_diff_params)}; received unexpected keys "
            f"{extra}"
        )
    numel = x_f32.numel()
    slang_dtype = _slang_dtype_str(orig_dtype)
    src = emit_bwd_diff_kernel(
        aten_op,
        dtype=slang_dtype,
        numthreads=numthreads,
    )
    fmt = "<" + "f" * len(entry.no_diff_params) + "I"
    values = [float(no_diff_kwargs[k]) for k in entry.no_diff_params]
    pc = struct.pack(fmt, *values, numel)
    wg_x = (numel + numthreads - 1) // numthreads
    compile_and_dispatch(
        src,
        tensors=[x_f32.contiguous(), go_f32.contiguous(), grad_in_f32],
        wg_x=wg_x,
        wg_y=1,
        wg_z=1,
        push_constants=pc,
        num_outputs=1,
        entry="bwd_op",
        cache_key=_cache_key(aten_op, orig_dtype, numthreads),
    )
    if out is not None:
        if out.dtype != orig_dtype:
            out.copy_(_narrow_from_f32(grad_in_f32, orig_dtype))
        else:
            out.copy_(grad_in_f32)
        return out
    return _narrow_from_f32(grad_in_f32, orig_dtype)
