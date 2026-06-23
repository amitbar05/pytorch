"""PF.6.b — binary bwd_diff dispatch."""

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


def dispatch_binary_bwd(
    aten_op: str,
    a: torch.Tensor,
    b: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    no_diff_kwargs: dict[str, float] | None = None,
    out_a: torch.Tensor | None = None,
    out_b: torch.Tensor | None = None,
    numthreads: int = _DEFAULT_NUMTHREADS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch the autodiff-emitted backward for a binary entry.

    Returns ``(grad_a, grad_b)`` — each shape/dtype/device matching
    its corresponding forward input. ``no_diff_kwargs`` supplies values
    for any ``no_diff_params`` declared on the entry (e.g.
    ``smooth_l1.beta``, ``huber.delta``); ``KeyError`` if a required
    key is missing.

    f16/bf16 inputs are widened to f32 for the compute kernel (T3.1).

    T4.2: Validates op routing against ``BWD_TEMPLATE_REGISTRY``.
    """
    # T4.2: Check BWD_TEMPLATE_REGISTRY before dispatching.
    resolved = resolve_backward_kind(aten_op)
    if resolved is not None and not resolved.is_bwd_diff:
        raise ValueError(
            f"T4.2: dispatch_binary_bwd called for {aten_op!r}, but "
            f"BWD_TEMPLATE_REGISTRY lists kind={resolved.kind}. "
            f"Use dispatch_template_bwd() for TEMPLATE_JINJA entries."
        )

    entry = _entry(aten_op, expected_arity=2)
    orig_dtype = a.dtype
    _check_float(a, b, grad_out)
    _check_vulkan(a, b, grad_out)
    # Scalar (0-D) grad_out arises when loss.backward() is called on a
    # mean/sum-reduced loss.  Expand to match a's shape before dispatch.
    if grad_out.dim() == 0 and a.dim() > 0:
        grad_out = grad_out.expand(a.shape)
    if a.shape != b.shape or a.shape != grad_out.shape:
        raise ValueError(
            f"PF.6.b: binary backward expects matching shapes for "
            f"a/b/grad_out; got {tuple(a.shape)}/{tuple(b.shape)}/"
            f"{tuple(grad_out.shape)}"
        )
    a_f32 = _ensure_f32(a)
    b_f32 = _ensure_f32(b)
    go_f32 = _ensure_f32(grad_out)
    grad_a_out = out_a.float() if out_a is not None else torch.empty_like(a_f32)
    grad_b_out = out_b.float() if out_b is not None else torch.empty_like(b_f32)
    _check_vulkan(grad_a_out, grad_b_out)
    if grad_a_out.shape != a.shape or grad_b_out.shape != b.shape:
        raise ValueError("PF.6.b: out_a/out_b shape mismatch with a/b")
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
    numel = a_f32.numel()
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
        tensors=[
            a_f32.contiguous(),
            b_f32.contiguous(),
            go_f32.contiguous(),
            grad_a_out,
            grad_b_out,
        ],
        wg_x=wg_x,
        wg_y=1,
        wg_z=1,
        push_constants=pc,
        num_outputs=2,
        entry="bwd_op",
        cache_key=_cache_key(aten_op, orig_dtype, numthreads),
    )
    ga = _narrow_from_f32(grad_a_out, orig_dtype)
    gb = _narrow_from_f32(grad_b_out, orig_dtype)
    if out_a is not None:
        out_a.copy_(ga)
    if out_b is not None:
        out_b.copy_(gb)
    return ga, gb
