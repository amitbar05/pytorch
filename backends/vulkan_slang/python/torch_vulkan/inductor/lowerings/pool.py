"""M17.3 — Adaptive average pooling lowering for Vulkan Inductor.

Forward: For the integer-divisible case, delegates to ``aten.avg_pool2d``
which creates a Reduction IR node that the scheduler fuses with adjacent
pointwise ops. Non-divisible / non-Vulkan cases fall back to the eager handler.

Backward: Replaces the upstream ``make_fallback`` for
``aten._adaptive_avg_pool2d_backward`` with a decomposition into broadcast
+ scale that fuses with adjacent pointwise ops.
"""

from __future__ import annotations

import torch
from torch._inductor.lowering import register_lowering

aten = torch.ops.aten


@register_lowering(aten._adaptive_avg_pool2d.default)
def _adaptive_avg_pool2d_vulkan(x, output_size):
    """Lower the forward pass for adaptive_avg_pool2d.

    For the integer-divisible case (``H_in % H_out == 0 and W_in % W_out == 0``),
    the op is equivalent to a standard ``avg_pool2d`` with
    ``kernel_size = (H_in//H_out, W_in//W_out)``.  Delegating to
    ``aten.avg_pool2d.default`` produces a Reduction IR node that the
    scheduler can fuse with adjacent pointwise ops.

    Non-divisible and non-Vulkan cases fall back to the eager handler.
    """
    from torch._inductor.ir import TensorBox
    from torch._inductor.virtualized import V

    assert isinstance(x, TensorBox)
    x.realize_hint()

    # Only intercept Vulkan tensors.
    try:
        if x.get_device().type != "vulkan":
            return _fallback_fwd(x, output_size)
    except Exception:
        return _fallback_fwd(x, output_size)

    *batch, h_in, w_in = x.get_size()
    h_out, w_out = output_size

    h_in_int = V.graph.sizevars.guard_int(h_in)
    w_in_int = V.graph.sizevars.guard_int(w_in)

    if h_in_int % h_out == 0 and w_in_int % w_out == 0:
        kH = h_in_int // h_out
        kW = w_in_int // w_out
        # Route to avg_pool2d which creates a fusable Reduction IR node.
        from torch._inductor.lowering import lowerings as _lowerings

        return _lowerings[aten.avg_pool2d.default](x, [kH, kW])

    # Non-divisible case: fall through to eager handler.
    return _fallback_fwd(x, output_size)


# NOTE (anti-goal #3): @register_lowering moved to bwd_lowerings.py.
# This function is imported there and registered via _register_pool_adaptive_bwd().
def _adaptive_avg_pool2d_backward_vulkan(grad_out, x, output_size):
    """Lower the backward pass for adaptive_avg_pool2d.

    For the integer-divisible case, the backward is simply broadcasting
    ``grad_out / (kH * kW)`` to the input shape.  Inductor's scheduler
    fuses the broadcast with upstream pointwise/reduction ops.
    Non-divisible / non-Vulkan cases fall back to the upstream handler.
    """
    from torch._inductor.ir import TensorBox
    from torch._inductor.virtualized import V, ops

    assert isinstance(grad_out, TensorBox)
    grad_out.realize_hint()

    # Only intercept Vulkan tensors.
    try:
        if grad_out.get_device().type != "vulkan":
            return _fallback(grad_out, x, output_size)
    except Exception:
        return _fallback(grad_out, x, output_size)

    h_out, w_out = output_size

    # Determine input spatial size from `x`.
    if isinstance(x, TensorBox):
        x.realize_hint()
        *_, h_in, w_in = x.get_size()
        h_in_int = V.graph.sizevars.guard_int(h_in)
        w_in_int = V.graph.sizevars.guard_int(w_in)
    else:
        return _fallback(grad_out, x, output_size)

    if h_in_int % h_out == 0 and w_in_int % w_out == 0:
        kH = h_in_int // h_out
        kW = w_in_int // w_out
        scale = 1.0 / float(kH * kW)

        # grad_scaled = grad_out * scale
        grad_scaled = ops.mul(grad_out, ops.constant(grad_out.get_dtype(), scale))

        # Broadcast each output pixel to a (kH, kW) block in the input.
        # reshape -> expand -> reshape
        *b, ho, wo = grad_out.get_size()
        reshaped = ops.reshape(grad_scaled, [*b, ho, 1, wo, 1])
        expanded = ops.expand(reshaped, [*b, ho, kH, wo, kW])
        result = ops.reshape(expanded, [*b, h_in_int, w_in_int])
        return result

    return _fallback(grad_out, x, output_size)


def _fallback_fwd(x, output_size):
    """Route to the eager handler for the forward pass."""
    from torch._inductor.lowering import fallback_handler

    return fallback_handler(aten._adaptive_avg_pool2d.default)(x, output_size)


def _fallback(grad_out, x, output_size):
    """Route to the upstream fallback handler (eager Vulkan dispatch)."""
    from torch._inductor.lowering import fallback_handler

    return fallback_handler(aten._adaptive_avg_pool2d_backward)(
        grad_out, x, output_size
    )


# ═══════════════════════════════════════════════════════════════════════
# CODEGEN.2 — avg_pool2d_backward (non-overlapping, pure codegen)
# ═══════════════════════════════════════════════════════════════════════


def avg_pool2d_backward_codegen(
    grad_output, x, kernel_size, stride, padding, ceil_mode,
    count_include_pad, divisor_override,
):
    """CODEGEN.2 — pure-codegen avg_pool2d_backward for non-overlapping pools.

    Returns a TensorBox when the parameters allow a decomposition that
    avoids ``ops.indirect_indexing`` (which generates incorrect SPIR-V on
    Vulkan).  Otherwise returns ``None`` so the caller falls back to the
    eager Vulkan handler.

    Supported parameter combinations (all sizes are compile-time static):
      * stride == kernel_size, no padding, no ceil_mode
      * stride == kernel_size, padding > 0, count_include_pad=True
        (or divisor_override set) — broadcast to padded shape then slice

    The decomposition is:
      1. scale grad_output by 1/N  (N = divisor or kH*kW)
      2. reshape → expand → reshape to broadcast from output spatial to
         input spatial dimensions
      3. (padding case) slice out the valid input region
    All IR nodes (mul / reshape / expand / slice) produce fusable Slang
    kernels — no indirect_indexing, no eager dispatch.
    """
    import torch
    from torch._inductor.ir import TensorBox
    from torch._inductor.virtualized import V, ops

    if not stride:
        stride = kernel_size
    if not padding:
        padding = [0, 0]

    # Only handle 2-D pooling with list args.
    if not (isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 2):
        return None
    if not (isinstance(stride, (list, tuple)) and len(stride) == 2):
        return None
    if not (isinstance(padding, (list, tuple)) and len(padding) == 2):
        return None

    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding

    # ── Gate: only non-overlapping cases ──────────────────────────────
    if sh != kh or sw != kw:
        return None  # overlapping windows need indirect_indexing
    if ceil_mode:
        return None  # irregular output sizes

    # When count_include_pad=False AND there is padding, the per-pixel
    # divisor varies across spatial positions → needs indirect_indexing.
    if (ph > 0 or pw > 0) and not count_include_pad and divisor_override is None:
        return None

    # ── Resolve spatial sizes to Python ints ──────────────────────────
    assert isinstance(grad_output, TensorBox)
    assert isinstance(x, TensorBox)

    x_size = x.get_size()
    go_size = grad_output.get_size()
    if len(x_size) not in (3, 4):
        return None

    h_in = V.graph.sizevars.guard_int(x_size[-2])
    w_in = V.graph.sizevars.guard_int(x_size[-1])
    h_out = V.graph.sizevars.guard_int(go_size[-2])
    w_out = V.graph.sizevars.guard_int(go_size[-1])
    prefix = go_size[:-2]  # batch dims: [] for 3-D or [N] for 4-D

    # ── Compute uniform scale factor ──────────────────────────────────
    if divisor_override is not None:
        scale = 1.0 / float(divisor_override)
    else:
        # count_include_pad or no padding → uniform kH*kW divisor.
        scale = 1.0 / float(kh * kw)

    # ── Decompose: scale → broadcast → (optional slice) ───────────────
    grad_scaled = ops.mul(
        grad_output, ops.constant(grad_output.get_dtype(), scale)
    )

    # Broadcast from (prefix, h_out, w_out) to (prefix, h_out, 1, w_out, 1)
    # then expand with (kh, kw) in the inserted dims, then flatten back.
    reshaped = ops.reshape(grad_scaled, [*prefix, h_out, 1, w_out, 1])
    expanded = ops.expand(reshaped, [*prefix, h_out, kh, w_out, kw])

    padded_h = h_out * kh
    padded_w = w_out * kw

    # Padding case: broadcast to padded shape then slice.
    # Skip if the padded shape doesn't match the expected relationship.
    if ph > 0 or pw > 0:
        # Verify padded dimensions are consistent: h_out*kh == h_in + 2*ph
        if padded_h != h_in + 2 * ph or padded_w != w_in + 2 * pw:
            return None  # Inconsistent sizes — fall back
        grad_padded = ops.reshape(expanded, [*prefix, padded_h, padded_w])

        # Try to use aten.slice lowering (creates SliceView — pure codegen).
        from torch._inductor.lowering import lowerings as _L

        _aten = torch.ops.aten
        # aten.slice may be keyed as .Tensor or .default depending on
        # PyTorch version; try both.
        slice_fn = _L.get(getattr(_aten.slice, "Tensor", None)) or _L.get(_aten.slice.default)
        if slice_fn is None:
            return None  # slice lowering not available — fall back
        ndim = len(prefix) + 2
        result = slice_fn(grad_padded, dim=ndim - 2, start=ph, end=ph + h_in, step=1)
        result = slice_fn(result, dim=ndim - 1, start=pw, end=pw + w_in, step=1)
        return result

    # No padding → reshape directly to input spatial shape.
    return ops.reshape(expanded, [*prefix, h_in, w_in])
