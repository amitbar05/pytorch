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
def _adaptive_avg_pool2d_backward_vulkan(grad_out, x):
    """Lower the backward pass for adaptive_avg_pool2d.

    aten::_adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) → Tensor
    (no output_size arg — derived from grad_out spatial dims).

    For the integer-divisible case each input pixel (h, w) receives the
    gradient of output pixel (h//kH, w//kW) scaled by 1/(kH*kW).  Uses
    Pointwise.create so ops.* runs inside the kernel body where it belongs
    — calling ops.* on TensorBox at lowering time produces invalid OpsValue.
    Non-divisible / non-Vulkan cases fall back to the C++ Vulkan handler.
    """
    from torch._inductor.ir import Pointwise, TensorBox
    from torch._inductor.virtualized import V, ops

    if not isinstance(grad_out, TensorBox):
        return _fallback(grad_out, x)

    grad_out.realize_hint()

    try:
        if grad_out.get_device().type != "vulkan":
            return _fallback(grad_out, x)
    except Exception:
        return _fallback(grad_out, x)

    *b, ho, wo = grad_out.get_size()
    try:
        h_out = V.graph.sizevars.guard_int(ho)
        w_out = V.graph.sizevars.guard_int(wo)
    except Exception:
        return _fallback(grad_out, x)

    if not isinstance(x, TensorBox):
        return _fallback(grad_out, x)

    x.realize_hint()
    *_, h_in, w_in = x.get_size()
    try:
        h_in_int = V.graph.sizevars.guard_int(h_in)
        w_in_int = V.graph.sizevars.guard_int(w_in)
    except Exception:
        return _fallback(grad_out, x)

    if h_in_int % h_out != 0 or w_in_int % w_out != 0:
        return _fallback(grad_out, x)

    kH = h_in_int // h_out
    kW = w_in_int // w_out
    scale = 1.0 / float(kH * kW)
    dtype = grad_out.get_dtype()
    device = grad_out.get_device()

    try:
        b_ints = [V.graph.sizevars.guard_int(d) for d in b]
    except Exception:
        return _fallback(grad_out, x)

    try:
        grad_out_loader = grad_out.make_loader()
        _kH, _kW, _scale, _dtype = kH, kW, scale, dtype

        def inner_fn(idx):
            *b_idx, h, w = idx
            # Affine index: input (h,w) → output (h//kH, w//kW).
            # kH/kW are compile-time ints → simple sympy floor-div, not indirect_indexing.
            val = grad_out_loader([*b_idx, h // _kH, w // _kW])
            return ops.mul(val, ops.constant(_scale, _dtype))

        return Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=[*b_ints, h_in_int, w_in_int],
        )
    except Exception:
        return _fallback(grad_out, x)


def _fallback_fwd(x, output_size):
    """Route to the eager handler for the forward pass."""
    from torch._inductor.lowering import fallback_handler

    return fallback_handler(aten._adaptive_avg_pool2d.default)(x, output_size)


def _fallback(grad_out, x):
    """Route to the upstream fallback handler (eager Vulkan dispatch)."""
    from torch._inductor.lowering import fallback_handler

    return fallback_handler(aten._adaptive_avg_pool2d_backward)(grad_out, x)


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
    Vulkan).  Otherwise returns ``None`` so the caller falls back.

    Supported parameter combinations (all sizes must be compile-time static):
      * stride == kernel_size, no padding, no ceil_mode
      * stride == kernel_size, padding > 0, count_include_pad=True
        (or divisor_override set)

    Uses Pointwise.create with an inner_fn for the index mapping — calling
    ops.* on TensorBox at lowering time produces invalid OpsValue nodes.
    """
    from torch._inductor.ir import Pointwise, TensorBox
    from torch._inductor.virtualized import V, ops

    if not stride:
        stride = kernel_size
    if not padding:
        padding = [0, 0]

    if not (isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 2):
        return None
    if not (isinstance(stride, (list, tuple)) and len(stride) == 2):
        return None
    if not (isinstance(padding, (list, tuple)) and len(padding) == 2):
        return None

    kh, kw = int(kernel_size[0]), int(kernel_size[1])
    sh, sw = int(stride[0]), int(stride[1])
    ph, pw = int(padding[0]), int(padding[1])

    if sh != kh or sw != kw:
        return None  # overlapping windows need indirect_indexing
    if ceil_mode:
        return None

    if (ph > 0 or pw > 0) and not count_include_pad and divisor_override is None:
        return None

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
    prefix_sym = list(go_size[:-2])
    try:
        prefix_ints = [V.graph.sizevars.guard_int(d) for d in prefix_sym]
    except Exception:
        return None

    if divisor_override is not None:
        scale = 1.0 / float(divisor_override)
    else:
        scale = 1.0 / float(kh * kw)

    dtype = grad_output.get_dtype()
    device = grad_output.get_device()
    grad_loader = grad_output.make_loader()
    _kh, _kw, _ph, _pw, _scale, _dtype = kh, kw, ph, pw, scale, dtype

    if ph > 0 or pw > 0:
        padded_h = h_out * kh
        padded_w = w_out * kw
        if padded_h != h_in + 2 * ph or padded_w != w_in + 2 * pw:
            return None

        def inner_fn_pad(idx):
            *pre, h, w = idx
            # Padded input coord → output coord via floor-div by compile-time stride.
            oh = (h + _ph) // _kh
            ow = (w + _pw) // _kw
            val = grad_loader([*pre, oh, ow])
            return ops.mul(val, ops.constant(_scale, _dtype))

        return Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn_pad,
            ranges=[*prefix_ints, h_in, w_in],
        )

    def inner_fn(idx):
        *pre, h, w = idx
        # Input coord → output coord via floor-div by compile-time kernel size.
        oh = h // _kh
        ow = w // _kw
        val = grad_loader([*pre, oh, ow])
        return ops.mul(val, ops.constant(_scale, _dtype))

    return Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=[*prefix_ints, h_in, w_in],
    )
