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


@register_lowering(aten._adaptive_avg_pool2d_backward)
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
