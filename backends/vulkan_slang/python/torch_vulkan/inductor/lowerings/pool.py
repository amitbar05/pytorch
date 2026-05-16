"""M17.3 — Adaptive average pooling lowering for Vulkan Inductor.

Replaces the upstream ``make_fallback`` for ``aten._adaptive_avg_pool2d_backward``
with a decomposition into broadcast + scale that fuses with adjacent pointwise ops.
"""

from __future__ import annotations

import torch
from torch._inductor.lowering import register_lowering

aten = torch.ops.aten


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


def _fallback(grad_out, x, output_size):
    """Route to the upstream fallback handler (eager Vulkan dispatch)."""
    from torch._inductor.lowering import fallback_handler

    return fallback_handler(aten._adaptive_avg_pool2d_backward)(
        grad_out, x, output_size
    )
