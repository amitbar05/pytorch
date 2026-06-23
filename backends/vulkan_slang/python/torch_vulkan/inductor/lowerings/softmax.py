"""Softmax / log-softmax forward lowerings.

Backward lowerings for _softmax_backward_data and _log_softmax_backward_data
have been moved to ``bwd_lowerings.py`` (TR.19 backward consolidation).
"""

from __future__ import annotations


def _register_softmax() -> None:
    import torch
    from torch._inductor import lowering as L
    from torch._inductor.lowering import register_lowering

    aten = torch.ops.aten

    # Save original lowerings before overriding.
    _orig_softmax = L.lowerings.get(aten._softmax)
    _orig_log_softmax = L.lowerings.get(aten._log_softmax)

    @register_lowering(aten._softmax, type_promotion_kind=None)
    def _vulkan_softmax(x, dim, half_to_float):
        if not _is_vulkan(x):
            if _orig_softmax is not None:
                return _orig_softmax(x, dim, half_to_float)
            return NotImplemented
        x_max = L.lowerings[aten.amax](x, [dim], True)
        x_shifted = L.lowerings[aten.sub.Tensor](x, x_max)
        x_exp = L.lowerings[aten.exp.default](x_shifted)
        x_sum = L.lowerings[aten.sum.dim_IntList](x_exp, [dim], keepdims=True)
        return L.lowerings[aten.div.Tensor](x_exp, x_sum)

    @register_lowering(aten._log_softmax, type_promotion_kind=None)
    def _vulkan_log_softmax(x, dim, half_to_float):
        if not _is_vulkan(x):
            if _orig_log_softmax is not None:
                return _orig_log_softmax(x, dim, half_to_float)
            return NotImplemented
        x_max = L.lowerings[aten.amax](x, [dim], True)
        x_shifted = L.lowerings[aten.sub.Tensor](x, x_max)
        x_exp = L.lowerings[aten.exp.default](x_shifted)
        x_sum = L.lowerings[aten.sum.dim_IntList](x_exp, [dim], keepdims=True)
        x_log_sum = L.lowerings[aten.log.default](x_sum)
        return L.lowerings[aten.sub.Tensor](x_shifted, x_log_sum)


# _softmax_backward_data and _log_softmax_backward_data lowerings
# have been consolidated into ``bwd_lowerings.py`` (TR.19).


def _register_softmax_backward() -> None:
    pass  # TR.19 — moved to bwd_lowerings.py


from . import _is_vulkan  # noqa: E402,F401
