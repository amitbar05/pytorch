"""Opaque conv2d_backward custom-op registration (M22b split from conv.py).

Contains ``_ensure_conv2d_backward_op_registered`` — the M17.8.d.2 opaque
non-autograd custom op that prevents AOTAutograd from decomposing the conv
backward into ``empty_like`` sub-ops that the partitioner collapses to zeros.

TRAIN.7 (2026-05-27): The eager impl now routes fp32 Vulkan groups==1 through
``_slang_tile_conv2d_bwd`` (single Slang compute dispatch via
``bwd_diff(conv_inner_madd)``), eliminating the prior CPU roundtrip through
``aten.convolution_backward`` → C++ ``vulkan_convolution_backward_overrideable``.
"""

from __future__ import annotations


def _ensure_conv2d_backward_op_registered() -> "object":
    """Register ``torch_vulkan::conv2d_backward`` as a non-autograd custom_op.

    M17.8.d.2 (2026-05-17): During AOTAutograd's joint-graph trace, the
    custom-op autograd ``_conv2d_backward`` runs with FakeTensors and
    previously fell through to ``torch.ops.aten.convolution_backward.default``.
    That call dispatched to our PrivateUse1 fake (``shape_ops.py::
    _convolution_backward_overrideable_fake``) whose body uses
    ``torch.empty_like(input)`` / ``torch.empty_like(weight)``. AOTAutograd's
    proxy tracer **recorded those sub-ops** into the FX graph instead of
    preserving a single op node — the joint-partitioner then saw
    ``empty_like(weight)`` as shape-only and dropped the primals from the
    backward partition. Inductor lowered the result as ``alloc + zero-init``,
    silently producing all-zero conv weight gradients in compile mode.

    This op is **non-autograd** and **opaque to the tracer**: a single
    ``torch_vulkan::conv2d_backward.default`` node lands in the FX graph,
    the joint-partitioner correctly preserves ``input`` / ``weight`` as
    backward inputs, and ``make_fallback`` (registered in
    ``lowerings/__init__.py``) makes Inductor emit a real
    ``extern_kernels.conv2d_backward(...)`` call that runs the C++ adapter
    at runtime.

    Idempotent — safe to call multiple times.
    """
    import torch
    from torch import Tensor

    op_name = "torch_vulkan::conv2d_backward"
    existing = getattr(torch.ops.torch_vulkan, "conv2d_backward", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    def _conv2d_backward_impl(
        input: Tensor,
        grad_output: Tensor,
        weight: Tensor,
        stride: list[int],
        padding: list[int],
        dilation: list[int],
        groups: int,
        has_bias: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Eager impl: route fp32 Vulkan groups==1 through _slang_tile_conv2d_bwd
        (TRAIN.7, 2026-05-27) — single Slang compute dispatch, no CPU roundtrip.
        Falls back to aten.convolution_backward for groups>1 or non-fp32.
        """
        use_slang_bwd = (
            int(groups) == 1
            and input.device.type == "vulkan"
            and input.dtype == torch.float32
            and grad_output.dtype == torch.float32
            and weight.dtype == torch.float32
        )

        if use_slang_bwd:
            from ..templates.caller.conv import _slang_tile_conv2d_bwd

            sH, sW = int(stride[0]), int(stride[-1] if len(stride) > 1 else stride[0])
            pH, pW = int(padding[0]), int(padding[-1] if len(padding) > 1 else padding[0])
            dH, dW = int(dilation[0]), int(dilation[-1] if len(dilation) > 1 else dilation[0])

            # TRAIN.5: route through pool_acquire for buffer reuse across steps
            from ...buffer_pool import pool_acquire

            _gi = pool_acquire(tuple(input.shape), input.dtype, input.device)
            grad_input = _gi.zero_() if _gi is not None else torch.zeros_like(input)
            _gw = pool_acquire(tuple(weight.shape), weight.dtype, weight.device)
            grad_weight = _gw.zero_() if _gw is not None else torch.zeros_like(weight)
            if has_bias:
                _gb = pool_acquire(
                    (weight.shape[0],), input.dtype, input.device,
                )
                grad_bias = _gb.zero_() if _gb is not None else torch.zeros(
                    weight.shape[0], device=input.device, dtype=input.dtype,
                )
            else:
                grad_bias = None

            _slang_tile_conv2d_bwd(
                input, weight, grad_output,
                grad_input, grad_weight,
                stride=(sH, sW),
                padding=(pH, pW),
                dilation=(dH, dW),
                grad_bias=grad_bias,
            )

            if not has_bias:
                grad_bias = grad_output.new_empty((0,))
            return grad_input, grad_weight, grad_bias

        result = torch.ops.aten.convolution_backward.default(
            grad_output,
            input,
            weight,
            None,
            list(stride),
            list(padding),
            list(dilation),
            False,
            [0] * len(stride),
            int(groups),
            [True, True, bool(has_bias)],
        )
        g_inp = (
            result[0]
            if result[0] is not None
            else input.new_empty(input.shape).zero_()
        )
        g_w = (
            result[1]
            if result[1] is not None
            else weight.new_empty(weight.shape).zero_()
        )
        if has_bias:
            g_b = (
                result[2]
                if len(result) > 2 and result[2] is not None
                else grad_output.new_empty((weight.shape[0],)).zero_()
            )
        else:
            # Return a zero-size bias so the tuple arity is stable. Callers
            # ignore this when has_bias=False.
            g_b = grad_output.new_empty((0,))
        return g_inp, g_w, g_b

    _conv2d_backward_impl.__annotations__ = {
        "input": Tensor,
        "grad_output": Tensor,
        "weight": Tensor,
        "stride": list[int],
        "padding": list[int],
        "dilation": list[int],
        "groups": int,
        "has_bias": bool,
        "return": tuple[Tensor, Tensor, Tensor],
    }
    bwd_op = torch.library.custom_op(op_name, mutates_args=())(_conv2d_backward_impl)

    def _conv2d_backward_fake(
        input, grad_output, weight, stride, padding, dilation, groups, has_bias
    ):
        # Shape inference for the opaque op. Use ``new_empty(shape)`` (M18.3
        # canonical) so the proxy tracer treats these as storage-bound
        # allocations rather than shape-only proxies.
        g_inp = input.new_empty(input.shape)
        g_w = weight.new_empty(weight.shape)
        if has_bias:
            g_b = grad_output.new_empty((weight.shape[0],))
        else:
            g_b = grad_output.new_empty((0,))
        return g_inp, g_w, g_b

    bwd_op.register_fake(_conv2d_backward_fake)
    return torch.ops.torch_vulkan.conv2d_backward.default
