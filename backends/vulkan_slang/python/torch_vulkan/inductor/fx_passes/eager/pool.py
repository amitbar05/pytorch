"""Pooling custom-op registrations for fused FX-pattern targets (PF.30.a/.d)."""

from __future__ import annotations


def _ensure_max_pool2d_op_registered() -> "object":
    """Register ``torch_vulkan::max_pool2d`` as a custom_op (PF.30.a follow-up).

    ``aten.max_pool2d.default`` carries an ``AutogradPrivateUse1`` dispatch
    key that runs the C++ ``vulkan_max_pool2d`` kernel *before* FakeTensorMode's
    ``__torch_dispatch__`` interception, so Dynamo's fake-trace hits a
    ``data_ptr()`` on a FakeTensor and graph-break-fails. Routing the
    Vulkan path through a Python custom_op with ``register_fake`` lets
    Dynamo treat it as opaque and shape-infer correctly.
    """
    import torch

    op_name = "torch_vulkan::max_pool2d"
    existing = getattr(torch.ops.torch_vulkan, "max_pool2d", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    Tensor = torch.Tensor

    def _max_pool2d_impl(
        input: Tensor,
        kernel_size: list[int],
        stride: list[int],
        padding: list[int],
        dilation: list[int],
        ceil_mode: bool,
    ) -> Tensor:
        return torch.ops.aten.max_pool2d.default(
            input,
            list(kernel_size),
            list(stride),
            list(padding),
            list(dilation),
            bool(ceil_mode),
        )

    _max_pool2d_impl.__annotations__ = {
        "input": Tensor,
        "kernel_size": list[int],
        "stride": list[int],
        "padding": list[int],
        "dilation": list[int],
        "ceil_mode": bool,
        "return": Tensor,
    }
    pool_op = torch.library.custom_op(op_name, mutates_args=())(_max_pool2d_impl)

    def _max_pool2d_fake(input, kernel_size, stride, padding, dilation, ceil_mode):
        import os

        if os.environ.get("TORCH_VULKAN_TRACE_MAXPOOL2D"):
            print(
                f"_max_pool2d_fake: input.shape={list(input.shape)} "
                f"ks={kernel_size} st={stride} pd={padding} dl={dilation} "
                f"ceil={ceil_mode}",
                flush=True,
            )
        N, C = input.shape[0], input.shape[1]
        H_in, W_in = input.shape[-2], input.shape[-1]
        # Tolerate scalar args (sometimes the dispatcher hands them in
        # un-normalized when the caller used integer kernel_size/stride).
        if isinstance(kernel_size, int):
            K_h = K_w = kernel_size
        else:
            K_h = kernel_size[0]
            K_w = kernel_size[-1] if len(kernel_size) > 1 else K_h
        if stride is None or (hasattr(stride, "__len__") and len(stride) == 0):
            s_h, s_w = K_h, K_w
        elif isinstance(stride, int):
            s_h = s_w = stride
        else:
            s_h = stride[0]
            s_w = stride[-1] if len(stride) > 1 else s_h
        if isinstance(padding, int):
            p_h = p_w = padding
        else:
            p_h = padding[0]
            p_w = padding[-1] if len(padding) > 1 else p_h
        if isinstance(dilation, int):
            d_h = d_w = dilation
        else:
            d_h = dilation[0]
            d_w = dilation[-1] if len(dilation) > 1 else d_h
        if ceil_mode:
            H_out = -(-(H_in + 2 * p_h - d_h * (K_h - 1) - 1) // s_h) + 1
            W_out = -(-(W_in + 2 * p_w - d_w * (K_w - 1) - 1) // s_w) + 1
        else:
            H_out = (H_in + 2 * p_h - d_h * (K_h - 1) - 1) // s_h + 1
            W_out = (W_in + 2 * p_w - d_w * (K_w - 1) - 1) // s_w + 1
        return input.new_empty((N, C, H_out, W_out))

    pool_op.register_fake(_max_pool2d_fake)

    # C2 follow-up: register autograd via Inductor decomposition path.
    # ``aten.max_pool2d_with_indices_backward`` is the standard backward
    # primitive — it scatters ``grad_output`` into the input-shaped grid
    # using the saved indices.  Both ``max_pool2d_with_indices`` and its
    # backward have working Inductor lowerings + Slang shaders, so the
    # backward graph trains entirely through auto-generated kernels (no
    # eager AutogradPrivateUse1 shim, no hand-written
    # ``max_pool2d_backward.slang``).
    #
    # Why we recompute indices in ``setup_context`` rather than threading
    # them through the forward signature: the custom_op forward returns a
    # single Tensor (matching ``F.max_pool2d``'s public contract). Running
    # ``max_pool2d_with_indices`` here under the same Vulkan FakeTensor /
    # PrivateUse1 dispatch reuses the C++ kernel's indices output without
    # changing the forward graph contract — and during AOTAutograd's
    # joint trace the indices land in ``ctx.saved_tensors`` as a vulkan
    # FakeTensor that the partitioner saves across the fw/bw boundary.
    def _max_pool2d_setup_context(ctx, inputs, output):
        inp, kernel_size, stride, padding, dilation, ceil_mode = inputs
        _y, idx = torch.ops.aten.max_pool2d_with_indices.default(
            inp,
            list(kernel_size),
            list(stride) if stride else list(kernel_size),
            list(padding),
            list(dilation),
            bool(ceil_mode),
        )
        ctx.save_for_backward(inp, idx)
        ctx.kernel_size = list(kernel_size)
        ctx.stride = list(stride) if stride else list(kernel_size)
        ctx.padding = list(padding)
        ctx.dilation = list(dilation)
        ctx.ceil_mode = bool(ceil_mode)

    def _max_pool2d_backward(ctx, grad_output):
        inp, idx = ctx.saved_tensors
        g_inp = torch.ops.aten.max_pool2d_with_indices_backward.default(
            grad_output,
            inp,
            ctx.kernel_size,
            ctx.stride,
            ctx.padding,
            ctx.dilation,
            ctx.ceil_mode,
            idx,
        )
        # custom_op autograd expects one grad per forward input.
        return g_inp, None, None, None, None, None

    pool_op.register_autograd(
        _max_pool2d_backward,
        setup_context=_max_pool2d_setup_context,
    )
    return torch.ops.torch_vulkan.max_pool2d.default


def _ensure_adaptive_avg_pool2d_op_registered() -> "object":
    """Register ``torch_vulkan::adaptive_avg_pool2d`` as a custom_op.

    Same pattern as ``_ensure_max_pool2d_op_registered`` (PF.30.d).
    ``aten.adaptive_avg_pool2d.default`` has ``PrivateUse1`` and
    ``AutogradPrivateUse1`` dispatch keys that run the Vulkan C++ kernel
    before FakeTensorMode's ``__torch_dispatch__`` interception, so
    Dynamo's fake-trace hits ``data_ptr()`` on a FakeTensor and fails.
    Routing the Vulkan path through a Python custom_op with
    ``register_fake`` lets Dynamo treat it as opaque and shape-infer
    correctly.
    """
    import torch

    op_name = "torch_vulkan::adaptive_avg_pool2d"
    existing = getattr(torch.ops.torch_vulkan, "adaptive_avg_pool2d", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    Tensor = torch.Tensor

    def _adaptive_avg_pool2d_impl(
        input: Tensor,
        output_size: list[int],
    ) -> Tensor:
        return torch.ops.aten.adaptive_avg_pool2d.default(
            input,
            list(output_size),
        )

    _adaptive_avg_pool2d_impl.__annotations__ = {
        "input": Tensor,
        "output_size": list[int],
        "return": Tensor,
    }
    pool_op = torch.library.custom_op(op_name, mutates_args=())(
        _adaptive_avg_pool2d_impl
    )

    def _adaptive_avg_pool2d_fake(input, output_size):
        N, C = input.shape[0], input.shape[1]
        if isinstance(output_size, int):
            oH = oW = output_size
        else:
            oH = output_size[0]
            oW = output_size[-1] if len(output_size) > 1 else oH
        return input.new_empty((N, C, oH, oW))

    pool_op.register_fake(_adaptive_avg_pool2d_fake)

    # Register autograd: delegate to aten's ``_adaptive_avg_pool2d_backward``.
    # The Vulkan backward kernel is registered on PrivateUse1 in
    # ``Registration.cpp``, so calling the aten op on Vulkan tensors
    # dispatches to ``vulkan_adaptive_avg_pool2d_backward``.
    def _adaptive_avg_pool2d_setup_context(ctx, inputs, output):
        inp, output_size = inputs
        ctx.save_for_backward(inp)
        ctx.output_size = (
            list(output_size)
            if not isinstance(output_size, int)
            else [output_size, output_size]
        )

    def _adaptive_avg_pool2d_backward(ctx, grad_output):
        inp = ctx.saved_tensors[0]
        g_inp = torch.ops.aten._adaptive_avg_pool2d_backward.default(grad_output, inp)
        return g_inp, None

    pool_op.register_autograd(
        _adaptive_avg_pool2d_backward,
        setup_context=_adaptive_avg_pool2d_setup_context,
    )
    return torch.ops.torch_vulkan.adaptive_avg_pool2d.default
