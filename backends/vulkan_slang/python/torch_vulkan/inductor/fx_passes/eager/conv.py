"""Conv2d / Conv1d custom-op registrations for fused FX-pattern targets (PF.30.a)."""

from __future__ import annotations


def _ensure_conv2d_with_optional_bias_op_registered() -> "object":
    """Register ``torch_vulkan::conv2d_with_optional_bias`` as a custom_op.

    PF.30.a — replaces the ``@torch.compiler.disable`` graph-break that
    ``_patched_conv2d`` carried in ``python/torch_vulkan/__init__.py``.
    Dynamo treats custom_ops as opaque (no trace-through) and uses the
    registered fake_impl for shape inference, so Inductor's fake-tensor
    pass never invokes the C++ ``vulkan_conv2d`` kernel — sidestepping
    the missing null-storage MetaGuard that broke Strike 1.

    The eager backing materializes ``bias=None`` to zeros and dtype-aligns
    weight/bias before forwarding to ``torch.ops.aten.convolution.default``.
    Calling aten directly avoids re-entering the patched ``F.conv2d``.
    """
    import torch

    op_name = "torch_vulkan::conv2d_with_optional_bias"
    existing = getattr(torch.ops.torch_vulkan, "conv2d_with_optional_bias", None)
    # NOTE: Always re-register to pick up source changes to the backward.
    # The early-return was preventing FakeTensor fixes from taking effect.
    if existing is not None and hasattr(existing, "default"):
        pass  # op exists; re-register autograd below

    Tensor = torch.Tensor

    def _conv2d_impl(
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        stride: list[int],
        padding: list[int],
        dilation: list[int],
        groups: int,
    ) -> Tensor:
        if weight.dtype != input.dtype:
            weight = weight.to(dtype=input.dtype)
        if bias is None:
            bias = torch.zeros(weight.shape[0], device=input.device, dtype=input.dtype)
        elif bias.dtype != input.dtype:
            bias = bias.to(dtype=input.dtype)
        return torch.ops.aten.convolution.default(
            input,
            weight,
            bias,
            list(stride),
            list(padding),
            list(dilation),
            False,
            [0, 0],
            int(groups),
        )

    _conv2d_impl.__annotations__ = {
        "input": Tensor,
        "weight": Tensor,
        "bias": Tensor | None,
        "stride": list[int],
        "padding": list[int],
        "dilation": list[int],
        "groups": int,
        "return": Tensor,
    }
    conv_op = torch.library.custom_op(op_name, mutates_args=())(_conv2d_impl)

    def _conv2d_fake(input, weight, bias, stride, padding, dilation, groups):
        N = input.shape[0]
        C_out = weight.shape[0]
        H_in, W_in = input.shape[-2], input.shape[-1]
        K_h, K_w = weight.shape[-2], weight.shape[-1]
        s_h, s_w = (stride[0], stride[-1])
        p_h, p_w = (padding[0], padding[-1])
        d_h, d_w = (dilation[0], dilation[-1])
        H_out = (H_in + 2 * p_h - d_h * (K_h - 1) - 1) // s_h + 1
        W_out = (W_in + 2 * p_w - d_w * (K_w - 1) - 1) // s_w + 1
        return input.new_empty((N, C_out, H_out, W_out))

    conv_op.register_fake(_conv2d_fake)

    # C2: Register autograd — delegate to aten's convolution_backward.
    # Without this, AOT Autograd can't trace through the custom op and
    # training fails with "no autograd formula was registered".
    def _conv2d_setup_context(ctx, inputs, output):
        inp, w, b, stride, padding, dilation, groups = inputs
        ctx.save_for_backward(inp, w, b if b is not None else None)
        ctx.stride = list(stride)
        ctx.padding = list(padding)
        ctx.dilation = list(dilation)
        ctx.groups = int(groups)

    @torch.compiler.disable
    def _conv2d_backward(ctx, grad_output):
        inp, w, saved_b = ctx.saved_tensors
        has_bias = saved_b is not None and saved_b.numel() > 0
        groups = int(ctx.groups)

        # CG.M6: Route through the [Differentiable]-based conv backward template
        # when groups==1, tensors are on Vulkan, f32, AND have real storage
        # (not FakeTensors during AOT Autograd tracing).
        #
        # During AOT Autograd's joint graph trace, all inputs are FakeTensors.
        # We detect this via untyped_storage() and fall through to
        # aten.convolution_backward which decomposes into primitives that
        # Inductor can compile (mm, sum, etc.).
        #
        # At execution time (real tensors), the Slang fused backward kernel
        # computes dX, dW, dB in a single dispatch via bwd_diff(conv_inner_madd).
        def _has_real_vulkan_storage(t):
            try:
                return t.untyped_storage().device.type != "meta"
            except Exception:
                return True

        use_slang_bwd = (
            groups == 1
            and inp.device.type == "vulkan"
            and inp.dtype == torch.float32
            and _has_real_vulkan_storage(inp)
        )
        # M6.6: Per-group Slang backward for depthwise conv (groups>1).
        # aten.convolution_backward fails with FunctionalTensor mismatch
        # during AOT Autograd trace.  Decompose into per-group group-1
        # _slang_tile_conv2d_bwd calls, then concatenate results.
        use_per_group_bwd = (
            groups > 1
            and inp.device.type == "vulkan"
            and inp.dtype == torch.float32
            and _has_real_vulkan_storage(inp)
        )

        if use_slang_bwd or use_per_group_bwd:
            from torch_vulkan.inductor.templates.caller import (
                _slang_tile_conv2d_bwd,
            )

            if use_slang_bwd:
                # groups==1: single dispatch
                g_inp = torch.zeros_like(inp)
                g_w = torch.zeros_like(w)
                g_b = (
                    torch.zeros(int(w.shape[0]), device=w.device, dtype=w.dtype)
                    if has_bias
                    else None
                )
                _slang_tile_conv2d_bwd(
                    inp,
                    w,
                    grad_output,
                    g_inp,
                    g_w,
                    stride=tuple(ctx.stride),
                    padding=tuple(ctx.padding),
                    dilation=tuple(ctx.dilation),
                    bias=saved_b if has_bias else None,
                    grad_bias=g_b,
                )
            else:
                # M6.6: groups>1 — per-group decomposition
                C_in_g = w.shape[1]
                C_out_g = w.shape[0] // groups
                g_inp_parts = []
                g_w_parts = []
                g_b_val = (
                    torch.zeros(int(w.shape[0]), device=w.device, dtype=w.dtype)
                    if has_bias
                    else None
                )
                for g in range(groups):
                    inp_s = inp[:, g * C_in_g : (g + 1) * C_in_g, :, :]
                    w_s = w[g * C_out_g : (g + 1) * C_out_g, :, :, :]
                    go_s = grad_output[:, g * C_out_g : (g + 1) * C_out_g, :, :]
                    b_s = (
                        saved_b[g * C_out_g : (g + 1) * C_out_g]
                        if has_bias and saved_b is not None
                        else None
                    )
                    gi = torch.zeros_like(inp_s)
                    gw = torch.zeros_like(w_s)
                    gb_s = (
                        g_b_val[g * C_out_g : (g + 1) * C_out_g]
                        if g_b_val is not None
                        else None
                    )
                    _slang_tile_conv2d_bwd(
                        inp_s,
                        w_s,
                        go_s,
                        gi,
                        gw,
                        stride=tuple(ctx.stride),
                        padding=tuple(ctx.padding),
                        dilation=tuple(ctx.dilation),
                        bias=b_s,
                        grad_bias=gb_s,
                    )
                    g_inp_parts.append(gi)
                    g_w_parts.append(gw)
                g_inp = torch.cat(g_inp_parts, dim=1)
                g_w = torch.cat(g_w_parts, dim=0)
                g_b = g_b_val
        else:
            # groups>1 non-Vulkan/non-f32: fall through to aten
            result = torch.ops.aten.convolution_backward.default(
                grad_output,
                inp,
                w,
                None,
                ctx.stride,
                ctx.padding,
                ctx.dilation,
                False,
                [0] * len(ctx.stride),
                int(groups),
                [True, True, has_bias],
            )
            g_inp = result[0] if result[0] is not None else torch.empty_like(inp)
            g_w = result[1] if result[1] is not None else torch.zeros_like(w)
            g_b = (
                result[2]
                if len(result) > 2 and result[2] is not None and has_bias
                else (
                    torch.zeros(int(w.shape[0]), device=w.device, dtype=w.dtype)
                    if has_bias
                    else None
                )
            )
        return g_inp, g_w, g_b if has_bias else None, None, None, None, None

    conv_op.register_autograd(_conv2d_backward, setup_context=_conv2d_setup_context)
    return torch.ops.torch_vulkan.conv2d_with_optional_bias.default


def _ensure_conv1d_with_optional_bias_op_registered() -> "object":
    """Register ``torch_vulkan::conv1d_with_optional_bias`` (PF.30.a).

    Mirrors the conv2d shim. Eager backing implements conv1d via the
    standard unsqueeze/conv2d/squeeze trick after materializing bias —
    matches the pre-PF.30.a body of ``_patched_conv1d``.
    """
    import torch

    op_name = "torch_vulkan::conv1d_with_optional_bias"
    existing = getattr(torch.ops.torch_vulkan, "conv1d_with_optional_bias", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    Tensor = torch.Tensor

    def _conv1d_impl(
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        stride: list[int],
        padding: list[int],
        dilation: list[int],
        groups: int,
    ) -> Tensor:
        if weight.dtype != input.dtype:
            weight = weight.to(dtype=input.dtype)
        if bias is None:
            bias = torch.zeros(weight.shape[0], device=input.device, dtype=input.dtype)
        elif bias.dtype != input.dtype:
            bias = bias.to(dtype=input.dtype)
        s = stride[0] if len(stride) == 1 else stride[-1]
        p = padding[0] if len(padding) == 1 else padding[-1]
        d = dilation[0] if len(dilation) == 1 else dilation[-1]
        input_4d = input.unsqueeze(2)
        weight_4d = weight.unsqueeze(2)
        out = torch.ops.aten.convolution.default(
            input_4d,
            weight_4d,
            bias,
            [1, int(s)],
            [0, int(p)],
            [1, int(d)],
            False,
            [0, 0],
            int(groups),
        )
        return out.squeeze(2)

    _conv1d_impl.__annotations__ = {
        "input": Tensor,
        "weight": Tensor,
        "bias": Tensor | None,
        "stride": list[int],
        "padding": list[int],
        "dilation": list[int],
        "groups": int,
        "return": Tensor,
    }
    conv_op = torch.library.custom_op(op_name, mutates_args=())(_conv1d_impl)

    def _conv1d_fake(input, weight, bias, stride, padding, dilation, groups):
        N = input.shape[0]
        C_out = weight.shape[0]
        L_in = input.shape[-1]
        K = weight.shape[-1]
        s = stride[0] if len(stride) == 1 else stride[-1]
        p = padding[0] if len(padding) == 1 else padding[-1]
        d = dilation[0] if len(dilation) == 1 else dilation[-1]
        L_out = (L_in + 2 * p - d * (K - 1) - 1) // s + 1
        return input.new_empty((N, C_out, L_out))

    conv_op.register_fake(_conv1d_fake)
    return torch.ops.torch_vulkan.conv1d_with_optional_bias.default
