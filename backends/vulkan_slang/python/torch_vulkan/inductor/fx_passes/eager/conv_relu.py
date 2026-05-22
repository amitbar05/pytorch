"""Conv2d+ReLU fused custom-op registration (M22b split from conv.py).

Contains ``_ensure_conv2d_relu_fused_op_registered`` — the M17.2 fused
conv2d+ReLU custom op that dispatches the conv and activation in a single
Slang kernel dispatch.
"""

from __future__ import annotations

import os

_FORCE_RELOAD = os.environ.get("TORCH_VULKAN_FORCE_CUSTOM_OP_RELOAD") == "1"


def _ensure_conv2d_relu_fused_op_registered() -> "object":
    """Register ``torch_vulkan::conv2d_relu_fused`` as a custom_op.

    M17.2 — Fused conv2d+ReLU in a single Slang dispatch.
    The eager backing calls ``_slang_tile_conv2d`` with ``epilogue="OpReLU"``,
    applying the activation at the shader store site instead of in a
    separate pointwise kernel.
    """
    import torch

    op_name = "torch_vulkan::conv2d_relu_fused"
    existing = getattr(torch.ops.torch_vulkan, "conv2d_relu_fused", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    Tensor = torch.Tensor

    def _conv2d_relu_impl(
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        stride: list[int],
        padding: list[int],
        dilation: list[int],
        groups: int,
    ) -> Tensor:
        if input.device.type != "vulkan" or input.dtype != torch.float32:
            y = torch.ops.aten.convolution.default(
                input,
                weight.to(dtype=input.dtype),
                bias
                if bias is not None
                else torch.zeros(
                    weight.shape[0], device=input.device, dtype=input.dtype
                ),
                list(stride),
                list(padding),
                list(dilation),
                False,
                [0, 0],
                int(groups),
            )
            return torch.relu(y)

        from torch_vulkan.inductor.templates.caller import _slang_tile_conv2d

        if groups != 1:
            C_in_g = input.shape[1] // groups
            C_out_g = weight.shape[0] // groups
            parts = []
            for g in range(groups):
                inp_s = input[:, g * C_in_g : (g + 1) * C_in_g, :, :].contiguous()
                w_s = weight[g * C_out_g : (g + 1) * C_out_g, :, :, :].contiguous()
                b_s = (
                    bias[g * C_out_g : (g + 1) * C_out_g] if bias is not None else None
                )
                N_s = inp_s.shape[0]
                C_out_s = w_s.shape[0]
                H_in_s, W_in_s = inp_s.shape[2], inp_s.shape[3]
                sH, sW = stride[0], stride[-1]
                pH, pW = padding[0], padding[-1]
                dH, dW = dilation[0], dilation[-1]
                kH, kW = weight.shape[2], weight.shape[3]
                H_out_s = (H_in_s + 2 * pH - dH * (kH - 1) - 1) // sH + 1
                W_out_s = (W_in_s + 2 * pW - dW * (kW - 1) - 1) // sW + 1
                out_s = torch.empty(
                    (N_s, C_out_s, H_out_s, W_out_s),
                    device=input.device,
                    dtype=input.dtype,
                )
                _slang_tile_conv2d(
                    inp_s,
                    w_s,
                    out_s,
                    stride=(sH, sW),
                    padding=(pH, pW),
                    dilation=(dH, dW),
                    groups=1,
                    bias=b_s,
                    epilogue="OpReLU",
                )
                parts.append(out_s)
            return torch.cat(parts, dim=1)

        N = input.shape[0]
        C_out = weight.shape[0]
        H_in, W_in = input.shape[2], input.shape[3]
        sH, sW = stride[0], stride[-1]
        pH, pW = padding[0], padding[-1]
        dH, dW = dilation[0], dilation[-1]
        kH, kW = weight.shape[2], weight.shape[3]
        H_out = (H_in + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W_in + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        out = torch.empty(
            (N, C_out, H_out, W_out),
            device=input.device,
            dtype=input.dtype,
        )
        _slang_tile_conv2d(
            input.contiguous() if not input.is_contiguous() else input,
            weight.contiguous() if not weight.is_contiguous() else weight,
            out,
            stride=(sH, sW),
            padding=(pH, pW),
            dilation=(dH, dW),
            groups=1,
            bias=bias,
            epilogue="OpReLU",
        )
        return out

    _conv2d_relu_impl.__annotations__ = {
        "input": Tensor,
        "weight": Tensor,
        "bias": Tensor | None,
        "stride": list[int],
        "padding": list[int],
        "dilation": list[int],
        "groups": int,
        "return": Tensor,
    }
    relu_op = torch.library.custom_op(op_name, mutates_args=())(_conv2d_relu_impl)

    def _conv2d_relu_fake(input, weight, bias, stride, padding, dilation, groups):
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

    relu_op.register_fake(_conv2d_relu_fake)

    # M17.2: Register autograd — uses the same Slang conv backward.
    # The ReLU backward mask is applied to grad_output before the conv
    # backward dispatch.
    def _conv2d_relu_setup_context(ctx, inputs, output):
        inp, w, b, stride, padding, dilation, groups = inputs
        ctx.save_for_backward(inp, w, b if b is not None else None, output)
        ctx.stride = list(stride)
        ctx.padding = list(padding)
        ctx.dilation = list(dilation)
        ctx.groups = int(groups)

    # M18.2 (2026-05-18): @torch.compiler.disable removed and the local
    # _has_real_vulkan_storage replaced with the shared M17.8.d.2-fixed
    # helper.  Same bug class as _conv2d_backward: the old
    # storage().device check returned True for FunctionalTensor wrappers
    # during AOTAutograd trace, so the joint graph saw only
    # torch.zeros_like(...) and collapsed the backward partition to
    # literal zeros (CPU=30.59 vs VK=12.67 on a tiny Conv+ReLU train
    # step, per Agent 1 audit).
    def _conv2d_relu_backward(ctx, grad_output):
        inp, w, saved_b, output = ctx.saved_tensors
        has_bias = saved_b is not None and saved_b.numel() > 0

        # ReLU backward: zero-out gradients where forward output <= 0.
        relu_mask = (output > 0).to(dtype=grad_output.dtype)
        grad_output = grad_output * relu_mask

        from ._common import _has_real_vulkan_storage

        use_slang_bwd = (
            ctx.groups == 1
            and inp.device.type == "vulkan"
            and inp.dtype == torch.float32
            and _has_real_vulkan_storage(inp)
        )
        if use_slang_bwd:
            from torch_vulkan.inductor.templates.caller import _slang_tile_conv2d_bwd

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
            return g_inp, g_w, g_b if has_bias else None, None, None, None, None

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
            int(ctx.groups),
            [True, True, has_bias],
        )
        # M18.3 (2026-05-18): safety fallbacks use new_empty(shape) so the
        # proxy tracer treats these as storage-bound rather than shape-only.
        g_inp = result[0] if result[0] is not None else inp.new_empty(inp.shape)
        g_w = result[1] if result[1] is not None else w.new_empty(w.shape).zero_()
        g_b = (
            result[2]
            if len(result) > 2 and result[2] is not None and has_bias
            else (
                w.new_empty((int(w.shape[0]),)).zero_()
                if has_bias
                else None
            )
        )
        return g_inp, g_w, g_b if has_bias else None, None, None, None, None

    relu_op.register_autograd(
        _conv2d_relu_backward, setup_context=_conv2d_relu_setup_context
    )
    return torch.ops.torch_vulkan.conv2d_relu_fused.default
