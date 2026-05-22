"""Conv2d+GroupNorm+ReLU fused custom-op registration (M22b split from conv.py).

Contains:
  - ``_dispatch_group_norm_slang`` — direct GN shader dispatch helper
  - ``_ensure_conv2d_gn_relu_fused_op_registered`` — M17.2 Phase 3 fused
    conv2d+GroupNorm+ReLU custom op (one vkQueueSubmit, zero intermediates)
"""

from __future__ import annotations


def _dispatch_group_norm_slang(
    input_t: "torch.Tensor",
    weight_t: "torch.Tensor",
    bias_t: "torch.Tensor",
    out: "torch.Tensor",
    num_groups: int,
    eps: float = 1e-5,
    save_mean: "torch.Tensor | None" = None,
    save_rstd: "torch.Tensor | None" = None,
) -> None:
    """Dispatch the ``group_norm.slang`` fused shader directly.

    Input:  [N, C, H, W] (reshaped internally to [N*G, group_size])
    Weight: [C] (gamma)
    Bias:   [C] (beta)
    Output: [N, C, H, W]

    The shader computes per-(batch,group) mean/variance via shared-memory
    reduction, normalizes, and applies per-channel affine in a single
    dispatch.  Optionally writes per-row mean/rstd for backward.
    """
    import os
    import struct

    import torch

    N, C, H, W = input_t.shape
    G = num_groups
    channels_per_group = C // G
    spatial_size = H * W
    group_size = channels_per_group * spatial_size
    num_rows = N * G

    # Reshape to [N*G, group_size] for the 2D shader
    inp_2d = input_t.reshape(N * G, group_size)
    w_1d = weight_t.view(-1)
    b_1d = bias_t.view(-1)
    out_2d = out.reshape(N * G, group_size)

    # Allocate mean/rstd buffers if not provided
    if save_mean is None:
        save_mean = torch.empty(num_rows, dtype=torch.float32, device=input_t.device)
    if save_rstd is None:
        save_rstd = torch.empty(num_rows, dtype=torch.float32, device=input_t.device)

    # Read the shader source
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    shader_path = os.path.join(
        _this_dir,
        "..",
        "..",
        "..",
        "..",
        "..",
        "shaders",
        "normalization",
        "group_norm.slang",
    )
    with open(shader_path) as f:
        src = f.read()

    from torch_vulkan.inductor.runtime import compile_and_dispatch

    # Pack push constants: 5 uint + 1 float (epsilon)
    pc = struct.pack(
        "5If",
        num_groups,
        group_size,
        num_rows,
        channels_per_group,
        spatial_size,
        float(eps),
    )

    buffers = [inp_2d, w_1d, b_1d, out_2d, save_mean, save_rstd]

    cache_key = f"group_norm_fused_{G}_{channels_per_group}_{spatial_size}_f32_m17"

    compile_and_dispatch(
        src,
        buffers,
        num_rows,  # grid_x = one workgroup per row
        1,
        1,
        push_constants=pc,
        num_outputs=1,
        entry="computeMain",
        cache_key=cache_key,
    )


def _ensure_conv2d_gn_relu_fused_op_registered() -> "object":
    """Register ``torch_vulkan::conv2d_gn_relu_fused`` as a custom_op.

    M17.2 Phase 3 — Fused conv2d + GroupNorm + ReLU in a single
    Vulkan dispatch (one vkQueueSubmit, zero intermediate buffers).

    The eager backing calls ``_slang_tile_conv2d_gn_relu`` which
      dispatches the combined ``conv_gn_relu.slang`` shader: conv compute,
      Welford reduction, normalization, affine, and store — all in one
    kernel.

    Backward chains GN backward + Conv backward using aten ops
    (delegates to eager Vulkan implementations under the hood).
    """
    import torch

    op_name = "torch_vulkan::conv2d_gn_relu_fused"
    existing = getattr(torch.ops.torch_vulkan, "conv2d_gn_relu_fused", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    Tensor = torch.Tensor

    def _conv2d_gn_relu_impl(
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        stride: list[int],
        padding: list[int],
        dilation: list[int],
        groups: int,
        gn_weight: Tensor,
        gn_bias: Tensor,
        num_groups: int,
        eps: float,
    ) -> Tensor:
        if input.device.type != "vulkan" or input.dtype != torch.float32:
            # M18.8.b (2026-05-18): fix forward op order. The FX matcher
            # ``_fuse_conv_gn_relu`` (decomposition_passes.py:557) replaces
            # the chain ``relu(group_norm(conv(...)))`` — i.e. produced by
            # ``nn.Sequential(Conv, GN, ReLU)`` — with this custom op. The
            # fused op must therefore compute ``conv → GN → ReLU``, NOT the
            # ``conv → ReLU → GN`` ordering that was here before (which
            # produced a different function and silently miscomputed
            # gradients with magnitude ~5.6× the CPU baseline).
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
            y = torch.nn.functional.group_norm(
                y, num_groups, gn_weight, gn_bias, eps
            )
            return torch.relu(y)

        from torch_vulkan.inductor.templates.caller import (
            _slang_tile_conv2d_gn_relu,
        )

        N = input.shape[0]
        C_out = weight.shape[0]
        sH, sW = stride[0], stride[-1]
        pH, pW = padding[0], padding[-1]
        dH, dW = dilation[0], dilation[-1]
        kH, kW = weight.shape[2], weight.shape[3]
        iH, iW = input.shape[2], input.shape[3]
        H_out = (iH + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (iW + 2 * pW - dW * (kW - 1) - 1) // sW + 1

        # M17.2 Phase 3: single-dispatch Conv2D + GroupNorm + ReLU.
        # The combined shader does conv compute, Welford reduction,
        # normalization, affine, and store in one vkQueueSubmit.
        gn_out = torch.empty(
            (N, C_out, H_out, W_out),
            device=input.device,
            dtype=input.dtype,
        )
        _slang_tile_conv2d_gn_relu(
            input.contiguous() if not input.is_contiguous() else input,
            weight.contiguous() if not weight.is_contiguous() else weight,
            bias,
            gn_weight,
            gn_bias,
            gn_out,
            stride=(sH, sW),
            padding=(pH, pW),
            dilation=(dH, dW),
            num_groups=num_groups,
            eps=eps,
        )
        return gn_out

    _conv2d_gn_relu_impl.__annotations__ = {
        "input": Tensor,
        "weight": Tensor,
        "bias": Tensor | None,
        "stride": list[int],
        "padding": list[int],
        "dilation": list[int],
        "groups": int,
        "gn_weight": Tensor,
        "gn_bias": Tensor,
        "num_groups": int,
        "eps": float,
        "return": Tensor,
    }
    gn_relu_op = torch.library.custom_op(op_name, mutates_args=())(_conv2d_gn_relu_impl)

    def _conv2d_gn_relu_fake(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        gn_weight,
        gn_bias,
        num_groups,
        eps,
    ):
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

    gn_relu_op.register_fake(_conv2d_gn_relu_fake)

    # Backward: chain GN backward + Conv backward.
    # We recompute the pre-GN tensor (conv+ReLU output) during backward
    # rather than saving it, trading extra compute for reduced memory.
    def _conv2d_gn_relu_setup_context(ctx, inputs, output):
        inp, w, b, stride, padding, dilation, groups, gn_w, gn_b, num_g, eps = inputs
        ctx.save_for_backward(inp, w, b if b is not None else None, gn_w, gn_b)
        ctx.stride = list(stride)
        ctx.padding = list(padding)
        ctx.dilation = list(dilation)
        ctx.groups = int(groups)
        ctx.num_groups = int(num_g) if num_g is not None else 1
        ctx.eps = float(eps) if eps is not None else 1e-5

    # M18.2 (2026-05-18): @torch.compiler.disable removed and the local
    # _has_real_vulkan_storage replaced with the shared M17.8.d.2-fixed
    # helper — see _conv2d_relu_backward / _conv2d_backward.
    #
    # M18.8.b (2026-05-18): backward op order fixed to match the corrected
    # forward (``conv → GN → ReLU``). Previously this assumed
    # ``conv → ReLU → GN`` — ``pre_gn = relu(conv(inp))`` was fed to GN
    # backward, and the ReLU backward mask was missing entirely. The result
    # was conv weight grads ~5.6× larger than the CPU baseline (because the
    # GN backward output was treated as ``grad(conv_out)`` rather than
    # ``grad(GN_out)``, skipping the cross-correlation reduction inside GN
    # backward and the ReLU mask).
    def _conv2d_gn_relu_backward(ctx, grad_output):
        inp, w, saved_b, gn_w, gn_b = ctx.saved_tensors
        has_bias = saved_b is not None and saved_b.numel() > 0

        # M-pipeline-1 (2026-05-18): N/C/HxW must describe the GN INPUT —
        # i.e. the conv OUTPUT — not the conv INPUT.  Previously this used
        # inp.shape[1] for C and inp.shape[2:] for HxW; once the M-pipeline-1
        # fix removed the Dynamo graph break, this codepath actually runs
        # and ``aten.native_group_norm`` chokes with
        # ``C_in (=3) % num_groups != 0`` for the SmallCNN test
        # (Conv2d(3, 8) → GroupNorm(2, 8)).  Compute the output spatial
        # dims here so both the GN call below and the recomputed conv_out
        # are sized correctly.
        sH, sW = ctx.stride[0], ctx.stride[-1]
        pH, pW = ctx.padding[0], ctx.padding[-1]
        dH, dW = ctx.dilation[0], ctx.dilation[-1]
        kH, kW = w.shape[2], w.shape[3]
        H_in, W_in = inp.shape[2], inp.shape[3]
        H_out = (H_in + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W_in + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        N = inp.shape[0]
        C = int(w.shape[0])  # conv-output channels (= GN input channels)
        G = ctx.num_groups
        HxW = H_out * W_out

        from ._common import _has_real_vulkan_storage

        use_slang = (
            ctx.groups == 1
            and inp.device.type == "vulkan"
            and inp.dtype == torch.float32
            and _has_real_vulkan_storage(inp)
        )

        # Recompute conv output for GN backward.  Forward is
        # ``conv → GN → ReLU``, so the GN input is the raw conv output
        # (no ReLU applied). We also recompute the GN output (used for
        # the ReLU mask).
        if use_slang:
            from torch_vulkan.inductor.templates.caller import _slang_tile_conv2d

            conv_out = torch.empty(
                (N, C, H_out, W_out),
                device=inp.device,
                dtype=inp.dtype,
            )
            _slang_tile_conv2d(
                inp.contiguous() if not inp.is_contiguous() else inp,
                w.contiguous() if not w.is_contiguous() else w,
                conv_out,
                stride=(sH, sW),
                padding=(pH, pW),
                dilation=(dH, dW),
                groups=1,
                bias=saved_b if has_bias else None,
                epilogue="OpIdent",
            )
        else:
            conv_out = torch.ops.aten.convolution.default(
                inp,
                w,
                saved_b
                if has_bias
                else torch.zeros(w.shape[0], device=inp.device, dtype=inp.dtype),
                ctx.stride,
                ctx.padding,
                ctx.dilation,
                False,
                [0, 0],
                int(ctx.groups),
            )

        # GN output (= ReLU input). Needed for ReLU mask AND to capture
        # the saved mean/rstd for the GN backward (passing ``None`` makes
        # aten error out; the backward needs the exact forward statistics).
        # Use the explicit aten call (not F.group_norm) because the proxy
        # tracer can mis-classify F.group_norm's saved-tensor args as
        # shape-only and drop ``gn_w`` from the joint graph.
        gn_out_tuple = torch.ops.aten.native_group_norm.default(
            conv_out, gn_w, gn_b, N, C, HxW, G, ctx.eps
        )
        gn_out = gn_out_tuple[0]
        gn_save_mean = gn_out_tuple[1]
        gn_save_rstd = gn_out_tuple[2]

        # ReLU backward: zero-out the gradient where the GN output (= ReLU
        # input) was <= 0.  This converts ``grad(ReLU_out)`` (the incoming
        # ``grad_output``) into ``grad(ReLU_in) = grad(GN_out)``.
        relu_mask = (gn_out > 0).to(dtype=grad_output.dtype)
        grad_gn_out = grad_output * relu_mask

        # GN backward via aten.  Input is the conv output (raw, no ReLU),
        # output is grad w.r.t. conv output (= GN input).
        gn_grad_input, gn_grad_w, gn_grad_b = (
            torch.ops.aten.native_group_norm_backward.default(
                grad_gn_out,
                conv_out,
                gn_save_mean,
                gn_save_rstd,
                gn_w,
                N,
                C,
                HxW,
                G,
                [True, True, True],
            )
        )

        # Conv backward
        if use_slang:
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
                gn_grad_input,
                g_inp,
                g_w,
                stride=tuple(ctx.stride),
                padding=tuple(ctx.padding),
                dilation=tuple(ctx.dilation),
                bias=saved_b if has_bias else None,
                grad_bias=g_b,
            )
            conv_grads = (g_inp, g_w, g_b if has_bias else None)
        else:
            result = torch.ops.aten.convolution_backward.default(
                gn_grad_input,
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
            conv_grads = (
                result[0] if result[0] is not None else inp.new_empty(inp.shape),
                result[1] if result[1] is not None else w.new_empty(w.shape).zero_(),
                result[2]
                if len(result) > 2 and result[2] is not None and has_bias
                else (
                    w.new_empty((int(w.shape[0]),)).zero_()
                    if has_bias
                    else None
                ),
            )

        return (
            conv_grads[0],  # grad_input
            conv_grads[1],  # grad_weight
            conv_grads[2] if has_bias else None,  # grad_bias
            None,  # stride
            None,  # padding
            None,  # dilation
            None,  # groups
            gn_grad_w,  # grad_gn_weight
            gn_grad_b,  # grad_gn_bias
            None,  # num_groups
            None,  # eps
        )

    gn_relu_op.register_autograd(
        _conv2d_gn_relu_backward, setup_context=_conv2d_gn_relu_setup_context
    )
    return torch.ops.torch_vulkan.conv2d_gn_relu_fused.default
