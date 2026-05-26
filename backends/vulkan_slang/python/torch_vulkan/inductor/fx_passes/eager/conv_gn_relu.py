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
    # setup_context computes and saves conv_out + GN statistics so the backward
    # can use them directly — no recomputation inside the compiled backward graph.
    # Recomputing inside the backward caused Inductor buffer-planner aliasing:
    # intermediates produced during the native_group_norm lowering (same shape
    # [N,C,H,W] as conv_out) were assigned to conv_out_bwd's buffer, corrupting
    # the GN backward's xhat = (conv_out - mean) * rstd computation and producing
    # conv/gn gradient errors of 0.07–0.21 (vs ~1e-7 for the unfused path).
    def _conv2d_gn_relu_setup_context(ctx, inputs, output):
        inp, w, b, stride, padding, dilation, groups, gn_w, gn_b, num_g, eps = inputs
        ctx.stride = list(stride)
        ctx.padding = list(padding)
        ctx.dilation = list(dilation)
        ctx.groups = int(groups)
        ctx.num_groups = int(num_g) if num_g is not None else 1
        ctx.eps = float(eps) if eps is not None else 1e-5
        has_bias = b is not None
        ctx.has_bias = has_bias

        # Compute conv_out and GN stats here (forward time, not inside the
        # compiled backward) so they are saved as proper save_for_backward
        # tensors.  As inputs to the backward graph they cannot be aliased
        # with backward intermediates, eliminating the Inductor buffer-planner
        # aliasing that caused 0.07–0.21 gradient errors in the old approach.
        #
        # IMPORTANT: detach inp/w/gn_w/gn_b before the setup_context ops.
        # Without detach, AOTAutograd traces backward through BOTH the custom
        # backward AND this convolution (using the same w), doubling
        # conv.weight.grad (~2× factor).  Detach breaks that extra gradient
        # path while leaving the numerical values of conv_out/mean/rstd intact.
        _b_d = (b.detach() if has_bias else torch.zeros(
            w.shape[0], device=inp.device, dtype=inp.dtype
        ))
        conv_out = torch.ops.aten.convolution.default(
            inp.detach(), w.detach(), _b_d,
            ctx.stride, ctx.padding, ctx.dilation,
            False, [0] * len(ctx.stride), ctx.groups,
        )
        N = inp.shape[0]
        C = int(w.shape[0])
        HxW = int(conv_out.shape[2]) * int(conv_out.shape[3])
        _, gn_save_mean, gn_save_rstd = torch.ops.aten.native_group_norm.default(
            conv_out, gn_w.detach(), gn_b.detach(), N, C, HxW, ctx.num_groups, ctx.eps
        )
        ctx.save_for_backward(
            inp, w, b if has_bias else None,   # NON-detached: needed by custom backward
            gn_w, gn_b, output,                # output = relu output (for mask)
            conv_out, gn_save_mean, gn_save_rstd,  # detached: no extra grad paths
        )

    from .conv_backward import _ensure_conv2d_backward_op_registered

    _ensure_conv2d_backward_op_registered()

    def _conv2d_gn_relu_backward(ctx, grad_output):
        (inp, w, saved_b, gn_w, gn_b, saved_output,
         conv_out, gn_save_mean, gn_save_rstd) = ctx.saved_tensors
        has_bias = ctx.has_bias

        N = inp.shape[0]
        C = int(w.shape[0])
        G = ctx.num_groups
        HxW = int(conv_out.shape[2]) * int(conv_out.shape[3])

        # ReLU backward: use the exact forward relu output for the mask.
        relu_mask = (saved_output > 0).to(dtype=grad_output.dtype)
        grad_gn_out = grad_output * relu_mask

        # GN backward: conv_out / mean / rstd are saved tensors (backward graph
        # inputs), so the memory planner cannot alias them with any intermediate.
        gn_grad_input, gn_grad_w, gn_grad_b = (
            torch.ops.aten.native_group_norm_backward.default(
                grad_gn_out, conv_out, gn_save_mean, gn_save_rstd,
                gn_w, N, C, HxW, G, [True, True, True],
            )
        )

        # Conv backward via opaque custom op (avoids M17.8.d.2 partitioner drop).
        g_inp, g_w, g_b_raw = torch.ops.torch_vulkan.conv2d_backward.default(
            inp, gn_grad_input, w,
            list(ctx.stride), list(ctx.padding), list(ctx.dilation),
            int(ctx.groups), has_bias,
        )

        return (
            g_inp, g_w,
            g_b_raw if has_bias else None,
            None, None, None, None,  # stride, padding, dilation, groups
            gn_grad_w, gn_grad_b,
            None, None,  # num_groups, eps
        )

    gn_relu_op.register_autograd(
        _conv2d_gn_relu_backward, setup_context=_conv2d_gn_relu_setup_context
    )
    return torch.ops.torch_vulkan.conv2d_gn_relu_fused.default
