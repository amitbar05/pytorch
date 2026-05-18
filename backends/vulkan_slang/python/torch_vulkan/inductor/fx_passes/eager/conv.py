"""Conv2d / Conv1d custom-op registrations for fused FX-pattern targets (PF.30.a)."""

from __future__ import annotations

import os

# M-pipeline-1 (2026-05-18): per-module idempotency guard.  Each
# ``_ensure_*`` below early-returns when the corresponding op already
# exists in ``torch.ops.torch_vulkan``.  Without this, the lazy
# ``_ensure_patch_custom_ops`` shim inside ``_patched_conv2d`` re-runs
# ``torch.library.custom_op(...)`` on every compile, and Dynamo's skip
# of ``_library.infer_schema`` produces a graph break — fragmenting
# ``nn.Sequential(Conv, GN, ReLU)`` into 3 subgraphs and stranding the
# ``_fuse_conv_patched_gn_relu`` pass / the Vulkan-native conv lowering.
#
# Set ``TORCH_VULKAN_FORCE_CUSTOM_OP_RELOAD=1`` to force re-registration
# during development when iterating on the backward implementation
# (it overrides the early return).
_FORCE_RELOAD = os.environ.get("TORCH_VULKAN_FORCE_CUSTOM_OP_RELOAD") == "1"


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
    # M-pipeline-1: short-circuit when the op is already registered, so
    # that any re-entry from ``_ensure_patch_custom_ops`` during a
    # Dynamo trace does NOT re-invoke ``torch.library.custom_op`` (which
    # would call ``_library.infer_schema``, marked
    # ``@torch._dynamo.skip``, and produce a graph break).
    if existing is not None and hasattr(existing, "default") and not _FORCE_RELOAD:
        return existing.default

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

    # M17.8.d.2 / M18.2 (2026-05-17/18): @torch.compiler.disable removed
    # because it made AOTAutograd emit ``aten.full(shape, 0)`` for
    # grad_weight / grad_bias (i.e. literal zeros) — Dynamo treated the
    # backward as opaque and bailed out of differentiation. Without the
    # decorator, AOTAutograd traces through the function and the
    # FakeTensor-detection branch (else of _has_real_vulkan_storage)
    # produces the proper backward via aten.convolution_backward (which
    # decomposes into primitives Inductor can compile).
    def _conv2d_backward(ctx, grad_output):
        inp, w, saved_b = ctx.saved_tensors
        has_bias = saved_b is not None and saved_b.numel() > 0
        groups = int(ctx.groups)

        # CG.M6: Route through the [Differentiable]-based conv backward template
        # when groups==1, tensors are on Vulkan, f32, AND have real storage
        # (not FakeTensors during AOT Autograd tracing).
        #
        # During AOT Autograd's joint graph trace, all inputs are FakeTensors
        # or FunctionalTensors.  Shared helper handles all three wrappers
        # (FakeTensor / FunctionalTensor / torch.compiler.is_compiling()).
        #
        # At execution time (real tensors), the Slang fused backward kernel
        # computes dX, dW, dB in a single dispatch via bwd_diff(conv_inner_madd).
        from ._common import _has_real_vulkan_storage

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
            # M17.8.d.2 — IDEAL ROUTING (currently blocked):
            # We would prefer to route through the opaque
            # ``torch.ops.torch_vulkan.conv2d_backward.default`` custom op
            # here so AOTAutograd's joint trace lands a single FX node
            # rather than decomposing into ``empty_like`` sub-ops that the
            # partitioner collapses to literal zeros. The op IS registered
            # (see ``_ensure_conv2d_backward_op_registered``) but Inductor
            # has no lowering for it yet — ``make_fallback`` must be added
            # in ``lowerings/__init__.py`` (Matmul-3D territory; queued for
            # parent-agent integration after M19.1).
            #
            # Until then, fall through to ``aten.convolution_backward.default``.
            # The M18.3 fix to ``op_registration.py:_layer_norm_bwd_meta`` /
            # ``_linear_bwd_meta`` / etc. (now using ``new_empty(shape)``
            # rather than ``empty_like``) keeps this path producing
            # non-zero gradients during AOT trace.
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
            g_inp = (
                result[0]
                if result[0] is not None
                else inp.new_empty(inp.shape).zero_()
            )
            g_w = (
                result[1]
                if result[1] is not None
                else w.new_empty(w.shape).zero_()
            )
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

    conv_op.register_autograd(_conv2d_backward, setup_context=_conv2d_setup_context)
    return torch.ops.torch_vulkan.conv2d_with_optional_bias.default


# ═══════════════════════════════════════════════════════════════════════════
# M17.2 — conv2d+ReLU fused custom op
# ═══════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════
# M17.8.d.2 — opaque conv2d_backward custom op
# ═══════════════════════════════════════════════════════════════════════════


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
        """Eager impl: route fp32 Vulkan to ``aten.convolution_backward.default``
        directly (which hits the working C++ ``vulkan_convolution_backward_overrideable``
        adapter). For non-Vulkan/non-f32 we fall back to plain aten too.
        """
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
