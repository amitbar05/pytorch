"""Conv2d / Conv1d custom-op registrations for fused FX-pattern targets (PF.30.a).

M22b: split from 1147-line monolith — this file keeps only the core conv2d
(forward + autograd) and conv1d ops. The other families live in:
  - conv_relu.py     — Conv2d+ReLU fused (M17.2)
  - conv_gn_relu.py  — Conv2d+GroupNorm+ReLU fused (M17.2 Phase 3)
  - conv_backward.py — Opaque non-autograd conv2d_backward (M17.8.d.2)

Re-exports from those modules are provided below so that any caller importing
directly from ``conv`` continues to work without modification.
"""

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
                # Bias gradient: skip Slang vk_atomic_add path (recycled Vulkan pool
                # buffers from torch.zeros(..., device='vulkan') are NOT zero-initialized,
                # so atomic accumulation into them produces wrong results). Compute via
                # a clean reduce-sum over spatial+batch dims instead.
                _slang_tile_conv2d_bwd(
                    inp,
                    w,
                    grad_output,
                    g_inp,
                    g_w,
                    stride=tuple(ctx.stride),
                    padding=tuple(ctx.padding),
                    dilation=tuple(ctx.dilation),
                    bias=None,
                    grad_bias=None,
                )
                g_b = grad_output.sum([0, 2, 3]) if has_bias else None
            else:
                # M6.6: groups>1 — per-group decomposition
                C_in_g = w.shape[1]
                C_out_g = w.shape[0] // groups
                g_inp_parts = []
                g_w_parts = []
                for g in range(groups):
                    inp_s = inp[:, g * C_in_g : (g + 1) * C_in_g, :, :]
                    w_s = w[g * C_out_g : (g + 1) * C_out_g, :, :, :]
                    go_s = grad_output[:, g * C_out_g : (g + 1) * C_out_g, :, :]
                    gi = torch.zeros_like(inp_s)
                    gw = torch.zeros_like(w_s)
                    _slang_tile_conv2d_bwd(
                        inp_s,
                        w_s,
                        go_s,
                        gi,
                        gw,
                        stride=tuple(ctx.stride),
                        padding=tuple(ctx.padding),
                        dilation=tuple(ctx.dilation),
                        bias=None,
                        grad_bias=None,
                    )
                    g_inp_parts.append(gi)
                    g_w_parts.append(gw)
                g_inp = torch.cat(g_inp_parts, dim=1)
                g_w = torch.cat(g_w_parts, dim=0)
                # Same bias fix as groups==1: clean reduction, not Slang atomic add
                g_b = grad_output.sum([0, 2, 3]) if has_bias else None
        else:
            # COMPILE.1: Route through torch_vulkan.conv2d_backward custom op.
            # This is an opaque FX node (make_fallback at lowerings/__init__.py:438)
            # that at runtime dispatches _slang_tile_conv2d_bwd in a single
            # Slang compute dispatch (no CPU roundtrip).
            _bwd_result = torch.ops.torch_vulkan.conv2d_backward.default(
                inp,
                grad_output,
                w,
                list(ctx.stride),
                list(ctx.padding),
                list(ctx.dilation),
                int(groups),
                has_bias,
            )
            g_inp = _bwd_result[0]
            g_w = _bwd_result[1]
            g_b = _bwd_result[2] if len(_bwd_result) > 2 else None
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

    def _conv1d_setup_ctx(ctx, inputs, output):
        input, weight, bias, stride, padding, dilation, groups = inputs
        ctx.save_for_backward(input, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.has_bias = bias is not None

    def _conv1d_backward(ctx, grad_output):
        # S3.5b: delegate to the new conv1d_backward_core opaque op.
        #
        # Old code: input_4d = input.unsqueeze(-1) + conv2d_backward + squeeze.
        # The view ``input_4d`` is an INTERMEDIATE in the AoT backward graph
        # (dead after conv2d_backward), so the Inductor memory planner reused
        # its buffer for conv2d_backward[0] (grad-input).  Since input_4d is a
        # view of primals_3 (primal x), grad-input aliased primals_3 and the
        # backward returned x itself as x.grad.
        #
        # conv1d_backward_core takes 3-D tensors directly → ``input`` is a
        # backward graph INPUT (never freed), eliminating the buffer-reuse
        # aliasing.  Its fake kernel derives outputs from ``input`` (a
        # FunctionalTensor proxy) so FunctionalTensorMode produces fresh FT
        # storage refs, preventing the conservative alias analysis that
        # concluded conv2d_backward[0] ≡ input_4d ≡ primals_3.
        from .conv_backward import _ensure_conv1d_backward_core_op_registered
        _ensure_conv1d_backward_core_op_registered()

        input, weight = ctx.saved_tensors
        s = ctx.stride[0] if ctx.stride else 1
        p = ctx.padding[0] if ctx.padding else 0
        d = ctx.dilation[0] if ctx.dilation else 1
        g_inp, g_w, g_b = torch.ops.torch_vulkan.conv1d_backward_core.default(
            input, grad_output, weight,
            [s], [p], [d],
            int(ctx.groups), ctx.has_bias,
        )
        gb = g_b if ctx.has_bias else None
        return g_inp, g_w, gb, None, None, None, None

    conv_op.register_autograd(_conv1d_backward, setup_context=_conv1d_setup_ctx)
    return torch.ops.torch_vulkan.conv1d_with_optional_bias.default


# ─── Re-exports for backward-compatible ``from .conv import …`` usage ────────
# Callers that import these names directly from ``conv`` continue to work
# without modification after the M22b split.

from .conv_relu import _ensure_conv2d_relu_fused_op_registered  # noqa: E402, F401
from .conv_gn_relu import (  # noqa: E402, F401
    _dispatch_group_norm_slang,
    _ensure_conv2d_gn_relu_fused_op_registered,
)
from .conv_backward import (  # noqa: E402, F401
    _ensure_conv1d_backward_core_op_registered,
    _ensure_conv2d_backward_op_registered,
)
