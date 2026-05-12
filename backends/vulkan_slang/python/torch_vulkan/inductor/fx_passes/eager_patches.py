"""Eager custom-op registrations for fused FX-pattern targets.

Each ``_ensure_*`` function idempotently registers a ``torch_vulkan::*``
custom_op so Inductor has a valid OpOverload target to replace matched
subgraphs with.
"""

from __future__ import annotations


def _ensure_flash_attention_op_registered() -> "object":
    """Register `torch_vulkan::flash_attention_fused` custom_op exactly once."""
    import torch

    op_name = "torch_vulkan::flash_attention_fused"
    existing = getattr(torch.ops.torch_vulkan, "flash_attention_fused", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    Tensor = torch.Tensor

    def _flash_impl(
        q: Tensor, k: Tensor, v: Tensor, scale: float, is_causal: bool
    ) -> Tensor:
        import torch_vulkan

        return torch_vulkan.flash_attention(q, k, v, scale, is_causal)

    _flash_impl.__annotations__ = {
        "q": Tensor,
        "k": Tensor,
        "v": Tensor,
        "scale": float,
        "is_causal": bool,
        "return": Tensor,
    }
    flash_op = torch.library.custom_op(op_name, mutates_args=())(_flash_impl)

    def _flash_fake(q, k, v, scale, is_causal):
        # q: (B, H, M, D), v: (B, H, N, D) → (B, H, M, D)
        return q.new_empty(q.shape)

    flash_op.register_fake(_flash_fake)
    return torch.ops.torch_vulkan.flash_attention_fused.default


def _ensure_swiglu_op_registered() -> "object":
    """Register `torch_vulkan::swiglu_fused` as a torch custom_op exactly once.

    Mirrors the scaled_bmm registration: the FX rewrite target must be an
    OpOverload, and we wrap the eager `torch_vulkan.swiglu` extern with a
    fake_impl so Inductor's shape inference accepts the rewritten graph.
    """
    import torch

    op_name = "torch_vulkan::swiglu_fused"
    existing = getattr(torch.ops.torch_vulkan, "swiglu_fused", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    Tensor = torch.Tensor

    def _swiglu_impl(gate: Tensor, up: Tensor) -> Tensor:
        import torch_vulkan

        return torch_vulkan.swiglu(gate, up)

    _swiglu_impl.__annotations__ = {"gate": Tensor, "up": Tensor, "return": Tensor}
    swiglu_op = torch.library.custom_op(op_name, mutates_args=())(_swiglu_impl)

    def _swiglu_fake(gate, up):
        return gate.new_empty(gate.shape)

    swiglu_op.register_fake(_swiglu_fake)
    return torch.ops.torch_vulkan.swiglu_fused.default


def _ensure_addmm_gelu_op_registered() -> "object":
    """Register `torch_vulkan::addmm_gelu_fused` (PF.5).

    Pattern target for `_fuse_addmm_gelu`. Eager backing dispatches the
    fused tiled-addmm+gelu Slang shader (single GPU dispatch for
    `gelu(a @ b + bias)`). Falls back to two ops on shape/dtype mismatch.
    """
    import torch

    op_name = "torch_vulkan::addmm_gelu_fused"
    existing = getattr(torch.ops.torch_vulkan, "addmm_gelu_fused", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    Tensor = torch.Tensor

    def _addmm_gelu_impl(bias: Tensor, mat1: Tensor, mat2: Tensor) -> Tensor:
        from .vulkan_template_caller import _pick_addmm_gelu_tile

        if (
            mat1.device.type != "vulkan"
            or mat1.dtype not in (torch.float32, torch.float16)
            or mat1.dim() != 2
            or mat2.dim() != 2
            or bias.dim() != 1
        ):
            return torch.nn.functional.gelu(torch.addmm(bias, mat1, mat2))

        caller = _pick_addmm_gelu_tile(mat1.shape[0], mat2.shape[1], mat1.shape[1])
        return caller(bias, mat1, mat2)

    _addmm_gelu_impl.__annotations__ = {
        "bias": Tensor,
        "mat1": Tensor,
        "mat2": Tensor,
        "return": Tensor,
    }
    fused_op = torch.library.custom_op(op_name, mutates_args=())(_addmm_gelu_impl)

    def _addmm_gelu_fake(bias, mat1, mat2):
        return mat1.new_empty((mat1.shape[0], mat2.shape[1]))

    fused_op.register_fake(_addmm_gelu_fake)
    return torch.ops.torch_vulkan.addmm_gelu_fused.default


def _ensure_scaled_bmm_op_registered() -> "object":
    """Register `torch_vulkan::scaled_bmm` as a torch custom_op exactly once.

    The FX pass needs an OpOverload as the rewrite target — Inductor's
    lowering machinery rejects plain Python functions. Wrapping the eager
    dispatch in a custom_op (with a fake_impl for shape inference) makes
    the post-rewrite graph compileable end-to-end. Returns the OpOverload.
    """
    import torch

    op_name = "torch_vulkan::scaled_bmm"
    existing = getattr(torch.ops.torch_vulkan, "scaled_bmm", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    Tensor = torch.Tensor

    def _scaled_bmm_impl(q: Tensor, k: Tensor, scale: float) -> Tensor:
        import torch_vulkan

        return torch_vulkan.scaled_bmm(q, k, scale)

    _scaled_bmm_impl.__annotations__ = {
        "q": Tensor,
        "k": Tensor,
        "scale": float,
        "return": Tensor,
    }
    scaled_bmm = torch.library.custom_op(op_name, mutates_args=())(_scaled_bmm_impl)

    def _scaled_bmm_fake(q, k, scale):
        # bmm(q, k.T): q is (B, M, K); k is (B, N, K) → output (B, M, N).
        b, m, _ = q.shape
        _, n, _ = k.shape
        return q.new_empty((b, m, n))

    scaled_bmm.register_fake(_scaled_bmm_fake)

    return torch.ops.torch_vulkan.scaled_bmm.default


def _ensure_qkv_cat_op_registered() -> "object":
    """Register ``torch_vulkan::qkv_cat3`` — concatenate 3 tensors along a dim
    via the eager Vulkan dispatch. We need this as an extern OpOverload so the
    QKV FX pass can inject a weight-pack node without going through Inductor's
    cat IR lowering, which crashes on Vulkan (masked-op `dtype=str` bug).
    """
    import torch

    op_name = "torch_vulkan::qkv_cat3"
    existing = getattr(torch.ops.torch_vulkan, "qkv_cat3", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    Tensor = torch.Tensor

    def _cat3_impl(a: Tensor, b: Tensor, c: Tensor, dim: int) -> Tensor:
        return torch.cat([a, b, c], dim=dim)

    _cat3_impl.__annotations__ = {
        "a": Tensor,
        "b": Tensor,
        "c": Tensor,
        "dim": int,
        "return": Tensor,
    }
    cat3_op = torch.library.custom_op(op_name, mutates_args=())(_cat3_impl)

    def _cat3_fake(a, b, c, dim):
        out_shape = list(a.shape)
        out_shape[dim] = a.shape[dim] + b.shape[dim] + c.shape[dim]
        return a.new_empty(out_shape)

    cat3_op.register_fake(_cat3_fake)
    return torch.ops.torch_vulkan.qkv_cat3.default


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
                return True  # real Vulkan tensor (untyped_storage raises)
        use_slang_bwd = (
            groups == 1
            and inp.device.type == "vulkan"
            and inp.dtype == torch.float32
            and _has_real_vulkan_storage(inp)
        )
        if use_slang_bwd:
            from torch_vulkan.inductor.vulkan_template_caller import (
                _slang_tile_conv2d_bwd,
            )

            g_inp = torch.empty_like(inp)
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
            # During AOT Autograd tracing (FakeTensors): route through
            # a custom op so inductor can lower it to the fused Slang
            # backward kernel.  This avoids decomposing into primitives
            # (aten.mm, aten.sum, etc.) which lose the fusion opportunity.
                    g_inp, g_w, g_b = torch.ops.torch_vulkan.conv2d_backward.default(
                grad_output,
                inp,
                w,
                saved_b if has_bias else None,
                ctx.stride,
                ctx.padding,
                ctx.dilation,
                int(groups),
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


def _ensure_sdpa_with_optional_mask_op_registered() -> "object":
    """Register ``torch_vulkan::sdpa_with_optional_mask`` (PF.30.b).

    Replaces the ``@torch.compiler.disable`` on ``_patched_sdpa``. The
    fake_impl returns ``query.new_empty(query.shape)`` so AOT never
    invokes the C++ ``vulkan_sdpa`` kernel during tracing — the kernel
    has no null-storage MetaGuard for ``attn_mask``.

    The eager backing forwards to the direct pybind ``_c_ext._sdpa``
    binding, which already handles ``attn_mask=None`` correctly.
    """
    import torch

    op_name = "torch_vulkan::sdpa_with_optional_mask"
    existing = getattr(torch.ops.torch_vulkan, "sdpa_with_optional_mask", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    Tensor = torch.Tensor

    def _sdpa_impl(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None,
        dropout_p: float,
        is_causal: bool,
        scale: float | None,
    ) -> Tensor:
        import torch_vulkan

        torch_vulkan._ensure_loaded()
        return torch_vulkan._c_ext._sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=float(dropout_p),
            is_causal=bool(is_causal),
            scale=scale,
        )

    _sdpa_impl.__annotations__ = {
        "query": Tensor,
        "key": Tensor,
        "value": Tensor,
        "attn_mask": Tensor | None,
        "dropout_p": float,
        "is_causal": bool,
        "scale": float | None,
        "return": Tensor,
    }
    sdpa_op = torch.library.custom_op(op_name, mutates_args=())(_sdpa_impl)

    def _sdpa_fake(query, key, value, attn_mask, dropout_p, is_causal, scale):
        return query.new_empty(query.shape)

    sdpa_op.register_fake(_sdpa_fake)

    # PF.23 — autograd formula via path (a): math decomp into mm +
    # softmax_backward + mm. AOT autograd captures the chain as primitives,
    # Inductor's existing reduction lowerings own the inner work. dropout_p
    # is unsupported in the bwd formula (no RNG-replay state in the shim's
    # forward output); the eager path silently accepts dropout_p>0 but the
    # gradient would be incorrect, so we guard it here. attn_mask gradient
    # is None (autograd-side: the mask is treated as a constant).
    def _sdpa_setup_context(ctx, inputs, output):
        query, key, value, attn_mask, dropout_p, is_causal, scale = inputs
        if float(dropout_p) > 0.0:
            raise NotImplementedError(
                "torch_vulkan::sdpa_with_optional_mask backward formula "
                "does not support dropout_p>0 (no RNG-replay state)."
            )
        # Save only forward inputs — they reach the bw_module as regular
        # tangent inputs, not lifted constants. The fwd output is *not*
        # saved (it would otherwise become a lifted ``_tensor_constantN``
        # buffer that AOT's bw_module deepcopy fails on for Vulkan
        # storage). The backward formula recomputes attn from q/k anyway,
        # mirroring the flash-attention "save q+k, recompute attn" pattern.
        if attn_mask is not None:
            ctx.save_for_backward(query, key, value, attn_mask)
        else:
            ctx.save_for_backward(query, key, value)
        ctx.scale = (
            float(scale) if scale is not None else 1.0 / (int(query.shape[-1]) ** 0.5)
        )
        ctx.is_causal = bool(is_causal)
        ctx.has_mask = attn_mask is not None

    def _sdpa_backward(ctx, grad_out):
        if ctx.has_mask:
            q, k, v, attn_mask = ctx.saved_tensors
        else:
            q, k, v = ctx.saved_tensors
            attn_mask = None
        scale = ctx.scale
        aten = torch.ops.aten
        # Recompute attn from saved q/k (flash-attention's "save q+k,
        # recompute attn" memory trick). All ops use raw aten overloads
        # so AOT autograd's joint trace records them as FX nodes
        # (high-level torch.* wrappers can sometimes get evaluated
        # eagerly during the joint trace, surfacing as
        # ``_tensor_constantN`` lifted constants).
        scores = aten.mul.Tensor(aten.matmul(q, aten.transpose.int(k, -2, -1)), scale)
        if ctx.is_causal:
            seq_len = int(q.shape[-2])
            mask = aten.triu.default(
                aten.full.default(
                    [seq_len, seq_len],
                    float("-inf"),
                    dtype=q.dtype,
                    device=q.device,
                    pin_memory=False,
                ),
                1,
            )
            scores = aten.add.Tensor(scores, mask)
        if attn_mask is not None:
            scores = aten.add.Tensor(scores, attn_mask)
        attn = aten._softmax.default(scores, -1, False)
        # Standard SDPA backward identities:
        #   grad_v   = attn^T @ grad_out
        #   d_attn   = grad_out @ v^T
        #   d_scores = _softmax_backward_data(d_attn, attn, dim=-1)
        #   grad_q   = (d_scores @ k) * scale
        #   grad_k   = (d_scores^T @ q) * scale
        grad_v = aten.matmul(aten.transpose.int(attn, -2, -1), grad_out)
        d_attn = aten.matmul(grad_out, aten.transpose.int(v, -2, -1))
        d_scores = aten._softmax_backward_data.default(
            d_attn,
            attn,
            -1,
            attn.dtype,
        )
        grad_q = aten.mul.Tensor(aten.matmul(d_scores, k), scale)
        grad_k = aten.mul.Tensor(
            aten.matmul(aten.transpose.int(d_scores, -2, -1), q),
            scale,
        )
        # 7 inputs: (query, key, value, attn_mask, dropout_p, is_causal, scale)
        return grad_q, grad_k, grad_v, None, None, None, None

    sdpa_op.register_autograd(_sdpa_backward, setup_context=_sdpa_setup_context)
    return torch.ops.torch_vulkan.sdpa_with_optional_mask.default


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


def register_eager_patch_custom_ops() -> None:
    """Register the conv2d / conv1d / sdpa / max_pool2d / adaptive_avg_pool2d
    custom_ops used by the eager monkey-patches in
    ``python/torch_vulkan/__init__.py`` (PF.30.a/.b/.d).

    Called once during backend init, before ``_register_optional_tensor_workarounds``
    swaps in the patched ``F.conv2d`` / ``F.conv1d`` /
    ``F.scaled_dot_product_attention`` / ``F.max_pool2d`` /
    ``F.adaptive_avg_pool2d``. Idempotent — each ``_ensure_*`` is a singleton.
    """
    _ensure_conv2d_with_optional_bias_op_registered()
    _ensure_conv1d_with_optional_bias_op_registered()
    _ensure_sdpa_with_optional_mask_op_registered()
    _ensure_max_pool2d_op_registered()
    _ensure_adaptive_avg_pool2d_op_registered()
    # T4.8 foreach optimizer custom ops — registered lazily by
    # install_external_optimizer() in vulkan_template_caller. They live in
    # this module (below) because eager_patches is the canonical home for
    # `torch_vulkan::*` custom_op factories.


# ═══════════════════════════════════════════════════════════════════════
# T4.8 — Foreach optimizer custom op registrations
# ═══════════════════════════════════════════════════════════════════════
#
# M24 fix: torch.library.infer_schema() rejects PEP 604 unions like
# `float | list[float]`. Per-param scalar lists must be `list[float]`
# (or `Sequence[float]` — both are accepted). Single-scalar callers
# normalize to `[scalar] * n_params` at the boundary before calling.
#
# All foreach ops mutate `params` (and optionally momentum/m/v buffers)
# in-place. We declare them via `mutates_args=("params", ...)` so that
# Inductor's functionalization understands the alias contract.


def _ensure_foreach_sgd_step_op_registered() -> "object":
    """Register `torch_vulkan::foreach_sgd_step` (T4.8).

    Vanilla SGD with optional weight decay over a list of parameters:
        p_i := p_i - lr_i * (g_i + wd_i * p_i)   for i in range(N)

    All scalar args are `list[float]` length N. Single-scalar callers
    normalize at the boundary.
    """
    import torch

    op_name = "torch_vulkan::foreach_sgd_step"
    existing = getattr(torch.ops.torch_vulkan, "foreach_sgd_step", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    Tensor = torch.Tensor

    def _foreach_sgd_step_impl(
        params: list[Tensor],
        grads: list[Tensor],
        lr: list[float],
        weight_decay: list[float],
    ) -> None:
        from ..vulkan_template_caller import _pick_foreach_optimizer_caller

        n = len(params)
        if n == 0:
            return
        # Boundary normalization — accept length-1 broadcasts.
        if len(lr) == 1 and n > 1:
            lr = list(lr) * n
        if len(weight_decay) == 1 and n > 1:
            weight_decay = list(weight_decay) * n
        # Vulkan path: dispatch the template (one shader for all params).
        if params[0].device.type == "vulkan":
            caller = _pick_foreach_optimizer_caller("sgd", n, "float")
            caller(
                list(params),
                list(grads),
                lr=list(lr),
                weight_decay=list(weight_decay),
            )
            return
        # CPU/other path: standard SGD update.
        for p, g, l, wd in zip(params, grads, lr, weight_decay):
            if wd != 0.0:
                p.add_(p, alpha=wd)  # in-place: g_eff = g + wd*p
            p.add_(g, alpha=-l)

    _foreach_sgd_step_impl.__annotations__ = {
        "params": list[Tensor],
        "grads": list[Tensor],
        "lr": list[float],
        "weight_decay": list[float],
        "return": None,
    }
    sgd_op = torch.library.custom_op(op_name, mutates_args=("params",))(
        _foreach_sgd_step_impl
    )

    def _foreach_sgd_step_fake(params, grads, lr, weight_decay):
        return None

    sgd_op.register_fake(_foreach_sgd_step_fake)
    return torch.ops.torch_vulkan.foreach_sgd_step.default


def _ensure_foreach_sgd_momentum_step_op_registered() -> "object":
    """Register `torch_vulkan::foreach_sgd_momentum_step` (T4.8).

    SGD + momentum:
        buf_i := momentum_i * buf_i + g_i
        p_i   := p_i - lr_i * buf_i
    """
    import torch

    op_name = "torch_vulkan::foreach_sgd_momentum_step"
    existing = getattr(torch.ops.torch_vulkan, "foreach_sgd_momentum_step", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    Tensor = torch.Tensor

    def _impl(
        params: list[Tensor],
        grads: list[Tensor],
        momentum_bufs: list[Tensor],
        lr: list[float],
        weight_decay: list[float],
        momentum: list[float],
    ) -> None:
        from ..vulkan_template_caller import _pick_foreach_optimizer_caller

        n = len(params)
        if n == 0:
            return
        if len(lr) == 1 and n > 1:
            lr = list(lr) * n
        if len(weight_decay) == 1 and n > 1:
            weight_decay = list(weight_decay) * n
        if len(momentum) == 1 and n > 1:
            momentum = list(momentum) * n
        if params[0].device.type == "vulkan":
            caller = _pick_foreach_optimizer_caller("sgd_momentum", n, "float")
            caller(
                list(params),
                list(grads),
                lr=list(lr),
                weight_decay=list(weight_decay),
                momentum=list(momentum),
                momentum_bufs=list(momentum_bufs),
            )
            return
        for p, g, buf, l, wd, m in zip(
            params, grads, momentum_bufs, lr, weight_decay, momentum
        ):
            g_eff = g.clone()
            if wd != 0.0:
                g_eff = g_eff.add(p, alpha=wd)
            buf.mul_(m).add_(g_eff)
            p.add_(buf, alpha=-l)

    _impl.__annotations__ = {
        "params": list[Tensor],
        "grads": list[Tensor],
        "momentum_bufs": list[Tensor],
        "lr": list[float],
        "weight_decay": list[float],
        "momentum": list[float],
        "return": None,
    }
    op = torch.library.custom_op(op_name, mutates_args=("params", "momentum_bufs"))(
        _impl
    )

    def _fake(params, grads, momentum_bufs, lr, weight_decay, momentum):
        return None

    op.register_fake(_fake)
    return torch.ops.torch_vulkan.foreach_sgd_momentum_step.default


def _ensure_foreach_adamw_step_op_registered() -> "object":
    """Register `torch_vulkan::foreach_adamw_step` (T4.8).

    AdamW (decoupled weight decay):
        m_i := beta1_i * m_i + (1 - beta1_i) * g_i
        v_i := beta2_i * v_i + (1 - beta2_i) * g_i^2
        p_i := p_i - lr_i * (m_i / (sqrt(v_i) + eps_i) + wd_i * p_i)
    """
    import torch

    op_name = "torch_vulkan::foreach_adamw_step"
    existing = getattr(torch.ops.torch_vulkan, "foreach_adamw_step", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    Tensor = torch.Tensor

    def _impl(
        params: list[Tensor],
        grads: list[Tensor],
        m_bufs: list[Tensor],
        v_bufs: list[Tensor],
        lr: list[float],
        weight_decay: list[float],
        beta1: list[float],
        beta2: list[float],
        eps: list[float],
    ) -> None:
        from ..vulkan_template_caller import _pick_foreach_optimizer_caller

        n = len(params)
        if n == 0:
            return
        # Length-1 broadcast normalization at the boundary.
        if len(lr) == 1 and n > 1:
            lr = list(lr) * n
        if len(weight_decay) == 1 and n > 1:
            weight_decay = list(weight_decay) * n
        if len(beta1) == 1 and n > 1:
            beta1 = list(beta1) * n
        if len(beta2) == 1 and n > 1:
            beta2 = list(beta2) * n
        if len(eps) == 1 and n > 1:
            eps = list(eps) * n
        if params[0].device.type == "vulkan":
            caller = _pick_foreach_optimizer_caller("adamw", n, "float")
            # Template uses `momentum` slot for beta1.
            caller(
                list(params),
                list(grads),
                lr=list(lr),
                weight_decay=list(weight_decay),
                momentum=list(beta1),
                beta2=list(beta2),
                eps=list(eps),
                momentum_bufs=list(m_bufs),
                v_bufs=list(v_bufs),
            )
            return
        # CPU reference.
        for p, g, m, v, l, wd, b1, b2, e in zip(
            params, grads, m_bufs, v_bufs, lr, weight_decay, beta1, beta2, eps
        ):
            m.mul_(b1).add_(g, alpha=1.0 - b1)
            v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
            denom = v.sqrt().add_(e)
            update = m / denom
            if wd != 0.0:
                update = update.add(p, alpha=wd)
            p.add_(update, alpha=-l)

    _impl.__annotations__ = {
        "params": list[Tensor],
        "grads": list[Tensor],
        "m_bufs": list[Tensor],
        "v_bufs": list[Tensor],
        "lr": list[float],
        "weight_decay": list[float],
        "beta1": list[float],
        "beta2": list[float],
        "eps": list[float],
        "return": None,
    }
    op = torch.library.custom_op(op_name, mutates_args=("params", "m_bufs", "v_bufs"))(
        _impl
    )

    def _fake(params, grads, m_bufs, v_bufs, lr, weight_decay, beta1, beta2, eps):
        return None

    op.register_fake(_fake)
    return torch.ops.torch_vulkan.foreach_adamw_step.default


def _ensure_foreach_lion_step_op_registered() -> "object":
    """Register `torch_vulkan::foreach_lion_step` (T4.8).

    Lion (EvoLved Sign Momentum):
        update = beta1_i * momentum_i + (1 - beta1_i) * g_i
        p_i   := p_i - lr_i * sign(update)
        momentum_i := beta2_i * momentum_i + (1 - beta2_i) * g_i
    """
    import torch

    op_name = "torch_vulkan::foreach_lion_step"
    existing = getattr(torch.ops.torch_vulkan, "foreach_lion_step", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    Tensor = torch.Tensor

    def _impl(
        params: list[Tensor],
        grads: list[Tensor],
        momentum_bufs: list[Tensor],
        lr: list[float],
        weight_decay: list[float],
        beta1: list[float],
        beta2: list[float],
    ) -> None:
        from ..vulkan_template_caller import _pick_foreach_optimizer_caller

        n = len(params)
        if n == 0:
            return
        if len(lr) == 1 and n > 1:
            lr = list(lr) * n
        if len(weight_decay) == 1 and n > 1:
            weight_decay = list(weight_decay) * n
        if len(beta1) == 1 and n > 1:
            beta1 = list(beta1) * n
        if len(beta2) == 1 and n > 1:
            beta2 = list(beta2) * n
        if params[0].device.type == "vulkan":
            caller = _pick_foreach_optimizer_caller("lion", n, "float")
            caller(
                list(params),
                list(grads),
                lr=list(lr),
                weight_decay=list(weight_decay),
                momentum=list(beta1),
                beta2=list(beta2),
                momentum_bufs=list(momentum_bufs),
            )
            return
        for p, g, mom, l, wd, b1, b2 in zip(
            params, grads, momentum_bufs, lr, weight_decay, beta1, beta2
        ):
            update = mom.mul(b1).add_(g, alpha=1.0 - b1)
            p.add_(update.sign(), alpha=-l)
            if wd != 0.0:
                p.add_(p, alpha=-l * wd)
            mom.mul_(b2).add_(g, alpha=1.0 - b2)

    _impl.__annotations__ = {
        "params": list[Tensor],
        "grads": list[Tensor],
        "momentum_bufs": list[Tensor],
        "lr": list[float],
        "weight_decay": list[float],
        "beta1": list[float],
        "beta2": list[float],
        "return": None,
    }
    op = torch.library.custom_op(op_name, mutates_args=("params", "momentum_bufs"))(
        _impl
    )

    def _fake(params, grads, momentum_bufs, lr, weight_decay, beta1, beta2):
        return None

    op.register_fake(_fake)
    return torch.ops.torch_vulkan.foreach_lion_step.default
