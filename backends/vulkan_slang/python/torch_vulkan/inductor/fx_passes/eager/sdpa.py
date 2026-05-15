"""Flash-attention / SDPA custom-op registrations for fused FX-pattern targets.

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
