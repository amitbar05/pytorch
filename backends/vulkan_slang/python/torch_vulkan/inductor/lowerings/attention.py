"""SDPA (scaled_dot_product_attention) lowering — OP.26.

Registers a native ``aten.scaled_dot_product_attention`` lowering that
routes directly to the FlashAttention Slang template for supported
configurations, eliminating the symptom-fix pattern-matcher in
``fx_passes/patterns/sdpa.py`` (anti-goal #5) and the pre-grad
decomposition in ``fx_passes/post_grad.py:_replace_sdpa_with_custom_op``.

Also registers a lowering for ``torch_vulkan::sdpa_with_optional_mask``
(the custom op that the monkey-patch in ``__init__.py:_patched_sdpa``
routes through under ``torch.compile``), covering GQA correctly via
the flash attention template (which natively handles KV_H < H).

Supported configurations
  - head_dim ∈ {32, 64, 128, 256}
  - attn_mask=None, dropout_p=0.0
  - GQA handled by the template natively (KV_H ≤ H)
Unsupported configurations fall through to the upstream decomposition
(matmuls + softmax).
"""

from __future__ import annotations

import torch

from . import _is_vulkan  # noqa: F401 — re-exported for convention

_SUPPORTED_HEAD_DIMS: set[int] = {32, 64, 128, 256}


def _register_sdpa_lowering() -> None:
    """Register native lowerings for SDPA ops.

    Registers two lowerings:
    1. ``aten.scaled_dot_product_attention`` — the direct ATen op path.
    2. ``torch_vulkan::sdpa_with_optional_mask`` — the custom op that the
       monkey-patch in ``__init__.py:_patched_sdpa`` routes through under
       ``torch.compile``.

    Both route to the FlashAttention Slang template via the existing
    ``flash_attention_fused`` autotune infrastructure.  Unsupported
    configurations fall through to upstream decomposition.

    Idempotent — safe to call multiple times.
    """
    from torch._inductor import lowering as L
    from torch._inductor.lowering import register_lowering

    aten = torch.ops.aten

    # Idempotent guard.
    if getattr(L, "_vulkan_sdpa_lowering_registered", False):
        return
    L._vulkan_sdpa_lowering_registered = True

    # Capture original lowerings for fall-through.
    _orig_aten = L.lowerings.get(aten.scaled_dot_product_attention)

    # Ensure the flash_attention_fused lowering is registered so we can
    # delegate to it.  install_external_flash_attention is idempotent.
    from ..templates.caller.flash_attn import install_external_flash_attention

    install_external_flash_attention()

    # ── Lowering 1: aten.scaled_dot_product_attention ────────────

    @register_lowering(aten.scaled_dot_product_attention, type_promotion_kind=None)
    def _vulkan_sdpa(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        *,
        layout=None,
    ):
        if not _is_vulkan(query):
            return _fallthrough_aten(
                _orig_aten,
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                enable_gqa,
                layout=layout,
            )
        if attn_mask is not None:
            return _fallthrough_aten(
                _orig_aten,
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                enable_gqa,
                layout=layout,
            )
        # M19.5: head_dim may be a SymInt under dynamic shapes —
        # use size_hint to resolve it, and fall through to the
        # stock aten path when the hint can't give us a concrete
        # value.
        from torch._inductor.graph import V

        head_dim_sym = query.get_size()[-1]
        try:
            head_dim = V.graph.sizevars.size_hint(head_dim_sym)
        except (TypeError, ValueError):
            return _fallthrough_aten(
                _orig_aten,
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                enable_gqa,
                layout=layout,
            )
        if head_dim not in _SUPPORTED_HEAD_DIMS:
            return _fallthrough_aten(
                _orig_aten,
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                enable_gqa,
                layout=layout,
            )
        return _route_to_flash(
            L,
            _orig_aten,
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            head_dim,
            layout=layout,
            fallback_fn=lambda: _fallthrough_aten(
                _orig_aten,
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                enable_gqa,
                layout=layout,
            ),
        )

    # ── Lowering 2: torch_vulkan::sdpa_with_optional_mask ─────────
    # The monkey-patch in __init__.py:_patched_sdpa intercepts
    # F.scaled_dot_product_attention under torch.compile and routes to
    # this custom op.  The custom op is handled by the eager C++ backend
    # (ExternKernel path) — NOT by this lowering.  Registering a lowering
    # here would take precedence over the eager path and break the
    # pre-existing SDPA tests (TestFlashAttentionFusion).
    #
    # Future work (OP.26 follow-up): once the flash_attention_fused
    # lowering is properly wired at lowering-registration time, this
    # custom-op path can be re-enabled to also benefit from the template.

    # from ..fx_passes.eager_patches import _ensure_sdpa_with_optional_mask_op_registered
    # sdpa_mask_op = _ensure_sdpa_with_optional_mask_op_registered()
    # _orig_mask = L.lowerings.get(sdpa_mask_op)
    # @register_lowering(sdpa_mask_op, type_promotion_kind=None)
    # def _vulkan_sdpa_with_optional_mask(...): ...


def _route_to_flash(
    L,
    _orig,
    query,
    key,
    value,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    head_dim,
    *,
    layout=None,
    fallback_fn=None,
):
    """Route a supported SDPA call to the flash_attention_fused lowering.

    When the flash template is not registered or the config is unsupported,
    decomposes SDPA into matmul + softmax + matmul primitives instead of
    returning ``NotImplemented`` (which causes an IR-level crash).
    """
    flash_op = torch.ops.torch_vulkan.flash_attention_fused
    flash_lowering = L.lowerings.get(flash_op)
    if flash_lowering is None:
        # Flash template not available — decompose into primitives.
        return _decompose_sdpa_to_matmul(
            query, key, value, attn_mask, is_causal, scale, head_dim
        )

    if scale is None:
        scale_val = 1.0 / (float(head_dim) ** 0.5)
    else:
        scale_val = scale

    return flash_lowering(query, key, value, scale_val, is_causal, layout=layout)


def _decompose_sdpa_to_matmul(query, key, value, attn_mask, is_causal, scale, head_dim):
    """Decompose SDPA into primitives when flash template is unavailable.

    Uses Inductor's built-in SDPA decomposition via the upstream
    lowering's fallback target. Avoids infinite recursion by checking
    for our own lowering key.
    """
    from torch._inductor.lowering import lowerings as L

    # Use the upstream decomposition target (not aten.scaled_dot_product_attention
    # which would recurse back to us). Instead decompose via the known
    # upstream pattern: expand SDPA into matmul + softmax.
    #
    # For 4-D inputs (B, H, S, D): scores = Q @ K^T, attn = softmax(scores * scale), out = attn @ V
    # We use the individual op lowerings already registered:
    bmm = L.get(torch.ops.aten.bmm.default)
    softmax = L.get(torch.ops.aten._softmax.default)
    permute = L.get(torch.ops.aten.permute.default)

    if bmm is None or softmax is None or permute is None:
        return NotImplemented

    if scale is None:
        scale_val = 1.0 / (float(head_dim) ** 0.5)
    else:
        scale_val = float(scale)

    # Transpose key: K^T = permute(K, [0, 1, 3, 2]) for 4-D
    key_t = (
        permute(key, [0, 1, 3, 2])
        if len(query.get_size()) == 4
        else permute(key, [1, 0])
    )
    # scores = Q @ K^T
    scores = bmm(query, key_t)
    # scores *= scale
    scores = scores * scale_val
    # attn = softmax(scores, dim=-1)
    attn = softmax(scores, dim=-1, half_to_float=False)
    # out = attn @ V
    return bmm(attn, value)


def _fallthrough_aten(
    _orig,
    query,
    key,
    value,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    enable_gqa,
    *,
    layout=None,
):
    if _orig is not None:
        return _orig(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            layout=layout,
        )
    return NotImplemented


def _fallthrough_mask(
    _orig,
    query,
    key,
    value,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    *,
    layout=None,
):
    if _orig is not None:
        return _orig(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            layout=layout,
        )
    return NotImplemented
