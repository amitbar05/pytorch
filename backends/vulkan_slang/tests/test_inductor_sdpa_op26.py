"""OP.26 — Native ``aten.scaled_dot_product_attention`` lowering regression tests.

Verifies that ``F.scaled_dot_product_attention`` under
``torch.compile(backend="inductor")`` routes through the native
lowering (``lowerings/attention.py``) and produces correct results.
The lowering dispatches to the FlashAttention Slang template for
supported head_dims {32, 64, 128, 256} with no attn_mask/dropout.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(autouse=True)
def setup():
    try:
        import torch_vulkan

        if not torch_vulkan.is_available():
            pytest.skip("No Vulkan device")
    except ImportError:
        pytest.skip("torch_vulkan not installed")


class TestNativeSdpaLowering:
    """OP.26 — Native aten.scaled_dot_product_attention lowering."""

    def test_sdpa_correctness_head_dim_64(self):
        """Basic SDPA correctness for head_dim=64 (most common case)."""

        @torch.compile(backend="inductor")
        def fn(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        q = torch.randn(1, 2, 8, 64, device="vulkan:0")
        k = torch.randn(1, 2, 8, 64, device="vulkan:0")
        v = torch.randn(1, 2, 8, 64, device="vulkan:0")
        y = fn(q, k, v)
        ref = F.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu(), is_causal=True)
        torch.testing.assert_close(y.cpu(), ref, rtol=1e-2, atol=1e-2)

    def test_sdpa_correctness_head_dim_32(self):
        """OP.26 — head_dim=32 correctness (small head)."""

        @torch.compile(backend="inductor")
        def fn(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=False)

        q = torch.randn(2, 4, 16, 32, device="vulkan:0")
        k = torch.randn(2, 4, 16, 32, device="vulkan:0")
        v = torch.randn(2, 4, 16, 32, device="vulkan:0")
        y = fn(q, k, v)
        ref = F.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu(), is_causal=False)
        torch.testing.assert_close(y.cpu(), ref, rtol=1e-2, atol=1e-2)

    def test_sdpa_correctness_head_dim_128(self):
        """OP.26 — head_dim=128 correctness (large head)."""

        @torch.compile(backend="inductor")
        def fn(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        q = torch.randn(1, 4, 4, 128, device="vulkan:0")
        k = torch.randn(1, 4, 4, 128, device="vulkan:0")
        v = torch.randn(1, 4, 4, 128, device="vulkan:0")
        y = fn(q, k, v)
        ref = F.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu(), is_causal=True)
        torch.testing.assert_close(y.cpu(), ref, rtol=1e-2, atol=1e-2)

    def test_sdpa_correctness_head_dim_256(self):
        """OP.26 — head_dim=256 correctness (max supported)."""

        @torch.compile(backend="inductor")
        def fn(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=False)

        q = torch.randn(1, 2, 4, 256, device="vulkan:0")
        k = torch.randn(1, 2, 4, 256, device="vulkan:0")
        v = torch.randn(1, 2, 4, 256, device="vulkan:0")
        y = fn(q, k, v)
        ref = F.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu(), is_causal=False)
        torch.testing.assert_close(y.cpu(), ref, rtol=1e-2, atol=1e-2)

    def test_sdpa_explicit_scale(self):
        """OP.26 — user-supplied scale parameter."""

        @torch.compile(backend="inductor")
        def fn(q, k, v, scale):
            return F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=False)

        q = torch.randn(1, 2, 8, 64, device="vulkan:0")
        k = torch.randn(1, 2, 8, 64, device="vulkan:0")
        v = torch.randn(1, 2, 8, 64, device="vulkan:0")
        scale = 0.125
        y = fn(q, k, v, scale)
        ref = F.scaled_dot_product_attention(
            q.cpu(), k.cpu(), v.cpu(), scale=0.125, is_causal=False
        )
        torch.testing.assert_close(y.cpu(), ref, rtol=1e-2, atol=1e-2)

    @pytest.mark.xfail(
        reason="Pre-existing: C++ _c_ext._sdpa does not support GQA (H != KV_H). "
        "The lowering routes to flash_attention_fused correctly, but the eager "
        "fallback in the custom-op shim hits the C++ backend which rejects "
        "mismatched Q/KV head counts. Tracked as part of OP.26 residue.",
        strict=True,
    )
    def test_sdpa_gqa_kv_heads_less_than_q_heads(self):
        """OP.26 — grouped-query attention: KV heads < Q heads."""

        @torch.compile(backend="inductor")
        def fn(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=False)

        q = torch.randn(1, 4, 8, 64, device="vulkan:0")
        k = torch.randn(1, 2, 8, 64, device="vulkan:0")
        v = torch.randn(1, 2, 8, 64, device="vulkan:0")
        y = fn(q, k, v)
        ref = F.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu(), is_causal=False)
        torch.testing.assert_close(y.cpu(), ref, rtol=1e-2, atol=1e-2)

    def test_sdpa_dispatch_count_with_lowering(self):
        """OP.26 — fused path stays at ≤5 dispatches (lowering active)."""
        import torch_vulkan

        torch_vulkan._c_ext._reset_perf_counters()

        @torch.compile(backend="inductor")
        def fn(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        q = torch.randn(1, 2, 8, 64, device="vulkan:0")
        k = torch.randn(1, 2, 8, 64, device="vulkan:0")
        v = torch.randn(1, 2, 8, 64, device="vulkan:0")
        fn(q, k, v)
        torch_vulkan._c_ext._reset_perf_counters()
        fn(q, k, v)
        torch_vulkan._c_ext._synchronize(0)
        n = torch_vulkan._c_ext._get_dispatch_count()
        assert n <= 5, f"expected ≤5 dispatches (native SDPA lowering), got {n}"

    def test_sdpa_fallthrough_unsupported_head_dim(self):
        """OP.26 — head_dim=48 falls through to upstream decomposition."""

        @torch.compile(backend="inductor")
        def fn(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=False)

        q = torch.randn(1, 2, 4, 48, device="vulkan:0")
        k = torch.randn(1, 2, 4, 48, device="vulkan:0")
        v = torch.randn(1, 2, 4, 48, device="vulkan:0")
        y = fn(q, k, v)
        ref = F.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu(), is_causal=False)
        torch.testing.assert_close(y.cpu(), ref, rtol=1e-2, atol=1e-2)

    def test_sdpa_fallthrough_with_mask(self):
        """OP.26 — attn_mask causes fall-through to upstream decomposition."""

        @torch.compile(backend="inductor")
        def fn(q, k, v, mask):
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=False
            )

        q = torch.randn(1, 2, 8, 64, device="vulkan:0")
        k = torch.randn(1, 2, 8, 64, device="vulkan:0")
        v = torch.randn(1, 2, 8, 64, device="vulkan:0")
        mask = torch.tril(torch.ones(8, 8, device="vulkan:0")).expand(1, 2, 8, 8)
        y = fn(q, k, v, mask)
        ref = F.scaled_dot_product_attention(
            q.cpu(), k.cpu(), v.cpu(), attn_mask=mask.cpu(), is_causal=False
        )
        torch.testing.assert_close(y.cpu(), ref, rtol=1e-2, atol=1e-2)

    def test_sdpa_lowering_is_registered(self):
        """OP.26 — confirmation that the lowering is wired."""
        import torch
        from torch._inductor import lowering as L

        aten = torch.ops.aten
        sdpa_lo = L.lowerings.get(aten.scaled_dot_product_attention)
        assert sdpa_lo is not None, (
            "aten.scaled_dot_product_attention lowering not registered"
        )
        assert (
            "_vulkan_sdpa" in sdpa_lo.__name__ or "sdpa" in sdpa_lo.__name__.lower()
        ), f"unexpected lowering: {sdpa_lo.__name__}"
