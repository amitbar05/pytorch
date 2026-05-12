"""Tests verifying Vulkan backend support for google/gemma-4-E2B architecture.

Gemma-4 E2B key features:
- Hybrid sliding-window (512 tok) + global full-context attention
- RMSNorm with eps=1e-6
- GELU exact (erf-based, approximate='none') activation
- GQA: 8 query heads, 1 KV head (8x expansion)
- Logit softcapping: softcap * tanh(logits / softcap)
- Proportional RoPE (partial_rotary_factor=0.25) for global layers
- Per-layer embeddings (PLE) via linear + mul + add
- KV sharing across layers
- Mixture-of-Experts (MoE): 8 experts, top-2 routing
- index_add_ for MoE output accumulation
- one_hot for expert masking
- nonzero / torch.where for expert selection
- vocab_size=262,144 (chunked LM head)
- head_dim=256 (local) / 512 (global queries)
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


RTOL = 1e-2
ATOL = 1e-2


@pytest.fixture(autouse=True)
def setup():
    try:
        import torch_vulkan
        if not torch_vulkan.is_available():
            pytest.skip("No Vulkan device")
    except ImportError:
        pytest.skip("torch_vulkan not installed")


def v(t):
    return t.to("vulkan:0")


def assert_close(a, b, rtol=RTOL, atol=ATOL):
    torch.testing.assert_close(a.cpu(), b if b.device.type == "cpu" else b.cpu(),
                                rtol=rtol, atol=atol)


# ── GELU exact (erf-based) ────────────────────────────────────────

class TestGeluExact:
    def test_gelu_none_forward(self):
        """F.gelu(approximate='none') uses erf-based exact formula."""
        x = torch.randn(64, 128)
        ref = F.gelu(x, approximate="none")
        out = F.gelu(v(x), approximate="none")
        assert_close(out, ref, rtol=1e-3, atol=1e-3)

    def test_gelu_none_backward(self):
        """Backward through exact GELU matches CPU autograd."""
        x_cpu = torch.randn(32, 64, requires_grad=True)
        vx = v(x_cpu.detach()).requires_grad_(True)

        F.gelu(vx, approximate="none").sum().backward()
        F.gelu(x_cpu, approximate="none").sum().backward()

        assert_close(vx.grad, x_cpu.grad, rtol=1e-2, atol=1e-2)

    def test_gelu_tanh_still_works(self):
        """approximate='tanh' path still produces correct results."""
        x = torch.randn(32, 64)
        ref = F.gelu(x, approximate="tanh")
        out = F.gelu(v(x), approximate="tanh")
        assert_close(out, ref)

    def test_gelu_values(self):
        """Spot-check known GELU values for approximate='none'."""
        x = torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0])
        ref = F.gelu(x, approximate="none")
        out = F.gelu(v(x), approximate="none")
        assert_close(out, ref, rtol=1e-3, atol=1e-3)


# ── index_add_ ────────────────────────────────────────────────────

class TestIndexAdd:
    def test_index_add_dim0_no_duplicates(self):
        """index_add_ dim=0, no duplicate indices."""
        self_cpu = torch.zeros(6, 4)
        src_cpu = torch.randn(3, 4)
        idx = torch.tensor([0, 2, 5])

        ref = self_cpu.clone().index_add_(0, idx, src_cpu)

        vs = v(self_cpu.clone())
        vs.index_add_(0, v(idx), v(src_cpu))
        assert_close(vs, ref)

    def test_index_add_dim0_with_duplicates(self):
        """index_add_ dim=0, duplicate indices (CAS atomics)."""
        self_cpu = torch.zeros(4, 8)
        src_cpu = torch.ones(6, 8)
        idx = torch.tensor([0, 0, 1, 2, 2, 3])  # duplicates at 0 and 2

        ref = self_cpu.clone().index_add_(0, idx, src_cpu)

        vs = v(self_cpu.clone())
        vs.index_add_(0, v(idx), v(src_cpu))
        assert_close(vs, ref)

    def test_index_add_alpha(self):
        """index_add_ with alpha scaling."""
        self_cpu = torch.zeros(5, 4)
        src_cpu = torch.randn(3, 4)
        idx = torch.tensor([1, 3, 4])
        alpha = 0.5

        ref = self_cpu.clone().index_add_(0, idx, src_cpu, alpha=alpha)

        vs = v(self_cpu.clone())
        vs.index_add_(0, v(idx), v(src_cpu), alpha=alpha)
        assert_close(vs, ref)

    def test_index_add_moe_pattern(self):
        """Simulate MoE expert output accumulation pattern."""
        N, D = 32, 64  # 32 tokens, 64 hidden dim
        num_experts, top_k = 4, 2

        # Fake routing: each token has top_k expert assignments
        expert_outputs = torch.randn(N * top_k, D)
        token_indices = torch.randint(0, N, (N * top_k,))

        ref = torch.zeros(N, D)
        ref.index_add_(0, token_indices, expert_outputs)

        vs = v(torch.zeros(N, D))
        vs.index_add_(0, v(token_indices), v(expert_outputs))
        assert_close(vs, ref)


# ── nonzero / torch.where ─────────────────────────────────────────

class TestNonzero:
    def test_nonzero_basic(self):
        """nonzero returns correct indices (CPU fallback)."""
        x = torch.tensor([0, 1, 0, 2, 0, 3], dtype=torch.float32)
        ref = x.nonzero()
        out = v(x).nonzero()
        assert_close(out.float(), ref.float())

    def test_nonzero_2d(self):
        """nonzero on 2D tensor."""
        x = torch.eye(4)
        ref = x.nonzero()
        out = v(x).nonzero()
        assert_close(out.float(), ref.float())

    def test_where_condition(self):
        """torch.where(condition) as nonzero wrapper."""
        mask = torch.tensor([True, False, True, True, False])
        ref = torch.where(mask)
        out = torch.where(v(mask))
        assert_close(out[0].float(), ref[0].float())


# ── one_hot ───────────────────────────────────────────────────────

class TestOneHot:
    def test_one_hot_basic(self):
        """F.one_hot produces correct one-hot encoding."""
        idx = torch.tensor([0, 2, 1, 3])
        ref = F.one_hot(idx, num_classes=4)
        out = F.one_hot(v(idx), num_classes=4)
        assert_close(out.float(), ref.float())

    def test_one_hot_moe_expert_mask(self):
        """MoE expert mask via one_hot (N*top_k indices → [N*top_k, num_experts])."""
        N, top_k, num_experts = 8, 2, 4
        selected = torch.randint(0, num_experts, (N * top_k,))
        ref = F.one_hot(selected, num_classes=num_experts)
        out = F.one_hot(v(selected), num_classes=num_experts)
        assert_close(out.float(), ref.float())


# ── Sliding window causal mask ────────────────────────────────────

class TestSlidingWindowMask:
    def _make_sliding_mask(self, S, window, device="cpu"):
        """Causal sliding-window mask: -inf outside [i-window+1, i]."""
        full_inf = torch.full((S, S), float("-inf"), device=device)
        causal = torch.triu(full_inf, 1)              # -inf where j > i
        left_cut = torch.tril(full_inf, -window)      # -inf where j <= i-window
        return causal + left_cut

    def test_sliding_window_shape(self):
        """Sliding-window mask has correct shape and values."""
        S, W = 16, 4
        mask = self._make_sliding_mask(S, W, "vulkan:0")
        mask_cpu = mask.cpu()
        assert mask_cpu.shape == (S, S)
        # Diagonal should be 0 (attend to self)
        assert mask_cpu[3, 3] == 0.0
        # Future positions: -inf
        assert mask_cpu[3, 4] == float("-inf")
        # Within window: 0
        assert mask_cpu[5, 2] == 0.0   # j=2, i=5, diff=3 < window=4 → attend
        # Beyond window left boundary: -inf
        assert mask_cpu[5, 1] == float("-inf")  # j=1, i=5, diff=4 >= window=4

    def test_sliding_window_attention(self):
        """Full sliding-window attention using composition of existing ops."""
        B, H, S, D = 1, 2, 16, 32
        window = 4

        q = torch.randn(B, H, S, D, device="vulkan:0")
        k = torch.randn(B, H, S, D, device="vulkan:0")
        vt = torch.randn(B, H, S, D, device="vulkan:0")

        mask = self._make_sliding_mask(S, window, "vulkan:0")
        scale = D ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale + mask
        weights = F.softmax(scores, dim=-1)
        out = weights @ vt

        assert out.shape == (B, H, S, D)
        assert torch.isfinite(out.cpu()).all()

    def test_sliding_vs_global_attention(self):
        """Sliding mask correctly restricts attention compared to global."""
        B, H, S, D = 1, 1, 8, 16
        window = 3
        q = torch.randn(B, H, S, D, device="vulkan:0")
        k = torch.randn(B, H, S, D, device="vulkan:0")
        vt = torch.randn(B, H, S, D, device="vulkan:0")
        scale = D ** -0.5

        # Sliding-window output
        mask = self._make_sliding_mask(S, window, "vulkan:0")
        out_sliding = F.softmax((q @ k.transpose(-2,-1)) * scale + mask, dim=-1) @ vt

        # Global causal output
        full_inf = torch.full((S, S), float("-inf"), device="vulkan:0")
        global_mask = torch.triu(full_inf, 1)
        out_global = F.softmax((q @ k.transpose(-2,-1)) * scale + global_mask, dim=-1) @ vt

        # Outputs differ (sliding window restricts context)
        assert not torch.allclose(out_sliding.cpu(), out_global.cpu(), atol=1e-4)


# ── Logit softcapping ─────────────────────────────────────────────

class TestLogitSoftcap:
    def test_softcap_attention(self):
        """Attention logit softcapping: softcap * tanh(logits / softcap)."""
        B, H, S, D = 2, 4, 16, 32
        softcap = 30.0
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        scale = D ** -0.5

        def apply_softcap(logits, cap):
            return cap * torch.tanh(logits / cap)

        logits_cpu = (q @ k.transpose(-2, -1)) * scale
        capped_cpu = apply_softcap(logits_cpu, softcap)

        vq, vk = v(q), v(k)
        logits_v = (vq @ vk.transpose(-2, -1)) * scale
        capped_v = apply_softcap(logits_v, softcap)

        assert_close(capped_v, capped_cpu)
        # Softcapped values must be in [-softcap, softcap]
        assert capped_v.cpu().abs().max().item() <= softcap + 1e-5

    def test_final_logit_softcap(self):
        """Final output logit softcapping (vocab dim)."""
        B, S, V = 2, 8, 1024
        softcap = 30.0
        logits = torch.randn(B, S, V)
        ref = softcap * torch.tanh(logits / softcap)
        out = softcap * torch.tanh(v(logits) / softcap)
        assert_close(out, ref)


# ── Partial RoPE (proportional_rotary_factor=0.25) ─────────────────

class TestPartialRoPE:
    def _rotate_half(self, x):
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def _apply_rope_partial(self, x, cos, sin, partial_factor=0.25):
        """Apply RoPE to only the first partial_factor fraction of dims."""
        D = x.shape[-1]
        r = int(D * partial_factor)
        x_rot = x[..., :r]
        x_pass = x[..., r:]
        cos_r = cos[..., :r]
        sin_r = sin[..., :r]
        x_rot_out = x_rot * cos_r + self._rotate_half(x_rot) * sin_r
        return torch.cat([x_rot_out, x_pass], dim=-1)

    def test_partial_rope_forward(self):
        """Partial RoPE (25% of dims) via slice+apply+cat on Vulkan."""
        B, H, S, D = 1, 4, 8, 64
        partial = 0.25
        r = int(D * partial)  # 16 dims rotated

        x = torch.randn(B, H, S, D)
        positions = torch.arange(S, dtype=torch.float32)
        # r//2 frequency pairs, each pair covers 2 dims → total r dims
        theta = torch.pow(10000.0, -torch.arange(0, r, 2).float() / r)
        freqs = torch.outer(positions, theta)          # [S, r//2]
        emb = torch.cat([freqs, freqs], dim=-1)        # [S, r] — duplicate for cos/sin
        cos = emb.cos().unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)  # [B,H,S,r]
        sin = emb.sin().unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)

        ref = self._apply_rope_partial(x, cos, sin, partial)
        out = self._apply_rope_partial(v(x), v(cos), v(sin), partial)

        assert_close(out, ref)
        # Unrotated suffix should be unchanged
        assert_close(out[..., r:], x[..., r:])


# ── GQA (8 queries, 1 KV head → 8x expansion) ─────────────────────

class TestGQAExpansion:
    def test_gqa_kv_expansion(self):
        """GQA KV expansion: [B, 1, S, D] → [B, 8, S, D]."""
        B, H, KVH, S, D = 2, 8, 1, 16, 32
        n_rep = H // KVH

        k = torch.randn(B, KVH, S, D, device="vulkan:0")
        # expand: unsqueeze → expand → reshape
        k_exp = k.unsqueeze(2).expand(B, KVH, n_rep, S, D).reshape(B, H, S, D)
        assert k_exp.shape == (B, H, S, D)
        assert torch.isfinite(k_exp.cpu()).all()

    def test_gqa_attention_forward(self):
        """Full GQA forward: 8 Q heads attend over 1 KV head (expanded 8x)."""
        B, H, KVH, S, D = 1, 8, 1, 16, 32
        n_rep = H // KVH

        q = torch.randn(B, H, S, D, device="vulkan:0")
        k = torch.randn(B, KVH, S, D, device="vulkan:0")
        vt = torch.randn(B, KVH, S, D, device="vulkan:0")

        k_exp = k.unsqueeze(2).expand(B, KVH, n_rep, S, D).reshape(B, H, S, D)
        v_exp = vt.unsqueeze(2).expand(B, KVH, n_rep, S, D).reshape(B, H, S, D)

        out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
        assert out.shape == (B, H, S, D)
        assert torch.isfinite(out.cpu()).all()


# ── RMSNorm (Gemma-4 uses eps=1e-6) ───────────────────────────────

class TestRMSNormGemma4:
    def test_rms_norm_eps(self):
        """RMSNorm with Gemma-4's eps=1e-6."""
        import torch_vulkan
        B, S, D = 2, 8, 64
        x = torch.randn(B, S, D)
        w = torch.ones(D)

        ref_var = x.pow(2).mean(-1, keepdim=True)
        ref = x * torch.rsqrt(ref_var + 1e-6) * w

        out = torch_vulkan.rms_norm(v(x), v(w), eps=1e-6)
        assert_close(out, ref, rtol=1e-3, atol=1e-3)

    def test_rms_norm_backward(self):
        """RMSNorm backward with eps=1e-6."""
        import torch_vulkan
        B, S, D = 1, 4, 32
        x = torch.randn(B, S, D)
        w = torch.ones(D)

        # CPU reference via decomposed formula
        def rms_norm_ref(x, w, eps=1e-6):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

        x_cpu = x.clone().requires_grad_(True)
        w_cpu = w.clone().requires_grad_(True)
        vx = v(x.clone()).requires_grad_(True)
        vw = v(w.clone()).requires_grad_(True)

        torch_vulkan.rms_norm(vx, vw, eps=1e-6).sum().backward()
        rms_norm_ref(x_cpu, w_cpu).sum().backward()

        assert_close(vx.grad, x_cpu.grad)
        assert_close(vw.grad, w_cpu.grad)


# ── MoE (Mixture of Experts) — full routing + accumulation ─────────

class TestMixtureOfExperts:
    def _moe_forward_cpu(self, hidden, router_w, expert_ws_gate,
                          expert_ws_up, expert_ws_down, num_experts, top_k):
        """Pure-PyTorch MoE reference."""
        B_S, D = hidden.shape
        router_logits = hidden @ router_w.T               # [B_S, E]
        routing_weights = F.softmax(router_logits, dim=-1)
        routing_weights, selected = torch.topk(routing_weights, top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(-1, keepdim=True)

        output = torch.zeros_like(hidden)
        for expert_idx in range(num_experts):
            mask = (selected == expert_idx)               # [B_S, top_k]
            token_mask = mask.any(dim=-1)                 # [B_S]
            if not token_mask.any():
                continue
            token_ids = token_mask.nonzero(as_tuple=True)[0]
            weights = routing_weights[token_mask]
            expert_in = hidden[token_mask]
            gate = F.gelu(expert_in @ expert_ws_gate[expert_idx].T, approximate="none")
            up = expert_in @ expert_ws_up[expert_idx].T
            expert_out = (gate * up) @ expert_ws_down[expert_idx].T
            # Weight and accumulate
            slot_weights = (mask[token_mask] * weights).sum(-1, keepdim=True)
            output.index_add_(0, token_ids, expert_out * slot_weights)
        return output

    def test_moe_routing(self):
        """MoE router produces top-k expert selection."""
        B_S, D, E, K = 16, 32, 4, 2
        hidden = torch.randn(B_S, D, device="vulkan:0")
        router_w = torch.randn(E, D, device="vulkan:0")

        router_logits = hidden @ router_w.T
        routing_weights = F.softmax(router_logits, dim=-1)
        weights, selected = torch.topk(routing_weights, K, dim=-1)

        assert selected.shape == (B_S, K)
        assert weights.shape == (B_S, K)
        # All selected experts are in [0, E)
        assert (selected.cpu() >= 0).all() and (selected.cpu() < E).all()

    def test_moe_index_add_accumulation(self):
        """MoE expert outputs accumulate via index_add_ on Vulkan."""
        N, D, E, K = 8, 16, 4, 2

        # Random expert assignments
        selected = torch.randint(0, E, (N, K))
        weights = F.softmax(torch.randn(N, K), dim=-1)
        expert_outputs = torch.randn(N, K, D)

        ref = torch.zeros(N, D)
        vs = v(torch.zeros(N, D))

        for k in range(K):
            contrib = expert_outputs[:, k] * weights[:, k:k+1]
            ref.index_add_(0, torch.arange(N), contrib)
            vs.index_add_(0, v(torch.arange(N)), v(contrib))

        assert_close(vs, ref)

    def test_moe_gelu_expert_ffn(self):
        """Expert FFN with exact GELU (Gemma-4 uses 'none' variant)."""
        B_S, D, I = 8, 32, 64
        hidden = torch.randn(B_S, D, device="vulkan:0")
        gate_w = torch.randn(I, D, device="vulkan:0")
        up_w = torch.randn(I, D, device="vulkan:0")
        down_w = torch.randn(D, I, device="vulkan:0")

        gate = F.gelu(hidden @ gate_w.T, approximate="none")
        up = hidden @ up_w.T
        out = (gate * up) @ down_w.T

        assert out.shape == (B_S, D)
        assert torch.isfinite(out.cpu()).all()


# ── Full Gemma-4 decoder block (text only) ────────────────────────

class TestGemma4TextDecoder:
    def _rms_norm(self, x, w, eps=1e-6):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

    def _softcap(self, x, cap):
        return cap * torch.tanh(x / cap)

    def _rotate_half(self, x):
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def _apply_rope(self, x, cos, sin):
        return x * cos + self._rotate_half(x) * sin

    def test_gemma4_local_attention_block(self):
        """Sliding-window local attention layer (head_dim=32, window=4)."""
        B, S, D, H, KVH, HD, W = 1, 16, 64, 4, 1, 32, 4
        n_rep = H // KVH

        hidden = torch.randn(B, S, D, device="vulkan:0")
        wq = torch.randn(H * HD, D, device="vulkan:0")
        wk = torch.randn(KVH * HD, D, device="vulkan:0")
        wv = torch.randn(KVH * HD, D, device="vulkan:0")
        wo = torch.randn(D, H * HD, device="vulkan:0")
        norm_w = torch.ones(D, device="vulkan:0")

        hn = self._rms_norm(hidden, norm_w)
        q = (hn @ wq.T).view(B, S, H, HD).transpose(1, 2)
        k = (hn @ wk.T).view(B, S, KVH, HD).transpose(1, 2)
        vt = (hn @ wv.T).view(B, S, KVH, HD).transpose(1, 2)

        # GQA expansion
        k = k.unsqueeze(2).expand(B, KVH, n_rep, S, HD).reshape(B, H, S, HD)
        vt = vt.unsqueeze(2).expand(B, KVH, n_rep, S, HD).reshape(B, H, S, HD)

        # Sliding-window causal mask
        full_inf = torch.full((S, S), float("-inf"), device="vulkan:0")
        mask = torch.triu(full_inf, 1) + torch.tril(full_inf, -W)

        scale = HD ** -0.5
        scores = self._softcap((q @ k.transpose(-2, -1)) * scale, 30.0) + mask
        weights = F.softmax(scores, dim=-1)
        attn_out = (weights @ vt).transpose(1, 2).reshape(B, S, H * HD)
        out = hidden + attn_out @ wo.T

        assert out.shape == (B, S, D)
        assert torch.isfinite(out.cpu()).all()

    def test_gemma4_moe_ffn_block(self):
        """MoE FFN block: top-2 routing + expert GELU FFN + index_add_."""
        B, S, D = 1, 8, 64
        E, K, I = 4, 2, 128
        hidden = torch.randn(B * S, D, device="vulkan:0")
        norm_w = torch.ones(D, device="vulkan:0")
        router_w = torch.randn(E, D, device="vulkan:0")

        hn = self._rms_norm(hidden, norm_w)
        router_logits = hn @ router_w.T
        routing_weights = F.softmax(router_logits, dim=-1)
        routing_weights, selected = torch.topk(routing_weights, K, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(-1, keepdim=True)

        output = torch.zeros_like(hn)
        for expert_idx in range(E):
            gate_w = torch.randn(I, D, device="vulkan:0")
            up_w = torch.randn(I, D, device="vulkan:0")
            down_w = torch.randn(D, I, device="vulkan:0")
            # find tokens routed here
            token_mask = (selected == expert_idx).any(dim=-1)
            if not token_mask.cpu().any():
                continue
            token_ids_cpu = token_mask.cpu().nonzero(as_tuple=True)[0]
            token_ids = token_ids_cpu.to("vulkan:0")
            expert_in = hn[token_ids]
            expert_out = F.gelu(expert_in @ gate_w.T, approximate="none") * (expert_in @ up_w.T)
            expert_out = expert_out @ down_w.T
            output.index_add_(0, token_ids, expert_out)

        output = output.view(B, S, D)
        assert output.shape == (B, S, D)
        assert torch.isfinite(output.cpu()).all()

    def test_gemma4_full_block_training_step(self):
        """End-to-end Gemma-4-style block: fwd + bwd + optimizer step."""
        B, S, D, H, KVH, HD = 1, 8, 64, 4, 1, 16
        W = 4  # sliding window

        class Gemma4Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = nn.Parameter(torch.ones(D))
                self.norm2 = nn.Parameter(torch.ones(D))
                self.wq = nn.Linear(D, H * HD, bias=False)
                self.wk = nn.Linear(D, KVH * HD, bias=False)
                self.wv = nn.Linear(D, KVH * HD, bias=False)
                self.wo = nn.Linear(H * HD, D, bias=False)
                self.gate = nn.Linear(D, D * 2, bias=False)
                self.up   = nn.Linear(D, D * 2, bias=False)
                self.down = nn.Linear(D * 2, D, bias=False)

            def rms_norm(self, x, w):
                return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w

            def forward(self, h):
                hn = self.rms_norm(h, self.norm1)
                q = self.wq(hn).view(B, S, H, HD).transpose(1, 2)
                k = self.wk(hn).view(B, S, KVH, HD).transpose(1, 2)
                vt = self.wv(hn).view(B, S, KVH, HD).transpose(1, 2)
                n_rep = H // KVH
                k = k.unsqueeze(2).expand(B, KVH, n_rep, S, HD).reshape(B, H, S, HD)
                vt = vt.unsqueeze(2).expand(B, KVH, n_rep, S, HD).reshape(B, H, S, HD)
                full_inf = torch.full((S, S), float("-inf"), device=h.device)
                mask = torch.triu(full_inf, 1) + torch.tril(full_inf, -W)
                scores = (q @ k.transpose(-2, -1)) * (HD ** -0.5)
                scores = 30.0 * torch.tanh(scores / 30.0) + mask
                attn = (F.softmax(scores, dim=-1) @ vt).transpose(1, 2).reshape(B, S, H * HD)
                h = h + self.wo(attn)
                hn2 = self.rms_norm(h, self.norm2)
                return h + self.down(F.gelu(self.gate(hn2), approximate="none") * self.up(hn2))

        block = Gemma4Block().to("vulkan:0")
        opt = torch.optim.SGD(block.parameters(), lr=1e-3)
        h = torch.randn(B, S, D, device="vulkan:0")
        target = torch.randn(B, S, D, device="vulkan:0")

        losses = []
        for _ in range(3):
            opt.zero_grad()
            out = block(h)
            loss = F.mse_loss(out, target)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses}"

    def test_large_vocab_lm_head(self):
        """LM head for Gemma-4's 262,144 vocab (chunked linear)."""
        B, S, D, V = 1, 4, 64, 1024  # scaled-down (real: V=262144)
        hidden = torch.randn(B * S, D, device="vulkan:0")
        lm_weight = torch.randn(V, D, device="vulkan:0")

        logits = hidden @ lm_weight.T
        assert logits.shape == (B * S, V)
        assert torch.isfinite(logits.cpu()).all()
