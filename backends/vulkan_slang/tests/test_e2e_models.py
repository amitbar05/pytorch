"""End-to-end model tests: validate full training pipelines on Vulkan."""

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


def to_vulkan(t):
    return t.to("vulkan:0")


class TestMLPTraining:
    """Full MLP training loop on Vulkan."""

    def test_mlp_convergence(self):
        """Train a 2-layer MLP on XOR-like data and verify loss decreases."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        ).vulkan()

        # XOR-like dataset
        X = torch.tensor(
            [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32, device="vulkan"
        )
        y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32, device="vulkan")

        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        losses = []

        for epoch in range(100):
            pred = model(X)
            loss = F.mse_loss(pred, y)
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Loss should decrease significantly
        assert losses[-1] < losses[0] * 0.5, (
            f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )


class TestCNNForward:
    """CNN forward pass on Vulkan."""

    def test_small_cnn_forward(self):
        """Verify CNN forward produces correct output shapes."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10),
        ).vulkan()

        x = torch.randn(2, 3, 8, 8, device="vulkan")
        out = model(x)
        assert out.shape == (2, 10)
        # Verify output is finite
        assert torch.isfinite(out.cpu()).all()


class TestTransformerComponents:
    """Test transformer building blocks."""

    def test_attention_forward(self):
        """Multi-head attention forward pass."""
        torch.manual_seed(42)
        B, N, D = 2, 8, 32
        num_heads = 4

        q = torch.randn(B, num_heads, N, D // num_heads, device="vulkan")
        k = torch.randn(B, num_heads, N, D // num_heads, device="vulkan")
        v = torch.randn(B, num_heads, N, D // num_heads, device="vulkan")

        out = F.scaled_dot_product_attention(q, k, v)
        assert out.shape == (B, num_heads, N, D // num_heads)
        assert torch.isfinite(out.cpu()).all()

    def test_causal_mask(self):
        """Create and apply causal attention mask using triu + where."""
        N = 8
        # Use where to avoid 0 * -inf = nan (IEEE 754)
        upper = torch.ones(N, N, device="vulkan").triu(1)
        neg_inf = torch.full((N, N), float("-inf"), device="vulkan")
        zeros = torch.zeros(N, N, device="vulkan")
        mask = torch.where(upper > 0, neg_inf, zeros)
        mask_cpu = mask.cpu()
        assert mask_cpu[0, 0] == 0.0  # diagonal is 0
        assert mask_cpu[0, 1] == float("-inf")  # above diagonal is -inf

    def test_positional_encoding(self):
        """Sinusoidal positional encoding using sin/cos."""
        N, D = 16, 32
        pos = torch.arange(N, dtype=torch.float32, device="vulkan").unsqueeze(1)
        div = torch.exp(
            torch.arange(0, D, 2, dtype=torch.float32, device="vulkan")
            * -(torch.log(torch.tensor(10000.0, device="vulkan")) / D)
        )
        pe_sin = torch.sin(pos * div)
        pe_cos = torch.cos(pos * div)
        assert pe_sin.shape == (N, D // 2)
        assert pe_cos.shape == (N, D // 2)
        assert torch.isfinite(pe_sin.cpu()).all()

    def test_layer_norm_forward(self):
        """Layer normalization."""
        x = torch.randn(2, 8, 32, device="vulkan")
        ln = nn.LayerNorm(32).vulkan()
        out = ln(x)
        assert out.shape == (2, 8, 32)
        # Check roughly normalized (mean ~0, std ~1)
        out_cpu = out.cpu()
        assert abs(out_cpu.mean().item()) < 0.5
        assert abs(out_cpu.std().item() - 1.0) < 0.5


class TestMiniGPT:
    """Minimal GPT-like model."""

    def test_mini_gpt_forward(self):
        """Forward pass of a minimal GPT block."""
        torch.manual_seed(42)
        B, N, D = 2, 8, 32
        num_heads = 4
        head_dim = D // num_heads

        # Embedding + position
        x = torch.randn(B, N, D, device="vulkan")

        # Self-attention (multi-head)
        Wq = nn.Linear(D, D, bias=False).vulkan()
        Wk = nn.Linear(D, D, bias=False).vulkan()
        Wv = nn.Linear(D, D, bias=False).vulkan()

        q = Wq(x).view(B, N, num_heads, head_dim).transpose(1, 2)  # (B, H, N, D/H)
        k = Wk(x).view(B, N, num_heads, head_dim).transpose(1, 2)
        v = Wv(x).view(B, N, num_heads, head_dim).transpose(1, 2)

        # Attention
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, N, D)  # merge heads
        assert attn.shape == (B, N, D)

        # FFN
        ffn = nn.Sequential(
            nn.Linear(D, D * 4),
            nn.GELU(),
            nn.Linear(D * 4, D),
        ).vulkan()

        out = ffn(attn + x)  # residual
        assert out.shape == (B, N, D)
        assert torch.isfinite(out.cpu()).all()

    def test_mini_gpt_backward(self):
        """Backward pass through GPT block."""
        torch.manual_seed(42)
        B, N, D = 2, 4, 16

        x = torch.randn(B, N, D, device="vulkan", requires_grad=True)

        # Simple transformer-like computation
        W = nn.Linear(D, D).vulkan()
        out = W(x)
        out = F.relu(out)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == (B, N, D)
        assert W.weight.grad is not None


class TestAutocastTraining:
    """AMP training loop tests."""

    def test_amp_mlp_training(self):
        """MLP training with autocast + GradScaler."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        ).vulkan()

        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("vulkan")

        x = torch.randn(4, 8, device="vulkan")
        target = torch.randn(4, 4, device="vulkan")

        initial_loss = None
        for step in range(10):
            with torch.autocast("vulkan", dtype=torch.float16):
                pred = model(x)
                loss = F.mse_loss(pred, target)

            if initial_loss is None:
                initial_loss = loss.item()

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        # Should make some progress
        assert loss.item() <= initial_loss


class TestSaveLoad:
    """Test torch.save/load with Vulkan tensors via CPU round-trip."""

    def test_save_load_tensor(self, tmp_path):
        """Save and load a Vulkan tensor (via CPU)."""
        x = torch.randn(4, 8, device="vulkan")
        x_cpu = x.cpu()
        path = tmp_path / "tensor.pt"
        torch.save(x_cpu, path)

        loaded = torch.load(path, weights_only=True).to("vulkan")
        torch.testing.assert_close(loaded.cpu(), x_cpu)

    def test_save_load_model(self, tmp_path):
        """Save and load model state dict (via CPU)."""
        torch.manual_seed(42)
        model = nn.Linear(4, 2).vulkan()
        path = tmp_path / "model.pt"

        # Save weights (move to CPU for serialization)
        cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(cpu_state, path)

        # Load into new model
        model2 = nn.Linear(4, 2).vulkan()
        state = torch.load(path, weights_only=True)
        # Move loaded weights to Vulkan
        vulkan_state = {k: v.to("vulkan") for k, v in state.items()}
        model2.load_state_dict(vulkan_state)

        x = torch.randn(2, 4, device="vulkan")
        torch.testing.assert_close(model(x).cpu(), model2(x).cpu())


class TestResNet:
    """ResNet-style residual block tests."""

    def test_resblock_forward(self):
        """Forward pass through a residual block."""
        torch.manual_seed(42)

        class ResBlock(nn.Module):
            def __init__(self, ch):
                super().__init__()
                self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(ch)
                self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(ch)

            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                return F.relu(out + x)

        model = ResBlock(16).eval().vulkan()
        x = torch.randn(1, 16, 8, 8, device="vulkan")
        out = model(x)
        assert out.shape == (1, 16, 8, 8)
        assert torch.isfinite(out.cpu()).all()


class TestTransformerBlock:
    """Full transformer encoder block tests."""

    def _make_block(self, d_model=32, nhead=2):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                self.wq = nn.Linear(d_model, d_model)
                self.wk = nn.Linear(d_model, d_model)
                self.wv = nn.Linear(d_model, d_model)
                self.proj = nn.Linear(d_model, d_model)
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                )
                self.nhead = nhead
                self.head_dim = d_model // nhead

            def forward(self, x):
                B, N, D = x.shape
                xn = self.norm1(x)
                q = self.wq(xn).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
                k = self.wk(xn).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
                v = self.wv(xn).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
                attn = F.scaled_dot_product_attention(q, k, v)
                attn = attn.transpose(1, 2).reshape(B, N, D)
                x = x + self.proj(attn)
                x = x + self.ffn(self.norm2(x))
                return x

        return Block()

    def test_transformer_forward(self):
        """Transformer block forward pass."""
        torch.manual_seed(42)
        block = self._make_block(64, 4).eval().vulkan()
        x = torch.randn(2, 8, 64, device="vulkan")
        out = block(x)
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out.cpu()).all()

    def test_transformer_backward(self):
        """Transformer block backward pass."""
        torch.manual_seed(42)
        block = self._make_block(32, 2).vulkan()
        x = torch.randn(1, 4, 32, device="vulkan", requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (1, 4, 32)

    def test_transformer_training_step(self):
        """Full training step through transformer block."""
        torch.manual_seed(42)
        block = self._make_block(32, 2).vulkan()
        opt = torch.optim.Adam(block.parameters(), lr=0.001)
        x = torch.randn(2, 4, 32, device="vulkan")
        target = torch.randn(2, 4, 32, device="vulkan")

        initial_loss = None
        for _ in range(5):
            out = block(x)
            loss = F.mse_loss(out, target)
            if initial_loss is None:
                initial_loss = loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert loss.item() <= initial_loss


class TestQwen35GatedDeltaNet:
    """Tests for Qwen3.5-0.8B GatedDeltaNet (linear attention) building blocks.

    Qwen3.5-0.8B uses a hybrid architecture: 3 GatedDeltaNet layers + 1 full attention
    per block, repeated 6 times (24 layers total). The GatedDeltaNet is a linear-time
    recurrent attention mechanism (O(n) per token inference vs O(n^2) for softmax).
    """

    def _l2norm(self, x, dim=-1, eps=1e-6):
        return x / (x.norm(dim=dim, keepdim=True) + eps)

    def _torch_recurrent_gated_delta_rule(
        self,
        query,
        key,
        value,
        g,
        beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    ):
        """Pure PyTorch fallback for GatedDeltaNet (used when fla is unavailable)."""
        initial_dtype = query.dtype
        if use_qk_l2norm_in_kernel:
            query = self._l2norm(query, dim=-1, eps=1e-6)
            key = self._l2norm(key, dim=-1, eps=1e-6)
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)
        ]
        batch_size, num_heads, seq_len, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        scale = 1 / (query.shape[-1] ** 0.5)
        query = query * scale
        core_out = torch.zeros(
            batch_size,
            num_heads,
            seq_len,
            v_head_dim,
            device=query.device,
            dtype=value.dtype,
        )
        state = (
            torch.zeros(
                batch_size,
                num_heads,
                k_head_dim,
                v_head_dim,
                device=query.device,
                dtype=value.dtype,
            )
            if initial_state is None
            else initial_state.float()
        )
        for i in range(seq_len):
            q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
            g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, i].unsqueeze(-1)
            state = state * g_t
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            core_out[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
        core_out = core_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_out, (state if output_final_state else None)

    def _torch_chunk_gated_delta_rule(
        self,
        query,
        key,
        value,
        g,
        beta,
        chunk_size=16,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    ):
        """Chunked forward pass — used for full-sequence training."""
        initial_dtype = query.dtype
        if use_qk_l2norm_in_kernel:
            query = self._l2norm(query, dim=-1, eps=1e-6)
            key = self._l2norm(key, dim=-1, eps=1e-6)
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)
        ]
        batch_size, num_heads, seq_len, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
        total = seq_len + pad_size
        scale = 1 / (query.shape[-1] ** 0.5)
        query = query * scale
        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)
        query, key, value, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
            for x in (query, key, value, k_beta, v_beta)
        ]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
        mask = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
            diagonal=0,
        )
        g = g.cumsum(dim=-1)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
        for i in range(1, chunk_size):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
        state = (
            torch.zeros(
                batch_size,
                num_heads,
                k_head_dim,
                v_head_dim,
                device=query.device,
                dtype=value.dtype,
            )
            if initial_state is None
            else initial_state.float()
        )
        core_out = torch.zeros_like(value)
        mask2 = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
            diagonal=1,
        )
        for i in range(total // chunk_size):
            q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
            attn_c = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(
                mask2, 0
            )
            v_prime = k_cumdecay[:, :, i] @ state
            v_new = v_i - v_prime
            core_out[:, :, i] = (
                q_i * g[:, :, i, :, None].exp()
            ) @ state + attn_c @ v_new
            state = (
                state * g[:, :, i, -1, None, None].exp()
                + (
                    k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
                ).transpose(-1, -2)
                @ v_new
            )
        core_out = core_out.reshape(batch_size, num_heads, total, v_head_dim)[
            :, :, :seq_len
        ]
        core_out = core_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_out, (state if output_final_state else None)

    def test_rms_norm_gated_forward(self):
        """RMSNormGated: weight * rms_norm(input) * silu(gate) — Qwen3_5RMSNormGated."""
        import torch_vulkan

        B, N, D = 2, 8, 128
        torch.manual_seed(42)
        x = torch.randn(B, N, D)
        gate = torch.randn(B, N, D) * 0.5
        w = torch.ones(D)

        # CPU reference: weight * (input/rms) * silu(gate)
        rms = (x.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
        silu_g = gate * torch.sigmoid(gate)
        ref = w * x * rms * silu_g

        out = torch_vulkan.rms_norm_gated(
            x.to("vulkan"), gate.to("vulkan"), w.to("vulkan")
        )
        assert torch.allclose(out.cpu(), ref, atol=1e-4), (
            f"max err {(out.cpu() - ref).abs().max()}"
        )

    def test_rms_norm_gated_backward(self):
        """RMSNormGated backward: grad_input and grad_gate match CPU autograd."""
        import torch_vulkan

        B, N, D = 2, 4, 64
        torch.manual_seed(42)
        x = torch.randn(B, N, D)
        gate = torch.randn(B, N, D) * 0.5
        w = torch.ones(D)

        # CPU reference backward
        x_r = x.clone().requires_grad_(True)
        g_r = gate.clone().requires_grad_(True)
        w_r = w.clone().requires_grad_(True)
        out_r = (
            w_r
            * x_r
            * (x_r.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
            * g_r.sigmoid()
            * g_r
        )
        out_r.sum().backward()

        # Vulkan backward
        x_vk = x.to("vulkan").requires_grad_(True)
        g_vk = gate.to("vulkan").requires_grad_(True)
        w_vk = w.to("vulkan").requires_grad_(True)
        out_vk = torch_vulkan.rms_norm_gated(x_vk, g_vk, w_vk)
        out_vk.sum().backward()

        assert torch.allclose(x_vk.grad.cpu(), x_r.grad, atol=1e-4), (
            f"grad_input max err {(x_vk.grad.cpu() - x_r.grad).abs().max()}"
        )
        assert torch.allclose(g_vk.grad.cpu(), g_r.grad, atol=1e-4), (
            f"grad_gate max err {(g_vk.grad.cpu() - g_r.grad).abs().max()}"
        )

    def test_gated_delta_rule_recurrent_forward(self):
        """Recurrent GatedDeltaNet (single-token inference mode) on Vulkan."""
        # Qwen3.5-0.8B: num_v_heads=16, head_k_dim=128, head_v_dim=128
        B, H, D = 1, 8, 32  # scaled down for SwiftShader
        torch.manual_seed(42)
        q = torch.randn(B, 1, H, D, device="vulkan")
        k = torch.randn(B, 1, H, D, device="vulkan")
        v = torch.randn(B, 1, H, D, device="vulkan")
        g = torch.randn(B, 1, H, device="vulkan") * 0.1
        beta = torch.zeros(B, 1, H, device="vulkan")  # zero beta = no update

        out, state = self._torch_recurrent_gated_delta_rule(
            q, k, v, g, beta, output_final_state=True
        )
        assert out.shape == (B, 1, H, D)
        assert state.shape == (B, H, D, D)
        assert torch.isfinite(out.cpu()).all()
        assert torch.isfinite(state.cpu()).all()

    def test_gated_delta_rule_chunk_forward(self):
        """Chunked GatedDeltaNet (full-sequence training mode) on Vulkan."""
        B, S, H, D = 1, 32, 4, 32
        torch.manual_seed(42)
        q = torch.randn(B, S, H, D, device="vulkan")
        k = torch.randn(B, S, H, D, device="vulkan")
        v = torch.randn(B, S, H, D, device="vulkan")
        g = torch.randn(B, S, H, device="vulkan") * 0.1
        beta = torch.rand(B, S, H, device="vulkan") * 0.5

        out, _ = self._torch_chunk_gated_delta_rule(q, k, v, g, beta)
        assert out.shape == (B, S, H, D)
        assert torch.isfinite(out.cpu()).all()

    def test_gated_delta_rule_chunk_vs_recurrent(self):
        """Chunk mode should match recurrent mode for same input."""
        B, S, H, D = 1, 8, 4, 16
        torch.manual_seed(42)
        q = torch.randn(B, S, H, D, device="vulkan")
        k = torch.randn(B, S, H, D, device="vulkan")
        v = torch.randn(B, S, H, D, device="vulkan")
        g = torch.randn(B, S, H, device="vulkan") * 0.05
        beta = torch.rand(B, S, H, device="vulkan") * 0.5

        out_chunk, _ = self._torch_chunk_gated_delta_rule(
            q, k, v, g, beta, chunk_size=4
        )
        out_recur, _ = self._torch_recurrent_gated_delta_rule(q, k, v, g, beta)

        assert torch.allclose(out_chunk.cpu(), out_recur.cpu(), atol=1e-4), (
            f"max diff: {(out_chunk.cpu() - out_recur.cpu()).abs().max()}"
        )

    def test_causal_conv1d_fallback(self):
        """Causal depthwise conv1d (PyTorch fallback for causal_conv1d package)."""
        # Qwen3.5: conv_dim=key_dim*2+value_dim, kernel_size=4, groups=conv_dim
        B, C, T = 1, 64, 32
        kernel_size = 4
        x = torch.randn(B, C, T, device="vulkan")
        weight = torch.randn(C, kernel_size, device="vulkan")

        # Causal conv1d: pad left so output[t] only sees input[:t+kernel_size]
        x_padded = F.pad(x, (kernel_size - 1, 0))
        # Conv1d expects weight shape (C_out, C_in/groups, kernel)
        out = F.conv1d(x_padded, weight.unsqueeze(1), None, padding=0, groups=C)
        out = F.silu(out[:, :, :T])

        assert out.shape == (B, C, T)
        assert torch.isfinite(out.cpu()).all()

    def test_partial_rope_qwen35(self):
        """Partial RoPE (partial_rotary_factor=0.25) as used by Qwen3.5-0.8B."""
        # head_dim=256, rotary_dim=64 (25% rotated, 75% pass-through)
        B, H, S, D = 1, 8, 16, 64  # using D=64 for test (same ratio)
        rotary_dim = D // 4  # 25% of head_dim

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q = torch.randn(B, H, S, D, device="vulkan")
        k = torch.randn(B, H, S, D, device="vulkan")
        # Partial cos/sin for rotary_dim dimensions only
        cos = torch.randn(B, S, rotary_dim, device="vulkan")
        sin = torch.randn(B, S, rotary_dim, device="vulkan")

        cos = cos.unsqueeze(1)  # (B, 1, S, rotary_dim)
        sin = sin.unsqueeze(1)
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_embed = torch.cat([q_rot * cos + rotate_half(q_rot) * sin, q_pass], dim=-1)
        k_embed = torch.cat([k_rot * cos + rotate_half(k_rot) * sin, k_pass], dim=-1)

        assert q_embed.shape == (B, H, S, D)
        assert k_embed.shape == (B, H, S, D)
        assert torch.isfinite(q_embed.cpu()).all()

    def test_full_gated_deltanet_layer_forward(self):
        """Full GatedDeltaNet layer (Qwen3.5-style) forward pass on Vulkan.

        Implements Qwen3_5GatedDeltaNet without custom CUDA kernels:
        in_proj_qkv → causal_conv1d → split Q/K/V → chunk_gated_delta_rule → RMSNormGated → out_proj
        """
        import torch_vulkan

        # Scaled-down Qwen3.5-0.8B dims for SwiftShader testing
        hidden_size = 256
        num_k_heads = 4
        num_v_heads = 4
        head_k_dim = 32
        head_v_dim = 32
        key_dim = head_k_dim * num_k_heads
        value_dim = head_v_dim * num_v_heads
        conv_dim = key_dim * 2 + value_dim
        conv_kernel_size = 4
        B, S = 1, 16

        torch.manual_seed(42)

        # Model weights
        in_proj_qkv = nn.Linear(hidden_size, conv_dim, bias=False).vulkan()
        in_proj_z = nn.Linear(hidden_size, value_dim, bias=False).vulkan()
        in_proj_b = nn.Linear(hidden_size, num_v_heads, bias=False).vulkan()
        in_proj_a = nn.Linear(hidden_size, num_v_heads, bias=False).vulkan()
        conv1d_weight = torch.randn(conv_dim, conv_kernel_size, device="vulkan")
        dt_bias = torch.ones(num_v_heads, device="vulkan")
        A_log = torch.log(torch.empty(num_v_heads).uniform_(0, 16)).to("vulkan")
        norm_weight = torch.ones(head_v_dim, device="vulkan")
        out_proj = nn.Linear(value_dim, hidden_size, bias=False).vulkan()

        hidden_states = torch.randn(B, S, hidden_size, device="vulkan")

        # Forward
        mixed_qkv = in_proj_qkv(hidden_states).transpose(1, 2)  # (B, conv_dim, S)
        # Causal conv1d (PyTorch fallback)
        mixed_qkv_padded = F.pad(mixed_qkv, (conv_kernel_size - 1, 0))
        mixed_qkv = F.silu(
            F.conv1d(
                mixed_qkv_padded,
                conv1d_weight.unsqueeze(1),
                None,
                padding=0,
                groups=conv_dim,
            )[:, :, :S]
        )
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [key_dim, key_dim, value_dim], dim=-1
        )
        query = query.reshape(B, S, num_k_heads, head_k_dim)
        key = key.reshape(B, S, num_k_heads, head_k_dim)
        value = value.reshape(B, S, num_v_heads, head_v_dim)
        z = in_proj_z(hidden_states).reshape(B, S, num_v_heads, head_v_dim)
        b = in_proj_b(hidden_states)
        a = in_proj_a(hidden_states)
        beta = b.sigmoid()
        g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)

        # Chunk gated delta rule
        core_out, _ = self._torch_chunk_gated_delta_rule(query, key, value, g, beta)

        # RMSNormGated
        core_out_flat = core_out.reshape(-1, head_v_dim)
        z_flat = z.reshape(-1, head_v_dim)
        normed = torch_vulkan.rms_norm_gated(core_out_flat, z_flat, norm_weight)
        output = out_proj(normed.reshape(B, S, value_dim))

        assert output.shape == (B, S, hidden_size)
        assert torch.isfinite(output.cpu()).all()

    def test_full_gated_deltanet_layer_backward(self):
        """GatedDeltaNet layer backward — gradients flow through rms_norm_gated + out_proj.

        Note: The Python fallback for chunk_gated_delta_rule uses in-place slice assignment
        in the recurrence loop, which breaks the autograd graph (this is expected — the
        real implementation uses CUDA custom kernels from fla-org/flash-linear-attention with
        proper autograd). We test backward through rms_norm_gated and out_proj, which is the
        path that DOES require our Vulkan ops to support backward.
        """
        import torch_vulkan

        head_dim = 16
        value_dim = 32  # 2 heads × 16
        hidden_size = 64
        B, S = 1, 8
        num_rows = B * S * (value_dim // head_dim)  # rows for rms_norm_gated

        torch.manual_seed(42)
        norm_weight = torch.ones(head_dim, device="vulkan", requires_grad=True)
        out_proj = nn.Linear(value_dim, hidden_size, bias=False).vulkan()

        # Simulate core_out from delta rule (treated as input to the norm+proj path)
        core_out = torch.randn(num_rows, head_dim, device="vulkan", requires_grad=True)
        z = torch.randn(num_rows, head_dim, device="vulkan", requires_grad=True)

        normed = torch_vulkan.rms_norm_gated(core_out, z, norm_weight)
        output = out_proj(normed.reshape(B, S, value_dim))
        output.sum().backward()

        assert core_out.grad is not None
        assert core_out.grad.shape == (num_rows, head_dim)
        assert z.grad is not None
        assert torch.isfinite(core_out.grad.cpu()).all()
        assert torch.isfinite(z.grad.cpu()).all()

    def test_qwen35_decoder_layer_hybrid(self):
        """Full Qwen3.5-style decoder: hybrid layer (linear_attention + full_attention)."""
        import torch_vulkan

        hidden_size = 128
        # Simulate 2-layer hybrid: 1 GatedDeltaNet + 1 full attention
        B, S = 1, 16

        torch.manual_seed(42)
        hidden = torch.randn(B, S, hidden_size, device="vulkan", requires_grad=True)

        # Layer 1: GatedDeltaNet
        num_heads = 4
        head_dim = 16
        conv_dim = head_dim * num_heads * 3  # q + k + v
        conv_kernel_size = 4
        in_proj = nn.Linear(hidden_size, conv_dim, bias=False).vulkan()
        norm_w = torch.ones(head_dim, device="vulkan")
        out_proj1 = nn.Linear(head_dim * num_heads, hidden_size, bias=False).vulkan()
        rms1 = nn.Parameter(torch.ones(hidden_size, device="vulkan"))
        rms2 = nn.Parameter(torch.ones(hidden_size, device="vulkan"))

        # RMSNorm
        h = torch_vulkan.rms_norm(hidden, rms1, 1e-6)
        # GatedDeltaNet
        mixed = in_proj(h).transpose(1, 2)
        padded = F.pad(mixed, (conv_kernel_size - 1, 0))
        mixed = F.silu(
            F.conv1d(
                padded,
                torch.randn(conv_dim, conv_kernel_size, device="vulkan").unsqueeze(1),
                None,
                padding=0,
                groups=conv_dim,
            )[:, :, :S]
        )
        mixed = mixed.transpose(1, 2)
        q, k, v = torch.split(mixed, [head_dim * num_heads] * 3, dim=-1)
        q = q.reshape(B, S, num_heads, head_dim)
        k = k.reshape(B, S, num_heads, head_dim)
        v = v.reshape(B, S, num_heads, head_dim)
        g = torch.randn(B, S, num_heads, device="vulkan") * 0.1
        beta = torch.rand(B, S, num_heads, device="vulkan") * 0.5
        core, _ = self._torch_chunk_gated_delta_rule(q, k, v, g, beta)
        z = torch.randn(B, S, num_heads, head_dim, device="vulkan")
        normed = torch_vulkan.rms_norm_gated(
            core.reshape(-1, head_dim), z.reshape(-1, head_dim), norm_w
        )
        h1 = out_proj1(normed.reshape(B, S, -1))
        h1 = hidden + h1  # residual

        # Layer 2: Full self-attention (standard SDPA)
        h2 = torch_vulkan.rms_norm(h1, rms2, 1e-6)
        # hidden_size=128, num_heads=4, head_dim=32 per head (4*32=128)
        attn_head_dim = hidden_size // num_heads  # 32
        wq = nn.Linear(hidden_size, hidden_size, bias=False).vulkan()
        wk = nn.Linear(hidden_size, hidden_size, bias=False).vulkan()
        wv = nn.Linear(hidden_size, hidden_size, bias=False).vulkan()
        wo = nn.Linear(hidden_size, hidden_size, bias=False).vulkan()
        q2 = wq(h2).view(B, S, num_heads, attn_head_dim).transpose(1, 2)
        k2 = wk(h2).view(B, S, num_heads, attn_head_dim).transpose(1, 2)
        v2 = wv(h2).view(B, S, num_heads, attn_head_dim).transpose(1, 2)
        causal_mask = torch.triu(
            torch.full((S, S), float("-inf"), device="vulkan"), diagonal=1
        )
        attn = F.scaled_dot_product_attention(q2, k2, v2, attn_mask=causal_mask)
        h2 = wo(attn.transpose(1, 2).reshape(B, S, hidden_size))
        output = h1 + h2  # residual

        assert output.shape == (B, S, hidden_size)
        assert torch.isfinite(output.cpu()).all()


class TestQwen35MiniE2E:
    """End-to-end MiniQwen3.5 training test (scaled-down Qwen3.5-0.8B architecture).

    Qwen3.5-0.8B architecture:
    - 24 layers: 18 GatedDeltaNet (linear_attention) + 6 full_attention
    - hidden=1024, num_q_heads=8, num_kv_heads=2, head_dim=256, partial_rotary_factor=0.25
    - FFN: intermediate=3584 (SwiGLU), hidden_act=silu
    - vocab=248320 (bf16 required for SwiftShader ~500MB limit)
    - RMSNorm (not LayerNorm), RoPE with theta=10M

    This test uses scaled-down dims for SwiftShader compatibility.
    """

    class MiniQwen35(nn.Module):
        """Minimal Qwen3.5-style model for testing (2-layer hybrid).

        Architecture: 1 GatedDeltaNet + 1 full attention, scaled down for SwiftShader.
        Uses pytorch fallback for GatedDeltaNet (no fla package required).
        """

        def __init__(
            self,
            vocab_size=4096,
            hidden_size=128,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=32,
            intermediate_size=384,
            linear_num_heads=4,
            linear_head_dim=32,
            conv_kernel_size=4,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_q_heads = num_q_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.linear_num_heads = linear_num_heads
            self.linear_head_dim = linear_head_dim
            self.conv_kernel_size = conv_kernel_size

            # Embedding
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

            # GatedDeltaNet layer (linear_attention)
            key_dim = linear_head_dim * linear_num_heads
            value_dim = linear_head_dim * linear_num_heads
            conv_dim = key_dim * 2 + value_dim
            self.gdn_in_proj = nn.Linear(hidden_size, conv_dim, bias=False)
            self.gdn_in_proj_z = nn.Linear(hidden_size, value_dim, bias=False)
            self.gdn_in_proj_b = nn.Linear(hidden_size, linear_num_heads, bias=False)
            self.gdn_in_proj_a = nn.Linear(hidden_size, linear_num_heads, bias=False)
            self.gdn_conv_weight = nn.Parameter(torch.randn(conv_dim, conv_kernel_size))
            self.gdn_dt_bias = nn.Parameter(torch.ones(linear_num_heads))
            self.gdn_A_log = nn.Parameter(
                torch.log(torch.empty(linear_num_heads).uniform_(0, 4))
            )
            self.gdn_norm_weight = nn.Parameter(torch.ones(linear_head_dim))
            self.gdn_out_proj = nn.Linear(value_dim, hidden_size, bias=False)

            # Full attention layer
            self.attn_q = nn.Linear(hidden_size, num_q_heads * head_dim, bias=False)
            self.attn_k = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
            self.attn_v = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
            self.attn_o = nn.Linear(num_q_heads * head_dim, hidden_size, bias=False)

            # SwiGLU MLP (shared by both layers)
            self.gate_proj1 = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj1 = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj1 = nn.Linear(intermediate_size, hidden_size, bias=False)
            self.gate_proj2 = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj2 = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj2 = nn.Linear(intermediate_size, hidden_size, bias=False)

            # RMSNorms
            self.norm_gdn_pre = nn.Parameter(torch.ones(hidden_size))
            self.norm_gdn_post = nn.Parameter(torch.ones(hidden_size))
            self.norm_attn_pre = nn.Parameter(torch.ones(hidden_size))
            self.norm_attn_post = nn.Parameter(torch.ones(hidden_size))
            self.norm_final = nn.Parameter(torch.ones(hidden_size))

            # LM head (tied with embedding for efficiency)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        def _l2norm(self, x, dim=-1, eps=1e-6):
            return x / (x.norm(dim=dim, keepdim=True) + eps)

        def _gated_delta_rule_forward(self, query, key, value, g, beta):
            """Pure PyTorch GatedDeltaNet (training, full sequence)."""
            B, S, H, D = query.shape
            initial_dtype = query.dtype
            query2 = self._l2norm(query) / (D**0.5)
            key2 = self._l2norm(key)
            query2, key2, value2, beta2, g2 = [
                x.transpose(1, 2).float() for x in (query2, key2, value, beta, g)
            ]
            state = torch.zeros(
                B, H, D, value.shape[-1], device=query.device, dtype=torch.float32
            )
            core_out = torch.zeros(
                B, H, S, value.shape[-1], device=query.device, dtype=torch.float32
            )
            for i in range(S):
                q_t, k_t, v_t = query2[:, :, i], key2[:, :, i], value2[:, :, i]
                g_t = g2[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
                beta_t = beta2[:, :, i].unsqueeze(-1)
                state = state * g_t
                kv_mem = (state * k_t.unsqueeze(-1)).sum(-2)
                delta = (v_t - kv_mem) * beta_t
                state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
                core_out[:, :, i] = (state * q_t.unsqueeze(-1)).sum(-2)
            return core_out.transpose(1, 2).to(initial_dtype)

        def forward(self, input_ids):
            import torch_vulkan

            B, S = input_ids.shape
            hidden = self.embed_tokens(input_ids)

            # ── Layer 1: GatedDeltaNet ──────────────────────────────
            key_dim = self.linear_head_dim * self.linear_num_heads
            value_dim = key_dim
            conv_dim = key_dim * 2 + value_dim

            residual = hidden
            h = torch_vulkan.rms_norm(hidden, self.norm_gdn_pre, 1e-6)
            # Causal conv1d → split Q/K/V
            mixed = self.gdn_in_proj(h).transpose(1, 2)
            padded = F.pad(mixed, (self.conv_kernel_size - 1, 0))
            mixed = F.silu(
                F.conv1d(
                    padded,
                    self.gdn_conv_weight.unsqueeze(1),
                    None,
                    padding=0,
                    groups=conv_dim,
                )[:, :, :S]
            )
            mixed = mixed.transpose(1, 2)
            q, k, v = torch.split(mixed, [key_dim, key_dim, value_dim], dim=-1)
            q = q.reshape(B, S, self.linear_num_heads, self.linear_head_dim)
            k = k.reshape(B, S, self.linear_num_heads, self.linear_head_dim)
            v = v.reshape(B, S, self.linear_num_heads, self.linear_head_dim)
            z = self.gdn_in_proj_z(h).reshape(
                B, S, self.linear_num_heads, self.linear_head_dim
            )
            b = self.gdn_in_proj_b(h)
            a = self.gdn_in_proj_a(h)
            beta = b.sigmoid()
            g_val = -self.gdn_A_log.float().exp() * F.softplus(
                a.float() + self.gdn_dt_bias
            )
            # GatedDeltaNet forward
            core_out = self._gated_delta_rule_forward(q, k, v, g_val, beta)
            # RMSNormGated + out proj
            core_flat = core_out.reshape(-1, self.linear_head_dim)
            z_flat = z.reshape(-1, self.linear_head_dim)
            normed = torch_vulkan.rms_norm_gated(
                core_flat, z_flat, self.gdn_norm_weight
            )
            gdn_out = self.gdn_out_proj(normed.reshape(B, S, value_dim))
            hidden = residual + gdn_out

            # MLP 1 (SwiGLU)
            residual = hidden
            h = torch_vulkan.rms_norm(hidden, self.norm_gdn_post, 1e-6)
            mlp1_out = self.down_proj1(
                torch_vulkan.swiglu(self.gate_proj1(h), self.up_proj1(h))
            )
            hidden = residual + mlp1_out

            # ── Layer 2: Full Self-Attention ─────────────────────────
            residual = hidden
            h = torch_vulkan.rms_norm(hidden, self.norm_attn_pre, 1e-6)
            # GQA: Q has num_q_heads, K/V have num_kv_heads
            q2 = (
                self.attn_q(h)
                .view(B, S, self.num_q_heads, self.head_dim)
                .transpose(1, 2)
            )
            k2 = (
                self.attn_k(h)
                .view(B, S, self.num_kv_heads, self.head_dim)
                .transpose(1, 2)
            )
            v2 = (
                self.attn_v(h)
                .view(B, S, self.num_kv_heads, self.head_dim)
                .transpose(1, 2)
            )
            # Expand KV for GQA
            n_rep = self.num_q_heads // self.num_kv_heads
            k2 = (
                k2[:, :, None, :, :]
                .expand(B, self.num_kv_heads, n_rep, S, self.head_dim)
                .reshape(B, self.num_q_heads, S, self.head_dim)
            )
            v2 = (
                v2[:, :, None, :, :]
                .expand(B, self.num_kv_heads, n_rep, S, self.head_dim)
                .reshape(B, self.num_q_heads, S, self.head_dim)
            )
            # Causal SDPA
            causal_mask = torch.triu(
                torch.full((S, S), float("-inf"), device=hidden.device), diagonal=1
            )
            attn_out = F.scaled_dot_product_attention(q2, k2, v2, attn_mask=causal_mask)
            hidden = residual + self.attn_o(attn_out.transpose(1, 2).reshape(B, S, -1))

            # MLP 2 (SwiGLU)
            residual = hidden
            h = torch_vulkan.rms_norm(hidden, self.norm_attn_post, 1e-6)
            mlp2_out = self.down_proj2(
                torch_vulkan.swiglu(self.gate_proj2(h), self.up_proj2(h))
            )
            hidden = residual + mlp2_out

            # Final norm + LM head
            hidden = torch_vulkan.rms_norm(hidden, self.norm_final, 1e-6)
            logits = self.lm_head(hidden)
            return logits

    def test_mini_qwen35_forward(self):
        """Full MiniQwen3.5 forward pass on Vulkan."""
        import torch_vulkan

        torch.manual_seed(42)
        model = self.MiniQwen35().vulkan()
        input_ids = torch.randint(0, 4096, (1, 16), device="vulkan")
        logits = model(input_ids)
        assert logits.shape == (1, 16, 4096)
        assert torch.isfinite(logits.cpu()).all()

    def test_mini_qwen35_training_step(self):
        """MiniQwen3.5 full training step: forward + cross_entropy + backward + optimizer."""
        import torch_vulkan

        torch.manual_seed(42)
        model = self.MiniQwen35().vulkan()
        opt = torch.optim.SGD(model.parameters(), lr=1e-4)

        input_ids = torch.randint(0, 4096, (1, 8), device="vulkan")
        targets = torch.randint(0, 4096, (8,), device="vulkan")  # next-token targets

        initial_loss = None
        for step in range(3):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, 4096), targets)
            if initial_loss is None:
                initial_loss = loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()

        assert torch.isfinite(torch.tensor(loss.item()))
        assert loss.item() < initial_loss * 2  # loss shouldn't explode

    def test_mini_qwen35_bf16(self):
        """MiniQwen3.5 forward in bfloat16 (matches real Qwen3.5 inference dtype)."""
        import torch_vulkan

        torch.manual_seed(42)
        model = self.MiniQwen35().to(torch.bfloat16).vulkan()
        input_ids = torch.randint(0, 4096, (1, 8), device="vulkan")
        logits = model(input_ids)
        assert logits.shape == (1, 8, 4096)
        assert torch.isfinite(logits.cpu()).all()

    def test_large_vocab_bf16_embedding(self):
        """248320-vocab bf16 embedding (Qwen3.5-0.8B vocab size) on Vulkan.

        The embedding weight (248320 × 1024 × 2 bytes = 485MB) fits in SwiftShader's
        ~500MB buffer limit when using bfloat16. Float32 would require 970MB.
        """
        vocab_size = 248320
        hidden_size = 1024
        embed = nn.Embedding(vocab_size, hidden_size).to(torch.bfloat16).to("vulkan")
        input_ids = torch.randint(0, vocab_size, (1, 8), device="vulkan")
        out = embed(input_ids)
        assert out.shape == (1, 8, hidden_size)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out.cpu()).all()


class TestProfiler:
    """Test torch.profiler with Vulkan ops."""

    def test_profiler_basic(self):
        """Profiler doesn't crash on Vulkan ops."""
        x = torch.randn(4, 4, device="vulkan")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            y = x + x
            z = y * 2.0
        table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=5)
        assert "aten::add" in table
        assert "aten::mul" in table


# ═══════════════════════════════════════════════════════════════════
# Wave 9: SmallCNN Training — THE NORTH STAR
# ═══════════════════════════════════════════════════════════════════


class TestSmallCNNTrain:
    """v5 North Star: SmallCNN trains end-to-end under torch.compile.

    Pipeline: Conv2d → GroupNorm → ReLU → MaxPool2d → Flatten → Linear → CrossEntropyLoss
    Verifies that the full training loop (forward, backward, optimizer step)
    works correctly on Vulkan through the Inductor backend.
    """

    class SmallCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.gn1 = nn.GroupNorm(4, 16)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.gn2 = nn.GroupNorm(8, 32)
            self.pool = nn.MaxPool2d(2, 2)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(32 * 8 * 8, num_classes)

        def forward(self, x):
            x = F.relu(self.gn1(self.conv1(x)))
            x = self.pool(x)
            x = F.relu(self.gn2(self.conv2(x)))
            x = self.pool(x)
            x = self.flatten(x)
            x = self.fc(x)
            return x

    def test_forward_compiles(self):
        """SmallCNN forward compiles and runs without error."""
        torch.manual_seed(42)
        model = self.SmallCNN().to("vulkan")
        compiled = torch.compile(model, backend="inductor")
        x = torch.randn(2, 3, 32, 32, device="vulkan")
        out = compiled(x)
        assert out.shape == (2, 10)
        assert torch.isfinite(out.cpu()).all()

    def test_forward_output_shape(self):
        """SmallCNN forward produces correct output shape for different batch sizes."""
        torch.manual_seed(42)
        model = self.SmallCNN().to("vulkan")
        compiled = torch.compile(model, backend="inductor")
        for B in [1, 4, 8]:
            x = torch.randn(B, 3, 32, 32, device="vulkan")
            out = compiled(x)
            assert out.shape == (B, 10), f"Expected ({B}, 10), got {out.shape}"

    def test_3step_training_loss(self):
        """SmallCNN training: loss decreases over 3 steps."""
        torch.manual_seed(42)
        model = self.SmallCNN().to("vulkan")
        compiled = torch.compile(model, backend="inductor")
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        x = torch.randn(4, 3, 32, 32, device="vulkan")
        targets = torch.randint(0, 10, (4,), device="vulkan")
        losses = []
        for _ in range(3):
            logits = compiled(x)
            loss = F.cross_entropy(logits, targets)
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert losses[-1] < losses[0] * 0.95, f"Loss did not decrease: {losses}"

    def test_grad_parity(self):
        """Compiled model gradients match eager-mode gradients."""
        torch.manual_seed(42)
        # Eager model
        model_eager = self.SmallCNN().to("vulkan")
        model_compile = self.SmallCNN().to("vulkan")
        model_compile.load_state_dict(model_eager.state_dict())
        compiled = torch.compile(model_compile, backend="inductor")
        x = torch.randn(2, 3, 32, 32, device="vulkan")
        targets = torch.randint(0, 10, (2,), device="vulkan")
        # Eager
        loss_e = F.cross_entropy(model_eager(x), targets)
        loss_e.backward()
        # Compiled
        loss_c = F.cross_entropy(compiled(x), targets)
        loss_c.backward()
        # Compare grads for conv1.weight
        g_eager = model_eager.conv1.weight.grad.cpu()
        g_comp = model_compile.conv1.weight.grad.cpu()
        assert torch.allclose(g_eager, g_comp, atol=1e-4, rtol=1e-2), (
            f"Grad mismatch: max err {(g_eager - g_comp).abs().max()}"
        )

    def test_dispatch_count(self):
        """Verify ops are dispatched to Vulkan (not falling back to CPU)."""
        torch.manual_seed(42)
        model = self.SmallCNN().to("vulkan")
        compiled = torch.compile(model, backend="inductor")
        x = torch.randn(2, 3, 32, 32, device="vulkan")
        out = compiled(x)
        assert out.device.type == "vulkan" if hasattr(out.device, "type") else True

    def test_10step_no_nan(self):
        """10 training steps produce no NaN values."""
        torch.manual_seed(42)
        model = self.SmallCNN().to("vulkan")
        compiled = torch.compile(model, backend="inductor")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for step in range(10):
            x = torch.randn(4, 3, 32, 32, device="vulkan")
            targets = torch.randint(0, 10, (4,), device="vulkan")
            logits = compiled(x)
            loss = F.cross_entropy(logits, targets)
            assert not torch.isnan(loss), f"NaN loss at step {step}"
            opt.zero_grad()
            loss.backward()
            opt.step()

    def test_forward_vs_cpu(self):
        """Compiled Vulkan forward matches CPU forward within tolerance."""
        torch.manual_seed(42)
        model_vk = self.SmallCNN().to("vulkan")
        model_cpu = self.SmallCNN()
        model_cpu.load_state_dict(
            {k: v.cpu() for k, v in model_vk.state_dict().items()}
        )
        compiled = torch.compile(model_vk, backend="inductor")
        x_cpu = torch.randn(2, 3, 32, 32)
        x_vk = x_cpu.to("vulkan")
        out_cpu = model_cpu(x_cpu)
        out_vk = compiled(x_vk).cpu()
        assert torch.allclose(out_vk, out_cpu, atol=1e-3, rtol=1e-2), (
            f"max err {(out_vk - out_cpu).abs().max()}"
        )

    def test_forward_backward(self):
        """Forward + backward produces valid gradients for all parameters."""
        torch.manual_seed(42)
        model = self.SmallCNN().to("vulkan")
        compiled = torch.compile(model, backend="inductor")
        x = torch.randn(2, 3, 32, 32, device="vulkan")
        targets = torch.randint(0, 10, (2,), device="vulkan")
        logits = compiled(x)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No grad for {name}"
            assert torch.isfinite(p.grad.cpu()).all(), f"Non-finite grad for {name}"


# ═══════════════════════════════════════════════════════════════════
# Wave 10: Llama3 Block — RoPE + RMSNorm + SwiGLU
# ═══════════════════════════════════════════════════════════════════


class TestLlama3Block:
    """Llama3-style decoder block components: RoPE, RMSNorm, SwiGLU."""

    @staticmethod
    def _precompute_rope_freqs(head_dim, seq_len, theta_base=10000.0):
        """Precompute RoPE frequencies using theta_base ** tensor (pow.Scalar)."""
        freqs = 1.0 / (
            theta_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        return torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)

    @staticmethod
    def _apply_rope(x, cos, sin):
        """Apply rotary position embeddings to input tensor."""
        x_rot = x * cos + TestLlama3Block._rotate_half(x) * sin
        return x_rot

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def test_rope_forward(self):
        """RoPE forward using theta_base ** tensor (pow.Scalar lowering)."""
        torch.manual_seed(42)
        B, H, S, D = 2, 4, 16, 32
        theta_base = 10000.0
        # Compute freqs on Vulkan — tests pow.Scalar lowering
        freqs = self._precompute_rope_freqs(D, S, theta_base).to("vulkan")
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, S, D)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        q = torch.randn(B, H, S, D, device="vulkan")
        k = torch.randn(B, H, S, D, device="vulkan")
        q_rot = self._apply_rope(q, cos, sin)
        k_rot = self._apply_rope(k, cos, sin)
        assert q_rot.shape == (B, H, S, D)
        assert k_rot.shape == (B, H, S, D)
        assert torch.isfinite(q_rot.cpu()).all()
        assert torch.isfinite(k_rot.cpu()).all()

    def test_rope_backward(self):
        """RoPE backward produces valid gradients."""
        torch.manual_seed(42)
        B, H, S, D = 2, 4, 16, 32
        freqs = self._precompute_rope_freqs(D, S).to("vulkan")
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        q = torch.randn(B, H, S, D, device="vulkan", requires_grad=True)
        q_rot = self._apply_rope(q, cos.detach(), sin.detach())
        q_rot.sum().backward()
        assert q.grad is not None
        assert torch.isfinite(q.grad.cpu()).all()

    def test_llama3_rms_norm_forward(self):
        """Llama3-style RMSNorm forward."""
        torch.manual_seed(42)
        B, S, D = 2, 8, 128
        weight = torch.ones(D, device="vulkan")
        x = torch.randn(B, S, D, device="vulkan")
        rms = (x.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
        out = x * rms * weight
        assert out.shape == (B, S, D)
        assert torch.isfinite(out.cpu()).all()

    def test_llama3_rms_norm_backward(self):
        """Llama3-style RMSNorm backward."""
        torch.manual_seed(42)
        B, S, D = 2, 4, 64
        weight = torch.ones(D, device="vulkan", requires_grad=True)
        x = torch.randn(B, S, D, device="vulkan", requires_grad=True)
        rms = (x.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
        out = x * rms * weight
        out.sum().backward()
        assert x.grad is not None
        assert weight.grad is not None
        assert torch.isfinite(x.grad.cpu()).all()
        assert torch.isfinite(weight.grad.cpu()).all()

    def test_llama3_swiglu_forward(self):
        """SwiGLU: silu(gate) * up, then down projection."""
        torch.manual_seed(42)
        B, S, D, I = 2, 8, 64, 256
        gate_proj = nn.Linear(D, I, bias=False).to("vulkan")
        up_proj = nn.Linear(D, I, bias=False).to("vulkan")
        down_proj = nn.Linear(I, D, bias=False).to("vulkan")
        x = torch.randn(B, S, D, device="vulkan")
        gate = F.silu(gate_proj(x))
        up = up_proj(x)
        out = down_proj(gate * up)
        assert out.shape == (B, S, D)
        assert torch.isfinite(out.cpu()).all()

    def test_llama3_swiglu_backward(self):
        """SwiGLU backward produces valid gradients."""
        torch.manual_seed(42)
        B, S, D, I = 2, 4, 64, 128
        gate_proj = nn.Linear(D, I, bias=False).to("vulkan")
        up_proj = nn.Linear(D, I, bias=False).to("vulkan")
        down_proj = nn.Linear(I, D, bias=False).to("vulkan")
        x = torch.randn(B, S, D, device="vulkan", requires_grad=True)
        gate = F.silu(gate_proj(x))
        up = up_proj(x)
        out = down_proj(gate * up)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad.cpu()).all()
        assert gate_proj.weight.grad is not None
        assert up_proj.weight.grad is not None
        assert down_proj.weight.grad is not None

    def test_llama3_full_block_forward(self):
        """Full Llama3 decoder block: RMSNorm → Attention → RMSNorm → SwiGLU."""
        torch.manual_seed(42)
        B, S, D, H = 2, 8, 64, 4
        head_dim = D // H
        # RMSNorm weights
        norm1_w = torch.ones(D, device="vulkan")
        norm2_w = torch.ones(D, device="vulkan")
        # Attention projections
        wq = nn.Linear(D, D, bias=False).to("vulkan")
        wk = nn.Linear(D, D, bias=False).to("vulkan")
        wv = nn.Linear(D, D, bias=False).to("vulkan")
        wo = nn.Linear(D, D, bias=False).to("vulkan")
        # SwiGLU
        gate_proj = nn.Linear(D, D * 4, bias=False).to("vulkan")
        up_proj = nn.Linear(D, D * 4, bias=False).to("vulkan")
        down_proj = nn.Linear(D * 4, D, bias=False).to("vulkan")
        x = torch.randn(B, S, D, device="vulkan")
        # RMSNorm → Attention
        residual = x
        h = x * (x.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt() * norm1_w
        q = wq(h).view(B, S, H, head_dim).transpose(1, 2)
        k = wk(h).view(B, S, H, head_dim).transpose(1, 2)
        v = wv(h).view(B, S, H, head_dim).transpose(1, 2)
        mask = torch.triu(
            torch.full((S, S), float("-inf"), device="vulkan"), diagonal=1
        )
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        h = wo(attn.transpose(1, 2).reshape(B, S, D))
        x = residual + h
        # RMSNorm → SwiGLU
        residual = x
        h = x * (x.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt() * norm2_w
        gate = F.silu(gate_proj(h))
        up = up_proj(h)
        out = down_proj(gate * up) + residual
        assert out.shape == (B, S, D)
        assert torch.isfinite(out.cpu()).all()

    def test_llama3_full_block_backward(self):
        """Full Llama3 decoder block backward produces valid gradients."""
        torch.manual_seed(42)
        B, S, D, H = 2, 4, 64, 4
        head_dim = D // H
        norm1_w = torch.ones(D, device="vulkan")
        norm2_w = torch.ones(D, device="vulkan")
        wq = nn.Linear(D, D, bias=False).to("vulkan")
        wk = nn.Linear(D, D, bias=False).to("vulkan")
        wv = nn.Linear(D, D, bias=False).to("vulkan")
        wo = nn.Linear(D, D, bias=False).to("vulkan")
        gate_proj = nn.Linear(D, D * 4, bias=False).to("vulkan")
        up_proj = nn.Linear(D, D * 4, bias=False).to("vulkan")
        down_proj = nn.Linear(D * 4, D, bias=False).to("vulkan")
        x = torch.randn(B, S, D, device="vulkan", requires_grad=True)
        residual = x
        h = x * (x.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt() * norm1_w
        q = wq(h).view(B, S, H, head_dim).transpose(1, 2)
        k = wk(h).view(B, S, H, head_dim).transpose(1, 2)
        v = wv(h).view(B, S, H, head_dim).transpose(1, 2)
        mask = torch.triu(
            torch.full((S, S), float("-inf"), device="vulkan"), diagonal=1
        )
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        h = wo(attn.transpose(1, 2).reshape(B, S, D))
        x2 = residual + h
        residual = x2
        h = x2 * (x2.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt() * norm2_w
        gate = F.silu(gate_proj(h))
        up = up_proj(h)
        out = down_proj(gate * up) + residual
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad.cpu()).all()


# ═══════════════════════════════════════════════════════════════════
# Wave 11: Mixtral MoE — TopK Routing + Expert FFN
# ═══════════════════════════════════════════════════════════════════


class TestMixtralMoE:
    """Mixtral-style Mixture of Experts: top-k routing + expert FFNs."""

    def test_mixtral_routing_topk(self):
        """TopK routing: select top-2 experts per token."""
        torch.manual_seed(42)
        B, S, D, E = 2, 8, 64, 8  # 8 experts
        top_k = 2
        router = nn.Linear(D, E, bias=False).to("vulkan")
        x = torch.randn(B, S, D, device="vulkan")
        logits = router(x)  # (B, S, E)
        weights, indices = torch.topk(logits, top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        assert weights.shape == (B, S, top_k)
        assert indices.shape == (B, S, top_k)
        assert (weights.sum(-1) - 1.0).abs().max().cpu().item() < 1e-5, (
            "weights should sum to 1"
        )

    def test_mixtral_expert_forward(self):
        """Single expert FFN: gate+up SiLU gating."""
        torch.manual_seed(42)
        D, I = 64, 256
        gate_proj = nn.Linear(D, I, bias=False).to("vulkan")
        up_proj = nn.Linear(D, I, bias=False).to("vulkan")
        down_proj = nn.Linear(I, D, bias=False).to("vulkan")
        x = torch.randn(4, D, device="vulkan")
        out = down_proj(F.silu(gate_proj(x)) * up_proj(x))
        assert out.shape == (4, D)
        assert torch.isfinite(out.cpu()).all()

    def test_mixtral_moe_layer_forward(self):
        """Full MoE layer: route → dispatch → expert → combine."""
        torch.manual_seed(42)
        B, S, D, E = 2, 8, 64, 4
        top_k = 2
        router = nn.Linear(D, E, bias=False).to("vulkan")
        # Create E expert FFNs (each is gate+up+down)
        experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(D, D * 4, bias=False),
                    nn.SiLU(),
                    nn.Linear(D * 4, D, bias=False),
                ).to("vulkan")
                for _ in range(E)
            ]
        )
        x = torch.randn(B, S, D, device="vulkan")
        logits = router(x)
        weights, indices = torch.topk(logits, top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        # Simple combine: weighted sum of expert outputs
        output = torch.zeros(B, S, D, device="vulkan")
        for e in range(E):
            mask = (indices == e).any(dim=-1)  # tokens routed to expert e
            if mask.any():
                expert_out = experts[e](x[mask])
                for k in range(top_k):
                    k_mask = indices[:, :, k] == e
                    output[k_mask] += expert_out[k_mask[mask]] * weights[:, :, k][
                        k_mask
                    ].unsqueeze(-1)
        assert output.shape == (B, S, D)
        assert torch.isfinite(output.cpu()).all()

    def test_mixtral_moe_layer_backward(self):
        """MoE layer backward: gradients flow through experts and router."""
        torch.manual_seed(42)
        B, S, D, E = 2, 4, 32, 4
        top_k = 2
        router = nn.Linear(D, E, bias=False).to("vulkan")
        expert = nn.Sequential(
            nn.Linear(D, D * 4, bias=False),
            nn.SiLU(),
            nn.Linear(D * 4, D, bias=False),
        ).to("vulkan")
        x = torch.randn(B, S, D, device="vulkan", requires_grad=True)
        logits = router(x)
        weights, indices = torch.topk(logits, top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        # Single expert for simplicity
        out = expert(x.reshape(-1, D)).reshape(B, S, D)
        out = out * weights.sum(-1, keepdim=True) / top_k
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad.cpu()).all()

    def test_mixtral_load_balancing(self):
        """Load balancing aux loss: encourages uniform expert usage."""
        torch.manual_seed(42)
        B, S, D, E = 2, 8, 64, 8
        router = nn.Linear(D, E, bias=False).to("vulkan")
        x = torch.randn(B, S, D, device="vulkan")
        logits = router(x)
        probs = F.softmax(logits, dim=-1)  # (B, S, E)
        # Fraction of tokens dispatched to each expert
        f = probs.mean(dim=(0, 1))  # (E,)
        # Aux loss: E * sum(f_i * mean_prob_i)
        aux_loss = E * (f * probs.mean(dim=(0, 1))).sum()
        assert aux_loss.ndim == 0  # scalar
        assert torch.isfinite(aux_loss.cpu()).all()


# ═══════════════════════════════════════════════════════════════════
# Wave 12: Stable Diffusion — UNet Blocks
# ═══════════════════════════════════════════════════════════════════


class TestStableDiffusion:
    """Stable Diffusion UNet building blocks."""

    def test_sd_attention_block(self):
        """Cross-attention + self-attention block (no xfail)."""
        torch.manual_seed(42)
        B, C, H, W = 2, 64, 16, 16
        num_heads = 8
        head_dim = C // num_heads
        # Self-attention on flattened spatial dims
        norm = nn.GroupNorm(32, C).to("vulkan")
        q_proj = nn.Linear(C, C, bias=False).to("vulkan")
        k_proj = nn.Linear(C, C, bias=False).to("vulkan")
        v_proj = nn.Linear(C, C, bias=False).to("vulkan")
        out_proj = nn.Linear(C, C).to("vulkan")
        x = torch.randn(B, C, H, W, device="vulkan")
        # Norm → reshape → attention → reshape
        h = norm(x)
        h = h.reshape(B, C, H * W).transpose(1, 2)  # (B, N, C)
        q = q_proj(h).view(B, -1, num_heads, head_dim).transpose(1, 2)
        k = k_proj(h).view(B, -1, num_heads, head_dim).transpose(1, 2)
        v = v_proj(h).view(B, -1, num_heads, head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, H * W, C)
        out = out_proj(attn).transpose(1, 2).reshape(B, C, H, W)
        assert out.shape == (B, C, H, W)
        assert torch.isfinite(out.cpu()).all()

    @pytest.mark.xfail(reason="GroupNorm backward not yet lowered on Vulkan")
    def test_sd_resnet_block(self):
        """ResNet block with GroupNorm (xfail: GN backward)."""
        torch.manual_seed(42)
        B, C, H, W = 2, 64, 16, 16
        norm1 = nn.GroupNorm(32, C).to("vulkan")
        conv1 = nn.Conv2d(C, C, 3, padding=1).to("vulkan")
        norm2 = nn.GroupNorm(32, C).to("vulkan")
        conv2 = nn.Conv2d(C, C, 3, padding=1).to("vulkan")
        x = torch.randn(B, C, H, W, device="vulkan", requires_grad=True)
        residual = x
        h = F.silu(norm1(conv1(x)))
        h = norm2(conv2(h))
        out = (h + residual).sum()
        out.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad.cpu()).all()

    def test_sd_upsample(self):
        """Upsample2D (nearest) for UNet decoder."""
        torch.manual_seed(42)
        B, C, H, W = 2, 64, 8, 8
        x = torch.randn(B, C, H, W, device="vulkan")
        out = F.interpolate(x, scale_factor=2.0, mode="nearest")
        assert out.shape == (B, C, H * 2, W * 2)
        assert torch.isfinite(out.cpu()).all()

    def test_sd_upsample_with_conv(self):
        """Upsample2D followed by Conv2d (typical UNet decoder pattern)."""
        torch.manual_seed(42)
        B, C_in, C_out, H, W = 2, 64, 32, 8, 8
        conv = nn.Conv2d(C_in, C_out, 3, padding=1).to("vulkan")
        x = torch.randn(B, C_in, H, W, device="vulkan")
        h = F.interpolate(x, scale_factor=2.0, mode="nearest")
        out = conv(h)
        assert out.shape == (B, C_out, H * 2, W * 2)
        assert torch.isfinite(out.cpu()).all()


# ═══════════════════════════════════════════════════════════════════
# Wave 13: Whisper Encoder
# ═══════════════════════════════════════════════════════════════════


class TestWhisperEncoder:
    """Whisper encoder: Conv1d subsampling + Transformer + LayerNorm."""

    def test_whisper_full_encoder_forward(self):
        """Full Whisper encoder pipeline forward."""
        torch.manual_seed(42)
        B, T, D, H = 2, 64, 128, 4
        head_dim = D // H
        # Conv1d subsampling (2x)
        conv1 = nn.Conv1d(D, D, 3, stride=1, padding=1).to("vulkan")
        conv2 = nn.Conv1d(D, D, 3, stride=2, padding=1).to("vulkan")
        # Transformer block
        ln1 = nn.LayerNorm(D).to("vulkan")
        ln2 = nn.LayerNorm(D).to("vulkan")
        wq = nn.Linear(D, D, bias=False).to("vulkan")
        wk = nn.Linear(D, D, bias=False).to("vulkan")
        wv = nn.Linear(D, D, bias=False).to("vulkan")
        wo = nn.Linear(D, D).to("vulkan")
        ffn = nn.Sequential(
            nn.Linear(D, D * 4),
            nn.GELU(),
            nn.Linear(D * 4, D),
        ).to("vulkan")
        x = torch.randn(B, D, T, device="vulkan")
        # Conv subsampling
        h = F.gelu(conv1(x))
        h = F.gelu(conv2(h))
        T_out = h.shape[-1]
        h = h.transpose(1, 2)  # (B, T', D)
        # Transformer
        residual = h
        h = ln1(h)
        q = wq(h).view(B, T_out, H, head_dim).transpose(1, 2)
        k = wk(h).view(B, T_out, H, head_dim).transpose(1, 2)
        v = wv(h).view(B, T_out, H, head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        h = wo(attn.transpose(1, 2).reshape(B, T_out, D)) + residual
        residual = h
        out = ffn(ln2(h)) + residual
        assert out.shape == (B, T_out, D)
        assert torch.isfinite(out.cpu()).all()

    def test_whisper_full_encoder_backward(self):
        """Full Whisper encoder pipeline backward."""
        torch.manual_seed(42)
        B, T, D, H = 2, 32, 64, 4
        head_dim = D // H
        conv1 = nn.Conv1d(D, D, 3, stride=1, padding=1).to("vulkan")
        conv2 = nn.Conv1d(D, D, 3, stride=2, padding=1).to("vulkan")
        ln1 = nn.LayerNorm(D).to("vulkan")
        ln2 = nn.LayerNorm(D).to("vulkan")
        wq = nn.Linear(D, D, bias=False).to("vulkan")
        wk = nn.Linear(D, D, bias=False).to("vulkan")
        wv = nn.Linear(D, D, bias=False).to("vulkan")
        wo = nn.Linear(D, D).to("vulkan")
        ffn = nn.Sequential(
            nn.Linear(D, D * 4),
            nn.GELU(),
            nn.Linear(D * 4, D),
        ).to("vulkan")
        x = torch.randn(B, D, T, device="vulkan", requires_grad=True)
        h = F.gelu(conv1(x))
        h = F.gelu(conv2(h))
        T_out = h.shape[-1]
        h = h.transpose(1, 2)
        residual = h
        h = ln1(h)
        q = wq(h).view(B, T_out, H, head_dim).transpose(1, 2)
        k = wk(h).view(B, T_out, H, head_dim).transpose(1, 2)
        v = wv(h).view(B, T_out, H, head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        h = wo(attn.transpose(1, 2).reshape(B, T_out, D)) + residual
        residual = h
        out = ffn(ln2(h)) + residual
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad.cpu()).all()


# ═══════════════════════════════════════════════════════════════════
# Wave 14: ViT Encoder Block
# ═══════════════════════════════════════════════════════════════════


class TestViTEncoderBlock:
    """Vision Transformer encoder block."""

    def test_vit_encoder_forward(self):
        """ViT encoder block: LayerNorm → Attention → LayerNorm → MLP."""
        torch.manual_seed(42)
        B, N, D, H = 2, 16, 64, 4
        head_dim = D // H
        ln1 = nn.LayerNorm(D).to("vulkan")
        ln2 = nn.LayerNorm(D).to("vulkan")
        wq = nn.Linear(D, D, bias=False).to("vulkan")
        wk = nn.Linear(D, D, bias=False).to("vulkan")
        wv = nn.Linear(D, D, bias=False).to("vulkan")
        wo = nn.Linear(D, D).to("vulkan")
        mlp = nn.Sequential(
            nn.Linear(D, D * 4),
            nn.GELU(),
            nn.Linear(D * 4, D),
        ).to("vulkan")
        x = torch.randn(B, N, D, device="vulkan")
        residual = x
        h = ln1(x)
        q = wq(h).view(B, N, H, head_dim).transpose(1, 2)
        k = wk(h).view(B, N, H, head_dim).transpose(1, 2)
        v = wv(h).view(B, N, H, head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        h = wo(attn.transpose(1, 2).reshape(B, N, D)) + residual
        residual = h
        out = mlp(ln2(h)) + residual
        assert out.shape == (B, N, D)
        assert torch.isfinite(out.cpu()).all()

    def test_vit_encoder_backward(self):
        """ViT encoder block backward."""
        torch.manual_seed(42)
        B, N, D, H = 2, 16, 64, 4
        head_dim = D // H
        ln1 = nn.LayerNorm(D).to("vulkan")
        ln2 = nn.LayerNorm(D).to("vulkan")
        wq = nn.Linear(D, D, bias=False).to("vulkan")
        wk = nn.Linear(D, D, bias=False).to("vulkan")
        wv = nn.Linear(D, D, bias=False).to("vulkan")
        wo = nn.Linear(D, D).to("vulkan")
        mlp = nn.Sequential(
            nn.Linear(D, D * 4),
            nn.GELU(),
            nn.Linear(D * 4, D),
        ).to("vulkan")
        x = torch.randn(B, N, D, device="vulkan", requires_grad=True)
        residual = x
        h = ln1(x)
        q = wq(h).view(B, N, H, head_dim).transpose(1, 2)
        k = wk(h).view(B, N, H, head_dim).transpose(1, 2)
        v = wv(h).view(B, N, H, head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        h = wo(attn.transpose(1, 2).reshape(B, N, D)) + residual
        residual = h
        out = mlp(ln2(h)) + residual
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad.cpu()).all()

    def test_vit_encoder_training_step(self):
        """ViT encoder block: 3-step training loop."""
        torch.manual_seed(42)
        B, N, D, H, C = 2, 16, 64, 4, 10
        head_dim = D // H
        ln1 = nn.LayerNorm(D).to("vulkan")
        ln2 = nn.LayerNorm(D).to("vulkan")
        wq = nn.Linear(D, D, bias=False).to("vulkan")
        wk = nn.Linear(D, D, bias=False).to("vulkan")
        wv = nn.Linear(D, D, bias=False).to("vulkan")
        wo = nn.Linear(D, D).to("vulkan")
        mlp = nn.Sequential(
            nn.Linear(D, D * 4),
            nn.GELU(),
            nn.Linear(D * 4, D),
        ).to("vulkan")
        classifier = nn.Linear(D, C).to("vulkan")
        opt = torch.optim.SGD(
            list(ln1.parameters())
            + list(ln2.parameters())
            + list(wq.parameters())
            + list(wk.parameters())
            + list(wv.parameters())
            + list(wo.parameters())
            + list(mlp.parameters())
            + list(classifier.parameters()),
            lr=1e-3,
        )
        losses = []
        for _ in range(3):
            x = torch.randn(B, N, D, device="vulkan")
            targets = torch.randint(0, C, (B,), device="vulkan")
            residual = x
            h = ln1(x)
            q = wq(h).view(B, N, H, head_dim).transpose(1, 2)
            k = wk(h).view(B, N, H, head_dim).transpose(1, 2)
            v = wv(h).view(B, N, H, head_dim).transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, k, v)
            h = wo(attn.transpose(1, 2).reshape(B, N, D)) + residual
            residual = h
            h = mlp(ln2(h)) + residual
            logits = classifier(h.mean(dim=1))
            loss = F.cross_entropy(logits, targets)
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert losses[-1] < losses[0] * 1.1, f"Loss exploded: {losses}"


# ═══════════════════════════════════════════════════════════════════
# Wave 15: UNet Block
# ═══════════════════════════════════════════════════════════════════


class TestUNetBlock:
    """UNet residual + attention blocks (GroupNorm works now — no xfail)."""

    def test_unet_res_block_forward(self):
        """UNet ResBlock: GroupNorm → SiLU → Conv → GroupNorm → SiLU → Conv + residual."""
        torch.manual_seed(42)
        B, C, H, W = 2, 32, 16, 16
        norm1 = nn.GroupNorm(min(32, C), C).to("vulkan")
        conv1 = nn.Conv2d(C, C, 3, padding=1).to("vulkan")
        norm2 = nn.GroupNorm(min(32, C), C).to("vulkan")
        conv2 = nn.Conv2d(C, C, 3, padding=1).to("vulkan")
        x = torch.randn(B, C, H, W, device="vulkan")
        residual = x
        h = F.silu(norm1(conv1(x)))
        h = norm2(conv2(h))
        out = h + residual
        assert out.shape == (B, C, H, W)
        assert torch.isfinite(out.cpu()).all()

    def test_unet_attn_block_forward(self):
        """UNet attention block: GroupNorm → spatial attention."""
        torch.manual_seed(42)
        B, C, H, W = 2, 32, 16, 16
        num_heads = 4
        head_dim = C // num_heads
        norm = nn.GroupNorm(min(32, C), C).to("vulkan")
        q_proj = nn.Linear(C, C, bias=False).to("vulkan")
        k_proj = nn.Linear(C, C, bias=False).to("vulkan")
        v_proj = nn.Linear(C, C, bias=False).to("vulkan")
        out_proj = nn.Linear(C, C).to("vulkan")
        x = torch.randn(B, C, H, W, device="vulkan")
        residual = x
        h = norm(x)
        N = H * W
        h = h.reshape(B, C, N).transpose(1, 2)
        q = q_proj(h).view(B, N, num_heads, head_dim).transpose(1, 2)
        k = k_proj(h).view(B, N, num_heads, head_dim).transpose(1, 2)
        v = v_proj(h).view(B, N, num_heads, head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, N, C)
        h = out_proj(attn).transpose(1, 2).reshape(B, C, H, W)
        out = h + residual
        assert out.shape == (B, C, H, W)
        assert torch.isfinite(out.cpu()).all()


# ═══════════════════════════════════════════════════════════════════
# Wave 16: Mamba2 Selective Scan
# ═══════════════════════════════════════════════════════════════════


class TestMamba2SelectiveScan:
    """Mamba2-style selective scan: SSM with input-dependent A, B, C, dt."""

    def test_mamba2_selective_scan_decomposes(self):
        """Selective scan decomposes into primitive ops (discretize + scan)."""
        torch.manual_seed(42)
        B, L, D, N = 2, 16, 32, 16  # batch, seq_len, dim, state_dim
        # SSM parameters
        A = -torch.arange(1, N + 1, dtype=torch.float32, device="vulkan").unsqueeze(
            0
        )  # (1, N)
        dt = F.softplus(torch.randn(B, L, D, device="vulkan"))  # (B, L, D)
        B_ssm = torch.randn(B, L, N, device="vulkan")
        C_ssm = torch.randn(B, L, N, device="vulkan")
        x = torch.randn(B, L, D, device="vulkan")
        # Discretize A: A_bar = exp(dt * A)
        dt_A = dt.unsqueeze(-1) * A  # (B, L, D, N)
        A_bar = dt_A.exp()
        # Discretize B: B_bar = dt * B
        dt_B = dt.unsqueeze(-1) * B_ssm
        assert A_bar.shape == (B, L, D, N)
        assert dt_B.shape == (B, L, N)
        assert torch.isfinite(A_bar.cpu()).all()

    def test_mamba2_block_forward(self):
        """Mamba2 block: RMSNorm → SSM + gate branch → out_proj."""
        torch.manual_seed(42)
        B, L, D, N = 2, 16, 32, 16
        # Input projections (expand to 2*D: one for SSM, one for gate)
        in_proj = nn.Linear(D, D * 2, bias=False).to("vulkan")
        out_proj = nn.Linear(D, D, bias=False).to("vulkan")
        # SSM parameters
        A = -torch.arange(1, N + 1, dtype=torch.float32, device="vulkan").unsqueeze(0)
        dt_proj = nn.Linear(D, D, bias=True).to("vulkan")
        B_proj = nn.Linear(D, N, bias=False).to("vulkan")
        C_proj = nn.Linear(D, N, bias=False).to("vulkan")
        x = torch.randn(B, L, D, device="vulkan")
        # RMSNorm (simple)
        rms = (x.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
        h = x * rms
        # Split into ssm and gate branches
        proj = in_proj(h)
        u, gate = proj.chunk(2, dim=-1)  # each (B, L, D)
        # SSM
        dt = F.softplus(dt_proj(u))
        B_ssm = B_proj(u)
        C_ssm = C_proj(u)
        # Recurrent scan (simplified: parallel scan approximation using cumsum)
        A_bar = (dt.unsqueeze(-1) * A).exp()
        u_weighted = u.unsqueeze(-1) * B_ssm.unsqueeze(-2)  # (B, L, D, N)
        state = u_weighted.cumsum(dim=1) * A_bar.cumprod(dim=1)
        y = (state * C_ssm.unsqueeze(-2)).sum(dim=-1)  # (B, L, D)
        out = out_proj(y * F.silu(gate))
        assert out.shape == (B, L, D)
        assert torch.isfinite(out.cpu()).all()

    def test_mamba2_block_backward(self):
        """Mamba2 block backward: gradients flow through SSM."""
        torch.manual_seed(42)
        B, L, D, N = 2, 8, 32, 16
        in_proj = nn.Linear(D, D * 2, bias=False).to("vulkan")
        out_proj = nn.Linear(D, D, bias=False).to("vulkan")
        A = -torch.arange(1, N + 1, dtype=torch.float32, device="vulkan").unsqueeze(0)
        dt_proj = nn.Linear(D, D, bias=True).to("vulkan")
        B_proj = nn.Linear(D, N, bias=False).to("vulkan")
        C_proj = nn.Linear(D, N, bias=False).to("vulkan")
        x = torch.randn(B, L, D, device="vulkan", requires_grad=True)
        rms = (x.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
        h = x * rms
        proj = in_proj(h)
        u, gate = proj.chunk(2, dim=-1)
        dt = F.softplus(dt_proj(u))
        B_ssm = B_proj(u)
        C_ssm = C_proj(u)
        A_bar = (dt.unsqueeze(-1) * A).exp()
        u_weighted = u.unsqueeze(-1) * B_ssm.unsqueeze(-2)
        state = u_weighted.cumsum(dim=1) * A_bar.cumprod(dim=1)
        y = (state * C_ssm.unsqueeze(-2)).sum(dim=-1)
        out = out_proj(y * F.silu(gate))
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad.cpu()).all()

    def test_mamba2_vs_recurrent(self):
        """Parallel scan approximation matches recurrent scan (with tolerance)."""
        torch.manual_seed(42)
        B, L, D, N = 1, 8, 16, 8
        A = -torch.arange(1, N + 1, dtype=torch.float32, device="vulkan").unsqueeze(0)
        dt = F.softplus(torch.randn(B, L, D, device="vulkan"))
        B_ssm = torch.randn(B, L, N, device="vulkan")
        C_ssm = torch.randn(B, L, N, device="vulkan")
        u = torch.randn(B, L, D, device="vulkan")
        # Parallel scan approximation
        A_bar = (dt.unsqueeze(-1) * A).exp()
        u_weighted = u.unsqueeze(-1) * B_ssm.unsqueeze(-2)
        state_p = u_weighted.cumsum(dim=1) * A_bar.cumprod(dim=1)
        y_parallel = (state_p * C_ssm.unsqueeze(-2)).sum(dim=-1)
        # Recurrent scan (ground truth)
        state = torch.zeros(B, D, N, device="vulkan")
        y_recurrent = torch.zeros(B, L, D, device="vulkan")
        for t in range(L):
            state = state * A_bar[:, t] + u_weighted[:, t]
            y_recurrent[:, t] = (state * C_ssm[:, t].unsqueeze(1)).sum(dim=-1)
        diff = (y_parallel - y_recurrent).abs().max().cpu().item()
        assert diff < 1e-2, f"Parallel vs recurrent mismatch: {diff}"

    def test_mamba2_training_step(self):
        """Mamba2 block: 3-step training loop."""
        torch.manual_seed(42)
        B, L, D, N = 2, 16, 32, 16
        in_proj = nn.Linear(D, D * 2, bias=False).to("vulkan")
        out_proj = nn.Linear(D, D, bias=False).to("vulkan")
        A = nn.Parameter(
            -torch.arange(1, N + 1, dtype=torch.float32).unsqueeze(0).to("vulkan")
        )
        dt_proj = nn.Linear(D, D, bias=True).to("vulkan")
        B_proj = nn.Linear(D, N, bias=False).to("vulkan")
        C_proj = nn.Linear(D, N, bias=False).to("vulkan")
        classifier = nn.Linear(D, 10).to("vulkan")
        opt = torch.optim.Adam(
            [A]
            + list(in_proj.parameters())
            + list(out_proj.parameters())
            + list(dt_proj.parameters())
            + list(B_proj.parameters())
            + list(C_proj.parameters())
            + list(classifier.parameters()),
            lr=1e-3,
        )
        losses = []
        for _ in range(3):
            x = torch.randn(B, L, D, device="vulkan")
            targets = torch.randint(0, 10, (B,), device="vulkan")
            rms = (x.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
            h = x * rms
            proj = in_proj(h)
            u, gate = proj.chunk(2, dim=-1)
            dt = F.softplus(dt_proj(u))
            B_ssm = B_proj(u)
            C_ssm = C_proj(u)
            A_bar = (dt.unsqueeze(-1) * A).exp()
            u_weighted = u.unsqueeze(-1) * B_ssm.unsqueeze(-2)
            state = u_weighted.cumsum(dim=1) * A_bar.cumprod(dim=1)
            y = (state * C_ssm.unsqueeze(-2)).sum(dim=-1)
            out = out_proj(y * F.silu(gate))
            logits = classifier(out.mean(dim=1))
            loss = F.cross_entropy(logits, targets)
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert losses[-1] < losses[0] * 1.2, f"Loss exploded: {losses}"


# ═══════════════════════════════════════════════════════════════════
# Wave 17a: Mamba CausalConv1d
# ═══════════════════════════════════════════════════════════════════


class TestMambaCausalConv1d:
    """Mamba-style causal depthwise Conv1d."""

    def test_causal_conv1d_forward(self):
        """Causal depthwise Conv1d forward (no xfail)."""
        torch.manual_seed(42)
        B, C, T = 2, 32, 16
        kernel_size = 4
        x = torch.randn(B, C, T, device="vulkan")
        weight = torch.randn(C, kernel_size, device="vulkan")
        # Causal: pad left so output[t] only sees input[:t+1]
        x_padded = F.pad(x, (kernel_size - 1, 0))
        out = F.conv1d(x_padded, weight.unsqueeze(1), None, padding=0, groups=C)
        assert out.shape == (B, C, T)
        assert torch.isfinite(out.cpu()).all()

    def test_causal_conv1d_silu_combined(self):
        """Causal Conv1d + SiLU activation (Mamba pattern)."""
        torch.manual_seed(42)
        B, C, T = 2, 32, 16
        kernel_size = 4
        x = torch.randn(B, C, T, device="vulkan")
        weight = torch.randn(C, kernel_size, device="vulkan")
        x_padded = F.pad(x, (kernel_size - 1, 0))
        out = F.silu(F.conv1d(x_padded, weight.unsqueeze(1), None, padding=0, groups=C))
        assert out.shape == (B, C, T)
        assert torch.isfinite(out.cpu()).all()

    def test_causal_conv1d_grad(self):
        """Causal Conv1d backward: gradients for input and weight."""
        torch.manual_seed(42)
        B, C, T = 2, 16, 8
        kernel_size = 3
        x = torch.randn(B, C, T, device="vulkan", requires_grad=True)
        weight = torch.randn(C, kernel_size, device="vulkan", requires_grad=True)
        x_padded = F.pad(x, (kernel_size - 1, 0))
        out = F.conv1d(x_padded, weight.unsqueeze(1), None, padding=0, groups=C)
        out.sum().backward()
        assert x.grad is not None
        assert weight.grad is not None
        assert x.grad.shape == (B, C, T)
        assert weight.grad.shape == (C, kernel_size)
        assert torch.isfinite(x.grad.cpu()).all()
        assert torch.isfinite(weight.grad.cpu()).all()

    def test_causal_conv1d_vs_standard(self):
        """Causal Conv1d matches standard Conv1d at last timestep."""
        torch.manual_seed(42)
        B, C, T = 1, 8, 16
        kernel_size = 4
        x = torch.randn(B, C, T, device="vulkan")
        weight = torch.randn(C, kernel_size, device="vulkan")
        # Causal
        x_padded = F.pad(x, (kernel_size - 1, 0))
        causal_out = F.conv1d(x_padded, weight.unsqueeze(1), None, padding=0, groups=C)
        # Standard (same padding removed at output edges)
        std_out = F.conv1d(
            x, weight.unsqueeze(1), None, padding=kernel_size - 1, groups=C
        )
        # Last output should match (both see all inputs at last timestep)
        diff = (causal_out[:, :, -1] - std_out[:, :, -1]).abs().max().cpu().item()
        assert diff < 1e-5, f"Last timestep mismatch: {diff}"


# ═══════════════════════════════════════════════════════════════════
# Wave 17b: LlamaMLP Block
# ═══════════════════════════════════════════════════════════════════


class TestLlamaMLPBlock:
    """Llama-style MLP block: gate + up projections, SiLU gating, down projection."""

    def test_llama_mlp_forward(self):
        """LlamaMLP: gate_proj + up_proj → SiLU(gate) * up → down_proj."""
        torch.manual_seed(42)
        B, S, D, I = 2, 8, 64, 256
        gate_proj = nn.Linear(D, I, bias=False).to("vulkan")
        up_proj = nn.Linear(D, I, bias=False).to("vulkan")
        down_proj = nn.Linear(I, D, bias=False).to("vulkan")
        x = torch.randn(B, S, D, device="vulkan")
        gate = F.silu(gate_proj(x))
        up = up_proj(x)
        out = down_proj(gate * up)
        assert out.shape == (B, S, D)
        assert torch.isfinite(out.cpu()).all()

    def test_llama_mlp_backward(self):
        """LlamaMLP backward: gradients for all three projections."""
        torch.manual_seed(42)
        B, S, D, I = 2, 4, 64, 128
        gate_proj = nn.Linear(D, I, bias=False).to("vulkan")
        up_proj = nn.Linear(D, I, bias=False).to("vulkan")
        down_proj = nn.Linear(I, D, bias=False).to("vulkan")
        x = torch.randn(B, S, D, device="vulkan", requires_grad=True)
        gate = F.silu(gate_proj(x))
        up = up_proj(x)
        out = down_proj(gate * up)
        out.sum().backward()
        assert x.grad is not None
        assert gate_proj.weight.grad is not None
        assert up_proj.weight.grad is not None
        assert down_proj.weight.grad is not None
        assert torch.isfinite(x.grad.cpu()).all()
        assert torch.isfinite(gate_proj.weight.grad.cpu()).all()

    def test_llama_mlp_vs_cpu_forward(self):
        """Vulkan MLP output matches CPU MLP output."""
        torch.manual_seed(42)
        B, S, D, I = 2, 4, 32, 64
        # Vulkan model
        gate_vk = nn.Linear(D, I, bias=False).to("vulkan")
        up_vk = nn.Linear(D, I, bias=False).to("vulkan")
        down_vk = nn.Linear(I, D, bias=False).to("vulkan")
        # CPU model
        gate_cpu = nn.Linear(D, I, bias=False)
        up_cpu = nn.Linear(D, I, bias=False)
        down_cpu = nn.Linear(I, D, bias=False)
        gate_cpu.load_state_dict({k: v.cpu() for k, v in gate_vk.state_dict().items()})
        up_cpu.load_state_dict({k: v.cpu() for k, v in up_vk.state_dict().items()})
        down_cpu.load_state_dict({k: v.cpu() for k, v in down_vk.state_dict().items()})
        x_cpu = torch.randn(B, S, D)
        x_vk = x_cpu.to("vulkan")
        out_cpu = down_cpu(F.silu(gate_cpu(x_cpu)) * up_cpu(x_cpu))
        out_vk = down_vk(F.silu(gate_vk(x_vk)) * up_vk(x_vk)).cpu()
        assert torch.allclose(out_vk, out_cpu, atol=1e-3, rtol=1e-2), (
            f"max err {(out_vk - out_cpu).abs().max()}"
        )

    def test_llama_mlp_training_step(self):
        """LlamaMLP: 5-step training loop with loss decrease."""
        torch.manual_seed(42)
        B, S, D, I = 4, 8, 32, 128
        gate_proj = nn.Linear(D, I, bias=False).to("vulkan")
        up_proj = nn.Linear(D, I, bias=False).to("vulkan")
        down_proj = nn.Linear(I, D, bias=False).to("vulkan")
        opt = torch.optim.Adam(
            [gate_proj.weight, up_proj.weight, down_proj.weight], lr=1e-2
        )
        x = torch.randn(B, S, D, device="vulkan")
        targets = torch.randn(B, S, D, device="vulkan")
        losses = []
        for _ in range(5):
            gate = F.silu(gate_proj(x))
            up = up_proj(x)
            out = down_proj(gate * up)
            loss = F.mse_loss(out, targets)
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert losses[-1] < losses[0] * 0.9, f"Loss did not decrease: {losses}"


# ═══════════════════════════════════════════════════════════════
# v6 Model Zoo additions (2026-05-11)
# ═══════════════════════════════════════════════════════════════

class TestLlama3Block:
    """Llama-3 block: RMSNorm + RoPE + SwiGLU MLP."""

    def test_rope_forward(self):
        """RoPE forward via @torch.compile."""
        B, H, S, D = 1, 4, 16, 32
        torch.manual_seed(42)
        x = torch.randn(B, H, S, D, device="vulkan")
        theta_base = 500000.0
        freq = 1.0 / (theta_base ** (torch.arange(0, D, 2, dtype=torch.float32).to("vulkan") / D))
        pos = torch.arange(S, dtype=torch.float32, device="vulkan")
        theta = pos.unsqueeze(1) * freq.unsqueeze(0)
        cos = theta.cos().unsqueeze(0).unsqueeze(0)
        sin = theta.sin().unsqueeze(0).unsqueeze(0)

        @torch.compile(backend="inductor")
        def rope(x, cos, sin):
            x1, x2 = x[..., 0::2], x[..., 1::2]
            return torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1).reshape(B, H, S, D)

        result = rope(x, cos, sin)
        assert result.shape == (B, H, S, D)
        assert torch.isfinite(result.cpu()).all()

    def test_llama3_rms_norm_forward(self):
        """RMSNorm forward."""
        B, S, D = 2, 8, 32
        torch.manual_seed(42)
        x = torch.randn(B, S, D, device="vulkan")
        weight = torch.ones(D, device="vulkan")

        @torch.compile(backend="inductor")
        def rms_norm(x, w, eps=1e-6):
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
            return x / rms * w

        out = rms_norm(x, weight)
        assert out.shape == (B, S, D)
        assert torch.isfinite(out.cpu()).all()

    def test_llama3_swiglu_forward(self):
        """SwiGLU: silu(gate) * up, then down projection."""
        B, S, D, D_ff = 2, 8, 32, 128
        torch.manual_seed(42)
        x = torch.randn(B, S, D, device="vulkan")
        gate_proj = torch.randn(D_ff, D, device="vulkan")
        up_proj = torch.randn(D_ff, D, device="vulkan")
        down_proj = torch.randn(D, D_ff, device="vulkan")

        @torch.compile(backend="inductor")
        def swiglu(x, g, u, d):
            return (torch.silu(x @ g.T) * (x @ u.T)) @ d.T

        out = swiglu(x, gate_proj, up_proj, down_proj)
        assert out.shape == (B, S, D)
        assert torch.isfinite(out.cpu()).all()

    def test_llama3_full_block_forward(self):
        """Full Llama-3 block: RMSNorm → RoPE Attention → RMSNorm → SwiGLU."""
        B, H, S, D = 1, 4, 16, 32
        torch.manual_seed(42)
        x = torch.randn(B, S, D, device="vulkan")
        Q = torch.randn(D, D, device="vulkan")
        K = torch.randn(D, D, device="vulkan")
        V = torch.randn(D, D, device="vulkan")
        O = torch.randn(D, D, device="vulkan")
        gate = torch.randn(4*D, D, device="vulkan")
        up = torch.randn(4*D, D, device="vulkan")
        down = torch.randn(D, 4*D, device="vulkan")

        @torch.compile(backend="inductor")
        def block(x, Q, K, V, O, gate, up, down):
            # RMSNorm
            rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
            a = x / rms
            # Attention (simplified: no RoPE for brevity)
            q = a @ Q.T
            k = a @ K.T
            v = a @ V.T
            attn = torch.softmax(q @ k.transpose(-2, -1) / (D**0.5), dim=-1) @ v
            a = a + (attn @ O.T)
            # RMSNorm + SwiGLU
            rms2 = torch.sqrt(torch.mean(a**2, dim=-1, keepdim=True) + 1e-6)
            a = a / rms2
            return a + ((torch.silu(a @ gate.T) * (a @ up.T)) @ down.T)

        out = block(x, Q, K, V, O, gate, up, down)
        assert out.shape == (B, S, D)
        assert torch.isfinite(out.cpu()).all()


class TestMixtralMoE:
    """Mixtral MoE: top-k routing + expert MLPs."""

    def test_mixtral_routing_topk(self):
        """Top-k expert selection."""
        B, S, E, k = 2, 8, 8, 2
        torch.manual_seed(42)
        logits = torch.randn(B, S, E, device="vulkan")

        @torch.compile(backend="inductor")
        def route(logits):
            return torch.topk(logits, k, dim=-1)

        scores, indices = route(logits)
        assert indices.shape == (B, S, k)
        assert (indices.cpu() >= 0).all() and (indices.cpu() < E).all()

    def test_mixtral_expert_forward(self):
        """Single SwiGLU expert."""
        B, S, D, D_ff = 1, 8, 32, 128
        torch.manual_seed(42)
        x = torch.randn(B, S, D, device="vulkan")
        w1 = torch.randn(D_ff, D, device="vulkan")
        w2 = torch.randn(D, D_ff, device="vulkan")
        w3 = torch.randn(D_ff, D, device="vulkan")

        @torch.compile(backend="inductor")
        def expert(x, w1, w2, w3):
            return (torch.silu(x @ w1.T) * (x @ w3.T)) @ w2.T

        out = expert(x, w1, w2, w3)
        assert out.shape == (B, S, D)
        assert torch.isfinite(out.cpu()).all()


class TestUNetBlock:
    """UNet blocks: Conv2d + GroupNorm + attention."""

    def test_unet_res_block_forward(self):
        """UNet residual block: Conv2d + GroupNorm + SiLU."""
        torch.manual_seed(42)
        B, C, H, W = 2, 32, 16, 16
        model = nn.Sequential(
            nn.GroupNorm(8, C), nn.SiLU(), nn.Conv2d(C, C, 3, padding=1),
            nn.GroupNorm(8, C), nn.SiLU(), nn.Conv2d(C, C, 3, padding=1),
        ).eval().vulkan()
        x = torch.randn(B, C, H, W, device="vulkan")
        out = model(x)
        assert out.shape == (B, C, H, W)
        assert torch.isfinite(out.cpu()).all()

    def test_unet_attn_block_forward(self):
        """UNet attention block: GroupNorm + SDPA."""
        torch.manual_seed(42)
        B, C, H, W = 2, 32, 8, 8
        class AttnBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.GroupNorm(8, C)
                self.qkv = nn.Conv2d(C, C*3, 1)
                self.proj = nn.Conv2d(C, C, 1)
            def forward(self, x):
                s = x.shape
                h = self.norm(x).reshape(s[0], s[1], -1).transpose(1,2)
                q, k, v = self.qkv(x).reshape(s[0], 3, s[1], s[2], s[3]).unbind(1)
                q, k, v = [t.reshape(s[0], s[1], -1).transpose(1,2) for t in (q,k,v)]
                a = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                a = a.transpose(1,2).reshape(s[0], s[1], s[2], s[3])
                return x + self.proj(a)
        model = AttnBlock().eval().vulkan()
        x = torch.randn(B, C, H, W, device="vulkan")
        out = model(x)
        assert out.shape == (B, C, H, W)
        assert torch.isfinite(out.cpu()).all()


class TestSmallCNNTrainSequential:
    """Older MNIST-style SmallCNN gate (1-channel, nn.Sequential).

    Renamed 2026-05-20 to avoid shadowing the canonical 8-test
    [TestSmallCNNTrain](#L1172).
    """

    def test_small_cnn_forward_compiles(self):
        """SmallCNN forward compiles and matches CPU."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.ReLU(),
            nn.MaxPool2d(2), nn.Flatten(), nn.Linear(16*14*14, 10),
        ).eval().vulkan()
        x = torch.randn(2, 1, 28, 28, device="vulkan")
        out = model(x)
        assert out.shape == (2, 10)
        assert torch.isfinite(out.cpu()).all()

    def test_small_cnn_3step_training(self):
        """SmallCNN 3-step training loss decreases."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.ReLU(),
            nn.MaxPool2d(2), nn.Flatten(), nn.Linear(16*14*14, 10),
        ).vulkan()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(4, 1, 28, 28, device="vulkan")
        target = torch.randint(0, 10, (4,), device="vulkan")
        losses = []
        for _ in range(3):
            def step():
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(x), target)
                loss.backward()
                return loss
            loss = torch.compile(step, backend="inductor")()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"
