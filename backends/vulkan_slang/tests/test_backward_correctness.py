"""Backward-pass and dtype correctness tests.

Covers gaps in test_correctness.py:
  - var/std forward (multiple shapes/dims) and backward
  - Activation backward: gelu, silu, leaky_relu, elu, tanh, softmax
  - Normalization backward: batch_norm, group_norm, layer_norm (various sizes)
  - Loss backward: cross_entropy (with ignore_index), mse, bce, bce_with_logits
  - SDPA backward
  - Non-contiguous tensor inputs
  - f16/bf16 forward correctness
  - Shape ops backward: permute, cat, gather
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

RTOL = 1e-3
ATOL = 1e-3


@pytest.fixture(autouse=True)
def _setup():
    try:
        import torch_vulkan
        if not torch_vulkan.is_available():
            pytest.skip("No Vulkan device")
    except ImportError:
        pytest.skip("torch_vulkan not installed")


def vk(t):
    return t.to("vulkan")


def check(cpu_out, vk_out, rtol=RTOL, atol=ATOL, msg=""):
    torch.testing.assert_close(
        vk_out.cpu() if vk_out.device.type != "cpu" else vk_out,
        cpu_out.cpu()  if cpu_out.device.type != "cpu" else cpu_out,
        rtol=rtol, atol=atol, msg=msg or None,
    )


def grad_pair(*shape, seed=42):
    """Return (cpu_tensor_requires_grad, vulkan_tensor_requires_grad)."""
    torch.manual_seed(seed)
    cpu = torch.randn(*shape, requires_grad=True)
    vk_t = cpu.detach().to("vulkan").requires_grad_(True)
    return cpu, vk_t


# ═══════════════════════════════════════════════════════════════════
# Variance / Std — forward correctness
# ═══════════════════════════════════════════════════════════════════

class TestVarStdCorrectness:

    @pytest.mark.parametrize("shape,dim,correction", [
        ((32,),    None, 1),
        ((8, 32),  0,    1),
        ((8, 32),  1,    1),
        ((8, 32),  1,    0),
        ((4, 8, 16), 2,  1),
        ((4, 8, 16), 0,  1),
    ])
    def test_var(self, shape, dim, correction):
        cpu = torch.randn(*shape)
        if dim is None:
            cpu_r = cpu.var(correction=correction)
            vk_r = vk(cpu).var(correction=correction)
        else:
            cpu_r = cpu.var(dim=dim, correction=correction)
            vk_r = vk(cpu).var(dim=dim, correction=correction)
        check(cpu_r, vk_r)

    @pytest.mark.parametrize("shape,dim", [
        ((8, 32), 1),
        ((4, 8, 16), 2),
    ])
    def test_var_keepdim(self, shape, dim):
        cpu = torch.randn(*shape)
        check(cpu.var(dim=dim, keepdim=True),
              vk(cpu).var(dim=dim, keepdim=True))

    @pytest.mark.parametrize("shape,dim,correction", [
        ((8, 32), 1, 1),
        ((4, 16), 0, 1),
    ])
    def test_std(self, shape, dim, correction):
        cpu = torch.randn(*shape)
        check(cpu.std(dim=dim, correction=correction),
              vk(cpu).std(dim=dim, correction=correction))

    def test_var_backward(self):
        cpu, vk_t = grad_pair(8, 16)
        cpu.var(dim=-1).sum().backward()
        vk_t.var(dim=-1).sum().backward()
        check(cpu.grad, vk_t.grad)

    def test_var_backward_correction0(self):
        cpu, vk_t = grad_pair(8, 16)
        cpu.var(dim=-1, correction=0).sum().backward()
        vk_t.var(dim=-1, correction=0).sum().backward()
        check(cpu.grad, vk_t.grad)

    def test_std_backward(self):
        cpu, vk_t = grad_pair(4, 16)
        cpu.std(dim=-1).sum().backward()
        vk_t.std(dim=-1).sum().backward()
        check(cpu.grad, vk_t.grad)

    def test_var_multi_dim_backward(self):
        """Multi-dim var backward (decomposed path)."""
        cpu, vk_t = grad_pair(4, 8)
        cpu.var(dim=[0, 1]).backward()
        vk_t.var(dim=[0, 1]).backward()
        check(cpu.grad, vk_t.grad)


# ═══════════════════════════════════════════════════════════════════
# Activation backward
# ═══════════════════════════════════════════════════════════════════

class TestActivationBackward:

    @pytest.mark.parametrize("fn,name", [
        (lambda x: F.gelu(x),                      "gelu"),
        (lambda x: F.gelu(x, approximate="tanh"),  "gelu_tanh"),
        (lambda x: F.silu(x),                      "silu"),
        (lambda x: F.leaky_relu(x, 0.01),          "leaky_relu"),
        (lambda x: F.elu(x),                       "elu"),
        (lambda x: F.softplus(x),                  "softplus"),
        (lambda x: F.hardtanh(x),                  "hardtanh"),
        (lambda x: F.hardswish(x),                 "hardswish"),
        (lambda x: F.mish(x),                      "mish"),
    ])
    def test_activation_backward(self, fn, name):
        cpu, vk_t = grad_pair(16, 16)
        fn(cpu).sum().backward()
        fn(vk_t).sum().backward()
        check(cpu.grad, vk_t.grad, msg=name)

    def test_softmax_backward(self):
        cpu, vk_t = grad_pair(8, 16)
        F.softmax(cpu, dim=-1).sum().backward()
        F.softmax(vk_t, dim=-1).sum().backward()
        check(cpu.grad, vk_t.grad)

    def test_log_softmax_backward(self):
        cpu, vk_t = grad_pair(8, 16)
        F.log_softmax(cpu, dim=-1).sum().backward()
        F.log_softmax(vk_t, dim=-1).sum().backward()
        check(cpu.grad, vk_t.grad)

    def test_softmax_backward_large(self):
        """Softmax backward for row_size > 256 (decomposed path)."""
        cpu, vk_t = grad_pair(4, 512)
        F.softmax(cpu, dim=-1).sum().backward()
        F.softmax(vk_t, dim=-1).sum().backward()
        check(cpu.grad, vk_t.grad, rtol=2e-3, atol=2e-3)


# ═══════════════════════════════════════════════════════════════════
# Normalization backward
# ═══════════════════════════════════════════════════════════════════

class TestNormalizationBackward:

    def test_batch_norm_train_backward(self):
        torch.manual_seed(0)
        bn_cpu = nn.BatchNorm2d(8)
        bn_vk = nn.BatchNorm2d(8).to("vulkan")
        bn_vk.load_state_dict(bn_cpu.state_dict())

        x_cpu, x_vk = grad_pair(2, 8, 4, 4)
        bn_cpu(x_cpu).sum().backward()
        bn_vk(x_vk).sum().backward()

        check(x_cpu.grad, x_vk.grad)
        check(bn_cpu.weight.grad, bn_vk.weight.grad.cpu())
        check(bn_cpu.bias.grad, bn_vk.bias.grad.cpu())

    def test_group_norm_backward(self):
        torch.manual_seed(0)
        gn_cpu = nn.GroupNorm(4, 8)
        gn_vk = nn.GroupNorm(4, 8).to("vulkan")
        gn_vk.load_state_dict(gn_cpu.state_dict())

        x_cpu, x_vk = grad_pair(2, 8, 4, 4)
        gn_cpu(x_cpu).sum().backward()
        gn_vk(x_vk).sum().backward()

        check(x_cpu.grad, x_vk.grad)
        check(gn_cpu.weight.grad, gn_vk.weight.grad.cpu())

    def test_layer_norm_backward_large(self):
        """Layer norm backward with norm_size > 256 (decomposed path)."""
        torch.manual_seed(0)
        ln_cpu = nn.LayerNorm(512)
        ln_vk = nn.LayerNorm(512).to("vulkan")
        ln_vk.load_state_dict(ln_cpu.state_dict())

        x_cpu, x_vk = grad_pair(2, 4, 512)
        ln_cpu(x_cpu).sum().backward()
        ln_vk(x_vk).sum().backward()

        check(x_cpu.grad, x_vk.grad, rtol=2e-3, atol=2e-3)
        check(ln_cpu.weight.grad, ln_vk.weight.grad.cpu(), rtol=2e-3, atol=2e-3)

    def test_rms_norm_backward_via_var(self):
        """RMSNorm backward (via x.var) gradient correctness."""
        cpu, vk_t = grad_pair(8, 32)
        w_cpu = torch.ones(32, requires_grad=True)
        w_vk = w_cpu.detach().to("vulkan").requires_grad_(True)

        def rms_norm(x, w):
            v = x.var(dim=-1, keepdim=True, correction=0)
            return x * torch.rsqrt(v + 1e-6) * w

        rms_norm(cpu, w_cpu).sum().backward()
        rms_norm(vk_t, w_vk).sum().backward()

        check(cpu.grad, vk_t.grad)
        check(w_cpu.grad, w_vk.grad)


# ═══════════════════════════════════════════════════════════════════
# Loss backward
# ═══════════════════════════════════════════════════════════════════

class TestLossBackward:

    def test_mse_backward(self):
        cpu, vk_t = grad_pair(8, 16)
        target = torch.randn(8, 16)
        F.mse_loss(cpu, target).backward()
        F.mse_loss(vk_t, vk(target)).backward()
        check(cpu.grad, vk_t.grad)

    def test_cross_entropy_backward(self):
        cpu, vk_t = grad_pair(8, 16)
        target = torch.randint(0, 16, (8,))
        F.cross_entropy(cpu, target).backward()
        F.cross_entropy(vk_t, vk(target)).backward()
        check(cpu.grad, vk_t.grad)

    def test_cross_entropy_ignore_index_backward(self):
        """CE backward with ignore_index=-100 (padding tokens)."""
        cpu, vk_t = grad_pair(8, 16)
        target = torch.randint(0, 16, (8,))
        target[2] = -100
        target[5] = -100
        F.cross_entropy(cpu, target, ignore_index=-100).backward()
        F.cross_entropy(vk_t, vk(target), ignore_index=-100).backward()
        check(cpu.grad, vk_t.grad)

    def test_bce_with_logits_backward(self):
        cpu, vk_t = grad_pair(8, 16)
        target = torch.rand(8, 16)  # binary targets in [0, 1]
        F.binary_cross_entropy_with_logits(cpu, target).backward()
        F.binary_cross_entropy_with_logits(vk_t, vk(target)).backward()
        check(cpu.grad, vk_t.grad)

    def test_nll_loss_backward(self):
        log_probs = F.log_softmax(torch.randn(8, 10), dim=-1)
        target = torch.randint(0, 10, (8,))
        lp_cpu = log_probs.detach().requires_grad_(True)
        lp_vk = log_probs.detach().to("vulkan").requires_grad_(True)
        F.nll_loss(lp_cpu, target).backward()
        F.nll_loss(lp_vk, vk(target)).backward()
        check(lp_cpu.grad, lp_vk.grad)

    def test_nll_loss_ignore_index_backward(self):
        log_probs = F.log_softmax(torch.randn(8, 10), dim=-1)
        target = torch.randint(0, 10, (8,))
        target[3] = -100
        lp_cpu = log_probs.detach().requires_grad_(True)
        lp_vk = log_probs.detach().to("vulkan").requires_grad_(True)
        F.nll_loss(lp_cpu, target, ignore_index=-100).backward()
        F.nll_loss(lp_vk, vk(target), ignore_index=-100).backward()
        check(lp_cpu.grad, lp_vk.grad)

    def test_smooth_l1_backward(self):
        cpu, vk_t = grad_pair(8, 16)
        target = torch.randn(8, 16)
        F.smooth_l1_loss(cpu, target).backward()
        F.smooth_l1_loss(vk_t, vk(target)).backward()
        check(cpu.grad, vk_t.grad)


# ═══════════════════════════════════════════════════════════════════
# SDPA backward
# ═══════════════════════════════════════════════════════════════════

class TestSDPABackward:

    def test_sdpa_backward_basic(self):
        torch.manual_seed(0)
        B, H, N, D = 2, 4, 8, 16
        q_cpu = torch.randn(B, H, N, D, requires_grad=True)
        k_cpu = torch.randn(B, H, N, D, requires_grad=True)
        v_cpu = torch.randn(B, H, N, D, requires_grad=True)
        q_vk = q_cpu.detach().to("vulkan").requires_grad_(True)
        k_vk = k_cpu.detach().to("vulkan").requires_grad_(True)
        v_vk = v_cpu.detach().to("vulkan").requires_grad_(True)

        F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu).sum().backward()
        F.scaled_dot_product_attention(q_vk, k_vk, v_vk).sum().backward()

        check(q_cpu.grad, q_vk.grad, rtol=2e-3, atol=2e-3)
        check(k_cpu.grad, k_vk.grad, rtol=2e-3, atol=2e-3)
        check(v_cpu.grad, v_vk.grad, rtol=2e-3, atol=2e-3)

    def test_sdpa_causal_backward(self):
        torch.manual_seed(0)
        B, H, N, D = 1, 4, 16, 32
        q_cpu = torch.randn(B, H, N, D, requires_grad=True)
        k_cpu = torch.randn(B, H, N, D, requires_grad=True)
        v_cpu = torch.randn(B, H, N, D, requires_grad=True)
        q_vk = q_cpu.detach().to("vulkan").requires_grad_(True)
        k_vk = k_cpu.detach().to("vulkan").requires_grad_(True)
        v_vk = v_cpu.detach().to("vulkan").requires_grad_(True)

        F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu, is_causal=True).sum().backward()
        F.scaled_dot_product_attention(q_vk, k_vk, v_vk, is_causal=True).sum().backward()

        check(q_cpu.grad, q_vk.grad, rtol=2e-3, atol=2e-3)
        check(v_cpu.grad, v_vk.grad, rtol=2e-3, atol=2e-3)


# ═══════════════════════════════════════════════════════════════════
# Non-contiguous tensor correctness
# ═══════════════════════════════════════════════════════════════════

class TestNonContiguousCorrectness:

    def test_var_transposed(self):
        """var on a transposed (non-contiguous) tensor."""
        cpu = torch.randn(8, 32).T  # shape (32, 8), stride (1, 32)
        assert not cpu.is_contiguous()
        check(cpu.var(dim=1), vk(cpu).var(dim=1))

    def test_sum_transposed(self):
        cpu = torch.randn(8, 16).T
        check(cpu.sum(dim=0), vk(cpu).sum(dim=0), rtol=1e-3, atol=1e-3)

    def test_mean_permuted(self):
        cpu = torch.randn(4, 8, 16).permute(2, 0, 1)  # (16, 4, 8)
        check(cpu.mean(dim=-1), vk(cpu).mean(dim=-1), rtol=1e-3, atol=1e-3)

    def test_add_strided(self):
        base = torch.randn(8, 16)
        cpu = base[:, ::2]  # stride-2 slice, non-contiguous
        assert not cpu.is_contiguous()
        b_cpu = torch.randn(8, 8)
        check(cpu + b_cpu, vk(cpu) + vk(b_cpu))

    def test_relu_non_contiguous(self):
        cpu = torch.randn(4, 4, 4).transpose(0, 2)
        check(F.relu(cpu), F.relu(vk(cpu)))

    def test_mm_transposed_input(self):
        """mm where one input is a transposed view."""
        a = torch.randn(16, 8)
        b = torch.randn(16, 4)
        # a.T @ b = (8, 16) @ (16, 4) = (8, 4)
        check(torch.mm(a.T, b), torch.mm(vk(a).T, vk(b)), rtol=2e-3, atol=2e-3)


# ═══════════════════════════════════════════════════════════════════
# f16 / bf16 forward correctness
# ═══════════════════════════════════════════════════════════════════

class TestHalfPrecisionCorrectness:

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_linear_half(self, dtype):
        torch.manual_seed(0)
        m = nn.Linear(16, 8, bias=True).to(dtype)
        m_vk = nn.Linear(16, 8, bias=True).to(dtype).to("vulkan")
        m_vk.load_state_dict(m.state_dict())
        x = torch.randn(4, 16, dtype=dtype)
        check(m(x), m_vk(vk(x)), rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_relu_half(self, dtype):
        x = torch.randn(16, 16, dtype=dtype)
        check(F.relu(x), F.relu(vk(x)))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_softmax_half(self, dtype):
        x = torch.randn(8, 32, dtype=dtype)
        check(F.softmax(x, dim=-1), F.softmax(vk(x), dim=-1), rtol=2e-3, atol=2e-3)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_layer_norm_half(self, dtype):
        torch.manual_seed(0)
        ln = nn.LayerNorm(16).to(dtype)
        ln_vk = nn.LayerNorm(16).to(dtype).to("vulkan")
        ln_vk.load_state_dict(ln.state_dict())
        x = torch.randn(4, 8, 16, dtype=dtype)
        check(ln(x), ln_vk(vk(x)), rtol=2e-3, atol=2e-3)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_var_half(self, dtype):
        # Vulkan var returns f32 (widen-compute-narrow); upcast CPU ref to match
        x = torch.randn(8, 32, dtype=dtype)
        check(x.var(dim=-1).float(), vk(x).var(dim=-1), rtol=2e-3, atol=2e-3)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_add_half(self, dtype):
        a = torch.randn(8, 16, dtype=dtype)
        b = torch.randn(8, 16, dtype=dtype)
        check(a + b, vk(a) + vk(b))

    def test_f16_training_step(self):
        """f16 training step: fwd + bwd + SGD step matches CPU trajectory."""
        torch.manual_seed(0)
        m_cpu = nn.Linear(16, 8).to(torch.float16)
        m_vk = nn.Linear(16, 8).to(torch.float16).to("vulkan")
        m_vk.load_state_dict(m_cpu.state_dict())

        x = torch.randn(4, 16, dtype=torch.float16)
        target = torch.randn(4, 8, dtype=torch.float16)

        out_cpu = m_cpu(x)
        F.mse_loss(out_cpu.float(), target.float()).backward()

        out_vk = m_vk(vk(x))
        F.mse_loss(out_vk.float(), vk(target).float()).backward()

        check(out_cpu, out_vk, rtol=2e-2, atol=2e-2)
        check(m_cpu.weight.grad, m_vk.weight.grad.cpu(), rtol=2e-2, atol=2e-2)


# ═══════════════════════════════════════════════════════════════════
# Shape ops backward
# ═══════════════════════════════════════════════════════════════════

class TestShapeOpsBackward:

    def test_permute_backward(self):
        cpu, vk_t = grad_pair(2, 4, 8)
        cpu.permute(2, 0, 1).sum().backward()
        vk_t.permute(2, 0, 1).sum().backward()
        check(cpu.grad, vk_t.grad)

    def test_cat_backward(self):
        a_cpu = torch.randn(4, 8, requires_grad=True)
        b_cpu = torch.randn(4, 8, requires_grad=True)
        a_vk = a_cpu.detach().to("vulkan").requires_grad_(True)
        b_vk = b_cpu.detach().to("vulkan").requires_grad_(True)
        torch.cat([a_cpu, b_cpu], dim=0).sum().backward()
        torch.cat([a_vk, b_vk], dim=0).sum().backward()
        check(a_cpu.grad, a_vk.grad)
        check(b_cpu.grad, b_vk.grad)

    def test_expand_backward(self):
        cpu, vk_t = grad_pair(1, 8)
        cpu.expand(4, 8).sum().backward()
        vk_t.expand(4, 8).sum().backward()
        check(cpu.grad, vk_t.grad)

    def test_reshape_backward(self):
        cpu, vk_t = grad_pair(4, 8)
        cpu.reshape(2, 16).sum().backward()
        vk_t.reshape(2, 16).sum().backward()
        check(cpu.grad, vk_t.grad)

    def test_gather_backward(self):
        cpu, vk_t = grad_pair(4, 8)
        idx = torch.randint(0, 8, (4, 4))
        torch.gather(cpu, 1, idx).sum().backward()
        torch.gather(vk_t, 1, vk(idx)).sum().backward()
        check(cpu.grad, vk_t.grad)

    def test_transpose_backward(self):
        cpu, vk_t = grad_pair(4, 8)
        cpu.T.sum().backward()
        vk_t.T.sum().backward()
        check(cpu.grad, vk_t.grad)


# ═══════════════════════════════════════════════════════════════════
# Reduction backward
# ═══════════════════════════════════════════════════════════════════

class TestReductionBackward:

    def test_sum_dim_backward(self):
        cpu, vk_t = grad_pair(4, 8, 16)
        cpu.sum(dim=1).sum().backward()
        vk_t.sum(dim=1).sum().backward()
        check(cpu.grad, vk_t.grad)

    def test_mean_dim_backward(self):
        cpu, vk_t = grad_pair(4, 8)
        cpu.mean(dim=0).sum().backward()
        vk_t.mean(dim=0).sum().backward()
        check(cpu.grad, vk_t.grad)

    def test_sum_multi_dim_backward(self):
        cpu, vk_t = grad_pair(4, 8, 16)
        cpu.sum(dim=[0, 2]).sum().backward()
        vk_t.sum(dim=[0, 2]).sum().backward()
        check(cpu.grad, vk_t.grad)

    def test_mean_backward(self):
        cpu, vk_t = grad_pair(8, 16)
        cpu.mean().backward()
        vk_t.mean().backward()
        check(cpu.grad, vk_t.grad)


# ═══════════════════════════════════════════════════════════════════
# Compound op correctness (multi-op chains)
# ═══════════════════════════════════════════════════════════════════

class TestCompoundOpsCorrectness:

    def test_rms_norm_forward(self):
        """RMSNorm forward: x / sqrt(mean(x^2) + eps) * weight."""
        torch.manual_seed(0)
        x_cpu = torch.randn(4, 32)
        w = torch.randn(32)
        x_vk = vk(x_cpu)
        w_vk = vk(w)

        def rms_norm(x, weight):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * weight

        check(rms_norm(x_cpu, w), rms_norm(x_vk, w_vk))

    def test_gelu_linear_chain_backward(self):
        """GELU + linear: backward through a common transformer FFN pattern."""
        torch.manual_seed(0)
        m_cpu = nn.Sequential(nn.Linear(16, 32, bias=False), nn.GELU(), nn.Linear(32, 16, bias=False))
        m_vk = nn.Sequential(nn.Linear(16, 32, bias=False), nn.GELU(), nn.Linear(32, 16, bias=False))
        m_vk.load_state_dict(m_cpu.state_dict())
        m_vk = m_vk.to("vulkan")

        x_cpu, x_vk = torch.randn(4, 16), torch.randn(4, 16).to("vulkan")
        x_cpu.requires_grad_(True)
        x_vk = vk(x_cpu.detach()).requires_grad_(True)

        m_cpu(x_cpu).sum().backward()
        m_vk(x_vk).sum().backward()

        check(x_cpu.grad, x_vk.grad)

    def test_attention_with_rms_norm_backward(self):
        """Transformer-style block: RMSNorm + scaled_dot_product_attention backward."""
        torch.manual_seed(0)
        B, H, S, D = 1, 4, 8, 16
        hidden = D * H  # 64

        x_cpu = torch.randn(B, S, hidden, requires_grad=True)
        x_vk = x_cpu.detach().to("vulkan").requires_grad_(True)

        def forward(x):
            # RMS norm
            normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
            # Reshape to (B, H, S, D) for attention
            q = normed.view(B, S, H, D).transpose(1, 2)
            k = normed.view(B, S, H, D).transpose(1, 2)
            v = normed.view(B, S, H, D).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return out.transpose(1, 2).reshape(B, S, hidden)

        forward(x_cpu).sum().backward()
        forward(x_vk).sum().backward()

        check(x_cpu.grad, x_vk.grad, rtol=2e-3, atol=2e-3)

    def test_var_in_rms_norm_train_correctness(self):
        """Training step with var-based RMSNorm matches CPU exactly."""
        torch.manual_seed(0)

        class RMSNormModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(16, 16, bias=False)
                self.scale = nn.Parameter(torch.ones(16))

            def forward(self, x):
                v = x.var(dim=-1, keepdim=True, correction=0)
                normed = x * torch.rsqrt(v + 1e-6) * self.scale
                return self.fc(normed)

        m_cpu = RMSNormModel()
        m_vk = RMSNormModel().to("vulkan")
        m_vk.load_state_dict(m_cpu.state_dict())

        x = torch.randn(4, 16)
        target = torch.randn(4, 16)

        opt_cpu = torch.optim.SGD(m_cpu.parameters(), lr=0.01)
        opt_vk = torch.optim.SGD(m_vk.parameters(), lr=0.01)

        for _ in range(3):
            loss_cpu = F.mse_loss(m_cpu(x), target)
            opt_cpu.zero_grad()
            loss_cpu.backward()
            opt_cpu.step()

            loss_vk = F.mse_loss(m_vk(vk(x)), vk(target))
            opt_vk.zero_grad()
            loss_vk.backward()
            opt_vk.step()

        for (n, p_cpu), (_, p_vk) in zip(m_cpu.named_parameters(), m_vk.named_parameters()):
            check(p_cpu.data, p_vk.data.cpu(), rtol=2e-3, atol=2e-3, msg=n)
