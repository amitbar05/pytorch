"""Tests comparing training on CPU vs Vulkan backend.

Verifies that:
- Loss curves track each other within tolerance
- Final parameters are numerically close
- Gradients match CPU reference
- Both devices converge on the same objective
"""

import copy
import time

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


def make_vulkan(t):
    return t.to("vulkan:0")


def assert_loss_close(cpu_losses, vk_losses, rtol=0.05, label=""):
    """Assert that loss curves track within relative tolerance."""
    for i, (c, v) in enumerate(zip(cpu_losses, vk_losses)):
        rel = abs(c - v) / (abs(c) + 1e-8)
        assert rel < rtol, (
            f"{label} step {i}: CPU loss={c:.6f}, Vulkan loss={v:.6f}, rel={rel:.4f}"
        )


class TestLinearCpuVsVulkan:
    """Compare CPU vs Vulkan for linear model training."""

    def test_sgd_linear_loss_curve(self):
        """SGD on a linear model: loss curves match CPU within 5%."""
        torch.manual_seed(42)
        model_cpu = nn.Linear(16, 8, bias=False)
        model_vk = copy.deepcopy(model_cpu).to("vulkan:0")

        opt_cpu = torch.optim.SGD(model_cpu.parameters(), lr=0.01)
        opt_vk = torch.optim.SGD(model_vk.parameters(), lr=0.01)

        torch.manual_seed(1)
        x = torch.randn(8, 16)
        y = torch.randn(8, 8)
        xv = x.to("vulkan:0")
        yv = y.to("vulkan:0")

        cpu_losses, vk_losses = [], []
        for _ in range(10):
            opt_cpu.zero_grad()
            loss_cpu = F.mse_loss(model_cpu(x), y)
            loss_cpu.backward()
            opt_cpu.step()
            cpu_losses.append(loss_cpu.item())

            opt_vk.zero_grad()
            loss_vk = F.mse_loss(model_vk(xv), yv)
            loss_vk.backward()
            opt_vk.step()
            vk_losses.append(loss_vk.item())

        assert_loss_close(cpu_losses, vk_losses, label="Linear-SGD")
        # Both should decrease
        assert cpu_losses[-1] < cpu_losses[0], "CPU loss did not decrease"
        assert vk_losses[-1] < vk_losses[0], "Vulkan loss did not decrease"

    def test_adam_mlp_params_match(self):
        """Adam on MLP: final parameters match CPU within tolerance."""
        torch.manual_seed(7)
        model_cpu = nn.Sequential(
            nn.Linear(8, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, 4, bias=True),
        )
        model_vk = copy.deepcopy(model_cpu).to("vulkan:0")

        opt_cpu = torch.optim.Adam(model_cpu.parameters(), lr=1e-3)
        opt_vk = torch.optim.Adam(model_vk.parameters(), lr=1e-3)

        torch.manual_seed(3)
        x = torch.randn(16, 8)
        y = torch.randn(16, 4)
        xv = x.to("vulkan:0")
        yv = y.to("vulkan:0")

        for _ in range(5):
            opt_cpu.zero_grad()
            F.mse_loss(model_cpu(x), y).backward()
            opt_cpu.step()

            opt_vk.zero_grad()
            F.mse_loss(model_vk(xv), yv).backward()
            opt_vk.step()

        for (n, p_cpu), (_, p_vk) in zip(
            model_cpu.named_parameters(), model_vk.named_parameters()
        ):
            torch.testing.assert_close(
                p_vk.cpu(),
                p_cpu,
                rtol=RTOL,
                atol=ATOL,
                msg=f"Parameter {n} mismatch after Adam",
            )

    def test_gradients_match_cpu(self):
        """Gradients from backward pass match CPU reference."""
        torch.manual_seed(0)
        model_cpu = nn.Sequential(
            nn.Linear(8, 16, bias=True),
            nn.GELU(),
            nn.Linear(16, 4, bias=True),
        )
        model_vk = copy.deepcopy(model_cpu).to("vulkan:0")

        x = torch.randn(4, 8)
        y = torch.randn(4, 4)
        xv = x.to("vulkan:0")
        yv = y.to("vulkan:0")

        F.mse_loss(model_cpu(x), y).backward()
        F.mse_loss(model_vk(xv), yv).backward()

        for (n, p_cpu), (_, p_vk) in zip(
            model_cpu.named_parameters(), model_vk.named_parameters()
        ):
            assert p_vk.grad is not None, f"Gradient missing for {n} on Vulkan"
            torch.testing.assert_close(
                p_vk.grad.cpu(),
                p_cpu.grad,
                rtol=RTOL,
                atol=ATOL,
                msg=f"Gradient mismatch for {n}",
            )


class TestTransformerCpuVsVulkan:
    """Compare CPU vs Vulkan for transformer-style training."""

    def _make_model(self, d=32, ff=64, n_layers=2):
        layers = []
        for _ in range(n_layers):
            layers.extend(
                [
                    nn.LayerNorm(d),
                    nn.Linear(d, ff, bias=False),
                    nn.GELU(),
                    nn.Linear(ff, d, bias=False),
                ]
            )
        return nn.Sequential(*layers)

    def test_ffn_loss_curve(self):
        """FFN with LayerNorm loss tracks CPU within 5%."""
        torch.manual_seed(10)
        model_cpu = self._make_model()
        model_vk = copy.deepcopy(model_cpu).to("vulkan:0")

        lr = 1e-3
        opt_cpu = torch.optim.Adam(model_cpu.parameters(), lr=lr)
        opt_vk = torch.optim.Adam(model_vk.parameters(), lr=lr)

        x = torch.randn(4, 8, 32)
        y = torch.randn(4, 8, 32)
        xv = x.to("vulkan:0")
        yv = y.to("vulkan:0")

        cpu_losses, vk_losses = [], []
        for _ in range(8):
            opt_cpu.zero_grad()
            F.mse_loss(model_cpu(x), y).backward()
            opt_cpu.step()
            cpu_losses.append(F.mse_loss(model_cpu(x), y).item())

            opt_vk.zero_grad()
            F.mse_loss(model_vk(xv), yv).backward()
            opt_vk.step()
            vk_losses.append(F.mse_loss(model_vk(xv), yv).item())

        assert_loss_close(cpu_losses, vk_losses, label="FFN-Adam")
        assert vk_losses[-1] < vk_losses[0], "Vulkan loss did not decrease"

    def test_cross_entropy_training(self):
        """Cross-entropy classification training matches CPU."""
        torch.manual_seed(5)
        model_cpu = nn.Sequential(
            nn.Linear(32, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 10, bias=True),
        )
        model_vk = copy.deepcopy(model_cpu).to("vulkan:0")

        opt_cpu = torch.optim.SGD(model_cpu.parameters(), lr=0.05, momentum=0.9)
        opt_vk = torch.optim.SGD(model_vk.parameters(), lr=0.05, momentum=0.9)

        x = torch.randn(32, 32)
        targets = torch.randint(0, 10, (32,))
        xv = x.to("vulkan:0")
        targets_v = targets.to("vulkan:0")

        cpu_losses, vk_losses = [], []
        for _ in range(10):
            opt_cpu.zero_grad()
            loss_cpu = F.cross_entropy(model_cpu(x), targets)
            loss_cpu.backward()
            opt_cpu.step()
            cpu_losses.append(loss_cpu.item())

            opt_vk.zero_grad()
            loss_vk = F.cross_entropy(model_vk(xv), targets_v)
            loss_vk.backward()
            opt_vk.step()
            vk_losses.append(loss_vk.item())

        assert_loss_close(cpu_losses, vk_losses, label="CrossEntropy-SGD")

    def test_rms_norm_model(self):
        """RMSNorm-based model training matches CPU."""
        torch.manual_seed(3)

        class RMSNormMLP(nn.Module):
            def __init__(self, d=32):
                super().__init__()
                self.w1 = nn.Linear(d, d * 2, bias=False)
                self.w2 = nn.Linear(d * 2, d, bias=False)
                self.scale = nn.Parameter(torch.ones(d))

            def forward(self, x):
                var = x.pow(2).mean(-1, keepdim=True)
                x = x * torch.rsqrt(var + 1e-6) * self.scale
                return self.w2(F.relu(self.w1(x)))

        model_cpu = RMSNormMLP()
        model_vk = copy.deepcopy(model_cpu).to("vulkan:0")

        opt_cpu = torch.optim.Adam(model_cpu.parameters(), lr=1e-3)
        opt_vk = torch.optim.Adam(model_vk.parameters(), lr=1e-3)

        x = torch.randn(8, 32)
        y = torch.randn(8, 32)
        xv = x.to("vulkan:0")
        yv = y.to("vulkan:0")

        cpu_losses, vk_losses = [], []
        for _ in range(8):
            opt_cpu.zero_grad()
            F.mse_loss(model_cpu(x), y).backward()
            opt_cpu.step()
            cpu_losses.append(F.mse_loss(model_cpu(x), y).item())

            opt_vk.zero_grad()
            F.mse_loss(model_vk(xv), yv).backward()
            opt_vk.step()
            vk_losses.append(F.mse_loss(model_vk(xv), yv).item())

        assert_loss_close(cpu_losses, vk_losses, label="RMSNorm-Adam")


class TestVulkanAdamWBf16:
    """Test batched bf16 AdamW optimizer correctness vs CPU reference."""

    def test_adamw_bf16_single_param(self):
        """bf16 AdamW single param: matches CPU f32 training within bf16 precision."""
        import torch_vulkan

        torch.manual_seed(0)
        p_cpu = torch.randn(64, dtype=torch.float32, requires_grad=True)
        grad_cpu = torch.randn(64, dtype=torch.float32)

        # CPU AdamW reference
        opt_cpu = torch.optim.AdamW([p_cpu], lr=1e-3, weight_decay=0.01)
        opt_cpu.zero_grad()
        p_cpu.grad = grad_cpu.clone()
        opt_cpu.step()

        # Vulkan bf16 with master weights
        p_vk = p_cpu.detach().clone().bfloat16().to("vulkan:0")
        opt_vk = torch_vulkan.AdamW(
            [p_vk], lr=1e-3, weight_decay=0.01, master_weights=True
        )
        grad_vk = grad_cpu.bfloat16().to("vulkan:0")
        p_vk.grad = grad_vk
        opt_vk.step()

        # bf16 has ~7-bit mantissa, so tolerance is ~1e-2
        torch.testing.assert_close(
            p_vk.cpu().float(),
            p_cpu.detach(),
            rtol=2e-2,
            atol=2e-2,
            msg="bf16 AdamW single param mismatch",
        )

    def test_adamw_bf16_batch_vs_sequential(self):
        """Batched bf16 AdamW produces same result as sequential per-param calls."""
        import torch_vulkan

        torch.manual_seed(1)
        N = 8
        params_init = [torch.randn(32, dtype=torch.bfloat16) for _ in range(N)]
        grads_init = [torch.randn(32, dtype=torch.bfloat16) for _ in range(N)]

        # Run batch optimizer (new path: multiple params share one dispatch per 4)
        params_batch = [p.clone().to("vulkan:0") for p in params_init]
        grads_batch = [g.clone().to("vulkan:0") for g in grads_init]
        opt_batch = torch_vulkan.AdamW(
            params_batch, lr=1e-3, weight_decay=0.01, master_weights=True
        )
        for pb, gb in zip(params_batch, grads_batch):
            pb.grad = gb.clone()
        opt_batch.step()

        # Run sequential optimizer (one optimizer per param)
        params_seq = [p.clone().to("vulkan:0") for p in params_init]
        grads_seq = [g.clone().to("vulkan:0") for g in grads_init]
        opts_seq = [
            torch_vulkan.AdamW([ps], lr=1e-3, weight_decay=0.01, master_weights=True)
            for ps in params_seq
        ]
        for ps, gs, opt_s in zip(params_seq, grads_seq, opts_seq):
            ps.grad = gs.clone()
            opt_s.step()

        for i, (pb, ps) in enumerate(zip(params_batch, params_seq)):
            torch.testing.assert_close(
                pb.cpu().float(),
                ps.cpu().float(),
                rtol=1e-5,
                atol=1e-5,
                msg=f"Batch vs sequential mismatch for param {i}",
            )

    def test_adamw_bf16_dispatch_count(self):
        """Batched bf16 AdamW: ceil(N/4) dispatches, not 3N."""
        import torch_vulkan

        torch.manual_seed(2)
        N = 8  # 8 params → ceil(8/4) = 2 dispatches
        params = [
            nn.Parameter(torch.randn(64, dtype=torch.bfloat16).to("vulkan:0"))
            for _ in range(N)
        ]
        grads = [torch.randn(64, dtype=torch.bfloat16).to("vulkan:0") for _ in range(N)]
        for p, g in zip(params, grads):
            p.grad = g

        opt = torch_vulkan.AdamW(
            params, lr=1e-3, weight_decay=0.01, master_weights=True
        )

        # Warm up (initializes state)
        for p, g in zip(params, grads):
            p.grad = g.clone()
        opt.step()

        # Measure
        for p, g in zip(params, grads):
            p.grad = g.clone()
        torch_vulkan._c_ext._reset_perf_counters()
        opt.step()
        d = torch_vulkan._c_ext._get_dispatch_count()

        expected_dispatches = (N + 3) // 4  # ceil(N/4)
        # Allow small overhead (zero_grad etc.)
        assert d <= expected_dispatches + 2, (
            f"bf16 AdamW batch: expected ~{expected_dispatches} dispatches, got {d}"
        )
        assert d < N, f"Not batching: dispatches {d} >= N={N}"

    def test_adamw_bf16_loss_curve(self):
        """Model with bf16 params trained with AdamW converges (matches CPU trend)."""
        import torch_vulkan

        torch.manual_seed(42)
        model_cpu = nn.Sequential(
            nn.Linear(16, 32, bias=False), nn.ReLU(), nn.Linear(32, 8, bias=False)
        )
        model_vk = copy.deepcopy(model_cpu).bfloat16().to("vulkan:0")
        model_cpu = model_cpu.float()

        opt_cpu = torch.optim.AdamW(model_cpu.parameters(), lr=1e-3, weight_decay=0.01)
        opt_vk = torch_vulkan.AdamW(
            list(model_vk.parameters()), lr=1e-3, weight_decay=0.01, master_weights=True
        )

        x_cpu = torch.randn(8, 16)
        y_cpu = torch.randn(8, 8)
        x_vk = x_cpu.bfloat16().to("vulkan:0")
        y_vk = y_cpu.bfloat16().to("vulkan:0")

        cpu_losses, vk_losses = [], []
        for _ in range(10):
            opt_cpu.zero_grad()
            F.mse_loss(model_cpu(x_cpu), y_cpu).backward()
            opt_cpu.step()
            cpu_losses.append(F.mse_loss(model_cpu(x_cpu), y_cpu).item())

            for p in model_vk.parameters():
                p.grad = None
            out_vk = model_vk(x_vk).float()
            y_f = y_vk.float()
            F.mse_loss(out_vk, y_f).backward()
            opt_vk.step()
            with torch.no_grad():
                vk_losses.append(F.mse_loss(model_vk(x_vk).float(), y_f).item())

        assert vk_losses[-1] < vk_losses[0], (
            f"bf16 AdamW model did not converge: {vk_losses[0]:.4f} -> {vk_losses[-1]:.4f}"
        )
        # Both should show downward trend
        assert cpu_losses[-1] < cpu_losses[0]

    def test_adamw_f16_dispatch_count(self):
        """Batched f16 AdamW: ceil(N/4) dispatches, not 3N."""
        import torch_vulkan

        torch.manual_seed(3)
        N = 4  # exactly 1 batch dispatch
        params = [
            nn.Parameter(torch.randn(32, dtype=torch.float16).to("vulkan:0"))
            for _ in range(N)
        ]
        grads = [torch.randn(32, dtype=torch.float16).to("vulkan:0") for _ in range(N)]

        opt = torch_vulkan.AdamW(
            params, lr=1e-3, weight_decay=0.01, master_weights=True
        )

        # Warm up
        for p, g in zip(params, grads):
            p.grad = g.clone()
        opt.step()

        # Measure
        for p, g in zip(params, grads):
            p.grad = g.clone()
        torch_vulkan._c_ext._reset_perf_counters()
        opt.step()
        d = torch_vulkan._c_ext._get_dispatch_count()

        assert d == 1, f"f16 AdamW 4 params expected 1 dispatch, got {d}"


class TestTimingComparison:
    """Compare wall-clock timing between CPU and Vulkan training."""

    def test_timing_mlp_training(self):
        """Report timing for MLP training on CPU vs Vulkan (informational)."""
        torch.manual_seed(0)
        model_base = nn.Sequential(
            nn.Linear(128, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 64, bias=True),
        )
        model_cpu = model_base
        model_vk = copy.deepcopy(model_base).to("vulkan:0")

        opt_cpu = torch.optim.Adam(model_cpu.parameters(), lr=1e-3)
        opt_vk = torch.optim.Adam(model_vk.parameters(), lr=1e-3)

        x = torch.randn(32, 128)
        y = torch.randn(32, 64)
        xv = x.to("vulkan:0")
        yv = y.to("vulkan:0")

        # Warmup
        for _ in range(3):
            opt_cpu.zero_grad()
            F.mse_loss(model_cpu(x), y).backward()
            opt_cpu.step()
            opt_vk.zero_grad()
            F.mse_loss(model_vk(xv), yv).backward()
            opt_vk.step()

        # Time CPU
        n_steps = 20
        t0 = time.perf_counter()
        for _ in range(n_steps):
            opt_cpu.zero_grad()
            F.mse_loss(model_cpu(x), y).backward()
            opt_cpu.step()
        cpu_ms = (time.perf_counter() - t0) * 1000 / n_steps

        # Time Vulkan (flush for sync)
        import torch_vulkan

        t0 = time.perf_counter()
        for _ in range(n_steps):
            opt_vk.zero_grad()
            F.mse_loss(model_vk(xv), yv).backward()
            opt_vk.step()
        torch_vulkan._c_ext._flush()
        vk_ms = (time.perf_counter() - t0) * 1000 / n_steps

        print(
            f"\nMLP training step: CPU={cpu_ms:.2f}ms, Vulkan={vk_ms:.2f}ms, ratio={cpu_ms / vk_ms:.2f}x"
        )
        # Just verify Vulkan runs without error; timing is informational
        assert True
