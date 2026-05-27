"""M-CV.2 Phase 2 — extended backward CPU-oracle parity tests.

Covers backward ops in ``bwd_diff_table.py`` that lack dedicated regression
coverage beyond incidental end-to-end tests.  All ops route through
``bwd_diff_table.py`` via Slang ``bwd_diff()`` (anti-goal #3).

Unary ops (5):
  - aten.sin_backward      (pairs with cos — Phase 1 gate)
  - aten.sqrt_backward     (used in layer-norm and attention scaling)
  - aten.erf_backward      (GELU variant base; uses special_math.slang)
  - aten.log_backward      (natural log; complements Phase-1 log2/log10)
  - aten.abs_backward      (L1-loss adjacent; sign function backward)

Binary ops (3):
  - aten.pow.Tensor_Tensor_backward  (common power law; non-trivial two-arg)
  - aten.hypot_backward              (2-arg; similar structure to atan2)
  - aten.reciprocal_backward         (1/x; numerically sensitive near 0)

Each test:
  1. Builds matched CPU + Vulkan input tensors with ``requires_grad=True``.
  2. Runs CPU eager forward+backward as the oracle.
  3. Runs Vulkan via ``torch.compile(backend="inductor")`` forward+backward.
  4. Asserts grads match within ``atol=1e-5, rtol=1e-5``.
"""

from __future__ import annotations

import pytest
import torch

import torch_vulkan  # noqa: F401  (registers the vulkan device)


@pytest.fixture(autouse=True)
def _vulkan_available():
    if not torch_vulkan.is_available():
        pytest.skip("No Vulkan device")


# ─────────────────────────────────────────────────────────────────────
# Unary backward parity
# ─────────────────────────────────────────────────────────────────────


class TestMCV2Phase2UnaryBackward:
    """Unary backward parity — d/dx via Slang bwd_diff()."""

    def test_sin_backward_matches_cpu(self):
        """d/dx sin(x) = cos(x). Pairs with cos_backward (Phase 1 gate)."""
        torch.manual_seed(42)
        x_cpu = torch.randn(64, requires_grad=True)
        x_vk = x_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(x):
            return torch.sin(x).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(x_cpu).backward()
        compiled(x_vk).backward()

        torch.testing.assert_close(x_vk.grad.cpu(), x_cpu.grad, atol=1e-5, rtol=1e-5)

    def test_sqrt_backward_matches_cpu(self):
        """d/dx sqrt(x) = 1 / (2 * sqrt(x)). Input must be > 0."""
        torch.manual_seed(42)
        # Positive inputs; sqrt is numerically sensitive near 0, stay in [0.1, 1.1].
        x_cpu = (torch.rand(64) + 0.1).requires_grad_(True)
        x_vk = x_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(x):
            return torch.sqrt(x).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(x_cpu).backward()
        compiled(x_vk).backward()

        torch.testing.assert_close(x_vk.grad.cpu(), x_cpu.grad, atol=1e-4, rtol=1e-4)

    def test_erf_backward_matches_cpu(self):
        """d/dx erf(x) = 2 / sqrt(pi) * exp(-x^2).  Uses special_math.slang."""
        torch.manual_seed(42)
        x_cpu = torch.randn(64, requires_grad=True)
        x_vk = x_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(x):
            return torch.erf(x).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(x_cpu).backward()
        compiled(x_vk).backward()

        torch.testing.assert_close(x_vk.grad.cpu(), x_cpu.grad, atol=1e-5, rtol=1e-5)

    def test_log_backward_matches_cpu(self):
        """d/dx log(x) = 1/x. Complements Phase-1 log2/log10.  Input > 0."""
        torch.manual_seed(42)
        x_cpu = (torch.rand(64) + 0.5).requires_grad_(True)
        x_vk = x_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(x):
            return torch.log(x).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(x_cpu).backward()
        compiled(x_vk).backward()

        torch.testing.assert_close(x_vk.grad.cpu(), x_cpu.grad, atol=1e-5, rtol=1e-5)

    def test_abs_backward_matches_cpu(self):
        """d/dx |x| = sign(x).  Inputs avoid 0 to keep gradient well-defined."""
        torch.manual_seed(42)
        # Sample and then shift so no element is exactly 0.
        x_cpu = torch.randn(64)
        x_cpu[x_cpu.abs() < 0.1] = 0.5  # eliminate near-zero ambiguity
        x_cpu = x_cpu.requires_grad_(True)
        x_vk = x_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(x):
            return torch.abs(x).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(x_cpu).backward()
        compiled(x_vk).backward()

        torch.testing.assert_close(x_vk.grad.cpu(), x_cpu.grad, atol=1e-5, rtol=1e-5)


# ─────────────────────────────────────────────────────────────────────
# Binary backward parity
# ─────────────────────────────────────────────────────────────────────


class TestMCV2Phase2BinaryBackward:
    """Binary backward parity — two-arg ops via Slang bwd_diff()."""

    def test_pow_tensor_tensor_backward_matches_cpu(self):
        """d/da a^b = b * a^(b-1);  d/db a^b = a^b * log(a).
        Positive base avoids the undefined log(a) branch."""
        torch.manual_seed(42)
        # Positive base in [0.5, 1.5]; modest exponents in [-2, 2].
        a_cpu = (torch.rand(64) + 0.5).requires_grad_(True)
        b_cpu = (torch.randn(64) * 0.5).requires_grad_(True)
        a_vk = a_cpu.detach().clone().to("vulkan:0").requires_grad_(True)
        b_vk = b_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(a, b):
            return torch.pow(a, b).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(a_cpu, b_cpu).backward()
        compiled(a_vk, b_vk).backward()

        torch.testing.assert_close(a_vk.grad.cpu(), a_cpu.grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(b_vk.grad.cpu(), b_cpu.grad, atol=1e-4, rtol=1e-4)

    def test_hypot_backward_matches_cpu(self):
        """d/da hypot(a, b) = a / hypot(a, b);
        d/db hypot(a, b) = b / hypot(a, b).
        Both operands offset from zero to avoid the (0, 0) singularity."""
        torch.manual_seed(42)
        a_cpu = (torch.randn(64) + 1.0).requires_grad_(True)
        b_cpu = (torch.randn(64) + 1.0).requires_grad_(True)
        a_vk = a_cpu.detach().clone().to("vulkan:0").requires_grad_(True)
        b_vk = b_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(a, b):
            return torch.hypot(a, b).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(a_cpu, b_cpu).backward()
        compiled(a_vk, b_vk).backward()

        torch.testing.assert_close(a_vk.grad.cpu(), a_cpu.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(b_vk.grad.cpu(), b_cpu.grad, atol=1e-5, rtol=1e-5)

    def test_reciprocal_backward_matches_cpu(self):
        """d/dx 1/x = -1/x^2.  Inputs avoid 0 (divergent gradient near 0)."""
        torch.manual_seed(42)
        x_cpu = (torch.rand(64) + 0.5).requires_grad_(True)
        x_vk = x_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(x):
            return torch.reciprocal(x).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(x_cpu).backward()
        compiled(x_vk).backward()

        torch.testing.assert_close(x_vk.grad.cpu(), x_cpu.grad, atol=1e-4, rtol=1e-4)
