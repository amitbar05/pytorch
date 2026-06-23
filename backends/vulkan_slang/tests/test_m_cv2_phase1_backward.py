"""M-CV.2 Phase 1 — high-priority backward CPU-oracle parity tests.

Covers 8 backward ops currently missing dedicated regression coverage:

Unary:
  - aten.cos_backward      (pair with covered sin_backward)
  - aten.log2_backward     (pair with covered log_backward; input > 0)
  - aten.log10_backward    (pair with covered log_backward; input > 0)
  - aten.exp2_backward     (pair with covered exp_backward)
  - aten.expm1_backward    (numerically stable variant; small-x regime)

Binary:
  - aten.atan2_backward    (high scientific-computing impact)
  - aten.maximum_backward  (piecewise routing; avoid tied values)
  - aten.minimum_backward  (piecewise routing; avoid tied values)

All ops route through ``bwd_diff_table.py`` via Slang ``bwd_diff()``
(anti-goal #3: no hand-written ``aten.<op>_backward`` lowerings). Forward
helpers live in ``shaders/lib/pointwise.slang`` and are float-only — no fp16
overloads — so these tests run at float32 only.

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


class TestMCV2UnaryBackward:
    """Unary backward parity — d/dx via Slang bwd_diff()."""

    def test_cos_backward_matches_cpu(self):
        """d/dx cos(x) = -sin(x). General-purpose input range."""
        torch.manual_seed(42)
        x_cpu = torch.randn(64, requires_grad=True)
        x_vk = x_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(x):
            return torch.cos(x).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(x_cpu).backward()
        compiled(x_vk).backward()

        torch.testing.assert_close(x_vk.grad.cpu(), x_cpu.grad, atol=1e-5, rtol=1e-5)

    def test_log2_backward_matches_cpu(self):
        """d/dx log2(x) = 1 / (x * ln 2). Input must be > 0."""
        torch.manual_seed(42)
        # Use rand() + 0.5 to keep input in [0.5, 1.5] — positive, away from 0.
        x_cpu = (torch.rand(64) + 0.5).requires_grad_(True)
        x_vk = x_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(x):
            return torch.log2(x).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(x_cpu).backward()
        compiled(x_vk).backward()

        torch.testing.assert_close(x_vk.grad.cpu(), x_cpu.grad, atol=1e-5, rtol=1e-5)

    def test_log10_backward_matches_cpu(self):
        """d/dx log10(x) = 1 / (x * ln 10). Input must be > 0."""
        torch.manual_seed(42)
        x_cpu = (torch.rand(64) + 0.5).requires_grad_(True)
        x_vk = x_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(x):
            return torch.log10(x).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(x_cpu).backward()
        compiled(x_vk).backward()

        torch.testing.assert_close(x_vk.grad.cpu(), x_cpu.grad, atol=1e-5, rtol=1e-5)

    def test_exp2_backward_matches_cpu(self):
        """d/dx 2^x = 2^x * ln 2. Bound input range to avoid overflow."""
        torch.manual_seed(42)
        # Keep |x| modest so exp2(x) stays well within f32 dynamic range.
        x_cpu = (torch.randn(64) * 0.5).requires_grad_(True)
        x_vk = x_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(x):
            return torch.exp2(x).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(x_cpu).backward()
        compiled(x_vk).backward()

        torch.testing.assert_close(x_vk.grad.cpu(), x_cpu.grad, atol=1e-5, rtol=1e-5)

    def test_expm1_backward_matches_cpu(self):
        """d/dx expm1(x) = exp(x). Test the numerically-stable small-x regime
        where expm1's accuracy advantage over exp(x) - 1 manifests in the
        forward pass; the backward derivative itself equals exp(x) and is
        mathematically identical to ``exp_backward``."""
        torch.manual_seed(42)
        # Small magnitudes: ~|x| < 0.05 — the regime expm1 exists for.
        x_cpu = (torch.randn(64) * 0.01).requires_grad_(True)
        x_vk = x_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(x):
            return torch.expm1(x).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(x_cpu).backward()
        compiled(x_vk).backward()

        torch.testing.assert_close(x_vk.grad.cpu(), x_cpu.grad, atol=1e-5, rtol=1e-5)


# ─────────────────────────────────────────────────────────────────────
# Binary backward parity
# ─────────────────────────────────────────────────────────────────────


class TestMCV2BinaryBackward:
    """Binary backward parity — two-arg ops via Slang bwd_diff()."""

    def test_atan2_backward_matches_cpu(self):
        """d/dy atan2(y, x) = x / (x^2 + y^2);
        d/dx atan2(y, x) = -y / (x^2 + y^2). Avoid (0,0) singularity by
        offsetting from zero."""
        torch.manual_seed(42)
        # Offset both inputs away from 0 to avoid the (0,0) singularity.
        a_cpu = (torch.randn(64) + 1.0).requires_grad_(True)
        b_cpu = (torch.randn(64) + 1.0).requires_grad_(True)
        a_vk = a_cpu.detach().clone().to("vulkan:0").requires_grad_(True)
        b_vk = b_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(a, b):
            return torch.atan2(a, b).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(a_cpu, b_cpu).backward()
        compiled(a_vk, b_vk).backward()

        torch.testing.assert_close(a_vk.grad.cpu(), a_cpu.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(b_vk.grad.cpu(), b_cpu.grad, atol=1e-5, rtol=1e-5)

    def test_maximum_backward_matches_cpu(self):
        """maximum(a, b) routes grad to whichever operand is larger.

        Inputs are drawn from disjoint distributions (``a`` from ``randn() - 2``,
        ``b`` from ``randn() + 2``) so no element is tied. PyTorch's gradient
        distribution at tie points is implementation-defined and testing
        exact parity there is fragile."""
        torch.manual_seed(42)
        a_cpu = (torch.randn(64) - 2.0).requires_grad_(True)
        b_cpu = (torch.randn(64) + 2.0).requires_grad_(True)
        a_vk = a_cpu.detach().clone().to("vulkan:0").requires_grad_(True)
        b_vk = b_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(a, b):
            return torch.maximum(a, b).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(a_cpu, b_cpu).backward()
        compiled(a_vk, b_vk).backward()

        torch.testing.assert_close(a_vk.grad.cpu(), a_cpu.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(b_vk.grad.cpu(), b_cpu.grad, atol=1e-5, rtol=1e-5)

    def test_minimum_backward_matches_cpu(self):
        """minimum(a, b) routes grad to whichever operand is smaller.

        Same tie-avoidance discipline as ``test_maximum_backward_matches_cpu``:
        disjoint distributions ensure no element ties, sidestepping
        implementation-defined behaviour at the kink."""
        torch.manual_seed(42)
        a_cpu = (torch.randn(64) - 2.0).requires_grad_(True)
        b_cpu = (torch.randn(64) + 2.0).requires_grad_(True)
        a_vk = a_cpu.detach().clone().to("vulkan:0").requires_grad_(True)
        b_vk = b_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(a, b):
            return torch.minimum(a, b).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(a_cpu, b_cpu).backward()
        compiled(a_vk, b_vk).backward()

        torch.testing.assert_close(a_vk.grad.cpu(), a_cpu.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(b_vk.grad.cpu(), b_cpu.grad, atol=1e-5, rtol=1e-5)
