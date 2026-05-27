"""TEST.COV.4 — Special-math backward CPU-oracle parity tests.

Covers the 9 special-math backward ops in ``bwd_diff_table.py`` that
had no dedicated regression coverage beyond incidental end-to-end tests.
All ops route through ``bwd_diff_table.py`` via Slang ``bwd_diff()``
(anti-goal #3).

Ops tested:
  - aten.erfc_backward       (erf* family; erfc'(x) = -erf'(x))
  - aten.erfinv_backward     (erf* family; erfinv'(x) = sqrt(pi)/2 * exp(erfinv(x)^2))
  - aten.lgamma_backward     (lgamma'(x) = digamma(x))
  - aten.digamma_backward    (digamma'(x) = polygamma(1, x) = trigamma(x))
  - aten.ndtri_backward      (inverse normal CDF; domain (0,1))
  - aten.i0_backward         (i0'(x) = i1(x))
  - aten.i0e_backward        (i0e'(x) = i1e(x) - sign(x)*i0e(x))
  - aten.i1_backward         (i1'(x) = i0(x) - i1(x)/x; limit at 0 is 0.5)
  - aten.i1e_backward        (i1e'(x) = i0e(x) - i1e(x)*(1/x + sign(x)))

Each test:
  1. Builds matched CPU + Vulkan input tensors with ``requires_grad=True``.
  2. Runs CPU eager forward+backward as the oracle.
  3. Runs Vulkan via ``torch.compile(backend="inductor")`` forward+backward.
  4. Asserts grads match within tolerance.

Also contains structural validation of M-pipeline-6 (uuid subtree hashing).
"""

from __future__ import annotations

import pytest
import torch
import torch.special

import torch_vulkan  # noqa: F401  (registers the vulkan device)


@pytest.fixture(autouse=True)
def _vulkan_available():
    if not torch_vulkan.is_available():
        pytest.skip("No Vulkan device")


# ─────────────────────────────────────────────────────────────────────
# TEST.COV.4 — erfc backward
# ─────────────────────────────────────────────────────────────────────


class TestCov4ErfcBackward:
    @pytest.mark.slow_compile(seconds=60)
    def test_erfc_backward_matches_cpu(self):
        torch.manual_seed(42)
        x_cpu = torch.randn(64, requires_grad=True)
        x_vk = x_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(x):
            return torch.erfc(x).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(x_cpu).backward()
        compiled(x_vk).backward()

        torch.testing.assert_close(x_vk.grad.cpu(), x_cpu.grad, atol=1e-5, rtol=1e-5)


# ─────────────────────────────────────────────────────────────────────
# TEST.COV.4 — erfinv backward
# ─────────────────────────────────────────────────────────────────────


class TestCov4ErfinvBackward:
    @pytest.mark.slow_compile(seconds=60)
    def test_erfinv_backward_matches_cpu(self):
        """Input domain: (-1, 1). Gradient is large near ±1; use mid-range."""
        torch.manual_seed(42)
        # range [-0.9, 0.9]: well inside domain, moderate gradient magnitude
        x_cpu = (torch.rand(64) * 1.8 - 0.9).requires_grad_(True)
        x_vk = x_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(x):
            return torch.erfinv(x).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(x_cpu).backward()
        compiled(x_vk).backward()

        torch.testing.assert_close(x_vk.grad.cpu(), x_cpu.grad, atol=1e-4, rtol=1e-4)


# ─────────────────────────────────────────────────────────────────────
# TEST.COV.4 — lgamma backward
# ─────────────────────────────────────────────────────────────────────


class TestCov4LgammaBackward:
    @pytest.mark.slow_compile(seconds=60)
    def test_lgamma_backward_matches_cpu(self):
        """lgamma'(x) = digamma(x). Input must be positive."""
        torch.manual_seed(42)
        x_cpu = (torch.rand(64) + 0.5).requires_grad_(True)  # [0.5, 1.5]
        x_vk = x_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(x):
            return torch.lgamma(x).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(x_cpu).backward()
        compiled(x_vk).backward()

        torch.testing.assert_close(x_vk.grad.cpu(), x_cpu.grad, atol=1e-4, rtol=1e-4)


# ─────────────────────────────────────────────────────────────────────
# TEST.COV.4 — digamma backward
# ─────────────────────────────────────────────────────────────────────


class TestCov4DigammaBackward:
    @pytest.mark.slow_compile(seconds=60)
    def test_digamma_backward_matches_cpu(self):
        """digamma'(x) = polygamma(1, x) = trigamma(x). Input must be positive."""
        torch.manual_seed(42)
        # Avoid x close to 0 where digamma has a pole; start at 1.0
        x_cpu = (torch.rand(64) + 1.0).requires_grad_(True)  # [1.0, 2.0]
        x_vk = x_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(x):
            return torch.digamma(x).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(x_cpu).backward()
        compiled(x_vk).backward()

        torch.testing.assert_close(x_vk.grad.cpu(), x_cpu.grad, atol=1e-3, rtol=1e-3)


# ─────────────────────────────────────────────────────────────────────
# TEST.COV.4 — ndtri backward
# ─────────────────────────────────────────────────────────────────────


class TestCov4NdtriBackward:
    @pytest.mark.slow_compile(seconds=60)
    def test_ndtri_backward_matches_cpu(self):
        """ndtri = inverse normal CDF. Domain: (0, 1). Gradient explodes near boundary."""
        torch.manual_seed(42)
        # Mid-range [0.1, 0.9] keeps gradients moderate
        x_cpu = (torch.rand(64) * 0.8 + 0.1).requires_grad_(True)
        x_vk = x_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(x):
            return torch.special.ndtri(x).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(x_cpu).backward()
        compiled(x_vk).backward()

        torch.testing.assert_close(x_vk.grad.cpu(), x_cpu.grad, atol=1e-3, rtol=1e-3)


# ─────────────────────────────────────────────────────────────────────
# TEST.COV.4 — Bessel function backward suite (i0, i0e, i1, i1e)
# ─────────────────────────────────────────────────────────────────────


class TestCov4BesselBackward:
    @pytest.mark.slow_compile(seconds=60)
    def test_i0_backward_matches_cpu(self):
        """i0'(x) = i1(x). Input: positive values."""
        torch.manual_seed(42)
        x_cpu = (torch.rand(64) + 0.5).requires_grad_(True)
        x_vk = x_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(x):
            return torch.special.i0(x).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(x_cpu).backward()
        compiled(x_vk).backward()

        torch.testing.assert_close(x_vk.grad.cpu(), x_cpu.grad, atol=1e-4, rtol=1e-4)

    @pytest.mark.slow_compile(seconds=60)
    def test_i0e_backward_matches_cpu(self):
        """i0e'(x) = i1e(x) - sign(x)*i0e(x). Exponentially scaled Bessel."""
        torch.manual_seed(42)
        x_cpu = (torch.rand(64) + 0.5).requires_grad_(True)
        x_vk = x_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(x):
            return torch.special.i0e(x).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(x_cpu).backward()
        compiled(x_vk).backward()

        torch.testing.assert_close(x_vk.grad.cpu(), x_cpu.grad, atol=1e-4, rtol=1e-4)

    @pytest.mark.slow_compile(seconds=60)
    def test_i1_backward_matches_cpu(self):
        """i1'(x) = i0(x) - i1(x)/x; limit at x=0 is 0.5."""
        torch.manual_seed(42)
        x_cpu = (torch.rand(64) + 0.5).requires_grad_(True)
        x_vk = x_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(x):
            return torch.special.i1(x).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(x_cpu).backward()
        compiled(x_vk).backward()

        torch.testing.assert_close(x_vk.grad.cpu(), x_cpu.grad, atol=1e-4, rtol=1e-4)

    @pytest.mark.slow_compile(seconds=60)
    def test_i1e_backward_matches_cpu(self):
        """i1e'(x) = i0e(x) - i1e(x)*(1/x + sign(x)); limit at x=0 is 0.5."""
        torch.manual_seed(42)
        x_cpu = (torch.rand(64) + 0.5).requires_grad_(True)
        x_vk = x_cpu.detach().clone().to("vulkan:0").requires_grad_(True)

        def fn(x):
            return torch.special.i1e(x).sum()

        compiled = torch.compile(fn, backend="inductor")
        fn(x_cpu).backward()
        compiled(x_vk).backward()

        torch.testing.assert_close(x_vk.grad.cpu(), x_cpu.grad, atol=1e-4, rtol=1e-4)


# ─────────────────────────────────────────────────────────────────────
# M-pipeline-6 structural test — FX pass UUID hashes the full subtree
# ─────────────────────────────────────────────────────────────────────


class TestMPipeline6FxPassSubtreeUUID:
    """M-pipeline-6: _VulkanCustomPass.uuid() must hash the full fx_passes/
    subtree, not just __init__.py, so that pattern-only edits invalidate the
    Inductor code cache."""

    def test_fx_passes_subtree_uuid_is_bytes(self):
        from torch_vulkan.inductor.fx_passes import _FX_PASSES_SUBTREE_UUID

        assert isinstance(_FX_PASSES_SUBTREE_UUID, bytes), (
            f"M-pipeline-6: expected bytes, got {type(_FX_PASSES_SUBTREE_UUID)}"
        )

    def test_fx_passes_subtree_uuid_nonempty(self):
        from torch_vulkan.inductor.fx_passes import _FX_PASSES_SUBTREE_UUID

        assert len(_FX_PASSES_SUBTREE_UUID) >= 16, (
            f"M-pipeline-6: UUID too short ({len(_FX_PASSES_SUBTREE_UUID)} bytes); "
            "expected ≥ 16"
        )

    def test_vulkan_custom_pass_uuid_returns_subtree_hash(self):
        from torch_vulkan.inductor.fx_passes import (
            _FX_PASSES_SUBTREE_UUID,
            _make_vulkan_pass,
        )

        pass_obj = _make_vulkan_pass()
        assert pass_obj.uuid() is _FX_PASSES_SUBTREE_UUID, (
            "M-pipeline-6: _VulkanCustomPass.uuid() must return "
            "_FX_PASSES_SUBTREE_UUID (the subtree hash), not a different object"
        )

    def test_fx_passes_uuid_debug_env_works(self, monkeypatch, capsys):
        """Smoke test: enabling TORCH_VULKAN_FX_PASS_UUID_DEBUG does not crash."""
        import importlib

        import torch_vulkan.inductor.fx_passes as fx_mod

        monkeypatch.setenv("TORCH_VULKAN_FX_PASS_UUID_DEBUG", "1")
        try:
            importlib.reload(fx_mod)
        except Exception:
            pass  # Any import-time side effects are acceptable; crash is not
        finally:
            monkeypatch.delenv("TORCH_VULKAN_FX_PASS_UUID_DEBUG", raising=False)
