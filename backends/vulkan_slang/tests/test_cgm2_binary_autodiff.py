"""CG.M2 — Pointwise binary [Differentiable] coverage tests.

Every binary aten op that needs explicit backward (pow, atan2, hypot,
copysign, max, min) gets ``[Differentiable]`` in ``lib/pointwise.slang``
with optional ``no_diff`` qualifier on non-differentiated args.

These tests verify that dispatch_binary_bwd produces correct gradients
matching CPU autograd for each binary op.

Stage tag: ``BUG_ROOT="autodiff"``.
"""

from __future__ import annotations

import torch
from torch.testing._internal.common_utils import TestCase, run_tests


class TestCGM2BinaryAutodiffCoverage(TestCase):
    """CG.M2 — Pointwise binary [Differentiable] coverage."""

    _BUG_ROOT_COMPONENT = "autodiff"

    def test_cgm2_bwd_diff_table_has_6_binary_entries(self):
        """CG.M2 gate: BWD_DIFF_TABLE should have 6 binary entries."""
        from torch_vulkan.inductor.bwd_diff_table import BWD_DIFF_TABLE

        expected = {
            "aten.pow.Tensor_Tensor_backward",
            "aten.atan2_backward",
            "aten.hypot_backward",
            "aten.copysign_tensor_backward",
            "aten.maximum_backward",
            "aten.minimum_backward",
        }
        missing = expected - set(BWD_DIFF_TABLE)
        self.assertEqual(
            len(missing), 0, f"CG.M2: missing BWD_DIFF_TABLE entries: {sorted(missing)}"
        )

    def _binary_bwd_test(
        self,
        aten_op,
        cpu_fn,
        *,
        a_lo=-3.0,
        a_hi=3.0,
        b_lo=-3.0,
        b_hi=3.0,
        no_diff_kwargs=None,
    ):
        """Shared helper: dispatch binary bwd and compare with CPU autograd."""
        from torch_vulkan.inductor.bwd_diff_dispatch import dispatch_binary_bwd

        torch.manual_seed(42)
        shape = (16, 64)

        a_cpu = torch.rand(shape) * (a_hi - a_lo) + a_lo
        b_cpu = torch.rand(shape) * (b_hi - b_lo) + b_lo
        grad_out_cpu = torch.randn(shape) * 0.5

        a_ref = a_cpu.detach().clone().requires_grad_(True)
        b_ref = b_cpu.detach().clone().requires_grad_(True)
        out = cpu_fn(a_ref, b_ref)
        out.backward(grad_out_cpu)

        a_v = a_cpu.to("vulkan:0")
        b_v = b_cpu.to("vulkan:0")
        gout_v = grad_out_cpu.to("vulkan:0")

        grad_a, grad_b = dispatch_binary_bwd(
            aten_op,
            a_v,
            b_v,
            gout_v,
            no_diff_kwargs=no_diff_kwargs or None,
        )
        torch.testing.assert_close(grad_a.cpu(), a_ref.grad, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(grad_b.cpu(), b_ref.grad, rtol=1e-3, atol=1e-3)

    def test_cgm2_pow_backward_autodiff(self):
        """CG.M2: pow backward via bwd_diff matches CPU (both args diff)."""
        self._binary_bwd_test(
            "aten.pow.Tensor_Tensor_backward",
            lambda a, b: torch.pow(a, b),
            a_lo=0.1,
            a_hi=3.0,
            b_lo=0.5,
            b_hi=2.5,
        )

    def test_cgm2_atan2_backward_autodiff(self):
        """CG.M2: atan2 backward via bwd_diff matches CPU."""
        self._binary_bwd_test(
            "aten.atan2_backward",
            lambda a, b: torch.atan2(a, b),
        )

    def test_cgm2_hypot_backward_autodiff(self):
        """CG.M2: hypot backward via bwd_diff matches CPU."""
        self._binary_bwd_test(
            "aten.hypot_backward",
            lambda a, b: torch.hypot(a, b),
        )

    def test_cgm2_max_backward_autodiff(self):
        """CG.M2: max backward routes gradients to the max element."""
        self._binary_bwd_test(
            "aten.maximum_backward",
            lambda a, b: torch.maximum(a, b),
        )

    def test_cgm2_min_backward_autodiff(self):
        """CG.M2: min backward routes gradients to the min element."""
        self._binary_bwd_test(
            "aten.minimum_backward",
            lambda a, b: torch.minimum(a, b),
        )

    def test_cgm2_copysign_backward_autodiff(self):
        """CG.M2: copysign backward — zero grad for sign arg (no_diff)."""
        from torch_vulkan.inductor.bwd_diff_dispatch import dispatch_binary_bwd

        torch.manual_seed(42)
        shape = (16, 64)

        mag_cpu = torch.randn(shape)
        sign_cpu = torch.randn(shape)
        grad_out_cpu = torch.randn(shape) * 0.5

        mag_ref = mag_cpu.detach().clone().requires_grad_(True)
        sign_ref = sign_cpu.detach().clone().requires_grad_(True)
        out = torch.copysign(mag_ref, sign_ref)
        out.backward(grad_out_cpu)

        mag_v = mag_cpu.to("vulkan:0")
        sign_v = sign_cpu.to("vulkan:0")
        gout_v = grad_out_cpu.to("vulkan:0")

        grad_mag, grad_sign = dispatch_binary_bwd(
            "aten.copysign_tensor_backward",
            mag_v,
            sign_v,
            gout_v,
        )
        # mag grad should match CPU autograd
        torch.testing.assert_close(grad_mag.cpu(), mag_ref.grad, rtol=1e-3, atol=1e-3)
        # sign arg is no_diff — Slang produces zero gradient
        torch.testing.assert_close(
            grad_sign.cpu(),
            torch.zeros_like(sign_cpu),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    run_tests()
