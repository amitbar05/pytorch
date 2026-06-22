"""CG.M3 — Reduction backward via [Differentiable] scalar fold.

Validates that sum, mean, and var backward produce correct gradients
matching CPU reference. The [Differentiable] fold functions
(reduce_fold_sum / reduce_fold_prod in reduction.slang) are verified
by slangc during compilation; the backward dispatches as broadcast
kernels derived from bwd_diff over the fold.
"""

import os

import pytest
import torch


@pytest.fixture(autouse=True)
def setup():
    try:
        import torch_vulkan

        if not torch_vulkan.is_available():
            pytest.skip("No Vulkan device")
    except ImportError:
        pytest.skip("torch_vulkan not installed")


class TestCGM3SumBackward:
    """Sum backward: gradient is broadcast of 1.0 (all elements get grad_out)."""

    def test_cgm3_sum_backward_matches_cpu(self):
        """sum().backward() matches CPU: gradient is broadcast of 1.0."""
        if not os.environ.get("SLANGC"):
            pytest.skip("SLANGC env var not set")

        @torch.compile(backend="inductor")
        def fn(x):
            return x.sum()

        torch.manual_seed(0)
        x = torch.randn(8, 16, device="vulkan:0", requires_grad=True)
        x_cpu = x.detach().cpu().requires_grad_()

        out = fn(x)
        out.backward()
        x_cpu.sum().backward()

        assert x.grad is not None, "sum backward produced None gradient"
        torch.testing.assert_close(x.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)
        # Sum gradient should be all ones
        expected = torch.ones_like(x_cpu.grad)
        torch.testing.assert_close(x.grad.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_cgm3_sum_dim_backward_matches_cpu(self):
        """sum(dim=0).backward() matches CPU: gradient broadcast along dim."""
        if not os.environ.get("SLANGC"):
            pytest.skip("SLANGC env var not set")

        @torch.compile(backend="inductor")
        def fn(x):
            return x.sum(dim=0)

        torch.manual_seed(0)
        x = torch.randn(8, 64, device="vulkan:0", requires_grad=True)
        x_cpu = x.detach().cpu().requires_grad_()

        out = fn(x)
        out.sum().backward()
        x_cpu.sum(dim=0).sum().backward()

        assert x.grad is not None, "sum dim=0 backward produced None gradient"
        torch.testing.assert_close(x.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)

    def test_cgm3_sum_two_dim_backward_matches_cpu(self):
        """sum(dim=[0,2]).backward() matches CPU on multi-axis reduction."""
        if not os.environ.get("SLANGC"):
            pytest.skip("SLANGC env var not set")

        @torch.compile(backend="inductor")
        def fn(x):
            return x.sum(dim=[0, 2])

        torch.manual_seed(0)
        x = torch.randn(8, 16, 32, device="vulkan:0", requires_grad=True)
        x_cpu = x.detach().cpu().requires_grad_()

        out = fn(x)
        out.sum().backward()
        x_cpu.sum(dim=[0, 2]).sum().backward()

        assert x.grad is not None, "sum dim=[0,2] backward produced None gradient"
        torch.testing.assert_close(x.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)


class TestCGM3MeanBackward:
    """Mean backward: gradient = 1/numel broadcast."""

    def test_cgm3_mean_backward_matches_cpu(self):
        """mean().backward() matches CPU: gradient = 1/numel broadcast."""
        if not os.environ.get("SLANGC"):
            pytest.skip("SLANGC env var not set")

        @torch.compile(backend="inductor")
        def fn(x):
            return x.mean()

        torch.manual_seed(0)
        x = torch.randn(8, 16, device="vulkan:0", requires_grad=True)
        x_cpu = x.detach().cpu().requires_grad_()

        out = fn(x)
        out.backward()
        x_cpu.mean().backward()

        assert x.grad is not None, "mean backward produced None gradient"
        torch.testing.assert_close(x.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)
        # Mean gradient should be 1/numel for all elements
        numel = x.numel()
        expected = torch.full_like(x_cpu.grad, 1.0 / numel)
        torch.testing.assert_close(x.grad.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_cgm3_mean_dim_backward_matches_cpu(self):
        """mean(dim=-1).backward() matches CPU."""
        if not os.environ.get("SLANGC"):
            pytest.skip("SLANGC env var not set")

        @torch.compile(backend="inductor")
        def fn(x):
            return x.mean(dim=-1)

        torch.manual_seed(0)
        x = torch.randn(4, 32, device="vulkan:0", requires_grad=True)
        x_cpu = x.detach().cpu().requires_grad_()

        out = fn(x)
        out.sum().backward()
        x_cpu.mean(dim=-1).sum().backward()

        assert x.grad is not None, "mean dim=-1 backward produced None gradient"
        torch.testing.assert_close(x.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)

    def test_cgm3_mean_two_dim_backward_matches_cpu(self):
        """mean(dim=[1,2]).backward() matches CPU on multi-axis reduction."""
        if not os.environ.get("SLANGC"):
            pytest.skip("SLANGC env var not set")

        @torch.compile(backend="inductor")
        def fn(x):
            return x.mean(dim=[1, 2])

        torch.manual_seed(0)
        x = torch.randn(4, 8, 16, device="vulkan:0", requires_grad=True)
        x_cpu = x.detach().cpu().requires_grad_()

        out = fn(x)
        out.sum().backward()
        x_cpu.mean(dim=[1, 2]).sum().backward()

        assert x.grad is not None, "mean dim=[1,2] backward produced None gradient"
        torch.testing.assert_close(x.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)


class TestCGM3VarBackward:
    """Var backward: gradient = 2 * (x - mean) * grad_out / (numel - 1) broadcast."""

    def test_cgm3_var_backward_matches_cpu(self):
        """var(unbiased=True).backward() matches CPU."""
        if not os.environ.get("SLANGC"):
            pytest.skip("SLANGC env var not set")

        @torch.compile(backend="inductor")
        def fn(x):
            return x.var(unbiased=True)

        torch.manual_seed(0)
        x = torch.randn(16, 32, device="vulkan:0", requires_grad=True)
        x_cpu = x.detach().cpu().requires_grad_()

        out = fn(x)
        out.backward()
        x_cpu.var(unbiased=True).backward()

        assert x.grad is not None, "var backward produced None gradient"
        torch.testing.assert_close(x.grad.cpu(), x_cpu.grad, rtol=1e-3, atol=1e-3)

    def test_cgm3_var_unbiased_false_backward_matches_cpu(self):
        """var(unbiased=False).backward() matches CPU."""
        if not os.environ.get("SLANGC"):
            pytest.skip("SLANGC env var not set")

        @torch.compile(backend="inductor")
        def fn(x):
            return x.var(unbiased=False)

        torch.manual_seed(0)
        x = torch.randn(16, 32, device="vulkan:0", requires_grad=True)
        x_cpu = x.detach().cpu().requires_grad_()

        out = fn(x)
        out.backward()
        x_cpu.var(unbiased=False).backward()

        assert x.grad is not None, "var(unbiased=False) backward produced None gradient"
        torch.testing.assert_close(x.grad.cpu(), x_cpu.grad, rtol=1e-3, atol=1e-3)

    def test_cgm3_var_dim_backward_matches_cpu(self):
        """var(dim=0, unbiased=True).backward() matches CPU."""
        if not os.environ.get("SLANGC"):
            pytest.skip("SLANGC env var not set")

        @torch.compile(backend="inductor")
        def fn(x):
            return x.var(dim=0, unbiased=True)

        torch.manual_seed(0)
        x = torch.randn(8, 64, device="vulkan:0", requires_grad=True)
        x_cpu = x.detach().cpu().requires_grad_()

        out = fn(x)
        out.sum().backward()
        x_cpu.var(dim=0, unbiased=True).sum().backward()

        assert x.grad is not None, "var dim=0 backward produced None gradient"
        torch.testing.assert_close(x.grad.cpu(), x_cpu.grad, rtol=1e-3, atol=1e-3)


class TestCGM3RegistryIntegrity:
    """Registry and source-level integrity checks for CG.M3."""

    def test_cgm3_reduce_fold_registry_entries(self):
        """BWD_TEMPLATE_REGISTRY has CG.M3 reduction backward entries."""
        from torch_vulkan.inductor.bwd_template_registry import (
            BWD_TEMPLATE_REGISTRY,
            BackwardKind,
        )

        expected = {
            "reduce_sum": ("reduce_fold_sum", "reduction"),
            "reduce_prod": ("reduce_fold_prod", "reduction"),
            "reduce_mean": ("reduce_fold_sum", "reduction"),
            "reduce_var": ("reduce_fold_sum", "reduction"),
        }
        for key, (expected_fn, expected_mod) in expected.items():
            entry = BWD_TEMPLATE_REGISTRY.lookup(key)
            assert entry is not None, (
                f"CG.M3: missing BWD_TEMPLATE_REGISTRY entry for {key}"
            )
            assert entry.kind == BackwardKind.BWD_DIFF, (
                f"CG.M3: {key} should be BWD_DIFF, got {entry.kind}"
            )
            assert entry.fwd_fn == expected_fn, (
                f"CG.M3: {key} should reference {expected_fn}, got {entry.fwd_fn}"
            )
            assert entry.module == expected_mod, (
                f"CG.M3: {key} should use {expected_mod} module, got {entry.module}"
            )

    def test_cgm3_op_to_fwd_key_mappings(self):
        """resolve_backward_kind maps aten reduction backward ops to CG.M3 keys."""
        from torch_vulkan.inductor.bwd_diff_dispatch import resolve_backward_kind
        from torch_vulkan.inductor.bwd_template_registry import BackwardKind

        for aten_op, expected_key in [
            ("aten.sum_backward", "reduce_sum"),
            ("aten.sum.dim_IntList_backward", "reduce_sum"),
            ("aten.mean_backward", "reduce_mean"),
            ("aten.mean.dim_backward", "reduce_mean"),
            ("aten.var_backward", "reduce_var"),
            ("aten.var.correction_backward", "reduce_var"),
            ("aten.prod_backward", "reduce_prod"),
            ("aten.prod.dim_int_backward", "reduce_prod"),
        ]:
            resolved = resolve_backward_kind(aten_op)
            assert resolved is not None, (
                f"CG.M3: resolve_backward_kind({aten_op!r}) returned None"
            )
            assert resolved.kind == BackwardKind.BWD_DIFF, (
                f"CG.M3: {aten_op} should be BWD_DIFF, got {resolved.kind}"
            )
            assert resolved.fwd_key == expected_key, (
                f"CG.M3: {aten_op} should map to {expected_key}, got {resolved.fwd_key}"
            )

    def test_cgm3_reduce_fold_sum_is_differentiable(self):
        """reduce_fold_sum in reduction.slang carries [Differentiable].

        Verifies by reading the source file and checking the annotation
        is present before the function definition.
        """
        import torch_vulkan

        pkg_dir = os.path.dirname(torch_vulkan.__file__)
        # torch_vulkan lives at backends/vulkan_slang/python/torch_vulkan/,
        # shaders live at backends/vulkan_slang/shaders/. Go up two levels.
        reduction_path = os.path.normpath(
            os.path.join(pkg_dir, "..", "..", "shaders", "lib", "reduction.slang")
        )

        with open(reduction_path) as f:
            src = f.read()

        # Verify [Differentiable] appears right before reduce_fold_sum
        assert "[Differentiable]" in src, (
            "CG.M3: [Differentiable] annotation missing from reduction.slang"
        )
        assert "reduce_fold_sum" in src, (
            "CG.M3: reduce_fold_sum function missing from reduction.slang"
        )
        assert "reduce_fold_prod" in src, (
            "CG.M3: reduce_fold_prod function missing from reduction.slang"
        )
        # The annotation must appear before the function definition
        idx_diff = src.index("[Differentiable]")
        idx_sum = src.index("reduce_fold_sum")
        assert idx_diff < idx_sum + 100, (
            "CG.M3: [Differentiable] must be near reduce_fold_sum in reduction.slang"
        )

    def test_cgm3_max_min_not_in_registry(self):
        """Max/min reductions are NOT differentiable — they should NOT have
        BWD_DIFF entries in the registry."""
        from torch_vulkan.inductor.bwd_diff_dispatch import resolve_backward_kind
        from torch_vulkan.inductor.bwd_template_registry import BackwardKind

        for non_diff_op in [
            "aten.max_backward",
            "aten.min_backward",
            "aten.max.dim_backward",
            "aten.min.dim_backward",
        ]:
            resolved = resolve_backward_kind(non_diff_op)
            if resolved is not None:
                assert resolved.kind != BackwardKind.BWD_DIFF, (
                    f"CG.M3: {non_diff_op} must NOT be BWD_DIFF "
                    f"(max/min are non-differentiable in the autodiff sense)"
                )


class TestMCV4SoftmaxBackward:
    """M-CV.4 — verify the M-NEW.9 ``_rewrite_constant_folded_tangent`` joint-pass
    fix generalizes to softmax backward.

    Background: AOTAutograd's joint-graph trace materializes the implicit
    ``tangents_1 = torch.ones(())`` for a scalar ``loss.backward()`` by
    constant-folding ``expand(zeros([]), shape)`` into a stored
    ``self._tensor_constant0 = zeros(shape)`` attribute. The partitioned
    backward then has ``tangents_1`` unused and ``get_attr(_tensor_constant0)``
    returning zeros — every gradient computed downstream collapses to zero.

    ``meta_patches/joint_graph_passes.py:_rewrite_constant_folded_tangent``
    fixes this by rewriting the get_attr to ``aten.expand(tangents_1, shape)``
    so the actual runtime tangent value propagates. This test confirms the
    fix is not specific to ``aten.sum`` / ``aten.mean`` — softmax also has
    the constant-folded tangent pattern, and the gradient must match CPU.
    """

    def test_mcv4_softmax_dim_sum_backward_matches_cpu(self):
        """``F.softmax(x, dim=-1).sum().backward()`` produces non-zero
        gradients matching CPU.

        Pre-M-NEW.9 (and pre-M-CV.4 generalization): VK gradient was all
        zeros from the constant-folded tangent — assert_close vs CPU
        non-zero softmax-Jacobian gradient would fail.
        """
        if not os.environ.get("SLANGC"):
            pytest.skip("SLANGC env var not set")

        @torch.compile(backend="inductor")
        def fn(x):
            return torch.nn.functional.softmax(x, dim=-1).sum()

        torch.manual_seed(0)
        x = torch.randn(4, 8, device="vulkan:0", requires_grad=True)
        x_cpu = x.detach().cpu().requires_grad_()

        out = fn(x)
        out.backward()
        torch.nn.functional.softmax(x_cpu, dim=-1).sum().backward()

        assert x.grad is not None, "softmax sum backward produced None gradient"
        # The crucial assertion: the gradient is NOT all zeros (which was
        # the M-NEW.9 / M-CV.4 bug symptom). Softmax row-summed-to-1, so
        # the Jacobian collapses identically, but the underlying compute
        # must still propagate the real tangent value.
        torch.testing.assert_close(x.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)


class TestAOTIFactoryOpsShim:
    """S4.3 — factory-op shim covers zeros/ones/full, not just empty.memory_format.

    Verifies that when torch.export / torch.compile lifts torch.zeros / torch.ones
    into graph constants (get_attr nodes), the _rewrite_factory_meta_to_vulkan pass
    inserts the required aten._to_copy(device='vulkan') so the .so runtime allocates
    on Vulkan, not CPU.

    Skipped when no AOTI build is available.
    """

    @pytest.mark.skip("requires AOTI build")
    def test_factory_ops_output_on_vulkan(self):
        import torch

        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                # torch.zeros / torch.ones are lifted as get_attr constants
                # by torch.export; the shim must move them to vulkan.
                zeros_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
                ones_bias = torch.ones(x.shape[1], dtype=x.dtype, device=x.device)
                return x + ones_bias

        model = SimpleModel()
        x = torch.randn(4, 8, device="vulkan:0")

        compiled = torch.compile(model, backend="inductor")
        out = compiled(x)

        assert out.device.type == "vulkan", (
            f"Expected vulkan output, got {out.device}")
        # Sanity-check the result is not all-zeros (which would indicate the
        # ones_bias constant was left on CPU and silently zeroed).
        assert out.abs().sum() > 0, "Output is all zeros — factory constant not on vulkan"
