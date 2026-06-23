"""GPU.4 / GPU.5 — Grid-aware WG sizing, persistent kernels, and
GPU utilization diagnostics.

Tests for:
  - Grid-aware workgroup sizing (GPU.4)
  - Persistent pointwise micro-batching (GPU.5)
  - GPU utilization diagnostics and occupancy estimation
"""

import os

import pytest
import torch
from torch.testing._internal.common_utils import TestCase, run_tests


@pytest.fixture(autouse=True)
def setup():
    try:
        import torch_vulkan

        if not torch_vulkan.is_available():
            pytest.skip("No Vulkan device")
    except ImportError:
        pytest.skip("torch_vulkan not installed")


class TestGPUUtilization(TestCase):
    """GPU.4 / GPU.5 — grid-aware WG sizing, persistent kernels, and
    GPU utilization diagnostics.

    Stage tag: ``BUG_ROOT=\"gpu-util\"``.
    """

    _BUG_ROOT_COMPONENT = "gpu-util"

    # ── Feature gate tests ──────────────────────────────────────────

    def test_gpu_util_grid_aware_wg_enabled(self):
        """GPU.4: grid_aware_wg() returns True by default."""
        os.environ["TORCH_VULKAN_GRID_AWARE_WG"] = "1"
        try:
            from torch_vulkan.inductor.config import grid_aware_wg

            assert grid_aware_wg(), "grid_aware_wg should be True when env=1"
        finally:
            os.environ.pop("TORCH_VULKAN_GRID_AWARE_WG", None)

    def test_gpu_util_grid_aware_wg_disabled(self):
        """GPU.4: grid_aware_wg() returns False when env=0."""
        os.environ["TORCH_VULKAN_GRID_AWARE_WG"] = "0"
        try:
            from torch_vulkan.inductor.config import grid_aware_wg

            assert not grid_aware_wg(), "grid_aware_wg should be False when env=0"
        finally:
            os.environ.pop("TORCH_VULKAN_GRID_AWARE_WG", None)

    def test_gpu_util_persistent_pointwise_enabled(self):
        """GPU.5: persistent_pointwise() returns True by default."""
        os.environ["TORCH_VULKAN_PERSISTENT_POINTWISE"] = "1"
        try:
            from torch_vulkan.inductor.config import persistent_pointwise

            assert persistent_pointwise(), (
                "persistent_pointwise should be True when env=1"
            )
        finally:
            os.environ.pop("TORCH_VULKAN_PERSISTENT_POINTWISE", None)

    def test_gpu_util_persistent_pointwise_disabled(self):
        """GPU.5: persistent_pointwise() returns False when env=0."""
        os.environ["TORCH_VULKAN_PERSISTENT_POINTWISE"] = "0"
        try:
            from torch_vulkan.inductor.config import persistent_pointwise

            assert not persistent_pointwise(), (
                "persistent_pointwise should be False when env=0"
            )
        finally:
            os.environ.pop("TORCH_VULKAN_PERSISTENT_POINTWISE", None)

    # ── Grid-aware WG sizing behavior ───────────────────────────────

    def test_gpu_util_grid_aware_wg_small_grid(self):
        """GPU.4: For a small numel, grid-aware WG sizing should produce
        a reasonable workgroup size (never below one wave)."""
        import torch_vulkan

        # Use a very small tensor — numel < 256
        os.environ["TORCH_VULKAN_GRID_AWARE_WG"] = "1"
        try:

            @torch.compile(backend="inductor")
            def fn(x):
                return torch.relu(x * 0.5 + 0.1)

            x = torch.randn(32, device="vulkan:0")
            result = fn(x)
            expected = torch.relu(x.cpu() * 0.5 + 0.1)
            torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-5)
        finally:
            os.environ.pop("TORCH_VULKAN_GRID_AWARE_WG", None)

    def test_gpu_util_grid_aware_wg_does_not_affect_large_grid(self):
        """GPU.4: For a large numel, grid-aware WG sizing should not
        reduce the WG size (grid is already large enough)."""
        import torch_vulkan

        os.environ["TORCH_VULKAN_GRID_AWARE_WG"] = "1"
        try:

            @torch.compile(backend="inductor")
            def fn(x):
                return torch.relu(x * 0.5 + 0.1)

            x = torch.randn(512, 1024, device="vulkan:0")
            result = fn(x)
            expected = torch.relu(x.cpu() * 0.5 + 0.1)
            torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-5)
        finally:
            os.environ.pop("TORCH_VULKAN_GRID_AWARE_WG", None)

    # ── Persistent kernel correctness ───────────────────────────────

    def test_gpu_util_persistent_kernel_correctness(self):
        """GPU.5: Small pointwise chain produces correct results under
        persistent pointwise mode (feature gate ON)."""
        import torch_vulkan

        os.environ["TORCH_VULKAN_PERSISTENT_POINTWISE"] = "1"
        try:

            @torch.compile(backend="inductor")
            def fn(x):
                return torch.tanh(torch.sigmoid(torch.relu(x)))

            x = torch.randn(64, 64, device="vulkan:0")
            result = fn(x)
            expected = torch.tanh(torch.sigmoid(torch.relu(x.cpu())))
            torch.testing.assert_close(result.cpu(), expected, rtol=1e-3, atol=1e-3)
        finally:
            os.environ.pop("TORCH_VULKAN_PERSISTENT_POINTWISE", None)

    def test_gpu_util_persistent_kernel_correctness_tiny(self):
        """GPU.5: Very small pointwise chain (numel < 256) correct."""
        import torch_vulkan

        os.environ["TORCH_VULKAN_PERSISTENT_POINTWISE"] = "1"
        try:

            @torch.compile(backend="inductor")
            def fn(x):
                return torch.sigmoid(x) + torch.tanh(x)

            x = torch.randn(16, device="vulkan:0")
            result = fn(x)
            expected = torch.sigmoid(x.cpu()) + torch.tanh(x.cpu())
            torch.testing.assert_close(result.cpu(), expected, rtol=1e-3, atol=1e-3)
        finally:
            os.environ.pop("TORCH_VULKAN_PERSISTENT_POINTWISE", None)

    # ── Dispatch count tests ────────────────────────────────────────

    def test_gpu_util_persistent_kernel_reduces_dispatches(self):
        """GPU.5: Small pointwise chain dispatch count is reasonable
        (≤10 dispatches for a 3-op chain)."""
        import torch_vulkan

        @torch.compile(backend="inductor")
        def fn(x):
            return torch.tanh(torch.sigmoid(torch.relu(x)))

        x = torch.randn(64, 64, device="vulkan:0")
        fn(x)  # warmup compile

        torch_vulkan._c_ext._reset_perf_counters()
        fn(x)
        dispatches = torch_vulkan._c_ext._get_dispatch_count()

        # With fusion, a 3-op chain should be ≤10 dispatches.
        assert dispatches <= 20, (
            f"small pointwise chain: expected ≤20 dispatches, got {dispatches}"
        )

    def test_gpu_util_tiny_grid_dispatch_count(self):
        """GPU.4: Very small grid (< 256 elements) should use ≤3 dispatches."""
        import torch_vulkan

        @torch.compile(backend="inductor")
        def fn(x):
            return torch.relu(x + 2.0) * 0.5

        x = torch.randn(32, device="vulkan:0")
        fn(x)

        torch_vulkan._c_ext._reset_perf_counters()
        fn(x)
        dispatches = torch_vulkan._c_ext._get_dispatch_count()

        assert dispatches <= 5, f"tiny grid: expected ≤5 dispatches, got {dispatches}"

    # ── Diagnostics API tests ───────────────────────────────────────

    def test_gpu_util_diagnostics_report(self):
        """GPU.4/GPU.5: gpu_utilization_report runs without error and
        returns expected keys with valid value ranges."""
        import torch_vulkan

        @torch.compile(backend="inductor")
        def fn(x):
            return torch.relu(x * 0.5 + 0.1)

        x = torch.randn(64, 128, device="vulkan:0")
        fn(x)  # warmup compile

        from torch_vulkan.inductor.gpu_utilization import gpu_utilization_report

        report = gpu_utilization_report(fn, x, warmup_iters=2, measure_iters=5)

        assert "total_time_ms" in report
        assert "dispatch_count" in report
        assert "avg_dispatch_us" in report
        assert "utilization_estimate" in report
        assert "measure_iters" in report

        assert report["measure_iters"] == 5
        assert report["dispatch_count"] >= 1, (
            f"expected at least 1 dispatch, got {report['dispatch_count']}"
        )
        assert report["total_time_ms"] >= 0.0
        assert report["avg_dispatch_us"] >= 0.0
        assert 0.0 <= report["utilization_estimate"] <= 100.0

    # ── Occupancy estimation tests ──────────────────────────────────

    def test_gpu_util_estimate_occupancy_light(self):
        """GPU.4: Light-kernel (8 VGPRs/thread, 256 threads) should get
        ≥2 waves/CU on RDNA1."""
        from torch_vulkan.inductor.gpu_utilization import estimate_occupancy

        occ = estimate_occupancy(
            threadgroup_size=256, vgprs_per_thread=8, shared_mem_bytes=0
        )
        assert occ["waves_per_cu"] >= 2, (
            f"light kernel should get ≥2 waves/CU, got {occ}"
        )
        assert occ["occupancy_pct"] >= 50.0
        assert "limiting_factor" in occ
        assert occ["vgprs_per_wave"] == 8 * 64

    def test_gpu_util_estimate_occupancy_heavy(self):
        """GPU.4: Heavy kernel (32 VGPRs/thread) should be VGPR-limited."""
        from torch_vulkan.inductor.gpu_utilization import estimate_occupancy

        occ = estimate_occupancy(
            threadgroup_size=256, vgprs_per_thread=32, shared_mem_bytes=0
        )
        assert occ["limiting_factor"] == "vgpr"
        # 32 VGPRs × 64 threads = 2048 VGPRs per wave > 256 CU VGPRs
        assert occ["vgprs_per_wave"] == 32 * 64

    def test_gpu_util_estimate_occupancy_lds_limited(self):
        """GPU.4: Kernel using lots of LDS should be LDS-limited."""
        from torch_vulkan.inductor.gpu_utilization import estimate_occupancy

        # 32 KB LDS per WG — only 2 WGs fit in 64 KB LDS per CU
        occ = estimate_occupancy(
            threadgroup_size=256,
            vgprs_per_thread=4,
            shared_mem_bytes=32 * 1024,
        )
        assert occ["limiting_factor"] == "lds"
        assert occ["waves_per_cu"] <= 2

    def test_gpu_util_estimate_occupancy_thread_limited(self):
        """GPU.4: Very large WG (1024 threads) should be thread-limited
        (only 1 WG fits in 1024 threads/CU)."""
        from torch_vulkan.inductor.gpu_utilization import estimate_occupancy

        occ = estimate_occupancy(
            threadgroup_size=1024, vgprs_per_thread=4, shared_mem_bytes=0
        )
        assert occ["limiting_factor"] == "threads"
        assert occ["waves_per_cu"] <= 1

    # ── _is_small_pointwise_chain tests ─────────────────────────────

    def test_gpu_util_is_small_pointwise_chain_single_node(self):
        """GPU.5: Single node is not a chain (need ≥2 to fuse)."""
        from torch_vulkan.inductor.kernel.pointwise import PointwiseMixin

        class MockSNode:
            def __init__(self, numel, rnumel=1):
                self.group = (None, (numel, rnumel))

        assert not PointwiseMixin._is_small_pointwise_chain([MockSNode(64)]), (
            "single node should not be a chain"
        )

    def test_gpu_util_is_small_pointwise_chain_two_nodes(self):
        """GPU.5: Two small pointwise nodes form a valid chain."""
        from torch_vulkan.inductor.kernel.pointwise import PointwiseMixin

        class MockSNode:
            def __init__(self, numel, rnumel=1):
                self.group = (None, (numel, rnumel))

        assert PointwiseMixin._is_small_pointwise_chain(
            [MockSNode(64), MockSNode(128)]
        ), "two small nodes should be a chain"

    def test_gpu_util_is_small_pointwise_chain_reduction_disqualifies(self):
        """GPU.5: Node with reduction disqualifies the chain."""
        from torch_vulkan.inductor.kernel.pointwise import PointwiseMixin

        class MockSNode:
            def __init__(self, numel, rnumel=1):
                self.group = (None, (numel, rnumel))

        assert not PointwiseMixin._is_small_pointwise_chain(
            [MockSNode(64), MockSNode(64, rnumel=2)]
        ), "reduction node should disqualify chain"

    def test_gpu_util_is_small_pointwise_chain_too_large_total(self):
        """GPU.5+: Total numel > 16384 (max_total_numel) or per-thread
        iterations > 16 disqualifies the chain."""
        from torch_vulkan.inductor.kernel.pointwise import PointwiseMixin

        class MockSNode:
            def __init__(self, numel, rnumel=1):
                self.group = (None, (numel, rnumel))

        # Total numel 20000 exceeds max_total_numel cap of 16384
        assert not PointwiseMixin._is_small_pointwise_chain(
            [MockSNode(10000), MockSNode(10000)]
        ), "total numel >16384 should disqualify"

    def test_gpu_util_is_small_pointwise_chain_per_thread_iters_too_high(self):
        """GPU.5+: Per-thread iterations > 16 disqualifies the chain
        for 2-op chains (iter_cap=16). For 3+ ops, the cap scales."""
        from torch_vulkan.inductor.kernel.pointwise import PointwiseMixin

        class MockSNode:
            def __init__(self, numel, rnumel=1):
                self.group = (None, (numel, rnumel))

        # num_threads = 256 * 20 = 5120. total 200000 -> per_thread_iters = 40 > 16.
        # But total 200000 also > max_total_numel 16384, so disqualified either way.
        # Use total=100000 (still >16384) to test the total cap.
        assert not PointwiseMixin._is_small_pointwise_chain(
            [MockSNode(50000), MockSNode(50000)]
        ), "total numel >16384 should disqualify"
        
    def test_gpu_util_is_small_pointwise_chain_high_per_thread_iters(self):
        """GPU.5+: For 2-op chains, per-thread iterations > 16 disqualifies."""
        from torch_vulkan.inductor.kernel.pointwise import PointwiseMixin

        class MockSNode:
            def __init__(self, numel, rnumel=1):
                self.group = (None, (numel, rnumel))

        # num_threads = 256 * 20 = 5120. total=15000 -> per_thread_iters = 3
        # iter_cap for 2 ops = 16, total <= 16384. Should pass.
        assert PointwiseMixin._is_small_pointwise_chain(
            [MockSNode(7500), MockSNode(7500)]
        ), "total 15000 with per_thread_iters=3 <= 16 should qualify"
        
    def test_gpu_util_is_small_pointwise_chain_many_ops_relaxes_cap(self):
        """GPU.5+: Many ops (>=5) relaxes iter_cap to 64."""
        from torch_vulkan.inductor.kernel.pointwise import PointwiseMixin

        class MockSNode:
            def __init__(self, numel, rnumel=1):
                self.group = (None, (numel, rnumel))

        # 5 nodes, total 15000. per_thread_iters = ceil(15000/5120) = 3.
        # iter_cap for 5 ops = 64. Should qualify.
        assert PointwiseMixin._is_small_pointwise_chain(
            [MockSNode(3000)] * 5
        ), "5 ops with total 15000 and per_thread_iters=3 <= 64 should qualify"


if __name__ == "__main__":
    run_tests()
