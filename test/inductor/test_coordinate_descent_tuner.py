# Owner(s): ["module: inductor"]

import sys
import unittest
from unittest import mock

import torch
from torch._inductor.runtime.hints import TRITON_MAX_BLOCK
from torch._inductor.test_case import TestCase, run_tests
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU

try:
    import triton  # @manual
except ImportError:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton")  # noqa: B904

from torch._inductor import config
from torch._inductor.runtime.coordinate_descent_tuner import (
    CoordescTuner,
    CoordescTunerConfig,
)

config.benchmark_kernel = True
config.coordinate_descent_tuning = True

orig_compare_config = CoordescTuner.compare_config


def mock_compare_config_prefer_larger_XBLOCK(
    self, func, candidate_config, best_config, best_timing
):
    """
    self is the CoordescTuner object
    """
    if "XBLOCK" in candidate_config.kwargs:
        if "XBLOCK" not in best_config.kwargs:
            raise AssertionError
        if candidate_config.kwargs["XBLOCK"] < best_config.kwargs["XBLOCK"]:
            func(candidate_config)  # run func so the launcher will be created
            return False, best_timing * 1.1
        elif candidate_config.kwargs["XBLOCK"] > best_config.kwargs["XBLOCK"]:
            func(candidate_config)
            return True, best_timing * 0.9

    return orig_compare_config(self, func, candidate_config, best_config, best_timing)


class TestCoordinateDescentTuner(TestCase):
    def test_abs_function(self):
        """
        The benchmark result is simply abs(XBLOCK - 15).
        The tuner should find XBLOCK=16.
        """
        tuner = CoordescTuner()
        baseline_config = triton.Config({"XBLOCK": 1}, num_warps=8, num_stages=1)

        def func(config):
            return abs(config.kwargs["XBLOCK"] - 15)

        best_config = tuner.autotune(func, baseline_config)
        self.assertTrue(best_config.kwargs.get("XBLOCK") == 16, str(best_config))

    def test_config_defaults(self):
        """Test that CoordescTunerConfig has sensible defaults."""
        cfg = CoordescTunerConfig()
        self.assertEqual(cfg.max_iterations, 100)
        self.assertEqual(cfg.early_stop_threshold, 0.001)
        self.assertEqual(cfg.early_stop_patience, 3)
        self.assertTrue(cfg.adaptive_step)
        self.assertFalse(cfg.multi_field_tuning)
        self.assertTrue(cfg.multi_field_tuning_for_mm)
        self.assertEqual(cfg.warmup_samples, 4)
        self.assertFalse(cfg.check_all_directions)
        self.assertEqual(cfg.search_radius, 1)
        self.assertEqual(cfg.correlated_fields, [])

    def test_adaptive_step_coarse_to_fine(self):
        """Test adaptive stepping converges to optimum reachable from start.

        Starts at XBLOCK=1, optimum at 128.
        With adaptive (factor 4): 1->4->16->64, then overshoot, shrink to factor 2: ->128
        """
        cfg = CoordescTunerConfig(adaptive_step=True, warmup_samples=0)
        tuner = CoordescTuner(tuner_config=cfg)
        baseline_config = triton.Config({"XBLOCK": 1}, num_warps=8, num_stages=1)

        def func(config):
            x = config.kwargs["XBLOCK"]
            return abs(x - 128) + 1.0

        best_config = tuner.autotune(func, baseline_config)
        self.assertEqual(best_config.kwargs.get("XBLOCK"), 128)

    def test_early_termination_stops_early(self):
        """Test that early termination stops when improvement is below threshold."""
        cfg = CoordescTunerConfig(
            early_stop_threshold=0.01,  # 1%
            early_stop_patience=2,
            max_iterations=100,
            warmup_samples=0,
        )
        tuner = CoordescTuner(tuner_config=cfg)
        baseline_config = triton.Config({"XBLOCK": 64}, num_warps=8, num_stages=1)

        # Function where optimum is at 64 - no improvement possible from neighbours
        def func(config):
            x = config.kwargs["XBLOCK"]
            return abs(x - 64) + 5.0

        best_config = tuner.autotune(func, baseline_config)
        # Should stop early and stay at 64
        self.assertEqual(best_config.kwargs.get("XBLOCK"), 64)
        # Should have terminated after patience iterations, not max
        self.assertLess(len(tuner._timing_history), 10)

    def test_early_termination_continues_when_improving(self):
        """Test that early termination does NOT stop when significant improvements exist."""
        cfg = CoordescTunerConfig(
            early_stop_threshold=0.001,
            early_stop_patience=2,
            max_iterations=100,
            warmup_samples=0,
        )
        tuner = CoordescTuner(tuner_config=cfg)
        baseline_config = triton.Config({"XBLOCK": 1}, num_warps=8, num_stages=1)

        def func(config):
            x = config.kwargs["XBLOCK"]
            return abs(x - 256) + 10.0

        best_config = tuner.autotune(func, baseline_config)
        # Should find a good value (way better than 1)
        self.assertGreater(best_config.kwargs.get("XBLOCK"), 1)

    def test_multi_field_tuning_pairs(self):
        """Test that multi-field tuning can find the optimum of a 2D valley."""
        cfg = CoordescTunerConfig(
            multi_field_tuning=True,
            multi_field_tuning_for_mm=True,
            warmup_samples=0,
            adaptive_step=False,
        )
        tuner = CoordescTuner(tuner_config=cfg)
        baseline_config = triton.Config(
            {"XBLOCK": 1, "R0_BLOCK": 1}, num_warps=8, num_stages=1
        )

        # Both fields are separable: XBLOCK optimum at 16, R0_BLOCK at 32.
        # Single-field tuning can find each independently, but multi-field
        # tuning can find them together faster.
        def func(config):
            x = config.kwargs["XBLOCK"]
            r = config.kwargs["R0_BLOCK"]
            return abs(x - 16) + abs(r - 32) + 1.0

        best_config = tuner.autotune(func, baseline_config)
        self.assertEqual(best_config.kwargs.get("XBLOCK"), 16)
        self.assertEqual(best_config.kwargs.get("R0_BLOCK"), 32)

    def test_warmup_finds_better_starting_point(self):
        """Test that warmup samples configs and can find a better starting point."""
        cfg = CoordescTunerConfig(
            warmup_samples=5,
            adaptive_step=False,
        )
        tuner = CoordescTuner(tuner_config=cfg)
        baseline_config = triton.Config({"XBLOCK": 1}, num_warps=8, num_stages=1)

        def func(config):
            x = config.kwargs["XBLOCK"]
            return abs(x - 128) + 5.0

        best_config = tuner.autotune(func, baseline_config)
        # Should find 128 (the global optimum)
        self.assertEqual(best_config.kwargs.get("XBLOCK"), 128)

    def test_max_iterations_respected(self):
        """Test that the tuner respects max_iterations."""
        cfg = CoordescTunerConfig(
            max_iterations=3,
            early_stop_patience=100,  # effectively disable early stop
            warmup_samples=0,
            adaptive_step=False,
        )
        tuner = CoordescTuner(tuner_config=cfg)
        baseline_config = triton.Config({"XBLOCK": 1}, num_warps=8, num_stages=1)

        def func(config):
            x = config.kwargs["XBLOCK"]
            return abs(x - 4096) + 10.0

        _ = tuner.autotune(func, baseline_config)
        # Verify we didn't exceed max_iterations: timing_history has
        # one entry for start + one per plateau iteration
        self.assertLessEqual(
            len(tuner._timing_history), 4
        )  # initial + at most 3 plateaus

    def test_no_neighbors(self):
        """
        Test the case that there is no available neighbor values for a field.
        """

        # size hint for x being 1 limits the max XBLOCK we try to be 1
        tuner = CoordescTuner(size_hints={"x": 1})
        baseline_config = triton.Config({"XBLOCK": 1}, num_warps=8, num_stages=1)

        def func(config):
            return abs(config.kwargs["XBLOCK"] - 15)

        best_config = tuner.autotune(func, baseline_config)
        self.assertTrue(best_config.kwargs.get("XBLOCK") == 1, str(best_config))

    def test_get_neighbour_values(self):
        tuner = CoordescTuner()

        neighbours = tuner.get_neighbour_values("num_stages", 2, radius=2)
        self.assertEqual(set(neighbours), {1, 3, 4})
        neighbours = tuner.get_neighbour_values("num_warps", 2, radius=2)
        self.assertEqual(set(neighbours), {1, 4, 8})

    def test_persistent_reduction(self):
        def f(x):
            return x / x.sum(dim=-1, keepdim=True)

        with mock.patch.object(
            CoordescTuner, "compare_config", mock_compare_config_prefer_larger_XBLOCK
        ):
            x = torch.ones(2, 256).to(GPU_TYPE)
            expected = f(x)
            # the first call get correct result when cache miss. Don't know why yet
            _ = torch.compile(f)(x)
            actual = torch.compile(f)(x)
            self.assertTrue(
                torch.allclose(expected, actual, atol=1e-4, rtol=1e-4),
                f"Expected:\n{expected}\nActual:\n{actual}",
            )

    def test_value_too_large(self):
        # Simulate a reduction
        size_hints = {"x": 2**20, "y": 2**20}

        tuner = CoordescTuner(size_hints=size_hints)

        max_block = TRITON_MAX_BLOCK
        self.assertFalse(tuner.value_too_large("XBLOCK", max_block["X"]))
        self.assertTrue(tuner.value_too_large("XBLOCK", max_block["X"] * 2))
        self.assertFalse(tuner.value_too_large("R0_BLOCK", max_block["R0_"]))
        self.assertTrue(tuner.value_too_large("R0_BLOCK", max_block["R0_"] * 2))

    def test_custom_correlated_fields(self):
        """Test that user-specified correlated fields are used."""
        cfg = CoordescTunerConfig(
            multi_field_tuning=True,
            correlated_fields=[("num_warps", "num_stages")],
            warmup_samples=0,
            adaptive_step=False,
        )
        tuner = CoordescTuner(tuner_config=cfg)
        pairs = tuner._get_correlated_field_pairs()
        self.assertEqual(pairs, [("num_warps", "num_stages")])

    def test_config_override_via_kwargs(self):
        """Test that all tuner_config fields can be set via constructor."""
        cfg = CoordescTunerConfig(
            max_iterations=10,
            early_stop_threshold=0.05,
            early_stop_patience=5,
            adaptive_step=False,
            multi_field_tuning=True,
            multi_field_tuning_for_mm=False,
            warmup_samples=8,
            check_all_directions=True,
            search_radius=2,
            correlated_fields=[("BLOCK_M", "BLOCK_N")],
        )
        tuner = CoordescTuner(tuner_config=cfg)
        self.assertEqual(tuner.tuner_config.max_iterations, 10)
        self.assertEqual(tuner.tuner_config.early_stop_threshold, 0.05)
        self.assertEqual(tuner.tuner_config.early_stop_patience, 5)
        self.assertFalse(tuner.tuner_config.adaptive_step)
        self.assertTrue(tuner.tuner_config.multi_field_tuning)
        self.assertFalse(tuner.tuner_config.multi_field_tuning_for_mm)
        self.assertEqual(tuner.tuner_config.warmup_samples, 8)
        self.assertTrue(tuner.tuner_config.check_all_directions)
        self.assertEqual(tuner.tuner_config.search_radius, 2)
        self.assertEqual(tuner.tuner_config.correlated_fields, [("BLOCK_M", "BLOCK_N")])


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU:
        run_tests()
