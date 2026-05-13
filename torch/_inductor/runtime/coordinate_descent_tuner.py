# mypy: allow-untyped-defs
import copy
import itertools
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import TYPE_CHECKING

from torch.utils._ordered_set import OrderedSet

from ..utils import get_max_numwarps
from .hints import TRITON_MAX_BLOCK
from .runtime_utils import red_text, triton_config_to_hashable

if TYPE_CHECKING:
    from .triton_compat import triton


log = logging.getLogger(__name__)


@dataclass
class CoordescTunerConfig:
    """Configuration for the coordinate descent autotuner."""

    # Maximum number of outer-loop iterations (each iteration tunes all fields once)
    max_iterations: int = 100

    # Early termination: stop if the best timing hasn't improved by more than
    # this fraction in the last `early_stop_patience` iterations.
    # e.g. 0.001 means 0.1% improvement threshold.
    early_stop_threshold: float = 0.001

    # Number of iterations with less than `early_stop_threshold` improvement
    # before stopping early.
    early_stop_patience: int = 3

    # Enable adaptive step sizes per field (like a per-field learning rate).
    # When True, step sizes shrink when a direction stops yielding improvements.
    adaptive_step: bool = True

    # Enable multi-field (joint) tuning when single-field tuning plateaus.
    # When True, pairs of correlated fields are tuned together.
    multi_field_tuning: bool = False

    # For matmul kernels, multi-field tuning is particularly useful because
    # BLOCK_M and BLOCK_N are correlated.
    multi_field_tuning_for_mm: bool = True

    # Number of warmup samples to evaluate before starting coordinate descent.
    # The tuner will sample `warmup_samples` random configs and start from the best.
    # Set to 0 to disable (start from the provided baseline config).
    warmup_samples: int = 4

    # After single-field tuning plateaus, check all direction combinations.
    check_all_directions: bool = False

    # Search radius for neighbour values.
    search_radius: int = 1

    # Correlated field pairs for multi-field tuning.
    # Each tuple is a pair of field names that should be tuned together.
    # If empty, sensible defaults are chosen based on kernel type.
    correlated_fields: list[tuple[str, str]] = dc_field(default_factory=list)


def get_field(config, name):
    if name == "num_warps":
        return config.num_warps
    elif name == "num_stages":
        return config.num_stages
    elif name == "waves_per_eu":
        return config.kwargs.get(name, int(8 // config.num_warps))
    else:
        return config.kwargs.get(name, None)


def set_field(config, name, value):
    if name == "num_warps":
        config.num_warps = value
    elif name == "num_stages":
        config.num_stages = value
    else:
        config.kwargs[name] = value


class CoordescTuner:
    """
    The coordinate descent tuner. Tune one field/coordinate at a time.

    Features:
    - Adaptive step sizes: coarse-to-fine search with per-field learning rates.
    - Multi-field tuning: tune correlated field pairs jointly when single-field plateaus.
    - Early termination: stop when improvement drops below a threshold.
    - Warmup: sample multiple initial configs and start from the best.
    """

    def __init__(
        self,
        is_mm=False,
        is_native_matmul=False,
        is_mix_order_reduction=False,
        name="unknown",
        size_hints=None,
        inductor_meta=None,
        frozen_fields=None,
        tuner_config: CoordescTunerConfig | None = None,
    ):
        self.is_mm = is_mm  # we will tune num_stages for mm

        # Native matmul codegen assumes ZBLOCK=1 always.
        # This is because 3d tl.dot is slow and so we want to tile y and x only.
        # tl.dot also does not support size smaller than 16; we put this restriction.
        self.is_native_matmul = is_native_matmul
        assert not (self.is_mm and self.is_native_matmul)
        self.is_mix_order_reduction = is_mix_order_reduction
        self.cached_benchmark_results = {}
        self.name = name
        self.size_hints = size_hints
        self.inductor_meta = inductor_meta or {}
        self.frozen_fields: OrderedSet[str] = (
            OrderedSet(frozen_fields) if frozen_fields is not None else OrderedSet()
        )

        # Tuner configuration
        if tuner_config is None:
            tuner_config = CoordescTunerConfig()
        self.tuner_config = tuner_config

        # Per-field step sizes for adaptive tuning (maps field_name -> current_step)
        self._step_sizes: dict[str, int] = {}

        # For early termination: track best timing per iteration
        self._timing_history: list[float] = []

    def _init_step_size(self, name: str) -> int:
        """Initialize the step size for a field."""
        if name in ("num_stages", "NUM_STAGES"):
            return 2  # start coarser for stages
        elif name == "num_warps":
            return 2  # powers of 2 step
        else:
            # For block sizes, start with a multiplicative factor of 4
            return 4

    def _get_correlated_field_pairs(self) -> list[tuple[str, str]]:
        """Return the list of correlated field pairs for multi-field tuning."""
        cfg = self.tuner_config
        if cfg.correlated_fields:
            return cfg.correlated_fields

        # Sensible defaults based on kernel type
        fields = self.tunable_fields
        pairs = []

        # For matmuls, BLOCK_M and BLOCK_N are highly correlated
        if self.is_mm and (cfg.multi_field_tuning or cfg.multi_field_tuning_for_mm):
            if "BLOCK_M" in fields and "BLOCK_N" in fields:
                pairs.append(("BLOCK_M", "BLOCK_N"))
            if "num_warps" in fields and "num_stages" in fields:
                pairs.append(("num_warps", "num_stages"))
        else:
            # For reductions: XBLOCK and R0_BLOCK are correlated
            if "XBLOCK" in fields and "R0_BLOCK" in fields:
                pairs.append(("XBLOCK", "R0_BLOCK"))
            if "num_warps" in fields and "num_stages" in fields:
                pairs.append(("num_warps", "num_stages"))

        return pairs

    def get_config_max(self, prefix: str) -> int:
        max_block = TRITON_MAX_BLOCK[prefix.upper()]
        size_hint = self.size_hints.get(prefix) if self.size_hints is not None else None
        return min(max_block, size_hint) if size_hint is not None else max_block

    def get_warpsmax(self):
        # Avoid querying device directly if device properties are populated in inductor_meta
        warp_size = self.inductor_meta.get("warp_size")
        max_threads_per_block = self.inductor_meta.get("max_threads_per_block")
        if warp_size and max_threads_per_block:
            return max_threads_per_block // warp_size
        else:
            return get_max_numwarps()

    def cache_benchmark_result(self, config, timing):
        self.cached_benchmark_results[triton_config_to_hashable(config)] = timing

    def lookup_in_cache(self, config):
        return self.cached_benchmark_results.get(triton_config_to_hashable(config))

    def call_func(self, func, config):
        found = self.lookup_in_cache(config)
        if found is not None:
            log.debug("  CACHED")
            return found
        timing = func(config)
        self.cache_benchmark_result(config, timing)
        return timing

    @property
    def tunable_fields(self):
        out = [
            "XBLOCK",
            "YBLOCK",
            "ZBLOCK",
            # NOTE: we should not tune R0_BLOCK for persistent reduction.
            # We rely on the fact that persistent reduction's triton.Config
            # does not have the R0_BLOCK field to guarantee that.
            "R0_BLOCK",
            "R1_BLOCK",
            # the following 3 are for mm
            "BLOCK_M",
            "BLOCK_N",
            "BLOCK_K",
            "num_warps",
        ]
        if self.is_mm:
            out.append("num_stages")
        if self.inductor_meta.get("is_hip") is True:
            out.append("waves_per_eu")
        if self.is_native_matmul:
            out.append("num_stages")
            out.remove("ZBLOCK")  # ZBLOCK=1 always in native matmul

        if self.is_mix_order_reduction:
            # unlike TritonConfig.num_stages, this one is
            # put in TritonConfig.kwargs["NUM_STAGES"] and is used to
            # control the stage of pipelining of tl.range.
            out.append("NUM_STAGES")

        return [f for f in out if f not in self.frozen_fields]

    def value_too_large(self, name: str, val: int) -> bool:
        block_suffix = "BLOCK"
        if name.endswith(block_suffix):
            prefix = name.strip(block_suffix).lower()
            return val > self.get_config_max(prefix)
        if name == "num_warps":
            return val > self.get_warpsmax()
        if name == "waves_per_eu":
            return val > 8

        return False

    def value_too_small(self, name: str, val: int) -> bool:
        # In native matmul, block size should be >= 16 for tl.dot
        if self.is_native_matmul:
            if name in ["YBLOCK", "XBLOCK", "R0_BLOCK"]:
                return val < 16

        # Break if value becomes 0/neg
        return val <= 0

    def get_neighbour_values(self, name, orig_val, radius=None, include_self=False):
        """
        Get neighbour values in 'radius' steps. The original value is not
        returned as it's own neighbour.

        When adaptive_step is enabled, uses per-field step sizes for multiplicative
        fields (blocks, num_warps) instead of fixed doubling/halving.
        """
        if radius is None:
            radius = self.tuner_config.search_radius
        if name == "NUM_STAGES":
            # we see cases that
            # NUM_STAGES=1 is better than NUM_STAGES=2
            # while NUM_STAGES=1 is worse than NUM_STAGES=3
            radius = max(radius, 2)

        assert radius >= 1

        # Determine the multiplicative step factor for adaptive tuning
        if self.tuner_config.adaptive_step and name not in (
            "num_stages",
            "NUM_STAGES",
        ):
            if name not in self._step_sizes:
                self._step_sizes[name] = self._init_step_size(name)
            step_factor = self._step_sizes[name]
        else:
            step_factor = 2  # default: double/halve

        def update(cur_val, inc=True):
            if name in ["num_stages", "NUM_STAGES"]:
                if inc:
                    return cur_val + 1
                else:
                    return cur_val - 1
            else:
                if inc:
                    return cur_val * step_factor
                else:
                    return cur_val // step_factor

        out = []
        # increment loop
        cur_val = orig_val
        for _ in range(radius):
            cur_val = update(cur_val, True)
            if self.value_too_large(name, cur_val):
                break
            out.append(cur_val)

        # decrement loop
        cur_val = orig_val
        for _ in range(radius):
            cur_val = update(cur_val, False)
            if self.value_too_small(name, cur_val):
                break
            out.append(cur_val)

        if include_self:
            out.append(orig_val)
        return out

    def _shrink_step(self, name: str) -> None:
        """Shrink the adaptive step size for a field (fine-grained search)."""
        if name in self._step_sizes and self._step_sizes[name] > 2:
            self._step_sizes[name] = max(2, self._step_sizes[name] // 2)
            log.debug("  Shrinking step for %s to %d", name, self._step_sizes[name])

    def has_improvement(self, baseline, test):
        threshold = self.tuner_config.early_stop_threshold
        return test is not None and test < baseline * (1 - threshold)

    def is_valid_config(self, config) -> bool:
        if self.is_mix_order_reduction:
            # Mix order reduction has an extra constraint that
            # we should not tune XBLOCK beyond RSPLIT_SIZE
            xblock = config.kwargs["XBLOCK"]
            split_size = config.kwargs["RSPLIT_SIZE"]
            return xblock <= split_size
        return True

    def check_all_tuning_directions(
        self,
        # pyrefly: ignore [missing-attribute]
        func: Callable[["triton.Config"], float],
        best_config,
        best_timing,
    ):
        """
        Check all directions. We only do this once the regular coordinate
        descent tuning find no better choices any more.
        We only have a few tunable fields, so this should be fine.
        """
        candidate_values_list = []
        effective_fields = []
        for field in self.tunable_fields:
            old_value = get_field(best_config, field)
            if old_value is None:
                continue
            radius = self.inductor_meta.get(
                "coordinate_descent_search_radius", self.tuner_config.search_radius
            )
            candidate_values = self.get_neighbour_values(
                field,
                old_value,
                radius=radius,
                include_self=True,
            )
            candidate_values_list.append(candidate_values)
            effective_fields.append(field)

        choices = itertools.product(*candidate_values_list)
        improved = False
        for choice in choices:
            assert len(choice) == len(effective_fields)
            candidate_config = copy.deepcopy(best_config)
            for new_val, field in zip(choice, effective_fields):
                set_field(candidate_config, field, new_val)
            if not self.is_valid_config(candidate_config):
                continue
            cmp_res, candidate_timing = self.compare_config(
                func, candidate_config, best_config, best_timing
            )
            if cmp_res:
                improved = True
                best_config = candidate_config
                best_timing = candidate_timing

        return improved, best_config, best_timing

    def _multi_field_tune(
        self,
        func: Callable[["triton.Config"], float],  # pyrefly: ignore
        best_config,
        best_timing,
    ) -> tuple[bool, "triton.Config", float]:  # pyrefly: ignore
        """
        Tune pairs of correlated fields jointly. This is useful when two fields
        interact (e.g., BLOCK_M and BLOCK_N for matmuls).

        For each correlated pair, we try all combinations of neighbour values
        for both fields simultaneously.
        """
        pairs = self._get_correlated_field_pairs()
        if not pairs:
            return False, best_config, best_timing

        improved = False
        for field_a, field_b in pairs:
            val_a = get_field(best_config, field_a)
            val_b = get_field(best_config, field_b)
            if val_a is None or val_b is None:
                continue

            neighbours_a = self.get_neighbour_values(field_a, val_a, include_self=True)
            neighbours_b = self.get_neighbour_values(field_b, val_b, include_self=True)

            log.debug(
                "  Multi-field tuning pair (%s, %s): %d x %d combinations",
                field_a,
                field_b,
                len(neighbours_a),
                len(neighbours_b),
            )

            for next_a in neighbours_a:
                for next_b in neighbours_b:
                    # Skip the current config — already benchmarked
                    if next_a == val_a and next_b == val_b:
                        continue

                    candidate_config = copy.deepcopy(best_config)
                    set_field(candidate_config, field_a, next_a)
                    set_field(candidate_config, field_b, next_b)

                    if not self.is_valid_config(candidate_config):
                        continue
                    cmp_res, candidate_timing = self.compare_config(
                        func, candidate_config, best_config, best_timing
                    )
                    if cmp_res:
                        improved = True
                        best_config, best_timing = candidate_config, candidate_timing

        return improved, best_config, best_timing

    def _check_early_termination(self, best_timing: float) -> bool:
        """
        Check if we should terminate early based on the timing history.
        Returns True if we should stop.
        """
        cfg = self.tuner_config
        if cfg.early_stop_patience <= 0:
            return False

        self._timing_history.append(best_timing)

        if len(self._timing_history) < cfg.early_stop_patience + 1:
            return False

        # Check if the improvement over the last K iterations is below threshold
        recent = self._timing_history[-cfg.early_stop_patience - 1 :]
        old_best = recent[0]
        new_best = min(recent[1:])

        if old_best > 0 and (old_best - new_best) / old_best < cfg.early_stop_threshold:
            log.debug(
                "  Early termination: improvement %.4f%% below threshold %.4f%% over last %d iterations",
                100 * (old_best - new_best) / old_best,
                100 * cfg.early_stop_threshold,
                cfg.early_stop_patience,
            )
            return True
        return False

    def _warmup(
        self,
        func: Callable[["triton.Config"], float],  # pyrefly: ignore
        baseline_config: "triton.Config",  # pyrefly: ignore
        baseline_timing: float,
    ) -> tuple["triton.Config", float]:  # pyrefly: ignore
        """
        Sample several configs before starting coordinate descent and return
        the best one as the starting point.
        """
        cfg = self.tuner_config
        if cfg.warmup_samples <= 0:
            return baseline_config, baseline_timing

        best_config = baseline_config
        best_timing = baseline_timing
        tunable_fields = self.tunable_fields

        log.debug(
            "  Warmup: sampling %d random configs (baseline: %f)",
            cfg.warmup_samples,
            baseline_timing,
        )

        # Seed random with kernel name for deterministic warmup across runs.
        # Save/restore global state to avoid interfering with other randomness.
        rng_state = random.getstate()
        try:
            # Hash the kernel name to get a deterministic seed
            seed_val = hash(self.name) & 0xFFFFFFFF
            random.seed(seed_val)

            for i in range(cfg.warmup_samples):
                candidate_config = copy.deepcopy(baseline_config)
                # Randomly perturb some fields
                fields_to_perturb = random.sample(
                    tunable_fields,
                    k=min(len(tunable_fields), random.randint(1, len(tunable_fields))),
                )
                for field in fields_to_perturb:
                    cur_val = get_field(candidate_config, field)
                    if cur_val is None:
                        continue
                    neighbours = self.get_neighbour_values(
                        field, cur_val, radius=self.tuner_config.search_radius
                    )
                    if neighbours:
                        new_val = random.choice(neighbours)
                        set_field(candidate_config, field, new_val)

                if not self.is_valid_config(candidate_config):
                    continue

                try:
                    timing = self.call_func(func, candidate_config)
                except Exception as e:
                    log.debug("  Warmup config %d failed: %s", i, e)
                    continue

                log.debug("  Warmup config %d: %f", i, timing)
                if self.has_improvement(best_timing, timing):
                    best_config = candidate_config
                    best_timing = timing
                    log.debug("  Warmup found better config: %f", timing)

        finally:
            random.setstate(rng_state)

        return best_config, best_timing

    def compare_config(self, func, candidate_config, best_config, best_timing):
        """
        Check if candidate_config is better than best_config.

        Return a tuple of (compare_result, candidate_timing).
        compare_result is true iff candidate_config is better.
        """
        log.debug("Try config %s", candidate_config)
        try:
            candidate_timing = self.call_func(func, candidate_config)
        except Exception as e:
            log.debug("Got exception %s", e)  # noqa: G200
            return False, float("inf")

        if self.has_improvement(best_timing, candidate_timing):
            log.debug(
                "Tune from %s %f -> %s %f",
                best_config,
                best_timing,
                candidate_config,
                candidate_timing,
            )

            return True, candidate_timing
        return False, candidate_timing

    def autotune(
        self,
        # pyrefly: ignore [missing-attribute]
        func: Callable[["triton.Config"], float],
        # pyrefly: ignore [missing-attribute]
        baseline_config: "triton.Config",
        baseline_timing: float | None = None,
    ) -> "triton.Config":  # pyrefly: ignore  # missing-attribute
        if baseline_timing is None:
            baseline_timing = self.call_func(func, baseline_config)

        log.debug("= Do coordinate descent tuning for %s =", self.name)
        log.debug(
            "%s: Baseline Config %s, baseline timing %f",
            self.name,
            baseline_config,
            baseline_timing,
        )

        # Phase 0: Warmup - sample configs to find a better starting point
        best_config, best_timing = self._warmup(func, baseline_config, baseline_timing)

        # Reset timing history for early termination tracking
        self._timing_history = [best_timing]

        tunable_fields = self.tunable_fields
        iteration = 0
        improved = True
        field_improved_in_iteration: dict[str, bool] = {}

        while improved and iteration < self.tuner_config.max_iterations:
            improved = False
            iteration += 1
            log.debug("  Iteration %d, best timing %f", iteration, best_timing)

            for name in tunable_fields:
                cur_val = get_field(best_config, name)
                # some kernel don't have R0_BLOCK/YBLOCK/ZBLOCK. So cur_val may be None
                if cur_val is None:
                    continue

                # It's possible that candidate_values is empty.
                # E.g., if XBLOCK is 1 initially and size_hint for x is also 1.
                # We would not try either larger or smaller XBLOCK in this case.
                candidate_values = self.get_neighbour_values(name, cur_val)

                field_improved = False
                for next_val in candidate_values:
                    candidate_config = copy.deepcopy(best_config)
                    set_field(candidate_config, name, next_val)

                    if not self.is_valid_config(candidate_config):
                        continue
                    cmp_res, candidate_timing = self.compare_config(
                        func, candidate_config, best_config, best_timing
                    )
                    if cmp_res:
                        improved = True
                        field_improved = True
                        best_config, best_timing = candidate_config, candidate_timing

                # Adaptive step: shrink step size when no improvement found for this field
                if (
                    self.tuner_config.adaptive_step
                    and not field_improved
                    and name in self._step_sizes
                ):
                    self._shrink_step(name)

                field_improved_in_iteration[name] = field_improved

            # Phase 2: Multi-field tuning when single-field plateaus
            if not improved and (
                self.tuner_config.multi_field_tuning
                or self.tuner_config.multi_field_tuning_for_mm
            ):
                log.debug("  Single-field tuning plateaued, trying multi-field tuning")
                improved, best_config, best_timing = self._multi_field_tune(
                    func, best_config, best_timing
                )

            # Phase 3: Check all directions
            if not improved and self.tuner_config.check_all_directions:
                old_best_timing = best_timing
                improved, best_config, best_timing = self.check_all_tuning_directions(
                    func, best_config, best_timing
                )

                if improved:
                    msg = red_text(
                        "%s: Coordinate descent tuning found improvement of %.3fx by looking in all directions."
                    )
                    log.debug(
                        msg,
                        self.name,
                        old_best_timing / best_timing,
                    )

            # Early termination check (also tracks timing history internally)
            if not improved and self._check_early_termination(best_timing):
                log.debug("  Stopping early at iteration %d", iteration)
                break

        log.debug(
            "%s: Improve from %s %f -> %s %f, %.3fx (iterations: %d)",
            self.name,
            baseline_config,
            baseline_timing,
            best_config,
            best_timing,
            baseline_timing / best_timing,
            iteration,
        )

        return best_config

    @staticmethod
    def autotune_single_field(fn, init_val, min_val=None, max_val=None):
        """
        fn is a function that takes the field value and returns the benchmarking result
        init_val is the starting point of autotuning.

        Should work well for parabola like curve. Here is a real example
        for split-size of mix-order-reduction: https://github.com/pytorch/pytorch/pull/166461
        """
        cache = {}

        def _bench(val):
            if val not in cache:
                cache[val] = fn(val)
                # print(f"split size {val} -> {cache[val]:.3f} ms")
            return cache[val]

        if min_val is None:
            min_val = 1
        if max_val is None:
            max_val = 2**30  # some arbitrary large value

        best_val = init_val
        improved = True
        while improved:
            improved = False
            candlist = [best_val // 2, best_val * 2]
            for cand in candlist:
                cand = max(cand, min_val)
                cand = min(cand, max_val)

                if _bench(cand) < _bench(best_val):
                    best_val = cand
                    improved = True

        return best_val
