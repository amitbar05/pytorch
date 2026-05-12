"""P8.17 — Heuristics cost-model skeleton.

Central plan/cost model for Vulkan kernel-shape decisions. Today the
skeleton wires only the pointwise op-class — workgroup-size selection
that today lives as a small static table in ``autotune.py`` /
``kernel.py``. P8.18 wires the `mm` tile selection (replacing
``vulkan_template.py:_MM_TILE_CONFIGS`` direct indexing) so the
caller asks the cost model for a tile rather than picking one by
heuristic. P8.19 layers online refinement over the same seam.

The skeleton is intentionally *prior-only* (no measurement feedback).
Every method takes a ``KernelDescriptor`` and returns a
``CostEstimate`` derived from priors known to be true on RDNA1
(1024-thread workgroup cap, wave64, 32 KiB groupshared limit,
single-issue scalar pipe). When `autotune.py` writes a measurement
back, the next call site's heuristic sees the refined estimate
through `update_with_measurement`. P8.19 is the item that lands the
update path.

Anti-pattern this avoids: per-codegen-site re-derivation of the same
priors. Today `_default_workgroup_size` lives in three places
(``kernel.py``, ``autotune.py``, ``vulkan_template_caller.py``) and
drifts. The skeleton is the single import.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# RDNA1 architecture priors. These are the constants every later
# heuristic decision derives from; they live here once instead of
# being duplicated across the codegen pipeline.
RDNA1_MAX_WORKGROUP_THREADS: int = 1024
RDNA1_WAVE_SIZE: int = 64
RDNA1_GROUPSHARED_BYTES: int = 32 * 1024


@dataclass(frozen=True)
class KernelDescriptor:
    """Common interface every kernel-class descriptor extends.

    The descriptor is what the caller hands to the cost model — it is
    *not* the codegen-side ``VulkanKernel`` (that owns SymPy / IR
    bookkeeping). The descriptor is a flat, hash-friendly value type
    that can key a cache.
    """

    # Defaults so dataclass inheritance with subclass-level new
    # fields works without forcing every subclass to re-declare
    # parent fields. Subclasses still override `op_class`.
    op_class: str = "unknown"
    dtype: str = "f32"


@dataclass(frozen=True)
class PointwiseDescriptor(KernelDescriptor):
    """Pointwise-kernel shape: a single numel and dtype."""

    numel: int = 0
    op_class: str = "pointwise"

    def __post_init__(self) -> None:
        if self.numel < 0:
            raise ValueError(f"numel must be ≥ 0, got {self.numel}")


@dataclass(frozen=True)
class CostEstimate:
    """Cost model output: a workgroup-size choice plus a relative
    cost score (lower is better). Optional fields hold the *why* so
    autotune can log + ratchet.

    The score is in arbitrary units — only the relative ordering
    between candidates is meaningful. Priors-only estimates produce
    deterministic scores so the regression suite can lock them.
    """

    workgroup_size: tuple[int, int, int]
    score: float
    reason: str = ""
    metadata: dict = field(default_factory=dict)


class PointwiseHeuristic:
    """Workgroup-size pick for pointwise kernels.

    Today: pick the largest power-of-two ≤ ``RDNA1_MAX_WORKGROUP_THREADS``
    that divides ``numel`` evenly, capped at 256 (the empirical sweet
    spot on RDNA1 for memory-bandwidth-bound pointwise; bigger
    workgroups don't add throughput because the limiter is the L1$
    line + scalar-issue throttle, not lane count). A 1-D dispatch
    geometry is implicit; multi-D dispatch is a P8.20 follow-up.
    """

    _CAP = 256
    _CANDIDATES = (256, 128, 64, 32, 16, 8, 4, 2, 1)

    def estimate(self, desc: PointwiseDescriptor) -> CostEstimate:
        if desc.numel <= 0:
            return CostEstimate(
                workgroup_size=(1, 1, 1),
                score=0.0,
                reason="degenerate-empty-workload",
            )
        for c in self._CANDIDATES:
            if c <= self._CAP and desc.numel % c == 0:
                return CostEstimate(
                    workgroup_size=(c, 1, 1),
                    score=float(desc.numel) / float(c),
                    reason=f"largest-divisor-le-{self._CAP}",
                    metadata={"divisor": c},
                )
        # numel is not divisible by any power-of-two we consider —
        # fall back to 32 (one wavefront) and pad in codegen.
        return CostEstimate(
            workgroup_size=(32, 1, 1),
            score=float(desc.numel) / 32.0,
            reason="indivisible-fallback-wave32",
        )


class StaticPriorCostModel:
    """Top-level entry point. Routes a descriptor to its op-class
    heuristic. P8.18 adds ``mm`` routing; P8.20 adds vec-load width.

    Today the model is pure-priors (no online refinement). The
    `update_with_measurement` hook is a no-op stub; P8.19 implements
    persistence + replay.
    """

    def __init__(self) -> None:
        self._pointwise = PointwiseHeuristic()

    def estimate(self, desc: KernelDescriptor) -> CostEstimate:
        if isinstance(desc, PointwiseDescriptor):
            return self._pointwise.estimate(desc)
        # Unknown op-class: return a deterministic fallback so the
        # caller can detect "model didn't know" and fall back to the
        # legacy heuristic without a hard crash.
        return CostEstimate(
            workgroup_size=(32, 1, 1),
            score=float("inf"),
            reason=f"no-heuristic-for-op-class:{desc.op_class}",
        )

    def update_with_measurement(
        self,
        desc: KernelDescriptor,
        actual_us: float,
        chosen: CostEstimate,
    ) -> None:
        """P8.19 hook — record a measurement so future estimates
        prefer faster plans. Today this is a no-op; the seam exists so
        callers can wire in unconditionally and P8.19 turns it on.
        """
        return None


_default_model: Optional[StaticPriorCostModel] = None


def default_model() -> StaticPriorCostModel:
    """Process-wide singleton. The skeleton is stateless today, so
    sharing one instance is safe; P8.19 will turn this into a real
    cache that benefits from singleton scope.
    """
    global _default_model
    if _default_model is None:
        _default_model = StaticPriorCostModel()
    return _default_model
