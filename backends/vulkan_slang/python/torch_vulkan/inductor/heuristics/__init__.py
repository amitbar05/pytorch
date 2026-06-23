"""Inductor heuristics engine — central plan/cost model for Vulkan codegen.

Today: thin skeleton. Future P8.18+ items wire the existing
``_MM_TILE_CONFIGS`` indexing in ``vulkan_template_caller.py``,
WG-size autotune in ``autotune.py``, and per-op-class strategy
selection through one entry point so individual codegen sites stop
re-deriving the same priors.
"""
from __future__ import annotations

from .cost_model import (
    CostEstimate,
    KernelDescriptor,
    PointwiseDescriptor,
    PointwiseHeuristic,
    StaticPriorCostModel,
)

__all__ = [
    "CostEstimate",
    "KernelDescriptor",
    "PointwiseDescriptor",
    "PointwiseHeuristic",
    "StaticPriorCostModel",
]
