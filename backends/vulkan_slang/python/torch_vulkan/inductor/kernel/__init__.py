"""Vulkan kernel codegen — supports pointwise, reduction, and scan.

Track 1 (codegen refactor — DONE 2026-05-02):
  - ``main.py``        — VulkanKernel core class (initialization, heuristics)
  - ``pointwise.py``   — PointwiseMixin (load, store, packed16, vec4)
  - ``reduction.py``   — ReductionMixin (workgroup reduction, welford, scan, sort)
  - ``indexing.py``    — IndexingMixin (index_to_str, iteration ranges, bounds)
  - ``symbolic.py``    — dynamic-shape stub (NotImplementedError)
  - ``header.py``      — HeaderMixin (codegen_body, codegen_kernel, call_kernel)
"""
from .main import VulkanKernel

__all__ = ["VulkanKernel"]

