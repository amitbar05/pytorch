"""Vulkan kernel codegen — supports pointwise, reduction, and scan.

Track 1 (codegen refactor — DONE 2026-05-02):
  - ``main.py``                    — VulkanKernel core class (initialization, heuristics)
  - ``pointwise.py``               — PointwiseMixin (store, bwd_diff, DCE, register tile)
  - ``pointwise_load_mixin.py``    — PointwiseLoadMixin (dtype dispatch, packed16, buffer load)
  - ``pointwise_vec4_mixin.py``    — PointwiseVec4Mixin (vec4 pack/unpack, epilogue)
  - ``reduction.py``               — ReductionMixin (workgroup reduction, welford, scan, sort)
  - ``reduction_load_mixin.py``    — ReductionLoadMixin (dtype sizing, groupshared allocation)
  - ``reduction_tile_picker.py``   — reduction-kind enum, bank-conflict padding, wave helpers
  - ``indexing.py``                — IndexingMixin (index_to_str, iteration ranges, bounds)
  - ``symbolic.py``                — dynamic-shape helpers (OP.22: forward + backward)
  - ``header.py``                  — HeaderMixin (codegen_body, codegen_kernel, call_kernel)
  - ``bwd_diff_inline.py``         — backward-diff inline emission helpers
"""

from .main import VulkanKernel

__all__ = ["VulkanKernel"]
