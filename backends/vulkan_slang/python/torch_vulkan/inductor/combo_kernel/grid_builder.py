"""Grid and numthreads computation for Vulkan combo-kernel dispatch.

The combo kernel uses a multi-dimensional grid (TRAIN.6-F1):
- X = max workgroups needed by any single subkernel.
- Y = number of subkernels (``gid.y`` selects which subkernel runs).
- Each workgroup (x, y) runs subkernel y, with ``gid.x`` as the
  subkernel's own workgroup ID. This preserves wave uniformity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..kernel import VulkanKernel


def compute_max_threadgroup_size(
    subkernels: list[tuple["VulkanKernel", int]],
) -> int:
    """Return the maximum threadgroup size across all subkernels."""
    max_tgs = 256
    for kernel, _ in subkernels:
        max_tgs = max(max_tgs, kernel.max_threadgroup_size)
    return max_tgs


def compute_max_workgroups(
    subkernels: list[tuple["VulkanKernel", int]],
    max_tgs: int,
) -> int:
    """Return the maximum workgroup count (X dimension) across subkernels.

    Reduction subkernels need one workgroup per output element (gid.x
    indexes the non-reduction axis). Pointwise uses ``ceil(numel / TGS)``.
    """
    return max(_wg_count(k, n, max_tgs) for k, n in subkernels)


def _wg_count(kernel: "VulkanKernel", numel: int, max_tgs: int) -> int:
    """Number of workgroups for a single subkernel."""
    if getattr(kernel, "inside_reduction", False):
        return numel  # one workgroup per output element
    return (numel + max_tgs - 1) // max_tgs


def compute_grid_dims(
    subkernels: list[tuple["VulkanKernel", int]],
) -> tuple[int, int, int]:
    """Return (wg_x, wg_y, wg_z) for a multi-dimensional combo-kernel dispatch.

    wg_x = max workgroups needed by any subkernel.
    wg_y = number of subkernels (selects which subkernel via gid.y).
    wg_z = 1 (unused).
    """
    max_tgs = compute_max_threadgroup_size(subkernels)
    wg_x = compute_max_workgroups(subkernels, max_tgs)
    wg_y = len(subkernels)
    wg_z = 1
    return wg_x, wg_y, wg_z
