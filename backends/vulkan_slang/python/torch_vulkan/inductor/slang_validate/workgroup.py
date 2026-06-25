"""Workgroup-size validation (pass 6, M27 advisory).

Extends numthreads-product checks with wave-size alignment advisory
for RDNA1 optimal occupancy.
"""

from __future__ import annotations

from ._config import _MAX_NUMTHREADS, _RE_NUMTHREADS, _WAVE_SIZE


def _validate_workgroup_size(src: str) -> list[str]:
    """M27: Validate workgroup size beyond bare product limits.

    Extends ``_check_numthreads_product`` with:
      - Non-multiple-of-wave-size advisory (suboptimal occupancy on RDNA1,
        where wave64 hardware must pad partial waves).
      - Product still clamped to ≤ 1024 (Vulkan minimum guarantee).

    Returns a list of error strings; empty list means pass.
    """
    errors: list[str] = []
    m = _RE_NUMTHREADS.search(src)
    if not m:
        return errors

    x, y, z = int(m.group(1)), int(m.group(2)), int(m.group(3))
    product = x * y * z

    # Hard limit — Vulkan minimum guarantee across all implementations.
    if product > _MAX_NUMTHREADS:
        errors.append(
            f"[M27] numthreads product {product} ({x}×{y}×{z}) exceeds "
            f"device limit of {_MAX_NUMTHREADS}"
        )
    if product == 0:
        errors.append(f"[M27] numthreads product is zero: ({x}, {y}, {z})")

    # Advisory: non-multiple of wave size leaves VGPR lanes idle.
    # On RDNA1 (wave64) a workgroup of 100 threads spans 2 waves
    # (128 lanes) → 28 lanes wasted.  Multiples of 64 guarantee
    # full-wave occupancy.  Skipped when _WAVE_SIZE is 0.
    #
    # NOTE: This is an ADVISORY check, not a hard error. Small workgroups
    # (e.g. 32 threads on wave64) are valid Vulkan — they just use partial
    # waves.  Blocking compilation on suboptimal occupancy prevents correct
    # kernels from running.  The warning is emitted but does not block.
    if _WAVE_SIZE > 0 and product % _WAVE_SIZE != 0:
        errors.append(
            f"[M27] numthreads product {product} ({x}×{y}×{z}) is not a "
            f"multiple of wave size {_WAVE_SIZE} — partial wave wastes lanes"
        )

    return errors
