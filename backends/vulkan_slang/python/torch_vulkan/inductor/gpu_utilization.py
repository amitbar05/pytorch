"""GPU.4/GPU.5 — GPU utilization diagnostics and reporting.

Provides ``gpu_utilization_report()`` which measures dispatch-level
GPU utilization metrics for a compiled model, and helper functions
for occupancy estimation from workgroup sizes and grid dimensions.
"""

from __future__ import annotations

import time
from typing import Any

import torch


def gpu_utilization_report(
    model: torch.nn.Module | callable,
    sample_input: tuple | torch.Tensor,
    *,
    warmup_iters: int = 3,
    measure_iters: int = 10,
) -> dict[str, Any]:
    """Report GPU utilization metrics for a compiled model.

    Runs the model in no_grad mode, collects dispatch counts and
    wall-clock timing, and returns a dictionary of utilization metrics.

    Args:
        model: A compiled ``torch.compile(backend="inductor")`` callable
               or a ``torch.nn.Module``.
        sample_input: Input tensors for the model.
        warmup_iters: Number of warm-up iterations (default 3).
        measure_iters: Number of measurement iterations (default 10).

    Returns:
        Dict with keys:
          - ``total_time_ms``: average per-iteration time (ms)
          - ``dispatch_count``: number of Vulkan dispatches per iteration
          - ``avg_dispatch_us``: average time per dispatch (µs)
          - ``utilization_estimate``: rough GPU utilization % estimate
          - ``measure_iters``: number of measurement iterations used
    """
    import torch_vulkan

    if not isinstance(sample_input, (tuple, list)):
        sample_input = (sample_input,)

    # Ensure Vulkan device
    device_inputs = []
    for inp in sample_input:
        if isinstance(inp, torch.Tensor) and inp.device.type != "vulkan":
            device_inputs.append(inp.to("vulkan:0"))
        else:
            device_inputs.append(inp)
    sample_input = tuple(device_inputs)

    with torch.no_grad():
        # Warm up — triggers compilation on first run
        for _ in range(warmup_iters):
            model(*sample_input)

        # Reset perf counters and measure
        torch_vulkan._c_ext._reset_perf_counters()
        start = time.perf_counter()
        for _ in range(measure_iters):
            model(*sample_input)
        elapsed = time.perf_counter() - start

        dispatch_count = torch_vulkan._c_ext._get_dispatch_count()

    avg_time_ms = (elapsed / measure_iters) * 1000
    avg_dispatch_us = (
        (elapsed / max(dispatch_count, 1)) * 1e6 if dispatch_count > 0 else 0
    )

    # Rough utilization estimate: how much of the wall-clock time
    # is spent inside Vulkan dispatches (vs Python overhead).
    # A low ratio (< 50%) indicates dispatch overhead dominates.
    # This is a heuristic — actual GPU utilization requires
    # GPU perf counters (VK_KHR_performance_query).
    if dispatch_count > 0 and elapsed > 0:
        # Assume each dispatch does at least some GPU work.
        # The overhead portion is Python dispatch overhead + command
        # buffer submission.  We estimate utilization as the inverse
        # of the overhead ratio.
        # Typical dispatch overhead is ~5-20 µs per dispatch on RDNA1.
        overhead_estimate_us = dispatch_count * 10  # 10 µs per dispatch
        utilization_estimate = max(
            0.0, min(100.0, (1.0 - overhead_estimate_us / (elapsed * 1e6)) * 100)
        )
    else:
        utilization_estimate = 0.0

    return {
        "total_time_ms": avg_time_ms,
        "dispatch_count": dispatch_count,
        "avg_dispatch_us": avg_dispatch_us,
        "utilization_estimate": utilization_estimate,
        "measure_iters": measure_iters,
    }


def estimate_occupancy(
    threadgroup_size: int,
    vgprs_per_thread: int,
    shared_mem_bytes: int = 0,
    *,
    simd_size: int = 64,
    cu_vgprs: int = 256,
    cu_lds_bytes: int = 64 * 1024,
    max_waves_per_cu: int = 4,
) -> dict[str, Any]:
    """Estimate GPU occupancy for a given workgroup configuration.

    Uses the RDNA1 occupancy model:
      - 64 VGPRs per SIMD, 256 VGPRs per CU (4 SIMDs)
      - 64 KB LDS per CU
      - Max 1024 threads per CU (4 wave64 waves)

    Args:
        threadgroup_size: Number of threads per workgroup.
        vgprs_per_thread: Estimated VGPRs used per thread.
        shared_mem_bytes: Groupshared memory used per workgroup (bytes).
        simd_size: Subgroup/wave size (default 64 for wave64).
        cu_vgprs: Total VGPRs per CU (default 256 for RDNA1).
        cu_lds_bytes: Total LDS per CU in bytes (default 64KB).
        max_waves_per_cu: Maximum concurrent waves per CU (default 4).

    Returns:
        Dict with estimated waves per CU, occupancy %, and limiting factor.
    """
    # VGPR-limited waves per CU
    vgprs_per_wave = vgprs_per_thread * simd_size
    waves_by_vgpr = cu_vgprs // max(vgprs_per_wave, 1)
    waves_by_vgpr = min(waves_by_vgpr, max_waves_per_cu)

    # LDS-limited waves per CU
    if shared_mem_bytes > 0:
        waves_by_lds = cu_lds_bytes // max(shared_mem_bytes, 1)
        waves_by_lds = min(waves_by_lds, max_waves_per_cu)
    else:
        waves_by_lds = max_waves_per_cu

    # Thread-limited waves per CU (1024 threads / CU max)
    waves_by_threads = min(1024 // max(threadgroup_size, 1), max_waves_per_cu)

    waves_per_cu = min(waves_by_vgpr, waves_by_lds, waves_by_threads)
    occupancy_pct = (waves_per_cu / max_waves_per_cu) * 100

    # Determine limiting factor
    if waves_per_cu == waves_by_vgpr:
        limit = "vgpr"
    elif waves_per_cu == waves_by_lds:
        limit = "lds"
    elif waves_per_cu == waves_by_threads:
        limit = "threads"
    else:
        limit = "unknown"

    return {
        "waves_per_cu": waves_per_cu,
        "occupancy_pct": occupancy_pct,
        "limiting_factor": limit,
        "vgprs_per_wave": vgprs_per_wave,
        "shared_mem_per_wg": shared_mem_bytes,
    }
