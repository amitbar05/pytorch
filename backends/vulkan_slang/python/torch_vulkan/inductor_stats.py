"""Public API for Inductor per-kernel performance counters.

Enable collection by setting ``TORCH_VULKAN_INDUCTOR_STATS=1`` before
importing torch_vulkan (the env var is read at import time in runtime.py).

Usage::

    import os
    os.environ["TORCH_VULKAN_INDUCTOR_STATS"] = "1"
    import torch_vulkan
    ...
    stats = torch_vulkan.inductor_stats.get_stats()
    # {key: {"call_count": N, "total_us": F, "last_args_len": M}}
    torch_vulkan.inductor_stats.reset_stats()
"""
from __future__ import annotations


def get_stats() -> dict[str, dict]:
    """Return a shallow copy of the per-kernel stats dict.

    Keys are the cache_key strings assigned to each Inductor-generated kernel
    at compile time.  Values are dicts with:
      * ``call_count``: number of times the kernel has been dispatched
      * ``total_us``:   cumulative wall-clock time spent inside the dispatch
                        wrapper, in microseconds (includes Python overhead)
      * ``last_args_len``: length of the last *args tuple (n_buffers + 3 for wg)
    """
    from torch_vulkan.inductor.runtime import _KERNEL_STATS
    return {k: dict(v) for k, v in _KERNEL_STATS.items()}


def reset_stats() -> None:
    """Clear all accumulated stats."""
    from torch_vulkan.inductor.runtime import _KERNEL_STATS
    _KERNEL_STATS.clear()


def is_enabled() -> bool:
    """Return True if stats collection is active (TORCH_VULKAN_INDUCTOR_STATS=1)."""
    from torch_vulkan.inductor.runtime import _INDUCTOR_STATS
    return _INDUCTOR_STATS
