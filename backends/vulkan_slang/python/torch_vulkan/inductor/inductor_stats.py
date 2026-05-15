"""Per-kernel performance statistics for the Vulkan Inductor backend.

Enable via ``TORCH_VULKAN_INDUCTOR_STATS=1`` environment variable.
Provides :func:`get_stats` / :func:`reset_stats` for programmatic access.
"""

from __future__ import annotations

import os
from typing import Any


def _stats_enabled() -> bool:
    return os.environ.get("TORCH_VULKAN_INDUCTOR_STATS") == "1"


def _get_stats_dict() -> dict[str, dict[str, Any]]:
    from .runtime import _KERNEL_STATS

    return _KERNEL_STATS


def get_stats() -> dict[str, dict[str, Any]]:
    """Return per-kernel stats collected since the last reset.

    Returns a dict mapping kernel cache key → ``{call_count, total_us, last_args_len}``.
    Only populated when ``TORCH_VULKAN_INDUCTOR_STATS=1``.
    """
    if not _stats_enabled():
        return {}
    return dict(_get_stats_dict())


def reset_stats() -> None:
    """Clear all collected kernel stats."""
    _get_stats_dict().clear()


def print_stats(top_n: int = 20) -> None:
    """Print the top-N kernels by cumulative dispatch time."""
    stats = get_stats()
    if not stats:
        return
    sorted_stats = sorted(stats.items(), key=lambda kv: kv[1]["total_us"], reverse=True)
    for i, (key, entry) in enumerate(sorted_stats[:top_n]):
        us = entry["total_us"]
        cnt = entry["call_count"]
        avg = us / cnt if cnt else 0
        print(
            f"  [{i + 1:2d}] {us:10.1f} us  ({cnt:5d} calls, {avg:7.1f} us/call) {key}"
        )


def compile_stats() -> dict[str, Any]:
    """Return the compile-time profiler counters.

    Always-on. Tracks ``cold_compiles`` (slangc subprocess invocations),
    ``cold_compile_us`` (cumulative wall time), ``in_memory_hits`` and
    ``disk_cache_hits`` (cache hits at the two levels), and
    ``prewarm_submits`` (entries the prewarm pool submitted to slangc).
    Includes derived ``cache_hit_rate`` for quick eyeballing.
    """
    from .runtime import _COMPILE_STATS

    s = dict(_COMPILE_STATS)
    hits = s["in_memory_hits"] + s["disk_cache_hits"]
    total = hits + s["cold_compiles"]
    s["cache_hit_rate"] = (hits / total) if total else 0.0
    s["avg_cold_compile_us"] = (
        s["cold_compile_us"] / s["cold_compiles"] if s["cold_compiles"] else 0.0
    )
    return s


def reset_compile_stats() -> None:
    """Zero out the compile-time profiler counters."""
    from .runtime import _COMPILE_STATS

    for k in _COMPILE_STATS:
        _COMPILE_STATS[k] = 0 if isinstance(_COMPILE_STATS[k], int) else 0.0


def print_full_report(top_n: int = 10) -> None:
    """Print a single-page diagnostic combining per-kernel + compile counters.

    The two numbers users actually want when inspecting a slow workload are
    "where is dispatch time going?" and "how much cold-compile am I paying?".
    Print both in one shot so a `print_full_report()` cell tells you both.
    """
    s = summary(top_n=top_n)
    cs = compile_stats()
    print(
        f"=== inductor_stats ({s['n_kernels']} kernels, "
        f"{s['total_calls']} calls, {s['total_us']:.1f} us total) ==="
    )
    if s["top"]:
        for k, us, cnt, avg, spv_hash in s["top"]:
            pct = 100.0 * us / s["total_us"] if s["total_us"] else 0.0
            tag = f"  [spv:{spv_hash}]" if spv_hash else ""
            print(
                f"  {pct:5.1f}%  {us:9.1f} us  ({cnt:5d}x, {avg:7.1f} us/call)  {k}{tag}"
            )
    print(
        f"=== compile_stats: {cs['cold_compiles']} cold "
        f"({cs['cold_compile_us'] / 1000:.1f} ms), "
        f"{cs['in_memory_hits'] + cs['disk_cache_hits']} hits "
        f"({100 * cs['cache_hit_rate']:.1f}%), "
        f"{cs['prewarm_submits']} prewarm ==="
    )
    from torch_vulkan.inductor.buffer_pool import pool_stats as _pool_stats

    p = _pool_stats()
    pool_hit_rate = (p["hits"] / p["acquires"]) if p["acquires"] else 0.0
    print(
        f"=== buffer_pool: {p['acquires']} acquires "
        f"({100 * pool_hit_rate:.1f}% hit), "
        f"{p['releases']} releases, "
        f"size_now={p['size_now']} peak={p['size_peak']} ==="
    )


def print_compile_stats() -> None:
    """Print a human-readable one-shot summary of the compile-time counters."""
    s = compile_stats()
    cold_ms = s["cold_compile_us"] / 1000.0
    avg_ms = s["avg_cold_compile_us"] / 1000.0
    print(
        f"slangc cold compiles: {s['cold_compiles']:5d}  "
        f"({cold_ms:7.1f} ms total, {avg_ms:6.1f} ms avg)"
    )
    print(
        f"cache hits: {s['in_memory_hits']:5d} mem + {s['disk_cache_hits']:5d} disk "
        f"({100 * s['cache_hit_rate']:5.1f}%)"
    )
    print(f"prewarm submits: {s['prewarm_submits']}")


def dump_waterfall(path: str) -> dict[str, Any]:
    """Write a per-kernel waterfall to ``path`` as JSON.

    Each entry: ``{name, dispatches, total_us, avg_us, percent_total}``,
    sorted by ``total_us`` descending. Returns the same payload (also
    contains ``total_us`` and ``total_calls`` aggregate fields). Suitable
    for grafana ingestion or a quick "what is the bottleneck" glance.

    Returns ``{"n_kernels": 0}`` when stats collection is disabled.
    """
    import json

    stats = get_stats()
    payload: dict[str, Any] = {
        "n_kernels": len(stats),
        "total_calls": sum(e["call_count"] for e in stats.values()),
        "total_us": sum(e["total_us"] for e in stats.values()),
        "kernels": [],
    }
    if stats:
        denom = payload["total_us"] or 1.0
        for k, e in sorted(
            stats.items(), key=lambda kv: kv[1]["total_us"], reverse=True
        ):
            cnt = e["call_count"]
            payload["kernels"].append(
                {
                    "name": k,
                    "dispatches": cnt,
                    "total_us": e["total_us"],
                    "avg_us": e["total_us"] / cnt if cnt else 0.0,
                    "percent_total": 100.0 * e["total_us"] / denom,
                }
            )
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


class MemoryTracker:
    """Context manager that samples Vulkan cached-memory before/after a region.

    The Vulkan caching allocator's ``memory_cached()`` returns the total bytes
    currently held in the cache (free + in-use). Sampling at the start and end
    of a workload + recording the max during dispatch gives a reasonable proxy
    for peak working set. P2.4.

    Usage:

        with MemoryTracker() as tr:
            compiled_fn(x)
        print(tr.peak_mib, tr.delta_mib)

    Caveat: this is a *sampled* peak — true peak requires C++ instrumentation
    inside ``VulkanAllocator::allocate``. The sample is taken at construction
    and on every ``poll()`` call; if the workload allocates and frees a large
    intermediate between two polls, the peak is missed.
    """

    def __init__(self):
        self.start = 0
        self.end = 0
        self.peak = 0

    def __enter__(self):
        import torch_vulkan

        self.start = torch_vulkan.memory_cached()
        self.peak = self.start
        return self

    def poll(self) -> int:
        import torch_vulkan

        cur = torch_vulkan.memory_cached()
        if cur > self.peak:
            self.peak = cur
        return cur

    def __exit__(self, *exc):
        import torch_vulkan

        self.end = torch_vulkan.memory_cached()
        if self.end > self.peak:
            self.peak = self.end

    @property
    def delta_mib(self) -> float:
        return (self.end - self.start) / (1024 * 1024)

    @property
    def peak_mib(self) -> float:
        return self.peak / (1024 * 1024)


def peak_memory_report() -> dict[str, Any]:
    """Return a snapshot of the Vulkan allocator's current state.

    Returns ``{cached_mib, n_kernels_recorded}``. Use as a Jupyter-friendly
    one-liner; for region-scoped peak, use ``MemoryTracker`` as a context
    manager. P2.4.
    """
    import torch_vulkan

    return {
        "cached_mib": torch_vulkan.memory_cached() / (1024 * 1024),
        "n_kernels_recorded": len(get_stats()),
    }


def summary(top_n: int = 20) -> dict[str, Any]:
    """Aggregate the per-kernel stats into a single summary dict.

    Returns ``{n_kernels, total_calls, total_us, avg_us_per_call,
    top: [(key, total_us, call_count, avg_us)]}``. Useful as a return value
    from a Jupyter cell — pretty-prints via the standard repr.

    Returns an empty dict when stats collection is disabled.
    """
    stats = get_stats()
    if not stats:
        from torch_vulkan.inductor.buffer_pool import pool_stats as _pool_stats

        return {
            "n_kernels": 0,
            "total_calls": 0,
            "total_us": 0.0,
            "avg_us_per_call": 0.0,
            "top": [],
            "buffer_pool": _pool_stats(),
        }
    from torch_vulkan.inductor.runtime import _KERNEL_SPIRV_HASH

    total_calls = sum(e["call_count"] for e in stats.values())
    total_us = sum(e["total_us"] for e in stats.values())
    sorted_stats = sorted(stats.items(), key=lambda kv: kv[1]["total_us"], reverse=True)
    top = [
        (
            k,
            e["total_us"],
            e["call_count"],
            e["total_us"] / e["call_count"] if e["call_count"] else 0.0,
            _KERNEL_SPIRV_HASH.get(k, ""),
            # M11.8: occupancy metadata (populated when reflection is available)
            e.get("wg_x", 0),
            e.get("vgprs", 0),
            e.get("lds_bytes", 0),
        )
        for k, e in sorted_stats[:top_n]
    ]
    from torch_vulkan.inductor.buffer_pool import pool_stats as _pool_stats

    return {
        "n_kernels": len(stats),
        "total_calls": total_calls,
        "total_us": total_us,
        "avg_us_per_call": total_us / total_calls if total_calls else 0.0,
        "top": top,
        "buffer_pool": _pool_stats(),
    }
