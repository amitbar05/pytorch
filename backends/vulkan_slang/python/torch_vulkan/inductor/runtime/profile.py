"""Dispatch profiling — per-kernel timing and prewarm infrastructure.

Provides ``_record_dispatch_time`` for the wrapper codegen to instrument
individual kernel dispatches, and ``dispatch_times`` for summary retrieval.
"""

import math

# Per-dispatch timing ring buffer for GPU.3 profiling.
# Keyed by kernel cache_key; stores list of (elapsed_us,) tuples.
_DISPATCH_TIMES: dict[str, list[float]] = {}
_DISPATCH_TIMES_MAX_SAMPLES = 10_000  # per-key cap


def _record_dispatch_time(kernel_key: str, elapsed_us: float) -> None:
    """Record a single dispatch's wall-clock time for profiling (GPU.3)."""
    times = _DISPATCH_TIMES.get(kernel_key)
    if times is None:
        times = []
        _DISPATCH_TIMES[kernel_key] = times
    if len(times) < _DISPATCH_TIMES_MAX_SAMPLES:
        times.append(elapsed_us)


def dispatch_times() -> dict[str, dict]:
    """Return per-kernel dispatch timing summaries.

    Returns a dict mapping kernel_key → {"min_us", "mean_us", "max_us",
    "count", "stdev_us"}.  Empty when GPU.3 profiling is disabled or no
    dispatches have been recorded.
    """
    result: dict[str, dict] = {}
    for key, times in list(_DISPATCH_TIMES.items()):
        if not times:
            continue
        n = len(times)
        mean = sum(times) / n
        variance = sum((t - mean) ** 2 for t in times) / n if n > 1 else 0.0
        result[key] = {
            "min_us": min(times),
            "mean_us": mean,
            "max_us": max(times),
            "count": n,
            "stdev_us": math.sqrt(variance),
        }
    return result


def _reset_dispatch_times() -> None:
    """Clear the dispatch timing buffer. Test hook."""
    _DISPATCH_TIMES.clear()
