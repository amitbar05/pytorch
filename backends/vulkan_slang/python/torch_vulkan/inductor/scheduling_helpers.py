"""Module-level helpers extracted from scheduling.py (M22c split, anti-goal #7)."""

from __future__ import annotations

import hashlib
from typing import Optional


def compute_combo_config_key(sub_config_keys) -> str:
    """M-pipeline-7: derive a content-aware cache key for a combo kernel
    from the tuple of its sub-kernels' ``config_key`` strings.

    Why this exists: the prior implementation used the literal string
    ``"combo"`` as the cache key for every combo kernel in the process.
    The ``_reflection_metrics_by_key`` cross-index in
    ``runtime/slangc.py`` then returned whichever combo happened to be
    compiled first, corrupting WG-sizing heuristics for every
    subsequent combo. Same bug class as M-pipeline-3 (single-kernel
    collisions) at a different cache layer.

    Order-sensitive: the combo's emitted Slang lays out gtid ranges and
    binding slots in sub-kernel iteration order, so combos that differ
    only in sub-kernel order ARE structurally different and must map to
    distinct cache slots.

    Key format: ``combo2_n{N}_{hash16}``. The ``combo2_`` prefix bumps
    the cache version so any in-memory entries from the old
    ``"combo"`` format cannot collide with new entries. The ``nN``
    component is human-debuggable (you can read the combo size off the
    key without rehashing). ``hash16`` is a 16-char SHA-1 prefix over
    ``repr(tuple(sub_config_keys))``.
    """
    sub_keys_tuple = tuple(sub_config_keys)
    n = len(sub_keys_tuple)
    return (
        f"combo2_n{n}_"
        f"{hashlib.sha1(repr(sub_keys_tuple).encode()).hexdigest()[:16]}"
    )


_cached_wave64_ok: Optional[bool] = None


def _wave64_persistent_ok() -> bool:
    """M-PERF.6: True iff device subgroup size is 64 (RDNA1/2/GCN).

    Gates the reduction+pointwise ``rnumel`` cap raise to 1024
    (16 waves of 64 fits RDNA1's 1024-thread WG ceiling). On wave32
    hardware (Nvidia/Intel/RDNA3) we keep the 256 cap. Cached on first
    call; falls back to False if the device probe fails.
    """
    global _cached_wave64_ok
    if _cached_wave64_ok is not None:
        return _cached_wave64_ok
    sgs: Optional[int] = None
    try:
        from .device_profile import current

        profile = current()
        if profile is not None:
            limits = profile.get("limits", {})
            if limits.get("subgroup_size_min") == limits.get("subgroup_size_max"):
                sgs = limits.get("subgroup_size_max")
    except Exception:
        pass
    if sgs is None:
        try:
            from torch._dynamo.device_interface import get_interface_for_device

            iface = get_interface_for_device("vulkan")
            props = iface.Worker.get_device_properties()
            sgs = getattr(props, "subgroup_size", None)
        except Exception:
            pass
    _cached_wave64_ok = sgs == 64
    return _cached_wave64_ok


_BENCHMARKER = None


def _get_benchmarker():
    """Cached `Benchmarker` instance. P2.2 — `benchmark_codegened_module` is
    called per autotune candidate; instantiating fresh per call repeated the
    benchmarker's init work for no benefit. The benchmarker is stateless
    across calls so a single instance is fine.
    """
    global _BENCHMARKER
    if _BENCHMARKER is None:
        from torch._inductor.runtime.benchmarking import Benchmarker

        _BENCHMARKER = Benchmarker()
    return _BENCHMARKER


def _reset_benchmarker_cache() -> None:
    """Test hook — clears the cached benchmarker."""
    global _BENCHMARKER
    _BENCHMARKER = None


def _register_vulkan_benchmarker_once() -> None:
    """Register Vulkan's wall-clock benchmarker exactly once at module load.

    Inductor's `Benchmarker` looks up a per-device entry in the registry on
    every benchmark call. Re-running `@register_benchmarker(..., override=True)`
    inside `benchmark_codegened_module` (the previous shape) replaced the
    entry on every autotune iteration, which is wasted work.
    """
    try:
        from torch._inductor.runtime.benchmarking import register_benchmarker

        @register_benchmarker("vulkan", override=True)
        def _vulkan_bench(self, f, *, warmup, rep, **kw):
            f()
            import time

            timings = []
            t0 = time.perf_counter()
            while True:
                start = time.perf_counter()
                f()
                end = time.perf_counter()
                timings.append((end - start) * 1000)
                if (end - t0) * 1000 > rep:
                    break
            from statistics import median

            return median(timings)
    except Exception:
        pass


_register_vulkan_benchmarker_once()


def _fusion_has_new_float_reads(node1, node2) -> bool:
    """Return True if node2 reads float buffers not in node1's input/output set.

    GPU.1 guard: On AMD RDNA1, after wg_welford's cross-wave LDS reduction,
    L2 cache state is corrupted for same-cache-line SSBO reads by WGs 1-N.
    Triggered by any broadcast buffer (weight/bias) read in the same dispatch,
    regardless of dtype — confirmed empirically for fp16/bf16, also observed
    for fp32 layer_norm with fresh cache compiles.
    """
    import torch

    _FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
    try:
        node1_produces: set[str] = set(node1.get_buffer_names())
        node1_reads: set[str] = set(node1.used_buffer_names())
        node2_reads: set[str] = set(node2.used_buffer_names())

        known = node1_produces | node1_reads
        extra = node2_reads - known
        if not extra:
            return False

        from torch._inductor.virtualized import V

        for name in extra:
            buf = V.graph.get_buffer(name)
            if buf is not None and buf.get_dtype() in _FLOAT_DTYPES:
                return True
        return False
    except Exception:
        return False


# Backwards-compat alias used by existing callers.
_fusion_has_new_half_reads = _fusion_has_new_float_reads
