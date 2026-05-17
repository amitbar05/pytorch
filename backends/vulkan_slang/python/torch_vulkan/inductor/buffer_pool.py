"""Recycle pool for Inductor-emitted Vulkan tensors.

Closes the per-allocation overhead gap on small compiled workloads: each
Inductor-generated buffer used to round-trip through ``torch.empty_strided``
→ PrivateUse1 dispatcher → ``VulkanAllocator::allocate`` → tensor wrap, costing
~17 us/call. On compiled MLP forward (8 buffers/step) that's ~140 us — the
entire excess vs eager. The C++ allocator already pools device memory by
size class, so the cost we still pay is the dispatch + wrap round-trip
itself, not the device-side allocation.

PF.41: the bucket key is ``(size_tuple, dtype, lifetime_class)`` —
*not* the old ``(size_tuple, stride_tuple, dtype)``. The lifetime tag
flows from PF.40's joint-graph FX annotation into the wrapper-codegen's
``empty_strided_vulkan(..., lifetime_class=...)`` call site, so a
``transient`` allocation can never collide with a ``save_for_backward``
bucket — the latter must outlive the forward pass and is the load-bearing
change for multi-step training memory plateau.

Stride is dropped from the key (see PF.44 for the underlying observation):
the C++ allocator already returns contiguous storage for our size class,
and `_empty_strided_fast` synthesizes the requested stride at tensor-wrap
time, so two acquires with the same shape/dtype but different strides can
share a bucket without correctness risk.

The wrapper emits ``vulkan_pool_release(buf_N, lifetime_class=...)``
instead of ``del buf_N`` at end-of-life; the next step's
``empty_strided_vulkan`` call hits the pool and skips the dispatch.

Per-key cap (4) and global cap (64) bound memory pressure on the 6 GB RDNA1.
The pool is **on by default** (set ``TORCH_VULKAN_BUFFER_POOL=0`` to disable).
PF.33 (T6.3) closed: template callers (mm/addmm/bmm/flash-attention/Philox)
now acquire output buffers from the pool via :func:`pool_acquire` before
falling back to ``torch.empty``; the wrapper's ``generate_extern_kernel_alloc``
pre-allocates via ``empty_strided_vulkan`` and passes ``out=`` to the extern
kernel, so extern-kernel outputs no longer bypass the pool.

PF.42 step-end release uses ``release_class("gradient")`` to drop every
gradient bucket on each ``optimizer.zero_grad()``; PF.41 alone reuses
``transient`` aggressively across step boundaries and keeps
``save_for_backward`` live across the fwd→bwd cut.

The hot path is performance-sensitive — Inductor wrappers run on a single
thread per compiled graph, so we skip threading.local and store the buckets
as module globals. The release/acquire helpers avoid all dict-of-dict
indirection by hoisting the stats dict to a flat module variable.

TRAIN.8: extern-kernel buffer pool integration.  Template callers for
mm / addmm / bmm / flash-attention / conv2d now acquire output buffers
from the pool via :func:`pool_acquire` before falling back to
``torch.empty``.  The convenience wrappers ``pool_acquire(size, dtype)``
and ``pool_release(buffer)`` keep the hot path readable.
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Optional

import torch

# Mirrors `torch_vulkan.inductor.lifetime.LIFETIME_CLASSES`. Hard-coded
# rather than imported to keep the hot-path module import-free of the FX
# subsystem. Validated against the lifetime module by the regression
# suite (``TestStepActivationPool`` keys on these names).
LIFETIME_CLASSES = frozenset(
    {"parameter", "gradient", "save_for_backward", "transient", "output", "scratch"}
)
_DEFAULT_LIFETIME = "transient"
# T6.7: ``scratch`` is the bucket for intermediate workspace tensors used
# inside a single extern-kernel dispatch (mm split-K accumulators, multi-
# stage reduction partials, philox second-output, flash-attention LSE,
# foreach-optimizer padding dummies). Distinct from ``transient`` so a
# scratch acquire can never collide with a user-visible buffer.
_SCRATCH_LIFETIME = "scratch"


_POOL_DISABLED = os.environ.get("TORCH_VULKAN_BUFFER_POOL", "1") == "0"
_PER_KEY_CAP_DEFAULT = int(os.environ.get("TORCH_VULKAN_BUFFER_POOL_PER_KEY", "4"))
_GLOBAL_CAP = int(os.environ.get("TORCH_VULKAN_BUFFER_POOL_SIZE", "64"))
# M9.6: adaptive per-key caps per lifetime class. Higher caps for high-churn
# classes (scratch/transient) reduce eviction pressure; lower caps for
# long-lived classes (save_for_backward) bound memory without hurting reuse.
# The env var TORCH_VULKAN_BUFFER_POOL_PER_KEY sets the fallback for classes
# not in this table (default 4).
# M17.7: increased per-key caps for training workloads. The LIFO hot-cache
# (see ``_lifo`` / ``_LIFO_MAX``) absorbs same-graph same-numel reuse and
# bypasses the per-key cap, so these caps mostly govern cross-graph reuse
# pressure. Raised from scratch=8/transient=6/save_for_backward=4.
_PER_KEY_CAPS: dict[str, int] = {
    _SCRATCH_LIFETIME: 16,
    "transient": 12,
    "save_for_backward": 8,
}
# TRAIN.8: detailed per-event pool stats, opt-in via TORCH_VULKAN_POOL_STATS=1.
# Tracks per-event timing (acquire/release hit/miss/evict), per-bucket histograms,
# and total bytes recycled. Access via :func:`pool_stats_detailed()`.
_POOL_STATS_ENABLED = os.environ.get("TORCH_VULKAN_POOL_STATS", "0") == "1"
_POOL_TRACE: list[dict] = []
_POOL_TRACE_MAX = 1024
_POOL_TRACE_INDEX = 0
_POOL_BYTES_ACQUIRED = 0
_POOL_BYTES_RELEASED = 0
_POOL_BYTES_EVICTED = 0
_POOL_HIT_SIZES: dict[int, int] = {}

_buckets: dict[tuple, deque] = {}

# M17.7: LIFO hot-cache for same-graph reuse.  When a buffer is released
# it lands here first (max ``_LIFO_MAX`` entries).  The next acquire in
# the same graph checks the LIFO *before* the per-class buckets, matching
# on ``(numel, dtype)`` only — lifetime_class is ignored for same-graph
# reuse because the original buffer is already dead (freed).  This
# collapses alloc→free→alloc chains within a single compiled graph and
# raises the effective hit rate from ~36 % toward ≥80 %.
#
# When the LIFO is full, the oldest entry is evicted to its regular
# per-class bucket.  On ``release_class``, matching entries are removed
# from the LIFO as well (the gradient bucket is reclaimed at the
# optimizer boundary).
_lifo: list[tuple[tuple, object]] = []  # [(key, tensor), ...]
_LIFO_MAX = 16

_size_now = 0
_stats = {
    "acquires": 0,
    "hits": 0,
    "lifo_hits": 0,
    "misses": 0,
    "releases": 0,
    "evictions": 0,
    "size_now": 0,
    "size_peak": 0,
}
# PF.42: per-class release counter, incremented every time `release_class`
# drops a bucket. Lets the FX-pass test floor verify the zero-grad release
# hook actually fired without round-tripping through pool_stats() (which
# tracks total releases, not class-keyed releases).
_release_counts: dict[str, int] = {cls: 0 for cls in LIFETIME_CLASSES}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _per_key_cap_for(lifetime_class: str) -> int:
    """Return the per-key capacity for a given lifetime class.

    M9.6: adaptive caps — scratch=8, transient=6, save_for_backward=4.
    If ``TORCH_VULKAN_BUFFER_POOL_PER_KEY`` is explicitly set in the
    environment, that value overrides all classes (preserving the legacy
    behaviour for tests and tuning). Otherwise, each class uses its own cap;
    classes not in ``_PER_KEY_CAPS`` fall back to ``_PER_KEY_CAP_DEFAULT``
    (4, or the env-var override).
    """
    # When the env var is explicitly set (not default), use it for all classes.
    if "TORCH_VULKAN_BUFFER_POOL_PER_KEY" in os.environ:
        return _PER_KEY_CAP_DEFAULT
    return _PER_KEY_CAPS.get(lifetime_class, _PER_KEY_CAP_DEFAULT)


def _numel(size) -> int:
    n = 1
    for d in size:
        n *= int(d)
    return n


def _key(size, dtype, lifetime_class: str) -> tuple:
    # M9.1: key on storage element count, not shape. The wrapper-codegen
    # path allocates with ``allocation_shape`` (e.g. ``(1, 8, 64)``) and
    # then ``.as_strided((8, 64), …)``-rewraps. By release-time the tensor
    # carries the view shape, so a shape-keyed bucket sees acquires with
    # ``(1, 8, 64)`` and releases with ``(8, 64)`` and never matches —
    # silent 0 % hit rate. ``as_strided`` is a view-only operation; the
    # underlying storage has the same element count on both sides, so
    # keying on numel is the correct invariant.
    return (_numel(size), dtype, lifetime_class)


def _compute_contiguous_stride(size: tuple[int, ...]) -> tuple[int, ...]:
    """Compute row-major contiguous strides for the given size."""
    if not size:
        return ()
    strides = [1]
    for d in reversed(size[1:]):
        strides.append(strides[-1] * d)
    return tuple(reversed(strides))


def _record_pool_event(event_type: str, **kwargs) -> None:
    """Append a structured event to the trace ring buffer when stats are enabled."""
    if not _POOL_STATS_ENABLED:
        return
    global _POOL_TRACE_INDEX
    entry = {"type": event_type, "ts_us": time.perf_counter() * 1e6, **kwargs}
    if len(_POOL_TRACE) < _POOL_TRACE_MAX:
        _POOL_TRACE.append(entry)
    else:
        _POOL_TRACE[_POOL_TRACE_INDEX % _POOL_TRACE_MAX] = entry
    _POOL_TRACE_INDEX += 1


# ═══════════════════════════════════════════════════════════════════════════
# Convenience hooks for extern-kernel callers (TRAIN.8)
# ═══════════════════════════════════════════════════════════════════════════


def pool_acquire(
    size: tuple[int, ...],
    dtype: torch.dtype,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Simpler pool-acquire for extern-kernel callers (mm/conv/SDPA).

    Computes a contiguous stride from ``size`` and delegates to
    :func:`vulkan_pool_acquire`.  Returns ``None`` on cache miss — the
    caller falls through to ``torch.empty``.

    ``device`` is accepted for call-site clarity but currently ignored
    (every Vulkan tensor is on ``vulkan:0``).
    """
    stride = _compute_contiguous_stride(size)
    t = vulkan_pool_acquire(size, stride, dtype, lifetime_class="transient")
    if t is not None and _POOL_STATS_ENABLED:
        global _POOL_BYTES_ACQUIRED
        _POOL_BYTES_ACQUIRED += t.element_size() * t.numel()
        size_class = t.numel()
        _POOL_HIT_SIZES[size_class] = _POOL_HIT_SIZES.get(size_class, 0) + 1
    return t


def pool_release(buffer: torch.Tensor) -> None:
    """Simpler pool-release for extern-kernel callers.

    Delegates to :func:`vulkan_pool_release` with ``lifetime_class="transient"``.
    The wrapper-codegen emits ``pool_release(buf)`` at end-of-life so the
    next dispatch can recycle the underlying storage without another
    PrivateUse1 round-trip.
    """
    if buffer is None:
        return
    if _POOL_STATS_ENABLED:
        global _POOL_BYTES_RELEASED
        _POOL_BYTES_RELEASED += buffer.element_size() * buffer.numel()
    vulkan_pool_release(buffer, lifetime_class="transient")


def pool_acquire_scratch(
    size: tuple[int, ...],
    dtype: torch.dtype,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """T6.7: pool-acquire for *intermediate workspace* tensors.

    Used by extern-kernel callers (philox normal mode second output,
    flash-attention LSE buffer, foreach-optimizer padding dummy, mm
    split-K accumulators, multi-stage reduction partials) that allocate
    a buffer purely for internal dispatch use — the caller never returns
    the scratch tensor and drops its reference immediately after the
    underlying op returns.

    Distinct from :func:`pool_acquire` (``transient`` bucket) so the
    scratch storage cannot collide with a user-visible Inductor buffer.
    Returns ``None`` on cache miss; caller falls through to ``torch.empty``.
    ``device`` is accepted for call-site clarity but currently ignored
    (every Vulkan tensor is on ``vulkan:0``).
    """
    stride = _compute_contiguous_stride(size)
    t = vulkan_pool_acquire(size, stride, dtype, lifetime_class=_SCRATCH_LIFETIME)
    if t is not None and _POOL_STATS_ENABLED:
        global _POOL_BYTES_ACQUIRED
        _POOL_BYTES_ACQUIRED += t.element_size() * t.numel()
        size_class = t.numel()
        _POOL_HIT_SIZES[size_class] = _POOL_HIT_SIZES.get(size_class, 0) + 1
    return t


def pool_release_scratch(buffer: torch.Tensor) -> None:
    """T6.7: companion release for :func:`pool_acquire_scratch`.

    Returns ``buffer`` to the ``scratch`` bucket. Caller must drop its
    own reference immediately afterward (otherwise the next acquire
    hands back a tensor someone else still holds).
    """
    if buffer is None:
        return
    if _POOL_STATS_ENABLED:
        global _POOL_BYTES_RELEASED
        _POOL_BYTES_RELEASED += buffer.element_size() * buffer.numel()
    vulkan_pool_release(buffer, lifetime_class=_SCRATCH_LIFETIME)


# ═══════════════════════════════════════════════════════════════════════════
# Core pool operations
# ═══════════════════════════════════════════════════════════════════════════


def _lifo_acquire(size, stride, dtype, lifetime_class: str):
    """M17.7: search the LIFO hot-cache for a matching ``(numel, dtype)`` entry.

    Lifetime class is ignored for same-graph reuse — the original buffer
    was freed, so its storage is available regardless of its former class.
    Returns ``(tensor, popped_index)`` on hit, ``(None, -1)`` on miss.
    The caller must ``as_strided`` the tensor if size/stride don't match.
    """
    target_numel = _numel(size)
    for i, (lifo_key, t) in enumerate(_lifo):
        if lifo_key[0] == target_numel and lifo_key[1] == dtype:
            del _lifo[i]
            return t
    return None


def _lifo_push(key, tensor):
    """M17.7: push a released tensor onto the LIFO hot-cache.

    When the LIFO is full, the oldest entry is evicted to its regular
    per-class bucket (respecting the per-key cap).  _size_now is NOT
    modified here — the caller (:func:`vulkan_pool_release`) handles
    the increment.  Evicting from LIFO to bucket is just a data-structure
    move, not a net size change.
    """
    if len(_lifo) >= _LIFO_MAX:
        evict_key, evict_t = _lifo.pop(0)
        _push_to_bucket(evict_key, evict_t)
    _lifo.append((key, tensor))


def _push_to_bucket(key, tensor):
    """Push ``tensor`` into the regular per-class bucket for ``key``.

    Respects the per-key cap — if the bucket is full the tensor is
    dropped (GC will free the underlying device memory).

    Does NOT modify ``_size_now`` — callers (:func:`_lifo_push` for
    LIFO eviction, future callers) handle the bookkeeping.  This is
    intentionally just the data-structure insert.
    """
    bucket = _buckets.get(key)
    if bucket is None:
        bucket = deque()
        _buckets[key] = bucket
    lt = key[2] if len(key) > 2 else _DEFAULT_LIFETIME
    if len(bucket) >= _per_key_cap_for(lt):
        _stats["evictions"] += 1
        _record_pool_event(
            "release_evict_per_key",
            key=str(key),
            per_key_cap=_per_key_cap_for(lt),
        )
        if _POOL_STATS_ENABLED:
            global _POOL_BYTES_EVICTED
            _POOL_BYTES_EVICTED += tensor.element_size() * tensor.numel()
        return
    bucket.append(tensor)


def vulkan_pool_acquire(size, stride, dtype, lifetime_class: str = _DEFAULT_LIFETIME):
    """Pop a recycled tensor matching ``(size, dtype, lifetime_class)`` from the pool.

    M17.7: searches the LIFO hot-cache first (``(numel, dtype)`` match,
    lifetime_class-agnostic) for same-graph reuse.  Falls back to the
    per-class bucket when the LIFO misses.

    ``stride`` is accepted for backwards-compat with the wrapper-codegen
    call signature but does not participate in the bucket key — the
    contiguous storage handed out by the C++ allocator can satisfy any
    requested stride at wrap time. ``lifetime_class`` defaults to
    ``"transient"`` so legacy callers that don't pass it land in the
    transient bucket (the most aggressively reused class).

    Returns ``None`` on miss (caller falls through to a fresh allocation).
    """
    if _POOL_DISABLED:
        return None
    global _size_now
    _stats["acquires"] += 1

    # M17.7: LIFO hot-cache first — lifetime_class-agnostic, same-graph reuse.
    t = _lifo_acquire(size, stride, dtype, lifetime_class)
    if t is not None:
        _size_now -= 1
        _stats["size_now"] = _size_now
        _stats["hits"] += 1
        _stats["lifo_hits"] += 1
        _record_pool_event(
            "acquire_hit_lifo",
            key=str(_key(size, dtype, lifetime_class)),
            size_now=_size_now,
        )
        if list(t.stride()) != list(stride) or list(t.size()) != list(size):
            t = t.as_strided(size, stride)
        return t

    key = _key(size, dtype, lifetime_class)
    bucket = _buckets.get(key)
    if not bucket:
        _stats["misses"] += 1
        _record_pool_event("acquire_miss", key=str(key))
        return None
    t = bucket.popleft()
    _size_now -= 1
    _stats["size_now"] = _size_now
    _stats["hits"] += 1
    _record_pool_event("acquire_hit", key=str(key), size_now=_size_now)
    # TRAIN.1 (2026-05-08): vulkan_empty_strided now creates tensors with
    # exact requested strides. A pooled tensor may have strides that don't
    # match the caller's request. Use as_strided to correct the view.
    if list(t.stride()) != list(stride) or list(t.size()) != list(size):
        t = t.as_strided(size, stride)
    return t


def vulkan_pool_release(tensor, lifetime_class: str = _DEFAULT_LIFETIME) -> None:
    """Return ``tensor`` to the pool keyed on its lifetime class.

    M17.7: lands in the LIFO hot-cache first (max ``_LIFO_MAX`` entries)
    so the next same-graph acquire can skip the per-class bucket.  When
    the LIFO is full the oldest LIFO entry is evicted to its regular
    bucket.

    ``lifetime_class`` controls which bucket the tensor lands in on
    eventual LIFO eviction. The wrapper-codegen reads
    ``node.meta["lifetime_class"]`` from PF.40's joint-graph annotation
    and emits the literal class name as a kwarg on the release call site.
    ``transient`` (the default) is the most aggressively reused class —
    same-step intra-graph allocations.  ``save_for_backward`` stays live
    until the bwd consumer fires (no cross-class reuse, so the fwd value
    can't be clobbered).  ``gradient`` releases at the
    ``optimizer.zero_grad()`` boundary via PF.42's :func:`release_class`.
    ``parameter`` never releases in practice (params stay live across
    steps); ``output`` is owned by the caller.

    Caller must drop its own reference (the wrapper emits
    ``vulkan_pool_release(buf, lifetime_class="..."); buf = None``);
    otherwise the next acquire hands back a tensor someone else still
    holds.

    Drops on the floor (lets refcount free it) when the pool is disabled,
    when the per-key bucket is full, or when the global cap is reached.
    """
    if _POOL_DISABLED or tensor is None:
        return
    global _size_now, _POOL_BYTES_EVICTED
    _stats["releases"] += 1

    if _size_now >= _GLOBAL_CAP:
        _stats["evictions"] += 1
        _record_pool_event("release_evict_global", size_now=_size_now)
        if _POOL_STATS_ENABLED:
            _POOL_BYTES_EVICTED += tensor.element_size() * tensor.numel()
        return
    # M9.1: use the underlying storage element count, not the view's
    # ``tensor.size()``. A buffer allocated as ``(1, 8, 64)`` and then
    # ``.as_strided((8, 64), …)``-rewrapped reports ``tensor.size() ==
    # (8, 64)`` but its storage still holds 512 elements — the next
    # acquire keys on 512 and must find this bucket.
    elem = tensor.element_size()
    storage_numel = (
        tensor.untyped_storage().nbytes() // elem if elem else tensor.numel()
    )
    key = _key((storage_numel,), tensor.dtype, lifetime_class)
    # M17.7: push to LIFO hot-cache first.  When full, the oldest LIFO
    # entry is evicted to its bucket.  The LIFO bypasses the per-key cap
    # for same-graph reuse — the regular bucket path still respects caps.
    _lifo_push(key, tensor)
    _size_now += 1
    if _size_now > _stats["size_peak"]:
        _stats["size_peak"] = _size_now
    _stats["size_now"] = _size_now
    _record_pool_event("release_accept", key=str(key), size_now=_size_now)


def _release_class_from_lifo(lifetime_class: str) -> int:
    """M17.7: remove entries with ``lifetime_class`` from the LIFO hot-cache.

    Iterates in reverse so ``del _lifo[i]`` does not shift unvisited indices.
    Returns the number of tensors removed.  Does NOT modify ``_size_now`` —
    the caller (:func:`release_class`) handles the accounting.
    """
    dropped = 0
    for i in range(len(_lifo) - 1, -1, -1):
        lifo_key, _t = _lifo[i]
        if lifo_key[2] == lifetime_class:
            del _lifo[i]
            dropped += 1
    return dropped


def release_class(lifetime_class: str) -> int:
    """Drop every bucket whose key carries ``lifetime_class``.

    M17.7: also purges matching entries from the LIFO hot-cache.

    PF.42's step-end hook calls ``release_class("gradient")`` after each
    ``optimizer.zero_grad()`` so the gradient buckets don't carry over
    into the next step's working set. Returns the number of tensors
    dropped (for stats / regression tests).

    No-op when the pool is disabled. Other classes are untouched.
    """
    if _POOL_DISABLED:
        return 0
    global _size_now
    # M17.7: purge LIFO first, then regular buckets.
    lifo_dropped = _release_class_from_lifo(lifetime_class)
    # Materialize keys before mutation — _buckets is the same dict we
    # walk and shrink.
    victims = [k for k in _buckets if k[2] == lifetime_class]
    bucket_dropped = 0
    for k in victims:
        bucket = _buckets.pop(k)
        bucket_dropped += len(bucket)
        _record_pool_event("release_class_drop", key=str(k), count=len(bucket))
    dropped = lifo_dropped + bucket_dropped
    _size_now -= dropped
    if _size_now < 0:
        # Defensive: counter drift should never put us negative. Reset
        # to the actual residual so subsequent stats stay coherent.
        _size_now = sum(len(b) for b in _buckets.values()) + len(_lifo)
    _stats["size_now"] = _size_now
    _release_counts[lifetime_class] = _release_counts.get(lifetime_class, 0) + 1
    return dropped


def release_count_for_class(lifetime_class: str) -> int:
    """Return the number of ``release_class`` calls observed for ``lifetime_class``.

    Counts the *number of release_class invocations* (not the total tensors
    dropped) — so the FX-pass test can verify the PF.42 zero-grad hook fired
    at least once for the gradient class. Returns 0 for unseen classes.
    """
    return _release_counts.get(lifetime_class, 0)


def reset_pool() -> None:
    """Drop every cached tensor and zero the stats. Test hook."""
    global _size_now, _POOL_TRACE_INDEX
    global _POOL_BYTES_ACQUIRED, _POOL_BYTES_RELEASED, _POOL_BYTES_EVICTED
    _buckets.clear()
    _lifo.clear()
    _size_now = 0
    for k in _stats:
        _stats[k] = 0
    for cls in _release_counts:
        _release_counts[cls] = 0
    _POOL_TRACE.clear()
    _POOL_TRACE_INDEX = 0
    _POOL_BYTES_ACQUIRED = 0
    _POOL_BYTES_RELEASED = 0
    _POOL_BYTES_EVICTED = 0
    _POOL_HIT_SIZES.clear()


def pool_stats() -> dict[str, int]:
    """Snapshot of the pool counters. Always returns the full schema."""
    return dict(_stats)


def pool_total_bytes() -> int:
    """Sum of bytes for every tensor currently held in the pool's buckets.

    M17.7: also counts LIFO hot-cache entries.

    Read-only probe used by the T6.4 50-step survival regression test to
    assert a memory plateau across steps 10-50: pool growth at any step
    indicates a lifetime-class leak (the prior step's bucket failed to
    release before this step's acquires). Walks ``_buckets`` and sums
    ``element_size() * numel()`` for each cached tensor — does not
    include device-allocator residency outside the pool (see
    :func:`torch_vulkan.memory_cached` for that).
    """
    total = 0
    for bucket in _buckets.values():
        for t in bucket:
            total += t.element_size() * t.numel()
    for _, t in _lifo:
        total += t.element_size() * t.numel()
    return total


def pool_stats_detailed() -> dict:
    """Return detailed pool statistics when ``TORCH_VULKAN_POOL_STATS=1``.

    Includes per-event trace (ring buffer), byte-level accounting,
    and per-size-class hit histogram. Always returns a dict with the
    same schema — fields are empty/zero when stats are disabled.
    """
    base = pool_stats()
    trace_snapshot = list(_POOL_TRACE) if _POOL_STATS_ENABLED else []
    return {
        **base,
        "stats_enabled": _POOL_STATS_ENABLED,
        "bytes_acquired": _POOL_BYTES_ACQUIRED,
        "bytes_released": _POOL_BYTES_RELEASED,
        "bytes_evicted": _POOL_BYTES_EVICTED,
        "hit_size_histogram": dict(_POOL_HIT_SIZES),
        "trace": trace_snapshot,
        "trace_len": len(trace_snapshot),
    }


def _reset_disabled_cache() -> None:
    """Re-read ``TORCH_VULKAN_BUFFER_POOL`` from the env. Test hook."""
    global _POOL_DISABLED, _PER_KEY_CAP_DEFAULT, _GLOBAL_CAP
    _POOL_DISABLED = os.environ.get("TORCH_VULKAN_BUFFER_POOL", "1") == "0"
    _PER_KEY_CAP_DEFAULT = int(os.environ.get("TORCH_VULKAN_BUFFER_POOL_PER_KEY", "4"))
    _GLOBAL_CAP = int(os.environ.get("TORCH_VULKAN_BUFFER_POOL_SIZE", "64"))
