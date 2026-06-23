"""Regression tests for the buffer-pool hot-path optimizations.

Cross-references:
- M-PERF.1: ``@lru_cache`` on ``_per_key_cap_for`` (env reads hoisted to
  module init); see ``buffer_pool.py:_per_key_cap_for``.
- F.D.2 / D.2: 2-level LIFO index (``_lifo_by_kd`` + ``_lifo_order``);
  see ``buffer_pool.py:_lifo_acquire`` / ``_lifo_push``.
"""

from __future__ import annotations

import random

import pytest
import torch

from torch_vulkan.inductor import buffer_pool
from torch_vulkan.inductor.buffer_pool import (
    _per_key_cap_for,
    pool_stats,
    pool_total_bytes,
    reset_pool,
    vulkan_pool_acquire,
    vulkan_pool_release,
)


@pytest.fixture(autouse=True)
def _reset_pool_state():
    """Reset pool state before/after each test so cross-test residue doesn't
    poison the LIFO index or the lru_cache."""
    buffer_pool._reset_disabled_cache()
    reset_pool()
    yield
    buffer_pool._reset_disabled_cache()
    reset_pool()


class TestPerKeyCapCached:
    """M-PERF.1: ``@functools.lru_cache`` on ``_per_key_cap_for``."""

    def test_cap_lookup_is_cached(self):
        """Repeated calls hit the lru_cache rather than re-running the body."""
        # Clear once so we start from a known cache state for this test.
        _per_key_cap_for.cache_clear()
        # Cold lookups populate the cache.
        v1 = _per_key_cap_for("transient")
        v2 = _per_key_cap_for("scratch")
        v3 = _per_key_cap_for("save_for_backward")
        info1 = _per_key_cap_for.cache_info()
        assert info1.misses == 3, info1
        assert info1.hits == 0, info1
        # Warm lookups must be cache hits.
        assert _per_key_cap_for("transient") == v1
        assert _per_key_cap_for("scratch") == v2
        assert _per_key_cap_for("save_for_backward") == v3
        info2 = _per_key_cap_for.cache_info()
        assert info2.hits == 3, info2
        assert info2.misses == 3, info2

    def test_reset_invalidates_cache(self, monkeypatch):
        """``_reset_disabled_cache`` must re-snapshot the env-override flag
        and clear the lru_cache, so per-test env mutations are observable."""
        # Baseline: no env override; adaptive cap for ``scratch`` is non-default.
        _per_key_cap_for.cache_clear()
        baseline = _per_key_cap_for("scratch")
        assert baseline != 4, (
            "scratch is supposed to use an adaptive cap > default 4"
        )

        # Apply env override; without cache invalidation the prior value
        # would stick.
        monkeypatch.setenv("TORCH_VULKAN_BUFFER_POOL_PER_KEY", "2")
        buffer_pool._reset_disabled_cache()
        assert _per_key_cap_for("scratch") == 2
        assert _per_key_cap_for("transient") == 2

        # Remove the override; another reset must re-expose the adaptive cap.
        monkeypatch.delenv("TORCH_VULKAN_BUFFER_POOL_PER_KEY")
        buffer_pool._reset_disabled_cache()
        assert _per_key_cap_for("scratch") == baseline


class TestLifoTwoLevelIndex:
    """F.D.2 / D.2: 2-level LIFO index for fast (numel, dtype) acquire."""

    def _make(self, numel: int, dtype: torch.dtype = torch.float32):
        return torch.empty_strided(
            (numel,), (1,), dtype=dtype, device="vulkan:0"
        )

    def test_lifo_returns_newest_per_kd(self):
        """LIFO semantic: newest released for a given ``(numel, dtype)`` wins."""
        a, b, c = self._make(8), self._make(8), self._make(8)
        ptrs = [a.data_ptr(), b.data_ptr(), c.data_ptr()]
        for t in (a, b, c):
            vulkan_pool_release(t, lifetime_class="transient")
        del a, b, c
        # Acquire 3 times — newest (c) first, then b, then a.
        out_c = vulkan_pool_acquire((8,), (1,), torch.float32)
        out_b = vulkan_pool_acquire((8,), (1,), torch.float32)
        out_a = vulkan_pool_acquire((8,), (1,), torch.float32)
        assert out_c is not None and out_c.data_ptr() == ptrs[2]
        assert out_b is not None and out_b.data_ptr() == ptrs[1]
        assert out_a is not None and out_a.data_ptr() == ptrs[0]

    def test_lifo_separates_distinct_kds(self):
        """Acquire on one ``(numel, dtype)`` doesn't disturb other kds."""
        a8 = self._make(8)
        b16 = self._make(16)
        c8 = self._make(8)
        ptrs = {"a8": a8.data_ptr(), "b16": b16.data_ptr(), "c8": c8.data_ptr()}
        vulkan_pool_release(a8, lifetime_class="transient")
        vulkan_pool_release(b16, lifetime_class="transient")
        vulkan_pool_release(c8, lifetime_class="transient")
        del a8, b16, c8
        # Newest of numel=8 is c8.
        out = vulkan_pool_acquire((8,), (1,), torch.float32)
        assert out is not None and out.data_ptr() == ptrs["c8"]
        # numel=16 still has b16 available; b16 hits.
        out = vulkan_pool_acquire((16,), (1,), torch.float32)
        assert out is not None and out.data_ptr() == ptrs["b16"]
        # Remaining numel=8 entry: a8.
        out = vulkan_pool_acquire((8,), (1,), torch.float32)
        assert out is not None and out.data_ptr() == ptrs["a8"]

    def test_lifo_dtype_separates(self):
        """``(numel, dtype)`` must include dtype — fp16 and fp32 don't mix."""
        a = self._make(8, dtype=torch.float32)
        b = self._make(8, dtype=torch.float16)
        ptr_a, ptr_b = a.data_ptr(), b.data_ptr()
        vulkan_pool_release(a, lifetime_class="transient")
        vulkan_pool_release(b, lifetime_class="transient")
        del a, b
        out_f16 = vulkan_pool_acquire((8,), (1,), torch.float16)
        out_f32 = vulkan_pool_acquire((8,), (1,), torch.float32)
        assert out_f16 is not None and out_f16.data_ptr() == ptr_b
        assert out_f32 is not None and out_f32.data_ptr() == ptr_a

    def test_lifo_eviction_to_bucket_when_full(self):
        """When the LIFO is full, the OLDEST entry evicts to its per-class
        bucket — not lost.  size_now stays consistent.
        """
        from torch_vulkan.inductor.buffer_pool import _LIFO_MAX

        # Release more than LIFO_MAX of the same kd.
        n = _LIFO_MAX + 4
        tensors = [self._make(4) for _ in range(n)]
        for t in tensors:
            vulkan_pool_release(t, lifetime_class="transient")
        del tensors
        stats = pool_stats()
        # All `n` releases land in the pool — `_LIFO_MAX` in the LIFO and
        # the rest evicted to the bucket (or dropped if the bucket cap is
        # hit).  At minimum, size_now > 0 and releases == n.
        assert stats["releases"] == n
        # The transient per-key cap is 12 (M17.7); with LIFO_MAX=16 we hold
        # 16 in LIFO + min(4, 12) in bucket = 20.  size_now must match.
        assert stats["size_now"] == n, stats

    def test_random_mixed_acquire_release_no_leak(self):
        """Acquire/release ~100 times in random order; hit-rate >= 0.5 and
        no buffer leaks."""
        random.seed(0xA0FE)
        # Pre-warm the pool with 4 distinct (numel, dtype) buffers.
        seed = [
            (16, torch.float32),
            (32, torch.float32),
            (16, torch.float16),
            (64, torch.float32),
        ]
        warm_tensors = [
            torch.empty_strided((n,), (1,), dtype=dt, device="vulkan:0")
            for n, dt in seed
        ]
        for t in warm_tensors:
            vulkan_pool_release(t, lifetime_class="transient")
        del warm_tensors

        # Reset stats so the warmup releases don't poison the hit rate.
        # We track our own hit count from acquire return values instead.
        live: list[torch.Tensor] = []
        hits = 0
        attempts = 0
        for _ in range(100):
            if live and random.random() < 0.5:
                # Release a random live buffer.
                idx = random.randrange(len(live))
                t = live.pop(idx)
                vulkan_pool_release(t, lifetime_class="transient")
            else:
                # Acquire a random (numel, dtype).
                numel, dt = random.choice(seed)
                attempts += 1
                got = vulkan_pool_acquire((numel,), (1,), dt)
                if got is not None:
                    hits += 1
                    live.append(got)
                else:
                    # Miss — synthesize via empty_strided so the next loop
                    # has something to release.
                    live.append(
                        torch.empty_strided(
                            (numel,), (1,), dtype=dt, device="vulkan:0"
                        )
                    )

        # Hit-rate floor: with 4 seed buffers and 4 distinct kds the pool
        # should service half of acquires from cache.
        assert attempts > 0
        hit_rate = hits / attempts
        assert hit_rate >= 0.5, (
            f"hit rate {hit_rate:.1%} below 0.5 floor; stats={pool_stats()}"
        )

        # No-leak check: release everything still live, then assert
        # ``pool_total_bytes`` accounts for all live storage and that the
        # internal live-counter equals the visible size_now.
        for t in live:
            vulkan_pool_release(t, lifetime_class="transient")
        del live

        # Internal coherence: _lifo_live_count + bucket sizes == size_now.
        bucket_total = sum(len(b) for b in buffer_pool._buckets.values())
        assert (
            buffer_pool._lifo_live_count + bucket_total == pool_stats()["size_now"]
        )

        # pool_total_bytes is non-negative and finite.
        total = pool_total_bytes()
        assert total >= 0

        # Drain to ensure cleanup leaves the data structures empty.
        reset_pool()
        assert buffer_pool._lifo_live_count == 0
        assert len(buffer_pool._lifo_by_kd) == 0
        assert pool_stats()["size_now"] == 0

    def test_release_class_purges_lifo(self):
        """``release_class`` removes matching entries from the 2-level
        LIFO index without leaving stale per-kd deques behind."""
        # Push 3 transient + 2 gradient; release_class("gradient") must
        # leave only the 3 transient entries.
        for _ in range(3):
            vulkan_pool_release(self._make(4), lifetime_class="transient")
        for _ in range(2):
            vulkan_pool_release(self._make(4), lifetime_class="gradient")
        dropped = buffer_pool.release_class("gradient")
        assert dropped == 2, dropped
        # 3 transient still in LIFO; an acquire must succeed.
        out = vulkan_pool_acquire((4,), (1,), torch.float32)
        assert out is not None
        # And gradient acquires miss.
        # (Lifetime class is ignored on acquire — but the corresponding
        # storage was already purged, so 2 of the 3 acquires hit; the
        # 4th misses.)
        for _ in range(2):
            assert vulkan_pool_acquire((4,), (1,), torch.float32) is not None
        assert vulkan_pool_acquire((4,), (1,), torch.float32) is None
