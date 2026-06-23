"""Pre-recorded command-buffer (Vulkan-graph) replay — P4.7 stub.

Wraps a callable for static-shape replay. The first call hashes the input
shapes/strides/dtypes, compiles the function via ``torch.compile`` with
``dynamic=False`` so the entire graph specializes, and records the resulting
command-buffer execution path. Subsequent calls with matching guard hash
re-submit the already-compiled graph; on guard miss, the call falls back to
re-compiling and replacing the cached entry.

This stub does **not** yet pre-record a primary Vulkan command buffer — that
requires a C++ ``RecordingAllocator`` plus a re-bindable command-buffer
submission path (next iteration). What it does ship:

- A stable shape-guard hashing scheme: ``(shape, stride, dtype, device)`` per
  input tensor, hashed into a single key.
- A guard cache keyed by that hash, mapping to the compiled callable.
- An ``info()`` method exposing the guard cache so users can inspect the
  hit/miss counters and decide if they want to opt out of static replay.

Future work: replace the per-call ``compiled_fn(*args)`` with the real
command-buffer replay (described in roadmap P4.7).
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable

import torch


class _CompiledGraph:
    """A static-shape-replay wrapper around a compiled function."""

    __slots__ = ("_fn", "_cache", "_hits", "_misses", "_re_records")

    def __init__(self, fn: Callable):
        self._fn = fn
        # Keyed on the tuple guard key directly (dict accepts tuple keys);
        # P6.4 — skips a per-call SHA1 + hexdigest compared to the prior
        # `_guard_hash` string-key path.
        self._cache: dict[tuple, Callable] = {}
        self._hits = 0
        self._misses = 0
        self._re_records = 0

    def __call__(self, *args, **kwargs):
        key = self._guard_key(args, kwargs)
        compiled = self._cache.get(key)
        if compiled is None:
            self._misses += 1
            compiled = torch.compile(self._fn, backend="inductor", dynamic=False)
            self._cache[key] = compiled
        else:
            self._hits += 1
        return compiled(*args, **kwargs)

    @staticmethod
    def _guard_key(args, kwargs) -> tuple:
        """Build a hashable shape/dtype/device key without hashing.

        Dict already hashes tuple keys via the per-element hash; doing a
        SHA1 over a string-encoded form on top added pure overhead.
        """
        arg_keys = []
        for a in args:
            if isinstance(a, torch.Tensor):
                arg_keys.append((
                    "T", tuple(a.shape), tuple(a.stride()), a.dtype, str(a.device),
                ))
            else:
                try:
                    arg_keys.append((type(a).__name__, a))
                except TypeError:
                    arg_keys.append((type(a).__name__, repr(a)))
        kw_keys = []
        for k in sorted(kwargs):
            v = kwargs[k]
            if isinstance(v, torch.Tensor):
                kw_keys.append((k, tuple(v.shape), v.dtype))
            else:
                try:
                    kw_keys.append((k, v))
                except TypeError:
                    kw_keys.append((k, repr(v)))
        return (tuple(arg_keys), tuple(kw_keys))

    @staticmethod
    def _guard_hash(args, kwargs) -> str:
        """Legacy SHA1 16-hex-char digest. Retained for tests / external
        callers that want a string key. The fast path no longer uses it."""
        h = hashlib.sha1()
        h.update(repr(_CompiledGraph._guard_key(args, kwargs)).encode())
        return h.hexdigest()[:16]

    def info(self) -> dict[str, Any]:
        """Return cache hit/miss counters and the number of recorded shapes."""
        return {
            "n_recordings": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (
                self._hits / (self._hits + self._misses)
                if (self._hits + self._misses) else 0.0
            ),
        }

    def reset_cache(self) -> int:
        """Drop all recorded shapes. Returns the count freed."""
        n = len(self._cache)
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        return n


def compile_graph(fn: Callable) -> _CompiledGraph:
    """Wrap ``fn`` for static-shape replay.

    Each unique input shape signature gets one ``torch.compile(dynamic=False)``
    invocation, then is replayed on every subsequent call with that signature.
    Shape-changing workloads will see a re-compile per shape — call
    ``.info()`` on the returned object to check hit rate.

    P4.7 — stub. The full Vulkan command-buffer pre-recording is the next
    item; today this just memoizes ``torch.compile`` per shape signature.
    """
    return _CompiledGraph(fn)
