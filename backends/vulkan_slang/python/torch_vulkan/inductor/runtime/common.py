"""Shared infrastructure for the Vulkan/Slang Inductor runtime.

This module holds the module-level state that is imported by all other
runtime submodules (``slangc``, ``dispatch``, ``reflection``, etc.).
Extracted from ``slangc.py`` as part of the M22a anti-goal #7 line-cap
split (Stage 1).

Concretely this file owns:
  - slangc binary resolver and ``_SLANGC`` singleton
  - Slang-source normalizer regexes + ``_normalize_slang_source()``
  - Device subgroup-size tag helper
  - Async/parallel compile flags (env-var driven)
  - In-memory SPIR-V caches (``_cache_by_key``, ``_cache_by_hash``)
  - Reflection metrics caches
  - SPV baseline caches
  - In-flight dedup set and lock
  - Cache lock
  - Compile-time stats counters + helpers
  - On-disk cache directory constant
  - Pool re-entrancy guard (``_pool_local``, ``_is_in_pool_worker``)
  - Thread pool getter ``_get_async_pool``
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional


# ---------------------------------------------------------------------------
# slangc binary resolver
# ---------------------------------------------------------------------------


def _resolve_slangc(raw: str) -> str:
    """PF.53 — resolve a possibly-relative ``SLANGC`` against well-known
    roots.

    The CLAUDE.md build/test recipe documents ``SLANGC=third_party/
    slang/build/.../slangc`` as a relative path; pytest's rootdir is
    ``backends/vulkan_slang/`` while the user's shell pwd is the repo
    root. Without this resolver, ``subprocess.run([raw, "--version"])``
    on a relative ``raw`` raises ``FileNotFoundError`` whenever cwd
    differs from either root, sending every dispatch through the
    "slangc not found" error path even though the binary exists.
    """
    if not raw:
        return "slangc"
    # Already absolute and exists → done.
    if os.path.isabs(raw) and os.path.exists(raw):
        return raw
    # Already on PATH → done.
    import shutil

    which = shutil.which(raw)
    if which:
        return which
    # Resolve relative to backend root (common.py is at
    # backends/vulkan_slang/python/torch_vulkan/inductor/runtime/common.py
    # → 4 dirname() steps up = backend root).
    backend_root = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
        )
    )
    # …repo root sits one more level up: 5 dirnames from this file.
    repo_root = os.path.normpath(os.path.join(backend_root, "..", ".."))
    candidates = [
        os.path.join(backend_root, raw),
        os.path.join(os.getcwd(), raw),
    ]
    # Auto-detect slangc.  Search BOTH ``third_party/slang/build/`` roots
    # (repo-root and backend-root) and prefer the newest version by parsed
    # ``slang-MAJOR.MINOR.PATCH-…`` tag.

    def _ver_key(path: str) -> tuple[int, ...]:
        m = re.search(r"slang-(\d+)\.(\d+)\.(\d+)", path)
        return tuple(int(g) for g in m.groups()) if m else (0, 0, 0)

    third_party_roots = [
        os.path.join(repo_root, "third_party", "slang", "build"),
        os.path.join(backend_root, "third_party", "slang", "build"),
    ]
    # Auto-detect from third_party only when the caller wants the generic
    # "slangc" binary — not when they named a specific (non-existent) binary.
    if raw == "slangc":
        seen: list[str] = []
        for tpr in third_party_roots:
            if not os.path.isdir(tpr):
                continue
            for entry in os.listdir(tpr):
                candidate = os.path.join(tpr, entry, "bin", "slangc")
                if os.path.isfile(candidate):
                    seen.append(candidate)
        if seen:
            seen.sort(key=_ver_key, reverse=True)
            candidates.append(seen[0])
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    # Fall through with the raw value; let `_slangc_available()` probe
    # report the failure with the original error.
    return raw


_SLANGC = _resolve_slangc(os.environ.get("SLANGC", "slangc"))


def _get_slangc() -> str:
    """Return the current slangc path, re-reading SLANGC from the env var.

    This ensures the path is fresh even when the env var is set after
    module import time (e.g. via conftest.py's pytest_configure).
    """
    env_val = os.environ.get("SLANGC")
    if env_val:
        resolved = _resolve_slangc(env_val)
        if resolved != "slangc":
            return resolved
    return _SLANGC


# PF.18: hard timeout on every `slangc` subprocess invocation so a hung
# slangc never wedges the agent's test loop indefinitely.
# Default 30 s per invocation; override via `TORCH_VULKAN_SLANGC_TIMEOUT_S`.
_SLANGC_TIMEOUT_S: float = float(os.environ.get("TORCH_VULKAN_SLANGC_TIMEOUT_S", "30"))


class SlangCompileTimeout(RuntimeError):
    """Raised when ``slangc`` exceeds ``TORCH_VULKAN_SLANGC_TIMEOUT_S``.

    Exposes the cache key (8-char prefix), invocation argv, and any
    stdout / stderr captured before the kill so the test surface fails
    fast with actionable context instead of hanging at pytest's
    signal-driven timeout.
    """

    def __init__(
        self,
        key: str,
        argv: list[str],
        partial_stdout: str = "",
        partial_stderr: str = "",
        timeout_s: float = _SLANGC_TIMEOUT_S,
    ) -> None:
        msg = (
            f"slangc timed out after {timeout_s:.1f}s for kernel {key[:8]}\n"
            f"  argv: {' '.join(argv)}\n"
            f"  partial stdout: {partial_stdout[:200]!r}\n"
            f"  partial stderr: {partial_stderr[:200]!r}"
        )
        super().__init__(msg)
        self.key = key
        self.argv = argv
        self.partial_stdout = partial_stdout
        self.partial_stderr = partial_stderr
        self.timeout_s = timeout_s


# ---------------------------------------------------------------------------
# Slang-source normalizer
# ---------------------------------------------------------------------------

# Slang-source minification used as the SPIR-V cache key. Two semantically
# identical sources that differ only in comments, blank lines, or trailing
# whitespace should map to the same hash.
_SLANG_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_SLANG_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_SLANG_TRAILING_WS_RE = re.compile(r"[ \t]+\n")
_SLANG_BLANK_LINES_RE = re.compile(r"\n{2,}")


def _normalize_slang_source(src: str) -> str:
    """Strip comments, blank-line runs, and trailing whitespace.

    Used as the cache-key normalizer so cosmetic codegen variation doesn't
    miss the SPIR-V cache.
    """
    src = _SLANG_BLOCK_COMMENT_RE.sub("", src)
    src = _SLANG_LINE_COMMENT_RE.sub("", src)
    src = _SLANG_TRAILING_WS_RE.sub("\n", src)
    src = _SLANG_BLANK_LINES_RE.sub("\n", src)
    return src.strip()


# ---------------------------------------------------------------------------
# Device subgroup-size tag
# ---------------------------------------------------------------------------

# N+1.12: Cached device subgroup-size tag for SPIR-V cache-key mixing.
_device_subgroup_size_tag: Optional[str] = None


def _get_device_subgroup_size_tag() -> str:
    """Return a stable tag like ``_sgs64`` for the current device.

    Fetched once and cached. Falls back to ``_sgs64`` if device
    properties are unavailable.
    """
    global _device_subgroup_size_tag
    if _device_subgroup_size_tag is not None:
        return _device_subgroup_size_tag
    try:
        from torch._dynamo.device_interface import get_interface_for_device

        iface = get_interface_for_device("vulkan")
        props = iface.Worker.get_device_properties()
        _device_subgroup_size_tag = f"_sgs{props.subgroup_size}"
    except Exception:
        _device_subgroup_size_tag = "_sgs64"
    return _device_subgroup_size_tag


# ---------------------------------------------------------------------------
# Async/parallel compile flags
# ---------------------------------------------------------------------------

_INDUCTOR_STATS = os.environ.get("TORCH_VULKAN_INDUCTOR_STATS") == "1"
# P8.13: default-on. Opt out with ``TORCH_VULKAN_ASYNC_COMPILE=0``.
_ASYNC_COMPILE = os.environ.get("TORCH_VULKAN_ASYNC_COMPILE", "1") != "0"
# M21: controls parallel slangc compilation across all paths.
_PARALLEL_COMPILE = os.environ.get("TORCH_VULKAN_PARALLEL_COMPILE", "1") != "0"
_TRACE = os.environ.get("TORCH_VULKAN_TRACE") == "1"
_ASYNC_POOL: Optional[ThreadPoolExecutor] = None


def _default_max_workers() -> int:
    """slangc workers default.

    **M22.16:** capped at ``min(2, cpu_count)`` to bound the worst-case
    cross-process slangc contention. Override via
    ``TORCH_VULKAN_SLANGC_WORKERS=<n>``.
    """
    override = os.environ.get("TORCH_VULKAN_SLANGC_WORKERS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    return max(1, min(2, (os.cpu_count() or 1)))


_ASYNC_MAX_WORKERS = _default_max_workers()
# N+1.6: public name for the max slangc workers knob.
_MAX_SLANGC_WORKERS = _ASYNC_MAX_WORKERS


# ---------------------------------------------------------------------------
# In-memory SPIR-V caches
# ---------------------------------------------------------------------------

# SPIR-V cache, keyed by the stable cache_key the generated wrapper owns.
# Falls back to SHA256(src) only for callers that don't supply a key.
_cache_by_key: dict[str, bytes] = {}
_cache_by_hash: dict[str, bytes] = {}

# P3.3/M13: Reflection metrics cache — per-hash-key dict of parsed metrics.
_reflection_metrics_by_hash: dict[str, dict] = {}
_reflection_metrics_by_key: dict[str, dict] = {}

# M6: SPIR-V perf regression baseline. Persisted to disk.
_spv_baselines: dict[str, dict] = {}
_spv_baselines_loaded: bool = False

# N+1.6: in-flight dedup — set of hash_keys currently being compiled.
_in_flight: dict[str, threading.Event] = {}
_in_flight_lock = threading.Lock()

# SP.1: carries compilation exceptions from pool workers back to waiters.
_compile_exceptions: dict[str, Exception] = {}
_compile_exceptions_lock = threading.Lock()

# TRAIN.8 / M21: guards _cache_by_key and _cache_by_hash for thread-safety.
_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Compile-time stats counters
# ---------------------------------------------------------------------------

# Always-on (cheap counter increments) so ``inductor_stats.compile_stats()``
# can answer "where did the cold-compile time go?" without an env var.
_COMPILE_STATS: dict[str, float] = {
    "cold_compiles": 0,
    "cold_compile_us": 0.0,
    "max_cold_compile_us": 0.0,
    "in_memory_hits": 0,
    "disk_cache_hits": 0,
    "prewarm_submits": 0,
}


def reset_compile_stats() -> None:
    for k in _COMPILE_STATS:
        _COMPILE_STATS[k] = (
            0.0 if k in ("cold_compile_us", "max_cold_compile_us") else 0
        )


def cache_hit_rate() -> float:
    """Ratio of compile requests that hit cache vs total.

    P5.6 — ``0.9`` is the post-warmup discipline floor.
    """
    hits = _COMPILE_STATS["in_memory_hits"] + _COMPILE_STATS["disk_cache_hits"]
    total = hits + _COMPILE_STATS["cold_compiles"]
    return float(hits) / float(total) if total > 0 else 0.0


# ---------------------------------------------------------------------------
# On-disk cache directory
# ---------------------------------------------------------------------------

# On-disk cache directory — lets subsequent Python sessions skip slangc
# entirely on kernels we've already compiled.
_DISK_CACHE_DIR = os.environ.get(
    "TORCH_VULKAN_SPIRV_CACHE",
    os.path.join(os.path.expanduser("~"), ".cache", "torch_vulkan", "spirv"),
)


def _get_disk_cache_dir() -> str:
    """Return _DISK_CACHE_DIR, resolving via the package for monkeypatch
    compatibility.

    Tests monkeypatch ``torch_vulkan.inductor.runtime._DISK_CACHE_DIR``;
    since this module is a sub-package, the patch lands on ``__init__.py``.
    This getter checks there first, falling back to the module-level default.
    """
    import sys

    pkg = sys.modules.get("torch_vulkan.inductor.runtime")
    if pkg is not None:
        val = getattr(pkg, "_DISK_CACHE_DIR", None)
        if val is not None:
            return val
    return _DISK_CACHE_DIR


# ---------------------------------------------------------------------------
# Pool re-entrancy guard
# ---------------------------------------------------------------------------

# PF.14: re-entrant submission guard. The thread-local ``_in_pool_worker``
# flag is set on entry to a worker callable; ``compile_slang_to_spirv``
# checks it and bypasses the pool, calling the inner compile directly.
_pool_local = threading.local()


def _is_in_pool_worker() -> bool:
    return getattr(_pool_local, "in_pool_worker", False)


def _wrap_pool_worker(fn):
    """Wrap a callable submitted to ``_ASYNC_POOL`` so the thread-local
    ``in_pool_worker`` flag is set for the duration of the call."""

    def _runner(*args, **kwargs):
        already = getattr(_pool_local, "in_pool_worker", False)
        _pool_local.in_pool_worker = True
        try:
            return fn(*args, **kwargs)
        finally:
            _pool_local.in_pool_worker = already

    return _runner


def _get_async_pool() -> ThreadPoolExecutor:
    global _ASYNC_POOL
    if _ASYNC_POOL is None:
        _ASYNC_POOL = ThreadPoolExecutor(max_workers=_ASYNC_MAX_WORKERS)
        import atexit
        atexit.register(_ASYNC_POOL.shutdown, wait=False)
    return _ASYNC_POOL
