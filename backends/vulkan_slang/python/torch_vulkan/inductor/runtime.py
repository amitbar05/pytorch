"""Runtime shim between Inductor-generated Slang source and Vulkan dispatch.

Compiles Slang source to SPIR-V via `slangc` subprocess, caches by source
hash, then dispatches the compute shader via the C++ `_jit_dispatch` pybind
entry point. The in-memory cache is augmented by an on-disk cache
(``~/.cache/torch_vulkan/spirv/``, overridable via ``TORCH_VULKAN_SPIRV_CACHE``)
so subsequent Python sessions bypass slangc entirely once a kernel has been
compiled.
"""

import hashlib
import os
import re
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch


def _resolve_slangc(raw: str) -> str:
    """PF.53 — resolve a possibly-relative ``SLANGC`` against well-known
    roots.

    The CLAUDE.md build/test recipe documents ``SLANGC=third_party/
    slang/build/.../slangc`` as a relative path; pytest's rootdir is
    ``backends/vulkan_slang/`` while the user's shell pwd is the repo
    root. Without this resolver, ``subprocess.run([raw, "--version"])``
    on a relative ``raw`` raises ``FileNotFoundError`` whenever cwd
    differs from either root, sending every dispatch through the
    "slangc not found" error path even though the binary exists. Test
    classes already side-step the bug by resolving against ``__file__``;
    this brings runtime.py to parity.
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
    # Resolve relative to backend root (runtime.py is at
    # backends/vulkan_slang/python/torch_vulkan/inductor/runtime.py
    # → 4 dirname() steps up = backend root).
    backend_root = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
        )
    )
    candidates = [
        os.path.join(backend_root, raw),
        os.path.join(os.getcwd(), raw),
    ]
    # Auto-detect slangc in third_party/slang/build/
    third_party_root = os.path.join(backend_root, "third_party", "slang", "build")
    if os.path.isdir(third_party_root):
        for entry in sorted(os.listdir(third_party_root), reverse=True):
            candidate = os.path.join(third_party_root, entry, "bin", "slangc")
            if os.path.isfile(candidate):
                candidates.append(candidate)
                break
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    # Fall through with the raw value; let `_slangc_available()` probe
    # report the failure with the original error.
    return raw


_SLANGC = _resolve_slangc(os.environ.get("SLANGC", "slangc"))

# PF.18: hard timeout on every `slangc` subprocess invocation so a hung
# slangc (rare, but possible — deeply nested generic specializations have
# been observed to thrash) never wedges the agent's test loop indefinitely.
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


# Slang-source minification used as the SPIR-V cache key. Two semantically
# identical sources that differ only in comments, blank lines, or trailing
# whitespace should map to the same hash, so equivalent codegen runs reuse
# the SPIR-V cache instead of re-compiling.
_SLANG_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_SLANG_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_SLANG_TRAILING_WS_RE = re.compile(r"[ \t]+\n")
_SLANG_BLANK_LINES_RE = re.compile(r"\n{2,}")


def _normalize_slang_source(src: str) -> str:
    """Strip comments, blank-line runs, and trailing whitespace.

    Used as the cache-key normalizer so cosmetic codegen variation doesn't
    miss the SPIR-V cache. Conservative — does not collapse intra-line
    whitespace because Slang preprocessor directives (`#include`, `#if`)
    are leading-whitespace-sensitive in some builds. Behavior preserves
    the source's lexical structure 1:1; only ignorable tokens drop.
    """
    src = _SLANG_BLOCK_COMMENT_RE.sub("", src)
    src = _SLANG_LINE_COMMENT_RE.sub("", src)
    src = _SLANG_TRAILING_WS_RE.sub("\n", src)
    src = _SLANG_BLANK_LINES_RE.sub("\n", src)
    return src.strip()


# N+1.12: Cached device subgroup-size tag for SPIR-V cache-key mixing.
# Different subgroup sizes can produce different SPIR-V from the same
# Slang source (barrier code, wave intrinsics), so the cache must
# distinguish wave32 from wave64 entries.
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


_INDUCTOR_STATS = os.environ.get("TORCH_VULKAN_INDUCTOR_STATS") == "1"
# P8.13: default-on. Opt out with ``TORCH_VULKAN_ASYNC_COMPILE=0`` if a
# user hits a slangc-pool bug; the long-term plan removes the env knob
# entirely once the pool is bulletproof.
_ASYNC_COMPILE = os.environ.get("TORCH_VULKAN_ASYNC_COMPILE", "1") != "0"
# M21: TORCH_VULKAN_PARALLEL_COMPILE controls parallel slangc compilation
# across all paths (single-compile submit + batch_compile). Default 1;
# set to 0 to force serial slangc for debugging / minimum-footprint runs.
_PARALLEL_COMPILE = os.environ.get("TORCH_VULKAN_PARALLEL_COMPILE", "1") != "0"
_TRACE = os.environ.get("TORCH_VULKAN_TRACE") == "1"
_ASYNC_POOL: Optional[ThreadPoolExecutor] = None


def _default_max_workers() -> int:
    """slangc workers default. Cold-compile of a 50-kernel graph spends
    most of its wall time in slangc subprocesses; the default cap of 4 (M21)
    keeps per-core load manageable while avoiding slangc thrash. M21."""
    override = os.environ.get("TORCH_VULKAN_SLANGC_WORKERS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    return max(1, min(4, (os.cpu_count() or 1)))


_ASYNC_MAX_WORKERS = _default_max_workers()
# N+1.6: public name for the max slangc workers knob. Alias of
# _ASYNC_MAX_WORKERS so callers have a stable, self-documenting handle.
_MAX_SLANGC_WORKERS = _ASYNC_MAX_WORKERS

# Per-kernel stats dict, populated only when _INDUCTOR_STATS is True.
# Key: kernel name (cache key). Value: dict with call_count, total_us, shapes.
_KERNEL_STATS: dict[str, dict] = {}
# Maps each Inductor-generated kernel cache_key to the 12-char prefix of its
# SPIR-V SHA256. Populated whenever a kernel is built; lets observability
# tooling (e.g. `inductor_stats.summary()`) correlate per-kernel timing back
# to specific compiled binaries when debugging cache-miss / autotune churn.
_KERNEL_SPIRV_HASH: dict[str, str] = {}
# SPIR-V cache, keyed by the stable cache_key the generated wrapper already owns.
# Falls back to SHA256(src) only for callers that don't supply a key.
_cache_by_key: dict[str, bytes] = {}
_cache_by_hash: dict[str, bytes] = {}

# P3.3/M13: Reflection metrics cache — per-hash-key dict of parsed metrics
# (vgprs, shared_mem, subgroup_size, loop_depth). Populated lazily after
# successful SPIR-V compilation when TORCH_VULKAN_REFLECTION=1.
_reflection_metrics_by_hash: dict[str, dict] = {}
_reflection_metrics_by_key: dict[str, dict] = {}

# M6: SPIR-V perf regression baseline. Persisted to disk at
# $HOME/.cache/torch_vulkan/spv_baselines.json. Maps hash_key[:12] → baseline dict.
_spv_baselines: dict[str, dict] = {}
_spv_baselines_loaded: bool = False

# N+1.6: in-flight dedup — set of hash_keys currently being compiled by a
# worker thread. Subsequent requests for the same key block on a per-key
# threading.Event instead of launching a duplicate slangc subprocess.
_in_flight: dict[str, threading.Event] = {}
_in_flight_lock = threading.Lock()

# TRAIN.8 / M21: guards _cache_by_key and _cache_by_hash for thread-safety
# when parallel compilation is enabled.  In CPython the GIL makes dict
# ops atomic, but this lock future-proofs for free-threaded Python.
_cache_lock = threading.Lock()

# Compile-time profiler counters. Always-on (cheap counter increments) so the
# `inductor_stats.compile_stats()` API can answer "where did the cold-compile
# time go?" without asking the user to set an env var first.
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


def reset_per_test_caches() -> None:
    """GAP 7.3 / PF.27.a.2 — clear in-memory caches that leak across tests.

    Called by the ``conftest.py`` per-test fixture so that dispatch-count
    and cold-compile-budget tests see deterministic cache state regardless
    of test ordering within a single xdist worker. Does *not* clear the
    on-disk SPIR-V cache (that's intentionally persistent across sessions).

    PF.27.a.2 extension (2026-05-02): also resets shader-lib module
    readiness, reflection cache, and async pool — globals that previously
    leaked worker-id-dependent state across tests within an xdist worker.
    """
    global _slangc_available_cache, _shader_lib_modules_ready
    _cache_by_key.clear()
    _cache_by_hash.clear()
    _KERNEL_SPIRV_HASH.clear()
    _reflection_cache.clear()
    _SHADER_LIB_MODULE_STATS["compiles"] = 0
    _SHADER_LIB_MODULE_STATS["cache_hits"] = 0
    reset_compile_stats()
    _slangc_available_cache = None
    _shader_lib_modules_ready = False
    _DISPATCH_TIMES.clear()


def cache_hit_rate() -> float:
    """Ratio of compile requests that hit cache (in-memory + disk) vs
    total slangc-bound compiles. P5.6 — `0.9` is the post-warmup
    discipline floor; anything below means cache-key churn.
    """
    hits = _COMPILE_STATS["in_memory_hits"] + _COMPILE_STATS["disk_cache_hits"]
    total = hits + _COMPILE_STATS["cold_compiles"]
    return float(hits) / float(total) if total > 0 else 0.0


# On-disk cache directory — lets subsequent Python sessions skip slangc
# entirely on kernels we've already compiled. `slangc` is ~100ms cold per
# kernel so this is worth ~1-5s on startup of a typical compiled module.
_DISK_CACHE_DIR = os.environ.get(
    "TORCH_VULKAN_SPIRV_CACHE",
    os.path.join(os.path.expanduser("~"), ".cache", "torch_vulkan", "spirv"),
)


def _wrap_stats(key: str, inner):
    """Wrap a kernel callable to collect per-kernel timing + call count.

    The stats entry is looked up by key on every call rather than captured
    at wrap-time so that `reset_stats()` (which clears `_KERNEL_STATS`)
    correctly repopulates entries for already-wrapped kernels — otherwise
    cached kernels would write to a dangling dict that's no longer in
    `_KERNEL_STATS` and `get_stats()` would always report empty.
    """

    def stats_kernel(*args):
        entry = _KERNEL_STATS.get(key)
        if entry is None:
            entry = {"call_count": 0, "total_us": 0.0, "last_args_len": 0}
            _KERNEL_STATS[key] = entry
        t0 = time.perf_counter()
        inner(*args)
        entry["total_us"] += (time.perf_counter() - t0) * 1e6
        entry["call_count"] += 1
        entry["last_args_len"] = len(args)

    return stats_kernel


def _disk_cache_read(hash_key: str) -> Optional[bytes]:
    path = os.path.join(_DISK_CACHE_DIR, hash_key[:2], hash_key[2:] + ".spv")
    try:
        with open(path, "rb") as f:
            return f.read()
    except (FileNotFoundError, OSError):
        return None


def _disk_cache_write(hash_key: str, spv: bytes) -> None:
    path = os.path.join(_DISK_CACHE_DIR, hash_key[:2], hash_key[2:] + ".spv")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Atomic write: rename from a pid-suffixed temp so we never leave a
        # truncated file if we're killed mid-write.
        tmp = path + f".tmp.{os.getpid()}"
        with open(tmp, "wb") as f:
            f.write(spv)
        os.replace(tmp, path)
    except OSError:
        pass  # cache is best-effort; fall back to recompiling next time


# Pre-resolved pybind entries — avoids a package import on every dispatch.
_jit_dispatch = None
_jit_dispatch_cached = None
_jit_dispatch_cached_nopc = None
_jit_dispatch_indexed = None
_jit_pipeline = None
_descriptor_indexing_probe: Optional[bool] = None


def _get_jit_dispatch():
    global _jit_dispatch
    if _jit_dispatch is None:
        from torch_vulkan import _C as _c

        _jit_dispatch = _c._jit_dispatch
    return _jit_dispatch


def _get_jit_dispatch_indexed():
    """Resolve the descriptor-array variant of `_jit_dispatch` (N+1.5).

    Returns the C++ pybind entry that accepts a per-binding
    ``descriptor_counts: vector<uint32_t>``. Auto-falls-back to the flat
    path inside the C++ runtime when every count == 1, so callers may use
    it unconditionally — but Python codegen still prefers the cheaper
    flat ``_jit_dispatch`` when all counts are 1, to skip the extra
    pybind conversion on the hot path.
    """
    global _jit_dispatch_indexed
    if _jit_dispatch_indexed is None:
        from torch_vulkan import _C as _c

        _jit_dispatch_indexed = getattr(_c, "_jit_dispatch_indexed", None)
    return _jit_dispatch_indexed


def _descriptor_indexing_supported() -> bool:
    """Cached probe of `VK_EXT_descriptor_indexing` availability.

    Returns ``True`` when the C++ runtime reports the extension is
    enabled and the FFI shim is present. False on older builds (no
    ``_descriptor_indexing_enabled`` symbol) or when the device driver
    rejected the extension.
    """
    global _descriptor_indexing_probe
    if _descriptor_indexing_probe is not None:
        return _descriptor_indexing_probe
    try:
        from torch_vulkan import _C as _c

        probe = getattr(_c, "_descriptor_indexing_enabled", None)
        _descriptor_indexing_probe = bool(probe()) if probe is not None else False
    except Exception:
        _descriptor_indexing_probe = False
    return _descriptor_indexing_probe


def _get_jit_dispatch_cached():
    """Resolve the cached-pipeline fast-path dispatches (no key lookup per call).

    Returns (dispatch_with_pc, dispatch_no_pc, get_pipeline). Generated
    kernels pick the no-pc entry when n_pc=0 (the common pointwise case),
    avoiding a pybind bytes conversion on every dispatch.
    """
    global _jit_dispatch_cached, _jit_dispatch_cached_nopc, _jit_pipeline
    if _jit_dispatch_cached is None:
        from torch_vulkan import _C as _c

        _jit_dispatch_cached = _c._jit_dispatch_cached
        _jit_dispatch_cached_nopc = _c._jit_dispatch_cached_nopc
        _jit_pipeline = _c._jit_pipeline
    return _jit_dispatch_cached, _jit_dispatch_cached_nopc, _jit_pipeline


# PF.14: re-entrant submission guard. The slangc thread pool dispatches
# `_safe_compile`; each worker calls `compile_slang_to_spirv`, which used
# to re-submit to the same pool and `.result()`-block — classic re-entrant
# `ThreadPoolExecutor` deadlock when all workers are themselves blocked on
# the inner submit. The thread-local `_in_pool_worker` flag is set on entry
# to a worker callable; `compile_slang_to_spirv` checks it and bypasses the
# pool, calling `_compile_slang_to_spirv_inner` directly. Net effect: the
# pool is used for *initial* dispatch only; nested compiles on workers go
# straight through.
_pool_local = threading.local()


def _is_in_pool_worker() -> bool:
    return getattr(_pool_local, "in_pool_worker", False)


def _wrap_pool_worker(fn):
    """Wrap a callable submitted to ``_ASYNC_POOL`` so the thread-local
    ``in_pool_worker`` flag is set for the duration of the call. Idempotent
    if already set (nested wraps are a no-op)."""

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
    return _ASYNC_POOL


def prewarm_compile(specs: list[tuple[str, str]], *, sync: bool = False) -> int:
    """Submit (cache_key, slang_src) pairs to the slangc thread pool.

    Returns the number of specs actually scheduled (already-cached entries are
    skipped). When ``sync=True``, blocks until every submitted compilation
    finishes — used by tests and by callers that want the SPIR-V cache fully
    populated before they continue. Otherwise returns immediately and the
    cache is populated by background workers.

    Pre-warm is best-effort: individual slangc failures (e.g. a buggy tile
    config that won't compile) are swallowed so they don't poison the rest
    of the cache. The user-facing dispatch path will surface any error
    cleanly when that specific kernel is actually requested.
    """
    pending: list[tuple[str, str]] = []
    for key, src in specs:
        if key in _cache_by_key:
            continue
        hash_key = hashlib.sha256(
            ("computeMain\n" + _normalize_slang_source(src)).encode()
        ).hexdigest()
        if hash_key in _cache_by_hash:
            _cache_by_key[key] = _cache_by_hash[hash_key]
            m = _reflection_metrics_by_hash.get(hash_key)
            if m is not None:
                _reflection_metrics_by_key[key] = m
            continue
        spv = _disk_cache_read(hash_key)
        if spv is not None:
            _cache_by_hash[hash_key] = spv
            _cache_by_key[key] = spv
            continue
        pending.append((key, src))

    if not pending:
        return 0

    def _safe_compile(key: str, src: str) -> None:
        try:
            compile_slang_to_spirv(src, "computeMain", cache_key=key)
        except Exception:
            pass

    pool = _get_async_pool()
    wrapped = _wrap_pool_worker(_safe_compile)
    futures = [pool.submit(wrapped, key, src) for key, src in pending]
    _COMPILE_STATS["prewarm_submits"] += len(pending)
    if sync:
        for f in futures:
            f.result()
    return len(pending)


_slangc_available_cache: Optional[bool] = None


def _slangc_available() -> bool:
    """Probe whether `slangc` resolves on PATH (or via SLANGC=).

    Cached on first call. The probe is a `subprocess.run([slangc, --version])`
    which costs ~1 ms; running it on every cold compile multiplies that by N
    kernels per graph for no benefit. Tests that swap `SLANGC=` mid-process
    can call `_reset_slangc_available_cache()` to re-probe.
    """
    global _slangc_available_cache
    if _slangc_available_cache is not None:
        return _slangc_available_cache
    try:
        subprocess.run(
            [_SLANGC, "--version"],
            capture_output=True,
            check=False,
            timeout=5,
        )
        _slangc_available_cache = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        _slangc_available_cache = False
    return _slangc_available_cache


def _reset_slangc_available_cache() -> None:
    """Force `_slangc_available()` to re-probe on its next call."""
    global _slangc_available_cache
    _slangc_available_cache = None


def _slangc_fingerprint() -> str:
    """Stable fingerprint of the slangc binary, mixed into the
    `.slang-module` cache key so a slangc upgrade or rebuild forces
    regen of every cached module.

    PF.27.a.1: slangc 2026.5.2 SIGSEGVs (rc=-11, "invalid RIFF" / silent)
    on `import X;` when the cached `.slang-module` was serialized by a
    different slangc build. Pre-fix, `precompile_shader_libs()` keyed the
    cache only on `sha256(<src.slang>)`, so a slangc swap left every
    artifact on disk and the next compile reused now-incompatible IR.

    Uses two layers for ABI-change detection:
    1. ``(realpath, size, mtime_ns)`` — cheap, no subprocess. Catches
       the common case of a rebuild that changes the binary.
    2. ``slangc --version`` output — catches ABI-incompatible rebuilds
       that happen to preserve size/mtime (e.g. CI cache restore,
       in-place hot patch, OS-level file dedup). The version string
       includes the build commit hash and changes with every build.

    The ``--version`` probe is cached in the module-level
    ``_slangc_fingerprint_cache`` dict so it runs at most once per
    process lifetime. Falls back to the stat-only key when slangc is
    unreachable (e.g. CI environments where slangc isn't on PATH).
    """
    try:
        st = os.stat(_SLANGC)
        stat_key = f"{os.path.realpath(_SLANGC)}:{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return f"unresolved:{_SLANGC}"
    version = _slangc_fingerprint_cache.get(stat_key)
    if version is None and _slangc_available():
        try:
            proc = subprocess.run(
                [_SLANGC, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            version = proc.stdout.strip() if proc.stdout else f"rc={proc.returncode}"
        except (subprocess.TimeoutExpired, OSError):
            version = "version_unknown"
        _slangc_fingerprint_cache[stat_key] = version
    if version:
        return f"{stat_key}::{version}"
    return stat_key


_slangc_fingerprint_cache: dict[str, str] = {}


_SHADERS_LIB_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "shaders", "lib")
)

_SHADER_LIB_MODULE_CACHE_DIR = os.environ.get(
    "TORCH_VULKAN_SLANG_MODULE_CACHE",
    os.path.join(os.path.expanduser("~"), ".cache", "torch_vulkan", "slang-modules"),
)

_SHADER_LIB_MODULE_STATS = {"compiles": 0, "cache_hits": 0}


def _shader_lib_sources() -> list[str]:
    if not os.path.isdir(_SHADERS_LIB_DIR):
        return []
    return sorted(
        os.path.join(_SHADERS_LIB_DIR, f)
        for f in os.listdir(_SHADERS_LIB_DIR)
        if f.endswith(".slang")
    )


def precompile_shader_libs(force: bool = False) -> dict:
    """Walk `shaders/lib/*.slang`, emit `<name>.slang-module` artifacts.

    Cache keyed on source SHA256: a module is recompiled only when its
    source content changes. Sidecar `.hash` files record the source hash
    that produced each cached module. Returns a summary dict
    ``{"compiled": [names], "cached": [names], "module_dir": str}`` and
    bumps `_SHADER_LIB_MODULE_STATS`.

    Output dir is `~/.cache/torch_vulkan/slang-modules` (override via
    ``TORCH_VULKAN_SLANG_MODULE_CACHE``). The same dir is added to slangc's
    `-I` so kernels that `import <name>;` resolve to the precompiled module
    instead of re-parsing the source.
    """
    os.makedirs(_SHADER_LIB_MODULE_CACHE_DIR, exist_ok=True)
    compiled, cached = [], []
    # PF.27.a.1: mix the slangc fingerprint into the cache key so a
    # slangc upgrade invalidates every cached `.slang-module` whose
    # serialized IR ABI may have changed. Hash mismatch fires on the
    # cache-hit read path below, exactly the requirement called out by
    # debug-coordinator.
    slangc_fp = _slangc_fingerprint()
    for src_path in _shader_lib_sources():
        name = os.path.splitext(os.path.basename(src_path))[0]
        with open(src_path, "rb") as f:
            src_bytes = f.read()
        src_hash = hashlib.sha256(
            src_bytes + b"\x00" + slangc_fp.encode("utf-8")
        ).hexdigest()
        mod_path = os.path.join(_SHADER_LIB_MODULE_CACHE_DIR, name + ".slang-module")
        hash_path = os.path.join(_SHADER_LIB_MODULE_CACHE_DIR, name + ".hash")
        prev_hash = None
        if os.path.exists(hash_path) and os.path.exists(mod_path):
            try:
                with open(hash_path) as f:
                    prev_hash = f.read().strip()
            except OSError:
                prev_hash = None
        if not force and prev_hash == src_hash:
            cached.append(name)
            _SHADER_LIB_MODULE_STATS["cache_hits"] += 1
            continue
        if not _slangc_available():
            raise RuntimeError(
                "slangc not found. Set SLANGC=/path/to/slangc; needed to "
                "precompile shader-lib modules at "
                f"{_SHADERS_LIB_DIR}."
            )
        argv = [_SLANGC, src_path, "-emit-ir", "-o", mod_path]
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=_SLANGC_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as e:
            raise SlangCompileTimeout(
                key=f"shader_lib_{name}",
                argv=argv,
                partial_stdout=(e.stdout or b"").decode("utf-8", errors="replace")
                if isinstance(e.stdout, bytes)
                else (e.stdout or ""),
                partial_stderr=(e.stderr or b"").decode("utf-8", errors="replace")
                if isinstance(e.stderr, bytes)
                else (e.stderr or ""),
            ) from None
        if proc.returncode != 0:
            raise RuntimeError(
                f"slangc failed precompiling shader lib {name}:\n{proc.stderr}"
            )
        tmp_hash = hash_path + f".tmp.{os.getpid()}"
        with open(tmp_hash, "w") as f:
            f.write(src_hash)
        os.replace(tmp_hash, hash_path)
        compiled.append(name)
        _SHADER_LIB_MODULE_STATS["compiles"] += 1
    return {
        "compiled": compiled,
        "cached": cached,
        "module_dir": _SHADER_LIB_MODULE_CACHE_DIR,
    }


_shader_lib_modules_ready = False


def _ensure_shader_lib_modules() -> str:
    """Lazy precompile on first kernel compile; returns module cache dir."""
    global _shader_lib_modules_ready
    if not _shader_lib_modules_ready:
        precompile_shader_libs()
        _shader_lib_modules_ready = True
    return _SHADER_LIB_MODULE_CACHE_DIR


def _reset_shader_lib_modules_ready() -> None:
    """Test hook — re-runs the lazy precompile pass on the next compile."""
    global _shader_lib_modules_ready
    _shader_lib_modules_ready = False


def _invalidate_shader_lib_modules() -> None:
    """Wipe `.slang-module` + `.hash` artifacts in the module-cache dir.

    PF.27.a.1 retry path: called from `_compile_slang_to_spirv_inner`
    when slangc dies with rc<0 on a kernel with the module cache on
    `-I`. Almost always indicates a stale-cache symptom that the
    fingerprint key in `precompile_shader_libs` somehow couldn't catch
    (e.g. ABI-incompatible slangc rebuild with identical stat).
    Resets the ready-flag so the next compile re-runs precompile from
    scratch.
    """
    global _shader_lib_modules_ready
    _shader_lib_modules_ready = False
    cache_dir = _SHADER_LIB_MODULE_CACHE_DIR
    if not os.path.isdir(cache_dir):
        return
    for fname in os.listdir(cache_dir):
        if fname.endswith(".slang-module") or fname.endswith(".hash"):
            try:
                os.remove(os.path.join(cache_dir, fname))
            except OSError:
                pass


# ── P3.2 / M14: Link-time specialization helpers ────────────────────────

# Cached capability flag: does this slangc support precompiled module imports?
# Determined by probing with a minimal wrapper at first use.
_slangc_modules_available: bool | None = None


def _slangc_supports_modules() -> bool:
    """Check if slangc supports importing precompiled .slang-module files.

    P3.2 / M14: The link-time specialization feature requires slangc to
    resolve ``import mm_tile;`` against a precompiled ``.slang-module``.
    Some slangc versions don't support this (returning "declaration not
    accessible" errors). This check probes with a minimal wrapper and
    caches the result for the session.
    """
    global _slangc_modules_available
    if _slangc_modules_available is not None:
        return _slangc_modules_available
    if not _slangc_available():
        _slangc_modules_available = False
        return False
    # Probe: try to compile a minimal wrapper that imports a known module.
    # We use pointwise (always available) for the probe.
    probe_src = (
        'import pointwise;\n[shader("compute")]\n[numthreads(1,1,1)]\nvoid main() {}\n'
    )
    try:
        with tempfile.TemporaryDirectory() as td:
            src_path = os.path.join(td, "probe.slang")
            out_path = os.path.join(td, "probe.spv")
            with open(src_path, "w") as f:
                f.write(probe_src)
            proc = subprocess.run(
                [
                    _SLANGC,
                    src_path,
                    "-target",
                    "spirv",
                    "-entry",
                    "main",
                    "-o",
                    out_path,
                    "-matrix-layout-row-major",
                    "-I",
                    _SHADERS_LIB_DIR,
                ],
                capture_output=True,
                text=True,
                timeout=_SLANGC_TIMEOUT_S,
            )
            _slangc_modules_available = proc.returncode == 0
    except Exception:
        _slangc_modules_available = False
    return _slangc_modules_available


# Regex to extract link-time specialization constants from mm_tile wrappers.
# Matches patterns like: static const int TILE_M = 64;
_LT_SPEC_CONST_RE = re.compile(
    r"(?:static\s+const\s+int|#define)\s+(TILE_[MNK]|M_PER_THREAD|N_PER_THREAD|NUM_STAGES)\s*=\s*(\d+)\s*;?"
)


def _mm_tile_module_path() -> str:
    """Return the path to the precompiled mm_tile.slang-module."""
    return os.path.join(_SHADER_LIB_MODULE_CACHE_DIR, "mm_tile.slang-module")


def _mm_tile_module_available() -> bool:
    """Check if the mm_tile.slang-module has been precompiled."""
    return os.path.isfile(_mm_tile_module_path())


def _ensure_mm_tile_module() -> str:
    """Ensure mm_tile.slang-module is precompiled. Returns the module path.

    P3.2 / M14: The mm_tile module is compiled once (per slangc version)
    and cached on disk. This function is called before any link-time
    specialized matmul kernel compilation to guarantee the module is
    available for linking.
    """
    mod_path = _mm_tile_module_path()
    if os.path.isfile(mod_path):
        return mod_path
    # Trigger precompilation of all shader libs (including mm_tile)
    _ensure_shader_lib_modules()
    if not os.path.isfile(mod_path):
        raise RuntimeError(
            "mm_tile.slang-module not found after precompilation. "
            "Ensure shaders/lib/mm_tile.slang exists and slangc is available."
        )
    return mod_path


def _extract_linktime_spec_constants(src: str) -> dict[str, int]:
    """Parse link-time specialization constants from a wrapper source.

    Scans for ``static const int TILE_M = 64;`` patterns and returns a
    dict mapping constant name to value. Used to construct
    ``-specialize-const`` slangc arguments for explicit link-time
    specialization.
    """
    constants: dict[str, int] = {}
    for m in _LT_SPEC_CONST_RE.finditer(src):
        name = m.group(1)
        try:
            value = int(m.group(2))
        except ValueError:
            continue
        constants[name] = value
    return constants


def _build_specialize_const_args(constants: dict[str, int]) -> list[str]:
    """Build ``-specialize-const`` slangc arguments from a constants dict.

    Example: ``{"TILE_M": 64, "TILE_K": 16}`` →
    ``["-specialize-const", "TILE_M=64", "-specialize-const", "TILE_K=16"]``.
    """
    args: list[str] = []
    for name, value in sorted(constants.items()):
        args.extend(["-specialize-const", f"{name}={value}"])
    return args


def _disk_reflection_path(hash_key: str) -> str:
    return os.path.join(_DISK_CACHE_DIR, hash_key[:2], hash_key[2:] + ".refl.json")


def _disk_reflection_read(hash_key: str) -> Optional[str]:
    try:
        with open(_disk_reflection_path(hash_key)) as f:
            return f.read()
    except (FileNotFoundError, OSError):
        return None


def _disk_reflection_write(hash_key: str, blob: str) -> None:
    path = _disk_reflection_path(hash_key)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + f".tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(blob)
        os.replace(tmp, path)
    except OSError:
        pass


_reflection_cache: dict[str, str] = {}

# ── P3.3/M13: Reflection metrics harvesting ───────────────────────────────
# Disk paths for per-kernel metrics JSON sidecar files.


def _disk_metrics_path(hash_key: str) -> str:
    return os.path.join(_DISK_CACHE_DIR, hash_key[:2], hash_key[2:] + ".metrics.json")


def _disk_metrics_read(hash_key: str):
    """Read parsed metrics from the disk sidecar. Returns None on miss."""
    import json

    try:
        with open(_disk_metrics_path(hash_key)) as f:
            return json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def _disk_metrics_write(hash_key: str, metrics: dict) -> None:
    """Write metrics dict as JSON sidecar. Best-effort."""
    import json

    path = _disk_metrics_path(hash_key)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + f".tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(metrics, f, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        pass


# ── M6: SPIR-V perf regression baseline tracking ───────────────────────────

_SPV_BASELINES_PATH = os.path.join(
    os.path.expanduser("~"), ".cache", "torch_vulkan", "spv_baselines.json"
)


def _load_spv_baselines() -> dict:
    """Load the on-disk SPIR-V perf baseline dict. Idempotent."""
    global _spv_baselines, _spv_baselines_loaded
    if _spv_baselines_loaded:
        return _spv_baselines
    _spv_baselines_loaded = True
    import json

    try:
        with open(_SPV_BASELINES_PATH) as f:
            _spv_baselines = json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        _spv_baselines = {}
    return _spv_baselines


def _save_spv_baselines() -> None:
    """Persist the in-memory baseline dict to disk. Best-effort."""
    import json

    try:
        os.makedirs(os.path.dirname(_SPV_BASELINES_PATH), exist_ok=True)
        tmp = _SPV_BASELINES_PATH + f".tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(_spv_baselines, f, sort_keys=True, indent=2)
        os.replace(tmp, _SPV_BASELINES_PATH)
    except OSError:
        pass


def _check_spv_regression(short_key: str, metrics: dict) -> None:
    """Check for SPIR-V perf regressions vs cached baseline (M6).

    Logs a warning when a kernel's VGPR count or shared-memory usage
    increases relative to a previously cached baseline. The baseline is
    stored in /home/amit/.cache/torch_vulkan/spv_baselines.json and
    updated on first sight of a new key.
    """
    import logging

    _load_spv_baselines()
    existing = _spv_baselines.get(short_key)
    if existing is None:
        # First sight — record as baseline.
        _spv_baselines[short_key] = {
            "vgprs": metrics.get("vgprs"),
            "shared_mem": metrics.get("shared_mem"),
            "subgroup_size": metrics.get("subgroup_size"),
            "loop_depth": metrics.get("loop_depth"),
        }
        _save_spv_baselines()
        return

    logger = logging.getLogger(__name__)
    vgprs = metrics.get("vgprs")
    prev_vgprs = existing.get("vgprs")
    if (
        vgprs is not None
        and prev_vgprs is not None
        and isinstance(vgprs, (int, float))
        and isinstance(prev_vgprs, (int, float))
        and vgprs > prev_vgprs
    ):
        logger.warning(
            "SPIR-V VGPR regression for kernel %s: %s -> %s (+%.0f%%). "
            "Workgroup occupancy may degrade.",
            short_key,
            prev_vgprs,
            vgprs,
            (vgprs - prev_vgprs) / max(prev_vgprs, 1) * 100,
        )
        existing["vgprs"] = vgprs
        _save_spv_baselines()

    sm = metrics.get("shared_mem")
    prev_sm = existing.get("shared_mem")
    if (
        sm is not None
        and prev_sm is not None
        and isinstance(sm, (int, float))
        and isinstance(prev_sm, (int, float))
        and sm > prev_sm
    ):
        logger.warning(
            "SPIR-V shared-memory increase for kernel %s: %s -> %s bytes (+%.0f%%).",
            short_key,
            prev_sm,
            sm,
            (sm - prev_sm) / max(prev_sm, 1) * 100,
        )
        existing["shared_mem"] = sm
        _save_spv_baselines()


# ── Reflection JSON parsing ───────────────────────────────────────────────


def _parse_reflection_metrics(refl_json: str) -> dict:
    """Parse slangc reflection JSON for key performance metrics (P3.3/M13).

    Extracts:
    - vgprs: estimated VGPR count (from SPIR-V analysis or
      reflection hints)
    - shared_mem: total groupshared / LDS bytes required
    - subgroup_size: wave size (32 or 64) from entry-point config
    - loop_depth: maximum nested loop depth

    Returns a dict; missing keys are set to None. The output schema is
    always {"vgprs": int|None, "shared_mem": int|None,
    "subgroup_size": int|None, "loop_depth": int|None}.
    """
    import json

    metrics: dict = {
        "vgprs": None,
        "shared_mem": None,
        "subgroup_size": None,
        "loop_depth": None,
    }
    try:
        data = json.loads(refl_json)
    except (json.JSONDecodeError, TypeError):
        return metrics

    # ── subgroup_size from entry points ──
    entry_points = data.get("entryPoints") or []
    for ep in entry_points:
        if ep.get("stage") == "compute":
            ss = ep.get("subgroupSize")
            if ss is not None:
                metrics["subgroup_size"] = int(ss)
            break

    # ── shared_mem from groupshared parameters ──
    gs_size = 0
    for p in data.get("parameters", []):
        b = p.get("binding") or {}
        if b.get("kind") == "groupshared":
            t = p.get("type", {})
            gs_size += int(t.get("size", 0) or 0)
    if gs_size > 0:
        metrics["shared_mem"] = gs_size

    # ── num_registers / vgpr_count ──
    for ep in entry_points:
        regs = ep.get("numRegisters") or ep.get("usedRegisters")
        if regs is not None:
            metrics["vgprs"] = int(regs)
            break

    # ── loop_depth ──
    for ep in entry_points:
        ld = ep.get("maxLoopDepth") or ep.get("loopDepth")
        if ld is not None:
            metrics["loop_depth"] = int(ld)
            break

    return metrics


# ── SPIR-V binary analysis fallback ──────────────────────────────────────


def _analyze_spirv_binary(spv: bytes) -> dict:
    """Estimate VGPR count and shared-memory usage from SPIR-V binary.

    Lightweight SPIR-V parser that walks the module to count:
    - OpVariable with StorageClass Function -> VGPRs lower-bound
      (each scalar variable costs at least 1 register)
    - OpVariable with StorageClass Workgroup -> shared-memory
      lower-bound

    Returns a partial metrics dict; only fields that could be estimated
    are populated. This is a fallback for when slangc reflection doesn't
    provide register counts.
    """
    metrics: dict = {"vgprs": None, "shared_mem": None}

    if len(spv) < 20:
        return metrics

    # Determine endianness from SPIR-V magic number
    magic = spv[0:4]
    if magic == b"\x03\x02\x23\x07":
        little = True
    elif magic == b"\x07\x23\x02\x03":
        little = False
    else:
        return metrics

    def w32(off: int) -> int:
        b = spv[off : off + 4]
        return int.from_bytes(b, "little" if little else "big")

    OP_VARIABLE = 59
    STORAGE_CLASS_FUNCTION = 7
    STORAGE_CLASS_WORKGROUP = 4
    OP_TYPE_FLOAT = 22
    OP_TYPE_INT = 21
    OP_TYPE_ARRAY = 28
    OP_DECORATE = 71

    n_words = len(spv) // 4
    i = 5  # skip 5-word header

    type_sizes: dict = {}
    func_vars = 0
    workgroup_bytes = 0

    while i < n_words:
        word = w32(i * 4)
        op = word & 0xFFFF
        wc = word >> 16
        if wc == 0:
            break

        if op == OP_TYPE_FLOAT and wc >= 3:
            tid = w32((i + 1) * 4)
            width = w32((i + 2) * 4)
            type_sizes[tid] = max(width // 8, 1)
        elif op == OP_TYPE_INT and wc >= 4:
            tid = w32((i + 1) * 4)
            width = w32((i + 2) * 4)
            type_sizes[tid] = max(width // 8, 1)
        elif op == OP_TYPE_ARRAY and wc >= 4:
            tid = w32((i + 1) * 4)
            elem_type = w32((i + 2) * 4)
            type_sizes[tid] = type_sizes.get(elem_type, 4)
        elif op == OP_VARIABLE and wc >= 4:
            result_type = w32((i + 1) * 4)
            storage_class = w32((i + 3) * 4)
            if storage_class == STORAGE_CLASS_FUNCTION:
                func_vars += 1
            elif storage_class == STORAGE_CLASS_WORKGROUP:
                elem_size = type_sizes.get(result_type, 4)
                workgroup_bytes += elem_size

        i += wc

    # Heuristic: each function-scope variable costs ≥1 VGPR + 1 temp.
    if func_vars > 0:
        metrics["vgprs"] = func_vars * 2
    if workgroup_bytes > 0:
        metrics["shared_mem"] = workgroup_bytes

    return metrics


def _analyze_slang_source_for_loop_depth(src: str) -> int:
    """Estimate maximum nested loop depth from Slang source.

    Scans for for and while keyword nesting. Conservative —
    doesn't parse preprocessor conditionals or generics, so may
    over-count in rare cases. Returns 0 for straight-line code.
    """
    import re

    # Remove comments and strings to avoid false positives
    cleaned = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    cleaned = re.sub(r"//[^\n]*", "", cleaned)
    cleaned = re.sub(r'"[^"]*"', '""', cleaned)

    max_depth = 0
    current_depth = 0
    i = 0
    while i < len(cleaned):
        if cleaned[i : i + 3] == "for" and (
            i + 3 >= len(cleaned) or not cleaned[i + 3].isalpha()
        ):
            j = cleaned.find("{", i)
            if j != -1:
                current_depth += 1
                max_depth = max(max_depth, current_depth)
                i = j + 1
                continue
        elif cleaned[i : i + 5] == "while" and (
            i + 5 >= len(cleaned) or not cleaned[i + 5].isalpha()
        ):
            j = cleaned.find("{", i)
            if j != -1:
                current_depth += 1
                max_depth = max(max_depth, current_depth)
                i = j + 1
                continue
        elif cleaned[i] == "}":
            current_depth = max(0, current_depth - 1)
        i += 1

    return max_depth


def _harvest_reflection_metrics(
    hash_key: str,
    refl_json: str,
    spv: bytes,
    src: str,
    config_key: str | None = None,
) -> dict:
    """Harvest and cache all available reflection metrics for a kernel.

    Combines slangc reflection JSON parsing with SPIR-V binary analysis
    and Slang source analysis. Writes the merged metrics to the in-memory
    and on-disk caches. Returns the metrics dict.

    DR.3: When ``config_key`` is provided, the metrics are also
    cross-indexed under this structural key so
    ``VulkanKernel._get_actual_vgprs`` can retrieve them on subsequent
    compiles with different numels but identical kernel structure.
    """
    # 1. Parse the reflection JSON
    metrics = _parse_reflection_metrics(refl_json)

    # 2. Fill gaps with SPIR-V binary analysis
    if metrics["vgprs"] is None or metrics["shared_mem"] is None:
        spv_metrics = _analyze_spirv_binary(spv)
        if metrics["vgprs"] is None:
            metrics["vgprs"] = spv_metrics.get("vgprs")
        if metrics["shared_mem"] is None:
            metrics["shared_mem"] = spv_metrics.get("shared_mem")

    # 3. Loop-depth from source analysis
    if metrics["loop_depth"] is None:
        metrics["loop_depth"] = _analyze_slang_source_for_loop_depth(src)

    # 4. Store in caches
    _reflection_metrics_by_hash[hash_key] = metrics
    _disk_metrics_write(hash_key, metrics)

    # DR.3: Cross-index under structural config_key so kernels with
    # identical structure but different numels can find cached data.
    if config_key is not None:
        _reflection_metrics_by_key[config_key] = metrics

    return metrics


def get_reflection_metrics(
    src=None,
    entry: str = "computeMain",
    cache_key=None,
    include_paths: tuple = (),
):
    """Get cached reflection metrics for a kernel (P3.3/M13 public API).

    Looks up the metrics by cache_key first, then by source-hash.
    Returns None when no metrics are available (e.g. compiled without
    reflection, or the kernel hasn't been compiled yet).

    Callers (e.g. kernel/main.py) can use the returned vgprs field
    to drive workgroup-size selection.
    """
    from ..inductor import config

    if not config.reflection_enabled():
        return None

    # Try by cache_key first
    if cache_key is not None:
        hit = _reflection_metrics_by_key.get(cache_key)
        if hit is not None:
            return hit

    # Try by source hash
    if src is not None:
        inc_tag = "" if not include_paths else "\nINC=" + "|".join(include_paths)
        hash_key = hashlib.sha256(
            (entry + "\n" + _normalize_slang_source(src) + inc_tag).encode()
        ).hexdigest()
        hit = _reflection_metrics_by_hash.get(hash_key)
        if hit is not None:
            if cache_key is not None:
                _reflection_metrics_by_key[cache_key] = hit
            return hit

        # Try disk
        disk = _disk_metrics_read(hash_key)
        if disk is not None:
            _reflection_metrics_by_hash[hash_key] = disk
            if cache_key is not None:
                _reflection_metrics_by_key[cache_key] = disk
            return disk

    return None


def get_cached_metrics_for_key(cache_key: str):
    """Fast-path lookup of cached reflection metrics by cache_key only.

    Designed for hot-path callers like kernel/main.py that already have
    a cache_key. Returns None on miss (caller falls back to
    heuristic).
    """
    from ..inductor import config

    if not config.reflection_enabled():
        return None
    return _reflection_metrics_by_key.get(cache_key)


def reset_reflection_baselines() -> None:
    """Clear the in-memory and on-disk SPIR-V baseline stores. Test hook."""
    global _spv_baselines, _spv_baselines_loaded
    _spv_baselines = {}
    _spv_baselines_loaded = True
    try:
        os.unlink(_SPV_BASELINES_PATH)
    except OSError:
        pass


# ── End of P3.3/M13 reflection metrics ─────────────────────────────────


# ── DR.7: Compile-time reflection routing helpers ───────────────────────

_NUMTHREADS_SRC_RE = re.compile(r"\[numthreads\((\d+),\s*(\d+),\s*(\d+)\)\]")


def _parse_numthreads_from_source(src: str) -> tuple[int, int, int] | None:
    """Extract numthreads tuple from Slang source.

    Returns ``(x, y, z)`` or ``None`` if no numthreads attribute is found.
    """
    m = _NUMTHREADS_SRC_RE.search(src)
    if m is None:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _rewrite_numthreads_in_source(src: str, new_nt: tuple[int, int, int]) -> str:
    """Replace the first numthreads attribute in *src* with *new_nt*."""
    replacement = f"[numthreads({new_nt[0]}, {new_nt[1]}, {new_nt[2]})]"
    return _NUMTHREADS_SRC_RE.sub(replacement, src, count=1)


def _pick_numthreads_from_reflection(
    vgprs: int | None,
    shared_mem: int | None = None,
    loop_depth: int | None = None,
    current_numthreads: tuple[int, int, int] = (256, 1, 1),
) -> tuple[int, int, int]:
    """DR.7: Pick optimal numthreads based on SPIR-V reflection metrics.

    RDNA1 occupancy heuristic (wave64, 256 VGPRs/CU, 1024 max threads/CU):

    - VGPRs ≤ 32  → 256 threads (4 waves/CU → high occupancy)
    - VGPRs 33–64 → 128 threads (2 waves/CU → balance)
    - VGPRs 65–128 → 64 threads  (1 wave/CU → avoid register spilling)
    - VGPRs > 128  → 32 threads  (minimum occupancy, avoid scratch)

    Falls back to *current_numthreads* when *vgprs* is ``None``.

    Args:
        vgprs: VGPR count from slangc reflection (numRegisters / usedRegisters).
        shared_mem: Groupshared / LDS bytes used (reserved for future use).
        loop_depth: Maximum nested loop depth (reserved for future use).
        current_numthreads: The numthreads currently in the source.

    Returns:
        Optimal ``(x, y, z)`` numthreads tuple.
    """
    if vgprs is None:
        return current_numthreads

    if vgprs <= 32:
        return (256, 1, 1)
    elif vgprs <= 64:
        return (128, 1, 1)
    elif vgprs <= 128:
        return (64, 1, 1)
    else:
        return (32, 1, 1)


# ── Core compilation ────────────────────────────────────────────────────


def _compile_slang_to_spirv_inner(
    src: str,
    entry: str,
    hash_key: str,
    include_paths: tuple[str, ...] = (),
    config_key: str | None = None,
) -> bytes:
    # T.7 (2026-05-08): pre-flight Slang source validation runs FIRST,
    # before slangc availability probes, tempdir creation, or any file
    # I/O.  The previous call site (post-tempfile-write but pre-subprocess)
    # still paid for the OS handles even on guaranteed-fail inputs; the
    # M15 docstring promised a "fast pre-check" — moving it to the top
    # delivers that.  Catches brace mismatches, binding gaps, size-symbol
    # leaks, groupshared budget overruns, and numthreads violations
    # without ever invoking the SPIR-V compiler.
    from torch_vulkan.inductor.slang_validator import validate_slang_source

    validation_errors = validate_slang_source(src)
    if validation_errors:
        raise RuntimeError(
            f"Slang source validation failed for kernel "
            f"{hash_key[:48]}:\n" + "\n".join(str(e) for e in validation_errors)
        )

    if not _slangc_available():
        raise RuntimeError(
            "slangc not found. Set SLANGC=/path/to/slangc or install slang. "
            "The torch_vulkan Inductor backend JIT-compiles Slang to SPIR-V "
            "via the slangc CLI."
        )

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        src_path = os.path.join(td, "kernel.slang")
        out_path = os.path.join(td, "kernel.spv")
        refl_path = os.path.join(td, "kernel.refl.json")
        with open(src_path, "w") as f:
            f.write(src)
        # Precompiled .slang-module dir is searched FIRST so kernels that
        # `import helpers;` or `import tensor_layout;` resolve to the
        # serialized IR (no per-kernel reparse). The source dir stays on
        # the -I list as a fallback for libraries that haven't been
        # precompiled yet (e.g. when slangc is unavailable in CI).
        try:
            module_dir = _ensure_shader_lib_modules()
            module_includes = [module_dir]
        except RuntimeError:
            module_includes = []
        cmd = [
            _SLANGC,
            src_path,
            "-target",
            "spirv",
            "-entry",
            entry,
            "-o",
            out_path,
            "-reflection-json",
            refl_path,
            "-matrix-layout-row-major",
        ]
        for ip in module_includes:
            cmd.extend(["-I", ip])
        cmd.extend(["-I", _SHADERS_LIB_DIR])
        for ip in include_paths:
            cmd.extend(["-I", ip])

        # P3.2 / M14: Ensure mm_tile module is precompiled for link-time
        # specialization. The wrapper source already defines tile-size
        # constants via "static const int" before "import mm_tile;", so
        # Slang's linker resolves them without additional flags.
        # We just need the .slang-module to exist on the -I path.
        if "import mm_tile;" in src:
            _ensure_mm_tile_module()

        # DR.7: track whether precompiled modules were on the include path
        # so the optional re-compile pass mirrors the same include structure.
        used_module_includes = bool(module_includes)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_SLANGC_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as e:
            raise SlangCompileTimeout(
                key=hash_key,
                argv=cmd,
                partial_stdout=(e.stdout or b"").decode("utf-8", errors="replace")
                if isinstance(e.stdout, bytes)
                else (e.stdout or ""),
                partial_stderr=(e.stderr or b"").decode("utf-8", errors="replace")
                if isinstance(e.stderr, bytes)
                else (e.stderr or ""),
            ) from None
        # PF.27.a.1: slangc SIGSEGV (rc<0) with the precompiled module
        # cache on `-I` is a stale-cache symptom — invalidate and retry
        # once from source. Belt-and-suspenders for the case the
        # `_slangc_fingerprint` cache key in `precompile_shader_libs`
        # couldn't catch (ABI-incompatible slangc rebuild with same
        # stat). Hard-fail on second crash to prevent any retry loop
        # and to avoid masking non-stale-cache slangc bugs.
        if proc.returncode < 0 and module_includes:
            _invalidate_shader_lib_modules()
            used_module_includes = False  # DR.7: retry without module includes
            cmd2 = [
                _SLANGC,
                src_path,
                "-target",
                "spirv",
                "-entry",
                entry,
                "-o",
                out_path,
                "-reflection-json",
                refl_path,
                "-matrix-layout-row-major",
                "-I",
                _SHADERS_LIB_DIR,
            ]
            for ip in include_paths:
                cmd2.extend(["-I", ip])
            try:
                proc = subprocess.run(
                    cmd2,
                    capture_output=True,
                    text=True,
                    timeout=_SLANGC_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired as e:
                raise SlangCompileTimeout(
                    key=hash_key,
                    argv=cmd2,
                    partial_stdout=(e.stdout or b"").decode("utf-8", errors="replace")
                    if isinstance(e.stdout, bytes)
                    else (e.stdout or ""),
                    partial_stderr=(e.stderr or b"").decode("utf-8", errors="replace")
                    if isinstance(e.stderr, bytes)
                    else (e.stderr or ""),
                ) from None
        if proc.returncode != 0:
            raise RuntimeError(
                f"slangc failed for kernel {hash_key[:8]}:\n{proc.stderr}\n"
                f"--- source ---\n{src}"
            )
        with open(out_path, "rb") as f:
            spv = f.read()
        try:
            with open(refl_path) as f:
                refl_blob = f.read()
            _reflection_cache[hash_key] = refl_blob
            _disk_reflection_write(hash_key, refl_blob)
            # P3.3/M13: harvest reflection metrics when enabled.
            # Gate on the config flag so TORCH_VULKAN_REFLECTION=0
            # bypasses the parsing + SPIR-V analysis overhead.
            from torch_vulkan.inductor import config as _cfg

            if _cfg.reflection_enabled():
                _harvest_reflection_metrics(hash_key, refl_blob, spv, src, config_key)
                # M6: check for SPIR-V perf regression vs baseline.
                short_key = hashlib.sha256(spv).hexdigest()[:12]
                _check_spv_regression(
                    short_key, _reflection_metrics_by_hash.get(hash_key, {})
                )
            # ── DR.7: Compile-time reflection routing ──────────────
            # Peek at VGPR count from SPIR-V reflection; if numthreads
            # should be adjusted for optimal occupancy, rewrite the
            # source and re-compile. The optimized SPIR-V replaces the
            # Pass-1 output and is cached under the original hash_key
            # so subsequent requests skip both passes entirely.
            if _cfg.reflection_enabled() and _cfg.reflection_routing():
                metrics = _reflection_metrics_by_hash.get(hash_key, {})
                vgprs = metrics.get("vgprs")
                if vgprs is not None:
                    current_nt = _parse_numthreads_from_source(src)
                    if current_nt is not None:
                        shared_mem = metrics.get("shared_mem")
                        loop_depth = metrics.get("loop_depth")
                        optimal_nt = _pick_numthreads_from_reflection(
                            vgprs,
                            shared_mem,
                            loop_depth,
                            current_nt,
                        )
                        if optimal_nt != current_nt:
                            # Rewrite source with optimized numthreads.
                            new_src = _rewrite_numthreads_in_source(src, optimal_nt)
                            new_src_path = os.path.join(td, "kernel_opt.slang")
                            new_out_path = os.path.join(td, "kernel_opt.spv")
                            new_refl_path = os.path.join(td, "kernel_opt.refl.json")
                            with open(new_src_path, "w") as f:
                                f.write(new_src)
                            cmd3 = [
                                _SLANGC,
                                new_src_path,
                                "-target",
                                "spirv",
                                "-entry",
                                entry,
                                "-o",
                                new_out_path,
                                "-reflection-json",
                                new_refl_path,
                                "-matrix-layout-row-major",
                            ]
                            if used_module_includes:
                                for ip in module_includes:
                                    cmd3.extend(["-I", ip])
                            cmd3.extend(["-I", _SHADERS_LIB_DIR])
                            for ip in include_paths:
                                cmd3.extend(["-I", ip])
                            try:
                                proc3 = subprocess.run(
                                    cmd3,
                                    capture_output=True,
                                    text=True,
                                    timeout=_SLANGC_TIMEOUT_S,
                                )
                            except subprocess.TimeoutExpired:
                                # DR.7: Pass-2 timeout → keep Pass-1 SPV.
                                pass
                            else:
                                if proc3.returncode == 0:
                                    with open(new_out_path, "rb") as f:
                                        spv = f.read()
                                    try:
                                        with open(new_refl_path) as f:
                                            refl_blob2 = f.read()
                                        _reflection_cache[hash_key] = refl_blob2
                                        _disk_reflection_write(hash_key, refl_blob2)
                                        if _cfg.reflection_enabled():
                                            _harvest_reflection_metrics(
                                                hash_key,
                                                refl_blob2,
                                                spv,
                                                new_src,
                                                config_key,
                                            )
                                    except (FileNotFoundError, OSError):
                                        pass
        except (FileNotFoundError, OSError):
            pass
    _COMPILE_STATS["cold_compiles"] += 1
    elapsed_us = (time.perf_counter() - t0) * 1e6
    _COMPILE_STATS["cold_compile_us"] += elapsed_us
    if elapsed_us > _COMPILE_STATS["max_cold_compile_us"]:
        _COMPILE_STATS["max_cold_compile_us"] = elapsed_us

    _cache_by_hash[hash_key] = spv
    _disk_cache_write(hash_key, spv)
    return spv


def compile_slang_to_spirv(
    src: str,
    entry: str = "computeMain",
    cache_key: Optional[str] = None,
    include_paths: tuple[str, ...] = (),
    config_key: Optional[str] = None,
) -> bytes:
    """Compile Slang source to SPIR-V bytes, cached.

    Uses `cache_key` when provided (avoids hashing the whole source on every
    hot-path dispatch); otherwise falls back to SHA256 of the source.

    When ``TORCH_VULKAN_ASYNC_COMPILE=1``, the slangc subprocess runs in a
    background thread. The in-memory and on-disk caches still apply so
    duplicate compilations are avoided.

    N+1.6: In-flight dedup — if another thread is already compiling the same
    hash_key, wait on its completion instead of launching a duplicate slangc.

    DR.3: ``config_key`` is passed through to the inner compiler so
    harvested reflection metrics are cross-referenced under this
    structural key.
    """
    if cache_key is not None:
        hit = _cache_by_key.get(cache_key)
        if hit is not None:
            _COMPILE_STATS["in_memory_hits"] += 1
            return hit

    # Normalize before hashing so cosmetic source variation (comments,
    # trailing whitespace, blank lines) doesn't fragment the cache.
    inc_tag = "" if not include_paths else "\nINC=" + "|".join(include_paths)
    # N+1.12: Mix in device subgroup-size tag so wave32 vs wave64
    # produce distinct SPIR-V cache entries (barrier / wave-intrinsic
    # code may differ across subgroup sizes).
    sgs_tag = _get_device_subgroup_size_tag()
    hash_key = hashlib.sha256(
        (entry + "\n" + _normalize_slang_source(src) + inc_tag + sgs_tag).encode()
    ).hexdigest()
    hit = _cache_by_hash.get(hash_key)
    if hit is not None:
        _COMPILE_STATS["in_memory_hits"] += 1
        if cache_key is not None:
            _cache_by_key[cache_key] = hit
        return hit

    spv = _disk_cache_read(hash_key)
    if spv is not None:
        _COMPILE_STATS["disk_cache_hits"] += 1
        _cache_by_hash[hash_key] = spv
        if cache_key is not None:
            with _cache_lock:
                _cache_by_key[cache_key] = spv
        return spv

    # N+1.6: in-flight dedup — avoid duplicate slangc subprocesses for the
    # same hash_key when multiple threads request the same kernel.
    we_own = False
    with _in_flight_lock:
        event = _in_flight.get(hash_key)
        if event is None:
            event = threading.Event()
            _in_flight[hash_key] = event
            we_own = True  # we created it — we must compile and set it

    if not we_own and not event.is_set():
        # Another thread owns this hash_key — wait for it.
        event.wait()
        spv = _cache_by_hash.get(hash_key)
        if spv is not None:
            if cache_key is not None:
                _cache_by_key[cache_key] = spv
            return spv
        # Edge case: the owning thread failed; fall through to compile.

    try:
        # PF.14: if we're already on a worker of `_ASYNC_POOL`, re-submitting
        # back to the same pool and blocking on `.result()` deadlocks when
        # every worker is itself blocked here. Bypass the pool in that case.
        if _PARALLEL_COMPILE and _ASYNC_COMPILE and not _is_in_pool_worker():
            pool = _get_async_pool()
            spv = pool.submit(
                _wrap_pool_worker(_compile_slang_to_spirv_inner),
                src,
                entry,
                hash_key,
                include_paths,
                config_key,
            ).result()
        else:
            spv = _compile_slang_to_spirv_inner(
                src, entry, hash_key, include_paths, config_key
            )
        if cache_key is not None:
            with _cache_lock:
                _cache_by_key[cache_key] = spv
        return spv
    finally:
        with _in_flight_lock:
            ev = _in_flight.pop(hash_key, None)
            if ev is not None:
                ev.set()  # wake any waiters


def batch_compile_slang_to_spirv(
    specs: list[tuple[str, str, str, tuple[str, ...]]],
    *,
    max_workers: Optional[int] = None,
) -> dict[str, bytes]:
    """Compile multiple Slang sources to SPIR-V in parallel.  N+1.6.

    Args:
        specs: List of ``(src, entry, cache_key, include_paths)`` tuples.
        max_workers: Max ThreadPoolExecutor workers (default: ``_ASYNC_MAX_WORKERS``).

    Returns:
        Dict mapping ``cache_key`` → SPIR-V bytes.  Already-cached entries
        are served from the in-memory / on-disk cache without a slangc
        subprocess.  In-flight dedup prevents duplicate slangc invocations
        for identical hash keys within the batch.

    Unlike ``prewarm_compile`` (fire-and-forget best-effort), this function
    always blocks until every spec is compiled and surfaces the first
    ``RuntimeError`` so callers can fail fast.
    """
    workers = max_workers if max_workers is not None else _ASYNC_MAX_WORKERS
    results: dict[str, bytes] = {}
    pending: list[tuple[str, str, str, tuple[str, ...]]] = []

    # Phase 1: check caches; collect cache misses.
    for src, entry, cache_key, include_paths in specs:
        if cache_key in _cache_by_key:
            results[cache_key] = _cache_by_key[cache_key]
            _COMPILE_STATS["in_memory_hits"] += 1
            continue
        inc_tag = "" if not include_paths else "\nINC=" + "|".join(include_paths)
        hash_key = hashlib.sha256(
            (entry + "\n" + _normalize_slang_source(src) + inc_tag).encode()
        ).hexdigest()
        if hash_key in _cache_by_hash:
            spv = _cache_by_hash[hash_key]
            results[cache_key] = spv
            _cache_by_key[cache_key] = spv
            _reflection_metrics_by_key.pop(cache_key, None)
            m = _reflection_metrics_by_hash.get(hash_key)
            if m is not None:
                _reflection_metrics_by_key[cache_key] = m
            _COMPILE_STATS["in_memory_hits"] += 1
            continue
        spv = _disk_cache_read(hash_key)
        if spv is not None:
            results[cache_key] = spv
            _cache_by_hash[hash_key] = spv
            _cache_by_key[cache_key] = spv
            _reflection_metrics_by_key.pop(cache_key, None)
            m = _disk_metrics_read(hash_key)
            if m is not None:
                _reflection_metrics_by_hash[hash_key] = m
                _reflection_metrics_by_key[cache_key] = m
            _COMPILE_STATS["disk_cache_hits"] += 1
            continue
        pending.append((src, entry, cache_key, include_paths))

    if not pending:
        return results

    # Phase 2: compile cache misses in parallel.
    errors: list[Exception] = []

    def _compile_one(
        src: str,
        entry: str,
        cache_key: str,
        include_paths: tuple[str, ...],
    ) -> Optional[bytes]:
        try:
            return compile_slang_to_spirv(
                src,
                entry=entry,
                cache_key=cache_key,
                include_paths=include_paths,
            )
        except Exception as e:
            errors.append(e)
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_compile_one, src, entry, ck, ip): ck
            for src, entry, ck, ip in pending
        }
        for fut in futures:
            ck = futures[fut]
            try:
                spv = fut.result()
                if spv is not None:
                    results[ck] = spv
            except Exception as e:
                errors.append(e)

    if errors:
        raise RuntimeError(
            f"batch_compile_slang_to_spirv: {len(errors)} compilation(s) failed. "
            f"First error: {errors[0]}"
        )
    return results


def _compile_slang_batch_parallel(
    sources: list[tuple[str, str]],
    *,
    max_workers: int | None = None,
    entry: str = "computeMain",
) -> list[bytes]:
    """Compile multiple (src, cache_key) pairs in parallel.  N+1.6.

    Thin convenience wrapper around ``batch_compile_slang_to_spirv`` for
    callers that have simple ``(src, cache_key)`` pairs without custom
    entry-points or include-paths.  Gated by the existing
    ``TORCH_VULKAN_ASYNC_COMPILE=1`` / ``TORCH_VULKAN_PARALLEL_COMPILE=1``
    (both default-on) which control the underlying thread-pool dispatch.

    Args:
        sources: List of ``(slang_src, cache_key)`` pairs.
        max_workers: Max ThreadPoolExecutor workers (default:
            ``_MAX_SLANGC_WORKERS``).
        entry: Slang entry-point name (default ``"computeMain"``).

    Returns:
        List of SPIR-V byte blobs in the same order as ``sources``.
        Already-cached entries are served from the in-memory / on-disk
        cache without a slangc subprocess.
    """
    workers = max_workers if max_workers is not None else _MAX_SLANGC_WORKERS
    specs: list[tuple[str, str, str, tuple[str, ...]]] = []
    for src, cache_key in sources:
        specs.append((src, entry, cache_key, ()))
    result_map = batch_compile_slang_to_spirv(specs, max_workers=workers)
    # Return in the same order as sources
    return [result_map[ck] for _src, ck in sources]


def parse_spec_constants(spv: bytes) -> list[tuple[int, int]]:
    """Walk a SPIR-V module and return ``[(spec_id, default_uint_value), ...]``.

    Reads `OpDecorate <id> SpecId N` (op 71) and `OpSpecConstant` (op 50)
    pairs. Default value is the 32-bit literal slangc bakes into the SPV
    for `[[vk::constant_id(N)]] const uint TILE = K;`. Sufficient for the
    P0.6 contract: prove the spec constants survived to SPIR-V so a
    future `VkSpecializationInfo` pass can override them at pipeline-
    creation time without recompiling slangc.
    """
    if len(spv) < 20 or spv[0:4] not in (b"\x03\x02\x23\x07", b"\x07\x23\x02\x03"):
        raise ValueError("not a SPIR-V module")
    little = spv[0:4] == b"\x03\x02\x23\x07"

    def w32(off: int) -> int:
        b = spv[off : off + 4]
        return int.from_bytes(b, "little" if little else "big")

    n_words = len(spv) // 4
    spec_ids: dict[int, int] = {}
    spec_defaults: dict[int, int] = {}
    i = 5  # skip 5-word header
    while i < n_words:
        word = w32(i * 4)
        op = word & 0xFFFF
        wc = word >> 16
        if wc == 0:
            break
        if op == 71 and wc >= 4:  # OpDecorate
            target = w32((i + 1) * 4)
            decoration = w32((i + 2) * 4)
            if decoration == 1:  # SpecId
                spec_ids[target] = w32((i + 3) * 4)
        elif op == 50 and wc >= 4:  # OpSpecConstant
            result_id = w32((i + 2) * 4)
            spec_defaults[result_id] = w32((i + 3) * 4)
        i += wc
    out: list[tuple[int, int]] = []
    for result_id, sid in spec_ids.items():
        out.append((sid, spec_defaults.get(result_id, 0)))
    out.sort()
    return out


def get_reflection_json(hash_key: str) -> Optional[str]:
    """Look up the cached slangc reflection JSON blob for a given key.

    First checks the in-memory cache, then the on-disk sidecar
    (`<spirv_cache_dir>/<prefix>/<rest>.refl.json`). Returns ``None`` when
    no reflection has ever been emitted for that key (e.g. compiled
    pre-P0.4 or via a slangc that doesn't support `-reflection-json`).
    """
    hit = _reflection_cache.get(hash_key)
    if hit is not None:
        return hit
    blob = _disk_reflection_read(hash_key)
    if blob is not None:
        _reflection_cache[hash_key] = blob
    return blob


def _binding_descriptor_count(param: dict) -> int:
    """Return the ``descriptorCount`` for a single reflection parameter.

    Looks for one of the recognised array shapes the slangc reflection
    JSON emits for ``RWStructuredBuffer<T> name[N]`` / nested
    ``ParameterBlock`` array members:

    - ``param["type"]["kind"] == "array"`` with ``elementCount: N`` —
      flat top-level array binding (the common case post-N+1.5).
    - ``param["binding"]["size"] >= 1`` — slangc occasionally inlines
      the count there for `descriptorTableSlot` kinds.
    - ``param["binding"]["subBindings"][...]`` — nested layout where the
      array element count lives one level deep (older slangc shapes).

    Returns ``1`` when none of the above match (i.e. a plain scalar
    binding).
    """
    t = param.get("type") or {}
    if t.get("kind") == "array":
        ec = t.get("elementCount")
        if ec is not None:
            try:
                n = int(ec)
                return n if n >= 1 else 1
            except (TypeError, ValueError):
                pass
    b = param.get("binding") or {}
    sz = b.get("size")
    if sz is not None:
        try:
            n = int(sz)
            if n >= 1:
                return n
        except (TypeError, ValueError):
            pass
    sub = b.get("subBindings") or []
    for s in sub:
        sb = s.get("binding") or {}
        ssz = sb.get("size")
        if ssz is not None:
            try:
                n = int(ssz)
                if n >= 1:
                    return n
            except (TypeError, ValueError):
                continue
        st = s.get("type") or {}
        if st.get("kind") == "array":
            ec = st.get("elementCount")
            if ec is not None:
                try:
                    n = int(ec)
                    if n >= 1:
                        return n
                except (TypeError, ValueError):
                    continue
    return 1


def reflection_layout(reflection_json: str) -> dict:
    """Extract the descriptor + push-constant layout from a slangc reflection JSON.

    Returns ``{"bindings": [(set, index, name), ...],
    "descriptor_counts": [int, ...], "push_constant_size": int}``.
    The ``descriptor_counts`` list is parallel to ``bindings`` (same
    length, same order after sort) and carries the per-binding
    ``descriptorCount`` extracted from the slangc reflection. A flat
    binding has count ``1``; an array binding (e.g.
    ``RWStructuredBuffer<float> outs[4]``) has count ``4``.

    Lets callers populate `VkDescriptorSetLayoutBinding` arrays without
    having to count tensors or guess push-constant sizes manually.
    """
    import json

    data = json.loads(reflection_json)
    paired: list[tuple[tuple[int, int, str], int]] = []
    pc_size = 0
    for p in data.get("parameters", []):
        b = p.get("binding") or {}
        kind = b.get("kind")
        if kind == "descriptorTableSlot":
            key = (b.get("space", 0), b.get("index", 0), p.get("name", ""))
            paired.append((key, _binding_descriptor_count(p)))
        elif kind == "pushConstantBuffer":
            t = p.get("type", {})
            elv = t.get("elementVarLayout", {})
            elb = elv.get("binding", {})
            pc_size = max(pc_size, int(elb.get("size", 0)))
    paired.sort(key=lambda kv: kv[0])
    bindings = [k for k, _ in paired]
    descriptor_counts = [c for _, c in paired]
    return {
        "bindings": bindings,
        "descriptor_counts": descriptor_counts,
        "push_constant_size": pc_size,
    }


def compile_slang_to_spirv_with_reflection(
    src: str,
    entry: str = "computeMain",
    cache_key: Optional[str] = None,
    include_paths: tuple[str, ...] = (),
) -> tuple[bytes, dict]:
    """Compile a Slang source and return ``(spv, layout_dict)``.

    Convenience wrapper for callers that want the binding/push-constant
    layout derived from compiler-truth instead of hand-counted from the
    source. Fully cached: repeat calls reuse both SPV and reflection.
    """
    spv = compile_slang_to_spirv(
        src, entry=entry, cache_key=cache_key, include_paths=include_paths
    )
    inc_tag = "" if not include_paths else "\nINC=" + "|".join(include_paths)
    hash_key = hashlib.sha256(
        (entry + "\n" + _normalize_slang_source(src) + inc_tag).encode()
    ).hexdigest()
    refl = get_reflection_json(hash_key)
    if refl is None:
        return spv, {"bindings": [], "push_constant_size": 0}
    return spv, reflection_layout(refl)


# ── D.3: Reflection-based buffer counting ───────────────────────────────────


def get_reflected_binding_count(spv: bytes) -> int | None:
    """Extract the number of storage-buffer bindings from SPIR-V reflection.

    Returns ``None`` when no reflection JSON is cached for this SPV (e.g.
    compiled with a slangc that doesn't support ``-reflection-json``).
    Callers should fall back to the hand-counted ``n_buffers`` in that case.
    """
    hash_key = hashlib.sha256(spv).hexdigest()
    refl_json = get_reflection_json(hash_key)
    if refl_json is None:
        # Try the full-src hash — the SPV hash alone may not match if the
        # reflection was keyed by (entry + src), not by SPV bytes.
        return None
    layout = reflection_layout(refl_json)
    return len(layout["bindings"])


def _get_reflected_buffer_count_from_cache_key(
    src: str,
    entry: str = "computeMain",
    include_paths: tuple[str, ...] = (),
) -> int | None:
    """Like :func:`get_reflected_binding_count` but keyed by source hash."""
    inc_tag = "" if not include_paths else "\nINC=" + "|".join(include_paths)
    hash_key = hashlib.sha256(
        (entry + "\n" + _normalize_slang_source(src) + inc_tag).encode()
    ).hexdigest()
    refl = get_reflection_json(hash_key)
    if refl is None:
        return None
    return len(reflection_layout(refl)["bindings"])


def get_reflected_descriptor_counts(spv: bytes) -> Optional[list[int]]:
    """N+1.5.a — extract per-binding ``descriptorCount`` from SPV reflection.

    Returns a list parallel to the binding list (same order as
    :func:`reflection_layout`'s ``bindings``). Each entry is the
    ``descriptorCount`` for that binding — ``1`` for a flat binding,
    ``N`` for an array binding (e.g. ``RWStructuredBuffer<T> arr[N]``).

    Returns ``None`` when no reflection JSON is cached for this SPV
    (e.g. compiled with a slangc that doesn't support
    ``-reflection-json``). Callers should fall back to ``[1] * n_buffers``
    in that case.
    """
    hash_key = hashlib.sha256(spv).hexdigest()
    refl_json = get_reflection_json(hash_key)
    if refl_json is None:
        return None
    layout = reflection_layout(refl_json)
    return list(layout.get("descriptor_counts") or [])


def _get_reflected_descriptor_counts_from_src(
    src: str,
    entry: str = "computeMain",
    include_paths: tuple[str, ...] = (),
) -> Optional[list[int]]:
    """Source-hash variant of :func:`get_reflected_descriptor_counts`.

    The reflection JSON is keyed by ``sha256(entry + normalized_src)``
    (see :func:`compile_slang_to_spirv`), so a SPV-hash lookup will
    miss. This helper performs the same lookup but using the source hash
    that ``compile_slang_to_spirv`` actually wrote.
    """
    inc_tag = "" if not include_paths else "\nINC=" + "|".join(include_paths)
    hash_key = hashlib.sha256(
        (entry + "\n" + _normalize_slang_source(src) + inc_tag).encode()
    ).hexdigest()
    refl_json = get_reflection_json(hash_key)
    if refl_json is None:
        return None
    layout = reflection_layout(refl_json)
    return list(layout.get("descriptor_counts") or [])


def _validate_no_null_storage(key: str, tensors: list[torch.Tensor]) -> bool:
    """PF.51 — fail-fast guard against null-storage (FakeTensor) leakage.

    A vulkan-tagged null-storage tensor (PF.13's ``make_vulkan_null``) reaches
    this layer only if an FX pass or wrapper-codegen step propagated a
    FakeTensor through to dispatch. The C++ layer would otherwise raise
    ``RuntimeError: Tensor has no backing Vulkan buffer`` with no
    indication of which arg was responsible. We surface the offender by
    name (and dispatch key) so the bug roots straight to the producing
    pipeline stage instead of to the runtime.
    """
    offenders: list[str] = []
    fake_count = 0
    vulkan_count = 0
    for i, t in enumerate(tensors):
        if t is None:
            continue
        # ``has_storage`` returns True for vulkan-null tensors (PF.13's
        # invariant — they carry a real Storage with a null DataPtr).
        # ``data_ptr() == 0`` is the canonical null-storage signal.
        try:
            dev_type = t.device.type
        except Exception:  # noqa: BLE001
            continue
        if dev_type not in ("vulkan", "privateuseone"):
            continue
        vulkan_count += 1
        try:
            ptr = t.data_ptr()
        except RuntimeError:
            # ``data_ptr`` raises on FakeTensor — tracing mode.
            fake_count += 1
            offenders.append(
                f"arg{i}: <FakeTensor> shape={list(t.shape)} dtype={t.dtype}"
            )
            continue
        if ptr == 0:
            offenders.append(
                f"arg{i}: shape={list(t.shape)} dtype={t.dtype} device={t.device}"
            )
    # If ALL Vulkan tensors are FakeTensors, we are in AOT Autograd tracing.
    # Skip the dispatch — outputs already have correct shapes.
    if fake_count > 0 and fake_count == vulkan_count:
        return True
    if offenders:
        raise RuntimeError(
            f"PF.51: vulkan-null-storage tensor reached dispatch '{key}' — "
            f"an FX pass or wrapper-codegen step is propagating a "
            f"FakeTensor through to the runtime. Offenders:\n  "
            + "\n  ".join(offenders)
        )
    return False


def dispatch(
    key: str,
    spirv: bytes,
    tensors: list[torch.Tensor],
    wg_x: int,
    wg_y: int = 1,
    wg_z: int = 1,
    push_constants: bytes = b"",
    num_outputs: int = 1,
) -> None:
    """Dispatch a pre-compiled SPIR-V compute shader on the Vulkan stream.

    Thin wrapper around the C++ `_jit_dispatch` pybind entry. Requires that
    every tensor is on the vulkan device and already contiguous.
    """
    if _TRACE:
        import sys

        print(
            f"[vk-jit] key={key} tensors={len(tensors)} wg=({wg_x},{wg_y},{wg_z}) "
            f"pc_bytes={len(push_constants)} num_out={num_outputs}",
            file=sys.stderr,
            flush=True,
        )
    if _validate_no_null_storage(key, tensors):
        return  # tracing mode, skip actual dispatch
    _get_jit_dispatch()(
        key, spirv, tensors, wg_x, wg_y, wg_z, push_constants, num_outputs
    )


def dispatch_indexed(
    key: str,
    spirv: bytes,
    tensors: list[torch.Tensor],
    descriptor_counts: list[int],
    wg_x: int,
    wg_y: int = 1,
    wg_z: int = 1,
    push_constants: bytes = b"",
    num_outputs: int = 1,
) -> None:
    """N+1.5.a — descriptor-array variant of :func:`dispatch`.

    Routes through the C++ ``_jit_dispatch_indexed`` pybind entry, which
    writes Vulkan descriptor sets with ``descriptorCount`` taken from
    ``descriptor_counts`` (parallel to the binding order). When every
    count is ``1`` the C++ runtime auto-falls-back to the flat path —
    callers may still want to use :func:`dispatch` directly for that
    case to skip the extra pybind list conversion.

    Raises ``RuntimeError`` when any count > 1 but
    ``VK_EXT_descriptor_indexing`` is unavailable on the device.
    """
    if any(c > 1 for c in descriptor_counts):
        if not _descriptor_indexing_supported():
            raise RuntimeError(
                f"N+1.5: dispatch '{key}' uses a descriptor-array binding "
                f"(descriptor_counts={list(descriptor_counts)}) but the "
                f"Vulkan runtime reports VK_EXT_descriptor_indexing is "
                f"unavailable on this device."
            )
    indexed_fn = _get_jit_dispatch_indexed()
    if indexed_fn is None:
        raise RuntimeError(
            f"N+1.5: `_jit_dispatch_indexed` FFI symbol not present. "
            f"Rebuild the C++ extension."
        )
    if _TRACE:
        import sys

        print(
            f"[vk-jit-idx] key={key} tensors={len(tensors)} "
            f"counts={list(descriptor_counts)} "
            f"wg=({wg_x},{wg_y},{wg_z}) pc_bytes={len(push_constants)} "
            f"num_out={num_outputs}",
            file=sys.stderr,
            flush=True,
        )
    if _validate_no_null_storage(key, tensors):
        return  # tracing mode, skip actual dispatch
    indexed_fn(
        key,
        spirv,
        tensors,
        list(int(c) for c in descriptor_counts),
        wg_x,
        wg_y,
        wg_z,
        push_constants,
        num_outputs,
    )


def compile_and_dispatch(
    src: str,
    tensors: list[torch.Tensor],
    wg_x: int,
    wg_y: int = 1,
    wg_z: int = 1,
    push_constants: bytes = b"",
    num_outputs: int = 1,
    entry: str = "computeMain",
    cache_key: str = "",
) -> None:
    """Compile Slang source (cached) and dispatch in one call.

    `cache_key` is required (used as both the SPIR-V cache key and the
    pipeline cache key). All in-tree callers supply one — the previous
    SHA1-of-SPIRV fallback was dead code.
    """
    if not cache_key:
        raise ValueError("compile_and_dispatch requires a non-empty cache_key")
    spv = compile_slang_to_spirv(src, entry=entry, cache_key=cache_key)
    dispatch(cache_key, spv, tensors, wg_x, wg_y, wg_z, push_constants, num_outputs)


# ═══════════════════════════════════════════════════════════════════════════
# GPU.1 — Batch Dispatch Submission
# ═══════════════════════════════════════════════════════════════════════════

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
    import math

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


class DispatchBatcher:
    """Batch multiple kernel dispatches into a single Vulkan submission.

    GPU.1 — When the wrapper codegen enters this context manager, calls to
    ``add(kernel_fn, *args)`` are collected instead of dispatched immediately.
    On ``__exit__``, all pending dispatches are submitted back-to-back with
    minimal Python overhead between them.

    Usage (emitted by wrapper codegen)::

        _batcher = DispatchBatcher()
        with _batcher:
            _batcher.add(kernel_0, arg0_0, arg0_1, ...)
            _batcher.add(kernel_1, arg1_0, arg1_1, ...)
        # All dispatches submitted on context exit.

    When the C++ ``_jit_dispatch_batch`` FFI is available, all dispatches are
    recorded into a single Vulkan command buffer and submitted with one
    ``vkQueueSubmit``.  Otherwise falls back to sequential individual dispatches
    (still beneficial: eliminates Python bytecode overhead between dispatches).
    """

    # Cached lookup of the batch FFI entry point (lazy, once per process).
    _batch_ffi = None
    _batch_ffi_probed: bool = False

    def __init__(self):
        self._pending: list[tuple] = []  # (kernel_callable, args_tuple)
        self._active: bool = False

    def __enter__(self):
        self._active = True
        self._pending.clear()
        return self

    def __exit__(self, *args):
        self._active = False
        if self._pending:
            self._flush()
        return False  # propagate exceptions

    def add(self, kernel_handle, *dispatch_args):
        """Collect a kernel dispatch for batched submission.

        When the batcher is active (inside a ``with`` block), the call is
        queued.  When inactive, dispatches immediately — this ensures
        correctness for callers that do not nest inside the batcher.
        """
        if self._active:
            self._pending.append((kernel_handle, dispatch_args))
        else:
            # Dispatch immediately (non-batched path).
            kernel_handle(*dispatch_args)

    def _flush(self):
        """Submit all pending dispatches.

        Tries the C++ ``_jit_dispatch_batch`` fast path first (single
        ``vkQueueSubmit`` for all kernels).  Falls back to sequential
        individual dispatches when the C++ FFI is unavailable.
        """
        if not self._pending:
            return

        # Try the C++ batch FFI path.
        batch_fn = self._resolve_batch_ffi()
        if batch_fn is not None:
            try:
                handles = [h for h, _ in self._pending]
                arg_lists = [list(a) for _, a in self._pending]
                batch_fn(handles, arg_lists)
                self._pending.clear()
                return
            except Exception:
                # Fall through to sequential path on any C++ error.
                pass

        # Sequential fallback: call each kernel in a tight loop.
        # Still beneficial vs. full Python wrapper overhead per dispatch.
        for kernel_handle, dispatch_args in self._pending:
            kernel_handle(*dispatch_args)
        self._pending.clear()

    @classmethod
    def _resolve_batch_ffi(cls):
        """Lazily resolve the ``_jit_dispatch_batch`` C++ FFI entry."""
        if cls._batch_ffi_probed:
            return cls._batch_ffi
        cls._batch_ffi_probed = True
        try:
            from torch_vulkan import _C as _c

            cls._batch_ffi = getattr(_c, "_jit_dispatch_batch", None)
        except Exception:
            cls._batch_ffi = None
        return cls._batch_ffi


def gc_spirv_cache(max_mib: int) -> dict:
    """Trim the on-disk SPIR-V cache to ``max_mib`` MiB by deleting LRU entries.

    Returns ``{"removed": int, "kept": int, "bytes_before": int, "bytes_after": int}``.
    Cache is sharded by 2-char hash prefix; we walk every ``.spv`` file, sort
    by mtime ascending, and delete oldest until under budget. Safe to call at
    any time — entries that get re-requested simply re-pay slangc once. P5.7.
    """
    if max_mib < 0:
        raise ValueError("max_mib must be >= 0")
    budget = max_mib * 1024 * 1024
    entries: list[tuple[float, int, str]] = []  # (mtime, size, path)
    if os.path.isdir(_DISK_CACHE_DIR):
        for shard in os.listdir(_DISK_CACHE_DIR):
            shard_dir = os.path.join(_DISK_CACHE_DIR, shard)
            if not os.path.isdir(shard_dir):
                continue
            for name in os.listdir(shard_dir):
                if not name.endswith(".spv"):
                    continue
                path = os.path.join(shard_dir, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                entries.append((st.st_mtime, st.st_size, path))
    bytes_before = sum(e[1] for e in entries)
    entries.sort(key=lambda e: e[0])  # oldest first
    removed = 0
    bytes_after = bytes_before
    for _, size, path in entries:
        if bytes_after <= budget:
            break
        try:
            os.unlink(path)
            bytes_after -= size
            removed += 1
        except OSError:
            continue
    return {
        "removed": removed,
        "kept": len(entries) - removed,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
    }


def make_vulkan_kernel(
    src: str,
    key: str,
    n_buffers: int | None,
    pc_size_bytes: int,
    n_pc: int,
    n_outputs: int = 1,
    config_key: str | None = None,
):
    """Build a generated-kernel wrapper that dispatches via _jit_dispatch.

    Uses the raw dispatch path (not cached) because pipeline pre-creation
    requires passing SPIR-V to _jit_pipeline which the closure doesn't store.

    N+1.5.a: when slangc reflection reports any binding with
    ``descriptorCount > 1`` (i.e. an array binding such as
    ``RWStructuredBuffer<T> arr[N]``), the closure routes through the
    descriptor-array FFI ``_jit_dispatch_indexed`` so each array slot
    receives its own buffer. Flat layouts (every count == 1) keep the
    original ``_jit_dispatch`` fast-path to avoid the extra pybind
    list-conversion per call.

    DR.3: ``config_key`` is threaded through to the SPIR-V compilation
    path so harvested reflection metrics are cross-indexed under this
    structural key, enabling ``_get_actual_vgprs`` to find cached data
    on subsequent compiles.
    """
    import struct

    pack = struct.Struct(f"{n_pc}I").pack if n_pc else None
    spv = compile_slang_to_spirv(src, cache_key=key, config_key=config_key)
    _KERNEL_SPIRV_HASH[key] = hashlib.sha256(spv).hexdigest()[:12]

    # ── D.3: Reflection-based buffer count ──
    # When n_buffers is None (or zero), derive the buffer count from
    # SPIR-V reflection so variable-arity kernels don't need a
    # hand-counted binding count.  The closure slices tensors from
    # ``args[:-(3+n_pc)]`` regardless, but callers can use this for
    # pre-dispatch validation or AOTI metadata.
    _n_buf = n_buffers
    if _n_buf is None or _n_buf == 0:
        _n_buf = get_reflected_binding_count(spv)
    # Cross-validate when both hand-count and reflection are available.
    if _n_buf is not None and _n_buf > 0 and n_buffers is not None and n_buffers > 0:
        if _n_buf != n_buffers:
            import warnings

            warnings.warn(
                f"D.3: kernel '{key}' hand-counted n_buffers={n_buffers} "
                f"but reflection reports {_n_buf} bindings. "
                f"Using reflection value."
            )
    # ── N+1.5.a: pick the dispatch FFI based on reflection ──
    # Reflection JSON is keyed by source-hash (see compile_slang_to_spirv);
    # the SPV-hash lookup will miss, so try source-hash first.
    descriptor_counts = _get_reflected_descriptor_counts_from_src(src)
    if descriptor_counts is None:
        descriptor_counts = get_reflected_descriptor_counts(spv)
    needs_indexed = bool(descriptor_counts) and any(c > 1 for c in descriptor_counts)
    if needs_indexed:
        if not _descriptor_indexing_supported():
            raise RuntimeError(
                f"N+1.5: kernel '{key}' uses a descriptor-array binding "
                f"(descriptor_counts={descriptor_counts}) but the Vulkan "
                f"runtime reports VK_EXT_descriptor_indexing is unavailable. "
                f"Either rebuild against a driver that exposes the extension, "
                f"or rewrite the shader to use only flat (descriptorCount=1) "
                f"bindings."
            )
        indexed_fn = _get_jit_dispatch_indexed()
        if indexed_fn is None:
            raise RuntimeError(
                f"N+1.5: kernel '{key}' needs `_jit_dispatch_indexed` but "
                f"the FFI symbol is missing. Rebuild the C++ extension."
            )
        # Freeze a tuple of uint32 for the closure (pybind picks up the
        # implicit conversion from list-of-int).
        dc = tuple(int(c) for c in descriptor_counts)
        if n_pc == 0:

            def kernel(*args):
                indexed_fn(
                    key,
                    spv,
                    list(args[:-3]),
                    list(dc),
                    args[-3],
                    args[-2],
                    args[-1],
                    b"",
                    n_outputs,
                )
        else:

            def kernel(*args):
                indexed_fn(
                    key,
                    spv,
                    list(args[: -(3 + n_pc)]),
                    list(dc),
                    args[-(3 + n_pc)],
                    args[-(3 + n_pc) + 1],
                    args[-(3 + n_pc) + 2],
                    pack(*args[-(3 + n_pc) + 3 : -3]) if pack else b"",
                    n_outputs,
                )

        stats_enabled = os.environ.get("TORCH_VULKAN_INDUCTOR_STATS") == "1"
        return kernel if not stats_enabled else _wrap_stats(key, kernel)

    # ── flat path: every binding has descriptorCount == 1 ──
    dispatch_fn = _get_jit_dispatch()

    # Build closure that captures key + spv + pack function
    if n_pc == 0:

        def kernel(*args):
            dispatch_fn(
                key, spv, list(args[:-3]), args[-3], args[-2], args[-1], b"", n_outputs
            )

        stats_enabled = os.environ.get("TORCH_VULKAN_INDUCTOR_STATS") == "1"
        return kernel if not stats_enabled else _wrap_stats(key, kernel)

    # With push constants. The wrapper (`kernel/header.py:call_kernel`)
    # emits ordered_args as ``[bufs..., sizevars..., dyn_numels..., wg_x,
    # wg_y, wg_z]`` — sizevars / dynamic numels come BEFORE the wg dims.
    # D.2.a (2026-05-09): the original slice math `args[-(3+n_pc):-(3+n_pc)+3]`
    # mistakenly treated PCs as if they were appended after the wg dims,
    # so for n_pc>0 the wg dims were read from the PC region and the PC
    # bytes came back empty (`args[-n_pc:-3]` is empty when n_pc<=3).
    # Fix: wg dims always live in the last 3 slots; PCs occupy the
    # n_pc slots immediately before.
    def kernel(*args):
        dispatch_fn(
            key,
            spv,
            list(args[: -(3 + n_pc)]),
            args[-3],
            args[-2],
            args[-1],
            pack(*args[-(3 + n_pc) : -3]) if pack else b"",
            n_outputs,
        )

    stats_enabled = os.environ.get("TORCH_VULKAN_INDUCTOR_STATS") == "1"
    return kernel if not stats_enabled else _wrap_stats(key, kernel)


def make_vulkan_kernel_via_aoti(
    src: str,
    key: str,
    n_buffers: int | None,
    pc_size_bytes: int,
    n_pc: int,
    n_outputs: int = 1,
):
    """PF.31 — same contract as ``make_vulkan_kernel`` but the dispatch
    closure routes through the C++ AOTI runtime ABI
    (``_aoti_make_kernel`` + ``_aoti_dispatch``) instead of the pybind
    JIT-pipeline path. The Python interpreter is still present to call the
    pybind wrappers — but the body of the closure is a single ABI call,
    matching what the AOTI-emitted C++ wrapper will do once PF.32 ships
    the SPV next to the `.so`. Used by the regression test that asserts
    a Python-free dispatch path.
    """
    import struct

    from torch_vulkan import _C as _c

    pack = struct.Struct(f"{n_pc}I").pack if n_pc else None
    spv = compile_slang_to_spirv(src, cache_key=key)
    _KERNEL_SPIRV_HASH[key] = hashlib.sha256(spv).hexdigest()[:12]
    # D.3: When n_buffers is None, derive from SPIR-V reflection.
    _nb_aoti = n_buffers
    if _nb_aoti is None:
        _nb_aoti = get_reflected_binding_count(spv)
        if _nb_aoti is None:
            _nb_aoti = _get_reflected_buffer_count_from_cache_key(src)
        if _nb_aoti is None:
            raise RuntimeError(
                "D.3: Cannot determine buffer count for kernel"
                f" '{key}' - reflection unavailable."
            )

    handle = _c._aoti_make_kernel(spv, key, _nb_aoti, pc_size_bytes)
    _no = n_outputs

    if n_pc == 0:

        def kernel(*args):
            tensors = list(args[:-3])
            _c._aoti_dispatch(handle, tensors, args[-3], args[-2], args[-1], b"", _no)
    else:
        pc_start = -3 - n_pc

        def kernel(*args):
            tensors = list(args[:pc_start])
            pc = pack(*args[pc_start:-3])
            _c._aoti_dispatch(handle, tensors, args[-3], args[-2], args[-1], pc, _no)

    kernel._aoti_handle = handle  # keep handle alive with the closure
    stats_enabled = os.environ.get("TORCH_VULKAN_INDUCTOR_STATS") == "1"
    if not stats_enabled:
        return kernel
    wrapped = _wrap_stats(key, kernel)
    wrapped._aoti_handle = handle
    return wrapped


# ── AOTI model export (P3.4) ────────────────────────────────────


def export_aoti_model(
    model: "torch.nn.Module",
    path: str,
    example_inputs: "tuple | None" = None,
) -> None:
    """Export a compiled model for AOTI deployment.

    Serializes compiled SPIR-V binaries, kernel metadata, buffer layouts,
    and dispatch order into a directory for later loading via
    ``_aoti_model_load``. The output is a directory containing:
        kernels.bin  — binary bundle of kernel SPIR-V + metadata
        metadata.json — human-readable dispatch order and buffer layouts

    The AOTI runtime can load this directory and execute all kernels
    without requiring the Python Inductor stack or slangc at runtime.

    Args:
        model: A ``torch.nn.Module`` compiled with the Vulkan Inductor backend.
        path: Output directory path. Created if it does not exist.
        example_inputs: Optional example inputs used to trigger tracing
                        if the model has not been pre-compiled.
    """
    import json
    import struct

    from torch_vulkan import _C as _c

    os.makedirs(path, exist_ok=True)

    # Collect all compiled kernels referenced by the model's generated code.
    # The Inductor wrapper calls make_vulkan_kernel which populates
    # _KERNEL_SPIRV_HASH. We walk the compile cache entries for each key.
    kernels: "list[dict]" = []
    seen_keys: "set[str]" = set()

    for key, spv_hash in _KERNEL_SPIRV_HASH.items():
        if key in seen_keys:
            continue
        seen_keys.add(key)

        # Try to get SPIR-V from disk cache or in-memory compile cache
        spv = None
        # Disk cache first (always available if previously compiled)
        spv = _disk_cache_read(key)
        if spv is None:
            # Re-compile if not cached
            sc = getattr(_get_jit_dispatch, "_source_cache", None)
            if sc is not None and key in sc:
                src = sc[key]
                spv = compile_slang_to_spirv(src, cache_key=key)
        if spv is None:
            raise RuntimeError(
                f"export_aoti_model: cannot find SPIR-V for kernel '{key}'. "
                f"Run the model forward at least once to compile kernels."
            )

        # Determine n_buffers from SPIR-V reflection
        n_buf = get_reflected_binding_count(spv)
        if n_buf is None:
            n_buf = _get_reflected_buffer_count_from_cache_key("") or 0

        kernels.append(
            {
                "key": key,
                "spv": spv,
                "n_buffers": n_buf,
                "pc_size_bytes": 0,
                "spv_hash": spv_hash,
            }
        )

    if not kernels:
        raise RuntimeError(
            "export_aoti_model: no compiled kernels found. "
            "Run the model forward at least once to compile kernels."
        )

    # Write kernels.bin
    bin_path = os.path.join(path, "kernels.bin")
    with open(bin_path, "wb") as f:
        f.write(b"vk_aoti\n")
        f.write(struct.pack("<I", len(kernels)))
        for k in kernels:
            spv = k["spv"]
            spv_words = len(spv) // 4
            key_bytes = k["key"].encode("utf-8")
            f.write(
                struct.pack(
                    "<IIII",
                    spv_words,
                    k["n_buffers"],
                    k["pc_size_bytes"],
                    len(key_bytes),
                )
            )
            f.write(key_bytes)
            f.write(spv)

    # Write metadata.json
    meta_path = os.path.join(path, "metadata.json")
    meta = {
        "version": 1,
        "kernel_count": len(kernels),
        "kernels": [
            {
                "key": k["key"],
                "spv_hash": k["spv_hash"],
                "n_buffers": k["n_buffers"],
                "spv_size_bytes": len(k["spv"]),
            }
            for k in kernels
        ],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Verify: load via C++ AOTI runtime, then free
    model_handle = _c._aoti_model_load(path)
    try:
        import sys

        total_spv = sum(len(k["spv"]) for k in kernels)
        print(
            f"[vk-aoti] export → {path}  kernels={len(kernels)}  "
            f"spv_total={total_spv / 1024:.1f} KiB",
            file=sys.stderr,
            flush=True,
        )
    finally:
        _c._aoti_model_free(model_handle)
