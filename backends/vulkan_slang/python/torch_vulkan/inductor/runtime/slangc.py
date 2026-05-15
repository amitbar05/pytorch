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
    # Resolve relative to backend root (slangc.py is at
    # backends/vulkan_slang/python/torch_vulkan/inductor/runtime/slangc.py
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


def _get_disk_cache_dir() -> str:
    """Return _DISK_CACHE_DIR, resolving via the package for monkeypatch compatibility.

    Tests monkeypatch ``torch_vulkan.inductor.runtime._DISK_CACHE_DIR``;
    since this module is now a sub-package, the patch lands on
    ``__init__.py``.  This getter checks there first, falling back to
    the module-level default.
    """
    import sys

    pkg = sys.modules.get("torch_vulkan.inductor.runtime")
    if pkg is not None:
        val = getattr(pkg, "_DISK_CACHE_DIR", None)
        if val is not None:
            return val
    return _DISK_CACHE_DIR


def _disk_cache_read(hash_key: str) -> Optional[bytes]:
    path = os.path.join(_get_disk_cache_dir(), hash_key[:2], hash_key[2:] + ".spv")
    try:
        with open(path, "rb") as f:
            return f.read()
    except (FileNotFoundError, OSError):
        return None


def _disk_cache_write(hash_key: str, spv: bytes) -> None:
    path = os.path.join(_get_disk_cache_dir(), hash_key[:2], hash_key[2:] + ".spv")
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
            [_get_slangc(), "-help"],
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
        st = os.stat(_get_slangc())
        stat_key = f"{os.path.realpath(_get_slangc())}:{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return f"unresolved:" + _get_slangc()
    version = _slangc_fingerprint_cache.get(stat_key)
    if version is None and _slangc_available():
        try:
            proc = subprocess.run(
                [_get_slangc(), "--version"],
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
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "shaders", "lib")
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


def precompile_shader_libs(force: bool = False, lax: bool = False) -> dict:
    """Walk `shaders/lib/*.slang`, emit `<name>.slang-module` artifacts.

    Cache keyed on source SHA256: a module is recompiled only when its
    source content changes. Sidecar `.hash` files record the source hash
    that produced each cached module. Returns a summary dict
    ``{"compiled": [names], "cached": [names], "failed": [(name, err)],
    "module_dir": str}`` and bumps `_SHADER_LIB_MODULE_STATS`.

    Output dir is `~/.cache/torch_vulkan/slang-modules` (override via
    ``TORCH_VULKAN_SLANG_MODULE_CACHE``). The same dir is added to slangc's
    `-I` so kernels that `import <name>;` resolve to the precompiled module
    instead of re-parsing the source.

    ``lax=True`` (M9.3): per-file slangc failures are collected into the
    ``failed`` list instead of raising. A file that fails precompile is
    not in ``compiled`` or ``cached``, so any kernel that ``import``-s it
    falls back to slangc's source-parse path (slower but correct).
    Strict mode (the default) preserves the historical contract — first
    failed file raises ``RuntimeError``.
    """
    os.makedirs(_SHADER_LIB_MODULE_CACHE_DIR, exist_ok=True)
    compiled, cached, failed = [], [], []
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
        argv = [_get_slangc(), src_path, "-emit-ir", "-o", mod_path]
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
            err_msg = f"slangc failed precompiling shader lib {name}:\n{proc.stderr}"
            if lax:
                failed.append((name, (proc.stderr or "").strip()))
                continue
            raise RuntimeError(err_msg)
        tmp_hash = hash_path + f".tmp.{os.getpid()}"
        with open(tmp_hash, "w") as f:
            f.write(src_hash)
        os.replace(tmp_hash, hash_path)
        compiled.append(name)
        _SHADER_LIB_MODULE_STATS["compiles"] += 1
    return {
        "compiled": compiled,
        "cached": cached,
        "failed": failed,
        "module_dir": _SHADER_LIB_MODULE_CACHE_DIR,
    }


_shader_lib_modules_ready = False
# M9.3: serialise the precompile pass so the import-time background prewarm
# (`prewarm_shader_libs`) cannot race the lazy first-compile path
# (`_ensure_shader_lib_modules`). Both write the same `.slang-module`
# artifacts via `subprocess.run(..., -o path)`; concurrent writes from two
# slangc processes targeting the same path are not atomic on POSIX and
# would corrupt the module cache.
_shader_lib_modules_lock = __import__("threading").Lock()


def _ensure_shader_lib_modules() -> str:
    """Lazy precompile on first kernel compile; returns module cache dir.

    Uses ``lax=True`` (M9.3): a per-file precompile failure becomes a
    cache miss for that one lib rather than aborting the whole pass,
    so a stale/dead lib (e.g. `norm.slang` references the legacy
    ``wg_reduce<…>`` 3-arg API) cannot block kernel compilation for
    every other module. Kernels that ``import`` a failed lib fall
    through to slangc's source-parse path — slower but correct.
    """
    global _shader_lib_modules_ready
    if _shader_lib_modules_ready:
        return _SHADER_LIB_MODULE_CACHE_DIR
    with _shader_lib_modules_lock:
        if not _shader_lib_modules_ready:
            precompile_shader_libs(lax=True)
            _shader_lib_modules_ready = True
    return _SHADER_LIB_MODULE_CACHE_DIR


def prewarm_shader_libs(*, sync: bool = False) -> bool:
    """M9.3: ensure the shader-lib module cache is populated.

    Called from the Inductor-backend registration step (import time) so
    the 8× ~800 ms slangc invocations that would otherwise fire on the
    first ``torch.compile`` dispatch happen before any user code runs.

    ``sync=False`` (the default) spawns a daemon background thread; the
    first kernel compile waits on `_shader_lib_modules_lock` if the
    background pass is still in flight, otherwise short-circuits via
    ``_shader_lib_modules_ready``. ``sync=True`` blocks until the
    precompile finishes — used by tests that need a populated cache.

    Returns True if a precompile pass was scheduled; False when
    disabled (``TORCH_VULKAN_NO_PREWARM=1``) or slangc is unavailable.
    """
    if os.environ.get("TORCH_VULKAN_NO_PREWARM") == "1":
        return False
    if not _slangc_available():
        return False
    if _shader_lib_modules_ready:
        return False

    if sync:
        _ensure_shader_lib_modules()
        return True

    import threading

    def _bg():
        try:
            _ensure_shader_lib_modules()
        except Exception:
            # Best-effort; the lazy path will retry on first compile.
            pass

    threading.Thread(target=_bg, daemon=True).start()
    return True


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
                    _get_slangc(),
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
    return os.path.join(
        _get_disk_cache_dir(), hash_key[:2], hash_key[2:] + ".refl.json"
    )


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
    return os.path.join(
        _get_disk_cache_dir(), hash_key[:2], hash_key[2:] + ".metrics.json"
    )


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
    from .. import config

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
    from .. import config

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

# DR.7 / M11.1: After Pass-2 optimizes numthreads, store the final value
# so the dispatch grid can be divided by the actual WG size instead of the
# codegen-time estimate.  Keyed by hash_key (same key as the SPV cache).
_optimized_numthreads_by_hash: dict[str, tuple[int, int, int]] = {}


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


def get_optimized_numthreads(hash_key: str) -> tuple[int, int, int] | None:
    """DR.7 / M11.1: Return the Pass-2 optimized numthreads for a kernel.

    Returns ``(x, y, z)`` if a Pass-2 recompile succeeded and stored
    optimized numthreads, or ``None`` if no optimization was applied.
    Callers (e.g. dispatch-grid codegen) use this to divide total numel
    by the ACTUAL workgroup size instead of the codegen-time estimate.
    """
    return _optimized_numthreads_by_hash.get(hash_key)


def _pick_numthreads_from_reflection(
    vgprs: int | None,
    shared_mem: int | None = None,
    loop_depth: int | None = None,
    current_numthreads: tuple[int, int, int] = (256, 1, 1),
) -> tuple[int, int, int]:
    """DR.7 / M11.1: Pick optimal numthreads based on SPIR-V reflection metrics.

    RDNA1 occupancy heuristic (wave64, 256 VGPRs/CU, 1024 max threads/CU,
    64 KiB LDS/CU):

    - VGPRs ≤ 32  → 256 threads (4 waves/CU → high occupancy)
    - VGPRs 33–64 → 128 threads (2 waves/CU → balance)
    - VGPRs 65–128 → 64 threads  (1 wave/CU → avoid register spilling)
    - VGPRs > 128  → 32 threads  (minimum occupancy, avoid scratch)

    When *shared_mem* exceeds 16 KiB (25 % of LDS budget), drop one tier
    to leave headroom for groupshared allocations.

    When *loop_depth* ≥ 3, drop one tier (deep loops increase register
    pressure beyond what the VGPR count captures). When ≥ 5, drop two.

    Falls back to *current_numthreads* when *vgprs* is ``None``.

    Args:
        vgprs: VGPR count from slangc reflection (numRegisters / usedRegisters).
        shared_mem: Groupshared / LDS bytes used (from reflection).
        loop_depth: Maximum nested loop depth (from reflection).
        current_numthreads: The numthreads currently in the source.

    Returns:
        Optimal ``(x, y, z)`` numthreads tuple.
    """
    if vgprs is None:
        return current_numthreads

    # Base tier from VGPR count
    if vgprs <= 32:
        base = 256
    elif vgprs <= 64:
        base = 128
    elif vgprs <= 128:
        base = 64
    else:
        base = 32

    # M11.1: Shared-memory penalty — when LDS usage is high, drop a tier
    # to leave more LDS per workgroup for groupshared allocations.
    if shared_mem is not None and shared_mem > 16384:  # > 16 KiB
        if base == 256:
            base = 128
        elif base == 128:
            base = 64

    # M11.1: Loop-depth penalty — deep nests blow register pressure.
    if loop_depth is not None:
        if loop_depth >= 5:
            for _ in range(2):
                if base == 256:
                    base = 128
                elif base == 128:
                    base = 64
                elif base == 64:
                    base = 32
        elif loop_depth >= 3:
            if base == 256:
                base = 128
            elif base == 128:
                base = 64
            elif base == 64:
                base = 32

    return (base, 1, 1)


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
            _get_slangc(),
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
                _get_slangc(),
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
                                _get_slangc(),
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
                                    # M11.1: Store optimized numthreads for dispatch grid
                                    _optimized_numthreads_by_hash[hash_key] = optimal_nt
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
                        else:
                            # M11.1: Store Pass-1 numthreads when no change needed
                            _optimized_numthreads_by_hash[hash_key] = current_nt
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
    # Lazy import from sibling reflection module within the runtime package.
    from .reflection import get_reflection_json, reflection_layout

    refl = get_reflection_json(hash_key)
    if refl is None:
        return spv, {"bindings": [], "push_constant_size": 0}
    return spv, reflection_layout(refl)


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
    cache_dir = _get_disk_cache_dir()
    if os.path.isdir(cache_dir):
        for shard in os.listdir(cache_dir):
            shard_dir = os.path.join(cache_dir, shard)
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
