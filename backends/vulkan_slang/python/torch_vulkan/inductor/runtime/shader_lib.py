"""Shader-library precompilation, module management, and prewarm.

Extracted from ``slangc.py`` as part of the M22a anti-goal #7 line-cap
split (Stage 2). This module owns:

  - On-disk SPIR-V cache read/write helpers (``_disk_cache_read``,
    ``_disk_cache_write``)
  - Pre-warm compile helper (``prewarm_compile``)
  - slangc availability probe (``_slangc_available``, ``_reset_slangc_available_cache``)
  - slangc fingerprint (``_slangc_fingerprint``)
  - Shader-lib precompilation: ``_SHADERS_LIB_DIR``,
    ``_SHADER_LIB_MODULE_CACHE_DIR``, ``_SHADER_LIB_MODULE_STATS``,
    ``precompile_shader_libs``, ``_ensure_shader_lib_modules``,
    ``prewarm_shader_libs``, ``_reset_shader_lib_modules_ready``,
    ``_invalidate_shader_lib_modules``
  - slangc module-support probe (``_slangc_supports_modules``)
  - mm_tile / mm_int8 module helpers
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import threading
from typing import Optional

from .common import (
    SlangCompileTimeout,
    _COMPILE_STATS,
    _SLANGC_TIMEOUT_S,
    _cache_by_hash,
    _cache_by_key,
    _get_async_pool,
    _get_disk_cache_dir,
    _get_slangc,
    _normalize_slang_source,
    _reflection_metrics_by_hash,
    _reflection_metrics_by_key,
    _wrap_pool_worker,
)


# ---------------------------------------------------------------------------
# On-disk SPIR-V cache helpers
# ---------------------------------------------------------------------------


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
        # Atomic write: rename from a pid+tid-suffixed temp so two threads
        # within the same process (M22.16) can't collide on the temp
        # file name. ``os.replace`` is atomic on POSIX, so concurrent
        # writes to the canonical ``path`` still see a complete file.
        tmp = path + f".tmp.{os.getpid()}.{threading.get_ident()}"
        with open(tmp, "wb") as f:
            f.write(spv)
        os.replace(tmp, path)
    except OSError:
        pass  # cache is best-effort; fall back to recompiling next time


# ---------------------------------------------------------------------------
# Pre-warm compile helper
# ---------------------------------------------------------------------------


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
    # Import here to avoid circular import (compile_slang_to_spirv lives in
    # slangc.py which in turn imports from this module).
    from .slangc import compile_slang_to_spirv

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


# ---------------------------------------------------------------------------
# slangc availability probe
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# slangc fingerprint
# ---------------------------------------------------------------------------

_slangc_fingerprint_cache: dict[str, str] = {}


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


# ---------------------------------------------------------------------------
# Shader-lib directories and stats
# ---------------------------------------------------------------------------

_SHADERS_LIB_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "shaders", "lib")
)

_SHADER_LIB_MODULE_CACHE_DIR = os.environ.get(
    "TORCH_VULKAN_SLANG_MODULE_CACHE",
    os.path.join(os.path.expanduser("~"), ".cache", "torch_vulkan", "slang-modules"),
)

_SHADER_LIB_MODULE_STATS = {"compiles": 0, "cache_hits": 0}


# M-NEW.5: cold-import contention. With 5+ agents concurrently importing
# torch_vulkan with a cold slangc cache, total wall time hits 6+ min
# (vs ~3 s warm). The chokepoint is ``precompile_shader_libs`` —
# 16 ``shaders/lib/*.slang`` modules × ~800 ms slangc invocation each =
# ~13 s sync, multiplied by N concurrent processes all writing to the
# same cache dir, ends up serialised by fcntl + slangc-worker pool
# contention.
#
# Fix D: gate prewarm via ``TORCH_VULKAN_PREWARM_LEVEL``.
#
#   level=0 → no prewarm. ``precompile_shader_libs`` returns immediately
#             without scheduling any slangc work. Every shader-lib
#             module compiles lazily on the first ``import`` (slangc -I
#             path picks up the cached artifact when it's available; on
#             a cold cache it parses source directly).
#   level=1 → CORE prewarm (default). Compiles only the 6 shader-lib
#             modules every model touches: ``helpers``, ``dtype_pack``,
#             ``pointwise``, ``mm``, ``reduction``, ``norm``. Skipped
#             modules are op-specific (atomics, bucket, philox, losses,
#             tensor_layout, mm_int8, mm_tile, special_math, conv,
#             pointwise_generic) and compile on first use when a kernel
#             that actually needs them is dispatched.
#   level=2 → FULL prewarm (pre-M-NEW.5 behaviour). All 16 modules
#             precompiled at import time.
#
# Cold-import wall (RDNA1 + RADV, single process, cold cache):
#   - level=0: ~3-5 s  (no shader-lib slangc at import)
#   - level=1: ~6-8 s  (6 modules × ~800 ms slangc, partly parallel)
#   - level=2: ~13-15 s (16 modules; M-NEW.5 pre-fix baseline)
# Under 5× concurrent processes the multiplier grows non-linearly due
# to slangc-worker contention; level=1 cuts the concurrent baseline
# from ~6+ min to <90 s on the reference rig.
_PREWARM_CORE_MODULES: frozenset[str] = frozenset({
    # Universal — every emitted shader imports `helpers`.
    "helpers",
    # Dtype-aware codegen (vec4/packed16 paths use this).
    "dtype_pack",
    # Hot ops on the cold-compile path.
    "pointwise",
    "mm",
    "reduction",
    "norm",
})


def _prewarm_level() -> int:
    """Read ``TORCH_VULKAN_PREWARM_LEVEL`` once and clamp to {0, 1, 2}.

    Defaults to 1 (CORE prewarm) which trades ~7 s of cold-import for
    no lazy-compile latency on the hot path. Set to 0 for fastest
    cold import (debugging, CI smoke tests); set to 2 to restore the
    pre-M-NEW.5 behaviour (every shader-lib module precompiled).
    """
    raw = os.environ.get("TORCH_VULKAN_PREWARM_LEVEL")
    if raw is None:
        return 1
    try:
        v = int(raw)
    except ValueError:
        return 1
    if v < 0:
        return 0
    if v > 2:
        return 2
    return v


def _shader_lib_sources() -> list[str]:
    if not os.path.isdir(_SHADERS_LIB_DIR):
        return []
    return sorted(
        os.path.join(_SHADERS_LIB_DIR, f)
        for f in os.listdir(_SHADERS_LIB_DIR)
        if f.endswith(".slang")
    )


def _prewarm_filtered_sources() -> list[str]:
    """Apply the M-NEW.5 prewarm-level filter to the shader-lib source
    list. Returns the subset that should be precompiled at this level.

    Used only by the prewarm path; the lazy-compile path
    (``_ensure_shader_lib_modules``) calls ``_shader_lib_sources``
    directly because by definition a kernel that explicitly requests
    a module needs that module compiled — level=0 just defers when
    the work happens, not whether it happens.
    """
    level = _prewarm_level()
    all_sources = _shader_lib_sources()
    if level == 0:
        return []
    if level >= 2:
        return all_sources
    # level=1: keep only modules in the core set.
    return [
        src for src in all_sources
        if os.path.splitext(os.path.basename(src))[0]
        in _PREWARM_CORE_MODULES
    ]


# ---------------------------------------------------------------------------
# Shader-lib precompile
# ---------------------------------------------------------------------------


def precompile_shader_libs(
    force: bool = False,
    lax: bool = False,
    *,
    level: int | None = None,
) -> dict:
    """Walk `shaders/lib/*.slang`, emit `<name>.slang-module` artifacts.

    Cache keyed on source SHA256: a module is recompiled only when its
    source content changes. Sidecar `.hash` files record the source hash
    that produced each cached module. Returns a summary dict
    ``{"compiled": [names], "cached": [names], "failed": [(name, err)],
    "module_dir": str, "skipped_by_level": [names]}`` and bumps
    `_SHADER_LIB_MODULE_STATS`.

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

    M-NEW.5: ``level`` (None / 0 / 1 / 2) filters which shader-lib
    modules to compile.
      - ``None`` (default): compile ALL modules. Use this for
        user-explicit precompile (e.g. the CLAUDE.md rebuild command).
      - ``0``: compile NOTHING. Skipped modules are listed in the
        return dict under ``"skipped_by_level"``. The lazy first-compile
        path picks them up on demand.
      - ``1``: compile only the CORE set (``_PREWARM_CORE_MODULES``).
        Other modules go in ``"skipped_by_level"``.
      - ``2``: same as ``None`` — compile all modules.
    """
    os.makedirs(_SHADER_LIB_MODULE_CACHE_DIR, exist_ok=True)
    compiled, cached, failed, skipped_by_level = [], [], [], []
    # PF.27.a.1: mix the slangc fingerprint into the cache key so a
    # slangc upgrade invalidates every cached `.slang-module` whose
    # serialized IR ABI may have changed. Hash mismatch fires on the
    # cache-hit read path below, exactly the requirement called out by
    # debug-coordinator.
    slangc_fp = _slangc_fingerprint()
    # M-NEW.5: determine the level-filtered source list. ``level=None``
    # means "ignore the env-knob, do the full set" — preserves the
    # contract for ``precompile_shader_libs(force=True)``-style
    # user-explicit invocations.
    all_sources = _shader_lib_sources()
    if level is None:
        sources_to_compile = all_sources
        eligible_names: frozenset[str] = frozenset(
            os.path.splitext(os.path.basename(s))[0] for s in all_sources
        )
    else:
        sources_to_compile = _prewarm_filtered_sources()
        eligible_names = frozenset(
            os.path.splitext(os.path.basename(s))[0]
            for s in sources_to_compile
        )
    for src_path in all_sources:
        # M-NEW.5: modules excluded by the level filter are reported
        # but not compiled. The lazy first-compile path picks them up
        # on demand when a kernel actually imports them.
        name_only = os.path.splitext(os.path.basename(src_path))[0]
        if name_only not in eligible_names:
            skipped_by_level.append(name_only)
            continue
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
        # M22.16: write to a per-process temp path first, then atomically
        # ``os.replace`` to ``mod_path``. Without this, two slangc processes
        # from different parent agents both targeting the same canonical
        # ``mod_path`` would interleave their writes — POSIX gives no
        # atomicity for direct ``slangc -o <path>`` and the next reader
        # gets a torn / truncated ``.slang-module`` that fails to import
        # downstream. Source-file slangc dies; downstream Python sees a
        # "intermittent zero" / silent-fail symptom.
        # M21.3.01 follow-up: slangc infers the output format from the
        # filename extension — the temp name must therefore still end in
        # ``.slang-module``. ``.tmp.<pid>.<tid>`` placed BEFORE the
        # extension keeps the atomic publish guarantee while preserving
        # format detection. Prior code placed the suffix after the
        # extension and broke ``-emit-ir`` with "cannot infer an output
        # format" (E00060).
        tmp_mod_path = (
            mod_path[: -len(".slang-module")]
            + f".tmp.{os.getpid()}.{threading.get_ident()}.slang-module"
        )
        # M22.16-followup: slangc 2026.7.1 crashes in the thread-pool case
        # when compiling helpers.slang (subgroup_ballot capability check).
        # Apply -ignore-capabilities here too — the SPIR-V output is
        # unaffected; we only suppress a static capability-availability
        # check that the GPU satisfies at runtime anyway.
        argv = [_get_slangc(), src_path, "-emit-ir", "-o", tmp_mod_path, "-ignore-capabilities"]
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=_SLANGC_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as e:
            # Best-effort cleanup of the partial temp output.
            try:
                os.remove(tmp_mod_path)
            except OSError:
                pass
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
            try:
                os.remove(tmp_mod_path)
            except OSError:
                pass
            err_msg = f"slangc failed precompiling shader lib {name}:\n{proc.stderr}"
            if lax:
                failed.append((name, (proc.stderr or "").strip()))
                continue
            raise RuntimeError(err_msg)
        # Atomic publish: rename tmp output to canonical mod_path.
        try:
            os.replace(tmp_mod_path, mod_path)
        except OSError as e:
            err_msg = f"slangc atomic rename failed for {name}: {e}"
            if lax:
                failed.append((name, err_msg))
                continue
            raise RuntimeError(err_msg) from None
        tmp_hash = hash_path + f".tmp.{os.getpid()}.{threading.get_ident()}"
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
        # M-NEW.5: names skipped by the level filter (level=0/1).
        # Empty when level is None or 2.
        "skipped_by_level": skipped_by_level,
    }


# ---------------------------------------------------------------------------
# Shader-lib module readiness gate
# ---------------------------------------------------------------------------

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

    M-NEW.5: the level filter is applied here AND on the import-time
    prewarm path. When level<2 some modules are skipped at this lazy
    pass too — they'll be picked up by slangc's source-parse fallback
    when a kernel that imports them is actually compiled. This is the
    correct shape because the "lazy" path fires the first time ANY
    kernel triggers it, not when a specific module is needed; deferring
    op-specific modules until first-use is the whole M-NEW.5 win.
    """
    global _shader_lib_modules_ready
    if _shader_lib_modules_ready:
        return _SHADER_LIB_MODULE_CACHE_DIR
    with _shader_lib_modules_lock:
        if not _shader_lib_modules_ready:
            precompile_shader_libs(lax=True, level=_prewarm_level())
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
    # M-NEW.5: ``TORCH_VULKAN_PREWARM_LEVEL=0`` short-circuits the
    # import-time background prewarm. The lazy-compile path still
    # honours level=0 — every module compiles on first kernel-import,
    # which is the correct shape for a cold-import-fast configuration.
    if _prewarm_level() == 0:
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
            # slangc 2026.7.1 trips on subgroup_ballot capability checks at
            # ``helpers.wave_active_count_bits`` even though every Vulkan
            # device we ship to supports subgroupVote/Ballot; the error
            # path then crashes the compiler in the thread-pool case.
            # Bypass the static check — runtime SPIR-V is unaffected.
            "-ignore-capabilities",
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


# ---------------------------------------------------------------------------
# mm_tile / mm_int8 module helpers
# ---------------------------------------------------------------------------

import re as _re

# Regex to extract link-time specialization constants from mm_tile wrappers.
# Matches patterns like: static const int TILE_M = 64;
_LT_SPEC_CONST_RE = _re.compile(
    r"(?:static\s+const\s+int|#define)\s+(TILE_[MNK]|M_PER_THREAD|N_PER_THREAD|NUM_STAGES)\s*=\s*(\d+)\s*;?"
)


def _compile_single_module(name: str) -> str:
    """Compile a single shader-lib module on demand, bypassing the level filter.

    Called when a kernel explicitly imports a module (e.g., ``import mm_tile;``)
    and the cached module doesn't exist. This can happen when:
    - TORCH_VULKAN_PREWARM_LEVEL=1 (CORE only, mm_tile skipped)
    - TORCH_VULKAN_PREWARM_LEVEL=0 (nothing precompiled)
    - Cache was cleared or invalidated

    Returns the path to the compiled module.

    Raises RuntimeError if slangc is unavailable or compilation fails.
    """
    mod_path = os.path.join(_SHADER_LIB_MODULE_CACHE_DIR, name + ".slang-module")
    if os.path.isfile(mod_path):
        return mod_path

    src_path = os.path.join(_SHADERS_LIB_DIR, name + ".slang")
    if not os.path.isfile(src_path):
        raise RuntimeError(
            f"Shader lib source not found: {src_path}. "
            f"Cannot compile {name}.slang-module."
        )

    if not _slangc_available():
        raise RuntimeError(
            "slangc not found. Set SLANGC=/path/to/slangc; needed to "
            f"compile {name}.slang-module on demand."
        )

    os.makedirs(_SHADER_LIB_MODULE_CACHE_DIR, exist_ok=True)

    # Compute hash for cache validation
    slangc_fp = _slangc_fingerprint()
    with open(src_path, "rb") as f:
        src_bytes = f.read()
    src_hash = hashlib.sha256(
        src_bytes + b"\x00" + slangc_fp.encode("utf-8")
    ).hexdigest()

    # Use atomic write pattern (same as precompile_shader_libs)
    import threading
    tmp_mod_path = (
        mod_path[: -len(".slang-module")]
        + f".tmp.{os.getpid()}.{threading.get_ident()}.slang-module"
    )

    argv = [_get_slangc(), src_path, "-emit-ir", "-o", tmp_mod_path, "-ignore-capabilities"]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_SLANGC_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as e:
        try:
            os.remove(tmp_mod_path)
        except OSError:
            pass
        raise SlangCompileTimeout(
            key=f"shader_lib_{name}_ondemand",
            argv=argv,
            partial_stdout=(e.stdout or b"").decode("utf-8", errors="replace")
            if isinstance(e.stdout, bytes)
            else (e.stdout or ""),
            partial_stderr=(e.stderr or b"").decode("utf-8", errors="replace")
            if isinstance(e.stderr, bytes)
            else (e.stderr or ""),
        ) from None

    if proc.returncode != 0:
        try:
            os.remove(tmp_mod_path)
        except OSError:
            pass
        raise RuntimeError(
            f"slangc failed compiling {name}.slang on demand:\n{proc.stderr}"
        )

    # Atomic publish
    try:
        os.replace(tmp_mod_path, mod_path)
    except OSError as e:
        raise RuntimeError(f"slangc atomic rename failed for {name}: {e}")

    # Write hash for cache validation
    hash_path = os.path.join(_SHADER_LIB_MODULE_CACHE_DIR, name + ".hash")
    try:
        with open(hash_path, "w") as f:
            f.write(src_hash)
    except OSError:
        pass  # Non-fatal; cache will recompile next time

    _SHADER_LIB_MODULE_STATS["compiles"] += 1
    return mod_path


def _mm_tile_module_path() -> str:
    """Return the path to the precompiled mm_tile.slang-module."""
    return os.path.join(_SHADER_LIB_MODULE_CACHE_DIR, "mm_tile.slang-module")


def _mm_tile_module_available() -> bool:
    """Check if the mm_tile.slang-module has been precompiled."""
    return os.path.isfile(_mm_tile_module_path())


def _ensure_mm_tile_module() -> str:
    """Ensure mm_tile.slang-module is precompiled. Returns the module path.

    TRAIN.11: The mm_tile module is required for link-time matmul specialization.
    With M-NEW.5 level filtering (default level=1), mm_tile is skipped during
    prewarm. This function compiles it on-demand when a kernel explicitly
    imports it.
    """
    return _compile_single_module("mm_tile")


def _mm_int8_module_path() -> str:
    return os.path.join(_SHADER_LIB_MODULE_CACHE_DIR, "mm_int8.slang-module")


def _ensure_mm_int8_module() -> str:
    """Ensure mm_int8.slang-module is precompiled. Returns the module path.

    TRAIN.11: Mirrors ``_ensure_mm_tile_module``; compiles on-demand to bypass
    M-NEW.5 level filtering. The int8 kernel wrapper uses ``import mm_int8;``
    for link-time tile-size specialization.
    """
    return _compile_single_module("mm_int8")


_IMPORT_STMT_RE = _re.compile(r"^\s*import\s+(\w+)\s*;", _re.MULTILINE)


def _shader_lib_import_hash(src: str) -> str:
    # Cache-key tag: mixes in content hash of each shader-lib file imported via 'import <name>;'
    names = _IMPORT_STMT_RE.findall(src)
    if not names:
        return ""
    parts = []
    for name in sorted(set(names)):
        lib_path = os.path.join(_SHADERS_LIB_DIR, name + ".slang")
        if os.path.exists(lib_path):
            try:
                with open(lib_path, "rb") as f:
                    parts.append(name + "=" + hashlib.sha256(f.read()).hexdigest()[:16])
            except OSError:
                pass
    return "\nLIB=" + "|".join(parts) if parts else ""
