"""Runtime shim between Inductor-generated Slang source and Vulkan dispatch.

Compiles Slang source to SPIR-V via `slangc` subprocess, caches by source
hash, then dispatches the compute shader via the C++ `_jit_dispatch` pybind
entry point. The in-memory cache is augmented by an on-disk cache
(``~/.cache/torch_vulkan/spirv/``, overridable via ``TORCH_VULKAN_SPIRV_CACHE``)
so subsequent Python sessions bypass slangc entirely once a kernel has been
compiled.

M22a Stage 1: module-level shared state extracted to ``common.py``.
M22a Stage 2: shader-lib precompile / module management extracted to
``shader_lib.py``.
M22a Stage 3: reflection metrics / SPIR-V baseline / numthreads cluster
extracted to ``reflection_ext.py``.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch

# M22a Stage 1: module-level shared state and infrastructure extracted to
# common.py. Import everything we need from there so external callers that
# do ``from torch_vulkan.inductor.runtime.slangc import X`` still work.
from .common import (  # noqa: F401
    SlangCompileTimeout,
    _ASYNC_COMPILE,
    _ASYNC_MAX_WORKERS,
    _ASYNC_POOL,
    _COMPILE_STATS,
    _DISK_CACHE_DIR,
    _INDUCTOR_STATS,
    _MAX_SLANGC_WORKERS,
    _PARALLEL_COMPILE,
    _SLANG_BLANK_LINES_RE,
    _SLANG_BLOCK_COMMENT_RE,
    _SLANG_LINE_COMMENT_RE,
    _SLANG_TRAILING_WS_RE,
    _SLANGC,
    _SLANGC_TIMEOUT_S,
    _TRACE,
    _cache_by_hash,
    _cache_by_key,
    _cache_lock,
    _default_max_workers,
    _device_subgroup_size_tag,
    _get_async_pool,
    _get_device_subgroup_size_tag,
    _get_disk_cache_dir,
    _get_slangc,
    _in_flight,
    _in_flight_lock,
    _is_in_pool_worker,
    _normalize_slang_source,
    _pool_local,
    _reflection_metrics_by_hash,
    _reflection_metrics_by_key,
    _resolve_slangc,
    _spv_baselines,
    _spv_baselines_loaded,
    _wrap_pool_worker,
    cache_hit_rate,
    reset_compile_stats,
)

# M22a Stage 2: shader-lib precompile / module management extracted to
# shader_lib.py. Re-export everything so external callers are unchanged.
from .shader_lib import (  # noqa: F401
    _LT_SPEC_CONST_RE,
    _PREWARM_CORE_MODULES,
    _SHADER_LIB_MODULE_CACHE_DIR,
    _SHADER_LIB_MODULE_STATS,
    _SHADERS_LIB_DIR,
    _disk_cache_read,
    _disk_cache_write,
    _ensure_mm_int8_module,
    _ensure_mm_tile_module,
    _ensure_shader_lib_modules,
    _invalidate_shader_lib_modules,
    _mm_int8_module_path,
    _mm_tile_module_available,
    _mm_tile_module_path,
    _shader_lib_import_hash,
    _prewarm_filtered_sources,
    _prewarm_level,
    _reset_shader_lib_modules_ready,
    _reset_slangc_available_cache,
    _shader_lib_modules_lock,
    _shader_lib_modules_ready,
    _shader_lib_sources,
    _slangc_available,
    _slangc_available_cache,
    _slangc_fingerprint,
    _slangc_fingerprint_cache,
    _slangc_modules_available,
    _slangc_supports_modules,
    precompile_shader_libs,
    prewarm_compile,
    prewarm_shader_libs,
)

# M22a Stage 3: reflection metrics / SPIR-V baseline / numthreads cluster
# extracted to reflection_ext.py. Re-export everything so external callers
# are unchanged.
from .reflection_ext import (  # noqa: F401
    _NUMTHREADS_SRC_RE,
    _SPV_BASELINES_PATH,
    _analyze_slang_source_for_loop_depth,
    _analyze_spirv_binary,
    _build_specialize_const_args,
    _check_spv_regression,
    _disk_metrics_path,
    _disk_metrics_read,
    _disk_metrics_write,
    _disk_reflection_path,
    _disk_reflection_read,
    _disk_reflection_write,
    _extract_linktime_spec_constants,
    _harvest_reflection_metrics,
    _load_spv_baselines,
    _optimized_numthreads_by_hash,
    _parse_numthreads_from_source,
    _parse_reflection_metrics,
    _pick_numthreads_from_reflection,
    _reflection_cache,
    _rewrite_numthreads_in_source,
    _save_spv_baselines,
    get_cached_metrics_for_key,
    get_optimized_numthreads,
    get_reflection_metrics,
    reset_reflection_baselines,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Modular compilation helpers (extracted from the monolithic inner function)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_compile_command(
    *,
    src_path: str,
    entry: str,
    out_path: str,
    refl_path: str,
    module_includes: list[str],
    include_paths: tuple[str, ...],
    src: str,
) -> list[str]:
    """Assemble the slangc argv for a single compile invocation."""
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
        # slangc 2026.7.1 trips on subgroup_ballot capability checks at
        # ``helpers.wave_active_count_bits`` even though every Vulkan
        # device we ship to supports subgroupVote/Ballot; the error
        # path then crashes the compiler in the thread-pool case.
        # Bypass the static check — runtime SPIR-V is unaffected.
        "-ignore-capabilities",
    ]
    for ip in module_includes:
        cmd.extend(["-I", ip])
    cmd.extend(["-I", _SHADERS_LIB_DIR])
    for ip in include_paths:
        cmd.extend(["-I", ip])

    # P3.2 / M14: Ensure mm_tile / mm_int8 modules are precompiled for
    # link-time specialization. The wrapper source already defines tile-size
    # constants via "static const int" before the import, so Slang's linker
    # resolves them without additional flags.
    if "import mm_tile;" in src:
        _ensure_mm_tile_module()
    if "import mm_int8;" in src:
        _ensure_mm_int8_module()
    return cmd


def _run_slangc(
    cmd: list[str],
    hash_key: str,
    timeout_s: float = _SLANGC_TIMEOUT_S,
) -> subprocess.CompletedProcess:
    """Run a single slangc invocation, raising SlangCompileTimeout on timeout."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
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


def _handle_stale_module_cache(
    src_path: str,
    out_path: str,
    refl_path: str,
    entry: str,
    hash_key: str,
    include_paths: tuple[str, ...],
    src: str,
    module_includes: list[str],
) -> subprocess.CompletedProcess:
    """PF.27.a.1: retry slangc without precompiled module cache on SIGSEGV."""
    _invalidate_shader_lib_modules()
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
        "-ignore-capabilities",
        "-I",
        _SHADERS_LIB_DIR,
    ]
    for ip in include_paths:
        cmd2.extend(["-I", ip])
    return _run_slangc(cmd2, hash_key)


def _read_spv_and_reflection(
    out_path: str,
    refl_path: str,
) -> tuple[bytes, str]:
    """Read compiled SPIR-V and reflection JSON from disk."""
    with open(out_path, "rb") as f:
        spv = f.read()
    with open(refl_path) as f:
        return spv, f.read()


def _process_reflection(
    hash_key: str,
    refl_blob: str,
    spv: bytes,
    src: str,
    config_key: str | None,
) -> None:
    """Harvest reflection metrics, check baselines, and possibly recompile."""
    from torch_vulkan.inductor import config as _cfg

    _reflection_cache[hash_key] = refl_blob
    _disk_reflection_write(hash_key, refl_blob)

    if not _cfg.reflection_enabled():
        return

    _harvest_reflection_metrics(hash_key, refl_blob, spv, src, config_key)
    short_key = hashlib.sha256(spv).hexdigest()[:12]
    _check_spv_regression(short_key, _reflection_metrics_by_hash.get(hash_key, {}))

    if not _cfg.reflection_routing():
        return

    metrics = _reflection_metrics_by_hash.get(hash_key, {})
    vgprs = metrics.get("vgprs")
    if vgprs is None:
        return

    current_nt = _parse_numthreads_from_source(src)
    if current_nt is None:
        return

    shared_mem = metrics.get("shared_mem")
    loop_depth = metrics.get("loop_depth")
    optimal_nt = _pick_numthreads_from_reflection(
        vgprs, shared_mem, loop_depth, current_nt,
    )
    if optimal_nt == current_nt:
        _optimized_numthreads_by_hash[hash_key] = current_nt
        return

    # Recompile with optimized numthreads.
    new_src = _rewrite_numthreads_in_source(src, optimal_nt)
    _compile_with_optimized_numthreads(
        new_src, optimal_nt, entry="computeMain", hash_key=hash_key,
        config_key=config_key, src=src,
    )
    _optimized_numthreads_by_hash[hash_key] = optimal_nt


def _compile_with_optimized_numthreads(
    new_src: str,
    optimal_nt: tuple[int, int, int],
    *,
    entry: str,
    hash_key: str,
    config_key: str | None,
    src: str,
) -> None:
    """Second-pass compile with rewritten numthreads; replace cached SPV on success."""
    with tempfile.TemporaryDirectory() as td:
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
            "-ignore-capabilities",
        ]
        try:
            proc3 = _run_slangc(cmd3, hash_key)
        except SlangCompileTimeout:
            return  # keep Pass-1 SPV
        if proc3.returncode != 0:
            return

        with open(new_out_path, "rb") as f:
            spv2 = f.read()
        _cache_by_hash[hash_key] = spv2
        _disk_cache_write(hash_key, spv2)

        try:
            with open(new_refl_path) as f:
                refl_blob2 = f.read()
            _reflection_cache[hash_key] = refl_blob2
            _disk_reflection_write(hash_key, refl_blob2)
            from torch_vulkan.inductor import config as _cfg
            if _cfg.reflection_enabled():
                _harvest_reflection_metrics(
                    hash_key, refl_blob2, spv2, new_src, config_key,
                )
        except (FileNotFoundError, OSError):
            pass


def _compile_slang_to_spirv_inner(
    src: str,
    entry: str,
    hash_key: str,
    include_paths: tuple[str, ...] = (),
    config_key: str | None = None,
) -> bytes:
    """Compile a single Slang source to SPIR-V with full retry / recompile logic.

    Modularized from the original 250-line function into discrete phases:
      1. validate source
      2. prepare command line
      3. run slangc (with stale-module retry)
      4. read SPV + reflection
      5. process reflection (metrics / numthreads recompile)
      6. update caches + stats
    """
    from torch_vulkan.inductor.slang_validator import validate_slang_source

    # Phase 0: normalize invalid Slang floating-point literals that the
    # backend codegen may produce (e.g., -inf/inf from reduction identity
    # values).  Slang has no `inf` keyword; the valid form is (1.0/0.0)
    # or (-1.0/0.0) or asfloat(0x7F800000u).
    import re
    src = re.sub(r'(?<!\w)(-inf)(?!\w)', '(-1.0/0.0)', src)
    src = re.sub(r'(?<!\w)(inf)(?!\w)', '(1.0/0.0)', src)
    src = re.sub(r'\(-\(1\.0/0\.0\)\)', '(-1.0/0.0)', src)

    # Phase 1: fast source validation (no file I/O, no subprocess)
    validation_errors = validate_slang_source(src)
    if validation_errors:
        raise RuntimeError(
            f"Slang source validation failed for kernel {hash_key[:48]}:\n"
            + "\n".join(str(e) for e in validation_errors)
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

        # Phase 2: resolve shader-lib module includes
        try:
            module_dir = _ensure_shader_lib_modules()
            module_includes = [module_dir]
        except RuntimeError:
            module_includes = []

        # Phase 3: compile (with stale-module retry)
        cmd = _build_compile_command(
            src_path=src_path,
            entry=entry,
            out_path=out_path,
            refl_path=refl_path,
            module_includes=module_includes,
            include_paths=include_paths,
            src=src,
        )
        used_module_includes = bool(module_includes)
        proc = _run_slangc(cmd, hash_key)

        if proc.returncode < 0 and module_includes:
            proc = _handle_stale_module_cache(
                src_path, out_path, refl_path, entry, hash_key,
                include_paths, src, module_includes,
            )
            used_module_includes = False

        if proc.returncode != 0:
            raise RuntimeError(
                f"slangc failed for kernel {hash_key[:8]}:\n{proc.stderr}\n"
                f"--- source ---\n{src}"
            )

        # Phase 4: read output
        spv, refl_blob = _read_spv_and_reflection(out_path, refl_path)

        # Phase 5: reflection + optional numthreads recompile
        try:
            _process_reflection(hash_key, refl_blob, spv, src, config_key)
        except (FileNotFoundError, OSError):
            pass

    # Phase 6: update stats + caches
    _COMPILE_STATS["cold_compiles"] += 1
    elapsed_us = (time.perf_counter() - t0) * 1e6
    _COMPILE_STATS["cold_compile_us"] += elapsed_us
    if elapsed_us > _COMPILE_STATS["max_cold_compile_us"]:
        _COMPILE_STATS["max_cold_compile_us"] = elapsed_us

    _cache_by_hash[hash_key] = spv
    _disk_cache_write(hash_key, spv)
    return spv


# ═══════════════════════════════════════════════════════════════════════════════
# Public API (unchanged signatures)
# ═══════════════════════════════════════════════════════════════════════════════

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
    # Mix in the content hash of each imported shader-lib file so that
    # changes to shaders/lib/*.slang automatically invalidate the disk
    # cache for all kernels that import those files (M22.16-cache-fix).
    lib_tag = _shader_lib_import_hash(src)
    hash_key = hashlib.sha256(
        (entry + "\n" + _normalize_slang_source(src) + inc_tag + sgs_tag + lib_tag).encode()
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
            we_own = True

    if not we_own and not event.is_set():
        event.wait()
        spv = _cache_by_hash.get(hash_key)
        if spv is not None:
            if cache_key is not None:
                _cache_by_key[cache_key] = spv
            return spv

    try:
        if _PARALLEL_COMPILE and _ASYNC_COMPILE and not _is_in_pool_worker():
            pool = _get_async_pool()
            spv = pool.submit(
                _wrap_pool_worker(_compile_slang_to_spirv_inner),
                src, entry, hash_key, include_paths, config_key,
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
                ev.set()


def async_precompile_slang(
    src: str,
    *,
    entry: str = "computeMain",
    cache_key: str = "",
    include_paths: tuple[str, ...] = (),
) -> None:
    """Submit Slang→SPIR-V compilation to the async pool without blocking (C1).

    The compilation runs in a background thread and populates the in-memory
    and on-disk SPIR-V caches.  Subsequent calls to :func:`compile_slang_to_spirv`
    with the same source will find the cached result and return instantly.

    If the async pool is unavailable (``_ASYNC_COMPILE=0``), this is a no-op
    — the first synchronous ``compile_slang_to_spirv`` call will pay the cost.
    """
    if not _ASYNC_COMPILE:
        return
    inc_tag = "" if not include_paths else "\nINC=" + "|".join(include_paths)
    sgs_tag = _get_device_subgroup_size_tag()
    lib_tag = _shader_lib_import_hash(src)
    hash_key = hashlib.sha256(
        (entry + "\n" + _normalize_slang_source(src) + inc_tag + sgs_tag + lib_tag).encode()
    ).hexdigest()
    if hash_key in _cache_by_hash:
        return  # Already compiled
    spv = _disk_cache_read(hash_key)
    if spv is not None:
        _cache_by_hash[hash_key] = spv
        if cache_key:
            with _cache_lock:
                _cache_by_key[cache_key] = spv
        return  # Already on disk
    with _in_flight_lock:
        if hash_key in _in_flight:
            return  # Already being compiled
    pool = _get_async_pool()
    pool.submit(
        _wrap_pool_worker(_compile_slang_to_spirv_inner),
        src, entry, hash_key, include_paths, cache_key,
    )
    # Intentionally do NOT call .result() — fire-and-forget


def batch_compile_slang_to_spirv(
    specs: list[tuple[str, str, str, tuple[str, ...]]],
    *,
    max_workers: Optional[int] = None,
) -> dict[str, bytes]:
    """Compile multiple Slang sources to SPIR-V in parallel.  N+1.6."""
    workers = max_workers if max_workers is not None else _ASYNC_MAX_WORKERS
    results: dict[str, bytes] = {}
    pending: list[tuple[str, str, str, tuple[str, ...]]] = []

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

    errors: list[Exception] = []

    def _compile_one(src, entry, cache_key, include_paths):
        try:
            return compile_slang_to_spirv(
                src, entry=entry, cache_key=cache_key, include_paths=include_paths,
            )
        except Exception as e:
            errors.append(e)
            return None

    wrapped = _wrap_pool_worker(_compile_one)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(wrapped, src, entry, ck, ip): ck
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
    """Compile multiple (src, cache_key) pairs in parallel.  N+1.6."""
    workers = max_workers if max_workers is not None else _MAX_SLANGC_WORKERS
    specs = [(src, entry, ck, ()) for src, ck in sources]
    result_map = batch_compile_slang_to_spirv(specs, max_workers=workers)
    return [result_map[ck] for _src, ck in sources]


def parse_spec_constants(spv: bytes) -> list[tuple[int, int]]:
    """Walk a SPIR-V module and return ``[(spec_id, default_uint_value), ...]``."""
    if len(spv) < 20 or spv[0:4] not in (b"\x03\x02\x23\x07", b"\x07\x23\x02\x03"):
        raise ValueError("not a SPIR-V module")
    little = spv[0:4] == b"\x03\x02\x23\x07"

    def w32(off: int) -> int:
        b = spv[off : off + 4]
        return int.from_bytes(b, "little" if little else "big")

    n_words = len(spv) // 4
    spec_ids: dict[int, int] = {}
    spec_defaults: dict[int, int] = {}
    i = 5
    while i < n_words:
        word = w32(i * 4)
        op = word & 0xFFFF
        wc = word >> 16
        if wc == 0:
            break
        if op == 71 and wc >= 4:
            target = w32((i + 1) * 4)
            decoration = w32((i + 2) * 4)
            if decoration == 1:
                spec_ids[target] = w32((i + 3) * 4)
        elif op == 50 and wc >= 4:
            result_id = w32((i + 2) * 4)
            spec_defaults[result_id] = w32((i + 3) * 4)
        i += wc
    out = [(sid, spec_defaults.get(result_id, 0)) for result_id, sid in spec_ids.items()]
    out.sort()
    return out


def compile_slang_to_spirv_with_reflection(
    src: str,
    entry: str = "computeMain",
    cache_key: Optional[str] = None,
    include_paths: tuple[str, ...] = (),
) -> tuple[bytes, dict]:
    """Compile a Slang source and return ``(spv, layout_dict)``."""
    spv = compile_slang_to_spirv(
        src, entry=entry, cache_key=cache_key, include_paths=include_paths
    )
    inc_tag = "" if not include_paths else "\nINC=" + "|".join(include_paths)
    hash_key = hashlib.sha256(
        (entry + "\n" + _normalize_slang_source(src) + inc_tag).encode()
    ).hexdigest()
    from .reflection import get_reflection_json, reflection_layout

    refl = get_reflection_json(hash_key)
    if refl is None:
        return spv, {"bindings": [], "push_constant_size": 0}
    return spv, reflection_layout(refl)


def gc_spirv_cache(max_mib: int) -> dict:
    """Trim the on-disk SPIR-V cache to ``max_mib`` MiB by deleting LRU entries."""
    if max_mib < 0:
        raise ValueError("max_mib must be >= 0")
    budget = max_mib * 1024 * 1024
    entries: list[tuple[float, int, str]] = []
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
    entries.sort(key=lambda e: e[0])
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
