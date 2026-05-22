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
        # We just need the .slang-module to exist on the -I path.
        if "import mm_tile;" in src:
            _ensure_mm_tile_module()
        if "import mm_int8;" in src:
            _ensure_mm_int8_module()

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
            # slangc 2026.7.1 trips on subgroup_ballot capability checks at
            # ``helpers.wave_active_count_bits`` even though every Vulkan
            # device we ship to supports subgroupVote/Ballot; the error
            # path then crashes the compiler in the thread-pool case.
            # Bypass the static check — runtime SPIR-V is unaffected.
            "-ignore-capabilities",
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
            # slangc 2026.7.1 trips on subgroup_ballot capability checks at
            # ``helpers.wave_active_count_bits`` even though every Vulkan
            # device we ship to supports subgroupVote/Ballot; the error
            # path then crashes the compiler in the thread-pool case.
            # Bypass the static check — runtime SPIR-V is unaffected.
            "-ignore-capabilities",
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
