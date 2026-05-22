"""Reflection metrics, SPIR-V baseline checks, and numthreads tuning.

Extracted from ``slangc.py`` as part of the M22a anti-goal #7 line-cap
split (Stage 3).  Owns:

  - Link-time spec-constant helpers
    (``_extract_linktime_spec_constants``, ``_build_specialize_const_args``)
  - Disk reflection cache
    (``_disk_reflection_path``, ``_disk_reflection_read``,
     ``_disk_reflection_write``)
  - Disk metrics sidecar cache
    (``_disk_metrics_path``, ``_disk_metrics_read``, ``_disk_metrics_write``)
  - SPIR-V perf baseline tracking
    (``_load_spv_baselines``, ``_save_spv_baselines``,
     ``_check_spv_regression``)
  - Reflection JSON / SPIR-V / source parsing
    (``_parse_reflection_metrics``, ``_analyze_spirv_binary``,
     ``_analyze_slang_source_for_loop_depth``)
  - Reflection metrics access API
    (``_harvest_reflection_metrics``, ``get_reflection_metrics``,
     ``get_cached_metrics_for_key``, ``reset_reflection_baselines``)
  - numthreads tuning
    (``_parse_numthreads_from_source``, ``_rewrite_numthreads_in_source``,
     ``get_optimized_numthreads``, ``_pick_numthreads_from_reflection``)
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Optional

from .common import (
    _get_disk_cache_dir,
    _normalize_slang_source,
    _reflection_metrics_by_hash,
    _reflection_metrics_by_key,
    _spv_baselines,
    _spv_baselines_loaded,
)
from .shader_lib import _LT_SPEC_CONST_RE


# ── Link-time spec-constant helpers ─────────────────────────────────────


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


# ── Disk reflection cache ────────────────────────────────────────────────


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
