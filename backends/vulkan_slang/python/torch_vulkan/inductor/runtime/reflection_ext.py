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
  - numthreads parsing
    (``_parse_numthreads_from_source``, ``_pick_numthreads_from_reflection``)
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
    """Parse slangc reflection JSON for key performance metrics (P3.3/M13/M20.5).

    slangc 2026.7.1 JSON schema:
      entryPoints[].{name, stage, parameters, threadGroupSize, bindings}
    Hardware counters (numRegisters, subgroupSize, numSgprs, …) are NOT
    emitted by slangc — those fields are filled later by
    ``_analyze_spirv_binary``.

    M20.5 additions: schema extended with num_sgprs, num_loads,
    num_stores, num_atomics (all None from JSON, filled by SPIR-V pass).
    subgroup_size is inferred from threadGroupSize[0] % 64/32.

    Returns a dict with keys:
      vgprs, shared_mem, subgroup_size, loop_depth,
      num_sgprs, num_loads, num_stores, num_atomics.
    All missing keys are None.
    """
    import json

    metrics: dict = {
        "vgprs": None,
        "shared_mem": None,
        "subgroup_size": None,
        "loop_depth": None,
        "num_sgprs": None,
        "num_loads": None,
        "num_stores": None,
        "num_atomics": None,
    }
    try:
        data = json.loads(refl_json)
    except (json.JSONDecodeError, TypeError):
        return metrics

    # ── subgroup_size from entry points ──
    # slangc 2026.7.1 does not emit subgroupSize in the JSON.
    # Infer from threadGroupSize: if X is a multiple of 64, assume wave64.
    entry_points = data.get("entryPoints") or []
    for ep in entry_points:
        if ep.get("stage") == "compute":
            ss = ep.get("subgroupSize")
            if ss is not None:
                metrics["subgroup_size"] = int(ss)
            else:
                # M20.5: infer from threadGroupSize X dimension
                tgs = ep.get("threadGroupSize")
                if tgs and len(tgs) >= 1:
                    x = tgs[0]
                    if x > 0:
                        if x % 64 == 0:
                            metrics["subgroup_size"] = 64
                        elif x % 32 == 0:
                            metrics["subgroup_size"] = 32
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
    # slangc 2026.7.1 does not emit numRegisters; filled by SPIR-V analysis.
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

    # ── M20.5: num_sgprs, num_loads, num_stores, num_atomics ──
    # slangc 2026.7.1 does not emit these; filled by _analyze_spirv_binary.
    for ep in entry_points:
        for field, keys in (
            ("num_sgprs", ("numSgprs", "numScalarRegisters")),
            ("num_loads", ("numLoads", "numMemoryLoads")),
            ("num_stores", ("numStores", "numMemoryStores")),
            ("num_atomics", ("numAtomics", "numAtomicOps")),
        ):
            for k in keys:
                v = ep.get(k)
                if v is not None:
                    metrics[field] = int(v)
                    break
        break

    return metrics


# ── SPIR-V binary analysis fallback ──────────────────────────────────────


def _analyze_spirv_binary(spv: bytes) -> dict:
    """Estimate VGPR/SGPR count, shared-memory, and I/O metrics from SPIR-V.

    M20.5: Extended to count OpLoad/OpStore/Atomics and approximate SGPR
    usage from uniform/input/push-constant variable count.  Always returns
    at least vgprs=1 (even for trivially simple kernels with 0 func vars).
    """
    metrics: dict = {
        "vgprs": None,
        "shared_mem": None,
        "num_sgprs": None,
        "num_loads": None,
        "num_stores": None,
        "num_atomics": None,
    }

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
    OP_LOAD = 61
    OP_STORE = 62
    STORAGE_CLASS_FUNCTION = 7
    STORAGE_CLASS_WORKGROUP = 4
    # Uniform/Input/PushConstant classes → SGPR-class resources
    STORAGE_CLASS_UNIFORM = 2
    STORAGE_CLASS_INPUT = 1
    STORAGE_CLASS_PUSH_CONSTANT = 9
    STORAGE_CLASS_UNIFORM_CONSTANT = 0
    OP_TYPE_FLOAT = 22
    OP_TYPE_INT = 21
    OP_TYPE_ARRAY = 28

    # Atomic opcode range: OpAtomicLoad=227 … OpAtomicFlagClear=240,
    # plus OpAtomicFAddEXT=6035 (rarely emitted by slangc).
    _ATOMIC_OPS: frozenset = frozenset(range(227, 241))

    n_words = len(spv) // 4
    i = 5  # skip 5-word header

    type_sizes: dict = {}
    func_vars = 0
    sgpr_vars = 0
    workgroup_bytes = 0
    num_loads = 0
    num_stores = 0
    num_atomics = 0

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
            elif storage_class in (
                STORAGE_CLASS_UNIFORM,
                STORAGE_CLASS_INPUT,
                STORAGE_CLASS_PUSH_CONSTANT,
                STORAGE_CLASS_UNIFORM_CONSTANT,
            ):
                sgpr_vars += 1
        elif op == OP_LOAD:
            num_loads += 1
        elif op == OP_STORE:
            num_stores += 1
        elif op in _ATOMIC_OPS:
            num_atomics += 1

        i += wc

    # Heuristic: each function-scope variable costs ≥1 VGPR + 1 temp.
    # M20.5: Even kernels with 0 function-scope vars have at least 1 VGPR
    # (the thread ID register).  Use 1 as the minimum so _pick_numthreads
    # doesn't fall back to the heuristic for trivially simple kernels.
    metrics["vgprs"] = max(func_vars * 2, 1)

    if workgroup_bytes > 0:
        metrics["shared_mem"] = workgroup_bytes

    # M20.5: SGPR approximation from uniform/input variable count.
    if sgpr_vars > 0:
        metrics["num_sgprs"] = sgpr_vars * 2  # each binding uses ~2 SGPRs

    metrics["num_loads"] = num_loads
    metrics["num_stores"] = num_stores
    metrics["num_atomics"] = num_atomics

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

    # 2. Fill gaps with SPIR-V binary analysis.
    # M20.5: _analyze_spirv_binary now also returns num_sgprs, num_loads,
    # num_stores, num_atomics — always run it to fill those fields.
    spv_metrics = _analyze_spirv_binary(spv)
    if metrics["vgprs"] is None:
        metrics["vgprs"] = spv_metrics.get("vgprs")
    if metrics["shared_mem"] is None:
        metrics["shared_mem"] = spv_metrics.get("shared_mem")
    # M20.5: hardware I/O counters — always from SPIR-V analysis.
    for _field in ("num_sgprs", "num_loads", "num_stores", "num_atomics"):
        if metrics.get(_field) is None:
            metrics[_field] = spv_metrics.get(_field)

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


def _parse_numthreads_from_source(src: str) -> tuple[int, int, int] | None:
    """Extract numthreads tuple from Slang source."""
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
    num_sgprs: int | None = None,
    num_loads: int | None = None,
    num_stores: int | None = None,
) -> tuple[int, int, int]:
    """DR.7 / M11.1 / M20.5: Pick optimal numthreads from SPIR-V metrics.

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

    M20.5 additions:
    - *num_sgprs* > 64: scalar-register pressure is high (many uniforms /
      push-constant fields), which on RDNA1 constrains the number of
      waves the hardware can schedule.  Drop one tier.
    - *num_loads* + *num_stores* > 128: memory-bandwidth-heavy kernel.
      Wider workgroups (more threads) hide latency better.  Raise the
      base one tier (but never above 256).

    Falls back to *current_numthreads* when *vgprs* is ``None`` and
    *num_sgprs* is ``None`` (no reflection data at all).

    Args:
        vgprs: VGPR count (from SPIR-V function-variable analysis).
        shared_mem: Groupshared / LDS bytes used.
        loop_depth: Maximum nested loop depth.
        current_numthreads: The numthreads currently in the source.
        num_sgprs: SGPR count estimate (uniform/input variable count × 2).
        num_loads: OpLoad instruction count from SPIR-V.
        num_stores: OpStore instruction count from SPIR-V.

    Returns:
        Optimal ``(x, y, z)`` numthreads tuple.
    """
    # Preserve 2D/3D workgroups — only adjust the X dimension.
    if current_numthreads[1] != 1 or current_numthreads[2] != 1:
        return current_numthreads

    if vgprs is None and num_sgprs is None:
        return current_numthreads

    # Base tier from VGPR count (fall back to 128 when only SGPR data)
    if vgprs is not None:
        if vgprs <= 32:
            base = 256
        elif vgprs <= 64:
            base = 128
        elif vgprs <= 128:
            base = 64
        else:
            base = 32
    else:
        base = 128  # conservative default when only SGPR data

    # M20.5: Memory-bandwidth-heavy kernels hide latency with more threads.
    # If num_loads + num_stores is large (>128 I/O ops), the kernel spends
    # significant time in memory transactions.  Raise one tier to expose
    # more in-flight requests.  Never exceed 256.
    total_io = (num_loads or 0) + (num_stores or 0)
    if total_io > 128:
        if base == 64:
            base = 128
        elif base == 128:
            base = 256

    # M11.1: Shared-memory penalty — when LDS usage is high, drop a tier
    # to leave more LDS per workgroup for groupshared allocations.
    if shared_mem is not None and shared_mem > 16384:  # > 16 KiB
        if base == 256:
            base = 128
        elif base == 128:
            base = 64

    # M20.5: SGPR pressure penalty — many uniform/input bindings increase
    # scalar register pressure; the hardware limits concurrent waves.
    if num_sgprs is not None and num_sgprs > 64:
        if base == 256:
            base = 128
        elif base == 128:
            base = 64
        elif base == 64:
            base = 32

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
