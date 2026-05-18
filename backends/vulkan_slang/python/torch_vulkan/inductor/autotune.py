"""Vulkan autotuning infrastructure for kernel configuration selection.

Provides ``VulkanAutotuner`` that benchmarks multiple workgroup-size variants
for pointwise and reduction kernels, and caches the winner per-device.  Usage is
controlled by the ``TORCH_VULKAN_MAX_AUTOTUNE`` environment variable:

    TORCH_VULKAN_MAX_AUTOTUNE=0  # disabled (default — use heuristics only)
    TORCH_VULKAN_MAX_AUTOTUNE=1  # fast: benchmark 2 variants (small + large WG)
    TORCH_VULKAN_MAX_AUTOTUNE=2  # exhaustive: benchmark all variants

The perf cache lives in ``~/.cache/torch_vulkan/autotune/`` as JSON files keyed by
kernel hash.  On first run (no cache hit), the first configuration is used without
benchmarking (warm-up).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional  # noqa: F401 — used by both pre-existing and M21.2 hooks

import torch


_AUTOTUNE_CACHE_DIR = Path.home() / ".cache" / "torch_vulkan" / "autotune"


def _autotune_level() -> int:
    """Read TORCH_VULKAN_MAX_AUTOTUNE env var. Returns 0, 1, or 2."""
    try:
        level = int(os.environ.get("TORCH_VULKAN_MAX_AUTOTUNE", "0"))
        return max(0, min(level, 2))
    except (ValueError, TypeError):
        return 0


def is_autotune_enabled() -> bool:
    return _autotune_level() > 0


def get_wg_size_variants(
    is_reduction: bool = False,
) -> list[int]:
    """Return candidate workgroup sizes for autotuning.

    For pointwise kernels: [256, 512]
    For reduction kernels: [64, 128, 256]
    """
    if _autotune_level() < 2:
        if is_reduction:
            return [128, 256]
        return [256]
    if is_reduction:
        return [64, 128, 256]
    return [128, 256, 512]


def _cache_key(kernel_hash: str, device_name: str) -> str:
    return f"{device_name}_{kernel_hash[:16]}"


def _cache_path(key: str) -> Path:
    return _AUTOTUNE_CACHE_DIR / f"{key}.json"


def lookup_cached_wg_size(kernel_hash: str, device_name: str) -> Optional[int]:
    """Return the cached best workgroup size, or None if not cached."""
    key = _cache_key(kernel_hash, device_name)
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("best_wg")
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def cache_wg_size(
    kernel_hash: str, device_name: str, best_wg: int, all_timings: dict[int, float]
) -> None:
    """Persist the best workgroup size and all timings to the cache."""
    _AUTOTUNE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(kernel_hash, device_name)
    path = _cache_path(key)
    try:
        with open(path, "w") as f:
            json.dump(
                {"best_wg": best_wg, "timings": all_timings},
                f,
                indent=2,
            )
    except OSError:
        pass


def benchmark_wg_sizes(
    call_fn,
    wg_sizes: list[int],
    kernel_hash: str,
    device_name: str,
    warmup: int = 2,
    rep_ms: float = 5.0,
    validation_body: Optional[str] = None,
    kernel_name: Optional[str] = None,
) -> int:
    """Benchmark multiple workgroup sizes and return the fastest.

    Args:
        call_fn: callable() -> None  (the kernel dispatch function)
        wg_sizes: list of int workgroup sizes to try
        kernel_hash: string hash for cache keying
        device_name: device identifier for cache keying
        warmup: number of warmup iterations
        rep_ms: total benchmark time budget in milliseconds
        validation_body: M21.2 — optional Python source body run in a
            subprocess with Vulkan validation layers enabled. When
            provided and ``TORCH_VULKAN_VALIDATE_CODEGEN != off``, the
            winner is dispatched once under validation before the cache
            commit. VUIDs trigger a warn or error per the env-var mode.
        kernel_name: optional friendly name for diagnostics.

    Returns:
        The fastest workgroup size.
    """
    cached = lookup_cached_wg_size(kernel_hash, device_name)
    if cached is not None and cached in wg_sizes:
        return cached

    wg_timings: dict[int, list[float]] = {wg: [] for wg in wg_sizes}
    best_wg = wg_sizes[0]
    best_ms = float("inf")

    for wg in wg_sizes:
        samples: list[float] = []
        try:
            for _ in range(warmup):
                call_fn()
            t0 = time.perf_counter()
            while (time.perf_counter() - t0) * 1000.0 < rep_ms:
                start = time.perf_counter()
                call_fn()
                samples.append((time.perf_counter() - start) * 1000.0)
            if not samples:
                call_fn()
                start = time.perf_counter()
                call_fn()
                samples.append((time.perf_counter() - start) * 1000.0)
            from statistics import median
            med = median(samples)
            wg_timings[wg] = samples
            if med < best_ms:
                best_ms = med
                best_wg = wg
        except Exception:
            wg_timings[wg] = [float("inf")]

    # M21.2 — validation-as-codegen-check on the winner.
    # ``validate_winner`` is a no-op when ``TORCH_VULKAN_VALIDATE_CODEGEN``
    # is unset or ``off`` (the default), so this path costs nothing in
    # the common case. When enabled, the winner is dispatched once under
    # validation layers + the M21.3.a debug-utils messenger so VUIDs
    # surface at autotune-commit time rather than at silent-corruption
    # time. Under ``mode=error`` a VUID raises ``RuntimeError`` and the
    # commit below is skipped (the next ``benchmark_wg_sizes`` call will
    # retry; the cache stays at whatever it had).
    if validation_body is not None:
        validate_winner(
            validation_body,
            kernel_name=kernel_name or kernel_hash[:16],
            wg=best_wg,
        )

    summary = {wg: (min(v) if v else float("inf")) for wg, v in wg_timings.items()}
    cache_wg_size(kernel_hash, device_name, best_wg, summary)
    return best_wg


def validate_winner(
    body: str,
    *,
    kernel_name: str,
    wg: int,
) -> None:
    """Run the M21.2 validation-as-codegen-check on the autotune winner.

    Splits out the env-var resolution and result-handling so non-WG-sweep
    paths (e.g. ``install_external_mm``'s autotune_select_algorithm
    selection) can call this directly with the winner's source.

    ``body`` is a Python source body that must dispatch the kernel of
    interest once. It's run in a subprocess so ``VK_INSTANCE_LAYERS``
    takes effect before ``import torch_vulkan``.

    No-op fast-path: when ``TORCH_VULKAN_VALIDATE_CODEGEN`` is ``off``
    (the default), this function returns immediately without spawning
    a subprocess. The cost only materializes when someone explicitly
    opts in.
    """
    from .runtime.validation_codegen import (
        get_codegen_validation_mode,
        handle_validation_result,
        validate_codegen_dispatch,
    )

    if get_codegen_validation_mode() == "off":
        return  # short-circuit before any logging or env walking

    name = f"{kernel_name}@wg={wg}"
    result = validate_codegen_dispatch(body, kernel_name=name, timeout_s=60)
    handle_validation_result(result, kernel_name=name)


# ── Cache management ─────────────────────────────────────────────────────────

def list_autotune_cache() -> list[dict]:
    """Return all cached autotune entries.

    Each entry: ``{"key", "best_wg", "timings"}``. Useful for "what tuning
    decisions has my workload baked in?" inspection from a Jupyter cell.
    """
    out: list[dict] = []
    if not _AUTOTUNE_CACHE_DIR.exists():
        return out
    for path in sorted(_AUTOTUNE_CACHE_DIR.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            out.append({
                "key": path.stem,
                "best_wg": data.get("best_wg"),
                "timings": data.get("timings", {}),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return out


def clear_autotune_cache() -> int:
    """Delete every cached autotune entry. Returns the count removed.

    Use this after upgrading slang/SPIR-V toolchain or after AMD driver
    updates that change the relative cost of WG sizes — stale tuning
    decisions otherwise get reused indefinitely.
    """
    if not _AUTOTUNE_CACHE_DIR.exists():
        return 0
    n = 0
    for path in _AUTOTUNE_CACHE_DIR.glob("*.json"):
        try:
            path.unlink()
            n += 1
        except OSError:
            continue
    return n
