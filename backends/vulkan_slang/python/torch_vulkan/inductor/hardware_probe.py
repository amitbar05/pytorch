"""M21.1.c — public hardware-probe orchestrator.

Bundles the three existing warm-up paths behind a single API the user can
call (or that the backend can fire on first import):

* **level 0** (~5 s) — microbenchmarks via ``device_profile.load_or_profile``.
  Captures launch latency, mem BW, LDS BW, atomic throughput, device limits.
  Cached at ``~/.cache/torch_vulkan/device_profile_<id>.json``.

* **level 1** (~30 s) — synchronous shader-lib precompile + matmul template
  SPIR-V cache fill. Both already exist as background paths; level 1 just
  blocks until they finish so the user knows the SPIR-V cache is hot.

* **level 2** (~3 min) — canonical-shape autotune sweep. Runs ``a @ b`` and
  ``F.conv2d(x, w, b)`` through ``torch.compile(backend="inductor")`` at a
  small grid of shapes × dtypes so the per-kernel WG-size cache (see
  ``inductor/autotune.py``) is populated before the user's first compile.

The probe writes a marker at ``~/.cache/torch_vulkan/probe_status_<id>.json``
recording the highest level completed for the current device, so subsequent
imports skip the work. The marker is keyed off the same ``compute_device_id``
hash as ``device_profile.py``.

Auto-run on import is gated by ``TORCH_VULKAN_PROFILE_DEVICE``:

* unset / ``"auto"`` — default. Run level 2 if no marker; otherwise read cache.
* ``"quick"``        — run level 0 only.
* ``"medium"``       — run level 1.
* ``"deep"``         — run level 2.
* ``"force"``        — re-run level 2 even if marker exists.
* ``"off"``          — skip entirely. ``current()`` from ``device_profile``
                       still returns ``None``.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional


_log = logging.getLogger(__name__)

LEVEL_QUICK = 0
LEVEL_MEDIUM = 1
LEVEL_DEEP = 2

# Canonical shapes for the level-2 autotune sweep. Sized to land in the
# ~3-minute budget on RDNA1 (16 CU, 1024 max WG). Shapes are powers-of-two
# multiples of 64 so they wave-align cleanly on both wave32 and wave64.
_MM_PROBE_SHAPES: list[tuple[int, int, int]] = [
    (128, 128, 128),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
]
_MM_PROBE_DTYPES: tuple[str, ...] = ("float32", "float16")

# (B, Cin, Cout, H, W, K, stride, padding)
_CONV_PROBE_SHAPES: list[tuple[int, int, int, int, int, int, int, int]] = [
    (1, 32, 64, 32, 32, 3, 1, 1),
    (1, 64, 128, 32, 32, 3, 1, 1),
    (1, 128, 128, 16, 16, 3, 1, 1),
]
_CONV_PROBE_DTYPES: tuple[str, ...] = ("float32", "float16")


# ── Status marker ──────────────────────────────────────────────────────


def _probe_status_path() -> Path:
    """Return the marker path for the current device."""
    from . import device_profile as _dp

    name = _dp._device_name_safe()
    props = {
        "device_name": name,
        "vendor_id": _dp._guess_vendor_id(name),
        "device_type": _dp._classify_device_type(name),
    }
    device_id = _dp.compute_device_id(props)
    return _dp.cache_root() / f"probe_status_{device_id}.json"


def _read_probe_status() -> Optional[dict[str, Any]]:
    path = _probe_status_path()
    if not path.exists():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def _write_probe_status(level: int, extra: Optional[dict[str, Any]] = None) -> None:
    path = _probe_status_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "completed_level": int(level),
        "captured_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


# ── Level implementations ──────────────────────────────────────────────


def _run_level_0(force: bool) -> Optional[dict[str, Any]]:
    """Run the microbench + capability probe. Returns the profile dict or None."""
    from . import device_profile as _dp

    return _dp.load_or_profile(mode="force" if force else "auto")


def _run_level_1_sync() -> dict[str, Any]:
    """Synchronously precompile shader libs + matmul template SPIR-V.

    Both prewarmers already exist as background paths invoked from
    ``inductor/__init__.py``. Calling them with ``sync=True`` reuses the same
    cache and blocks until done. The on-disk SPIR-V cache survives across
    imports so this is paid at most once per (slangc, slang version) pair.
    """
    out: dict[str, Any] = {}

    t0 = time.perf_counter()
    try:
        from .templates.caller.gemm.install import prewarm_matmul_templates

        out["matmul_specs_compiled"] = int(prewarm_matmul_templates(sync=True) or 0)
    except Exception as e:
        _log.warning("matmul template prewarm failed: %s", e)
        out["matmul_specs_compiled"] = 0
        out["matmul_error"] = type(e).__name__
    out["matmul_ms"] = (time.perf_counter() - t0) * 1e3

    t1 = time.perf_counter()
    try:
        from .runtime import prewarm_shader_libs

        out["shader_libs_prewarmed"] = bool(prewarm_shader_libs(sync=True))
    except Exception as e:
        _log.warning("shader lib prewarm failed: %s", e)
        out["shader_libs_prewarmed"] = False
        out["shader_libs_error"] = type(e).__name__
    out["shader_libs_ms"] = (time.perf_counter() - t1) * 1e3

    return out


def _dtype_from_name(name: str):
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _run_level_2_autotune() -> dict[str, Any]:
    """Compile + run canonical mm and conv2d shapes through Inductor.

    This populates the per-kernel WG-size autotune cache
    (``~/.cache/torch_vulkan/autotune/*.json``) and the inductor compile cache
    for the standard training-shape grid. Failures per-shape are logged and
    counted but never raise — the probe is best-effort.
    """
    import torch

    out: dict[str, Any] = {
        "mm_shapes_probed": 0,
        "mm_ms": 0.0,
        "conv_shapes_probed": 0,
        "conv_ms": 0.0,
        "failures": [],
    }

    @torch.compile(backend="inductor", dynamic=False)
    def _mm(a, b):
        return a @ b

    t0 = time.perf_counter()
    with torch.no_grad():
        for M, N, K in _MM_PROBE_SHAPES:
            for dt_name in _MM_PROBE_DTYPES:
                dt = _dtype_from_name(dt_name)
                try:
                    a = torch.randn(M, K, dtype=dt, device="vulkan")
                    b = torch.randn(K, N, dtype=dt, device="vulkan")
                    _ = _mm(a, b)
                    out["mm_shapes_probed"] += 1
                except Exception as e:
                    _log.warning(
                        "mm probe (%d,%d,%d,%s) failed: %s",
                        M,
                        N,
                        K,
                        dt_name,
                        e,
                    )
                    out["failures"].append(
                        f"mm[{M},{N},{K},{dt_name}]: {type(e).__name__}"
                    )
    out["mm_ms"] = (time.perf_counter() - t0) * 1e3

    @torch.compile(backend="inductor", dynamic=False)
    def _conv(x, w, b, stride, padding):
        import torch.nn.functional as F

        return F.conv2d(x, w, b, stride=stride, padding=padding)

    t1 = time.perf_counter()
    with torch.no_grad():
        for B, Cin, Cout, H, W, K, stride, padding in _CONV_PROBE_SHAPES:
            for dt_name in _CONV_PROBE_DTYPES:
                dt = _dtype_from_name(dt_name)
                try:
                    x = torch.randn(B, Cin, H, W, dtype=dt, device="vulkan")
                    w = torch.randn(Cout, Cin, K, K, dtype=dt, device="vulkan")
                    bias = torch.zeros(Cout, dtype=dt, device="vulkan")
                    _ = _conv(x, w, bias, stride, padding)
                    out["conv_shapes_probed"] += 1
                except Exception as e:
                    _log.warning(
                        "conv probe (B=%d,Cin=%d,Cout=%d,H=%d,W=%d,K=%d,%s) failed: %s",
                        B,
                        Cin,
                        Cout,
                        H,
                        W,
                        K,
                        dt_name,
                        e,
                    )
                    out["failures"].append(
                        f"conv[B={B},Cin={Cin},Cout={Cout},H={H},W={W},K={K},{dt_name}]:"
                        f" {type(e).__name__}"
                    )
    out["conv_ms"] = (time.perf_counter() - t1) * 1e3

    return out


# ── Public API ─────────────────────────────────────────────────────────


def profile_device(
    level: int = LEVEL_DEEP,
    *,
    force: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the hardware probe at the requested level.

    See module docstring for a per-level breakdown and budgets. Results are
    cached in ``~/.cache/torch_vulkan/`` so a second call with the same level
    short-circuits to a cache read.

    Args:
        level: 0 (microbench), 1 (+ compile prewarm), or 2 (+ autotune sweep).
        force: re-run even if the marker says the level is already complete.
        verbose: print per-stage progress to stdout (useful from the CLI).

    Returns:
        ``{"level": int, "cached": bool, ...stage results}``.
    """
    if level not in (LEVEL_QUICK, LEVEL_MEDIUM, LEVEL_DEEP):
        raise ValueError(
            f"profile_device level must be 0, 1, or 2 (got {level!r})"
        )

    if not force:
        status = _read_probe_status()
        if status and int(status.get("completed_level", -1)) >= level:
            if verbose:
                print(
                    f"torch_vulkan.profile_device: level={level} cached (run at"
                    f" {status.get('captured_at', '?')})"
                )
            return {
                "level": level,
                "cached": True,
                "completed_level": int(status.get("completed_level", level)),
                "captured_at": status.get("captured_at"),
            }

    result: dict[str, Any] = {"level": level, "cached": False}
    t_total = time.perf_counter()

    if verbose:
        print(f"torch_vulkan.profile_device: starting level={level}")

    t = time.perf_counter()
    result["device_profile"] = _run_level_0(force=force)
    result["level_0_ms"] = (time.perf_counter() - t) * 1e3
    if verbose:
        print(f"  level 0 (microbench)     : {result['level_0_ms']:>8.0f} ms")

    if level >= LEVEL_MEDIUM:
        t = time.perf_counter()
        result["compile"] = _run_level_1_sync()
        result["level_1_ms"] = (time.perf_counter() - t) * 1e3
        if verbose:
            n = result["compile"].get("matmul_specs_compiled", 0)
            print(
                f"  level 1 (shader compile) : {result['level_1_ms']:>8.0f} ms"
                f" ({n} mm specs)"
            )

    if level >= LEVEL_DEEP:
        t = time.perf_counter()
        result["autotune"] = _run_level_2_autotune()
        result["level_2_ms"] = (time.perf_counter() - t) * 1e3
        if verbose:
            f = len(result["autotune"].get("failures", []))
            mm_n = result["autotune"].get("mm_shapes_probed", 0)
            conv_n = result["autotune"].get("conv_shapes_probed", 0)
            print(
                f"  level 2 (autotune sweep) : {result['level_2_ms']:>8.0f} ms"
                f" (mm={mm_n} conv={conv_n} failures={f})"
            )

    result["total_ms"] = (time.perf_counter() - t_total) * 1e3
    _write_probe_status(
        level,
        extra={
            "total_ms": result["total_ms"],
            "device_name": (result.get("device_profile") or {}).get(
                "device_name", "unknown"
            ),
        },
    )

    if verbose:
        print(
            f"torch_vulkan.profile_device: done ({result['total_ms']/1000.0:.1f} s)"
        )
    return result


def _resolve_auto_level(mode: str) -> Optional[int]:
    """Map ``TORCH_VULKAN_PROFILE_DEVICE`` to a level, or None for skip.

    Default (env unset / ``auto``) runs the level-0 microbench only — about
    5 seconds on RDNA1.  The compile + autotune sweeps (levels 1/2) take
    minutes on a cold slangc cache so we don't run them implicitly on
    every fresh install.  Users who want the full warm-up should call
    :func:`torch_vulkan.profile_and_warmup` explicitly, or set
    ``TORCH_VULKAN_PROFILE_DEVICE=deep`` to opt in for auto-import.
    """
    m = (mode or "auto").lower()
    if m in ("0", "off", "no", "false"):
        return None
    if m in ("quick", "1"):
        return LEVEL_QUICK
    if m in ("medium", "2"):
        return LEVEL_MEDIUM
    if m in ("deep", "3", "force"):
        return LEVEL_DEEP
    # "auto" / "true" / "yes" / unknown — default to QUICK so first
    # import populates the device-profile cache without paying for
    # shader compile or autotune. The marker prevents re-running.
    return LEVEL_QUICK


def auto_probe_on_import() -> Optional[dict[str, Any]]:
    """Entry point for ``inductor/__init__.py`` on first import.

    Reads ``TORCH_VULKAN_PROFILE_DEVICE`` and runs at the resolved level.
    Returns the result dict, or ``None`` when disabled. Errors are swallowed
    and logged — never let the probe block backend registration.
    """
    mode = os.environ.get("TORCH_VULKAN_PROFILE_DEVICE", "auto")
    level = _resolve_auto_level(mode)
    if level is None:
        return None

    force = mode.lower() == "force"

    # Print a one-line "what's happening" notice when we're about to do real
    # work — first import on a new device. Cached calls stay silent. The
    # cold-slangc cost dominates the budget on a fresh install (matmul
    # template SPIR-V compile = ~5-10 min on a 16-CU box); after that the
    # SPIR-V cache amortises and the steady-state cost is much lower.
    if level >= LEVEL_MEDIUM and not force:
        status = _read_probe_status()
        if not status or int(status.get("completed_level", -1)) < level:
            secs = {
                LEVEL_MEDIUM: "~30 s warm / up to ~10 min on cold slangc cache",
                LEVEL_DEEP: "~3 min warm / up to ~15 min on cold slangc cache",
            }.get(level, "")
            print(
                f"torch_vulkan: running hardware probe (level={level}, {secs})"
                " — cached on disk for future imports."
            )
            print(
                "torch_vulkan:   set TORCH_VULKAN_PROFILE_DEVICE=quick to skip"
                " the compile/autotune sweep next time."
            )

    try:
        return profile_device(level=level, force=force, verbose=False)
    except Exception as e:
        _log.warning("auto-probe-on-import failed at level=%d: %s", level, e)
        return None


# ── Test hooks ─────────────────────────────────────────────────────────


def reset_for_test() -> None:
    """Delete the status marker so the next call re-runs. Test-only."""
    path = _probe_status_path()
    try:
        path.unlink()
    except FileNotFoundError:
        pass
