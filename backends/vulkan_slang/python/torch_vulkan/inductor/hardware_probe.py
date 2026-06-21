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
# D1: Expanded to cover square, tall-skinny, short-wide, batched, and
# large-dimension mm shapes plus 1×1/3×3/5×5 conv kernels.
_MM_PROBE_SHAPES: list[tuple[int, int, int]] = [
    # Training-typical square matmuls
    (128, 128, 128),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    # Tall-skinny (common in attention: Q@K^T with large seq_len)
    (2048, 128, 128),
    (4096, 256, 256),
    # Short-wide (common in FFN: hidden @ weight^T)
    (128, 2048, 128),
    (256, 4096, 256),
    # Batched dim-2 (common for small-batch Linear in training)
    (64, 512, 256),
    (256, 256, 1024),
    # Small matmuls (common in GN/LN weight grad, loss backward)
    (32, 32, 32),
    (8, 32, 64),
]
_MM_PROBE_DTYPES: tuple[str, ...] = ("float32", "float16")

# (B, Cin, Cout, H, W, K, stride, padding)
# D1: Expanded to cover 1×1, 3×3, 5×5 kernels at common training resolutions.
_CONV_PROBE_SHAPES: list[tuple[int, int, int, int, int, int, int, int]] = [
    # Standard 3×3 conv (ResNet / VGG style)
    (1, 32, 64, 32, 32, 3, 1, 1),
    (2, 64, 128, 32, 32, 3, 1, 1),
    (2, 128, 128, 16, 16, 3, 1, 1),
    # 1×1 conv (bottleneck / projection — very common)
    (2, 64, 64, 32, 32, 1, 1, 0),
    (2, 128, 256, 16, 16, 1, 1, 0),
    # 5×5 conv (larger receptive field)
    (1, 32, 64, 28, 28, 5, 1, 2),
    # Small conv (first-layer style, few channels)
    (2, 3, 16, 64, 64, 3, 1, 1),
    # Large batch conv
    (4, 64, 64, 32, 32, 3, 1, 1),
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
    """Compile + run canonical training shapes through Inductor.

    Sweeps mm, conv2d (fwd+bwd), linear (addmm), and bmm at a grid of
    shapes × dtypes (fp32, fp16) to populate the per-kernel WG-size autotune
    cache (``~/.cache/torch_vulkan/autotune/*.json``) and the Inductor
    compile cache. Failures per-shape are logged and counted but never
    raise — the probe is best-effort.

    D1: When TORCH_VULKAN_MM_TILES=expanded (or unset), uses the
    expanded tile config sweep (16 basic + 4 register tiles) so the
    autotune cache has per-shape winners for the full tile space.
    Set TORCH_VULKAN_MM_TILES=default to use only the small default set.
    """
    # D1: enable expanded tile sweep for warm-up autotune
    _prev_mm_tiles = os.environ.get("TORCH_VULKAN_MM_TILES")
    if _prev_mm_tiles is None:
        os.environ["TORCH_VULKAN_MM_TILES"] = "expanded"

    # D1: enable WG-size autotune during warm-up (benchmarks numthreads
    # variants for every pointwise/reduction kernel)
    _prev_wg = os.environ.get("TORCH_VULKAN_WG_AUTOTUNE")
    if _prev_wg is None:
        os.environ["TORCH_VULKAN_WG_AUTOTUNE"] = "1"

    try:
        return _run_level_2_autotune_impl()
    finally:
        if _prev_mm_tiles is None:
            del os.environ["TORCH_VULKAN_MM_TILES"]
        else:
            os.environ["TORCH_VULKAN_MM_TILES"] = _prev_mm_tiles
        if _prev_wg is None:
            del os.environ["TORCH_VULKAN_WG_AUTOTUNE"]
        else:
            os.environ["TORCH_VULKAN_WG_AUTOTUNE"] = _prev_wg


def _run_level_2_autotune_impl() -> dict[str, Any]:
    """Internal: runs the actual autotune probe (after env setup)."""
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

    # D1 — Linear (addmm) sweep: input @ weight.T + bias.
    # This is the most common op in transformer FFN and MLP blocks.
    # Uses a subset of MM shapes converted to linear-compatible dims.
    _LINEAR_PROBE_SHAPES: list[tuple[int, int, int]] = [
        (128, 512, 256),   # small batch, medium hidden
        (64, 2048, 512),   # tiny batch, large hidden
        (256, 256, 1024),  # medium batch, deep hidden
        (32, 1024, 4096),  # tiny batch, large FFN
    ]

    @torch.compile(backend="inductor", dynamic=False)
    def _linear(input_t, weight, bias):
        return torch.nn.functional.linear(input_t, weight, bias)

    t2 = time.perf_counter()
    linear_probed = 0
    with torch.no_grad():
        for B, in_features, out_features in _LINEAR_PROBE_SHAPES:
            for dt_name in _MM_PROBE_DTYPES:
                dt = _dtype_from_name(dt_name)
                try:
                    inp = torch.randn(B, in_features, dtype=dt, device="vulkan")
                    wt = torch.randn(out_features, in_features, dtype=dt, device="vulkan")
                    bias = torch.zeros(out_features, dtype=dt, device="vulkan")
                    _ = _linear(inp, wt, bias)
                    linear_probed += 1
                except Exception as e:
                    _log.warning(
                        "linear probe (B=%d,in=%d,out=%d,%s) failed: %s",
                        B, in_features, out_features, dt_name, e,
                    )
                    out["failures"].append(
                        f"linear[B={B},in={in_features},out={out_features},{dt_name}]:"
                        f" {type(e).__name__}"
                    )
    out["linear_shapes_probed"] = linear_probed
    out["linear_ms"] = (time.perf_counter() - t2) * 1e3

    # D1 — BMM (batched matmul) sweep: common in attention (Q@K^T),
    # multi-head projections, and batched linear layers.
    _BMM_PROBE_SHAPES: list[tuple[int, int, int, int]] = [
        (4, 128, 128, 128),     # small batch, square
        (8, 64, 512, 64),       # attention: Q@K^T style (heads, seq, head_dim)
        (2, 256, 256, 256),     # medium batch, square
        (16, 32, 128, 32),      # many small matmuls
    ]

    @torch.compile(backend="inductor", dynamic=False)
    def _bmm(a, b):
        return torch.bmm(a, b)

    t3 = time.perf_counter()
    bmm_probed = 0
    with torch.no_grad():
        for B, M, N, K in _BMM_PROBE_SHAPES:
            for dt_name in _MM_PROBE_DTYPES:
                dt = _dtype_from_name(dt_name)
                try:
                    a = torch.randn(B, M, K, dtype=dt, device="vulkan")
                    b = torch.randn(B, K, N, dtype=dt, device="vulkan")
                    _ = _bmm(a, b)
                    bmm_probed += 1
                except Exception as e:
                    _log.warning(
                        "bmm probe (B=%d,M=%d,N=%d,K=%d,%s) failed: %s",
                        B, M, N, K, dt_name, e,
                    )
                    out["failures"].append(
                        f"bmm[B={B},M={M},N={N},K={K},{dt_name}]: {type(e).__name__}"
                    )
    out["bmm_shapes_probed"] = bmm_probed
    out["bmm_ms"] = (time.perf_counter() - t3) * 1e3

    # D1 — Conv2d backward sweep: compiles both fwd and bwd by running a
    # loss.backward() through a tiny Conv2d module. This populates the
    # autotune cache for conv backward templates (slang_conv_bwd + bwd_diff).
    _CONV_BWD_PROBE_SHAPES: list[tuple[int, int, int, int, int, int]] = [
        (2, 32, 64, 16, 16, 3),    # medium conv
        (1, 64, 128, 32, 32, 3),   # larger spatial
        (2, 16, 32, 8, 8, 1),      # 1×1 conv backward
    ]

    t4 = time.perf_counter()
    conv_bwd_probed = 0
    for B, Cin, Cout, H, W, K in _CONV_BWD_PROBE_SHAPES:
        for dt_name in _CONV_PROBE_DTYPES:
            dt = _dtype_from_name(dt_name)
            try:
                conv = torch.nn.Conv2d(
                    Cin, Cout, K, padding=K // 2, device="vulkan"
                )
                if dt == torch.float16:
                    conv = conv.half()
                compiled_conv = torch.compile(conv, backend="inductor", dynamic=False)
                x = torch.randn(B, Cin, H, W, dtype=dt, device="vulkan")
                y = compiled_conv(x)
                loss = y.sum()
                loss.backward()
                conv.zero_grad(set_to_none=True)
                conv_bwd_probed += 1
            except Exception as e:
                _log.warning(
                    "conv bwd probe (B=%d,Cin=%d,Cout=%d,H=%d,W=%d,K=%d,%s) failed: %s",
                    B, Cin, Cout, H, W, K, dt_name, e,
                )
                out["failures"].append(
                    f"conv_bwd[B={B},Cin={Cin},Cout={Cout},H={H},W={W},K={K},{dt_name}]:"
                    f" {type(e).__name__}"
                )
    out["conv_bwd_shapes_probed"] = conv_bwd_probed
    out["conv_bwd_ms"] = (time.perf_counter() - t4) * 1e3

    # D1 — Reduction/pointwise sweep: GroupNorm, Softmax, GELU.
    # These are the most common non-matmul ops in training (norm layers,
    # attention softmax, activation functions).  Compiling them during
    # warm-up populates the SPIR-V cache for the reduction and pointwise
    # kernel templates, which otherwise compile cold on the first step.
    _REDUCTION_PROBE_SHAPES: list[tuple[int, int, int, int]] = [
        (2, 16, 32, 32),    # small GN
        (4, 32, 16, 16),    # medium GN
        (2, 64, 8, 8),      # large channels, small spatial
    ]

    @torch.compile(backend="inductor", dynamic=False)
    def _gn(x, weight, bias):
        return torch.nn.functional.group_norm(x, 4, weight, bias)

    @torch.compile(backend="inductor", dynamic=False)
    def _softmax(x):
        return torch.nn.functional.softmax(x, dim=-1)

    @torch.compile(backend="inductor", dynamic=False)
    def _gelu(x):
        return torch.nn.functional.gelu(x)

    t5 = time.perf_counter()
    gn_probed = softmax_probed = gelu_probed = 0
    with torch.no_grad():
        # GroupNorm
        for B, C, H, W in _REDUCTION_PROBE_SHAPES:
            for dt_name in _MM_PROBE_DTYPES:
                dt = _dtype_from_name(dt_name)
                try:
                    x = torch.randn(B, C, H, W, dtype=dt, device="vulkan")
                    w = torch.ones(C, dtype=dt, device="vulkan")
                    b = torch.zeros(C, dtype=dt, device="vulkan")
                    _ = _gn(x, w, b)
                    gn_probed += 1
                except Exception as e:
                    _log.warning("gn probe failed: %s", e)
                    out["failures"].append(f"gn: {type(e).__name__}")

        # Softmax (reduction along last dim — common in attention)
        for shape in [(2, 8, 128, 128), (4, 16, 256, 256), (2, 64, 512)]:
            for dt_name in _MM_PROBE_DTYPES:
                dt = _dtype_from_name(dt_name)
                try:
                    x = torch.randn(*shape, dtype=dt, device="vulkan")
                    _ = _softmax(x)
                    softmax_probed += 1
                except Exception as e:
                    _log.warning("softmax probe failed: %s", e)
                    out["failures"].append(f"softmax: {type(e).__name__}")

        # GELU (pointwise activation — common in transformer FFN)
        for shape in [(128, 512), (256, 1024), (64, 2048)]:
            for dt_name in _MM_PROBE_DTYPES:
                dt = _dtype_from_name(dt_name)
                try:
                    x = torch.randn(*shape, dtype=dt, device="vulkan")
                    _ = _gelu(x)
                    gelu_probed += 1
                except Exception as e:
                    _log.warning("gelu probe failed: %s", e)
                    out["failures"].append(f"gelu: {type(e).__name__}")

    out["gn_shapes_probed"] = gn_probed
    out["softmax_shapes_probed"] = softmax_probed
    out["gelu_shapes_probed"] = gelu_probed
    out["reduction_ms"] = (time.perf_counter() - t5) * 1e3

    # D1 — Conv2d tile config sweep: try different tile_w × tile_h × tile_c
    # combinations for a small set of canonical conv shapes.  The env var
    # TORCH_VULKAN_CONV_TILE controls the tile config in the conv lowering.
    # Dynamo is reset between tile configs to force a fresh compile.
    _CONV_TILE_CONFIGS: list[tuple[int, int, int]] = [
        (8, 8, 8),    # default — square
        (16, 8, 4),   # wide tile, shallow channels
        (4, 16, 4),   # tall tile, shallow channels
        (8, 8, 16),   # square, deep channels
    ]
    # Use a subset of conv probe shapes for the tile sweep (avoid
    # blowing the time budget — 4 tiles × 2 shapes × 2 dtypes = 16 extra).
    _TILE_SWEEP_SHAPES = _CONV_PROBE_SHAPES[:2]

    t6 = time.perf_counter()
    tile_probed = 0
    prev_tile_env = os.environ.get("TORCH_VULKAN_CONV_TILE")
    try:
        for B, Cin, Cout, H, W, K, stride, padding in _TILE_SWEEP_SHAPES:
            for dt_name in _CONV_PROBE_DTYPES:
                dt = _dtype_from_name(dt_name)
                for tw, th, tc in _CONV_TILE_CONFIGS:
                    try:
                        os.environ["TORCH_VULKAN_CONV_TILE"] = f"{tw}x{th}x{tc}"
                        torch._dynamo.reset()

                        @torch.compile(backend="inductor", dynamic=False)
                        def _conv_tile(x, w, b):
                            import torch.nn.functional as F
                            return F.conv2d(x, w, b, stride=stride, padding=padding)

                        x = torch.randn(B, Cin, H, W, dtype=dt, device="vulkan")
                        w = torch.randn(Cout, Cin, K, K, dtype=dt, device="vulkan")
                        b = torch.zeros(Cout, dtype=dt, device="vulkan")
                        with torch.no_grad():
                            _ = _conv_tile(x, w, b)
                        tile_probed += 1
                    except Exception as e:
                        _log.warning(
                            "conv tile probe (tile=%dx%dx%d, shape=%d,%d,%d,%d,%d,%d) failed: %s",
                            tw, th, tc, B, Cin, Cout, H, W, K, e,
                        )
                        out["failures"].append(
                            f"conv_tile[{tw}x{th}x{tc},"
                            f"B={B},Cin={Cin},Cout={Cout},H={H},W={W},K={K}]:"
                            f" {type(e).__name__}"
                        )
    finally:
        if prev_tile_env is not None:
            os.environ["TORCH_VULKAN_CONV_TILE"] = prev_tile_env
        elif "TORCH_VULKAN_CONV_TILE" in os.environ:
            del os.environ["TORCH_VULKAN_CONV_TILE"]

    out["conv_tile_shapes_probed"] = tile_probed
    out["conv_tile_ms"] = (time.perf_counter() - t6) * 1e3

    return out


# ── Public API ─────────────────────────────────────────────────────────


def profile_device(
    level: int = LEVEL_DEEP,
    *,
    force: bool = False,
    verbose: bool = False,
    validate: bool = False,
) -> dict[str, Any]:
    """Run the hardware probe at the requested level.

    See module docstring for a per-level breakdown and budgets. Results are
    cached in ``~/.cache/torch_vulkan/`` so a second call with the same level
    short-circuits to a cache read.

    Args:
        level: 0 (microbench), 1 (+ compile prewarm), or 2 (+ autotune sweep).
        force: re-run even if the marker says the level is already complete.
        verbose: print per-stage progress to stdout (useful from the CLI).
        validate: when True, enable ``TORCH_VULKAN_VUID_AS_ERROR=1`` during
            warm-up so that any VUID emitted by pre-compiled or autotuned
            shaders fails the warm-up call. Warns if VK_INSTANCE_LAYERS is
            not also set (Vulkan validation requires restart to take effect).

    Returns:
        ``{"level": int, "cached": bool, ...stage results}``.
    """
    if level not in (LEVEL_QUICK, LEVEL_MEDIUM, LEVEL_DEEP):
        raise ValueError(f"profile_device level must be 0, 1, or 2 (got {level!r})")

    # W4: Vulkan validation during warm-up.
    # When validate=True, set TORCH_VULKAN_VUID_AS_ERROR=1 for the duration
    # so any shader bugs surface at warm-up time.  Note: VK_INSTANCE_LAYERS
    # is read at Vulkan instance creation time (during import torch_vulkan)
    # so it cannot be changed here — we just warn if it's missing.
    _prev_vuid = os.environ.get("TORCH_VULKAN_VUID_AS_ERROR")
    _validation_active = False
    if validate:
        os.environ["TORCH_VULKAN_VUID_AS_ERROR"] = "1"
        _validation_active = True
        if verbose:
            _layers = os.environ.get("VK_INSTANCE_LAYERS", "")
            if _layers and "validation" in _layers.lower():
                print("  VUID-as-error + Vulkan validation layers: ON")
            else:
                print(
                    "  VUID-as-error: ON "
                    "(VK_INSTANCE_LAYERS not set — restart required for full "
                    "Vulkan validation layer)"
                )
    try:
        return _profile_device_impl(level, force, verbose, validate=validate)
    finally:
        if _prev_vuid is not None:
            os.environ["TORCH_VULKAN_VUID_AS_ERROR"] = _prev_vuid
        elif "TORCH_VULKAN_VUID_AS_ERROR" in os.environ:
            del os.environ["TORCH_VULKAN_VUID_AS_ERROR"]


def _profile_device_impl(
    level: int,
    force: bool,
    verbose: bool,
    validate: bool = False,
) -> dict[str, Any]:
    """Internal implementation of profile_device (after validation env setup)."""
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
        # W4: When validate=True, enable autotune validation so the
        # autotune subprocesses run with VK_INSTANCE_LAYERS active.
        # The subprocess gets a fresh Vulkan instance and CAN enable
        # validation layers even when the parent did not.
        _prev_validate_codegen = os.environ.get("TORCH_VULKAN_VALIDATE_CODEGEN")
        if validate:
            os.environ["TORCH_VULKAN_VALIDATE_CODEGEN"] = "error"
            if verbose:
                print("  (level 2 will validate autotune candidates in subprocess)")
        try:
            result["autotune"] = _run_level_2_autotune()
        finally:
            if _prev_validate_codegen is not None:
                os.environ["TORCH_VULKAN_VALIDATE_CODEGEN"] = _prev_validate_codegen
            elif "TORCH_VULKAN_VALIDATE_CODEGEN" in os.environ:
                del os.environ["TORCH_VULKAN_VALIDATE_CODEGEN"]
        result["level_2_ms"] = (time.perf_counter() - t) * 1e3
        if verbose:
            f = len(result["autotune"].get("failures", []))
            mm_n = result["autotune"].get("mm_shapes_probed", 0)
            conv_n = result["autotune"].get("conv_shapes_probed", 0)
            lin_n = result["autotune"].get("linear_shapes_probed", 0)
            bmm_n = result["autotune"].get("bmm_shapes_probed", 0)
            cbd_n = result["autotune"].get("conv_bwd_shapes_probed", 0)
            red_n = (
                result["autotune"].get("gn_shapes_probed", 0)
                + result["autotune"].get("softmax_shapes_probed", 0)
                + result["autotune"].get("gelu_shapes_probed", 0)
            )
            print(
                f"  level 2 (autotune sweep) : {result['level_2_ms']:>8.0f} ms"
                f" (mm={mm_n} conv={conv_n} linear={lin_n}"
                f" bmm={bmm_n} conv_bwd={cbd_n} reduct={red_n} failures={f})"
            )

    result["total_ms"] = (time.perf_counter() - t_total) * 1e3
    extra: dict[str, Any] = {
        "total_ms": result["total_ms"],
        "device_name": (result.get("device_profile") or {}).get(
            "device_name", "unknown"
        ),
    }
    # S2.4: record the tuning knobs that were active during warm-up so that
    # subsequent training imports can apply the same defaults → same Slang
    # source hash → SPIR-V cache hit instead of cold slangc.
    if level >= LEVEL_DEEP:
        extra["mm_tiles_mode"] = os.environ.get("TORCH_VULKAN_MM_TILES", "expanded")
    _write_probe_status(level, extra=extra)

    if verbose:
        print(
            f"torch_vulkan.profile_device: done ({result['total_ms'] / 1000.0:.1f} s)"
        )
    return result


def _resolve_auto_level(mode: str) -> Optional[int]:
    """Map ``TORCH_VULKAN_PROFILE_DEVICE`` to a level, or None for skip.

    M-PROBE.2 (v7): Default (env unset / ``auto``) is OFF — no implicit
    probe on import.  Users should call ``torch_vulkan.prepare_device()``
    explicitly.  Set ``TORCH_VULKAN_PROFILE_DEVICE=quick`` to restore the
    pre-v7 implicit level-0 microbench.
    """
    m = (mode or "auto").lower()
    if m in ("0", "off", "no", "false", "auto"):
        return None
    if m in ("quick", "1"):
        return LEVEL_QUICK
    if m in ("medium", "2"):
        return LEVEL_MEDIUM
    if m in ("deep", "3", "force"):
        return LEVEL_DEEP
    # Unknown value — default to off (safer than surprising auto-benchmark).
    return None


def _restore_probe_defaults() -> None:
    """S2.4: Apply tuning defaults from a previous warm-up (soft env defaults).

    Reads probe_status.json and sets env vars that were active during the
    level-2 autotune sweep, only when the user has NOT explicitly set them.
    This ensures torch.compile kernels generate the same Slang source as
    warm-up → same SPIR-V cache key → hit instead of cold slangc.

    Called on every import (before any compile) so that training inherits
    the warm-up tile config without the user needing to remember to set env vars.
    """
    status = _read_probe_status()
    if not status:
        return
    mm_mode = status.get("mm_tiles_mode")
    if mm_mode and "TORCH_VULKAN_MM_TILES" not in os.environ:
        os.environ["TORCH_VULKAN_MM_TILES"] = mm_mode


def auto_probe_on_import() -> Optional[dict[str, Any]]:
    """Entry point for ``inductor/__init__.py`` on first import.

    Reads ``TORCH_VULKAN_PROFILE_DEVICE`` and runs at the resolved level.
    Returns the result dict, or ``None`` when disabled. Errors are swallowed
    and logged — never let the probe block backend registration.
    """
    # S2.4: always restore tuning defaults from the last warm-up, even when
    # auto-probe is disabled. This is a fast disk read (no GPU work) and
    # ensures cache coherence between warm-up and training processes.
    try:
        _restore_probe_defaults()
    except Exception:
        pass

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


# ── W5: Per-model warm-up ─────────────────────────────────────────────


def prepare_model(
    model: "torch.nn.Module",
    sample_input: "tuple | torch.Tensor",
    *,
    loss_fn: "callable | None" = None,
    verbose: bool = True,
) -> "torch.nn.Module":
    """Compile and warm up all kernels for a specific model before training.

    Traces the model through ``torch.compile(backend="inductor")`` with the
    provided sample input, runs forward + backward to trigger SPIR-V
    compilation and caching for every kernel the model will use during
    training.  After this call, ``torch.compile(model, backend="inductor")``
    finds 100% SPIR-V cache hits — zero cold slangc latency.

    Args:
        model: The model to warm up (e.g., ``nn.Sequential(Conv2d(...), GN(...), ReLU())``).
        sample_input: A single tensor or tuple of tensors matching the
            model's forward signature.  Only shapes/dtypes matter; values
            can be random.
        loss_fn: Optional loss function ``(output, target) -> loss`` used to
            trigger backward compilation.  If ``None``, uses
            ``lambda out, tgt: out.sum()`` (fake loss, compiles fwd only if
            no grad).  For full fwd+bwd warm-up, pass a real loss.
        verbose: Print progress (compiled module, kernel count, timing).

    Returns:
        The ``torch.compile``-wrapped model, ready for training.  The
        returned module is functional — use it directly in your training
        loop.

    Example::

        import torch, torch_vulkan
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.GroupNorm(4, 16),
            torch.nn.ReLU(),
        ).to("vulkan")
        x = torch.randn(2, 3, 32, 32, device="vulkan")
        loss_fn = torch.nn.MSELoss()

        # One-shot warm-up: compiles all fwd+bwd kernels
        compiled = torch_vulkan.prepare_model(model, x, loss_fn=loss_fn)

        # Training loop — zero cold slangc
        for batch in train_loader:
            out = compiled(batch.x)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
    """
    import torch

    t0 = time.perf_counter()

    if isinstance(sample_input, torch.Tensor):
        args = (sample_input,)
    else:
        args = tuple(sample_input)

    # Create a target for the loss (use same device/dtype as input)
    first_arg = args[0]
    device = first_arg.device
    dtype = first_arg.dtype

    if verbose:
        print(
            f"torch_vulkan.prepare_model: tracing model with "
            f"input shape={tuple(first_arg.shape)}, dtype={dtype} ..."
        )

    # 1. Compile the model
    compiled = torch.compile(model, backend="inductor", dynamic=False)

    # 2. Run forward pass to compile all forward kernels
    with torch.no_grad():
        try:
            out = compiled(*args)
        except Exception as e:
            _log.warning("prepare_model forward pass failed: %s", e)
            if verbose:
                print(f"  forward FAILED: {e}")
                print(
                    "  (returning uncompiled model — kernels may still be "
                    "cached from partial compile)"
                )
            return compiled

    fwd_ms = (time.perf_counter() - t0) * 1e3
    if verbose:
        print(f"  forward compiled: {fwd_ms:>8.0f} ms")

    # 3. Run backward pass to trigger backward kernel compilations
    t_bwd = time.perf_counter()
    bwd_succeeded = False
    try:
        if loss_fn is not None:
            # Create a dummy target from the output shape/device
            if isinstance(out, torch.Tensor):
                target = torch.zeros_like(out)
            else:
                target = torch.zeros_like(out[0])

            loss = loss_fn(out, target)
            if loss.requires_grad:
                loss.backward()
                # Zero grads to avoid polluting the model
                model.zero_grad(set_to_none=True)
                bwd_succeeded = True
        else:
            # Fake backward: sum and backward to compile backward kernels
            # without needing a real loss function
            if isinstance(out, (list, tuple)):
                fake_loss = sum(o.sum() for o in out if isinstance(o, torch.Tensor))
            elif isinstance(out, torch.Tensor):
                fake_loss = out.sum()
            else:
                fake_loss = None

            if fake_loss is not None and fake_loss.requires_grad:
                fake_loss.backward()
                model.zero_grad(set_to_none=True)
                bwd_succeeded = True
    except Exception as e:
        _log.warning("prepare_model backward pass failed: %s", e)
        if verbose:
            print(f"  backward FAILED: {e}")
            print(
                "  (forward kernels are cached; backward will compile on "
                "first training step)"
            )

    bwd_ms = (time.perf_counter() - t_bwd) * 1e3
    total_s = (time.perf_counter() - t0)
    if verbose:
        if bwd_succeeded:
            print(f"  backward compiled: {bwd_ms:>8.0f} ms")
        print(
            f"  total warm-up: {total_s:.1f} s"
        )

    return compiled
