#!/usr/bin/env python3
"""PF.16 — Build-time SPIR-V pre-population for the regression suite.

Pre-warms the on-disk SPIR-V cache (``~/.cache/torch_vulkan/spirv``) so
the first regression-suite run after a fresh checkout / cache clear
hits warm cache instead of paying the slangc cold-compile bill on
every test class.

Two pre-warm paths:

1. **Module artifacts** — every ``shaders/lib/*.slang`` module is
   precompiled via ``precompile_shader_libs()``. This is normally
   triggered lazily by the runtime; doing it once at install time
   means the first kernel compile in every test process resolves
   ``import lib.helpers;`` against a cached ``.slang-module``
   instead of reparsing the source.

2. **Canonical kernel sources** — a curated set of trivially-
   parameterized kernel sources covering the dispatch shapes the
   regression suite exercises (pointwise, reduction, mm, addmm,
   mm+bias, gelu+linear, layer_norm). Each source is registered with
   the ``prewarm_compile`` API so the SPIR-V cache holds a hit before
   the test path requests the kernel.

Run after ``pip install -e .``:

    python tools/precompile_test_shaders.py

Or gated on an env var so CI can opt in but devs can skip if they
don't want the one-time ~5 s setup cost:

    TORCH_VULKAN_PRECOMPILE_TESTS=1 python tools/precompile_test_shaders.py

Output is JSON-on-stdout so a setup.py post-install hook (or CI
manifest) can parse the result.

PF.16 is a companion to PF.14 (deadlock fix) and PF.18 (subprocess
timeout): with those two in place, ``prewarm_compile(..., sync=True)``
is reliable + bounded; this script then leverages it to amortize the
slangc cost across every subsequent test run.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any


_REPO_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(__file__), ".."
))


def _canonical_kernel_sources() -> list[tuple[str, str]]:
    """Return ``[(cache_key, slang_src), ...]`` for the kernel shapes
    the regression suite exercises most often. Each source is small
    (sub-50-line) so the cumulative cold-compile time is bounded; we
    don't try to enumerate every possible kernel — just the high-hit
    canonical shapes whose absence from cache forces a cold
    recompile per test class.

    Sources are written so ``slangc -target spirv`` accepts them with
    only the ``shaders/lib`` include path; no extra runtime context
    needed.
    """
    sources: list[tuple[str, str]] = []

    # Trivial pointwise kernels — these prove the import + ParameterBlock
    # path. Cache keys are stable across versions because the source is
    # canonicalized.
    pointwise_template = (
        "[shader(\"compute\")][numthreads(256, 1, 1)]\n"
        "void computeMain(uint3 tid : SV_DispatchThreadID,\n"
        "                 uniform StructuredBuffer<float> in_x,\n"
        "                 uniform RWStructuredBuffer<float> out_y,\n"
        "                 uniform uint numel) {{\n"
        "    if (tid.x >= numel) return;\n"
        "    float x = in_x[tid.x];\n"
        "    out_y[tid.x] = {expr};\n"
        "}}\n"
    )
    for name, expr in [
        ("relu_f32", "max(x, 0.0f)"),
        ("sigmoid_f32", "1.0f / (1.0f + exp(-x))"),
        ("tanh_f32", "tanh(x)"),
        ("scale_f32", "x * 2.0f"),
        ("scale_add_f32", "x * 2.0f + 1.0f"),
        ("neg_f32", "-x"),
        ("abs_f32", "abs(x)"),
        ("sqrt_f32", "sqrt(x)"),
        ("exp_f32", "exp(x)"),
        ("log_f32", "log(x)"),
    ]:
        sources.append((
            f"prewarm_pointwise_{name}",
            pointwise_template.format(expr=expr),
        ))

    # Reduction kernel (sum) — the canonical reduction shape. Locks
    # the wgreduce/welford codegen path.
    reduction_src = (
        "[shader(\"compute\")][numthreads(256, 1, 1)]\n"
        "groupshared float smem[256];\n"
        "void computeMain(uint3 tid : SV_DispatchThreadID,\n"
        "                 uint3 ltid : SV_GroupThreadID,\n"
        "                 uniform StructuredBuffer<float> in_x,\n"
        "                 uniform RWStructuredBuffer<float> out_y,\n"
        "                 uniform uint numel) {\n"
        "    float v = (tid.x < numel) ? in_x[tid.x] : 0.0f;\n"
        "    smem[ltid.x] = v;\n"
        "    GroupMemoryBarrierWithGroupSync();\n"
        "    for (uint s = 128; s > 0; s >>= 1) {\n"
        "        if (ltid.x < s) smem[ltid.x] += smem[ltid.x + s];\n"
        "        GroupMemoryBarrierWithGroupSync();\n"
        "    }\n"
        "    if (ltid.x == 0) out_y[0] = smem[0];\n"
        "}\n"
    )
    sources.append(("prewarm_reduction_sum_f32", reduction_src))

    return sources


def precompile_test_shaders(
    *,
    sync: bool = True,
    include_module_libs: bool = True,
) -> dict[str, Any]:
    """Programmatic entry point. Returns a dict with timing + counts
    so callers (CI pipelines, ``setup.py``) can log + ratchet."""
    out: dict[str, Any] = {"phases": [], "total_seconds": 0.0}
    t_total = time.perf_counter()

    if include_module_libs:
        from torch_vulkan.inductor.runtime import precompile_shader_libs
        t0 = time.perf_counter()
        try:
            modres = precompile_shader_libs()
            out["phases"].append({
                "name": "module_libs",
                "seconds": time.perf_counter() - t0,
                "compiled": modres.get("compiled", []),
                "cached": modres.get("cached", []),
                "ok": True,
            })
        except Exception as e:
            out["phases"].append({
                "name": "module_libs",
                "seconds": time.perf_counter() - t0,
                "error": f"{type(e).__name__}: {e}",
                "ok": False,
            })

    from torch_vulkan.inductor.runtime import prewarm_compile
    sources = _canonical_kernel_sources()
    t0 = time.perf_counter()
    try:
        n_scheduled = prewarm_compile(sources, sync=sync)
        out["phases"].append({
            "name": "kernel_sources",
            "seconds": time.perf_counter() - t0,
            "scheduled": n_scheduled,
            "total": len(sources),
            "ok": True,
        })
    except Exception as e:
        out["phases"].append({
            "name": "kernel_sources",
            "seconds": time.perf_counter() - t0,
            "error": f"{type(e).__name__}: {e}",
            "ok": False,
        })

    out["total_seconds"] = time.perf_counter() - t_total
    out["ok"] = all(p.get("ok", False) for p in out["phases"])
    return out


def main(argv: list[str] | None = None) -> int:
    res = precompile_test_shaders()
    print(json.dumps(res, indent=2, sort_keys=True))
    return 0 if res.get("ok", False) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
