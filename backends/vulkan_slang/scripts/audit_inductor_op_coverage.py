#!/usr/bin/env python3
"""P7.4 — Inductor op-coverage audit.

Walks the Inductor backend's Python sources and reports which aten ops
are explicitly handled vs which would fall back to eager. This is the
discovery loop entry-point: every aten op the backend doesn't lower
(or doesn't have a meta_patch for) is a fallback risk that shows up as
extra dispatches under `torch.compile`.

Output is consumable by humans (CLI summary) and by the regression
suite (`coverage_summary()` returns structured stats so a CI test can
ratchet "no new uncovered ops" without anyone running the script
manually).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass


_INDUCTOR_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "python", "torch_vulkan", "inductor"
))

_ATEN_OP_RE = re.compile(r"\baten\.([A-Za-z_][A-Za-z_0-9]*)")
_OVERRIDE_OP_RE = re.compile(r"^\s*(?:def\s+)?([a-z_][a-z_0-9]*)\s*=?\s*(?:_)?(?:make_)?")


@dataclass(frozen=True)
class CoverageRow:
    op: str
    in_lowerings: bool
    in_meta_patches: bool
    in_overrides: bool


def _extract_aten_ops(path: str) -> set[str]:
    seen: set[str] = set()
    try:
        with open(path) as f:
            for m in _ATEN_OP_RE.finditer(f.read()):
                seen.add(m.group(1))
    except OSError:
        pass
    return seen


def coverage_rows() -> list[CoverageRow]:
    lowerings = _extract_aten_ops(os.path.join(_INDUCTOR_DIR, "lowerings.py"))
    meta = _extract_aten_ops(os.path.join(_INDUCTOR_DIR, "meta_patches.py"))
    overrides = _extract_aten_ops(os.path.join(_INDUCTOR_DIR, "overrides.py"))
    universe = lowerings | meta | overrides
    rows = [
        CoverageRow(
            op=op,
            in_lowerings=op in lowerings,
            in_meta_patches=op in meta,
            in_overrides=op in overrides,
        )
        for op in sorted(universe)
    ]
    return rows


def coverage_summary() -> dict:
    rows = coverage_rows()
    return {
        "total_ops": len(rows),
        "lowered": sum(1 for r in rows if r.in_lowerings),
        "fake_impl": sum(1 for r in rows if r.in_meta_patches),
        "overridden": sum(1 for r in rows if r.in_overrides),
        "any_coverage": sum(
            1 for r in rows
            if r.in_lowerings or r.in_meta_patches or r.in_overrides
        ),
    }


# `torch.library.custom_op("torch_vulkan::<name>", ...)` shims registered in
# `fx_passes.py` (PF.30.a/.b/.d, PF.5, etc.) — each one is a coverage dimension
# distinct from lowerings/meta/overrides. An aten op routed through such a shim
# (e.g. `aten.convolution.default` → `torch_vulkan::conv2d_with_optional_bias`
# via the `_VULKAN_OP_PREWARM_REGISTRY` / FX-pass rewrites) IS covered, even
# though it doesn't appear in `lowerings.py`.
_OP_NAME_RE = re.compile(
    r'op_name\s*=\s*"(torch_vulkan::[a-z_][a-z_0-9]*)"',
)


def _extract_shim_names() -> set[str]:
    """Return the set of `torch_vulkan::*` custom-op shim names declared in
    Inductor source files. The discovery is regex-based (string literal match)
    so it works without importing the modules — keeps the audit cheap and
    immune to in-flight runtime breakage.
    """
    seen: set[str] = set()
    for fname in ("fx_passes.py", "lowerings.py", "overrides.py", "wrapper.py"):
        path = os.path.join(_INDUCTOR_DIR, fname)
        try:
            with open(path) as f:
                for m in _OP_NAME_RE.finditer(f.read()):
                    seen.add(m.group(1))
        except OSError:
            pass
    return seen


# Mapping aten op (full-name minus `aten.` prefix and overload) → shim name
# the FX-pass rewrites it to. Documented at audit time so the gap report can
# attribute coverage to the right shim. Update when a new shim is added.
_ATEN_TO_SHIM: dict[str, str] = {
    # PF.30.a/.b/.d shims.
    "convolution.default": "torch_vulkan::conv2d_with_optional_bias",
    "convolution_overrideable": "torch_vulkan::conv2d_with_optional_bias",
    "max_pool2d.default": "torch_vulkan::max_pool2d",
    "max_pool2d_with_indices.default": "torch_vulkan::max_pool2d",
    "scaled_dot_product_attention": "torch_vulkan::sdpa_with_optional_mask",
    # PF.5 epilogue shim (mm + bias + gelu).
    "addmm.default": "torch_vulkan::addmm_gelu_fused",
    # PF.B fused shims.
    "_scaled_dot_product_efficient_attention": "torch_vulkan::sdpa_with_optional_mask",
}


def _model_op_usage_uncached() -> dict[str, set[str]]:
    """FX-trace each model in `benchmarks/inductor_train.py` (on CPU,
    real tensors — fastest path; we throw away the result) and collect
    the set of aten ops each model touches.

    Returns ``{model_name: {aten.op.overload, ...}}``. Failures (e.g.
    `torchvision` not installed for ``resnet18``) are silently skipped —
    the model's entry is omitted from the result rather than poisoning
    the audit.
    """
    import importlib
    import sys
    bench_dir = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "benchmarks",
    ))
    if bench_dir not in sys.path:
        sys.path.insert(0, bench_dir)
    try:
        import inductor_train as it
    except Exception:
        return {}

    try:
        import torch
        from torch.fx.experimental.proxy_tensor import make_fx
    except Exception:
        return {}

    out: dict[str, set[str]] = {}
    for name, builder in it._MODELS.items():
        try:
            model, make_inputs = builder()
            model = model.eval()
            # Force everything to CPU + no_grad — FX-tracing must not touch
            # Vulkan (which would import the runtime, defeating the
            # build-break-immunity).
            inputs = make_inputs()
            for p in model.parameters():
                p.requires_grad_(False)
            def _fn(*args):
                with torch.no_grad():
                    return model(*args)
            gm = make_fx(_fn, tracing_mode="real")(*inputs)
        except (Exception, SystemExit):
            # `SystemExit` covers builders like `_resnet18` that raise it
            # when `torchvision` isn't installed. Treat as "model
            # unavailable, skip" rather than failing the whole audit.
            continue
        ops: set[str] = set()
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            try:
                target = node.target
                ns = target.namespace
                opname = target._opname
                overload = target._overloadname
                ops.add(f"{ns}.{opname}.{overload}")
            except AttributeError:
                # builtin ops or non-OpOverload; record by str(target)
                ops.add(str(target))
        out[name] = ops
    return out


_MODEL_OP_USAGE_CACHE: dict[str, set[str]] | None = None


def model_op_usage() -> dict[str, set[str]]:
    """Cached wrapper around ``_model_op_usage_uncached``. The trace is
    deterministic given the bench registry, so caching keeps the audit
    cheap when called from multiple test methods.
    """
    global _MODEL_OP_USAGE_CACHE
    if _MODEL_OP_USAGE_CACHE is None:
        _MODEL_OP_USAGE_CACHE = _model_op_usage_uncached()
    return _MODEL_OP_USAGE_CACHE


@dataclass(frozen=True)
class UndeclaredOpRow:
    """One aten op used by ≥1 benchmark model with no Vulkan lowering,
    no fake_impl, no override, and no `torch_vulkan::*` shim. Each row
    is a future PF.54/PF.55-class blocker waiting to happen.
    """

    op: str  # full overload name, e.g. "aten.flatten.using_ints"
    bare_op: str  # bare name (after `aten.`, before overload), e.g. "flatten"
    models_using_it: tuple[str, ...]
    suggested_owner: str  # which file the lowering / fake should live in


def _suggested_owner(bare_op: str) -> str:
    """Heuristic: `view`/`reshape`/`flatten`/`squeeze`-class ops → meta_patches
    (FakeTensor view-fast-path is the historical pain point). All other
    structural ops → lowerings. Pure pointwise → overrides.
    """
    view_class = {
        "view", "reshape", "flatten", "unflatten", "squeeze", "unsqueeze",
        "permute", "transpose", "expand", "select", "slice", "as_strided",
        "_unsafe_view", "narrow", "split", "chunk",
    }
    pointwise_class = {
        "add", "sub", "mul", "div", "neg", "abs", "exp", "log", "sqrt",
        "rsqrt", "pow", "sigmoid", "tanh", "relu", "gelu", "silu", "erf",
    }
    if bare_op in view_class:
        return "meta_patches.py"
    if bare_op in pointwise_class:
        return "overrides.py"
    return "lowerings.py"


def _vulkan_overrides_methods() -> set[str]:
    """Return method names defined on `VulkanOverrides` (the
    `OpOverrides` subclass that maps Inductor's internal pointwise ops
    to Slang snippets). An aten op covered by an entry here is
    considered "overridden" — Inductor's pointwise decomposition
    routes through this class.
    """
    seen: set[str] = set()
    path = os.path.join(_INDUCTOR_DIR, "overrides.py")
    method_re = re.compile(
        r"^[ \t]+(?:@staticmethod\s*\n[ \t]+)?def\s+([a-z_][a-z_0-9]*)\s*\(",
        re.MULTILINE,
    )
    try:
        with open(path) as f:
            for m in method_re.finditer(f.read()):
                seen.add(m.group(1))
    except OSError:
        pass
    return seen


# Aten ops that Inductor's *built-in* decomposition handles before they
# reach our backend — i.e. they're never observed as a call_function in a
# Vulkan-bound graph. Treat as "covered upstream".
_BUILTIN_DECOMPOSED: frozenset[str] = frozenset({
    # View / reshape family — Inductor decomposes these into
    # `as_strided` / `view` / strided tensor ops at lowering time;
    # they never hit our backend as call_functions in a real
    # `torch.compile` graph (FX-tracing via `make_fx` keeps them, but
    # only for the upstream view layer).
    "select", "transpose", "squeeze", "unsqueeze", "expand", "t",
    "permute", "view", "reshape", "flatten", "_unsafe_view",
    "as_strided",
    # Builtin pointwise that Inductor decomposes through its Pointwise
    # IR layer using `VulkanOverrides` methods (gelu, relu, etc.).
    "gelu", "relu", "leaky_relu",
    # In-place scalar variants — Inductor decomposes
    # `aten.add_.Scalar` to `aten.add_.Scalar(self, scalar)` which is
    # pointwise + write-back in our wrapper.
    "add_", "sub_", "mul_", "div_",
})


def undeclared_op_rows() -> list[UndeclaredOpRow]:
    """Return aten ops used by ≥1 benchmark model that have no coverage
    in any of the four dimensions: lowering, fake_impl, override,
    `torch_vulkan::*` shim, plus the upstream-decomposed allow-list.
    Sorted by `op` for deterministic output.
    """
    usage = model_op_usage()
    if not usage:
        return []

    cov_aten = {r.op for r in coverage_rows()}  # bare-aten names
    overrides = _vulkan_overrides_methods()
    shims = _extract_shim_names()

    # Aggregate per-op model membership.
    op_to_models: dict[str, set[str]] = {}
    for model, ops in usage.items():
        for op in ops:
            op_to_models.setdefault(op, set()).add(model)

    rows: list[UndeclaredOpRow] = []
    for op, models in sorted(op_to_models.items()):
        # Only aten.* ops are in scope; builtin Python ops are noise.
        if not op.startswith("aten."):
            continue
        # Strip aten. prefix.
        rest = op[len("aten."):]
        # CPU-only ops surface during FX-tracing (which runs on CPU)
        # but never reach the Vulkan backend — the dispatcher never
        # picks the `_for_cpu` overload for a Vulkan tensor. Filter
        # them so the audit reports actionable Vulkan-side gaps.
        if "_for_cpu" in rest:
            continue
        # Bare op name = first dotted component (for `gelu.default`,
        # bare = `gelu`; for `add.Scalar`, bare = `add`).
        bare = rest.split(".")[0] if "." in rest else rest
        # Coverage check 1: aten-name match (covers
        # `register_lowering(aten.X.Y, ...)` style).
        if bare in cov_aten:
            continue
        # Coverage check 2: VulkanOverrides method.
        if bare in overrides:
            continue
        # Coverage check 3: upstream-decomposed allow-list.
        if bare in _BUILTIN_DECOMPOSED:
            continue
        # Coverage check 4: `torch_vulkan::*` custom-op shim.
        if rest in _ATEN_TO_SHIM and _ATEN_TO_SHIM[rest] in shims:
            continue
        if any(
            rest.startswith(k + ".") and _ATEN_TO_SHIM[k] in shims
            for k in _ATEN_TO_SHIM
        ):
            continue
        rows.append(UndeclaredOpRow(
            op=op,
            bare_op=bare,
            models_using_it=tuple(sorted(models)),
            suggested_owner=_suggested_owner(bare),
        ))
    return rows


def undeclared_op_summary() -> dict:
    rows = undeclared_op_rows()
    return {
        "total_undeclared": len(rows),
        "rows": [
            {
                "op": r.op, "bare_op": r.bare_op,
                "models_using_it": list(r.models_using_it),
                "suggested_owner": r.suggested_owner,
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# PF.61 — Backward-extern audit
# ---------------------------------------------------------------------------

_BWD_OP_RE = re.compile(r"\baten\.([a-z_0-9]*_backward[a-z_0-9]*)\b")


def _covered_backward_ops() -> set[str]:
    """Bare aten backward-op names that have a lowering/fake/override."""
    covered: set[str] = set()
    for fname in ("lowerings.py", "meta_patches.py", "overrides.py"):
        path = os.path.join(_INDUCTOR_DIR, fname)
        try:
            with open(path) as f:
                for m in _BWD_OP_RE.finditer(f.read()):
                    covered.add(m.group(1))
        except OSError:
            pass
    return covered


def _make_joint_capture(ops_set: set) -> "Callable":
    """Return a compiler callback that records all call_function op names."""
    def _capture(gm, _example_inputs):
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            t = node.target
            try:
                ops_set.add(f"{t.namespace}.{t._opname}.{t._overloadname}")
            except AttributeError:
                pass
        return gm.forward
    return _capture


def _model_joint_op_usage() -> dict[str, set[str]]:
    """Trace each benchmark model in training mode (fwd + bwd) on CPU.

    Uses AOT autograd to capture the joint fwd+bwd graph.
    Build-break-immune: any model that fails to trace is silently skipped.
    """
    import sys
    bench_dir = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "benchmarks",
    ))
    if bench_dir not in sys.path:
        sys.path.insert(0, bench_dir)
    try:
        import inductor_train as it
    except (Exception, SystemExit):
        return {}
    try:
        import torch
        from torch._functorch.aot_autograd import aot_function
    except (Exception, SystemExit):
        return {}

    out: dict[str, set[str]] = {}
    for name, builder in it._MODELS.items():
        ops: set[str] = set()
        try:
            model, make_inputs = builder()
            model = model.train()
            inputs = make_inputs()
            for p in model.parameters():
                p.requires_grad_(True)

            capture = _make_joint_capture(ops)

            def fn(*args):
                return model(*args).sum()

            compiled = aot_function(fn, fw_compiler=capture, bw_compiler=capture)
            result = compiled(*inputs)
            result.backward()
        except (Exception, SystemExit):
            pass
        if ops:
            out[name] = ops
    return out


def backward_extern_summary() -> dict:
    """PF.61 — count aten.*_backward ops in training graphs with no lowering.

    Traces each benchmark model's joint fwd+bwd graph on CPU via AOT
    autograd, then cross-references discovered backward ops against the
    four coverage dimensions (lowerings, meta_patches, overrides, shims).
    An op that appears in the joint graph but has no coverage entry is
    counted as a ``backward extern`` — it will route through
    ``extern_kernels.<op>`` under Inductor, breaking fusion on the
    backward graph.

    Returns ``{"total_backward_extern": N, "rows": [...]}``.
    Build-break-immune: trace failures contribute zero rows; only
    successfully traced models contribute to the count.
    """
    covered = _covered_backward_ops()
    usage = _model_joint_op_usage()

    op_to_models: dict[str, set[str]] = {}
    for model_name, ops in usage.items():
        for op in ops:
            if not (op.startswith("aten.") and "_backward" in op):
                continue
            op_to_models.setdefault(op, set()).add(model_name)

    rows = []
    for op, models in sorted(op_to_models.items()):
        bare = op[len("aten."):].split(".")[0] if "." in op[len("aten."):] else op[len("aten."):]
        if bare in covered:
            continue
        rows.append({"op": op, "models_using_it": sorted(models)})

    return {"total_backward_extern": len(rows), "rows": rows}


def markdown_report() -> str:
    """Render a Markdown report bundling all four audit dimensions:
    op-coverage summary, wrapper-emit imports, slangc smoke (CLI surface),
    and undeclared-op gaps with `(op, models, suggested_owner)`. Suitable
    for pasting into a roadmap entry or a PR description.
    """
    cov = coverage_summary()
    emit = emit_import_summary()
    undec = undeclared_op_summary()
    lines = [
        "# Inductor op-coverage audit",
        "",
        "## Coverage summary",
        "",
        f"- total_ops: **{cov['total_ops']}**",
        f"- lowered: {cov['lowered']}",
        f"- fake_impl: {cov['fake_impl']}",
        f"- overridden: {cov['overridden']}",
        f"- any_coverage: **{cov['any_coverage']}**",
        "",
        "## Wrapper-emit imports",
        "",
        f"- total_imports: {emit['total_imports']}",
        f"- ok: {emit['ok']}",
        f"- broken: **{emit['broken']}**",
    ]
    if emit["broken"]:
        lines.append("")
        for r in emit["broken_rows"]:
            lines.append(
                f"  - {r['file']}:{r['line']} "
                f"`from {r['module']} import {r['name']}` — {r['error']}"
            )
    lines += [
        "",
        "## Undeclared aten ops used by benchmark models",
        "",
    ]
    if not undec["rows"]:
        lines.append("_None — every aten op used by a benchmark has at "
                     "least one coverage dimension._")
    else:
        lines.append("| Op | Bare | Models | Suggested owner |")
        lines.append("|----|------|--------|------------------|")
        for r in undec["rows"]:
            models = ", ".join(r["models_using_it"])
            lines.append(
                f"| `{r['op']}` | `{r['bare_op']}` | {models} | "
                f"`{r['suggested_owner']}` |"
            )
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class EmitImportRow:
    """One `from <module> import <name>` line emitted by the wrapper-codegen
    headers. ``ok`` is True when ``getattr(import_module(module), name)``
    resolves at audit time; False when the symbol is undefined (the bug
    class that broke MLP/CNN compile in PF.54-era).
    """

    source_file: str
    line: int
    module: str
    name: str
    alias: str
    ok: bool
    error: str  # empty when ok


# `from <module> import <name> [as <alias>]` inside a Python string
# literal (single OR multi-line). The wrapper emits these as part of
# `header.splice("...")`. Captures: module, name, optional alias.
_EMIT_IMPORT_RE = re.compile(
    r'"\s*from\s+([\w.]+)\s+import\s+([\w]+)(?:\s+as\s+([\w]+))?\s*'
    r'\\?n?\s*"',
)


def _wrapper_emit_sources() -> list[str]:
    """Files whose string literals get spliced into the Inductor-generated
    wrapper Python source. Today the only emitter is `wrapper.py`; if a
    new emitter ships (e.g. `wrapper_subgraph.py`), add it here.
    """
    return [
        os.path.join(_INDUCTOR_DIR, "wrapper.py"),
    ]


def emit_import_rows() -> list[EmitImportRow]:
    """Scan wrapper-emit sources for `from <module> import <name>` lines
    embedded in Python string literals, then verify each one resolves.

    Catches the PF.54-era bug class: wrapper emits an import that no
    longer exists in the package surface (e.g. ``from torch_vulkan
    import _empty_strided_vulkan`` after a refactor strips the symbol),
    breaking every compiled graph silently until a runtime ImportError
    surfaces inside torch.compile's wrapper-execution path. The audit
    surfaces the failure at static-analysis time instead.
    """
    import importlib
    rows: list[EmitImportRow] = []
    for src in _wrapper_emit_sources():
        try:
            with open(src) as f:
                lines = f.readlines()
        except OSError:
            continue
        for ln_idx, line in enumerate(lines, 1):
            for m in _EMIT_IMPORT_RE.finditer(line):
                module, name, alias = m.group(1), m.group(2), m.group(3)
                if alias is None:
                    alias = name
                ok, err = True, ""
                try:
                    mod = importlib.import_module(module)
                    if not hasattr(mod, name):
                        ok, err = False, (
                            f"module {module!r} has no attribute {name!r}"
                        )
                except Exception as e:
                    ok, err = False, f"{type(e).__name__}: {e}"
                rows.append(EmitImportRow(
                    source_file=os.path.relpath(src, _INDUCTOR_DIR),
                    line=ln_idx, module=module, name=name, alias=alias,
                    ok=ok, error=err,
                ))
    return rows


def emit_import_summary() -> dict:
    """Structured summary the regression test asserts on. ``broken``
    is the count that matters — a non-zero value means a future commit
    will silently break torch.compile.
    """
    rows = emit_import_rows()
    return {
        "total_imports": len(rows),
        "ok": sum(1 for r in rows if r.ok),
        "broken": sum(1 for r in rows if not r.ok),
        "broken_rows": [
            {
                "file": r.source_file,
                "line": r.line,
                "module": r.module,
                "name": r.name,
                "alias": r.alias,
                "error": r.error,
            }
            for r in rows if not r.ok
        ],
    }


# Canonical slangc smoke snippets — minimal proxies for the kernel
# shapes Inductor's wrapper-codegen actually emits. Each snippet must
# compile clean against the current `slangc` + `shaders/lib` modules.
# A teammate change that breaks slangc compile of one of these (e.g.
# helpers.slang loses an exported symbol, or an mm template mutation
# tickles a slangc bug) flips the audit's `slangc_smoke_summary()`
# from green to broken — the regression surfaces *before* a compiled
# graph fails inside `torch.compile`.
#
# Snippets must be self-contained — no Inductor wrapper headers, no
# push-constant struct chasing into runtime helpers. Each one mirrors
# a real wrapper-emitted kernel pattern.
_SLANGC_SMOKE_SNIPPETS: dict[str, str] = {
    # Bare-pointwise: relu. The "no helpers needed" path. Locks that
    # the simplest possible compile path stays clean.
    "smoke_pointwise_relu": (
        "[[vk::binding(0)]] StructuredBuffer<float> in_x;\n"
        "[[vk::binding(1)]] RWStructuredBuffer<float> out_y;\n"
        "[shader(\"compute\")] [numthreads(256, 1, 1)]\n"
        "void computeMain(uint3 tid : SV_DispatchThreadID) {\n"
        "    out_y[tid.x] = max(in_x[tid.x], 0.0f);\n"
        "}\n"
    ),
    # Module import + helpers symbol. Locks that `import helpers;` +
    # the special_math `.erf()` extension method resolves. The exact
    # snippet that failed slangc during the in-flight PF.5 (the helper
    # was looked up against a stale `helpers.slang-module`).
    # NOTE: erf is exposed as a float extension method `(x).erf()` via
    # special_math.slang, re-exported by helpers.slang. There is no
    # standalone `c10_vulkan_erf()` free function.
    "smoke_pointwise_gelu_import_helpers": (
        "[[vk::binding(0)]] StructuredBuffer<float4> in_x;\n"
        "[[vk::binding(1)]] RWStructuredBuffer<float4> out_y;\n"
        "import helpers;\n"
        "[shader(\"compute\")] [numthreads(256, 1, 1)]\n"
        "void computeMain(uint3 gtid : SV_DispatchThreadID) {\n"
        "    float4 v = in_x[gtid.x];\n"
        "    float4 r;\n"
        "    [unroll] for (uint k = 0u; k < 4u; ++k) {\n"
        "        float x = v[k];\n"
        "        r[k] = 0.5f * x * (1.0f + (x * 0.7071067811865476f).erf());\n"
        "    }\n"
        "    out_y[gtid.x] = r;\n"
        "}\n"
    ),
    # Push-constant + bias add. Mirrors the mm-template epilogue
    # pattern (PF.5's fused `mm + bias + gelu` codegen). Locks that
    # push-constant struct + multi-binding + `import helpers;` all
    # compile together.
    "smoke_mm_epilogue_pc": (
        "import helpers;\n"
        "struct PC { uint M; uint N; uint stride_bias_n; };\n"
        "[[vk::push_constant]] PC pc;\n"
        "[[vk::binding(0)]] StructuredBuffer<float> in_a;\n"
        "[[vk::binding(1)]] StructuredBuffer<float> bias;\n"
        "[[vk::binding(2)]] RWStructuredBuffer<float> out_c;\n"
        "[shader(\"compute\")] [numthreads(64, 1, 1)]\n"
        "void computeMain(uint3 tid : SV_DispatchThreadID) {\n"
        "    if (tid.x >= pc.M * pc.N) return;\n"
        "    uint col = tid.x % pc.N;\n"
        "    float v = in_a[tid.x] + bias[col * pc.stride_bias_n];\n"
        "    out_c[tid.x] = 0.5f * v * (1.0f + (v * 0.7071067811865476f).erf());\n"
        "}\n"
    ),
}


@dataclass(frozen=True)
class SlangcSmokeRow:
    """Result of one canonical slangc smoke compile.

    ``ok`` is True when slangc returned exit-0 *and* produced a
    non-empty SPV blob. ``error`` carries the first ~400 chars of
    slangc stderr when ``ok`` is False — enough to point at the
    failing line in the canonical snippet without dumping the entire
    compiler trace into pytest output.
    """

    name: str
    ok: bool
    spv_bytes: int  # 0 when ok=False
    error: str  # empty when ok


def slangc_smoke_rows() -> list[SlangcSmokeRow]:
    """Compile every canonical snippet via the runtime's slangc entry
    point. Returns one row per snippet. When slangc isn't available
    the audit returns rows with ``ok=False, error="slangc unavailable"``
    so callers can skip rather than treating it as a hard regression.
    """
    rows: list[SlangcSmokeRow] = []
    try:
        from torch_vulkan.inductor.runtime import (
            _slangc_available, compile_slang_to_spirv,
        )
    except Exception as e:  # pragma: no cover — import path
        for name in _SLANGC_SMOKE_SNIPPETS:
            rows.append(SlangcSmokeRow(
                name=name, ok=False, spv_bytes=0,
                error=f"runtime import failed: {type(e).__name__}: {e}"[:400],
            ))
        return rows
    if not _slangc_available():
        for name in _SLANGC_SMOKE_SNIPPETS:
            rows.append(SlangcSmokeRow(
                name=name, ok=False, spv_bytes=0,
                error="slangc unavailable",
            ))
        return rows
    for name, src in _SLANGC_SMOKE_SNIPPETS.items():
        try:
            spv = compile_slang_to_spirv(
                src, "computeMain", cache_key=f"audit_{name}",
            )
            rows.append(SlangcSmokeRow(
                name=name, ok=bool(spv), spv_bytes=len(spv or b""),
                error="" if spv else "empty SPV blob",
            ))
        except Exception as e:
            rows.append(SlangcSmokeRow(
                name=name, ok=False, spv_bytes=0,
                error=f"{type(e).__name__}: {e}"[:400],
            ))
    return rows


def slangc_smoke_summary() -> dict:
    """Structured summary the regression test asserts on. ``broken``
    is the count that matters — non-zero means a teammate change
    has broken slangc compile of a canonical snippet, and every
    inductor-emitted kernel of that shape will fail at compile time.
    ``unavailable`` separates "slangc isn't installed" (skip) from
    "slangc rejected our source" (fail).
    """
    rows = slangc_smoke_rows()
    unavailable = sum(
        1 for r in rows if r.error == "slangc unavailable"
    )
    return {
        "total": len(rows),
        "ok": sum(1 for r in rows if r.ok),
        "broken": sum(
            1 for r in rows
            if not r.ok and r.error != "slangc unavailable"
        ),
        "unavailable": unavailable,
        "broken_rows": [
            {"name": r.name, "error": r.error}
            for r in rows
            if not r.ok and r.error != "slangc unavailable"
        ],
    }


def main(argv: list[str] | None = None) -> int:
    import sys
    args = list(argv) if argv is not None else sys.argv[1:]
    if "--markdown" in args:
        print(markdown_report())
        return 0

    s = coverage_summary()
    print("Inductor op-coverage audit")
    print("==========================")
    for k, v in s.items():
        print(f"  {k:14s} = {v}")
    print()
    rows = coverage_rows()
    uncovered = [r.op for r in rows if not (
        r.in_lowerings or r.in_meta_patches or r.in_overrides
    )]
    if uncovered:
        print("Aten ops referenced but with no lowering/meta/override:")
        for op in uncovered:
            print(f"  - aten.{op}")
    print()
    es = emit_import_summary()
    print("Wrapper-emit import audit (PF.54-era class)")
    print("===========================================")
    for k in ("total_imports", "ok", "broken"):
        print(f"  {k:14s} = {es[k]}")
    if es["broken"]:
        print()
        print("Broken wrapper-emitted imports (would fail torch.compile):")
        for row in es["broken_rows"]:
            print(
                f"  - {row['file']}:{row['line']} "
                f"`from {row['module']} import {row['name']}` — "
                f"{row['error']}"
            )
    print()
    ss = slangc_smoke_summary()
    print("Slangc smoke audit (PF.5-era class)")
    print("===================================")
    for k in ("total", "ok", "broken", "unavailable"):
        print(f"  {k:12s} = {ss[k]}")
    if ss["broken"]:
        print()
        print("Broken canonical snippets (would fail every compiled "
              "graph that emits a similar kernel):")
        for row in ss["broken_rows"]:
            print(f"  - {row['name']}: {row['error']}")
    print()
    us = undeclared_op_summary()
    print("Undeclared aten ops used by benchmark models")
    print("=============================================")
    print(f"  total_undeclared = {us['total_undeclared']}")
    if us["rows"]:
        print()
        print("(Each row is a future PF.54/PF.55-class blocker.)")
        for row in us["rows"]:
            models = ", ".join(row["models_using_it"])
            print(
                f"  - {row['op']}  (models: {models}; "
                f"suggested owner: {row['suggested_owner']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
