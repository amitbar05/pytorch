"""PF.58 — custom-op registration ↔ eager monkey-patch consistency audit.

Walks every `_ensure_<opname>_op_registered` function in
`python/torch_vulkan/inductor/fx_passes.py` and asserts that, for ops
which have an eager-side counterpart, `python/torch_vulkan/__init__.py`
contains a matching `F.<opname> = _patched_<opname>` assignment.

Defends against the half-done-ship pattern (observed twice in one loop)
where the FX-pass / Inductor-lowering half of a two-site contract ships
but the eager-side `__init__.py` patch is silently deferred — leaving
the regression suite GREEN on the custom_op contract while live
`torch.compile` of the op fails because eager dispatch never reaches
the registered shim.

The audit's allowlist makes the contract explicit:
  - REQUIREMENTS: ops that rewrite an existing `F.<x>` (conv2d, sdpa,
    max_pool2d, etc.). Each requires `F.<x>` and `_patched_<x>` in the
    package surface.
  - FUSED_ONLY: graph-rewrite targets with no eager counterpart
    (swiglu, addmm_gelu, qkv_cat, scaled_bmm, flash_attention). No
    `F.<x>` assignment expected.

A new `_ensure_*` site that is neither in REQUIREMENTS nor in
FUSED_ONLY is reported as "unclassified" — fail-loud so the author
adds the new entry consciously.

CLI: `python3 scripts/audit_custom_op_eager_patches.py` prints a human-
readable section. Test class
`TestCustomOpEagerPatchConsistency::test_every_registered_custom_op_has_eager_patch`
asserts `eager_patch_summary()["broken"] == 0`.
"""
from __future__ import annotations

import os
import re
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.normpath(os.path.join(_SCRIPTS_DIR, ".."))
# Track 1 (codegen-refactor) split the original `fx_passes.py` monolith
# into a package; the `_ensure_*_op_registered` factories now live in
# `fx_passes/eager_patches.py` (and `fx_passes/__init__.py` re-exports
# them).  Look at the package directory; the audit walks every `*.py`
# file under it so future moves are picked up automatically.
_FX_PASSES_DIR = os.path.join(
    _BACKEND_ROOT, "python", "torch_vulkan", "inductor", "fx_passes",
)
_INIT_PY = os.path.join(_BACKEND_ROOT, "python", "torch_vulkan", "__init__.py")

_ENSURE_RE = re.compile(r"^def\s+(_ensure_(\w+)_op_registered)\s*\(", re.MULTILINE)


# Ops that wrap an existing `F.<name>` and need an eager-side
# `__init__.py` patch. Maps `_ensure_*` function name → expected eager
# `F.<name>` symbol. The `_patched_<name>` definition must also live in
# `__init__.py`.
REQUIREMENTS: dict[str, str] = {
    "_ensure_conv2d_with_optional_bias_op_registered": "conv2d",
    "_ensure_conv1d_with_optional_bias_op_registered": "conv1d",
    "_ensure_sdpa_with_optional_mask_op_registered": "scaled_dot_product_attention",
    "_ensure_max_pool2d_op_registered": "max_pool2d",
    # TRAIN.12 (2026-05-08) — adaptive_avg_pool2d gained a custom op
    # with `register_autograd` + `register_fake` so AOTAutograd can
    # trace through it. Mirrors the conv2d/sdpa pattern; eager
    # callers go through `F.adaptive_avg_pool2d`.
    "_ensure_adaptive_avg_pool2d_op_registered": "adaptive_avg_pool2d",
}

# Graph-rewrite-only custom_ops with no eager `F.<name>` counterpart —
# Inductor's FX passes are the only callers; eager users never invoke
# them by name. New fused-only ops join this set.
FUSED_ONLY: set[str] = {
    "_ensure_scaled_bmm_op_registered",
    "_ensure_flash_attention_op_registered",
    "_ensure_swiglu_op_registered",
    "_ensure_addmm_gelu_op_registered",
    "_ensure_qkv_cat_op_registered",
    # T4.8 (2026-05-05) — foreach optimizer custom_ops are reached
    # exclusively via the FX-pass `_route_foreach_add_to_template`
    # rewrite; there's no `F.foreach_sgd_step` etc.  Eager users would
    # call `torch.optim.{SGD,AdamW,Lion}.step()` which lowers to
    # `aten._foreach_*` then gets routed by the pass.
    "_ensure_foreach_sgd_step_op_registered",
    "_ensure_foreach_sgd_momentum_step_op_registered",
    "_ensure_foreach_adamw_step_op_registered",
    "_ensure_foreach_lion_step_op_registered",
}


def _ensure_sites() -> list[tuple[str, str]]:
    """Return [(ensure_fn_name, opname)] discovered in the fx_passes
    package (any `.py` file).  Track 1 split moved these out of the
    original `fx_passes.py` into `fx_passes/eager_patches.py`."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for root, _dirs, files in os.walk(_FX_PASSES_DIR):
        # Skip __pycache__
        if "__pycache__" in root:
            continue
        for fname in files:
            if not fname.endswith(".py"):
                continue
            with open(os.path.join(root, fname)) as f:
                src = f.read()
            for m in _ENSURE_RE.finditer(src):
                fn, op = m.group(1), m.group(2)
                if fn in seen:
                    continue
                seen.add(fn)
                out.append((fn, op))
    return out


def _init_has_patch(opname: str) -> tuple[bool, str]:
    """Verify `python/torch_vulkan/__init__.py` reassigns `F.<opname>`
    to a Vulkan-aware shim (any `_patched_<X>` callable).

    The audit defends the contract "F.<x> is patched" — the load-bearing
    line is the assignment that diverts eager dispatch into the shim.
    The shim's def name is incidental (e.g. `_patched_sdpa` is bound to
    `F.scaled_dot_product_attention`).

    Returns ``(ok, error)``. ``error`` is empty when ``ok`` is True.
    """
    with open(_INIT_PY) as f:
        src = f.read()
    assign_pat = re.compile(
        rf"\bF\.{re.escape(opname)}\s*=\s*(_patched_\w+)\b"
    )
    m = assign_pat.search(src)
    if not m:
        return False, f"missing `F.{opname} = _patched_*` assignment"
    bound_name = m.group(1)
    def_pat = re.compile(rf"\bdef\s+{re.escape(bound_name)}\s*\(")
    if not def_pat.search(src):
        return False, (
            f"`F.{opname} = {bound_name}` assigned but "
            f"`def {bound_name}(...)` not defined in __init__.py"
        )
    return True, ""


def eager_patch_summary() -> dict:
    """Audit driver — returns counts + broken-row + unclassified-row
    detail consumed by the regression test class."""
    sites = _ensure_sites()
    broken_rows: list[dict] = []
    unclassified_rows: list[str] = []
    checked = 0
    for ensure_fn, _opname in sites:
        if ensure_fn in FUSED_ONLY:
            continue
        if ensure_fn not in REQUIREMENTS:
            unclassified_rows.append(ensure_fn)
            continue
        expected = REQUIREMENTS[ensure_fn]
        checked += 1
        ok, err = _init_has_patch(expected)
        if not ok:
            broken_rows.append({
                "ensure_fn": ensure_fn,
                "expected_fn": expected,
                "error": err,
            })
    return {
        "total_ensure_sites": len(sites),
        "checked": checked,
        "broken": len(broken_rows),
        "broken_rows": broken_rows,
        "unclassified": len(unclassified_rows),
        "unclassified_rows": unclassified_rows,
    }


def _main() -> int:
    s = eager_patch_summary()
    print("Custom-op eager-patch audit (PF.58)")
    print(f"  ensure-sites discovered: {s['total_ensure_sites']}")
    print(f"  checked against requirement-map: {s['checked']}")
    print(f"  broken: {s['broken']}")
    for r in s["broken_rows"]:
        print(
            f"    - {r['ensure_fn']} → F.{r['expected_fn']} = "
            f"_patched_{r['expected_fn']}: {r['error']}"
        )
    print(f"  unclassified: {s['unclassified']}")
    for fn in s["unclassified_rows"]:
        print(f"    - {fn} (add to REQUIREMENTS or FUSED_ONLY)")
    return 0 if s["broken"] == 0 and s["unclassified"] == 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
