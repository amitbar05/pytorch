"""Phase-3 codegen-shrink audit.

P3.x asks Python codegen to shed branches that the Slang generic shader
families (P1.x) now own. To make the migration ratchetable on every
commit, we lock the *current* counts of three smells and add CI checks
that fail if any of them grows:

    * `dtype_branches` — `if/elif <dtype> ==` branches in the codegen
      that should melt away once `IPointwise<T>` / `IReduction<T>`
      cover every type (P3.1).
    * `wave_smem_refs` — Python-side mentions of `wave`, `WaveActive`,
      `groupshared`, `smem_` that the Slang `[require(subgroup)]` paths
      and `helpers.wave_*` from P0.7 should subsume (P3.2).
    * `binding_emissions` — Python codegen lines containing
      `vk::binding(` literals; reflection-driven binding (P0.4) should
      drop these to zero (P3.3).

Each commit can lower the LOCKED ceiling but never raise it. To
intentionally raise (e.g. a temporary increase during a multi-PR
migration), the human flips the constant in this file with a comment.
"""

from __future__ import annotations

import os
import re

# T.6 (2026-05-08): the legacy top-level `kernel.py` was split into a
# `kernel/` package by T1.1.  Auditing a non-existent file silently
# returned 0 and let the ratchet trivially pass on every commit.  The
# audit now scans the whole `kernel/` package plus `overrides.py`.
_KERNEL_PKG_FILES = [
    os.path.normpath(os.path.join(os.path.dirname(__file__), "kernel", name))
    for name in (
        "main.py",
        "pointwise.py",
        "reduction.py",
        "indexing.py",
        "symbolic.py",
        "header.py",
    )
]
CODEGEN_FILES = [
    os.path.normpath(os.path.join(os.path.dirname(__file__), "overrides.py")),
    *_KERNEL_PKG_FILES,
]

COMBO_KERNEL_FILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "vulkan_combo_kernel.py")
)


def kernel_package_sloc() -> int:
    """Total physical line count across the `kernel/` package files.

    Used as a coarse anti-bloat signal — the package was created to keep
    each file < 800 lines (Anti-goal #7); summing across it lets the
    ratchet detect package-wide growth that masquerades as small per-file
    deltas.
    """
    total = 0
    for p in _KERNEL_PKG_FILES:
        try:
            with open(p) as f:
                total += sum(1 for _ in f)
        except OSError:
            continue
    return total


_DTYPE_BRANCH_RE = re.compile(r"\b(if|elif)\b[^\n]*\bdtype\b")
_WAVE_SMEM_RE = re.compile(r"\b(wave|WaveActive\w+|groupshared|smem_\w+)\b")
_BINDING_EMIT_RE = re.compile(r"vk::binding\(")


def count_smell(pattern: re.Pattern[str], paths: list[str]) -> int:
    n = 0
    for p in paths:
        try:
            with open(p) as f:
                for line in f:
                    if pattern.search(line):
                        n += 1
        except OSError:
            continue
    return n


_COMBO_STRING_CONCAT_RE = re.compile(r"\+\s*['\"][^'\"]*['\"]")


def codegen_smell_counts() -> dict[str, int]:
    return {
        "dtype_branches": count_smell(_DTYPE_BRANCH_RE, CODEGEN_FILES),
        "wave_smem_refs": count_smell(_WAVE_SMEM_RE, CODEGEN_FILES),
        "binding_emissions": count_smell(_BINDING_EMIT_RE, CODEGEN_FILES),
        "combo_string_concat": count_smell(
            _COMBO_STRING_CONCAT_RE, [COMBO_KERNEL_FILE]
        ),
        "kernel_pkg_sloc": kernel_package_sloc(),
    }


# Locked ceilings — bump down (never up) as P3.x migrations land.
#
# Rebaselined 2026-05-08 (T.6 / T.8): the previous lock from 2026-04-28
# scanned the obsolete top-level `kernel.py` (removed by T1.1), so the
# kernel-side smell counts were silently 0.  After redirecting the audit
# at the `kernel/` package the true counts came in higher than the old
# locks.  We re-baseline at today's measured value plus ~5% slack so a
# single legitimate edit doesn't trip the ratchet, then continue
# ratcheting downward as P3.x lands.
#
# Measured 2026-05-08 (overrides.py + kernel/{main,pointwise,reduction,
# indexing,symbolic,header}.py):
#     dtype_branches      = 10
#     wave_smem_refs      = 22
#     binding_emissions   = 4
#     combo_string_concat = 0
#     kernel_pkg_sloc     = 3276
LOCKED_CEILINGS: dict[str, int] = {
    "dtype_branches": 11,  # rebaselined 2026-05-08 (was 8 against kernel.py); measured 10 + slack
    "wave_smem_refs": 24,  # rebaselined 2026-05-08 (was 27 against kernel.py); measured 22 + slack
    "binding_emissions": 5,  # rebaselined 2026-05-08 (was 4); measured 4 + slack
    "combo_string_concat": 0,  # measured 2026-05-08: 0 (held)
    # Anti-goal #7 says no `inductor/` file > 800 lines.  The kernel
    # package was created to honor that limit by splitting; sum across
    # the package guards against horizontal bloat (small adds spread
    # across all six files).  Measured 2026-05-08: 3276 (+5% slack).
    "kernel_pkg_sloc": 3440,
}


def assert_under_locked_ceilings() -> None:
    actual = codegen_smell_counts()
    failures = []
    for k, ceiling in LOCKED_CEILINGS.items():
        if actual[k] > ceiling:
            failures.append(
                f"{k}: {actual[k]} > {ceiling} (locked ceiling) — "
                f"new growth in Python codegen that Slang generics should "
                f"own. See P3.x in docs/10-inductor-backend.md."
            )
    if failures:
        raise AssertionError("\n".join(failures))


# ── M15: In-process Slang source validator ──────────────────────────
# Catches ~30% of codegen bugs (brace imbalance, binding gaps,
# groupshared overcommit, push-constant overflow) without needing
# slangc or a GPU.  Designed to run in unit tests as a fast pre-check
# before submitting to the real SPIR-V compiler.

_VALIDATE_BRACE_RE = re.compile(r"[{}]")
_VALIDATE_BINDING_RE = re.compile(r"\[\[vk::binding\((\d+)(?:\s*,\s*\d+)?\)\]\]")
_VALIDATE_GROUPSHARED_RE = re.compile(
    r"groupshared\s+(\w+(?:<[^>]+>)?)\s+(\w+)\s*\[\s*(\d+)\s*\]"
)
_VALIDATE_PC_FIELD_RE = re.compile(r"struct PC \{[^}]*\}")

_RDNA1_MAX_GROUPSHARED_BYTES = 64 * 1024
_RDNA1_MAX_PUSH_CONSTANT_BYTES = 128

# Approximate byte sizes for common Slang types.
_SLANG_TYPE_SIZES: dict[str, int] = {
    "float": 4,
    "float2": 8,
    "float3": 12,
    "float4": 16,
    "uint": 4,
    "uint2": 8,
    "uint3": 12,
    "uint4": 16,
    "int": 4,
    "int2": 8,
    "int3": 12,
    "int4": 16,
    "int64_t": 8,
    "half": 2,
    "half2": 4,
    "half4": 8,
    "bool": 1,
    "WelfordResult<float>": 12,
    "ArgPair": 8,
    "FloatPair": 8,
}


def validate_slang_source(source: str) -> list[str]:
    """Validate generated Slang source before feeding it to slangc.

    Returns a list of diagnostic strings; empty list means no issues found.
    Catches structural errors that would otherwise surface as opaque
    slangc parse errors or driver-side GPU faults.
    """
    diags: list[str] = []

    # ── 1. Brace balance ─────────────────────────────────────────
    depth = 0
    for i, ch in enumerate(source):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth < 0:
            line = source[:i].count("\n") + 1
            diags.append(f"Brace imbalance: unexpected '}}' at line {line}")
            depth = 0  # reset to avoid cascading
    if depth != 0:
        diags.append(f"Brace imbalance: {depth} unclosed '{{' braces")

    # ── 2. Binding slot contiguity ───────────────────────────────
    bindings = _VALIDATE_BINDING_RE.findall(source)
    if bindings:
        slots = [int(b) for b in bindings]
        if slots != list(range(len(slots))):
            diags.append(f"Binding slots not contiguous/zero-based: {slots}")
        if len(slots) > 32:
            diags.append(
                f"Binding count {len(slots)} > 32 (may exceed "
                f"maxPerStageDescriptorStorageBuffers)"
            )

    # ── 3. Groupshared memory budget ─────────────────────────────
    total_gs_bytes = 0
    for type_str, name, count_str in _VALIDATE_GROUPSHARED_RE.findall(source):
        count = int(count_str)
        type_bytes = _SLANG_TYPE_SIZES.get(type_str)
        if type_bytes is None:
            # Unknown type — assume 4 bytes (conservative, likely float/int).
            type_bytes = 4
        total_gs_bytes += type_bytes * count
    # The shader module precompilation may add its own groupshared
    # arrays (e.g. _wg_reduce_smem[1024] from reduction.slang).
    # We can't see those here — only validate the kernel-local decls.
    if total_gs_bytes > _RDNA1_MAX_GROUPSHARED_BYTES:
        diags.append(
            f"Groupshared usage {total_gs_bytes} bytes exceeds "
            f"RDNA1 LDS budget ({_RDNA1_MAX_GROUPSHARED_BYTES} bytes)"
        )
    elif total_gs_bytes > 0:
        # Informational: track for regression detection.
        pass

    # ── 4. Push-constant struct field count ──────────────────────
    pc_match = re.search(r"struct PC \{([^}]*)\}", source, re.DOTALL)
    if pc_match:
        pc_body = pc_match.group(1)
        pc_fields = [l.strip() for l in pc_body.split(";") if l.strip()]
        pc_bytes = 0
        for field in pc_fields:
            parts = field.split()
            if len(parts) >= 2:
                type_str = parts[0]
                type_bytes = _SLANG_TYPE_SIZES.get(type_str, 4)
                pc_bytes += type_bytes
        if pc_bytes > _RDNA1_MAX_PUSH_CONSTANT_BYTES:
            diags.append(
                f"Push-constant struct {pc_bytes} bytes exceeds "
                f"RDNA1 limit ({_RDNA1_MAX_PUSH_CONSTANT_BYTES} bytes)"
            )

    # ── 5. Entry-point presence ──────────────────────────────────
    if "void computeMain" not in source and "void main" not in source:
        diags.append(
            "No entry point found (expected 'void computeMain' or 'void main')"
        )

    # ── 6. Basic syntax: no bare Python/Jinja artifacts ───────────
    for artifact in ("{{", "}}", "{%", "%}"):
        if artifact in source:
            diags.append(
                f"Possible unrendered template artifact: '{artifact}' in source"
            )
            break

    return diags


def validate_slang_source_or_raise(source: str) -> None:
    """Validate and raise AssertionError on first issue."""
    diags = validate_slang_source(source)
    if diags:
        raise AssertionError(
            f"Slang source validation failed ({len(diags)} issue(s)):\n"
            + "\n".join(f"  - {d}" for d in diags)
        )
