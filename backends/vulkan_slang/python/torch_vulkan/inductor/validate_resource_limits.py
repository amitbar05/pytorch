"""Resource-limit checks extracted from ``validate.py`` (anti-goal #7 split).

Houses the two device-limit checks (``groupshared`` budget and
``numthreads`` product) and the small ``_type_size_bytes`` helper they share.
Imported and called by ``validate.SlangValidator``; the public entry point
remains ``validate.validate_slang_source``.

Pure code move: no semantic changes.
"""

from __future__ import annotations

import re

from torch_vulkan.inductor.validate_types import SlangValidationIssue


# ── Regex patterns (local to the resource-limit checks) ───────────────────

_GROUPSHARED_RE = re.compile(
    r"groupshared\s+(\w+(?:\s*<[^>]+>)?)\s+(\w+)\s*\[([^\]]+)\]\s*;"
)
_NUMTHREADS_RE = re.compile(r"\[numthreads\((\d+),\s*(\d+),\s*(\d+)\)\]")


def _type_size_bytes(type_str: str) -> int:
    """Return size in bytes for a Slang type string."""
    type_str = type_str.strip()
    # Exact matches
    exact = {
        "float": 4,
        "int": 4,
        "uint": 4,
        "float32_t": 4,
        "int32_t": 4,
        "uint32_t": 4,
        "half": 2,
        "float16_t": 2,
        "int16_t": 2,
        "uint16_t": 2,
        "double": 8,
        "float64_t": 8,
        "int64_t": 8,
        "uint64_t": 8,
        "int8_t": 1,
        "uint8_t": 1,
        "bool": 1,
        "bfloat16": 2,
    }
    if type_str in exact:
        return exact[type_str]
    # Vector types: float2, float3, float4, int2, etc.
    vec_match = re.match(r"^(float|int|uint|half|double|bool)(\d)$", type_str)
    if vec_match:
        base = vec_match.group(1)
        count = int(vec_match.group(2))
        base_size = exact.get(base, 4)
        return base_size * count
    # Matrix types: float2x2, float3x3, float4x4, etc.
    mat_match = re.match(r"^(float|half|double)(\d)x(\d)$", type_str)
    if mat_match:
        base = mat_match.group(1)
        rows = int(mat_match.group(2))
        cols = int(mat_match.group(3))
        base_size = exact.get(base, 4)
        return base_size * rows * cols
    # Default: assume 4 bytes (float)
    return 4


def check_groupshared_budget(
    src: str, max_groupshared_bytes: int
) -> list[SlangValidationIssue]:
    """Sum groupshared array sizes and compare against device limit."""
    issues: list[SlangValidationIssue] = []
    total_bytes = 0
    for m in _GROUPSHARED_RE.finditer(src):
        type_str = m.group(1).strip()
        # name = m.group(2)
        size_expr = m.group(3)
        # Determine element size in bytes
        elem_bytes = _type_size_bytes(type_str)
        # Try to evaluate the size expression (simple integer check)
        try:
            num_elems = int(size_expr)
        except ValueError:
            # Could be an expression with template variables like {{ BQ }} * {{ head_dim }}
            # Try evaluating the Jinja-rendered form
            try:
                num_elems = eval(size_expr, {"__builtins__": {}}, {})
            except Exception:
                # Can't evaluate — just note it for manual review
                continue
        total_bytes += elem_bytes * num_elems

    if total_bytes > max_groupshared_bytes:
        issues.append(
            SlangValidationIssue(
                category="groupshared",
                message=(
                    f"groupshared budget exceeded: {total_bytes} bytes "
                    f"(limit: {max_groupshared_bytes} bytes)"
                ),
            )
        )
    return issues


def check_numthreads_product(
    src: str, max_invocations: int
) -> list[SlangValidationIssue]:
    """Check that numthreads.x * y * z ≤ device max invocations."""
    issues: list[SlangValidationIssue] = []
    for m in _NUMTHREADS_RE.finditer(src):
        x, y, z = int(m.group(1)), int(m.group(2)), int(m.group(3))
        product = x * y * z
        if product > max_invocations:
            issues.append(
                SlangValidationIssue(
                    category="numthreads",
                    message=(
                        f"numthreads product ({x}×{y}×{z} = {product}) "
                        f"exceeds max invocations ({max_invocations})"
                    ),
                )
            )
        if (
            x > max_invocations
            or y > max_invocations
            or z > max_invocations
        ):
            issues.append(
                SlangValidationIssue(
                    category="numthreads",
                    message=(
                        f"numthreads dimension ({x},{y},{z}) exceeds "
                        f"max invocations per dimension ({max_invocations})"
                    ),
                )
            )
    return issues
