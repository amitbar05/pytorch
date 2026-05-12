"""Dynamic-shape support foundation — P1.1 (dynamic shape codegen).

Provides helpers for detecting dynamic (symbolic) expressions, emitting
bounds guards, dynamic loops, and runtime workgroup-count computation.

Gated behind ``TORCH_VULKAN_DYNAMIC_SHAPES=1`` (default 0).  When
disabled, all numels are assumed static and integer casts are used.

Usage entry points:
  ``is_dynamic()``        — detect sympy Symbol (vs Integer) expressions
  ``dynamic_wg_counts()`` — compute (wg_x, wg_y) from a dynamic numel expr
  ``emit_guard()``        — emit a bounds-check guard for out-of-range threads
  ``emit_dynamic_loop()`` — emit a dynamic for-loop header
  ``get_static_numel()``  — extract a static int or return None / fallback
"""

from __future__ import annotations

import sympy


class DynamicShapeNotImplemented(Exception):
    """Raised when a dynamic shape path is hit that hasn't been implemented yet."""

    pass


def raise_dynamic_not_implemented(reason: str) -> None:
    raise DynamicShapeNotImplemented(reason)


def is_dynamic(expr: sympy.Expr) -> bool:
    """Check if an expression involves runtime-determined (dynamic) sizes."""
    return not isinstance(expr, (sympy.Integer, int))


def is_dynamic_iteration_ranges(ranges: list) -> bool:
    """Check if any iteration range is dynamic."""
    return any(is_dynamic(e.length) for e in ranges)


def dynamic_numel_name(prefix: str) -> str:
    """Return the push-constant name for a dynamic numel."""
    return f"{prefix}numel"


def emit_guard(var_name: str, total_expr: str) -> str:
    """Emit a bounds guard: if var >= total, skip this thread."""
    return f"if ({var_name} >= ({total_expr})) return;"


def emit_dynamic_loop(prefix: str, numel_expr: str, stride: int) -> str:
    """Emit a dynamic loop header."""
    return (
        f"for (uint {prefix}_cnt = 0; "
        f"{prefix}_cnt < (({numel_expr}) + {stride - 1}) / {stride}; "
        f"++{prefix}_cnt)"
    )


def get_static_numel(expr: sympy.Expr, fallback: int | None = None) -> int | None:
    """Get a static integer from an expression, or None if dynamic."""
    if isinstance(expr, (sympy.Integer, int)):
        return int(expr)
    return fallback


# ── D.2: Dynamic workgroup count helpers ────────────────────────────────────

MAX_COMPUTE_WG_X = 65535


def dynamic_wg_counts(
    total_numel_expr: str,
    threadgroup_size: int,
) -> tuple[str, str]:
    """Return ``(wg_x, wg_y)`` Python expressions for a dynamic total numel.

    Splits work across the primary axis X (clamped to 65535, Vulkan's
    ``maxComputeWorkGroupCount[0]``) and overflows into Y.
    """
    t = threadgroup_size
    total_wgs = f"(({total_numel_expr}) + {t - 1}) // {t}"
    wg_x = f"min({total_wgs}, {MAX_COMPUTE_WG_X})"
    wg_y = f"(({total_wgs}) + {MAX_COMPUTE_WG_X - 1}) // {MAX_COMPUTE_WG_X}"
    return wg_x, wg_y


def is_all_static(trees) -> bool:
    """Check if every range tree has a static (integer) numel."""
    return all(isinstance(t.numel, (sympy.Integer, int)) for t in trees)


def static_total_numel(trees) -> int:
    """Compute the static total numel (product of all tree numels).

    Raises :class:`ValueError` if any tree is dynamic.
    """
    n = 1
    for t in trees:
        n *= int(t.numel)
    return n
