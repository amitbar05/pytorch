"""CG.M8 — Inline bwd_diff emission for Inductor's codegen pipeline.

Provides methods that emit ``bwd_diff(fwd_fn)`` directly into the
generated Slang kernel body, eliminating the Python custom-op shim.

The emission is split into two parts:
1. **Body lines**: ``DifferentialPair`` declaration + ``bwd_diff(fwd_fn)`` call
   (emitted as raw lines into ``kernel.compute``).
2. **Result expression**: ``dp.getDifferential()``, which is passed to
   ``kernel.cse.generate()`` so CSE creates a proper variable for the store.

This separation ensures the CSE layer is aware of the output variable
and can properly cache/reuse it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch_vulkan.inductor.bwd_diff_table import BwdDiffEntry


def emit_inline_unary_bwd(
    entry: BwdDiffEntry,
    *,
    x_var: str,
    grad_out_var: str,
    dtype: str = "float",
) -> tuple[str, str]:
    """Emit inline Slang code for a unary bwd_diff.

    Produces body lines::

        DifferentialPair<float> dp_N = diffPair(x_var, 0.0f);
        bwd_diff(silu_fwd)(dp_N, grad_out_var);

    Returns (body_lines, result_expr) where result_expr is something
    like ``"_dp_bwd_0.getDifferential()"`` that the caller passes to
    ``kernel.cse.generate()``.
    """
    import itertools

    if not hasattr(emit_inline_unary_bwd, "_counter"):
        emit_inline_unary_bwd._counter = itertools.count()  # type: ignore[attr-defined]
    dp_name = f"_dp_bwd_{next(emit_inline_unary_bwd._counter)}"  # type: ignore[attr-defined]

    body_lines = (
        f"DifferentialPair<{dtype}> {dp_name} = diffPair({x_var}, ({dtype})0);\n"
        f"bwd_diff({entry.fwd_fn})({dp_name}, {grad_out_var});"
    )
    result_expr = f"{dp_name}.getDifferential()"
    return body_lines, result_expr


def emit_inline_binary_bwd(
    entry: BwdDiffEntry,
    *,
    a_var: str,
    b_var: str,
    grad_out_var: str,
    dtype: str = "float",
) -> tuple[str, str, str]:
    """Emit inline Slang code for a binary bwd_diff.

    Returns (body_lines, result_a_expr, result_b_expr).
    """
    import itertools

    if not hasattr(emit_inline_binary_bwd, "_counter"):
        emit_inline_binary_bwd._counter = itertools.count()  # type: ignore[attr-defined]
    cnt = next(emit_inline_binary_bwd._counter)  # type: ignore[attr-defined]
    dpa_name = f"_dpa_bwd_{cnt}"
    dpb_name = f"_dpb_bwd_{cnt}"

    no_diff_args = ", ".join(entry.no_diff_params) if entry.no_diff_params else ""
    if no_diff_args:
        no_diff_args += ", "

    body_lines = (
        f"DifferentialPair<{dtype}> {dpa_name} = diffPair({a_var}, ({dtype})0);\n"
        f"DifferentialPair<{dtype}> {dpb_name} = diffPair({b_var}, ({dtype})0);\n"
        f"bwd_diff({entry.fwd_fn})({dpa_name}, {dpb_name}, {no_diff_args}{grad_out_var});"
    )
    result_a_expr = f"{dpa_name}.getDifferential()"
    result_b_expr = f"{dpb_name}.getDifferential()"
    return body_lines, result_a_expr, result_b_expr


def bwd_diff_module_import(entry: BwdDiffEntry) -> str:
    """Return the ``import <module>;`` line needed for this bwd_diff entry."""
    return f"import {entry.module};"


def can_inline_bwd_diff(aten_op: str) -> bool:
    """Check if an aten backward op can be emitted inline via bwd_diff.

    Returns False for ops that are not in BWD_DIFF_TABLE, are excluded,
    or need groupshared/barriers.
    """
    from torch_vulkan.inductor.bwd_diff_table import (
        BWD_DIFF_TABLE,
        EXCLUDED_DIFFERENTIABLE_FWDS,
        is_bwd_diff_eligible,
    )

    if not is_bwd_diff_eligible(aten_op):
        return False

    entry = BWD_DIFF_TABLE[aten_op]
    if entry.fwd_fn in EXCLUDED_DIFFERENTIABLE_FWDS:
        return False

    return True
