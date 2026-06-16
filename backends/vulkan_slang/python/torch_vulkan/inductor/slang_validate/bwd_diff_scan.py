"""Backward-derivative annotation scanning (pass 8).

Checks that every ``[Differentiable]``-annotated function has a paired
``[BackwardDerivative]`` or ``[ForwardDerivative]`` declaration, **and**
that the parameter count and types of the backward function match the forward.
"""

from __future__ import annotations

import re

from ._config import (
    _RE_BACKWARD_DERIVATIVE,
    _RE_DIFFERENTIABLE,
    _RE_FORWARD_DERIVATIVE,
)


# ── Regex for parsing function signatures ───────────────────────────────────
# Matches "ret_type func_name(params)" where ret_type is a simple word.
_RE_FUNC_SIG = re.compile(r"(\w+)\s+(\w+)\s*\(([^)]*)\)")

# Parameter regexes — forward params may be "no_diff type name".
_RE_FWD_PARAM = re.compile(r"^\s*(no_diff\s+)?(\S+)\s+(\w+)\s*$")
# Backward params may be "inout no_diff type name".
_RE_BWD_PARAM = re.compile(r"^\s*(inout\s+)?(no_diff\s+)?(\S+)\s+(\w+)\s*$")


def _split_param_list(params_str: str) -> list[str]:
    """Split a parameter-list string by commas, respecting angle-bracket
    nesting (e.g. ``DifferentialPair<float>`` must not be split)."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in params_str:
        if ch == "<":
            depth += 1
            current.append(ch)
        elif ch == ">":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def _check_differentiable_pairs(src: str) -> list[str]:
    """Check that every ``[Differentiable]`` function has a paired
    ``[BackwardDerivative]`` or ``[ForwardDerivative]`` declaration.

    This is a WARNING-level check — missing backward derivatives are
    not fatal but indicate incomplete autodiff coverage.
    """
    errors: list[str] = []

    # Extract all [Differentiable] function names
    differentiable_funcs: list[tuple[str, int]] = []  # (name, lineno)
    # Find [Differentiable] and then the function declaration that follows
    for m in _RE_DIFFERENTIABLE.finditer(src):
        # Find the function name after this attribute
        remainder = src[m.end() :]
        # Skip whitespace and other attributes
        func_match = re.search(r"[\w<>]+\s+(\w+)\s*\(", remainder)
        if func_match:
            name = func_match.group(1)
            lineno = src[: m.start()].count("\n") + 1
            differentiable_funcs.append((name, lineno))

    # Extract all [BackwardDerivative(name)] and [ForwardDerivative(name)] targets
    backward_targets: set[str] = set()
    for m in _RE_BACKWARD_DERIVATIVE.finditer(src):
        backward_targets.add(m.group(1))
    for m in _RE_FORWARD_DERIVATIVE.finditer(src):
        backward_targets.add(m.group(1))

    # Also find which function names have a [BackwardDerivative] or
    # [ForwardDerivative] attribute directly (the function following
    # the attribute).
    has_derivative_attr: set[str] = set()
    _RE_DERIV_ATTR = re.compile(r"\[(?:Backward|Forward)Derivative\(\w+\)\]")
    for m in _RE_DERIV_ATTR.finditer(src):
        remainder = src[m.end() :]
        func_match = re.search(r"[\w<>]+\s+(\w+)\s*\(", remainder)
        if func_match:
            has_derivative_attr.add(func_match.group(1))

    # Check each differentiable function has a paired derivative
    for name, lineno in differentiable_funcs:
        if name not in backward_targets and name not in has_derivative_attr:
            errors.append(
                f"L{lineno}: [Differentiable] function '{name}' has no "
                f"paired [BackwardDerivative] or [ForwardDerivative] — "
                f"autodiff coverage may be incomplete"
            )

    return errors


def validate_bwd_diff_signatures(src: str) -> list[str]:
    """Validate that backward-derivative functions have matching parameter
    count and types with their forward counterparts.

    For each ``[Differentiable]`` / ``[BackwardDerivative(fn_bwd)]`` pair:

    * Forward param of type ``T`` (differentiable) →
      backward param must be ``inout DifferentialPair<T>``.
    * Forward param of ``no_diff T`` →
      backward param must be ``no_diff T`` (same type).
    * Backward must have exactly one extra output-gradient param (scalar,
      no ``inout``, no ``DifferentialPair``).

    Returns a list of error strings (empty = all valid).
    """
    errors: list[str] = []

    for m_diff in _RE_DIFFERENTIABLE.finditer(src):
        # ── Locate the [BackwardDerivative(name)] annotation ────────────
        remainder = src[m_diff.end() :]
        bwd_annotation = _RE_BACKWARD_DERIVATIVE.search(remainder)
        if not bwd_annotation:
            continue  # ForwardDerivative or no annotation → skip

        # ── Locate the forward function signature ───────────────────────
        match_fwd = _RE_FUNC_SIG.search(remainder)
        if not match_fwd:
            continue

        # The [BackwardDerivative] must appear *before* the forward function
        if bwd_annotation.start() > match_fwd.start():
            continue

        fwd_ret = match_fwd.group(1)
        fwd_name = match_fwd.group(2)
        fwd_params_str = match_fwd.group(3)
        bwd_name = bwd_annotation.group(1)
        base_lineno = src[: m_diff.start()].count("\n") + 1

        # ── Parse forward parameters ────────────────────────────────────
        fwd_raw = _split_param_list(fwd_params_str)
        fwd_params: list[dict[str, object]] = []
        for p in fwd_raw:
            pm = _RE_FWD_PARAM.match(p)
            if pm:
                fwd_params.append(
                    {
                        "no_diff": pm.group(1) is not None,
                        "type": pm.group(2).strip(),
                        "name": pm.group(3),
                    }
                )
        if not fwd_params:
            continue

        # ── Locate the backward function by name ────────────────────────
        bwd_func_pattern = re.compile(
            rf"(\w+)\s+{re.escape(bwd_name)}\s*\(([^)]*)\)"
        )
        match_bwd = bwd_func_pattern.search(src)
        if not match_bwd:
            errors.append(
                f"L{base_lineno}: [Differentiable] function '{fwd_name}' "
                f"references [BackwardDerivative({bwd_name})] but backward "
                f"function not found in source"
            )
            continue

        bwd_params_str = match_bwd.group(2)

        # ── Parse backward parameters ───────────────────────────────────
        bwd_raw = _split_param_list(bwd_params_str)
        bwd_params: list[dict[str, object]] = []
        for p in bwd_raw:
            pm = _RE_BWD_PARAM.match(p)
            if pm:
                bwd_params.append(
                    {
                        "inout": pm.group(1) is not None,
                        "no_diff": pm.group(2) is not None,
                        "type": pm.group(3).strip(),
                        "name": pm.group(4),
                    }
                )

        # ── Rule 1: correct number of backward parameters ───────────────
        expected_bwd_count = len(fwd_params) + 1  # +1 for output gradient
        if len(bwd_params) != expected_bwd_count:
            errors.append(
                f"L{base_lineno}: backward '{bwd_name}' has "
                f"{len(bwd_params)} param(s), expected {expected_bwd_count} "
                f"({len(fwd_params)} forward + 1 output gradient)"
            )
            # Keep validating what we can, but skip per-param checks if
            # counts don't match — they'd produce cascading noise.
            if len(bwd_params) != expected_bwd_count:
                continue

        # ── Rule 2: per-parameter matching ──────────────────────────────
        for i, fp in enumerate(fwd_params):
            bp = bwd_params[i]

            if fp["no_diff"]:  # type: ignore[operator]
                if not bp["no_diff"]:  # type: ignore[operator]
                    errors.append(
                        f"L{base_lineno}: param {i + 1} of '{bwd_name}' "
                        f"should be no_diff (forward param "
                        f"'{fp['name']}' is no_diff)"
                    )
                elif bp["type"] != fp["type"]:  # type: ignore[operator]
                    errors.append(
                        f"L{base_lineno}: param {i + 1} of '{bwd_name}' "
                        f"type mismatch: expected no_diff {fp['type']}, "
                        f"got no_diff {bp['type']}"
                    )
            else:
                # Differentiable forward param → inout DifferentialPair<T>
                if not bp["inout"]:  # type: ignore[operator]
                    errors.append(
                        f"L{base_lineno}: param {i + 1} of '{bwd_name}' "
                        f"should be inout (forward param "
                        f"'{fp['name']}' is differentiable)"
                    )
                expected_type = f"DifferentialPair<{fp['type']}>"
                if bp["type"] != expected_type:  # type: ignore[operator]
                    errors.append(
                        f"L{base_lineno}: param {i + 1} of '{bwd_name}' "
                        f"type mismatch: expected inout {expected_type}, "
                        f"got {bp['type']}"
                    )

        # ── Rule 3: output-gradient param (last backward param) ─────────
        og = bwd_params[-1]
        if og["inout"]:  # type: ignore[operator]
            errors.append(
                f"L{base_lineno}: last param of '{bwd_name}' "
                f"(output gradient '{og['name']}') should not be inout"
            )
        if og["no_diff"]:  # type: ignore[operator]
            errors.append(
                f"L{base_lineno}: last param of '{bwd_name}' "
                f"(output gradient '{og['name']}') should not be no_diff"
            )
        if "DifferentialPair" in str(og["type"]):
            errors.append(
                f"L{base_lineno}: last param of '{bwd_name}' "
                f"(output gradient '{og['name']}') should not be "
                f"DifferentialPair, got {og['type']}"
            )

    return errors
