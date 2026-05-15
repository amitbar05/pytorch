"""Backward-derivative annotation scanning (pass 8).

Checks that every ``[Differentiable]``-annotated function has a paired
``[BackwardDerivative]`` or ``[ForwardDerivative]`` declaration.
"""

from __future__ import annotations

import re

from ._config import (
    _RE_BACKWARD_DERIVATIVE,
    _RE_DIFFERENTIABLE,
    _RE_FORWARD_DERIVATIVE,
)


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
