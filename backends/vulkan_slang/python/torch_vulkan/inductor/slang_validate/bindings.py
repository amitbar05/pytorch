"""Binding-contiguity validation for Slang source (pass 2).

Checks that ``[[vk::binding(N)]]`` declarations are contiguous from 0
with no gaps.
"""

from __future__ import annotations

from ._config import _RE_BINDING


def _check_binding_contiguity(src: str) -> list[str]:
    """Check that ``[[vk::binding(N)]]`` values are contiguous from 0.

    Returns a list of error messages (empty = pass).
    """
    errors: list[str] = []
    bindings: list[int] = []
    for m in _RE_BINDING.finditer(src):
        bindings.append(int(m.group(1)))

    if not bindings:
        return errors

    expected = 0
    for b in sorted(bindings):
        if b != expected:
            errors.append(
                f"Binding gap: expected {expected}, got {b}. "
                f"Declared bindings: {sorted(bindings)}"
            )
            break
        expected += 1

    return errors
