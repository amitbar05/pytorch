"""Groupshared-budget and numthreads-product validation (passes 4–5).

Checks that total ``groupshared`` memory stays within the device budget
and that ``[numthreads(X, Y, Z)]`` product is within device limits.
"""

from __future__ import annotations

from ._config import (
    _MAX_GROUPSHARED,
    _MAX_NUMTHREADS,
    _RE_GROUPSHARED,
    _RE_NUMTHREADS,
    SIZEOF_MAP,
)


def _check_groupshared_budget(src: str) -> list[str]:
    """Check that total ``groupshared`` memory usage stays within budget.

    Parses declarations like::

        groupshared float tile_a[2 * 32 * 32];
        groupshared float4 vecs[64];

    Multiplies element count by ``sizeof(type)`` and sums.
    """
    errors: list[str] = []
    total_bytes = 0

    for m in _RE_GROUPSHARED.finditer(src):
        type_str = src[m.start() : m.end()].split()[1]
        # Strip template params
        if "<" in type_str:
            type_str = type_str[: type_str.index("<")]
        elem_size = SIZEOF_MAP.get(type_str, 4)
        count_str = m.group(2)
        try:
            # Evaluate constant expressions like "2 * 32 * 32"
            count = eval(count_str, {"__builtins__": {}}, {})
            total_bytes += int(count) * elem_size
        except Exception:
            # Dynamic size — assume worst case of 4096 elements
            total_bytes += 4096 * elem_size

    if total_bytes > _MAX_GROUPSHARED:
        errors.append(
            f"groupshared budget exceeded: {total_bytes} bytes > "
            f"{_MAX_GROUPSHARED} bytes limit"
        )

    return errors


def _check_numthreads_product(src: str) -> list[str]:
    """Check that ``[numthreads(X, Y, Z)]`` product is within device limits."""
    errors: list[str] = []
    m = _RE_NUMTHREADS.search(src)
    if m:
        x, y, z = int(m.group(1)), int(m.group(2)), int(m.group(3))
        product = x * y * z
        if product > _MAX_NUMTHREADS:
            errors.append(
                f"numthreads product {product} ({x}×{y}×{z}) exceeds "
                f"device limit of {_MAX_NUMTHREADS}"
            )
        if product == 0:
            errors.append(f"numthreads product is zero: ({x}, {y}, {z})")
    return errors
