"""Push-constant struct size validation (pass 7).

Parses ``[[vk::push_constant]]`` usage and checks that the referenced
struct fits within the Vulkan minimum guarantee (128 bytes).
"""

from __future__ import annotations

import re

from ._config import _MAX_PUSH_CONSTANT_BYTES, _RE_STRUCT_MEMBER, SIZEOF_MAP


def _check_push_constant_size(src: str) -> list[str]:
    """Check that push-constant struct size is ≤ 128 bytes.

    Parses ``struct PC { ... }`` declarations that are used with
    ``[[vk::push_constant]]`` and sums member sizes.
    """
    errors: list[str] = []

    # Find push-constant usage: [[vk::push_constant]] Type name;
    pc_usages: list[tuple[str, int]] = []  # (struct_type, line_no)
    for m in re.finditer(r"\[\[vk::push_constant\]\]", src):
        # Find the line containing this match
        line_start = src.rfind("\n", 0, m.start()) + 1
        line_end = src.find("\n", m.end())
        if line_end == -1:
            line_end = len(src)
        line = src[line_start:line_end]
        # Parse: [[vk::push_constant]] Type name;
        rest = src[m.end() : line_end].strip()
        parts = rest.split()
        if len(parts) >= 1:
            pc_type = parts[0]
            if pc_type == "struct":
                # Inline struct: [[vk::push_constant]] struct { ... } name;
                if len(parts) >= 2 and parts[1].startswith("{"):
                    # Parse inline struct body
                    brace_start = rest.find("{")
                    brace_end = rest.find("}", brace_start)
                    if brace_end != -1:
                        body = rest[brace_start + 1 : brace_end]
                        total = _sum_struct_members(body)
                        if total > _MAX_PUSH_CONSTANT_BYTES:
                            lineno = src[: m.start()].count("\n") + 1
                            errors.append(
                                f"L{lineno}: push-constant struct size {total}B "
                                f"exceeds limit of {_MAX_PUSH_CONSTANT_BYTES}B"
                            )
                else:
                    pc_usages.append((parts[0], m.start()))
            else:
                pc_usages.append((pc_type, m.start()))

    # Find struct declarations matching the push-constant types
    for pc_type, pc_pos in pc_usages:
        # Search for struct PC { ... };
        struct_pattern = re.compile(
            r"struct\s+" + re.escape(pc_type) + r"\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}\s*;"
        )
        sm = struct_pattern.search(src)
        if sm:
            body = sm.group(1)
            total = _sum_struct_members(body)
            if total > _MAX_PUSH_CONSTANT_BYTES:
                lineno = src[: sm.start()].count("\n") + 1
                errors.append(
                    f"L{lineno}: push-constant struct '{pc_type}' size {total}B "
                    f"exceeds limit of {_MAX_PUSH_CONSTANT_BYTES}B"
                )

    return errors


def _sum_struct_members(body: str) -> int:
    """Sum the sizes of struct member declarations in ``body``."""
    total = 0
    for m in _RE_STRUCT_MEMBER.finditer(body):
        type_str = m.group(1)
        array_dim = m.group(3)  # e.g. "4" or "2*3"

        elem_size = SIZEOF_MAP.get(type_str, 4)
        if array_dim:
            try:
                count = eval(array_dim, {"__builtins__": {}}, {})
            except Exception:
                count = 1
            total += int(count) * elem_size
        else:
            total += elem_size
    return total
