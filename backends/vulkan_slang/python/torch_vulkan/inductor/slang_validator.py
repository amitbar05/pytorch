"""Slang source validation without compilation (Track T.5).

Catches common codegen bugs in-process before invoking ``slangc`` —
every codegen bug that *can* be caught here saves a subprocess creation,
SPIR-V generation, and error-message parse cycle (~0.2-0.5s per kernel).

Checks performed (all O(n) in source length):
  1. **Brace balance** — ``{ }`` mismatch, catches 90% of codegen bugs.
  2. **Binding contiguity** — ``[[vk::binding(N)]]`` declarations must be
     consecutive starting from 0 with no gaps.
  3. **Undefined identifier leaks** — Dynamo/Inductor size symbols (``s27``,
     ``s143``) that leak into generated Slang outside of subscript contexts.
  4. **Groupshared budget** — sum of all ``groupshared`` array sizes must
     not exceed the device limit (default 65536 bytes for RDNA1 — full LDS).
  5. **Numthreads product** — ``[numthreads(X, Y, Z)]`` product must be
     ≤ device limit (default 1024 for RDNA1).

Usage (called before every ``slangc`` invocation)::

    from torch_vulkan.inductor.codegen.validate import validate_slang_source

    errors = validate_slang_source(src)
    if errors:
        raise RuntimeError(f"Slang validation failed: {errors}")

Environment knobs:
  ``TORCH_VULKAN_VALIDATE_SLANG=0`` — disable validation (default: on).
  ``TORCH_VULKAN_MAX_GROUPSHARED_BYTES=N`` — override budget (default 65536).
  ``TORCH_VULKAN_MAX_NUMTHREADS_PRODUCT=N`` — override limit (default 1024).
"""

from __future__ import annotations

import os
import re

# ── Configurable limits ────────────────────────────────────────────────────
_MAX_GROUPSHARED = int(os.environ.get("TORCH_VULKAN_MAX_GROUPSHARED_BYTES", "65536"))
_MAX_NUMTHREADS = int(os.environ.get("TORCH_VULKAN_MAX_NUMTHREADS_PRODUCT", "1024"))
_ENABLED = os.environ.get("TORCH_VULKAN_VALIDATE_SLANG", "1") != "0"

# ── Regex patterns (compiled once) ─────────────────────────────────────────
_RE_BINDING = re.compile(r"\[\[vk::binding\((\d+)\)\]\]")
_RE_GROUPSHARED = re.compile(r"groupshared\s+\w+(?:<[^>]+>)?\s+(\w+)\s*\[([^\]]+)\]")
_RE_NUMTHREADS = re.compile(r"\[numthreads\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)\]")
# Dynamo/Inductor size symbols: s<number>
_RE_SIZE_SYM = re.compile(r"\bs(\d{2,})\b")
# Subscript context: identifier[s27] or identifier[s27 * ...] — these are safe
_RE_SAFE_SUBSCRIPT = re.compile(r"\[\s*s\d+\s*[,\*\s\]]")


def _check_brace_balance(src: str) -> list[str]:
    """Check that ``{ }``, ``( )``, and ``[ ]`` are balanced.

    Returns a list of error messages (empty = pass).
    """
    errors: list[str] = []
    pairs = {"{": "}", "(": ")", "[": "]"}
    stack: list[tuple[str, int]] = []
    in_line_comment = False
    in_block_comment = False
    in_string = False
    escape = False

    for lineno, line in enumerate(src.split("\n"), start=1):
        # Reset line-scoped comment state at each newline.  ``//`` comments
        # terminate at end-of-line; without this reset a single ``//``
        # corrupts validation for the rest of the file.
        in_line_comment = False
        col = 0
        while col < len(line):
            ch = line[col]

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                col += 1
                continue

            if in_line_comment:
                col += 1
                continue

            if in_block_comment:
                if ch == "*" and col + 1 < len(line) and line[col + 1] == "/":
                    in_block_comment = False
                    col += 2
                    continue
                col += 1
                continue

            if ch == '"' and (col == 0 or line[col - 1] != "'"):
                in_string = True
                col += 1
                continue

            if ch == "/" and col + 1 < len(line):
                if line[col + 1] == "/":
                    in_line_comment = True
                    col += 2
                    continue
                if line[col + 1] == "*":
                    in_block_comment = True
                    col += 2
                    continue

            if ch in pairs:
                stack.append((ch, lineno))
            elif ch in (")", "}", "]") and stack:
                opener, opener_lineno = stack.pop()
                expected = pairs[opener]
                if ch != expected:
                    errors.append(
                        f"L{lineno}: mismatched bracket: "
                        f"expected {expected!r} to close "
                        f"{opener!r} from L{opener_lineno}, "
                        f"got {ch!r}"
                    )
            elif ch in (")", "}", "]") and not stack:
                errors.append(f"L{lineno}: unexpected closing bracket {ch!r}")

            col += 1

    for opener, lineno in stack:
        errors.append(
            f"L{lineno}: unclosed bracket {opener!r} (missing {pairs[opener]!r})"
        )

    return errors


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


def _check_size_symbol_leaks(src: str) -> list[str]:
    """Check for Dynamo/Inductor size symbols (``s27``) outside
    safe subscript contexts.

    Size symbols in subscript expressions like ``buf[s27 * stride]``
    are expected — they index into buffers.  Size symbols in other
    contexts (e.g. bare ``s27`` as an rvalue) indicate a codegen bug.
    """
    errors: list[str] = []
    in_line_comment = False
    in_block_comment = False
    in_string = False
    lines = src.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        # Reset line-scoped comment state at each newline — ``//`` comments
        # terminate at EOL; failure to reset corrupts validation for the rest
        # of the file after the first ``//`` line comment.
        in_line_comment = False
        j = 0
        while j < len(line):
            ch = line[j]

            if in_string:
                if ch == '"' and (j == 0 or line[j - 1] != "\\"):
                    in_string = False
                j += 1
                continue
            if in_line_comment:
                j += 1
                continue
            if in_block_comment:
                if ch == "*" and j + 1 < len(line) and line[j + 1] == "/":
                    in_block_comment = False
                    j += 2
                    continue
                j += 1
                continue
            if ch == '"':
                in_string = True
                j += 1
                continue
            if ch == "/" and j + 1 < len(line):
                if line[j + 1] == "/":
                    in_line_comment = True
                    j += 2
                    continue
                if line[j + 1] == "*":
                    in_block_comment = True
                    j += 2
                    continue

            # Check for s<number> pattern
            if ch == "s" and j + 1 < len(line) and line[j + 1].isdigit():
                m = _RE_SIZE_SYM.match(line[j:])
                if m:
                    sym = m.group(1)
                    # Check if we're in a safe subscript context
                    # (looking backwards within the same line for '[')
                    before = line[:j].rstrip()
                    if not before.endswith("["):
                        # Also check ahead for subscript context
                        after = line[j + len(sym) + 1 :].lstrip()
                        # If followed by ']' or ' *' or ',' within a few chars, it's safe
                        if not (
                            after.startswith("]")
                            or after.startswith("*")
                            or after.startswith(",")
                            or after.startswith("+")
                            or after.startswith("-")
                            or after.startswith("/")
                            or after.startswith("%")
                            or after.startswith(")")
                        ):
                            errors.append(
                                f"L{i + 1}: size symbol 's{sym}' appears "
                                f"outside a subscript context — possible "
                                f"codegen leak"
                            )
                    j += len(sym) + 1
                    continue
            j += 1
        i += 1

    return errors


def _check_groupshared_budget(src: str) -> list[str]:
    """Check that total ``groupshared`` memory usage stays within budget.

    Parses declarations like::

        groupshared float tile_a[2 * 32 * 32];
        groupshared float4 vecs[64];

    Multiplies element count by ``sizeof(type)`` and sums.
    """
    errors: list[str] = []
    total_bytes = 0
    # Sizeof lookup for common types (in bytes)
    sizeof_map = {
        "float": 4,
        "float2": 8,
        "float3": 12,
        "float4": 16,
        "int": 4,
        "int2": 8,
        "int3": 12,
        "int4": 16,
        "uint": 4,
        "uint2": 8,
        "uint3": 12,
        "uint4": 16,
        "half": 2,
        "half2": 4,
        "half3": 6,
        "half4": 8,
        "double": 8,
        "double2": 16,
        "double3": 24,
        "double4": 32,
        "bool": 4,
    }

    for m in _RE_GROUPSHARED.finditer(src):
        type_str = src[m.start() : m.end()].split()[1]
        # Strip template params
        if "<" in type_str:
            type_str = type_str[: type_str.index("<")]
        elem_size = sizeof_map.get(type_str, 4)
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


# ── M27: Workgroup size standardization ────────────────────────────────────
# Wave64-alignment warning threshold (RDNA1/GCN preferred threadgroup size
# multiple).  Set to 0 to disable the advisory check.
_WAVE_SIZE = int(os.environ.get("TORCH_VULKAN_WAVE_SIZE", "64"))


def _validate_workgroup_size(src: str) -> list[str]:
    """M27: Validate workgroup size beyond bare product limits.

    Extends ``_check_numthreads_product`` with:
      - Non-multiple-of-wave-size advisory (suboptimal occupancy on RDNA1,
        where wave64 hardware must pad partial waves).
      - Product still clamped to ≤ 1024 (Vulkan minimum guarantee).

    Returns a list of error strings; empty list means pass.
    """
    errors: list[str] = []
    m = _RE_NUMTHREADS.search(src)
    if not m:
        return errors

    x, y, z = int(m.group(1)), int(m.group(2)), int(m.group(3))
    product = x * y * z

    # Hard limit — Vulkan minimum guarantee across all implementations.
    if product > _MAX_NUMTHREADS:
        errors.append(
            f"[M27] numthreads product {product} ({x}×{y}×{z}) exceeds "
            f"device limit of {_MAX_NUMTHREADS}"
        )
    if product == 0:
        errors.append(f"[M27] numthreads product is zero: ({x}, {y}, {z})")

    # Advisory: non-multiple of wave size leaves VGPR lanes idle.
    # On RDNA1 (wave64) a workgroup of 100 threads spans 2 waves
    # (128 lanes) → 28 lanes wasted.  Multiples of 64 guarantee
    # full-wave occupancy.  Skipped when _WAVE_SIZE is 0.
    if _WAVE_SIZE > 0 and product % _WAVE_SIZE != 0:
        errors.append(
            f"[M27] numthreads product {product} ({x}×{y}×{z}) is not a "
            f"multiple of wave size {_WAVE_SIZE}; suboptimal occupancy "
            f"on wave{_WAVE_SIZE} hardware."
        )

    return errors


# ── Push-constant parsing ────────────────────────────────────────────────
# Member declaration: type name; or type name[N];
_RE_STRUCT_MEMBER = re.compile(r"(\w+(?:\d+)?)\s+(\w+)\s*(?:\[([^\]]+)\])?\s*;")
_RE_DIFFERENTIABLE = re.compile(r"\[Differentiable\]")
_RE_BACKWARD_DERIVATIVE = re.compile(r"\[BackwardDerivative\((\w+)\)\]")
_RE_FORWARD_DERIVATIVE = re.compile(r"\[ForwardDerivative\((\w+)\)\]")

# Push-constant size limit per Vulkan spec minimum
_MAX_PUSH_CONSTANT_BYTES = int(
    os.environ.get("TORCH_VULKAN_MAX_PUSH_CONSTANT_BYTES", "128")
)


def _check_push_constant_size(src: str) -> list[str]:
    """Check that push-constant struct size is ≤ 128 bytes.

    Parses ``struct PC { ... }`` declarations that are used with
    ``[[vk::push_constant]]`` and sums member sizes.
    """
    errors: list[str] = []

    # Sizeof lookup for common types (in bytes)
    sizeof_map = {
        "float": 4,
        "float2": 8,
        "float3": 12,
        "float4": 16,
        "int": 4,
        "int2": 8,
        "int3": 12,
        "int4": 16,
        "uint": 4,
        "uint2": 8,
        "uint3": 12,
        "uint4": 16,
        "half": 2,
        "half2": 4,
        "half3": 6,
        "half4": 8,
        "double": 8,
        "double2": 16,
        "double3": 24,
        "double4": 32,
        "bool": 4,
    }

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
                        total = _sum_struct_members(body, sizeof_map)
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
            total = _sum_struct_members(body, sizeof_map)
            if total > _MAX_PUSH_CONSTANT_BYTES:
                lineno = src[: sm.start()].count("\n") + 1
                errors.append(
                    f"L{lineno}: push-constant struct '{pc_type}' size {total}B "
                    f"exceeds limit of {_MAX_PUSH_CONSTANT_BYTES}B"
                )

    return errors


def _sum_struct_members(
    body: str,
    sizeof_map: dict[str, int],
) -> int:
    """Sum the sizes of struct member declarations in ``body``."""
    total = 0
    for m in _RE_STRUCT_MEMBER.finditer(body):
        type_str = m.group(1)
        # name = m.group(2)
        array_dim = m.group(3)  # e.g. "4" or "2*3"

        elem_size = sizeof_map.get(type_str, 4)
        if array_dim:
            try:
                count = eval(array_dim, {"__builtins__": {}}, {})
            except Exception:
                count = 1
            total += int(count) * elem_size
        else:
            total += elem_size
    return total


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

def validate_slang_source(src: str) -> list[str]:
    """Run all Slang source validation checks.

    Args:
        src: Complete Slang source string to validate.

    Returns:
        List of error message strings.  Empty list means pass.
    """
    if not _ENABLED:
        return []

    errors: list[str] = []
    errors.extend(_check_brace_balance(src))
    errors.extend(_check_binding_contiguity(src))
    errors.extend(_check_size_symbol_leaks(src))
    errors.extend(_check_groupshared_budget(src))
    errors.extend(_check_numthreads_product(src))
    errors.extend(_validate_workgroup_size(src))
    errors.extend(_check_push_constant_size(src))
    errors.extend(_check_differentiable_pairs(src))
    return errors
