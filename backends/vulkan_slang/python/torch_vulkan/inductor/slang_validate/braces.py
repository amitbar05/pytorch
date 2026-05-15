"""Brace-balance validation for Slang source (pass 1).

Checks that ``{ }``, ``( )``, and ``[ ]`` are balanced, with awareness of
line comments, block comments, and string literals.
"""

from __future__ import annotations


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
