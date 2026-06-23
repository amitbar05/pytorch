"""Size-symbol leak detection for Slang source (pass 3).

Detects Dynamo/Inductor size symbols (``s27``, ``s143``) that leak into
generated Slang outside of subscript contexts.
"""

from __future__ import annotations

from ._config import _RE_SIZE_SYM


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
