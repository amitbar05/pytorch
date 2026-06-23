"""Shared types for the Slang source validator (split out of ``validate.py``).

``SlangValidationIssue`` lives here so sibling check modules
(``validate_resource_limits.py``, ``validate_identifiers.py``) can import
it without forming an import cycle with ``validate.py``. The dataclass is
re-exported from ``validate.py`` so existing callers see no API change.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class SlangValidationIssue:
    """A single validation issue found in Slang source."""

    category: (
        str  # "brace", "binding", "identifier", "groupshared", "numthreads", "syntax"
    )
    message: str
    line: int | None = None
    context: str | None = None  # relevant source snippet

    def __str__(self) -> str:
        parts = [f"[{self.category}] {self.message}"]
        if self.line is not None:
            parts.insert(1, f"line {self.line}")
        if self.context:
            parts.append(f"  context: {self.context}")
        return " ".join(parts)

    def __contains__(self, substring: object) -> bool:
        # Lets legacy callers use ``"unclosed" in issue`` and ``"gap" in
        # issue.lower()`` against an issue object the same way they used to
        # against a plain error string. The shim path (slang_validator.py)
        # imports SlangValidationIssue and re-exports validate_slang_source
        # without any string conversion, so older tests keep working.
        return isinstance(substring, str) and substring in str(self)

    def lower(self) -> str:
        # Older tests do ``e.lower()`` on each error.  Mirror str semantics.
        return str(self).lower()
