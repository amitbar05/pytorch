"""Lightweight in-process Slang source validator (Track T.5).

Catches common codegen bugs WITHOUT running slangc — no GPU, no SPIR-V
compilation, no subprocess. Designed to run in CI as part of the regression
test suite.

Checks performed:
  1. Brace balance (``{`` vs ``}``)
  2. Binding contiguity (``[[vk::binding(N)]]`` numbers are 0,1,2,…)
  3. Undefined identifier detection (single-use identifiers — likely typos)
  4. ``groupshared`` budget check (sum of groupshared arrays ≤ device limit)
  5. ``numthreads`` product check (numthreads.x * y * z ≤ device max invocations)
  6. Simple syntax checks (unclosed string literals, unclosed block comments)

The heavier checks live in sibling modules to keep this file under the
800-line anti-goal cap:

* ``validate_resource_limits`` — groupshared / numthreads.
* ``validate_identifiers`` — single-use identifier detection.
"""

from __future__ import annotations

import re

from torch_vulkan.inductor.validate_identifiers import (
    _SLANG_RESERVED,
    check_undefined_identifiers,
)
from torch_vulkan.inductor.validate_resource_limits import (
    check_groupshared_budget,
    check_numthreads_product,
)
from torch_vulkan.inductor.validate_types import SlangValidationIssue


__all__ = [
    "SlangValidationError",
    "SlangValidationIssue",
    "SlangValidator",
    "_SLANG_RESERVED",
    "inject_and_validate",
    "validate_slang_source",
]


# ── Device limits (conservative) ──────────────────────────────────────────

# Vulkan spec minimum for maxComputeSharedMemorySize is 32768 bytes.
# In practice most devices support 49152 (48KB) or 65536 (64KB). RDNA1
# is 64KB, Adreno 7xx is 32KB. Use the conservative floor.
_DEFAULT_MAX_GROUPSHARED_BYTES = 32768

# Vulkan spec minimum for maxComputeWorkGroupInvocations is 1024.
_DEFAULT_MAX_INVOCATIONS = 1024


class SlangValidationError(Exception):
    """Raised when validation finds issues, with a summary message."""

    def __init__(self, issues: list[SlangValidationIssue]) -> None:
        self.issues = issues
        summary = f"Slang validation found {len(issues)} issue(s):\n"
        summary += "\n".join(f"  {i}" for i in issues)
        super().__init__(summary)


# ── Regex patterns ────────────────────────────────────────────────────────

_BINDING_RE = re.compile(r"\[\[vk::binding\((\d+)(?:\s*,\s*\d+)?\)\]\]")
_STRING_LITERAL_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
_BLOCK_COMMENT_START_RE = re.compile(r"/\*")
_BLOCK_COMMENT_END_RE = re.compile(r"\*/")
_LINE_COMMENT_RE = re.compile(r"//.*$", re.MULTILINE)


class SlangValidator:
    """Stateful validator for Slang compute shader source.

    Usage::

        v = SlangValidator()
        issues = v.validate(src)
        if issues:
            raise SlangValidationError(issues)
    """

    def __init__(
        self,
        *,
        max_groupshared_bytes: int = _DEFAULT_MAX_GROUPSHARED_BYTES,
        max_invocations: int = _DEFAULT_MAX_INVOCATIONS,
    ) -> None:
        self._max_groupshared_bytes = max_groupshared_bytes
        self._max_invocations = max_invocations

    def validate(self, src: str) -> list[SlangValidationIssue]:
        """Run all checks and return a list of issues found."""
        issues: list[SlangValidationIssue] = []
        issues.extend(self._check_brace_balance(src))
        issues.extend(self._check_binding_contiguity(src))
        issues.extend(self._check_string_literals(src))
        issues.extend(self._check_block_comments(src))
        issues.extend(self._check_numthreads_product(src))
        issues.extend(self._check_groupshared_budget(src))
        issues.extend(self._check_undefined_identifiers(src))
        return issues

    def validate_or_raise(self, src: str) -> None:
        """Run all checks and raise `SlangValidationError` if issues found."""
        issues = self.validate(src)
        if issues:
            raise SlangValidationError(issues)

    # ── Individual checks ─────────────────────────────────────────────

    @staticmethod
    def _check_brace_balance(src: str) -> list[SlangValidationIssue]:
        """Check that ``{`` and ``}`` counts match."""
        issues: list[SlangValidationIssue] = []
        # Strip strings and comments first to avoid false positives
        cleaned = _strip_strings_and_comments(src)
        opens = cleaned.count("{")
        closes = cleaned.count("}")
        if opens != closes:
            lines = src.split("\n")
            # Find the line with the likely mismatch
            balance = 0
            suspect_line = None
            for i, line in enumerate(lines, start=1):
                stripped_line = _strip_strings_and_comments(line)
                balance += stripped_line.count("{") - stripped_line.count("}")
                if balance < 0:
                    suspect_line = i
                    break
            issues.append(
                SlangValidationIssue(
                    category="brace",
                    message=f"Brace mismatch: {opens} open vs {closes} close braces",
                    line=suspect_line,
                    context=(
                        lines[suspect_line - 1].strip()[:120]
                        if suspect_line and suspect_line <= len(lines)
                        else None
                    ),
                )
            )
        return issues

    @staticmethod
    def _check_binding_contiguity(src: str) -> list[SlangValidationIssue]:
        """Check that ``[[vk::binding(N)]]`` numbers are 0,1,2,… without gaps."""
        issues: list[SlangValidationIssue] = []
        bindings = [int(m.group(1)) for m in _BINDING_RE.finditer(src)]
        if not bindings:
            return issues
        # Check for duplicates
        seen: set[int] = set()
        dupes = [b for b in bindings if b in seen or seen.add(b)]  # type: ignore[func-returns-value]
        if dupes:
            issues.append(
                SlangValidationIssue(
                    category="binding",
                    message=f"Duplicate binding(s): {sorted(set(dupes))}",
                )
            )
        # Check contiguity starting from 0
        sorted_bindings = sorted(set(bindings))
        expected = list(range(len(sorted_bindings)))
        if sorted_bindings != expected:
            gaps = sorted(set(expected) - set(sorted_bindings))
            extra = sorted(set(sorted_bindings) - set(expected))
            detail = []
            if gaps:
                detail.append(f"missing bindings: {gaps}")
            if extra:
                detail.append(f"unexpected high bindings: {extra}")
            issues.append(
                SlangValidationIssue(
                    category="binding",
                    message=f"Non-contiguous bindings — expected 0..{len(bindings) - 1} "
                    f"but got {sorted_bindings}. {'; '.join(detail)}",
                )
            )
        return issues

    @staticmethod
    def _check_string_literals(src: str) -> list[SlangValidationIssue]:
        """Check for unclosed string literals."""
        issues: list[SlangValidationIssue] = []
        # Remove block comments, then line comments
        cleaned = _BLOCK_COMMENT_START_RE.sub(" ", src)
        cleaned = _BLOCK_COMMENT_END_RE.sub(" ", cleaned)
        cleaned = _LINE_COMMENT_RE.sub(" ", cleaned)

        in_string = False
        i = 0
        while i < len(cleaned):
            ch = cleaned[i]
            if ch == '"' and not in_string:
                in_string = True
            elif ch == '"' and in_string:
                # Check it's not escaped
                if i > 0 and cleaned[i - 1] == "\\":
                    # Count consecutive backslashes before this quote
                    bs = 0
                    j = i - 1
                    while j >= 0 and cleaned[j] == "\\":
                        bs += 1
                        j -= 1
                    if bs % 2 == 1:
                        # Odd count → this quote is escaped
                        pass
                    else:
                        in_string = False
                else:
                    in_string = False
            i += 1

        if in_string:
            issues.append(
                SlangValidationIssue(
                    category="syntax",
                    message="Unclosed string literal detected",
                )
            )
        return issues

    @staticmethod
    def _check_block_comments(src: str) -> list[SlangValidationIssue]:
        """Check for unclosed block comments (``/* ... */``)."""
        issues: list[SlangValidationIssue] = []
        # Remove string literals first to avoid false positives
        cleaned = _STRING_LITERAL_RE.sub('""', src)
        starts = list(_BLOCK_COMMENT_START_RE.finditer(cleaned))
        ends = list(_BLOCK_COMMENT_END_RE.finditer(cleaned))

        if len(starts) != len(ends):
            # Find the line of the unclosed comment
            lines = src.split("\n")
            balance = 0
            suspect_line = None
            for i, line in enumerate(lines, start=1):
                stripped = _STRING_LITERAL_RE.sub('""', line)
                balance += len(_BLOCK_COMMENT_START_RE.findall(stripped))
                balance -= len(_BLOCK_COMMENT_END_RE.findall(stripped))
                if balance < 0:
                    suspect_line = i
                    break
            if balance > 0:
                # Unclosed comment — find the last `/*`
                if starts:
                    last_start = starts[-1].start()
                    # Count lines up to the last start
                    suspect_line = src[:last_start].count("\n") + 1

            issues.append(
                SlangValidationIssue(
                    category="syntax",
                    message=(
                        f"Unclosed block comment: {len(starts)} `/*` vs "
                        f"{len(ends)} `*/`"
                    ),
                    line=suspect_line,
                    context=(
                        lines[suspect_line - 1].strip()[:120]
                        if suspect_line and suspect_line <= len(lines)
                        else None
                    ),
                )
            )
        return issues

    def _check_groupshared_budget(self, src: str) -> list[SlangValidationIssue]:
        """Sum groupshared array sizes and compare against device limit."""
        return check_groupshared_budget(src, self._max_groupshared_bytes)

    def _check_numthreads_product(self, src: str) -> list[SlangValidationIssue]:
        """Check that numthreads.x * y * z ≤ device max invocations."""
        return check_numthreads_product(src, self._max_invocations)

    @staticmethod
    def _check_undefined_identifiers(src: str) -> list[SlangValidationIssue]:
        """Detect identifiers that appear exactly once — likely typos."""
        return check_undefined_identifiers(src)


# ── Helper functions ──────────────────────────────────────────────────────


def _strip_strings_and_comments(src: str) -> str:
    """Remove string literals and comments from source to avoid false matches."""
    # Remove string literals
    cleaned = _STRING_LITERAL_RE.sub('""', src)
    # Remove block comments
    cleaned = _BLOCK_COMMENT_START_RE.sub("  ", cleaned)
    cleaned = _BLOCK_COMMENT_END_RE.sub("  ", cleaned)
    # Remove line comments
    cleaned = _LINE_COMMENT_RE.sub(" ", cleaned)
    return cleaned


# ── Public API ────────────────────────────────────────────────────────────

# Default validator instance
_default_validator = SlangValidator()


def validate_slang_source(
    src: str,
    *,
    max_groupshared_bytes: int | None = None,
    max_invocations: int | None = None,
) -> list[SlangValidationIssue]:
    """Validate a Slang source string and return issues found.

    Args:
        src: The Slang compute shader source code.
        max_groupshared_bytes: Override the groupshared budget limit.
        max_invocations: Override the max invocations limit.

    Returns:
        List of ``SlangValidationIssue`` objects. Empty list = clean.
    """
    if max_groupshared_bytes is not None or max_invocations is not None:
        validator = SlangValidator(
            max_groupshared_bytes=max_groupshared_bytes
            or _DEFAULT_MAX_GROUPSHARED_BYTES,
            max_invocations=max_invocations or _DEFAULT_MAX_INVOCATIONS,
        )
        return validator.validate(src)
    return _default_validator.validate(src)


# ── Test helper: inject known errors and verify they're caught ────────────

# Injectable error snippets keyed by error type.
_INJECTABLE_ERRORS: dict[str, str] = {
    "brace": "void bad() { if (true) { return; }",
    "binding": "[[vk::binding(0)]] StructuredBuffer<float> a;\n"
    "[[vk::binding(2)]] StructuredBuffer<float> b;\n"
    "[[vk::binding(5)]] StructuredBuffer<float> c;",
    "string": 'float3x3 a = "unclosed;',
    "comment": "/* unclosed",
    "numthreads": "[numthreads(2048, 2, 1)]",
    "groupshared": "groupshared float big[16384];\ngroupshared float also_big[16384];",
    "identifier": "float xyzzy_missing_var = s27;",  # s27 is a typo
}


def inject_and_validate() -> tuple[dict[str, bool], str]:
    """Inject each known error type into a minimal shader skeleton and verify
    the validator catches it.

    Returns:
        Tuple of ``(results_dict, report_str)`` where ``results_dict`` maps
        error type → caught (bool), and ``report_str`` is a human-readable
        summary.
    """
    skeleton = """\
[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 gtid : SV_DispatchThreadID,
                 uint3 lid : SV_GroupThreadID,
                 uint3 gid : SV_GroupID) {
    float x = (float)gtid.x;
    // placeholder for injected error
}
"""
    results: dict[str, bool] = {}
    lines: list[str] = []

    # Map sub-error-types to actual validator categories
    _SUBCATEGORY_TO_CATEGORY = {
        "string": "syntax",
        "comment": "syntax",
    }
    for error_type, snippet in sorted(_INJECTABLE_ERRORS.items()):
        test_src = skeleton.replace("// placeholder for injected error", snippet)
        issues = validate_slang_source(test_src)
        expected_cat = _SUBCATEGORY_TO_CATEGORY.get(error_type, error_type)
        caught = any(issue.category == expected_cat for issue in issues)
        results[error_type] = caught
        status = "✓ CAUGHT" if caught else "✗ MISSED"
        lines.append(f"  [{status}] {error_type}: {len(issues)} issue(s)")
        if not caught:
            for i in issues:
                lines.append(f"          unexpected: [{i.category}] {i.message[:100]}")

    all_caught = all(results.values())
    summary = (
        f"Injected error detection: {'ALL CAUGHT' if all_caught else 'SOME MISSED'}\n"
    )
    summary += "\n".join(lines)

    return results, summary
