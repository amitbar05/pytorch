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
"""

from __future__ import annotations

import dataclasses
import re

# ── Device limits (conservative) ──────────────────────────────────────────

# Vulkan spec minimum for maxComputeSharedMemorySize is 32768 bytes.
# In practice most devices support 49152 (48KB) or 65536 (64KB). RDNA1
# is 64KB, Adreno 7xx is 32KB. Use the conservative floor.
_DEFAULT_MAX_GROUPSHARED_BYTES = 32768

# Vulkan spec minimum for maxComputeWorkGroupInvocations is 1024.
_DEFAULT_MAX_INVOCATIONS = 1024


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


class SlangValidationError(Exception):
    """Raised when validation finds issues, with a summary message."""

    def __init__(self, issues: list[SlangValidationIssue]) -> None:
        self.issues = issues
        summary = f"Slang validation found {len(issues)} issue(s):\n"
        summary += "\n".join(f"  {i}" for i in issues)
        super().__init__(summary)


# ── Regex patterns ────────────────────────────────────────────────────────

_BINDING_RE = re.compile(r"\[\[vk::binding\((\d+)(?:\s*,\s*\d+)?\)\]\]")
_GROUPSHARED_RE = re.compile(
    r"groupshared\s+(\w+(?:\s*<[^>]+>)?)\s+(\w+)\s*\[([^\]]+)\]\s*;"
)
_NUMTHREADS_RE = re.compile(r"\[numthreads\((\d+),\s*(\d+),\s*(\d+)\)\]")
_STRING_LITERAL_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
_BLOCK_COMMENT_START_RE = re.compile(r"/\*")
_BLOCK_COMMENT_END_RE = re.compile(r"\*/")
_LINE_COMMENT_RE = re.compile(r"//.*$", re.MULTILINE)
_IDENTIFIER_RE = re.compile(r"\b([a-zA-Z_]\w*)\b")


# Slang keywords and built-in identifiers that are always "defined"
_SLANG_RESERVED = frozenset(
    {
        # Slang keywords
        "if",
        "else",
        "for",
        "while",
        "do",
        "switch",
        "case",
        "default",
        "break",
        "continue",
        "return",
        "struct",
        "class",
        "interface",
        "void",
        "bool",
        "int",
        "uint",
        "float",
        "double",
        "half",
        "int8_t",
        "int16_t",
        "int64_t",
        "uint8_t",
        "uint16_t",
        "uint64_t",
        "float16_t",
        "float32_t",
        "float64_t",
        "bfloat16",
        "true",
        "false",
        "null",
        "this",
        "const",
        "static",
        "inline",
        "public",
        "private",
        "protected",
        "virtual",
        "override",
        "abstract",
        "typedef",
        "using",
        "namespace",
        "import",
        "export",
        "module",
        "enum",
        "union",
        "operator",
        "explicit",
        "noexcept",
        "__import",
        "__generic",
        "__generic_type",
        "__extension",
        "extension",
        "as",
        "is",
        "sizeof",
        "in",
        "out",
        "inout",
        "ref",
        "property",
        "get",
        "set",
        "where",
        "associatedtype",
        "capability",
        # Slang attributes
        "vk",
        "binding",
        "push_constant",
        "location",
        "descriptorset",
        "shader",
        "compute",
        "ForceInline",
        "unroll",
        "StructuredBuffer",
        "RWStructuredBuffer",
        "ByteAddressBuffer",
        "RWByteAddressBuffer",
        "Texture2D",
        "RWTexture2D",
        "ConstantBuffer",
        "groupshared",
        # SPIR-V / HLSL built-ins
        "numthreads",
        "SV_DispatchThreadID",
        "SV_GroupThreadID",
        "SV_GroupID",
        "SV_GroupIndex",
        # Slang built-in functions (subset)
        "abs",
        "max",
        "min",
        "clamp",
        "sqrt",
        "rsqrt",
        "exp",
        "log",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "sinh",
        "cosh",
        "tanh",
        "floor",
        "ceil",
        "round",
        "trunc",
        "frac",
        "sign",
        "copysign",
        "fmod",
        "lerp",
        "step",
        "smoothstep",
        "pow",
        "mod",
        "fma",
        "fwidth",
        "ddx",
        "ddy",
        "dot",
        "cross",
        "normalize",
        "length",
        "distance",
        "reflect",
        "refract",
        "transpose",
        "determinant",
        "asfloat",
        "asint",
        "asuint",
        "asdouble",
        "countbits",
        "firstbithigh",
        "firstbitlow",
        "reversebits",
        "GroupMemoryBarrierWithGroupSync",
        "GroupMemoryBarrier",
        "InterlockedAdd",
        "InterlockedExchange",
        "InterlockedCompareExchange",
        "InterlockedMin",
        "InterlockedMax",
        "InterlockedAnd",
        "InterlockedOr",
        "InterlockedXor",
        "WaveGetLaneIndex",
        "WaveReadLaneFirst",
        "WaveReadLaneAt",
        "WaveActiveSum",
        "WaveActiveProduct",
        "WaveActiveMax",
        "WaveActiveMin",
        "WaveActiveAllEqual",
        "WaveActiveBitAnd",
        "WaveActiveBitOr",
        "WaveActiveBitXor",
        "WavePrefixSum",
        "WavePrefixProduct",
        "WaveIsFirstLane",
        "WaveBroadcastLaneAt",
        "tex2D",
        "tex2Dlod",
        "sampler",
        "float2",
        "float3",
        "float4",
        "float2x2",
        "float3x3",
        "float4x4",
        "int2",
        "int3",
        "int4",
        "uint2",
        "uint3",
        "uint4",
        "bool2",
        "bool3",
        "bool4",
        "double2",
        "double3",
        "double4",
        "matrix",
        # GPU coordinate builtins (SV_ prefix is in reserved set above)
        "gtid",
        "lid",
        "gid",
        "tid",
    }
)


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
        issues: list[SlangValidationIssue] = []
        total_bytes = 0
        for m in _GROUPSHARED_RE.finditer(src):
            type_str = m.group(1).strip()
            # name = m.group(2)
            size_expr = m.group(3)
            # Determine element size in bytes
            elem_bytes = _type_size_bytes(type_str)
            # Try to evaluate the size expression (simple integer check)
            try:
                num_elems = int(size_expr)
            except ValueError:
                # Could be an expression with template variables like {{ BQ }} * {{ head_dim }}
                # Try evaluating the Jinja-rendered form
                try:
                    num_elems = eval(size_expr, {"__builtins__": {}}, {})
                except Exception:
                    # Can't evaluate — just note it for manual review
                    continue
            total_bytes += elem_bytes * num_elems

        if total_bytes > self._max_groupshared_bytes:
            issues.append(
                SlangValidationIssue(
                    category="groupshared",
                    message=(
                        f"groupshared budget exceeded: {total_bytes} bytes "
                        f"(limit: {self._max_groupshared_bytes} bytes)"
                    ),
                )
            )
        return issues

    def _check_numthreads_product(self, src: str) -> list[SlangValidationIssue]:
        """Check that numthreads.x * y * z ≤ device max invocations."""
        issues: list[SlangValidationIssue] = []
        for m in _NUMTHREADS_RE.finditer(src):
            x, y, z = int(m.group(1)), int(m.group(2)), int(m.group(3))
            product = x * y * z
            if product > self._max_invocations:
                issues.append(
                    SlangValidationIssue(
                        category="numthreads",
                        message=(
                            f"numthreads product ({x}×{y}×{z} = {product}) "
                            f"exceeds max invocations ({self._max_invocations})"
                        ),
                    )
                )
            if (
                x > self._max_invocations
                or y > self._max_invocations
                or z > self._max_invocations
            ):
                issues.append(
                    SlangValidationIssue(
                        category="numthreads",
                        message=(
                            f"numthreads dimension ({x},{y},{z}) exceeds "
                            f"max invocations per dimension ({self._max_invocations})"
                        ),
                    )
                )
        return issues

    @staticmethod
    def _check_undefined_identifiers(src: str) -> list[SlangValidationIssue]:
        """Detect identifiers that appear exactly once — likely typos.

        An identifier used **exactly once** in the whole shader source is
        suspicious: either it's a function/variable that's never used (dead
        code), or a typo (e.g., ``s27`` misspelled as ``s28`` in one place).
        We flag these but filter out common single-use patterns like:

        - Entry-point name (``computeMain``)
        - Struct/parameter names in declarations
        - Slang reserved words
        - Common singleton patterns (``PC``, ``pc``)
        """
        issues: list[SlangValidationIssue] = []
        # Strip strings and comments
        cleaned = _strip_strings_and_comments(src)
        # Find all identifiers
        identifiers: dict[str, list[tuple[int, int]]] = {}
        for m in _IDENTIFIER_RE.finditer(cleaned):
            ident = m.group(1)
            if ident in _SLANG_RESERVED:
                continue
            if ident.startswith("_"):
                # Allow single-use underscore-prefixed identifiers
                # (they're often intentionally generated)
                continue
            if len(ident) <= 1:
                continue
            identifiers.setdefault(ident, []).append((m.start(), m.end()))

        # Find identifiers used exactly once
        single_use: dict[str, tuple[int, int]] = {
            k: v[0] for k, v in identifiers.items() if len(v) == 1
        }

        # Filter known-good single-use patterns
        _known_single_use = {
            "computeMain",  # entry point
            "PC",
            "pc",  # push-constant struct and var (may appear once per shader)
            "main",
        }
        suspicious = {
            k: pos for k, pos in single_use.items() if k not in _known_single_use
        }

        # Don't flag identifiers that look like type names (capitalized,
        # used before struct/class declarations)
        suspicious_filtered = {}
        for k, pos in suspicious.items():
            if k[0].isupper() and len(k) > 1:
                # Capitalized identifiers may be type names used once
                # Check context: if it appears near "struct" or "class", skip
                ctx_start = max(0, pos[0] - 20)
                ctx_end = min(len(cleaned), pos[1] + 20)
                ctx = cleaned[ctx_start:ctx_end].lower()
                if "struct" in ctx or "class" in ctx:
                    continue
            suspicious_filtered[k] = pos

        if suspicious_filtered:
            # Get line numbers for the suspicious identifiers
            line_map = _build_line_map(src)
            details = []
            for ident, (start_pos, _) in sorted(
                suspicious_filtered.items(), key=lambda x: x[1][0]
            ):
                line_no = _pos_to_line(line_map, start_pos)
                details.append(f"`{ident}` at line {line_no}")
            issues.append(
                SlangValidationIssue(
                    category="identifier",
                    message=(
                        f"Single-use identifiers (possible typos): "
                        f"{', '.join(details[:8])}"  # cap at 8 to avoid flooding
                    ),
                )
            )
        return issues


# ── Helper functions ──────────────────────────────────────────────────────


def _type_size_bytes(type_str: str) -> int:
    """Return size in bytes for a Slang type string."""
    type_str = type_str.strip()
    # Exact matches
    exact = {
        "float": 4,
        "int": 4,
        "uint": 4,
        "float32_t": 4,
        "int32_t": 4,
        "uint32_t": 4,
        "half": 2,
        "float16_t": 2,
        "int16_t": 2,
        "uint16_t": 2,
        "double": 8,
        "float64_t": 8,
        "int64_t": 8,
        "uint64_t": 8,
        "int8_t": 1,
        "uint8_t": 1,
        "bool": 1,
        "bfloat16": 2,
    }
    if type_str in exact:
        return exact[type_str]
    # Vector types: float2, float3, float4, int2, etc.
    vec_match = re.match(r"^(float|int|uint|half|double|bool)(\d)$", type_str)
    if vec_match:
        base = vec_match.group(1)
        count = int(vec_match.group(2))
        base_size = exact.get(base, 4)
        return base_size * count
    # Matrix types: float2x2, float3x3, float4x4, etc.
    mat_match = re.match(r"^(float|half|double)(\d)x(\d)$", type_str)
    if mat_match:
        base = mat_match.group(1)
        rows = int(mat_match.group(2))
        cols = int(mat_match.group(3))
        base_size = exact.get(base, 4)
        return base_size * rows * cols
    # Default: assume 4 bytes (float)
    return 4


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


def _build_line_map(src: str) -> list[int]:
    """Build a list where index i is the start position of line i (0-based)."""
    line_map = [0]
    for i, ch in enumerate(src):
        if ch == "\n":
            line_map.append(i + 1)
    return line_map


def _pos_to_line(line_map: list[int], pos: int) -> int:
    """Convert a character position to a 1-based line number."""
    import bisect

    return bisect.bisect_right(line_map, pos)


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
