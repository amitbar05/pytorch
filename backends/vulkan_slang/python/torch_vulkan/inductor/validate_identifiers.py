"""Undefined-identifier check extracted from ``validate.py`` (anti-goal #7 split).

Houses the biggest check function — ``_check_undefined_identifiers`` —
together with the Slang reserved-words set, the identifier regex, and the
line-mapping helpers it relies on. Imported and called by
``validate.SlangValidator``; the public entry point remains
``validate.validate_slang_source``.

Pure code move: no semantic changes.
"""

from __future__ import annotations

import re

from torch_vulkan.inductor.validate_types import SlangValidationIssue


# ── Regex patterns ────────────────────────────────────────────────────────

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


def check_undefined_identifiers(src: str) -> list[SlangValidationIssue]:
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
