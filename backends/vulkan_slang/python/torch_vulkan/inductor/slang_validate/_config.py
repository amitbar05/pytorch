"""Shared constants, regex patterns, and type-size lookup for slang_validate."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


# M15.1.i: ValidationIssue dataclass carries a category tag so callers
# can programmatically distinguish issue types without parsing message
# strings.  The ``message`` field is a human-readable error description.
@dataclass(frozen=True)
class ValidationIssue:
    category: str
    message: str

    def __str__(self) -> str:
        return f"[{self.category}] {self.message}"

    def __contains__(self, substring: object) -> bool:
        # Lets older tests use ``"unclosed" in issue`` (and ``"gap" in
        # issue.lower()``) against an issue object the same way they
        # used to against a plain error string.  Without this, the
        # dataclass falls back to default __contains__ which raises
        # ``TypeError: argument of type 'ValidationIssue' is not iterable``.
        return isinstance(substring, str) and substring in str(self)

    def lower(self) -> str:
        # Older tests do ``e.lower()`` on each error.  Mirror str semantics.
        return str(self).lower()


# ── Configurable limits ────────────────────────────────────────────────────
_MAX_GROUPSHARED = int(os.environ.get("TORCH_VULKAN_MAX_GROUPSHARED_BYTES", "65536"))
_MAX_NUMTHREADS = int(os.environ.get("TORCH_VULKAN_MAX_NUMTHREADS_PRODUCT", "1024"))
_ENABLED = os.environ.get("TORCH_VULKAN_VALIDATE_SLANG", "1") != "0"
_WAVE_SIZE = int(os.environ.get("TORCH_VULKAN_WAVE_SIZE", "64"))
_MAX_PUSH_CONSTANT_BYTES = int(
    os.environ.get("TORCH_VULKAN_MAX_PUSH_CONSTANT_BYTES", "128")
)

# ── Regex patterns (compiled once) ─────────────────────────────────────────
_RE_BINDING = re.compile(r"\[\[vk::binding\((\d+)\)\]\]")
_RE_GROUPSHARED = re.compile(r"groupshared\s+\w+(?:<[^>]+>)?\s+(\w+)\s*\[([^\]]+)\]")
_RE_NUMTHREADS = re.compile(r"\[numthreads\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)\]")
# Dynamo/Inductor size symbols: s<number>
_RE_SIZE_SYM = re.compile(r"\bs(\d{2,})\b")
# Subscript context: identifier[s27] or identifier[s27 * ...] — these are safe
_RE_SAFE_SUBSCRIPT = re.compile(r"\[\s*s\d+\s*[,\*\s\]]")
# Member declaration: type name; or type name[N];
_RE_STRUCT_MEMBER = re.compile(r"(\w+(?:\d+)?)\s+(\w+)\s*(?:\[([^\]]+)\])?\s*;")
_RE_DIFFERENTIABLE = re.compile(r"\[Differentiable\]")
_RE_BACKWARD_DERIVATIVE = re.compile(r"\[BackwardDerivative\((\w+)\)\]")
_RE_FORWARD_DERIVATIVE = re.compile(r"\[ForwardDerivative\((\w+)\)\]")

# ── Type-size lookup (shared across groupshared-budget + push-constant) ──
SIZEOF_MAP: dict[str, int] = {
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
