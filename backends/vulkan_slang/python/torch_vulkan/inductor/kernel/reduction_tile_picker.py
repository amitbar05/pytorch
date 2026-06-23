"""Reduction tile-picker — typed reduction-kind enum, metadata table,
bank-conflict padding analysis, and wave-compute-dtype helpers.

Extracted from ``reduction.py`` (M15.1.g — Track 1 anti-goal #7 split).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union

import sympy
import torch

# ── M29: Typed reduction-kind enum + metadata table ────────────────
# Replaces 4 parallel string-keyed dicts (_NEUTRAL_ELEMENT, _NEUTRAL_LITERAL,
# _REDUCTION_OP_TEMPLATE, _ACCUMULATE_OP) that all keyed on the same set of
# reduction names ("sum", "prod", "max", "min").  Each kind now has a single
# typed metadata record, eliminating stringly-typed dispatch where a typo
# in one dict would silently produce wrong codegen.
#
# Scope: only the simple wave-reduction kinds that flow through the unified
# wg_reduce_wave<Op> codegen path.  welford/any/argmin/argmax/xor_sum each
# have their own dedicated helper methods and are NOT modeled here.
#
# The Slang side (lib/reduction.slang) still receives the op_template name
# as a string template parameter — the more aggressive Slang-side refactor
# (`enum class ReductionKind`) is deferred until the codegen path actually
# emits enum-typed parameters.


class ReductionKind(Enum):
    """Simple wave-reduction kinds dispatched through ``wg_reduce_wave<Op>``."""

    SUM = "sum"
    PROD = "prod"
    MAX = "max"
    MIN = "min"

    @classmethod
    def from_str(cls, name: str) -> "ReductionKind":
        return cls(name)


@dataclass(frozen=True)
class ReductionMeta:
    """Per-kind metadata for the unified wave-reduction codegen path.

    Attributes:
        op_template: Slang lib struct name passed to ``wg_reduce_wave<Op>``.
        neutral_element: Initial value expression for multistage accumulation
            (used as the ``default_value`` for a per-thread accumulator).
        neutral_literal: Slang-literal form used to neutralize out-of-range
            lanes when ``red_numel < max_threadgroup_size``.
        accumulate_op: Local accumulation operator/template.  For ``+`` and
            ``*`` reductions this is a compound-assignment operator (e.g.
            ``"+="``).  For ``max``/``min`` the codegen branches and emits a
            ``= max({local}, {value})`` form instead — the string stored
            here matches that template for symmetry.
    """

    op_template: str
    neutral_element: str
    neutral_literal: str
    accumulate_op: str


REDUCTION_TABLE: dict[ReductionKind, ReductionMeta] = {
    ReductionKind.SUM: ReductionMeta(
        op_template="OpSum",
        neutral_element="0",
        neutral_literal="0.0f",
        accumulate_op="+=",
    ),
    ReductionKind.PROD: ReductionMeta(
        op_template="OpProd",
        neutral_element="1",
        neutral_literal="1.0f",
        accumulate_op="*=",
    ),
    ReductionKind.MAX: ReductionMeta(
        op_template="OpMaxReduce",
        neutral_element="(-3.4e38f)",
        neutral_literal="(-3.4e38f)",
        accumulate_op="= max({local}, {value})",
    ),
    ReductionKind.MIN: ReductionMeta(
        op_template="OpMinReduce",
        neutral_element="(3.4e38f)",
        neutral_literal="(3.4e38f)",
        accumulate_op="= min({local}, {value})",
    ),
}


def _reduction_meta(kind: Union[str, ReductionKind]) -> ReductionMeta:
    """Look up reduction metadata, accepting either a string or an enum."""
    if isinstance(kind, str):
        kind = ReductionKind.from_str(kind)
    return REDUCTION_TABLE[kind]


# ── DEPRECATED string-keyed compat shims ──────────────────────────
# Existing string-keyed callers (none remain in this file as of M29 — all
# internal lookups now go through ``_reduction_meta``) can still index these
# dicts by reduction-name string.  Prefer ``REDUCTION_TABLE[ReductionKind.X]``
# for new code.

# DEPRECATED — use REDUCTION_TABLE[ReductionKind.X].neutral_element
_NEUTRAL_ELEMENT = {k.value: m.neutral_element for k, m in REDUCTION_TABLE.items()}

# DEPRECATED — use REDUCTION_TABLE[ReductionKind.X].neutral_literal
_NEUTRAL_LITERAL = {k.value: m.neutral_literal for k, m in REDUCTION_TABLE.items()}

# DEPRECATED — use REDUCTION_TABLE[ReductionKind.X].op_template
# P3.6: uses reduction.wg_reduce_wave generic instead of per-op inline helpers.
_REDUCTION_OP_TEMPLATE = {k.value: m.op_template for k, m in REDUCTION_TABLE.items()}

# DEPRECATED — use REDUCTION_TABLE[ReductionKind.X].accumulate_op
_ACCUMULATE_OP = {k.value: m.accumulate_op for k, m in REDUCTION_TABLE.items()}


# ── M20: Bank-conflict padding for groupshared arrays ──────────────
# RDNA1 LDS has 32 banks.  Without padding, column-major access patterns
# (reduction tree stride, tile loads) land multiple threads on the same
# bank serializing access.  Adding +32 elements shifts the bank mapping
# for elements past offset 32, eliminating bank conflicts.
#
# Gate: TORCH_VULKAN_BANK_CONFLICT_PAD=1 enables +32 padding (default: 1
# on GPU, 0 on Lavapipe where LDS is smaller and has no bank conflicts).

_BANK_CONFLICT_PAD: int = 32  # DR.6: actual padding constant (num elems)

# M20 — Per-pattern bank conflict analysis.
# RDNA1: 32 banks, 4-byte bank width.
# Bank conflict when (stride * element_size / 4) % 32 == 0.
# Instead of always padding by 32, we analyze the access stride and only
# add padding when actually needed — reducing LDS waste.
_RDNA1_NUM_BANKS: int = 32
_RDNA1_BANK_WIDTH_BYTES: int = 4


def _compute_padding_for_reduction(
    num_threads: int,
    element_size: int,
    num_banks: int = _RDNA1_NUM_BANKS,
) -> int:
    """Compute recommended padding for reduction tree access.

    Reduction pattern: thread tid accesses ``smem[tid]`` and ``smem[tid + s]``
    where ``s`` starts at ``(size+1)/2`` and halves each iteration.

    Bank conflict occurs when ``(s * element_size / 4) % num_banks == 0``,
    i.e. when the stride in banks is a multiple of num_banks.

    For float (4 bytes): conflict when ``s % 32 == 0``.
    For WelfordResult (12 bytes): conflict when ``(s * 3) % 32 == 0``
    (same condition since 3 and 32 are coprime).

    Returns padding amount in elements, 0 if no significant risk.
    """
    elements_per_bank = _RDNA1_BANK_WIDTH_BYTES // element_size
    if elements_per_bank == 0:
        # Element larger than bank width (e.g. 8-byte double)
        elements_per_bank = 1

    # Reduction tree strides: s starts at ~num_threads/2, halves to 1.
    # Conflict when s * (element_size / 4) % num_banks == 0.
    # For float: s % 32 == 0; first conflict-capable s is 32.
    # If num_threads < 64, max stride is < 32 → no conflict possible.
    conflict_stride = num_banks // max(elements_per_bank, 1)
    if elements_per_bank >= num_banks:
        # Element spans ≥1 full bank cycle → every access touches
        # multiple banks; padding won't help linear tree patterns.
        return 0

    # Walk the reduction tree strides to check for conflicts.
    max_stride = (num_threads + 1) // 2
    has_conflict = False
    s = max_stride
    while s > 0:
        if (s * element_size // _RDNA1_BANK_WIDTH_BYTES) % num_banks == 0:
            has_conflict = True
            break
        if s == 1:
            break
        s = (s + 1) // 2

    if not has_conflict:
        return 0

    # When conflicts exist, pad by 1 element to shift the base address
    # alignment of the groupshared allocation. The compiler allocates
    # groupshared arrays sequentially; changing the total allocation
    # size by 1 element (4 bytes) shifts the base address of subsequent
    # arrays by 4 bytes = 1 bank position, breaking the alignment.
    # For the array itself, we recommend a padding that makes the
    # access pattern avoid multiples of 32 in common strides.
    #
    # Conservative: pad by 1 element (shifts base alignment by 4 bytes).
    # This is sufficient to move elements off bank aliasing for simple
    # linear access but costs only 4 bytes instead of 128 (32*4).
    return 1


def _compute_padding_for_welford(
    num_threads: int,
    num_banks: int = _RDNA1_NUM_BANKS,
) -> int:
    """Compute recommended padding for welford reduction.

    WelfordResult is 3 floats = 12 bytes = 3 banks per element.
    Since 3 and 32 are coprime, stride-based bank conflicts only occur
    when ``stride % 32 == 0`` — same condition as float.

    However, the wider element (3 banks) means each access touches 3
    banks concurrently, which naturally reduces intra-access contention.
    Many welford configurations can get away with zero padding.
    """
    return _compute_padding_for_reduction(num_threads, 12, num_banks)


def _compute_padding_for_scan(
    num_threads: int,
    element_size: int,
    num_banks: int = _RDNA1_NUM_BANKS,
) -> int:
    """Compute recommended padding for inclusive scan pattern.

    Scan pattern (tree-based): thread tid accesses ``smem[tid]`` and
    ``smem[tid - s]`` where ``s`` doubles from 1 up to size/2.

    For wave-prefix scan (IScan), the shared memory usage is minimal
    (only 32 elements for cross-wave propagation).

    Returns padding amount, 0 if no significant risk.
    """
    elements_per_bank = _RDNA1_BANK_WIDTH_BYTES // element_size
    if elements_per_bank == 0:
        elements_per_bank = 1

    # Scan strides: s = 1, 2, 4, 8, 16, 32, 64, ...
    # Conflict when s * (element_size / 4) % num_banks == 0.
    # For float: s = 32 is the first conflict stride.
    if num_threads < 64:
        return 0

    conflict_stride = num_banks // max(elements_per_bank, 1)
    if elements_per_bank >= num_banks:
        return 0

    if conflict_stride <= num_threads:
        return 1
    return 0


def _compute_padding_for_sort(
    num_threads: int,
    element_size: int,
    num_banks: int = _RDNA1_NUM_BANKS,
) -> int:
    """Compute recommended padding for bitonic sort pattern.

    Bitonic sort: thread lid accesses ``smem[lid]`` and ``smem[lid XOR j]``
    where ``j`` is a power-of-2 step. The XOR pattern naturally distributes
    accesses across banks for power-of-2 array sizes, so bank conflicts
    are minimal.

    Returns padding amount, usually 0.
    """
    # Power-of-2 sizes: XOR pattern distributes well across banks.
    # For non-power-of-2 sizes, slight padding can help.
    if num_threads > 0 and (num_threads & (num_threads - 1)) == 0:
        return 0
    # Non-power-of-2: may have stride aliasing; pad minimally.
    return 0  # Typically zero — sort has good bank distribution


def _analyze_smem_bank_conflict_risk(
    num_threads: int,
    element_size: int,
    access_pattern: str = "generic",
    num_banks: int = _RDNA1_NUM_BANKS,
) -> int:
    """Return recommended padding in elements to avoid bank conflicts.

    Args:
        num_threads: Number of threads accessing the groupshared array.
        element_size: Bytes per element (4 for float, 12 for WelfordResult).
        access_pattern: One of ``"reduction"``, ``"welford"``, ``"scan"``,
            ``"sort"``, or ``"generic"``.
        num_banks: Number of LDS banks (32 on RDNA1).

    Returns:
        Recommended padding in elements (0 = no padding needed).
        Never exceeds the legacy blanket pad of 32.
    """
    if num_threads <= 0:
        return 0

    # Dispatch to pattern-specific helpers.
    if access_pattern == "reduction":
        pad = _compute_padding_for_reduction(num_threads, element_size, num_banks)
    elif access_pattern == "welford":
        pad = _compute_padding_for_welford(num_threads, num_banks)
    elif access_pattern == "scan":
        pad = _compute_padding_for_scan(num_threads, element_size, num_banks)
    elif access_pattern == "sort":
        pad = _compute_padding_for_sort(num_threads, element_size, num_banks)
    else:
        # Generic/conservative: check if any stride up to num_threads/2
        # could cause a conflict.
        elements_per_bank = _RDNA1_BANK_WIDTH_BYTES // element_size
        if elements_per_bank > 0 and num_threads >= num_banks // elements_per_bank:
            pad = 1
        else:
            pad = 0

    # Never exceed legacy blanket pad of 32.
    return min(pad, _BANK_CONFLICT_PAD)


# ── N.3.a — Wave-native compute dtype (generic wg_reduce_wave<W,T>) ──
# Maps PyTorch source dtype → PyTorch dtype used for the wave reduction path.
# f16 stays native (half); bf16 widens to f32 (no bf16 wave intrinsics).
_WAVE_COMPUTE_DTYPE: dict[torch.dtype, torch.dtype] = {
    torch.float16: torch.float16,  # native half — no widening
    torch.bfloat16: torch.float32,  # widen: bf16 wave intrinsics not universal
    torch.float32: torch.float32,
    torch.float64: torch.float64,
}


def _slang_wave_type(dtype: torch.dtype) -> str:
    """Return the Slang built-in type name for use as a generic parameter."""
    if dtype == torch.float16:
        return "half"
    if dtype == torch.float32:
        return "float"
    if dtype == torch.float64:
        return "double"
    return "float"


def _neutral_literal_for(literal: str, slang_type: str) -> str:
    """Convert a float neutral literal to the appropriate typed form.

    ``"0.0f"`` → ``"half(0.0)"`` for half, ``"0.0f"`` for float, etc.
    """
    if slang_type == "half":
        inner = literal.replace("f)", ")").rstrip("f")
        return f"half({inner})"
    if slang_type == "float":
        return literal
    if slang_type == "double":
        inner = literal.replace("f)", ")").rstrip("f")
        return f"double({inner})"
    return literal


def _op_template_generic(op_template: str, slang_type: str) -> str:
    """Return the Slang type name for generic instantiation (CG.M13).

    The returned string is used in Slang generic syntax, e.g.
    ``wg_reduce_wave<OpSum>(...)``. This is NOT string-based code
    templating — the Slang function ``wg_reduce_wave<W : IWaveReduction>``
    is a proper generic, and the type name selects the concrete
    implementation at slangc compile time.

    ``slang_type`` is reserved for future dtype-parameterized variants
    (e.g. ``OpSum<half>`` vs ``OpSum<float>``); currently all op structs
    are float-monomorphized so the parameter is unused.
    """
    return op_template
