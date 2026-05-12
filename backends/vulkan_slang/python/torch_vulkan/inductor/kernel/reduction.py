"""Reduction codegen — workgroup reduction, welford, scan.

Extracted from ``VulkanKernel`` via ``ReductionMixin`` (Track 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Union

import sympy
import torch
from torch._inductor.codegen.common import DTYPE_TO_COMPUTATION_DTYPE
from torch._inductor.virtualized import V
from torch.utils._sympy.value_ranges import ValueRanges

if TYPE_CHECKING:
    from torch._inductor.ops_handler import ReductionType


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
    """Return the op template name (N.3.a generic disabled — use float-monomorphized)."""
    return op_template


class ReductionMixin:
    """Mixin providing reduction, welford, scan, sort, and bucketize codegen."""

    @staticmethod
    def _slang_dtype_bytes(dtype_str: str) -> int:
        return {
            "double": 8,
            "int64_t": 8,
            "uint64_t": 8,
            "float": 4,
            "int": 4,
            "uint": 4,
            "half": 2,
            "int16_t": 2,
            "uint16_t": 2,
            "int8_t": 1,
            "uint8_t": 1,
            "bool": 1,
        }.get(dtype_str, 4)

    def _new_idxvar(
        self,
        dtype,
        elem_count: Optional[int] = None,
        default_value: Optional[Any] = None,
        is_threadgroup: bool = True,
        bounds: ValueRanges = ValueRanges.unknown(),
        access_pattern: str = "generic",
    ) -> "CSEVariable":
        """Create a CSE variable, optionally backed by groupshared memory.

        Args:
            access_pattern: One of ``"reduction"``, ``"welford"``,
                ``"scan"``, ``"sort"``, or ``"generic"``. Used by M20
                bank-conflict analysis to compute appropriate padding.
        """
        from torch._inductor.codegen.common import CSEVariable

        if isinstance(dtype, torch.dtype):
            dtype = self.dtype_to_str(dtype)
        var_name = f"tmp_acc_{next(self.acc_var_ids)}"
        var = V.kernel.create_cse_var(var_name, bounds, dtype)
        if is_threadgroup:
            self._pw_uses_groupshared = True
            count = 1
            # DR.6: resolve bank-padding gate once (used for both budget
            # tracking and array declaration).
            from .. import config as _cfg

            _use_pad = _cfg.bank_conflict_pad()
            if elem_count is not None and isinstance(elem_count, (int, sympy.Integer)):
                count = max(1, int(elem_count))
                if _use_pad:
                    # M20: use access-pattern-aware padding instead of
                    # blanket +32.  Fall back to 32 if analysis is
                    # unavailable or returns an unexpected value.
                    _elem_bytes = self._slang_dtype_bytes(dtype)
                    _analyzed_pad = _analyze_smem_bank_conflict_risk(
                        num_threads=int(elem_count),
                        element_size=_elem_bytes,
                        access_pattern=access_pattern,
                    )
                    _pad = _analyzed_pad
                    count = count + _pad
            raw_bytes = self._slang_dtype_bytes(dtype) * count
            new_bytes = (raw_bytes + 15) & ~15
            if (
                self._groupshared_bytes_used + new_bytes
                > self._groupshared_budget_bytes
            ):
                raise NotImplementedError(
                    f"Vulkan Inductor: groupshared LDS budget exceeded "
                    f"({self._groupshared_bytes_used + new_bytes} bytes > "
                    f"{self._groupshared_budget_bytes} budget). The "
                    f"driver would spill to scratch — disable persistent "
                    f"reduction for this kernel or shrink rnumel."
                )
            self._groupshared_bytes_used += new_bytes
            decl = f"groupshared {dtype} {var_name}"
            # M20 / DR.6: bank-conflict padding (analysis-driven)
            if elem_count:
                if _use_pad:
                    decl += f"[{self.sexpr(elem_count)} + {_pad}]"
                    # Debug-level log of the analysis decision.
                    import logging

                    _log = logging.getLogger(__name__)
                    _log.debug(
                        "M20 bank-conflict: pattern=%s, dtype=%s, "
                        "threads=%s → pad=%s (was blanket %s)",
                        access_pattern,
                        dtype,
                        int(elem_count),
                        _pad,
                        _BANK_CONFLICT_PAD,
                    )
                else:
                    decl += f"[{self.sexpr(elem_count)}]"
            self.module_scope_decls.writeline(decl + self.suffix)
        else:
            decl = f"{dtype} {var_name}"
            if elem_count:
                decl += f"[{self.sexpr(elem_count)}]"
            if default_value is not None:
                decl += f" = {default_value}"
            self.indexing_code.writeline(decl + self.suffix)
        return var

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: "ReductionType",
        value,
    ):
        cache_key = (src_dtype, reduction_type, value)
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]
        result = self._reduction_nocache(dtype, src_dtype, reduction_type, value)
        self.cse.reduction_cache[cache_key] = result
        return result

    def _reduction_nocache(self, dtype, src_dtype, reduction_type, value):
        """Workgroup reduction using Wave intrinsics + groupshared scratch.

        Layout:
          * One workgroup reduces across the reduction dim.
          * WaveActiveSum / WaveActiveMax condense 64 lanes → 1.
          * `THREADS / simd_group_size` lane-0 writes go to groupshared.
          * Wave 0 reduces the lane-0 values → final scalar.
        """
        assert self.inside_reduction
        if reduction_type in ("welford_reduce", "welford_combine"):
            return self._welford_reduction(dtype, src_dtype, reduction_type, value)
        if reduction_type == "any":
            return self._any_reduction(src_dtype, value)
        if reduction_type in ("argmax", "argmin"):
            if not (isinstance(value, (tuple, list)) and len(value) == 2):
                raise NotImplementedError(
                    "Vulkan Inductor argmax/argmin single-axis codegen "
                    "is pending — see kernel.py:_reduction_nocache. "
                    "Falls back to upstream extern decomp."
                )
            return self._argmin_argmax_reduction(
                dtype, src_dtype, reduction_type, value[0], value[1]
            )
        if reduction_type == "xor_sum":
            return self._xor_sum_reduction(src_dtype, value)
        if reduction_type not in {"sum", "prod", "max", "min"}:
            raise NotImplementedError(
                f"Vulkan Inductor reduction '{reduction_type}' not implemented"
            )

        # N.3.a — Determine wave-native compute dtype.
        # f16 stays native (half); bf16 widens to float.
        acc_dtype = DTYPE_TO_COMPUTATION_DTYPE[src_dtype]
        wave_dtype = _WAVE_COMPUTE_DTYPE.get(src_dtype, acc_dtype)
        wave_slang = _slang_wave_type(wave_dtype)
        # Accumulator Slang type for per-thread partials.
        # For f16 native: use half accumulator (not widened float).
        if wave_dtype != acc_dtype:
            acc_slang = _slang_wave_type(wave_dtype)
        else:
            acc_slang = self.dtype_to_str(acc_dtype)

        reduction_idx = ""
        acc_buf_size = 1
        for rd in self.range_trees:
            if not rd.is_reduction:
                continue
            if reduction_idx:
                reduction_idx += " + "
            reduction_idx += f"{rd.name} * {acc_buf_size}"
            if isinstance(rd.numel, sympy.Integer):
                acc_buf_size *= int(rd.numel)
            else:
                acc_buf_size *= sympy.Symbol(
                    f"{rd.prefix}numel", integer=True, positive=True
                )

        kind = ReductionKind.from_str(reduction_type)
        meta = REDUCTION_TABLE[kind]

        _neutral_element = meta.neutral_element
        if wave_slang != "float":
            _neutral_element = _neutral_literal_for(meta.neutral_literal, wave_slang)

        if self.multistage_reduction_entry:
            local = self._new_idxvar(
                acc_slang, is_threadgroup=False, default_value=_neutral_element
            )
            if kind in (ReductionKind.MAX, ReductionKind.MIN):
                self.compute.writeline(f"{local} = {kind.value}({local}, {value});")
            else:
                self.compute.writeline(f"{local} {meta.accumulate_op} {value};")
            val = local
        else:
            val = value

        neutral_lit = meta.neutral_literal
        if wave_slang != "float":
            neutral_lit = _neutral_literal_for(neutral_lit, wave_slang)
        red_numel, _ = self._compute_red_numel()
        red_size = min(red_numel, self.max_threadgroup_size)
        n_waves = max(1, self.max_threadgroup_size // self.simd_group_size)
        simd = self.simd_group_size
        op_tmpl = _op_template_generic(meta.op_template, wave_slang)
        layout_2d = self._persistent_2d_layout()
        partitioned = getattr(self, "_partitioned_2d_layout_values", None)
        if partitioned is not None and self.multistage_reduction_entry:
            ty, tx, loop_y, loop_x = partitioned
            guarded_val = str(val)
            result = self.cse.generate(
                self.stores,
                f"wg_reduce_wave_2d<{op_tmpl}>({guarded_val}, uint2(lid.x, lid.y), uint2({tx}, {ty}), {simd}u)",
                dtype=DTYPE_TO_COMPUTATION_DTYPE[dtype],
            )
            self.headers.add("wgreduce2d")
        elif layout_2d is not None and not self.multistage_reduction_entry:
            ty, tx = layout_2d
            linear_tid = f"lid.y * {tx} + lid.x"
            guarded_val = str(val)
            result = self.cse.generate(
                self.stores,
                f"wg_reduce_wave<{op_tmpl}>({guarded_val}, {linear_tid}, {n_waves}u, {simd}u)",
                dtype=DTYPE_TO_COMPUTATION_DTYPE[dtype],
            )
            self.headers.add("wgreduce")
        else:
            guarded_val = (
                f"((lid.x < {red_numel}) ? ({val}) : ({neutral_lit}))"
                if red_numel < self.max_threadgroup_size
                else str(val)
            )
            result = self.cse.generate(
                self.stores,
                f"wg_reduce_wave<{op_tmpl}>({guarded_val}, lid.x, {n_waves}u, {simd}u)",
                dtype=DTYPE_TO_COMPUTATION_DTYPE[dtype],
            )
            self.headers.add("wgreduce")
        return result

    def _welford_reduction(self, dtype, src_dtype, reduction_type, value):
        """Welford mean+var reduction — used by LayerNorm/GroupNorm."""
        self.headers.add("welford")
        self.has_welford = True
        red_numel, _ = self._compute_red_numel()
        red_size = min(red_numel, self.max_threadgroup_size)

        if self.multistage_reduction_entry:
            acc_name = f"_wf_acc_{next(self.acc_var_ids)}"
            self.indexing_code.writeline(
                f"WelfordResult<float> {acc_name} = WelfordResult<float>(0.0f, 0.0f, 0.0f);"
            )
            u = next(self.acc_var_ids)
            if reduction_type == "welford_reduce":
                self.compute.splice(f"""
                    {{
                        float _wf_x_{u} = {value};
                        {acc_name}.n += 1.0f;
                        float _wf_d_{u} = _wf_x_{u} - {acc_name}.mean;
                        {acc_name}.mean += _wf_d_{u} / {acc_name}.n;
                        float _wf_d2_{u} = _wf_x_{u} - {acc_name}.mean;
                        {acc_name}.m2 += _wf_d_{u} * _wf_d2_{u};
                    }}
                """)
            else:  # welford_combine
                mean, m2, cnt = value
                self.compute.splice(f"""
                    {{
                        float _wf_bmean_{u} = {mean};
                        float _wf_bm2_{u} = {m2};
                        float _wf_bcnt_{u} = {cnt};
                        float _wf_n_{u} = {acc_name}.n + _wf_bcnt_{u};
                        float _wf_inv_{u} = _wf_n_{u} > 0.0f ? 1.0f / _wf_n_{u} : 0.0f;
                        float _wf_del_{u} = _wf_bmean_{u} - {acc_name}.mean;
                        float _wf_nm_{u} = {acc_name}.mean + _wf_del_{u} * _wf_bcnt_{u} * _wf_inv_{u};
                        float _wf_nm2_{u} = {acc_name}.m2 + _wf_bm2_{u} +
                            _wf_del_{u} * _wf_del_{u} * {acc_name}.n * _wf_bcnt_{u} * _wf_inv_{u};
                        {acc_name} = WelfordResult<float>( _wf_nm_{u}, _wf_nm2_{u}, _wf_n_{u} );
                    }}
                """)
            input_triple = acc_name
        elif reduction_type == "welford_reduce":
            input_triple = (
                f"(lid.x < {red_numel} ? WelfordResult<float>( {value}, 0.0f, 1.0f ) : "
                f"WelfordResult<float>( 0.0f, 0.0f, 0.0f ))"
                if red_numel < self.max_threadgroup_size
                else f"WelfordResult<float>( {value}, 0.0f, 1.0f )"
            )
        else:  # welford_combine, persistent
            mean, m2, cnt = value
            input_triple = (
                f"(lid.x < {red_numel} ? WelfordResult<float>( {mean}, {m2}, {cnt} ) : "
                f"WelfordResult<float>( 0.0f, 0.0f, 0.0f ))"
                if red_numel < self.max_threadgroup_size
                else f"WelfordResult<float>( {mean}, {m2}, {cnt} )"
            )

        tup_name = f"tmp_wf_{next(self.acc_var_ids)}"
        self.stores.writeline(
            f"WelfordResult<float> {tup_name} = "
            f"wg_welford({input_triple}, lid.x, {red_size});"
        )
        comp_dtype = DTYPE_TO_COMPUTATION_DTYPE[dtype]
        mean_v = V.kernel.create_cse_var(
            f"{tup_name}.mean", ValueRanges.unknown(), comp_dtype
        )
        m2_v = V.kernel.create_cse_var(
            f"{tup_name}.m2", ValueRanges.unknown(), comp_dtype
        )
        cnt_v = V.kernel.create_cse_var(
            f"{tup_name}.n", ValueRanges.unknown(), comp_dtype
        )
        return (mean_v, m2_v, cnt_v)

    def _any_reduction(self, src_dtype, value):
        """Boolean-OR reduction (any) via shared memory."""
        self.headers.add("wgreduce_any")
        red_numel, _ = self._compute_red_numel()
        red_size = min(red_numel, self.max_threadgroup_size)
        return self.cse.generate(
            self.stores,
            f"vk_wg_reduce_any({value} ? 1.0f : 0.0f, lid.x, {red_size}) != 0.0f",
            dtype=torch.bool,
        )

    def _xor_sum_reduction(self, src_dtype, value):
        """Bitwise XOR reduction via shared memory."""
        self.headers.add("wgreduce_xor")
        red_numel, _ = self._compute_red_numel()
        red_size = min(red_numel, self.max_threadgroup_size)
        layout_2d = self._persistent_2d_layout()
        if layout_2d is not None and not self.multistage_reduction_entry:
            ty, tx = layout_2d
            self.headers.add("wgreduce2d_xor")
            result = self.cse.generate(
                self.stores,
                f"vk_wg_reduce_xor_2d({value}, {tx}, lid.y * {tx} + lid.x, {tx * ty})",
                dtype=DTYPE_TO_COMPUTATION_DTYPE[src_dtype],
            )
        else:
            result = self.cse.generate(
                self.stores,
                f"vk_wg_reduce_xor({value}, lid.x, {red_size})",
                dtype=DTYPE_TO_COMPUTATION_DTYPE[src_dtype],
            )
        return result

    def _argmin_argmax_reduction(self, dtype, src_dtype, reduction_type, value, index):
        """argmin/argmax reduction — returns (value, index) tuple."""
        assert reduction_type in ("argmax", "argmin")
        is_max = reduction_type == "argmax"
        red_numel, _ = self._compute_red_numel()
        red_size = min(red_numel, self.max_threadgroup_size)
        cmp_op = ">" if is_max else "<"
        suffix = "max" if is_max else "min"
        self.headers.add(f"wgreduce_arg{suffix}")

        n_waves = max(1, self.max_threadgroup_size // self.simd_group_size)
        neutral_val = "(-3.4e38f)" if is_max else "(3.4e38f)"
        guarded_val = (
            f"((lid.x < {red_numel}) ? float2({value}, float({index})) : float2({neutral_val}, 0.0f))"
            if red_numel < self.max_threadgroup_size
            else f"float2({value}, float({index}))"
        )

        if self.multistage_reduction_entry:
            acc_name = f"_arg_{suffix}_{next(self.acc_var_ids)}"
            self.indexing_code.writeline(f"float2 {acc_name} = {guarded_val};")
            op_str = f"(({value}) {cmp_op} ({acc_name}).x ? float2({value}, float({index})) : {acc_name})"
            self.compute.writeline(f"{acc_name} = {op_str};")
            result_val = V.kernel.create_cse_var(
                f"{acc_name}.x", ValueRanges.unknown(), dtype
            )
            result_idx = V.kernel.create_cse_var(
                f"{acc_name}.y", ValueRanges.unknown(), torch.int64
            )
            return (result_val, result_idx)

        result = self.cse.generate(
            self.stores,
            f"vk_wg_reduce_arg{suffix}({guarded_val}, lid.x, {red_size})",
            dtype=dtype,
        )
        result_val = V.kernel.create_cse_var(
            f"{result}.x", ValueRanges.unknown(), dtype
        )
        result_idx = V.kernel.create_cse_var(
            f"{result}.y", ValueRanges.unknown(), torch.int64
        )
        return (result_val, result_idx)

    def partial_accumulate(self, name: str, reduction_type, val, extra_meta):
        from torch._inductor.codegen.simd import PartialAccumulate

        self.saved_partial_accumulate.append(
            PartialAccumulate(name, reduction_type, val)
        )

    # ── P2.1/M1: Wave primitive codegen ───────────────────────────

    def _emit_wave_broadcast(self, value, dtype=None) -> "CSEVariable":
        """Broadcast lane-0's value to all lanes via WaveReadLaneFirst.

        This is faster than shared memory for distributing a uniform
        value computed by lane 0 to the rest of the wave.  Useful for:
        - Filling uniform constants without shared memory
        - Leader election in scan
        - Distributing the result of a wave-level reduction

        Emits: ``wave_broadcast({value})`` which compiles to
        ``WaveReadLaneFirst({value})`` via ``reduction.slang``.
        """
        self.headers.add("wave_broadcast")
        result = self.cse.generate(
            self.compute,
            f"wave_broadcast({value})",
            dtype=dtype,
        )
        return result

    def emit_wave_scan_exclusive(
        self, value, scan_op: str, dtype=None
    ) -> "CSEVariable":
        """Wave-level exclusive prefix scan via WavePrefixSum/Product/Min/Max.

        Returns the exclusive prefix: for lane i, the result is the
        reduction (according to ``scan_op``) of all lanes with index < i.
        Lane 0 always returns the identity element.

        Args:
            value: The CSE variable or expression to scan.
            scan_op: One of ``"IScanAdd"``, ``"IScanMul"``, ``"IScanMax"``,
                     ``"IScanMin"``.
            dtype: Output dtype for CSE registration.
        """
        self.headers.add("wg_scan_exclusive")
        result = self.cse.generate(
            self.compute,
            f"wg_scan_exclusive<{scan_op}>"
            f"({value}, lid.x & ({self.simd_group_size}u - 1u), {self.simd_group_size}u)",
            dtype=dtype,
        )
        return result

    def emit_wave_scan_inclusive(
        self, value, scan_op: str, size: int, dtype=None
    ) -> "CSEVariable":
        """Workgroup-level inclusive scan via IScan wave-prefix intrinsics.

        Uses the two-phase approach in ``wg_scan_inclusive<S>``:
        Phase 1 computes per-wave inclusive scan via wave_prefix;
        Phase 2 propagates per-wave totals across waves via shared memory.

        Args:
            value: The CSE variable or expression to scan.
            scan_op: One of ``"IScanAdd"``, ``"IScanMul"``, ``"IScanMax"``,
                     ``"IScanMin"``.
            size: Total number of elements to scan (≤ workgroup size).
            dtype: Output dtype for CSE registration.
        """
        self.headers.add("wg_scan_inclusive")
        result = self.cse.generate(
            self.compute,
            f"wg_scan_inclusive<{scan_op}>"
            f"({value}, lid.x, {size}u, {self.simd_group_size}u)",
            dtype=dtype,
        )
        return result

    def scan(self, dtypes, combine_fn, values):
        red_numel, _ = self._compute_red_numel()
        if not isinstance(red_numel, int) or red_numel > self.max_threadgroup_size:
            raise NotImplementedError(
                "Vulkan Inductor: scan requires reduction numel <= "
                f"{self.max_threadgroup_size} (got {red_numel}). "
                "Set TORCHINDUCTOR_DISABLE_COMBO_KERNEL=1 to fall back to eager dispatch."
            )
        if len(values) != 1:
            raise NotImplementedError(
                "Vulkan Inductor: scan only supports single-value scans."
            )
        # P2.1/M1: route scan through IScan generic (uses WavePrefixSum/
        # WavePrefixProduct for add/mul, shared-memory tree for max/min).
        # _probe_scan_op now returns IScan* struct names for the new
        # wg_scan_inclusive generic path.
        self.headers.add("wg_scan_inclusive")
        struct_name = self._probe_scan_op(combine_fn)
        val = values[0]
        red_size = min(red_numel, self.max_threadgroup_size)
        guarded_val = (
            f"((lid.x < {red_numel}) ? ({val}) : {struct_name}::identity())"
            if red_numel < self.max_threadgroup_size
            else str(val)
        )
        result = self.cse.generate(
            self.compute,
            f"wg_scan_inclusive<{struct_name}>"
            f"({guarded_val}, lid.x, {red_size}u, {self.simd_group_size}u)",
            dtype=dtypes[0],
        )
        return (result,)

    @staticmethod
    def _probe_scan_op(combine_fn) -> str:
        """Determine the Slang scan struct name from the combine_fn.

        Evaluates combine_fn with a minimal ops handler that records which
        binary op was called. Maps the op name to the corresponding Slang
        struct: add→IScanAdd, mul→IScanMul, maximum→IScanMax, minimum→IScanMin.

        P2.1/M1: Uses the new IScan interface (with wave_prefix) for unified
        wave-prefix → inclusive scan path via wg_scan_inclusive.
        """
        from torch._inductor.virtualized import V

        # Map recorded op → Slang IScan struct name.
        _SCAN_OP_MAP = {
            "add": "IScanAdd",
            "mul": "IScanMul",
            "maximum": "IScanMax",
            "minimum": "IScanMin",
        }

        class _ScanProbeHandler:
            """Minimal ops handler that records the first binary scan op."""

            def __init__(self):
                self.op_name = None

            def _default(self, name, args, kwargs):
                if name in _SCAN_OP_MAP and self.op_name is None:
                    self.op_name = name
                return 0

            # Delegate everything else through _default.
            def __getattr__(self, name):
                if name.startswith("_") or name in ("_default",):
                    raise AttributeError(name)
                return lambda *a, **kw: self._default(name, a, kw)

        probe = _ScanProbeHandler()
        with V.set_ops_handler(probe):
            combine_fn(("_sa",), ("_sb",))

        if probe.op_name is None or probe.op_name not in _SCAN_OP_MAP:
            raise NotImplementedError(
                f"Vulkan Inductor: scan combine_fn not recognized (op={probe.op_name})."
            )
        return _SCAN_OP_MAP[probe.op_name]

    def sort(self, dtypes, values, stable, descending):
        red_numel, _ = self._compute_red_numel()
        if not isinstance(red_numel, int) or red_numel > self.max_threadgroup_size:
            raise NotImplementedError(
                "Vulkan Inductor: sort requires reduction numel <= "
                f"{self.max_threadgroup_size} (got {red_numel}). "
                "Set TORCHINDUCTOR_DISABLE_COMBO_KERNEL=1 to fall back to eager dispatch."
            )
        if len(values) != 2:
            raise NotImplementedError(
                "Vulkan Inductor: sort only supports (key, value) pair sorting."
            )
        self.headers.add("wg_sort")
        key, val = values[0], values[1]
        red_size = min(red_numel, self.max_threadgroup_size)
        guarded_key = (
            f"((lid.x < {red_numel}) ? ({key}) : "
            f"{'asfloat(0x7F7FFFFFu)' if not descending else 'asfloat(0xFF7FFFFFu)'})"
            if red_numel < self.max_threadgroup_size
            else str(key)
        )
        guarded_val = (
            f"((lid.x < {red_numel}) ? ({val}) : 0.0f)"
            if red_numel < self.max_threadgroup_size
            else str(val)
        )
        pair = self.cse.generate(
            self.compute,
            f"float2({guarded_key}, {guarded_val})",
            dtype=dtypes[0],
        )
        sorted_pair = self.cse.generate(
            self.compute,
            f"wg_bitonic_sort_float2({pair}, lid.x, {red_size}, "
            f"{'true' if descending else 'false'})",
            dtype=dtypes[0],
        )
        sorted_key = self.cse.newvar(dtypes[0])
        sorted_val = self.cse.newvar(dtypes[1])
        self.compute.writeline(f"float {sorted_key} = ({sorted_pair}).x;")
        self.compute.writeline(f"float {sorted_val} = ({sorted_pair}).y;")
        return (sorted_key, sorted_val)

    def bucketize(
        self,
        values,
        boundaries,
        boundary_indices,
        indexing_dtype,
        right,
        sorter,
        sorter_indices,
    ):
        raise NotImplementedError(
            "Vulkan Inductor: bucketize not yet implemented via Slang codegen. "
            "Set TORCHINDUCTOR_DISABLE_COMBO_KERNEL=1 to avoid searchsorted/topk patterns."
        )
