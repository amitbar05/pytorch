"""Reduction codegen — workgroup reduction, welford, scan.

Extracted from ``VulkanKernel`` via ``ReductionMixin`` (Track 1).
Load mixin and tile-picker extracted to
``reduction_load_mixin`` / ``reduction_tile_picker``
(M15.1.g — Track 1 anti-goal #7 split).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

import sympy
import torch
from torch._inductor.codegen.common import DTYPE_TO_COMPUTATION_DTYPE
from torch._inductor.virtualized import V
from torch.utils._sympy.value_ranges import ValueRanges

from .reduction_load_mixin import ReductionLoadMixin
from .reduction_tile_picker import (
    _WAVE_COMPUTE_DTYPE,
    REDUCTION_TABLE,
    ReductionKind,
    _neutral_literal_for,
    _op_template_generic,
    _reduction_meta,
    _slang_wave_type,
)

if TYPE_CHECKING:
    from torch._inductor.ops_handler import ReductionType


class ReductionMixin(ReductionLoadMixin):
    """Mixin providing reduction, welford, scan, sort, and bucketize codegen.

    Inherits ``_slang_dtype_bytes`` and ``_new_idxvar`` from
    ``ReductionLoadMixin``.
    """

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

        # M11.2: Subgroup-only fast path — when the reduction fits in a
        # single wave, emit a direct WaveActive intrinsic instead of
        # wg_reduce_wave.  This avoids importing reduction.slang (and its
        # groupshared float _wg_reduce_smem[1024+32] LDS declaration),
        # reducing register pressure and eliminating unused LDS.
        if (
            not partitioned
            and layout_2d is None
            and not self.multistage_reduction_entry
            and isinstance(red_numel, int)
            and red_numel <= simd
        ):
            guarded_val = (
                f"((lid.x < {red_numel}) ? ({val}) : ({neutral_lit}))"
                if red_numel < self.max_threadgroup_size
                else str(val)
            )
            # Map reduction kind to the direct wave intrinsic.
            # wave_sum / wave_prod / wave_max / wave_min are in
            # helpers.slang (via import helpers;).
            _wave_fn = {
                ReductionKind.SUM: "wave_sum",
                ReductionKind.PROD: "wave_prod",
                ReductionKind.MAX: "wave_max",
                ReductionKind.MIN: "wave_min",
            }[kind]
            result = self.cse.generate(
                self.stores,
                f"{_wave_fn}({guarded_val})",
                dtype=DTYPE_TO_COMPUTATION_DTYPE[dtype],
            )
            self.headers.add(_wave_fn)
            return result

        # M20.6: wave size flows through the VK_SUBGROUP_SIZE spec const
        # (declared in lib/helpers.slang at constant_id=100, default 64).
        # The rendered Slang reads ``VK_SUBGROUP_SIZE`` instead of a
        # baked ``{simd}u`` literal, so future wave32 dispatch on
        # NAVI21+ silicon overrides the spec const at pipeline create
        # time without any codegen surgery.
        if partitioned is not None and self.multistage_reduction_entry:
            ty, tx, loop_y, loop_x = partitioned
            guarded_val = str(val)
            result = self.cse.generate(
                self.stores,
                f"wg_reduce_wave_2d<{op_tmpl}>({guarded_val}, lid.x, lid.y, {tx}, {ty}, VK_SUBGROUP_SIZE)",
                dtype=DTYPE_TO_COMPUTATION_DTYPE[dtype],
            )
            self.headers.add("wgreduce2d")
        elif layout_2d is not None and not self.multistage_reduction_entry:
            ty, tx = layout_2d
            # n_waves must reflect the ACTUAL threadgroup size (ty * tx), not
            # max_threadgroup_size. When ty * tx < max_threadgroup_size the
            # header emits numthreads(tx, ty, 1), so only (ty*tx//simd) waves
            # actually exist. Using max_threadgroup_size//simd causes
            # wg_reduce_wave to read smem slots that were never written →
            # garbage accumulates into the reduction output.
            n_waves_2d = max(1, (ty * tx) // self.simd_group_size)
            linear_tid = f"lid.y * {tx} + lid.x"
            guarded_val = str(val)
            result = self.cse.generate(
                self.stores,
                f"wg_reduce_wave<{op_tmpl}>({guarded_val}, {linear_tid}, {n_waves_2d}u, VK_SUBGROUP_SIZE)",
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
                f"wg_reduce_wave<{op_tmpl}>({guarded_val}, lid.x, {n_waves}u, VK_SUBGROUP_SIZE)",
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

        # M22.15 fix: when the reduction uses a 2D thread block (numthreads.y > 1),
        # the Welford thread-id must be the *flat* index lid.y*tx + lid.x, not
        # just lid.x.  Using only lid.x makes all threads in wave 1 (lid.y==1)
        # claim wave-index 0 in the cross-wave groupshared reduction, producing
        # a data race and wrong mean/m2 values.
        layout_2d = self._persistent_2d_layout()
        if layout_2d is not None:
            _ty, _tx = layout_2d
            _linear_tid = f"lid.y * {_tx} + lid.x"
        else:
            _linear_tid = "lid.x"

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
                f"({_linear_tid} < {red_numel} ? WelfordResult<float>( {value}, 0.0f, 1.0f ) : "
                f"WelfordResult<float>( 0.0f, 0.0f, 0.0f ))"
                if red_numel < self.max_threadgroup_size
                else f"WelfordResult<float>( {value}, 0.0f, 1.0f )"
            )
        else:  # welford_combine, persistent
            mean, m2, cnt = value
            input_triple = (
                f"({_linear_tid} < {red_numel} ? WelfordResult<float>( {mean}, {m2}, {cnt} ) : "
                f"WelfordResult<float>( 0.0f, 0.0f, 0.0f ))"
                if red_numel < self.max_threadgroup_size
                else f"WelfordResult<float>( {mean}, {m2}, {cnt} )"
            )

        tup_name = f"tmp_wf_{next(self.acc_var_ids)}"
        self.stores.writeline(
            f"WelfordResult<float> {tup_name} = "
            f"wg_welford({input_triple}, {_linear_tid}, {red_size}, VK_SUBGROUP_SIZE);"
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
        """Boolean-OR reduction (any) via shared memory or wave intrinsic."""
        red_numel, _ = self._compute_red_numel()
        red_size = min(red_numel, self.max_threadgroup_size)
        # M20.4.b: when data fits in one wave, use wave_active_any directly
        # (no LDS needed, no vk_wg_reduce_any overhead).
        simd = self.simd_group_size or 64  # RDNA1 wave64 fallback
        if red_size <= simd:
            self.headers.add("wave_active_any")
            return self.cse.generate(
                self.stores,
                f"wave_active_any({value})",
                dtype=torch.bool,
            )
        self.headers.add("wgreduce_any")
        return self.cse.generate(
            self.stores,
            f"vk_wg_reduce_any({value} ? 1.0f : 0.0f, lid.x, {red_size}, VK_SUBGROUP_SIZE) != 0.0f",
            dtype=torch.bool,
        )

    def _xor_sum_reduction(self, src_dtype, value):
        """Bitwise XOR reduction via wave intrinsic or shared memory."""
        red_numel, _ = self._compute_red_numel()
        red_size = min(red_numel, self.max_threadgroup_size)
        layout_2d = self._persistent_2d_layout()
        # M20.4.b: wave fast path for small 1D reductions
        simd = self.simd_group_size or 64  # RDNA1 wave64 fallback
        if red_size <= simd and layout_2d is None:
            self.headers.add("wave_active_bit_xor")
            return self.cse.generate(
                self.stores,
                f"wave_active_bit_xor({value})",
                dtype=DTYPE_TO_COMPUTATION_DTYPE[src_dtype],
            )
        if layout_2d is not None and not self.multistage_reduction_entry:
            ty, tx = layout_2d
            self.headers.add("wgreduce2d_xor")
            result = self.cse.generate(
                self.stores,
                f"vk_wg_reduce_xor_2d({value}, {tx}, lid.y * {tx} + lid.x, {tx * ty}, VK_SUBGROUP_SIZE)",
                dtype=DTYPE_TO_COMPUTATION_DTYPE[src_dtype],
            )
        else:
            self.headers.add("wgreduce_xor")
            result = self.cse.generate(
                self.stores,
                f"vk_wg_reduce_xor({value}, lid.x, {red_size}, VK_SUBGROUP_SIZE)",
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
            f"vk_wg_reduce_arg{suffix}({guarded_val}, lid.x, {red_size}, VK_SUBGROUP_SIZE)",
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
        # M20.6: wave-size literal replaced with VK_SUBGROUP_SIZE spec const
        # (lib/helpers.slang, constant_id=100, default 64).
        result = self.cse.generate(
            self.compute,
            f"wg_scan_exclusive<{scan_op}>"
            f"({value}, lid.x & (VK_SUBGROUP_SIZE - 1u), VK_SUBGROUP_SIZE)",
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
        # M20.6: wave-size literal replaced with VK_SUBGROUP_SIZE spec const.
        result = self.cse.generate(
            self.compute,
            f"wg_scan_inclusive<{scan_op}>({value}, lid.x, {size}u, VK_SUBGROUP_SIZE)",
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
            # M20.6: wave-size literal replaced with VK_SUBGROUP_SIZE spec const.
            f"({guarded_val}, lid.x, {red_size}u, VK_SUBGROUP_SIZE)",
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
        # wg_bitonic_sort_wave(inout float2 v, uint lane) sorts ascending
        # in-place within the wave using WaveReadLaneAt shuffle.
        # For descending sort: negate keys so smallest negated key = largest
        # original key comes first; sentinel for unused slots is +FLT_MAX
        # (after negation, so -(original_guard) → +FLT_MAX → sorts last ✓).
        if descending:
            key_expr = f"-({guarded_key})"
        else:
            key_expr = guarded_key
        pair = self.cse.newvar(dtype=dtypes[0])
        self.compute.writeline(f"float2 {pair} = float2({key_expr}, {guarded_val});")
        sorted_pair = self.cse.newvar(dtype=dtypes[0])
        self.compute.writeline(f"float2 {sorted_pair} = {pair};")
        self.compute.writeline(f"wg_bitonic_sort_wave({sorted_pair}, lid.x);")
        sorted_key = self.cse.newvar(dtype=dtypes[0])
        sorted_val = self.cse.newvar(dtype=dtypes[1])
        if descending:
            self.compute.writeline(f"float {sorted_key} = -({sorted_pair}.x);")
        else:
            self.compute.writeline(f"float {sorted_key} = {sorted_pair}.x;")
        self.compute.writeline(f"float {sorted_val} = {sorted_pair}.y;")
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
