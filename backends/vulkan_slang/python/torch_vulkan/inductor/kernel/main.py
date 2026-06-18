"""VulkanKernel — SIMDKernel subclass emitting Slang compute-shader source.

Track 1 codegen refactor: methods extracted into mixin modules
(``pointwise.py``, ``reduction.py``, ``indexing.py``, ``header.py``).
This file retains the core class definition, initialization, and
heuristic selectors.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Optional

import sympy
import torch
from torch._inductor.codegen.common import (
    CSE,
    IndentedBuffer,
    PythonPrinter,
)
from torch._inductor.codegen.simd import (
    SIMDKernel,
)
from torch.utils._ordered_set import OrderedSet

from ..expr_printer import VulkanExprPrinter
from ..overrides import DTYPE_TO_SLANG, VulkanOverrides
from .header import HeaderMixin
from .indexing import IndexingMixin
from .pointwise import PointwiseMixin
from .reduction import ReductionMixin
from .symbolic import get_static_numel, is_dynamic
from .threadgroup_sizing import ThreadgroupSizingMixin

if TYPE_CHECKING:
    from torch._inductor.ops_handler import ReductionType, StoreMode


# ── M18: Dtype-aware CSE variable declaration ──────────────────────
# Upstream CSE uses a static ``newvar_prefix`` for all variable declarations.
# int64 index arithmetic stored in a ``float`` variable loses precision
# for tensors with >2**24 elements.  VulkanCSE overrides ``generate`` to
# inspect the ``dtype`` kwarg and emit the correct Slang type prefix.

_DTYPE_TO_CSE_PREFIX: dict[torch.dtype, str] = {
    torch.float32: "float ",
    torch.float16: "float ",
    torch.bfloat16: "float ",
    torch.int64: "int64_t ",
    torch.int32: "int ",
    torch.int16: "int ",
    torch.int8: "int ",
    torch.uint8: "int ",
    torch.bool: "bool ",
}


class VulkanCSE(CSE):
    """Dtype-aware CSE that emits correct Slang type prefix per-variable.

    When ``dtype`` is passed to ``generate``, the declaration prefix
    is switched from the default ``float`` to the appropriate Slang type
    (e.g. ``int64_t`` for int64 indices).  Float-like dtypes keep the
    ``float`` prefix to avoid churn in existing codegen output.
    """

    def generate(
        self,
        buffer,
        expr,
        *,
        bounds=None,
        write=True,
        assignment=True,
        dtype=None,
        shape=None,
    ):
        from torch._inductor.codegen.common import ValueRanges

        if bounds is None:
            bounds = ValueRanges.unknown()
        if dtype is not None and dtype in _DTYPE_TO_CSE_PREFIX:
            old_prefix = self.prefix
            self.prefix = _DTYPE_TO_CSE_PREFIX[dtype]
            try:
                return super().generate(
                    buffer,
                    expr,
                    bounds=bounds,
                    write=write,
                    assignment=assignment,
                    dtype=dtype,
                    shape=shape,
                )
            finally:
                self.prefix = old_prefix
        return super().generate(
            buffer,
            expr,
            bounds=bounds,
            write=write,
            assignment=assignment,
            dtype=dtype,
            shape=shape,
        )


class VulkanKernel(
    PointwiseMixin,
    ReductionMixin,
    IndexingMixin,
    HeaderMixin,
    ThreadgroupSizingMixin,
    SIMDKernel,
):
    """SIMDKernel subclass emitting Slang compute-shader source."""

    overrides = VulkanOverrides  # type: ignore[assignment]
    suffix = ";"
    newvar_prefix = "float "
    max_threadgroup_size = 256
    simd_group_size = 64
    pexpr = PythonPrinter().doprint
    _vk_printer = VulkanExprPrinter()
    sexpr = _vk_printer.doprint
    kexpr = sexpr
    headers: OrderedSet[str] = OrderedSet()
    multistage_reduction_entry = []

    _device_simd_group_size: Optional[int] = None

    @classmethod
    def _get_device_simd_group_size(cls) -> int:
        if cls._device_simd_group_size is not None:
            return cls._device_simd_group_size
        try:
            from torch._dynamo.device_interface import get_interface_for_device

            iface = get_interface_for_device("vulkan")
            props = iface.Worker.get_device_properties()
            cls._device_simd_group_size = props.subgroup_size
        except Exception:
            cls._device_simd_group_size = 64
        return cls._device_simd_group_size

    def should_use_persistent_reduction(self) -> bool:
        """Determine whether the reduction loop should use a persistent
        grid-stride pattern (one threadgroup loops over chunks).

        Currently not wired into codegen — the reduction path always
        uses the cooperative/non-cooperative branching from
        ``should_use_cooperative_reduction`` instead.  This method is
        retained for a future C3 task (see ROADMAP.md pillar C3).
        """
        rnumel = sympy.S.One
        for rd in self.range_trees:
            if rd.is_reduction:
                rnumel = rnumel * rd.numel
        if is_dynamic(rnumel):
            return True
        rn = int(rnumel)
        if rn > 8192:
            return False
        return True

    def _has_welford_reduction(self) -> bool:
        """True if any reduction in this kernel is a welford variant.

        AMD RADV miscompiles per-thread dynamic-indexed local arrays in
        shaders that combine `GroupMemoryBarrier` with welford's groupshared
        cross-thread combine. Detecting welford ahead of codegen lets us
        bias toward cooperative single-workgroup reduction (which uses
        wave intrinsics, no groupshared barrier sequence).
        """
        try:
            for rn in self.features.reduction_nodes():
                rt = (
                    rn.get_reduction_type()
                    if hasattr(rn, "get_reduction_type")
                    else None
                )
                if rt and "welford" in rt:
                    return True
        except Exception:
            pass
        return False

    def should_use_cooperative_reduction(self) -> bool:
        rnumel = sympy.S.One
        for rd in self.range_trees:
            if rd.is_reduction:
                rnumel = rnumel * rd.numel
        # Dynamic rnumel always uses cooperative reduction regardless of
        # reflection metadata — the threshold is not applicable.
        if is_dynamic(rnumel):
            return True
        rn = int(rnumel)
        numel_hint = 1
        for rd in self.range_trees:
            if not rd.is_reduction:
                if is_dynamic(rd.numel):
                    numel_hint = 1
                    break
                numel_hint *= int(rd.numel)

        # M20.9: reflection-aware threshold adjustment.
        #
        # Cooperative reduction adds per-wave sync + extra register pressure.
        # Two orthogonal signals from SPIR-V reflection refine the static
        # thresholds:
        #
        #   SGPR pressure penalty (num_sgprs > 64):
        #     A register-heavy kernel has little headroom left; enabling
        #     cooperative mode would push more waves into spill territory.
        #     Lower the rnumel threshold by 50 % so we switch to cooperative
        #     only for larger reductions where the benefit outweighs the cost.
        #
        #   I/O pressure boost (num_loads + num_stores > 128):
        #     A memory-bandwidth-dominated kernel benefits from cooperative
        #     reduction because the sync overhead is hidden by memory latency.
        #     Raise the rnumel threshold by 2× so more kernels get the
        #     latency-hiding benefit.
        #
        # Fallback: when reflection data is not yet available (first compile),
        # the unmodified static thresholds are used.
        _threshold_scale = 1.0
        try:
            # _get_cached_io_pressure returns num_loads + num_stores.
            io_pressure = self._get_cached_io_pressure()
            # _get_cached_num_sgprs returns the SGPR count from reflection.
            num_sgprs = self._get_cached_num_sgprs()
            if num_sgprs is not None and num_sgprs > 64:
                # Register-heavy: lower threshold (fewer kernels switch to
                # cooperative) — avoids the additional pressure from sync.
                _threshold_scale *= 0.5
            if io_pressure is not None and io_pressure > 128:
                # Memory-bandwidth-heavy: raise threshold (more kernels switch
                # to cooperative) — sync overhead is hidden by memory latency.
                _threshold_scale *= 2.0
        except Exception:
            # Defensive: never let reflection failures affect correctness.
            _threshold_scale = 1.0

        def _scaled(base_threshold: int) -> int:
            return int(base_threshold * _threshold_scale)

        if self._has_welford_reduction():
            if numel_hint <= 16:
                return rn <= _scaled(65536)
            return rn <= _scaled(8192)
        if numel_hint <= 4:
            return rn <= _scaled(131072)
        if numel_hint <= 16:
            return rn <= _scaled(65536)
        return rn <= _scaled(4096)

    def __init__(self, tiling: dict[str, sympy.Expr], **kwargs: Any) -> None:
        super().__init__(tiling, **kwargs)
        # M18: Replace upstream CSE with dtype-aware VulkanCSE so int64
        # index variables are declared as ``int64_t`` instead of ``float``,
        # preserving precision for large tensors (>2**24 elements).
        self.cse = VulkanCSE(self.newvar_prefix, self.suffix)
        self.simd_group_size = self._get_device_simd_group_size()
        self._packed16: Optional[bool] = None
        self.has_welford = False  # P5.1: set before _pick_threadgroup_size
        self.max_threadgroup_size = self._pick_threadgroup_size()
        # M11.7: occupancy gate — warn if estimated occupancy < 50 %.
        self._check_occupancy_gate()
        # M11.3: register-tile size (0 = disabled, set during codegen_kernel).
        self._register_tile_size: int = 0
        # TRAIN.6-F1: Per-instance multistage_reduction_entry to prevent
        # state leakage between VulkanKernel instances. Previously a class
        # variable, causing entries from one kernel to persist into the next
        # (e.g. combo-kernel subkernels sharing the same list).
        self.multistage_reduction_entry: list = []
        self.acc_var_ids = itertools.count()
        self.module_scope_decls = IndentedBuffer()
        self.multistage_load_cache: dict[tuple[str, str], str] = {}
        self.multistage_load_seq = itertools.count()
        self._packed16_dtype: Optional[torch.dtype] = None
        self._packed16_bufs: set[str] = set()
        self._atomic_out_bufs: set[str] = set()
        self._packed16_load_only = False
        self._vec_width: int = 1
        self._vec4_pw_active: bool = False
        self._vec4_pw_bufs: set[str] = set()
        self._packed16_vw_active: bool = False
        self._p16_load_records: list[tuple[str, str, str]] = []
        self._p16_store_records: list[tuple[str, str, str]] = []
        self._groupshared_bytes_used: int = 0
        self._groupshared_budget_bytes: int = 64 * 1024
        self._reduction_type: Optional[str] = None
        # Per-entry brace counter so mixed partitioned (2 braces)
        # and standard (1 brace) entries accumulate correctly.
        self._multistage_brace_count: int = 0
        # Track which index variables have been hoisted to function
        # scope so the epilogue can reference them after the loop.
        self._hoisted_vars: set[str] = set()
        self._hoisted_decls: IndentedBuffer = IndentedBuffer()
        # Loop template replay: when disable_reduction flushes the
        # reduction loop (first codegen_body call), the for-loop
        # structure is saved.  The second codegen_body call replays
        # it so pointwise epilogue code runs inside the same loop.
        self._loop_template: IndentedBuffer = IndentedBuffer()
        self._loop_template_saved: Optional[str] = None
        self._loop_brace_count: int = 0

        # Track 5.7: Structural eligibility tracking for vec4/packed16.
        # Per-buffer index records for BlockPatternMatcher analysis:
        #   (buffer_inner_name, sympy_index_expr, is_load: bool)
        self._pw_index_records: list[tuple[str, sympy.Expr, bool]] = []
        # Operation-level flags for ineligibility detection:
        self._pw_has_early_return: bool = False
        self._pw_has_atomic_op: bool = False
        self._pw_has_scan_or_linear: bool = False
        self._pw_has_wave_ops: bool = False
        self._pw_uses_subbyte_packing: bool = False
        self._pw_uses_groupshared: bool = False
        # CG.M8: inline bwd_diff tracking
        # Maps aten_op strings (e.g. "aten.silu_backward") to sets of
        # (input_cse_var, grad_out_cse_var, output_buf_name) tuples.
        # Populated during inner_fn codegen; consumed during body
        # emission to inject bwd_diff() calls instead of generic arithmetic.
        self._bwd_diff_unary_ops: dict[str, list[tuple[str, str, str]]] = {}
        self._bwd_diff_binary_ops: dict[str, list[tuple[str, str, str, str, str]]] = {}
        # Modules that need to be imported (e.g. "pointwise", "losses")
        self._bwd_diff_imports: set[str] = set()
        # GPU.5: Persistent pointwise mode — when True, the pointwise body
        # is wrapped in a grid-stride loop so one kernel handles multiple
        # small operations.  Set by the scheduler via _enable_persistent_mode().
        self._persistent_mode: bool = False
        # P3.1/M9: MUST be set BEFORE node.codegen() → load()/store()
        # calls (during codegen_node_schedule_with_kernel) so that
        # _buf_path() can prefix buffer accesses with ``args.`` when
        # ParameterBlock is active.
        from .. import config

        self._use_parameter_block: bool = config.parameter_block()
        for _rn in self.features.reduction_nodes():
            _rt = (
                _rn.get_reduction_type() if hasattr(_rn, "get_reduction_type") else None
            )
            if _rt in ("sum", "prod", "max", "min"):
                self._reduction_type = _rt
                break

    index_dtype = "uint"

    def _estimate_vgprs(self) -> int:
        """Estimate VGPR count from kernel configuration (M4).

        Called during __init__ before body codegen, so we can't count
        actual CSE variables. Instead we use feature-based estimation:

        - Base: 4 VGPRs (lane index, global index, 2 temps)
        - Per I/O buffer: 1 VGPR (for the loaded value)
        - Reduction accumulator: 2-4 VGPRs depending on type
        - Welford: +7 VGPRs (mean, M2, count, 4 temps)
        - f64: ×2 multiplier on all estimates
        - Reduction loop overhead: +4 VGPRs
        """
        base = 4
        n_in = len(self.args.input_buffers)
        n_out = len(self.args.output_buffers)
        n_io = n_in + n_out

        # Accumulator cost by reduction type
        acc_cost = 0
        if self.inside_reduction:
            acc_cost = 2  # sum/prod/min/max: 1 acc + 1 temp
            if self.has_welford:
                acc_cost = 7  # mean, M2, count, 4 temps for welford
            acc_cost += 4  # loop overhead: counter, stride, 2 temps

        vgprs = base + n_io + acc_cost

        # f64 doubles register pressure (2×32-bit regs per value)
        dtype_bytes = 2 if self._packed16 else 4
        if dtype_bytes >= 8:
            vgprs *= 2
        # f16/packed16 halves pressure (2 values per 32-bit reg)
        elif dtype_bytes <= 2 and self._packed16:
            vgprs = max(vgprs // 2, base)

        if getattr(self, "_partitioned_2d_active", False):
            vgprs += 6  # Y-axis index, 2 boundary checks, 3 temps

        return vgprs

    def _compute_config_key(self) -> str:
        """Compute a stable config-key from kernel characteristics.

        P3.3 / DR.3: This key is independent of workgroup size (unlike the Slang
        source hash) so cached reflection data from a prior compile can
        be found even when WG size changes.  Used to look up actual VGPR
        counts from SPIR-V reflection to refine WG sizing.

        DR.3: Threads subgroup_size and a structural loop-depth proxy
        into the key so register-pressure-divergent kernel variants
        get distinct buckets, eliminating false cache hits.
        """
        return self.config_key

    @property
    def config_key(self) -> str:
        import hashlib

        from torch._inductor.virtualized import V

        # Structural loop-depth proxy: number of reduction axes × max
        # reduction depth correlates with loop nesting in emitted code.
        # Two kernels with identical buffer layouts but different
        # reduction arity (e.g. sum(dim=0) vs sum(dim=[0,1])) produce
        # different loop nests and different register pressure.
        _loop_depth_proxy = 0
        for rd in self.range_trees:
            if rd.is_reduction:
                _loop_depth_proxy += 1
        # Note: cooperative/persistent reduction distinction is deliberately
        # excluded from _loop_depth_proxy here.  should_use_cooperative_reduction
        # calls _get_cached_io_pressure / _get_cached_num_sgprs, both of which
        # call _compute_config_key → config_key, creating a circular dependency
        # that causes infinite recursion.  The reduction arity from range_trees
        # above is sufficient differentiation; inside_reduction is already in
        # the `parts` list below.

        # M-pipeline-3: include the input + output buffer DTYPES in the
        # cache key. Two kernels with identical shape / reduction
        # structure but different dtypes (e.g. fp32 vs fp16 vs int64)
        # emit different SPIR-V with different VGPR pressure — without
        # hashing dtype here, the reflection-metrics cross-index
        # (`_reflection_metrics_by_key` in `runtime/slangc.py`) returns
        # the wrong kernel's reflection data and M11.1's WG sizing
        # silently picks a suboptimal numthreads.
        #
        # `V.graph.get_dtype(outer_name)` is the canonical resolver
        # (same path `kernel/header.py` uses to emit buffer types).
        # Fail-soft: if the graph context is unavailable (rare path —
        # e.g. when computing the key before the graph is bound, or
        # when called on a mock kernel during unit testing), fall back
        # to a marker so the key still differentiates kernels from the
        # cached-but-graphless case.
        def _resolve_dtype(outer_name) -> str:
            try:
                d = V.graph.get_dtype(outer_name)
                return str(d)
            except Exception:  # noqa: BLE001 — defensive against ambient-state misses
                return "?"

        def _outer_names(buf_container):
            """Iterate outer-name keys from either a dict (production
            KernelArgs) or a plain list/tuple (test mocks). Stable
            order: dicts preserve insertion order in Python 3.7+, lists
            preserve their natural order."""
            if hasattr(buf_container, "keys"):
                return list(buf_container.keys())
            return list(buf_container)

        _input_dtypes = tuple(
            _resolve_dtype(outer)
            for outer in _outer_names(self.args.input_buffers)
        )
        _output_dtypes = tuple(
            _resolve_dtype(outer)
            for outer in _outer_names(self.args.output_buffers)
        )

        # M-pipeline-3: hash the push-constant layout (sorted set of
        # sizevar inner names). Two kernels with identical buffer
        # counts but different dynamic-dim sets (e.g. one with a
        # dynamic batch dim N, another with a dynamic seq-len S)
        # produce different PC structs and reflection metrics.
        # `args.sizevars` is a dict { sympy_expr → inner_name }; we
        # hash the inner-name set (which corresponds to PC field names
        # in the emitted struct). Defensive against mocks that omit it.
        _sizevars = getattr(self.args, "sizevars", None)
        if _sizevars is None:
            _pc_layout: tuple = ()
        elif hasattr(_sizevars, "values"):
            _pc_layout = tuple(sorted(str(v) for v in _sizevars.values()))
        else:
            _pc_layout = tuple(sorted(str(v) for v in _sizevars))

        # M-pipeline-3: hash a proxy for descriptor_counts. The true
        # `descriptor_counts` comes from post-compile SPIR-V reflection
        # so it is not available here. Two pre-compile signals together
        # determine the binding layout: the count of in-place buffers
        # (which collapse two logical bindings into one descriptor) and
        # the ParameterBlock flag (which packs all storage buffers into
        # one `ParameterBlock<>` slot via `[[vk::binding(0)]]`).
        _inplace_count = len(getattr(self.args, "inplace_buffers", {}) or {})
        _use_param_block = int(bool(getattr(self, "_use_parameter_block", False)))

        parts = [
            str(len(self.args.input_buffers)),
            str(len(self.args.output_buffers)),
            str(int(self.inside_reduction)),
            str(int(self.has_welford)),
            str(int(self._packed16 if self._packed16 is not None else 0)),
            str(getattr(self, "_reduction_type", None) or "none"),
            str(int(getattr(self, "_partitioned_2d_active", False))),
            # DR.3: subgroup_size distinguishes wave32 vs wave64 codegen
            str(self.simd_group_size or 64),
            # DR.3: loop-depth proxy — distinct buckets for different
            # reduction arity / persistent vs cooperative
            str(_loop_depth_proxy),
            # M-pipeline-3: dtype + PC layout + descriptor-binding proxy
            # — these were silent collision sources before. See block
            # comment above for the bug and rationale.
            "in_dt:" + ",".join(_input_dtypes),
            "out_dt:" + ",".join(_output_dtypes),
            "pc:" + ",".join(_pc_layout),
            "ip:" + str(_inplace_count),
            "pb:" + str(_use_param_block),
        ]
        raw = "|".join(parts)
        # M-pipeline-3: prefix bumped `cfg_` → `cfg2_` so any cached
        # entries from the old key format (which omitted dtype / PC /
        # descriptor-count dimensions) cannot collide with new entries
        # in process-lifetime caches like `_reflection_metrics_by_key`.
        return "cfg2_" + hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _get_actual_vgprs(self) -> Optional[int]:
        """Query cached reflection metrics for this kernel's config.

        Returns the actual VGPR count from SPIR-V reflection if a
        prior compilation of the same kernel config exists. Returns
        None on first compile (no cached reflection yet).
        """
        from .. import config as _cfg

        if not _cfg.reflection_enabled():
            return None

        config_key = self._compute_config_key()
        from torch_vulkan.inductor.runtime import get_cached_metrics_for_key

        metrics = get_cached_metrics_for_key(config_key)
        if metrics is None:
            return None
        vgprs = metrics.get("vgprs")
        if vgprs is None:
            return None
        try:
            return int(vgprs)
        except (TypeError, ValueError):
            return None

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        return DTYPE_TO_SLANG.get(dtype, "float")
