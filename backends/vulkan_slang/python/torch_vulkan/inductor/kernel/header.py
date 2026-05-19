"""Slang buffer-binding, header-emission, and codegen-call surface.

Extracted from ``VulkanKernel`` via ``HeaderMixin`` (Track 1).

Owns:
  - ``codegen_body`` — assembles load/compute/store blocks
  - ``codegen_kernel`` — full shader source generation
  - ``call_kernel`` — wrapper-side dispatch grid + arg emission
  - Helper emitter and misc stubs

**M27 — Workgroup Size Convention**

All generated shaders use ``[numthreads(..., ..., 1)]`` with the following
guidelines:

  - **Pointwise kernels**: ``[numthreads(256, 1, 1)]`` by default;
    ``_pick_threadgroup_size_pointwise()`` may select smaller sizes
    (down to wave32 = 64) for tiny problems or register-heavy chains.
  - **Reduction kernels**: ``[numthreads(256, 1, 1)]`` for 1D reductions;
    ``_pick_threadgroup_size_reduction()`` may use fewer threads when
    VGPR pressure is high (welford, multi-axis).
  - **2D persistent reductions**: ``[numthreads(TX, TY, 1)]`` with a
    2-D layout for better data locality and reduced lane-crossing.
  - **Matmul / Conv templates**: ``[numthreads(WG_N, WG_M, 1)]`` or
    ``[numthreads(TILE, TILE, 1)]`` driven by tile configuration.
  - **RDNA1 wave64**: All sizes are multiples of 64 wherever practical
    (256 = 4 waves, 64 = 1 wave) to guarantee full-wave occupancy.
  - **Hard cap**: All sizes ≤ 1024 (Vulkan minimum guarantee);
    ``_validate_workgroup_size()`` in ``slang_validator.py`` enforces
    this plus the wave64-alignment advisory.

Rationale:
  - 256 threads = 4 waves on RDNA1 (wave64) — excellent occupancy.
  - 16×16 = 256 threads with 2D grid — better for 2D data locality.
  - Pointwise kernels benefit from more threads to hide memory latency.
  - Reduction kernels with high VGPR count need fewer threads per WG
    to stay under the VGPR budget (fewer waves per CU = more regs/wave).

**M28 — Push-Constant Struct Field Ordering Convention**

Push-constant struct fields are ordered by alignment requirements to
minimize padding:

  1. **uint32_t fields** (4-byte aligned): numels, strides, flags
  2. **float fields** (4-byte aligned): scale factors, alpha, beta
  3. **uint64_t fields** (8-byte aligned): large counters (if any)
  4. **float2 / float4 fields** (8/16-byte aligned): compound values

Within each alignment group, fields are ordered by semantic group:

  - Shape metadata (M, N, K, numels)
  - Stride metadata (stride_a, stride_b, …)
  - Computation parameters (alpha, beta, scale)
  - Optional / conditional fields last

This convention minimizes struct size (no hidden padding between
groups) and keeps related fields together for readability.  Templates
that deviate are noted in the inline comments of each template.
"""

from __future__ import annotations

from typing import Any, Optional

import sympy
import torch
from torch._inductor.codegen.common import (
    IndentedBuffer,
    InplacedBuffer,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..slang_helpers import emit_helpers

# Max Vulkan compute workgroup count per dimension.
# X is the primary launch axis; Y serves as overflow.
MAX_COMPUTE_WG_X = 65535


class HeaderMixin:
    """Mixin providing body assembly, shader codegen, and dispatch-call codegen."""

    def codegen_body(self) -> None:
        # CG.M8: emit any registered inline bwd_diff operations into
        # the compute buffer BEFORE assembling the body.  These Slang
        # calls replace the generic arithmetic that the inner_fn
        # emitted (the inner_fn's ops are still in loads/compute but
        # will be dead-code eliminated by _eliminate_dead_code).
        self._emit_inline_bwd_diff_body()

        # Track roots that were fully processed (for-loop emitted + closed)
        # so the force-codegen pass below doesn't re-emit an empty for-loop.
        _processed_roots: set = set()  # names of fully-processed reduction roots

        if self.multistage_reduction_entry:
            # Prepend hoisted declarations to body so they appear before
            # the for loop at function scope.
            if self._hoisted_decls._lines:
                self.body._lines[0:0] = self._hoisted_decls._lines
                self._hoisted_decls = IndentedBuffer()
            with self.body.indent():
                self.body.splice(self.loads)
                self.body.splice(self.compute)
            brace_count = self._multistage_brace_count
            self._multistage_brace_count = 0
            self.body.writeline("}" * brace_count)
            self.body.splice(self.stores)
            # Save loop template for pointwise epilogue replay.
            # Freeze the current template and reset for next use.
            self._loop_template_saved = self._loop_template.getvalue()
            self._loop_template = IndentedBuffer()
            self._loop_brace_count = brace_count
            self.cse.invalidate(
                OrderedSet(
                    v
                    for item in self.cse.reduction_cache.values()
                    for v in (item if isinstance(item, tuple) else (item,))
                )
            )
            while self.multistage_reduction_entry:
                popped = self.multistage_reduction_entry.pop()
                _processed_roots.add(popped.root.prefix)
                popped.cache_clear()
        elif self._loop_template_saved is not None:
            # Pointwise epilogue after a reduction: replay the loop
            # structure so index variables are in scope.
            self.body.splice(self._loop_template_saved)
            with self.body.indent():
                self.body.splice(self.loads)
                self.body.splice(self.compute)
                self.body.splice(self.stores)
            self.body.writeline("}" * self._loop_brace_count)
            # Invalidate all CSE variables generated inside the replayed
            # loop so subsequent code re-emits their declarations at
            # function scope.
            self.cse.invalidate(OrderedSet())
            self._loop_template_saved = None
            self._loop_brace_count = 0
        else:
            self.body.splice(self.loads)
            self.body.splice(self.compute)
            self.body.splice(self.stores)

        # GPU.5: Persistent pointwise micro-batching.
        if getattr(self, "_persistent_mode", False) and not self.inside_reduction:
            persistent_body = self._emit_persistent_grid_stride_loop()
            if persistent_body is not None:
                self.body = IndentedBuffer()
                self.body.splice(persistent_body)

        # M22: Dead code elimination — strip CSE assignments whose LHS
        # is never referenced by any live computation or output store.
        # Reduces SPIR-V bloat from unused loads and computations that
        # survive upstream DeferredLine pruning.
        body_str = self.body.getvalue()
        optimized = self._eliminate_dead_code(body_str)
        if optimized != body_str:
            self.body = IndentedBuffer()
            self.body.splice(optimized)

        self.loads.clear()
        self.compute.clear()
        self.stores.clear()

        # T4.12: Ensure ALL range-tree entries are codegened before the
        # body is spliced.  Range-tree symbols used only in conditions
        # (compute buffer) never trigger codegen_indexing(), so they would
        # be missing declarations.  Force-codegen every entry to guarantee
        # every referenced axis variable (x0, x1, ...) has a declaration.
        # Skip roots that were already fully processed during multistage
        # reduction codegen to avoid emitting a second empty for-loop.
        for node in getattr(self, "range_tree_nodes", {}).values():
            if (
                hasattr(node, "codegen")
                and getattr(node, "root", None).prefix not in _processed_roots
            ):
                node.codegen()

    def _emit_helpers(self, code: IndentedBuffer) -> None:
        emit_helpers(
            code, self.headers, self.max_threadgroup_size, self.simd_group_size
        )

    def _is_fully_static(self) -> bool:
        """N+1.7: Return True if all sizevars and numels are sympy.Integer.

        When True, the push-constant struct can be skipped entirely — all
        dims/numels are emitted as ``static const uint`` module-scope
        declarations and slangc constant-folds everything.
        """
        idx_vars = self.active_range_trees()
        if not idx_vars:
            # No iteration ranges → no push constants needed anyway.
            return True
        # All range-tree numels must be known integers.
        if not all(isinstance(v.numel, sympy.Integer) for v in idx_vars):
            return False
        # All sizevars must also be known integers.
        for expr in self.args.sizevars:
            if not isinstance(expr, sympy.Integer):
                return False
        return True

    def codegen_kernel(self, name: Optional[str] = None) -> str:
        # P3.1/M9: Set ParameterBlock mode BEFORE codegen_body() so
        # pointwise.py load/store can prefix buffer accesses
        # (``args.in_ptr0[idx]`` vs ``in_ptr0[idx]``).
        self._use_parameter_block = config.parameter_block()

        self.codegen_body()
        code = IndentedBuffer()

        slot = 0
        inout_decls: list[tuple[str, str]] = []
        out_decls: list[tuple[str, str]] = []
        in_decls: list[tuple[str, str]] = []
        sv_decls: list[tuple[str, str]] = []

        for outer, inner in self.args.output_buffers.items():
            if outer in self.removed_buffers:
                continue
            if outer in self.args.inplace_buffers:
                continue
            if inner in self._packed16_bufs or inner in self._atomic_out_bufs:
                dtype_str = "uint"
            else:
                d = V.graph.get_dtype(outer)
                dtype_str = "float" if d == torch.bfloat16 else self.dtype_to_str(d)
            out_decls.append((dtype_str, inner))

        for outer, inner in self.args.input_buffers.items():
            if outer in self.args.inplace_buffers:
                continue
            dtype = V.graph.get_dtype(outer)
            if inner in self._packed16_bufs:
                dtype_str = "uint"
            else:
                dtype_str = (
                    "float" if dtype == torch.bfloat16 else self.dtype_to_str(dtype)
                )
            in_decls.append((dtype_str, inner))

        seen_inout: set[str] = set()
        for inplaced in self.args.inplace_buffers.values():
            if not isinstance(inplaced, InplacedBuffer):
                continue
            if inplaced.inner_name in seen_inout:
                continue
            seen_inout.add(inplaced.inner_name)
            outer = inplaced.other_names[-1]
            d = V.graph.get_dtype(outer)
            if inplaced.inner_name in self._packed16_bufs:
                dtype_str = "uint"
            else:
                dtype_str = "float" if d == torch.bfloat16 else self.dtype_to_str(d)
            inout_decls.append((dtype_str, inplaced.inner_name))

        idx_vars = self.active_range_trees()
        # M2 / M17 / N+1.7: split range-tree numels into static and dynamic.
        # Static numels (sympy.Integer) are emitted as ``static const uint``
        # at module scope so slangc constant-folds loop bounds and eliminates
        # dead branches.  Dynamic numels stay in the push-constant struct.
        # When every numel is static and there are no sizevars, the kernel
        # has no push constants at all (fully-static specialization).
        #
        # P1.1: When TORCH_VULKAN_DYNAMIC_SHAPES=0, all numels are forced
        # to the static path (int casts).  When enabled, sympy Symbols are
        # classified as dynamic and routed through push constants.
        #
        if config.dynamic_shapes():
            static_numel = [
                (f"{v.prefix}numel", int(v.numel))
                for v in idx_vars
                if isinstance(v.numel, sympy.Integer)
            ]
            dyn_numel = [
                ("uint", f"{v.prefix}numel")
                for v in idx_vars
                if not isinstance(v.numel, sympy.Integer)
            ]
        else:
            static_numel = [(f"{v.prefix}numel", int(v.numel)) for v in idx_vars]
            dyn_numel = []

        body_code = IndentedBuffer()
        layout_2d = self._persistent_2d_layout()
        # GAP-1.1: Store header-time layout_2d for codegen_iteration_ranges_entry.
        # _persistent_2d_layout depends on lazily-created range-tree entries
        # that don't exist yet.  If entries appear later during body codegen,
        # the indexing code would see a different layout_2d and emit mismatched assignments.
        self._header_layout_2d = layout_2d

        with body_code.indent():
            red_vars = [v for v in idx_vars if v.is_reduction]
            non_red = [v for v in idx_vars if not v.is_reduction]

            if self.inside_reduction and red_vars:
                axes = ["x", "y", "z"]
                for i, v in enumerate(non_red[:3]):
                    body_code.writeline(f"uint {v.name} = gid.{axes[i]};")
                if layout_2d is not None:
                    # M18.10: emit declarations for the reduction-axis root
                    # variables so downstream codegen in
                    # ``indexing.py:codegen_iteration_ranges_entry`` doesn't
                    # reference an undeclared ``r0_index`` symbol on the
                    # RHS of a derived entry expression (e.g. ``r0_1 =
                    # ((r0_index) % (16))`` from upstream Inductor's
                    # CSE-flattened index).
                    #
                    # Two persistent-2D shapes (per ``_persistent_2d_layout``):
                    #   (a) ``len(red_vars) == 2`` — two separate root trees;
                    #       each root maps directly to one axis. The first
                    #       root takes ``lid.y``, the second ``lid.x``.
                    #   (b) ``len(red_vars) == 1`` — one root tree with two
                    #       sub-entries (e.g. ``sum(dim=(0, 2))`` flattens
                    #       two reduction axes into one root). The root
                    #       index is the flat linearization
                    #       ``lid.y * tx + lid.x`` where ``tx`` is the
                    #       inner-axis size (the second element of
                    #       ``layout_2d``).
                    thread_y, thread_x = layout_2d
                    if len(red_vars) == 1:
                        body_code.writeline(
                            f"uint {red_vars[0].name} = "
                            f"lid.y * {thread_x}u + lid.x;"
                        )
                    else:
                        for i, v in enumerate(red_vars[:2]):
                            axis = "y" if i == 0 else "x"
                            body_code.writeline(
                                f"uint {v.name} = lid.{axis};"
                            )
                    # Tracker for indexing.py: don't re-declare these in
                    # the per-entry code path. The hoist set is the
                    # canonical "already declared at function scope" map.
                    if not hasattr(self, "_hoisted_vars"):
                        self._hoisted_vars = set()
                    for v in red_vars:
                        self._hoisted_vars.add(v.name)
                else:
                    for v in red_vars:
                        body_code.writeline(f"uint {v.name} = lid.x;")
            else:
                thr = self.max_threadgroup_size
                total_numel_static = None
                if all(isinstance(v.numel, sympy.Integer) for v in idx_vars):
                    total_numel_static = 1
                    for v in idx_vars:
                        total_numel_static *= int(v.numel)
                one_d = (
                    total_numel_static is not None and total_numel_static <= 65535 * thr
                )
                if len(idx_vars) == 1:
                    if one_d:
                        body_code.writeline(f"uint {idx_vars[0].name} = gtid.x;")
                    else:
                        body_code.writeline(
                            f"uint {idx_vars[0].name} = gtid.x + gid.y * (65535u * {thr}u);"
                        )
                else:
                    if one_d:
                        body_code.writeline("uint _vk_linear_orig = gtid.x;")
                    else:
                        body_code.writeline(
                            f"uint _vk_linear_orig = gtid.x + gid.y * (65535u * {thr}u);"
                        )
                    body_code.writeline("uint _vk_linear = _vk_linear_orig;")
                    self._pw_has_scan_or_linear = True
                    for i in range(len(idx_vars) - 1, -1, -1):
                        v = idx_vars[i]
                        if i == 0:
                            body_code.writeline(f"uint {v.name} = _vk_linear;")
                        else:
                            numel_str = self.sexpr(v.numel)
                            body_code.writeline(
                                f"uint {v.name} = _vk_linear % ({numel_str});"
                            )
                            body_code.writeline(
                                f"_vk_linear = _vk_linear / ({numel_str});"
                            )

            if idx_vars and not self.inside_reduction:
                static_total = None
                if all(isinstance(v.numel, sympy.Integer) for v in idx_vars):
                    static_total = 1
                    for v in idx_vars:
                        static_total *= int(v.numel)
                if (
                    static_total is None
                    or static_total % self.max_threadgroup_size != 0
                ):
                    total = self.sexpr(idx_vars[0].numel)
                    for v in idx_vars[1:]:
                        total = f"({total}) * ({self.sexpr(v.numel)})"
                    guard_var = (
                        idx_vars[0].name if len(idx_vars) == 1 else "_vk_linear_orig"
                    )
                    body_code.writeline(f"if ({guard_var} >= ({total})) return;")
                    self._pw_has_early_return = True

            body_code.splice(self.indexing_code)
            body_code.splice(self.body)

        # N+1.7: Fully-static specialization — when ALL dims and numels
        # are known integers, emit sizevars as ``static const uint``
        # module-scope constants instead of push-constant struct members.
        # This lets slangc constant-fold loop bounds and eliminate dead
        # branches. Gated behind TORCH_VULKAN_STATIC_SPECIALIZATION=1
        # (default ON). Even one dynamic dim keeps the PC path.
        _fully_static = config.static_specialization() and self._is_fully_static()
        if _fully_static:
            # Collect static const declarations for sizevars.
            _static_sv: list[tuple[str, int]] = []
            for expr, inner in self.args.sizevars.items():
                val = int(expr)
                _static_sv.append((inner, val))
            sv_decls = []  # no push-constant members for sizevars
        else:
            _static_sv = []
            for inner in self.args.sizevars.values():
                sv_decls.append(("uint", inner))

        # T.12 / CNN out_ptr0 — strip body lines that reference an
        # ``out_ptrN`` whose binding was elided. Multi-stage reduction
        # codegen calls ``codegen_body()`` once per stage; on the first
        # call ``DeferredLine`` stores get materialized into plain strings
        # via ``getvalue() → splice(optimized)``. Later stages add the
        # corresponding output buffer to ``removed_buffers`` (when its
        # next-stage consumer is fused inplace), but the body now holds
        # plain-string ``out_ptrN[idx] = ...`` writes that the binding
        # loop above skipped — slangc then errors with
        # ``undefined identifier 'out_ptrN'``.
        #
        # Filter at the final-emission boundary: any ``out_ptr<digits>``
        # token not present in the declared in/out/inout binding set is a
        # dangling reference and the line is dropped.
        declared_inners: set[str] = set()
        for _, inner in in_decls:
            declared_inners.add(inner)
        for _, inner in out_decls:
            declared_inners.add(inner)
        for _, inner in inout_decls:
            declared_inners.add(inner)
        if body_code._lines:
            import re as _re

            _ptr_pat = _re.compile(r"\b(out_ptr\d+|in_ptr\d+|in_out_ptr\d+)\b")
            kept_body: list[Any] = []
            for line in body_code._lines:
                line_str = (
                    line.line
                    if hasattr(line, "line") and not isinstance(line, str)
                    else str(line)
                )
                tokens = set(_ptr_pat.findall(line_str))
                if tokens and any(t not in declared_inners for t in tokens):
                    continue
                kept_body.append(line)
            body_code._lines = kept_body

        body_str = body_code.getvalue()
        if self._vec4_pw_eligible(body_str, in_decls, out_decls, inout_decls):
            new_body = self._vec4_pw_rewrite(body_str, in_decls, out_decls, inout_decls)
            if new_body is not None:
                body_code = new_body
                self._vec4_pw_active = True

        # M11.3: Register-tile pointwise — unroll scalar body 2-4×.
        # Runs after the vec4 check; only applies when vec4/packed16
        # did NOT activate (register tiling is the fallback).
        _rt_size = config.register_tile()
        if (
            _rt_size > 0
            and not getattr(self, "_vec4_pw_active", False)
            and not getattr(self, "_packed16", False)
            and not getattr(self, "_persistent_mode", False)
            and not self.inside_reduction
            and not self.multistage_reduction_entry
        ):
            non_red = [t for t in self.range_trees if not t.is_reduction]
            if (
                len(non_red) == 1
                and isinstance(non_red[0].numel, sympy.Integer)
                and int(non_red[0].numel) % (self.max_threadgroup_size * _rt_size) == 0
            ):
                tiled = self._apply_register_tile(body_code.getvalue(), _rt_size)
                if tiled is not None:
                    body_code = IndentedBuffer()
                    body_code.splice(tiled)
                    self._register_tile_size = _rt_size

        # DR.6: Vec4/packed16 vectorization audit — count loads/stores
        # and compute hit rates after the rewrite pass.  Gated by
        # TORCH_VULKAN_VEC4_AUDIT=1 (default: off).
        if config.vec4_audit_enabled():
            from ..heuristics.vectorization_audit import audit_kernel

            _post_rewrite_body = body_code.getvalue()
            _kname = name or getattr(self, "_kernel_name", "")
            audit_kernel(
                _post_rewrite_body,
                in_decls,
                out_decls,
                inout_decls,
                getattr(self, "_vec4_pw_bufs", set()),
                getattr(self, "_packed16_bufs", set()),
                getattr(self, "_vec4_pw_active", False),
                getattr(self, "_packed16_vw_active", False),
                _kname,
            )

        def _vec4_dtype(inner: str, dtype_str: str) -> str:
            return "float4" if inner in self._vec4_pw_bufs else dtype_str

        def _binding_dtype(inner: str, dtype_str: str) -> str:
            base = _vec4_dtype(inner, dtype_str)
            if base == "int64_t":
                return "uint2"
            return base

        # P3.1/M9: When ParameterBlock is enabled, collect all buffer
        # declarations into a struct and emit ParameterBlock<KernelArgs>.
        # Slang auto-assigns binding indices in struct-field order.
        if self._use_parameter_block:
            code.writeline("struct KernelArgs {")
            with code.indent():
                for dtype_str, inner in in_decls:
                    code.writeline(
                        f"StructuredBuffer<{_binding_dtype(inner, dtype_str)}> {inner};"
                    )
                for dtype_str, inner in out_decls:
                    code.writeline(
                        f"RWStructuredBuffer<{_binding_dtype(inner, dtype_str)}> {inner};"
                    )
                for dtype_str, inner in inout_decls:
                    code.writeline(
                        f"RWStructuredBuffer<{_binding_dtype(inner, dtype_str)}> {inner};"
                    )
                for ws_arg in self.args.workspace_args:
                    ws_dtype_str = self.dtype_to_str(ws_arg.dtype)
                    code.writeline(
                        f"RWStructuredBuffer<{ws_dtype_str}> {ws_arg.inner_name};"
                    )
            code.writeline("};")
            # CG.M14: ParameterBlock<KernelArgs> groups all buffer bindings
            # into a single descriptor table. Slang auto-assigns binding
            # indices in struct-field order.
            #
            # M21.3.01: slangc 2026.7.1 unconditionally assigns
            # ``ParameterBlock<KernelArgs>`` to ``DescriptorSet=1`` even
            # when ``[[vk::constant_id]]`` is absent — the C++ pipeline
            # layout only declares Set 0, which yields
            # ``VUID-VkComputePipelineCreateInfo-layout-07988`` under the
            # validation layer and unbound / stale descriptor reads on
            # RADV in production (EAGER.1.b). Explicit
            # ``[[vk::binding(0, 0)]]`` forces Set 0 in the SPIR-V.
            code.writeline(
                "// M21.3.01: explicit Set 0 binding "
                "(slangc 2026.7.1 defaults to Set 1)"
            )
            code.writeline(
                "[[vk::binding(0, 0)]] ParameterBlock<KernelArgs> args;"
            )
            slot = 0  # unused; keep for compatibility
        else:
            for dtype_str, inner in in_decls:
                code.writeline(
                    f"[[vk::binding({slot})]] StructuredBuffer<{_binding_dtype(inner, dtype_str)}> {inner};"
                )
                slot += 1
            for dtype_str, inner in out_decls:
                code.writeline(
                    f"[[vk::binding({slot})]] RWStructuredBuffer<{_binding_dtype(inner, dtype_str)}> {inner};"
                )
                slot += 1
            for dtype_str, inner in inout_decls:
                code.writeline(
                    f"[[vk::binding({slot})]] RWStructuredBuffer<{_binding_dtype(inner, dtype_str)}> {inner};"
                )
                slot += 1

        pc_fields = sv_decls + dyn_numel

        # M17: When TORCH_VULKAN_SPEC_CONSTANTS=1 and every range-tree
        # numel is static (no dynamic numels, no sizevars), emit
        # ``[[vk::constant_id(N)]] const uint xnumel;`` instead of
        # push-constant struct members or ``static const uint``.
        # Slangc constant-folds loop bounds and eliminates dead branches
        # at SPIR-V emission time.
        #
        # CG.M14: Disable spec constants when ParameterBlock is active.
        # Slang places ParameterBlock in descriptor set 1 when
        # [[vk::constant_id]] is present anywhere in the module, but the
        # Vulkan pipeline layout expects all storage buffers at set 0.
        # Using ``static const uint`` instead avoids the set mismatch.
        use_spec_constants = (
            config.spec_constants()
            and not self._use_parameter_block
            and not dyn_numel
            and not sv_decls
        )

        if pc_fields:
            code.writeline("struct PC {")
            with code.indent():
                for t, n in pc_fields:
                    code.writeline(f"{t} {n};")
            code.writeline("};")
            code.writeline("[[vk::push_constant]] PC pc;")

        code.splice(self.module_scope_decls)

        # P3.1/M9: When ParameterBlock is enabled, workspace args are
        # already inside the KernelArgs struct in the binding block above.
        # Only emit separate bindings on the legacy path.
        if not self._use_parameter_block:
            for ws_arg in self.args.workspace_args:
                ws_dtype_str = self.dtype_to_str(ws_arg.dtype)
                code.writeline(
                    f"[[vk::binding({slot})]] RWStructuredBuffer<{ws_dtype_str}> {ws_arg.inner_name};"
                )
                slot += 1

        # ── Descriptor indexing guard (N+1.5.c) ────────────────
        # When VK_EXT_descriptor_indexing is enabled the binding cap is
        # lifted; when it is off we warn if the kernel exceeds the pre-
        # indexing default of 16 storage-buffer bindings per stage.
        _total_bindings = (
            len(in_decls)
            + len(out_decls)
            + len(inout_decls)
            + len(self.args.workspace_args)
        )
        if _total_bindings > 16:
            from ..config import descriptor_indexing_enabled

            if not descriptor_indexing_enabled():
                import warnings

                warnings.warn(
                    f"Kernel uses {_total_bindings} buffer bindings, "
                    f"exceeding the pre-descriptor-indexing limit of 16. "
                    f"Descriptor indexing is disabled (TORCH_VULKAN_DESCRIPTOR_INDEXING=0). "
                    f"The kernel may fail at Vulkan pipeline creation.",
                    stacklevel=2,
                )

        self._emit_helpers(code)

        # Module-scope static specialization for fully-known range-tree
        # numels.  Emitted AFTER helpers so any helper that references a
        # <prefix>numel symbol (e.g. mix-order reduction store) resolves
        # against the constant.  Slang folds these at SPIR-V emission time;
        # any unrolled loop bound or comparison against the value collapses
        # to a literal.
        # Emit static const declarations for sizevars when the kernel
        # is fully-static (N+1.7). These go AFTER helpers (same reasoning
        # as static_numel — helpers may reference these symbols).
        for name_, value_ in _static_sv:
            code.writeline(f"static const uint {name_} = {value_};")

        if use_spec_constants:
            for i, (name_, _value_) in enumerate(static_numel):
                code.writeline(f"[[vk::constant_id({i})]] const uint {name_};")
        else:
            for name_, value_ in static_numel:
                code.writeline(f"static const uint {name_} = {value_};")

        if layout_2d is not None:
            ty, tx = layout_2d
            code.writeline(f'[shader("compute")] [numthreads({tx}, {ty}, 1)]')
        else:
            thread_count = self.max_threadgroup_size
            # M-PERF.3: Reflection-driven numthreads override at emit
            # time.  When SPIR-V reflection has reported >128 actual
            # VGPRs for a prior compile of this config, force a 64-
            # thread WG so RDNA1 keeps 2-4 waves/CU of headroom
            # (256 VGPRs/CU ÷ 128 VGPRs/lane = 2 lanes/CU at saturation;
            # a 256-thread WG would force 1 wave/CU and tank occupancy).
            # The 64-thread floor matches wave64 = 1 wave/WG, giving the
            # scheduler the most flexibility to interleave waves.
            try:
                _actual_vgprs = self._get_actual_vgprs()
            except Exception:
                _actual_vgprs = None
            if (
                _actual_vgprs is not None
                and _actual_vgprs > 128
                and thread_count > 64
            ):
                thread_count = 64
            code.writeline(f'[shader("compute")] [numthreads({thread_count}, 1, 1)]')
        code.writeline(
            "void computeMain(uint3 gtid : SV_DispatchThreadID, "
            "uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID) {"
        )
        with code.indent():
            if _fully_static:
                # N+1.7: Fully-static kernel — all sizevars are already
                # module-scope ``static const uint``. No PC struct exists,
                # so no local aliases needed.
                pass
            elif use_spec_constants:
                # Spec constants are already module-scope globals — no
                # local alias (``uint n = pc.n;``) needed.
                pass
            else:
                for _, n in pc_fields:
                    code.writeline(f"uint {n} = pc.{n};")
        code.splice(body_code)
        code.writeline("}")
        return code.getvalue()

    def call_kernel(
        self, name: str, node: Any = None, deallocate_ws: bool = True
    ) -> None:
        wrapper = V.graph.wrapper_code
        for v in self.args.sizevars:
            wrapper.ensure_size_computed(v)

        # Emit args in binding order: all inputs first, then all outputs.
        ordered_args: list[str] = []
        ordered_args.extend(
            outer
            for outer, inner in self.args.input_buffers.items()
            if outer not in self.removed_buffers
        )
        ordered_args.extend(
            outer
            for outer, inner in self.args.output_buffers.items()
            if outer not in self.args.inplace_buffers
            and outer not in self.removed_buffers
        )
        ordered_args.extend(
            outer
            for outer, inner in self.args.inplace_buffers.items()
            if isinstance(inner, InplacedBuffer)
        )

        # Substitute buffer aliases (the Inductor memory planner may alias
        # buffers via ``buf1 = reinterpret_tensor(div); del div`` before
        # the kernel call). Resolve transitively so a chain
        # ``buf9 → buf10 → buf11`` (where buf10 is itself reused into
        # buf11) collapses to the final live name. A naive one-step
        # substitution leaves the intermediate buf10 in args, which
        # then references a deleted variable at runtime.
        freed = getattr(wrapper, "freed", set())
        reuses = getattr(wrapper, "reuses", {})
        if freed:
            old_to_new = {}
            for new_name, old_name in reuses.items():
                old_to_new[old_name] = new_name

            def _resolve(name):
                # Walk the alias chain, guarding against cycles.
                seen = set()
                while name in old_to_new and name in freed and name not in seen:
                    seen.add(name)
                    name = old_to_new[name]
                return name

            ordered_args = [_resolve(a) for a in ordered_args]

        # N+1.7: For fully-static kernels, sizevars are emitted as
        # ``static const uint`` module-scope declarations — no push
        # constants needed. Skip appending them to ordered_args so
        # the wrapper passes n_pc=0.
        if not (config.static_specialization() and self._is_fully_static()):
            for v in self.args.sizevars:
                ordered_args.append(str(v))

        # P1.1: Only pass dynamic numels as push constants when
        # dynamic shapes are enabled.  When the gate is off all numels
        # are treated as static and no extra push constants are needed.
        if config.dynamic_shapes():
            for tree in self.range_trees:
                if isinstance(tree.numel, (sympy.Integer, int)):
                    continue
                if not isinstance(tree.numel, sympy.Symbol):
                    continue
                if tree.is_reduction and not self.inside_reduction:
                    continue
                ordered_args.append(str(tree.numel))

        for ws_arg in self.args.workspace_args:
            wrapper.generate_workspace_allocation(ws_arg)

        red = [v for v in self.active_range_trees() if v.is_reduction]
        non_red = [v for v in self.active_range_trees() if not v.is_reduction]
        layout_2d = self._persistent_2d_layout()

        # Compute workgroup counts.
        # P1.1: When dynamic shapes are enabled, the dispatch grid is
        # computed at runtime from push-constant values.  The total numel
        # expression references the sizevar names (e.g. ks27) and is
        # divided by the threadgroup size to yield the workgroup count.
        thr = self.max_threadgroup_size
        if config.dynamic_shapes() and non_red:
            # Build a single Python expression for the total numel across
            # all non-reduction dimensions.  Dynamic numels reference the
            # sizevar name (rendered via sexpr); static numels are literals.
            total_numel_expr = self.sexpr(non_red[0].numel)
            for v in non_red[1:]:
                total_numel_expr = f"({total_numel_expr}) * ({self.sexpr(v.numel)})"

            if red:
                # D.2.a — For reductions, one workgroup per output element
                # (no division by threadgroup_size).  The threads within
                # each WG collaborate to reduce the reduction dimension.
                from .symbolic import MAX_COMPUTE_WG_X

                wg_x = f"min(({total_numel_expr}), {MAX_COMPUTE_WG_X})"
                wg_y = (
                    f"((({total_numel_expr}) + {MAX_COMPUTE_WG_X - 1})"
                    f" // {MAX_COMPUTE_WG_X})"
                )
            else:
                # Pointwise: ceil(total_elements / threadgroup_size)
                from .symbolic import dynamic_wg_counts

                wg_x, wg_y = dynamic_wg_counts(total_numel_expr, thr)
            wg_z = "1"
        else:
            if non_red:
                wg_x_str = self.sexpr(non_red[0].numel)
                for v in non_red[1:]:
                    wg_x_str = f"({wg_x_str}) * ({self.sexpr(v.numel)})"
            else:
                wg_x_str = "1"

            if red:
                wg_x_str = f"({wg_x_str})"

            wg_x = wg_x_str
            wg_y = "1"
            wg_z = "1"

        ordered_args.append(wg_x)
        ordered_args.append(wg_y)
        ordered_args.append(wg_z)

        # M11.3: Register-tile grid adjustment — divide innermost grid
        # axis by tile_size since each thread processes multiple elements.
        _tile = getattr(self, "_register_tile_size", 0)
        if _tile > 0 and not red and non_red:
            # wg_x currently holds the total numel (or per-dim product).
            # The C++ dispatch layer divides by WG to get the actual grid,
            # so we divide by tile_size here to get: numel / (WG * tile).
            ordered_args[-3] = f"(({ordered_args[-3]}) // {_tile})"

        wrapper.generate_kernel_call(
            name,
            ordered_args,
            device=torch.device("vulkan"),
            triton=False,
            arg_types=None,
        )

        if deallocate_ws:
            self.deallocate_workspaces()
