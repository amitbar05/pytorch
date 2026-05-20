"""Vulkan combo kernel — merges multiple pointwise kernels into one Slang shader.

Reduces dispatch overhead (~0.03-0.07 ms per dispatch) by combining independent
pointwise subkernels into a single compute shader with a gtid-based dispatch
routing if-ladder.

Each subkernel emits its body referencing local identifiers (`xindex`, `x0`,
`in_ptr0`, `out_ptr1`, ...) that collide across subkernels. We resolve this by:

  1. Globally renaming each subkernel's buffer references via a deduped
     outer→global-name map (same outer buffer reuses the same binding, distinct
     outers get unique slot names).
  2. Wrapping each subkernel body in its own `{}` scope and seeding the index
     variables (`xindex`, `gid.x`) from the combo's `_vk_gtid_local`.
  3. A lightweight tokenizer distinguishes declarations from references so
     renaming is precise — a variable named `xindex_sub0` in a subkernel body
     is left alone instead of being corrupted by the old regex approach.

Implementation split across ``combo_kernel/`` subpackage:
- ``body_rewriter.py`` — tokenizer, identifier classifier, body rewriting
- ``binding_map.py`` — global binding-map construction
- ``grid_builder.py`` — grid dimensions and numthreads computation
"""

from __future__ import annotations

from typing import Optional

import sympy
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.virtualized import V

from .combo_kernel.binding_map import build_global_binding_map
from .combo_kernel.body_rewriter import (
    _KEYWORDS,
    _NEVER_RENAME,
    _T_COMMENT,
    _T_IDENT,
    _T_KEYWORD,
    _T_NUMBER,
    _T_OPERATOR,
    _T_PUNCT,
    _T_SPACE,
    _T_STRING,
    _TYPE_KEYWORDS,
    _is_buffer_name,
    _is_local_to_rename,
    _rewrite_body,
    _Token,
    _tokenize,
)
from .combo_kernel.grid_builder import (
    _wg_count,
    compute_grid_dims,
    compute_max_threadgroup_size,
    compute_max_workgroups,
)
from .kernel import VulkanKernel
from .slang_helpers import emit_helpers

# Re-export all public symbols from submodules so existing imports continue
# to work (e.g. ``from torch_vulkan.inductor.vulkan_combo_kernel import
# VulkanComboKernel``).
__all__ = [
    "VulkanComboKernel",
    "_rewrite_body",
    "_tokenize",
    "_is_buffer_name",
    "_is_local_to_rename",
    "_Token",
    "_NEVER_RENAME",
    "_TYPE_KEYWORDS",
    "_KEYWORDS",
    "build_global_binding_map",
    "compute_grid_dims",
    "compute_max_threadgroup_size",
    "compute_max_workgroups",
]


class VulkanComboKernel:
    """Merges multiple VulkanKernel instances into a single Slang shader."""

    # M16: class-level debug flag. When True, the rewriter verifies renamed
    # variables don't collide with any existing declaration in any scope.
    # Set via environment variable ``TORCH_VULKAN_COMBO_DEBUG_ASSERT=1``.
    debug_assert_rename: bool = False

    def __init__(self) -> None:
        import os

        self.subkernels: list[tuple[VulkanKernel, int]] = []
        # Filled in by `codegen_kernel` so the wrapper define-kernel pass can
        # see the merged (n_buffers, n_outputs) — they don't match any single
        # subkernel's args.
        self.n_buffers: int = 0
        self.n_outputs: int = 0
        self._debug_rename = (
            VulkanComboKernel.debug_assert_rename
            or os.environ.get("TORCH_VULKAN_COMBO_DEBUG_ASSERT", "0") == "1"
        )
        self.n_inputs: int = 0

    def share_cse_from(
        self, source_kernel: VulkanKernel, target_kernel: VulkanKernel
    ) -> None:
        """Share only the CSE counter (not cache) from source to target.

        Each subkernel keeps its own CSE cache entries (independent variable
        declarations), but the shared counter prevents tmpN name collisions
        across subkernels.  This avoids cross-subkernel variable references
        that would require scope hoisting — each subkernel is self-contained.

        Must be called BEFORE target_kernel's process_kernel/body generation.
        """
        target_kernel.cse.iter_buffer_ids = source_kernel.cse.iter_buffer_ids

    def create_sub_kernel(self, kernel: VulkanKernel, numel: int) -> VulkanKernel:
        self.subkernels.append((kernel, numel))
        return kernel

    def _rename_subkernel_locals(self, body: str, subkernel_idx: int) -> str:
        """Rename local variables in a subkernel body to avoid collisions.

        Uses a simple tokenizer that distinguishes:
        - Declarations: ``float tmp0 = ...`` → ``float tmp0_sub{idx} = ...``
        - References: ``tmp0`` → ``tmp0_sub{idx}``
        - Already-renamed: ``tmp0_sub3`` → leave as-is (don't double-rename)
        - String literals: ``"tmp0"`` → not renamed (preserved verbatim)
        - Slang keywords and built-in types: never renamed

        This is a convenience wrapper around the module-level ``_rewrite_body``
        that only handles local renaming (no buffer name mapping, no cross-
        subkernel scope tracking).  For full combo-kernel body rewriting with
        buffer name mapping and cross-decl tracking, use ``_rewrite_body``
        directly.

        Args:
            body: The subkernel body source code.
            subkernel_idx: Integer index of the subkernel (for suffix generation).

        Returns:
            The renamed body string.
        """
        # Use _rewrite_body with an empty name_map (no buffer renames) and
        # no cross_decls (isolated scope).  The empty name_map ensures only
        # local-to-rename identifiers get the _sub{idx} suffix.
        rewritten, _decls = _rewrite_body(
            body,
            {},
            subkernel_idx,
            cross_decls=None,
            debug_assert=self._debug_rename,
        )
        return rewritten

    @staticmethod
    def _coalesce_orphan_pointwise(
        nodes: "list[SchedulerNode]",
    ) -> "list[SchedulerNode]":
        """Merge independent orphan pointwise nodes into combo kernels.

        Orphan pointwise ops are ``SchedulerNode`` instances that:
        - Are pointwise (not reductions, not templates)
        - Are not already in a ``ForeachKernelSchedulerNode``
        - Have no data dependencies between them (same topological level)
        - Can share the same workgroup grid (same numel)
        - Have compatible threadgroup sizes

        Groups of compatible orphans are wrapped into
        ``ForeachKernelSchedulerNode`` instances, which the downstream
        ``codegen_combo_kernel`` path merges into a single Slang shader.

        Args:
            nodes: Ordered list of ``SchedulerNode`` objects (one per dispatch).

        Returns:
            Updated node list with orphan pointwise groups coalesced into
            ``ForeachKernelSchedulerNode`` instances.
        """
        # Import here to avoid circular imports at module load time.
        from torch._inductor.scheduler import (
            BaseSchedulerNode,
            ForeachKernelSchedulerNode,
        )

        # --- Phase 1: identify orphan pointwise nodes ---
        orphans: list[BaseSchedulerNode] = []
        for node in nodes:
            if isinstance(node, ForeachKernelSchedulerNode):
                continue
            if node.is_template():
                continue
            if node.is_reduction():
                continue
            if node.is_extern():
                continue
            orphans.append(node)

        if len(orphans) < 2:
            return nodes

        # --- Phase 2: group orphans by compatible grid size ---
        # Two pointwise ops can share a combo kernel when they have the same
        # numel (same workgroup grid).  Threadgroup size differences are
        # handled by the combo kernel codegen which uses the maximum TGS
        # across all subkernels.
        from collections import defaultdict

        # Key: numel string.  Pointwise nodes with the same numel are
        # co-schedulable in a single combo kernel dispatch.
        buckets: dict[str, list[BaseSchedulerNode]] = defaultdict(list)
        for node in orphans:
            _, (numel, rnumel) = node.group
            numel_str = str(numel)
            buckets[numel_str].append(node)

        # --- Phase 3: build ForeachKernelSchedulerNode for each group ---
        if not buckets:
            return nodes

        # Track which orphans have been consumed (assigned to a group).
        consumed: set[int] = set()  # indices into the orphans list
        orphan_index: dict[int, int] = {id(n): i for i, n in enumerate(orphans)}
        result: list[BaseSchedulerNode] = []

        for node in nodes:
            oid = id(node)
            if oid not in orphan_index:
                # Not an orphan — pass through unchanged.
                result.append(node)
                continue
            oi = orphan_index[oid]
            if oi in consumed:
                # Already consumed by a prior bucket emission.
                continue

            key = str(node.group[1][0])
            bucket = buckets.get(key)
            if bucket is None or len(bucket) < 2:
                result.append(node)
                consumed.add(oi)
                if bucket is not None:
                    buckets.pop(key, None)
                continue

            # Create a ForeachKernelSchedulerNode for this bucket.
            # Use the first node's scheduler.
            try:
                group_snode = ForeachKernelSchedulerNode(
                    bucket[0].scheduler,
                    list(bucket),
                    use_custom_partition_algo=True,
                    enable_autotune=False,
                )
                result.append(group_snode)
            except Exception:
                # If grouping fails (e.g., validation rejects it), keep
                # nodes as individual dispatches.
                result.extend(bucket)
            # Mark all bucket members as consumed and remove the bucket.
            for bn in bucket:
                bi = orphan_index.get(id(bn))
                if bi is not None:
                    consumed.add(bi)
            buckets.pop(key, None)

        return result

    def _build_global_binding_map(
        self,
    ) -> tuple[
        list[
            tuple[str, str, str]
        ],  # in_decls: [(dtype_str, global_name, outer)] read-only
        list[
            tuple[str, str, str]
        ],  # rw_decls: [(dtype_str, global_name, outer)] read-write (inplace + output)
        list[dict[str, str]],  # per-subkernel inner->global rename map
    ]:
        """Build the global binding map across all subkernels.

        Delegates to ``combo_kernel.binding_map.build_global_binding_map``.
        """
        return build_global_binding_map(self.subkernels)

    def codegen_kernel(self) -> str:
        code = IndentedBuffer()
        in_decls, rw_decls, per_sub_maps = build_global_binding_map(self.subkernels)
        self.n_inputs = len(in_decls)
        self.n_outputs = len(rw_decls)
        self.n_buffers = self.n_inputs + self.n_outputs

        # CG.M14: Use ParameterBlock<KernelArgs> for clean binding emission
        # (matching the single-kernel codegen path in kernel/header.py).
        # Slang auto-assigns binding indices in struct field declaration order.
        from . import config as _cfg

        if _cfg.parameter_block():
            code.writeline("struct KernelArgs {")
            with code.indent():
                for dtype_str, name, _outer in in_decls:
                    code.writeline(f"StructuredBuffer<{dtype_str}> {name};")
                for dtype_str, name, _outer in rw_decls:
                    code.writeline(f"RWStructuredBuffer<{dtype_str}> {name};")
            code.writeline("};")
            # Blocker E: pin ParameterBlock to Set 0 — slangc 2026.7.1
            # otherwise places it on Set 1, which doesn't match the C++
            # pipeline layout (Set 0 only).
            code.writeline("[[vk::binding(0, 0)]] ParameterBlock<KernelArgs> args;")
        else:
            slot = 0
            # Blocker E: explicit Set 0 (`, 0`) on every binding.
            for dtype_str, name, _outer in in_decls:
                code.writeline(
                    f"[[vk::binding({slot}, 0)]] StructuredBuffer<{dtype_str}> {name};"
                )
                slot += 1
            for dtype_str, name, _outer in rw_decls:
                code.writeline(
                    f"[[vk::binding({slot}, 0)]] RWStructuredBuffer<{dtype_str}> {name};"
                )
                slot += 1

        max_tgs = compute_max_threadgroup_size(self.subkernels)

        # Emit module-scope helpers (imports + inline) as the union of every
        # subkernel's required headers. Without this, bodies that reference
        # `wg_reduce_wave<OpSum>(...)` or other reduction helpers fail slangc
        # with "undefined identifier". `emit_helpers` routes known headers
        # through `import reduction;` / `import helpers;` / `import atomics;`
        # and falls back to inline emission for anything else, matching the
        # single-kernel codegen path in `kernel/header.py:HeaderMixin._emit_helpers`.
        union_headers: set[str] = set()
        simd_group_size = 64
        for kernel, _ in self.subkernels:
            union_headers |= set(getattr(kernel, "headers", set()))
            simd_group_size = max(
                simd_group_size, getattr(kernel, "simd_group_size", 64)
            )
        if union_headers:
            emit_helpers(code, union_headers, max_tgs, simd_group_size)

        code.writeline(f'[shader("compute")] [numthreads({max_tgs}, 1, 1)]')
        code.writeline(
            "void computeMain(uint3 gtid : SV_DispatchThreadID, "
            "uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID) {"
        )
        with code.indent():
            # TRAIN.6-F1: Wave-uniform dispatch via multi-dimensional grid.
            # gid.y selects which subkernel this workgroup runs, so ALL threads
            # in a workgroup execute the SAME subkernel body. This preserves
            # wave uniformity for reduction intrinsics (WaveActiveSum, etc.).
            # gid.x is the subkernel's own workgroup ID — no remapping needed.
            code.writeline("uint _vk_subkernel = gid.y;")
            # TR.16.A (2026-05-09): gtid.x = SV_DispatchThreadID.x = gid.x *
            # numthreads.x + lid.x ALREADY. The previous form
            # `gtid.x + gid.x * max_tgs` double-counted gid.x, so workgroups
            # with gid.x >= 1 wrote past their bounds-check `< numel` and left
            # output slots uninitialized (= buffer-pool garbage). Repro:
            # Conv+BN(eval) compiled produced max diff 2.72 vs 3.6e-7 with this
            # fix.
            code.writeline("uint _vk_gtid = gtid.x;")

            cross_decls: list[dict[str, str]] = []
            for idx, (kernel, numel) in enumerate(self.subkernels):
                # SIMD codegen reaches V.kernel.codegen_iteration_ranges_entry;
                # without re-pushing the handler here it's NullKernelHandler
                # at the outer scope and AttributeError fires.
                with V.set_kernel_handler(kernel):
                    kernel.codegen_body()
                # `kernel.body` holds loads/compute/stores. The single-kernel
                # path additionally emits per-range-tree index assignments
                # (header.py lines 137-181) and splices `kernel.indexing_code`
                # before the body. We must do the same for any subkernel whose
                # body references those index symbols (`r0_1`, `r0_index`,
                # `x1`, `x3`, …) — otherwise slangc fails with "undefined
                # identifier". The rewriter sees the prepended declarations
                # first and renames them to `_sub{idx}` consistently with the
                # references that follow.
                indexing_src = kernel.indexing_code.getvalue()
                body_src = kernel.body.getvalue().strip()

                # TRAIN.6-F1: Reduction subkernels use gid.x directly for
                # workgroup indexing (one workgroup per output element).
                # Pointwise subkernels use flat _vk_gtid with TGS threads.
                inside_reduction = getattr(kernel, "inside_reduction", False)
                if inside_reduction:
                    cond = (
                        f"if (_vk_subkernel == {idx}u && gid.x < {numel}u) {{"
                        if idx == 0
                        else f"}} else if (_vk_subkernel == {idx}u && gid.x < {numel}u) {{"
                    )
                else:
                    cond = (
                        f"if (_vk_subkernel == {idx}u && _vk_gtid < {numel}u) {{"
                        if idx == 0
                        else f"}} else if (_vk_subkernel == {idx}u && _vk_gtid < {numel}u) {{"
                    )
                code.writeline(cond)

                # Build seed indexing declarations.  We need these even
                # when body_src is empty so cross_decls captures the
                # index variable names for later subkernels.
                seed = IndentedBuffer()
                inside_reduction = getattr(kernel, "inside_reduction", False)
                if inside_reduction:
                    seed.writeline(f"uint xindex = gid.x;")
                else:
                    trees = list(kernel.active_range_trees())
                    non_red_trees = [t for t in trees if not t.is_reduction]
                    if len(non_red_trees) > 1:
                        seed.writeline(
                            f"uint _vk_linear_orig = _vk_gtid_local_sub{idx};"
                        )
                        seed.writeline("uint _vk_linear = _vk_linear_orig;")
                        for i in range(len(non_red_trees) - 1, -1, -1):
                            v = non_red_trees[i]
                            if i == 0:
                                seed.writeline(f"uint {v.name} = _vk_linear;")
                            else:
                                numel_str = kernel.sexpr(v.numel)
                                seed.writeline(
                                    f"uint {v.name} = _vk_linear % ({numel_str});"
                                )
                                seed.writeline(
                                    f"_vk_linear = _vk_linear / ({numel_str});"
                                )
                    else:
                        seed.writeline(f"uint xindex = _vk_gtid_local_sub{idx};")
                        if "x0" not in indexing_src:
                            seed.writeline("uint x0 = xindex;")
                if inside_reduction:
                    try:
                        trees = list(kernel.active_range_trees())
                    except Exception:
                        trees = list(getattr(kernel, "range_trees", []))
                    non_red = [t for t in trees if not t.is_reduction]
                    red = [t for t in trees if t.is_reduction]
                    # T5.10: The combo kernel dispatch uses gid.y as
                    # _vk_subkernel (subkernel selector) and gid.z is
                    # always 1.  Mapping non-red axes to gid.{y,z}
                    # would give each axis the wrong value.  Instead
                    # decompose flat gid.x (= xindex) into
                    # multi-dimensional non-red indices via arithmetic,
                    # matching the pointwise path's approach for multi-
                    # axis gtid decomposition.
                    if len(non_red) > 1:
                        # Multi non-red axes: linearize gid.x and
                        # decompose into per-axis indices.
                        seed.writeline("uint _vk_rlinear = xindex;")
                        for i in range(len(non_red) - 1, -1, -1):
                            t = non_red[i]
                            skip = (
                                t.name in ("xindex", "x0")
                                or f"uint {t.name} " in indexing_src
                            )
                            if skip:
                                if i == 0:
                                    # x0 was set to xindex=flat; reassign
                                    # to the decomposed first-axis value.
                                    if "x0" not in indexing_src:
                                        seed.writeline("x0 = _vk_rlinear;")
                            else:
                                if i == 0:
                                    seed.writeline(f"uint {t.name} = _vk_rlinear;")
                                else:
                                    numel_str = kernel.sexpr(t.numel)
                                    seed.writeline(
                                        f"uint {t.name} = _vk_rlinear % ({numel_str});"
                                    )
                                    seed.writeline(
                                        f"_vk_rlinear = _vk_rlinear / ({numel_str});"
                                    )
                    else:
                        # Single non-red axis: xindex = gid.x IS the
                        # axis value (numel from _wg_count equals
                        # the output-element count = that axis).
                        if "x0" not in indexing_src:
                            seed.writeline("uint x0 = xindex;")
                    for t in red:
                        if f"uint {t.name} " not in indexing_src:
                            seed.writeline(f"uint {t.name} = lid.x;")

                # Always process merged source to collect declarations,
                # even when body_src is empty.
                merged = seed.getvalue() + indexing_src + body_src
                rewritten, sub_scope = _rewrite_body(
                    merged,
                    per_sub_maps[idx],
                    idx,
                    cross_decls[:idx] if idx > 0 else None,
                    debug_assert=self._debug_rename,
                )
                cross_decls.append(sub_scope)

                if body_src:
                    with code.indent():
                        if not inside_reduction:
                            code.writeline(f"uint _vk_gtid_local_sub{idx} = _vk_gtid;")
                        # TRAIN.6-F1: Reduction subkernels may share a shader
                        # with pointwise subkernels that need a larger TGS.
                        # Guard lanes beyond the reduction's own TGS to prevent
                        # out-of-bounds memory access via lid.x.
                        red_tgs = getattr(kernel, "max_threadgroup_size", 256)
                        needs_lane_guard = inside_reduction and red_tgs < max_tgs
                        if needs_lane_guard:
                            code.writeline(f"if (lid.x < {red_tgs}u) {{")
                            code.do_indent()
                        for line in rewritten.splitlines():
                            if line.strip():
                                code.writeline(line)
                            else:
                                code.writeline("")
                        if needs_lane_guard:
                            code.do_unindent()
                            code.writeline("}")

            if self.subkernels:
                code.writeline("}")

        code.writeline("}")
        return code.getvalue()

    def call_kernel(self, name: str, node=None, deallocate_ws: bool = True) -> None:
        wrapper = V.graph.wrapper_code
        import torch

        max_tgs = compute_max_threadgroup_size(self.subkernels)

        for kernel, _ in self.subkernels:
            for v in kernel.args.sizevars:
                wrapper.ensure_size_computed(v)

        # Args must be passed in the same order the Slang shader binds them:
        # all in_decls (read-only) first, then rw_decls (read-write). The
        # `_outer` field on each decl is the wrapper-visible buffer name, so we
        # just emit those in the same order `build_global_binding_map` did.
        in_decls, rw_decls, _ = build_global_binding_map(self.subkernels)
        all_args: list[str] = [outer for _, _, outer in in_decls]
        all_args.extend(outer for _, _, outer in rw_decls)

        # PF.13.b.4-CODG: The Inductor memory planner may alias two buffers
        # via ``buf1 = reinterpret_tensor(div); del div`` before the kernel
        # call.  If any kernel argument name was freed by a reuse line,
        # substitute the new name so the emitted call doesn't reference a
        # deleted variable. Resolve transitively so a chain
        # ``buf9 → buf10 → buf11`` collapses to the final live name —
        # a naive one-step substitution leaves the intermediate buf10
        # in args, which references a deleted variable at runtime
        # (UnboundLocalError seen in MultiheadAttention forward graphs).
        freed: set = getattr(wrapper, "freed", set())
        reuses: dict = getattr(wrapper, "reuses", {})
        if freed:
            old_to_new: dict[str, str] = {}
            for new_name, old_name in reuses.items():
                old_to_new[old_name] = new_name

            def _resolve(name: str) -> str:
                seen: set[str] = set()
                while name in old_to_new and name in freed and name not in seen:
                    seen.add(name)
                    name = old_to_new[name]
                return name

            all_args = [_resolve(a) for a in all_args]

        for v in self.subkernels[0][0].args.sizevars:
            all_args.append(str(v))

        seen_args: set[str] = set()
        for kernel, _ in self.subkernels:
            for tree in kernel.range_trees:
                if isinstance(tree.numel, (sympy.Integer, int)):
                    continue
                if not isinstance(tree.numel, sympy.Symbol):
                    continue
                if tree.is_reduction and not kernel.inside_reduction:
                    continue
                sv = str(tree.numel)
                if sv not in seen_args:
                    seen_args.add(sv)
                    all_args.append(sv)

        for ws in self.subkernels[0][0].args.workspace_args:
            wrapper.generate_workspace_allocation(ws)

        # TRAIN.6-F1: Multi-dimensional grid dispatch.
        wg_x, wg_y, wg_z = compute_grid_dims(self.subkernels)
        all_args.append(str(wg_x))
        all_args.append(str(wg_y))
        all_args.append(str(wg_z))

        wrapper.generate_kernel_call(
            name,
            all_args,
            device=torch.device("vulkan"),
            triton=False,
            arg_types=None,
        )

        if deallocate_ws:
            for kernel, _ in self.subkernels:
                kernel.deallocate_workspaces()
