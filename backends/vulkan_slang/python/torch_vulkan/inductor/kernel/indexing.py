"""Indexing codegen — multi-axis decomposition, bounds checks, strides.

Extracted from ``VulkanKernel`` via ``IndexingMixin`` (Track 1).
"""

from __future__ import annotations

import sympy
from torch._inductor.codegen.simd import IterationRangesEntry

from .symbolic import is_dynamic


class IndexingMixin:
    """Mixin providing index expression rendering and iteration-range decomposition."""

    def index_to_str(self, index: sympy.Expr) -> str:
        """Render an index expression. Enters the printer's subscript context
        so `tmp\\d+` symbols (CSE'd int64 indices stored as float) get the
        `((int)(...))` cast they need for safe array subscripting (P6.3)."""
        if isinstance(index, list):
            return f"[{', '.join(map(self.index_to_str, index))}]"
        with self._vk_printer.subscript():
            return self.kexpr(self.rename_indexing(index))

    def check_bounds(
        self,
        expr: sympy.Expr,
        size: sympy.Expr,
        lower: bool,
        upper: bool,
    ) -> None:
        """Emit a runtime bounds check for an indirect-indexing expression.

        Upstream Inductor calls this from ``CSEProxy`` whenever an
        ``indirect_indexing`` op is emitted with ``check=True`` (the default
        when ``config.assert_indirect_indexing`` is on, which it is by
        default).  Used for ``aten.gather`` / ``aten.scatter`` /
        ``aten.index_put`` / ``nn.Embedding`` / ``embedding_bag`` —
        anywhere a tensor of indices feeds a load or store.

        Strategy
        --------
        Slang/SPIR-V has no ``device_assert`` primitive in the vanilla
        compute pipeline (no ``trap``, no ``abort``).  ``OpKill`` only
        exists in fragment shaders, and ``printf`` requires
        ``GL_EXT_debug_printf`` which we do not enable.  The closest we
        can do is what Metal's MPS backend does: emit an early-return
        guard (``if (oob) return;``) that turns out-of-bounds threads
        into no-ops.  This avoids undefined behavior (Vulkan robust
        buffer access already returns 0 on OOB SSBO reads, but only on
        drivers that advertise the feature; the explicit guard makes
        the contract enforceable on every driver).

        We intentionally do NOT mutate the index variable to a clamped
        value — the variable was CSE'd upstream and may already be
        referenced; rewriting it would invalidate downstream uses.  The
        early-return guard is functionally equivalent for pointwise
        kernels (the OOB thread skips its store).  Strict abort with
        ``OpKill``/``printf`` is deferred — see TODO below.

        TODO(inductor-bounds-strict): When debug printf or VK_EXT_robustness2
        with explicit OOB reporting is wired up, switch the guard body to
        emit ``printf("OOB at %u: idx=%d size=%d", gl_GlobalInvocationID.x,
        idx, size)`` and ``return;`` so failures surface at runtime.
        """
        if not (lower or upper):
            return
        # Don't emit checks before the kernel has a body to write into;
        # this can happen during template kernel preamble.
        compute = getattr(self, "compute", None)
        if compute is None:
            return

        expr_str = self.index_to_str(expr)
        if lower and upper:
            size_str = self.index_to_str(size)
            cond = f"(({expr_str}) < 0 || ({expr_str}) >= ({size_str}))"
            note = f"0 <= {expr_str} < {size_str}"
        elif lower:
            cond = f"(({expr_str}) < 0)"
            note = f"0 <= {expr_str}"
        else:
            size_str = self.index_to_str(size)
            cond = f"(({expr_str}) >= ({size_str}))"
            note = f"{expr_str} < {size_str}"

        # Block-style comment so it's safe even on Slang sources processed by
        # line-based validators that don't reset comment state at newline.
        compute.writeline(f"/* CHECK: {note} */ if ({cond}) {{ return; }}")

    def _compute_red_numel(self) -> tuple[int, bool]:
        red_numel = 1
        has_dynamic = False
        for rd in self.range_trees:
            if rd.is_reduction:
                if is_dynamic(rd.numel):
                    red_numel = self.max_threadgroup_size
                    has_dynamic = True
                    break
                else:
                    red_numel *= int(rd.numel)
        return red_numel, has_dynamic

    @staticmethod
    def _slang_entries_sorted(root) -> list[IterationRangesEntry]:
        """Return root's entries sorted by divisor descending, matching
        ``construct_entries`` order (outer dimension first, inner second)."""
        entries = list(root.nodes.values())
        entries.sort(
            key=lambda e: (
                0 if isinstance(e.divisor, sympy.Integer) else 1,
                int(e.divisor) if isinstance(e.divisor, sympy.Integer) else 0,
            ),
            reverse=True,
        )
        return entries

    def _persistent_2d_layout(self) -> tuple[int, int] | None:
        """Check if this is a multi-axis persistent reduction eligible for 2D tiling.

        Returns (thread_y, thread_x) if eligible, None otherwise.
        Two cases: (a) 2 separate reduction root trees; (b) 1 root tree with
        2 entries (e.g. ``sum(dim=(0,2))``).  Product must be <= max_threadgroup_size.
        """
        if not self.inside_reduction or not self.range_trees:
            return None
        red_vars = [v for v in self.range_trees if v.is_reduction]
        if len(red_vars) == 2:
            if any(is_dynamic(v.numel) for v in red_vars):
                return None
            sizes = [int(v.numel) for v in red_vars]
            product = sizes[0] * sizes[1]
            if product == 0 or product > self.max_threadgroup_size:
                return None
            return (sizes[0], sizes[1])
        if len(red_vars) == 1:
            entries = self._slang_entries_sorted(red_vars[0])
            if len(entries) != 2:
                return None
            if any(is_dynamic(e.length) for e in entries):
                return None
            sizes = [int(e.length) for e in entries]
            product = sizes[0] * sizes[1]
            if product == 0 or product > self.max_threadgroup_size:
                return None
            return (sizes[0], sizes[1])
        return None

    def _partitioned_2d_layout(self) -> tuple[int, int, int, int] | None:
        """Check for 2-axis reduction where product > max_threadgroup_size.

        Returns (ty, tx, loop_y, loop_x) where ty * tx is the 2D thread
        layout and each thread processes loop_y rows × loop_x columns.
        Returns None if not a 2-axis reduction or sizes are dynamic.
        Two cases: (a) 2 separate reduction root trees; (b) 1 root tree with
        2 entries (e.g. ``sum(dim=(0,2))``).
        """
        if not self.inside_reduction or not self.range_trees:
            return None
        red_vars = [v for v in self.range_trees if v.is_reduction]
        if len(red_vars) == 2:
            if any(is_dynamic(v.numel) for v in red_vars):
                return None
            outer = int(red_vars[0].numel)
            inner = int(red_vars[1].numel)
        elif len(red_vars) == 1:
            entries = self._slang_entries_sorted(red_vars[0])
            if len(entries) != 2:
                return None
            if any(is_dynamic(e.length) for e in entries):
                return None
            outer = int(entries[0].length)
            inner = int(entries[1].length)
        else:
            return None
        product = outer * inner
        if product <= self.max_threadgroup_size:
            return None  # _persistent_2d_layout handles this case

        max_t = self.max_threadgroup_size
        simd = self.simd_group_size

        # Favor wider inner (x) dimension since wave reduce operates along x.
        # tx should be a multiple of simd for efficient wave ops.
        if inner <= max_t:
            tx = max(inner, simd)
            tx = min(tx, max_t)
            ty = min(outer, max_t // tx)
        elif outer <= max_t:
            ty = max(outer, 1)
            tx = min(max_t // ty, max_t)
            tx = (tx // simd) * simd
            tx = max(tx, simd)
        else:
            # Both dims large — use square-ish partition
            aspect = max(1, int((outer / inner) ** 0.5))
            ty = min(outer, max(4, aspect * 4))
            tx = min(inner, max_t // ty)
            tx = (tx // simd) * simd
            tx = max(tx, simd)

        if ty < 1 or tx < simd:
            return None

        loop_y = (outer + tx - 1) // tx
        loop_x = (inner + ty - 1) // ty
        return (ty, tx, loop_y, loop_x)

    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry) -> None:
        index_expr = self.rename_indexing(entry.expr)
        index_str = self.sexpr(index_expr)

        layout_2d = self._persistent_2d_layout()
        header_2d = getattr(self, "_header_layout_2d", None)
        if layout_2d is not None and header_2d is None:
            layout_2d = None
        partitioned = self._partitioned_2d_layout()

        if not entry.is_reduction or (
            not is_dynamic(entry.root.numel)
            and int(entry.root.numel) <= self.max_threadgroup_size
        ):
            if layout_2d is not None and entry.root.is_reduction:
                red_vars = [v for v in self.range_trees if v.is_reduction]
                if len(red_vars) == 1:
                    entries = self._slang_entries_sorted(entry.root)
                    for i, e in enumerate(entries):
                        if e.name == entry.name:
                            if i == 0:
                                self.indexing_code.writeline(
                                    f"{self.index_dtype} {entry.name} = lid.y;"
                                )
                            else:
                                self.indexing_code.writeline(
                                    f"{self.index_dtype} {entry.name} = lid.x;"
                                )
                            return
                else:
                    for i, rt in enumerate(self.range_trees):
                        if rt.is_reduction and rt.root is entry.root:
                            if i == 0:
                                self.indexing_code.writeline(
                                    f"{self.index_dtype} {entry.name} = lid.y;"
                                )
                            else:
                                self.indexing_code.writeline(
                                    f"{self.index_dtype} {entry.name} = lid.x;"
                                )
                            return
            self.indexing_code.writeline(
                f"{self.index_dtype} {entry.name} = {index_str};"
            )
            return

        # T5.1: 2D partitioned path for large multi-axis reductions
        if partitioned is not None and entry.root.is_reduction:
            ty, tx, loop_y, loop_x = partitioned
            red_vars = [v for v in self.range_trees if v.is_reduction]
            is_single_tree = len(red_vars) == 1
            root_already_processed = any(
                e.root is entry.root for e in self.multistage_reduction_entry
            )
            if not root_already_processed:
                self.multistage_reduction_entry.append(entry)
                self._partitioned_2d_layout_values = (ty, tx, loop_y, loop_x)
                self._partitioned_2d_active = True
                self._multistage_brace_count += 2
                # Hoist declarations before for loops so epilogue code
                # (e.g. batch-norm backward normalization) can reference
                # these variables after the loops close.
                # Only declare each variable once at function scope.
                # Write to _hoisted_decls which codegen_body prepends.
                if is_single_tree:
                    entries = self._slang_entries_sorted(entry.root)
                    for e in entries:
                        if e is not entry and e.name not in self._hoisted_vars:
                            self._hoisted_vars.add(e.name)
                            self._hoisted_decls.writeline(
                                f"{self.index_dtype} {e.name} = 0;"
                            )
                if entry.name not in self._hoisted_vars:
                    self._hoisted_vars.add(entry.name)
                    self._hoisted_decls.writeline(
                        f"{self.index_dtype} {entry.name} = 0;"
                    )
                outer_for = f"for (uint _ry = 0; _ry < {loop_y}u; ++_ry) {{"
                self.body.writeline(outer_for)
                self._loop_template.writeline(outer_for)
                with self.body.indent():
                    with self._loop_template.indent():
                        inner_for = f"for (uint _rx = 0; _rx < {loop_x}u; ++_rx) {{"
                        self.body.writeline(inner_for)
                        self._loop_template.writeline(inner_for)
                        with self.body.indent():
                            with self._loop_template.indent():
                                if is_single_tree:
                                    entries = self._slang_entries_sorted(entry.root)
                                    for i, e in enumerate(entries):
                                        if i == 0:
                                            idx_str = f"_ry * {tx}u + lid.x"
                                        else:
                                            idx_str = f"_rx * {ty}u + lid.y"
                                        var_line = (
                                            f"{self.index_dtype} {e.name} = {idx_str};"
                                        )
                                        self.body.writeline(var_line)
                                    for e in entries:
                                        if e is not entry:
                                            e.set_name(e.name)
                                    return
                                for i, rt in enumerate(self.range_trees):
                                    if rt.is_reduction and rt.root is entry.root:
                                        if i == 0:
                                            idx_str = f"_ry * {tx}u + lid.x"
                                        else:
                                            idx_str = f"_rx * {ty}u + lid.y"
                                        var_line = f"{self.index_dtype} {entry.name} = {idx_str};"
                                        self.body.writeline(var_line)
                                return
            else:
                return

        acc_size = entry.root.numel
        # D.2.a — Dynamic reduction numels now flow through the
        # multi-stage loop path below.  The key invariants:
        #   (a) Per-thread index: entry.root.name (e.g. r0) is already
        #       ``lid.x`` from the kernel-header preamble (codegen_kernel
        #       assigns ``uint r0 = lid.x;`` for every reduction var).
        #       The loop body uses ``{entry.root.name}`` directly, so
        #       every thread gets its own offset.
        #   (b) Dispatch grid: call_kernel() in header.py computes
        #       wg = product(non_red) for reductions (not divided by
        #       threadgroup_size — one WG per output element).
        #   (c) OOB mask: the static_divisible check below takes the
        #       dynamic branch and emits ``if (linear_idx >= numel) break;``
        #       which guards the last workgroup's partial wave.
        #
        # Force vw=1 for dynamic numels — we can't statically verify
        # divisibility for vec4 widening.
        if is_dynamic(acc_size):
            vw = 1
        else:
            vw = self._vec_width
        stride = self.max_threadgroup_size * vw
        loop_size = (
            (acc_size + stride - 1) // stride if not is_dynamic(acc_size) else None
        )

        root_already_processed = any(
            e.root is entry.root for e in self.multistage_reduction_entry
        )
        linear_idx_name = f"{entry.root.prefix}_linear_idx"

        if not root_already_processed:
            self.multistage_reduction_entry.append(entry)
            self._multistage_brace_count += 1
            if (
                vw == 1
                and not is_dynamic(acc_size)
                and int(acc_size) % 4 == 0
                and not self.has_welford
                and self._reduction_type in ("sum", "prod", "max", "min")
                and self.range_trees
                and self.range_trees[-1].is_reduction
            ):
                vw = 4
                stride = self.max_threadgroup_size * vw
                loop_size = (int(acc_size) + stride - 1) // stride
                self._vec_width = 4
            loop_size_str = (
                str(loop_size)
                if loop_size is not None
                else f"(({self.sexpr(acc_size)} + {stride - 1}) / {stride})"
            )
            unroll_attr = ""
            # DR.3: Cap unroll based on VGPR pressure + loop_depth.
            # Heavy kernels (welford, multi-axis reduction, deep loops)
            # use a lower cap to avoid VGPR explosion on RDNA1.
            _max_unroll = getattr(self, "_max_unroll_factor", lambda: 16)()
            if loop_size is not None and 1 < loop_size <= _max_unroll:
                unroll_attr = f"[unroll({loop_size})] "
            for_header = (
                f"{unroll_attr}for (uint {entry.root.prefix}_cnt = 0; "
                f"{entry.root.prefix}_cnt < {loop_size_str}; "
                f"++{entry.root.prefix}_cnt) {{"
            )
            # Hoist declarations before for loop so epilogue code
            # (e.g. batch-norm backward normalization) can reference
            # these variables after the loop closes.
            # Only declare each variable once at function scope.
            # Write to _hoisted_decls which codegen_body prepends at
            # function scope (before any for loops).
            if linear_idx_name not in self._hoisted_vars:
                self._hoisted_vars.add(linear_idx_name)
                self._hoisted_decls.writeline(
                    f"{self.index_dtype} {linear_idx_name} = 0;"
                )
            if entry.name not in self._hoisted_vars:
                self._hoisted_vars.add(entry.name)
                self._hoisted_decls.writeline(f"{self.index_dtype} {entry.name} = 0;")
            self.body.writeline(for_header)
            self._loop_template.writeline(for_header)
            static_divisible = not is_dynamic(acc_size) and int(acc_size) % stride == 0
            with self.body.indent():
                with self._loop_template.indent():
                    linear_idx_line = (
                        f"{linear_idx_name} = "
                        f"{stride} * {entry.root.prefix}_cnt + "
                        f"{vw} * {entry.root.name};"
                    )
                    self.body.writeline(linear_idx_line)
                    self._loop_template.writeline(linear_idx_line)
                if not static_divisible:
                    guard_line = f"if ({linear_idx_name} + {vw - 1} >= ({self.sexpr(acc_size)})) break;"
                    self.body.writeline(guard_line)
                    self._loop_template.writeline(guard_line)
                sub_index_str = index_str.replace(entry.root.name, linear_idx_name)
                entry_line = f"{entry.name} = {sub_index_str};"
                self.body.writeline(entry_line)
                self._loop_template.writeline(entry_line)
        else:
            # Hoist this entry's name to function scope so epilogue code
            # can reference it.  Write to _hoisted_decls which codegen_body
            # prepends before the for loop.
            if entry.name not in self._hoisted_vars:
                self._hoisted_vars.add(entry.name)
                self._hoisted_decls.writeline(f"{self.index_dtype} {entry.name} = 0;")
            # T5.14: When the partitioned-2D layout claimed this root via an
            # earlier entry, no `r0__cnt` loop emitted the flat
            # ``r0__linear_idx`` linearization.  A late-arriving entry whose
            # length equals the root numel (e.g. a flattened CSE'd index) lands
            # here and references ``linear_idx_name`` -> slangc fails with
            # ``undefined identifier``.  Synthesize the linearization from the
            # active partitioned-2D coords (`(_ry * tx + lid.x) * inner_len +
            # (_rx * ty + lid.y)`) so the substitution lands on a defined var.
            partitioned_active = (
                getattr(self, "_partitioned_2d_active", False)
                and getattr(self, "_partitioned_2d_layout_values", None) is not None
                and any(e.root is entry.root for e in self.multistage_reduction_entry)
            )
            if partitioned_active and linear_idx_name not in self._hoisted_vars:
                self._hoisted_vars.add(linear_idx_name)
                self._hoisted_decls.writeline(
                    f"{self.index_dtype} {linear_idx_name} = 0;"
                )
                ty_p, tx_p, _, _ = self._partitioned_2d_layout_values
                root_entries = self._slang_entries_sorted(entry.root)
                # Need exactly two ranged entries (the partitioned-2D shape)
                # for a clean flat linearization.  If the structure is
                # different, fall back to leaving the var at 0 (no crash).
                if len(root_entries) >= 2:
                    inner_len = int(root_entries[1].length)
                    flat_expr = (
                        f"((_ry * {tx_p}u + lid.x) * {inner_len}u "
                        f"+ (_rx * {ty_p}u + lid.y))"
                    )
                    self.body.writeline(f"{linear_idx_name} = {flat_expr};")
                    self._loop_template.writeline(f"{linear_idx_name} = {flat_expr};")
            with self.body.indent():
                sub_index_str = index_str.replace(entry.root.name, linear_idx_name)
                entry_line = f"{entry.name} = {sub_index_str};"
                self.body.writeline(entry_line)


class IndexingNotImplemented(NotImplementedError):
    """Raised when an indexing pattern can't be codegen'd for Vulkan."""


def raise_indexing_not_implemented(reason: str) -> None:
    raise IndexingNotImplemented(
        f"Vulkan Inductor indexing: {reason}. "
        "Set TORCHINDUCTOR_DISABLE_COMBO_KERNEL=1 to fall back."
    )
