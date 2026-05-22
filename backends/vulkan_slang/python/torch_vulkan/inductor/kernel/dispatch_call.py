"""Wrapper-side dispatch grid + kernel-call emission for :class:`VulkanKernel`.

Extracted from ``kernel/header.py`` as ``CallKernelMixin`` to keep
``header.py`` under the 800-line anti-goal #7 cap (see
``backends/vulkan_slang/CLAUDE.md``).

Owns the ``call_kernel`` method that emits the wrapper-side dispatch
call — resolving alias chains, ordering input/output/inplace arguments,
appending sizevars and dynamic numels as push constants, and computing
the workgroup-count grid (including dynamic-shape, register-tile, and
overflow-into-y handling).

All ``self.xxx`` attribute accesses resolve via MRO against the host
class (``VulkanKernel``); this mixin contributes no state of its own.
"""

from __future__ import annotations

from typing import Any

import sympy
import torch
from torch._inductor.codegen.common import InplacedBuffer
from torch._inductor.virtualized import V

from .. import config


class CallKernelMixin:
    """Mixin providing wrapper-side dispatch-call codegen for :class:`VulkanKernel`."""

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
        # Deduplicate by inner_name: when a buffer is aliased via
        # reinterpret_tensor (e.g. buf2 → buf5), both the pre- and
        # post-alias outer names appear as separate entries in
        # inplace_buffers, all pointing to the same inner_name.  After
        # alias resolution both collapse to the live name (buf5) and the
        # duplicate gets the WRONG buffer into the next descriptor slot.
        # Mirror header.py's seen_inout guard: pick the last other_name
        # (the most-recent live outer name after aliasing) per inner.
        _seen_inout: set[str] = set()
        for outer, inner in self.args.inplace_buffers.items():
            if not isinstance(inner, InplacedBuffer):
                continue
            if inner.inner_name in _seen_inout:
                continue
            _seen_inout.add(inner.inner_name)
            ordered_args.append(inner.other_names[-1])

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
