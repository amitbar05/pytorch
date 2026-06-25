"""Pointwise codegen — store, register tile, persistent mode.

Extracted from ``VulkanKernel`` via ``PointwiseMixin`` (Track 1).
Load and vec4 logic extracted to ``pointwise_load_mixin`` / ``pointwise_vec4_mixin``
(M15.1.d — Track 1 anti-goal #7 split).
bwd_diff and DCE logic extracted to ``pointwise_bwd`` (CG.5 anti-goal #7 split).
"""

import logging
import re
from typing import TYPE_CHECKING

import sympy
import torch


logger = logging.getLogger(__name__)
from torch._inductor.codegen.common import CSEVariable, DeferredLine, IndentedBuffer
from torch._inductor.virtualized import V

from .pointwise_bwd import PointwiseBwdMixin
from .pointwise_load_mixin import PointwiseLoadMixin
from .pointwise_vec4_mixin import PointwiseVec4Mixin
from .symbolic import is_dynamic, is_dynamic_stride


if TYPE_CHECKING:
    from torch._inductor.ops_handler import StoreMode

_STORE_NO_CAST_DTYPES = frozenset(
    {
        "float",
        "half",
        "int",
        "uint",
    }
)


class PointwiseMixin(PointwiseLoadMixin, PointwiseVec4Mixin, PointwiseBwdMixin):
    """Mixin providing pointwise store, register tile, and persistent mode.

    bwd_diff and DCE methods live in PointwiseBwdMixin (pointwise_bwd.py).
    """

    # Suppress type-checker complaints about attributes defined in other
    # mixins or the base SIMDKernel — all resolved via self at runtime.

    def _build_body_var_deps(self, body_str: str) -> dict[str, set[str]]:
        """Parse body lines and build a variable dependency graph.

        Scans for assignments of the form ``<type> <var> = <rhs>;``
        and records which variables appear on the RHS for each LHS.
        Returns ``{var_name: {vars_it_depends_on}}``.

        Variables that reference lane/thread IDs (``lid.x``, ``lid.y``,
        ``ltid``) are marked with a synthetic ``__lane_id__`` dependency
        so transitive checks can detect them.
        """
        deps: dict[str, set[str]] = {}
        for line in body_str.splitlines():
            m = self._ASSIGN_RE.match(line)
            if not m:
                continue
            lhs = m.group(1)
            rhs = m.group(2)
            # Skip obvious non-variable tokens
            rhs_vars: set[str] = set()
            for tok in re.findall(r"\b([a-zA-Z_]\w*)\b", rhs):
                if tok in self._LANE_ID_TOKENS:
                    rhs_vars.add("__lane_id__")
                elif not tok[0].isdigit():
                    rhs_vars.add(tok)
            # Also check for direct lane-id references like "lid.x"
            if any(t in rhs for t in ("lid.x", "lid.y", "lid.z", "ltid")):
                rhs_vars.add("__lane_id__")
            deps[lhs] = rhs_vars
        return deps

    def _transitive_dep_closure(
        self, deps: dict[str, set[str]], roots: set[str]
    ) -> set[str]:
        """Compute the transitive closure of all variables reachable from
        ``roots`` through the dependency graph ``deps``."""
        result: set[str] = set()
        stack: list[str] = list(roots)
        while stack:
            var = stack.pop()
            if var in result:
                continue
            result.add(var)
            for dep in deps.get(var, set()):
                if dep not in result:
                    stack.append(dep)
        return result

    def _check_index_lane_dependency(
        self, body_str: str, rt_name: str, all_inners: list[str]
    ) -> bool:
        """Return True if any buffer's index variable transitively depends
        on lane/thread IDs (lid.x, lid.y, ltid).

        Builds a dependency graph from the body and checks each buffer
        access ``buf_name[idx_var]`` — if ``idx_var`` (or any variable
        it depends on) references a lane/thread ID, the kernel is
        ineligible for vec4 rewriting (vec4 processes 4 consecutive
        global elements per thread; lane-ID-indexed access would pick
        wrong elements).
        """
        deps = self._build_body_var_deps(body_str)
        if "__lane_id__" not in self._transitive_dep_closure(deps, set(deps.keys())):
            return False  # No lane-id dependencies at all

        # Now check: for each I/O buffer, does its index variable depend
        # on a lane-ID?  We look at patterns like `buf_name[<var>]`.
        buf_access_re = re.compile(
            r"\b(" + "|".join(re.escape(n) for n in all_inners) + r")\s*\[\s*(.+?)\s*\]"
        )
        for m in buf_access_re.finditer(body_str):
            # The index may be a composite expression (e.g. base + xindex).
            # Extract every identifier token and check each transitively.
            for idx_var in re.findall(r"\b([a-zA-Z_]\w*)\b", m.group(2)):
                # Check if idx_var or any of its transitive deps reference lane IDs
                closure = self._transitive_dep_closure(deps, {idx_var})
                if "__lane_id__" in closure:
                    return True
        return False

    def _can_register_tile(self, tile_size: int) -> bool:
        """Check whether register tiling is applicable to this kernel.

        Conditions (all must hold):
        1. Single non-reduction range-tree axis.
        2. Static numel, divisible by max_threadgroup_size * tile_size.
        3. Not inside a reduction or multistage entry.
        4. Not vec4-eligible (vec4 is a stronger optimization).
        5. Not packed16 (packed16 has its own vectorization path).
        6. Not in persistent mode (persistent already does grid-stride).
        """

        if tile_size < 2 or tile_size > 4:
            return False
        if self.inside_reduction:
            return False
        if self.multistage_reduction_entry:
            return False
        if getattr(self, "_packed16", False):
            return False
        if getattr(self, "_persistent_mode", False):
            return False

        non_red = [t for t in self.range_trees if not t.is_reduction]
        if len(non_red) != 1:
            return False
        rt = non_red[0]
        if not isinstance(rt.numel, sympy.Integer):
            return False
        numel = int(rt.numel)
        if numel <= 0:
            return False

        wg = self.max_threadgroup_size
        if numel % (wg * tile_size) != 0:
            return False

        if getattr(self, "_vec4_pw_bufs", None):
            return False

        body_str = self.body.getvalue()
        if "gtid.x" not in body_str:
            return False
        anchor = f"uint {rt.name} = gtid.x;"
        if anchor not in body_str:
            return False

        return True

    def _apply_register_tile(self, body_str: str, tile_size: int) -> str | None:
        """Rewrite the scalar pointwise body for register tiling.

        Wraps the body in ``[unroll] for (uint _rt = 0u; _rt < T; ++_rt)``
        with ``xindex = xbase + _rt`` re-declared inside the loop.
        Returns the new body string, or None on failure.

        M-PERF.2: The unroll factor is gated on VGPR pressure.  For
        ``heavy`` kernels (estimated VGPRs > 32, e.g. f64 / welford /
        deep-loop chains) we emit ``[unroll(2)]`` instead of a full
        ``[unroll]`` so slangc keeps a small inner loop rather than
        expanding the body T-fold — which on RDNA1 (256 VGPRs/CU)
        regularly drops occupancy from 2 waves/CU to 1 wave/CU and
        gives back the 5-10% the tile was meant to win.
        """
        non_red = [t for t in self.range_trees if not t.is_reduction]
        rt = non_red[0]
        rt_name = rt.name

        anchor = f"uint {rt_name} = gtid.x;"
        anchor_idx = body_str.find(anchor)
        if anchor_idx < 0:
            return None

        head = body_str[:anchor_idx]
        tail = body_str[anchor_idx + len(anchor) :]
        xbase_line = f"uint xbase = gtid.x * {tile_size}u;"

        # M-PERF.2: VGPR-pressure-gated unroll attribute.  Falls back to
        # ``[unroll]`` (full) when the classifier is unavailable or the
        # kernel is light/normal pressure.
        dtype_bytes = 2 if getattr(self, "_packed16", False) else 4
        try:
            vgpr_class, _ = self._classify_vgpr_pressure(dtype_bytes)
        except Exception:
            vgpr_class = "normal"
        if vgpr_class == "heavy" and tile_size > 2:
            unroll_attr = "[unroll(2)]"
        else:
            unroll_attr = "[unroll]"

        new_buf = IndentedBuffer()
        new_buf.splice(head)
        new_buf.writeline(xbase_line)
        new_buf.writeline(
            f"{unroll_attr} for (uint _rt = 0u; _rt < {tile_size}u; ++_rt) {{"
        )
        with new_buf.indent():
            new_buf.writeline(f"uint {rt_name} = xbase + _rt;")
            new_buf.splice(tail)
        new_buf.writeline("}")

        return new_buf.getvalue()

    def _enable_persistent_mode(self) -> None:
        """Enable grid-stride-loop wrapping for this kernel.

        Called by the scheduler when a chain of small pointwise ops
        is detected.  When enabled, the body is wrapped in a for-loop
        so each thread processes multiple elements.
        """
        self._persistent_mode = True

    def _emit_persistent_grid_stride_loop(self) -> str | None:
        """Emit a grid-stride loop wrapper for the pointwise body.

        When _persistent_mode is True, this wraps the compute body
        in a for-loop that lets each thread process multiple elements
        across potentially multiple operations.

        Returns the modified body source, or None if persistent mode
        is not active.
        """
        from .. import config

        if not self._persistent_mode:
            return None
        if not config.persistent_pointwise():
            return None
        if self.inside_reduction:
            return None

        # Compute total numel from numels dict
        total = 1
        for v in self.numels.values():
            if is_dynamic(v):
                return None  # dynamic shapes not yet supported
            total *= int(v)

        wg_size = self.max_threadgroup_size
        # M11.4: Scale persistent WG count by actual CU count, not hardcoded 20.
        # More WGs → more wave slots filled → better occupancy for persistent
        # kernels that stay resident across the grid-stride loop.
        try:
            from torch._dynamo.device_interface import get_interface_for_device

            iface = get_interface_for_device("vulkan")
            props = iface.Worker.get_device_properties()
            num_cus = getattr(props, "num_compute_units", 16)
        except Exception:
            # M-PERF.5: RDNA1 (RX 5600 XT) has 16 CUs — use as default.
            num_cus = 16

        # M-PERF.5: Replace the hard ``total > 16384`` reject with a
        # persistent-WG-count clamp.  For large numels we previously
        # bailed out to a plain elementwise dispatch (one launch per
        # bucket, full overhead per op).  Now we keep persistence
        # enabled for any numel ≥ wg_size and clamp the grid to
        # ``cu_count * 4`` total resident waves on RDNA1
        # (4 waves/CU × wave64 lanes / wg_size = WG count).  One
        # persistent dispatch then chews through tensors up to and
        # beyond 64M elements via the grid-stride loop, amortizing
        # launch / barrier overhead across the whole tensor.
        # Expected gain: 20-30% on large-batch element-wise ops.
        sgs = self.simd_group_size or 64  # RDNA1 wave64
        waves_per_cu = 4  # RDNA1 hardware cap
        total_resident_lanes = num_cus * waves_per_cu * sgs
        persistent_wg_count = max(1, total_resident_lanes // max(1, wg_size))
        # Never request more WGs than the work itself can fill.
        num_wgs = max(1, min(persistent_wg_count, (total + wg_size - 1) // wg_size))

        body_str = self.body.getvalue()
        if not body_str.strip():
            return None

        # Wrap the body in a grid-stride loop.
        # Each thread computes: for (i = tid; i < total; i += grid_stride)
        # The original body is preserved but with i replacing the
        # original global index.
        grid_stride = wg_size * num_wgs

        # Heuristic: for very small numels (< wg_size), use a single WG
        # and let threads loop over the elements.
        if total < wg_size:
            grid_stride = wg_size

        loop_body = IndentedBuffer()
        loop_body.writeline(
            f"for (uint _pi = gtid.x; _pi < {total}u; _pi += {grid_stride}u) {{"
        )
        with loop_body.indent():
            # Replace gtid.x references with _pi in the body
            # We use a simple substitution — the body uses gtid.x for
            # global indexing in single-axis pointwise kernels.
            adjusted = body_str.replace("gtid.x", "_pi")
            # Also handle cases where gtid is used as a uint3
            adjusted = adjusted.replace("gtid", "_pi")
            loop_body.splice(adjusted)
        loop_body.writeline("}")

        return loop_body.getvalue()

    @staticmethod
    def _is_small_pointwise_chain(nodes) -> bool:
        """Check if a list of scheduler nodes form a small pointwise chain
        suitable for persistent kernel micro-batching.

        GPU.5+ — Improved criteria:
        - All nodes are pointwise (no reductions)
        - At least 2 nodes (single op doesn't benefit)
        - Per-thread work: total_numel / (wg_size * target_wgs) <= 16
          (each thread does at most 16 iterations; more = overhead dominates)
        - Number of ops scales the benefit: more ops = more dispatches saved
        """
        if len(nodes) < 2:
            return False

        # Estimate workgroup size for per-thread work calculation.
        # Default to 256 threads (pointwise kernels typically use
        # max_threadgroup_size=256).
        wg_size = 256
        # M11.4: Use actual CU count for target WGs.
        try:
            from torch._dynamo.device_interface import get_interface_for_device

            iface = get_interface_for_device("vulkan")
            props = iface.Worker.get_device_properties()
            num_cus = getattr(props, "num_compute_units", 20)
        except Exception:
            num_cus = 20
        target_wgs = num_cus
        num_threads = wg_size * target_wgs

        total_numel = 0
        for sn in nodes:
            _, (numel, rnumel) = sn.group
            if rnumel != 1:
                return False  # has reduction
            if is_dynamic_stride(numel):
                return False
            n = int(numel)
            total_numel += n

        # Per-thread iterations: how many elements each thread processes.
        per_thread_iters = (total_numel + num_threads - 1) // num_threads

        # GPU.5+: Tune the per-thread-iteration cap by number of ops.
        # More ops in the chain = more dispatches saved by fusing,
        # so we can tolerate higher per-thread work.
        #   - 2 ops: cap at 16 iterations/thread (save 1 dispatch)
        #   - 3-4 ops: cap at 32 iterations/thread (save 2-3 dispatches)
        #   - 5+ ops: cap at 64 iterations/thread (save 4+ dispatches)
        num_ops = len(nodes)
        if num_ops >= 5:
            iter_cap = 64
        elif num_ops >= 3:
            iter_cap = 32
        else:
            iter_cap = 16

        # Also check: even with many ops, don't go beyond a total numel
        # that would produce excessive register pressure from live
        # variables across all ops in the fused kernel.
        # C6.4 (2026-06-18): raise cap from 16384→65536. The persistent
        # grid-stride loop already handles larger numel by looping;
        # register pressure from live variables is bounded by the number
        # of ops (max 64 iterations/thread × ~4 registers/iter = 256 VGPRs,
        # within RDNA1 budget). This allows more ops in the loss backward
        # (GN backward sub/mul/sum chains) to fuse.
        max_total_numel = 65536

        return per_thread_iters <= iter_cap and total_numel <= max_total_numel

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: CSEVariable,
        mode: "StoreMode" = None,
    ) -> None:
        var = self.args.output(name)
        index = self.prepare_indexing(index)
        out_dtype = V.graph.get_dtype(name)
        idx_str = self.index_to_str(index)

        # Track 5.7: Record sympy index for BlockPatternMatcher analysis.
        self._pw_index_records.append((var, index, False))

        if (
            mode is None
            and self._decide_packed16(out_dtype)
            and not self._packed16_load_only
        ):
            self._pw_uses_subbyte_packing = True
            # M19.4 (Gate C): only mark the kernel as wave-op-using when
            # the packed16-vector-write rewrite is NOT going to elide
            # this scalar-store path. When `_packed16_vw_active` is set,
            # `_packed16_vw_rewrite` replaces the WaveReadLaneAt-based
            # pack/unpack with a `_pvw_out_*[_k] = …` scratch store —
            # so the wave-op never reaches the emitted shader.  Setting
            # the flag unconditionally here used to cause circular
            # rejection at `pointwise_vec4_mixin.py:199` (packed16 vec4
            # path) for every f16 contiguous pointwise kernel.
            if not getattr(self, "_packed16_vw_active", False):
                self._pw_has_wave_ops = True
            self._packed16_bufs.add(var)
            suffix = "f16" if out_dtype == torch.float16 else "bf16"
            self.headers.add(f"packed16_{suffix}")
            uid = f"{abs(hash((var, idx_str))) & 0xFFFF:04x}"
            line = (
                f"{{ float _p16_odd_{uid} = WaveReadLaneAt((float)({value}), "
                f"WaveGetLaneIndex() ^ 1u); "
                f"if (({idx_str}) % 2u == 0u) "
                f"{self._buf_path(var)}[({idx_str}) >> 1u] = _vk_pack_{suffix}((float)({value}), _p16_odd_{uid}); }}"
            )
            target_buf = self.compute if self.inside_reduction else self.stores
            target_buf.writeline(DeferredLine(name, line))
            self._p16_store_records.append((var, str(value), suffix))
            return

        dtype_str = self.dtype_to_str(out_dtype)
        if out_dtype == torch.bfloat16 and not self.inside_reduction:
            # Fallback packed16 store for bf16 output when packed16 loads were
            # disabled (e.g. mixed fp32/bf16 inputs disabling _decide_packed16).
            # DTYPE_TO_SLANG[bfloat16]="uint", so the generic cast_val path
            # would emit ((uint)(value)) which TRUNCATES the float to integer.
            # Correct bf16 packing requires reading the adjacent lane's value
            # (WaveReadLaneAt) and packing two bf16 values per uint slot.
            # This is identical to the packed16 store path above but entered
            # unconditionally when out_dtype is bfloat16.
            self._pw_uses_subbyte_packing = True
            # CG.2: mirror the guard from the primary packed16 path — when
            # _packed16_vw_active is True, _packed16_vw_rewrite() replaces
            # this WaveReadLaneAt body with a gtid.x vector write; signalling
            # _pw_has_wave_ops in that state causes a structural conflict.
            if not getattr(self, "_packed16_vw_active", False):
                self._pw_has_wave_ops = True
            self.headers.add("packed16_bf16")
            self._packed16_bufs.add(var)
            uid = f"{abs(hash((var, idx_str))) & 0xFFFF:04x}"
            line = (
                f"{{ float _p16_odd_{uid} = WaveReadLaneAt((float)({value}), "
                f"WaveGetLaneIndex() ^ 1u); "
                f"if (({idx_str}) % 2u == 0u) "
                f"{self._buf_path(var)}[({idx_str}) >> 1u] = "
                f"_vk_pack_bf16((float)({value}), _p16_odd_{uid}); }}"
            )
            self.stores.writeline(DeferredLine(name, line))
            self._p16_store_records.append((var, str(value), "bf16"))
            return
        if out_dtype == torch.bool:
            # M-NEW.13 (2026-05-22) — comment refresh: bool output buffers
            # are declared as ``RWStructuredBuffer<uint8_t>`` (1 B/slot)
            # per ``DTYPE_TO_SLANG[torch.bool] = "uint8_t"`` (M18.4-followup-C),
            # matching PyTorch's 1 B/element bool storage. Slang implicitly
            # narrows the ``uint`` cast to ``uint8_t`` at store time, so
            # the existing ``((uint)({value}))`` cast still writes the
            # correct 0/1 byte. Refresh kept here to align with the load
            # side (``pointwise_load_mixin.py``) which now routes all
            # bool reads through the native ``((float)(v[i]))`` path
            # against the ``uint8_t`` slot.
            cast_val = f"((uint)({value}))"
        elif out_dtype == torch.int64:
            cast_val = f"uint2(uint(int({value})), uint(int({value}) >> 31))"
        else:
            val_dtype = getattr(value, "dtype", None)
            if val_dtype is not None and val_dtype == out_dtype:
                cast_val = f"{value}"
            else:
                cast_val = f"(({dtype_str})({value}))"
        guard = ""
        if self.inside_reduction:
            red_numel, has_dynamic = self._compute_red_numel()
            # OP.22: when red_numel is dynamic, emit a guard against
            # the push-constant numel so OOB threads don't write to
            # the output buffer.  For cooperative reductions where
            # multiple WGs contribute, this prevents corruption when
            # the last WG has a partial wave.
            if has_dynamic:
                reduction_root = None
                for rd in self.range_trees:
                    if rd.is_reduction:
                        reduction_root = rd
                        break
                if reduction_root is not None:
                    from .symbolic import dynamic_reduction_guard

                    guard = dynamic_reduction_guard(
                        reduction_root.name, self.max_threadgroup_size
                    )
            elif red_numel < self.max_threadgroup_size:
                guard = f"if (lid.x < {red_numel}) "
        if mode is None:
            line = f"{guard}{self._buf_path(var)}[{idx_str}] = {cast_val};"
        elif mode == "atomic_add":
            self._pw_has_atomic_op = True
            self.headers.add("atomic_add")
            self._atomic_out_bufs.add(var)
            line = f"vk_atomic_add({self._buf_path(var)}, {idx_str}, ({value}));"
            target_buf = self.compute if self.inside_reduction else self.stores
            target_buf.writeline(DeferredLine(name, line))
            return
        else:
            raise RuntimeError(f"Unimplemented store mode {mode}")
        if self.inside_reduction:
            self.compute.writeline(DeferredLine(name, line))
        else:
            self.stores.writeline(DeferredLine(name, line))

    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable) -> None:
        var = self.args.output(name)
        index = self.prepare_indexing(index)
        # Track 5.7: Record sympy index for BlockPatternMatcher analysis.
        self._pw_index_records.append((var, index, False))
        out_dtype = V.graph.get_dtype(name)
        if out_dtype == torch.bool:
            cast_expr = f"((uint)({value}))"
        elif out_dtype == torch.int64:
            cast_expr = f"uint2(uint(int({value})), uint(int({value}) >> 31))"
        elif out_dtype == torch.bfloat16:
            cast_expr = f"((float)({value}))"
        else:
            dtype_str = self.dtype_to_str(out_dtype)
            cast_expr = f"(({dtype_str})({value}))"
        layout_2d = self._persistent_2d_layout()
        if layout_2d is not None:
            line = (
                f"if (lid.y == 0 && lid.x == 0) "
                f"{self._buf_path(var)}[{self.index_to_str(index)}] = {cast_expr};"
            )
        else:
            reduction_dim = next(t for t in self.range_trees if t.is_reduction)
            line = (
                f"if ({reduction_dim.name} == 0) "
                f"{self._buf_path(var)}[{self.index_to_str(index)}] = {cast_expr};"
            )
        self.stores.writeline(DeferredLine(name, line))
