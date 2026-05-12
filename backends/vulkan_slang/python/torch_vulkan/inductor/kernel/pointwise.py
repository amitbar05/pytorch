"""Pointwise codegen — load, store, packed16, vec4.

Extracted from ``VulkanKernel`` via ``PointwiseMixin`` (Track 1).
"""

import logging
import re
from typing import TYPE_CHECKING, Any, Optional

import sympy
import torch

logger = logging.getLogger(__name__)
from torch._inductor.codegen.block_analysis import BlockPatternMatcher
from torch._inductor.codegen.common import (
    CSEVariable,
    DeferredLine,
    IndentedBuffer,
)
from torch._inductor.virtualized import V

from .symbolic import is_dynamic

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

# dtype → (emit_fn, header_tag) dispatch table used by VulkanKernel.load().
# ``emit_fn(var, idx_str)`` returns a Slang expression that loads one
# element from ``var`` at index ``idx_str`` into a float register.
# ``header_tag``, if non-None, is added to ``self.headers`` so
# ``slang_helpers.emit_helpers`` emits the supporting function.
_LOAD_DISPATCH: dict = {}  # populated lazily for torch.dtype imports


def _init_load_dispatch() -> None:
    import torch

    global _LOAD_DISPATCH
    if _LOAD_DISPATCH:
        return
    # Sub-32-bit dtypes (bool / uint8 / int8 / int16) bind through Slang
    # as ``StructuredBuffer<uint>`` (32-bit slots) regardless of PyTorch's
    # 1- or 2-byte itemsize.  The pointwise STORE path writes ``out[idx] =
    # cast<uint>(v)`` — one uint per element — so the LOAD path MUST read
    # the same way; otherwise Inductor-internal buffers (e.g. saved
    # ``bool`` masks for ``WhereBackward0``) read garbage every time the
    # idx is not 4-aligned.  This was the root cause of partially-correct
    # gradients in compiled ``relu().sum().backward()`` (got
    # ``[1,0,0,0,1,0,0,0]`` instead of ``[1,1,1,0,1,0,0,0]``) — the FW
    # kernel wrote 4 B per bool, the BW kernel read 1 B per bool via
    # ``_vk_unpack_u8``, so 3 of every 4 reads picked up zero-padding.
    #
    # Read 1 uint per element to match the STORE side.  ``_vk_unpack_*``
    # helpers in ``helpers.slang`` are kept for future external-tensor
    # paths (CPU bool masks copied with native packed-byte storage), but
    # never used for compile-internal pointwise loads.
    _LOAD_DISPATCH.update(
        {
            torch.bool: (lambda v, i: f"((float)({v}[{i}]))", None),
            torch.uint8: (lambda v, i: f"((float)({v}[{i}]))", None),
            torch.int8: (
                lambda v, i: f"((float)((int({v}[{i}]) << 24) >> 24))",
                None,
            ),
            torch.int16: (
                lambda v, i: f"((float)((int({v}[{i}]) << 16) >> 16))",
                None,
            ),
            torch.float16: (lambda v, i: f"((float)({v}[{i}]))", None),
            torch.bfloat16: (lambda v, i: f"((float)({v}[{i}]))", None),
            torch.int32: (lambda v, i: f"((float)({v}[{i}]))", None),
            torch.int64: (lambda v, i: f"((float)(int)({v}[{i}].x))", None),
        }
    )


class PointwiseMixin:
    """Mixin providing pointwise load, store, packed16, and vec4 codegen."""

    # Suppress type-checker complaints about attributes defined in other
    # mixins or the base SIMDKernel — all resolved via self at runtime.

    # ── CG.M8: Inline bwd_diff emission ─────────────────────────────

    def register_inline_unary_bwd(
        self,
        aten_op: str,
        x_var: str,
        grad_out_var: str,
        grad_in_var: str,
    ) -> None:
        """Record a unary bwd_diff operation for inline emission.

        Called during inner_fn codegen. The actual Slang emission happens
        later when ``_emit_inline_bwd_diff_body`` is called before body
        finalization.
        """
        from torch_vulkan.inductor.bwd_diff_table import BWD_DIFF_TABLE

        entry = BWD_DIFF_TABLE.get(aten_op)
        if entry is not None:
            self._bwd_diff_imports.add(entry.module)
        self._bwd_diff_unary_ops.setdefault(aten_op, []).append(
            (x_var, grad_out_var, grad_in_var)
        )

    def register_inline_binary_bwd(
        self,
        aten_op: str,
        a_var: str,
        b_var: str,
        grad_out_var: str,
        grad_a_var: str,
        grad_b_var: str,
    ) -> None:
        """Record a binary bwd_diff operation for inline emission."""
        from torch_vulkan.inductor.bwd_diff_table import BWD_DIFF_TABLE

        entry = BWD_DIFF_TABLE.get(aten_op)
        if entry is not None:
            self._bwd_diff_imports.add(entry.module)
        self._bwd_diff_binary_ops.setdefault(aten_op, []).append(
            (a_var, b_var, grad_out_var, grad_a_var, grad_b_var)
        )

    def _emit_inline_bwd_diff_body(self) -> None:
        """Emit all registered inline bwd_diff operations into the compute
        buffer. Called during ``codegen_body`` before body finalization.

        This replaces the generic arithmetic that inner_fn emitted with
        actual ``bwd_diff(fwd_fn)(...)`` Slang calls.

        Also adds ``import <module>;`` declarations to
        ``self.module_scope_decls`` so the Slang shader can resolve
        the forward functions (e.g. ``silu_fwd`` from ``pointwise``).
        """
        if not self._bwd_diff_unary_ops and not self._bwd_diff_binary_ops:
            return

        # Add module imports to module_scope_decls (idempotent via set)
        for mod in sorted(self._bwd_diff_imports):
            import_line = f"import {mod};"
            self.module_scope_decls.writeline(import_line)

        from torch_vulkan.inductor.bwd_diff_table import BWD_DIFF_TABLE
        from torch_vulkan.inductor.kernel.bwd_diff_inline import (
            emit_inline_binary_bwd,
            emit_inline_unary_bwd,
        )

        for aten_op, ops in self._bwd_diff_unary_ops.items():
            entry = BWD_DIFF_TABLE.get(aten_op)
            if entry is None:
                continue
            for x_var, grad_out_var, grad_in_var in ops:
                body_lines, result_expr = emit_inline_unary_bwd(
                    entry,
                    x_var=x_var,
                    grad_out_var=grad_out_var,
                    dtype="float",
                )
                self.compute.writeline(body_lines)
                # Use CSE to create the result variable for the store
                self.cse.generate(
                    self.compute,
                    result_expr,
                    dtype=torch.float32,
                )

        for aten_op, ops in self._bwd_diff_binary_ops.items():
            entry = BWD_DIFF_TABLE.get(aten_op)
            if entry is None:
                continue
            for a_var, b_var, grad_out_var, grad_a_var, grad_b_var in ops:
                body_lines, result_a_expr, _result_b_expr = emit_inline_binary_bwd(
                    entry,
                    a_var=a_var,
                    b_var=b_var,
                    grad_out_var=grad_out_var,
                    dtype="float",
                )
                self.compute.writeline(body_lines)
                self.cse.generate(
                    self.compute,
                    result_a_expr,
                    dtype=torch.float32,
                )

        # Clear after emission
        self._bwd_diff_unary_ops.clear()
        self._bwd_diff_binary_ops.clear()

    def _buf_path(self, inner_name: str) -> str:
        """Return the buffer access path for codegen emission.

        When ParameterBlock is enabled (P3.1/M9), buffer accesses use
        args.in_ptr0[idx] instead of in_ptr0[idx]. This helper
        returns args.{inner_name} or just {inner_name} based on
        the current mode.
        """
        if getattr(self, "_use_parameter_block", False):
            return f"args.{inner_name}"
        return inner_name

    def _decide_packed16(self, dtype: torch.dtype) -> bool:
        """Lazily decide whether this kernel uses packed16 mode.

        Called on every load/store.  Returns True only when eligible AND the
        dtype of the new buffer matches the dtype already locked in.  Flips
        self._packed16 to False permanently on the first disqualifying event.

        Eligibility rules (all must hold):
        - No reduction, multistage, or welford (unless _packed16_load_only).
        - All I/O buffers share the same half dtype (f16 or bf16).
        - Innermost non-reduction axis has even numel (so pairs of adjacent
          elements can be packed into one uint32 word).
        - For small persistent reductions (rnumel <= simd_group_size, even
          rnumel): packed16 is load-only — stores remain f32.
        """
        from .. import config

        if self._packed16 is False:
            return False

        if (
            self.has_welford
            or self.multistage_reduction_entry
            or config.no_packed16()
            or not config.prefer_packed16()
        ):
            self._packed16 = False
            return False

        if dtype not in (torch.float16, torch.bfloat16):
            self._packed16 = False
            return False

        if self._packed16 is None:
            axes = self.active_range_trees()
            non_red = [t for t in axes if not t.is_reduction]
            red = [t for t in axes if t.is_reduction]

            if self.inside_reduction and red:
                rnumel, _ = self._compute_red_numel()
                if (
                    isinstance(rnumel, sympy.Integer)
                    and int(rnumel) > 0
                    and int(rnumel) <= self.simd_group_size
                    and int(rnumel) % 2 == 0
                ):
                    self._packed16_load_only = True
                else:
                    self._packed16 = False
                    return False

            if not non_red:
                self._packed16 = False
                return False

            innermost = non_red[-1]
            if (
                not isinstance(innermost.numel, sympy.Integer)
                or int(innermost.numel) % 2 != 0
            ):
                self._packed16 = False
                return False

            self._packed16 = True
            self._packed16_dtype = dtype

            non_red_trees = [t for t in axes if not t.is_reduction]
            if (
                not self.inside_reduction
                and len(non_red_trees) == 1
                and isinstance(non_red_trees[0].numel, sympy.Integer)
                and int(non_red_trees[0].numel) % (self.max_threadgroup_size * 4) == 0
            ):
                self._packed16_vw_active = True

            return True

        if dtype != self._packed16_dtype:
            self._packed16 = False
            return False
        return True

    # ── M23: Variable dependency tracking for vec4 eligibility ──────
    # The body is a mix of plain strings and DeferredLine objects in
    # IndentedBuffer._lines.  We parse simple assignment patterns to
    # build a dependency graph, then check whether any buffer index
    # variable transitively depends on lane/thread IDs (lid.x, lid.y,
    # ltid).  This is more precise than string-scanning for "lid.x"
    # because it tracks transitive dependencies through CSE variables.

    _LANE_ID_TOKENS = frozenset({"lid.x", "lid.y", "lid.z", "ltid"})
    _ASSIGN_RE = re.compile(
        r"^\s*(?:float|int|int64_t|uint|bool|half)\s+(\w+)\s*=\s*(.+);\s*$"
    )

    # ── M22: Dead code elimination (DCE) ───────────────────────────
    # After body codegen, scan the generated Slang source for CSE
    # variable declarations.  Build a use-def chain, mark variables
    # transitively reachable from output stores as "live", and strip
    # assignments whose LHS is dead.  This eliminates unused loads and
    # computations that survive upstream DeferredLine pruning (which
    # only removes stores to removed buffers).
    #
    # Gated by TORCH_VULKAN_DCE=1 (default: 1).

    # Pattern for any typed CSE assignment (float/int/int64_t/uint/bool/half).
    # Group 1: variable name, Group 2: RHS expression (without trailing ;).
    _DCE_ASSIGN_RE = re.compile(
        r"^\s*(?:float|int|int64_t|uint|bool|half)\s+(\w+)\s*=\s*(.+);\s*$"
    )
    # Variables that are always live (never eliminated).
    _DCE_ALWAYS_LIVE = frozenset(
        {
            "lid",
            "gtid",
            "gid",  # built-in thread/group IDs
            "xindex",
            "yindex",
            "zindex",  # range-tree variables
            "rindex",
            "x0index",
            "y0index",
            "_vk_linear",
            "_vk_linear_orig",
            "xbase",
            "_k",  # vec4 rewrite artifacts
            "ltid",  # linear thread ID
        }
    )
    # Prefixes whose variables are always live.
    _DCE_LIVE_PREFIXES = ("lid.", "gtid.", "gid.", "pc.", "Wave", "Group")

    @staticmethod
    def _dce_parse_assignments(body_str: str) -> dict[str, str]:
        """Parse ``type var = expr;`` assignments from body text.

        Returns ``{var_name: rhs_expression}`` for every CSE assignment
        found.  Skips lines that don't match the assignment pattern.
        """
        assignments: dict[str, str] = {}
        for line in body_str.splitlines():
            m = PointwiseMixin._DCE_ASSIGN_RE.match(line)
            if not m:
                continue
            lhs = m.group(1)
            rhs = m.group(2)
            assignments[lhs] = rhs
        return assignments

    @staticmethod
    def _dce_build_use_def(
        assignments: dict[str, str],
    ) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        """Build use-def and def-use chains from assignment map.

        Returns ``(uses, defs)`` where:
        - ``uses[var]`` = set of variables that ``var`` references on its RHS
        - ``defs[rhs_var]`` = set of variables whose RHS references ``rhs_var``
        """
        uses: dict[str, set[str]] = {}
        defs: dict[str, set[str]] = {}
        for lhs, rhs in assignments.items():
            # Extract variable-like tokens from RHS
            rhs_vars: set[str] = set()
            for tok in re.findall(r"\b([a-zA-Z_]\w*)\b", rhs):
                if tok in assignments:
                    rhs_vars.add(tok)
            uses[lhs] = rhs_vars
            for rv in rhs_vars:
                defs.setdefault(rv, set()).add(lhs)
        return uses, defs

    @staticmethod
    def _dce_compute_live_set(
        body_str: str,
        assignments: dict[str, str],
        uses: dict[str, set[str]],
        defs: dict[str, set[str]],
    ) -> set[str]:
        """Compute the set of live variables via reverse reachability.

        A variable is live if:
        1. It appears in a non-assignment context (e.g. store, function
           call argument, condition) AND that context is reachable, OR
        2. It is used by another live variable, OR
        3. It is in the always-live set (built-in IDs, range-tree vars).

        We seed the live set with variables referenced in non-assignment
        lines and always-live tokens, then transitively close.
        """
        live: set[str] = set(PointwiseMixin._DCE_ALWAYS_LIVE)

        # Seed: any variable referenced in a non-assignment line is live.
        # A non-assignment line is any line that doesn't match _DCE_ASSIGN_RE.
        for line in body_str.splitlines():
            if PointwiseMixin._DCE_ASSIGN_RE.match(line):
                continue
            # Extract tokens that look like known CSE variables
            for tok in re.findall(r"\b([a-zA-Z_]\w*)\b", line):
                if tok in assignments:
                    live.add(tok)

        # Transitively close: any variable that defines a live variable
        # is itself live.
        changed = True
        while changed:
            changed = False
            for lhs in list(live):
                for dep in uses.get(lhs, set()):
                    if dep not in live:
                        live.add(dep)
                        changed = True

        return live

    def _eliminate_dead_code(self, body_str: str) -> str:
        """Eliminate dead CSE variable assignments from body text.

        Returns ``body_str`` with dead assignments removed.  A dead
        assignment is one whose LHS is never referenced by any live
        computation or output store.

        Gated by ``TORCH_VULKAN_DCE=1`` (default: 1).  Set to 0 to
        disable for debugging.
        """
        from .. import config

        if not config.dce_enabled():
            return body_str

        assignments = self._dce_parse_assignments(body_str)
        if len(assignments) < 2:
            return body_str  # Nothing to eliminate

        uses, defs = self._dce_build_use_def(assignments)
        live = self._dce_compute_live_set(body_str, assignments, uses, defs)

        # Filter out dead assignment lines
        lines = body_str.splitlines()
        result: list[str] = []
        for line in lines:
            m = self._DCE_ASSIGN_RE.match(line)
            if m:
                lhs = m.group(1)
                if lhs not in live:
                    continue  # Dead — skip this line
            result.append(line)
        return "\n".join(result)

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
            r"\b(" + "|".join(re.escape(n) for n in all_inners) + r")\s*\[\s*(\w+)\s*\]"
        )
        for m in buf_access_re.finditer(body_str):
            idx_var = m.group(2)
            # Check if idx_var or any of its transitive deps reference lane IDs
            closure = self._transitive_dep_closure(deps, {idx_var})
            if "__lane_id__" in closure:
                return True
        return False

    # ── M23: Vec4 eligibility — structural + legacy fallback ──────

    def _vec4_pw_eligible_structural(
        self,
        rt: Any,
        all_inners: list[str],
        out_inners: set[str],
    ) -> Optional[bool]:
        """Use BlockPatternMatcher to determine vec4 eligibility structurally.

        Analyzes collected sympy index expressions from ``_pw_index_records``
        to determine whether every I/O buffer has stride-1 contiguous access
        along the innermost iteration dimension *rt*.

        Returns:
            ``True``  — all buffers have stride-1 coalesced access; vec4 OK.
            ``False`` — at least one buffer definitively does NOT have
                        stride-1 access; vec4 ineligible.
            ``None``  — structural analysis could not determine eligibility
                        (e.g. no index records available); caller should
                        fall back to string-matching.
        """
        if not self._pw_index_records:
            logger.debug(
                "M23: vec4 structural → None (no _pw_index_records); "
                "falling back to legacy string-matching."
            )
            return None

        # Verify every declared I/O buffer is actually referenced.
        indexed_bufs = {name for name, _, _ in self._pw_index_records}
        for inner in all_inners:
            if inner not in indexed_bufs:
                logger.debug(
                    "M23: vec4 structural → False: buffer %s declared "
                    "but never loaded/stored.",
                    inner,
                )
                return False

        # Classify each buffer as vec4-coalesced or broadcast w.r.t. *rt*.
        # This is the core BlockPatternMatcher analysis:
        #   1. Extract subexpression involving rt's symbol
        #   2. Match as affine (stride × symbol)
        #   3. Verify stride == 1
        #   4. Verify no other range-tree symbols leak in
        classification = self._classify_pw_buffers_by_axis(rt)
        if classification is None:
            logger.debug(
                "M23: vec4 structural → False: buffer classification "
                "failed (non-affine index, non-unit stride, mixed access, "
                "or foreign range-tree symbol)."
            )
            return False

        vec4_bufs, broadcast_bufs = classification
        if not vec4_bufs:
            logger.debug(
                "M23: vec4 structural → False: no vec4-coalesced buffers "
                "(all broadcast)."
            )
            return False

        # Output buffers MUST be vec4-coalesced — the rewrite assumes
        # ``out[gtid.x] = _v_out`` writeback at stride 4.
        if any(inner not in vec4_bufs for inner in out_inners):
            logger.debug(
                "M23: vec4 structural → False: output buffer is not "
                "in vec4-coalesced set."
            )
            return False

        logger.debug(
            "M23: vec4 structural → True: %d vec4 buffer(s), %d broadcast buffer(s).",
            len(vec4_bufs),
            len(broadcast_bufs),
        )
        return True

    def _vec4_pw_eligible_legacy(
        self,
        body_str: str,
        rt: Any,
        all_inners: list[str],
    ) -> bool:
        """String-matching fallback for vec4 eligibility.

        Used when structural analysis via BlockPatternMatcher cannot
        determine eligibility (``_vec4_pw_eligible_structural`` returned
        ``None``).  Performs:

        1. Lane/thread-ID dependency check (transitive dep on lid.x/lid.y/ltid).
        2. Alias detection — finds variables that are direct aliases of the
           range-tree variable (``uint x = xindex;``).

        Returns ``True`` if string-based checks pass, ``False`` otherwise.
        """
        logger.debug(
            "M23: vec4 legacy string-matching fallback active for rt=%s.",
            rt.name,
        )

        # Lane-ID dependency: if any buffer index depends on lid.x/lid.y/ltid,
        # vec4 would pick wrong elements.
        if self._check_index_lane_dependency(body_str, "", all_inners):
            logger.debug("M23: vec4 legacy → False: lane-id dependency detected.")
            return False

        # Compute alias set via precise regex matching.
        # An alias is a variable directly assigned from rt_name
        # with no operators (e.g. uint x = xindex;).
        # Derived symbols like x0 = xindex / 512 are NOT aliases.
        rt_name = rt.name
        direct_alias_re = re.compile(
            r"^\s*uint\s+(\w+)\s*=\s*" + re.escape(rt_name) + r"\s*;\s*$"
        )
        aliases = {rt_name}
        for ln in body_str.splitlines():
            m = direct_alias_re.match(ln)
            if m:
                aliases.add(m.group(1))
        self._vec4_pw_aliases = tuple(aliases)

        logger.debug(
            "M23: vec4 legacy → True: aliases=%s.",
            tuple(sorted(aliases)),
        )
        return True

    def _vec4_pw_eligible(
        self,
        body_str: str,
        in_decls: list[tuple[str, str]],
        out_decls: list[tuple[str, str]],
        inout_decls: list[tuple[str, str]],
    ) -> bool:
        """Check eligibility for the vec4 pointwise path.

        Uses BlockPatternMatcher structural analysis (M23) as the primary
        decision mechanism; falls back to string-matching only when
        structural analysis cannot determine eligibility.

        All conditions must hold (any failure → scalar codegen, the safe
        default).
        """
        from .. import config

        if config.no_vec4_pointwise():
            return False
        if self.inside_reduction:
            return False
        non_red_trees = [t for t in self.range_trees if not t.is_reduction]

        # ── packed16 (half-precision) vec4 path ──────────────────────
        if self._packed16 is True:
            if not self._packed16_vw_active:
                return False
            all_decls = in_decls + out_decls + inout_decls
            if not all_decls:
                return False
            for dt, _ in all_decls:
                if dt != "uint":
                    return False
            if self.args.workspace_args:
                return False

            # Structural operation checks.
            if self._pw_has_atomic_op:
                return False
            if self._pw_has_early_return:
                return False
            if self._pw_has_wave_ops:
                return False
            if self._pw_uses_groupshared:
                return False
            if self._pw_has_scan_or_linear:
                return False

            if len(non_red_trees) != 1:
                return False
            rt = non_red_trees[0]
            if not isinstance(rt.numel, sympy.Integer):
                return False
            if int(rt.numel) == 0:
                return False

            all_inners_p16 = [n for _, n in all_decls]
            out_inners_p16 = {n for _, n in out_decls + inout_decls}

            # M23: Try structural analysis first.
            structural = self._vec4_pw_eligible_structural(
                rt, all_inners_p16, out_inners_p16
            )
            if structural is not None:
                if not structural:
                    return False
            else:
                # Structural can't determine — fall back to legacy.
                # _vec4_pw_eligible_legacy validates lane-id
                # dependency internally.
                if not self._vec4_pw_eligible_legacy(body_str, rt, all_inners_p16):
                    return False

            if not self._p16_load_records or not self._p16_store_records:
                return False
            return True

        # ── float vec4 path ──────────────────────────────────────────
        if self.has_welford:
            return False
        if len(non_red_trees) != 1:
            return False
        rt = non_red_trees[0]
        if not isinstance(rt.numel, sympy.Integer):
            return False
        n = int(rt.numel)
        thr4 = self.max_threadgroup_size * 4
        if n == 0 or n % thr4 != 0:
            return False

        # All I/O buffers must be plain `float`. Half / packed / atomic /
        # workspace decls have different types and would alias incorrectly
        # under a `float4` reinterpretation.
        all_decls = in_decls + out_decls + inout_decls
        if not all_decls:
            return False
        for dt, _ in all_decls:
            if dt != "float":
                return False
        if self.args.workspace_args:
            return False

        # Structural operation checks.
        if self._pw_has_atomic_op:
            return False
        if self._pw_has_early_return:
            return False
        if self._pw_has_wave_ops:
            return False
        if self._pw_uses_groupshared:
            return False
        if self._pw_uses_subbyte_packing:
            return False
        if self._pw_has_scan_or_linear:
            return False

        # Verify the body will have the expected single-axis iteration
        # pattern (uint <rt> = gtid.x;). This is guaranteed when the
        # problem size fits in a single dispatch dimension (one_d).
        # See HeaderMixin.codegen_kernel for the one_d condition.
        if n > 65535 * self.max_threadgroup_size:
            return False

        all_inners_f32 = [n for _, n in all_decls]
        out_inners_f32 = {n for _, n in out_decls + inout_decls}

        # M23: Try structural analysis first.
        structural = self._vec4_pw_eligible_structural(
            rt, all_inners_f32, out_inners_f32
        )
        if structural is False:
            logger.debug("M23: vec4 eligibility → False (structural).")
            return False
        if structural is None:
            # Structural can't determine — fall back to legacy string-matching.
            logger.debug("M23: vec4 structural returned None; falling back to legacy.")
            if not self._vec4_pw_eligible_legacy(body_str, rt, all_inners_f32):
                return False
            # Legacy path also validates lane-id dependency (already done
            # by _vec4_pw_eligible_legacy above, which sets _vec4_pw_aliases).
        else:
            # Structural passed — alias detection is still needed for the
            # string-based rewrite in _vec4_pw_rewrite.  Detect aliases
            # here since the structural path skipped the legacy method.
            rt_name = rt.name
            direct_alias_re = re.compile(
                r"^\s*uint\s+(\w+)\s*=\s*" + re.escape(rt_name) + r"\s*;\s*$"
            )
            aliases = {rt_name}
            for ln in body_str.splitlines():
                m = direct_alias_re.match(ln)
                if m:
                    aliases.add(m.group(1))
            self._vec4_pw_aliases = tuple(aliases)
            logger.debug(
                "M23: vec4 structural passed; aliases=%s.",
                tuple(sorted(aliases)),
            )

        logger.debug("M23: vec4 eligibility → True.")
        return True

    # ── BlockPatternMatcher structural analysis helpers ──────────────

    def _check_stride_one_coalescing(self, rt: Any) -> bool:
        """Verify all collected buffer indices are stride-1 w.r.t. *rt*.

        Uses BlockPatternMatcher to structurally analyze each recorded
        load/store index expression:

        1. Extract the subexpression involving *rt*'s symbol.
        2. Match it as an affine expression (stride × symbol).
        3. Verify the stride is exactly 1.
        4. Verify no other range-tree symbols leak into the index.

        Broadcast loads (index with no rt symbol) are allowed as they
        are trivially stride-1 — but they must NOT mix with vec4-strided
        accesses on the same buffer (slangc rejects scalar indexing into
        a ``StructuredBuffer<float4>``).
        """
        classification = self._classify_pw_buffers_by_axis(rt)
        if classification is None:
            return False
        return True

    def _classify_pw_buffers_by_axis(
        self, rt: Any
    ) -> Optional[tuple[set[str], set[str]]]:
        """Classify each I/O buffer as vec4-coalesced or broadcast w.r.t. *rt*.

        Returns ``(vec4_bufs, broadcast_bufs)`` on success, or ``None`` if any
        buffer fails the structural checks (non-rt range-tree symbol leaked
        in, non-affine index, non-unit stride, or mixed access patterns on
        the same buffer — vec4-strided AND broadcast loads of the same
        underlying buffer would force a single binding to be both
        ``StructuredBuffer<float4>`` and scalar-indexed, which slangc rejects).
        """
        sym_to_tree: dict[sympy.Symbol, Any] = {}
        for sym, node in self.range_tree_nodes.items():
            sym_to_tree[sym] = node.parent

        per_buf_kinds: dict[str, set[str]] = {}
        for buf_name, index_expr, _is_load in self._pw_index_records:
            idx_syms = index_expr.free_symbols

            seen_trees: set[int] = set()  # use id() — IterationRangesRoot unhashable
            rt_sym_in_idx: Optional[sympy.Symbol] = None
            for s in idx_syms:
                tree = sym_to_tree.get(s)
                if tree is not None:
                    seen_trees.add(id(tree))
                    if tree is rt:
                        rt_sym_in_idx = s

            if seen_trees - {id(rt)}:
                return None

            if rt_sym_in_idx is None:
                # No rt symbol → broadcast / constant index.
                per_buf_kinds.setdefault(buf_name, set()).add("broadcast")
                continue

            sub = BlockPatternMatcher.get_subexpr_involving_symbol(
                index_expr, rt_sym_in_idx
            )
            if sub is None or sub == sympy.S.Zero:
                per_buf_kinds.setdefault(buf_name, set()).add("broadcast")
                continue

            stride = BlockPatternMatcher.match_affine_block_expr(sub, rt_sym_in_idx)
            if stride is None:
                return None

            try:
                if not V.graph.sizevars.statically_known_equals(stride, sympy.S.One):
                    return None
            except Exception:
                return None

            per_buf_kinds.setdefault(buf_name, set()).add("vec4")

        vec4_bufs: set[str] = set()
        broadcast_bufs: set[str] = set()
        for buf, kinds in per_buf_kinds.items():
            if kinds == {"vec4"}:
                vec4_bufs.add(buf)
            elif kinds == {"broadcast"}:
                broadcast_bufs.add(buf)
            else:
                # Mixed — same buffer is loaded both vec4-coalesced and
                # broadcast-scalar.  Cannot pick a single binding type.
                return None
        return vec4_bufs, broadcast_bufs

    def _vec4_pw_rewrite(
        self,
        body_str: str,
        in_decls: list[tuple[str, str]],
        out_decls: list[tuple[str, str]],
        inout_decls: list[tuple[str, str]],
    ) -> Optional[IndentedBuffer]:
        """Rewrite a scalar pointwise body into vec4 form.

        Pre-conditions: `_vec4_pw_eligible` returned True.

        Returns a new IndentedBuffer with absolute indentation matching
        the original (so the downstream `code.splice(body_code)` keeps
        producing correctly-indented Slang). Returns None on failure.
        """
        if self._packed16 is True:
            return self._packed16_vw_rewrite(body_str, in_decls, out_decls, inout_decls)
        non_red_trees = [t for t in self.range_trees if not t.is_reduction]
        rt = non_red_trees[0]
        rt_name = rt.name
        in_inners = [n for _, n in in_decls]
        out_inners = [n for _, n in out_decls]
        inout_inners = [n for _, n in inout_decls]
        all_inners = in_inners + out_inners + inout_inners

        # Per-buffer classification: vec4 buffers get a float4 prefetch and
        # binding-type rewrite; broadcast buffers stay scalar `<float>`
        # bindings and are loaded inside the unroll loop unchanged.
        classification = self._classify_pw_buffers_by_axis(rt)
        if classification is None:
            return None
        vec4_bufs, _broadcast_bufs = classification

        rewritten = body_str
        aliases = getattr(self, "_vec4_pw_aliases", (rt_name,))
        for inner in all_inners:
            if inner not in vec4_bufs:
                continue
            for a in aliases:
                rewritten = rewritten.replace(
                    f"{inner}[((int)({a}))]", f"_v_{inner}[_k]"
                )
                rewritten = rewritten.replace(f"{inner}[{a}]", f"_v_{inner}[_k]")

        anchor = f"uint {rt_name} = gtid.x;"
        lines = rewritten.splitlines()
        anchor_idx = next(
            (i for i, ln in enumerate(lines) if ln.strip() == anchor),
            -1,
        )
        if anchor_idx < 0:
            return None

        head_lines = [ln.strip() for ln in lines[:anchor_idx] if ln.strip()]
        tail_lines = [ln.strip() for ln in lines[anchor_idx + 1 :] if ln.strip()]

        alias_decl = re.compile(r"^uint\s+(\w+)\s*=\s*" + re.escape(rt_name) + r"\s*;$")
        live_tail: list[str] = []
        for i, ln in enumerate(tail_lines):
            m = alias_decl.match(ln)
            if m:
                alias = m.group(1)
                if not any(
                    re.search(r"\b" + re.escape(alias) + r"\b", other)
                    for j, other in enumerate(tail_lines)
                    if j != i
                ):
                    continue
            live_tail.append(ln)

        # Only promote buffers whose body accesses will all be rewritten
        # to _v_ prefetch form. A buffer classified as "vec4" by the
        # sympy analysis may still have scalar accesses via derived
        # symbols (e.g. x1 = (xindex // N) % M) that aren't rt_name or
        # its aliases. Those accesses are NOT rewritten by the string
        # replacements above, so declaring the buffer as float4 would
        # cause slangc to reject the scalar indexing (TRAIN.5).
        # Build the set of all index patterns that the rewrite handles.
        _rewritable_idx: set[str] = {rt_name} | set(aliases)
        _rewritable_idx.add(f"((int)({rt_name}))")
        # Scan live_tail for non-rewritable accesses to each vec4 buffer.
        _buf_re = re.compile(r"\b(\w+)\s*\[")  # buffer_name[idx...
        for inner in list(vec4_bufs):
            for ln in live_tail:
                for m in _buf_re.finditer(ln):
                    if m.group(1) == inner:
                        # Extract the index expression by finding matching ']'
                        start = m.end()
                        depth = 1
                        end = start
                        while end < len(ln) and depth > 0:
                            if ln[end] == "[":
                                depth += 1
                            elif ln[end] == "]":
                                depth -= 1
                            end += 1
                        idx_str = ln[start : end - 1].strip()
                        # If idx not rewritable, this buffer has scalar
                        # accesses that won't be rewritten -> not vec4.
                        if idx_str not in _rewritable_idx:
                            vec4_bufs.discard(inner)
                            break
                if inner not in vec4_bufs:
                    break

        for inner in all_inners:
            if inner in vec4_bufs:
                self._vec4_pw_bufs.add(inner)

        axis_used = any(
            re.search(r"\b" + re.escape(rt_name) + r"\b", ln) for ln in live_tail
        )

        new_buf = IndentedBuffer()
        with new_buf.indent():
            for ln in head_lines:
                new_buf.writeline(ln)
            if axis_used:
                new_buf.writeline("uint xbase = gtid.x * 4u;")
            for inner in in_inners:
                if inner in vec4_bufs:
                    new_buf.writeline(f"float4 _v_{inner} = {inner}[gtid.x];")
            for inner in inout_inners:
                if inner in vec4_bufs:
                    new_buf.writeline(f"float4 _v_{inner} = {inner}[gtid.x];")
            for inner in out_inners:
                if inner in vec4_bufs:
                    new_buf.writeline(f"float4 _v_{inner};")
            new_buf.writeline("[unroll] for (uint _k = 0u; _k < 4u; ++_k) {")
            with new_buf.indent():
                if axis_used:
                    new_buf.writeline(f"uint {rt_name} = xbase + _k;")
                for ln in live_tail:
                    new_buf.writeline(ln)
            new_buf.writeline("}")
            for inner in out_inners:
                if inner in vec4_bufs:
                    new_buf.writeline(f"{inner}[gtid.x] = _v_{inner};")
            for inner in inout_inners:
                if inner in vec4_bufs:
                    new_buf.writeline(f"{inner}[gtid.x] = _v_{inner};")
        return new_buf

    def _packed16_vw_rewrite(
        self,
        body_str: str,
        in_decls: list[tuple[str, str]],
        out_decls: list[tuple[str, str]],
        inout_decls: list[tuple[str, str]],
    ) -> Optional[IndentedBuffer]:
        """Rewrite a scalar packed16 pointwise body into vectorized form.

        Each thread processes 4 consecutive half-precision elements
        (2 uint32 words).  Loaded elements are pre-unpacked into float
        scratch arrays before the compute loop, and results are packed
        back and written after the loop — eliminating the wave-based
        pack/unpack sequence and halving the workgroup count.

        Returns a new IndentedBuffer, or None on failure.
        """
        non_red_trees = [t for t in self.range_trees if not t.is_reduction]
        rt_name = non_red_trees[0].name
        in_inners = [n for _, n in in_decls]
        out_inners = [n for _, n in out_decls]
        inout_inners = [n for _, n in inout_decls]

        suffix = "f16" if self._packed16_dtype == torch.float16 else "bf16"
        pfx = "f16" if suffix == "f16" else "bf16"

        in_bufs = {r[1] for r in self._p16_load_records}
        out_bufs = {r[0] for r in self._p16_store_records}

        anchor = f"uint {rt_name} = gtid.x;"
        lines = body_str.splitlines()
        anchor_idx = next(
            (i for i, ln in enumerate(lines) if ln.strip() == anchor),
            -1,
        )
        if anchor_idx < 0:
            return None

        head_lines = [ln for ln in lines[:anchor_idx] if ln.strip()]
        tail_lines = [ln for ln in lines[anchor_idx + 1 :] if ln.strip()]

        alias_decl = re.compile(r"^uint\s+(\w+)\s*=\s*" + re.escape(rt_name) + r"\s*;$")
        live_tail: list[str] = []
        for i, ln in enumerate(tail_lines):
            m = alias_decl.match(ln)
            if m:
                alias = m.group(1)
                if not any(
                    re.search(r"\b" + re.escape(alias) + r"\b", other)
                    for j, other in enumerate(tail_lines)
                    if j != i
                ):
                    continue
            live_tail.append(ln)

        load_line_re = re.compile(
            r"float\s+(\w+)\s*=\s*_vk_unpack_(f16|bf16)\((\w+)\[\([^)]+\)\s*>>\s*1u\],"
        )
        store_line_re = re.compile(r".*\b_vk_pack_(f16|bf16)\b.*")

        rewritten: list[str] = []
        for ln in live_tail:
            ml = load_line_re.search(ln)
            if ml:
                cse_name = ml.group(1)
                buf_name = ml.group(3)
                rewritten.append(f"float {cse_name} = _pvw_in_{buf_name}[_k];")
                continue
            if store_line_re.search(ln):
                for buf_inner, value_cse, _sfx in self._p16_store_records:
                    if (
                        buf_inner in ln
                        and f"_vk_pack_{_sfx}((float)({value_cse})," in ln
                    ):
                        rewritten.append(
                            f"_pvw_out_{buf_inner}[_k] = (float)({value_cse});"
                        )
                        break
                continue
            rewritten.append(ln)

        axis_used = any(
            re.search(r"\b" + re.escape(rt_name) + r"\b", ln) for ln in rewritten
        )

        new_buf = IndentedBuffer()
        with new_buf.indent():
            for ln in head_lines:
                new_buf.writeline(ln)
            new_buf.writeline(f"uint xbase = gtid.x * 4u;")

            for buf_inner in sorted(in_bufs):
                new_buf.writeline(f"float _pvw_in_{buf_inner}[4];")
                new_buf.writeline("{")
                with new_buf.indent():
                    new_buf.writeline(f"uint _pvw_w0 = {buf_inner}[gtid.x * 2u];")
                    new_buf.writeline(f"uint _pvw_w1 = {buf_inner}[gtid.x * 2u + 1u];")
                    new_buf.writeline(
                        f"_pvw_in_{buf_inner}[0] = _vk_unpack_{pfx}(_pvw_w0, 0u);"
                    )
                    new_buf.writeline(
                        f"_pvw_in_{buf_inner}[1] = _vk_unpack_{pfx}(_pvw_w0, 1u);"
                    )
                    new_buf.writeline(
                        f"_pvw_in_{buf_inner}[2] = _vk_unpack_{pfx}(_pvw_w1, 0u);"
                    )
                    new_buf.writeline(
                        f"_pvw_in_{buf_inner}[3] = _vk_unpack_{pfx}(_pvw_w1, 1u);"
                    )
                new_buf.writeline("}")

            for buf_inner in sorted(out_bufs):
                new_buf.writeline(f"float _pvw_out_{buf_inner}[4];")

            new_buf.writeline("[unroll] for (uint _k = 0u; _k < 4u; ++_k) {")
            with new_buf.indent():
                if axis_used:
                    new_buf.writeline(f"uint {rt_name} = xbase + _k;")
                for ln in rewritten:
                    new_buf.writeline(ln)
            new_buf.writeline("}")

            for buf_inner in sorted(out_bufs):
                new_buf.writeline(
                    f"{buf_inner}[gtid.x * 2u] = "
                    f"_vk_pack_{pfx}(_pvw_out_{buf_inner}[0], _pvw_out_{buf_inner}[1]);"
                )
                new_buf.writeline(
                    f"{buf_inner}[gtid.x * 2u + 1u] = "
                    f"_vk_pack_{pfx}(_pvw_out_{buf_inner}[2], _pvw_out_{buf_inner}[3]);"
                )

        return new_buf

    # ── GPU.5: Persistent pointwise micro-batching ─────────────────

    def _enable_persistent_mode(self) -> None:
        """Enable grid-stride-loop wrapping for this kernel.

        Called by the scheduler when a chain of small pointwise ops
        is detected.  When enabled, the body is wrapped in a for-loop
        so each thread processes multiple elements.
        """
        self._persistent_mode = True

    def _emit_persistent_grid_stride_loop(self) -> Optional[str]:
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

        # GPU.5+: Relaxed threshold — consider per-thread work, not just total.
        # Check if total numel is too large for efficient persistent execution.
        # Use the same heuristic as _is_small_pointwise_chain:
        # per_thread_iters = total / (wg_size * target_wgs) <= 16 for 2 ops,
        # scaled up for more ops. Since we don't have the op count here,
        # use a conservative cap: total <= 16384 (matching _is_small_pointwise_chain).
        if total > 16384:
            # Too large for persistent kernel — overhead of grid-stride
            # loop dispatch (finding op per element) outweighs benefit.
            return None

        wg_size = self.max_threadgroup_size
        # For persistent kernels, use fewer WGs to keep the kernel
        # resident longer and amortize the per-op lookup overhead.
        num_wgs = max(1, min(20, (total + wg_size - 1) // wg_size))

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
        # Target ~20 workgroups to saturate typical GPU CUs.
        target_wgs = 20
        num_threads = wg_size * target_wgs

        total_numel = 0
        for sn in nodes:
            _, (numel, rnumel) = sn.group
            if rnumel != 1:
                return False  # has reduction
            if is_dynamic(numel):
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
        # Cap total numel at ~16K to stay safe.
        max_total_numel = 16384

        return per_thread_iters <= iter_cap and total_numel <= max_total_numel

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        var = self.args.input(name)
        index = self.prepare_indexing(index)
        dtype = V.graph.get_dtype(name)
        idx_str = self.index_to_str(index)

        # Track 5.7: Record sympy index for BlockPatternMatcher analysis.
        self._pw_index_records.append((var, index, True))

        if self._decide_packed16(dtype):
            self._pw_uses_subbyte_packing = True
            self._packed16_bufs.add(var)
            suffix = "f16" if dtype == torch.float16 else "bf16"
            self.headers.add(f"packed16_{suffix}")
            line = f"_vk_unpack_{suffix}({self._buf_path(var)}[({idx_str}) >> 1u], ({idx_str}) & 1u)"
            dtype = torch.float32
            cse_var = self.cse.generate(self.loads, line, dtype=dtype)
            self._p16_load_records.append((str(cse_var), var, suffix))
            return cse_var
        else:
            if (
                self._vec_width > 1
                and self.inside_reduction
                and self.multistage_reduction_entry
                and dtype in (torch.float32, torch.float16, torch.bfloat16)
                and self._reduction_type in ("sum", "prod", "max", "min")
            ):
                self.headers.add(f"vec4_reduce_{self._reduction_type}")
                rt = self._reduction_type
                if rt == "sum":
                    line = f"vk_vec4_hsum({self._buf_path(var)}, {idx_str})"
                elif rt == "max":
                    line = f"vk_vec4_hmax({self._buf_path(var)}, {idx_str})"
                elif rt == "min":
                    line = f"vk_vec4_hmin({self._buf_path(var)}, {idx_str})"
                else:
                    line = f"vk_vec4_hprod({self._buf_path(var)}, {idx_str})"
                if dtype != torch.float32:
                    line = f"((float)({line}))"
                    dtype = torch.float32
            else:
                _init_load_dispatch()
                # OP.1.b — external bool inputs are byte-packed.  PyTorch eager
                # allocates 1 byte per ``torch.bool`` element; the
                # ``dispatch_copy_buffer`` upload path (4 B/element) packs four
                # consecutive bool bytes into each uint32 slot of the SSBO.
                # Reading the slot as a single uint (the compile-internal
                # contract) returns 0x00010001 for ``[T,F,T,F]`` instead of the
                # per-element value.  For graph-input bool buffers we unpack
                # the byte at ``idx`` via ``_vk_unpack_u8`` (helpers.slang).
                # Compile-internal bool buffers (produced by a prior pointwise
                # STORE that writes 1 uint/element) keep the legacy
                # 1-uint-per-element read.
                if dtype == torch.bool and name in V.graph.graph_inputs:
                    self._pw_uses_subbyte_packing = True
                    self.headers.add("subdtype_unpack")
                    line = f"_vk_unpack_u8({self._buf_path(var)}, {idx_str})"
                    dtype = torch.float32
                else:
                    spec = _LOAD_DISPATCH.get(dtype)
                    if spec is not None:
                        emit_fn, hdr = spec
                        if hdr is not None:
                            self._pw_uses_subbyte_packing = True
                            self.headers.add(hdr)
                        line = emit_fn(self._buf_path(var), idx_str)
                        dtype = torch.float32
                    else:
                        line = f"{self._buf_path(var)}[{idx_str}]"

        from .. import config

        if (
            self.inside_reduction
            and self.multistage_reduction_entry
            and not getattr(self, "_partitioned_2d_active", False)
            and not self.has_welford
            and not config.no_load_hoist()
        ):
            key = (var, idx_str)
            cached = self.multistage_load_cache.get(key)
            if cached is not None:
                return self.cse.generate(self.loads, cached, dtype=dtype)
            root = self.multistage_reduction_entry[0].root
            if isinstance(root.numel, sympy.Integer):
                _hoist_stride = self.max_threadgroup_size * self._vec_width
                loop_size = (int(root.numel) + _hoist_stride - 1) // _hoist_stride
                # P5.4: Load-hoist threshold keyed on dtype and simd size.
                # Smaller dtypes → smaller per-element register footprint →
                # larger cache tolerated.  Smaller simd → more per-lane
                # registers available → larger cache.  Cap at 256 to prevent
                # pathological register spilling on RDNA1 (64 VGPRs/SIMD).
                _elt_bytes = 2 if self._packed16 else 4
                if dtype == torch.float64:
                    _elt_bytes = 8
                _dtype_scale = 4.0 / _elt_bytes
                _simd_scale = 64.0 / self.simd_group_size
                _base_limit = int(64 * _dtype_scale * _simd_scale)
                hoist_limit = min(max(_base_limit, 32), 256)
                if loop_size > hoist_limit:
                    return self.cse.generate(self.loads, line, dtype=dtype)
                arr_name = f"_ml_cache_{next(self.multistage_load_seq)}"
                cnt_name = f"{root.prefix}_cnt"
                self.indexing_code.writeline(f"float {arr_name}[{loop_size}];")
                cse_var = self.cse.generate(self.loads, line, dtype=dtype)
                self.loads.writeline(f"{arr_name}[{cnt_name}] = {cse_var};")
                self.multistage_load_cache[key] = f"{arr_name}[{cnt_name}]"
                return cse_var

        return self.cse.generate(self.loads, line, dtype=dtype)

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
        if out_dtype == torch.bool:
            # Comparison ops produce Slang bool expressions.  Bool output
            # buffers are declared as StructuredBuffer<uint> (1-uint-per-element),
            # so an explicit cast is always required.
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
            red_numel, _ = self._compute_red_numel()
            if red_numel < self.max_threadgroup_size:
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
