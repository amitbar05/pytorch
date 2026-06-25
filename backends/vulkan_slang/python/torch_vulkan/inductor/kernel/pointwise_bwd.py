"""Pointwise bwd_diff and DCE mixin — extracted from pointwise.py (CG.5 anti-goal #7 split)."""

import re

import torch


class PointwiseBwdMixin:
    """Mixin providing bwd_diff registration/emission and DCE for pointwise kernels."""

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

    # Variable dependency tracking for vec4 eligibility:
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

    # Dead code elimination (DCE): after body codegen, scan the generated Slang for CSE
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
            m = PointwiseBwdMixin._DCE_ASSIGN_RE.match(line)
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
        live: set[str] = set(PointwiseBwdMixin._DCE_ALWAYS_LIVE)

        # Seed: any variable referenced in a non-assignment line is live.
        # A non-assignment line is any line that doesn't match _DCE_ASSIGN_RE.
        for line in body_str.splitlines():
            if PointwiseBwdMixin._DCE_ASSIGN_RE.match(line):
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
