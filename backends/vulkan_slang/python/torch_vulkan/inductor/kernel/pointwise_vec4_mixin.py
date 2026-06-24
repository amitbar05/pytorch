"""Pointwise vec4 mixin — vec4 eligibility analysis and body rewrites.

Extracted from ``PointwiseMixin`` (M15.1.d — Track 1 anti-goal #7 split).
"""

import logging
import re
from typing import Any

import sympy
import torch
from torch._inductor.codegen.block_analysis import BlockPatternMatcher
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.virtualized import V


logger = logging.getLogger(__name__)


class PointwiseVec4Mixin:
    """Mixin providing vec4 vectorization logic for pointwise kernels.

    Handles eligibility analysis (structural via BlockPatternMatcher,
    legacy string-matching fallback), buffer classification by stride
    pattern, and the Scalar→vec4 body rewrite (including the packed16
    variant).
    """

    # ── M23: Vec4 eligibility — structural + legacy fallback ──────

    def _vec4_pw_eligible_structural(
        self,
        rt: Any,
        all_inners: list[str],
        out_inners: set[str],
    ) -> bool | None:
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

            if self.has_welford:
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

    def _classify_pw_buffers_by_axis(self, rt: Any) -> tuple[set[str], set[str]] | None:
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
            rt_sym_in_idx: sympy.Symbol | None = None
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
    ) -> IndentedBuffer | None:
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

        aliases = getattr(self, "_vec4_pw_aliases", (rt_name,))
        # Build the full set of index expressions that the rewrite can handle.
        # These include both bare aliases ("xindex") and (int)-cast forms
        # ("((int)(xindex))") that the scalar load path emits.
        _rewritable_idx_full: set[str] = set(aliases)
        for a in aliases:
            _rewritable_idx_full.add(f"((int)({a}))")
        rewritten = self._apply_vec4_body_rewrite(
            body_str, vec4_bufs, _rewritable_idx_full
        )

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
        # its aliases. Those accesses are NOT rewritten by the line-by-line
        # rewrite above, so declaring the buffer as float4 would
        # cause slangc to reject the scalar indexing (TRAIN.5).
        # Scan live_tail for non-rewritable accesses to each vec4 buffer.
        for inner in list(vec4_bufs):
            if self._line_has_non_rewritable_access(
                inner, live_tail, _rewritable_idx_full
            ):
                vec4_bufs.discard(inner)

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
                    new_buf.writeline(
                        f"float4 _v_{inner} = {self._buf_path(inner)}[gtid.x];"
                    )
            for inner in inout_inners:
                if inner in vec4_bufs:
                    new_buf.writeline(
                        f"float4 _v_{inner} = {self._buf_path(inner)}[gtid.x];"
                    )
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
                    new_buf.writeline(f"{self._buf_path(inner)}[gtid.x] = _v_{inner};")
            for inner in inout_inners:
                if inner in vec4_bufs:
                    new_buf.writeline(f"{self._buf_path(inner)}[gtid.x] = _v_{inner};")
        return new_buf

    # ── CG.M17: Typed vec4 rewrite helpers ─────────────────────────

    def _apply_vec4_body_rewrite(
        self,
        body_str: str,
        vec4_bufs: set[str],
        rewritable_idx: set[str],
    ) -> str:
        """Rewrite scalar buffer accesses to vec4 prefetch form, line by line.

        Replaces ``buf_name[rewritable_idx]`` with ``_v_buf_name[_k]`` on each
        line individually.  Uses regex with ``(?<!\\w)`` lookbehind to ensure
        buffer names match as whole identifiers, avoiding the substring-matching
        fragility of ``str.replace()``.
        """
        if not vec4_bufs:
            return body_str
        lines = body_str.splitlines()
        rewritten_lines: list[str] = [
            self._rewrite_line_vec4_accesses(ln, vec4_bufs, rewritable_idx)
            for ln in lines
        ]
        return "\n".join(rewritten_lines)

    @staticmethod
    def _rewrite_line_vec4_accesses(
        line: str,
        vec4_bufs: set[str],
        rewritable_idx: set[str],
    ) -> str:
        """Replace vec4-buffer accesses on a single line with prefetch form.

        Matches ``buf_name[index]`` (or ``args.buf_name[index]`` when
        ParameterBlock is enabled) where *buf_name* is a vec4 buffer and
        *index* (stripped) appears in *rewritable_idx*.  Uses
        ``(?<!\\w)`` to prevent matching substrings of longer identifiers.

        CG.M14: The optional ``args.`` prefix is consumed so the replacement
        is always the unprefixed ``_v_{buf_name}[_k]`` local variable.
        """
        _buf_access_re = re.compile(
            r"(?<!\w)(?:args\.)?("
            + "|".join(re.escape(b) for b in vec4_bufs)
            + r")\[([^\]]+)\]"
        )

        def _replace(m: re.Match) -> str:
            buf_name = m.group(1)
            idx = m.group(2).strip()
            if idx in rewritable_idx:
                return f"_v_{buf_name}[_k]"
            return m.group(0)

        return _buf_access_re.sub(_replace, line)

    @staticmethod
    def _line_has_non_rewritable_access(
        inner: str,
        lines: list[str],
        rewritable_idx: set[str],
    ) -> bool:
        """Checks if *inner* has any buffer access with a non-rewritable index.

        Scans each line for ``inner[index]`` patterns.  Returns ``True`` if any
        index expression is NOT in *rewritable_idx* \u2014 meaning the line cannot
        be converted to vec4 prefetch form and the buffer must be demoted to
        scalar binding type (TRAIN.5).
        """
        _buf_re = re.compile(r"(?<!\w)" + re.escape(inner) + r"\s*\[")
        for ln in lines:
            for m in _buf_re.finditer(ln):
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
                if idx_str not in rewritable_idx:
                    return True
        return False

    def _packed16_vw_rewrite(
        self,
        body_str: str,
        in_decls: list[tuple[str, str]],
        out_decls: list[tuple[str, str]],
        inout_decls: list[tuple[str, str]],
    ) -> IndentedBuffer | None:
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
            new_buf.writeline("uint xbase = gtid.x * 4u;")

            for buf_inner in sorted(in_bufs):
                buf_path = self._buf_path(buf_inner)
                new_buf.writeline(f"float _pvw_in_{buf_inner}[4];")
                new_buf.writeline("{")
                with new_buf.indent():
                    new_buf.writeline(f"uint _pvw_w0 = {buf_path}[gtid.x * 2u];")
                    new_buf.writeline(f"uint _pvw_w1 = {buf_path}[gtid.x * 2u + 1u];")
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
                buf_path = self._buf_path(buf_inner)
                new_buf.writeline(
                    f"{buf_path}[gtid.x * 2u] = "
                    f"_vk_pack_{pfx}(_pvw_out_{buf_inner}[0], _pvw_out_{buf_inner}[1]);"
                )
                new_buf.writeline(
                    f"{buf_path}[gtid.x * 2u + 1u] = "
                    f"_vk_pack_{pfx}(_pvw_out_{buf_inner}[2], _pvw_out_{buf_inner}[3]);"
                )

        return new_buf
