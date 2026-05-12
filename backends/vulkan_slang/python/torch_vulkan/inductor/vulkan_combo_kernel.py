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
"""

from __future__ import annotations

from typing import Optional

import sympy
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.virtualized import V

from .kernel import VulkanKernel
from .slang_helpers import emit_helpers

# Identifiers we leave alone (combo-kernel-scope or shader builtins).
_NEVER_RENAME = frozenset(
    {
        "gid",
        "lid",
        "gtid",
        "_vk_gtid",
        "_vk_gtid_local",
        "gtid_lin",
        "tid",
        "subgroup_id",
        "lane_id",
        # PF.13.b.4-CODG: _vk_linear / _vk_linear_orig are per-subkernel
        # scoped locals emitted by the combo-kernel seed for multi-dimensional
        # index decomposition.  They must not be renamed.
        "_vk_linear",
        "_vk_linear_orig",
    }
)

# Slang type keywords that introduce variable declarations.
_TYPE_KEYWORDS = frozenset(
    {
        "float",
        "int",
        "uint",
        "bool",
        "half",
        "double",
        "void",
        "int64_t",
        "float16_t",
        "int8_t",
        "uint8_t",
        "int16_t",
        "uint16_t",
        "float2",
        "float3",
        "float4",
        "int2",
        "int3",
        "int4",
        "uint2",
        "uint3",
        "uint4",
    }
)

# All known Slang keywords (superset of type keywords).
_KEYWORDS = _TYPE_KEYWORDS | frozenset(
    {
        "for",
        "while",
        "do",
        "if",
        "else",
        "switch",
        "case",
        "default",
        "return",
        "break",
        "continue",
        "static",
        "const",
        "struct",
        "uniform",
        "in",
        "out",
        "inout",
        "true",
        "false",
        "sizeof",
        "typedef",
        "StructuredBuffer",
        "RWStructuredBuffer",
        "ByteAddressBuffer",
        "RWByteAddressBuffer",
        "groupshared",
        "nointerpolation",
        "linear",
        "centroid",
        "sample",
        "discard",
        "cbuffer",
        "tbuffer",
        "register",
        "packoffset",
        "unroll",
        "loop",
        "branch",
        "flatten",
    }
)

# ---- lightweight Slang/HLSL tokenizer -----------------------------------

# Token type constants (small integers for fast comparison).
_T_KEYWORD = 1
_T_IDENT = 2
_T_OPERATOR = 3
_T_NUMBER = 4
_T_PUNCT = 5
_T_STRING = 6
_T_SPACE = 7
_T_COMMENT = 8


class _Token:
    """A single token from the Slang/HLSL source."""

    __slots__ = ("type", "value")

    def __init__(self, typ: int, value: str) -> None:
        self.type = typ
        self.value = value


def _tokenize(source: str):
    """Yield _Token objects from a Slang/HLSL source string.

    Handles: identifiers, keywords, numeric literals (decimal, hex, float
    with suffixes), string literals, line comments (``//``), block comments
    (``/* … */``), multi-character operators, and single-character
    punctuation. Whitespace is preserved as ``_T_SPACE`` tokens so the
    rewriter can reconstruct exact formatting.
    """
    i = 0
    n = len(source)

    while i < n:
        ch = source[i]

        # Whitespace — collapse contiguous runs into one token.
        if ch in " \t\n\r":
            j = i
            while j < n and source[j] in " \t\n\r":
                j += 1
            yield _Token(_T_SPACE, source[i:j])
            i = j
            continue

        # Line comment  // …
        if ch == "/" and i + 1 < n and source[i + 1] == "/":
            j = i
            while j < n and source[j] != "\n":
                j += 1
            yield _Token(_T_COMMENT, source[i:j])
            i = j
            continue

        # Block comment  /* … */
        if ch == "/" and i + 1 < n and source[i + 1] == "*":
            j = source.find("*/", i + 2)
            if j == -1:
                j = n
            else:
                j += 2
            yield _Token(_T_COMMENT, source[i:j])
            i = j
            continue

        # String literal  "…"  (with backslash escapes).
        if ch == '"':
            j = i + 1
            while j < n and source[j] != '"':
                if source[j] == "\\":
                    j += 1  # skip escaped char
                j += 1
            j += 1  # closing quote
            yield _Token(_T_STRING, source[i:j])
            i = j
            continue

        # Numeric literal: decimal / hex / float with optional suffix.
        if ch.isdigit() or (ch == "." and i + 1 < n and source[i + 1].isdigit()):
            j = i
            if source[j] == "0" and j + 1 < n and source[j + 1] in ("x", "X"):
                j += 2
                while j < n and source[j] in "0123456789abcdefABCDEF":
                    j += 1
            else:
                while j < n and source[j].isdigit():
                    j += 1
                if j < n and source[j] == ".":
                    j += 1
                    while j < n and source[j].isdigit():
                        j += 1
                if j < n and source[j] in ("e", "E"):
                    j += 1
                    if j < n and source[j] in "+-":
                        j += 1
                    while j < n and source[j].isdigit():
                        j += 1
            if j < n and source[j] in "fFuU":
                j += 1
            yield _Token(_T_NUMBER, source[i:j])
            i = j
            continue

        # Identifier or keyword.
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (source[j].isalnum() or source[j] == "_"):
                j += 1
            ident = source[i:j]
            typ = _T_KEYWORD if ident in _KEYWORDS else _T_IDENT
            yield _Token(typ, ident)
            i = j
            continue

        # Multi-character operators (longest match first).
        for op in (
            ">>=",
            "<<=",
            "<=",
            ">=",
            "==",
            "!=",
            "&&",
            "||",
            "++",
            "--",
            "<<",
            ">>",
            "+=",
            "-=",
            "*=",
            "/=",
            "%=",
            "&=",
            "|=",
            "^=",
        ):
            if source.startswith(op, i):
                yield _Token(_T_OPERATOR, op)
                i += len(op)
                break
        else:
            # Single-character operator or punctuation.
            if ch in "=+-*/%&|^~!<>?:.,;(){}[]":
                typ = _T_OPERATOR if ch in "=+-*/%&|^~!<>?:" else _T_PUNCT
                yield _Token(typ, ch)
            else:
                # Any unrecognized char — emit verbatim so we never drop bytes.
                yield _Token(_T_SPACE, ch)
            i += 1


# ---- local-variable classifier (zero regex) -----------------------------


def _is_buffer_name(name: str) -> bool:
    """Internal Inductor buffer slot names like in_ptr0 / out_ptr1 / inout_ptr2."""
    for prefix in ("in_ptr", "out_ptr", "inout_ptr"):
        if name.startswith(prefix) and name[len(prefix) :].isdigit():
            return True
    return False


def _is_local_to_rename(name: str) -> bool:
    """Return True if *name* is an Inductor-generated scalar local that needs
    per-subkernel namespacing.  Deliberately zero regex — uses only
    character-class inspection and ``str.isdigit``."""
    if name in _NEVER_RENAME:
        return False

    # xindex / yindex / rindex
    if name in ("xindex", "yindex", "rindex"):
        return True

    # tmp\d+   (tmp0, tmp1, ..., tmp123)
    if name.startswith("tmp") and len(name) > 3 and name[3:].isdigit():
        return True

    # x\d+ / y\d+   (x0, x1, y0, y1, ...)
    if len(name) >= 2 and name[0] in ("x", "y") and name[1:].isdigit():
        return True

    # r\d+_\d+   (r0_0, r1_123, ...)
    if name.startswith("r") and "_" in name:
        rest = name[1:]
        digit_part, sep, suffix = rest.partition("_")
        if sep and digit_part.isdigit() and suffix.isdigit():
            return True

    # r\d+_index   (r0_index, r1_index, ...)
    if name.startswith("r") and name.endswith("_index"):
        rest = name[1:-6]
        if rest.isdigit():
            return True

    # tmp_acc_\d+   (tmp_acc_0, tmp_acc_1, ...)
    if name.startswith("tmp_acc_") and len(name) > 8 and name[8:].isdigit():
        return True

    return False


# ---- token-based body rewriter ------------------------------------------


def _rewrite_body(
    body_src: str,
    name_map: dict[str, str],
    idx: int,
    cross_decls: list[dict[str, str]] | None = None,
    debug_assert: bool = False,
) -> tuple[str, dict[str, str]]:
    """Token-based rename of buffer names and Inductor-generated locals.

    Replaces the old regex approach with a lightweight tokenization pass
    that distinguishes declarations from references and never corrupts
    identifiers that happen to contain a matching substring (e.g.
    ``xindex_sub0``).

    Uses a single pass — safe because HLSL requires declarations to
    precede references within every scope.

    M16 additions:
      - Struct member access (``obj.field``) is preserved — identifiers
        following a ``.`` operator are never renamed.
      - ``debug_assert=True`` enables a collision check that verifies
        renamed variables don't shadow any existing declaration in the
        current scope chain.
    """
    tokens = list(_tokenize(body_src))
    parts: list[str] = []

    # Scope stack: each scope is a dict mapping original_name → renamed_name.
    # The top-level scope is pre-seeded with the index locals that the combo
    # kernel wrapper declares *outside* the body (xindex, x0, …).
    scopes: list[dict[str, str]] = [
        {
            n: f"{n}_sub{idx}"
            for n in ("xindex", "yindex", "rindex", "x0", "x1", "y0", "y1")
            if n not in _NEVER_RENAME
        }
    ]

    # Declaration-context state machine.
    # ``in_decl_stmt`` is True from a type keyword until a statement
    # terminator (``;``, ``{``, ``}``) or a non-type keyword.  This keeps
    # commas inside function-call argument lists from being mistaken for
    # declaration-list commas (e.g. ``foo(x, y)`` vs ``float x, y;``).
    in_decl_stmt = False
    saw_type = False  # just consumed a type keyword; next IDENT is a declarator

    # M16: track the previous non-space/non-comment token to detect
    # struct member access (``obj.field``) and array subscript context.
    # Identifiers following ``.`` are member names — never rename them.
    prev_significant_type: int | None = None
    prev_significant_value: str = ""

    for tok in tokens:
        if tok.type in (_T_SPACE, _T_COMMENT):
            parts.append(tok.value)
            continue

        if tok.type == _T_KEYWORD:
            parts.append(tok.value)
            if tok.value in _TYPE_KEYWORDS:
                in_decl_stmt = True
                saw_type = True
            else:
                # Non-type keyword (for, while, if, else, return, …)
                # ends any ongoing declaration statement.
                in_decl_stmt = False
                saw_type = False

        elif tok.type == _T_PUNCT:
            parts.append(tok.value)
            if tok.value == "{":
                scopes.append({})
                in_decl_stmt = False
                saw_type = False
            elif tok.value == "}":
                if len(scopes) > 1:
                    scopes.pop()
                in_decl_stmt = False
                saw_type = False
            elif tok.value == ";":
                in_decl_stmt = False
                saw_type = False
            elif tok.value == "," and in_decl_stmt:
                # Comma inside a declaration statement: next IDENT is
                # another declarator (e.g. ``float x, y, z;`` or
                # ``for (int i = 0, j = 0; …)``).
                saw_type = True
            # Other punctuation ``( ) [ ]`` does not affect decl tracking.
            # ``.`` is tracked via prev_significant_type below.

        elif tok.type == _T_OPERATOR:
            parts.append(tok.value)
            # Operators never affect declaration tracking — the ``=`` in
            # ``float x = …`` does *not* end the declaration statement
            # (commas after the initializer still extend the declarator list).

        elif tok.type == _T_IDENT:
            name = tok.value

            # M16: if the previous significant token was a ``.``, this
            # identifier is a struct/object member name — never rename it.
            is_member_access = (
                prev_significant_type == _T_PUNCT and prev_significant_value == "."
            )

            if is_member_access:
                # Member access: emit verbatim, do NOT track as declaration
                # or apply rename maps.
                parts.append(name)
                saw_type = False
            else:
                is_decl = saw_type and in_decl_stmt

                if is_decl:
                    # Declaration: record in current scope and emit.
                    if _is_local_to_rename(name):
                        new_name = f"{name}_sub{idx}"
                        # M16 debug assertion: verify renamed name doesn't
                        # collide with any existing declaration.
                        if debug_assert:
                            for scope in scopes:
                                if name in scope and scope[name] == new_name:
                                    continue  # same rename, ok
                            for scope in scopes:
                                if new_name in scope or new_name == name:
                                    raise RuntimeError(
                                        f"Combo kernel sub{idx}: renamed "
                                        f"variable '{name}' → '{new_name}' "
                                        f"collides with existing "
                                        f"declaration in scope."
                                    )
                        scopes[-1][name] = new_name
                        parts.append(new_name)
                    else:
                        scopes[-1][name] = name
                        parts.append(name)
                    saw_type = False  # first declarator consumed
                else:
                    # Reference: buffer rename map has priority, then scope chain.
                    if name in name_map and name_map[name] != name:
                        parts.append(name_map[name])
                    else:
                        renamed = name
                        for scope in reversed(scopes):
                            if name in scope:
                                renamed = scope[name]
                                break
                        if renamed == name and cross_decls:
                            # Check previous subkernels' declarations
                            for prev_scope in reversed(cross_decls):
                                if name in prev_scope:
                                    renamed = prev_scope[name]
                                    break
                        parts.append(renamed)

        else:
            # NUMBER, STRING — emit verbatim.
            parts.append(tok.value)

        prev_significant_type = tok.type
        prev_significant_value = tok.value

    all_decls: dict[str, str] = {}
    for scope in scopes:
        all_decls.update(scope)
    return "".join(parts), all_decls


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

        Each outer buffer name (the wrapper-visible buf name) gets one binding,
        even if multiple subkernels reference it. Inner names (`in_ptr0`,
        `out_ptr1`, `inout_ptr0`) collide across subkernels and get prefixed
        with `s{idx}_` if already taken. Inplace buffers are declared `RW` so
        the body's `<inner>[idx] = ...` writes are valid l-values.
        """
        import torch
        from torch._inductor.codegen.common import InplacedBuffer

        from .overrides import DTYPE_TO_SLANG

        outer_to_global: dict[str, str] = {}
        outer_is_rw: dict[str, bool] = {}
        in_decls: list[tuple[str, str, str]] = []
        rw_decls: list[tuple[str, str, str]] = []
        per_sub_maps: list[dict[str, str]] = []
        used_globals: set[str] = set()

        def _dtype_str(outer: str) -> str:
            dtype = V.graph.get_dtype(outer)
            if dtype in (torch.float16, torch.bfloat16):
                return "float"
            base = DTYPE_TO_SLANG.get(dtype, "float")
            # int64_t buffers must be declared as uint2 because the
            # pointwise store emits uint2(...) for int64 values (Slang
            # on Vulkan lacks native 64-bit integer atomics, so we
            # bitcast through uint2).  Matches _binding_dtype in
            # kernel/header.py.
            if base == "int64_t":
                return "uint2"
            return base

        # First pass: discover which outer buffers are used as outputs anywhere
        # (so we know to declare them as RW). An outer that appears in BOTH
        # input_buffers and output_buffers in the same subkernel is a
        # read-modify-write (inplace) — we must declare it RW even though
        # `inplace_buffers` is empty.
        outers_written: set[str] = set()
        for kernel, _ in self.subkernels:
            for outer, inplaced in kernel.args.inplace_buffers.items():
                if isinstance(inplaced, InplacedBuffer):
                    outers_written.add(outer)
            for outer in kernel.args.output_buffers:
                if outer in kernel.removed_buffers:
                    continue
                outers_written.add(outer)

        for idx, (kernel, _) in enumerate(self.subkernels):
            name_map: dict[str, str] = {}

            def _declare(outer: str, inner: str) -> None:
                if outer in outer_to_global:
                    name_map[inner] = outer_to_global[outer]
                    return
                # GAP-1.1-B: loop until we find a name not in used_globals.
                # The naive f"s{idx}_{inner}" can itself collide when
                # the same subkernel has TWO different outer buffers that
                # share the same inner name (e.g. both input and output
                # buffers named "in_out_ptr0" in subkernel 1).
                candidate = inner
                while candidate in used_globals:
                    candidate = f"s{idx}_{candidate}"
                global_name = candidate
                used_globals.add(global_name)
                outer_to_global[outer] = global_name
                name_map[inner] = global_name
                if outer in outers_written:
                    outer_is_rw[outer] = True
                    rw_decls.append((_dtype_str(outer), global_name, outer))
                else:
                    outer_is_rw[outer] = False
                    in_decls.append((_dtype_str(outer), global_name, outer))

            for outer, inplaced in kernel.args.inplace_buffers.items():
                if not isinstance(inplaced, InplacedBuffer):
                    continue
                _declare(outer, inplaced.inner_name)

            for outer, inner in kernel.args.input_buffers.items():
                if outer in kernel.args.inplace_buffers:
                    continue
                _declare(outer, inner)

            for outer, inner in kernel.args.output_buffers.items():
                if (
                    outer in kernel.removed_buffers
                    or outer in kernel.args.inplace_buffers
                ):
                    continue
                _declare(outer, inner)

            per_sub_maps.append(name_map)

        return in_decls, rw_decls, per_sub_maps

    def codegen_kernel(self) -> str:
        code = IndentedBuffer()
        in_decls, rw_decls, per_sub_maps = self._build_global_binding_map()
        self.n_inputs = len(in_decls)
        self.n_outputs = len(rw_decls)
        self.n_buffers = self.n_inputs + self.n_outputs

        # T5.5 (future cleanup): The `slot += 1` pattern below duplicates the
        # binding-counter logic in `kernel/header.py:HeaderMixin.codegen_kernel`.
        # Both places maintain their own counter and emit `[[vk::binding(N)]]`
        # annotations.  When we introduce reflection-driven binding (P0.4),
        # unify this into a single shared helper so the two paths can't
        # drift apart.
        slot = 0
        for dtype_str, name, _outer in in_decls:
            code.writeline(
                f"[[vk::binding({slot})]] StructuredBuffer<{dtype_str}> {name};"
            )
            slot += 1
        for dtype_str, name, _outer in rw_decls:
            code.writeline(
                f"[[vk::binding({slot})]] RWStructuredBuffer<{dtype_str}> {name};"
            )
            slot += 1

        max_tgs = 256
        for kernel, _ in self.subkernels:
            max_tgs = max(max_tgs, kernel.max_threadgroup_size)

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

        max_tgs = 256
        for kernel, _ in self.subkernels:
            max_tgs = max(max_tgs, kernel.max_threadgroup_size)

        for kernel, _ in self.subkernels:
            for v in kernel.args.sizevars:
                wrapper.ensure_size_computed(v)

        # Args must be passed in the same order the Slang shader binds them:
        # all in_decls (read-only) first, then rw_decls (read-write). The
        # `_outer` field on each decl is the wrapper-visible buffer name, so we
        # just emit those in the same order `_build_global_binding_map` did.
        in_decls, rw_decls, _ = self._build_global_binding_map()
        all_args: list[str] = [outer for _, _, outer in in_decls]
        all_args.extend(outer for _, _, outer in rw_decls)

        # PF.13.b.4-CODG: The Inductor memory planner may alias two buffers
        # via ``buf1 = div; del div`` before the kernel call.  If any kernel
        # argument name was freed by a reuse line, substitute the new name
        # so the emitted call doesn't reference a deleted variable.
        freed: set = getattr(wrapper, "freed", set())
        reuses: dict = getattr(wrapper, "reuses", {})
        if freed:
            old_to_new: dict[str, str] = {}
            for new_name, old_name in reuses.items():
                old_to_new[old_name] = new_name
            all_args = [old_to_new.get(a, a) for a in all_args]

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
        # X = max workgroups needed by any single subkernel.
        # Y = number of subkernels (gid.y selects which subkernel runs).
        # Each workgroup (x, y) runs subkernel y, with gid.x as the
        # subkernel's own workgroup ID. This preserves wave uniformity.
        # Reduction subkernels need one workgroup per output element (gid.x
        # indexes the non-reduction axis). Pointwise uses ceil(numel/TGS).
        def _wg_count(kernel, numel):
            if getattr(kernel, "inside_reduction", False):
                return numel  # one workgroup per output element
            return (numel + max_tgs - 1) // max_tgs

        max_wgs = max(_wg_count(k, n) for k, n in self.subkernels)
        wg_x = str(max_wgs)
        wg_y = str(len(self.subkernels))

        all_args.append(wg_x)
        all_args.append(wg_y)
        all_args.append("1")

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
