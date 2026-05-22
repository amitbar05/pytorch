"""Token-based body rewriting for Vulkan combo-kernel fusion.

Resolves identifier collisions when merging multiple pointwise subkernel bodies
into a single Slang shader. Each subkernel emits its body referencing local
identifiers (``xindex``, ``x0``, ``in_ptr0``, ``out_ptr1``, ...) that collide
across subkernels.

We resolve this by:

  1. A lightweight tokenizer that distinguishes declarations from references
     so renaming is precise — a variable named ``xindex_sub0`` in a subkernel
     body is left alone instead of being corrupted by the old regex approach.
  2. Globally renaming each subkernel's buffer references via a deduped
     outer→global-name map.
  3. Per-subkernel scope tracking for Inductor-generated local variables.
"""

from __future__ import annotations

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
        "bfloat16",
        "int8_t",
        "uint8_t",
        "int16_t",
        "uint16_t",
        "int32_t",
        "int64_t",
        "uint64_t",
        "float16_t",
        "float32_t",
        "float64_t",
        "float2",
        "float3",
        "float4",
        "half2",
        "half3",
        "half4",
        "double2",
        "double3",
        "double4",
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
    # Identifiers following ``.`` are member names — never rename them,
    # UNLESS the object is ``args`` (ParameterBlock<KernelArgs>), whose
    # fields are buffer references that still need binding-map renaming.
    # CG.M14: prev_significant_value tracks the last non-space/non-comment
    # token value, so we can detect ``args.`` and allow name_map lookup.
    # member_object tracks the identifier token immediately before a ``.'',
    # so that when processing "args.in_ptr0" the field handler can check
    # whether the object was "args" even though prev_significant_value has
    # already been updated to ".".
    prev_significant_type: int | None = None
    prev_significant_value: str = ""
    member_object: str = ""  # identifier token before the most recent "."

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
            elif tok.value == ".":
                # Record the current object identifier for use when the
                # field token is processed (prev_significant_value will be
                # "." by then, so we need a separate slot).
                if prev_significant_type == _T_IDENT:
                    member_object = prev_significant_value
                else:
                    member_object = ""
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
                # or apply rename maps, UNLESS the object is ``args``
                # (ParameterBlock<KernelArgs>), whose fields are buffer
                # references that still need binding-map renaming (CG.M14).
                # Use member_object (captured when "." was processed) because
                # prev_significant_value is "." at this point, not "args".
                if (
                    member_object == "args"
                    and name in name_map
                    and name_map[name] != name
                ):
                    parts.append(name_map[name])
                else:
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
                            # M9.9: If a buffer name (in_ptrN / out_ptrN /
                            # inout_ptrN) is not in name_map, pre-seed it
                            # with a per-subkernel suffix instead of using
                            # cross_decls (which would pick up a previous
                            # subkernel's _sub{other} rename).  This prevents
                            # UnboundLocalError / undefined-identifier
                            # collisions when the combo batcher merges
                            # transformer subkernels that share buffer inner
                            # names.
                            if _is_buffer_name(name):
                                renamed = f"{name}_sub{idx}"
                            else:
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
