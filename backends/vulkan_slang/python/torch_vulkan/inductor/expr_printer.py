"""SymPy → Slang expression printer for Inductor codegen."""
from __future__ import annotations

import contextlib
import math

import sympy

from torch.utils._sympy.printers import ExprPrinter as ExprPrinter_
from torch._inductor.virtualized import V


def _kernel_is_fully_static(kernel) -> bool:
    """Check whether *kernel* will emit sizevars as ``static const uint``.

    When True, sizevar inner names (e.g. ``ks0``) are module-scope
    constants and do not need a ``pc.`` prefix.  When False, they are
    push-constant struct members and MUST be referenced as ``pc.ks0``.
    """
    idx_vars = getattr(kernel, "active_range_trees", lambda: [])()
    if not idx_vars:
        return True
    if not all(isinstance(v.numel, sympy.Integer) for v in idx_vars):
        return False
    sv = getattr(getattr(kernel, "args", None), "sizevars", None)
    if sv is not None:
        for expr in sv:
            if not isinstance(expr, sympy.Integer):
                return False
    return True


class VulkanExprPrinter(ExprPrinter_):
    """SymPy → Slang expression printer.

    Slang's arithmetic is HLSL-like, so most sympy operators map directly to
    `+`, `-`, `*`, `/`, `%`. Integer division requires manual handling —
    same story as CUDA/Metal.

    Subscript context: `_subscript_depth > 0` means the expression is being
    rendered as an array subscript (set by load/store via `subscript()`).
    `tmp\\d+` CSE locals from `indirect_indexing` hold int64 index values
    stored as float — they need an `((int)(...))` cast on subscript use to
    avoid float→uint truncation. Outside subscript context the cast would
    *destroy* precision on non-index float CSE locals (P6.3).
    """

    _subscript_depth: int = 0

    # DYN.1: When True (default) — the Slang source emission context —
    # sizevar inner names get a ``pc.`` prefix so the generated Slang
    # references push-constant struct members.  When False — the Python
    # wrapper codegen context — the bare inner name is used for wrapper
    # variable lookup.  Toggle via ``_disable_pc_prefix()`` context manager.
    _use_pc_prefix: bool = True

    @contextlib.contextmanager
    def _disable_pc_prefix(self):
        """Context manager for Python-wrapper codegen: sizevars are bare names."""
        prev = self._use_pc_prefix
        self._use_pc_prefix = False
        try:
            yield
        finally:
            self._use_pc_prefix = prev

    @contextlib.contextmanager
    def subscript(self):
        """Mark the upcoming `doprint` call as emitting an array subscript so
        `tmp\\d+` symbols get the int cast they need for safe indexing."""
        self._subscript_depth += 1
        try:
            yield
        finally:
            self._subscript_depth -= 1

    @staticmethod
    def _mul_needs_parens(s: str) -> bool:
        """True when the printed string `s` has a binary +/- or / at top level.

        Inductor sometimes wraps Add expressions in an `Identity` node so that
        ``isinstance(factor, sympy.Add)`` returns False.  Rather than chasing
        every wrapper type, we check the already-printed string: if it contains
        a `+` or a `-` that is not the very first character (i.e. unary minus
        on a plain negative literal), the factor is an addition/subtraction/division and
        must be wrapped in parentheses when used as a Mul operand.
        """
        depth = 0
        for i, c in enumerate(s):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif depth == 0 and c in '+-/' and i > 0:
                return True
        return False

    def _print_Identity(self, expr: sympy.Expr) -> str:
        """Unwrap Inductor's Identity() wrapper and print the inner expression."""
        return self.doprint(expr.args[0])

    def _print_Mul(self, expr: sympy.Expr) -> str:
        # SymPy normally constant-folds 0*x → 0 and 1*x → x, but Inductor
        # sometimes constructs Mul nodes with `evaluate=False` (e.g. for
        # symbolic strides where `stride*idx` is kept atomic). Filter out
        # the trivial multipliers so the emitted Slang doesn't ship dead
        # `0 * tmp17` or `1 * tmp17` scalar ops that bloat the SPIR-V.
        from sympy import S
        args = list(expr.args)
        for a in args:
            if a == S.Zero:
                return "0"
        non_one = [a for a in args if a != S.One]
        if not non_one:
            return "1"
        if len(non_one) == 1:
            return self.doprint(non_one[0])
        parts_list = []
        for a in non_one:
            s = self.doprint(a)
            parts_list.append(f"({s})" if self._mul_needs_parens(s) else s)
        return " * ".join(parts_list)

    def _print_Add(self, expr: sympy.Expr) -> str:
        from sympy import S
        args = [a for a in expr.args if a != S.Zero]
        if not args:
            return "0"
        parts = []
        for a in args:
            s = self.doprint(a)
            if parts and s.startswith("-"):
                parts.append(f" - {s[1:]}")
            else:
                parts.append(f" + " + s if parts else s)
        return "".join(parts)

    def _print_Pow(self, expr: sympy.Expr) -> str:
        base, exp = expr.args
        if exp == 0.5:
            return f"sqrt({self.doprint(base)})"
        if exp == -1:
            return f"(1.0f / ({self.doprint(base)}))"
        if exp == 2:
            return f"(({self.doprint(base)}) * ({self.doprint(base)}))"
        return f"pow({self.doprint(base)}, {self.doprint(exp)})"

    def _print_Mod(self, expr: sympy.Expr) -> str:
        return f"({self.doprint(expr.args[0])} % {self.doprint(expr.args[1])})"

    def _print_Integer(self, expr: sympy.Expr) -> str:
        return str(int(expr))

    def _print_Float(self, expr: sympy.Expr) -> str:
        val = float(expr)
        if math.isinf(val):
            return "(1.0/0.0)" if val > 0 else "(-1.0/0.0)"
        if math.isnan(val):
            return "asfloat(0x7FC00000u)"
        return f"{val!r}f"

    def _print_Rational(self, expr: sympy.Expr) -> str:
        return f"({float(expr.p)}f / {float(expr.q)}f)"

    def _print_floor(self, expr: sympy.Expr) -> str:
        return f"floor({self.doprint(expr.args[0])})"

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        """FloorDiv(a, b) → int floor division. All Vulkan index values
        are positive, so integer truncation IS floor division."""
        a = self.doprint(expr.args[0])
        b = self.doprint(expr.args[1])
        return f"({a} / {b})"

    def _print_ModularIndexing(self, expr: sympy.Expr) -> str:
        """ModularIndexing(a, b, c) → (a / b) % c.
        All Vulkan index values are positive."""
        a = self.doprint(expr.args[0])
        b = self.doprint(expr.args[1])
        c = self.doprint(expr.args[2])
        return f"(({a} / {b}) % {c})"

    def _print_ceiling(self, expr: sympy.Expr) -> str:
        return f"ceil({self.doprint(expr.args[0])})"

    def _print_Symbol(self, expr: sympy.Expr) -> str:
        # Register symbolic shape symbols (e.g. `s27` from dynamic=True) as
        # kernel sizevars so they get a push-constant slot. Without this the
        # raw sympy name ("s27") leaks into shader source and slangc fails
        # with "undefined identifier 's27'". Mirrors the CUDA Triton codegen
        # pattern — size symbols are passed in through the push-constant
        # struct as `ks0`, `ks1`, ... Only size symbols (conventionally
        # start with `s` or `u` in Inductor) are registered; axis variables
        # like `xindex` / `tmp0` already exist in scope.
        #
        # DYN.1 fix: when the kernel is NOT fully static, sizevars are
        # push-constant members (``pc.ks0``).  The expression printer must
        # emit the ``pc.`` prefix so the generated Slang body resolves
        # against the correct struct member.  Without it, slangc fails with
        # "undefined identifier 'ks0'" / "'s77u'" on dynamic-batch compiles.
        name = expr.name
        if name[:1] in ("s", "u") and name[1:2].isdigit():
            k = V.kernel if hasattr(V, "kernel") else None
            if k is not None and getattr(k, "args", None) is not None:
                inner = k.args.size(expr)
                if not _kernel_is_fully_static(k) and self._use_pc_prefix:
                    return f"pc.{inner}"
                return inner
        # `tmp\d+` symbols are CSE locals from indirect_indexing on int64
        # gather/scatter indices. The CSE machinery declares them `float`
        # (kernel newvar_prefix) but they hold integer index values. Used
        # as array subscripts they need an int cast, otherwise Slang either
        # rejects the subscript or silently truncates via float→uint with
        # the wrong rounding behavior. Cast in subscript context only —
        # outside that the same `tmp` may be a real float CSE value and the
        # cast would destroy precision (P6.3 / P1.5).
        if (
            self._subscript_depth > 0
            and name.startswith("tmp") and name[3:].isdigit()
        ):
            return f"((int)({name}))"
        return name
