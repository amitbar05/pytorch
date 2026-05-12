"""SymPy → Slang expression printer for Inductor codegen."""
from __future__ import annotations

import contextlib

import sympy

from torch.utils._sympy.printers import ExprPrinter as ExprPrinter_
from torch._inductor.virtualized import V


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

    @contextlib.contextmanager
    def subscript(self):
        """Mark the upcoming `doprint` call as emitting an array subscript so
        `tmp\\d+` symbols get the int cast they need for safe indexing."""
        self._subscript_depth += 1
        try:
            yield
        finally:
            self._subscript_depth -= 1

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
            return self._print(non_one[0])
        return self.stringify(non_one, "*", sympy.printing.precedence.precedence(expr))

    def _print_Add(self, expr: sympy.Expr, order: str | None = None) -> str:
        from sympy import S
        non_zero = [a for a in expr.args if a != S.Zero]
        if not non_zero:
            return "0"
        if len(non_zero) == 1:
            return self._print(non_zero[0])
        s = self.stringify(non_zero, " + ", sympy.printing.precedence.precedence(expr))
        # P10 / replication_pad2d: Slang's uint arithmetic wraps on overflow.
        # The expression ``(-1) + x1`` with ``uint x1`` produces 0xFFFFFFFF for
        # x1=0 instead of -1, causing min()/max() indexing to read from the
        # wrong memory location. When any arg is a negative Integer, wrap the
        # entire sum in ``(int)(...)`` to force signed arithmetic.
        if any(isinstance(a, sympy.Integer) and a < 0 for a in expr.args):
            return f"(int)({s})"
        return s

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        x, div = expr.args
        # P11.5 / P6.6 — Dead-axis elimination for size-1 dims:
        # `FloorDiv(x, 1) = x` and `FloorDiv(0, _) = 0`. SymPy normally
        # folds these but `keepdim=True` reductions can construct them
        # with `evaluate=False`.
        if div == 1:
            return self.doprint(x)
        if x == 0:
            return "0"
        xs = self.doprint(x)
        ds = self.doprint(div)
        if expr.is_integer:
            # For negative ints Slang follows C (`/` rounds toward 0). Use a
            # branchless floor-divide for correctness.
            return f"(({xs}) / ({ds}) - ((({xs}) % ({ds})) != 0 && (({xs}) < 0) != (({ds}) < 0) ? 1 : 0))"
        return f"floor(({xs}) / ({ds}))"

    def _print_ModularIndexing(self, expr: sympy.Expr) -> str:
        x, div, mod = expr.args
        # P11.5 — `x % 1 = 0` (any value modulo 1 is 0). Same dead-axis
        # case as FloorDiv above.
        if mod == 1:
            return "0"
        xs = self.doprint(x)
        if div != 1:
            ds = self.doprint(div)
            xs = f"(({xs}) / ({ds}))" if expr.is_integer else f"(floor({xs}) / ({ds}))"
        ms = self.doprint(mod)
        return f"(({xs}) % ({ms}))"

    def _print_Min(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 2:
            raise RuntimeError("Slang min only supported for 2 args")
        a, b = map(self._print, expr.args)
        return f"min({a}, {b})"

    def _print_Max(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 2:
            raise RuntimeError("Slang max only supported for 2 args")
        a, b = map(self._print, expr.args)
        return f"max({a}, {b})"

    def _print_Abs(self, expr: sympy.Expr) -> str:
        (a,) = expr.args
        return f"abs({self.doprint(a)})"

    def _print_Where(self, expr: sympy.Expr) -> str:
        c, t, f = expr.args
        return f"({self.doprint(c)} ? {self.doprint(t)} : {self.doprint(f)})"

    def _print_floor(self, expr: sympy.Expr) -> str:
        return f"floor({self.doprint(expr.args[0])})"

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
        name = expr.name
        if name[:1] in ("s", "u") and name[1:2].isdigit():
            k = V.kernel if hasattr(V, "kernel") else None
            if k is not None and getattr(k, "args", None) is not None:
                return k.args.size(expr)
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
