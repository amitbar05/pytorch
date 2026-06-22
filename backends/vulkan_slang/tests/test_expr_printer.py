"""Unit tests for VulkanExprPrinter precedence/parenthesization rules."""
from __future__ import annotations

import sympy

from torch_vulkan.inductor.expr_printer import VulkanExprPrinter


class TestMulNeedsParens:
    """_mul_needs_parens guards factors that would change meaning when
    concatenated with ` * ` in a Mul chain."""

    def test_add_factor_wrapped(self):
        """(a + b) * c → the Add factor must be parenthesized regardless of factor order."""
        a = sympy.Symbol("a")
        b = sympy.Symbol("b")
        c = sympy.Symbol("c")
        expr = sympy.Mul(a + b, c)
        out = VulkanExprPrinter().doprint(expr)
        # sympy may reorder factors; what matters is the Add sub-expression has parens
        assert "(a + b)" in out, f"expected parenthesized '(a + b)' in output, got {out!r}"
        assert "*" in out, f"expected '*' in output, got {out!r}"

    def test_plain_mul_no_spurious_parens(self):
        """a * b → no parentheses needed."""
        a = sympy.Symbol("a")
        b = sympy.Symbol("b")
        expr = sympy.Mul(a, b)
        out = VulkanExprPrinter().doprint(expr)
        assert out == "a * b", f"expected 'a * b', got {out!r}"

    def test_rational_factor_has_parens(self):
        """sympy.Rational(1, 2) prints with /; must be parenthesized in Mul."""
        a = sympy.Symbol("a")
        r = sympy.Rational(1, 2)
        expr = sympy.Mul(r, a)
        out = VulkanExprPrinter().doprint(expr)
        # _print_Rational emits "(0.5f / 1.0f)" which already has outer parens,
        # but _mul_needs_parens must also catch any bare '/' in a factor string.
        assert "/" in out, f"expected '/' in output, got {out!r}"
        # Verify the factor is parenthesized
        assert out.startswith("("), f"expected parenthesized factor, got {out!r}"

    def test_negative_integer_no_spurious_parens(self):
        """-1 at position 0 (unary minus) must not trigger spurious parens."""
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        expr = sympy.Mul(-x, y)
        out = VulkanExprPrinter().doprint(expr)
        # The leading '-' is unary; _mul_needs_parens must NOT wrap it.
        assert "(-1)" not in out, f"spurious parens around -1 in {out!r}"
        assert out.startswith("-"), f"expected leading minus, got {out!r}"
