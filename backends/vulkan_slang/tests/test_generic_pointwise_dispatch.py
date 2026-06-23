"""Tests for generic pointwise dispatch via Slang IPointwise interfaces (T2.4).

Verifies that unary and binary pointwise ops can be dispatched through the
generic ``lib/pointwise.slang`` template instead of individual G-category shaders.
"""
from __future__ import annotations

import unittest as ut


class TestGenericPointwiseDispatch(ut.TestCase):
    """T2.4 — generic pointwise dispatch replaces per-op G-category shaders."""

    def test_pointwise_table_coverage(self):
        """Verify POINTWISE_TABLE covers all expected ops."""
        from torch_vulkan.inductor.generic_dispatch_table import POINTWISE_TABLE

        expected_unary = [
            "aten.relu", "aten.sigmoid", "aten.tanh", "aten.gelu",
            "aten.silu", "aten.elu", "aten.hardswish", "aten.hardsigmoid",
            "aten.mish", "aten.softplus", "aten.relu6",
            "aten.abs", "aten.neg", "aten.exp", "aten.log",
            "aten.sqrt", "aten.rsqrt", "aten.reciprocal",
        ]
        expected_binary = [
            "aten.add", "aten.sub", "aten.mul", "aten.div",
            "aten.minimum", "aten.maximum", "aten.pow",
        ]

        for op in expected_unary:
            entry = POINTWISE_TABLE.get(op)
            self.assertIsNotNone(entry, f"Missing {op} in POINTWISE_TABLE")
            self.assertEqual(entry.arity, 1, f"{op} should have arity 1")

        for op in expected_binary:
            entry = POINTWISE_TABLE.get(op)
            self.assertIsNotNone(entry, f"Missing {op} in POINTWISE_TABLE")
            self.assertEqual(entry.arity, 2, f"{op} should have arity 2")

    def test_pointwise_entry_emit_call(self):
        """Verify emit_call generates correct Slang code."""
        from torch_vulkan.inductor.generic_dispatch_table import (
            POINTWISE_TABLE, emit_pointwise_via_table,
        )

        # Unary
        out = emit_pointwise_via_table("aten.abs", "x")
        self.assertIn("OpAbs", out)
        self.assertIn("apply(x)", out)

        # Binary
        out = emit_pointwise_via_table("aten.add", "a", "b")
        self.assertIn("OpAdd", out)
        self.assertIn("apply(a, b)", out)

        # With imports
        self.assertIn("helpers", POINTWISE_TABLE["aten.gelu"].imports)

    def test_collect_imports(self):
        """Verify import collection returns correct modules."""
        from torch_vulkan.inductor.generic_dispatch_table import collect_imports

        imports = collect_imports(["aten.gelu", "aten.mish", "aten.softplus"])
        self.assertIn("pointwise", imports)
        self.assertIn("helpers", imports)

    def test_pointwise_dispatchtable_has_no_unknown_ops(self):
        """Every entry in POINTWISE_TABLE must have a valid Slang struct."""
        from torch_vulkan.inductor.generic_dispatch_table import POINTWISE_TABLE

        for op, entry in POINTWISE_TABLE.items():
            self.assertTrue(
                entry.op_struct.startswith("Op"),
                f"{op} → {entry.op_struct} should start with 'Op'",
            )
            self.assertIn(
                entry.arity, (1, 2),
                f"{op} arity={entry.arity} must be 1 or 2",
            )


if __name__ == "__main__":
    ut.main()
