"""CG.M10 — ``interface IDifferentiable`` for generic templates.

CG.M10 defines ``IDifferentiable`` in ``pointwise.slang`` so that
templates parameterized over ``<Epilogue : IDifferentiable>`` get
``bwd_diff`` support automatically.  The mm template's generic
constraint is changed from ``IPointwise`` to ``IDifferentiable``,
and ``IDifferentiableReduction`` is added for reductions.

Tests verify:
- ``IDifferentiable`` interface exists and is syntactically valid.
- All differentiable structs implement ``IDifferentiable``.
- Non-differentiable structs correctly omit the interface.
- The mm template's generic constraint is ``IDifferentiable``.
- ``IDifferentiableReduction`` exists and OpSum/OpProd implement it.
"""

import os
import re

import pytest
from torch.testing._internal.common_utils import TestCase, run_tests


class TestCGM10IDifferentiableInterface(TestCase):
    """CG.M10 — ``interface IDifferentiable`` for generic templates."""

    _DIFFERENTIABLE_STRUCTS: frozenset[str] = frozenset(
        {
            "OpIdentity",
            "OpReLU",
            "OpSigmoid",
            "OpTanh",
            "OpGELU",
            "OpSiLU",
            "OpELU",
            "OpHardSwish",
            "OpHardSigmoid",
            "OpMish",
            "OpSoftplus",
            "OpLeakyReLU",
            "OpAbs",
            "OpNeg",
            "OpExp",
            "OpLog",
            "OpSqrt",
            "OpRsqrt",
            "OpReciprocal",
            "OpCos",
            "OpSin",
            "OpTan",
            "OpAtan",
            "OpLog2",
            "OpLog10",
            "OpLog1p",
            "OpExp2",
            "OpExpm1",
            "OpAcos",
            "OpAsin",
            "OpCosh",
            "OpSinh",
            "OpAsinh",
            "OpAcosh",
            "OpAtanh",
        }
    )

    _NON_DIFFERENTIABLE_STRUCTS: frozenset[str] = frozenset(
        {
            "OpRelu6",
            "OpCeil",
            "OpFloor",
            "OpRound",
            "OpSign",
            "OpTrunc",
            "OpFrac",
            "OpLogicalNot",
            "OpBitwiseNot",
        }
    )

    _DIFFERENTIABLE_COUNT: int = 35
    _NON_DIFFERENTIABLE_COUNT: int = 9

    def _read_pointwise_slang(self) -> str:
        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "shaders",
            "lib",
            "pointwise.slang",
        )
        with open(path) as f:
            return f.read()

    def _read_reduction_slang(self) -> str:
        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "shaders",
            "lib",
            "reduction.slang",
        )
        with open(path) as f:
            return f.read()

    # ── Interface existence ────────────────────────────────────────────

    def test_cgm10_idifferentiable_interface_exists(self):
        """IDifferentiable is defined in pointwise.slang."""
        content = self._read_pointwise_slang()
        self.assertIn(
            "interface IDifferentiable",
            content,
            "CG.M10: interface IDifferentiable not found in pointwise.slang",
        )
        # The [Differentiable] attribute should be near the interface.
        idx = content.index("interface IDifferentiable")
        near = content[idx : idx + 150]
        self.assertIn(
            "[Differentiable]",
            near,
            "CG.M10: [Differentiable] attribute expected near IDifferentiable interface",
        )

    def test_cgm10_opgelu_implements_idifferentiable(self):
        """OpGELU satisfies the IDifferentiable constraint."""
        content = self._read_pointwise_slang()
        self.assertIn(
            "struct OpGELU : IPointwise, IDifferentiable",
            content,
            "CG.M10: OpGELU must implement IDifferentiable",
        )
        # Verify the [Differentiable] annotation is present on the apply method.
        idx = content.index("struct OpGELU : IPointwise, IDifferentiable")
        block = content[idx : idx + 300]
        self.assertIn(
            "[Differentiable]",
            block,
            "CG.M10: OpGELU::apply missing [Differentiable] annotation",
        )

    # ── Template constraint ────────────────────────────────────────────

    def test_cgm10_mm_template_uses_idifferentiable(self):
        """The mm template accepts IDifferentiable epilogue."""
        template_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "python",
            "torch_vulkan",
            "inductor",
            "templates",
        )
        for fname in ("slang_mm.py.jinja", "slang_mm.slang"):
            path = os.path.join(template_dir, fname)
            if os.path.exists(path):
                with open(path) as f:
                    content = f.read()
                self.assertIn(
                    "computeMain<Epilogue : IDifferentiable>",
                    content,
                    f"CG.M10: {fname} must use <Epilogue : IDifferentiable> constraint",
                )

    # ── Struct coverage ────────────────────────────────────────────────

    def test_cgm10_all_differentiable_structs_implement_idifferentiable(self):
        """All differentiable IPointwise structs also implement IDifferentiable."""
        content = self._read_pointwise_slang()
        missing = []
        for struct_name in sorted(self._DIFFERENTIABLE_STRUCTS):
            pattern = f"struct {struct_name} : IPointwise, IDifferentiable"
            if pattern not in content:
                missing.append(struct_name)
        self.assertEqual(
            missing,
            [],
            f"CG.M10: {len(missing)} differentiable struct(s) do NOT implement "
            f"IDifferentiable:\n" + "\n".join(f"  - {n}" for n in missing),
        )

    def test_cgm10_non_differentiable_structs_do_not_implement_idifferentiable(
        self,
    ):
        """Step/discrete function structs must NOT implement IDifferentiable."""
        content = self._read_pointwise_slang()
        incorrectly_differentiable = []
        for struct_name in sorted(self._NON_DIFFERENTIABLE_STRUCTS):
            pattern = f"struct {struct_name} : IPointwise, IDifferentiable"
            if pattern in content:
                incorrectly_differentiable.append(struct_name)
        self.assertEqual(
            incorrectly_differentiable,
            [],
            f"CG.M10: {len(incorrectly_differentiable)} non-differentiable struct(s) "
            f"incorrectly implement IDifferentiable:\n"
            + "\n".join(f"  - {n}" for n in incorrectly_differentiable),
        )

    def test_cgm10_differentiable_count_matches_expected(self):
        """Exactly 35 structs should implement IDifferentiable."""
        content = self._read_pointwise_slang()
        count = content.count(", IDifferentiable")
        self.assertEqual(
            count,
            self._DIFFERENTIABLE_COUNT,
            f"CG.M10: Expected {self._DIFFERENTIABLE_COUNT} IDifferentiable structs, "
            f"found {count}",
        )

    # ── IDifferentiableReduction ───────────────────────────────────────

    def test_cgm10_idifferentiable_reduction_exists(self):
        """IDifferentiableReduction is defined in reduction.slang."""
        content = self._read_reduction_slang()
        self.assertIn(
            "interface IDifferentiableReduction",
            content,
            "CG.M10: IDifferentiableReduction not found in reduction.slang",
        )
        self.assertIn(
            "IDifferentiableReduction : IReduction",
            content,
            "CG.M10: IDifferentiableReduction must inherit from IReduction",
        )

    def test_cgm10_opsum_implements_idifferentiable_reduction(self):
        """OpSum implements IDifferentiableReduction."""
        content = self._read_reduction_slang()
        self.assertIn(
            "struct OpSum : IReduction, IWaveReduction, IDifferentiableReduction",
            content,
            "CG.M10: OpSum must implement IDifferentiableReduction",
        )

    def test_cgm10_opprod_implements_idifferentiable_reduction(self):
        """OpProd implements IDifferentiableReduction."""
        content = self._read_reduction_slang()
        self.assertIn(
            "struct OpProd : IReduction, IWaveReduction, IDifferentiableReduction",
            content,
            "CG.M10: OpProd must implement IDifferentiableReduction",
        )

    # ── Python-side validation ─────────────────────────────────────────

    def test_cgm10_vulkan_template_caller_exports_differentiable_set(self):
        """vulkan_template_caller exports _VALID_IDIFFERENTIABLE_STRUCTS."""
        from torch_vulkan.inductor.vulkan_template_caller import (
            _VALID_IDIFFERENTIABLE_STRUCTS,
            _VALID_IPOINTWISE_STRUCTS,
        )

        # Every differentiable struct must also be in the IPointwise set.
        for s in _VALID_IDIFFERENTIABLE_STRUCTS:
            self.assertIn(
                s,
                _VALID_IPOINTWISE_STRUCTS,
                f"CG.M10: {s} is in _VALID_IDIFFERENTIABLE_STRUCTS "
                f"but not in _VALID_IPOINTWISE_STRUCTS",
            )

        # The differentiable set must be a proper subset (non-diff structs exist).
        self.assertLess(
            len(_VALID_IDIFFERENTIABLE_STRUCTS),
            len(_VALID_IPOINTWISE_STRUCTS),
            "CG.M10: _VALID_IDIFFERENTIABLE_STRUCTS should be a proper subset "
            "of _VALID_IPOINTWISE_STRUCTS",
        )

        # Specific check: OpGELU must be in the differentiable set.
        self.assertIn(
            "OpGELU",
            _VALID_IDIFFERENTIABLE_STRUCTS,
            "CG.M10: OpGELU must be in _VALID_IDIFFERENTIABLE_STRUCTS",
        )

    def test_cgm10_validate_epilogue_rejects_non_differentiable(self):
        """_validate_epilogue_struct rejects IPointwise-only structs."""
        from torch_vulkan.inductor.vulkan_template_caller import (
            _validate_epilogue_struct,
        )

        # OpCeil is IPointwise but NOT IDifferentiable.
        with self.assertRaises(ValueError) as cm:
            _validate_epilogue_struct("OpCeil")
        self.assertIn(
            "NOT IDifferentiable",
            str(cm.exception),
            f"CG.M10: error message should mention IDifferentiable, got: {cm.exception}",
        )

        # OpGELU should pass validation.
        result = _validate_epilogue_struct("OpGELU")
        self.assertEqual(
            result,
            "OpGELU",
            f"CG.M10: OpGELU should pass validation, got: {result}",
        )

        # None should pass through.
        self.assertIsNone(
            _validate_epilogue_struct(None),
            "CG.M10: None should bypass validation",
        )


if __name__ == "__main__":
    run_tests()
