"""Aten op → generic-shader dispatch table.

P1.6 foundation: pulls per-op string-emit logic out of `overrides.py`
into a single source-of-truth mapping, so the codegen path becomes:

    table[aten_op].emit_call(args)

instead of dozens of branches that each splice an inline math expression.
The full overrides.py rewrite consuming this table lands as
`P1.6-followup-overrides-migration` under P3.x.

Each entry names the `pointwise.slang` struct that implements the op so
the kernel codegen can emit a `pointwise_unary_apply<OpX>` /
`pointwise_binary_apply<OpX>` call against the generic shader family
shipped in P1.1.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PointwiseEntry:
    """One row of the generic pointwise dispatch table.

    `op_struct`: the Slang struct in `pointwise.slang` (e.g. "OpGELU").
    `arity`: 1 for unary `T -> T`, 2 for binary `(T, T) -> T`.
    `imports`: extra Slang module names this op pulls in beyond
    `pointwise` itself (e.g. GELU needs `helpers` for `c10_vulkan_erf`).
    """

    op_struct: str
    arity: int
    imports: tuple[str, ...] = ()

    def emit_call(self, *args: str) -> str:
        if len(args) != self.arity:
            raise ValueError(
                f"{self.op_struct}: expected {self.arity} args, got {len(args)}"
            )
        return f"{self.op_struct}::apply({', '.join(args)})"


# All 25 ops shipped in `shaders/lib/pointwise.slang` (P1.1). Keep this
# table in sync with that file — adding an op there *and* here is the
# single place new pointwise ops register.
POINTWISE_TABLE: dict[str, PointwiseEntry] = {
    # Unary activations
    "aten.relu":         PointwiseEntry("OpReLU", 1),
    "aten.sigmoid":      PointwiseEntry("OpSigmoid", 1),
    "aten.tanh":         PointwiseEntry("OpTanh", 1),
    "aten.gelu":         PointwiseEntry("OpGELU", 1, imports=("helpers",)),
    "aten.silu":         PointwiseEntry("OpSiLU", 1),
    "aten.elu":          PointwiseEntry("OpELU", 1),
    "aten.hardswish":    PointwiseEntry("OpHardSwish", 1),
    "aten.hardsigmoid":  PointwiseEntry("OpHardSigmoid", 1),
    "aten.mish":         PointwiseEntry("OpMish", 1, imports=("helpers",)),
    "aten.softplus":     PointwiseEntry("OpSoftplus", 1, imports=("helpers",)),
    "aten.relu6":        PointwiseEntry("OpRelu6", 1),
    # Unary math
    "aten.abs":          PointwiseEntry("OpAbs", 1),
    "aten.neg":          PointwiseEntry("OpNeg", 1),
    "aten.exp":          PointwiseEntry("OpExp", 1),
    "aten.log":          PointwiseEntry("OpLog", 1),
    "aten.sqrt":         PointwiseEntry("OpSqrt", 1),
    "aten.rsqrt":        PointwiseEntry("OpRsqrt", 1),
    "aten.reciprocal":   PointwiseEntry("OpReciprocal", 1),
    # Additional unary math
    "aten.cos":          PointwiseEntry("OpCos", 1),
    "aten.sin":          PointwiseEntry("OpSin", 1),
    "aten.tan":          PointwiseEntry("OpTan", 1),
    "aten.acos":         PointwiseEntry("OpAcos", 1),
    "aten.asin":         PointwiseEntry("OpAsin", 1),
    "aten.atan":         PointwiseEntry("OpAtan", 1),
    "aten.cosh":         PointwiseEntry("OpCosh", 1),
    "aten.sinh":         PointwiseEntry("OpSinh", 1),
    "aten.asinh":        PointwiseEntry("OpAsinh", 1),
    "aten.acosh":        PointwiseEntry("OpAcosh", 1),
    "aten.atanh":        PointwiseEntry("OpAtanh", 1),
    "aten.ceil":         PointwiseEntry("OpCeil", 1),
    "aten.floor":        PointwiseEntry("OpFloor", 1),
    "aten.round":        PointwiseEntry("OpRound", 1),
    "aten.sign":         PointwiseEntry("OpSign", 1),
    "aten.sgn":          PointwiseEntry("OpSign", 1),
    "aten.log2":         PointwiseEntry("OpLog2", 1),
    "aten.log10":        PointwiseEntry("OpLog10", 1),
    "aten.log1p":        PointwiseEntry("OpLog1p", 1),
    "aten.exp2":         PointwiseEntry("OpExp2", 1),
    "aten.expm1":        PointwiseEntry("OpExpm1", 1),
    "aten.trunc":        PointwiseEntry("OpTrunc", 1),
    "aten.frac":         PointwiseEntry("OpFrac", 1),
    "aten.logical_not":  PointwiseEntry("OpLogicalNot", 1),
    "aten.bitwise_not":  PointwiseEntry("OpBitwiseNot", 1),
    # Binary
    "aten.add":          PointwiseEntry("OpAdd", 2),
    "aten.sub":          PointwiseEntry("OpSub", 2),
    "aten.mul":          PointwiseEntry("OpMul", 2),
    "aten.div":          PointwiseEntry("OpDiv", 2),
    "aten.minimum":      PointwiseEntry("OpMin", 2),
    "aten.maximum":      PointwiseEntry("OpMax", 2),
    "aten.pow":          PointwiseEntry("OpPow", 2),
    # Additional binary
    "aten.fmod":         PointwiseEntry("OpFmod", 2),
    "aten.remainder":    PointwiseEntry("OpRemainder", 2),
    "aten.atan2":        PointwiseEntry("OpAtan2", 2),
    "aten.hypot":        PointwiseEntry("OpHypot", 2),
    "aten.nextafter":    PointwiseEntry("OpNextafter", 2),
    "aten.copysign":     PointwiseEntry("OpCopysign", 2),
}


def collect_imports(aten_ops: list[str]) -> tuple[str, ...]:
    """Return the Slang module imports required to compile a fused kernel
    that calls each op in ``aten_ops``. ``pointwise`` is always included
    because every entry routes through that module."""
    seen: set[str] = {"pointwise"}
    for op in aten_ops:
        entry = POINTWISE_TABLE.get(op)
        if entry is None:
            continue
        seen.update(entry.imports)
    return tuple(sorted(seen))


def emit_pointwise_via_table(aten_op: str, *args: str) -> str:
    """Render a Slang call site for ``aten_op`` using the generic shader.

    Raises KeyError if the op isn't in the table — callers must fall
    back to the legacy inline-string path until every op migrates.
    """
    entry = POINTWISE_TABLE[aten_op]
    return entry.emit_call(*args)
