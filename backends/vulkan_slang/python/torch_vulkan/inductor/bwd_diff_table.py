"""PF.6 — `bwd_diff(fwd)` codegen for autodiff-eligible backwards.

Maps each `aten.<op>_backward` whose forward primal lives in
`shaders/lib/*.slang` as `[Differentiable]` to a wrapper-kernel emitter
that invokes `bwd_diff(<op>_fwd)(...)`. Slang autodiff produces the
gradient — the corresponding hand-written backward shader (or the
hand-derived gradient algebra in `lowerings.py`) is replaced by codegen
once the per-op autodiff-vs-hand benchmark in PF.11 picks autodiff.

The emitted shader puts ``numel`` (and any ``no_diff`` scalar params)
into a ``[[vk::push_constant]] cbuffer Push`` block — matching the
existing project convention so the runtime dispatcher
(``bwd_diff_dispatch.dispatch_unary_bwd`` /
``dispatch_binary_bwd``) can invoke ``compile_and_dispatch`` with
``push_constants=`` directly. Bare ``uniform <scalar>`` entry-point
parameters are intentionally avoided — slangc maps those to a UBO
descriptor (``kind: "uniform"``), not a push constant.

Today this module emits the codegen path *and* is wired through the
dispatcher in ``bwd_diff_dispatch.py``; default lowerings are still
hand-rolled. PF.6.b.iii benchmarks autodiff vs hand-rolled wall-clock
to flip the swap op-by-op.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BwdDiffEntry:
    fwd_fn: str
    module: str
    arity: int
    # Names of `no_diff`-annotated scalar params in the forward
    # signature (e.g. `smooth_l1_elem(pred, target, no_diff float beta)`
    # → `("beta",)`). Slang's bwd_diff threads each no_diff scalar
    # through unchanged (no DifferentialPair wrap), so the emitter
    # materializes each as a member of the `[[vk::push_constant]]
    # cbuffer Push { ... };` block (in declaration order, before
    # `numel`) and forwards it to the bwd_diff call site between the
    # DifferentialPair args and the trailing `dOut`. The dispatcher
    # packs scalars in the same order: `<f * len(no_diff_params)> + <I>`.
    no_diff_params: tuple[str, ...] = ()


BWD_DIFF_TABLE: dict[str, BwdDiffEntry] = {
    # ── Activation backward (existing, PF.11) ──────────────────────────
    "aten.relu_backward": BwdDiffEntry("relu_fwd", "pointwise", 1),
    # C1: threshold_backward(grad, output, 0) IS relu backward — same Slang.
    "aten.threshold_backward": BwdDiffEntry("relu_fwd", "pointwise", 1),
    # B.5.C: leaky_relu_fwd carries a ``no_diff float alpha`` param that
    # Slang's autodiff threads through unchanged. The unary emitter
    # materializes ``negative_slope`` in the push-constant block (before
    # ``numel``) and forwards it to ``bwd_diff(leaky_relu_fwd)``. The
    # caller passes ``no_diff_kwargs={"negative_slope": ...}`` to
    # ``dispatch_unary_bwd``.
    "aten.leaky_relu_backward": BwdDiffEntry(
        "leaky_relu_fwd",
        "pointwise",
        1,
        no_diff_params=("negative_slope",),
    ),
    "aten.sigmoid_backward": BwdDiffEntry("sigmoid_fwd", "pointwise", 1),
    "aten.tanh_backward": BwdDiffEntry("tanh_fwd", "pointwise", 1),
    "aten.gelu_backward": BwdDiffEntry("gelu_fwd", "pointwise", 1),
    "aten.silu_backward": BwdDiffEntry("silu_fwd", "pointwise", 1),
    "aten.elu_backward": BwdDiffEntry("elu_fwd", "pointwise", 1),
    "aten.hardswish_backward": BwdDiffEntry("hardswish_fwd", "pointwise", 1),
    "aten.hardsigmoid_backward": BwdDiffEntry("hardsigmoid_fwd", "pointwise", 1),
    # M-AG5.1 Tier-2 (2026-05-22): ``aten.softplus_backward`` removed from
    # the autodiff route. The aten signature carries ``beta`` and
    # ``threshold`` Scalar params, but ``softplus_fwd`` in
    # ``shaders/lib/pointwise.slang`` is the ``beta=1, threshold=20`` form
    # ``log(1 + exp(x))`` with neither param materialised — wiring
    # ``no_diff_params=("beta", "threshold")`` here would emit a
    # ``bwd_diff(softplus_fwd)(dp, beta, threshold, dOut)`` call against a
    # shader that does not declare those parameters and slangc would
    # reject it. Until ``softplus_fwd`` grows the two ``no_diff float``
    # params (Group G work), softplus_backward routes via the algebraic
    # lowering in ``bwd_lowerings.py::_register_algebraic_backward_lowerings``
    # — the leaky_relu pattern — and ``softplus_fwd`` is listed in
    # ``EXCLUDED_DIFFERENTIABLE_FWDS`` below.
    "aten.mish_backward": BwdDiffEntry("mish_fwd", "pointwise", 1),
    # ── CG.M1: basic trig ─────────────────────────────────────────────
    "aten.sin_backward": BwdDiffEntry("sin_fwd", "pointwise", 1),
    "aten.cos_backward": BwdDiffEntry("cos_fwd", "pointwise", 1),
    "aten.tan_backward": BwdDiffEntry("tan_fwd", "pointwise", 1),
    "aten.asin_backward": BwdDiffEntry("asin_fwd", "pointwise", 1),
    "aten.acos_backward": BwdDiffEntry("acos_fwd", "pointwise", 1),
    "aten.atan_backward": BwdDiffEntry("atan_fwd", "pointwise", 1),
    # ── CG.M1: hyperbolic ─────────────────────────────────────────────
    "aten.sinh_backward": BwdDiffEntry("sinh_fwd", "pointwise", 1),
    "aten.cosh_backward": BwdDiffEntry("cosh_fwd", "pointwise", 1),
    "aten.asinh_backward": BwdDiffEntry("asinh_fwd", "pointwise", 1),
    "aten.acosh_backward": BwdDiffEntry("acosh_fwd", "pointwise", 1),
    "aten.atanh_backward": BwdDiffEntry("atanh_fwd", "pointwise", 1),
    # ── CG.M1: exp / log / power ──────────────────────────────────────
    "aten.exp_backward": BwdDiffEntry("exp_fwd", "pointwise", 1),
    "aten.expm1_backward": BwdDiffEntry("expm1_fwd", "pointwise", 1),
    "aten.exp2_backward": BwdDiffEntry("exp2_fwd", "pointwise", 1),
    "aten.log_backward": BwdDiffEntry("log_fwd", "pointwise", 1),
    "aten.log2_backward": BwdDiffEntry("log2_fwd", "pointwise", 1),
    "aten.log10_backward": BwdDiffEntry("log10_fwd", "pointwise", 1),
    "aten.log1p_backward": BwdDiffEntry("log1p_fwd", "pointwise", 1),
    "aten.sqrt_backward": BwdDiffEntry("sqrt_fwd", "pointwise", 1),
    "aten.rsqrt_backward": BwdDiffEntry("rsqrt_fwd", "pointwise", 1),
    "aten.reciprocal_backward": BwdDiffEntry("reciprocal_fwd", "pointwise", 1),
    # ── CG.M1: abs / neg ──────────────────────────────────────────────
    "aten.abs_backward": BwdDiffEntry("abs_fwd", "pointwise", 1),
    "aten.neg_backward": BwdDiffEntry("neg_fwd", "pointwise", 1),
    # ── CG.M1: special functions ──────────────────────────────────────
    "aten.erf_backward": BwdDiffEntry("erf_fwd", "pointwise", 1),
    "aten.erfc_backward": BwdDiffEntry("erfc_fwd", "pointwise", 1),
    "aten.erfinv_backward": BwdDiffEntry("erfinv_fwd", "pointwise", 1),
    "aten.lgamma_backward": BwdDiffEntry("lgamma_fwd", "pointwise", 1),
    "aten.digamma_backward": BwdDiffEntry("digamma_fwd", "pointwise", 1),
    "aten.ndtri_backward": BwdDiffEntry("ndtri_fwd", "pointwise", 1),
    "aten.i0_backward": BwdDiffEntry("i0_fwd", "pointwise", 1),
    "aten.i0e_backward": BwdDiffEntry("i0e_fwd", "pointwise", 1),
    "aten.i1_backward": BwdDiffEntry("i1_fwd", "pointwise", 1),
    "aten.i1e_backward": BwdDiffEntry("i1e_fwd", "pointwise", 1),
    # ── CG.M2: binary pointwise backward ───────────────────────────────
    "aten.pow.Tensor_Tensor_backward": BwdDiffEntry("pow_fwd", "pointwise", 2),
    "aten.atan2_backward": BwdDiffEntry("atan2_fwd", "pointwise", 2),
    "aten.hypot_backward": BwdDiffEntry("hypot_fwd", "pointwise", 2),
    "aten.copysign_tensor_backward": BwdDiffEntry(
        "copysign_fwd", "pointwise", 2, no_diff_params=("sign",)
    ),
    "aten.maximum_backward": BwdDiffEntry("max_fwd", "pointwise", 2),
    "aten.minimum_backward": BwdDiffEntry("min_fwd", "pointwise", 2),
    # ── Loss backward (existing, T2.11) ───────────────────────────────
    # T2.11 (2026-05-08): all 7 loss elementals now carry
    # [BackwardDerivative] in `lib/losses.slang`. The table entries below
    # are unchanged — `bwd_diff(<fn>)` automatically resolves to the
    # closed-form `<fn>_bwd` instead of auto-deriving (and allocating an
    # `exp`-tape for `bce_with_logits`). No emitter change needed.
    "aten.mse_loss_backward": BwdDiffEntry("mse_elem", "losses", 2),
    "aten.l1_loss_backward": BwdDiffEntry("l1_elem", "losses", 2),
    "aten.binary_cross_entropy_backward": BwdDiffEntry("bce_elem", "losses", 2),
    "aten.binary_cross_entropy_with_logits_backward": BwdDiffEntry(
        "bce_with_logits_elem", "losses", 2
    ),
    "aten.smooth_l1_loss_backward": BwdDiffEntry(
        "smooth_l1_elem",
        "losses",
        2,
        no_diff_params=("beta",),
    ),
    "aten.huber_loss_backward": BwdDiffEntry(
        "huber_elem",
        "losses",
        2,
        no_diff_params=("delta",),
    ),
    "aten.kl_div_backward": BwdDiffEntry("kl_div_elem", "losses", 2),
}


def is_bwd_diff_eligible(aten_op: str) -> bool:
    return aten_op in BWD_DIFF_TABLE


# Forwards in `shaders/lib/*.slang` that carry `[Differentiable]` but
# intentionally do not have an entry in `BWD_DIFF_TABLE`. Each exclusion
# documents the reason so the coverage gate
# (`TestBwdDiffCoverageGate`) doesn't silently drift — a future
# annotation drop on a forward shader without a corresponding table or
# exclusion update breaks the gate.
EXCLUDED_DIFFERENTIABLE_FWDS: dict[str, str] = {
    # T3.5 (2026-05-02): [BackwardDerivative] annotations added to all 4 norm
    # elementals. Fast backward formulas for per-element dx/dw/db gradients.
    # Still excluded from BWD_DIFF_TABLE because the full norm backward requires
    # reduction ops (sum over normalized dims) that bwd_diff codegen doesn't emit.
    # Inductor lowerings handle norm backward via decomposed pointwise+reduction IR.
    "ln_affine_elem": "norm: needs reduction ops for full backward (T3.5)",
    "ln_no_affine_elem": "norm: needs reduction ops for full backward (T3.5)",
    "rms_affine_elem": "norm: needs reduction ops for full backward (T3.5)",
    "rms_no_affine_elem": "norm: Welford reduction not autodiff-safe (P2.3)",
    # M-AG5.1 Tier-2 (2026-05-22): ``softplus_fwd`` keeps its
    # ``[Differentiable]`` annotation (the autograd-eligible math is well
    # defined for ``beta=1`` / threshold disabled), but the aten op
    # ``softplus_backward(grad_output, self, beta, threshold)`` cannot be
    # routed through this table because the shader signature
    # ``softplus_fwd(float x)`` does not carry the two ``no_diff`` scalars.
    # softplus_backward is lowered algebraically in ``bwd_lowerings.py``;
    # see the M-AG5.1 entry there.
    "softplus_fwd": (
        "M-AG5.1 Tier-2: aten signature carries beta/threshold "
        "scalars that softplus_fwd does not declare; backward "
        "lowered algebraically in bwd_lowerings.py"
    ),
    # ── M12 / reduction-autodiff internal helpers ──────────────────────
    # These carry [Differentiable] so Slang's autodiff engine can chain
    # through them when differentiating reduction kernels. They are NOT
    # direct aten ops — the aten-level ops (sum/mean/prod/etc.) are
    # routed via their own BWD_DIFF_TABLE entries. Keeping [Differentiable]
    # on the helpers enables bwd_diff(reduce_sum)(...) to trace through
    # combine_sum_nan automatically, which is the M12 autodiff payoff.
    "combine_max": "reduction helper: [Differentiable] for autodiff chaining; not a direct aten op",
    "combine_min": "reduction helper: [Differentiable] for autodiff chaining; not a direct aten op",
    "combine_prod_nan": "reduction helper: [Differentiable] for autodiff chaining; not a direct aten op",
    "combine_sum_nan": "reduction helper: [Differentiable] for autodiff chaining; not a direct aten op",
    "welford_combine": "Welford combine step: [Differentiable] for internal chaining; not a direct aten op",
    "reduce_fold_prod": "reduction fold helper: [Differentiable] for autodiff chaining; not a direct aten op",
    "reduce_fold_sum": "reduction fold helper: [Differentiable] for autodiff chaining; not a direct aten op",
    # ── Norm chain helpers ─────────────────────────────────────────────
    # Full fused forward (Welford + normalize + affine). [Differentiable]
    # enables autodiff through the chain, but the full norm backward
    # also requires a reduction over normalized dims — not emittable from
    # bwd_diff codegen alone. Norm backward routes via decomposed IR.
    "layer_norm_chain": "norm chain: needs reduction for full bwd; routes via decomposed IR",
    "layer_norm_chain_no_affine": "norm chain: needs reduction for full bwd; routes via decomposed IR",
    "rms_norm_chain": "norm chain: needs reduction for full bwd; routes via decomposed IR",
    "rms_norm_chain_no_affine": "norm chain: needs reduction for full bwd; routes via decomposed IR",
    # ── SDPA / softmax helpers ─────────────────────────────────────────
    # Element-wise steps inside SDPA backward and softmax backward.
    # [Differentiable] enables Slang autodiff to chain through them.
    # The SDPA / softmax aten ops are not yet in BWD_DIFF_TABLE.
    "sdpa_score": "SDPA helper: [Differentiable] for attention-bwd chaining; aten op not yet in table",
    "sdpa_output": "SDPA helper: [Differentiable] for attention-bwd chaining; aten op not yet in table",
    "softmax_elem": "softmax helper: [Differentiable] for bwd chaining; aten op not yet in table",
    "softmax_exp_sub": "softmax helper: [Differentiable] for bwd chaining; aten op not yet in table",
    # ── Conv / mm inner helpers ────────────────────────────────────────
    # Inner multiply-add steps. [Differentiable] enables bwd_diff to
    # trace through the inner loop. The conv/mm aten backward ops are
    # handled by hand-rolled lowerings, not via bwd_diff codegen.
    "conv_inner_madd": "conv inner helper: [Differentiable] for conv-bwd autodiff chaining",
    "tile_inner_madd": "mm-tile inner helper: [Differentiable] for mm-bwd autodiff chaining",
}


def emit_bwd_diff_kernel(
    aten_op: str,
    *,
    dtype: str = "float",
    numthreads: int = 256,
) -> str:
    """Render a self-contained Slang shader source whose body invokes
    ``bwd_diff(<fwd_fn>)`` to produce the gradient for ``aten_op``.

    The forward primal must carry ``[Differentiable]``; sigmoid resolves
    to its ``[BackwardDerivative(sigmoid_fast_bwd)]`` override.
    """
    entry = BWD_DIFF_TABLE[aten_op]
    if entry.arity == 1:
        return _emit_unary(entry, dtype=dtype, numthreads=numthreads)
    if entry.arity == 2:
        return _emit_binary(entry, dtype=dtype, numthreads=numthreads)
    raise ValueError(f"unsupported arity {entry.arity} for {aten_op}")


def _emit_unary(entry: BwdDiffEntry, *, dtype: str, numthreads: int) -> str:
    # B.5.C: ``no_diff`` scalars (e.g. ``leaky_relu_fwd``'s
    # ``negative_slope``) are emitted as fields of the push-constant
    # block — in declaration order, before ``numel`` — and forwarded
    # to ``bwd_diff(<fwd>)`` between the DifferentialPair argument and
    # the trailing ``dOut``. Mirrors ``_emit_binary``.
    pc_fields = "".join(f"    {dtype} {name};\n" for name in entry.no_diff_params)
    no_diff_args = "".join(f"{name}, " for name in entry.no_diff_params)
    return (
        f"import {entry.module};\n"
        f"\n"
        f"[[vk::push_constant]] cbuffer Push {{\n"
        f"{pc_fields}"
        f"    uint numel;\n"
        f"}};\n"
        f"\n"
        f'[shader("compute")][numthreads({numthreads},1,1)]\n'
        f"void bwd_op(\n"
        f"    uniform StructuredBuffer<{dtype}> x,\n"
        f"    uniform StructuredBuffer<{dtype}> grad_out,\n"
        f"    uniform RWStructuredBuffer<{dtype}> grad_in,\n"
        f"    uint3 tid : SV_DispatchThreadID\n"
        f") {{\n"
        f"    if (tid.x >= numel) return;\n"
        f"    DifferentialPair<{dtype}> dp = "
        f"diffPair(x[tid.x], ({dtype})0);\n"
        f"    bwd_diff({entry.fwd_fn})("
        f"dp, {no_diff_args}grad_out[tid.x]);\n"
        f"    grad_in[tid.x] = dp.getDifferential();\n"
        f"}}\n"
    )


def _emit_binary(entry: BwdDiffEntry, *, dtype: str, numthreads: int) -> str:
    pc_fields = "".join(f"    {dtype} {name};\n" for name in entry.no_diff_params)
    no_diff_args = "".join(f"{name}, " for name in entry.no_diff_params)
    return (
        f"import {entry.module};\n"
        f"\n"
        f"[[vk::push_constant]] cbuffer Push {{\n"
        f"{pc_fields}"
        f"    uint numel;\n"
        f"}};\n"
        f"\n"
        f'[shader("compute")][numthreads({numthreads},1,1)]\n'
        f"void bwd_op(\n"
        f"    uniform StructuredBuffer<{dtype}> a,\n"
        f"    uniform StructuredBuffer<{dtype}> b,\n"
        f"    uniform StructuredBuffer<{dtype}> grad_out,\n"
        f"    uniform RWStructuredBuffer<{dtype}> grad_a,\n"
        f"    uniform RWStructuredBuffer<{dtype}> grad_b,\n"
        f"    uint3 tid : SV_DispatchThreadID\n"
        f") {{\n"
        f"    if (tid.x >= numel) return;\n"
        f"    DifferentialPair<{dtype}> dpa = "
        f"diffPair(a[tid.x], ({dtype})0);\n"
        f"    DifferentialPair<{dtype}> dpb = "
        f"diffPair(b[tid.x], ({dtype})0);\n"
        f"    bwd_diff({entry.fwd_fn})("
        f"dpa, dpb, {no_diff_args}grad_out[tid.x]);\n"
        f"    grad_a[tid.x] = dpa.getDifferential();\n"
        f"    grad_b[tid.x] = dpb.getDifferential();\n"
        f"}}\n"
    )
