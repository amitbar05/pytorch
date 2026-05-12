"""T3.8 — backward template registry for forward-template partners.

Every forward template registered in ``template_registry.py`` must also
register a paired backward template here.  The backward can be:

* **``bwd_diff``-generated** — the forward carries ``[Differentiable]``
  and Slang's autodiff produces the backward kernel.
* **``[BackwardDerivative]``-bound** — the forward has a hand-written
  fast backward annotation in ``shaders/lib/*.slang`` (e.g.
  ``sigmoid_fast_bwd``).

Once a ``(forward_template_key, backward_template_entry)`` pair is
registered, the lowering infrastructure can route compile-time backward
graphs through templates instead of hand-written ``lowerings/`` modules.
The exit goal: zero ``aten.<op>_backward`` lowerings in ``lowerings/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class BackwardKind(Enum):
    BWD_DIFF = auto()  # Slang autodiff over a [Differentiable] primal
    BACKWARD_DERIVATIVE = auto()  # Hand-written [BackwardDerivative(fast_bwd)]
    TEMPLATE_JINJA = (
        auto()
    )  # Backward generated from Jinja template (mm/conv/attention)


@dataclass(frozen=True)
class BwdTemplateEntry:
    fwd_template_key: str  # matches TemplateRegistry key (e.g. "matmul_f32_square")
    kind: BackwardKind
    # For BWD_DIFF: the lib function name (e.g. "tile_loop" in lib/mm.slang)
    # For BACKWARD_DERIVATIVE: the annotation name (e.g. "sigmoid_fast_bwd")
    fwd_fn: str
    module: str  # lib module (e.g. "mm", "pointwise")

    @property
    def is_bwd_diff(self) -> bool:
        return self.kind == BackwardKind.BWD_DIFF

    @property
    def is_backward_derivative(self) -> bool:
        return self.kind == BackwardKind.BACKWARD_DERIVATIVE


class BwdTemplateRegistry:
    """Paired (forward-template, backward-template) registry.

    Indexed by the forward template key (matching
    ``template_registry.TemplateKey``).  Each entry specifies how the
    backward kernel is produced.

    Singleton via ``BWD_TEMPLATE_REGISTRY``.
    """

    def __init__(self) -> None:
        self._entries: dict[str, BwdTemplateEntry] = {}

    def register(self, fwd_key: str, entry: BwdTemplateEntry) -> None:
        if fwd_key in self._entries:
            raise KeyError(
                f"BwdTemplateRegistry: duplicate backward entry for "
                f"forward template key {fwd_key!r}"
            )
        self._entries[fwd_key] = entry

    def lookup(self, fwd_key: str) -> Optional[BwdTemplateEntry]:
        return self._entries.get(fwd_key)

    def keys(self):
        return self._entries.keys()

    def __contains__(self, fwd_key: str) -> bool:
        return fwd_key in self._entries

    def __len__(self) -> int:
        return len(self._entries)


BWD_TEMPLATE_REGISTRY = BwdTemplateRegistry()


def register_bwd_template(
    fwd_key: str,
    kind: BackwardKind,
    fwd_fn: str,
    module: str,
) -> BwdTemplateEntry:
    entry = BwdTemplateEntry(
        fwd_template_key=fwd_key,
        kind=kind,
        fwd_fn=fwd_fn,
        module=module,
    )
    BWD_TEMPLATE_REGISTRY.register(fwd_key, entry)
    return entry


def lookup_bwd_template(fwd_key: str) -> Optional[BwdTemplateEntry]:
    return BWD_TEMPLATE_REGISTRY.lookup(fwd_key)


# ── CG.M5: Matmul backward via [Differentiable] tile_inner_madd ──────────
# The backward template ``slang_mm_bwd.py.jinja`` computes BOTH dA and dB
# in a single dispatch by wrapping ``bwd_diff(tile_inner_madd)`` in a tiled
# K-loop.  This replaces the naive decomposition (dA = dC @ B^T, dB = A^T @ dC
# → 2 dispatches) with 1 fused dispatch.
#
# The fwd_fn field names the [Differentiable] scalar inner in
# shaders/lib/mm_tile.slang.  The module field is "__template__" to mark
# that the backward shader body is emitted from a Jinja template rather
# than imported from a .slang-module.

register_bwd_template(
    "mm_default",
    BackwardKind.TEMPLATE_JINJA,
    "tile_inner_madd",
    "__template__",
)
register_bwd_template(
    "bmm_default",
    BackwardKind.TEMPLATE_JINJA,
    "tile_inner_madd",
    "__template__",
)
register_bwd_template(
    "addmm_default",
    BackwardKind.TEMPLATE_JINJA,
    "tile_inner_madd",
    "__template__",
)

# ── Track 4: new template backward registrations ──────────────────────────
# T4.6: Philox RNG — philox_rng.py.jinja template exists; backward via bwd_diff
register_bwd_template(
    "philox_rng_f32_uniform", BackwardKind.BWD_DIFF, "philox_rand", "helpers"
)
# CG.M7: SDPA backward — flash_attention_bwd.py.jinja template computes
# dQ, dK, dV in a single fused dispatch by recomputing the softmax P from
# saved Q/K/V/LSE, then applying the softmax backward formula:
#   dS = P * (dP - rowsum(P * dP))
#   dQ = dS @ K,  dK = dS^T @ Q,  dV = P^T @ dO
# The template uses InterlockedAdd for dK/dV (multiple Q tiles contribute
# to the same K/V positions) and register accumulation for dQ (same pattern
# as forward O accumulation).
#
# The fwd_fn field names the scalar inner in the backward template.
# The module field is the sentinel ``"__template__"`` — the backward
# shader body is emitted from the Jinja template rather than imported
# from a .slang module.
register_bwd_template(
    "flash_attention_f32_bhsd",
    BackwardKind.TEMPLATE_JINJA,
    "sdpa_score",
    "__template__",
)
# T4.8: Foreach optimizer — foreach_optimizer.py.jinja template exists,
# wired via install_external_optimizer() in vulkan_template_caller.
# T2.9 (2026-05-08): shaders/lib/training.slang stub deleted; module field
# is the sentinel ``"__template__"`` (see flash_attention entry above).
register_bwd_template(
    "foreach_sgd_f32", BackwardKind.TEMPLATE_JINJA, "sgd_step", "__template__"
)
register_bwd_template(
    "foreach_adamw_f32", BackwardKind.TEMPLATE_JINJA, "adamw_step", "__template__"
)

# ── CG.M6: Conv backward via [Differentiable] conv_inner_madd ──────────
# The backward template ``slang_conv_bwd.py.jinja`` computes dX, dW, and dB
# in a single dispatch by wrapping ``bwd_diff(conv_inner_madd)`` in the same
# tiled loop structure as the forward conv.  This replaces the naive
# im2col→mm→col2im decomposition (~8-10 dispatches) with 1 fused dispatch.
#
# The fwd_fn field names the [Differentiable] scalar inner in
# shaders/lib/conv.slang.  The module field is "__template__" to mark
# that the backward shader body is emitted from a Jinja template rather
# than imported from a .slang-module.

register_bwd_template(
    "conv_im2col_f32",
    BackwardKind.TEMPLATE_JINJA,
    "conv_inner_madd",
    "__template__",
)
register_bwd_template(
    "conv2d_default",
    BackwardKind.TEMPLATE_JINJA,
    "conv_inner_madd",
    "__template__",
)

# ── CG.M3: Reduction backward via [Differentiable] scalar fold ──────────
# The per-element fold functions reduce_fold_sum / reduce_fold_prod in
# shaders/lib/reduction.slang are annotated [Differentiable].  Slang
# autodiff over these produces the gradient contribution for each
# participating element: bwd_diff(reduce_fold_sum)(dpa, dpb, dout) yields
# dpa.d = dout, dpb.d = dout → every element's gradient = grad_output.
#
# The backward of a reduction IS a broadcast of the output gradient (sum),
# optionally scaled (mean: /numel, var: 2*(x-mean)/(n-1), prod: *prod/x_i).
# Max/min are NOT differentiable — they route gradient to the argmax/argmin
# position (sparse backward, handled separately).
#
# BWD_DIFF entries: the fwd_fn names the [Differentiable] fold in
# shaders/lib/reduction.slang.  The module is "reduction".  The backward
# dispatch emits a broadcast kernel that implicitly chains through
# bwd_diff(reduce_fold_<op>) to produce the correct per-element gradient.

register_bwd_template(
    "reduce_sum", BackwardKind.BWD_DIFF, "reduce_fold_sum", "reduction"
)
register_bwd_template(
    "reduce_prod", BackwardKind.BWD_DIFF, "reduce_fold_prod", "reduction"
)
# mean and var backward reuse reduce_fold_sum (their autodiff chain
# differs only in the scaling factor applied to the broadcast gradient).
register_bwd_template(
    "reduce_mean", BackwardKind.BWD_DIFF, "reduce_fold_sum", "reduction"
)
register_bwd_template(
    "reduce_var", BackwardKind.BWD_DIFF, "reduce_fold_sum", "reduction"
)

# ── CG.M4: Norm backward via [Differentiable] composable ────────────────
# The per-element norm chains (layer_norm_chain, layer_norm_chain_no_affine,
# rms_norm_chain, rms_norm_chain_no_affine) in shaders/lib/norm.slang are
# annotated [Differentiable] with fully differentiable stats (mean, rstd are
# NOT no_diff).  bwd_diff over these chains produces per-element gradients for
# x, weight, bias, AND the stats contributions (d_mean, d_rstd).
#
# The welford reduction step uses [BackwardDerivative(welford_combine_bwd)]
# on welford_combine in shaders/lib/reduction.slang so bwd_diff can propagate
# gradients through the reduction tree.

register_bwd_template(
    "layer_norm_chain",
    BackwardKind.BWD_DIFF,
    "layer_norm_chain",
    "norm",
)
register_bwd_template(
    "layer_norm_chain_no_affine",
    BackwardKind.BWD_DIFF,
    "layer_norm_chain_no_affine",
    "norm",
)
register_bwd_template(
    "rms_norm_chain",
    BackwardKind.BWD_DIFF,
    "rms_norm_chain",
    "norm",
)
register_bwd_template(
    "rms_norm_chain_no_affine",
    BackwardKind.BWD_DIFF,
    "rms_norm_chain_no_affine",
    "norm",
)
# The welford reduction step uses BACKWARD_DERIVATIVE kind:
# bwd_diff(welford_combine) emits the hand-written welford_combine_bwd.
register_bwd_template(
    "welford_combine",
    BackwardKind.BACKWARD_DERIVATIVE,
    "welford_combine",
    "reduction",
)
