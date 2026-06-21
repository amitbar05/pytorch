"""TR.19 — Consolidated backward lowerings (single source of truth).

Every ``aten.<op>_backward`` lowering lives here — NOT in ``lowerings/``.
This satisfies anti-goal #3: zero ``@register_lowering(aten.*_backward)``
decorators in ``lowerings/``.

The file is organized in three sections:

1. **Bwd-diff auto-generated** — unary/binary ops whose forward has a
   ``[Differentiable]`` annotation.  A single auto-registration loop
   walks ``BWD_DIFF_TABLE`` and registers a thin redirect through
   ``_ensure_unary_bwd_diff_op`` / ``_ensure_binary_loss_bwd_diff_op``.

2. **Algebraic / non-bwd-diff** — ops that CANNOT use bwd_diff because:
   - They receive the forward *output* (y), not the input (x)
     (``sigmoid_backward``, ``tanh_backward``).
   - They are not autodiff-eligible (``native_dropout_backward``).
   - ``leaky_relu_backward`` when ``self_is_result=True`` (saved tensor
     is y, not x).  The inline bwd_diff path in
     ``bwd_diff_inline_lowering.py`` handles the common
     ``self_is_result=False`` case; this algebraic lowering is the
     fallback.

3. **Complex decompositions** — norm/softmax backward ops that genuinely
   need reductions + pointwise chains.  The decomposition logic is the same
   that previously lived in ``lowerings/{norm,softmax}.py``.
   Norm backward (layer_norm, group_norm, batch_norm) is in
   ``bwd_lowerings_norm.py`` (M22.6 split to stay under the 800-line cap).

Exit gate: ``git grep '@register_lowering(aten.*_backward' lowerings/``
returns zero hits.
"""

from __future__ import annotations

from .bwd_lowerings_norm import register_norm_backward_lowerings
from .lowerings.bwd_diff import (
    _ensure_binary_loss_bwd_diff_op,
    _ensure_unary_bwd_diff_op,
)
from .lowerings.embedding import (
    _get_embedding_bag_backward_impl,
    _get_embedding_dense_backward_impl,
)
from .lowerings.loss import _get_nll_loss_backward_impl
from .lowerings.pool import _adaptive_avg_pool2d_backward_vulkan
from .lowerings.conv_backward import _get_conv_backward_lowering_impl


def _is_vulkan(x) -> bool:
    try:
        return x.get_device().type == "vulkan"
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════
# 1.  Auto-generated bwd_diff lowerings
# ═══════════════════════════════════════════════════════════════════════

# Subsets of BWD_DIFF_TABLE keys that need @register_lowering.  Not every
# entry needs a lowering — many backward ops (sin_backward, cos_backward,
# etc.) are decomposed by AOTAutograd into primitives before Inductor sees
# them.  Only ops that actually appear in compiled backward graphs need
# registration.

_UNARY_BWD_DIFF_LOWERING_OPS: set[str] = {
    "aten.relu_backward",
    "aten.threshold_backward",
    "aten.silu_backward",
    "aten.elu_backward",
    "aten.hardswish_backward",
    "aten.hardsigmoid_backward",
    # M-AG5.1 Tier-3 (2026-05-24): aten.mish_backward removed from the
    # bwd_diff compile path. slangc v2026.7.1 does not correctly propagate
    # [BackwardDerivative(mish_fast_bwd)] across module import boundaries —
    # bwd_diff(mish_fwd) returns all-zero gradients in compile mode.
    # mish_fwd stays in BWD_DIFF_TABLE so dispatch_unary_bwd (eager path)
    # still works. Compile mode lowers algebraically in
    # _register_algebraic_backward_lowerings below.
}
# NOTE: aten.gelu_backward is NOT in the auto-generated set because
# gelu_fwd in pointwise.slang uses the tanh approximation, while
# PyTorch's default approximate="none" uses the exact erf formula.
# gelu_backward is handled in _register_algebraic_backward_lowerings().
#
# NOTE: aten.softplus_backward is now routed through BWD_DIFF_TABLE
# (2026-05-29). softplus_fwd in pointwise.slang declares
# ``no_diff float beta, no_diff float threshold`` parameters, matching
# the aten signature. The inline bwd_diff lowering in
# bwd_diff_inline_lowering.py handles it with a custom registration
# (similar to leaky_relu pattern).

_BINARY_BWD_DIFF_LOWERING_OPS: set[str] = {
    "aten.mse_loss_backward",
    "aten.l1_loss_backward",
    # TRAIN.1 (2026-05-27): binary_cross_entropy_with_logits_backward was
    # missing from lowering registration. The custom_op factory in bwd_diff.py
    # creates torch_vulkan::binary_cross_entropy_with_logits_backward_bwd_diff,
    # but no @register_lowering was installed to route aten op -> custom op.
    # Adding here fixes reachability for multi-label classification training.
    "aten.binary_cross_entropy_with_logits_backward",
    "aten.smooth_l1_loss_backward",
    "aten.huber_loss_backward",
}


def _register_bwd_diff_lowerings() -> None:
    """Register a thin redirect for every bwd_diff-eligible backward op.

    Each lowering calls ``_ensure_unary_bwd_diff_op(aten_op)`` (or the
    binary variant) and routes through ``L.fallback_handler(op)`` —
    identical to what the previous per-op lowerings in
    ``lowerings/{activation,loss}.py`` did.
    """
    import torch
    from torch._inductor import lowering as L
    from torch._inductor.lowering import register_lowering

    aten = torch.ops.aten

    # ── Unary ──────────────────────────────────────────────────────────
    for aten_op in sorted(_UNARY_BWD_DIFF_LOWERING_OPS):
        op = _ensure_unary_bwd_diff_op(aten_op)
        # Capture op in closure — the default arg binds at definition time.
        _make_unary_bwd_diff_lowering(aten_op, op, register_lowering, L, aten)

    # ── Binary loss ────────────────────────────────────────────────────
    for aten_op in sorted(_BINARY_BWD_DIFF_LOWERING_OPS):
        op = _ensure_binary_loss_bwd_diff_op(aten_op)
        _make_binary_bwd_diff_lowering(aten_op, op, register_lowering, L, aten)

    # ── Special: binary_cross_entropy_backward with weight=None only ────
    # Registered separately because the weight≠None case falls through.
    _register_bce_backward_special(register_lowering, L, aten)
    # TRAIN.1 (2026-05-27): same pattern for BCE with logits backward —
    # weight/pos_weight kwargs must trigger NotImplemented fallthrough.
    _register_bce_with_logits_backward_special(register_lowering, L, aten)


def _make_unary_bwd_diff_lowering(
    aten_op: str,
    op,
    register_lowering,
    L,
    aten,
) -> None:
    """Install a ``@register_lowering`` for a unary bwd_diff op."""
    short = aten_op.split(".", 1)[1]  # e.g. "silu_backward"
    try:
        target = getattr(aten, short)
    except AttributeError:
        return

    @register_lowering(target, type_promotion_kind=None)
    def _lowering(grad_output, self, *_scalars, _op=op, _L=L):
        if not _is_vulkan(grad_output):
            return NotImplemented
        return _L.fallback_handler(_op)(grad_output, self)

    # Attach a stable __name__ for debugging / audit tools.
    _lowering.__name__ = f"_vulkan_{short}_bwd_diff"


def _make_binary_bwd_diff_lowering(
    aten_op: str,
    op,
    register_lowering,
    L,
    aten,
) -> None:
    """Install a ``@register_lowering`` for a binary loss bwd_diff op.

    Loss backward ops accept ``(grad_output, self, target, reduction, ...)``
    but the bwd_diff kernel only needs ``(self, target)`` — the ``grad_output``
    and ``reduction`` are handled by the upstream decomposition.
    """
    short = aten_op.split(".", 1)[1]  # e.g. "mse_loss_backward"
    try:
        target = getattr(aten, short)
    except AttributeError:
        return

    # Some loss backward ops have extra scalar args (beta, delta).
    extra_params = {
        "aten.smooth_l1_loss_backward": ["beta"],
        "aten.huber_loss_backward": ["delta"],
    }.get(aten_op, [])

    @register_lowering(target, type_promotion_kind=None)
    def _lowering(grad_output, self, target_tensor, reduction, *args, **kwargs):
        if not _is_vulkan(self):
            return NotImplemented
        # Collect extra scalar params from args/kwargs.
        if extra_params:
            scalar_args = list(args) + [kwargs[p] for p in extra_params if p in kwargs]
            return L.fallback_handler(op)(
                grad_output, self, target_tensor, *scalar_args
            )
        return L.fallback_handler(op)(grad_output, self, target_tensor)

    _lowering.__name__ = f"_vulkan_{short}_bwd_diff"


def _register_bce_backward_special(register_lowering, L, aten) -> None:
    """BCE backward with optional weight — bwd_diff only when weight is None."""
    op = _ensure_binary_loss_bwd_diff_op("aten.binary_cross_entropy_backward")

    @register_lowering(aten.binary_cross_entropy_backward, type_promotion_kind=None)
    def _vulkan_bce_backward(grad_output, self, target, weight=None, reduction=1):
        if not _is_vulkan(self):
            return NotImplemented
        if weight is not None:
            return NotImplemented
        return L.fallback_handler(op)(grad_output, self, target)

    _vulkan_bce_backward.__name__ = "_vulkan_binary_cross_entropy_backward_bwd_diff"


def _register_bce_with_logits_backward_special(register_lowering, L, aten) -> None:
    """BCE-with-logits backward with optional weight/pos_weight — bwd_diff only
    when both weight and pos_weight are None.

    TRAIN.1 (2026-05-27): Without this special handler, the generic lowering in
    _make_binary_bwd_diff_lowering silently drops weight/pos_weight arguments,
    producing incorrect gradients when they are non-None. The NotImplemented
    fallthrough lets the upstream lowering (or the AOT decomposition) handle
    the weighted case correctly.
    """
    if not hasattr(aten, "binary_cross_entropy_with_logits_backward"):
        return  # Op not available in this PyTorch version

    op = _ensure_binary_loss_bwd_diff_op(
        "aten.binary_cross_entropy_with_logits_backward"
    )

    @register_lowering(
        aten.binary_cross_entropy_with_logits_backward,
        type_promotion_kind=None,
    )
    def _vulkan_bce_logits_backward(
        grad_output, self, target, weight=None, reduction=1, pos_weight=None
    ):
        if not _is_vulkan(self):
            return NotImplemented
        if weight is not None or pos_weight is not None:
            return NotImplemented
        return L.fallback_handler(op)(grad_output, self, target)

    _vulkan_bce_logits_backward.__name__ = (
        "_vulkan_binary_cross_entropy_with_logits_backward_bwd_diff"
    )


# ═══════════════════════════════════════════════════════════════════════
# 2.  Algebraic / non-bwd-diff lowerings
# ═══════════════════════════════════════════════════════════════════════


def _register_algebraic_backward_lowerings() -> None:
    """Register backward lowerings that CANNOT use bwd_diff.

    Reasons vary per op — see inline comments.
    """
    import torch
    from torch._inductor import lowering as L
    from torch._inductor.lowering import register_lowering

    aten = torch.ops.aten

    # ── C1: threshold_backward AOT decomp ──────────────────────────
    from torch._decomp import decomposition_table

    if torch.ops.aten.threshold_backward.default not in decomposition_table:

        def _decomp_threshold_backward(grad_output, self, threshold):
            return torch.where(self > threshold, grad_output, 0.0)

        decomposition_table[torch.ops.aten.threshold_backward.default] = (
            _decomp_threshold_backward
        )

    # ── gelu_backward ─────────────────────────────────────────────
    @register_lowering(aten.gelu_backward, type_promotion_kind=None)
    def _vulkan_gelu_backward(grad_output, self, *, approximate="none"):
        if not _is_vulkan(grad_output):
            return NotImplemented
        import math

        if approximate == "none":
            sqrt2 = 1.4142135623730951
            inv_sqrt_2pi = 0.3989422804014327
            x_div_sqrt2 = L.lowerings[aten.div.Scalar](self, sqrt2)
            erf_term = L.lowerings[aten.erf.default](x_div_sqrt2)
            half_erf = L.lowerings[aten.mul.Scalar](erf_term, 0.5)
            phi_cdf = L.lowerings[aten.add.Scalar](half_erf, 0.5)
            neg_half = L.lowerings[aten.mul.Scalar](self, -0.5)
            x_sq = L.lowerings[aten.mul.Tensor](neg_half, self)
            exp_term = L.lowerings[aten.exp.default](x_sq)
            phi_pdf = L.lowerings[aten.mul.Scalar](exp_term, inv_sqrt_2pi)
            x_times_pdf = L.lowerings[aten.mul.Tensor](self, phi_pdf)
            dgelu = L.lowerings[aten.add.Tensor](phi_cdf, x_times_pdf)
            return L.lowerings[aten.mul.Tensor](grad_output, dgelu)
        else:
            k = 0.7978845608028654
            c = 0.044715
            x3 = L.lowerings[aten.mul.Tensor](
                self, L.lowerings[aten.mul.Tensor](self, self)
            )
            c_x3 = L.lowerings[aten.mul.Scalar](x3, c)
            u = L.lowerings[aten.add.Tensor](self, c_x3)
            k_u = L.lowerings[aten.mul.Scalar](u, k)
            tanh_ku = L.lowerings[aten.tanh.default](k_u)
            one_plus_tanh = L.lowerings[aten.add.Scalar](tanh_ku, 1.0)
            tanh_sq = L.lowerings[aten.mul.Tensor](tanh_ku, tanh_ku)
            neg_tanh_sq = L.lowerings[aten.neg.default](tanh_sq)
            d_tanh = L.lowerings[aten.add.Scalar](neg_tanh_sq, 1.0)
            three_c = 3.0 * c
            three_c_x2 = L.lowerings[aten.mul.Scalar](
                L.lowerings[aten.mul.Tensor](self, self), three_c
            )
            one_plus_3cx2 = L.lowerings[aten.add.Scalar](three_c_x2, 1.0)
            k_term = L.lowerings[aten.mul.Scalar](one_plus_3cx2, k)
            x_dtanh = L.lowerings[aten.mul.Tensor](self, d_tanh)
            term2 = L.lowerings[aten.mul.Tensor](x_dtanh, k_term)
            inner_sum = L.lowerings[aten.add.Tensor](one_plus_tanh, term2)
            dgelu = L.lowerings[aten.mul.Scalar](inner_sum, 0.5)
            return L.lowerings[aten.mul.Tensor](grad_output, dgelu)

    # ── sigmoid_backward ───────────────────────────────────────────
    # CANNOT use bwd_diff(sigmoid_fwd): PyTorch saves the *output* y =
    # sigmoid(x), not x. bwd_diff would differentiate sigmoid(y) instead
    # of computing y * (1 - y).  Algebraic form is required.
    @register_lowering(aten.sigmoid_backward, type_promotion_kind=None)
    def _vulkan_sigmoid_backward(grad_output, output):
        if not _is_vulkan(grad_output):
            return NotImplemented
        # y * (1 - y)
        neg_y = L.lowerings[aten.neg.default](output)
        one_minus_y = L.lowerings[aten.add.Scalar](neg_y, 1.0)
        out_times_one_minus = L.lowerings[aten.mul.Tensor](output, one_minus_y)
        return L.lowerings[aten.mul.Tensor](grad_output, out_times_one_minus)

    # ── tanh_backward ──────────────────────────────────────────────
    # CANNOT use bwd_diff(tanh_fwd): same reason as sigmoid — receives
    # the output y = tanh(x).  Algebraic form: grad * (1 - y^2).
    @register_lowering(aten.tanh_backward, type_promotion_kind=None)
    def _vulkan_tanh_backward(grad_output, output):
        if not _is_vulkan(grad_output):
            return NotImplemented
        sq = L.lowerings[aten.mul.Tensor](output, output)
        neg_sq = L.lowerings[aten.neg.default](sq)
        one_minus_sq = L.lowerings[aten.add.Scalar](neg_sq, 1.0)
        return L.lowerings[aten.mul.Tensor](grad_output, one_minus_sq)

    # ── leaky_relu_backward (fallback) ─────────────────────────────
    # The inline bwd_diff lowering in bwd_diff_inline_lowering.py
    # handles the common case (self_is_result=False, saved tensor is x).
    # This algebraic path is the fallback for self_is_result=True or
    # when inline bwd_diff is disabled (TORCH_VULKAN_INLINE_BWD_DIFF=0).
    @register_lowering(aten.leaky_relu_backward, type_promotion_kind=None)
    def _vulkan_leaky_relu_backward(
        grad_output, self_or_result, negative_slope, self_is_result
    ):
        if not _is_vulkan(grad_output):
            return NotImplemented
        gt0 = L.lowerings[aten.gt.Scalar](self_or_result, 0)
        scaled = L.lowerings[aten.mul.Scalar](grad_output, negative_slope)
        return L.lowerings[aten.where.self](gt0, grad_output, scaled)

    # ── softplus_backward (removed 2026-05-29) ────────────────────────
    # M-AG5.1 Tier-2: aten.softplus_backward is now routed through
    # BWD_DIFF_TABLE with no_diff_params=("beta", "threshold").
    # The inline bwd_diff lowering in bwd_diff_inline_lowering.py
    # handles the 4-arg signature with a custom registration.

    # ── mish_backward ──────────────────────────────────────────────
    # M-AG5.1 Tier-3 (2026-05-24): slangc v2026.7.1 does not correctly
    # propagate [BackwardDerivative(mish_fast_bwd)] across module import
    # boundaries — bwd_diff(mish_fwd) returns all-zero gradients in
    # compile mode. mish_fwd stays in BWD_DIFF_TABLE so dispatch_unary_bwd
    # (eager path) still works. Compile mode lowers algebraically:
    #
    #   mish'(x) = tanh(sp) + x * (1 - tanh(sp)^2) * sigmoid(x)
    #   where sp = log(1 + exp(x))  [softplus(x)]
    #
    @register_lowering(aten.mish_backward, type_promotion_kind=None)
    def _vulkan_mish_backward(grad_output, self):
        if not _is_vulkan(grad_output):
            return NotImplemented
        sp = L.lowerings[aten.log1p.default](L.lowerings[aten.exp.default](self))
        th = L.lowerings[aten.tanh.default](sp)
        th_sq = L.lowerings[aten.mul.Tensor](th, th)
        neg_th_sq = L.lowerings[aten.mul.Scalar](th_sq, -1.0)
        sech2 = L.lowerings[aten.add.Scalar](neg_th_sq, 1.0)  # 1 - tanh(sp)^2
        sig = L.lowerings[aten.sigmoid.default](self)
        x_sech2_sig = L.lowerings[aten.mul.Tensor](
            self, L.lowerings[aten.mul.Tensor](sech2, sig)
        )
        grad_fn = L.lowerings[aten.add.Tensor](th, x_sech2_sig)
        return L.lowerings[aten.mul.Tensor](grad_output, grad_fn)

    # ── native_dropout_backward ────────────────────────────────────
    # Not autodiff-eligible — simple mask * scale * grad.
    @register_lowering(aten.native_dropout_backward, type_promotion_kind=None)
    def _vulkan_native_dropout_backward(grad_output, mask, scale):
        if not _is_vulkan(grad_output):
            return NotImplemented
        if mask.get_dtype() == torch.bool:
            mask = L.lowerings[aten.to.dtype](mask, grad_output.get_dtype())
        gm = L.lowerings[aten.mul.Tensor](grad_output, mask)
        return L.lowerings[aten.mul.Scalar](gm, float(scale))


# ═══════════════════════════════════════════════════════════════════════
# 3.  Complex decompositions (softmax backward)
# ═══════════════════════════════════════════════════════════════════════
# Norm backward (layer_norm / group_norm / batch_norm) lives in
# bwd_lowerings_norm.py — imported above as register_norm_backward_lowerings.


def _register_reduction_backward() -> None:
    """CG.M3 — Reduction backward lowerings decomposed into primitives.

    Decomposes ``aten.{sum,mean,var,prod}_backward`` into pointwise
    and reduction primitives that the Vulkan backend already supports.
    This closes anti-goal #3 for the reduction class — backward routes
    through the decomposition here rather than hand-rolled lowerings.

    The [Differentiable] fold functions in shaders/lib/reduction.slang
    (reduce_fold_sum / reduce_fold_prod) provide the autodiff proof;
    the runtime dispatch for the broadcast step goes through
    bwd_diff_dispatch.dispatch_reduction_bwd when the decomposition
    is suppressed.
    """
    import torch
    from torch._inductor import lowering as L
    from torch._inductor.lowering import register_lowering

    aten = torch.ops.aten

    # ── sum_backward(grad, sizes, dim, keepdim) ──────────────────────
    # Just expand grad to match the input shape.
    try:
        _sum_bwd_target = aten.sum_backward.default
    except AttributeError:
        _sum_bwd_target = None

    if _sum_bwd_target is not None:

        @register_lowering(_sum_bwd_target, type_promotion_kind=None)
        def _vulkan_sum_backward(grad_output, sizes, dim, keepdim):
            if not _is_vulkan(grad_output):
                return NotImplemented
            # expand grad_output to sizes
            return L.lowerings[aten.expand.default](grad_output, list(sizes))

    # ── mean_backward(grad, sizes, dim, numel, keepdim) ──────────────
    # grad / numel expanded to sizes.
    try:
        _mean_bwd_target = aten.mean_backward.default
    except AttributeError:
        _mean_bwd_target = None

    if _mean_bwd_target is not None:

        @register_lowering(_mean_bwd_target, type_promotion_kind=None)
        def _vulkan_mean_backward(grad_output, sizes, dim, numel, keepdim):
            if not _is_vulkan(grad_output):
                return NotImplemented
            inv_numel = 1.0 / float(numel)
            scaled = L.lowerings[aten.mul.Scalar](grad_output, inv_numel)
            return L.lowerings[aten.expand.default](scaled, list(sizes))

    # ── var_backward(grad, self, dim, correction, keepdim) ───────────
    # grad * 2 * (self - mean) / (N - correction)
    # Registered for both aten.var_backward and aten.var.correction_backward.
    for _var_bwd_name in ("var_backward", "var.correction_backward"):
        try:
            _var_bwd_target = getattr(aten, _var_bwd_name).default
        except AttributeError:
            continue

        @register_lowering(_var_bwd_target, type_promotion_kind=None)
        def _vulkan_var_backward(
            grad_output,
            self,
            dim,
            correction,
            keepdim,
            _var_name=_var_bwd_name,
        ):
            if not _is_vulkan(grad_output):
                return NotImplemented
            # Compute mean over the reduced dimensions.
            mean = L.lowerings[aten.mean.dim](self, dim, True)
            # centered = self - mean
            centered = L.lowerings[aten.sub.Tensor](self, mean)
            # N = product of sizes over dim
            dims = dim if dim is not None else list(range(self.get_ndim()))
            N = 1
            for d in dims:
                N *= self.get_size()[d]
            denom = float(N) - float(correction)
            if denom <= 0.0:
                denom = 1.0
            scale = 2.0 / denom
            # grad_in = grad_output * centered * scale
            scaled_grad = L.lowerings[aten.mul.Scalar](grad_output, scale)
            return L.lowerings[aten.mul.Tensor](scaled_grad, centered)

    # ── prod_backward(grad, self, result[, dim, keepdim]) ────────────
    # grad * result.expand_as(self) / self
    for _prod_bwd_name in ("prod_backward", "prod.dim_int_backward"):
        try:
            _prod_bwd_target = getattr(aten, _prod_bwd_name).default
        except AttributeError:
            continue

        @register_lowering(_prod_bwd_target, type_promotion_kind=None)
        def _vulkan_prod_backward(
            grad_output,
            self,
            result,
            *args,
            _prod_name=_prod_bwd_name,
        ):
            if not _is_vulkan(grad_output):
                return NotImplemented
            # result may be a scalar or reduced tensor; expand to match self.
            if result.get_ndim() < self.get_ndim():
                result = L.lowerings[aten.expand.default](result, list(self.get_size()))
            div = L.lowerings[aten.div.Tensor](result, self)
            return L.lowerings[aten.mul.Tensor](grad_output, div)


def _register_softmax_backward() -> None:
    """P0.1 — ``_softmax_backward_data`` / ``_log_softmax_backward_data``.

    (Moved from ``lowerings/softmax.py`` per TR.19.)
    """
    import torch
    from torch._inductor import lowering as L
    from torch._inductor.lowering import register_lowering

    aten = torch.ops.aten

    @register_lowering(aten._softmax_backward_data, type_promotion_kind=None)
    def _vulkan_softmax_backward(grad_output, output, dim, input_dtype):
        if not _is_vulkan(grad_output):
            return NotImplemented
        prod = L.lowerings[aten.mul.Tensor](grad_output, output)
        s = L.lowerings[aten.sum.dim_IntList](prod, [dim], keepdims=True)
        diff = L.lowerings[aten.sub.Tensor](grad_output, s)
        return L.lowerings[aten.mul.Tensor](output, diff)

    @register_lowering(aten._log_softmax_backward_data, type_promotion_kind=None)
    def _vulkan_log_softmax_backward(grad_output, output, dim, input_dtype):
        if not _is_vulkan(grad_output):
            return NotImplemented
        s = L.lowerings[aten.sum.dim_IntList](grad_output, [dim], keepdims=True)
        ex = L.lowerings[aten.exp.default](output)
        prod = L.lowerings[aten.mul.Tensor](ex, s)
        return L.lowerings[aten.sub.Tensor](grad_output, prod)


def _register_pool_backward() -> None:
    """M22.15 + TRAIN.2 + CODEGEN.2 — pool backward ops.

    avg_pool2d_backward: CODEGEN.2 adds a pure-codegen path for the
    non-overlapping case (stride==kernel_size, no ceil_mode).  The
    upstream lowering uses ops.indirect_indexing which generates incorrect
    SPIR-V on Vulkan; the codegen path avoids it via broadcast+scale.
    Overlapping / complex cases still route through FallbackKernel.

    max_pool2d_with_indices (forward): AOTAutograd rematerialises indices in
    the backward graph rather than saving them from the forward.  The upstream
    Inductor lowering uses ops.indirect_indexing → wrong Vulkan SPIR-V for
    the index output; route through FallbackKernel so the remat produces
    correct int64 flat indices.

    max_pool2d_with_indices_backward (TRAIN.2): replaced the old FallbackKernel
    path (C++ GPU shader + int64→uint32 CPU roundtrip) with a GPU-only scatter
    custom op ``torch_vulkan::max_pool2d_scatter_bwd``.  The custom op computes
    int32 global flat indices on GPU and dispatches ``scatter_add`` via the
    ``scatter_atomic.slang`` template.  No CPU roundtrip, no int64 SPIR-V.
    """
    import torch
    from torch._inductor.lowering import (
        fallback_handler,
        register_lowering,
    )

    from .fx_passes.eager.pool import (
        _ensure_max_pool2d_scatter_bwd_op_registered,
        _ensure_avg_pool2d_scatter_bwd_op_registered,
    )

    aten = torch.ops.aten

    _vk_avg_pool2d_bwd_fallback = fallback_handler(
        aten.avg_pool2d_backward.default,
        add_to_fallback_set=False,
    )

    # CODEGEN.2: scatter_bwd path for overlapping avg_pool2d backward.
    _ensure_avg_pool2d_scatter_bwd_op_registered()
    _vk_avg_pool2d_scatter_bwd_fb = fallback_handler(
        torch.ops.torch_vulkan.avg_pool2d_scatter_bwd.default,
        add_to_fallback_set=False,
    )

    # Use .default (specific overload) instead of the packet so
    # register_lowering's get_overloads() guard does not skip the
    # entry when the upstream lowering has already registered .default.
    @register_lowering(aten.avg_pool2d_backward.default, type_promotion_kind=None)
    def _vulkan_avg_pool2d_bwd(
        grad_output, x, kernel_size, stride, padding, ceil_mode,
        count_include_pad, divisor_override
    ):
        if not _is_vulkan(grad_output):
            return NotImplemented
        # CODEGEN.2: pure-codegen backward for non-overlapping avg_pool2d.
        # When stride == kernel_size and no padding/ceil_mode, each input
        # pixel maps to exactly one output pixel — the gradient is simply
        # broadcast(grad_output / scale) to input spatial dims.
        # Avoids the upstream lowering's ops.indirect_indexing which
        # generates incorrect SPIR-V on Vulkan.
        from .lowerings.pool import avg_pool2d_backward_codegen

        # M23.2-spinoff: when the forward output is a DonatedBuffer (e.g.,
        # conv2d output reused as pool input), the ops.mul/reshape/expand
        # chain inside avg_pool2d_backward_codegen produces OpsValue nodes
        # that the lowering framework can't wrap into top-level IR.
        # Route directly to the scatter_bwd fallback which handles donated
        # buffers correctly via FallbackKernel.
        from torch._inductor.ir import StorageBox, DonatedBuffer, TensorBox
        _x_is_donated = False
        if isinstance(x, TensorBox):
            inner = x.data
            if isinstance(inner, StorageBox):
                if isinstance(inner.data, DonatedBuffer):
                    _x_is_donated = True

        if not _x_is_donated:
            result = avg_pool2d_backward_codegen(
                grad_output, x, kernel_size, stride, padding, ceil_mode,
                count_include_pad, divisor_override,
            )
            if result is not None:
                return result

        # CODEGEN.2 (overlapping): use scatter_add via custom op.
        # This avoids the old FallbackKernel eager Vulkan path and routes
        # through the scatter_atomic Slang template in the compile path.
        if not stride:
            stride = kernel_size
        if not padding:
            padding = [0, 0]
        if not (isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 2):
            return _vk_avg_pool2d_bwd_fallback(
                grad_output, x, kernel_size, stride, padding, ceil_mode,
                count_include_pad, divisor_override,
            )
        if not (isinstance(stride, (list, tuple)) and len(stride) == 2):
            return _vk_avg_pool2d_bwd_fallback(
                grad_output, x, kernel_size, stride, padding, ceil_mode,
                count_include_pad, divisor_override,
            )
        if not (isinstance(padding, (list, tuple)) and len(padding) == 2):
            return _vk_avg_pool2d_bwd_fallback(
                grad_output, x, kernel_size, stride, padding, ceil_mode,
                count_include_pad, divisor_override,
            )

        kh, kw = kernel_size
        sh, sw = stride
        ph, pw = padding

        x_size = x.get_size()
        if len(x_size) != 4:
            return _vk_avg_pool2d_bwd_fallback(
                grad_output, x, kernel_size, stride, padding, ceil_mode,
                count_include_pad, divisor_override,
            )
        _N = int(x_size[0])
        _C = int(x_size[1])
        _iH = int(x_size[2])
        _iW = int(x_size[3])
        _div_ovr = divisor_override if divisor_override is not None else 0
        return _vk_avg_pool2d_scatter_bwd_fb(
            grad_output, _N, _C, _iH, _iW, kh, kw, sh, sw, ph, pw,
            count_include_pad, _div_ovr,
        )

    # Forward: max_pool2d_with_indices — AOTAutograd rematerialises indices
    # in the backward, so this op appears in the backward graph too.  The
    # upstream Inductor lowering uses ops.indirect_indexing which produces
    # wrong SPIR-V on Vulkan for the indices output; route through
    # FallbackKernel instead.
    _vk_max_pool_fwd_fallback = fallback_handler(
        aten.max_pool2d_with_indices.default,
        add_to_fallback_set=False,
    )

    @register_lowering(aten.max_pool2d_with_indices.default, type_promotion_kind=None)
    def _vulkan_max_pool_with_indices(
        x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False
    ):
        if not _is_vulkan(x):
            return NotImplemented
        # Normalize scalar pool args to 2-element lists.  Inductor's
        # FallbackKernel validates that pool parameters are list/tuple
        # (ir.py:_check_kernel_args_for_fallback); the ATen schema
        # expects IntList for these slots.
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = [stride, stride]
        if isinstance(padding, int):
            padding = [padding, padding]
        if isinstance(dilation, int):
            dilation = [dilation, dilation]
        return _vk_max_pool_fwd_fallback(
            x, kernel_size, stride, padding, dilation, ceil_mode
        )

    # A5 (2026-06-16) — Forward: aten.max_pool2d and aten.avg_pool2d.
    # These route through FallbackKernel (C++ Vulkan kernel) to avoid the
    # upstream Inductor lowerings which use ops.indirect_indexing → wrong
    # SPIR-V on Vulkan.  This is a stepping stone toward full Slang codegen.
    #
    # aten.max_pool2d: upstream is CompositeImplicitAutograd — when not
    # decomposed by AOT it delegates to max_pool2d_with_indices.  The
    # suppression in lowerings/__init__.py keeps the raw op alive so our
    # FallbackKernel intercepts it directly.
    _vk_max_pool2d_fallback = fallback_handler(
        aten.max_pool2d.default,
        add_to_fallback_set=False,
    )

    @register_lowering(aten.max_pool2d.default, type_promotion_kind=None)
    def _vulkan_max_pool2d(
        x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False
    ):
        if not _is_vulkan(x):
            return NotImplemented
        # Normalize scalar pool args to 2-element lists (same as
        # max_pool2d_with_indices pattern above).
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = [stride, stride]
        if isinstance(padding, int):
            padding = [padding, padding]
        if isinstance(dilation, int):
            dilation = [dilation, dilation]
        return _vk_max_pool2d_fallback(
            x, kernel_size, stride, padding, dilation, ceil_mode
        )

    # aten.avg_pool2d: upstream Inductor lowering (_avg_poolnd) uses
    # make_loader+indirect_indexing for the complex case, producing wrong
    # SPIR-V on Vulkan.  Override with FallbackKernel (eager C++ dispatch).
    # S2.5 (open): replace with a torch_vulkan custom op to avoid the aten
    # extern while keeping eager C++ dispatch.
    _vk_avg_pool2d_fallback = fallback_handler(
        aten.avg_pool2d.default,
        add_to_fallback_set=False,
    )

    @register_lowering(aten.avg_pool2d.default, type_promotion_kind=None)
    def _vulkan_avg_pool2d(
        x, kernel_size, stride=None, padding=0, ceil_mode=False,
        count_include_pad=True, divisor_override=None,
    ):
        if not _is_vulkan(x):
            return NotImplemented
        # Normalize scalar pool args to 2-element lists.
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        if stride is None or (isinstance(stride, (list, tuple)) and len(stride) == 0):
            stride = kernel_size
        elif isinstance(stride, int):
            stride = [stride, stride]
        if isinstance(padding, int):
            padding = [padding, padding]
        return _vk_avg_pool2d_fallback(
            x, kernel_size, stride, padding, ceil_mode,
            count_include_pad, divisor_override,
        )

    # Backward: max_pool2d_with_indices_backward — TRAIN.2.
    #
    # Replaced the FallbackKernel path (C++ GPU shader with int64→uint32
    # CPU roundtrip in backward_ops.cpp:436-445) with a GPU-only scatter
    # custom op (``torch_vulkan::max_pool2d_scatter_bwd``).  The custom op
    # computes global int32 indices on GPU and dispatches the scatter_add
    # Slang template — no CPU roundtrip, no int64 SPIR-V, M-CG compliant.
    _ensure_max_pool2d_scatter_bwd_op_registered()
    _vk_max_pool_scatter_bwd_fb = fallback_handler(
        torch.ops.torch_vulkan.max_pool2d_scatter_bwd.default,
        add_to_fallback_set=False,
    )

    @register_lowering(aten.max_pool2d_with_indices_backward.default, type_promotion_kind=None)
    def _vulkan_max_pool_bwd(
        grad_output, x, kernel_size, stride, padding, dilation, ceil_mode, indices
    ):
        if not _is_vulkan(grad_output):
            return NotImplemented
        # Extract static shapes from IR tensors for the custom op scalars.
        x_shape = list(x.get_size())
        if len(x_shape) != 4:
            return NotImplemented  # only 4D NCHW supported
        _N, _C, _iH, _iW = (int(s) for s in x_shape)
        return _vk_max_pool_scatter_bwd_fb(
            grad_output, indices, _N, _C, _iH, _iW,
        )


# ═══════════════════════════════════════════════════════════════════════
# 4.  Consolidated backward lowering registrations (anti-goal #3)
# ═══════════════════════════════════════════════════════════════════════
# These implementations live in lowerings/*.py but their @register_lowering
# decorators are moved HERE to satisfy the exit gate:
#   git grep '@register_lowering(aten.*_backward' lowerings/ => 0 hits


def _register_consolidated_backward_impls() -> None:
    """Register backward lowerings whose implementations are in lowerings/*.py.

    Anti-goal #3 requires zero ``@register_lowering(aten.*_backward)`` in
    ``lowerings/``.  The implementations are defined there; this function
    is the single place where they are registered with Inductor.
    """
    import torch
    from torch._inductor.lowering import register_lowering

    aten = torch.ops.aten

    # ── aten._adaptive_avg_pool2d_backward (pool.py) ──────────────────────
    # Module-level function — import already at top of file.
    register_lowering(aten._adaptive_avg_pool2d_backward, type_promotion_kind=None)(
        _adaptive_avg_pool2d_backward_vulkan
    )
    # get_overloads() skips overloads already in lowerings, so .default stays
    # pointing at the upstream make_fallback handler.  Force-override so
    # backward pool dispatch actually hits our Pointwise.create implementation.
    from torch._inductor import lowering as _L
    if aten._adaptive_avg_pool2d_backward.default in _L.lowerings:
        _L.lowerings[aten._adaptive_avg_pool2d_backward.default] = (
            _L.lowerings[aten._adaptive_avg_pool2d_backward]
        )

    # ── aten.embedding_dense_backward (embedding.py) ─────────────────────
    _emb_dense_impl = _get_embedding_dense_backward_impl()
    _emb_dense_target = aten.embedding_dense_backward
    register_lowering(_emb_dense_target, type_promotion_kind=None)(_emb_dense_impl)

    # ── aten._embedding_bag_backward (embedding.py) ──────────────────────
    _emb_bag_impl = _get_embedding_bag_backward_impl()
    _emb_bag_target = aten._embedding_bag_backward
    register_lowering(_emb_bag_target, type_promotion_kind=None)(_emb_bag_impl)

    # ── aten.nll_loss_backward (loss.py) ─────────────────────────────────
    _nll_impl = _get_nll_loss_backward_impl()
    _nll_target = aten.nll_loss_backward
    register_lowering(_nll_target, type_promotion_kind=None)(_nll_impl)

    # ── aten.convolution_backward.default (conv_backward.py) ─────────────
    _conv_impl = _get_conv_backward_lowering_impl()
    _conv_target = aten.convolution_backward.default
    register_lowering(_conv_target, type_promotion_kind=None)(_conv_impl)


# ═══════════════════════════════════════════════════════════════════════
# 5.  Master registration entry point
# ═══════════════════════════════════════════════════════════════════════


def register() -> None:
    """Register ALL backward lowerings.

    Called from ``lowerings/__init__.py:register()``.  Idempotent via
    the per-op ``_ensure_*_bwd_diff_op`` caches and Inductor's own
    lowering dedup guard.
    """
    _register_bwd_diff_lowerings()
    _register_algebraic_backward_lowerings()
    _register_reduction_backward()
    register_norm_backward_lowerings()  # layer_norm / group_norm / batch_norm
    _register_softmax_backward()
    _register_pool_backward()  # M22.15 — max_pool2d_with_indices_backward
    _register_consolidated_backward_impls()  # anti-goal #3 exit gate
