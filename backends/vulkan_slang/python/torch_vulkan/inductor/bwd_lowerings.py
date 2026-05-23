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
   - They have a ``no_diff`` scalar that the unary emitter doesn't handle
     (``leaky_relu_backward``).
   - They are not autodiff-eligible (``native_dropout_backward``).

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
from .lowerings.embedding import _register_embedding_bag_backward


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
    "aten.mish_backward",
}
# NOTE: aten.gelu_backward is NOT in the auto-generated set because
# gelu_fwd in pointwise.slang uses the tanh approximation, while
# PyTorch's default approximate="none" uses the exact erf formula.
# gelu_backward is handled in _register_algebraic_backward_lowerings().
#
# NOTE: aten.softplus_backward is also NOT in the auto-generated set
# (M-AG5.1 Tier-2, 2026-05-22). The aten signature carries
# ``beta``/``threshold`` Scalar params, but the unary bwd_diff lowering
# template ``_lowering(grad_output, self)`` has no slots for them, and
# the ``softplus_fwd`` shader does not declare matching ``no_diff``
# scalars. softplus_backward is lowered algebraically (leaky_relu
# pattern) in ``_register_algebraic_backward_lowerings``.

_BINARY_BWD_DIFF_LOWERING_OPS: set[str] = {
    "aten.mse_loss_backward",
    "aten.binary_cross_entropy_backward",
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

    # ── leaky_relu_backward ────────────────────────────────────────
    # CANNOT use bwd_diff: leaky_relu_fwd takes a ``no_diff float alpha``
    # that the unary bwd_diff emitter doesn't handle (CG.M1 exclusion).
    @register_lowering(aten.leaky_relu_backward, type_promotion_kind=None)
    def _vulkan_leaky_relu_backward(
        grad_output, self_or_result, negative_slope, self_is_result
    ):
        if not _is_vulkan(grad_output):
            return NotImplemented
        gt0 = L.lowerings[aten.gt.Scalar](self_or_result, 0)
        scaled = L.lowerings[aten.mul.Scalar](grad_output, negative_slope)
        return L.lowerings[aten.where.self](gt0, grad_output, scaled)

    # ── softplus_backward ──────────────────────────────────────────
    # M-AG5.1 Tier-2 (2026-05-22). CANNOT use bwd_diff:
    # ``softplus_backward(grad_output, self, beta, threshold)`` carries
    # two Scalar args that ``softplus_fwd`` in
    # ``shaders/lib/pointwise.slang`` does not declare (the shader is the
    # ``beta=1`` form ``log(1 + exp(x))``). Wiring those through
    # ``no_diff_params`` would mismatch the shader signature and slangc
    # would reject the emitted ``bwd_diff(softplus_fwd)(dp, beta,
    # threshold, dOut)`` call.
    #
    # Math (matches the deleted symptom-fix in
    # ``meta_patches/decomposition_passes.py::_softplus_bwd``):
    #
    #     bx = beta * self
    #     dy/dx = 1                       if bx > threshold
    #           = sigmoid(bx)             otherwise
    #     grad_input = grad_output * dy/dx
    #
    # PyTorch's softplus default is ``beta=1, threshold=20``; we honour
    # whatever AOTAutograd hands us so the lowering stays correct for
    # ``nn.Softplus(beta=…, threshold=…)`` configurations.
    @register_lowering(aten.softplus_backward, type_promotion_kind=None)
    def _vulkan_softplus_backward(grad_output, self, beta, threshold):
        if not _is_vulkan(grad_output):
            return NotImplemented
        bx = L.lowerings[aten.mul.Scalar](self, beta)
        sig_bx = L.lowerings[aten.sigmoid.default](bx)
        gt_thr = L.lowerings[aten.gt.Scalar](bx, threshold)
        sig_branch = L.lowerings[aten.mul.Tensor](grad_output, sig_bx)
        return L.lowerings[aten.where.self](gt_thr, grad_output, sig_branch)

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


# ═══════════════════════════════════════════════════════════════════════
# 4.  Master registration entry point
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
    _register_embedding_bag_backward()  # OP.21 — scatter backward via decomposition
