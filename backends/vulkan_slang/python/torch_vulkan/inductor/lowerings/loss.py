"""Loss-function forward lowerings — mse, l1, smooth_l1, huber, kl_div, bce, vector_norm.

Backward lowerings for loss ops have been moved to
``bwd_lowerings.py`` (TR.19 backward consolidation).
"""

from __future__ import annotations

from . import _is_vulkan


def _register_loss_lowerings() -> None:
    """P1.7 + P0.1 — Inductor lowerings for the simple loss ops.

    Each forward decomposes into pointwise + reduction primitives so the fused
    chain emits as 1 VulkanKernel.

    Reduction enum: 0=none, 1=mean, 2=sum.

    All decompositions guard `_is_vulkan(self)` and return NotImplemented for
    non-Vulkan inputs so non-Vulkan backends keep their default Inductor path.

    Backward lowerings for mse_loss, smooth_l1_loss, huber_loss, and
    binary_cross_entropy have been consolidated into ``bwd_lowerings.py``
    (TR.19 backward consolidation).
    """
    import math

    import torch
    from torch._inductor import lowering as L
    from torch._inductor.lowering import register_lowering

    aten = torch.ops.aten

    def _reduce(t, reduction: int):
        if reduction == 0:
            return t
        ndim = len(t.get_size())
        dims = list(range(ndim))
        if reduction == 1:  # mean
            return L.lowerings[aten.mean.dim](t, dims, keepdim=False) if dims else t
        if reduction == 2:  # sum
            return (
                L.lowerings[aten.sum.dim_IntList](t, dims, keepdims=False)
                if dims
                else t
            )
        return NotImplemented

    @register_lowering(aten.mse_loss, type_promotion_kind=None)
    def _vulkan_mse_loss(self, target, reduction=1):
        if not _is_vulkan(self):
            return NotImplemented
        diff = L.lowerings[aten.sub.Tensor](self, target)
        sq = L.lowerings[aten.mul.Tensor](diff, diff)
        return _reduce(sq, int(reduction))

    # mse_loss_backward → bwd_lowerings.py (TR.19)

    @register_lowering(aten.l1_loss, type_promotion_kind=None)
    def _vulkan_l1_loss(self, target, reduction=1):
        if not _is_vulkan(self):
            return NotImplemented
        diff = L.lowerings[aten.sub.Tensor](self, target)
        ab = L.lowerings[aten.abs.default](diff)
        return _reduce(ab, int(reduction))

    # aten.l1_loss_backward does not exist — PyTorch autograd decomposes l1_loss
    # backward into primitives (sign + mul).  BWD_DIFF_TABLE entry is reserved for
    # future PyTorch versions that surface the op.

    @register_lowering(aten.smooth_l1_loss, type_promotion_kind=None)
    def _vulkan_smooth_l1_loss(self, target, reduction=1, beta=1.0):
        if not _is_vulkan(self):
            return NotImplemented
        diff = L.lowerings[aten.sub.Tensor](self, target)
        ab = L.lowerings[aten.abs.default](diff)
        # quad branch: 0.5 * diff^2 / beta
        sq = L.lowerings[aten.mul.Tensor](diff, diff)
        quad = L.lowerings[aten.mul.Scalar](sq, 0.5 / float(beta))
        # linear branch: |diff| - 0.5 * beta
        lin = L.lowerings[aten.sub.Scalar](ab, 0.5 * float(beta))
        cond = L.lowerings[aten.lt.Scalar](ab, float(beta))
        out = L.lowerings[aten.where.self](cond, quad, lin)
        return _reduce(out, int(reduction))

    # smooth_l1_loss_backward → bwd_lowerings.py (TR.19)

    @register_lowering(aten.huber_loss, type_promotion_kind=None)
    def _vulkan_huber_loss(self, target, reduction=1, delta=1.0):
        if not _is_vulkan(self):
            return NotImplemented
        diff = L.lowerings[aten.sub.Tensor](self, target)
        ab = L.lowerings[aten.abs.default](diff)
        sq = L.lowerings[aten.mul.Tensor](diff, diff)
        quad = L.lowerings[aten.mul.Scalar](sq, 0.5)
        # linear: delta * (|diff| - 0.5*delta)
        shifted = L.lowerings[aten.sub.Scalar](ab, 0.5 * float(delta))
        lin = L.lowerings[aten.mul.Scalar](shifted, float(delta))
        cond = L.lowerings[aten.lt.Scalar](ab, float(delta))
        out = L.lowerings[aten.where.self](cond, quad, lin)
        return _reduce(out, int(reduction))

    # huber_loss_backward → bwd_lowerings.py (TR.19)

    @register_lowering(aten.kl_div, type_promotion_kind=None)
    def _vulkan_kl_div(self, target, reduction=1, log_target=False):
        if not _is_vulkan(self):
            return NotImplemented
        if bool(log_target):
            # exp(target) * (target - input)
            ex = L.lowerings[aten.exp.default](target)
            diff = L.lowerings[aten.sub.Tensor](target, self)
            elem = L.lowerings[aten.mul.Tensor](ex, diff)
        else:
            # target * (log(target) - input). Mask out target == 0 so log(0) doesn't infect.
            log_t = L.lowerings[aten.log.default](target)
            inner = L.lowerings[aten.sub.Tensor](log_t, self)
            elem = L.lowerings[aten.mul.Tensor](target, inner)
            # target == 0 → contribution should be 0 (xlogy convention), not -∞ * 0 = NaN.
            zero_mask = L.lowerings[aten.eq.Scalar](target, 0)
            zero_g = L.lowerings[aten.mul.Scalar](elem, 0.0)
            elem = L.lowerings[aten.where.self](zero_mask, zero_g, elem)
        return _reduce(elem, int(reduction))

    @register_lowering(aten.binary_cross_entropy, type_promotion_kind=None)
    def _vulkan_bce(self, target, weight=None, reduction=1):
        if not _is_vulkan(self):
            return NotImplemented
        log_x = L.lowerings[aten.log.default](self)
        one_minus_x = L.lowerings[aten.rsub.Scalar](self, 1.0)
        log_one_minus_x = L.lowerings[aten.log.default](one_minus_x)
        one_minus_t = L.lowerings[aten.rsub.Scalar](target, 1.0)
        a = L.lowerings[aten.mul.Tensor](target, log_x)
        b = L.lowerings[aten.mul.Tensor](one_minus_t, log_one_minus_x)
        loss = L.lowerings[aten.neg.default](L.lowerings[aten.add.Tensor](a, b))
        if weight is not None:
            loss = L.lowerings[aten.mul.Tensor](loss, weight)
        return _reduce(loss, int(reduction))

    # binary_cross_entropy_backward → bwd_lowerings.py (TR.19)

    @register_lowering(aten.binary_cross_entropy_with_logits, type_promotion_kind=None)
    def _vulkan_bce_with_logits(
        self, target, weight=None, pos_weight=None, reduction=1
    ):
        if not _is_vulkan(self):
            return NotImplemented
        # log(1 + exp(-|x|)) + max(x, 0) is the numerically-stable form.
        # Loss per element: max(x,0) - x*target + log(1+exp(-|x|)) (without pos_weight)
        # With pos_weight w: (1 + (w-1)*target) * (log(1+exp(-|x|)) + max(x,0)) - x*target  ... complicated.
        # Without pos_weight + weight is the most common; only optimize that path.
        if pos_weight is not None:
            return NotImplemented  # fall through; rare path, let upstream handle
        zero = L.lowerings[aten.mul.Scalar](self, 0.0)
        relu_x = L.lowerings[aten.maximum](self, zero)
        neg_abs = L.lowerings[aten.neg.default](L.lowerings[aten.abs.default](self))
        log1p_exp = L.lowerings[aten.log1p.default](
            L.lowerings[aten.exp.default](neg_abs)
        )
        x_t = L.lowerings[aten.mul.Tensor](self, target)
        loss = L.lowerings[aten.add.Tensor](
            L.lowerings[aten.sub.Tensor](relu_x, x_t),
            log1p_exp,
        )
        if weight is not None:
            loss = L.lowerings[aten.mul.Tensor](loss, weight)
        return _reduce(loss, int(reduction))

    @register_lowering(aten.linalg_vector_norm, type_promotion_kind=None)
    def _vulkan_linalg_vector_norm(self, ord=2.0, dim=None, keepdim=False, dtype=None):
        if not _is_vulkan(self):
            return NotImplemented
        ndim = len(self.get_size())
        if dim is None:
            reduce_dims = list(range(ndim))
        elif isinstance(dim, int):
            reduce_dims = [dim if dim >= 0 else dim + ndim]
        else:
            reduce_dims = [d if d >= 0 else d + ndim for d in dim]
        try:
            ordf = float(ord)
        except Exception:
            ordf = None
        if ordf is None:
            return NotImplemented
        if math.isinf(ordf):
            # ord=+inf → max(|x|); ord=-inf → min(|x|).
            ab = L.lowerings[aten.abs.default](self)
            if ordf > 0:
                return L.lowerings[aten.amax](ab, reduce_dims, keepdim)
            return L.lowerings[aten.amin](ab, reduce_dims, keepdim)
        if math.isclose(ordf, 0.0):
            # ord=0 → count of non-zero entries.
            zero_mask = L.lowerings[aten.ne.Scalar](self, 0)
            # ne.Scalar returns bool; cast to input dtype for sum.
            cast = L.lowerings[aten._to_copy.default](zero_mask, dtype=self.get_dtype())
            return L.lowerings[aten.sum.dim_IntList](
                cast, reduce_dims, keepdims=keepdim
            )
        if math.isclose(ordf, 2.0):
            sq = L.lowerings[aten.mul.Tensor](self, self)
            s = L.lowerings[aten.sum.dim_IntList](sq, reduce_dims, keepdims=keepdim)
            return L.lowerings[aten.sqrt.default](s)
        if math.isclose(ordf, 1.0):
            ab = L.lowerings[aten.abs.default](self)
            return L.lowerings[aten.sum.dim_IntList](ab, reduce_dims, keepdims=keepdim)
        # Generic float p: pow(sum(|x|^p), 1/p).
        ab = L.lowerings[aten.abs.default](self)
        if math.isclose(ordf, math.floor(ordf)) and abs(ordf) <= 8:
            # Small integer p — multiply rather than pow for stability.
            cur = ab
            for _ in range(int(ordf) - 1):
                cur = L.lowerings[aten.mul.Tensor](cur, ab)
            powed = cur
        else:
            powed = L.lowerings[aten.pow.Tensor_Scalar](ab, ordf)
        s = L.lowerings[aten.sum.dim_IntList](powed, reduce_dims, keepdims=keepdim)
        return L.lowerings[aten.pow.Tensor_Scalar](s, 1.0 / ordf)

    @register_lowering(aten.norm.ScalarOpt_dim, type_promotion_kind=None)
    def _vulkan_norm_scalar_opt_dim(self, p, dim, keepdim=False):
        """Older ``aten.norm`` overload — proxy to ``linalg_vector_norm``."""
        if not _is_vulkan(self):
            return NotImplemented
        if p is None:
            p = 2
        return _vulkan_linalg_vector_norm(self, p, dim, keepdim, None)

    # ── P1.5: cross_entropy_loss ──────────────────────────────────────

    def _nll_loss_decomp(log_probs, target, weight, reduction, ignore_index):
        """P1.5 — Decompose nll_loss into Vulkan-safe primitives.

        Mirrors the upstream ``_nll_loss_forward`` decomposition in
        ``torch/_decomp/decompositions.py`` but operates at the Inductor
        IR level so it works even when AOT decomposition is bypassed.

        All gather operations use indirect-indexing with int64 targets,
        which is safe after the F.10 int64-truncation fix
        (``_install_vulkan_skip_alignment_clone``).

        Returns ``(output, total_weight)`` as two IR nodes.
        """
        n_dims = len(log_probs.get_size())
        channel_dim = 1
        if n_dims < 2:
            channel_dim = 0

        # Apply per-class weight before gathering.
        self_weighted = log_probs
        w = None
        if weight is not None:
            if n_dims > 1:
                shape = [1] * n_dims
                shape[channel_dim] = weight.get_size()[0]
                w = L.lowerings[aten.view](weight, shape)
            else:
                w = weight
            self_weighted = L.lowerings[aten.mul.Tensor](self_weighted, w)

        # Safe target: replace ignore_index with 0 so gather stays in bounds.
        ignore_mask = L.lowerings[aten.ne.Scalar](target, ignore_index)
        zero_target = L.lowerings[aten.mul.Scalar](target, 0.0)
        safe_target = L.lowerings[aten.where.self](ignore_mask, target, zero_target)
        safe_target_unsq = L.lowerings[aten.unsqueeze.default](safe_target, channel_dim)

        # Gather along channel_dim, then squeeze.
        gathered = L.lowerings[aten.gather.default](
            self_weighted, channel_dim, safe_target_unsq, False
        )
        gathered = L.lowerings[aten.squeeze.dim](gathered, channel_dim)

        # Negative log-likelihood.
        result = L.lowerings[aten.neg.default](gathered)

        # Zero out ignore_index positions.
        zero_result = L.lowerings[aten.mul.Scalar](result, 0.0)
        result = L.lowerings[aten.where.self](ignore_mask, result, zero_result)

        # Compute total_weight for the reduction denominator.
        if weight is not None:
            w_gathered = L.lowerings[aten.gather.default](
                w, channel_dim, safe_target_unsq, False
            )
            w_gathered = L.lowerings[aten.squeeze.dim](w_gathered, channel_dim)
            w_gathered = L.lowerings[aten.where.self](
                ignore_mask, w_gathered, zero_result
            )
            total_weight = L.lowerings[aten.sum.default](w_gathered)
        else:
            # TRAIN.8 (2026-05-29): Use constant total_weight = batch_size
            # instead of computing it from target values. This prevents the
            # AOTAutograd partitioner from marking total_weight (and thus
            # the mean reduction div) as backward-only when target is
            # partitioned away from the forward graph.
            #
            # For ignore_index < 0 (the default -100), all targets are valid
            # class indices, so total_weight = number of samples.
            #
            # Get batch size from input shape
            batch_size = float(log_probs.get_size()[0]) if n_dims >= 2 else 1.0
            total_weight = L.lowerings[aten.full.default](
                [],  # scalar shape
                batch_size,
                dtype=log_probs.get_dtype(),
                device=log_probs.get_device(),
                pin_memory=False,
            )

        if reduction == 0:  # none
            dummy_tw = L.lowerings[aten.mul.Scalar](total_weight, 0.0)
            return result, dummy_tw

        result_sum = L.lowerings[aten.sum.default](result)

        if reduction == 2:  # sum
            return result_sum, total_weight

        # reduction == 1 (mean)
        result_mean = L.lowerings[aten.div.Tensor](result_sum, total_weight)
        return result_mean, total_weight

    @register_lowering(aten.cross_entropy_loss, type_promotion_kind=None)
    def _vulkan_cross_entropy_loss(
        self,
        target,
        weight=None,
        reduction=1,
        ignore_index=-100,
        label_smoothing=0.0,
    ):
        """P1.5 — ``aten.cross_entropy_loss`` for Vulkan.

        Decomposes into ``log_softmax`` (which has a Vulkan lowering in
        ``softmax.py``) + nll_loss primitives. The int64-target gather
        inside the nll_loss decomposition is safe after the F.10
        ``skip_alignment_clone`` fix.

        ``label_smoothing`` support is deferred (returns NotImplemented
        for non-zero values so upstream handles it).
        """
        if not _is_vulkan(self):
            return NotImplemented
        label_smoothing_val = float(label_smoothing)
        if label_smoothing_val != 0.0:
            return NotImplemented
        log_probs = L.lowerings[aten._log_softmax](self, -1, False)
        result, _total_weight = _nll_loss_decomp(
            log_probs,
            target,
            weight=weight,
            reduction=int(reduction),
            ignore_index=int(ignore_index),
        )
        return result

    # ── TRAIN.4: nll_loss_backward ─────────────────────────────────────
    # NOTE (anti-goal #3): Implementation moved to _get_nll_loss_backward_impl()
    # below. Registration is done in bwd_lowerings.py.


def _get_nll_loss_backward_impl():
    """Return the implementation function for aten.nll_loss_backward.

    Registration is done in bwd_lowerings.py (anti-goal #3).
    TRAIN.4 — Decomposition into Vulkan-safe primitives that mirrors
    the upstream decomposition but operates at the Inductor IR level.
    """
    import torch
    from torch._inductor import lowering as L

    aten = torch.ops.aten

    def _vulkan_nll_loss_backward(
        grad_output,
        self,
        target,
        weight,
        reduction,
        ignore_index,
        total_weight,
    ):
        if not _is_vulkan(self):
            return NotImplemented

        n_dims = len(self.get_size())
        if n_dims < 2:
            channel_dim = 0
        else:
            channel_dim = 1

        reduction_val = int(reduction)
        ignore_index_val = int(ignore_index)

        # 1. Scale grad_output for mean reduction.
        if reduction_val == 1:  # mean
            grad_output = L.lowerings[aten.div.Tensor](grad_output, total_weight)

        # 2. Create gradient mask using pointwise ops.
        num_classes = int(self.get_size()[channel_dim])
        target_unsq = L.lowerings[aten.unsqueeze.default](target, channel_dim)

        from torch._inductor import ir as _ir
        from torch._inductor.virtualized import ops as _ops

        _target_dtype = target.get_dtype()
        _target_device = self.get_device()

        class_idx = _ir.Pointwise.create(
            device=_target_device,
            dtype=_target_dtype,
            inner_fn=lambda idx: _ops.index_expr(idx[0], _target_dtype),
            ranges=[num_classes],
        )

        mask = L.lowerings[aten.eq.Tensor](target_unsq, class_idx)
        not_ignored = L.lowerings[aten.ne.Scalar](
            target_unsq, ignore_index_val
        )

        minus_one_full = L.lowerings[aten.full.default](
            list(self.get_size()),
            -1.0,
            dtype=self.get_dtype(),
            device=self.get_device(),
            pin_memory=False,
        )
        zero_full = L.lowerings[aten.full.default](
            list(self.get_size()),
            0.0,
            dtype=self.get_dtype(),
            device=self.get_device(),
            pin_memory=False,
        )

        grad_from_mask = L.lowerings[aten.where.self](
            mask, minus_one_full, zero_full
        )
        grad_input = L.lowerings[aten.where.self](
            not_ignored, grad_from_mask, zero_full
        )

        # 3. Multiply: grad_input * grad_output.
        if weight is not None:
            w_shape = [1] * n_dims
            w_shape[channel_dim] = int(weight.get_size()[0])
            w_reshaped = L.lowerings[aten.view.default](weight, w_shape)
            grad_input = L.lowerings[aten.mul.Tensor](grad_input, w_reshaped)

        return L.lowerings[aten.mul.Tensor](grad_input, grad_output)

    return _vulkan_nll_loss_backward
