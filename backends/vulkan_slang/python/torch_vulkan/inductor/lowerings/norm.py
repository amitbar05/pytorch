"""Layer norm, group norm, batch norm forward lowerings.

Backward lowerings for layer_norm, group_norm, and batch_norm have been
moved to ``bwd_lowerings.py`` (TR.19 backward consolidation).
"""

from __future__ import annotations

from . import _is_vulkan


def _register_layer_norm() -> None:
    import torch
    from torch._inductor import lowering as L
    from torch._inductor.lowering import register_lowering

    aten = torch.ops.aten

    @register_lowering(aten.native_layer_norm, type_promotion_kind=None)
    def _vulkan_native_layer_norm(x, normalized_shape, weight, bias, eps):
        if not _is_vulkan(x):
            # No original lowering registered — fall back to ExternKernel path
            return NotImplemented

        ndim = len(x.get_size())
        norm_ndim = len(normalized_shape)
        reduce_dims = list(range(ndim - norm_ndim, ndim))

        # DR.1+: Use var_mean welford reduction (single dispatch) instead
        # of two separate mean reductions (mean + var).  Saves 1 dispatch
        # and eliminates the intermediate dx*dx buffer.
        var_kd, mean_kd = L.lowerings[aten.var_mean.correction](
            x, reduce_dims, correction=0, keepdim=True
        )
        # dx = x - mean
        dx = L.lowerings[aten.sub.Tensor](x, mean_kd)
        # rstd = rsqrt(var + eps)
        var_eps = L.lowerings[aten.add.Scalar](var_kd, eps)
        rstd_kd = L.lowerings[aten.rsqrt.default](var_eps)
        # normalize
        out = L.lowerings[aten.mul.Tensor](dx, rstd_kd)
        if weight is not None:
            out = L.lowerings[aten.mul.Tensor](out, weight)
        if bias is not None:
            out = L.lowerings[aten.add.Tensor](out, bias)

        # ``aten.native_layer_norm`` returns ``(output, mean, rstd)`` where
        # mean/rstd keep the reduction axes with size 1 (not collapsed). For
        # input [N, *, normalized_shape] with normalized_shape having
        # ``norm_ndim`` axes, mean/rstd shape is [N, *, 1, 1, ...] —
        # ``batch_shape`` for the leading dims plus ``[1] * norm_ndim``.
        # Eager `at::native_layer_norm` returns shape [..., 1] (verified on
        # 2.11.0+cu130). var_mean(keepdim=True) already produces this shape;
        # passing it through directly preserves rank for downstream
        # ``require_stride_order`` checks (Inductor's
        # ``stride_ordered_for_memory_format`` asserts rank match against
        # the FX node's ``meta['val'].stride()``).
        return [out, mean_kd, rstd_kd]


def _register_group_norm() -> None:
    import torch
    from torch._inductor import lowering as L
    from torch._inductor.lowering import register_lowering

    aten = torch.ops.aten

    @register_lowering(aten.native_group_norm, type_promotion_kind=None)
    def _vulkan_native_group_norm(x, weight, bias, N, C, HxW, num_groups, eps):
        if not _is_vulkan(x):
            return NotImplemented

        group_size = int(C * HxW) // int(num_groups)
        x_reshaped = L.lowerings[aten.view.default](x, [N, num_groups, group_size])

        # DR.1+: Use var_mean welford reduction (single reduction dispatch)
        # instead of two separate mean reductions (mean + var).  The welford
        # path computes both statistics in one pass via TUPLE_REDUCTION,
        # saving 1 dispatch and the intermediate (dx, dx*dx) buffers.
        reduce_dims = [2]
        var_kd, mean_kd = L.lowerings[aten.var_mean.correction](
            x_reshaped, reduce_dims, correction=0, keepdim=True
        )
        dx = L.lowerings[aten.sub.Tensor](x_reshaped, mean_kd)
        var_eps = L.lowerings[aten.add.Scalar](var_kd, eps)
        rstd_kd = L.lowerings[aten.rsqrt.default](var_eps)
        out = L.lowerings[aten.mul.Tensor](dx, rstd_kd)

        # Reshape back to original before applying affine transform
        out = L.lowerings[aten.view.default](out, list(x.get_size()))

        if weight is not None or bias is not None:
            # weight/bias shape [C]; reshape to [1, C, 1, ...] for broadcast
            orig_sizes = list(x.get_size())
            spatial_rank = len(orig_sizes) - 2
            affine_shape = [1, C] + [1] * spatial_rank
            if weight is not None:
                w_broad = L.lowerings[aten.view.default](weight, affine_shape)
                out = L.lowerings[aten.mul.Tensor](out, w_broad)
            if bias is not None:
                b_broad = L.lowerings[aten.view.default](bias, affine_shape)
                out = L.lowerings[aten.add.Tensor](out, b_broad)

        mean_out = L.lowerings[aten.view.default](mean_kd, [N, num_groups])
        rstd_out = L.lowerings[aten.view.default](rstd_kd, [N, num_groups])
        return [out, mean_out, rstd_out]


def _register_batch_norm_forward() -> None:
    """B1 — Inductor lowering for ``aten.native_batch_norm.default`` (forward).

    Without this, compiled models using ``nn.BatchNorm*d`` in training mode
    extern-fall on the forward path (even though the backward IS lowered via
    PF.24). Decomposes into mean/var/normalize/scale+bias so the scheduler
    fuses the entire chain into a single VulkanKernel.

    Schema:
      ``native_batch_norm(input, weight, bias, running_mean, running_var,
        training, momentum, eps) -> (output, save_mean, save_invstd)``

    In training mode, computes per-channel mean/invstd from the input batch
    **and** updates ``running_mean`` / ``running_var`` in-place via
    momentum-driven exponential moving average (matching eager semantics).

    In eval mode, uses running_mean/running_var and returns zero-element
    save_mean/save_invstd (matching the eager C++ convention).

    BN.1 (2026-05-09): Added running_stat copy_ mutations so compiled
    training steps update ``running_mean`` / ``running_var``, matching the
    eager ``aten.native_batch_norm`` side-effect.
    """
    import torch
    from torch._inductor import lowering as L
    from torch._inductor.lowering import register_lowering

    aten = torch.ops.aten

    @register_lowering(aten.native_batch_norm, type_promotion_kind=None)
    def _vulkan_native_batch_norm(
        inp, weight, bias, running_mean, running_var, training, momentum, eps
    ):
        if not _is_vulkan(inp):
            return NotImplemented

        sizes = list(inp.get_size())
        ndim = len(sizes)
        if ndim < 2:
            return NotImplemented

        try:
            C = int(sizes[1])
        except Exception:
            return NotImplemented

        reduce_dims = [d for d in range(ndim) if d != 1]
        bcast_shape = [1] * ndim
        bcast_shape[1] = C

        if bool(training):
            # DR.1+: Use var_mean welford reduction (single dispatch) instead
            # of two separate mean reductions (mean + var).  Saves 1 dispatch
            # and eliminates the intermediate dx*dx buffer.
            var_kd, mean_kd = L.lowerings[aten.var_mean.correction](
                inp, reduce_dims, correction=0, keepdim=True
            )
            mean_1d = L.lowerings[aten.view.default](mean_kd, [C])
            dx = L.lowerings[aten.sub.Tensor](inp, mean_kd)
            var_1d = L.lowerings[aten.view.default](var_kd, [C])
            var_eps = L.lowerings[aten.add.Scalar](var_kd, float(eps))
            rstd_kd = L.lowerings[aten.rsqrt.default](var_eps)
            rstd_1d = L.lowerings[aten.view.default](rstd_kd, [C])
            normalized = L.lowerings[aten.mul.Tensor](dx, rstd_kd)

            # BN.1 — update running_mean / running_var in-place (eager parity).
            # The eager aten::native_batch_norm mutates these buffers via
            # momentum-driven exponential moving average even though the
            # schema does not formally annotate them as mutable.  The
            # lowering decomposes the forward computation into pointwise +
            # reduction primitives, so we must explicitly emit the copy_
            # mutations here; otherwise ``nn.BatchNorm2d`` buffers stay at
            # their initial values (0 / 1) forever, causing multi-step
            # training loss drift.
            #
            # Formula (matches eager C++ Normalization.cpp):
            #   running_mean = (1−m) · running_mean + m · batch_mean
            #   running_var  = (1−m) · running_var  + m · unbiased_batch_var
            #   unbiased_batch_var = biased_var · N / (N−1)
            if running_mean is not None and running_var is not None:
                total_numel = 1
                for s in sizes:
                    total_numel *= int(s)
                N_eff = total_numel // C
                mom = float(momentum)
                one_minus_mom = 1.0 - mom

                # updated_rm = (1-m) * running_mean + m * mean_1d
                updated_rm = L.lowerings[aten.add.Tensor](
                    L.lowerings[aten.mul.Scalar](running_mean, one_minus_mom),
                    L.lowerings[aten.mul.Scalar](mean_1d, mom),
                )
                L.lowerings[aten.copy_](running_mean, updated_rm)

                # unbiased_var = var_1d * N / (N-1)   (N_eff ≥ 2 for BN)
                if N_eff > 1:
                    corr = float(N_eff) / float(N_eff - 1)
                    unbiased_var_1d = L.lowerings[aten.mul.Scalar](var_1d, corr)
                else:
                    unbiased_var_1d = var_1d

                # updated_rv = (1-m) * running_var + m * unbiased_var_1d
                updated_rv = L.lowerings[aten.add.Tensor](
                    L.lowerings[aten.mul.Scalar](running_var, one_minus_mom),
                    L.lowerings[aten.mul.Scalar](unbiased_var_1d, mom),
                )
                L.lowerings[aten.copy_](running_var, updated_rv)
        else:
            if running_mean is None or running_var is None:
                return NotImplemented
            mean_b = L.lowerings[aten.view.default](running_mean, bcast_shape)
            var_eps = L.lowerings[aten.add.Scalar](
                L.lowerings[aten.view.default](running_var, bcast_shape),
                float(eps),
            )
            rstd_kd = L.lowerings[aten.rsqrt.default](var_eps)
            dx = L.lowerings[aten.sub.Tensor](inp, mean_b)
            normalized = L.lowerings[aten.mul.Tensor](dx, rstd_kd)
            mean_1d = L.lowerings[aten.full.default](
                [C],
                0.0,
                dtype=inp.get_dtype(),
                device=inp.get_device(),
                pin_memory=False,
            )
            rstd_1d = L.lowerings[aten.full.default](
                [C],
                0.0,
                dtype=inp.get_dtype(),
                device=inp.get_device(),
                pin_memory=False,
            )

        out = normalized
        if weight is not None:
            weight_b = L.lowerings[aten.view.default](weight, bcast_shape)
            out = L.lowerings[aten.mul.Tensor](out, weight_b)
        if bias is not None:
            bias_b = L.lowerings[aten.view.default](bias, bcast_shape)
            out = L.lowerings[aten.add.Tensor](out, bias_b)

        return [out, mean_1d, rstd_1d]


# Backward lowerings for layer_norm, group_norm, and batch_norm have
# been consolidated into ``bwd_lowerings.py`` (TR.19).  The no-op
# stubs below preserve the import contract for callers that still
# reference them.


def _register_layer_norm_backward() -> None:
    pass  # TR.19 — moved to bwd_lowerings.py


def _register_group_norm_backward() -> None:
    pass  # TR.19 — moved to bwd_lowerings.py


def _register_batch_norm_backward() -> None:
    pass  # TR.19 — moved to bwd_lowerings.py


def _register_batch_norm_backward() -> None:
    """Register batch_norm backward lowerings (stub — TR.19).

    The full batch_norm backward lowering has been moved to
    ``bwd_lowerings.py`` (TR.19 backward consolidation).  This stub
    keeps the import contract in ``lowerings/__init__.py`` working
    until the consolidation is complete.
    """


def _register_group_norm_backward() -> None:
    """Register group_norm backward lowerings (stub — TR.19).

    The full group_norm backward lowering has been moved to
    ``bwd_lowerings.py`` (TR.19 backward consolidation).  This stub
    keeps the import contract in ``lowerings/__init__.py`` working
    until the consolidation is complete.
    """
