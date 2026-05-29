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

        # GPU.1: AMD RDNA1 (NAVI10) L2 cache bug — after WaveActiveSum
        # (used by wg_welford), WGs 1-N cannot read the same SSBO cache line
        # as WG 0 in the SAME dispatch.  weight/bias share the same [C]
        # addresses across all batch WGs, triggering this race regardless of
        # dtype (confirmed empirically for fp16/bf16; also observed for fp32
        # with fresh cache compiles).
        # Confirmed via diag_l2_race.py: Case A (no reduction) always passes;
        # Case B (reduction then same-address read) always fails.
        # Fix: realize() forces the normalized intermediate to be materialized
        # in a concrete buffer, so the weight/bias application runs in a
        # SEPARATE kernel dispatch with no preceding WaveActiveSum.
        if weight is not None or bias is not None:
            out.realize()

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

        try:
            total_numel = 1
            for s in sizes:
                total_numel *= int(s)
            N_eff = total_numel // C
        except Exception:
            return NotImplemented
        if N_eff <= 0:
            return NotImplemented

        if bool(training):
            # M22.15 v3: 2-step reshape-reduce to avoid the combo-kernel
            # scheduler merging the mean-sum and var-sum into a buggy
            # Welford combo kernel.  The two sequential single-dim sums
            # previously had identical xnumel=C / r0_numel=N_eff, which
            # caused the scheduler to fuse them into a combo Welford kernel
            # whose generated struct was missing out_ptr1 (the variance
            # output), causing silent wrong values.
            #
            # Fix: reshape [N,C,H*] to [N,C,H_W] and use two reductions
            # at DIFFERENT scales so the combo scheduler never merges them:
            #   step1 → xnumel=N*C, r0_numel=H_W  (reduces spatial dims)
            #   step2 → xnumel=C,   r0_numel=N     (reduces batch dim)
            N = int(sizes[0])
            if ndim >= 3:
                H_W = N_eff // N
                inp_3d = L.lowerings[aten.view.default](inp, [N, C, H_W])
                s1 = L.lowerings[aten.sum.dim_IntList](inp_3d, [2], keepdims=False)
                sum_inp = L.lowerings[aten.sum.dim_IntList](s1, [0], keepdims=False)
            else:
                # ndim == 2: no spatial dims — single reduction over batch
                sum_inp = L.lowerings[aten.sum.dim_IntList](inp, [0], keepdims=False)
            # sum_inp is now shape [C]
            inv_N = 1.0 / float(N_eff)
            mean_1d = L.lowerings[aten.mul.Scalar](sum_inp, inv_N)
            mean_kd = L.lowerings[aten.view.default](mean_1d, bcast_shape)
            dx = L.lowerings[aten.sub.Tensor](inp, mean_kd)
            dx_sq = L.lowerings[aten.mul.Tensor](dx, dx)
            if ndim >= 3:
                dx_sq_3d = L.lowerings[aten.view.default](dx_sq, [N, C, H_W])
                sq1 = L.lowerings[aten.sum.dim_IntList](dx_sq_3d, [2], keepdims=False)
                sum_dx_sq = L.lowerings[aten.sum.dim_IntList](sq1, [0], keepdims=False)
            else:
                sum_dx_sq = L.lowerings[aten.sum.dim_IntList](dx_sq, [0], keepdims=False)
            # sum_dx_sq is now shape [C]
            var_1d = L.lowerings[aten.mul.Scalar](sum_dx_sq, inv_N)
            var_kd = L.lowerings[aten.view.default](var_1d, bcast_shape)
            var_eps = L.lowerings[aten.add.Scalar](var_kd, float(eps))
            rstd_kd = L.lowerings[aten.rsqrt.default](var_eps)
            rstd_1d = L.lowerings[aten.view.default](rstd_kd, [C])
            normalized = L.lowerings[aten.mul.Tensor](dx, rstd_kd)

            # MODEL.2: running_mean / running_var EMA update (training mode).
            # Compute new stats using pointwise ops that produce plain IR
            # buffers (no InplacedBuffer), then use copy_ to write back.
            # realize() on both new_* tensors before copy_ forces separate
            # dispatches so the copy_ mutations don't alias with the
            # normalization kernel's intermediates.
            if running_mean is not None and running_var is not None:
                one_minus_mom = 1.0 - float(momentum)
                mom = float(momentum)

                # new_running_mean = running_mean * (1 - momentum) + mean_1d * momentum
                rm_decay = L.lowerings[aten.mul.Scalar](running_mean, one_minus_mom)
                rm_new_contrib = L.lowerings[aten.mul.Scalar](mean_1d, mom)
                new_rm = L.lowerings[aten.add.Tensor](rm_decay, rm_new_contrib)

                # new_running_var = running_var * (1 - momentum) + unbiased_var * momentum
                # PyTorch uses unbiased variance (Bessel correction) for running_var
                if N_eff > 1:
                    unbiased_factor = float(N_eff) / float(N_eff - 1)
                    unbiased_var = L.lowerings[aten.mul.Scalar](var_1d, unbiased_factor)
                else:
                    unbiased_var = var_1d

                rv_decay = L.lowerings[aten.mul.Scalar](running_var, one_minus_mom)
                rv_new_contrib = L.lowerings[aten.mul.Scalar](unbiased_var, mom)
                new_rv = L.lowerings[aten.add.Tensor](rv_decay, rv_new_contrib)

                # Force materialization before copy_ to avoid InplacedBuffer
                # aliasing with the normalization kernel's Welford intermediates.
                new_rm.realize()
                new_rv.realize()

                # Write back via copy_ (in-place mutation on running stats)
                L.lowerings[aten.copy_.default](running_mean, new_rm)
                L.lowerings[aten.copy_.default](running_var, new_rv)
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

    # MODEL.2 (2026-05-29): Register _native_batch_norm_legit to delegate
    # to the same lowering as native_batch_norm. The "legit" variant has
    # proper mutation annotations so AOT autograd functionalizes it
    # correctly, but with the decomp suppressed our lowering handles the
    # running_mean/running_var copy_ mutations directly.
    if hasattr(aten, "_native_batch_norm_legit"):
        @register_lowering(aten._native_batch_norm_legit, type_promotion_kind=None)
        def _vulkan_batch_norm_legit(
            inp, weight, bias, running_mean, running_var,
            training, momentum, eps,
        ):
            if not _is_vulkan(inp):
                return NotImplemented
            return _vulkan_native_batch_norm(
                inp, weight, bias, running_mean, running_var,
                training, momentum, eps,
            )

    if hasattr(aten, "_native_batch_norm_legit_functional"):
        @register_lowering(
            aten._native_batch_norm_legit_functional,
            type_promotion_kind=None,
        )
        def _vulkan_batch_norm_legit_functional(
            inp, weight, bias, running_mean, running_var,
            training, momentum, eps,
        ):
            if not _is_vulkan(inp):
                return NotImplemented
            result = _vulkan_native_batch_norm(
                inp, weight, bias, running_mean, running_var,
                training, momentum, eps,
            )
            if result is NotImplemented:
                return NotImplemented
            # The non-functional variant mutates running_mean/running_var
            # in-place via copy_.  The functional variant must return
            # the new running stats as additional outputs.
            # Since _vulkan_native_batch_norm already did the copy_,
            # we can return the (now-updated) running stats directly.
            out, save_mean, save_rstd = result
            return [out, save_mean, save_rstd, running_mean, running_var]


# Backward lowerings for layer_norm, group_norm, and batch_norm have
# been consolidated into ``bwd_lowerings.py`` (TR.19).  The no-op
# stubs below preserve the import contract for callers that still
# reference them (M22.5 — duplicate definitions collapsed).


def _register_layer_norm_backward() -> None:
    pass  # TR.19 — moved to bwd_lowerings.py


def _register_group_norm_backward() -> None:
    pass  # TR.19 — moved to bwd_lowerings.py


def _register_batch_norm_backward() -> None:
    pass  # TR.19 — moved to bwd_lowerings.py
