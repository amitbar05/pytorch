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

        # Keep symbolic — C, HxW may be sympy.Expr under dynamic-shape compile.
        # Sympy supports *, // natively; aten.view.default lowering accepts
        # sympy size expressions.
        group_size = C * HxW // num_groups
        x_reshaped = L.lowerings[aten.view.default](
            x, [N, num_groups, group_size]
        )

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


def _register_group_norm_fused() -> None:
    """GN.1 — Fused GroupNorm forward via standalone Slang shader (ExternKernelOut).

    Replaces the ~10 dispatch decomposition above with 1 fused dispatch
    using ``group_norm.slang``.  Gated behind ``TORCH_VULKAN_GN_FUSED_FWD=1``
    until GPU-verified on RDNA1.

    The fused shader computes per-(batch,group) mean/variance via shared-memory
    reduction, normalizes, and applies per-channel affine in a single
    kernel dispatch.  save_mean [N*G] and save_rstd [N*G] are written to
    pre-allocated buffers for the backward pass.
    """
    import os
    import torch
    from torch._inductor import ir
    from torch._inductor import lowering as L
    from torch._inductor.lowering import register_lowering, lowerings

    aten = torch.ops.aten

    @register_lowering(aten.native_group_norm, type_promotion_kind=None)
    def _vulkan_native_group_norm_fused(x, weight, bias, N, C, HxW, num_groups, eps):
        if not _is_vulkan(x):
            return NotImplemented

        N_val = int(N)
        C_val = int(C)
        G = int(num_groups)
        if N_val <= 0 or C_val <= 0 or G <= 0 or C_val % G != 0:
            return NotImplemented

        from torch_vulkan.inductor.lowerings.gn_forward_extern import (
            _VulkanGNFwdExternKernel,
        )

        dev = x.get_device()
        dtype = x.get_dtype()

        # Pre-allocate save_mean [N*G] and save_rstd [N*G]
        num_rows = N_val * G
        sm_size = [num_rows]
        sm_stride = [1]
        sm_layout = ir.FixedLayout(
            device=dev, dtype=dtype, size=sm_size, stride=sm_stride
        )
        save_mean_buf = lowerings[aten.empty.memory_format](
            sm_size, dtype=dtype, device=dev
        )
        save_rstd_buf = lowerings[aten.empty.memory_format](
            sm_size, dtype=dtype, device=dev
        )

        # GN forward ExternKernelOut — fused dispatch
        out_size = list(x.get_size())
        out_stride = [C_val * int(HxW), int(HxW), int(x.get_size()[-1]), 1]
        out_layout = ir.FixedLayout(
            device=dev, dtype=dtype, size=out_size, stride=out_stride
        )

        gn_inputs = [x, weight, bias, save_mean_buf, save_rstd_buf]
        gn_kernel = _VulkanGNFwdExternKernel(
            layout=out_layout,
            inputs=gn_inputs,
            num_groups=G,
            eps=float(eps),
        )

        # Reshape save_mean/save_rstd from [N*G] to [N, G] for backward
        sm_reshaped = L.lowerings[aten.view.default](
            save_mean_buf, [N_val, G]
        )
        sr_reshaped = L.lowerings[aten.view.default](
            save_rstd_buf, [N_val, G]
        )

        # The primary output is the ExternKernelOut wrapped in TensorBox
        out_ret = ir.TensorBox.create(gn_kernel)
        return [out_ret, sm_reshaped, sr_reshaped]


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
    import os
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
            # TRAIN.12 (2026-06-02): Simplified reduction path.  The original
            # M22.15 v3 workaround used 2-step reshape-reduce to avoid a buggy
            # combo-kernel merge (identical-shape reductions → Welford combo
            # missing out_ptr1 for variance).  With combo_kernels=False (TR.20,
            # set in __init__.py), the combo scheduler never creates these merges.
            # We can now use the simpler var_mean Welford reduction — 1 dispatch
            # instead of 4 (reshape + sum + sum + div → var_mean).
            #
            # Reshape to [N_eff, C] for contiguous reduction over dim 0.
            # Reducing over non-contiguous dims [0, 2, 3] causes Inductor
            # scheduler hangs.  N_eff = total_numel // C.
            inp_2d = L.lowerings[aten.view.default](inp, [N_eff, C])
            var_1d, mean_1d = L.lowerings[aten.var_mean.correction](
                inp_2d, [0], correction=0, keepdim=False,
            )
            mean_kd = L.lowerings[aten.view.default](mean_1d, bcast_shape)
            var_kd = L.lowerings[aten.view.default](var_1d, bcast_shape)
            dx = L.lowerings[aten.sub.Tensor](inp, mean_kd)
            var_eps = L.lowerings[aten.add.Scalar](var_kd, float(eps))
            rstd_kd = L.lowerings[aten.rsqrt.default](var_eps)
            rstd_1d = L.lowerings[aten.view.default](rstd_kd, [C])
            normalized = L.lowerings[aten.mul.Tensor](dx, rstd_kd)

            # BN.2 (2026-06-01): running_mean / running_var EMA update.
            # Was SKIPPED because copy_ on running_stats created InplacedBuffer
            # nodes that caused slangc "undefined identifier" errors. Root cause
            # was the missing spvGroupNonUniform capability in helpers.slang
            # (fixed RNN.1/2, same session). The InplacedBuffer handling in
            # dispatch_call.py/scheduling.py/header.py already deduplicates
            # aliased buffers — the slangc errors were cascading from the
            # broken helpers import.
            #
            # Gate behind TORCH_VULKAN_BN_EMA_UPDATE=1 until GPU-verified.
            # Without this flag, running_mean/running_var are NOT updated
            # during compiled training (backward still works — it uses
            # save_mean/save_invstd computed fresh per batch).
            if (
                running_mean is not None
                and running_var is not None
                and os.environ.get("TORCH_VULKAN_BN_EMA_UPDATE") == "1"
            ):
                decay = 1.0 - float(momentum)
                # running_mean update
                updated_rm = L.lowerings[aten.mul.Scalar](running_mean, decay)
                batch_rm_contrib = L.lowerings[aten.mul.Scalar](
                    mean_1d, float(momentum)
                )
                new_rm = L.lowerings[aten.add.Tensor](updated_rm, batch_rm_contrib)
                L.lowerings[aten.copy_.default](running_mean, new_rm)
                # running_var update (unbiased estimate)
                if N_eff > 1:
                    correction = float(N_eff) / float(N_eff - 1)
                    unbiased_var = L.lowerings[aten.mul.Scalar](var_1d, correction)
                else:
                    unbiased_var = var_1d
                updated_rv = L.lowerings[aten.mul.Scalar](running_var, decay)
                batch_rv_contrib = L.lowerings[aten.mul.Scalar](
                    unbiased_var, float(momentum)
                )
                new_rv = L.lowerings[aten.add.Tensor](updated_rv, batch_rv_contrib)
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
    # correctly.  EMA update gated behind TORCH_VULKAN_BN_EMA_UPDATE=1
    # (BN.2, 2026-06-01 — pending GPU verification after RNN capability fix).
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
            # in-place via copy_ (when TORCH_VULKAN_BN_EMA_UPDATE=1).
            # The functional variant must return the new running stats
            # as additional outputs.  Since the EMA update is done inline,
            # we return the (now-updated) running stats directly.
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
