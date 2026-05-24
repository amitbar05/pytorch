"""M22.6 — Norm backward lowerings split from bwd_lowerings.py.

Contains layer_norm, group_norm, and batch_norm backward.  Extracted to
keep bwd_lowerings.py under the 800-line anti-goal #7 cap.
"""

from __future__ import annotations


def _is_vulkan(x) -> bool:
    try:
        return x.get_device().type == "vulkan"
    except Exception:
        return False


def _register_layer_norm_backward() -> None:
    """P0.1 — Inductor lowering for ``aten.native_layer_norm_backward``.

    (Moved from ``lowerings/norm.py`` per TR.19.)
    """
    import torch
    from torch._inductor import lowering as L
    from torch._inductor.lowering import register_lowering

    aten = torch.ops.aten

    @register_lowering(aten.native_layer_norm_backward, type_promotion_kind=None)
    def _vulkan_native_layer_norm_backward(
        grad_out, inp, normalized_shape, mean, rstd, weight, bias, output_mask
    ):
        if not _is_vulkan(grad_out):
            return NotImplemented

        ndim = len(inp.get_size())
        norm_ndim = len(normalized_shape)
        axis = ndim - norm_ndim
        inner_dims = list(range(axis, ndim))
        outer_dims = list(range(axis))

        N = 1
        for s in normalized_shape:
            try:
                N *= int(s)
            except Exception:
                return NotImplemented
        if N <= 0:
            return NotImplemented
        inv_N = 1.0 / float(N)

        outer_shape = list(inp.get_size()[:axis])
        bcast_shape = outer_shape + [1] * norm_ndim
        mean_b = L.lowerings[aten.view.default](mean, bcast_shape)
        rstd_b = L.lowerings[aten.view.default](rstd, bcast_shape)

        dx = L.lowerings[aten.sub.Tensor](inp, mean_b)
        x_hat = L.lowerings[aten.mul.Tensor](dx, rstd_b)

        if weight is not None:
            grad_x_hat = L.lowerings[aten.mul.Tensor](grad_out, weight)
        else:
            grad_x_hat = grad_out

        c1 = L.lowerings[aten.mul.Tensor](grad_x_hat, x_hat)

        outputs: list = []

        if output_mask[0]:
            b = L.lowerings[aten.sum.dim_IntList](grad_x_hat, inner_dims, keepdims=True)
            c2 = L.lowerings[aten.sum.dim_IntList](c1, inner_dims, keepdims=True)
            a = L.lowerings[aten.mul.Scalar](grad_x_hat, float(N))
            c3 = L.lowerings[aten.mul.Tensor](x_hat, c2)
            inner = L.lowerings[aten.sub.Tensor](L.lowerings[aten.sub.Tensor](a, b), c3)
            scaled = L.lowerings[aten.mul.Tensor](rstd_b, inner)
            d_input = L.lowerings[aten.mul.Scalar](scaled, inv_N)
            outputs.append(d_input)
        else:
            outputs.append(None)

        if output_mask[1] and weight is not None:
            grad_w_elem = L.lowerings[aten.mul.Tensor](grad_out, x_hat)
            if outer_dims:
                d_weight = L.lowerings[aten.sum.dim_IntList](
                    grad_w_elem, outer_dims, keepdims=False
                )
            else:
                d_weight = grad_w_elem
            outputs.append(d_weight)
        else:
            outputs.append(None)

        if output_mask[2] and bias is not None:
            if outer_dims:
                d_bias = L.lowerings[aten.sum.dim_IntList](
                    grad_out, outer_dims, keepdims=False
                )
            else:
                d_bias = L.lowerings[aten.clone.default](grad_out)
            outputs.append(d_bias)
        else:
            outputs.append(None)

        return outputs


def _register_group_norm_backward() -> None:
    """P0.1 — Inductor lowering for ``aten.native_group_norm_backward``.

    (Moved from ``lowerings/norm.py`` per TR.19.)
    """
    import torch
    from torch._inductor import lowering as L
    from torch._inductor.lowering import register_lowering

    aten = torch.ops.aten

    @register_lowering(aten.native_group_norm_backward, type_promotion_kind=None)
    def _vulkan_native_group_norm_backward(
        grad_output, inp, mean, rstd, gamma, N, C, HxW, group, output_mask
    ):
        if not _is_vulkan(grad_output):
            return NotImplemented

        N = int(N)
        C = int(C)
        HxW = int(HxW)
        group = int(group)
        if N <= 0 or C <= 0 or HxW <= 0 or group <= 0 or C % group != 0:
            return NotImplemented
        cpg = C // group
        s = 1.0 / float(cpg * HxW)

        # Work in 4D [N, group, cpg, HxW] throughout so mean/rstd broadcast
        # cleanly over (cpg, HxW) and reductions land on the right dims.
        in_4d = L.lowerings[aten.view.default](inp, [N, group, cpg, HxW])
        go_4d = L.lowerings[aten.view.default](grad_output, [N, group, cpg, HxW])
        mean_4d = L.lowerings[aten.view.default](mean, [N, group, 1, 1])
        rstd_4d = L.lowerings[aten.view.default](rstd, [N, group, 1, 1])

        # xhat = (x - mean) * rstd  — the per-group normalised input
        xhat_4d = L.lowerings[aten.mul.Tensor](
            L.lowerings[aten.sub.Tensor](in_4d, mean_4d), rstd_4d
        )

        def _sum2(t, d0, d1):
            """Reduce t over two dims sequentially (d0 first, then d1)."""
            t = L.lowerings[aten.sum.dim_IntList](t, [d0], keepdims=False)
            # Removing dim d0 shifts all dims > d0 down by 1.
            adj = d1 - 1 if d1 > d0 else d1
            return L.lowerings[aten.sum.dim_IntList](t, [adj], keepdims=False)

        def _zero_like_shape(shape):
            return L.lowerings[aten.full.default](
                shape,
                0,
                dtype=grad_output.get_dtype(),
                device=grad_output.get_device(),
                pin_memory=False,
            )

        outputs: list = [
            _zero_like_shape(list(inp.get_size())),
            _zero_like_shape([C]),
            _zero_like_shape([C]),
        ]

        # Correct PyTorch GroupNorm backward (matches ATen group_norm.cpp):
        #   d_x_i = rstd * (gamma_i * dy_i
        #             - (1/n) * sum_j(gamma_j * dy_j)
        #             - xhat_i * (1/n) * sum_j(gamma_j * dy_j * xhat_j))
        # where n = cpg * HxW.  gamma must be INSIDE the group-level sums;
        # factoring it outside is only valid when cpg == 1.

        # go_gamma = gamma * dy  (or just dy when gamma is None)
        if gamma is not None:
            gamma_4d = L.lowerings[aten.view.default](gamma, [1, group, cpg, 1])
            go_gamma = L.lowerings[aten.mul.Tensor](go_4d, gamma_4d)
        else:
            go_gamma = go_4d

        # dy * xhat — needed for d_gamma (no gamma factor)
        dy_xhat = L.lowerings[aten.mul.Tensor](go_4d, xhat_4d)

        if output_mask[0]:
            # group sums with gamma inside
            go_gamma_xhat = L.lowerings[aten.mul.Tensor](go_gamma, xhat_4d)
            # ds_g[n,g] = sum_{cpg,HxW}(gamma * dy * xhat)
            ds_g = _sum2(go_gamma_xhat, 3, 2)  # [N, group]
            # db_g[n,g] = sum_{cpg,HxW}(gamma * dy)
            db_g = _sum2(go_gamma, 3, 2)  # [N, group]
            ds_g_4d = L.lowerings[aten.view.default](ds_g, [N, group, 1, 1])
            db_g_4d = L.lowerings[aten.view.default](db_g, [N, group, 1, 1])
            db_term = L.lowerings[aten.mul.Scalar](db_g_4d, s)
            ds_term = L.lowerings[aten.mul.Scalar](ds_g_4d, s)
            xhat_ds = L.lowerings[aten.mul.Tensor](xhat_4d, ds_term)
            correction = L.lowerings[aten.sub.Tensor](go_gamma, db_term)
            correction = L.lowerings[aten.sub.Tensor](correction, xhat_ds)
            d_input_4d = L.lowerings[aten.mul.Tensor](rstd_4d, correction)
            outputs[0] = L.lowerings[aten.view.default](
                d_input_4d, list(inp.get_size())
            )

        if output_mask[1]:
            # d_gamma_c = sum_{N,HxW}(dy_c * xhat_c)  — no gamma factor
            tmp = _sum2(dy_xhat, 3, 0)  # sum HxW then N: [group, cpg]
            outputs[1] = L.lowerings[aten.view.default](tmp, [C])

        if output_mask[2]:
            # d_bias_c = sum_{N,HxW}(dy_c)  — no gamma factor
            tmp = _sum2(go_4d, 3, 0)  # sum HxW then N: [group, cpg]
            outputs[2] = L.lowerings[aten.view.default](tmp, [C])

        return outputs


def _register_batch_norm_backward() -> None:
    """PF.24 — Inductor lowering for ``aten.native_batch_norm_backward``.

    (Moved from ``lowerings/norm.py`` per TR.19.)
    """
    import torch
    from torch._inductor import lowering as L
    from torch._inductor.lowering import register_lowering

    aten = torch.ops.aten

    @register_lowering(aten.native_batch_norm_backward, type_promotion_kind=None)
    def _vulkan_native_batch_norm_backward(
        grad_out,
        inp,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        train,
        eps,
        output_mask,
    ):
        if not _is_vulkan(grad_out):
            return NotImplemented

        sizes = list(inp.get_size())
        ndim = len(sizes)
        if ndim < 2:
            return NotImplemented

        try:
            C = int(sizes[1])
            total_numel = 1
            for s in sizes:
                total_numel *= int(s)
        except Exception:
            return NotImplemented
        if C <= 0 or total_numel <= 0 or total_numel % C != 0:
            return NotImplemented
        N_eff = total_numel // C
        if N_eff <= 0:
            return NotImplemented

        reduce_dims = [d for d in range(ndim) if d != 1]
        bcast_shape = [1] * ndim
        bcast_shape[1] = C

        if bool(train):
            invstd_1d = save_invstd
            mean_b = L.lowerings[aten.view.default](save_mean, bcast_shape)
            invstd_b = L.lowerings[aten.view.default](save_invstd, bcast_shape)
        else:
            var_eps = L.lowerings[aten.add.Tensor](
                running_var,
                L.lowerings[aten.full.default](
                    [C],
                    float(eps),
                    dtype=running_var.get_dtype(),
                    device=running_var.get_device(),
                    pin_memory=False,
                ),
            )
            invstd_1d = L.lowerings[aten.rsqrt.default](var_eps)
            mean_b = L.lowerings[aten.view.default](running_mean, bcast_shape)
            invstd_b = L.lowerings[aten.view.default](invstd_1d, bcast_shape)

        x_centered = L.lowerings[aten.sub.Tensor](inp, mean_b)

        # M22.15 v3: 2-step reshape-reduce — same fix as BN forward.
        # Reshape [N,C,H*] to [N,C,H_W] then:
        #   step1 sum(dim=2) → [N,C]   (xnumel=N*C, r0_numel=H_W)
        #   step2 sum(dim=0) → [C]     (xnumel=C,   r0_numel=N)
        # The two steps have DIFFERENT xnumel/r0_numel so the combo-kernel
        # scheduler never merges dY_sum and dY_xc into a buggy Welford combo.
        dY_xc_elem = L.lowerings[aten.mul.Tensor](grad_out, x_centered)
        N = int(sizes[0])
        if ndim >= 3:
            H_W = N_eff // N
            go_3d = L.lowerings[aten.view.default](grad_out, [N, C, H_W])
            xc_3d = L.lowerings[aten.view.default](dY_xc_elem, [N, C, H_W])
            dY_sum_nc = L.lowerings[aten.sum.dim_IntList](go_3d, [2], keepdims=False)
            dY_xc_nc = L.lowerings[aten.sum.dim_IntList](xc_3d, [2], keepdims=False)
            dY_sum = L.lowerings[aten.sum.dim_IntList](dY_sum_nc, [0], keepdims=False)
            dY_xc = L.lowerings[aten.sum.dim_IntList](dY_xc_nc, [0], keepdims=False)
        else:
            dY_sum = L.lowerings[aten.sum.dim_IntList](grad_out, [0], keepdims=False)
            dY_xc = L.lowerings[aten.sum.dim_IntList](dY_xc_elem, [0], keepdims=False)
        # dY_sum and dY_xc are now shape [C]
        dY_sum_b = L.lowerings[aten.view.default](dY_sum, bcast_shape)
        dY_xc_sum_b = L.lowerings[aten.view.default](dY_xc, bcast_shape)

        outputs: list = [None, None, None]

        if output_mask[0]:
            if bool(train):
                inv_N = 1.0 / float(N_eff)
                dY_sum_scaled = L.lowerings[aten.mul.Scalar](dY_sum_b, inv_N)
                a = L.lowerings[aten.sub.Tensor](grad_out, dY_sum_scaled)
                invstd_sq = L.lowerings[aten.mul.Tensor](invstd_b, invstd_b)
                dY_xc_scaled = L.lowerings[aten.mul.Scalar](dY_xc_sum_b, inv_N)
                k = L.lowerings[aten.mul.Tensor](invstd_sq, dY_xc_scaled)
                b = L.lowerings[aten.mul.Tensor](x_centered, k)
                inner = L.lowerings[aten.sub.Tensor](a, b)
                scaled = L.lowerings[aten.mul.Tensor](inner, invstd_b)
                if weight is not None:
                    weight_b = L.lowerings[aten.view.default](weight, bcast_shape)
                    d_input = L.lowerings[aten.mul.Tensor](scaled, weight_b)
                else:
                    d_input = scaled
            else:
                if weight is not None:
                    weight_b = L.lowerings[aten.view.default](weight, bcast_shape)
                    k = L.lowerings[aten.mul.Tensor](invstd_b, weight_b)
                else:
                    k = invstd_b
                d_input = L.lowerings[aten.mul.Tensor](grad_out, k)
            outputs[0] = d_input

        if output_mask[1]:
            outputs[1] = L.lowerings[aten.mul.Tensor](dY_xc, invstd_1d)

        if output_mask[2]:
            outputs[2] = dY_sum

        return outputs


def register_norm_backward_lowerings() -> None:
    """Register all norm backward lowerings.  Called from bwd_lowerings.register()."""
    _register_layer_norm_backward()
    _register_group_norm_backward()
    _register_batch_norm_backward()
