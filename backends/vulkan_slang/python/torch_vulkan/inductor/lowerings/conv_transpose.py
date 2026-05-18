"""Transposed conv (1D / 2D / 3D) lowerings via decomposition (M6 Phase 4).

conv_transpose is the backward of regular conv w.r.t. its input. We
decompose into: flip kernel spatially → swap in/out channels → upsample
input (for stride>1) → regular conv with adjusted padding → output pad.

Supported envelope:
  * 2D — arbitrary stride / padding / dilation / groups / output_padding
  * 1D — reshape to 2D and reuse the 2D path
  * 3D — KD==1 path (per-frame, depth preserved); KD>1 returns
    NotImplemented (depth coupling can't be reduced to flat Conv2d).
"""

from __future__ import annotations


def _register_conv_transpose_lowerings() -> None:
    """Register conv_transpose1d/2d/3d lowerings on top of the Conv2d
    custom op (``torch_vulkan::conv2d_with_optional_bias``).

    Called from ``_register_conv_and_pool_lowerings`` so the Conv2d custom
    op is already resolved and registered.
    """
    import torch
    from torch._inductor.lowering import lowerings, make_fallback, register_lowering

    aten = torch.ops.aten

    # ``aten.flip`` is only in the decomposition table upstream — not in
    # the lowering registry — so a recursive ``_lowerings[aten.flip.default]``
    # lookup KeyErrors. Register a fallback (overriding the decomp) so
    # the conv_transpose decomposition can call into it. The fallback
    # routes to the eager Vulkan ``flip`` op at runtime.
    if aten.flip.default not in lowerings:
        make_fallback(aten.flip, override_decomp=True, warn=False)

    # M-pipeline-2: the OpOverload-identity-safe helper now lives in
    # ``_conv_common.py``. The previous inline definition (M19.5-followup-1
    # vintage) was duplicated between this module and ``conv.py``; the
    # extraction keeps both call sites in lockstep.
    from ._conv_common import _get_conv2d_lowering_by_name as _get_conv2d_lowering  # noqa: F401

    def _impl_2d(
        input, weight, bias, stride, padding, output_padding, groups, dilation
    ):
        from torch._inductor.lowering import lowerings as _lowerings
        from torch_vulkan.inductor.kernel.symbolic import get_static_numel

        if len(input.get_size()) != 4 or len(weight.get_size()) != 4:
            return NotImplemented

        sH = int(stride[0])
        sW = int(stride[-1] if len(stride) > 1 else stride[0])
        pH = int(padding[0])
        pW = int(padding[-1] if len(padding) > 1 else padding[0])
        oH = int(output_padding[0])
        oW = int(output_padding[-1] if len(output_padding) > 1 else output_padding[0])
        dH = int(dilation[0])
        dW = int(dilation[-1] if len(dilation) > 1 else dilation[0])
        g = int(groups)

        # M19.5 — keep weight + input dims as sympy when dynamic. The
        # weight's spatial kernel size (kH, kW) is always concrete from
        # the module; the channel counts (C_in, C_out_per_g) are
        # almost always concrete too. The input batch / spatial dims
        # may be SymInt under dynamic-shape compile.
        kH = int(weight.get_size()[2])
        kW = int(weight.get_size()[3])
        H_in = input.get_size()[2]
        W_in = input.get_size()[3]
        N = input.get_size()[0]
        C_in = input.get_size()[1]
        C_out_per_g = int(weight.get_size()[1])
        C_out = C_out_per_g * g

        # Per-group decomposition: slice channels, recurse with groups=1,
        # then concat along channel axis. Bias is sliced per group.
        if g != 1:
            C_in_static = get_static_numel(C_in)
            if C_in_static is None:
                return NotImplemented
            if C_in_static % g != 0 or C_out % g != 0:
                return NotImplemented
            C_in_per_g = C_in_static // g
            outs = []
            for i in range(g):
                inp_g = _lowerings[aten.slice.Tensor](
                    input, 1, i * C_in_per_g, (i + 1) * C_in_per_g, 1
                )
                w_g = _lowerings[aten.slice.Tensor](
                    weight, 0, i * C_in_per_g, (i + 1) * C_in_per_g, 1
                )
                bias_g = None
                if bias is not None:
                    bias_g = _lowerings[aten.slice.Tensor](
                        bias, 0, i * C_out_per_g, (i + 1) * C_out_per_g, 1
                    )
                out_g = _impl_2d(
                    inp_g, w_g, bias_g,
                    [sH, sW], [pH, pW], [oH, oW], 1, [dH, dW],
                )
                if out_g is NotImplemented:
                    return NotImplemented
                outs.append(out_g)
            return _lowerings[aten.cat.default](outs, 1)

        # Step 1: Flip weight spatially (reverse kernel dims). Uses the
        # fallback registered above (eager Vulkan flip op).
        weight_flipped = _lowerings[aten.flip.default](weight, [2, 3])
        # Step 2: Swap in/out channels: [C_in, C_out, kH, kW] →
        #         [C_out, C_in, kH, kW]. groups==1 here. Use permute
        #         instead of transpose.int since transpose has no lowering.
        weight_t = _lowerings[aten.permute.default](weight_flipped, [1, 0, 2, 3])
        weight_t = _lowerings[aten.clone.default](weight_t)
        # The conv2d ExternKernelOut requires realized inputs (FixedLayout
        # buffers). Both clone and permute return Pointwise/BaseView IR
        # which fails ``isinstance(self.data, BaseView)`` checks downstream.
        # Realize explicitly via ``ExternKernel.realize_input``.
        from torch._inductor import ir as _ir
        weight_t = _ir.ExternKernel.realize_input(weight_t)

        # Adjusted padding for the equivalent regular conv. With dilation,
        # the effective kernel extent is dilation*(K-1)+1, so adj_p =
        # dilation*(K-1) - p (reduces to k-1-p when dilation==1).
        adj_pH = dH * (kH - 1) - pH
        adj_pW = dW * (kW - 1) - pW
        if adj_pH < 0 or adj_pW < 0:
            # Negative adjusted padding would require cropping the input —
            # not supported by the conv2d template. Fall through to extern.
            return NotImplemented

        # Step 3: Upsample input if stride > 1 by inserting zeros between
        # spatial elements, then dispatch to regular conv2d with stride=1.
        if sH > 1 or sW > 1:
            input_up = _lowerings[aten.unsqueeze.default](input, 3)
            input_up = _lowerings[aten.unsqueeze.default](input_up, -1)
            input_up = _lowerings[aten.constant_pad_nd.default](
                input_up, [0, sW - 1, 0, 0, 0, sH - 1, 0, 0], 0.0
            )
            input_up = _lowerings[aten.reshape.default](
                input_up, [N, C_in, H_in * sH, W_in * sW]
            )
            H_up = (H_in - 1) * sH + 1
            W_up = (W_in - 1) * sW + 1
            if H_up < H_in * sH or W_up < W_in * sW:
                input_up = _lowerings[aten.slice.default](input_up, 2, 0, H_up)
                input_up = _lowerings[aten.slice.default](input_up, 3, 0, W_up)
            input_for_conv = input_up
        else:
            input_for_conv = input

        conv2d_lower = _get_conv2d_lowering()
        if conv2d_lower is None:
            # conv2d custom op lowering not registered yet — fall through
            # to the upstream extern path (will likely fail with `out=`,
            # but that's a separate bug).
            return NotImplemented
        result = conv2d_lower(
            input_for_conv, weight_t, bias, [1, 1], [adj_pH, adj_pW], [dH, dW], 1
        )
        if result is NotImplemented:
            return NotImplemented

        # output_padding adds zeros to the right/bottom of the output (one
        # side only, by PyTorch convention).
        if oH != 0 or oW != 0:
            result = _lowerings[aten.constant_pad_nd.default](
                result, [0, oW, 0, oH], 0.0
            )
        return result

    def _impl_1d(input, weight, bias, stride, padding, output_padding, groups, dilation):
        """conv_transpose1d via reshape to 2D impl."""
        from torch._inductor.lowering import lowerings as _lowerings

        if len(input.get_size()) != 3 or len(weight.get_size()) != 3:
            return NotImplemented

        s = int(stride[0]) if hasattr(stride, "__len__") else int(stride)
        p = int(padding[0]) if hasattr(padding, "__len__") else int(padding)
        o = (
            int(output_padding[0])
            if hasattr(output_padding, "__len__")
            else int(output_padding)
        )
        d = int(dilation[0]) if hasattr(dilation, "__len__") else int(dilation)

        # Reshape to 4-D ([..., L] → [..., L, 1]) and delegate to the 2D
        # impl with passthrough W (stride=1, pad=0, dilation=1).
        input_4d = _lowerings[aten.unsqueeze.default](input, -1)
        weight_4d = _lowerings[aten.unsqueeze.default](weight, -1)

        result_4d = _impl_2d(
            input_4d, weight_4d, bias,
            [s, 1], [p, 0], [o, 0], int(groups), [d, 1],
        )
        if result_4d is NotImplemented:
            return NotImplemented
        result_4d = _lowerings[aten.clone.default](result_4d)
        return _lowerings[aten.squeeze.default](result_4d, -1)

    def _impl_3d(input, weight, bias, stride, padding, output_padding, groups, dilation):
        """conv_transpose3d via per-frame Conv2d (KD==1 only)."""
        from torch._inductor.lowering import lowerings as _lowerings

        if len(input.get_size()) != 5 or len(weight.get_size()) != 5:
            return NotImplemented

        def _triple(v):
            if hasattr(v, "__len__"):
                if len(v) == 3:
                    return [int(v[0]), int(v[1]), int(v[2])]
                if len(v) == 1:
                    return [int(v[0]), int(v[0]), int(v[0])]
                return None
            return [int(v), int(v), int(v)]

        st = _triple(stride)
        pd = _triple(padding)
        op = _triple(output_padding)
        dl = _triple(dilation)
        if st is None or pd is None or op is None or dl is None:
            return NotImplemented
        sD, sH, sW = st
        pD, pH, pW = pd
        oD, oH, oW = op
        dD, dH, dW = dl

        kD = int(weight.get_size()[2])
        # KD>1 / sD>1 / pD>0 / oD>0 / dD>1: depth coupling can't reduce to
        # flat per-frame Conv2d. Defer to extern.
        if kD != 1 or sD != 1 or pD != 0 or oD != 0 or dD != 1:
            return NotImplemented

        # M19.5 — input dims may be SymInt under dynamic-shape compile.
        # Sympy expressions flow through reshape/squeeze. The
        # ``C_in != C_in_w`` channel-count check still requires concrete
        # ints to compare safely against the weight's static channel.
        from torch_vulkan.inductor.kernel.symbolic import get_static_numel

        t1_sizes = input.get_size()
        w_sizes = weight.get_size()
        N = t1_sizes[0]
        C_in = t1_sizes[1]
        D = t1_sizes[2]
        H_in = t1_sizes[3]
        W_in = t1_sizes[4]
        C_in_w = int(w_sizes[0])
        C_out_per_g = int(w_sizes[1])
        C_in_static = get_static_numel(C_in)
        if C_in_static is not None and C_in_static != C_in_w:
            return NotImplemented
        g = int(groups)
        C_out = C_out_per_g * g

        # Per-frame: reshape [N,C,D,H,W] → [N*D,C,H,W], transpose-conv2d,
        # reshape back to [N,C_out,D,H_out,W_out].
        input_4d = _lowerings[aten.reshape.default](
            input, [N * D, C_in, H_in, W_in]
        )
        # Drop the singleton kD dim: [C_in, C_out, 1, KH, KW] →
        # [C_in, C_out, KH, KW].
        weight_4d = _lowerings[aten.squeeze.default](weight, 2)
        result_4d = _impl_2d(
            input_4d, weight_4d, bias,
            [sH, sW], [pH, pW], [oH, oW], g, [dH, dW],
        )
        if result_4d is NotImplemented:
            return NotImplemented
        # H_out / W_out flow into the reshape size list as sympy.
        H_out = result_4d.get_size()[2]
        W_out = result_4d.get_size()[3]
        result_4d = _lowerings[aten.clone.default](result_4d)
        return _lowerings[aten.reshape.default](
            result_4d, [N, C_out, D, H_out, W_out]
        )

    # Per-overload lowerings — fire when AOTAutograd preserves the
    # high-level conv_transposeNd op (rare; usually decomposed to
    # ``aten.convolution`` first — see ``_vulkan_convolution`` below).
    @register_lowering(aten.conv_transpose2d.input, type_promotion_kind=None)
    def _vulkan_conv_transpose2d(
        input, weight, bias=None,
        stride=1, padding=0, output_padding=0, groups=1, dilation=1,
    ):
        return _impl_2d(
            input, weight, bias, stride, padding, output_padding, groups, dilation,
        )

    @register_lowering(aten.conv_transpose1d.default, type_promotion_kind=None)
    def _vulkan_conv_transpose1d(
        input, weight, bias=None,
        stride=1, padding=0, output_padding=0, groups=1, dilation=1,
    ):
        return _impl_1d(
            input, weight, bias, stride, padding, output_padding, groups, dilation,
        )

    @register_lowering(aten.conv_transpose3d.input, type_promotion_kind=None)
    def _vulkan_conv_transpose3d(
        input, weight, bias=None,
        stride=1, padding=0, output_padding=0, groups=1, dilation=1,
    ):
        return _impl_3d(
            input, weight, bias, stride, padding, output_padding, groups, dilation,
        )

    # NOTE — ``aten.convolution.default`` interceptor not yet enabled.
    # ``F.conv_transpose1d/2d/3d`` decompose to ``aten.convolution`` with
    # ``transposed=True`` before reaching the lowering registry. The
    # planned interceptor would route 1D/3D transposed convs through the
    # ``_impl_{1,2,3}d`` decomposition above (flip + permute + clone +
    # upsample → conv2d). The decomposition reaches the per-frame
    # ``_VulkanConv2dExternKernel`` but triggers a segfault during the
    # downstream slangc compile of the intermediate kernel. Re-enable
    # once the IR realization path is stable — likely via a custom op
    # ``torch_vulkan::conv_transpose{1,2,3}d`` wrapping the eager flow,
    # or a dedicated ``slang_conv_transpose`` template.
    # Tests in ``TestM6Phase4ConvTranspose`` xfail on the upstream
    # ``TypeError: convolution() got an unexpected keyword argument 'out'``.
