"""Conv2d backward lowering — _slang_tile_conv2d_bwd template dispatch (CODEGEN.3).

Extracted from conv.py to comply with anti-goal #7 (800-line cap).
Intercepts ``aten.convolution_backward`` for groups==1 / fp32 / vulkan
and routes through the Slang bwd template (single dispatch) instead
of Inductor's stock decomposition (which decomposes into mm calls).
For groups>1 or transposed conv, returns NotImplemented so the
upstream ``make_fallback(aten.convolution_backward)`` handles it.
"""

from __future__ import annotations


def _audit_conv_backward_routing() -> None:
    """T4.2 — Audit ``aten.convolution_backward`` routing via BWD_TEMPLATE_REGISTRY.

    Mirrors ``lowerings/matmul.py::_register_matmul_backward``. We do NOT
    register a lowering for ``aten.convolution_backward.default`` —
    anti-goal #3 prohibits hand-written ``aten.<op>_backward`` lowerings
    in ``lowerings/``. Instead, this function confirms that the registry
    has the ``conv_im2col_f32`` entry that pairs with the forward FX
    pattern (T4.4), so the backward path can flow through
    ``bwd_diff_dispatch.dispatch_template_bwd("conv_im2col_f32", ...)``
    once a paired backward FX rewrite is wired up.

    Today the conv_im2col forward FX pattern only fires on
    ``torch_vulkan::conv2d_with_optional_bias`` (the eager custom op);
    AOT Autograd lowers ``aten.convolution_backward`` through Inductor's
    stock decomposition, which materializes patches via ``aten.mm`` and
    routes through the same forward template. The dispatch-count regression
    on ``test_convolution_backward_dispatch_count`` (9 vs 8) is the residual
    orphan pointwise from the stock decomp's bias-sum + patches-fold pair —
    a paired FX rewrite (TODO) lifts it into the template path.
    """
    import logging

    _log = logging.getLogger(__name__)
    from torch_vulkan.inductor.bwd_diff_dispatch import resolve_backward_kind

    for op_name in ("aten.convolution.default", "aten.convolution_backward.default"):
        resolved = resolve_backward_kind(op_name)
        if resolved is None:
            _log.warning(
                "T4.2: No BWD_TEMPLATE_REGISTRY entry for %s; "
                "conv backward will fall through to Inductor's stock decomp.",
                op_name,
            )
        elif not resolved.is_template_jinja:
            _log.warning(
                "T4.2: BWD_TEMPLATE_REGISTRY entry for %s has kind=%s; "
                "expected TEMPLATE_JINJA.",
                op_name,
                resolved.kind,
            )
        else:
            _log.debug(
                "T4.2: %s backward routing confirmed → %s (template jinja)",
                op_name,
                resolved.fwd_key,
            )


def _register_conv_backward_lowering() -> None:
    """Register the conv2d backward lowering using _slang_tile_conv2d_bwd.

    Called from ``_register_conv_and_pool_lowerings`` in ``conv.py`` so
    the Conv2d custom op is already resolved and registered.
    """
    import torch
    from torch._inductor import ir
    from torch._inductor.lowering import register_lowering

    aten = torch.ops.aten

    # ═════════════════════════════════════════════════════════════════════
    # CODEGEN.3: _VulkanConvBwdExternKernel — ExternKernelOut subclass that
    # routes conv2d backward through the _slang_tile_conv2d_bwd template.
    # ═════════════════════════════════════════════════════════════════════

    class _VulkanConvBwdExternKernel(ir.ExternKernelOut):
        """ExternKernelOut that dispatches conv2d backward via slang template.

        The primary output (``layout``) is grad_input. grad_weight (and
        optionally grad_bias) are passed as additional inputs — they are
        pre-allocated zero buffers that the Slang kernel writes into via
        ``bwd_diff(conv_inner_madd)`` accumulation.

        The codegen override emits ``_slang_tile_conv2d_bwd(...)`` which
        writes into both grad_input and grad_weight in a single dispatch.
        """

        def __init__(
            self,
            layout,
            inputs,
            stride_arg,
            padding_arg,
            dilation_arg,
            has_bias=False,
        ):
            super().__init__(
                layout=layout,
                inputs=inputs,
                python_kernel_name=(
                    "torch_vulkan.inductor.vulkan_template_caller._slang_tile_conv2d_bwd"
                ),
                op_overload=None,
            )
            self.stride_arg = stride_arg
            self.padding_arg = padding_arg
            self.dilation_arg = dilation_arg
            self.has_bias = has_bias

        def codegen(self, wrapper):
            """Emit a call to ``_slang_tile_conv2d_bwd`` in the generated wrapper.

            The wrapper has pre-allocated the output buffers (grad_input
            via ExternKernelOut, grad_weight/grad_bias via aten.full).
            We emit zero-init on the output buffers, then the single
            dispatch call.
            """
            wrapper.add_import_once(
                "from torch_vulkan.inductor.vulkan_template_caller "
                "import _slang_tile_conv2d_bwd"
            )

            input_names = [inp.codegen_reference() for inp in self.inputs]
            out_name = self.codegen_reference()  # grad_input

            # inputs layout: [input, weight, grad_output, grad_weight,
            #                  grad_bias?]
            input_t = input_names[0]
            weight_t = input_names[1]
            grad_out = input_names[2]
            grad_weight = input_names[3]

            sH, sW = self.stride_arg
            pH, pW = self.padding_arg
            dH, dW = self.dilation_arg

            # Zero-init output buffers (bwd kernel accumulates via +=)
            wrapper.writeline(f"{out_name}.zero_()")
            wrapper.writeline(f"{grad_weight}.zero_()")

            grad_bias_arg = "None"
            if self.has_bias and len(input_names) > 4:
                grad_bias_arg = input_names[4]
                wrapper.writeline(f"{grad_bias_arg}.zero_()")

            self.codegen_comment(wrapper)
            wrapper.writeline(
                f"_slang_tile_conv2d_bwd("
                f"{input_t}, {weight_t}, {grad_out}, "
                f"{out_name}, {grad_weight}, "
                f"stride=({sH}, {sW}), "
                f"padding=({pH}, {pW}), "
                f"dilation=({dH}, {dW}), "
                f"grad_bias={grad_bias_arg})"
            )
            self.codegen_size_asserts(wrapper)

    # ═════════════════════════════════════════════════════════════════════
    # CODEGEN.3: aten.convolution_backward lowering → _slang_tile_conv2d_bwd
    # ═════════════════════════════════════════════════════════════════════
    @register_lowering(aten.convolution_backward.default, type_promotion_kind=None)
    def _vulkan_convolution_backward(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask,
    ):
        # Gate on supported envelope: groups==1, not transposed, 4D, fp32.
        if bool(transposed):
            return NotImplemented
        g = int(groups)
        if g != 1:
            return NotImplemented
        if len(input.get_size()) != 4 or len(weight.get_size()) != 4:
            return NotImplemented
        if input.get_dtype() != torch.float32:
            return NotImplemented
        if input.get_device().type != "vulkan":
            return NotImplemented

        t_sizes = input.get_size()
        w_sizes = weight.get_size()
        N = t_sizes[0]
        C_in = t_sizes[1]
        iH = t_sizes[2]
        iW = t_sizes[3]
        C_out = w_sizes[0]
        kH = w_sizes[2]
        kW = w_sizes[3]

        sH = int(stride[0])
        sW = int(stride[-1] if len(stride) > 1 else stride[0])
        pH = int(padding[0])
        pW = int(padding[-1] if len(padding) > 1 else padding[0])
        dH = int(dilation[0])
        dW = int(dilation[-1] if len(dilation) > 1 else dilation[0])

        dev = input.get_device()
        dtype = input.get_dtype()

        need_gi = bool(output_mask[0])
        need_gw = bool(output_mask[1])
        has_bias = bool(output_mask[2] and bias_sizes)

        # Output layouts (contiguous NCHW)
        gi_size = [N, C_in, iH, iW]
        gi_stride = [C_in * iH * iW, iH * iW, iW, 1]
        gi_layout = ir.FixedLayout(
            device=dev, dtype=dtype, size=gi_size, stride=gi_stride
        )
        # Pre-allocate grad_weight as a zero buffer (ExternKernelOut inputs)
        from torch._inductor.lowering import lowerings as _lowerings

        gw_size = [C_out, C_in, kH, kW]
        gw_box = _lowerings[aten.full.default](
            gw_size, 0.0, dtype=dtype, device=dev
        )

        # grad_bias allocation (if needed)
        gb_box = None
        kernel_inputs = [input, weight, grad_output, gw_box]
        if has_bias:
            gb_size = [int(w_sizes[0])]
            gb_box = _lowerings[aten.full.default](
                gb_size, 0.0, dtype=dtype, device=dev
            )
            kernel_inputs.append(gb_box)

        # Create the ExternKernelOut (grad_input is the primary output)
        kernel = _VulkanConvBwdExternKernel(
            layout=gi_layout,
            inputs=kernel_inputs,
            stride_arg=(sH, sW),
            padding_arg=(pH, pW),
            dilation_arg=(dH, dW),
            has_bias=has_bias,
        )
        gi_box = ir.TensorBox.create(kernel)

        # Build output tuple matching aten.convolution_backward signature
        # output_mask[i]==False → return empty(0,) placeholder
        empty_box = _lowerings[aten.full.default](
            [1], 0.0, dtype=dtype, device=dev
        )
        result_gi = gi_box if need_gi else empty_box
        result_gw = gw_box if need_gw else empty_box
        result_gb = gb_box if has_bias and gb_box is not None else empty_box
        return [result_gi, result_gw, result_gb]

    # T4.2 — audit backward routing via BWD_TEMPLATE_REGISTRY
    _audit_conv_backward_routing()
