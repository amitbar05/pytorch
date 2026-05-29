"""Conv3d backward lowering — _slang_tile_conv3d_bwd template dispatch (MODEL.1).

Intercepts ``aten.convolution_backward`` for 5D inputs (groups==1 / fp32 / vulkan)
and routes through the Conv3d Slang bwd template (single dispatch) instead of
Inductor's stock decomposition.

For 4D inputs, returns NotImplemented so the existing Conv2d backward lowering
(conv_backward.py) handles them.
"""

from __future__ import annotations


def _get_conv3d_backward_extern_kernel_class():
    """Return the _VulkanConv3dBwdExternKernel class."""
    import torch
    from torch._inductor import ir

    class _VulkanConv3dBwdExternKernel(ir.ExternKernelOut):
        """ExternKernelOut that dispatches conv3d backward via slang template.

        The primary output (``layout``) is grad_input. grad_weight (and
        optionally grad_bias) are passed as additional inputs — they are
        pre-allocated zero buffers that the Slang kernel writes into via
        ``bwd_diff(conv_inner_madd)`` accumulation.
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
                    "torch_vulkan.inductor.vulkan_template_caller._slang_tile_conv3d_bwd"
                ),
                op_overload=None,
            )
            self.stride_arg = stride_arg
            self.padding_arg = padding_arg
            self.dilation_arg = dilation_arg
            self.has_bias = has_bias

        def codegen(self, wrapper):
            """Emit ``_slang_tile_conv3d_bwd`` in the generated wrapper."""
            wrapper.add_import_once(
                "from torch_vulkan.inductor.vulkan_template_caller "
                "import _slang_tile_conv3d_bwd"
            )

            input_names = [inp.codegen_reference() for inp in self.inputs]
            out_name = self.codegen_reference()  # grad_input

            # inputs: [input, weight, grad_output, grad_weight, grad_bias?]
            input_t = input_names[0]
            weight_t = input_names[1]
            grad_out = input_names[2]
            grad_weight = input_names[3]

            sD, sH, sW = self.stride_arg
            pD, pH, pW = self.padding_arg
            dD, dH, dW = self.dilation_arg

            # Zero-init output buffers (bwd kernel accumulates via +=)
            wrapper.writeline(f"{out_name}.zero_()")
            wrapper.writeline(f"{grad_weight}.zero_()")

            grad_bias_arg = "None"
            if self.has_bias and len(input_names) > 4:
                grad_bias_arg = input_names[4]
                wrapper.writeline(f"{grad_bias_arg}.zero_()")

            self.codegen_comment(wrapper)
            wrapper.writeline(
                f"_slang_tile_conv3d_bwd("
                f"{input_t}, {weight_t}, {grad_out}, "
                f"{out_name}, {grad_weight}, "
                f"stride=({sD}, {sH}, {sW}), "
                f"padding=({pD}, {pH}, {pW}), "
                f"dilation=({dD}, {dH}, {dW}), "
                f"grad_bias={grad_bias_arg})"
            )
            self.codegen_size_asserts(wrapper)

    return _VulkanConv3dBwdExternKernel


def _get_conv3d_backward_lowering_impl():
    """Return the implementation function for 5D aten.convolution_backward.

    Called from bwd_lowerings.py during registration.
    Returns NotImplemented for 4D inputs (handled by conv_backward.py).
    """
    import torch
    from torch._inductor import ir

    aten = torch.ops.aten

    def _vulkan_conv3d_backward(
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
        """5D convolution backward lowering via slang_conv3d_bwd template.

        Returns NotImplemented for non-5D inputs (Conv2d backward is handled
        by conv_backward.py::_get_conv_backward_lowering_impl).
        """
        # Only handle 5D (Conv3d) inputs
        if len(input.get_size()) != 5 or len(weight.get_size()) != 5:
            return NotImplemented

        if bool(transposed):
            return NotImplemented
        g = int(groups)
        if g != 1:
            return NotImplemented
        if input.get_dtype() != torch.float32:
            return NotImplemented
        if input.get_device().type != "vulkan":
            return NotImplemented

        t_sizes = input.get_size()
        w_sizes = weight.get_size()
        N = t_sizes[0]
        C_in = t_sizes[1]
        iD = t_sizes[2]
        iH = t_sizes[3]
        iW = t_sizes[4]
        C_out = w_sizes[0]
        kD = int(w_sizes[2])
        kH = int(w_sizes[3])
        kW = int(w_sizes[4])

        sD = int(stride[0])
        sH = int(stride[1]) if len(stride) > 1 else sD
        sW = int(stride[2]) if len(stride) > 2 else sH

        pD = int(padding[0])
        pH = int(padding[1]) if len(padding) > 1 else pD
        pW = int(padding[2]) if len(padding) > 2 else pH

        dD = int(dilation[0])
        dH = int(dilation[1]) if len(dilation) > 1 else dD
        dW = int(dilation[2]) if len(dilation) > 2 else dH

        dev = input.get_device()
        dtype = input.get_dtype()

        need_gi = bool(output_mask[0])
        need_gw = bool(output_mask[1])
        has_bias = bool(output_mask[2] and bias_sizes)

        # Output layouts (contiguous NCDHW)
        gi_size = [N, C_in, iD, iH, iW]
        gi_stride = [C_in * iD * iH * iW, iD * iH * iW, iH * iW, iW, 1]
        gi_layout = ir.FixedLayout(
            device=dev, dtype=dtype, size=gi_size, stride=gi_stride
        )

        # Pre-allocate grad_weight as a zero buffer
        from torch._inductor.lowering import lowerings as _lowerings

        gw_size = [C_out, C_in, kD, kH, kW]
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
        ExternKernelClass = _get_conv3d_backward_extern_kernel_class()
        kernel = ExternKernelClass(
            layout=gi_layout,
            inputs=kernel_inputs,
            stride_arg=(sD, sH, sW),
            padding_arg=(pD, pH, pW),
            dilation_arg=(dD, dH, dW),
            has_bias=has_bias,
        )
        gi_box = ir.TensorBox.create(kernel)

        # Build output tuple matching aten.convolution_backward signature
        empty_box = _lowerings[aten.full.default](
            [1], 0.0, dtype=dtype, device=dev
        )
        result_gi = gi_box if need_gi else empty_box
        result_gw = gw_box if need_gw else empty_box
        result_gb = gb_box if has_bias and gb_box is not None else empty_box
        return [result_gi, result_gw, result_gb]

    return _vulkan_conv3d_backward
