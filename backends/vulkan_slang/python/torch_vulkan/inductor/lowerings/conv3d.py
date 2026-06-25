"""Conv3d native lowering — _slang_tile_conv3d template dispatch (MODEL.1).

Provides full 3D convolution support (including KD > 1) via the dedicated
slang_conv3d.slang template. This replaces the previous NotImplemented fallback
for KD > 1 that the reshape-to-Conv2d path in conv.py returned.

Supported envelope:
  - groups == 1
  - fp32
  - 5D tensors (NCDHW)
  - Arbitrary padding, stride, dilation in all 3 spatial dims

The lowering creates a _VulkanConv3dExternKernel that emits a direct call
to _slang_tile_conv3d during codegen.
"""

from __future__ import annotations


def _get_conv3d_native_extern_kernel_class():
    """Return the _VulkanConv3dExternKernel class.

    Deferred import to avoid circular deps during module loading.
    """
    import torch
    from torch._inductor import ir

    class _VulkanConv3dExternKernel(ir.ExternKernelOut):
        """ExternKernelOut that dispatches conv3d via the slang_conv3d template.

        Holds conv3d parameters (stride, padding, dilation, epilogue) as
        instance attributes so the codegen path can call ``_slang_tile_conv3d``.
        """

        def __init__(
            self,
            layout,
            inputs,
            stride_arg,
            padding_arg,
            dilation_arg,
            epilogue=None,
        ):
            super().__init__(
                layout=layout,
                inputs=inputs,
                python_kernel_name=(
                    "torch_vulkan.inductor.vulkan_template_caller._slang_tile_conv3d"
                ),
                op_overload=None,
            )
            self.stride_arg = stride_arg
            self.padding_arg = padding_arg
            self.dilation_arg = dilation_arg
            self.epilogue = epilogue

        def codegen(self, wrapper):
            """Emit a call to ``_slang_tile_conv3d`` in the generated wrapper.

            In AOTI mode, compiles the Slang template to SPIR-V at codegen
            time and emits C++ dispatch via ``emit_aoti_extern_dispatch``.
            """
            from torch._inductor import graph as _inductor_graph

            if getattr(_inductor_graph.V.graph, 'aot_mode', False):
                self._codegen_aoti(wrapper)
                return

            # M-NEW.12: flush batcher before direct Vulkan dispatch
            wrapper._flush_batcher_before_direct_call()

            wrapper.add_import_once(
                "from torch_vulkan.inductor.vulkan_template_caller "
                "import _slang_tile_conv3d"
            )

            input_names = [inp.codegen_reference() for inp in self.inputs]
            out_name = self.codegen_reference()

            input_t = input_names[0]
            weight_t = input_names[1]
            bias_t = input_names[2] if len(input_names) > 2 else "None"

            sD, sH, sW = self.stride_arg
            pD, pH, pW = self.padding_arg
            dD, dH, dW = self.dilation_arg

            epilogue_kwarg = (
                f', epilogue="{self.epilogue}"' if self.epilogue else ""
            )

            self.codegen_comment(wrapper)
            wrapper.writeline(
                f"_slang_tile_conv3d("
                f"{input_t}, {weight_t}, {out_name}, "
                f"stride=({sD}, {sH}, {sW}), "
                f"padding=({pD}, {pH}, {pW}), "
                f"dilation=({dD}, {dH}, {dW}), "
                f"bias={bias_t}"
                f"{epilogue_kwarg})"
            )
            self.codegen_size_asserts(wrapper)

        def _codegen_aoti(self, wrapper):
            """Emit C++ AOTI dispatch for conv3d forward via pre-compiled SPIR-V."""
            from ..templates.caller.conv3d import _render_conv3d_slang

            input_names = [inp.codegen_reference() for inp in self.inputs]
            out_name = self.codegen_reference()

            input_t = input_names[0]
            weight_t = input_names[1]
            has_bias = len(input_names) > 2
            bias_t = input_names[2] if has_bias else None

            in_layout = self.inputs[0].get_layout()
            w_layout = self.inputs[1].get_layout()
            out_layout = self.get_layout()

            N, C_in, iD, iH, iW = in_layout.size
            C_out, _, kD, kH, kW = w_layout.size
            oD, oH, oW = out_layout.size[2], out_layout.size[3], out_layout.size[4]

            sD, sH, sW = self.stride_arg
            pD, pH, pW = self.padding_arg
            dD, dH, dW = self.dilation_arg

            tile_w = tile_h = tile_c = 4
            threads_w = threads_h = 8

            slang_src = _render_conv3d_slang(
                tile_w=tile_w, tile_h=tile_h, tile_c=tile_c,
                threads_w=threads_w, threads_h=threads_h,
            )

            dtype_s = "f32"
            cache_key = f"slang_conv3d_{dtype_s}_aoti"

            in_stride = in_layout.stride
            w_stride = w_layout.stride
            out_stride = out_layout.stride

            pc_values = [
                int(N), int(C_in), int(C_out),
                int(iD), int(iH), int(iW),
                int(oD), int(oH), int(oW),
                int(kD), int(kH), int(kW),
                int(sD), int(sH), int(sW),
                int(pD), int(pH), int(pW),
                int(dD), int(dH), int(dW),
                int(tile_w), int(tile_h), int(tile_c),
                1 if has_bias else 0,  # stride_bias
                0, 0, 0,  # padding
            ]

            grid_x = (int(oW) + tile_w - 1) // tile_w
            grid_y = (int(oH) + tile_h - 1) // tile_h
            tile_c_count = (int(C_out) + tile_c - 1) // tile_c
            grid_z = int(N) * tile_c_count * int(oD)

            buffer_names = [input_t, weight_t]
            output_allocations = []
            if has_bias:
                buffer_names.append(bias_t)
            else:
                buffer_names.append("_conv3d_dummy_bias")
                output_allocations.append({
                    "name": "_conv3d_dummy_bias",
                    "shape": [1], "stride": [1], "dtype": "float32",
                })
            buffer_names.append(out_name)
            output_allocations.append({
                "name": out_name,
                "shape": [int(s) for s in out_layout.size],
                "stride": [int(s) for s in out_layout.stride],
                "dtype": "float32",
            })

            wrapper.emit_aoti_extern_dispatch(
                slang_src=slang_src,
                cache_key=cache_key,
                buffer_names=buffer_names,
                pc_values=pc_values,
                grid_x=grid_x,
                grid_y=grid_y,
                grid_z=grid_z,
                num_outputs=1,
                output_allocations=output_allocations if output_allocations else None,
            )

    return _VulkanConv3dExternKernel


def _vulkan_conv3d_native_lowering(
    input, weight, bias, stride, padding, dilation, groups
):
    """Native Conv3d lowering using slang_conv3d template.

    Handles all KD values (including KD > 1 which the reshape-to-conv2d
    path cannot handle). Called from the existing Conv3d lowering in
    conv.py when the reshape-to-conv2d path returns NotImplemented.

    Supported: groups == 1, fp32, 5D tensors only.
    Returns NotImplemented for anything outside this envelope.
    """
    import torch
    from torch._inductor import ir

    if len(input.get_size()) != 5 or len(weight.get_size()) != 5:
        raise NotImplementedError("vulkan conv3d: non-5D input/weight")

    g = int(groups)
    if g != 1:
        raise NotImplementedError("vulkan conv3d: groups != 1")

    if input.get_dtype() != torch.float32:
        raise NotImplementedError("vulkan conv3d: non-fp32 dtype")

    t1_sizes = input.get_size()
    w_sizes = weight.get_size()
    N = t1_sizes[0]
    C_in = t1_sizes[1]
    iD = t1_sizes[2]
    iH = t1_sizes[3]
    iW = t1_sizes[4]

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

    # Compute output sizes
    D_out = (iD + 2 * pD - dD * (kD - 1) - 1) // sD + 1
    H_out = (iH + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    W_out = (iW + 2 * pW - dW * (kW - 1) - 1) // sW + 1

    dev = input.get_device()
    dtype = input.get_dtype()

    # Output layout: contiguous NCDHW
    out_layout = ir.FixedLayout(
        device=dev,
        dtype=dtype,
        size=[N, C_out, D_out, H_out, W_out],
        stride=[
            C_out * D_out * H_out * W_out,
            D_out * H_out * W_out,
            H_out * W_out,
            W_out,
            1,
        ],
    )

    inputs = [input, weight]
    if bias is not None:
        inputs.append(bias)

    ExternKernelClass = _get_conv3d_native_extern_kernel_class()
    kernel = ExternKernelClass(
        layout=out_layout,
        inputs=inputs,
        stride_arg=(sD, sH, sW),
        padding_arg=(pD, pH, pW),
        dilation_arg=(dD, dH, dW),
    )
    return ir.TensorBox.create(kernel)
