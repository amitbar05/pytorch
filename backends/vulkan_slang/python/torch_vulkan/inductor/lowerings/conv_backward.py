"""Conv2d backward lowering — _slang_tile_conv2d_bwd template dispatch (CODEGEN.3).

Extracted from conv.py to comply with anti-goal #7 (800-line cap).
Intercepts ``aten.convolution_backward`` for groups==1 / fp32 / vulkan
and routes through the Slang bwd template (single dispatch) instead
of Inductor's stock decomposition (which decomposes into mm calls).
For groups>1 or transposed conv, returns NotImplemented so the
upstream ``make_fallback(aten.convolution_backward)`` handles it.
"""

from __future__ import annotations

import torch as _torch_module
from torch._inductor import ir as _ir_module

_aten = _torch_module.ops.aten


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


def _vk_realize_then_unwrap(x):
    """Realize Pointwise/Reduction, then unwrap StorageBox → data.

    Returns Buffer/ReinterpretView (what ExternKernel expects).
    Used by ``_VulkanConvBwdExternKernel``.

    Handles nested StorageBox (TensorBox → StorageBox → StorageBox → Pointwise)
    by looping until the innermost data is reached.
    Also unwraps View nodes to their inner data.
    """
    if isinstance(x, _ir_module.TensorBox):
        x = x.data
    # S2.0b: unwrap StorageBox/View layers to a fixpoint (interleaved nesting
    # like StorageBox → View → StorageBox → Buffer would otherwise leave a
    # trailing StorageBox that crashes decide_layout).
    while True:
        if isinstance(x, _ir_module.StorageBox):
            x = x.data
            continue
        if isinstance(x, _ir_module.BaseView) and hasattr(x, 'data'):
            x = x.data
            continue
        break
    # If result is not a real Buffer (Pointwise/Reduction/etc.),
    # wrap in a ComputedBuffer so it gets codegen_reference() and allocation.
    if not isinstance(x, (_ir_module.Buffer, _ir_module.ReinterpretView)):
        from torch._inductor.graph import V
        layout = _ir_module.FlexibleLayout(
            device=x.get_device(),
            dtype=x.get_dtype(),
            size=list(x.get_size()),
        )
        buf = _ir_module.ComputedBuffer(name=None, layout=layout, data=x)
        V.graph.register_buffer(buf, set_name=True)
        V.graph.register_operation(buf)  # Sets operation_name for scheduler
        return buf
    return x


# ═════════════════════════════════════════════════════════════════════
# CODEGEN.3: _VulkanConvBwdExternKernel — ExternKernelOut subclass that
# routes conv2d backward through the _slang_tile_conv2d_bwd template.
# At module level so closures in _get_conv2d_backward_custom_op_lowering
# resolve the real class, not a function-local shadow.
# ═════════════════════════════════════════════════════════════════════

class _VulkanConvBwdExternKernel(_ir_module.ExternKernelOut):
    """ExternKernelOut that dispatches conv2d backward via slang template.

    The primary output (``layout``) is grad_input. grad_weight (and
    optionally grad_bias) are passed as additional inputs — they are
    pre-allocated zero buffers that the Slang kernel writes into via
    ``bwd_diff(conv_inner_madd)`` accumulation.

    The codegen override emits ``_slang_tile_conv2d_bwd(...)`` which
    writes into both grad_input and grad_weight in a single dispatch.
    """

    @staticmethod
    def unwrap_storage(inputs):
        """Only unwrap input/weight/grad_output (indices 0-2).
        Keep gw_box/gb_box (indices 3+) as StorageBoxes so the scheduler
        preserves the data dependency and allocates them before the kernel.
        """
        inputs_new = []
        for i, x in enumerate(inputs):
            if i <= 2:
                x = _vk_realize_then_unwrap(x)
            # else: keep as StorageBox to preserve dependency
            inputs_new.append(x)
        return inputs_new

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
        # CODEGEN.3-fix: mark grad_weight (inputs[3]) and grad_bias (inputs[4])
        # as mutation outputs so the scheduler knows this kernel WRITES to them.
        # Without this, the scheduler treats them as read-only inputs, and
        # downstream codegen creates a separate buffer for the output — the
        # kernel's writes never reach the parameter's .grad attribute.
        from torch._inductor.ir import MutationOutput, NoneLayout
        if len(self.inputs) > 3:
            gw_node = self.inputs[3]
            self.mutation_outputs.append(
                MutationOutput(NoneLayout(device=layout.device), gw_node, self)
            )
        if has_bias and len(self.inputs) > 4:
            gb_node = self.inputs[4]
            self.mutation_outputs.append(
                MutationOutput(NoneLayout(device=layout.device), gb_node, self)
            )

    def codegen(self, wrapper):
        """Emit a call to ``_slang_tile_conv2d_bwd`` in the generated wrapper.

        A2.5: When ``V.graph.aot_mode`` is True, emits C++ AOTI dispatch
        calls with pre-compiled SPIR-V instead of Python function calls.
        """
        from torch._inductor import graph as _inductor_graph

        if getattr(_inductor_graph.V.graph, 'aot_mode', False):
            self._codegen_aoti(wrapper)
            return

        # M-NEW.12: flush batcher before this direct Vulkan dispatch.
        # Batched pointwise/foreach kernels queued before this ExternKernelOut
        # (e.g., ReLU backward mask for Conv-GN-ReLU chains) must be flushed
        # so their output buffers are populated before the conv backward reads
        # the GN grad_input.
        wrapper._flush_batcher_before_direct_call()

        wrapper.add_import_once(
            "from torch_vulkan.inductor.vulkan_template_caller "
            "import _slang_tile_conv2d_bwd"
        )

        # TR.20: the wrapper's codegen_allocation won't trigger for
        # extern-kernel primary outputs when they're the final output
        # (no downstream consumer reads them).  Emit allocation directly.
        out_name = self.codegen_reference()
        layout = self.get_layout()
        size = list(layout.size)
        stride = list(layout.stride)
        dtype_str = "torch.float32"
        wrapper.writeline(
            f"{out_name} = empty_strided_vulkan({size}, {stride}, {dtype_str}, "
            f"lifetime_class='transient')"
        )

        input_names = [inp.codegen_reference() for inp in self.inputs]

        # inputs layout: [input, weight, grad_output, grad_weight, grad_bias?]
        input_t = input_names[0]
        weight_t = input_names[1]
        grad_out = input_names[2]
        grad_weight = input_names[3]

        sH, sW = self.stride_arg
        pH, pW = self.padding_arg
        dH, dW = self.dilation_arg

        # M23.2: output buffers are zero-initialized by the allocator
        # (M23.1, commit 60541e0e1e8).  Explicit .zero_() calls are redundant
        # dispatches and push the dispatch count above the 2-dispatch budget.
        # The sync below guarantees any prior operations are drained before
        # the kernel reads the output buffers.

        grad_bias_arg = "None"
        if self.has_bias and len(input_names) > 4:
            grad_bias_arg = input_names[4]

        # C1.2: before-sync is redundant — _flush_batcher_before_direct_call()
        # already submitted prior dispatches, and Vulkan single-queue in-order
        # execution guarantees they complete before this kernel runs.
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
        # C1.2: after-sync is redundant — Vulkan single-queue in-order
        # execution guarantees downstream dispatches wait for this kernel.
        # The optimizer's foreach dispatch goes to the same queue.

    def _codegen_aoti(self, wrapper):
        """Emit C++ AOTI dispatch for conv2d backward via pre-compiled SPIR-V."""
        from torch._inductor.virtualized import V

        from ...templates.caller.conv import _render_conv_bwd_slang

        input_names = [inp.codegen_reference() for inp in self.inputs]
        out_name = self.codegen_reference()

        input_t = input_names[0]
        weight_t = input_names[1]
        grad_out = input_names[2]
        grad_weight = input_names[3]
        grad_bias = input_names[4] if self.has_bias and len(input_names) > 4 else None

        in_layout = self.inputs[0].get_layout()
        w_layout = self.inputs[1].get_layout()
        go_layout = self.inputs[2].get_layout()
        out_layout = self.get_layout()

        N, C_in, iH, iW = in_layout.size
        C_out, C_in_w, kH, kW = w_layout.size
        sH, sW = self.stride_arg
        pH, pW = self.padding_arg
        dH, dW = self.dilation_arg
        oH, oW = go_layout.size[2], go_layout.size[3]

        tile_w = tile_h = tile_c = 8
        threads_w = threads_h = 16

        slang_src = _render_conv_bwd_slang(
            tile_w=tile_w, tile_h=tile_h, tile_c=tile_c,
            threads_w=threads_w, threads_h=threads_h,
        )

        dtype_s = "f32"
        cache_key = f"slang_conv_bwd_m20p3_{dtype_s}"

        in_stride = in_layout.stride
        w_stride = w_layout.stride
        go_stride = go_layout.stride

        common_fields = [
            int(N), int(C_in), int(C_out),
            int(iH), int(iW), int(oH), int(oW),
            int(kH), int(kW),
            int(sH), int(sW), int(pH), int(pW),
            int(dH), int(dW),
            int(in_stride[0]), int(in_stride[1]),
            int(in_stride[2]), int(in_stride[3]),
            int(w_stride[0]), int(w_stride[1]),
            int(w_stride[2]), int(w_stride[3]),
            int(go_stride[0]), int(go_stride[1]),
            int(go_stride[2]), int(go_stride[3]),
        ]
        stride_grad_bias = 1 if grad_bias is not None else 0
        pc_values = list(common_fields) + [stride_grad_bias]

        grid_x = (int(oW) + tile_w - 1) // tile_w
        grid_y = (int(oH) + tile_h - 1) // tile_h
        tile_c_count = (int(C_out) + tile_c - 1) // tile_c
        grid_z = int(N) * tile_c_count

        # buffers: input_f32, weight_f32, grad_out_f32, gi_f32, gw_f32, gb_f32
        buffer_names = [input_t, weight_t, grad_out, out_name, grad_weight]
        if grad_bias:
            buffer_names.append(grad_bias)
        else:
            buffer_names.append("_conv_bwd_dummy_gb")

        output_allocations = [
            {"name": out_name, "shape": [int(s) for s in out_layout.size],
             "stride": [int(s) for s in out_layout.stride], "dtype": "float32"},
        ]
        if not grad_bias:
            output_allocations.append({
                "name": "_conv_bwd_dummy_gb",
                "shape": [1], "stride": [1], "dtype": "float32",
            })

        wrapper.emit_aoti_extern_dispatch(
            slang_src=slang_src,
            cache_key=cache_key,
            buffer_names=buffer_names,
            pc_values=pc_values,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_z=grid_z,
            num_outputs=3,
            output_allocations=output_allocations,
        )


def _get_conv_backward_lowering_impl():
    """Return the lowering for aten.convolution_backward.default.

    Registration is done in bwd_lowerings.py (anti-goal #3).
    Conv2d backward lowering using _slang_tile_conv2d_bwd template (CODEGEN.3).
    """
    import torch
    from torch._inductor import ir

    aten = torch.ops.aten

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
        # Gate on supported envelope: groups==1, not transposed.
        # PF.70 / FP16.1: accept fp16/bf16 inputs — the template caller
        # handles upcast→float32→downcast internally.
        if bool(transposed):
            return NotImplemented
        g = int(groups)
        if g != 1:
            return NotImplemented
        if input.get_device().type != "vulkan":
            return NotImplemented

        # MODEL.1: delegate 5D inputs to Conv3d backward lowering.
        if len(input.get_size()) == 5 and len(weight.get_size()) == 5:
            from .conv3d_backward import _get_conv3d_backward_lowering_impl

            conv3d_impl = _get_conv3d_backward_lowering_impl()
            return conv3d_impl(
                grad_output, input, weight, bias_sizes,
                stride, padding, dilation, transposed,
                output_padding, groups, output_mask,
            )

        # Conv1d backward: reshape 3D tensors to 4D, delegate to 4D path, then
        # squeeze results back. Mirrors the _conv1d_to_conv2d_lowering pattern
        # in conv.py. Without this, the 3D guard below returns NotImplemented and
        # the FallbackKernel produces zero gradients for conv1d.
        if len(input.get_size()) == 3 and len(weight.get_size()) == 3:
            from torch._inductor.lowering import lowerings as _lowerings
            unsq = _lowerings[aten.unsqueeze.default]
            # squeeze.dim squeezes a specific dim; squeeze.default (no-arg) squeezes
            # all size-1 dims but its lowering signature doesn't accept a dim arg.
            sq = _lowerings[aten.squeeze.dim]
            clone_fn = _lowerings[aten.clone.default]

            go_4d = unsq(grad_output, -1)
            inp_4d = unsq(input, -1)
            w_4d = unsq(weight, -1)

            def _1d_to_2d(v, second):
                if isinstance(v, (list, tuple)):
                    return (int(v[0]), second)
                return (int(v), second)

            stride_2d = _1d_to_2d(stride, 1)
            padding_2d = _1d_to_2d(padding, 0)
            dilation_2d = _1d_to_2d(dilation, 1)
            output_padding_2d = _1d_to_2d(output_padding, 0)

            results_4d = _vulkan_convolution_backward(
                go_4d, inp_4d, w_4d, bias_sizes,
                stride_2d, padding_2d, dilation_2d,
                transposed, output_padding_2d, groups, output_mask,
            )
            if results_4d is NotImplemented:
                return NotImplemented

            # Squeeze dummy W dim from grad_input and grad_weight.
            # Only squeeze if result is genuinely 4D (output_mask[i]==False
            # returns a 1-element placeholder that must not be squeezed).
            gi_4d, gw_4d, gb = results_4d
            gi_3d = sq(clone_fn(gi_4d), -1) if len(gi_4d.get_size()) == 4 else gi_4d
            gw_3d = sq(clone_fn(gw_4d), -1) if len(gw_4d.get_size()) == 4 else gw_4d
            return [gi_3d, gw_3d, gb]

        if len(input.get_size()) != 4 or len(weight.get_size()) != 4:
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
        gw_box = _lowerings[aten.empty.memory_format](
            gw_size, dtype=dtype, device=dev
        )
        # M23.2: do NOT call gw_box.realize() — the ExternKernelOut codegen
        # handles zero-init and scheduling. Pre-realizing causes the scheduler
        # to treat gw_box as already-finalized, dropping the kernel's writes
        # (zero gradients / sign-flipped weight grads). Same pattern as
        # _get_conv2d_backward_custom_op_lowering at line 430-433.

        # grad_bias allocation (if needed)
        gb_box = None
        kernel_inputs = [input, weight, grad_output, gw_box]
        if has_bias:
            gb_size = [int(w_sizes[0])]
            gb_box = _lowerings[aten.empty.memory_format](
                gb_size, dtype=dtype, device=dev
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

    return _vulkan_convolution_backward


def _get_conv2d_backward_custom_op_lowering():
    """Return a lowering for ``torch_vulkan::conv2d_backward.default``.

    Replaces the ``make_fallback`` in ``lowerings/__init__.py`` (M17.8.d.2)
    with a proper Inductor ExternKernelOut lowering that routes through
    ``_slang_tile_conv2d_bwd`` — the same verified Slang template the
    eager path uses directly.  The ``make_fallback`` path emitted a
    raw ``torch.ops.torch_vulkan.conv2d_backward.default(...)`` call
    that, despite calling the same Python function, produced wrong
    gradients at runtime (conv weight ~7% off, bias ~28% off).  Routing
    through an ExternKernelOut lets Inductor manage the buffer lifecycle
    and emit ``_slang_tile_conv2d_bwd`` via the wrapper's batcher.
    """
    # _VulkanConvBwdExternKernel is at module level — no need to call
    # _get_conv_backward_lowering_impl() first. The closure below will
    # resolve it from the module namespace.

    def _lower_vulkan_conv2d_backward(
        input,
        grad_output,
        weight,
        stride,
        padding,
        dilation,
        groups,
        has_bias,
    ):
        if int(groups) != 1:
            return NotImplemented
        if input.get_device().type != "vulkan":
            return NotImplemented

        sH = int(stride[0])
        sW = int(stride[-1] if len(stride) > 1 else stride[0])
        pH = int(padding[0])
        pW = int(padding[-1] if len(padding) > 1 else padding[0])
        dH = int(dilation[0])
        dW = int(dilation[-1] if len(dilation) > 1 else dilation[0])

        gi_size = input.get_size()
        N, C_in, iH, iW = gi_size
        w_sizes = weight.get_size()
        C_out = w_sizes[0]

        dev = input.get_device()
        dtype = input.get_dtype()

        gi_layout = _ir_module.FixedLayout(
            device=dev, dtype=dtype, size=gi_size,
            stride=[C_in * iH * iW, iH * iW, iW, 1],
        )

        from torch._inductor.lowering import lowerings as _lowerings

        gw_size = [C_out, C_in, int(w_sizes[2]), int(w_sizes[3])]
        gw_box = _lowerings[_aten.empty.memory_format](
            gw_size, dtype=dtype, device=dev
        )
        # Note: do NOT call gw_box.realize() here — the ExternKernelOut
        # codegen handles zero-init and scheduling. Pre-realizing causes
        # the scheduler to treat gw_box as already-finalized, dropping
        # the kernel's writes (zero gradients).

        kernel_inputs = [input, weight, grad_output, gw_box]
        hb = bool(has_bias)
        if hb:
            gb_size = [C_out]
            gb_box = _lowerings[_aten.empty.memory_format](
                gb_size, dtype=dtype, device=dev
            )
            kernel_inputs.append(gb_box)

        kernel = _VulkanConvBwdExternKernel(
            layout=gi_layout,
            inputs=kernel_inputs,
            stride_arg=(sH, sW),
            padding_arg=(pH, pW),
            dilation_arg=(dH, dW),
            has_bias=hb,
        )
        gi_box = _ir_module.TensorBox.create(kernel)

        empty_box = _lowerings[_aten.full.default](
            [1], 0.0, dtype=dtype, device=dev
        )
        result_gw = gw_box
        result_gb = gb_box if hb else empty_box
        return [gi_box, result_gw, result_gb]

    return _lower_vulkan_conv2d_backward
