r"""Conv2d + max_pool2d lowerings (PF.3 Strike 2).

NOTE — Conv2d lowering is now the **active** path for groups==1 (Path 2):

    The ``conv_im2col`` FX pattern (priority 15 in
    ``fx_passes/patterns/builtin_patterns.py``) now skips groups==1 nodes,
    letting this lowering fire instead.  The lowering routes through the
    dedicated ``slang_conv2d`` template — a direct tiled conv2d compute
    shader that eliminates the 6-copy dispatch overhead of the old
    im2col+mm decomposition.

    For groups>1, the ``conv_im2col`` FX pattern still fires and decomposes
    into per-group im2col+mm.

    The ``max_pool2d`` lowering (below) has **no** corresponding FX pattern,
    so it is the active decomposition path for ``torch_vulkan::max_pool2d``.

Convolution Lowering Support Matrix (T4.12):

+--------------------------------+--------------------+-----------------------------+
| Feature                        | Status             | Notes                       |
+================================+====================+=============================+
| Conv2d, groups=1               | ✅ Full fwd+bwd    | ``slang_conv2d.slang`` +    |
|                                |                    | CG.M6 bwd template          |
+--------------------------------+--------------------+-----------------------------+
| Conv2d, groups>1 (depthwise)   | ✅ Fwd (M6 Phase 2) | Decomposed to per-group      |
|  (incl. grouped conv)          |                    | im2col+mm via                |
|                                |                    | ``conv_im2col`` FX pattern.  |
|                                |                    | Bwd via standard autograd.   |
+--------------------------------+--------------------+-----------------------------+
| Conv1d                          | ✅ Fwd+bwd (M6 P1) | Reshape to Conv2d + dedicated|
|                                |                    | ``slang_conv2d.slang``      |
|                                |                    | template. Backward inherited|
|                                |                    | from Conv2d bwd (CG.M6).    |
+--------------------------------+--------------------+-----------------------------+
| Conv3d                          | ✅ Fwd+bwd (MODEL.1) | KD==1: reshape to Conv2d;    |
|                                |                    | KD>1: native slang_conv3d    |
|                                |                    | template. Backward via       |
|                                |                    | slang_conv3d_bwd template.   |
+--------------------------------+--------------------+-----------------------------+
| Transposed conv                 | ❌ Not supported   | Falls to ``aten.convolution``|
|                                |                    | extern. No Vulkan-native    |
|                                |                    | ``conv_transpose`` path.    |
+--------------------------------+--------------------+-----------------------------+
| Dilation > 1                    | ✅ Supported       | im2col handles arbitrary    |
|                                |                    | dilation; direct conv       |
|                                |                    | template also supports it.  |
+--------------------------------+--------------------+-----------------------------+
| Stride > 1                      | ✅ Supported       | Both im2col and direct      |
|                                |                    | template paths.             |
+--------------------------------+--------------------+-----------------------------+
| Padding                          | ✅ Supported       | Direct template handles     |
|                                |                    | padding natively (no        |
|                                |                    | separate ``constant_pad_nd`` |
|                                |                    | dispatch).                  |
+--------------------------------+--------------------+-----------------------------+
| Bias                             | ✅ Supported       | Optional bias fused into    |
|                                |                    | template or applied via     |
|                                |                    | broadcast add.              |
+--------------------------------+--------------------+-----------------------------+
| MaxPool2d                        | ✅ Supported       | Decomposed to ``as_strided``|
|                                |                    | + ``amax`` reduction.       |
+--------------------------------+--------------------+-----------------------------+

Blockers for Transposed conv (T4.12):

- **Conv3d**: ✅ RESOLVED (MODEL.1) — native slang_conv3d template supports
  all KD values. KD==1 still uses the faster reshape-to-Conv2d path;
  KD>1 delegates to the native 3D template.
- **Transposed conv**: No Vulkan-native ``conv_transpose`` path.

M6 Phase 2 (T4.12 groups>1) — COMPLETE:
- Groups>1 is handled by the existing ``conv_im2col`` FX pattern which
  decomposes grouped convs into per-group im2col+mm. The resulting mm
  calls leverage the existing ``_slang_tile_mm`` template.
- Groups=1 continues to use the dedicated ``slang_conv2d.slang`` direct
  template (via this lowering).
- Future: A dedicated depthwise Slang template could reduce dispatch
  count further for ``groups == C_in == C_out`` cases.
"""

from __future__ import annotations


def _register_conv_and_pool_lowerings() -> None:
    """PF.3 (Strike 2) — Inductor lowerings for the PF.30 conv2d / max_pool2d
    custom_op shims.

    **Path 2 (2026-05-06)**: The conv2d lowering now routes through the
    dedicated ``slang_conv2d`` template — a direct tiled conv2d compute
    shader that eliminates the 6-copy dispatch overhead of the old
    im2col+mm decomposition. The template handles padding natively
    (no separate ``constant_pad_nd`` dispatch needed).

    The FX pattern ``conv_im2col`` (builtin_patterns.py) is gated to only
    fire for groups>1, so this lowering is the **active** path for groups==1.

    Supported envelope (anything else returns ``NotImplemented``, falling
    through to extern):
      conv2d:    groups == 1, ``input.dim() == 4``, ``weight.dim() == 4``;
                 dilation arbitrary; padding >= 0
      max_pool2d: dilation == 1; ceil_mode == False; ``input.dim() == 4``
    """
    import torch
    from torch._inductor import ir
    from torch._inductor import lowering as L
    from torch._inductor.lowering import register_lowering

    aten = torch.ops.aten

    # The custom_op schemas are registered eagerly by
    # ``register_eager_patch_custom_ops()`` at backend init. The ordering
    # in ``inductor/__init__.py`` calls our register() *before* that
    # factory, so the OpOverloads may not yet exist on the
    # ``torch.ops.torch_vulkan`` namespace. Force-resolve them here so
    # ``register_lowering`` sees the real OpOverload, not None.
    try:
        from ..fx_passes import register_eager_patch_custom_ops

        register_eager_patch_custom_ops()
    except (ImportError, AttributeError, RuntimeError):
        pass
    try:
        conv_op = torch.ops.torch_vulkan.conv2d_with_optional_bias.default
        pool_op = torch.ops.torch_vulkan.max_pool2d.default
    except (AttributeError, RuntimeError):
        return
    # T4.12 — conv1d_with_optional_bias may not exist if the eager-patch
    # factory wasn't run (e.g. inductor-only test paths). Resolve lazily
    # so we still register conv2d if conv1d is unavailable.
    try:
        conv1d_op = torch.ops.torch_vulkan.conv1d_with_optional_bias.default
    except (AttributeError, RuntimeError):
        conv1d_op = None

    # M-pipeline-2: OpOverload-identity-safe lowering lookup helpers
    # live in ``lowerings/_conv_common.py``. ``_get_conv2d_lowering_by_name``
    # is the zero-arg alias preserved for compatibility with the
    # M19.5-followup-1 call sites below; the generalised
    # ``get_lowering_by_name(lowerings, target)`` is the canonical entry
    # point for new code. Both survive ``register_eager_patch_custom_ops()``
    # re-binding of ``torch_vulkan::conv2d_with_optional_bias``, which
    # otherwise produces a NEW ``OpOverload`` object whose Python identity
    # mismatches the key our ``@register_lowering(conv_op)`` decorator
    # stamped into ``L.lowerings``.
    from ._conv_common import (  # noqa: F401 — alias used below
        _get_conv2d_lowering_by_name,
        get_lowering_by_name,
    )

    # ═════════════════════════════════════════════════════════════════════
    # Path 2: _VulkanConv2dExternKernel — ExternKernelOut subclass that
    # routes conv2d through the dedicated slang_conv2d template.
    # ═════════════════════════════════════════════════════════════════════

    def _vk_realize_then_unwrap(x):
        """Realize Pointwise/Reduction, then unwrap StorageBox → data.
        Returns Buffer/ReinterpretView (what ExternKernel expects).

        Handles nested StorageBox (TensorBox → StorageBox → StorageBox → Pointwise)
        by looping until the innermost data is reached.
        """
        import torch._inductor.ir as _ir

        # Unwrap TensorBox → StorageBox
        if isinstance(x, _ir.TensorBox):
            x = x.data

        # If StorageBox contains Pointwise/Reduction, realize it first
        while isinstance(x, _ir.StorageBox):
            x = x.data  # Unwrap — ComputedBuffer fallback handles Pointwise below

        # Unwrap to raw data (Buffer/ReinterpretView)
        if isinstance(x, _ir.StorageBox):
            x = x.data
        # Unwrap View layers (View → inner data)
        while isinstance(x, _ir.BaseView) and hasattr(x, 'data'):
            x = x.data
        # If result is not a real Buffer (Pointwise/Reduction/etc.),
        # wrap in a ComputedBuffer so it gets codegen_reference() and allocation.
        if not isinstance(x, (_ir.Buffer, _ir.ReinterpretView)):
            from torch._inductor.graph import V
            layout = _ir.FlexibleLayout(
                device=x.get_device(),
                dtype=x.get_dtype(),
                size=list(x.get_size()),
            )
            buf = _ir.ComputedBuffer(name=None, layout=layout, data=x)
            V.graph.register_buffer(buf, set_name=True)
            V.graph.register_operation(buf)  # Sets operation_name for scheduler
            return buf
        return x

    class _VulkanConv2dExternKernel(ir.ExternKernelOut):
        """ExternKernelOut that dispatches conv2d via the slang_conv2d template.

        Holds conv2d parameters (stride, padding, dilation, epilogue) as
        instance attributes so the codegen path can call ``_slang_tile_conv2d``.

        The output buffer is pre-allocated by the wrapper's
        ``codegen_allocation`` path (routed through ``empty_strided_vulkan``).
        The ``codegen`` override emits a direct call to ``_slang_tile_conv2d``
        which writes into the pre-allocated output.

        M17.2: Optional ``epilogue`` param (e.g. ``"OpReLU"``) for
        fused conv+activation in a single dispatch.
        """

        @staticmethod
        def unwrap_storage(inputs):
            """Override to realize Pointwise before unwrapping.
            The base class unwraps StorageBox → data, then asserts
            Buffer|ReinterpretView. We intercept to realize first.
            """
            from collections.abc import Sequence

            inputs_new = []
            for x in inputs:
                if isinstance(x, Sequence):
                    x = [
                        _vk_realize_then_unwrap(i) for i in x
                    ]
                else:
                    x = _vk_realize_then_unwrap(x)
                inputs_new.append(x)
            return inputs_new

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
                python_kernel_name="torch_vulkan.inductor.vulkan_template_caller._slang_tile_conv2d",
                op_overload=None,
            )
            self.stride_arg = stride_arg
            self.padding_arg = padding_arg
            self.dilation_arg = dilation_arg
            self.epilogue = epilogue

        def codegen(self, wrapper):
            """Emit a call to ``_slang_tile_conv2d`` in the generated wrapper.

            Overrides the standard ``ExternKernelOut.codegen`` (which would
            go through ``ExternKernelOutLine``) because our function
            signature differs from the standard ``kernel(out=...)``
            convention — we pass the output as a positional arg.

            M17.2: When ``self.epilogue`` is set, the ``epilogue`` kwarg
            is emitted, routing through the fused conv+activation path.

            A2.5: When ``V.graph.aot_mode`` is True, emits C++ AOTI dispatch
            calls with pre-compiled SPIR-V instead of Python function calls.
            """
            from torch._inductor import graph as _inductor_graph

            if getattr(_inductor_graph.V.graph, 'aot_mode', False):
                self._codegen_aoti(wrapper)
                return

            # M-NEW.12: flush batcher before direct Vulkan dispatch
            wrapper._flush_batcher_before_direct_call()

            wrapper.add_import_once(
                "from torch_vulkan.inductor.vulkan_template_caller "
                "import _slang_tile_conv2d"
            )

            input_names = [inp.codegen_reference() for inp in self.inputs]
            out_name = self.codegen_reference()

            input_t = input_names[0]
            weight_t = input_names[1]
            bias_t = input_names[2] if len(input_names) > 2 else "None"

            sH, sW = self.stride_arg
            pH, pW = self.padding_arg
            dH, dW = self.dilation_arg

            epilogue_kwarg = f', epilogue="{self.epilogue}"' if self.epilogue else ""

            self.codegen_comment(wrapper)
            wrapper.writeline(
                f"_slang_tile_conv2d("
                f"{input_t}, {weight_t}, {out_name}, "
                f"stride=({sH}, {sW}), "
                f"padding=({pH}, {pW}), "
                f"dilation=({dH}, {dW}), "
                f"bias={bias_t}"
                f"{epilogue_kwarg})"
            )
            self.codegen_size_asserts(wrapper)

        def _codegen_aoti(self, wrapper):
            """Emit C++ AOTI dispatch for conv2d forward via pre-compiled SPIR-V.

            Replicates the flow of ``_slang_tile_conv2d`` from
            ``templates/caller/conv.py`` but emits C++ code instead of
            Python dispatch calls.  Renders the Slang template, compiles to
            SPIR-V, and emits ``torch_vulkan_aoti_make_kernel`` + ``dispatch``.
            """
            import struct

            from torch._inductor.virtualized import V

            from ...templates.caller.conv import (
                _render_conv2d_slang,
                _validate_epilogue_struct,
                _dtype_to_slang,
            )

            # 1. Collect buffer names and layout info
            input_names = [inp.codegen_reference() for inp in self.inputs]
            out_name = self.codegen_reference()

            input_t = input_names[0]
            weight_t = input_names[1]
            bias_t = input_names[2] if len(input_names) > 2 else None

            in_layout = self.inputs[0].get_layout()
            w_layout = self.inputs[1].get_layout()
            out_layout = self.get_layout()

            N, C_in, iH, iW = in_layout.size
            C_out, C_in_w, kH, kW = w_layout.size
            sH, sW = self.stride_arg
            pH, pW = self.padding_arg
            dH, dW = self.dilation_arg
            oH, oW = out_layout.size[2], out_layout.size[3]

            has_bias = bias_t is not None
            epilogue = _validate_epilogue_struct(self.epilogue)
            dtype_s = "float32"  # Inductor IR is always float32

            tile_w = tile_h = tile_c = 8
            threads_w = threads_h = 16

            # 2. Render Slang source
            slang_src = _render_conv2d_slang(
                tile_w=tile_w,
                tile_h=tile_h,
                tile_c=tile_c,
                threads_w=threads_w,
                threads_h=threads_h,
                has_bias=has_bias,
                epilogue_struct=epilogue,
            )

            cache_key = (
                f"slang_conv2d_{tile_w}x{tile_h}x{tile_c}"
                f"_t{threads_w}x{threads_h}_{dtype_s}"
                f"{'_' + epilogue if epilogue else ''}_msf5"
            )

            # 3. Compute push constants (matching _slang_tile_conv2d layout)
            in_stride = in_layout.stride
            w_stride = w_layout.stride
            out_stride = out_layout.stride

            pc_values = [
                int(N), int(C_in), int(C_out),
                int(iH), int(iW), int(oH), int(oW),
                int(kH), int(kW),
                int(sH), int(sW), int(pH), int(pW),
                1,  # groups
                int(dH), int(dW),
                int(in_stride[0]), int(in_stride[1]),
                int(in_stride[2]), int(in_stride[3]),
                int(w_stride[0]), int(w_stride[1]),
                int(w_stride[2]), int(w_stride[3]),
                int(out_stride[0]), int(out_stride[1]),
                int(out_stride[2]), int(out_stride[3]),
                int(tile_w), int(tile_h), int(tile_c),
            ]
            # stride_bias = 1 if bias present, else 0
            stride_bias = 1 if has_bias else 0
            pc_values.append(stride_bias)

            # 4. Compute grid
            grid_x = (int(oW) + tile_w - 1) // tile_w
            grid_y = (int(oH) + tile_h - 1) // tile_h
            tile_c_count = (int(C_out) + tile_c - 1) // tile_c
            grid_z = int(N) * tile_c_count

            # 5. Collect buffer names in the order the shader expects:
            #    input, weight, bias?, output
            buffer_names = [input_t, weight_t]
            if has_bias:
                buffer_names.append(bias_t)
            else:
                # Dummy bias — allocated below
                buffer_names.append("_conv_dummy_bias")
            buffer_names.append(out_name)

            # 6. Allocate output and dummy bias (if needed)
            output_allocations = []
            if not has_bias:
                output_allocations.append({
                    "name": "_conv_dummy_bias",
                    "shape": [1],
                    "stride": [1],
                    "dtype": "float32",
                })
            output_allocations.append({
                "name": out_name,
                "shape": [int(s) for s in out_layout.size],
                "stride": [int(s) for s in out_layout.stride],
                "dtype": "float32",
            })

            # 7. Emit AOTI dispatch
            entry = (
                f"computeMain<{epilogue}>"
                if epilogue is not None
                else "computeMain<OpIdentity>"
            )
            wrapper.emit_aoti_extern_dispatch(
                slang_src=slang_src,
                cache_key=cache_key,
                buffer_names=buffer_names,
                pc_values=pc_values,
                grid_x=grid_x,
                grid_y=grid_y,
                grid_z=grid_z,
                num_outputs=1,
                output_allocations=output_allocations,
            )

    # M-pipeline-2: this lowering targets a CUSTOM op
    # (``torch_vulkan::conv2d_with_optional_bias``) whose ``OpOverload``
    # identity is NOT stable across ``register_eager_patch_custom_ops()``
    # re-runs. Any caller that resolves the op fresh
    # (``torch.ops.torch_vulkan.conv2d_with_optional_bias.default``) and
    # does ``lowerings[op]`` will MISS — they must use
    # ``get_lowering_by_name(lowerings, target)`` from `_conv_common.py`
    # instead. See the conv1d / conv1d_to_conv2d delegations below for
    # the canonical pattern.
    @register_lowering(conv_op, type_promotion_kind=None)
    def _vulkan_conv2d_with_optional_bias(
        input, weight, bias, stride, padding, dilation, groups
    ):
        if len(input.get_size()) != 4 or len(weight.get_size()) != 4:
            return NotImplemented

        # M19.5 — input dims (N especially) may be ``SymInt``/``sympy.Expr``
        # under dynamic-shape compile (e.g. ``torch._dynamo.mark_dynamic
        # (x, 0)``). Don't ``int()``-coerce — sympy propagates through
        # the H_out / W_out arithmetic below and ``ir.FixedLayout``
        # accepts sympy size/stride expressions. Weight dims are
        # almost always static (nn.Conv2d's kernel shape is concrete
        # at module-build time); we still keep their values raw rather
        # than ``int()`` to avoid spuriously dropping a static SymInt
        # to a Python int (the rest of the lowering tolerates either).
        from torch_vulkan.inductor.kernel.symbolic import get_static_numel

        g = int(groups)
        t1_sizes = input.get_size()
        w_sizes = weight.get_size()
        N = t1_sizes[0]
        C_in = t1_sizes[1]
        H_in = t1_sizes[2]
        W_in = t1_sizes[3]
        C_out = w_sizes[0]
        kH = w_sizes[2]
        kW = w_sizes[3]

        # For the groups>1 decomposition path we still need a concrete
        # C_in/C_out to compute per-group channel slice indices.
        # ``mark_dynamic`` on the batch dim doesn't affect channels, so
        # this is normally fine; if a model marks channels dynamic
        # we fall through to extern (g != 1 branch below will
        # NotImplement on the % g check anyway).
        C_in_static = get_static_numel(C_in)
        C_out_static = get_static_numel(C_out)

        if g != 1:
            # M6 Phase 2 — decompose grouped/depthwise conv into per-group
            # group-1 Conv2d calls.  This bypasses im2col+as_strided whose
            # strided views materialize as all-zeros on Vulkan.
            #
            # * Depthwise (g == C_in == C_out): per-channel (g groups of
            #   C_in_per_group == 1).
            # * Grouped conv (g > 1, g < C_in): per-group slices of
            #   C_in//g channels each.
            #
            # Each per-group call uses groups=1 and routes through the
            # dedicated ``slang_conv2d`` template.  Backward is inherited
            # automatically from the Conv2d bwd template (CG.M6) because
            # each per-group call has groups=1.
            if C_in_static is None or C_out_static is None:
                # Dynamic channels under grouped conv — can't compute
                # per-group slice indices. Fall through to extern.
                return NotImplemented
            if C_in_static % g != 0 or C_out_static % g != 0:
                return NotImplemented

            from torch._inductor.lowering import lowerings as _lowerings

            C_in_per_g = C_in_static // g
            C_out_per_g = C_out_static // g

            groups_out = []
            for i in range(g):
                inp_g = _lowerings[aten.slice.Tensor](
                    input, 1, i * C_in_per_g, (i + 1) * C_in_per_g, 1
                )
                w_g = _lowerings[aten.slice.Tensor](
                    weight, 0, i * C_out_per_g, (i + 1) * C_out_per_g, 1
                )
                bias_g = None
                if bias is not None:
                    bias_g = _lowerings[aten.slice.Tensor](
                        bias, 0, i * C_out_per_g, (i + 1) * C_out_per_g, 1
                    )
                # M-pipeline-2: use identity-safe lookup. `conv_op` is
                # the OpOverload captured by the @register_lowering
                # decorator above, but a future re-binding of the
                # `torch_vulkan::conv2d_with_optional_bias` custom op
                # could invalidate the `_lowerings[conv_op]` dict lookup.
                # The helper iterates by string form, surviving that.
                _conv2d_lower = get_lowering_by_name(_lowerings, conv_op)
                if _conv2d_lower is None:
                    return NotImplemented
                out_g = _conv2d_lower(
                    inp_g, w_g, bias_g, stride, padding, dilation, 1
                )
                if out_g is NotImplemented:
                    return NotImplemented
                groups_out.append(out_g)
            result = _lowerings[aten.cat.default](groups_out, 1)
            # Bias is already handled per-group above; no post-hoc bias add needed.
            return result
        sH = int(stride[0])
        sW = int(stride[-1] if len(stride) > 1 else stride[0])
        pH = int(padding[0])
        pW = int(padding[-1] if len(padding) > 1 else padding[0])
        dH = int(dilation[0])
        dW = int(dilation[-1] if len(dilation) > 1 else dilation[0])

        H_out = (H_in + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W_in + 2 * pW - dW * (kW - 1) - 1) // sW + 1

        # Path 2: Use the dedicated slang_conv2d template for direct
        # conv2d — eliminates the 6-copy dispatch overhead of the
        # im2col+mm decomposition.  The template handles padding natively
        # (no separate constant_pad_nd dispatch needed).
        dev = input.get_device()
        dtype = input.get_dtype()

        out_layout = ir.FixedLayout(
            device=dev,
            dtype=dtype,
            size=[N, C_out, H_out, W_out],
            stride=[C_out * H_out * W_out, H_out * W_out, W_out, 1],
        )

        inputs = [input, weight]
        if bias is not None:
            inputs.append(bias)

        kernel = _VulkanConv2dExternKernel(
            layout=out_layout,
            inputs=inputs,
            stride_arg=(sH, sW),
            padding_arg=(pH, pW),
            dilation_arg=(dH, dW),
        )
        return ir.TensorBox.create(kernel)

    # ═════════════════════════════════════════════════════════════════════
    # A4: Wire aten.convolution.default → _slang_tile_conv2d via
    # _VulkanConv2dExternKernel.  Replaces the default
    # extern_kernels.convolution path (→ eager C++ Vulkan) with the
    # Slang template dispatch for groups==1, non-transposed, 4D input.
    # For groups>1 or transposed conv, returns NotImplemented so the
    # existing fallback paths handle them.
    # ═════════════════════════════════════════════════════════════════════
    @register_lowering(aten.convolution.default, type_promotion_kind=None)
    def _vulkan_aten_convolution(
        input, weight, bias, stride, padding, dilation,
        transposed, output_padding, groups,
    ):
        if bool(transposed):
            return NotImplemented
        if int(groups) != 1:
            return NotImplemented
        if len(input.get_size()) != 4 or len(weight.get_size()) != 4:
            return NotImplemented
        if input.get_device().type != "vulkan":
            return NotImplemented
        # Delegate to the existing conv2d custom-op lowering which
        # creates a _VulkanConv2dExternKernel → _slang_tile_conv2d.
        return _vulkan_conv2d_with_optional_bias(
            input, weight, bias, stride, padding, dilation, groups
        )

    # ═════════════════════════════════════════════════════════════════════
    # M6 Phase 1 — Conv1d compile-path lowering via reshape to Conv2d.
    #
    # Conv1d is lowered by reshaping the 3-D tensors to 4-D (adding a
    # dummy spatial dim of size 1), dispatching to the already-working
    # Conv2d path, then squeezing the dummy dim out of the result.
    #
    # This is simpler than the previous im2col+mm decomposition and
    # automatically inherits backward support from the Conv2d path
    # (CG.M6 conv2d bwd template).
    #
    # Pipeline:
    #   1. input:  [N, C, L]     → unsqueeze(-1) → [N, C, L, 1]
    #   2. weight: [O, I, K]     → unsqueeze(-1) → [O, I, K, 1]
    #   3. stride: (s,)          → (1, s)
    #   4. padding: (p,)         → (0, p)
    #   5. dilation: (d,)        → (1, d)
    #   6. Dispatch to ``torch_vulkan::conv2d_with_optional_bias`` (the
    #      dedicated Slang template for groups==1, eager fallback otherwise).
    #   7. result: [N, O, L_out, 1] → squeeze(-1) → [N, O, L_out]
    #
    # Groups > 1 falls through to the Conv2d path which routes to the
    # eager ``aten.convolution`` extern for groups != 1.
    # ═════════════════════════════════════════════════════════════════════
    if conv1d_op is not None:

        @register_lowering(conv1d_op, type_promotion_kind=None)
        def _vulkan_conv1d_with_optional_bias(
            input, weight, bias, stride, padding, dilation, groups
        ):
            if len(input.get_size()) != 3 or len(weight.get_size()) != 3:
                return NotImplemented

            from torch._inductor.lowering import lowerings as _lowerings

            # Step 1 — reshape input: [N, C, L] → [N, C, L, 1]
            input_4d = _lowerings[aten.unsqueeze.default](input, -1)

            # Step 2 — reshape weight: [C_out, C_in // groups, K] →
            #          [C_out, C_in // groups, K, 1]
            weight_4d = _lowerings[aten.unsqueeze.default](weight, -1)

            # Step 3 — adjust stride / padding / dilation to 2-D.
            # The dummy dim (H) is stride=1, pad=0, dilation=1 — a passthrough.
            s = int(stride[0])
            p = int(padding[0])
            d = int(dilation[0])
            stride_4d = [1, s]
            padding_4d = [0, p]
            dilation_4d = [1, d]

            # Step 4 — dispatch to Conv2d custom-op lowering.
            # M19.5-followup-1: avoid OpOverload-identity drift by looking
            # up by string-form instead of fresh ``torch.ops.…default``
            # (which may produce a different ``OpOverload`` instance than
            # the dict key we registered against).
            conv2d_lower = _get_conv2d_lowering_by_name()
            if conv2d_lower is None:
                return NotImplemented
            result_4d = conv2d_lower(
                input_4d,
                weight_4d,
                bias,
                stride_4d,
                padding_4d,
                dilation_4d,
                int(groups),
            )
            if result_4d is NotImplemented:
                return NotImplemented

            # Step 5 — squeeze the dummy dim back:
            #          [N, C_out, L_out, 1] → [N, C_out, L_out]
            return _lowerings[aten.squeeze.default](result_4d, -1)

    # ═══════════════════════════════════════════════════════════════════
    # T4.12 — Native ``aten.conv1d`` / ``aten.conv1d.padding`` lowerings.
    #
    # When ``F.conv1d`` is called directly under ``torch.compile`` (not
    # via the patched ``nn.Conv1d`` module), the FX graph contains
    # ``aten.conv1d`` (or ``aten.conv1d.padding``).  These lowerings
    # reshape the 3-D tensors to 4-D, then dispatch to
    # ``aten.convolution`` which routes through the existing Conv2d path
    # (``slang_conv2d.slang`` for groups==1, eager fallback otherwise).
    #
    # This is the same reshape strategy as the custom-op lowering above,
    # but targeting ``aten.convolution`` directly so the indirection
    # through ``torch_vulkan::conv2d_with_optional_bias`` isn't needed.
    # Backward is inherited automatically from the Conv2d backward path
    # (CG.M6 conv2d bwd template).
    # ═══════════════════════════════════════════════════════════════════
    def _conv1d_to_conv2d_lowering(
        input, weight, bias, stride, padding, dilation, groups
    ):
        """Reshape Conv1d args → Conv2d, dispatch, squeeze back."""
        if len(input.get_size()) != 3 or len(weight.get_size()) != 3:
            return NotImplemented

        from torch._inductor.lowering import lowerings as _lowerings

        # Step 1 — reshape input:  [N, C, L] → [N, C, L, 1]
        input_4d = _lowerings[aten.unsqueeze.default](input, -1)
        # Step 2 — reshape weight: [C_out, C_in // groups, K] →
        #          [C_out, C_in // groups, K, 1]
        weight_4d = _lowerings[aten.unsqueeze.default](weight, -1)

        # Step 3 — convert stride / padding / dilation to 2-D.
        # The dummy H dim gets stride=1, pad=0, dilation=1 (passthrough).
        if isinstance(stride, int):
            stride_2d = (stride, 1)
        else:
            stride_2d = (int(stride[0]), 1)
        if isinstance(padding, int):
            padding_2d = (padding, 0)
        else:
            padding_2d = (int(padding[0]), 0)
        if isinstance(dilation, int):
            dilation_2d = (dilation, 1)
        else:
            dilation_2d = (int(dilation[0]), 1)

        # Step 4 — dispatch to Conv2d custom-op lowering.
        # M19.5-followup-1: avoid OpOverload-identity drift by looking
        # up by string-form (see ``_get_conv2d_lowering_by_name`` above)
        # instead of re-resolving via ``torch.ops.…default`` which can
        # return a different ``OpOverload`` instance.
        conv2d_lower = _get_conv2d_lowering_by_name()
        if conv2d_lower is None:
            return NotImplemented
        result_4d = conv2d_lower(
            input_4d,
            weight_4d,
            bias,
            stride_2d,
            padding_2d,
            dilation_2d,
            int(groups),
        )
        if result_4d is NotImplemented:
            return NotImplemented

        # Step 5 — add clone to ensure contiguous strides before squeeze.
        # The conv2d extern kernel may return a tensor with strides that
        # don't survive the SqueezeView stride-length assertion in
        # significant_strides_equal (ir.py:470).
        result_4d = _lowerings[aten.clone.default](result_4d)

        # Step 6 — squeeze the dummy dim back:
        #          [N, C_out, L_out, 1] → [N, C_out, L_out]
        return _lowerings[aten.squeeze.default](result_4d, -1)

    @register_lowering(aten.conv1d.default, type_promotion_kind=None)
    def _vulkan_conv1d_default(
        input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
    ):
        return _conv1d_to_conv2d_lowering(
            input, weight, bias, stride, padding, dilation, groups
        )

    @register_lowering(aten.conv1d.padding, type_promotion_kind=None)
    def _vulkan_conv1d_padding(
        input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
    ):
        return _conv1d_to_conv2d_lowering(
            input, weight, bias, stride, padding, dilation, groups
        )

    # ═════════════════════════════════════════════════════════════════════
    # T4.12 / MODEL.1 — Conv3d compile-path lowering.
    #
    # KD==1 case: reshape input [N, C, D, H, W] → [N*D, C, H, W] and
    # apply Conv2d. This is the fast path for per-frame spatial convolutions.
    #
    # KD>1 case: delegate to native slang_conv3d template (MODEL.1) which
    # handles full 3D kernels including arbitrary kernel depth, stride,
    # and dilation along the depth dimension.
    #
    # Both paths support groups==1 and fp32.
    # ═════════════════════════════════════════════════════════════════════

    def _conv3d_to_conv2d_lowering(
        input, weight, bias, stride, padding, dilation, groups
    ):
        """Reshape Conv3d args → Conv2d, dispatch, reshape back.

        KD==1 case: reshape [N,C,D,H,W] → [N*D,C,H,W], apply Conv2d.
        KD>1 case: delegate to native Conv3d template (MODEL.1).
        """
        if len(input.get_size()) != 5 or len(weight.get_size()) != 5:
            return NotImplemented

        from torch._inductor.lowering import lowerings as _lowerings
        from torch_vulkan.inductor.kernel.symbolic import get_static_numel

        # M19.5 — input dims may be SymInt under dynamic-shape compile.
        # Keep raw expressions; sympy propagates through the H/W/D_out
        # arithmetic and ``reshape``'s size list accepts sympy. The
        # ``KD == 1 and dD == 1 and sD == 1`` gate below requires
        # concrete weight/stride/dilation values — kernel shape and
        # stride/padding/dilation are always concrete from the module.
        t1_sizes = input.get_size()
        w_sizes = weight.get_size()
        N = t1_sizes[0]
        C_in = t1_sizes[1]
        D = t1_sizes[2]
        H = t1_sizes[3]
        W = t1_sizes[4]

        C_out = w_sizes[0]
        KD = int(w_sizes[2])
        KH = int(w_sizes[3])
        KW = int(w_sizes[4])

        sD = int(stride[0])
        sH = int(stride[1]) if len(stride) > 1 else sD
        sW = int(stride[2]) if len(stride) > 2 else sH

        pD = int(padding[0])
        pH = int(padding[1]) if len(padding) > 1 else pD
        pW = int(padding[2]) if len(padding) > 2 else pH

        dD = int(dilation[0])
        dH = int(dilation[1]) if len(dilation) > 1 else dD
        dW = int(dilation[2]) if len(dilation) > 2 else dH

        g = int(groups)

        # Compute output depth dim.
        D_out = (D + 2 * pD - dD * (KD - 1) - 1) // sD + 1
        H_out = (H + 2 * pH - dH * (KH - 1) - 1) // sH + 1
        W_out = (W + 2 * pW - dW * (KW - 1) - 1) // sW + 1

        # C_in_per_g / C_out_per_g require concrete channel counts —
        # channels are almost always static, but if a model marks them
        # dynamic we can't compute the slice indices.
        C_in_static = get_static_numel(C_in)
        C_out_static = get_static_numel(C_out)
        if C_in_static is None or C_out_static is None:
            return NotImplemented
        C_in_per_g = C_in_static // g
        C_out_per_g = C_out_static // g
        # M19.5-followup-1: look up the conv2d lowering by op-name string
        # rather than via fresh ``torch.ops.…default`` which can produce
        # a stale ``OpOverload`` identity after
        # ``register_eager_patch_custom_ops()`` re-registration.
        conv2d_lower = _get_conv2d_lowering_by_name()
        if conv2d_lower is None:
            return NotImplemented

        if KD == 1 and dD == 1 and sD == 1:
            # Optimised path: reshape [N,C,D,H,W] → [N*D,C,H,W].
            # Clone after reshape to avoid as_strided view materialization
            # issues (the reshape from [N,C,D,H,W] to [N*D,C,H,W] merges
            # non-adjacent dims N and D, creating a non-contiguous view).
            input_4d = _lowerings[aten.reshape.default](input, [N * D, C_in, H, W])
            input_4d = _lowerings[aten.clone.default](input_4d)
            weight_4d = _lowerings[aten.reshape.default](
                weight, [C_out, C_in_per_g, KH, KW]
            )
            weight_4d = _lowerings[aten.clone.default](weight_4d)
            result_4d = conv2d_lower(
                input_4d, weight_4d, bias, [sH, sW], [pH, pW], [dH, dW], g
            )
            if result_4d is NotImplemented:
                return NotImplemented
            # Clone before reshaping back to avoid SqueezeView stride-length
            # assertion failures (same pattern as Conv1d lowering).
            result_4d = _lowerings[aten.clone.default](result_4d)
            # H_out_actual / W_out_actual are passed straight into
            # the reshape size list — sympy expressions flow through
            # without coercion.
            H_out_actual = result_4d.get_size()[2]
            W_out_actual = result_4d.get_size()[3]
            return _lowerings[aten.reshape.default](
                result_4d, [N, C_out, D_out, H_out_actual, W_out_actual]
            )

        # KD > 1 or dD > 1 or sD > 1 — full 3D kernel support via native
        # Conv3d template (MODEL.1). Delegates to the dedicated
        # slang_conv3d.slang template which handles arbitrary 3D kernels.
        from .conv3d import _vulkan_conv3d_native_lowering

        return _vulkan_conv3d_native_lowering(
            input, weight, bias, stride, padding, dilation, groups
        )

    # Register for aten.conv3d overloads so torch.nn.Conv3d and F.conv3d
    # both route through the reshape-to-Conv2d path.
    @register_lowering(aten.conv3d.default, type_promotion_kind=None)
    def _vulkan_conv3d_default(
        input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
    ):
        return _conv3d_to_conv2d_lowering(
            input, weight, bias, stride, padding, dilation, groups
        )

    @register_lowering(aten.conv3d.padding, type_promotion_kind=None)
    def _vulkan_conv3d_padding(
        input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
    ):
        return _conv3d_to_conv2d_lowering(
            input, weight, bias, stride, padding, dilation, groups
        )

    # M6 Phase 4 — conv_transpose1d/2d/3d via decomposition (split out for
    # the file-size cap; see ``conv_transpose.py``).
    from .conv_transpose import _register_conv_transpose_lowerings

    _register_conv_transpose_lowerings()

    # CODEGEN.3 — conv2d backward lowering (extracted to conv_backward.py
    # for anti-goal #7 compliance).
    # NOTE (anti-goal #3): Registration moved to bwd_lowerings.py.
    # _get_conv_backward_lowering_impl() is called from bwd_lowerings.py.

    @register_lowering(pool_op, type_promotion_kind=None)
    def _vulkan_max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode):
        # Mirrors the canonical Inductor pattern for max_pool2d (see
        # ``_max_pool_with_offsets`` in ``torch/_inductor/lowering.py``):
        # build the windowed access via a custom ``inner_fn`` rather than
        # ``as_strided``, then drop into ``Reduction.create`` with
        # ``reduction_type="max"``. The inner_fn uses
        # ``constant_boundary_condition`` for padded cases — no real
        # ``aten.constant_pad_nd`` materialization is needed.
        from torch._inductor.lowering import constant_boundary_condition

        if bool(ceil_mode):
            return NotImplemented
        if len(input.get_size()) != 4:
            return NotImplemented

        kH = int(kernel_size[0])
        kW = int(kernel_size[-1] if len(kernel_size) > 1 else kernel_size[0])
        sH = int(stride[0]) if stride else kH
        sW = int(stride[-1]) if (stride and len(stride) > 1) else sH
        pH = int(padding[0])
        pW = int(padding[-1]) if len(padding) > 1 else pH
        dH = int(dilation[0])
        dW = int(dilation[-1]) if len(dilation) > 1 else dH

        # M19.5 — input dims (N, H_in, W_in especially) may be SymInt
        # under dynamic-shape compile. Don't ``int()``-coerce — sympy
        # propagates through the H/W_out arithmetic and ``Reduction.create``
        # accepts sympy ranges.
        t1_sizes = input.get_size()
        N = t1_sizes[0]
        C = t1_sizes[1]
        H_in = t1_sizes[2]
        W_in = t1_sizes[3]
        H_out = (H_in + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W_in + 2 * pW - dW * (kW - 1) - 1) // sW + 1

        input.realize_hint()
        dtype = input.get_dtype()
        if pH > 0 or pW > 0 or dH > 1 or dW > 1:
            x_loader = constant_boundary_condition(input, float("-inf"), dim=2)
        else:
            x_loader = input.make_loader()

        stride_t = (sH, sW)
        pad_t = (pH, pW)
        dil_t = (dH, dW)

        def inner_fn(idx, reduction_idx):
            n, c = idx[0], idx[1]
            h_o, w_o = idx[2], idx[3]
            k_h, k_w = reduction_idx[0], reduction_idx[1]
            ih = h_o * stride_t[0] + k_h * dil_t[0] - pad_t[0]
            iw = w_o * stride_t[1] + k_w * dil_t[1] - pad_t[1]
            return x_loader([n, c, ih, iw])

        result = ir.Reduction.create(
            reduction_type="max",
            input_node=input,
            device=input.get_device(),
            dst_dtype=dtype,
            src_dtype=dtype,
            inner_fn=inner_fn,
            ranges=[N, C, H_out, W_out],
            reduction_ranges=[kH, kW],
        )
        if isinstance(result.data.data, ir.Reduction):
            result.realize()
        return result
