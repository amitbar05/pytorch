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
| Conv3d                          | ❌ Not supported   | Falls to ``aten.convolution``|
|                                |                    | extern. 5-D tensor support  |
|                                |                    | would require a new Slang   |
|                                |                    | template + dispatch caller. |
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

Blockers for Conv3d / Transposed conv (T4.12):

- **Conv3d**: 5-D tensor support would require a new Slang
  template + dispatch caller.
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
            """
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
    # T4.12 — Conv3d compile-path lowering via reshape to Conv2d.
    #
    # Conv3d input [N, C, D, H, W] is reshaped to [N*D, C, H, W], and
    # weight [C_out, C_in, KD, KH, KW] is reshaped to
    # [C_out, C_in, KD*KH, KW].  This only preserves the original Conv3d
    # semantics when KD == 1 (no cross-depth coupling).  For KD > 1 the
    # 3-D kernel would couple depth slices incorrectly under a 2-D
    # sliding window, so we return NotImplemented for those cases.
    #
    # This is a pragmatic lowering that handles the common 1×K×K Conv3d
    # pattern (e.g. video models with 2-D spatial conv applied per-frame).
    # Full 3-D template support is deferred to a future milestone.
    # ═════════════════════════════════════════════════════════════════════

    def _conv3d_to_conv2d_lowering(
        input, weight, bias, stride, padding, dilation, groups
    ):
        """Reshape Conv3d args → Conv2d, dispatch, reshape back.

        KD==1 case: reshape [N,C,D,H,W] → [N*D,C,H,W], apply Conv2d.
        KD>1 case: returns NotImplemented (full 3-D template deferred).
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

        # KD > 1 or dD > 1 or sD > 1 — full 3-D kernel support is deferred.
        # The previous per-depth-slice decomposition created many Conv2d
        # dispatches and hit the same aten.convolution out= kwarg issue.
        # A proper 3-D tiled template is planned for a future milestone.
        return NotImplemented

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

    _audit_conv_backward_routing()

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
