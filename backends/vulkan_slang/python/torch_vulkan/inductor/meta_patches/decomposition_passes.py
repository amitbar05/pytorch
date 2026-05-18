"""Decomposition overrides and pre-grad passes.

- Real decomposition replacements for activation backward ops
- Pre-grad relu→where rewrite to avoid threshold_backward meta cascades
- Pre-grad optimizer foreach step pattern fusion
"""

from __future__ import annotations

import torch


def _patch_decompositions() -> None:
    """Override decomposition table for activation backward ops.

    M15.2 audit: 12 entries are (b) — real-computation decompositions that
    exist because PyTorch's default decompositions fail on Vulkan FakeTensors
    (device-mixing: meta saved-input × vulkan grad). Should route through
    autodiff (bwd_diff_table.py) per M12. 10 entries are (a) — shape-only
    decompositions redundant with _OP_IMPLS fake_impls (see M15.2.b).
    """
    import torch
    from torch._decomp import decomposition_table

    aten = torch.ops.aten

    def _hardtanh_bwd(grad_output, self, min_val, max_val):
        mask = (self > min_val) & (self < max_val)
        return grad_output * mask

    def _hardswish_bwd(grad_output, self):
        return torch.where(
            self < -3.0,
            torch.zeros_like(grad_output),
            torch.where(self > 3.0, grad_output, grad_output * (self / 3.0 + 0.5)),
        )

    def _hardsigmoid_bwd(grad_output, self):
        mask = (self > -3.0) & (self < 3.0)
        return grad_output * mask * (1.0 / 6.0)

    def _softplus_bwd(grad_output, self, beta, threshold):
        bx = beta * self
        return torch.where(bx > threshold, grad_output, grad_output * torch.sigmoid(bx))

    def _mish_bwd(grad_output, self):
        sp = torch.nn.functional.softplus(self)
        tsp = torch.tanh(sp)
        return grad_output * (tsp + self * (1.0 - tsp * tsp) * torch.sigmoid(self))

    # C1: Provide REAL decompositions, not shape-only proxies.
    # The previous approach (torch.empty_like for all backwards) caused
    # ReluBackward0 and similar to return [] because the compiled backward
    # never fills these empty tensors. Inductor lowerings handle the actual
    # computation via bwd_diff or primitive decomposition.
    def _relu_fwd_decomp(self):
        # C1 safety net: decompose relu → where(x > 0, x, 0) at the
        # AOT decomposition level so ReluBackward0 never enters the
        # joint trace even if the pre-grad rewrite misses.
        return torch.where(self > 0, self, torch.zeros_like(self))

    def _threshold_bwd(grad_output, self, threshold):
        return torch.where(self > threshold, grad_output, 0.0)

    def _gelu_bwd(grad_output, self, approximate="none"):
        # Use the built-in gelu_backward which inductor can lower
        return torch.ops.aten.gelu_backward(grad_output, self, approximate=approximate)

    def _silu_bwd(grad_output, self):
        sig = torch.sigmoid(self)
        return grad_output * sig * (1.0 + self * (1.0 - sig))

    def _tanh_bwd(grad_output, self):
        return grad_output * (1.0 - self * self)

    def _sigmoid_bwd(grad_output, self):
        return grad_output * self * (1.0 - self)

    def _leaky_relu_bwd(grad_output, self, negative_slope, self_is_result):
        return torch.where(self > 0, grad_output, grad_output * negative_slope)

    def _elu_bwd(
        grad_output, alpha, scale, input_scale, self_or_result, self_is_result
    ):
        return torch.where(
            self_or_result > 0,
            grad_output * scale,
            grad_output * scale * (self_or_result + alpha / input_scale),
        )

    # M18.3 (2026-05-18): torch.empty_like(t) → t.new_empty(t.shape).
    # Same bug class as M17.8.d.2: under AOTAutograd's proxy tracer the
    # symbolic ``aten.empty_like`` call can be recorded into the FX graph
    # and (mis-)classified as shape-only — the joint partitioner then
    # collapses to ``aten.full(shape, 0)`` and produces literal-zero
    # gradients at runtime. ``new_empty(shape)`` materialises through the
    # input tensor's factory, which the partitioner treats as
    # storage-bound rather than shape-derived.
    def _softmax_bwd(grad_output, output, dim, input_dtype):
        return grad_output.new_empty(grad_output.shape)

    def _log_softmax_bwd(grad_output, output, dim, input_dtype):
        return grad_output.new_empty(grad_output.shape)

    def _avg_pool2d_bwd(
        grad_output,
        self,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    ):
        return self.new_empty(self.shape)

    def _max_pool_bwd(
        grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices
    ):
        return self.new_empty(self.shape)

    def _linear_bwd(input, grad_output, weight, output_mask):
        gi = (
            input.new_empty(input.shape)
            if output_mask[0]
            else input.new_empty((0,))
        )
        gw = (
            weight.new_empty(weight.shape)
            if output_mask[1]
            else weight.new_empty((0,))
        )
        gb = (
            grad_output.new_empty((weight.size(0),))
            if output_mask[2]
            else grad_output.new_empty((0,))
        )
        return gi, gw, gb

    def _layer_norm_bwd(
        grad_out,
        input,
        normalized_shape,
        mean,
        rstd,
        weight=None,
        bias=None,
        output_mask=(True, True, True),
    ):
        norm_size = 1
        for s in normalized_shape:
            norm_size *= int(s)
        gi = (
            input.new_empty(input.shape)
            if output_mask[0]
            else input.new_empty((0,))
        )
        gw = (
            grad_out.new_empty((norm_size,))
            if output_mask[1]
            else grad_out.new_empty((0,))
        )
        gb = (
            grad_out.new_empty((norm_size,))
            if output_mask[2]
            else grad_out.new_empty((0,))
        )
        return gi, gw, gb

    def _group_norm_bwd(
        grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask
    ):
        gi = (
            input.new_empty(input.shape)
            if output_mask[0]
            else input.new_empty((0,))
        )
        gw = (
            grad_out.new_empty((int(C),))
            if output_mask[1]
            else grad_out.new_empty((0,))
        )
        gb = (
            grad_out.new_empty((int(C),))
            if output_mask[2]
            else grad_out.new_empty((0,))
        )
        return gi, gw, gb

    def _batch_norm_bwd(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        train,
        eps,
        output_mask,
    ):
        C = input.shape[1]
        gi = (
            input.new_empty(input.shape)
            if output_mask[0]
            else input.new_empty((0,))
        )
        gw = (
            grad_out.new_empty((C,))
            if output_mask[1]
            else grad_out.new_empty((0,))
        )
        gb = (
            grad_out.new_empty((C,))
            if output_mask[2]
            else grad_out.new_empty((0,))
        )
        return gi, gw, gb

    replacements = {
        aten.hardtanh_backward.default: _hardtanh_bwd,
        aten.hardswish_backward.default: _hardswish_bwd,
        aten.hardsigmoid_backward.default: _hardsigmoid_bwd,
        aten.softplus_backward.default: _softplus_bwd,
        aten.mish_backward.default: _mish_bwd,
        aten.relu.default: _relu_fwd_decomp,
        aten.threshold_backward.default: _threshold_bwd,
        # Shape-only backward decompositions (input-saving ops that would
        # mix meta saved-input device with vulkan:0 gradient device)
        aten.gelu_backward.default: _gelu_bwd,
        aten.silu_backward.default: _silu_bwd,
        aten.leaky_relu_backward.default: _leaky_relu_bwd,
        aten.elu_backward.default: _elu_bwd,
        aten.sigmoid_backward.default: _sigmoid_bwd,
        aten.tanh_backward.default: _tanh_bwd,
        aten._softmax_backward_data.default: _softmax_bwd,
        aten._log_softmax_backward_data.default: _log_softmax_bwd,
        aten.avg_pool2d_backward.default: _avg_pool2d_bwd,
        aten.max_pool2d_with_indices_backward.default: _max_pool_bwd,
        aten.linear_backward.default: _linear_bwd,
        aten.native_layer_norm_backward.default: _layer_norm_bwd,
        aten.native_group_norm_backward.default: _group_norm_bwd,
        aten.native_batch_norm_backward.default: _batch_norm_bwd,
    }
    decomposition_table.update(replacements)


def _patch_pre_grad_passes_for_optimizer_foreach() -> None:
    """T4.8 pre-grad pass: detect optimizer-step patterns on the forward
    graph BEFORE AOTAutograd functionalizes in-place mutations.

    M15.2 audit (b): workaround for missing foreach lowerings. Should be
    replaced by proper foreach_add/foreach_mul/etc. lowerings in the
    Inductor backend. See roadmap M14 (op coverage).
    """

    # Dynamo traces ``p.add_(g, alpha=-lr)`` as ``aten.add_.Tensor``.
    # AOTAutograd then functionalizes this into the
    # ``(mul.Tensor -> add.Tensor -> copy_)`` triplet.  If we intercept
    # the pre-grad graph, we can rewrite contiguous per-param add_/mul_/
    # addcdiv_/addcmul_ sequences directly into ``torch_vulkan::foreach_*``
    # custom ops, bypassing the functionalization dance.

    # Vulkan-only: only fires when the graph's tensors originate from a
    # Vulkan device, detected by the same three strategies used in
    # ``_patch_pre_grad_passes_for_relu_rewrite``.
    import torch
    import torch._inductor.compile_fx as _cfx

    if getattr(_cfx, "_vulkan_foreach_rewrite_patched", False):
        return

    _orig = _cfx.run_pre_grad_passes

    def _patched(model_, example_inputs_):
        from ..fx_passes.functional.optimizer import (
            _fuse_optimizer_step_to_foreach,
        )

        # Reuse the relu-rewrite detector (same Vulkan-detection logic).
        # Must import the function here to avoid circular imports.
        try:
            # Quick device check — reuse the same strategies as
            # _patch_pre_grad_passes_for_relu_rewrite.
            inputs = example_inputs_ or ()
            is_vulkan = False
            for t in inputs:
                if not isinstance(t, torch.Tensor):
                    continue
                try:
                    if t.device.type in ("vulkan", "privateuseone"):
                        is_vulkan = True
                        break
                except Exception:
                    pass
                try:
                    fd = getattr(t, "fake_device", None)
                    if fd is not None and fd.type in ("vulkan", "privateuseone"):
                        is_vulkan = True
                        break
                except Exception:
                    pass
            if not is_vulkan and isinstance(model_, torch.fx.GraphModule):
                try:
                    for node in model_.graph.nodes:
                        if node.op != "placeholder":
                            continue
                        val = node.meta.get("val") if hasattr(node, "meta") else None
                        if val is None:
                            continue
                        for v in val if isinstance(val, (list, tuple)) else [val]:
                            if not isinstance(v, torch.Tensor):
                                continue
                            try:
                                if v.device.type in ("vulkan", "privateuseone"):
                                    is_vulkan = True
                                    break
                            except Exception:
                                pass
                            try:
                                fd = getattr(v, "fake_device", None)
                                if fd is not None and fd.type in (
                                    "vulkan",
                                    "privateuseone",
                                ):
                                    is_vulkan = True
                                    break
                            except Exception:
                                pass
                        if is_vulkan:
                            break
                except Exception:
                    pass

            if is_vulkan and isinstance(model_, torch.fx.GraphModule):
                try:
                    _fuse_optimizer_step_to_foreach(model_)
                except Exception as e:
                    import logging

                    logging.getLogger(__name__).warning(
                        "Vulkan pre-grad foreach rewrite failed: %s", e
                    )
        except Exception:
            pass

        return _orig(model_, example_inputs_)

    _cfx.run_pre_grad_passes = _patched
    _cfx._vulkan_foreach_rewrite_patched = True


def _patch_pre_grad_passes_for_relu_rewrite() -> None:
    """C1 fix: rewrite ``aten.relu`` in the pre-grad FX graph BEFORE
    AOTAutograd traces the joint forward+backward.

    M15.2 audit (b): workaround for ReluBackward0/threshold_backward meta
    cascade. Should be replaced by proper relu backward through autodiff
    (bwd_diff_table.py). See roadmap M12 (autodiff).
    """

    # The default ``ReluBackward0`` saves the forward output and computes
    # backward via ``aten.threshold_backward(grad_out, result, 0)``. Under
    # AOTAutograd's joint trace the saved ``result`` propagates as a
    # ``device=meta`` FakeTensor through PF.13's view-op cascade, so the
    # threshold-backward decomposition's ``result > 0`` lands on meta and
    # ``where(meta_cond, vulkan_grad, 0.0)`` collapses to a ``[]``-shape
    # tensor — surfacing as ``"ReluBackward0 returned an invalid gradient
    # at index 0 - got [] but expected shape compatible with [...]"``.

    # Decompose ``relu(x) -> where(x > 0, x, full_like(x, 0))`` at the
    # pre-grad stage so AOTAutograd traces the joint graph against
    # pointwise primitives whose backwards never call threshold_backward
    # nor save a meta-cascaded forward output.

    # The Vulkan-detection check uses three strategies because
    # ``run_pre_grad_passes`` is called from two different contexts:

    # 1. **Direct call** (``compile_fx.py:2883``, AOT path):
    #    ``example_inputs_`` are real tensors — ``t.device.type`` works.
    # 2. **Inside ``aot_autograd()``** (``aot_autograd.py:1108/1134``):
    #    ``example_inputs_`` are ``FakeTensors``.  FakeTensor's ``device``
    #    property returns ``fake_device`` when not inside a kernel
    #    invocation (the usual case during pre-grad passes), but a
    #    defensive check of the ``fake_device`` attribute is included for
    #    the rare kernel-invocation path where ``.device`` returns
    #    ``"meta"``.
    # 3. **Graph placeholder metadata**: inspect ``node.meta['val']``
    #    on placeholder nodes as a last-resort fallback.

    # Vulkan-only: only fires when the graph's tensors originate from a
    # Vulkan device, so non-Vulkan compiles are unaffected.
    import torch
    import torch._inductor.compile_fx as _cfx

    if getattr(_cfx, "_vulkan_relu_rewrite_patched", False):
        return

    _orig = _cfx.run_pre_grad_passes

    def _detect_vulkan(example_inputs_, model_) -> bool:
        """Return True if this compilation involves Vulkan tensors."""
        inputs = example_inputs_ or ()
        # Strategy 1: real tensors (direct-call path in compile_fx.py:2883)
        for t in inputs:
            if not isinstance(t, torch.Tensor):
                continue
            try:
                if t.device.type in ("vulkan", "privateuseone"):
                    return True
            except Exception:
                pass
            # Strategy 2: FakeTensor with fake_device (aot_autograd path)
            # FakeTensor.device returns fake_device when not in kernel
            # invocation, but we check both for robustness.
            try:
                fd = getattr(t, "fake_device", None)
                if fd is not None and fd.type in ("vulkan", "privateuseone"):
                    return True
            except Exception:
                pass
        # Strategy 3: graph placeholder metadata (last-resort fallback)
        if isinstance(model_, torch.fx.GraphModule):
            try:
                for node in model_.graph.nodes:
                    if node.op != "placeholder":
                        continue
                    val = node.meta.get("val") if hasattr(node, "meta") else None
                    if val is None:
                        continue
                    for v in val if isinstance(val, (list, tuple)) else [val]:
                        if not isinstance(v, torch.Tensor):
                            continue
                        try:
                            if v.device.type in ("vulkan", "privateuseone"):
                                return True
                        except Exception:
                            pass
                        try:
                            fd = getattr(v, "fake_device", None)
                            if fd is not None and fd.type in (
                                "vulkan",
                                "privateuseone",
                            ):
                                return True
                        except Exception:
                            pass
            except Exception:
                pass
        return False

    def _patched(model_, example_inputs_):
        from ..fx_passes.post_grad import _replace_relu_with_clamp_min

        if _detect_vulkan(example_inputs_, model_) and isinstance(
            model_, torch.fx.GraphModule
        ):
            try:
                _replace_relu_with_clamp_min(model_)
            except Exception as e:  # pragma: no cover
                import logging

                logging.getLogger(__name__).warning(
                    "Vulkan pre-grad relu rewrite failed: %s", e
                )
        return _orig(model_, example_inputs_)

    _cfx.run_pre_grad_passes = _patched
    _cfx._vulkan_relu_rewrite_patched = True


def _patch_pre_grad_passes_for_conv_gn_relu_fusion() -> None:
    """M17.2 Phase 2 pre-grad pass: fuse conv → group_norm → relu into
    ``torch_vulkan::conv2d_gn_relu_fused``.

    Runs BEFORE AOTAutograd decomposition so ``aten.native_group_norm``
    is still a single node in the graph.  Matches:

        conv2d_with_optional_bias(x, w, b, ...)
          → native_group_norm(y, gn_w, gn_b, ...)
            → relu(z)

    and replaces the chain with a single fused custom-op call.
    """
    import torch
    import torch._inductor.compile_fx as _cfx

    if getattr(_cfx, "_vulkan_conv_gn_relu_fusion_patched", False):
        return

    _orig = _cfx.run_pre_grad_passes

    def _detect_vulkan(example_inputs_, model_) -> bool:
        """Return True if this compilation involves Vulkan tensors."""
        inputs = example_inputs_ or ()
        for t in inputs:
            if not isinstance(t, torch.Tensor):
                continue
            try:
                if t.device.type in ("vulkan", "privateuseone"):
                    return True
            except Exception:
                pass
            try:
                fd = getattr(t, "fake_device", None)
                if fd is not None and fd.type in ("vulkan", "privateuseone"):
                    return True
            except Exception:
                pass
        if isinstance(model_, torch.fx.GraphModule):
            try:
                for node in model_.graph.nodes:
                    if node.op != "placeholder":
                        continue
                    val = node.meta.get("val") if hasattr(node, "meta") else None
                    if val is None:
                        continue
                    for v in val if isinstance(val, (list, tuple)) else [val]:
                        if not isinstance(v, torch.Tensor):
                            continue
                        try:
                            if v.device.type in ("vulkan", "privateuseone"):
                                return True
                        except Exception:
                            pass
                        try:
                            fd = getattr(v, "fake_device", None)
                            if fd is not None and fd.type in (
                                "vulkan",
                                "privateuseone",
                            ):
                                return True
                        except Exception:
                            pass
            except Exception:
                pass
        return False

    def _patched(model_, example_inputs_):
        if _detect_vulkan(example_inputs_, model_) and isinstance(
            model_, torch.fx.GraphModule
        ):
            try:
                _fuse_conv_gn_relu(model_)
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(
                    "Vulkan pre-grad conv→GN→ReLU fusion failed: %s", e
                )
        return _orig(model_, example_inputs_)

    _cfx.run_pre_grad_passes = _patched
    _cfx._vulkan_conv_gn_relu_fusion_patched = True


def _fuse_conv_gn_relu(gm: "torch.fx.GraphModule") -> None:
    """Scan the FX graph for conv → group_norm → relu chains and replace
    with ``torch_vulkan::conv2d_gn_relu_fused``.
    """
    import operator

    import torch

    aten = torch.ops.aten

    # Conv targets we recognize (pre-decomposition, pre-AOTAutograd).
    _CONV_TARGETS = set()
    try:
        _CONV_TARGETS.add(torch.ops.torch_vulkan.conv2d_with_optional_bias.default)
    except AttributeError:
        pass
    try:
        _CONV_TARGETS.add(aten.convolution.default)
    except AttributeError:
        pass

    # Walk the graph looking for relu nodes whose input is a native_group_norm
    # whose input is a conv.
    graph = gm.graph
    changed = True
    while changed:
        changed = False
        for node in list(graph.nodes):
            if node.op != "call_function":
                continue

            # Match: relu
            if node.target != aten.relu.default:
                continue
            relu_node = node

            # M17.2 Phase 2: Match native_group_norm (returns tuple).
            # Dynamo wraps multi-output ops with operator.getitem(gn_call, 0),
            # so look through getitem to find the real native_group_norm call.
            gn_node = None
            for arg in relu_node.args:
                if isinstance(arg, torch.fx.Node) and arg.op == "call_function":
                    if arg.target == aten.native_group_norm.default:
                        gn_node = arg
                        break
                    # Look through operator.getitem(gn_call, idx) wrapper
                    if (
                        arg.target == operator.getitem
                        and len(arg.args) >= 1
                        and isinstance(arg.args[0], torch.fx.Node)
                        and arg.args[0].op == "call_function"
                        and arg.args[0].target == aten.native_group_norm.default
                    ):
                        gn_node = arg.args[0]
                        break
            if gn_node is None:
                continue

            # Extract GN args: input, num_groups, weight, bias, eps
            gn_args = gn_node.args
            if len(gn_args) < 5:
                continue
            gn_input = gn_args[0]
            num_groups = gn_args[1]
            gn_weight = gn_args[2]
            gn_bias = gn_args[3]
            eps = gn_args[4] if len(gn_args) > 4 else 1e-5

            if not isinstance(gn_input, torch.fx.Node):
                continue
            if gn_input.op != "call_function":
                continue

            # Match: conv (conv2d_with_optional_bias or aten.convolution)
            conv_node = gn_input
            if conv_node.target not in _CONV_TARGETS:
                continue

            # Extract conv args
            conv_args = conv_node.args
            if conv_node.target == aten.convolution.default:
                # aten.convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)
                if len(conv_args) < 9:
                    continue
                conv_input = conv_args[0]
                conv_weight = conv_args[1]
                conv_bias = conv_args[2]
                conv_stride = conv_args[3]
                conv_padding = conv_args[4]
                conv_dilation = conv_args[5]
                conv_groups = conv_args[8]
            else:
                # torch_vulkan::conv2d_with_optional_bias(input, weight, bias, stride, padding, dilation, groups)
                if len(conv_args) < 7:
                    continue
                conv_input = conv_args[0]
                conv_weight = conv_args[1]
                conv_bias = conv_args[2]
                conv_stride = conv_args[3]
                conv_padding = conv_args[4]
                conv_dilation = conv_args[5]
                conv_groups = conv_args[6]

            # Build the fused op call
            try:
                fused_op = torch.ops.torch_vulkan.conv2d_gn_relu_fused.default
            except AttributeError:
                continue

            # Ensure list types for stride/padding/dilation
            if not isinstance(conv_stride, (list, tuple)):
                conv_stride = [conv_stride, conv_stride]
            if not isinstance(conv_padding, (list, tuple)):
                conv_padding = [conv_padding, conv_padding]
            if not isinstance(conv_dilation, (list, tuple)):
                conv_dilation = [conv_dilation, conv_dilation]

            with graph.inserting_before(relu_node):
                fused = graph.call_function(
                    fused_op,
                    args=(
                        conv_input,
                        conv_weight,
                        conv_bias,
                        list(conv_stride),
                        list(conv_padding),
                        list(conv_dilation),
                        int(conv_groups),
                        gn_weight,
                        gn_bias,
                        int(num_groups) if num_groups is not None else 1,
                        float(eps) if eps is not None else 1e-5,
                    ),
                )
                fused.meta = dict(relu_node.meta)

            relu_node.replace_all_uses_with(fused)
            graph.erase_node(relu_node)
            graph.erase_node(gn_node)
            graph.erase_node(conv_node)
            graph.lint()
            gm.recompile()
            changed = True
            break  # restart scan after graph mutation
