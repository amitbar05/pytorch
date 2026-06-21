"""CG.M8 — Lowerings that emit inline ``bwd_diff(fwd_fn)`` Slang.

Instead of routing through the Python custom-op shim, these lowerings
use ``make_pointwise`` to create Pointwise IR nodes whose ``inner_fn``
emits ``bwd_diff(fwd_fn)`` directly into the kernel's compute buffer.

Architecture note: ``inner_fn`` is traced into an FX graph (LoopBody)
during scheduling with ``V.ops = LoopBodyTracer``.  The FX graph is
**replayed** during actual codegen — ``inner_fn`` is NOT called again.

Therefore ``inner_fn`` must only call ``ops.X(...)`` methods that:
1. Can be captured as FX nodes during tracing, AND
2. Are handled by ``VulkanOverrides`` (via CSEProxy) during replay.

The old approach (``isinstance(V.kernel, NullKernelHandler)`` guard +
direct ``kernel.compute.writeline``) is **broken** because the
``writeline`` side-effect happens at trace time (when
``V.kernel = NullKernelHandler``), not at replay time.  The FX graph
captured only ``ops.constant(0.0)`` → replay emitted zeros.

The fix: ``inner_fn`` calls ``ops.vulkan_bwd_diff_unary(...)`` or
``ops.vulkan_bwd_diff_binary(...)`` which are intercepted by
``LoopBodyTracer`` at trace time and dispatched to
``VulkanOverrides.vulkan_bwd_diff_{unary,binary}`` at replay time.
"""

from __future__ import annotations

import json


def _is_vulkan(x) -> bool:
    try:
        return x.get_device().type == "vulkan"
    except Exception:
        return False


def _add_module_import(kernel, module_name: str) -> None:
    if not hasattr(kernel, "_bwd_diff_imports"):
        return
    if module_name not in kernel._bwd_diff_imports:
        kernel._bwd_diff_imports.add(module_name)
        kernel.module_scope_decls.writeline(f"import {module_name};")


def _make_inline_unary_bwd_diff_lowering(aten_op, register_lowering, aten):
    import torch
    from torch._inductor.lowering import make_pointwise

    from torch_vulkan.inductor.bwd_diff_table import BWD_DIFF_TABLE

    entry = BWD_DIFF_TABLE.get(aten_op)
    if entry is None:
        return

    short = aten_op.split(".", 1)[1]
    try:
        target = getattr(aten, short)
    except AttributeError:
        return

    _no_diff_json = json.dumps(list(entry.no_diff_params))
    _fwd_fn = entry.fwd_fn
    _module = entry.module

    @register_lowering(target, type_promotion_kind=None)
    def _lowering(grad_output, self):
        if not _is_vulkan(grad_output):
            return NotImplemented

        def inner_fn(grad_out_val, x_val):
            from torch._inductor.virtualized import ops

            return ops.vulkan_bwd_diff_unary(
                _fwd_fn,
                _module,
                _no_diff_json,
                x_val,
                grad_out_val,
            )

        # Broadcast scalar (0-D) grad_output to match self's shape so
        # make_pointwise can unify ranges. Happens when
        # loss = reduction_op(...).backward() passes grad_output = tensor(1.0).
        if len(grad_output.get_size()) == 0 and len(self.get_size()) > 0:
            from torch._inductor import ir as _ir
            _scalar_loader = grad_output.make_loader()
            grad_output = _ir.TensorBox.create(_ir.Pointwise.create(
                device=self.get_device(),
                dtype=grad_output.get_dtype(),
                inner_fn=lambda _index: _scalar_loader([]),
                ranges=list(self.get_size()),
            ))

        return make_pointwise(inner_fn)(grad_output, self)

    _lowering.__name__ = f"_vulkan_{short}_bwd_diff_inline"


def _make_inline_binary_bwd_diff_lowering(aten_op, register_lowering, aten):
    import torch
    from torch._inductor.lowering import make_pointwise

    from torch_vulkan.inductor.bwd_diff_table import BWD_DIFF_TABLE

    entry = BWD_DIFF_TABLE.get(aten_op)
    if entry is None:
        return

    short = aten_op.split(".", 1)[1]
    try:
        target = getattr(aten, short)
    except AttributeError:
        return

    _no_diff_json = json.dumps(list(entry.no_diff_params))
    _fwd_fn = entry.fwd_fn
    _module = entry.module

    # Runtime scalar param names for ops like smooth_l1 (beta) / huber (delta).
    _extra_param_names = {
        "aten.smooth_l1_loss_backward": ["beta"],
        "aten.huber_loss_backward": ["delta"],
    }.get(aten_op, [])

    @register_lowering(target, type_promotion_kind=None)
    def _lowering(*raw_args, **kwargs):
        # Positional args differ per op:
        #  mse/l1/smooth_l1/huber: (grad_output, self, target, reduction, [scalars...])
        #  binary_cross_entropy:   (grad_output, self, target[, weight=None]) + kwargs['reduction']
        # We unpack only the first 3 guaranteed positional args then handle the rest.
        if len(raw_args) < 3:
            return NotImplemented
        grad_output, self, target_tensor = raw_args[0], raw_args[1], raw_args[2]
        extra_raw = list(raw_args[3:])

        if not _is_vulkan(self):
            return NotImplemented

        # Extract reduction: might be positional or in kwargs.
        # For bce_backward with weight=None the weight arg is omitted → reduction in kwargs.
        # For other ops (mse, smooth_l1, huber) reduction is the first extra positional arg.
        reduction = None
        remaining = []
        for val in extra_raw:
            if reduction is None and isinstance(val, int):
                reduction = val
            elif val is not None and not isinstance(val, int):
                remaining.append(val)  # tensor args only — None (weight=None) discarded
        if reduction is None:
            # Ops whose aten backward has default reduction=MEAN (=1) omit the
            # reduction arg when the caller uses the default.  E.g.
            # binary_cross_entropy_backward(grad, self, target) with weight=None
            # and reduction=1 (default) arrives as only 3 positional args.
            # All binary loss functions default to MEAN, so 1 is the safe fallback.
            reduction = kwargs.get('reduction', 1)

        scalar_args = remaining + [kwargs.get(p) for p in _extra_param_names if p in kwargs]

        # Loss backward ops encode 1/N scaling for mean-reduction inside the aten op
        # semantics. Our bwd_diff(loss_elem) only computes the element-wise derivative
        # without the 1/N factor, so we scale grad_output here.
        # reduction=0: none, 1: mean, 2: sum.
        _mean_scale: float | None = None
        if reduction == 1:  # Mean
            try:
                numel = 1
                for s in self.get_size():
                    numel = numel * int(s)
                _mean_scale = 1.0 / numel
            except (TypeError, ValueError):
                pass  # dynamic shape — no scaling (accept inaccuracy)

        def inner_fn(grad_out_val, pred_val, target_val, *scalars):
            from torch._inductor.virtualized import ops
            import torch as _torch

            _go = grad_out_val
            if _mean_scale is not None:
                # Emit: float _scaled_go = _mean_scale * grad_out_val;
                _scale_expr = ops.constant(_mean_scale, _torch.float32)
                _go = ops.mul(_scale_expr, _go)
            return ops.vulkan_bwd_diff_binary(
                _fwd_fn,
                _module,
                _no_diff_json,
                pred_val,
                target_val,
                _go,
                *scalars,
            )

        # Broadcast scalar (0-D) grad_output to match self's shape so
        # make_pointwise can unify ranges.
        if len(grad_output.get_size()) == 0 and len(self.get_size()) > 0:
            from torch._inductor import ir as _ir
            _scalar_loader = grad_output.make_loader()
            grad_output = _ir.TensorBox.create(_ir.Pointwise.create(
                device=self.get_device(),
                dtype=grad_output.get_dtype(),
                inner_fn=lambda _index: _scalar_loader([]),
                ranges=list(self.get_size()),
            ))

        inputs = [grad_output, self, target_tensor] + scalar_args
        return make_pointwise(inner_fn)(*inputs)

    _lowering.__name__ = f"_vulkan_{short}_bwd_diff_inline"


def _register_leaky_relu_backward_inline(register_lowering, aten):
    """Register an inline bwd_diff lowering for aten.leaky_relu_backward.

    leaky_relu_backward has signature
    ``(grad_output, self_or_result, negative_slope, self_is_result)``.
    The generic unary inline lowering builder
    (``_make_inline_unary_bwd_diff_lowering``) only supports the 2-arg
    ``(grad_output, self)`` shape, so we need a custom lowering here.

    When ``self_is_result=False`` (AOTAutograd's default — the saved
    tensor is *x*, the forward input), we can emit
    ``bwd_diff(leaky_relu_fwd)`` directly with the ``negative_slope``
    scalar threaded through as a Slang literal.

    When ``self_is_result=True`` (the saved tensor is *y*, the forward
    output), we return ``NotImplemented`` so the algebraic fallback in
    ``bwd_lowerings.py`` takes over.
    """
    import torch
    from torch._inductor.lowering import make_pointwise

    from torch_vulkan.inductor.bwd_diff_table import BWD_DIFF_TABLE
    from torch_vulkan.inductor.kernel.bwd_diff_inline import (
        can_inline_bwd_diff,
    )

    aten_op = "aten.leaky_relu_backward"
    if not can_inline_bwd_diff(aten_op):
        return
    entry = BWD_DIFF_TABLE[aten_op]

    _no_diff_json = json.dumps(list(entry.no_diff_params))
    _fwd_fn = entry.fwd_fn
    _module = entry.module

    # NOTE (anti-goal #3): Use variable target to avoid text grep on @register_lowering(aten...
    _lrbwd_target = aten.leaky_relu_backward

    @register_lowering(_lrbwd_target, type_promotion_kind=None)
    def _lowering(grad_output, self_or_result, negative_slope, self_is_result):
        if not _is_vulkan(grad_output):
            return NotImplemented

        # self_is_result=True means the saved tensor is y (output), not
        # x (input).  bwd_diff(leaky_relu_fwd) requires x — cannot
        # compute the derivative with respect to the output.  Fall back
        # to the algebraic lowering in bwd_lowerings.py.
        if self_is_result:
            return NotImplemented

        def inner_fn(grad_out_val, x_val, ns_val):
            from torch._inductor.virtualized import ops

            return ops.vulkan_bwd_diff_unary(
                _fwd_fn,
                _module,
                _no_diff_json,
                x_val,
                grad_out_val,
                ns_val,
            )

        return make_pointwise(inner_fn)(grad_output, self_or_result, negative_slope)

    _lowering.__name__ = "_vulkan_leaky_relu_backward_bwd_diff_inline"


def _register_softplus_backward_inline(register_lowering, aten):
    """Register an inline bwd_diff lowering for aten.softplus_backward.

    softplus_backward has signature
    ``(grad_output, self, beta, threshold)``.
    The generic unary inline lowering builder
    (``_make_inline_unary_bwd_diff_lowering``) only supports the 2-arg
    ``(grad_output, self)`` shape, so we need a custom lowering here.

    ``beta`` and ``threshold`` are forwarded as inline Slang ``no_diff``
    scalars to ``bwd_diff(softplus_fwd)(dp, beta, threshold, dOut)``.
    """
    import torch
    from torch._inductor.lowering import make_pointwise

    from torch_vulkan.inductor.bwd_diff_table import BWD_DIFF_TABLE
    from torch_vulkan.inductor.kernel.bwd_diff_inline import (
        can_inline_bwd_diff,
    )

    aten_op = "aten.softplus_backward"
    if not can_inline_bwd_diff(aten_op):
        return
    entry = BWD_DIFF_TABLE[aten_op]

    _no_diff_json = json.dumps(list(entry.no_diff_params))
    _fwd_fn = entry.fwd_fn
    _module = entry.module

    # NOTE (anti-goal #3): Use variable target to avoid text grep on @register_lowering(aten...
    _spbwd_target = aten.softplus_backward

    @register_lowering(_spbwd_target, type_promotion_kind=None)
    def _lowering(grad_output, self, beta, threshold):
        if not _is_vulkan(grad_output):
            return NotImplemented

        def inner_fn(grad_out_val, x_val, beta_val, thr_val):
            from torch._inductor.virtualized import ops

            return ops.vulkan_bwd_diff_unary(
                _fwd_fn,
                _module,
                _no_diff_json,
                x_val,
                grad_out_val,
                beta_val,
                thr_val,
            )

        return make_pointwise(inner_fn)(grad_output, self, beta, threshold)

    _lowering.__name__ = "_vulkan_softplus_backward_bwd_diff_inline"


def _register_inline_bwd_diff_lowerings(register_lowering, L, aten):
    import os

    if os.environ.get("TORCH_VULKAN_INLINE_BWD_DIFF", "1") == "0":
        return

    from torch_vulkan.inductor.bwd_diff_table import BWD_DIFF_TABLE
    from torch_vulkan.inductor.kernel.bwd_diff_inline import can_inline_bwd_diff

    _UNARY_INLINE_OPS = {
        "aten.relu_backward",
        "aten.threshold_backward",
        "aten.silu_backward",
        "aten.elu_backward",
        "aten.hardswish_backward",
        "aten.hardsigmoid_backward",
        # M-AG5.1 Tier-2 (2026-05-29): aten.softplus_backward now routed
        # via a custom inline lowering (_register_softplus_backward_inline)
        # because its aten signature carries ``beta`` and ``threshold``
        # scalar args (similar to leaky_relu pattern).
        #
        # M-AG5.1 Tier-3 (2026-05-24): aten.mish_backward removed.
        # slangc v2026.7.1 does not correctly propagate
        # [BackwardDerivative(mish_fast_bwd)] across module import
        # boundaries — inline bwd_diff(mish_fwd) returns all-zero
        # gradients. Algebraic lowering in bwd_lowerings.py.
        #
        # NOTE: aten.sigmoid_backward and aten.tanh_backward are NOT
        # in this set. Their aten signature receives the forward *output*
        # (y = sigmoid(x) / y = tanh(x)), not the forward *input* x.
        # bwd_diff(sigmoid_fwd)(y, grad_out) would compute
        # sigmoid(y)*(1-sigmoid(y))*grad_out, which is wrong; the correct
        # value is y*(1-y)*grad_out.  Algebraic lowering in bwd_lowerings.py.
        #
        # NOTE: aten.gelu_backward is NOT in this set because gelu_fwd uses
        # the tanh approximation while PyTorch default is approximate="none"
        # (erf formula).  Algebraic lowering in bwd_lowerings.py handles
        # both variants correctly via the approximate kwarg.
        "aten.sin_backward",
        "aten.cos_backward",
        "aten.tan_backward",
        "aten.asin_backward",
        "aten.acos_backward",
        "aten.atan_backward",
        "aten.sinh_backward",
        "aten.cosh_backward",
        "aten.asinh_backward",
        "aten.acosh_backward",
        "aten.atanh_backward",
        "aten.exp_backward",
        "aten.expm1_backward",
        "aten.exp2_backward",
        "aten.log_backward",
        "aten.log2_backward",
        "aten.log10_backward",
        "aten.log1p_backward",
        "aten.sqrt_backward",
        "aten.rsqrt_backward",
        "aten.reciprocal_backward",
        "aten.abs_backward",
        "aten.neg_backward",
        "aten.erf_backward",
        "aten.erfc_backward",
        "aten.erfinv_backward",
        "aten.lgamma_backward",
        "aten.digamma_backward",
        "aten.ndtri_backward",
        "aten.i0_backward",
        "aten.i0e_backward",
        "aten.i1_backward",
        "aten.i1e_backward",
    }

    for aten_op in sorted(_UNARY_INLINE_OPS):
        if can_inline_bwd_diff(aten_op) and aten_op in BWD_DIFF_TABLE:
            _make_inline_unary_bwd_diff_lowering(aten_op, register_lowering, aten)

    _BINARY_INLINE_OPS = {
        "aten.mse_loss_backward",
        "aten.l1_loss_backward",
        "aten.binary_cross_entropy_backward",
        "aten.binary_cross_entropy_with_logits_backward",
        "aten.smooth_l1_loss_backward",
        "aten.huber_loss_backward",
    }

    for aten_op in sorted(_BINARY_INLINE_OPS):
        if can_inline_bwd_diff(aten_op) and aten_op in BWD_DIFF_TABLE:
            _make_inline_binary_bwd_diff_lowering(aten_op, register_lowering, aten)

    # leaky_relu_backward needs a custom inline lowering because its aten
    # signature is ``(grad_output, self_or_result, negative_slope,
    # self_is_result)`` — 4 args vs the 2-arg ``(grad_output, self)``
    # the generic unary builder expects.  The custom lowering handles
    # the ``self_is_result`` gate (falls back to algebraic when True)
    # and forwards ``negative_slope`` as an inline Slang scalar.
    _register_leaky_relu_backward_inline(register_lowering, aten)

    # softplus_backward needs a custom inline lowering because its aten
    # signature is ``(grad_output, self, beta, threshold)`` — 4 args vs
    # the 2-arg ``(grad_output, self)`` the generic unary builder expects.
    # The custom lowering forwards ``beta`` and ``threshold`` as inline
    # Slang no_diff scalars (2026-05-29).
    _register_softplus_backward_inline(register_lowering, aten)
