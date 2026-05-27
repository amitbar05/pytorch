"""CG.M8 — Lowerings that emit inline ``bwd_diff(fwd_fn)`` Slang.

Instead of routing through the Python custom-op shim, these lowerings
use ``make_pointwise`` to create Pointwise IR nodes whose ``inner_fn``
emits ``bwd_diff(fwd_fn)`` directly into the kernel's compute buffer.

The emitted Slang is split into:
1. Body lines (DifferentialPair + bwd_diff call) → ``kernel.compute``
2. Result expression (dp.getDifferential()) → ``kernel.cse.generate()``

This lets CSE properly track the output variable while the bwd_diff
mechanism is emitted as raw Slang.
"""

from __future__ import annotations


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
    from torch._inductor.virtualized import V

    from torch_vulkan.inductor.bwd_diff_table import BWD_DIFF_TABLE

    entry = BWD_DIFF_TABLE.get(aten_op)
    if entry is None:
        return

    short = aten_op.split(".", 1)[1]
    try:
        target = getattr(aten, short)
    except AttributeError:
        return

    @register_lowering(target, type_promotion_kind=None)
    def _lowering(grad_output, self):
        if not _is_vulkan(grad_output):
            return NotImplemented

        def inner_fn(grad_out_val, x_val):
            from torch._inductor.virtualized import NullKernelHandler

            kernel = V.kernel
            # ``inner_fn`` is invoked twice: once during real codegen (with
            # a live VulkanKernel context) and once from
            # ``Pointwise.inner_fn_opcount`` to count read/write ops with a
            # NullKernelHandler V.kernel. The opcount call only looks at
            # ``ops.X(...)`` reads — it doesn't need the actual bwd_diff
            # body emitted to ``kernel.compute``. Return a constant
            # placeholder so opcount sees a valid op tree.
            if isinstance(kernel, NullKernelHandler):
                return ops.constant(0.0, torch.float32)

            _add_module_import(kernel, entry.module)

            from torch_vulkan.inductor.kernel.bwd_diff_inline import (
                emit_inline_unary_bwd,
            )

            body_lines, result_expr = emit_inline_unary_bwd(
                entry,
                x_var=str(x_val),
                grad_out_var=str(grad_out_val),
                dtype="float",
            )
            kernel.compute.writeline(body_lines)
            return kernel.cse.generate(
                kernel.compute,
                result_expr,
                dtype=torch.float32,
            )

        from torch._inductor.virtualized import ops

        return make_pointwise(inner_fn)(grad_output, self)

    _lowering.__name__ = f"_vulkan_{short}_bwd_diff_inline"


def _make_inline_binary_bwd_diff_lowering(aten_op, register_lowering, aten):
    import torch
    from torch._inductor.lowering import make_pointwise
    from torch._inductor.virtualized import V

    from torch_vulkan.inductor.bwd_diff_table import BWD_DIFF_TABLE

    entry = BWD_DIFF_TABLE.get(aten_op)
    if entry is None:
        return

    short = aten_op.split(".", 1)[1]
    try:
        target = getattr(aten, short)
    except AttributeError:
        return

    @register_lowering(target, type_promotion_kind=None)
    def _lowering(grad_output, self, target_tensor, reduction, *args, **kwargs):
        if not _is_vulkan(self):
            return NotImplemented

        extra_params = {
            "aten.smooth_l1_loss_backward": ["beta"],
            "aten.huber_loss_backward": ["delta"],
        }.get(aten_op, [])
        scalar_args = list(args) + [kwargs.get(p) for p in extra_params if p in kwargs]

        def inner_fn(grad_out_val, pred_val, target_val, *scalars):
            from torch._inductor.virtualized import NullKernelHandler, ops

            kernel = V.kernel
            # See unary case above: ``inner_fn_opcount`` calls this without
            # a real kernel context. Return a constant placeholder for the
            # opcount pass; only emit real bwd_diff body during codegen.
            if isinstance(kernel, NullKernelHandler):
                return ops.constant(0.0, torch.float32)

            _add_module_import(kernel, entry.module)

            from torch_vulkan.inductor.kernel.bwd_diff_inline import (
                emit_inline_binary_bwd,
            )

            body_lines, result_a_expr, _result_b_expr = emit_inline_binary_bwd(
                entry,
                a_var=str(pred_val),
                b_var=str(target_val),
                grad_out_var=str(grad_out_val),
                dtype="float",
            )
            kernel.compute.writeline(body_lines)
            return kernel.cse.generate(
                kernel.compute,
                result_a_expr,
                dtype=torch.float32,
            )

        inputs = [grad_output, self, target_tensor] + scalar_args
        return make_pointwise(inner_fn)(*inputs)

    _lowering.__name__ = f"_vulkan_{short}_bwd_diff_inline"


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
        # M-AG5.1 Tier-2 (2026-05-22): aten.softplus_backward removed.
        # The inline emitter's lowering signature is ``(grad_output, self)``,
        # incompatible with the aten op's ``(grad_output, self, beta,
        # threshold)``. Algebraic lowering in bwd_lowerings.py.
        #
        # M-AG5.1 Tier-3 (2026-05-24): aten.mish_backward removed.
        # slangc v2026.7.1 does not correctly propagate
        # [BackwardDerivative(mish_fast_bwd)] across module import
        # boundaries — inline bwd_diff(mish_fwd) returns all-zero
        # gradients. Algebraic lowering in bwd_lowerings.py.
        "aten.sigmoid_backward",
        "aten.tanh_backward",
        "aten.gelu_backward",
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
