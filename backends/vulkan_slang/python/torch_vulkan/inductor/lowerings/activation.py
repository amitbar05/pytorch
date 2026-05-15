"""Activation lowerings -- clamp, clamp_min, clamp_max, pow.Scalar.

Backward lowerings for activation ops have been moved to
``bwd_lowerings.py`` (TR.19 backward consolidation).
"""

from __future__ import annotations

import math

import torch


def _register_pow_scalar_lowering() -> None:
    """Register aten.pow.Scalar -- scalar ** tensor.

    Decomposes ``base ** exponent`` (scalar base, tensor exponent)
    into ``exp(exponent * log(base))``.  This is used by RoPE frequency
    computation and any model that computes ``theta_base ** freqs``.
    """
    from torch._inductor.lowering import lowerings, register_lowering

    @register_lowering(torch.ops.aten.pow.Scalar, type_promotion_kind=None)
    def _pow_scalar(exponent, base):
        return lowerings[torch.ops.aten.exp.default](
            lowerings[torch.ops.aten.mul.Tensor](exponent, math.log(float(base)))
        )


def _register_clamp_lowerings() -> None:
    """Register aten.clamp / clamp_min / clamp_max lowerings.

    Without these, Inductor decomposes aten.clamp(x, min, max) into
    clamp_max(clamp_min(x, min), max) -- two pointwise ops each emitting
    NaN-guard branches via maximum/minimum expression printers.
    """
    from torch._inductor.lowering import register_lowering

    @register_lowering(torch.ops.aten.clamp_min.default, type_promotion_kind=None)
    def _clamp_min(x, min_val):
        return torch.ops.aten.maximum.default(x, min_val)

    @register_lowering(torch.ops.aten.clamp_max.default, type_promotion_kind=None)
    def _clamp_max(x, max_val):
        return torch.ops.aten.minimum.default(x, max_val)

    @register_lowering(torch.ops.aten.clamp.default, type_promotion_kind=None)
    def _clamp(x, min_val=None, max_val=None):
        if min_val is not None:
            x = torch.ops.aten.maximum.default(x, min_val)
        if max_val is not None:
            x = torch.ops.aten.minimum.default(x, max_val)
        return x


def _register_pointwise_math_lowerings() -> None:
    """Register lowerings for common pointwise math ops.

    These are trivially decomposable ops that the upstream Inductor
    doesn't automatically decompose, causing graph breaks on Vulkan.
    """
    from torch._inductor.lowering import lowerings, register_lowering

    # lerp.Scalar: input + weight * (end - input)
    @register_lowering(torch.ops.aten.lerp.Scalar, type_promotion_kind=None)
    def _lerp_scalar(input, end, weight):
        diff = lowerings[torch.ops.aten.sub.Tensor](end, input)
        scaled = lowerings[torch.ops.aten.mul.Tensor](diff, weight)
        return lowerings[torch.ops.aten.add.Tensor](input, scaled)

    # lerp.Tensor: input + weight * (end - input)
    @register_lowering(torch.ops.aten.lerp.Tensor, type_promotion_kind=None)
    def _lerp_tensor(input, end, weight):
        diff = lowerings[torch.ops.aten.sub.Tensor](end, input)
        scaled = lowerings[torch.ops.aten.mul.Tensor](diff, weight)
        return lowerings[torch.ops.aten.add.Tensor](input, scaled)

    # OP.23: Register lowerings for the _out variants of lerp that
    # Inductor's ForeachKernelSchedulerNode generates for each sub-tensor
    # when processing _foreach_lerp.{Scalar,List}.  Without these,
    # the scheduler falls through to eager dispatch which fails because
    # the Vulkan backend doesn't have C++ kernels for *_out ops.
    @register_lowering(torch.ops.aten.lerp.Scalar_out, type_promotion_kind=None)
    def _lerp_scalar_out(input, end, weight, *, out=None):
        diff = lowerings[torch.ops.aten.sub.Tensor](end, input)
        scaled = lowerings[torch.ops.aten.mul.Tensor](diff, weight)
        result = lowerings[torch.ops.aten.add.Tensor](input, scaled)
        if out is not None:
            return lowerings[torch.ops.aten.copy_.default](out, result)
        return result

    @register_lowering(torch.ops.aten.lerp.Tensor_out, type_promotion_kind=None)
    def _lerp_tensor_out(input, end, weight, *, out=None):
        diff = lowerings[torch.ops.aten.sub.Tensor](end, input)
        scaled = lowerings[torch.ops.aten.mul.Tensor](diff, weight)
        result = lowerings[torch.ops.aten.add.Tensor](input, scaled)
        if out is not None:
            return lowerings[torch.ops.aten.copy_.default](out, result)
        return result

    # addcmul: input + value * tensor1 * tensor2
    @register_lowering(torch.ops.aten.addcmul.default, type_promotion_kind=None)
    def _addcmul(input, tensor1, tensor2, value=1):
        # Use tensor1 * float(value) so Python dispatches through
        # TensorBox.__mul__(scalar) rather than float.__mul__(TensorBox).
        return input + tensor1 * tensor2 * float(value)

    # addcdiv: input + value * tensor1 / tensor2
    @register_lowering(torch.ops.aten.addcdiv.default, type_promotion_kind=None)
    def _addcdiv(input, tensor1, tensor2, value=1):
        return input + tensor1 / tensor2 * float(value)

    # rot90: k=1 rotates 90deg CCW = flip(transpose(x, 0, 1), 1)
    # k>1 repeats the operation k times
    @register_lowering(torch.ops.aten.rot90.default, type_promotion_kind=None)
    def _rot90(input, k=1, dims=[0, 1]):
        k = k % 4
        if k == 0:
            return input
        result = input
        for _ in range(k):
            result = torch.ops.aten.flip.default(
                torch.ops.aten.transpose.int(result, dims[0], dims[1]), [dims[1]]
            )
        return result
