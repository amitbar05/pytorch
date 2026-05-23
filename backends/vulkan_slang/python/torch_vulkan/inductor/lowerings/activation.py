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
    """Register aten.clamp lowering.

    We register aten.clamp (2-arg min+max form) as a composite of
    clamp_min + clamp_max so Inductor sees a single pointwise IR node
    rather than two chained ones.  We do NOT override clamp_min / clamp_max
    themselves — the upstream Inductor lowerings (via register_pointwise) for
    aten.maximum / aten.minimum already handle scalar broadcast correctly.

    aten.clamp is suppressed from Inductor's decomposition table
    (``_suppress_upstream_decomps`` in ``lowerings/__init__.py``) so that
    it reaches this lowering rather than being split by the upstream
    clamp → clamp_min → clamp_max path.
    """
    from torch._inductor.lowering import lowerings, register_lowering

    aten = torch.ops.aten

    @register_lowering(aten.clamp.default, type_promotion_kind=None)
    def _clamp(x, min_val=None, max_val=None):
        if min_val is not None:
            x = lowerings[aten.clamp_min.default](x, min_val)
        if max_val is not None:
            x = lowerings[aten.clamp_max.default](x, max_val)
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

    # M19.R — rot90 correctness fix + dispatch reduction.
    #
    # The previous iterative implementation applied
    # ``flip(transpose(x, d0, d1), [d1])`` k times. That per-iteration
    # step is actually a 270° (= 90° CW) rotation, not a 90° CCW
    # rotation — so the loop computed ``rot90(x, 3*k)`` instead of
    # ``rot90(x, k)``. With ``k mod 4 ∈ {0, 2}`` the bug happens to
    # cancel; with ``k mod 4 ∈ {1, 3}`` the result is the rotation in
    # the opposite direction. Eager-mode tests didn't catch this
    # because ``torch.rot90`` in eager goes through the C++ kernel,
    # not this lowering — only the compile path was wrong. See the
    # 2026-05-21 lowering survey at
    # ``agent_space/lowering_survey_2026_05_21.md`` §2.1.
    #
    # Correct identities (verified against ``torch.rot90`` semantics
    # at ``torch/functional.py::rot90``):
    #   k%4 == 0 → x                                       (0 dispatches)
    #   k%4 == 1 → flip(transpose(x, d0, d1), [d0])        (1 flip + 1 view)
    #   k%4 == 2 → flip(flip(x, [d0]), [d1])               (2 flips, no transpose)
    #   k%4 == 3 → transpose(flip(x, [d0]), d0, d1)        (1 flip + 1 view)
    #
    # Dispatch win: k=3 drops from 3 flips → 1 flip; k=2 drops the
    # doubly-nested transpose-view chain (still 2 flips but the IR is
    # flat). Pointwise fusion already collapses the k=2 chain into one
    # kernel; the value is cleaner IR + correctness fix.
    @register_lowering(torch.ops.aten.rot90.default, type_promotion_kind=None)
    def _rot90(input, k=1, dims=[0, 1]):
        k = int(k) % 4
        if k == 0:
            return input
        d0, d1 = int(dims[0]), int(dims[1])
        if k == 1:
            return torch.ops.aten.flip.default(
                torch.ops.aten.transpose.int(input, d0, d1), [d0]
            )
        if k == 2:
            return torch.ops.aten.flip.default(
                torch.ops.aten.flip.default(input, [d0]), [d1]
            )
        # k == 3
        return torch.ops.aten.transpose.int(
            torch.ops.aten.flip.default(input, [d0]), d0, d1
        )
