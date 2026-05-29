"""Parametrized conv2d+epilogue custom-op factory.

Generalises ``conv_relu.py`` to support all 9 IDifferentiable epilogue
activations defined in ``shaders/lib/pointwise.slang``.  The Slang conv2d
template is already generic over ``<Epilogue : IDifferentiable>``; this
module creates one ``torch_vulkan::conv2d_{name}_fused`` custom_op per
epilogue activation, each with:

  * Forward: fused conv2d + activation in a single Slang dispatch.
  * Fake impl: shape-only meta computation.
  * Autograd: epilogue-specific grad transform → conv2d backward dispatch.

Two categories of backward:
  - **Output-sufficient** (ReLU, Sigmoid, Tanh, ELU, LeakyReLU): the grad
    mask depends only on the saved forward *output*.
  - **Input-required** (GELU, SiLU, HardSigmoid, HardSwish): the grad
    mask depends on the saved forward *input* (pre-activation x).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

_FORCE_RELOAD = os.environ.get("TORCH_VULKAN_FORCE_CUSTOM_OP_RELOAD") == "1"

# Module-level Tensor alias so nested closures can use it in annotations.
Tensor = torch.Tensor


@dataclass(frozen=True)
class EpilogueSpec:
    """Specification for a conv2d + activation fused custom op."""

    slang_struct: str
    aten_op_name: str
    needs_input_for_bwd: bool
    grad_transform: Callable[..., torch.Tensor]
    extra_params: dict[str, Any] = field(default_factory=dict)


def _grad_relu(go: torch.Tensor, y: torch.Tensor, x: torch.Tensor | None) -> torch.Tensor:
    return go * (y > 0).to(dtype=go.dtype)


def _grad_sigmoid(go: torch.Tensor, y: torch.Tensor, x: torch.Tensor | None) -> torch.Tensor:
    return go * y * (1.0 - y)


def _grad_tanh(go: torch.Tensor, y: torch.Tensor, x: torch.Tensor | None) -> torch.Tensor:
    return go * (1.0 - y * y)


def _grad_elu(go: torch.Tensor, y: torch.Tensor, x: torch.Tensor | None) -> torch.Tensor:
    return go * torch.where(y > 0, torch.ones_like(y), y + 1.0)


def _grad_leaky_relu(
    go: torch.Tensor, y: torch.Tensor, x: torch.Tensor | None, *, alpha: float = 0.01
) -> torch.Tensor:
    return go * torch.where(
        y > 0, torch.ones_like(y), torch.full_like(y, alpha)
    )


def _grad_gelu(go: torch.Tensor, y: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
    # GELU backward: grad * sigmoid(1.702 * x) * (1 + 1.702 * x * (1 - sigmoid(1.702 * x)))
    # Using tanh approximation: grad * 0.5 * (1 + tanh(c * (x + 0.044715*x^3)))
    #   where c = sqrt(2/pi) ≈ 0.7978845608
    cdf = 0.5 * (1.0 + torch.erf(x / 1.4142135624))
    pdf = torch.exp(-0.5 * x * x) / 2.5066282746  # 1/sqrt(2*pi)
    return go * (cdf + x * pdf)


def _grad_silu(go: torch.Tensor, y: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
    sig = torch.sigmoid(x)
    return go * (sig + x * sig * (1.0 - sig))


def _grad_hardsigmoid(
    go: torch.Tensor, y: torch.Tensor | None, x: torch.Tensor
) -> torch.Tensor:
    return go * torch.where(
        (x > -3.0) & (x < 3.0),
        torch.full_like(x, 1.0 / 6.0),
        torch.zeros_like(x),
    )


def _grad_hardswish(
    go: torch.Tensor, y: torch.Tensor | None, x: torch.Tensor
) -> torch.Tensor:
    return go * torch.where(
        x >= 3.0,
        torch.ones_like(x),
        torch.where(x <= -3.0, torch.zeros_like(x), (2.0 * x + 3.0) / 6.0),
    )


EPILOGUE_SPECS: dict[str, EpilogueSpec] = {
    "OpReLU": EpilogueSpec(
        slang_struct="OpReLU",
        aten_op_name="conv2d_relu_fused",
        needs_input_for_bwd=False,
        grad_transform=_grad_relu,
    ),
    "OpSigmoid": EpilogueSpec(
        slang_struct="OpSigmoid",
        aten_op_name="conv2d_sigmoid_fused",
        needs_input_for_bwd=False,
        grad_transform=_grad_sigmoid,
    ),
    "OpTanh": EpilogueSpec(
        slang_struct="OpTanh",
        aten_op_name="conv2d_tanh_fused",
        needs_input_for_bwd=False,
        grad_transform=_grad_tanh,
    ),
    "OpELU": EpilogueSpec(
        slang_struct="OpELU",
        aten_op_name="conv2d_elu_fused",
        needs_input_for_bwd=False,
        grad_transform=_grad_elu,
    ),
    "OpLeakyReLU": EpilogueSpec(
        slang_struct="OpLeakyReLU",
        aten_op_name="conv2d_leaky_relu_fused",
        needs_input_for_bwd=False,
        grad_transform=_grad_leaky_relu,
        extra_params={"alpha": 0.01},
    ),
    "OpGELU": EpilogueSpec(
        slang_struct="OpGELU",
        aten_op_name="conv2d_gelu_fused",
        needs_input_for_bwd=True,
        grad_transform=_grad_gelu,
    ),
    "OpSiLU": EpilogueSpec(
        slang_struct="OpSiLU",
        aten_op_name="conv2d_silu_fused",
        needs_input_for_bwd=True,
        grad_transform=_grad_silu,
    ),
    "OpHardSigmoid": EpilogueSpec(
        slang_struct="OpHardSigmoid",
        aten_op_name="conv2d_hardsigmoid_fused",
        needs_input_for_bwd=True,
        grad_transform=_grad_hardsigmoid,
    ),
    "OpHardSwish": EpilogueSpec(
        slang_struct="OpHardSwish",
        aten_op_name="conv2d_hardswish_fused",
        needs_input_for_bwd=True,
        grad_transform=_grad_hardswish,
    ),
}

# ── Registry of already-registered ops ──────────────────────────────────

_registered: dict[str, object] = {}


def _ensure_conv2d_epilogue_op_registered(spec: EpilogueSpec) -> "object":
    """Register a ``torch_vulkan::conv2d_{name}_fused`` custom op.

    Generic factory parametrised by ``EpilogueSpec``.  Each op fuses
    ``conv2d + activation`` in a single Slang dispatch and provides
    autograd support with the epilogue-specific gradient transform.
    """
    op_name = f"torch_vulkan::{spec.aten_op_name}"

    # Idempotent guard.
    existing = _registered.get(spec.aten_op_name)
    if existing is not None and not _FORCE_RELOAD:
        return existing
    existing_op = getattr(torch.ops.torch_vulkan, spec.aten_op_name, None)
    if existing_op is not None and hasattr(existing_op, "default"):
        _registered[spec.aten_op_name] = existing_op.default
        return existing_op.default

    Tensor = torch.Tensor

    def _conv2d_epi_impl(
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        stride: list[int],
        padding: list[int],
        dilation: list[int],
        groups: int,
    ) -> Tensor:
        if input.device.type != "vulkan" or input.dtype != torch.float32:
            y = torch.ops.aten.convolution.default(
                input,
                weight.to(dtype=input.dtype),
                bias
                if bias is not None
                else torch.zeros(
                    weight.shape[0], device=input.device, dtype=input.dtype
                ),
                list(stride),
                list(padding),
                list(dilation),
                False,
                [0, 0],
                int(groups),
            )
            # CPU fallback: apply the activation via aten.
            from torch._inductor.decomposition import decompositions as _decomps

            aten = torch.ops.aten
            # Map Slang struct to aten activation op.
            _slang_to_aten = {
                "OpReLU": aten.relu.default,
                "OpGELU": aten.gelu.default,
                "OpSiLU": aten.silu.default,
                "OpSigmoid": aten.sigmoid.default,
                "OpTanh": aten.tanh.default,
                "OpHardSigmoid": aten.hardsigmoid.default,
                "OpHardSwish": aten.hardswish.default,
                "OpLeakyReLU": aten.leaky_relu.default,
                "OpELU": aten.elu.default,
            }
            act_op = _slang_to_aten.get(spec.slang_struct)
            if act_op is not None:
                return act_op(y)
            return y

        from torch_vulkan.inductor.templates.caller import _slang_tile_conv2d

        if groups != 1:
            C_in_g = input.shape[1] // groups
            C_out_g = weight.shape[0] // groups
            parts = []
            for g in range(groups):
                inp_s = input[:, g * C_in_g : (g + 1) * C_in_g, :, :].contiguous()
                w_s = weight[g * C_out_g : (g + 1) * C_out_g, :, :, :].contiguous()
                b_s = (
                    bias[g * C_out_g : (g + 1) * C_out_g]
                    if bias is not None
                    else None
                )
                N_s = inp_s.shape[0]
                C_out_s = w_s.shape[0]
                H_in_s, W_in_s = inp_s.shape[2], inp_s.shape[3]
                sH, sW = stride[0], stride[-1]
                pH, pW = padding[0], padding[-1]
                dH, dW = dilation[0], dilation[-1]
                kH, kW = weight.shape[2], weight.shape[3]
                H_out_s = (H_in_s + 2 * pH - dH * (kH - 1) - 1) // sH + 1
                W_out_s = (W_in_s + 2 * pW - dW * (kW - 1) - 1) // sW + 1
                out_s = torch.empty(
                    (N_s, C_out_s, H_out_s, W_out_s),
                    device=input.device,
                    dtype=input.dtype,
                )
                _slang_tile_conv2d(
                    inp_s, w_s, out_s,
                    stride=(sH, sW), padding=(pH, pW), dilation=(dH, dW),
                    groups=1, bias=b_s, epilogue=spec.slang_struct,
                )
                parts.append(out_s)
            return torch.cat(parts, dim=1)

        N = input.shape[0]
        C_out = weight.shape[0]
        H_in, W_in = input.shape[2], input.shape[3]
        sH, sW = stride[0], stride[-1]
        pH, pW = padding[0], padding[-1]
        dH, dW = dilation[0], dilation[-1]
        kH, kW = weight.shape[2], weight.shape[3]
        H_out = (H_in + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W_in + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        out = torch.empty(
            (N, C_out, H_out, W_out),
            device=input.device,
            dtype=input.dtype,
        )
        _slang_tile_conv2d(
            input if input.is_contiguous() else input.contiguous(),
            weight if weight.is_contiguous() else weight.contiguous(),
            out,
            stride=(sH, sW), padding=(pH, pW), dilation=(dH, dW),
            groups=1, bias=bias, epilogue=spec.slang_struct,
        )
        return out

    _conv2d_epi_impl.__annotations__ = {
        "input": Tensor,
        "weight": Tensor,
        "bias": Tensor | None,
        "stride": list[int],
        "padding": list[int],
        "dilation": list[int],
        "groups": int,
        "return": Tensor,
    }
    op = torch.library.custom_op(op_name, mutates_args=())(_conv2d_epi_impl)

    def _conv2d_epi_fake(input, weight, bias, stride, padding, dilation, groups):
        N = input.shape[0]
        C_out = weight.shape[0]
        H_in, W_in = input.shape[-2], input.shape[-1]
        K_h, K_w = weight.shape[-2], weight.shape[-1]
        s_h, s_w = (stride[0], stride[-1])
        p_h, p_w = (padding[0], padding[-1])
        d_h, d_w = (dilation[0], dilation[-1])
        H_out = (H_in + 2 * p_h - d_h * (K_h - 1) - 1) // s_h + 1
        W_out = (W_in + 2 * p_w - d_w * (K_w - 1) - 1) // s_w + 1
        return input.new_empty((N, C_out, H_out, W_out))

    op.register_fake(_conv2d_epi_fake)

    # ── Autograd ─────────────────────────────────────────────────────
    def _setup_context(ctx, inputs, output):
        inp, w, b, stride, padding, dilation, groups = inputs
        if spec.needs_input_for_bwd:
            # Save input (pre-activation) for grad transform.
            ctx.save_for_backward(
                inp, w, b if b is not None else None,
                torch.empty(0, device=inp.device),  # placeholder for output slot
            )
        else:
            # Save output for grad transform.
            ctx.save_for_backward(inp, w, b if b is not None else None, output)
        ctx.stride = list(stride)
        ctx.padding = list(padding)
        ctx.dilation = list(dilation)
        ctx.groups = int(groups)
        ctx.has_bias = b is not None

    def _backward(ctx, grad_output):
        inp, w, saved_b, saved_extra = ctx.saved_tensors
        has_bias = ctx.has_bias

        # Compute epilogue-specific grad transform.
        if spec.needs_input_for_bwd:
            grad_output = spec.grad_transform(grad_output, None, inp)
        else:
            grad_output = spec.grad_transform(grad_output, saved_extra, None)

        from ._common import _has_real_vulkan_storage

        use_slang_bwd = (
            ctx.groups == 1
            and inp.device.type == "vulkan"
            and inp.dtype == torch.float32
            and _has_real_vulkan_storage(inp)
        )
        if use_slang_bwd:
            from torch_vulkan.inductor.templates.caller import (
                _slang_tile_conv2d_bwd,
            )

            g_inp = torch.zeros_like(inp)
            g_w = torch.zeros_like(w)
            g_b = (
                torch.zeros(int(w.shape[0]), device=w.device, dtype=w.dtype)
                if has_bias
                else None
            )
            _slang_tile_conv2d_bwd(
                inp, w, grad_output, g_inp, g_w,
                stride=tuple(ctx.stride),
                padding=tuple(ctx.padding),
                dilation=tuple(ctx.dilation),
                bias=saved_b if has_bias else None,
                grad_bias=g_b,
            )
            return g_inp, g_w, g_b if has_bias else None, None, None, None, None

        result = torch.ops.aten.convolution_backward.default(
            grad_output, inp, w, None,
            ctx.stride, ctx.padding, ctx.dilation,
            False, [0] * len(ctx.stride), int(ctx.groups),
            [True, True, has_bias],
        )
        g_inp = result[0] if result[0] is not None else inp.new_empty(inp.shape)
        g_w = result[1] if result[1] is not None else w.new_empty(w.shape).zero_()
        g_b = (
            result[2]
            if len(result) > 2 and result[2] is not None and has_bias
            else (w.new_empty((int(w.shape[0]),)).zero_() if has_bias else None)
        )
        return g_inp, g_w, g_b if has_bias else None, None, None, None, None

    op.register_autograd(_backward, setup_context=_setup_context)
    result = getattr(torch.ops.torch_vulkan, spec.aten_op_name).default
    _registered[spec.aten_op_name] = result
    return result


def register_all_conv_epilogue_ops() -> dict[str, "object"]:
    """Register custom ops for all supported conv2d+epilogue combinations."""
    ops: dict[str, object] = {}
    for epilogue_name, spec in EPILOGUE_SPECS.items():
        ops[epilogue_name] = _ensure_conv2d_epilogue_op_registered(spec)
    return ops
