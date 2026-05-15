"""Conv2d template callers.

Provides rendering and dispatch for the Conv2d forward and backward Slang templates.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

from ...vulkan_template import _load_slang_template
from ...vulkan_template_caller import _dtype_to_slang

_conv2d_cache: dict[tuple, str] = {}

_conv_bwd_cache: dict[tuple, str] = {}


def _render_conv2d_slang(
    tile_w: int = 8,
    tile_h: int = 8,
    tile_c: int = 8,
    threads_w: int = 16,
    threads_h: int = 16,
    has_bias: bool = False,
) -> str:
    """Render the slang_conv2d Jinja2 template with tile configuration."""
    from jinja2 import Environment

    key = (tile_w, tile_h, tile_c, threads_w, threads_h, has_bias)
    if key in _conv2d_cache:
        return _conv2d_cache[key]

    src = _load_slang_template("slang_conv2d")
    if not src:
        raise RuntimeError("slang_conv2d.slang template not found")

    env = Environment()
    tmpl = env.from_string(src)
    rendered = tmpl.render(
        tile_w=tile_w,
        tile_h=tile_h,
        tile_c=tile_c,
        threads_w=threads_w,
        threads_h=threads_h,
        has_bias=has_bias,
    )
    _conv2d_cache[key] = rendered
    return rendered


def _slang_tile_conv2d(
    input_t: torch.Tensor,
    weight_t: torch.Tensor,
    out: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int = 1,
    bias: torch.Tensor | None = None,
    tile_w: int = 8,
    tile_h: int = 8,
    tile_c: int = 8,
    threads_w: int = 16,
    threads_h: int = 16,
) -> None:
    """Execute direct tiled conv2d via Slang template shader.

    Input:  [N, C_in, iH, iW]  (NCHW)
    Weight: [C_out, C_in, kH, kW]  (NCHW, groups=1)
    Output: [N, C_out, oH, oW]  (NCHW)
    """
    from ...runtime import compile_and_dispatch

    N, C_in, iH, iW = input_t.shape
    C_out, C_in_w, kH, kW = weight_t.shape
    assert C_in == C_in_w, f"weight C_in mismatch: {C_in} vs {C_in_w}"
    assert groups == 1, "Only groups=1 supported"

    sH, sW = stride
    pH, pW = padding
    dH, dW = dilation

    oH = (iH + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    oW = (iW + 2 * pW - dW * (kW - 1) - 1) // sW + 1

    has_bias = bias is not None
    dtype_s = _dtype_to_slang(input_t.dtype)
    src = _render_conv2d_slang(
        tile_w=tile_w,
        tile_h=tile_h,
        tile_c=tile_c,
        threads_w=threads_w,
        threads_h=threads_h,
        has_bias=has_bias,
    )
    cache_key = (
        f"slang_conv2d_{tile_w}x{tile_h}x{tile_c}"
        f"_t{threads_w}x{threads_h}_{dtype_s}"
        f"{'_bias' if has_bias else ''}_m15"
    )

    # CG.M15: spec_constants for [[vk::constant_id]] overrides.
    spec_constants = [
        (30, tile_w),
        (31, tile_h),
        (32, tile_c),
        (33, threads_w),
        (34, threads_h),
    ]

    # Ensure contiguous for direct buffer access
    if not input_t.is_contiguous():
        input_t = input_t.contiguous()
    if not weight_t.is_contiguous():
        weight_t = weight_t.contiguous()

    # Pack push constants: 15 uint fields for no-bias, 17 for bias
    common_fields = (
        N,
        C_in,
        C_out,
        iH,
        iW,
        oH,
        oW,
        kH,
        kW,
        sH,
        sW,
        pH,
        pW,
        dH,
        dW,
        input_t.stride(0),
        input_t.stride(1),
        input_t.stride(2),
        input_t.stride(3),
        weight_t.stride(0),
        weight_t.stride(1),
        weight_t.stride(2),
        weight_t.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        tile_w,
        tile_h,
        tile_c,
    )
    if has_bias:
        bias_1d = bias.view(-1)
        pc = struct.pack(
            "32I",
            *common_fields,
            bias_1d.stride(0),
            0,  # _pad
        )
    else:
        pc = struct.pack("30I", *common_fields)

    grid_x = (oW + tile_w - 1) // tile_w
    grid_y = (oH + tile_h - 1) // tile_h
    tile_c_count = (C_out + tile_c - 1) // tile_c
    grid_z = N * tile_c_count

    buffers = [input_t, weight_t]
    if has_bias:
        buffers.append(bias.view(-1))
    buffers.append(out)

    compile_and_dispatch(
        src,
        buffers,
        grid_x,
        grid_y,
        grid_z,
        push_constants=pc,
        num_outputs=1,
        entry="computeMain",
        cache_key=cache_key,
        spec_constants=spec_constants,
    )


def _render_conv_bwd_slang(
    tile_w: int = 8,
    tile_h: int = 8,
    tile_c: int = 8,
    threads_w: int = 16,
    threads_h: int = 16,
    has_bias: bool = False,
) -> str:
    """Render the slang_conv_bwd Jinja2 template with tile configuration."""
    from jinja2 import Environment

    key = (tile_w, tile_h, tile_c, threads_w, threads_h, has_bias)
    if key in _conv_bwd_cache:
        return _conv_bwd_cache[key]

    src = _load_slang_template("slang_conv_bwd")
    if not src:
        raise RuntimeError("slang_conv_bwd.py.jinja template not found")

    env = Environment()
    tmpl = env.from_string(src)
    rendered = tmpl.render(
        tile_w=tile_w,
        tile_h=tile_h,
        tile_c=tile_c,
        threads_w=threads_w,
        threads_h=threads_h,
        has_bias=has_bias,
    )
    _conv_bwd_cache[key] = rendered
    return rendered


def _slang_tile_conv2d_bwd(
    input_t: torch.Tensor,
    weight_t: torch.Tensor,
    grad_out: torch.Tensor,
    grad_input: torch.Tensor,
    grad_weight: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    bias: torch.Tensor | None = None,
    grad_bias: torch.Tensor | None = None,
    tile_w: int = 8,
    tile_h: int = 8,
    tile_c: int = 8,
    threads_w: int = 16,
    threads_h: int = 16,
) -> None:
    """Execute conv2d backward via Slang template shader.

    Computes dX (grad_input), dW (grad_weight), and optionally dB (grad_bias)
    in a single dispatch using ``bwd_diff(conv_inner_madd)``.

    FakeTensor-aware: during AOT Autograd tracing, all inputs are FakeTensors.
    We detect this and return early — the caller already allocated output tensors
    with correct shapes; no actual computation is needed during tracing.

    Args:
        input_t:    [N, C_in, iH, iW]  (NCHW) — saved forward input
        weight_t:   [C_out, C_in, kH, kW] — saved forward weight
        grad_out:   [N, C_out, oH, oW] — upstream gradient
        grad_input: [N, C_in, iH, iW] — output: input gradient (zero-initialized)
        grad_weight:[C_out, C_in, kH, kW] — output: weight gradient (zero-initialized)
        grad_bias:  [C_out] — output: bias gradient (zero-initialized), optional
    """
    # PF.51 guard: during AOT Autograd tracing, inputs are FakeTensors with
    # meta-device storage.  Skip the actual dispatch — the caller already
    # allocated output tensors with correct shapes.
    try:
        if input_t.untyped_storage().device.type == "meta":
            return  # tracing mode, outputs already allocated
    except Exception:
        # FakeTensors raise on untyped_storage() — treat as tracing mode.
        return

    from ...runtime import compile_and_dispatch

    N, C_in, iH, iW = input_t.shape
    C_out, C_in_w, kH, kW = weight_t.shape
    assert C_in == C_in_w, f"weight C_in mismatch: {C_in} vs {C_in_w}"

    sH, sW = stride
    pH, pW = padding
    dH, dW = dilation

    oH = (iH + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    oW = (iW + 2 * pW - dW * (kW - 1) - 1) // sW + 1

    has_bias = bias is not None
    dtype_s = _dtype_to_slang(input_t.dtype)
    src = _render_conv_bwd_slang(
        tile_w=tile_w,
        tile_h=tile_h,
        tile_c=tile_c,
        threads_w=threads_w,
        threads_h=threads_h,
        has_bias=has_bias,
    )
    cache_key = (
        f"slang_conv_bwd_{tile_w}x{tile_h}x{tile_c}"
        f"_t{threads_w}x{threads_h}_{dtype_s}"
        f"{'_bias' if has_bias else ''}"
    )

    # Ensure contiguous for direct buffer access
    if not input_t.is_contiguous():
        input_t = input_t.contiguous()
    if not weight_t.is_contiguous():
        weight_t = weight_t.contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    # Pack push constants: 27 uint fields (no bias) or 28 (with bias)
    # Layout matches BwdPC in slang_conv_bwd.py.jinja:
    #   dims (15) + stride_in (4) + stride_w (4) + stride_go (4) = 27
    #   + _pad_bwd (1) with bias = 28
    common_fields = (
        N,
        C_in,
        C_out,
        iH,
        iW,
        oH,
        oW,
        kH,
        kW,
        sH,
        sW,
        pH,
        pW,
        dH,
        dW,
        input_t.stride(0),
        input_t.stride(1),
        input_t.stride(2),
        input_t.stride(3),
        weight_t.stride(0),
        weight_t.stride(1),
        weight_t.stride(2),
        weight_t.stride(3),
        grad_out.stride(0),
        grad_out.stride(1),
        grad_out.stride(2),
        grad_out.stride(3),
    )
    if has_bias:
        pc = struct.pack("28I", *common_fields, 0)  # _pad_bwd
    else:
        pc = struct.pack("27I", *common_fields)

    grid_x = (oW + tile_w - 1) // tile_w
    grid_y = (oH + tile_h - 1) // tile_h
    tile_c_count = (C_out + tile_c - 1) // tile_c
    grid_z = N * tile_c_count

    buffers = [input_t, weight_t, grad_out, grad_input, grad_weight]
    if has_bias:
        buffers.append(grad_bias.view(-1))

    compile_and_dispatch(
        src,
        buffers,
        grid_x,
        grid_y,
        grid_z,
        push_constants=pc,
        num_outputs=2 if not has_bias else 3,
        entry="computeMain",
        cache_key=cache_key,
    )
