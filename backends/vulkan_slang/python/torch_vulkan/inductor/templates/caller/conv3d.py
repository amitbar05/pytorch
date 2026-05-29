"""Conv3d template callers.

Provides rendering and dispatch for the Conv3d forward and backward Slang templates.
Following the pattern of templates/caller/conv.py (Conv2d).
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

from ...vulkan_template import _load_slang_template
from ...vulkan_template_caller import (
    _dtype_to_slang,
    _validate_epilogue_struct,
)

_conv3d_cache: dict[tuple, str] = {}
_conv3d_bwd_cache: dict[tuple, str] = {}


# ======================================================================
# Forward
# ======================================================================


def _render_conv3d_slang(
    tile_w: int = 4,
    tile_h: int = 4,
    tile_c: int = 4,
    threads_w: int = 8,
    threads_h: int = 8,
    epilogue_struct: str | None = None,
) -> str:
    """Render the slang_conv3d Jinja2 template with tile configuration."""
    from jinja2 import Environment

    epilogue_struct = _validate_epilogue_struct(epilogue_struct)

    key = (tile_w, tile_h, tile_c, threads_w, threads_h, epilogue_struct)
    if key in _conv3d_cache:
        return _conv3d_cache[key]

    src = _load_slang_template("slang_conv3d")
    if not src:
        raise RuntimeError("slang_conv3d.slang template not found")

    env = Environment()
    tmpl = env.from_string(src)
    rendered = tmpl.render(
        tile_w=tile_w,
        tile_h=tile_h,
        tile_c=tile_c,
        threads_w=threads_w,
        threads_h=threads_h,
        epilogue=epilogue_struct is not None,
    )
    _conv3d_cache[key] = rendered
    return rendered


def _slang_tile_conv3d(
    input_t: torch.Tensor,
    weight_t: torch.Tensor,
    out: torch.Tensor,
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    dilation: tuple[int, int, int],
    groups: int = 1,
    bias: torch.Tensor | None = None,
    tile_w: int = 4,
    tile_h: int = 4,
    tile_c: int = 4,
    threads_w: int = 8,
    threads_h: int = 8,
    epilogue: str | None = None,
) -> None:
    """Execute direct tiled conv3d via Slang template shader.

    Input:  [N, C_in, iD, iH, iW]  (NCDHW)
    Weight: [C_out, C_in, kD, kH, kW]  (groups=1)
    Output: [N, C_out, oD, oH, oW]  (NCDHW)

    ``epilogue`` is an optional ``IDifferentiable`` struct name
    (e.g. ``"OpReLU"``).  When set, the entry point becomes
    ``computeMain<OpReLU>`` and the shader applies the activation.
    """
    from ...runtime import compile_and_dispatch

    N, C_in, iD, iH, iW = input_t.shape
    C_out, C_in_w, kD, kH, kW = weight_t.shape
    assert C_in == C_in_w, f"weight C_in mismatch: {C_in} vs {C_in_w}"
    assert groups == 1, "Only groups=1 supported for Conv3d"

    sD, sH, sW = stride
    pD, pH, pW = padding
    dD, dH, dW = dilation

    oD = (iD + 2 * pD - dD * (kD - 1) - 1) // sD + 1
    oH = (iH + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    oW = (iW + 2 * pW - dW * (kW - 1) - 1) // sW + 1

    epilogue = _validate_epilogue_struct(epilogue)
    has_bias = bias is not None
    dtype_s = _dtype_to_slang(input_t.dtype)
    src = _render_conv3d_slang(
        tile_w=tile_w,
        tile_h=tile_h,
        tile_c=tile_c,
        threads_w=threads_w,
        threads_h=threads_h,
        epilogue_struct=epilogue,
    )
    cache_key = (
        f"slang_conv3d_{tile_w}x{tile_h}x{tile_c}"
        f"_t{threads_w}x{threads_h}_{dtype_s}"
        f"{'_' + epilogue if epilogue else ''}"
    )

    # CG.SF.01: spec constants for [[vk::constant_id]] overrides.
    spec_constants = [
        (50, tile_w),
        (51, tile_h),
        (52, tile_c),
        (53, threads_w),
        (54, threads_h),
    ]

    # Ensure contiguous for direct buffer access (strides computed from sizes)
    if not input_t.is_contiguous():
        input_t = input_t.contiguous()
    if not weight_t.is_contiguous():
        weight_t = weight_t.contiguous()

    # Pack push constants: 28 uint fields = 112B.
    # Layout matches PC in slang_conv3d.slang:
    #   dims (21) + runtime tiles (3) + stride_bias + pad(3)
    bias_1d = (
        bias.view(-1)
        if has_bias
        else torch.zeros(1, device=input_t.device, dtype=input_t.dtype)
    )
    stride_bias = int(bias_1d.stride(0)) if has_bias else 0
    pc = struct.pack(
        "28I",
        int(N), int(C_in), int(C_out),
        int(iD), int(iH), int(iW),
        int(oD), int(oH), int(oW),
        int(kD), int(kH), int(kW),
        int(sD), int(sH), int(sW),
        int(pD), int(pH), int(pW),
        int(dD), int(dH), int(dW),
        int(tile_w), int(tile_h), int(tile_c),
        stride_bias,
        0,  # _pad0
        0,  # _pad1
        0,  # extra pad to align to 28 fields
    )

    # Grid decomposition: each workgroup handles one od slice.
    grid_x = (oW + tile_w - 1) // tile_w
    grid_y = (oH + tile_h - 1) // tile_h
    tile_c_count = (C_out + tile_c - 1) // tile_c
    grid_z = N * tile_c_count * oD

    # Always include bias buffer (dummy when no bias).
    buffers = [input_t, weight_t]
    if has_bias:
        buffers.append(bias.view(-1))
    else:
        buffers.append(torch.zeros(1, device=input_t.device, dtype=input_t.dtype))
    buffers.append(out)

    # Resolve entry point.
    entry_point = (
        f"computeMain<{epilogue}>"
        if epilogue is not None
        else "computeMain<OpIdentity>"
    )

    compile_and_dispatch(
        src,
        buffers,
        grid_x,
        grid_y,
        grid_z,
        push_constants=pc,
        num_outputs=1,
        entry=entry_point,
        cache_key=cache_key,
        spec_constants=spec_constants,
    )


# ======================================================================
# Backward
# ======================================================================


def _render_conv3d_bwd_slang(
    tile_w: int = 4,
    tile_h: int = 4,
    tile_c: int = 4,
    threads_w: int = 8,
    threads_h: int = 8,
    has_bias: bool = False,
) -> str:
    """Render the slang_conv3d_bwd Jinja2 template with tile configuration."""
    from jinja2 import Environment

    key = (tile_w, tile_h, tile_c, threads_w, threads_h, has_bias)
    if key in _conv3d_bwd_cache:
        return _conv3d_bwd_cache[key]

    src = _load_slang_template("slang_conv3d_bwd")
    if not src:
        raise RuntimeError("slang_conv3d_bwd.slang template not found")

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
    _conv3d_bwd_cache[key] = rendered
    return rendered


def _slang_tile_conv3d_bwd(
    input_t: torch.Tensor,
    weight_t: torch.Tensor,
    grad_out: torch.Tensor,
    grad_input: torch.Tensor,
    grad_weight: torch.Tensor,
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    dilation: tuple[int, int, int],
    bias: torch.Tensor | None = None,
    grad_bias: torch.Tensor | None = None,
    tile_w: int = 4,
    tile_h: int = 4,
    tile_c: int = 4,
    threads_w: int = 8,
    threads_h: int = 8,
) -> None:
    """Execute conv3d backward via Slang template shader.

    Computes dX (grad_input), dW (grad_weight), and optionally dB (grad_bias)
    in a single dispatch using ``bwd_diff(conv_inner_madd)``.

    Args:
        input_t:    [N, C_in, iD, iH, iW]  (NCDHW) -- saved forward input
        weight_t:   [C_out, C_in, kD, kH, kW] -- saved forward weight
        grad_out:   [N, C_out, oD, oH, oW] -- upstream gradient
        grad_input: [N, C_in, iD, iH, iW] -- output: input gradient
        grad_weight:[C_out, C_in, kD, kH, kW] -- output: weight gradient
        grad_bias:  [C_out] -- output: bias gradient (optional)
    """

    # PF.51 guard: during AOT Autograd tracing, skip actual dispatch.
    def _is_meta(t):
        if t is None:
            return False
        try:
            return t.untyped_storage().device.type == "meta"
        except Exception:
            return True

    if any(
        _is_meta(t)
        for t in (input_t, weight_t, grad_out, grad_input, grad_weight, bias, grad_bias)
    ):
        return

    from ...runtime import compile_and_dispatch

    N, C_in, iD, iH, iW = input_t.shape
    C_out, C_in_w, kD, kH, kW = weight_t.shape
    assert C_in == C_in_w, f"weight C_in mismatch: {C_in} vs {C_in_w}"

    sD, sH, sW = stride
    pD, pH, pW = padding
    dD, dH, dW = dilation

    oD = (iD + 2 * pD - dD * (kD - 1) - 1) // sD + 1
    oH = (iH + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    oW = (iW + 2 * pW - dW * (kW - 1) - 1) // sW + 1

    has_bias = grad_bias is not None
    dtype_s = _dtype_to_slang(input_t.dtype)
    src = _render_conv3d_bwd_slang(
        tile_w=tile_w,
        tile_h=tile_h,
        tile_c=tile_c,
        threads_w=threads_w,
        threads_h=threads_h,
        has_bias=has_bias,
    )
    cache_key = f"slang_conv3d_bwd_{dtype_s}{'_bias' if has_bias else ''}"

    # CG.SF.01: spec constants.
    spec_constants = [
        (50, tile_w),
        (51, tile_h),
        (52, tile_c),
        (53, threads_w),
        (54, threads_h),
    ]

    # Ensure contiguous for direct buffer access
    if not input_t.is_contiguous():
        input_t = input_t.contiguous()
    if not weight_t.is_contiguous():
        weight_t = weight_t.contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    # Pack push constants: 28 uint fields = 112B.
    pc = struct.pack(
        "28I",
        int(N), int(C_in), int(C_out),
        int(iD), int(iH), int(iW),
        int(oD), int(oH), int(oW),
        int(kD), int(kH), int(kW),
        int(sD), int(sH), int(sW),
        int(pD), int(pH), int(pW),
        int(dD), int(dH), int(dW),
        int(tile_w), int(tile_h), int(tile_c),
        1 if has_bias else 0,  # has_bias
        0,  # _pad0
        0,  # _pad1
        0,  # extra pad to align to 28 fields
    )

    # Grid: each workgroup handles one od slice.
    grid_x = (oW + tile_w - 1) // tile_w
    grid_y = (oH + tile_h - 1) // tile_h
    tile_c_count = (C_out + tile_c - 1) // tile_c
    grid_z = N * tile_c_count * oD

    buffers = [input_t, weight_t, grad_out, grad_input, grad_weight]
    if has_bias:
        buffers.append(grad_bias.view(-1))
    else:
        # Dummy for the grad_bias slot (shader won't write when has_bias==0)
        buffers.append(torch.zeros(1, device=input_t.device, dtype=input_t.dtype))

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
        spec_constants=spec_constants,
    )
