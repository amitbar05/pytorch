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
from ...vulkan_template_caller import (
    _dtype_to_slang,
    _validate_epilogue_struct,
)

_conv2d_cache: dict[tuple, str] = {}

_conv_bwd_cache: dict[tuple, str] = {}

_conv_gn_relu_cache: dict[tuple, str] = {}


def _render_conv_gn_relu_slang(
    has_bias: bool = False,
) -> str:
    """Render the conv_gn_relu.slang Jinja2 template.

    M17.2 Phase 3: combined Conv2D + GroupNorm + ReLU in a single
    Slang compute shader.  The template only has one Jinja2 toggle
    (``has_bias``); all dimensions come through push constants so
    the same SPIR-V module serves any shape that shares the bias
    presence.
    """
    from jinja2 import Environment

    key = (has_bias,)
    if key in _conv_gn_relu_cache:
        return _conv_gn_relu_cache[key]

    src = _load_slang_template("conv_gn_relu")
    if not src:
        raise RuntimeError("conv_gn_relu.slang template not found")

    env = Environment()
    tmpl = env.from_string(src)
    rendered = tmpl.render(has_bias=has_bias)
    _conv_gn_relu_cache[key] = rendered
    return rendered


def _slang_tile_conv2d_gn_relu(
    input_t: torch.Tensor,
    weight_t: torch.Tensor,
    bias: torch.Tensor | None,
    gn_weight: torch.Tensor,
    gn_bias: torch.Tensor,
    out: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    num_groups: int,
    eps: float,
) -> None:
    """Execute combined Conv2D + GroupNorm + ReLU.

    M17.2 Phase 3 originally introduced a single Slang shader
    (``conv_gn_relu.slang``) that does conv compute, Welford reduction,
    GN normalisation + affine, and the final ReLU in one
    ``vkQueueSubmit``.  That shader has two known bugs investigated in
    detail on 2026-05-21:

      1. Op-order: the shader pre-clamps with ReLU before the Welford
         accumulation, computing ``gn(relu(conv(x)))`` instead of
         ``relu(gn(conv(x)))`` — fixed in the shader body.
      2. slangc 2026.7.1 write-coverage miscompile: Wave 1 (lanes
         64..127 on RDNA1 wave64) silently drops its ``OpStore`` when
         Pass 3's stored value depends on both the conv-load chain and
         the welford results.  Verified via
         ``agent_space/probe_cgr_write_pattern.py``: a constant store
         hits every cell, but a store dependent on ``sum``/``mean``/
         ``rstd`` leaves the second channel-per-WG entirely at the
         pre-fill value.  Loop-shape rewrites did not lift the
         miscompile.

    Until a clean shader-side fix lands, this entry point falls back to
    a 3-dispatch decomp using the existing eager aten ops
    (``aten.convolution`` + ``F.group_norm`` + ``F.relu``).  Functional
    correctness is preserved; the dispatch-count win from M17.2-Phase-3
    is temporarily forfeit.  The original PC packing and dispatch wiring
    below the early-return is intentionally retained so the dispatch
    path can be re-armed by deleting the fallback once the underlying
    shader is fixed.

    Input:  [N, C_in, iH, iW]  (NCHW)
    Weight: [C_out, C_in, kH, kW]  (groups=1)
    Bias:   [C_out] or None
    GN weight: [C_out]  (gamma)
    GN bias:   [C_out]  (beta)
    Output: [N, C_out, oH, oW]  (NCHW)
    """
    import torch.nn.functional as F

    from ...runtime import compile_and_dispatch

    # ── Group-D fix (2026-05-27): reduced WG 256→64, avoids slangc ──
    # 2026.7.1 multi-wave write-coverage miscompile.  The fused shader
    # (conv_gn_relu.slang) is now re-armed.
    # Previous fallback (3-dispatch aten decomp) is removed.

    N, C_in, iH, iW = input_t.shape
    C_out, C_in_w, kH, kW = weight_t.shape
    assert C_in == C_in_w, f"weight C_in mismatch: {C_in} vs {C_in_w}"

    sH, sW = stride
    pH, pW = padding
    dH, dW = dilation

    oH = (iH + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    oW = (iW + 2 * pW - dW * (kW - 1) - 1) // sW + 1

    G = num_groups
    channels_per_group = C_out // G
    spatial_size = oH * oW
    group_size = channels_per_group * spatial_size
    num_rows = N * G

    has_bias = bias is not None
    dtype_s = _dtype_to_slang(input_t.dtype)
    src = _render_conv_gn_relu_slang(has_bias=has_bias)
    # ``_relufix2`` (2026-05-21, Group D): bumped twice in one session —
    # (1) corrected the op-order bug (was ``gn(relu(conv(x)))``,
    #     should be ``relu(gn(conv(x)))``), and
    # (2) reshaped the Pass-3 loop nest to work around a slangc 2026.7.1
    #     miscompile that dropped Wave 1's stores when Pass 3's
    #     ``d``-stride loop mirrored Pass 1's.
    # Cache-busting tag prevents stale SPIR-V blobs from being reused.
    # Bumped for M-CG.3 WG 256→64 fix (avoids slangc write-coverage bug).
    cache_key = f"conv_gn_relu_{dtype_s}{'_bias' if has_bias else ''}_mcg3_wg64"

    # Ensure contiguous for direct buffer access
    if not input_t.is_contiguous():
        input_t = input_t.contiguous()
    if not weight_t.is_contiguous():
        weight_t = weight_t.contiguous()

    # Pack push constants (33 uint fields with bias, 32 without)
    # Layout must match PC struct in conv_gn_relu.slang:
    #   conv dims (15) + strides_in (4) + strides_w (4) + strides_out (4)
    #   + GN params (4 uint + 1 float = 5 fields) + bias_stride + _pad
    #
    # M-pipeline-1-followup: every integer field is wrapped with ``int(...)``
    # so AOT-passed ``SymInt`` values (from tensor metadata via the
    # post-AOT compile-mode wrapper) coerce to plain ints. Without the
    # wrap, ``struct.pack("32IfI", ...)`` raises
    # ``struct.error: required argument is not an integer`` because
    # ``SymInt`` doesn't satisfy the ``I`` format's int-conversion
    # protocol on PyTorch 2.11.
    common_fields = (
        int(N),
        int(C_in),
        int(C_out),
        int(iH),
        int(iW),
        int(oH),
        int(oW),
        int(kH),
        int(kW),
        int(sH),
        int(sW),
        int(pH),
        int(pW),
        int(dH),
        int(dW),
        int(input_t.stride(0)),
        int(input_t.stride(1)),
        int(input_t.stride(2)),
        int(input_t.stride(3)),
        int(weight_t.stride(0)),
        int(weight_t.stride(1)),
        int(weight_t.stride(2)),
        int(weight_t.stride(3)),
        int(out.stride(0)),
        int(out.stride(1)),
        int(out.stride(2)),
        int(out.stride(3)),
        int(G),
        int(group_size),
    )
    # PC layout (conv_gn_relu.slang::PC) — trimmed to 124B (≤128B RDNA1 cap):
    #   15 conv dims + 4 stride_in + 4 stride_w + 4 stride_out
    #   + 2 GN params (num_groups, group_size)               = 29 uints
    #   + 1 float eps                                         =  4 bytes
    #   + 1 uint stride_bias                                  =  4 bytes
    # Total = 29*4 + 4 + 4 = 124 bytes.
    # (Removed: spatial_size, channels_per_group, _pad — derived in shader.)
    bias_stride = int(bias.view(-1).stride(0)) if has_bias else 0
    pc = struct.pack(
        "29IfI",
        *common_fields,
        float(eps),
        bias_stride,
    )

    buffers = [input_t, weight_t]
    if has_bias:
        buffers.append(bias.view(-1))
    else:
        # Pass a dummy buffer for the bias slot — the shader won't read it
        # because stride_bias == 0, but Vulkan requires all declared bindings.
        buffers.append(torch.empty(1, dtype=torch.float32, device=input_t.device))
    buffers.extend([gn_weight.view(-1), gn_bias.view(-1), out])

    compile_and_dispatch(
        src,
        buffers,
        num_rows,  # grid_x = N * G workgroups
        1,  # grid_y
        1,  # grid_z
        push_constants=pc,
        num_outputs=1,
        entry="computeMain",
        cache_key=cache_key,
    )


def _render_conv2d_slang(
    tile_w: int = 8,
    tile_h: int = 8,
    tile_c: int = 8,
    threads_w: int = 16,
    threads_h: int = 16,
    has_bias: bool = False,
    epilogue_struct: str | None = None,
) -> str:
    """Render the slang_conv2d Jinja2 template with tile configuration.

    M17.2: ``epilogue_struct`` is a validated ``IDifferentiable`` struct name
    (e.g. ``"OpReLU"``).  When non-``None``, the Jinja2 ``epilogue`` flag is
    set, enabling the ``Epilogue::apply(...)`` call site in the generated
    Slang source.  The concrete type is resolved at SPIR-V compile time via
    the ``entry`` parameter of ``compile_and_dispatch``.
    """
    from jinja2 import Environment

    epilogue_struct = _validate_epilogue_struct(epilogue_struct)

    key = (tile_w, tile_h, tile_c, threads_w, threads_h, has_bias, epilogue_struct)
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
        epilogue=epilogue_struct is not None,
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
    epilogue: str | None = None,
) -> None:
    """Execute direct tiled conv2d via Slang template shader.

    Input:  [N, C_in, iH, iW]  (NCHW)
    Weight: [C_out, C_in, kH, kW]  (NCHW, groups=1)
    Output: [N, C_out, oH, oW]  (NCHW)

    M17.2: ``epilogue`` is an optional ``IDifferentiable`` struct name
    (e.g. ``"OpReLU"``).  When set, the entry point becomes
    ``computeMain<OpReLU>`` and the shader applies the activation in the
    store path.
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

    epilogue = _validate_epilogue_struct(epilogue)
    has_bias = bias is not None
    dtype_s = _dtype_to_slang(input_t.dtype)
    src = _render_conv2d_slang(
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
        f"{'_bias' if has_bias else ''}"
        f"{'_' + epilogue if epilogue else ''}_m17"
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

    # Pack push constants: 15 uint fields for no-bias, 17 for bias.
    # M-pipeline-1-followup: wrap every int field with ``int(...)`` so
    # AOT-passed ``SymInt`` shape / stride metadata coerces cleanly.
    common_fields = (
        int(N),
        int(C_in),
        int(C_out),
        int(iH),
        int(iW),
        int(oH),
        int(oW),
        int(kH),
        int(kW),
        int(sH),
        int(sW),
        int(pH),
        int(pW),
        int(dH),
        int(dW),
        int(input_t.stride(0)),
        int(input_t.stride(1)),
        int(input_t.stride(2)),
        int(input_t.stride(3)),
        int(weight_t.stride(0)),
        int(weight_t.stride(1)),
        int(weight_t.stride(2)),
        int(weight_t.stride(3)),
        int(out.stride(0)),
        int(out.stride(1)),
        int(out.stride(2)),
        int(out.stride(3)),
        int(tile_w),
        int(tile_h),
        int(tile_c),
    )
    if has_bias:
        bias_1d = bias.view(-1)
        pc = struct.pack(
            "32I",
            *common_fields,
            int(bias_1d.stride(0)),
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

    # M17.2: Resolve the entry point.  When an epilogue is set, the entry
    # becomes ``computeMain<OpReLU>`` (etc.) so slangc selects the correct
    # generic instantiation.  When no epilogue, ``computeMain<OpIdentity>``
    # applies the identity function.
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
    #
    # Why each tensor is checked: when AOT Autograd traces a depthwise
    # backward, ``input_t`` (saved from forward) can stay real while
    # ``grad_input``/``grad_weight`` get re-materialized as FakeTensors via
    # ``torch.zeros_like`` under FakeTensorMode. Checking just ``input_t``
    # missed that path.
    def _is_meta(t):
        if t is None:
            return False
        try:
            return t.untyped_storage().device.type == "meta"
        except Exception:
            # FakeTensors raise on untyped_storage() — treat as tracing.
            return True

    if any(
        _is_meta(t)
        for t in (input_t, weight_t, grad_out, grad_input, grad_weight, bias, grad_bias)
    ):
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
    # M20.3: tile sizes are spec constants (IDs 40-44) — they don't
    # affect the SPIR-V hash, so the cache key collapses to (dtype,
    # has_bias). The same SPIR-V module serves every tile combo;
    # the tuple is applied as a pipeline spec-constant override at
    # dispatch time.
    cache_key = f"slang_conv_bwd_m20p3_{dtype_s}{'_bias' if has_bias else ''}"

    # M20.3: Vulkan specialization constant overrides for the tile
    # tuple (constant_id 40-44). One pipeline per tuple, but the same
    # SPIR-V module is reused — slangc cost amortises across every
    # autotuned shape that hits this template.
    spec_constants = [
        (40, tile_w),
        (41, tile_h),
        (42, tile_c),
        (43, threads_w),
        (44, threads_h),
    ]

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
    #
    # M-pipeline-1-followup: wrap every int field with ``int(...)`` so
    # AOT-passed ``SymInt`` shape / stride metadata coerces cleanly.
    common_fields = (
        int(N),
        int(C_in),
        int(C_out),
        int(iH),
        int(iW),
        int(oH),
        int(oW),
        int(kH),
        int(kW),
        int(sH),
        int(sW),
        int(pH),
        int(pW),
        int(dH),
        int(dW),
        int(input_t.stride(0)),
        int(input_t.stride(1)),
        int(input_t.stride(2)),
        int(input_t.stride(3)),
        int(weight_t.stride(0)),
        int(weight_t.stride(1)),
        int(weight_t.stride(2)),
        int(weight_t.stride(3)),
        int(grad_out.stride(0)),
        int(grad_out.stride(1)),
        int(grad_out.stride(2)),
        int(grad_out.stride(3)),
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
        spec_constants=spec_constants,
    )
