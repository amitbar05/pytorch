"""conv_im2col — decompose grouped torch_vulkan.conv2d_with_optional_bias.

Priority 15 — runs after most fusions.

M6 Phase 2: Only matches groups>1 conv2d nodes.  Groups==1 conv2d is handled
by the dedicated ``slang_conv2d.slang`` direct template (lowering in conv.py).

For groups>1, the rewrite decomposes into per-group ``conv2d_with_optional_bias``
calls with groups=1 (each routing through the ``slang_conv2d`` template) instead
of the old im2col+mm approach that required ``as_strided`` views.  This
eliminates the M6 as_strided view materialization blocker for both forward
and backward.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch.fx import GraphModule, Node

from .registry import register_fx_pattern


def _fx_to_int(x: Any) -> int:
    """Extract a concrete int from a value that may be an FX Node or SymInt.

    A grouped/depthwise conv is structurally specialized on its groups,
    kernel extent, and channel counts — the per-group decomposition slices
    a concrete number of channels and the ``slang_conv2d`` template bakes
    the kernel window in. Under dynamic shapes these arrive as symbols, so
    guard (specialize) them to a concrete value rather than crashing.
    """
    if isinstance(x, Node):
        val = x.meta.get("val")
        if val is None:
            raise ValueError(f"Cannot extract int from FX Node {x} with no val meta")
        x = val
    if isinstance(x, int):
        return x
    if isinstance(x, torch.SymInt):
        from torch.fx.experimental.symbolic_shapes import guard_int

        return int(guard_int(x))
    return int(x)


def _match_conv_im2col(gm: GraphModule) -> Iterable[tuple[Node, dict[str, Any]]]:
    """Match ``torch_vulkan.conv2d_with_optional_bias`` calls whose input
    and weight tensors are 4-D with valid group division."""
    try:
        conv_target = torch.ops.torch_vulkan.conv2d_with_optional_bias.default
    except (AttributeError, RuntimeError):
        return iter(())

    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        if node.target != conv_target:
            continue
        if len(node.args) < 7:
            continue
        inp, weight, bias, stride, padding, dilation, groups = node.args[:7]
        if not isinstance(inp, Node) or not isinstance(weight, Node):
            continue
        inp_val = inp.meta.get("val") if hasattr(inp, "meta") else None
        w_val = weight.meta.get("val") if hasattr(weight, "meta") else None
        if inp_val is None or w_val is None:
            continue
        if inp_val.dim() != 4 or w_val.dim() != 4:
            continue
        g = _fx_to_int(groups)
        if g < 1:
            continue
        # T4.12 / M6 Phase 2: groups==1 convs go through the dedicated
        # ``slang_conv2d.slang`` direct template (lowering in conv.py);
        # the FX pattern only decomposes grouped convs (groups>1) to
        # im2col+mm.
        if g == 1:
            continue
        if _fx_to_int(w_val.shape[0]) % g != 0:
            continue
        if _fx_to_int(inp_val.shape[1]) % g != 0:
            continue
        # All checks passed — yield the match context.
        yield (
            node,
            {
                "inp": inp,
                "weight": weight,
                "bias": bias,
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "groups": g,
            },
        )


def _rewrite_conv_im2col(
    gm: GraphModule, root: Node, ctx: dict[str, Any]
) -> GraphModule:
    """groups == 1  ->  im2col (as_strided) + mm
    groups  > 1  ->  per-group conv2d (groups=1) + cat

    M6 Phase 2: Per-group decomposition uses the
    ``torch_vulkan::conv2d_with_optional_bias`` custom op with groups=1
    so the dedicated ``slang_conv2d`` template handles each group's
    forward/backward without any ``as_strided`` views.
    """
    aten = torch.ops.aten
    inp: Node = ctx["inp"]
    weight: Node = ctx["weight"]
    bias: Node = ctx["bias"]
    stride = ctx["stride"]
    padding = ctx["padding"]
    dilation = ctx["dilation"]
    groups: int = _fx_to_int(ctx["groups"])
    conv_target = root.target

    sH = _fx_to_int(stride[0])
    sW = _fx_to_int(stride[-1]) if len(stride) > 1 else sH
    pH = _fx_to_int(padding[0])
    pW = _fx_to_int(padding[-1]) if len(padding) > 1 else pH
    dH = _fx_to_int(dilation[0])
    dW = _fx_to_int(dilation[-1]) if len(dilation) > 1 else dH

    inp_val = inp.meta["val"]
    w_val = weight.meta["val"]
    N, C_in, H_in, W_in = (int(s) for s in inp_val.shape)
    C_out, C_in_per_g, kH, kW = (int(s) for s in w_val.shape)
    assert C_in_per_g * groups == C_in
    assert C_out % groups == 0
    M_per_g = C_out // groups
    H_out = (H_in + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    W_out = (W_in + 2 * pW - dW * (kW - 1) - 1) // sW + 1

    out_val = root.meta.get("val")
    out_dtype = out_val.dtype if out_val is not None else inp_val.dtype
    out_device = out_val.device if out_val is not None else inp_val.device

    has_bias = bias is not None and isinstance(bias, Node)

    def _empty_meta(node, shape, dtype, device):
        node.meta["val"] = torch.empty(*shape, dtype=dtype, device=device)

    def _im2col_mm(x_node, x_C_in, w_node, w_C_out):
        if pH > 0 or pW > 0:
            pad_node = gm.graph.call_function(
                aten.constant_pad_nd.default,
                args=(x_node, [pW, pW, pH, pH], 0.0),
            )
            pad_node.meta = dict(x_node.meta)
            _empty_meta(
                pad_node,
                (N, x_C_in, H_in + 2 * pH, W_in + 2 * pW),
                inp_val.dtype,
                inp_val.device,
            )
            xn = pad_node
            H_pad, W_pad = H_in + 2 * pH, W_in + 2 * pW
        else:
            xn = x_node
            H_pad, W_pad = H_in, W_in
        s_n_p = x_C_in * H_pad * W_pad
        s_c_p = H_pad * W_pad
        s_h_p = W_pad
        s_w_p = 1

        win_size = [N, H_out, W_out, x_C_in, kH, kW]
        win_stride = [s_n_p, sH * s_h_p, sW * s_w_p, s_c_p, dH * s_h_p, dW * s_w_p]
        windowed = gm.graph.call_function(
            aten.as_strided.default,
            args=(xn, win_size, win_stride),
        )
        windowed.meta = dict(xn.meta)
        _empty_meta(
            windowed,
            (N, H_out, W_out, x_C_in, kH, kW),
            inp_val.dtype,
            inp_val.device,
        )

        patches = gm.graph.call_function(
            aten.reshape.default,
            args=(windowed, [N * H_out * W_out, x_C_in * kH * kW]),
        )
        patches.meta = dict(windowed.meta)
        _empty_meta(
            patches,
            (N * H_out * W_out, x_C_in * kH * kW),
            inp_val.dtype,
            inp_val.device,
        )

        w_flat = gm.graph.call_function(
            aten.reshape.default,
            args=(w_node, [w_C_out, x_C_in * kH * kW]),
        )
        w_flat.meta = dict(w_node.meta)
        _empty_meta(w_flat, (w_C_out, x_C_in * kH * kW), w_val.dtype, w_val.device)

        w_t = gm.graph.call_function(
            aten.permute.default,
            args=(w_flat, [1, 0]),
        )
        w_t.meta = dict(w_flat.meta)
        _empty_meta(w_t, (x_C_in * kH * kW, w_C_out), w_val.dtype, w_val.device)

        out_flat = gm.graph.call_function(
            aten.mm.default,
            args=(patches, w_t),
        )
        out_flat.meta = dict(root.meta)
        _empty_meta(out_flat, (N * H_out * W_out, w_C_out), out_dtype, out_device)
        return out_flat

    with gm.graph.inserting_before(root):
        if groups == 1:
            out_flat = _im2col_mm(inp, C_in, weight, C_out)
            out_nhwc = gm.graph.call_function(
                aten.reshape.default,
                args=(out_flat, [N, H_out, W_out, C_out]),
            )
            out_nhwc.meta = dict(root.meta)
            out = gm.graph.call_function(
                aten.permute.default,
                args=(out_nhwc, [0, 3, 1, 2]),
            )
            out.meta = dict(root.meta)
        else:
            # M6 Phase 2: Per-group decomposition using group-1 conv2d
            # custom op calls.  Each call routes through the dedicated
            # ``slang_conv2d`` template — no as_strided views needed.
            # Backward is handled automatically: each per-group call
            # has groups=1, so ``_conv2d_backward`` uses the CG.M6
            # bwd template.
            Cg = C_in_per_g
            per_group_outs: list = []
            for g in range(groups):
                inp_g = gm.graph.call_function(
                    aten.slice.Tensor,
                    args=(inp, 1, g * Cg, (g + 1) * Cg),
                )
                inp_g.meta = dict(inp.meta)
                _empty_meta(inp_g, (N, Cg, H_in, W_in), inp_val.dtype, inp_val.device)
                w_g = gm.graph.call_function(
                    aten.slice.Tensor,
                    args=(weight, 0, g * M_per_g, (g + 1) * M_per_g),
                )
                w_g.meta = dict(weight.meta)
                _empty_meta(w_g, (M_per_g, Cg, kH, kW), w_val.dtype, w_val.device)

                if has_bias:
                    bias_g = gm.graph.call_function(
                        aten.slice.Tensor,
                        args=(bias, 0, g * M_per_g, (g + 1) * M_per_g),
                    )
                    bias_g.meta = dict(bias.meta)
                    _empty_meta(
                        bias_g,
                        (M_per_g,),
                        bias.meta["val"].dtype,
                        bias.meta["val"].device,
                    )
                else:
                    bias_g = None

                out_g = gm.graph.call_function(
                    conv_target,
                    args=(inp_g, w_g, bias_g, stride, padding, dilation, 1),
                )
                out_g.meta = dict(root.meta)
                _empty_meta(out_g, (N, M_per_g, H_out, W_out), out_dtype, out_device)
                per_group_outs.append(out_g)

            out = gm.graph.call_function(
                aten.cat.default,
                args=(per_group_outs, 1),
            )
            out.meta = dict(root.meta)

        # Bias is already handled per-group above when groups > 1.
        # For groups == 1, apply bias as before.
        if groups == 1 and has_bias:
            bias_view = gm.graph.call_function(
                aten.reshape.default,
                args=(bias, [1, C_out, 1, 1]),
            )
            bias_view.meta = (
                dict(bias.meta) if hasattr(bias, "meta") else dict(root.meta)
            )
            final = gm.graph.call_function(
                aten.add.Tensor,
                args=(out, bias_view),
            )
            final.meta = dict(root.meta)
        else:
            final = out

    root.replace_all_uses_with(final)
    gm.graph.erase_node(root)
    gm.graph.lint()
    gm.recompile()
    return gm


register_fx_pattern(
    "conv_im2col",
    _match_conv_im2col,
    _rewrite_conv_im2col,
    priority=15,
)
