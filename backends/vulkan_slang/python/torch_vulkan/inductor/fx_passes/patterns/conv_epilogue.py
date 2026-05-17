"""conv_epilogue — fuse conv2d + pointwise activation into a single
fused conv+epilogue dispatch via the Slang conv2d template's
``<Epilogue : IDifferentiable>`` generic parameter.

Priority 9 — runs AFTER ``op_class_fusion`` (8) annotates the graph, but
BEFORE ``addmm_gelu`` (10) and most structural rewrites.

M17.2: Works by matching conv2d_with_optional_bias → activation and
replacing the pair with ``torch_vulkan::conv2d_relu_fused`` (single
Slang dispatch).  Additional activation fusions (gelu, silu, etc.) are
extended in the same pattern.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Set

import torch
from torch.fx import GraphModule, Node

from .registry import register_fx_pattern

# ── Activation targets we can fuse into conv epilogue ──────────────────
_CONV_EPILOGUE_ACTIVATIONS = {
    torch.ops.aten.relu.default: "OpReLU",
    torch.ops.aten.gelu.default: "OpGELU",
    torch.ops.aten.silu.default: "OpSiLU",
    torch.ops.aten.sigmoid.default: "OpSigmoid",
    torch.ops.aten.tanh.default: "OpTanh",
    torch.ops.aten.hardsigmoid.default: "OpHardSigmoid",
    torch.ops.aten.hardswish.default: "OpHardSwish",
    torch.ops.aten.leaky_relu.default: "OpLeakyReLU",
    torch.ops.aten.elu.default: "OpELU",
}

# Conv targets in the FX graph (post-AOTAutograd).
# Populated lazily because the custom op may not be registered
# at module import time.
_CONV_TARGETS: Optional[Set] = None


def _get_conv_targets() -> Set:
    """Return the set of conv op targets, populating lazily."""
    global _CONV_TARGETS
    if _CONV_TARGETS is None:
        _CONV_TARGETS = set()
        try:
            _CONV_TARGETS.add(torch.ops.torch_vulkan.conv2d_with_optional_bias.default)
        except AttributeError:
            pass
        # Also match aten.convolution.default (used when the eager
        # patch is not active).
        try:
            _CONV_TARGETS.add(torch.ops.aten.convolution.default)
        except AttributeError:
            pass
    return _CONV_TARGETS


def _match_conv_epilogue(gm: GraphModule) -> Iterable[tuple[Node, dict[str, Any]]]:
    """Match conv2d with optional bias whose *sole* consumer is a pointwise
    activation (relu, gelu, silu, etc.)."""
    conv_targets = _get_conv_targets()

    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        if node.target not in _CONV_EPILOGUE_ACTIVATIONS:
            continue
        if len(node.users) == 0:
            continue

        pw_node = node
        # The activation should have exactly one tensor input from a conv.
        pw_args = [
            a for a in pw_node.args if isinstance(a, Node) and a.op == "call_function"
        ]
        if len(pw_args) != 1:
            continue
        conv_node = pw_args[0]

        if conv_node.target not in conv_targets:
            continue

        # The conv output must have exactly one consumer (the activation).
        if len(conv_node.users) != 1:
            continue

        # Validate conv args: (input, weight, bias, stride, padding, dilation, groups)
        if len(conv_node.args) < 2:
            continue

        epilogue_name = _CONV_EPILOGUE_ACTIVATIONS[pw_node.target]

        yield (
            pw_node,
            {
                "conv_node": conv_node,
                "pw_node": pw_node,
                "epilogue_name": epilogue_name,
            },
        )


def _rewrite_conv_epilogue(
    gm: GraphModule, root: Node, ctx: dict[str, Any]
) -> GraphModule:
    """Replace conv + activation with ``torch_vulkan::conv2d_relu_fused``.

    For activations other than ReLU, the general fused template path
    is a stub for now (the custom-op dispatch table only has ReLU).
    Other activations fall back to the original graph unchanged.
    """
    conv_node: Node = ctx["conv_node"]
    pw_node: Node = ctx["pw_node"]
    epilogue_name: str = ctx["epilogue_name"]

    # For now, only ReLU has a dedicated fused custom op.
    if epilogue_name != "OpReLU":
        # Future: register conv2d_gelu_fused, conv2d_silu_fused, etc.
        # and dispatch here.  For now, leave the graph unchanged so the
        # scheduler fuses the pointwise into a downstream kernel.
        return gm

    from ..eager.conv import _ensure_conv2d_relu_fused_op_registered

    fused_op = _ensure_conv2d_relu_fused_op_registered()

    # Conv node args: (input, weight, bias, stride, padding, dilation, groups)
    conv_args = conv_node.args
    with gm.graph.inserting_before(pw_node):
        fused = gm.graph.call_function(fused_op, args=conv_args)
        fused.meta = dict(pw_node.meta)

    pw_node.replace_all_uses_with(fused)
    gm.graph.erase_node(pw_node)
    gm.graph.erase_node(conv_node)
    gm.graph.lint()
    gm.recompile()
    return gm


register_fx_pattern(
    "conv_epilogue",
    _match_conv_epilogue,
    _rewrite_conv_epilogue,
    priority=9,
)
