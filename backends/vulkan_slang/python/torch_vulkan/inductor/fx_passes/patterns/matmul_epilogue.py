"""matmul_epilogue — fuse mm/addmm + pointwise activation into a single
fused matmul+epilogue dispatch.

Priority -1 (runs before almost everything) so that pointwise fusions happen
early, exposing cleaner graphs for later structural rewrites.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

import torch
from torch.fx import GraphModule, Node

from .registry import register_fx_pattern


def _match_matmul_epilogue(gm: GraphModule) -> Iterable[tuple[Node, dict[str, Any]]]:
    """Match ``mm`` or ``addmm`` whose *sole* consumer is a pointwise op:
    relu, gelu (pre-decomp), silu, sigmoid, tanh, hardsigmoid, hardswish,
    leaky_relu, or elu."""
    aten = torch.ops.aten
    mm_targets = {aten.mm.default, aten.addmm.default}
    pointwise_targets = {
        aten.relu.default,
        aten.gelu.default,
        aten.silu.default,
        aten.sigmoid.default,
        aten.tanh.default,
        aten.hardsigmoid.default,
        aten.hardswish.default,
        aten.leaky_relu.default,
        aten.elu.default,
    }

    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        if node.target not in pointwise_targets:
            continue
        pw_node = node
        if len(pw_node.users) == 0:
            continue

        # The pointwise op should have exactly one tensor input.
        pw_args = [
            a for a in pw_node.args if isinstance(a, Node) and a.op == "call_function"
        ]
        if len(pw_args) != 1:
            continue
        mm_node = pw_args[0]

        if mm_node.target not in mm_targets:
            continue
        if len(mm_node.users) != 1:
            continue

        # Collect context.
        ctx: dict[str, Any] = {
            "mm_node": mm_node,
            "pw_node": pw_node,
            "pw_target": pw_node.target,
        }
        if mm_node.target == aten.addmm.default and len(mm_node.args) >= 3:
            ctx["bias"] = mm_node.args[0]
            ctx["mat1"] = mm_node.args[1]
            ctx["mat2"] = mm_node.args[2]
        elif mm_node.target == aten.mm.default and len(mm_node.args) >= 2:
            ctx["mat1"] = mm_node.args[0]
            ctx["mat2"] = mm_node.args[1]
        else:
            continue

        yield (pw_node, ctx)


def _rewrite_matmul_epilogue(
    gm: GraphModule, root: Node, ctx: dict[str, Any]
) -> GraphModule:
    """Replace mm/addmm + pointwise with a fused dispatch.

    The fused op is chosen based on the pointwise activation.  When the
    pointwise is ``gelu`` and the matmul is ``addmm``, the specialised
    ``addmm_gelu_fused`` custom-op is preferred; other combinations
    route through the template registry (falling back to the original
    pattern if no template is registered)."""
    aten = torch.ops.aten
    mm_node: Node = ctx["mm_node"]
    pw_node: Node = ctx["pw_node"]
    pw_target = ctx["pw_target"]

    is_addmm = mm_node.target == aten.addmm.default
    mat1: Node = ctx["mat1"]
    mat2: Node = ctx["mat2"]
    bias: Optional[Node] = ctx.get("bias")

    # Specialised path: addmm + gelu → addmm_gelu_fused
    if is_addmm and pw_target == aten.gelu.default and bias is not None:
        from ..eager_patches import _ensure_addmm_gelu_op_registered

        fused_op = _ensure_addmm_gelu_op_registered()
        with gm.graph.inserting_before(pw_node):
            fused = gm.graph.call_function(fused_op, args=(bias, mat1, mat2))
            fused.meta = dict(pw_node.meta)
        pw_node.replace_all_uses_with(fused)
        gm.graph.erase_node(pw_node)
        gm.graph.erase_node(mm_node)
        gm.graph.lint()
        gm.recompile()
        return gm

    # General path: fused matmul+epilogue via template dispatch (stub).
    # When the template registry is wired up, this extracts a TemplateKey
    # and dispatches.  For now, leave the graph unchanged.
    return gm


register_fx_pattern(
    "matmul_epilogue",
    _match_matmul_epilogue,
    _rewrite_matmul_epilogue,
    priority=-1,
)
