"""swiglu — fuse ``silu(gate) * up`` into ``torch_vulkan.swiglu_fused``.

Priority 20 — one of the last fusions, after QKV and other structural
rewrites have run.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch
from torch.fx import GraphModule, Node

from ._common import _template_key_mm
from .registry import register_fx_pattern


def _match_swiglu(gm: GraphModule) -> Iterable[tuple[Node, dict[str, Any]]]:
    """Match ``silu(gate) * up`` where silu has exactly one user (the mul)
    and gate / up are tensor nodes with matching shapes and dtypes."""
    aten = torch.ops.aten
    silu_targets = {aten.silu.default}
    mul_targets = {aten.mul.Tensor}

    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in mul_targets:
            continue
        lhs, rhs = node.args[0], node.args[1]

        silu_node = None
        other_node = None
        if (
            isinstance(lhs, Node)
            and lhs.op == "call_function"
            and lhs.target in silu_targets
            and len(lhs.users) == 1
        ):
            silu_node = lhs
            other_node = rhs
        elif (
            isinstance(rhs, Node)
            and rhs.op == "call_function"
            and rhs.target in silu_targets
            and len(rhs.users) == 1
        ):
            silu_node = rhs
            other_node = lhs

        if silu_node is None or not isinstance(other_node, Node):
            continue

        gate_node = silu_node.args[0]
        if not isinstance(gate_node, Node):
            continue

        gate_val = getattr(gate_node, "meta", {}).get("val")
        other_val = getattr(other_node, "meta", {}).get("val")
        if gate_val is None or other_val is None:
            continue
        if tuple(gate_val.shape) != tuple(other_val.shape):
            continue
        if gate_val.dtype != other_val.dtype:
            continue

        yield (
            node,
            {
                "gate": gate_node,
                "up": other_node,
                "silu_node": silu_node,
                "mul_node": node,
            },
        )


def _rewrite_swiglu(gm: GraphModule, root: Node, ctx: dict[str, Any]) -> GraphModule:
    """Replace ``silu(gate) * up`` with ``swiglu_fused(gate, up)``."""
    from ..eager_patches import _ensure_swiglu_op_registered

    swiglu_op = _ensure_swiglu_op_registered()
    gate: Node = ctx["gate"]
    up: Node = ctx["up"]
    silu_node: Node = ctx["silu_node"]

    with gm.graph.inserting_before(root):
        fused = gm.graph.call_function(swiglu_op, args=(gate, up))
        fused.meta = dict(root.meta)

    root.replace_all_uses_with(fused)
    gm.graph.erase_node(root)
    if len(silu_node.users) == 0:
        gm.graph.erase_node(silu_node)

    gm.graph.lint()
    gm.recompile()
    return gm


register_fx_pattern(
    "swiglu",
    _match_swiglu,
    _rewrite_swiglu,
    template_key_fn=lambda ctx: _template_key_mm(
        ctx.get("gate", None).meta.get("val").dtype
        if ctx.get("gate") is not None and hasattr(ctx["gate"], "meta")
        else None
    ),
    priority=20,
)
