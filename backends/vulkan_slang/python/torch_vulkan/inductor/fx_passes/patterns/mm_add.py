"""mm_add — fold ``mm(a, b) + bias`` into ``addmm(bias, a, b)``.

Priority 5 — runs after epilogue and QKV fusions, folding any remaining
mm+add pairs into a single addmm dispatch.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch
from torch.fx import GraphModule, Node

from .registry import register_fx_pattern


def _match_mm_add_to_addmm(
    gm: GraphModule,
) -> Iterable[tuple[Node, dict[str, Any]]]:
    """Match ``mm(a, b) + bias`` where the mm has exactly one user (the add)
    and bias is a 0-D or 1-D tensor node."""
    aten = torch.ops.aten

    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target != aten.add.Tensor:
            continue
        lhs, rhs = node.args[0], node.args[1]
        mm_node = None
        bias_node = None
        if (
            isinstance(lhs, Node)
            and lhs.op == "call_function"
            and lhs.target == aten.mm.default
            and len(lhs.users) == 1
        ):
            mm_node = lhs
            bias_node = rhs
        elif (
            isinstance(rhs, Node)
            and rhs.op == "call_function"
            and rhs.target == aten.mm.default
            and len(rhs.users) == 1
        ):
            mm_node = rhs
            bias_node = lhs

        if mm_node is None:
            continue

        a, b = mm_node.args
        if not isinstance(bias_node, Node):
            continue

        yield (
            node,
            {
                "a": a,
                "b": b,
                "bias": bias_node,
                "mm_node": mm_node,
                "add_node": node,
            },
        )


def _rewrite_mm_add_to_addmm(
    gm: GraphModule, root: Node, ctx: dict[str, Any]
) -> GraphModule:
    """Replace ``mm(a, b) + bias`` with ``addmm(bias, a, b)``."""
    aten = torch.ops.aten
    a: Node = ctx["a"]
    b: Node = ctx["b"]
    bias: Node = ctx["bias"]
    mm_node: Node = ctx["mm_node"]

    with gm.graph.inserting_before(root):
        addmm_node = gm.graph.call_function(aten.addmm.default, args=(bias, a, b))
        addmm_node.meta = dict(root.meta)

    root.replace_all_uses_with(addmm_node)
    gm.graph.erase_node(root)
    gm.graph.erase_node(mm_node)

    gm.graph.lint()
    gm.recompile()
    return gm


register_fx_pattern(
    "mm_add_to_addmm",
    _match_mm_add_to_addmm,
    _rewrite_mm_add_to_addmm,
    priority=5,
)
