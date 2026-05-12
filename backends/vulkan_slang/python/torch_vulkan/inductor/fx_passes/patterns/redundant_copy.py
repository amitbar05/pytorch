"""redundant_copy — remove ``aten.clone`` nodes that are true no-ops
(same device, dtype, shape).

Priority 2 — cheap cleanup that simplifies the graph for later passes.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch
from torch.fx import GraphModule, Node

from .registry import register_fx_pattern


def _match_redundant_copy(
    gm: GraphModule,
) -> Iterable[tuple[Node, dict[str, Any]]]:
    """Match ``aten.clone.default`` whose output is semantically identical
    to its input (same device, dtype, shape)."""
    aten = torch.ops.aten

    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target != aten.clone.default:
            continue
        inp = node.args[0]
        if not isinstance(inp, Node):
            continue
        in_meta = inp.meta.get("val") if hasattr(inp, "meta") else None
        out_meta = node.meta.get("val")
        if (
            in_meta is not None
            and out_meta is not None
            and in_meta.device == out_meta.device
            and in_meta.dtype == out_meta.dtype
            and in_meta.shape == out_meta.shape
        ):
            yield (node, {"inp": inp})


def _rewrite_redundant_copy(
    gm: GraphModule, root: Node, ctx: dict[str, Any]
) -> GraphModule:
    """Replace the clone node with its input directly."""
    inp: Node = ctx["inp"]
    root.replace_all_uses_with(inp)
    gm.graph.erase_node(root)
    gm.graph.lint()
    gm.recompile()
    return gm


register_fx_pattern(
    "redundant_copy",
    _match_redundant_copy,
    _rewrite_redundant_copy,
    priority=2,
)
