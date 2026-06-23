"""relu_clamp_min — canonicalise ``relu(x)`` to ``clamp_min(x, 0)``.

Priority 0 — runs early so downstream fusions can match a single pointwise
signature (clamp_min) instead of relu.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Union

import torch
from torch.fx import GraphModule, Node

from .registry import register_fx_pattern


def _match_single_op(
    gm: GraphModule, target: Union[Callable, str]
) -> Iterable[tuple[Node, dict[str, Any]]]:
    """Generic single-op matcher — yields ``(node, {})`` for every
    ``call_function`` node whose ``.target`` equals *target*."""
    for node in list(gm.graph.nodes):
        if node.op == "call_function" and node.target == target:
            yield (node, {})


def _match_relu_to_clamp_min(
    gm: GraphModule,
) -> Iterable[tuple[Node, dict[str, Any]]]:
    """Match ``aten.relu.default`` nodes."""
    aten = torch.ops.aten
    return _match_single_op(gm, aten.relu.default)


def _rewrite_relu_to_clamp_min(
    gm: GraphModule, root: Node, ctx: dict[str, Any]
) -> GraphModule:
    """Replace ``relu(x)`` with ``clamp_min(x, 0)``."""
    aten = torch.ops.aten
    inp = root.args[0]

    with gm.graph.inserting_before(root):
        clamped = gm.graph.call_function(aten.clamp_min.default, args=(inp, 0))
        clamped.meta = dict(root.meta)

    root.replace_all_uses_with(clamped)
    gm.graph.erase_node(root)
    gm.graph.lint()
    gm.recompile()
    return gm


register_fx_pattern(
    "relu_to_clamp_min",
    _match_relu_to_clamp_min,
    _rewrite_relu_to_clamp_min,
    priority=0,
)
