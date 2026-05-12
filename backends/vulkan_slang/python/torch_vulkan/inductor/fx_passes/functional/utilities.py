"""Small utility FX passes — copy removal, topological resort, scaled_bmm extern."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _remove_redundant_copy(gm: "torch.fx.GraphModule") -> "torch.fx.GraphModule":
    """Remove `aten.clone.default` nodes that copy a tensor to itself (same device, same dtype).

    The graph compiler sometimes inserts redundant clone nodes when views are
    materialized; on Vulkan every tensor from the opaque allocator is already
    contiguous, so these clones produce a second allocation + GPU copy for zero
    semantic benefit. We remove them, letting the downstream use the original
    tensor directly.
    """
    import torch
    from torch.fx import Node

    aten = torch.ops.aten
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target != aten.clone.default:
            continue
        inp = node.args[0]
        if not isinstance(inp, Node):
            continue
        in_meta = inp.meta.get("val") if isinstance(inp, Node) else None
        out_meta = node.meta.get("val")
        if (
            in_meta is not None
            and out_meta is not None
            and in_meta.device == out_meta.device
            and in_meta.dtype == out_meta.dtype
            and in_meta.shape == out_meta.shape
        ):
            node.replace_all_uses_with(inp)
            gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()
    return gm


def _scaled_bmm_extern(q, k, scale: float):
    """Runtime dispatch for the fused scaled bmm: scale * (q @ k.T)."""
    import torch_vulkan
    return torch_vulkan.scaled_bmm(q, k, scale)


def _topological_resort(gm: "torch.fx.GraphModule") -> None:
    """Re-order ``gm.graph`` so all nodes appear after their inputs.

    Used after rewrites that produce a graph with edges pointing backwards in
    the linked-list order (e.g. when the new fused ops live near a late
    dependency, but original consumers — now retargeted to the fused outputs
    — still sit at their old earlier positions). FX's ``Graph.lint`` rejects
    such graphs; we fix them by Kahn's algorithm.
    """
    nodes = list(gm.graph.nodes)
    in_deg = {n: 0 for n in nodes}
    for n in nodes:
        for u in n.users:
            if u in in_deg:
                in_deg[u] += 1

    ready = [n for n in nodes if in_deg[n] == 0]
    ordered: list = []
    while ready:
        n = ready.pop(0)
        ordered.append(n)
        for u in n.users:
            if u not in in_deg:
                continue
            in_deg[u] -= 1
            if in_deg[u] == 0:
                ready.append(u)
    if len(ordered) != len(nodes):
        return  # cycle — shouldn't happen, leave as-is

    for i in range(1, len(ordered)):
        prev, cur = ordered[i - 1], ordered[i]
        if cur.prev is not prev:
            prev.append(cur)
