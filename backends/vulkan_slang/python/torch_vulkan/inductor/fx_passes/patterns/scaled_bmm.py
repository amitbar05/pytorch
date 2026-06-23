"""scaled_bmm — fuse ``scale * bmm(q, transpose(k, -2, -1))`` into
``scaled_bmm(q, k, scale)``.

Priority 15 — runs alongside conv_im2col.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch
from torch.fx import GraphModule, Node

from ._common import _template_key_bmm
from .registry import register_fx_pattern


def _match_scaled_bmm(gm: GraphModule) -> Iterable[tuple[Node, dict[str, Any]]]:
    """Match ``scale * bmm(q, k.T)`` where the transpose dims are (-2, -1)
    (or the AOT-lowered ``permute([0, 2, 1])``), and scale is a Python
    numeric constant."""
    aten = torch.ops.aten

    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in (
            aten.mul.Tensor,
            aten.mul.Scalar,
        ):
            continue

        lhs, rhs = node.args[0], node.args[1]

        bmm_node, scale_val = None, None
        if (
            isinstance(lhs, Node)
            and lhs.op == "call_function"
            and lhs.target == aten.bmm.default
            and len(lhs.users) == 1
            and not isinstance(rhs, Node)
        ):
            bmm_node, scale_val = lhs, rhs
        elif (
            isinstance(rhs, Node)
            and rhs.op == "call_function"
            and rhs.target == aten.bmm.default
            and len(rhs.users) == 1
            and not isinstance(lhs, Node)
        ):
            bmm_node, scale_val = rhs, lhs

        if bmm_node is None:
            continue

        try:
            scale_f = float(scale_val)
        except (TypeError, ValueError):
            continue

        q_node, k_t_node = bmm_node.args

        if not (
            isinstance(k_t_node, Node)
            and k_t_node.op == "call_function"
            and k_t_node.target in (aten.transpose.int, aten.permute.default)
            and len(k_t_node.users) == 1
        ):
            continue

        t_args = k_t_node.args
        if k_t_node.target == aten.transpose.int:
            if len(t_args) < 3:
                continue
            k_node = t_args[0]
            dim0 = int(t_args[1]) if not isinstance(t_args[1], Node) else None
            dim1 = int(t_args[2]) if not isinstance(t_args[2], Node) else None
            if dim0 is None or dim1 is None:
                continue
            k_val = getattr(k_node, "meta", {}).get("val")
            if k_val is not None:
                ndim = k_val.dim()
                dim0 = dim0 if dim0 >= 0 else ndim + dim0
                dim1 = dim1 if dim1 >= 0 else ndim + dim1
            if not ({dim0, dim1} == {1, 2} or {dim0, dim1} == {-2, -1}):
                continue
        else:
            # permute path
            if len(t_args) < 2:
                continue
            k_node = t_args[0]
            perm = t_args[1]
            if not isinstance(perm, (list, tuple)):
                continue
            k_val = getattr(k_node, "meta", {}).get("val")
            if k_val is None or k_val.dim() != 3:
                continue
            if list(perm) != [0, 2, 1]:
                continue

        yield (
            node,
            {
                "q": q_node,
                "k": k_node,
                "scale": scale_f,
                "bmm_node": bmm_node,
                "k_t_node": k_t_node,
                "mul_node": node,
            },
        )


def _rewrite_scaled_bmm(
    gm: GraphModule, root: Node, ctx: dict[str, Any]
) -> GraphModule:
    """Replace ``scale * bmm(q, k.T)`` with ``scaled_bmm(q, k, scale)``."""
    from ..eager_patches import _ensure_scaled_bmm_op_registered

    scaled_op = _ensure_scaled_bmm_op_registered()
    q: Node = ctx["q"]
    k: Node = ctx["k"]
    scale: float = ctx["scale"]
    bmm_node: Node = ctx["bmm_node"]
    k_t_node: Node = ctx["k_t_node"]

    with gm.graph.inserting_before(root):
        fused = gm.graph.call_function(scaled_op, args=(q, k, scale))
        fused.meta = dict(root.meta)

    root.replace_all_uses_with(fused)
    gm.graph.erase_node(root)
    gm.graph.erase_node(bmm_node)
    if len(k_t_node.users) == 0:
        gm.graph.erase_node(k_t_node)

    gm.graph.lint()
    gm.recompile()
    return gm


register_fx_pattern(
    "scaled_bmm",
    _match_scaled_bmm,
    _rewrite_scaled_bmm,
    template_key_fn=lambda ctx: _template_key_bmm(
        ctx.get("q", None).meta.get("val").dtype
        if ctx.get("q") is not None and hasattr(ctx["q"], "meta")
        else None
    ),
    priority=15,
)
