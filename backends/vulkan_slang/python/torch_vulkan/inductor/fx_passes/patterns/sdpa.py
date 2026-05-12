"""sdpa — fuse ``aten.scaled_dot_product_attention`` into
``flash_attention_fused``.

Priority 12 — runs before bmm-oriented patterns so the SDPA is caught
before its decomposition to bmm + softmax.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch
from torch.fx import GraphModule, Node

from .registry import register_fx_pattern


def _match_sdpa(gm: GraphModule) -> Iterable[tuple[Node, dict[str, Any]]]:
    """Match ``aten.scaled_dot_product_attention`` with 4-D inputs and a
    supported head_dim (32, 64, 128).  Skips masked / dropout variants."""
    aten = torch.ops.aten
    sdpa_targets = {aten.scaled_dot_product_attention.default}

    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in sdpa_targets:
            continue
        if len(node.args) < 3:
            continue
        q, k, v = node.args[:3]
        if not all(isinstance(t, Node) for t in (q, k, v)):
            continue
        q_val = getattr(q, "meta", {}).get("val")
        if q_val is None or q_val.dim() != 4:
            continue
        head_dim = int(q_val.shape[-1])
        if head_dim not in (32, 64, 128):
            continue

        kwargs = dict(node.kwargs or {})
        scale = kwargs.get("scale", None)
        is_causal = kwargs.get("is_causal", False)
        attn_mask = kwargs.get("attn_mask", None)
        dropout_p = kwargs.get("dropout_p", 0.0)
        if attn_mask is not None or (dropout_p and float(dropout_p) > 0.0):
            continue
        if scale is None:
            scale = 1.0 / (head_dim**0.5)
        try:
            scale_f = float(scale)
        except (TypeError, ValueError):
            continue
        is_causal_b = bool(is_causal)

        yield (
            node,
            {
                "q": q,
                "k": k,
                "v": v,
                "scale": scale_f,
                "is_causal": is_causal_b,
            },
        )


def _rewrite_sdpa(gm: GraphModule, root: Node, ctx: dict[str, Any]) -> GraphModule:
    """Replace SDPA with ``flash_attention_fused``."""
    from ..eager_patches import _ensure_flash_attention_op_registered

    flash_op = _ensure_flash_attention_op_registered()
    q: Node = ctx["q"]
    k: Node = ctx["k"]
    v: Node = ctx["v"]
    scale: float = ctx["scale"]
    is_causal: bool = ctx["is_causal"]

    with gm.graph.inserting_before(root):
        fused = gm.graph.call_function(flash_op, args=(q, k, v, scale, is_causal))
        fused.meta = dict(root.meta)

    root.replace_all_uses_with(fused)
    gm.graph.erase_node(root)

    gm.graph.lint()
    gm.recompile()
    return gm


register_fx_pattern(
    "sdpa",
    _match_sdpa,
    _rewrite_sdpa,
    priority=12,
)
