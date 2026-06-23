"""qkv_cat — fuse three mm/addmm ops sharing the same input into one fused
mm + split.

Priority 8 — runs before individual mm fusions (mm_add_to_addmm at 5,
matmul_epilogue at -1).
"""
from __future__ import annotations

from typing import Any, Iterable

import torch
from torch.fx import GraphModule, Node

from .registry import register_fx_pattern


def _match_qkv_cat(gm: GraphModule) -> Iterable[tuple[Node, dict[str, Any]]]:
    """Match three ``mm`` or ``addmm`` nodes sharing the same input tensor.

    Returns the *last* of the three nodes as root so the rewrite inserts
    the fused op after all three."""
    aten = torch.ops.aten

    # Collect all mm/addmm nodes.
    mm_nodes: list[Node] = []
    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        if node.target not in (aten.mm.default, aten.addmm.default):
            continue
        mm_nodes.append(node)

    # Group by input activation.
    by_input: dict[Node, list[Node]] = {}
    for n in mm_nodes:
        if n.target == aten.mm.default:
            if len(n.args) >= 2 and isinstance(n.args[0], Node):
                by_input.setdefault(n.args[0], []).append(n)
        elif n.target == aten.addmm.default:
            if len(n.args) >= 3 and isinstance(n.args[1], Node):
                by_input.setdefault(n.args[1], []).append(n)

    for inp, group in by_input.items():
        if len(group) != 3:
            continue

        # All must be the same flavour.
        if all(n.target == aten.mm.default for n in group):
            kind = "mm"
        elif all(n.target == aten.addmm.default for n in group):
            kind = "addmm"
        else:
            continue

        # Collect weights (and biases) preserving user ordering.
        weights: list[Node] = []
        biases: list[Node] = []
        ok = True
        for n in group:
            if kind == "mm":
                w = n.args[1]
                if not isinstance(w, Node):
                    ok = False
                    break
                weights.append(w)
            else:
                w = n.args[2]
                b = n.args[0]
                if not isinstance(w, Node) or not isinstance(b, Node):
                    ok = False
                    break
                weights.append(w)
                biases.append(b)
        if not ok:
            continue

        # Shape sanity: input must be at least 2-D with reasonable dims.
        # mat2 in mm/addmm has shape (K, N); the output feature dim is
        # the LAST axis (`shape[-1]`).  The earlier `shape[0]` read was
        # K, not N — the check only worked for square weights.
        inp_val = getattr(inp, "meta", {}).get("val")
        if inp_val is None or inp_val.dim() < 2:
            continue
        M = int(inp_val.shape[-2])
        K = int(inp_val.shape[-1])
        w_val = getattr(weights[0], "meta", {}).get("val")
        N_est = (
            int(w_val.shape[-1])
            if w_val is not None and w_val.dim() >= 2
            else 0
        )
        if M < 4 or K < 8 or N_est < 8:
            continue
        # Cache N for the rewrite to size slices correctly.
        N_per = N_est

        # Pick the last node in graph order as the root for insertion.
        root = max(group, key=lambda n: list(gm.graph.nodes).index(n))

        yield (
            root,
            {
                "inp": inp,
                "kind": kind,
                "weights": weights,
                "biases": biases,
                "group": group,
                "N_per": N_per,
            },
        )


def _rewrite_qkv_cat(gm: GraphModule, root: Node, ctx: dict[str, Any]) -> GraphModule:
    """Replace three mm/addmm nodes with one fused mm/addmm + split."""
    aten = torch.ops.aten
    from ..eager_patches import _ensure_qkv_cat_op_registered

    cat_op = _ensure_qkv_cat_op_registered()
    inp: Node = ctx["inp"]
    kind: str = ctx["kind"]
    weights: list[Node] = ctx["weights"]
    biases: list[Node] = ctx["biases"]
    group: list[Node] = ctx["group"]
    N_per: int = ctx["N_per"]

    # Concatenate weights along the OUTPUT feature dim of mat2 (dim=-1
    # of `(K, N)`).  Earlier code used dim=0 (K-axis) which produced an
    # invalid stack of weights and silently misroutes the matmul output.
    with gm.graph.inserting_before(root):
        cat_w = gm.graph.call_function(
            cat_op, args=(weights[0], weights[1], weights[2], -1)
        )

        if kind == "addmm":
            cat_b = gm.graph.call_function(
                cat_op, args=(biases[0], biases[1], biases[2], 0)
            )
            fused = gm.graph.call_function(aten.addmm.default, args=(cat_b, inp, cat_w))
        else:
            fused = gm.graph.call_function(aten.mm.default, args=(inp, cat_w))
        fused.meta = dict(root.meta)

    # Replace each original mm/addmm with `aten.slice.Tensor(fused, -1,
    # i*N, (i+1)*N)`.  Using `slice` rather than `split.Tensor + getitem`
    # avoids tuple-aware downstream lowering and matches the form in
    # `_fuse_qkv_linears` (the test-facing helper).  LLONG_MAX for the
    # last slice is the conventional sentinel.
    offsets = [
        (0, N_per),
        (N_per, 2 * N_per),
        (2 * N_per, 9223372036854775807),
    ]
    for i, n in enumerate(group):
        start, end = offsets[i]
        with gm.graph.inserting_after(fused):
            slc = gm.graph.call_function(
                aten.slice.Tensor, args=(fused, -1, start, end)
            )
            slc.meta = dict(n.meta)
        n.replace_all_uses_with(slc)
        gm.graph.erase_node(n)

    # Fix topological order.
    from ..functional.utilities import _topological_resort

    _topological_resort(gm)
    gm.graph.lint()
    gm.recompile()
    return gm


register_fx_pattern(
    "qkv_cat",
    _match_qkv_cat,
    _rewrite_qkv_cat,
    priority=8,
)
