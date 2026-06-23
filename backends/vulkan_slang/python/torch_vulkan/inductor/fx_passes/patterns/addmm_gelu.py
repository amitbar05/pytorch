"""addmm_gelu — match the *decomposed* (post-AOT) gelu(addmm) subgraph and
replace with ``torch_vulkan.addmm_gelu_fused``.

Priority 10 runs after ``matmul_epilogue`` (-1) so the specialised gelu
pattern wins over the generic epilogue pass when gelu survives decomposition.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import torch
from torch.fx import GraphModule, Node

from .registry import register_fx_pattern


def _match_addmm_gelu(gm: GraphModule) -> Iterable[tuple[Node, dict[str, Any]]]:
    """Match the exact post-AOT gelu decomposition rooted at an addmm:

    t1 = mul(addmm, 0.7071067811865475)   # GELU_C
    t2 = erf(t1)
    t3 = add(t2, 1.0)
    t4 = mul(addmm, 0.5)
    out = mul(t4, t3)
    """
    aten = torch.ops.aten
    addmm_targets = {aten.addmm.default}
    GELU_C = 0.7071067811865475

    def _is_const(n, value: float, rtol: float = 1e-7) -> bool:
        return isinstance(n, (int, float)) and math.isclose(
            float(n), value, rel_tol=rtol, abs_tol=1e-9
        )

    def _other_arg(node: Node, target: Node):
        a, b = node.args[0], node.args[1]
        if a is target:
            return b
        if b is target:
            return a
        return None

    for erf_node in list(gm.graph.nodes):
        if erf_node.op != "call_function" or erf_node.target != aten.erf.default:
            continue
        if len(erf_node.users) != 1:
            continue

        t1 = erf_node.args[0]
        if (
            not isinstance(t1, Node)
            or t1.op != "call_function"
            or t1.target != aten.mul.Tensor
            or len(t1.users) != 1
        ):
            continue

        addmm_node = None
        for cand in (t1.args[0], t1.args[1]):
            if (
                isinstance(cand, Node)
                and cand.op == "call_function"
                and cand.target in addmm_targets
            ):
                addmm_node = cand
                break
        if addmm_node is None:
            continue
        const_c = _other_arg(t1, addmm_node)
        if not _is_const(const_c, GELU_C, rtol=1e-4):
            continue

        t3 = next(iter(erf_node.users))
        if (
            t3.op != "call_function"
            or t3.target != aten.add.Tensor
            or len(t3.users) != 1
        ):
            continue
        const_one = _other_arg(t3, erf_node)
        if not _is_const(const_one, 1.0, rtol=1e-7):
            continue

        t4 = None
        t5 = None
        for u in addmm_node.users:
            if u is t1:
                continue
            if (
                u.op == "call_function"
                and u.target == aten.mul.Tensor
                and len(u.users) == 1
            ):
                other = _other_arg(u, addmm_node)
                if _is_const(other, 0.5, rtol=1e-7):
                    t4 = u
                    cand_t5 = next(iter(u.users))
                    if (
                        cand_t5.op == "call_function"
                        and cand_t5.target == aten.mul.Tensor
                        and (cand_t5.args[0] is t3 or cand_t5.args[1] is t3)
                        and (cand_t5.args[0] is u or cand_t5.args[1] is u)
                    ):
                        t5 = cand_t5
                        break
        if t4 is None or t5 is None:
            continue

        if set(addmm_node.users) != {t1, t4}:
            continue

        if len(addmm_node.args) != 3:
            continue
        bias, mat1, mat2 = addmm_node.args[0], addmm_node.args[1], addmm_node.args[2]
        if not all(isinstance(x, Node) for x in (bias, mat1, mat2)):
            continue
        alpha = addmm_node.kwargs.get("alpha", 1)
        beta = addmm_node.kwargs.get("beta", 1)
        if alpha != 1 or beta != 1:
            continue

        mat1_val = getattr(mat1, "meta", {}).get("val")
        mat2_val = getattr(mat2, "meta", {}).get("val")
        bias_val = getattr(bias, "meta", {}).get("val")
        if mat1_val is None or mat2_val is None or bias_val is None:
            continue
        if mat1_val.dim() != 2 or mat2_val.dim() != 2 or bias_val.dim() != 1:
            continue
        if mat1_val.dtype not in (torch.float32, torch.float16):
            continue
        if mat1_val.dtype != mat2_val.dtype or mat1_val.dtype != bias_val.dtype:
            continue

        yield (
            t5,
            {
                "addmm_node": addmm_node,
                "bias": bias,
                "mat1": mat1,
                "mat2": mat2,
                "erf_node": erf_node,
                "t1": t1,
                "t3": t3,
                "t4": t4,
                "t5": t5,
            },
        )


def _rewrite_addmm_gelu(
    gm: GraphModule, root: Node, ctx: dict[str, Any]
) -> GraphModule:
    """Replace the decomposed gelu(addmm) subgraph with a single
    ``addmm_gelu_fused`` call."""
    from ..eager_patches import _ensure_addmm_gelu_op_registered

    fused_op = _ensure_addmm_gelu_op_registered()
    bias: Node = ctx["bias"]
    mat1: Node = ctx["mat1"]
    mat2: Node = ctx["mat2"]

    with gm.graph.inserting_before(root):
        fused = gm.graph.call_function(fused_op, args=(bias, mat1, mat2))
        fused.meta = dict(root.meta)

    root.replace_all_uses_with(fused)
    for dead in (
        root,
        ctx["t4"],
        ctx["t3"],
        ctx["erf_node"],
        ctx["t1"],
        ctx["addmm_node"],
    ):
        if dead is not None and len(dead.users) == 0:
            gm.graph.erase_node(dead)

    gm.graph.lint()
    gm.recompile()
    return gm


register_fx_pattern(
    "addmm_gelu",
    _match_addmm_gelu,
    _rewrite_addmm_gelu,
    priority=10,
)
