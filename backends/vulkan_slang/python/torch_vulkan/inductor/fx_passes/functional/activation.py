"""FX passes for activation fusion — addmm_gelu, silu_mul→swiglu."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _fuse_addmm_gelu(gm: "torch.fx.GraphModule") -> "torch.fx.GraphModule":
    """Pattern: gelu(addmm(bias, mat1, mat2))  →  torch_vulkan.addmm_gelu_fused.

    PF.5 epilogue fusion. Collapses two dispatches (addmm + gelu) into one
    (the Slang tiled-addmm shader with `epilogue="OpGELU"` baked in).

    By the time the post-grad custom pass runs, the upstream
    decomposition has expanded `aten.gelu.default` into the exact
    (non-approximate) primitives:

        t1 = mul(addmm, 0.7071067811865475)
        t2 = erf(t1)
        t3 = add(t2, 1.0)
        t4 = mul(addmm, 0.5)
        out = mul(t4, t3)

    (`mul`'s arg order may swap on either side; `add` may carry the
    constant on either side too.) We walk this graph from each `aten.erf`
    node back to its addmm and forward to the final `mul`, and replace
    the entire subgraph with a single `addmm_gelu_fused` call.

    Pre-conditions:
      - `addmm` is consumed *exactly* by t1 and t4 (the two gelu-decomp
        muls) — any other consumer means we'd lose a needed value.
      - All decomp ops have a single user (no shared intermediates).
      - Inputs are 2-D float32/float16 tensors (matches the Slang
        template's binding layout) with a 1-D bias.
    """
    import math
    import torch
    from torch.fx import Node

    aten = torch.ops.aten
    addmm_targets = {aten.addmm.default}
    GELU_C = 0.7071067811865475

    def _is_const(n, value: float, rtol: float = 1e-7) -> bool:
        return isinstance(n, (int, float)) and math.isclose(float(n), value, rel_tol=rtol, abs_tol=1e-9)

    def _other_arg(node: Node, target: Node):
        a, b = node.args[0], node.args[1]
        if a is target:
            return b
        if b is target:
            return a
        return None

    changed = True
    while changed:
        changed = False
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

            from ..eager_patches import _ensure_addmm_gelu_op_registered
            fused_op = _ensure_addmm_gelu_op_registered()
            with gm.graph.inserting_before(t5):
                fused = gm.graph.call_function(fused_op, args=(bias, mat1, mat2))
                fused.meta = dict(t5.meta)

            t5.replace_all_uses_with(fused)
            for dead in (t5, t4, t3, erf_node, t1, addmm_node):
                if len(dead.users) == 0:
                    gm.graph.erase_node(dead)
            changed = True
            break

    gm.graph.lint()
    gm.recompile()
    return gm


def _fuse_silu_mul_to_swiglu(gm: "torch.fx.GraphModule") -> "torch.fx.GraphModule":
    """Pattern: silu(gate) * up  →  torch_vulkan.swiglu_fused(gate, up).

    Three-op pattern (silu + mul) → single fused dispatch. Matches the
    transformer FFN gate path. Only fires when the silu output has exactly
    one user (the mul) and both silu input and the mul's other operand are
    real tensor nodes with matching shapes. Either pre-AOT (`aten.silu`) or
    post-AOT (`aten.silu.default`) form is accepted.
    """
    import torch
    from torch.fx import Node

    aten = torch.ops.aten
    silu_targets = {aten.silu.default}
    mul_targets = {aten.mul.Tensor}

    changed = True
    while changed:
        changed = False
        for node in list(gm.graph.nodes):
            if node.op != "call_function" or node.target not in mul_targets:
                continue
            lhs, rhs = node.args[0], node.args[1]

            silu_node, other_node = None, None
            if (
                isinstance(lhs, Node)
                and lhs.op == "call_function"
                and lhs.target in silu_targets
                and len(lhs.users) == 1
            ):
                silu_node, other_node = lhs, rhs
            elif (
                isinstance(rhs, Node)
                and rhs.op == "call_function"
                and rhs.target in silu_targets
                and len(rhs.users) == 1
            ):
                silu_node, other_node = rhs, lhs

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

            from ..eager_patches import _ensure_swiglu_op_registered
            swiglu_op = _ensure_swiglu_op_registered()
            with gm.graph.inserting_before(node):
                fused = gm.graph.call_function(
                    swiglu_op, args=(gate_node, other_node)
                )
                fused.meta = dict(node.meta)

            node.replace_all_uses_with(fused)
            gm.graph.erase_node(node)
            if len(silu_node.users) == 0:
                gm.graph.erase_node(silu_node)
            changed = True
            break

    gm.graph.lint()
    gm.recompile()
    return gm
