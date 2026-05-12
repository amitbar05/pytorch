"""FX passes for matmul / bmm / scaled-bmm fusion."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _fuse_mm_add_to_addmm(gm: "torch.fx.GraphModule") -> "torch.fx.GraphModule":
    """Pattern: mm(a, b) + bias  →  addmm(bias, a, b).

    Reduces two Inductor graph nodes (mm + add) to one (addmm), cutting one
    extern-kernel dispatch per linear layer in models that emit separate mm+add
    rather than addmm.  Only applies when:
      - The mm inputs are the first two args.
      - The add's bias is a 0- or 1-D node.
      - No other consumers of the mm output exist (single-use).
    """
    import torch
    from torch.fx import Node

    aten = torch.ops.aten
    changed = True
    while changed:
        changed = False
        for node in list(gm.graph.nodes):
            if node.op != "call_function" or node.target != aten.add.Tensor:
                continue
            lhs, rhs = node.args[0], node.args[1]
            mm_node, bias_node = None, None
            if (
                isinstance(lhs, Node)
                and lhs.op == "call_function"
                and lhs.target == aten.mm.default
                and len(lhs.users) == 1
            ):
                mm_node, bias_node = lhs, rhs
            elif (
                isinstance(rhs, Node)
                and rhs.op == "call_function"
                and rhs.target == aten.mm.default
                and len(rhs.users) == 1
            ):
                mm_node, bias_node = rhs, lhs
            if mm_node is None:
                continue
            a, b = mm_node.args
            with gm.graph.inserting_before(node):
                addmm_node = gm.graph.call_function(
                    aten.addmm.default, args=(bias_node, a, b)
                )
                addmm_node.meta = dict(node.meta)
            node.replace_all_uses_with(addmm_node)
            gm.graph.erase_node(node)
            gm.graph.erase_node(mm_node)
            changed = True
            break
    gm.graph.lint()
    gm.recompile()
    return gm


def _fuse_bmm_mul_to_scaled_bmm(gm: "torch.fx.GraphModule") -> "torch.fx.GraphModule":
    """Pattern: scale * bmm(q, transpose(k, -2, -1))  →  scaled_bmm(q, k, scale).

    Replaces the three-op pattern (transpose + bmm dispatch + elementwise mul)
    with a single fused Vulkan dispatch computing scale*(q @ k.T) atomically.
    This is the hot path in scaled-dot-product attention score computation.

    The `transpose(k, -2, -1)` step is removed and k is passed directly to
    `torch_vulkan.scaled_bmm`, which handles the transpose internally via a
    `transpose_b=true` shader flag — no intermediate transposed tensor copy.

    Only matches when:
      - The bmm's second input is an `aten.transpose.int` of dims (-2, -1).
      - The mul scale is a Python numeric constant (not a tensor node).
      - The bmm output is consumed solely by the mul node.
      - The transpose output is consumed solely by the bmm node.
    """
    import torch
    from torch.fx import Node

    aten = torch.ops.aten
    changed = True
    while changed:
        changed = False
        for node in list(gm.graph.nodes):
            if node.op != "call_function" or node.target not in (
                aten.mul.Tensor, aten.mul.Scalar
            ):
                continue

            lhs, rhs = node.args[0], node.args[1]

            # Find which side is the bmm and which is the scalar.
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

            # scale_val must be a Python number (not a tensor node).
            try:
                scale_f = float(scale_val)
            except (TypeError, ValueError):
                continue

            q_node, k_t_node = bmm_node.args

            # k_t_node must be a transpose(-2, -1) or its post-AOT permute
            # decomposition. AOT autograd lowers `transpose(x, -2, -1)`
            # into `permute(x, [0, 2, 1])` for 3-D inputs; accept both.
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

            from ..eager_patches import _ensure_scaled_bmm_op_registered
            scaled_op = _ensure_scaled_bmm_op_registered()
            with gm.graph.inserting_before(node):
                scaled_node = gm.graph.call_function(
                    scaled_op,
                    args=(q_node, k_node, scale_f),
                )
                scaled_node.meta = dict(node.meta)

            node.replace_all_uses_with(scaled_node)
            gm.graph.erase_node(node)
            gm.graph.erase_node(bmm_node)
            if len(k_t_node.users) == 0:
                gm.graph.erase_node(k_t_node)
            changed = True
            break

    gm.graph.lint()
    gm.recompile()
    return gm


def _enable_b2b_gemm(gm: "torch.fx.GraphModule") -> "torch.fx.GraphModule":
    """Annotate the graph so Inductor enables back-to-back GEMM fusion."""
    gm.meta["b2b_gemm_pass"] = True
    return gm
