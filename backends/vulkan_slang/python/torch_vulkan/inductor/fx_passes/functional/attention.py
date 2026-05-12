"""FX passes for attention fusion — SDPA→flash_attention, QKV linear fusion."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _fuse_sdpa_to_flash_attention(gm: "torch.fx.GraphModule") -> "torch.fx.GraphModule":
    """Pattern: aten.scaled_dot_product_attention -> torch_vulkan.flash_attention_fused.

    Routes the standard SDPA op to the eager Vulkan flash-attention extern
    when the head_dim qualifies (32, 64, 128) and the attention shape is
    4-D (B, H, M, D). On shape mismatch, leaves the SDPA call untouched
    so it falls through to the upstream decomposition.
    """
    import torch
    from torch.fx import Node

    aten = torch.ops.aten
    sdpa_targets = {aten.scaled_dot_product_attention.default}

    changed = True
    while changed:
        changed = False
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
                scale = 1.0 / (head_dim ** 0.5)
            try:
                scale_f = float(scale)
            except (TypeError, ValueError):
                continue
            is_causal_b = bool(is_causal)

            from ..eager_patches import _ensure_flash_attention_op_registered
            flash_op = _ensure_flash_attention_op_registered()
            with gm.graph.inserting_before(node):
                fused = gm.graph.call_function(
                    flash_op, args=(q, k, v, scale_f, is_causal_b)
                )
                fused.meta = dict(node.meta)
            node.replace_all_uses_with(fused)
            gm.graph.erase_node(node)
            changed = True
            break

    gm.graph.lint()
    gm.recompile()
    return gm


def _fuse_qkv_linears(gm: "torch.fx.GraphModule") -> "torch.fx.GraphModule":
    """Pattern: 3 (mm | addmm) ops sharing the same input -> 1 fused mm + slices.

    For QKV projection in transformers, three independent ``mm(x, w_qT)`` (or
    ``addmm(b, x, w_qT)``) calls all read the same activation ``x``. We
    concatenate the three weight matrices along the K-output dim into one
    ``cat_w (K, Hq+Hk+Hv)``, run a single ``mm(x, cat_w)`` (or addmm with
    concatenated bias), then ``aten.split.Tensor`` the result into three
    views.

    This cuts two redundant reads of ``x`` (which is typically the dominant
    tensor — ``B*S*H`` elements) and replaces three matmul dispatches with one.

    Only matches when all three linears are the same flavour (all mm or all
    addmm), share the same input tensor node, and have non-degenerate
    shapes (M >= 4, K >= 8, N >= 8).  The fusion saves 2× redundant reads
    of the input activation, so even small shapes benefit; the lower
    bounds rule out true degenerate cases (e.g. 1×1 mm).
    """
    import torch
    from torch.fx import Node

    aten = torch.ops.aten

    changed = True
    while changed:
        changed = False
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
            if all(n.target == aten.mm.default for n in group):
                kind = "mm"
            elif all(n.target == aten.addmm.default for n in group):
                kind = "addmm"
            else:
                continue

            # Collect weights (and biases) in consistent order.
            weights = []
            biases = []
            for n in group:
                if kind == "mm":
                    w = n.args[1]
                    if not isinstance(w, Node):
                        break
                    weights.append(w)
                else:
                    w = n.args[2]
                    b = n.args[0]
                    if not isinstance(w, Node) or not isinstance(b, Node):
                        break
                    weights.append(w)
                    biases.append(b)
            else:
                shape_ok = True
                for n in group:
                    v = getattr(inp, "meta", {}).get("val")
                    if v is None or v.dim() < 2:
                        shape_ok = False
                        break
                    M = int(v.shape[-2])
                    K = int(v.shape[-1])
                    # mat2 in mm/addmm has shape (K, N); the output
                    # feature dim is the LAST axis (`shape[-1]`).  The
                    # earlier `shape[0]` read was wrong (that's K) and
                    # only worked for square weights.
                    w_val = getattr(weights[0], "meta", {}).get("val")
                    N_est = (
                        int(w_val.shape[-1])
                        if w_val is not None and w_val.dim() >= 2
                        else 0
                    )
                    if M < 4 or K < 8 or N_est < 8:
                        shape_ok = False
                        break
                if not shape_ok:
                    continue

                from ..eager_patches import _ensure_qkv_cat_op_registered
                cat_w = gm.graph.call_function(
                    _ensure_qkv_cat_op_registered(),
                    args=(weights[0], weights[1], weights[2], 1),
                )
                if kind == "addmm":
                    cat_b = gm.graph.call_function(
                        _ensure_qkv_cat_op_registered(),
                        args=(biases[0], biases[1], biases[2], 0),
                    )

                with gm.graph.inserting_after(group[-1]):
                    if kind == "mm":
                        fused = gm.graph.call_function(
                            aten.mm.default, args=(inp, cat_w)
                        )
                    else:
                        fused = gm.graph.call_function(
                            aten.addmm.default, args=(cat_b, inp, cat_w)
                        )
                    fused.meta = dict(group[-1].meta)

                # Each consumer reads a contiguous slice of the fused
                # output along the last (output-feature) dim.  Use
                # `aten.slice.Tensor(input, dim, start, end)` so the
                # downstream consumer (often a `view`/`reshape` reaping
                # contiguous head dims) sees exactly the per-Q/K/V chunk
                # the original mm produced — `aten.split.Tensor` yields a
                # tuple and would require an extra `getitem` per use.
                # mat2 has shape (K, N); the output dim is `shape[-1]`.
                # Fixed offsets per group member: 0..N, N..2N, 2N..3N.
                # The slice end uses 9223372036854775807 (LLONG_MAX) for
                # the third slice as is conventional in Inductor.
                w0_val = getattr(weights[0], "meta", {}).get("val")
                N_per = (
                    int(w0_val.shape[-1])
                    if w0_val is not None and w0_val.dim() >= 2
                    else 0
                )
                offsets = [
                    (0, N_per),
                    (N_per, 2 * N_per),
                    (2 * N_per, 9223372036854775807),
                ]
                for i, n in enumerate(group):
                    start, end = offsets[i]
                    with gm.graph.inserting_after(fused):
                        slc = gm.graph.call_function(
                            aten.slice.Tensor,
                            args=(fused, -1, start, end),
                        )
                        # Preserve the original mm's val for shape prop.
                        slc.meta = dict(n.meta)
                        n.replace_all_uses_with(slc)
                        gm.graph.erase_node(n)

            changed = True
            break

    from .utilities import _topological_resort
    _topological_resort(gm)
    gm.graph.lint()
    gm.recompile()
    return gm
