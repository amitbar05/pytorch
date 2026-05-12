"""op_class_fusion — DR.1 fusion-aware scheduler: annotate FX nodes with
``vulkan_fusion_group`` metadata so the Vulkan scheduler can make
template-aware vertical fusion decisions.

Priority 8 — runs AFTER structural rewrites (mm→addmm, conv→im2col+mm)
but BEFORE the scheduler inspects the graph. This pass annotates rather
than rewrites; the scheduler consumes the metadata in
``can_fuse_vertical``.

Patterns:
    * conv_output → bias_add → norm → activation  (FusionGroup.conv_epilogue)
    * mm/addmm → bias_add                         (FusionGroup.mm_bias)
    * log_softmax → nll_loss                      (FusionGroup.softmax_nll)
    * reduction → pointwise_tail                  (FusionGroup.reduction_pointwise)
    * bw_chain — reduction backward + pointwise epilogue (FusionGroup.bw_chain)
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any, Iterable, Optional

import torch
from torch.fx import GraphModule, Node

from .registry import register_fx_pattern


class FusionGroup(Enum):
    """Known template-supported epilogue compositions.

    The scheduler uses these to decide whether vertical fusion is
    legal: nodes that share a fusion group may be fused even when
    base heuristics would reject the merge; nodes that cross an
    op-class boundary (different groups) are blocked from fusing.
    """

    # No fusion group — default, no annotation.
    none = auto()
    # Conv + bias-add + norm + activation as a single epilogue.
    conv_epilogue = auto()
    # mm/addmm + bias-add fused.
    mm_bias = auto()
    # softmax/log_softmax + nll/cross_entropy loss fused.
    softmax_nll = auto()
    # Reduction + pointwise tail (e.g. sum → div → add).
    reduction_pointwise = auto()
    # Backward chain: reduction backward + pointwise epilogue.
    bw_chain = auto()


def _tag_fusion_group(node: Node, group: FusionGroup) -> None:
    """Annotate *node* with a fusion-group tag.

    Uses ``vulkan_fusion_group`` as the meta key so the scheduler's
    ``can_fuse_vertical`` can consume it without importing this module
    (avoids a circular import between scheduling.py and fx_passes).
    """
    if not hasattr(node, "meta"):
        node.meta = {}
    node.meta["vulkan_fusion_group"] = group.name


def _get_fusion_group(node: Node) -> Optional[str]:
    """Return the fusion-group annotation on *node*, or None."""
    if not hasattr(node, "meta"):
        return None
    return node.meta.get("vulkan_fusion_group")


# ---------------------------------------------------------------------------
# Pattern 1 — Conv → bias → norm → activation
# ---------------------------------------------------------------------------


def _match_conv_norm_activation(
    gm: GraphModule,
) -> Iterable[tuple[Node, dict[str, Any]]]:
    """Match conv2d output that feeds into (optional bias_add) → norm → activation.

    The canonical pattern from SmallCNN:
        conv2d(x, w) → add(bias) → group_norm → relu

    We match the activation node and walk backward to find the conv root,
    then annotate the entire chain with ``conv_epilogue`` so the scheduler
    fuses them into a single dispatch.
    """
    aten = torch.ops.aten

    # Norm targets we can fuse into a conv epilogue.
    norm_targets = {
        aten.native_group_norm.default,
        aten.native_layer_norm.default,
        aten.native_batch_norm.default,
    }

    # Activation targets that can sit after a norm.
    activation_targets = {
        aten.relu.default,
        aten.gelu.default,
        aten.silu.default,
        aten.sigmoid.default,
        aten.tanh.default,
        aten.hardsigmoid.default,
        aten.hardswish.default,
        aten.leaky_relu.default,
        aten.elu.default,
        aten.clamp_min.default,
        aten.clamp.default,
    }

    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        if node.target not in activation_targets:
            continue

        # The activation should consume a norm output (possibly through a
        # view/permute, but in practice the norm output goes straight into
        # the activation).
        act_inputs = [
            a for a in node.args if isinstance(a, Node) and a.op == "call_function"
        ]
        if len(act_inputs) != 1:
            # Maybe the activation is part of a wider block (e.g. where(cond, x, 0)
            # for relu_decomp); only handle simple single-input activations.
            continue
        act_in = act_inputs[0]

        # Walk through optional bias-add between norm and activation.
        norm_node: Optional[Node] = None
        bias_node: Optional[Node] = None

        if act_in.target in norm_targets:
            norm_node = act_in
        elif act_in.target == aten.add.Tensor:
            # Check if one operand is a norm and the other is a bias.
            lhs, rhs = act_in.args[0], act_in.args[1]
            if (
                isinstance(lhs, Node)
                and lhs.op == "call_function"
                and lhs.target in norm_targets
            ):
                norm_node = lhs
                bias_node = rhs
            elif (
                isinstance(rhs, Node)
                and rhs.op == "call_function"
                and rhs.target in norm_targets
            ):
                norm_node = rhs
                bias_node = lhs

        if norm_node is None:
            continue

        # Walk from norm input back to conv. The norm's first input
        # is the feature tensor.
        norm_inputs = [
            a for a in norm_node.args if isinstance(a, Node) and a.op == "call_function"
        ]
        if not norm_inputs:
            continue
        norm_in = norm_inputs[0]

        # The norm input could be a conv2d, or conv2d + bias add.
        conv_node: Optional[Node] = None
        conv_bias_node: Optional[Node] = None

        conv_targets = {
            aten.conv2d.default,
            aten.convolution.default,
            torch.ops.torch_vulkan.conv2d_with_optional_bias.default
            if hasattr(torch.ops, "torch_vulkan")
            else None,
        }
        conv_targets.discard(None)

        if norm_in.target in conv_targets:
            conv_node = norm_in
        elif norm_in.target == aten.add.Tensor:
            lhs, rhs = norm_in.args[0], norm_in.args[1]
            if (
                isinstance(lhs, Node)
                and lhs.op == "call_function"
                and lhs.target in conv_targets
            ):
                conv_node = lhs
                conv_bias_node = rhs
            elif (
                isinstance(rhs, Node)
                and rhs.op == "call_function"
                and rhs.target in conv_targets
            ):
                conv_node = rhs
                conv_bias_node = lhs
        # Also handle the im2col decomposed form: conv may have been decomposed
        # to im2col→reshape→mm chain, but we tag the mm output for the fusion
        # pass since that's where the conv output lives in the FX graph after
        # conv_im2col pass runs.
        elif norm_in.target in (aten.mm.default, aten.addmm.default):
            # The conv was decomposed to mm/addmm; tag the mm output.
            conv_node = norm_in

        if conv_node is None:
            continue

        # Found the chain! Yield once per activation node.
        ctx: dict[str, Any] = {
            "conv_node": conv_node,
            "conv_bias_node": conv_bias_node,
            "norm_node": norm_node,
            "norm_bias_node": bias_node,
            "activation_node": node,
        }
        yield (node, ctx)


def _rewrite_conv_norm_activation(
    gm: GraphModule, root: Node, ctx: dict[str, Any]
) -> GraphModule:
    """Annotate the conv→norm→activation chain with ``conv_epilogue``
    fusion group so the scheduler fuses them into a single dispatch."""
    conv_node = ctx["conv_node"]
    norm_node = ctx["norm_node"]
    activation_node = ctx["activation_node"]

    # Tag every node in the chain.
    _tag_fusion_group(conv_node, FusionGroup.conv_epilogue)
    _tag_fusion_group(norm_node, FusionGroup.conv_epilogue)
    _tag_fusion_group(activation_node, FusionGroup.conv_epilogue)

    if ctx.get("conv_bias_node") is not None:
        _tag_fusion_group(ctx["conv_bias_node"], FusionGroup.conv_epilogue)
    if ctx.get("norm_bias_node") is not None:
        _tag_fusion_group(ctx["norm_bias_node"], FusionGroup.conv_epilogue)

    return gm  # No graph mutation — annotation only.


register_fx_pattern(
    "op_class_conv_norm_activation",
    _match_conv_norm_activation,
    _rewrite_conv_norm_activation,
    priority=8,
)


# ---------------------------------------------------------------------------
# Pattern 2 — mm/addmm + bias add
# ---------------------------------------------------------------------------


def _match_mm_bias(
    gm: GraphModule,
) -> Iterable[tuple[Node, dict[str, Any]]]:
    """Match ``mm(a, b) + bias`` or ``addmm(bias, a, b) + extra_bias``.

    This runs AFTER ``mm_add.py`` (priority 5) which already folded
    mm+add→addmm.  This pass tags the addmm node and any downstream
    bias-add so the scheduler knows the chain is fusible.
    """
    aten = torch.ops.aten

    mm_targets = {aten.mm.default, aten.addmm.default}

    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        if node.target != aten.add.Tensor:
            continue

        lhs, rhs = node.args[0], node.args[1]
        mm_node = None
        bias_node = None

        if (
            isinstance(lhs, Node)
            and lhs.op == "call_function"
            and lhs.target in mm_targets
        ):
            mm_node = lhs
            bias_node = rhs
        elif (
            isinstance(rhs, Node)
            and rhs.op == "call_function"
            and rhs.target in mm_targets
        ):
            mm_node = rhs
            bias_node = lhs

        if mm_node is None:
            continue
        if not isinstance(bias_node, Node):
            continue

        ctx = {
            "mm_node": mm_node,
            "add_node": node,
            "bias_node": bias_node,
        }
        yield (node, ctx)


def _rewrite_mm_bias(gm: GraphModule, root: Node, ctx: dict[str, Any]) -> GraphModule:
    """Tag the mm→bias chain as ``mm_bias`` so the scheduler fuses them."""
    mm_node = ctx["mm_node"]
    add_node = ctx["add_node"]
    bias_node = ctx["bias_node"]

    _tag_fusion_group(mm_node, FusionGroup.mm_bias)
    _tag_fusion_group(add_node, FusionGroup.mm_bias)
    if isinstance(bias_node, Node):
        _tag_fusion_group(bias_node, FusionGroup.mm_bias)

    return gm


register_fx_pattern(
    "op_class_mm_bias",
    _match_mm_bias,
    _rewrite_mm_bias,
    priority=8,
)


# ---------------------------------------------------------------------------
# Pattern 3 — log_softmax → nll_loss (cross_entropy decomposition)
# ---------------------------------------------------------------------------


def _match_softmax_nll(
    gm: GraphModule,
) -> Iterable[tuple[Node, dict[str, Any]]]:
    """Match ``log_softmax → nll_loss_forward`` which is the canonical
    cross-entropy decomposition under torch.compile."""
    aten = torch.ops.aten

    softmax_targets = {
        aten._softmax.default,
        aten._log_softmax.default,
    }
    nll_targets = {aten.nll_loss_forward.default, aten.nll_loss2d_forward.default}

    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        if node.target not in nll_targets:
            continue

        # nll_loss_forward(args: output, target, ...)
        nll_args = [
            a for a in node.args if isinstance(a, Node) and a.op == "call_function"
        ]
        if not nll_args:
            continue
        log_softmax_node = nll_args[0]

        if log_softmax_node.target not in softmax_targets:
            continue

        # Check that log_softmax feeds ONLY into nll (single consumer).
        if len(log_softmax_node.users) != 1:
            continue

        ctx = {
            "log_softmax_node": log_softmax_node,
            "nll_node": node,
        }
        yield (node, ctx)


def _rewrite_softmax_nll(
    gm: GraphModule, root: Node, ctx: dict[str, Any]
) -> GraphModule:
    """Tag the softmax→nll chain as ``softmax_nll``."""
    log_softmax_node = ctx["log_softmax_node"]
    nll_node = ctx["nll_node"]

    _tag_fusion_group(log_softmax_node, FusionGroup.softmax_nll)
    _tag_fusion_group(nll_node, FusionGroup.softmax_nll)

    return gm


register_fx_pattern(
    "op_class_softmax_nll",
    _match_softmax_nll,
    _rewrite_softmax_nll,
    priority=8,
)


# ---------------------------------------------------------------------------
# Pattern 4 — reduction + pointwise tail
# ---------------------------------------------------------------------------


def _match_reduction_pointwise(
    gm: GraphModule,
) -> Iterable[tuple[Node, dict[str, Any]]]:
    """Match reduction output that feeds into a pointwise tail.

    Canonical example: ``sum(x, dim) → div(N) → sub(mean)`` in
    variance computation. The reduction produces a tensor that is
    consumed by a chain of pointwise operations.

    We tag the entire reduction→pointwise chain with
    ``reduction_pointwise`` so the scheduler can fuse the pointwise
    tail into the reduction kernel's epilogue.
    """
    aten = torch.ops.aten

    reduction_targets = {
        aten.sum.default,
        aten.sum.dim_IntList,
        aten.mean.default,
        aten.mean.dim,
        aten.var.default,
        aten.var.dim,
        aten._softmax.default,
        aten._log_softmax.default,
        aten.max.default,
        aten.max.dim,
        aten.min.default,
        aten.min.dim,
        aten.any.default,
        aten.any.dim,
    }

    pointwise_targets = {
        aten.mul.Tensor,
        aten.div.Tensor,
        aten.add.Tensor,
        aten.sub.Tensor,
        aten.relu.default,
        aten.gelu.default,
        aten.silu.default,
        aten.sigmoid.default,
        aten.tanh.default,
        aten.clamp.default,
        aten.clamp_min.default,
        aten.sqrt.default,
        aten.exp.default,
        aten.log.default,
        aten.neg.default,
        aten.abs.default,
    }

    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        if node.target not in reduction_targets:
            continue

        # Check if any consumer is a pointwise op.
        pw_consumers = [
            u
            for u in node.users
            if u.op == "call_function" and u.target in pointwise_targets
        ]
        if not pw_consumers:
            continue

        ctx = {
            "reduction_node": node,
            "pointwise_nodes": pw_consumers,
        }
        yield (node, ctx)


def _rewrite_reduction_pointwise(
    gm: GraphModule, root: Node, ctx: dict[str, Any]
) -> GraphModule:
    """Tag the reduction→pointwise chain."""
    reduction_node = ctx["reduction_node"]
    pointwise_nodes = ctx["pointwise_nodes"]

    _tag_fusion_group(reduction_node, FusionGroup.reduction_pointwise)
    for pw_node in pointwise_nodes:
        _tag_fusion_group(pw_node, FusionGroup.reduction_pointwise)

    return gm


register_fx_pattern(
    "op_class_reduction_pointwise",
    _match_reduction_pointwise,
    _rewrite_reduction_pointwise,
    priority=9,
)


# ---------------------------------------------------------------------------
# Pattern 5 — Backward chain (reduction backward + pointwise epilogue)
# ---------------------------------------------------------------------------


def _match_bw_chain(
    gm: GraphModule,
) -> Iterable[tuple[Node, dict[str, Any]]]:
    """Match backward reduction ops followed by pointwise epilogues.

    In the backward graph, many ``aten.*_backward`` ops produce gradients
    that are then combined via pointwise add/mul/div.  Tagging these chains
    helps the scheduler fuse the pointwise tail into the reduction/backward
    kernel.

    We look for any backward op whose output feeds solely into pointwise
    consumers.
    """
    aten = torch.ops.aten

    # Backward ops that produce a gradient tensor.
    bwd_targets = {
        aten.native_group_norm_backward.default,
        aten.native_layer_norm_backward.default,
        aten.native_batch_norm_backward.default,
        aten._softmax_backward_data.default,
        aten._log_softmax_backward_data.default,
        aten.threshold_backward.default,
        aten.gelu_backward.default,
        aten.tanh_backward.default,
        aten.sigmoid_backward.default,
        aten.elu_backward.default,
        aten.hardsigmoid_backward.default,
        aten.hardswish_backward.default,
        aten.leaky_relu_backward.default,
        aten.max_pool2d_with_indices_backward.default,
    }

    pointwise_targets = {
        aten.mul.Tensor,
        aten.div.Tensor,
        aten.add.Tensor,
        aten.sub.Tensor,
        aten.relu.default,
        aten.gelu.default,
        aten.silu.default,
    }

    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        target = node.target
        # Handle both direct aten targets and registered ops.
        is_bwd = target in bwd_targets or (
            hasattr(target, "_opname") and "_backward" in target._opname
        )
        if not is_bwd:
            continue

        # Find pointwise consumers.
        pw_consumers = [
            u
            for u in node.users
            if u.op == "call_function" and u.target in pointwise_targets
        ]
        if not pw_consumers:
            continue

        ctx = {
            "bwd_node": node,
            "pointwise_nodes": pw_consumers,
        }
        yield (node, ctx)


def _rewrite_bw_chain(gm: GraphModule, root: Node, ctx: dict[str, Any]) -> GraphModule:
    """Tag the backward→pointwise chain as ``bw_chain``."""
    bwd_node = ctx["bwd_node"]
    pointwise_nodes = ctx["pointwise_nodes"]

    _tag_fusion_group(bwd_node, FusionGroup.bw_chain)
    for pw_node in pointwise_nodes:
        _tag_fusion_group(pw_node, FusionGroup.bw_chain)

    return gm


register_fx_pattern(
    "op_class_bw_chain",
    _match_bw_chain,
    _rewrite_bw_chain,
    priority=10,
)
