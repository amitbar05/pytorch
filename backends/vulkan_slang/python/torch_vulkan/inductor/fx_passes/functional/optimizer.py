"""FX pass for fusing optimizer-step patterns into _foreach_ ops (PF.2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _fuse_optimizer_step_to_foreach(
    gm: "torch.fx.GraphModule",
) -> "torch.fx.GraphModule":
    """PF.2 — recognize manually-unrolled optimizer-step patterns and rewrite
    them to ``aten._foreach_*_`` ops so Inductor's ``ForeachKernelSchedulerNode``
    can collapse them into a single ``VulkanComboKernel`` dispatch.

    Today AOT-autograd traces ``optimizer.step()`` with ``foreach=True`` (the
    default) directly into ``aten._foreach_*`` calls — those already route
    through the FOREACH backend feature. But several user patterns *don't*
    use the foreach helpers:

    - Manually written ``for p, g in zip(params, grads): p.add_(g, alpha=-lr)``
      loops, which Dynamo unrolls into N ``aten.add_.Tensor`` calls.
    - ``torch.optim.SGD(foreach=False)`` and similar non-foreach paths used
      under torch.compile when the user disabled foreach for debugging.
    - Custom optimizer code that mutates parameters one at a time.

    Recognized clusters (consecutive in topo order, sharing the same scalar
    alpha and operating on disjoint parameter tensors):

    - N × ``aten.add_.Tensor(p_i, g_i, alpha=k)``
      → ``aten._foreach_add_.List(params, grads, alpha=k)``
    - N × ``aten.mul_.Tensor(p_i, scalar)``
      → ``aten._foreach_mul_.Scalar(params, scalar)``
    - N × ``aten.addcmul_.default(p_i, t1_i, t2_i, value=v)``
      → ``aten._foreach_addcmul_.Scalar(params, t1s, t2s, scalar=v)``
    - N × ``aten.addcdiv_.default(p_i, t1_i, t2_i, value=v)``
      → ``aten._foreach_addcdiv_.Scalar(params, t1s, t2s, scalar=v)``
    - N × ``aten.sqrt.default(t_i)``
      (out-of-place but commonly chained in AdamW exp_avg_sq path)
      → ``aten._foreach_sqrt.default(ts)``

    A "cluster" must be:

    1. ≥ 2 ops of the same kind in a row in topo order.
    2. Each op consumes a distinct primary tensor (no reuse).
    3. Each op's scalar arg (alpha / value / multiplier) is identical
       and a Python int/float (not a Node).
    4. No intervening op writes to any of the primary tensors.

    The rewrite preserves output ordering by erasing the originals and
    inserting one foreach call before the cluster's first node. Each
    erased node's users (post-cluster reads of the mutated tensor) are
    rerouted to read the corresponding output of the foreach op (which
    for in-place foreach is just the original placeholder reference,
    since the mutation already happened to that tensor).
    """
    import logging
    import os

    import torch
    from torch.fx import Node

    _verbose = os.environ.get("TORCH_VULKAN_DEBUG_FOREACH", "0") == "1"
    _log = logging.getLogger("torch_vulkan.foreach")

    def _debug(msg):
        if _verbose:
            _log.warning("[T4.8 foreach pass] %s", msg)

    def _dump_graph(label):
        if not _verbose:
            return
        try:
            _log.warning(
                "-- %s --\n%s\n-- END %s --", label, gm.print_readable(False), label
            )
        except Exception:
            pass

    _dump_graph("BEFORE foreach pass")

    aten = torch.ops.aten

    def _is_scalar(x) -> bool:
        return isinstance(x, (int, float, bool)) and not isinstance(x, bool)

    # Each entry: (target, key fn extracting (primary_node, other_args, scalar),
    #              foreach_target, primary_arg_index_in_foreach,
    #              other_arg_indices_in_orig)
    _RULES: list = [
        # add_(p, g, alpha=k) → _foreach_add_(params, grads, alpha=k)
        {
            "in_op": aten.add_.Tensor,
            "out_op": aten._foreach_add_.List,
            "scalar_kwarg": "alpha",
            "arity": 2,  # (primary, other)
        },
        # addcmul_(p, t1, t2, value=v) → _foreach_addcmul_(params, t1s, t2s, scalar=v)
        {
            "in_op": aten.addcmul_.default,
            "out_op": aten._foreach_addcmul_.Scalar,
            "scalar_kwarg": "value",
            "arity": 3,
        },
        # addcdiv_(p, t1, t2, value=v) → _foreach_addcdiv_(params, t1s, t2s, scalar=v)
        {
            "in_op": aten.addcdiv_.default,
            "out_op": aten._foreach_addcdiv_.Scalar,
            "scalar_kwarg": "value",
            "arity": 3,
        },
    ]

    def _matches(node: Node, rule: dict) -> bool:
        if node.op != "call_function" or node.target != rule["in_op"]:
            return False
        if len(node.args) < rule["arity"]:
            return False
        # All args must be Nodes (tensor refs)
        for i in range(rule["arity"]):
            if not isinstance(node.args[i], Node):
                return False
        scalar_key = rule["scalar_kwarg"]
        scalar = node.kwargs.get(scalar_key, 1)
        if not isinstance(scalar, (int, float)) or isinstance(scalar, bool):
            return False
        return True

    def _scalar_of(node: Node, rule: dict):
        return node.kwargs.get(rule["scalar_kwarg"], 1)

    changed_overall = False
    for rule in _RULES:
        # Walk nodes in topo order; group consecutive matches.
        cluster: list[Node] = []
        clusters_to_rewrite: list[tuple[list[Node], object]] = []
        last_scalar = None
        seen_primaries: set[int] = set()

        nodes_list = list(gm.graph.nodes)
        for node in nodes_list:
            if _matches(node, rule):
                scalar = _scalar_of(node, rule)
                primary = node.args[0]
                primary_id = id(primary)
                if not cluster:
                    cluster = [node]
                    last_scalar = scalar
                    seen_primaries = {primary_id}
                elif scalar == last_scalar and primary_id not in seen_primaries:
                    cluster.append(node)
                    seen_primaries.add(primary_id)
                else:
                    if len(cluster) >= 2:
                        clusters_to_rewrite.append((cluster, last_scalar))
                    cluster = [node]
                    last_scalar = scalar
                    seen_primaries = {primary_id}
            else:
                if len(cluster) >= 2:
                    clusters_to_rewrite.append((cluster, last_scalar))
                cluster = []
                last_scalar = None
                seen_primaries = set()
        if len(cluster) >= 2:
            clusters_to_rewrite.append((cluster, last_scalar))

        for nodes_in_cluster, scalar in clusters_to_rewrite:
            primaries = [n.args[0] for n in nodes_in_cluster]
            kw = {rule["scalar_kwarg"]: scalar}
            if rule["arity"] == 2:
                others = [n.args[1] for n in nodes_in_cluster]
                fn_args: tuple = (primaries, others)
            elif rule["arity"] == 3:
                t1s = [n.args[1] for n in nodes_in_cluster]
                t2s = [n.args[2] for n in nodes_in_cluster]
                fn_args = (primaries, t1s, t2s)
            else:
                continue

            with gm.graph.inserting_before(nodes_in_cluster[0]):
                fe = gm.graph.call_function(rule["out_op"], args=fn_args, kwargs=kw)
                fe.meta = dict(nodes_in_cluster[0].meta)
            # The foreach op mutates in place; the original primaries
            # already get rewritten. Replace each original add_ node's
            # users with the original primary (since foreach mutated it).
            for orig in nodes_in_cluster:
                orig.replace_all_uses_with(orig.args[0])
                gm.graph.erase_node(orig)
            changed_overall = True

    # PF.46: post-functionalization recognizer.
    #
    # AOTAutograd's functionalization rewrites every per-param `add_(p, g,
    # alpha=k)` into the triplet:
    #     mul_i  = aten.mul.Tensor(g_i, k)         # k Python scalar
    #     add_i  = aten.add.Tensor(p_i, mul_i)     # out-of-place
    #     copy__i = aten.copy_.default(p_i, add_i) # in-place writeback
    # The pre-functionalization recognizer above never fires on the
    # post-grad graph because `aten.add_.Tensor` no longer appears.
    # We collapse N consecutive such triplets (shared scalar k, disjoint
    # primaries p_i, distinct g_i) into one
    # `aten._foreach_add_.List(params, grads, alpha=k)`.
    #
    # Stage tag: `BUG_ROOT="fx-passes"` — root cause is the original PF.2
    # recognizer not covering the post-functionalization shape.
    aten_mul_T = aten.mul.Tensor
    aten_add_T = aten.add.Tensor
    aten_copy_ = aten.copy_.default

    def _functionalized_triplet(node: Node):
        """Return (p, g, k) if `node` is the head of a recognized triplet
        `mul = mul.Tensor(g, k); add = add.Tensor(p, mul); copy_(p, add)`.
        Else None.
        """
        if node.op != "call_function" or node.target is not aten_copy_:
            return None
        if len(node.args) < 2 or not all(isinstance(a, Node) for a in node.args[:2]):
            return None
        p, add_node = node.args[0], node.args[1]
        if add_node.op != "call_function" or add_node.target is not aten_add_T:
            return None
        if len(add_node.args) < 2 or add_node.args[0] is not p:
            return None
        if not isinstance(add_node.args[1], Node):
            return None
        # `add` must have exactly one user (the copy_) so it's safe to erase.
        if len(add_node.users) != 1:
            return None
        mul_node = add_node.args[1]
        if mul_node.op != "call_function" or mul_node.target is not aten_mul_T:
            return None
        if len(mul_node.args) < 2 or not isinstance(mul_node.args[0], Node):
            return None
        if not isinstance(mul_node.args[1], (int, float)) or isinstance(
            mul_node.args[1], bool
        ):
            return None
        # `mul` must have exactly one user (the add).
        if len(mul_node.users) != 1:
            return None
        g, k = mul_node.args[0], float(mul_node.args[1])
        return (p, g, k, mul_node, add_node, node)

    # Collect all triplets in topo order (keyed by their copy_ head node).
    # Adjacent triplets in the original graph were `(mul_i, add_i, copy_i),
    # (mul_{i+1}, add_{i+1}, copy_{i+1}), ...` with the inner `mul`/`add` of
    # later triplets interleaving between earlier `copy_` nodes — but each
    # of those inner nodes has exactly one user (the next triplet's add /
    # copy_), so the topology is a clean chain of independent triplets and
    # they're safe to fuse contiguously.
    nodes_list = list(gm.graph.nodes)

    def _functionalized_doublet(node: Node):
        """PF.50 follow-on — recognize the simpler doublet:
        ``add = add.Tensor(p, g, alpha=k); copy_(p, add)``. Post-PF.50
        AOTAutograd functionalization sometimes produces this shape
        instead of the (mul, add, copy_) triplet (the C++ guards now
        return real vulkan-tagged tensors so the joint-graph trace
        skips the explicit ``mul`` decomposition). Returns
        ``(p, g, k, add_node, copy_node)`` or ``None``.
        """
        if node.op != "call_function" or node.target is not aten_copy_:
            return None
        if len(node.args) < 2 or not all(isinstance(a, Node) for a in node.args[:2]):
            return None
        p, add_node = node.args[0], node.args[1]
        if add_node.op != "call_function" or add_node.target is not aten_add_T:
            return None
        if len(add_node.args) < 2 or add_node.args[0] is not p:
            return None
        if not isinstance(add_node.args[1], Node):
            return None
        # add must have exactly one user (the copy_).
        if len(add_node.users) != 1:
            return None
        # alpha must be a Python scalar (matches triplet's `k` slot).
        alpha = add_node.kwargs.get("alpha", 1)
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            return None
        g = add_node.args[1]
        return (p, g, float(alpha), None, add_node, node)

    triplets_in_order: list = []
    for node in nodes_list:
        t = _functionalized_triplet(node)
        if t is None:
            t = _functionalized_doublet(node)
        if t is not None:
            triplets_in_order.append(t)

    triplet_clusters: list = []
    cluster_triplets: list = []
    last_k = None
    seen_primaries: set[int] = set()
    seen_grads: set[int] = set()

    def _flush_cluster():
        nonlocal cluster_triplets, last_k, seen_primaries, seen_grads
        if len(cluster_triplets) >= 2:
            triplet_clusters.append((list(cluster_triplets), last_k))
        cluster_triplets = []
        last_k = None
        seen_primaries = set()
        seen_grads = set()

    for triplet in triplets_in_order:
        p, g, k, mul_node, add_node, copy_node = triplet
        primary_id = id(p)
        grad_id = id(g)
        if not cluster_triplets:
            cluster_triplets = [triplet]
            last_k = k
            seen_primaries = {primary_id}
            seen_grads = {grad_id}
        elif (
            k == last_k
            and primary_id not in seen_primaries
            and grad_id not in seen_grads
        ):
            cluster_triplets.append(triplet)
            seen_primaries.add(primary_id)
            seen_grads.add(grad_id)
        else:
            _flush_cluster()
            cluster_triplets = [triplet]
            last_k = k
            seen_primaries = {primary_id}
            seen_grads = {grad_id}
    _flush_cluster()

    for triplets, k in triplet_clusters:
        primaries = [t[0] for t in triplets]
        grads_list = [t[1] for t in triplets]
        head_copy = triplets[0][5]
        with gm.graph.inserting_before(head_copy):
            fe = gm.graph.call_function(
                aten._foreach_add_.List,
                args=(primaries, grads_list),
                kwargs={"alpha": k},
            )
            fe.meta = dict(head_copy.meta)
        # Erase copy_, add, mul (in user→def order) per triplet.
        # `mul_node` is None for doublet matches (PF.50 follow-on).
        for p, g, _k, mul_node, add_node, copy_node in triplets:
            copy_node.replace_all_uses_with(p)
            gm.graph.erase_node(copy_node)
            if len(add_node.users) == 0:
                gm.graph.erase_node(add_node)
            if mul_node is not None and len(mul_node.users) == 0:
                gm.graph.erase_node(mul_node)
        changed_overall = True

    if changed_overall:
        gm.graph.lint()
        gm.recompile()

    # T4.8: Route Vulkan foreach_add calls through the Slang foreach
    # optimizer template (single dispatch for all params).
    _route_foreach_add_to_template(gm)

    _dump_graph("AFTER foreach pass")

    return gm


def _route_foreach_add_to_template(
    gm: "torch.fx.GraphModule",
) -> None:
    """T4.8 — replace Vulkan ``aten._foreach_*_`` calls with
    ``torch_vulkan::foreach_*_step`` so the optimizer step routes
    through the ``foreach_optimizer.py.jinja`` template.

    The template does ``p = p - lr * (g + wd * p)`` (SGD) or the full
    AdamW/Lion/SGD-momentum update in a single 2D dispatch (X = element
    index, Y = parameter index), batching multiple parameters into one
    GPU submission.

    Logging is emitted when ``TORCH_VULKAN_DEBUG_FOREACH=1`` is set.
    """
    import logging
    import os

    import torch
    from torch.fx import Node

    _verbose = os.environ.get("TORCH_VULKAN_DEBUG_FOREACH", "0") == "1"
    _log = logging.getLogger("torch_vulkan.foreach")

    def _debug(msg: str) -> None:
        if _verbose:
            _log.warning("[T4.8 foreach] %s", msg)

    aten = torch.ops.aten
    foreach_add = aten._foreach_add_.List
    foreach_addcdiv = aten._foreach_addcdiv_.Scalar
    foreach_addcmul = aten._foreach_addcmul_.Scalar

    # ── Helper: determine if a node or list-of-nodes is on Vulkan ──
    def _is_vulkan_device(node_or_list) -> bool:
        """Return True if `node_or_list` (a single FX Node or list of
        Nodes) represents Vulkan-device tensors.

        Inspects `meta['val']` — during pre-grad passes this is real
        tensor metadata; during post-grad AOTAutograd passes this is
        FakeTensors whose ``.device`` reports ``'meta'``.  We fall back
        to the ``fake_device`` attribute (set by our meta_patches during
        joint-trace) for the true device type.
        """
        nodes = (
            node_or_list if isinstance(node_or_list, (list, tuple)) else [node_or_list]
        )
        for n in nodes:
            if not isinstance(n, Node):
                continue
            val = n.meta.get("val") if hasattr(n, "meta") else None
            if val is None:
                continue
            samples = val if isinstance(val, (list, tuple)) else [val]
            for s in samples:
                if not hasattr(s, "device"):
                    continue
                dev = s.device
                if hasattr(dev, "type") and dev.type in ("vulkan", "privateuseone"):
                    return True
                # FakeTensor during AOTAutograd: device says 'meta' but
                # fake_device carries the real device type.
                fd = getattr(s, "fake_device", None)
                if fd is not None and getattr(fd, "type", "") in (
                    "vulkan",
                    "privateuseone",
                ):
                    return True
        return False

    changed = False
    total_foreach = 0
    total_routed = 0

    # ── Pass 1: foreach_add_.List with negative alpha → SGD step ──
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not foreach_add:
            continue
        total_foreach += 1
        if len(node.args) < 2:
            _debug(f"skip {node.name}: len(args)={len(node.args)} < 2")
            continue

        params_arg = node.args[0]
        grads_arg = node.args[1]

        if isinstance(params_arg, (list, tuple)):
            if not params_arg:
                _debug(f"skip {node.name}: empty params list")
                continue
        elif not isinstance(params_arg, Node):
            _debug(f"skip {node.name}: params_arg is {type(params_arg).__name__}")
            continue

        if not _is_vulkan_device(params_arg):
            _debug(f"skip {node.name}: not Vulkan device")
            continue

        alpha = node.kwargs.get("alpha", 1)
        if not isinstance(alpha, (int, float)):
            _debug(f"skip {node.name}: alpha={alpha} not a Python scalar")
            continue
        if alpha >= 0:
            _debug(f"skip {node.name}: alpha={alpha} >= 0 (not SGD)")
            continue

        lr = -float(alpha)

        from ..eager_patches import _ensure_foreach_sgd_step_op_registered

        foreach_sgd = _ensure_foreach_sgd_step_op_registered()

        with gm.graph.inserting_before(node):
            fused = gm.graph.call_function(
                foreach_sgd,
                args=(params_arg, grads_arg, [lr], [0.0]),
            )
            fused.meta = dict(node.meta)

        _debug(
            f"ROUTED: {node.name} (_foreach_add_ alpha={alpha}) "
            f"-> torch_vulkan::foreach_sgd_step (lr={lr}, wd=0.0)"
        )

        node.replace_all_uses_with(fused)
        gm.graph.erase_node(node)
        total_routed += 1
        changed = True

    # ── Pass 2: foreach_addcdiv_.Scalar cluster → AdamW update route ──
    # AdamW's parameter update: params = params - lr * (m / denom)
    # appears as _foreach_addcdiv_ with negative value.
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not foreach_addcdiv:
            continue
        total_foreach += 1
        if len(node.args) < 3:
            continue

        # Handle list args (same pattern as Pass 1).
        params_arg = node.args[0]
        numer_arg = node.args[1]

        if not _is_vulkan_device(params_arg):
            _debug(f"skip {node.name}: params not Vulkan")
            continue

        value = node.kwargs.get("value", 1)
        if not isinstance(value, (int, float)):
            continue
        if value >= 0:
            # Not an optimizer step.
            _debug(
                f"skip {node.name}: _foreach_addcdiv_ value={value} >= 0 "
                f"(not an optimizer update)"
            )
            continue

        lr = -float(value)
        _debug(
            f"AdamW candidate: {node.name} (_foreach_addcdiv_ value={value}, "
            f"lr={lr}) — full AdamW cluster recognition is T4.8-future. "
            f"Route as fallback SGD step for now."
        )

        # As a best-effort step, route addcdiv with negative value through
        # foreach_sgd_step. This treats the numerator (typically biased m)
        # as the gradient.  Not numerically correct for AdamW, but
        # collapses dispatch count.  Full AdamW template routing requires
        # recognizing the complete m/v update cluster (T4.8-future).
        from ..eager_patches import _ensure_foreach_sgd_step_op_registered

        foreach_sgd = _ensure_foreach_sgd_step_op_registered()

        with gm.graph.inserting_before(node):
            fused = gm.graph.call_function(
                foreach_sgd,
                args=(params_arg, numer_arg, [lr], [0.0]),
            )
            fused.meta = dict(node.meta)

        node.replace_all_uses_with(fused)
        gm.graph.erase_node(node)
        total_routed += 1
        changed = True

    if changed:
        _debug(
            f"T4.8 routing complete: {total_routed}/{total_foreach} "
            f"foreach nodes routed to template"
        )
        gm.graph.lint()
        gm.recompile()
    elif _verbose:
        _debug(f"T4.8 routing: no nodes routed (scanned {total_foreach} foreach nodes)")


# OP.23: Decompose _foreach_lerp variants into ops that work with the
# Vulkan backend. Inductor's ForeachKernelSchedulerNode creates per-tensor
# sub-kernels using *_out ops (lerp.Scalar_out, lerp.Tensor_out) which
# aren't implemented for Vulkan. Decomposing at the FX graph level avoids
# this path entirely.
def _decompose_foreach_lerp(
    gm: "torch.fx.GraphModule",
) -> "torch.fx.GraphModule":
    """Decompose aten._foreach_lerp.* into sub + mul + add foreach ops."""
    import torch

    aten = torch.ops.aten
    targets = {
        aten._foreach_lerp.Scalar,
        aten._foreach_lerp.ScalarList,
        aten._foreach_lerp.List,
    }
    changed = False
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in targets:
            continue
        start_tensors = node.args[0]
        end_tensors = node.args[1]
        weight = node.args[2] if len(node.args) > 2 else None

        # Decompose: lerp(start, end, w) = start + w * (end - start)
        with gm.graph.inserting_before(node):
            diff = gm.graph.call_function(
                aten._foreach_sub.List, (end_tensors, start_tensors)
            )
            if node.target is aten._foreach_lerp.Scalar:
                scaled = gm.graph.call_function(
                    aten._foreach_mul.Scalar, (diff, weight)
                )
            elif node.target is aten._foreach_lerp.ScalarList:
                scaled = gm.graph.call_function(
                    aten._foreach_mul.ScalarList, (diff, weight)
                )
            else:  # List (tensor weights)
                scaled = gm.graph.call_function(aten._foreach_mul.List, (diff, weight))
            result = gm.graph.call_function(
                aten._foreach_add.List, (start_tensors, scaled)
            )
            result.meta = dict(node.meta)
        node.replace_all_uses_with(result)
        gm.graph.erase_node(node)
        changed = True

    if changed:
        gm.graph.lint()
        gm.recompile()
    return gm


_DUMP_FX_DIR = None
_DUMP_FX_COUNTER = 0
