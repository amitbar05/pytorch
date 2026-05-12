"""PF.40 — Joint-graph lifetime-class annotation.

Tags every FX node in the AOTAutograd joint graph with
``node.meta["lifetime_class"]`` drawn from one of:

- ``parameter`` — primal placeholder (model parameter or activation input);
  never released across training steps.
- ``gradient`` — joint-graph output that is a backward-graph result
  (aliases an ``accumulate_grad_`` target); released at the
  ``optimizer.zero_grad()`` boundary.
- ``save_for_backward`` — forward node consumed by a backward node;
  released after the backward step that consumes it.
- ``transient`` — fwd-or-bwd intermediate consumed within the same side
  of the graph; released eagerly.
- ``output`` — joint-graph output that is *not* a gradient (the user's
  forward result); owned by the caller.
- ``scratch`` — intermediate workspace allocated and released *inside* a
  single extern-kernel dispatch (mm split-K accumulators, multi-stage
  reduction partials, conv im2col, philox second-output, flash-attention
  log-sum-exp). Never crosses a Python statement boundary; the caller
  immediately drops it after the dispatch returns. Distinct from
  ``transient`` — the scratch bucket can be reused freely *within* a
  single training step without colliding with any user-visible buffer.

Implementation strategy: identify the bwd subgraph by walking the joint
graph backwards from the gradient outputs. AOTAutograd's joint convention
is ``output = (*forward_outputs, *gradients)`` where the gradient slice
length equals the number of primals that require grad (≈ ``len(tangents)``
for typical scalar-loss graphs). With ``joint_inputs`` we know exactly
how many forward outputs there are. Without it we fall back to
tangent-placeholder reachability.

Consumed by PF.41 (``StepActivationPool`` keyed on ``lifetime_class``),
PF.42 (step-end release hook), PF.43 (activation checkpointing).
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from torch_vulkan.inductor import buffer_pool as _buffer_pool

if TYPE_CHECKING:
    import torch.fx as _fx


LIFETIME_CLASSES = frozenset(
    {"parameter", "gradient", "save_for_backward", "transient", "output", "scratch"}
)

_TANGENT_RE = re.compile(r"^tangents?_\d+$")
_LIFETIME_KEY = "lifetime_class"
_GUARD_KEY = "_pf40_lifetime_annotated"


def _count_tangents(gm: "_fx.GraphModule", joint_inputs: Any) -> int:
    """Number of tangent inputs in the joint graph.

    Prefer ``joint_inputs[1]`` when AOT-autograd hands us its native
    ``(primals, tangents)`` tuple; fall back to counting tangent-pattern
    placeholders. Returns 0 when nothing matches (forward-only graph).
    """
    if isinstance(joint_inputs, (list, tuple)) and len(joint_inputs) == 2:
        tangents = joint_inputs[1]
        if isinstance(tangents, (list, tuple)):
            return len(tangents)
    return sum(
        1 for n in gm.graph.nodes
        if n.op == "placeholder" and _TANGENT_RE.match(n.name)
    )


def _split_output_args(output_args: tuple, num_fwd_outputs: int) -> tuple:
    """Split the joint-graph output tuple into (forward_outputs, gradients).

    AOTAutograd's joint convention puts forward outputs first, gradients
    last. ``num_fwd_outputs`` is derived from ``len(joint_inputs[1])`` —
    i.e. the count of tangents. When the count is unknown we treat all
    outputs as forward outputs (degenerate but safe).
    """
    if num_fwd_outputs <= 0 or num_fwd_outputs > len(output_args):
        return output_args, ()
    return output_args[:num_fwd_outputs], output_args[num_fwd_outputs:]


def annotate_lifetime_classes(
    gm: "_fx.GraphModule",
    joint_inputs: Any | None = None,
) -> "_fx.GraphModule":
    """Annotate every node in ``gm.graph`` with a ``lifetime_class``.

    Idempotent — guarded by ``gm.meta["_pf40_lifetime_annotated"]``.
    Operates on the joint graph as captured at ``joint_custom_pass``
    time; the annotations propagate into ``fw_module`` / ``bw_module``
    after the partitioner runs because the partitioner copies node meta.

    ``joint_inputs`` (when supplied) carries AOT-autograd's native
    ``(primals, tangents)`` tuple, which lets us split the output node
    cleanly. Without it we fall back to tangent-placeholder counting,
    which suffices for dynamic-tangent graphs but degrades when AOT
    autograd lifts a constant tangent into a ``get_attr`` (the
    placeholder is then dead and we under-count).
    """
    if gm.meta.get(_GUARD_KEY):
        return gm

    graph = gm.graph

    # Locate the (single) output node and unpack its return tuple.
    output_node = next(
        (n for n in graph.nodes if n.op == "output"), None
    )
    output_args: tuple = ()
    if output_node is not None and output_node.args:
        first = output_node.args[0]
        if isinstance(first, (list, tuple)):
            output_args = tuple(first)
        else:
            output_args = (first,)

    num_tangents = _count_tangents(gm, joint_inputs)
    # Forward outputs come first; gradients (one per primal-with-grad)
    # come last. The tangent count equals the forward-output count for
    # all standard AOT-autograd traces.
    fwd_out_args, grad_args = _split_output_args(output_args, num_tangents)

    # AOTAutograd uses ``None`` as the "no gradient" sentinel for primals
    # that don't require grad — strip it (and any other non-Node sentinel
    # like ints/Symbols) before building the reachability seeds.
    fwd_out_set = {a for a in fwd_out_args if hasattr(a, "op")}
    grad_set = {a for a in grad_args if hasattr(a, "op")}

    def _arg_nodes(node):
        # Yields Node-typed args (skip primitives, dtypes, devices).
        for arg in node.args:
            if hasattr(arg, "op"):
                yield arg
            elif isinstance(arg, (list, tuple)):
                for a in arg:
                    if hasattr(a, "op"):
                        yield a
        for v in node.kwargs.values():
            if hasattr(v, "op"):
                yield v
            elif isinstance(v, (list, tuple)):
                for a in v:
                    if hasattr(a, "op"):
                        yield a

    def _backward_reachable_from(seed: set) -> set:
        out = set(seed)
        frontier = list(seed)
        while frontier:
            node = frontier.pop()
            for arg in _arg_nodes(node):
                if arg not in out:
                    out.add(arg)
                    frontier.append(arg)
        return out

    # Walk producer→consumer in two BFS sweeps. ``fwd_producers`` is
    # everything that contributes to a forward output (the fwd subgraph
    # the user sees). ``bwd_producers`` is everything that contributes to
    # a gradient. A node in *both* is a fwd-produced value that the bwd
    # depends on — i.e. ``save_for_backward``.
    fwd_producers = _backward_reachable_from(fwd_out_set)
    bwd_producers = _backward_reachable_from(grad_set)

    # The bwd subgraph at runtime is "bwd_producers minus fwd_producers"
    # plus the saved-for-backward tensors that cross the cut. We use
    # ``bwd_producers`` directly to detect bwd-only consumers.
    bwd_only = bwd_producers - fwd_producers

    # Tangent placeholders always live in the bwd subgraph at runtime
    # even if their downstream uses have been rewritten; their dataflow
    # role is bwd-input regardless. Force-include for completeness.
    for n in graph.nodes:
        if n.op == "placeholder" and _TANGENT_RE.match(n.name):
            bwd_only.add(n)

    for node in graph.nodes:
        if node.op == "placeholder":
            if _TANGENT_RE.match(node.name):
                node.meta[_LIFETIME_KEY] = "transient"
            else:
                node.meta[_LIFETIME_KEY] = "parameter"
            continue

        if node.op == "output":
            node.meta[_LIFETIME_KEY] = "output"
            continue

        if node in fwd_out_set:
            node.meta[_LIFETIME_KEY] = "output"
            continue

        if node in grad_set:
            node.meta[_LIFETIME_KEY] = "gradient"
            continue

        # save_for_backward: fwd-produced node with at least one
        # bwd-only consumer. The bwd consumer side is what makes this
        # a cross-cut tensor (must outlive forward dispatch).
        if node in fwd_producers and any(
            user in bwd_only or user in grad_set
            for user in node.users
        ):
            node.meta[_LIFETIME_KEY] = "save_for_backward"
            continue

        # bwd-only intermediate.
        if node in bwd_only:
            node.meta[_LIFETIME_KEY] = "transient"
            continue

        # fwd-only intermediate (consumed within forward, never feeds bwd).
        node.meta[_LIFETIME_KEY] = "transient"

    gm.meta[_GUARD_KEY] = True
    _install_zero_grad_release_hook(graph)
    return gm


def _is_zero_inplace(node) -> bool:
    """Match ``aten.zero_.default`` (and its OverloadPacket) as a call_function target.

    The optimizer's ``zero_grad`` lowers (under set_to_none=False) to an
    ``aten.zero_.default`` op against the gradient tensor. Different graph
    capture surfaces hand us either the OpOverload or the OverloadPacket
    object as ``node.target``; match both shapes to stay capture-agnostic.
    """
    if node.op != "call_function":
        return False
    target = node.target
    name = getattr(target, "_opname", None) or getattr(target, "__name__", "")
    qualname = getattr(target, "name", lambda: "")
    qual = qualname() if callable(qualname) else ""
    return (
        name in ("zero_", "zero")
        or qual == "aten::zero_"
        or qual == "aten.zero_.default"
    )


def _install_zero_grad_release_hook(graph) -> int:
    """Fire ``release_class("gradient")`` once per joint-graph annotation pass.

    PF.42: when the joint graph contains an ``aten.zero_.default`` call whose
    target tensor was tagged ``lifetime_class="gradient"`` by
    :func:`annotate_lifetime_classes`, the gradient bucket can be reclaimed —
    the in-place zero_ writes through the storage so the buffer's old contents
    are dead. We collapse the per-tensor signal into a single
    ``release_class("gradient")`` call: the buffer pool drops every gradient
    bucket together, which matches optimizer semantics (zero_grad zeroes all
    grads in one optimizer.step boundary).

    Returns the number of zero-on-gradient sites detected (≥1 implies the
    hook fired). The actual release happens at FX-pass time — this is a
    synthetic-FX floor surface that proves the detection chain works end-to-
    end without requiring a compiled training-loop dispatch (which is gated
    on PF.63 storage-clone repair).
    """
    sites = 0
    for node in graph.nodes:
        if not _is_zero_inplace(node):
            continue
        # The first positional arg of ``aten.zero_.default`` is the target
        # tensor. Skip if it's not a Node (e.g. a get_attr literal).
        if not node.args:
            continue
        target_arg = node.args[0]
        if not hasattr(target_arg, "meta"):
            continue
        if target_arg.meta.get(_LIFETIME_KEY) == "gradient":
            sites += 1
    if sites > 0:
        _buffer_pool.release_class("gradient")
    return sites


# T6.2: runtime hook on ``Optimizer.zero_grad``. The FX-pass-time hook above
# only fires when the joint graph is *compiled*; once compilation is cached
# the user's training loop calls ``optimizer.zero_grad()`` every step
# without re-tripping the FX pass, so gradient buckets pile up. Patch
# ``torch.optim.Optimizer.zero_grad`` so each call drops the gradient
# class regardless of whether the step was compiled or eager.
_RUNTIME_HOOK_INSTALLED = False
_orig_zero_grad = None  # type: ignore[var-annotated]


def install_zero_grad_runtime_hook() -> bool:
    """Patch ``torch.optim.Optimizer.zero_grad`` to fire ``release_class("gradient")``.

    T6.2: every call to ``optimizer.zero_grad()`` (eager or compiled, any
    Optimizer subclass that uses the base method) releases the buffer pool's
    gradient bucket *after* the original zero_grad runs. This complements
    :func:`_install_zero_grad_release_hook` (which fires once per joint-graph
    compile, not per training step).

    Idempotent — second call is a no-op. Returns True if the patch was just
    installed, False if it was already in place. Use
    :func:`uninstall_zero_grad_runtime_hook` to revert (test hook).

    The wrapper preserves zero_grad's exact semantics (the original runs
    first, gradients are still zeroed/None'd as the user expects); the only
    addition is the trailing ``release_class("gradient")`` call. Failures
    inside ``release_class`` are swallowed so a buffer-pool-disabled or
    Vulkan-less environment never breaks user training loops.
    """
    global _RUNTIME_HOOK_INSTALLED, _orig_zero_grad
    if _RUNTIME_HOOK_INSTALLED:
        return False
    import logging

    import torch.optim

    _log = logging.getLogger(__name__)
    _orig_zero_grad = torch.optim.Optimizer.zero_grad

    def _patched_zero_grad(self, *args, **kwargs):
        result = _orig_zero_grad(self, *args, **kwargs)
        try:
            _buffer_pool.release_class("gradient")
        except Exception:
            # Never let a pool-side failure perturb training. The gradient
            # bucket would just stay live until the next compile-time hook
            # firing — degraded memory but correct numerics.
            pass
        return result

    _patched_zero_grad._torch_vulkan_patched = True  # type: ignore[attr-defined]
    torch.optim.Optimizer.zero_grad = _patched_zero_grad
    _RUNTIME_HOOK_INSTALLED = True
    _log.info("T6.2: patched Optimizer.zero_grad to release gradient class")
    return True


def uninstall_zero_grad_runtime_hook() -> bool:
    """Revert :func:`install_zero_grad_runtime_hook`. Test hook.

    Returns True if a patch was uninstalled, False if none was installed.
    Restores the original ``Optimizer.zero_grad``.
    """
    global _RUNTIME_HOOK_INSTALLED, _orig_zero_grad
    if not _RUNTIME_HOOK_INSTALLED:
        return False
    import torch.optim

    if _orig_zero_grad is not None:
        torch.optim.Optimizer.zero_grad = _orig_zero_grad
    _RUNTIME_HOOK_INSTALLED = False
    _orig_zero_grad = None
    return True
