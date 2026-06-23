"""COMPILE.2 fix: mark 0-d ``aten.div.Tensor`` nodes as ``must_be_in_forward``.

Root cause
----------
When a model uses ``conv2d → cross_entropy`` and is compiled with
``torch.compile(backend="inductor")``, the joint forward+backward graph
contains a scalar (0-d) ``aten.div.Tensor`` that computes
``loss / numel`` inside ``nll_loss_backward``.  The AOT min-cut
partitioner may decide to place this node in the *backward* sub-graph.

However, the backward sub-graph's inputs only include placeholders that
the partitioner explicitly forwards.  A 0-d div whose inputs are a loss
scalar and an integer-scalar (numel) often ends up depending on nodes
that the partitioner considers "forward-only".  When the partitioner
copies the joint graph into the backward sub-graph it marks them
``InvalidNode`` — the div then has invalid inputs but is still an output
of the backward, triggering:

    AssertionError: Node was invalid, but is output: ...

The Fix
-------
Before partitioning, scan the joint graph for every ``aten.div.Tensor``
call whose ``node.meta["val"]`` is a 0-d tensor (``ndim == 0``).  Tag
those nodes with::

    node.meta["partitioner_tag"] = "must_be_in_forward"

The partitioner's ``_must_be_in_forward`` / ``_has_tag_must_be_in_forward``
helpers (``torch/_functorch/partitioners.py``) then force the node
into the forward sub-graph, preventing the invalid-node crash.

This is a minimal surgical fix — it does not change the partitioner's
cost model or the joint-graph passes.  It only adds an annotation that
overrides the partitioner's default placement for this specific pattern.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

_PARTITIONER_TAG_KEY = "partitioner_tag"
_MUST_BE_IN_FORWARD = "must_be_in_forward"


def mark_0d_div_must_be_in_forward(
    fx_g: "torch.fx.GraphModule",
) -> "torch.fx.GraphModule":
    """Scan *fx_g* and tag 0-d ``aten.div.Tensor`` nodes for the forward sub-graph.

    Idempotent — nodes already carrying ``partitioner_tag`` are skipped.
    Returns *fx_g* unchanged when no matches are found.
    """
    import torch

    aten = torch.ops.aten

    # Support both the overload and the overloadpacket so we catch
    # aten.div.Tensor.default as well as any future overload.
    _DIV_TARGETS: set = {
        aten.div.Tensor,
        aten.div.default,
        aten.div.Scalar,
    }
    # Also match via overloadpacket in case tracing emits a bare packet.
    _DIV_OVERLOAD_PACKETS: set = {
        aten.div,
    }

    tagged = 0
    for node in fx_g.graph.nodes:
        if node.op != "call_function":
            continue

        # Fast path: direct target comparison (most common).
        if node.target not in _DIV_TARGETS:
            # Slow path: check overloadpacket for wrapped overloads.
            target = node.target
            target_pkg = getattr(target, "overloadpacket", None)
            if target_pkg not in _DIV_OVERLOAD_PACKETS:
                continue

        # Check that the output is a 0-d tensor.
        val = node.meta.get("val")
        if val is None:
            continue
        if not isinstance(val, torch.Tensor):
            continue
        if val.ndim != 0:
            continue

        # Already tagged — don't overwrite.
        if _PARTITIONER_TAG_KEY in node.meta:
            continue

        node.meta[_PARTITIONER_TAG_KEY] = _MUST_BE_IN_FORWARD
        tagged += 1
        log.debug(
            "COMPILE.2: tagged %s (target=%s, shape=%s) as %s",
            node.name,
            node.target,
            val.shape,
            _MUST_BE_IN_FORWARD,
        )

    if tagged:
        log.info(
            "COMPILE.2: marked %d 0-d div node(s) as must_be_in_forward",
            tagged,
        )

    return fx_g
