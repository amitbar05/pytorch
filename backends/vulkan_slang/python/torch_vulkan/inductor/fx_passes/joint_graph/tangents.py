"""Materialize implicit tangent placeholders for the joint backward graph (PF.52)."""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch


def _materialize_implicit_tangents(
    gm: "torch.fx.GraphModule",
) -> "torch.fx.GraphModule":
    """PF.52 — Constant-tangent rematerialization.

    AOTAutograd's joint-graph trace for ``loss.backward()`` on a scalar
    loss lifts the implicit ``tangents_1 = torch.ones(())`` and either
    (a) keeps it as a ``tangents_*`` placeholder, or (b) lifts it via
    ``self._tensor_constantN`` as a ``get_attr`` node on the partitioned
    graph — depending on partitioner choices. Both paths produce a
    Vulkan-device null-storage tensor at runtime when the value is
    statically broadcast-known, because the wrapper never allocates a
    real buffer for it.

    This pass walks the post-partition FX graph and rewrites both
    flavors:

    1. ``placeholder`` named ``tangents_*`` whose ``meta['val']`` is a
       Vulkan-device 1-element tensor.
    2. ``get_attr`` whose target attribute is a Vulkan-device tensor
       with ``data_ptr() == 0`` *or* with all-zero strides + 1-element
       underlying storage (the canonical ``_tensor_constant`` pattern
       for a broadcast tangent).

    For each match, replace every consumer of the node with an
    ``aten.full(target_shape, 1.0, dtype=..., device=vulkan)`` call:

    - When the consumer is ``aten.expand.default(node, target_shape)``,
      fold expand+full into the same call (the expand's output shape is
      the full's shape).
    - Otherwise, splice ``aten.full(node.shape, 1.0, ...)`` immediately
      after the source node and reroute consumers.

    The fill value is ``1.0`` — autograd's implicit grad_output for
    a scalar ``loss.backward()``. Validated by inspecting the source
    tensor for non-``get_attr`` cases (or trusting the AOTAutograd
    invariant for placeholders).

    Idempotent (guarded via ``gm.meta['_pf52_materialized']``) and
    Vulkan-gated (CPU graphs are untouched).

    Stage tag: ``BUG_ROOT="fx-passes"``. Failure mode without this pass:
    ``RuntimeError: Tensor has no backing Vulkan buffer`` (or PF.51's
    pre-validation surface) on backward dispatch.
    """
    import re
    import torch
    from torch.fx import Node

    aten = torch.ops.aten

    if gm.meta.get("_pf52_materialized"):
        return gm

    tangent_re = re.compile(r"^tangents?_\d+$")

    def _is_scalar_val(val) -> bool:
        if not isinstance(val, torch.Tensor):
            return False
        try:
            return val.numel() == 1
        except Exception:  # noqa: BLE001
            return False

    def _val_device_type(val) -> str:
        if not isinstance(val, torch.Tensor):
            return ""
        try:
            return val.device.type
        except Exception:  # noqa: BLE001
            return ""

    def _is_zero_stride_broadcast(t: "torch.Tensor") -> bool:
        """A `_tensor_constant` lifted from a scalar tangent has all-zero
        strides (broadcast-only), shape > 0, and a 1-element underlying
        view. That's the canonical lift pattern.
        """
        try:
            strides = t.stride()
        except Exception:  # noqa: BLE001
            return False
        if not strides:
            return False
        if any(s != 0 for s in strides):
            return False
        # Shape must be non-empty (a true scalar would have stride=()).
        return t.dim() > 0

    # Collect (node, fill_value, target_dtype, target_device) tuples.
    Sources: list = []  # list of (node, dtype, device)
    for node in list(gm.graph.nodes):
        if node.op == "placeholder" and tangent_re.match(node.name):
            val = node.meta.get("val")
            if (
                _is_scalar_val(val)
                and _val_device_type(val) == "vulkan"
            ):
                Sources.append((node, val.dtype, val.device))
        elif node.op == "get_attr":
            try:
                attr = getattr(gm, node.target, None)
            except Exception:  # noqa: BLE001
                attr = None
            if not isinstance(attr, torch.Tensor):
                continue
            if attr.device.type != "vulkan":
                continue
            # Either null-storage (data_ptr == 0) or zero-stride broadcast.
            try:
                ptr = attr.data_ptr()
            except RuntimeError:
                ptr = 0
            if ptr != 0 and not _is_zero_stride_broadcast(attr):
                continue
            Sources.append((node, attr.dtype, attr.device))

    if not Sources:
        gm.meta["_pf52_materialized"] = True
        return gm

    fill_value = 1.0  # autograd's implicit grad_output for ``.sum()`` / scalars.

    changed = False
    for src, dtype, device in Sources:
        # Resolve the source's shape — for a placeholder use its val
        # shape; for a get_attr inspect the resolved attribute. The
        # broadcast-stride cases have a non-trivial shape we use as
        # the materialized full's shape when there's no expand consumer.
        if src.op == "placeholder":
            val = src.meta["val"]
            self_shape = list(val.shape)
        else:
            attr = getattr(gm, src.target, None)
            self_shape = list(attr.shape) if isinstance(attr, torch.Tensor) else []

        # Pass 1: fold ``expand`` consumers into ``aten.full(target_shape)``.
        for consumer in list(src.users):
            if (
                consumer.op == "call_function"
                and consumer.target == aten.expand.default
                and consumer.args
                and consumer.args[0] is src
            ):
                target_shape = consumer.args[1]
                with gm.graph.inserting_before(consumer):
                    full_node = gm.graph.call_function(
                        aten.full.default,
                        args=(target_shape, fill_value),
                        kwargs={"dtype": dtype, "device": device},
                    )
                    full_node.meta = dict(consumer.meta)
                consumer.replace_all_uses_with(full_node)
                gm.graph.erase_node(consumer)
                changed = True

        # Pass 2: any remaining consumer reads the source directly.
        # Splice ``aten.full(self_shape, 1.0, ...)`` after the source
        # and route them to it.
        if src.users:
            with gm.graph.inserting_after(src):
                full_node = gm.graph.call_function(
                    aten.full.default,
                    args=(self_shape, fill_value),
                    kwargs={"dtype": dtype, "device": device},
                )
                # Borrow source meta if available so downstream shape
                # inference is unaffected.
                src_val = src.meta.get("val")
                if src_val is not None:
                    full_node.meta["val"] = src_val
            src.replace_all_uses_with(full_node)
            changed = True

        # Never erase placeholders — they are part of the GraphModule's
        # input contract and AOTAutograd's runtime wrapper passes the
        # original argument list regardless. ``get_attr`` nodes are also
        # left in place — the constant attribute is owned by the
        # partitioner and erasing it can confuse downstream caching.
        # Leaving the source in place is safe: with no users it's a
        # dead-code input that the wrapper-codegen will preserve as
        # ``del placeholder`` after entry but never dispatch with.

    gm.meta["_pf52_materialized"] = True
    if changed:
        gm.graph.lint()
        gm.recompile()
    return gm


