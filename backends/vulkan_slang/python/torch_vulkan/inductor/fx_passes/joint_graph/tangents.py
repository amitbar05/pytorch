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
    graph — depending on partitioner choices.  Both paths produce a
    Vulkan-device null-storage tensor at runtime when the value is
    statically broadcast-known, because the wrapper never allocates a
    real buffer for it.

    This pass walks the post-partition FX graph and rewrites all three
    flavors:

    1. ``placeholder`` named ``tangents_*`` whose ``meta['val']`` is a
       Vulkan-device 1-element tensor.
    2. ``get_attr`` whose target attribute is a Vulkan-device tensor
       with all-zero strides + non-empty shape (the CPU-side canonical
       ``_tensor_constant`` pattern for a broadcast tangent lifted via
       ``tangents_1.expand([N])``).
    3. ``get_attr`` for ``_tensor_constant*`` attributes on Vulkan device
       whose Python-level storage is invalid (``untyped_storage().data_ptr()``
       raises).  These are the implicit grad_output=1.0 tangents lifted by
       the AOTAutograd partitioner for binary-backward ops (maximum, minimum,
       atan2, etc.) that use two separate lifted constant nodes.

    For each match, replace every consumer of the node with an
    ``aten.full(target_shape, 1.0, dtype=..., device=vulkan)`` call.

    The fill value is ``1.0`` — autograd's implicit grad_output for
    a scalar ``loss.backward()``.

    Idempotent (guarded via ``gm.meta['_pf52_materialized']``) and
    Vulkan-gated (CPU graphs are untouched).

    **Critical structural note**: Fix 3 (null-storage _tensor_constant*
    detection) MUST NOT be guarded by ``if not sources: return gm``.
    The backward graph for binary ops (maximum, minimum, atan2) typically
    has NO ``tangents_*`` placeholders — only ``_tensor_constant*`` attrs —
    so ``sources`` is always empty for those graphs.  Fix 3 must run
    unconditionally to handle them.

    Stage tag: ``BUG_ROOT="fx-passes"``.
    """
    import re
    import torch

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
        """Return True for a tensor with all-zero strides and non-empty shape."""
        try:
            strides = t.stride()
        except Exception:  # noqa: BLE001
            return False
        if not strides:
            return False
        if any(s != 0 for s in strides):
            return False
        return t.dim() > 0

    fill_value = 1.0  # autograd's implicit grad_output for ``.sum()`` / scalars.
    changed = False

    # Collect expand-target (shape, dtype) pairs seen in Fix 1 Pass 1.
    # Fix 3 uses this set to identify co-lifted _tensor_constant* nodes
    # without relying on untyped_storage().data_ptr() (which returns a
    # non-zero VkBuffer handle even for null-storage Vulkan FakeTensors).
    _tangent_shapes: set = set()

    # ── Fix 1 + Fix 2: tangents_* placeholders and zero-stride get_attr ──────
    sources: list = []
    for node in list(gm.graph.nodes):
        if node.op == "placeholder" and tangent_re.match(node.name):
            val = node.meta.get("val")
            if (
                _is_scalar_val(val)
                and _val_device_type(val) == "vulkan"
            ):
                sources.append((node, val.dtype, val.device))
        elif node.op == "get_attr":
            try:
                attr = getattr(gm, node.target, None)
            except Exception:  # noqa: BLE001
                attr = None
            if not isinstance(attr, torch.Tensor):
                continue
            if attr.device.type != "vulkan":
                continue
            # Only match zero-stride broadcast constants.
            # On Vulkan, ``tensor.expand([N])`` produces stride=(1,) tensors
            # (not stride=(0,)), so this branch currently never fires; kept
            # for forward-compatibility.
            if not _is_zero_stride_broadcast(attr):
                continue
            sources.append((node, attr.dtype, attr.device))

    for src, dtype, device in sources:
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
                    # Preserve the Vulkan FakeTensor in meta["val"] — do NOT
                    # overwrite with a real CPU tensor (which would cause
                    # FakeTensorUpdater.incremental_update to pass a real tensor
                    # to ops under FakeTensorMode, crashing with
                    # "Please convert all Tensors to FakeTensors first").
                consumer.replace_all_uses_with(full_node)
                gm.graph.erase_node(consumer)
                # Record this expand target shape so Fix 3 can match the
                # co-lifted _tensor_constant* node for the other gradient.
                try:
                    _tangent_shapes.add((tuple(int(s) for s in target_shape), dtype))
                except Exception:  # noqa: BLE001
                    pass
                changed = True

        # Pass 2: remaining consumers read the source directly.
        if src.users:
            with gm.graph.inserting_after(src):
                full_node = gm.graph.call_function(
                    aten.full.default,
                    args=(self_shape, fill_value),
                    kwargs={"dtype": dtype, "device": device},
                )
                # Borrow source meta (Vulkan FakeTensor) — do NOT set to
                # real CPU tensor.
                src_val = src.meta.get("val")
                if src_val is not None:
                    full_node.meta["val"] = src_val
            src.replace_all_uses_with(full_node)
            changed = True

        # Never erase placeholders — they are part of the GraphModule's
        # input contract. get_attr nodes: also left in place (owned by
        # partitioner).

    # ── Fix 3: null-storage _tensor_constant* get_attr nodes ─────────────────
    # This fix runs UNCONDITIONALLY — it must not be gated by ``if not sources``
    # because backward graphs for binary ops (maximum, minimum, atan2) have NO
    # tangents_* placeholders; their implicit grad_outputs are lifted as
    # _tensor_constant* module attributes.
    #
    # Discriminant: ``untyped_storage().data_ptr()`` raises for FakeTensors
    # with invalid Python storage (the null-storage implicit tangents), while
    # it succeeds for real saved-activation tensors.  This matches the EXACT
    # same condition that causes ``add_tensor_constant`` → ``is_same_tensor``
    # to crash: ``data.untyped_storage().data_ptr()`` → RuntimeError.
    _const_re = re.compile(r"^_tensor_constant\d+$")
    for node in list(gm.graph.nodes):
        if node.op != "get_attr" or not _const_re.match(node.target):
            continue
        try:
            attr = getattr(gm, node.target, None)
        except Exception:  # noqa: BLE001
            attr = None
        if not isinstance(attr, torch.Tensor):
            continue
        if attr.device.type not in ("vulkan", "privateuseone"):
            continue
        # On Vulkan, untyped_storage().data_ptr() returns a non-zero VkBuffer
        # handle even for null-storage FakeTensors — it never raises, so it
        # cannot discriminate tangent constants from real saved activations.
        # Primary discriminant: shape-based matching.  Fix 1's Pass 1 records
        # each tangent expand target in _tangent_shapes; a _tensor_constant*
        # with the same (shape, dtype) is the co-lifted tangent for the other
        # gradient in a binary-backward graph (atan2, maximum, minimum, …).
        _shape_key = (tuple(int(s) for s in attr.shape), attr.dtype)
        if _shape_key not in _tangent_shapes:
            # No shape match — fallback to Python-storage check for future
            # PyTorch versions where null-storage may raise.
            try:
                attr.untyped_storage().data_ptr()
                continue  # valid storage → real saved activation
            except Exception:  # noqa: BLE001
                pass  # invalid storage → implicit tangent → fall through
        if not attr.shape:
            continue  # skip 0-dim scalars
        # Use the original node's meta["val"] (Vulkan FakeTensor) as the
        # meta for the replacement aten.full node.  Do NOT use a real CPU
        # tensor — FakeTensorUpdater.incremental_update reads meta["val"]
        # and passes it to ops under V.fake_mode.
        src_val = node.meta.get("val")
        with gm.graph.inserting_before(node):
            const_full = gm.graph.call_function(
                aten.full.default,
                args=(list(attr.shape), fill_value),
                kwargs={"dtype": attr.dtype, "device": attr.device},
            )
            if src_val is not None:
                const_full.meta["val"] = src_val
        node.replace_all_uses_with(const_full)
        # Erase the dead get_attr node so GraphLowering.run_node never
        # processes it.  The module attribute is left intact so cache keys
        # remain stable.
        try:
            gm.graph.erase_node(node)
        except Exception:  # noqa: BLE001
            pass
        changed = True

    gm.meta["_pf52_materialized"] = True
    if changed:
        try:
            gm.graph.lint()
            gm.recompile()
        except Exception:  # noqa: BLE001
            pass
    return gm
