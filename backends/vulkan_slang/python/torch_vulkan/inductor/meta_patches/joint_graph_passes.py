"""Joint-graph and backward-compilation device-translation passes.

Rewrites device='meta' to device='vulkan' in the joint graph before
partitioning, hooks into AOT custom passes, and handles lifetime
annotation for Vulkan tensors.
"""

from __future__ import annotations

import threading

import torch

_tls = threading.local()


class _joint_trace_ctx:
    __slots__ = ()

    def __enter__(self):
        _tls._in_joint_trace = True
        return self

    def __exit__(self, *exc):
        _tls._in_joint_trace = False
        return False


class _FixMetaDevicePass:
    """Rewrite device='meta' to device='vulkan' in the joint graph.

    M15.2 audit (b): workaround for AOTAutograd producing meta-device tensors
    during joint tracing. Should be fixed by proper joint tracing that
    preserves Vulkan device info throughout. See roadmap M15 (cleanup).
    """

    __bases__ = ()

    def __call__(self, gm: torch.fx.GraphModule) -> None:
        _vulkan_dev = torch.device("vulkan", 0)
        _ft = torch._subclasses.fake_tensor

        fm = None
        for node in gm.graph.nodes:
            val = node.meta.get("val")
            if isinstance(val, _ft.FakeTensor):
                fm = val.fake_mode
                break

        modified = False

        for node in gm.graph.nodes:
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor) and val.device.type == "meta":
                if fm is not None and isinstance(val, _ft.FakeTensor):
                    node.meta["val"] = _ft.FakeTensor.__new__(
                        _ft.FakeTensor,
                        fm,
                        val,
                        device=_vulkan_dev,
                    )
                else:
                    node.meta["val"] = torch.empty_strided(
                        val.shape,
                        val.stride(),
                        dtype=val.dtype,
                        device=_vulkan_dev,
                    )
                modified = True
            elif isinstance(val, (list, tuple)):
                changed = False
                new_list = []
                for v in val:
                    if isinstance(v, torch.Tensor) and v.device.type == "meta":
                        if fm is not None and isinstance(v, _ft.FakeTensor):
                            new_list.append(
                                _ft.FakeTensor.__new__(
                                    _ft.FakeTensor,
                                    fm,
                                    v,
                                    device=_vulkan_dev,
                                )
                            )
                        else:
                            new_list.append(
                                torch.empty_strided(
                                    v.shape,
                                    v.stride(),
                                    dtype=v.dtype,
                                    device=_vulkan_dev,
                                )
                            )
                        changed = True
                    else:
                        new_list.append(v)
                if changed:
                    node.meta["val"] = type(val)(new_list)
                    modified = True

        def _fix(obj):
            if isinstance(obj, torch.device) and obj.type == "meta":
                nonlocal modified
                modified = True
                return _vulkan_dev
            if isinstance(obj, (list, tuple)):
                return type(obj)(_fix(x) for x in obj)
            if isinstance(obj, dict):
                return {k: _fix(v) for k, v in obj.items()}
            return obj

        for node in gm.graph.nodes:
            new_args = _fix(node.args)
            if new_args is not node.args:
                node.args = new_args
            new_kwargs = _fix(node.kwargs)
            if new_kwargs is not node.kwargs:
                node.kwargs = new_kwargs

        if modified:
            gm.graph.lint()
            gm.recompile()

    def uuid(self):
        return None

    @classmethod
    def _make_compatible(cls):
        from torch._inductor.custom_graph_pass import CustomGraphModulePass

        class _Compatible(CustomGraphModulePass):
            _inner = cls()

            def __call__(self, gm):
                self._inner.__call__(gm)

            def uuid(self):
                return self._inner.uuid()

        return _Compatible()


def _install_joint_partition_device_fix() -> None:
    """PF.1 / P8.1: rewrite ``device='meta'`` to ``device='vulkan'`` in the
    joint graph BEFORE partitioning, then deeply propagate vulkan device
    tags through every downstream FakeTensor val that consumes a vulkan
    input.

    M15.2 audit (b): workaround for AOTAutograd joint-trace device loss.
    The 3-stage pass (factory rewrite, deep device propagation, arg/kwarg
    rewrite) papers over the root cause: the joint trace doesn't preserve
    Vulkan device info. Should be fixed by proper joint tracing.
    The nested _rewrite_empty_meta_to_tangent_expand is also (b) — it
    works around expand-on-0-dim-FakeTensor losing the device tag.
    See roadmap M15 (cleanup).
    """

    # Stage 1 (narrow factory-op rewrite): walk the joint graph, find the
    # factory ops (``aten.empty*``, ``aten.zeros*``, ``aten.ones*``,
    # ``aten.full*``) whose kwargs contain ``torch.device('meta')``, and
    # rewrite to ``vulkan:0``. Patches ``node.meta['val']`` in lock-step.

    # Stage 2 (deep device propagation, PF.1 / B1'): walk every remaining
    # ``call_function`` node in topo order. If its ``meta['val']`` carries
    # ``device.type == 'meta'`` AND any of its Tensor arg-vals already
    # carries ``device.type == 'vulkan'``, restamp the val to vulkan via
    # ``_ft.FakeTensor.__new__`` (the FakeMode-preserving path — never
    # ``empty_strided``, which would silently install a real allocation
    # into the partitioner's cost-model input and collapse the backward).
    # Also recurse into list/tuple vals for multi-output ops.

    # Stage 3 (arg/kwarg device-literal rewrite): replaces any remaining
    # ``torch.device('meta')`` literals embedded in node args/kwargs with
    # the vulkan device, mirroring ``_FixMetaDevicePass`` so downstream
    # passes that introspect args also see consistent device info.

    # Why the deep propagation is correct: AOT autograd traced the joint
    # graph once with a meta-device tangent, so the factory op (e.g.
    # ``aten.empty.memory_format(..., device='meta')``) and every
    # downstream node that consumed it carry meta vals. After Stage 1
    # rewrites the factory device, the partitioner's ``get_device``
    # (`torch/_functorch/partitioners.py:1721`) inspects ``node.meta['val'].device``
    # on consumer nodes and would still see meta — placing the entire
    # backward branch on a CPU-default device. Stage 2 fixes the
    # consumers without re-running the dispatcher, so the partitioner's
    # cost model sees a single coherent vulkan device while all
    # FakeTensors stay FakeTensors.

    # Why ``empty_strided`` is forbidden during this pass: the previous
    # iteration of this code fell back to ``empty_strided(...)`` when
    # ``fm is None`` or the val wasn't a FakeTensor — that creates a real
    # Vulkan allocation that the partitioner's cost model treats as a
    # static input, dropping the node from the backward graph and yielding
    # an empty bw_module. We only restamp via FakeTensor.__new__; if no
    # FakeMode is in scope we skip the deep pass entirely.

    # Hooks into ``torch._functorch.config.joint_custom_pass``. Chains
    # with any pre-existing pass; short-circuits on non-Vulkan graphs.
    import torch
    import torch._functorch.config as _fc

    existing = _fc.joint_custom_pass
    if getattr(existing, "_vulkan_partition_pass", False):
        return  # idempotent

    _vulkan_dev = torch.device("vulkan", 0)
    _ft = torch._subclasses.fake_tensor

    _FACTORY_OPS = (
        torch.ops.aten.empty.memory_format,
        torch.ops.aten.empty_strided.default,
        torch.ops.aten.zeros.default,
        torch.ops.aten.ones.default,
        torch.ops.aten.full.default,
        torch.ops.aten.empty_like.default,
        torch.ops.aten.zeros_like.default,
        torch.ops.aten.ones_like.default,
        torch.ops.aten.full_like.default,
    )

    def _has_vulkan_input(joint_inputs) -> bool:
        def _check(t) -> bool:
            return isinstance(t, torch.Tensor) and t.device.type == "vulkan"

        if isinstance(joint_inputs, (list, tuple)):
            for inp in joint_inputs:
                if _check(inp):
                    return True
                if isinstance(inp, (list, tuple)):
                    for sub in inp:
                        if _check(sub):
                            return True
        return False

    def _restamp_to_vulkan(val, fm):
        """Return a new vulkan-device FakeTensor mirroring ``val``'s
        shape/dtype/strides, sharing FakeMode ``fm``. ``val`` must be a
        meta-device FakeTensor; other inputs are returned unchanged.
        """
        if isinstance(val, _ft.FakeTensor) and val.device.type == "meta":
            return _ft.FakeTensor.__new__(
                _ft.FakeTensor,
                fm,
                val,
                device=_vulkan_dev,
            )
        return val

    def _val_has_vulkan_tensor(val) -> bool:
        if isinstance(val, torch.Tensor):
            return val.device.type == "vulkan"
        if isinstance(val, (list, tuple)):
            return any(_val_has_vulkan_tensor(v) for v in val)
        return False

    def _val_has_meta_tensor(val) -> bool:
        if isinstance(val, torch.Tensor):
            return val.device.type == "meta"
        if isinstance(val, (list, tuple)):
            return any(_val_has_meta_tensor(v) for v in val)
        return False

    def _restamp_val(val, fm):
        if isinstance(val, _ft.FakeTensor):
            return _restamp_to_vulkan(val, fm)
        if isinstance(val, list):
            return [_restamp_val(v, fm) for v in val]
        if isinstance(val, tuple):
            return tuple(_restamp_val(v, fm) for v in val)
        return val

    def _stamp_factory_devices(fx_g):
        # Discover the FakeMode in scope so we mint Vulkan FakeTensors
        # that share it with the rest of the graph's metadata.
        fm = None
        for node in fx_g.graph.nodes:
            val = node.meta.get("val")
            if isinstance(val, _ft.FakeTensor):
                fm = val.fake_mode
                break

        # Stage 1: rewrite factory-op device kwargs from meta → vulkan
        # and re-stamp the matching node.meta['val'] in lock-step.
        code_modified = False
        for node in fx_g.graph.nodes:
            if node.op != "call_function" or node.target not in _FACTORY_OPS:
                continue
            new_kwargs = dict(node.kwargs)
            kw_device = new_kwargs.get("device")
            if isinstance(kw_device, torch.device) and kw_device.type == "meta":
                new_kwargs["device"] = _vulkan_dev
                node.kwargs = new_kwargs
                code_modified = True
                val = node.meta.get("val")
                if (
                    fm is not None
                    and isinstance(val, _ft.FakeTensor)
                    and val.device.type == "meta"
                ):
                    node.meta["val"] = _restamp_to_vulkan(val, fm)

        # Stage 2 (PF.1): deep device propagation. Walk in topo order
        # (graph.nodes is topologically sorted). For each call_function
        # node whose val is meta and whose input vals contain at least
        # one vulkan tensor, restamp the meta vals to vulkan.
        if fm is not None:
            for node in fx_g.graph.nodes:
                if node.op != "call_function":
                    continue
                val = node.meta.get("val")
                if not _val_has_meta_tensor(val):
                    continue
                # Look at every input node's val; if any is vulkan, this
                # node is a vulkan computation that the joint trace
                # mistakenly stamped as meta.
                has_vulkan_input = False
                for input_node in node.all_input_nodes:
                    if _val_has_vulkan_tensor(input_node.meta.get("val")):
                        has_vulkan_input = True
                        break
                if has_vulkan_input:
                    node.meta["val"] = _restamp_val(val, fm)

        # Stage 3: rewrite any torch.device('meta') literals embedded
        # in node args/kwargs (e.g. cast-style ops with explicit device
        # kwargs that the joint trace captured as meta).
        def _fix(obj):
            nonlocal code_modified
            if isinstance(obj, torch.device) and obj.type == "meta":
                code_modified = True
                return _vulkan_dev
            if isinstance(obj, list):
                return [_fix(x) for x in obj]
            if isinstance(obj, tuple):
                return tuple(_fix(x) for x in obj)
            if isinstance(obj, dict):
                return {k: _fix(v) for k, v in obj.items()}
            return obj

        for node in fx_g.graph.nodes:
            if node.op != "call_function":
                continue
            new_args = _fix(node.args)
            if new_args is not node.args:
                node.args = new_args
            new_kwargs = _fix(node.kwargs)
            if new_kwargs is not node.kwargs:
                node.kwargs = new_kwargs

        if code_modified:
            fx_g.graph.lint()
            fx_g.recompile()
        return fx_g

    def _rewrite_empty_meta_to_tangent_expand(fx_g):
        """PF.13 root fix: replace uninitialized ``aten.empty(shape, device=meta)``
        nodes in the joint graph with ``aten.expand(tangents_X, shape)``.

        Why this is needed: when AOT autograd's joint trace evaluates a
        backward formula like ``SumBackward0 -> grad.expand_symint(self_sizes)``
        with ``grad`` a vulkan FakeTensor of shape ``[]``, the proxy-tensor
        tracer captures the result as ``aten.empty.memory_format(shape,
        device='meta')`` instead of ``aten.expand.default(grad, shape)`` —
        because expand on a 0-dim FakeTensor under ``in_kernel_invocation``
        loses the device tag and proxy_tensor materializes the result as a
        fresh meta-empty allocation. The resulting BW graph has the
        ``tangents_X`` input completely unused and reads uninitialized
        memory in its place — the canonical NaN/0.5×-grad bug.

        Heuristic match: any ``aten.empty.memory_format`` node with no
        tensor inputs (purely a sizes/dtype/device factory call) whose
        ``device`` argument is meta. Pair it with the unique tangent
        placeholder of matching dtype, expanding from the tangent's
        smaller shape to the empty's target shape.

        Vulkan-only: only fires after ``_has_vulkan_input`` has gated us in.
        """
        # Collect tangent placeholders. AOT autograd names them
        # ``tangents_1``, ``tangents_2``, .... We match by dtype because
        # the trace doesn't preserve which tangent the empty was meant
        # to materialize.
        tangent_placeholders = []
        for node in fx_g.graph.nodes:
            if node.op != "placeholder":
                continue
            if not str(node.target).startswith("tangents"):
                continue
            tangent_placeholders.append(node)

        if not tangent_placeholders:
            return fx_g

        def _tangent_for_dtype(dtype):
            for t in tangent_placeholders:
                v = t.meta.get("val")
                if isinstance(v, torch.Tensor) and v.dtype == dtype:
                    return t
            return None

        modified = False
        for node in list(fx_g.graph.nodes):
            if node.op != "call_function":
                continue
            if node.target is not torch.ops.aten.empty.memory_format:
                continue
            # Pure factory: shape is args[0], dtype/device/etc in kwargs.
            kw_device = node.kwargs.get("device")
            if not (isinstance(kw_device, torch.device) and kw_device.type == "meta"):
                continue
            target_shape = node.args[0]
            target_dtype = node.kwargs.get("dtype", torch.float32)
            tangent = _tangent_for_dtype(target_dtype)
            if tangent is None:
                continue
            # Build ``aten.expand.default(tangent, target_shape)``. expand
            # broadcasts a 0-dim or smaller-rank source to the requested
            # shape using stride-0 views — exactly the semantics
            # ``grad.expand_symint(sizes)`` was meant to capture.
            with fx_g.graph.inserting_before(node):
                expand_node = fx_g.graph.call_function(
                    torch.ops.aten.expand.default,
                    (tangent, list(target_shape)),
                )
            # Carry over the val so downstream FakeTensor-aware passes
            # (lifetime annotation, partitioner cost model) see a vulkan
            # tensor of the right shape/dtype/device.
            expand_node.meta = dict(node.meta)
            tangent_val = tangent.meta.get("val")
            if isinstance(tangent_val, torch.Tensor):
                fm = getattr(tangent_val, "fake_mode", None)
                if fm is not None:
                    expand_node.meta["val"] = fm.from_tensor(
                        torch.empty(
                            list(target_shape),
                            dtype=target_dtype,
                            device=_vulkan_dev,
                        ),
                        static_shapes=True,
                    )
            node.replace_all_uses_with(expand_node)
            fx_g.graph.erase_node(node)
            modified = True

        if modified:
            fx_g.graph.lint()
            fx_g.recompile()
        return fx_g

    def _rewrite_constant_folded_tangent(fx_g):
        """M-NEW.9 + M-AUDIT-PERF.1-followup — undo AOTAutograd's constant
        fold of the implicit upstream gradient.

        When the user calls ``loss.backward()`` on a scalar loss, AOT autograd
        traces the joint graph with a *concrete* ``tangents_1 = torch.ones(())``
        on the vulkan device. The partitioner's constant-folder then evaluates
        ``expand(tangents_1, target_shape)`` and lifts the result as
        ``self._tensor_constantN`` on the GraphModule — a vulkan tensor with
        ``data_ptr() == 0`` (null storage). At runtime the wrapper never
        allocates a real buffer for it, so every read returns garbage / zero
        and **every reachable gradient is zero**.

        The placeholder ``tangents_N`` is still present in the graph but has
        ``users == 0``. The backward module's actual upstream gradient (passed
        in at run time by the autograd engine) is silently discarded.

        Fix: walk the joint graph, find each ``tangents_N`` placeholder with
        zero users and a matching ``get_attr`` whose:

          - dtype matches the placeholder val's dtype,
          - shape is broadcast-compatible *from* the placeholder val's shape
            (i.e. the placeholder shape is a suffix of the get_attr shape,
            or the placeholder is 0-dim), and
          - the underlying tensor has ``data_ptr() == 0`` OR all-zero
            strides (the canonical broadcast-only constant-fold pattern).

        Replace every consumer of the get_attr with
        ``aten.expand.default(tangents_N, target_shape)``. The placeholder is
        now live, so the runtime wrapper threads the real upstream gradient
        through. The get_attr is left in place — the constant attribute is
        owned by the partitioner; erasing it can confuse downstream cache
        keys, and once it has no users it costs nothing.

        Handles both:
          - **Scalar tangent** (``sum().backward()`` etc.): placeholder
            ``tangents_1`` shape ``[]``, get_attr shape e.g. ``[8, 16]``.
          - **Non-scalar tangent** (``sum(dim=0).sum().backward()`` etc., or
            ``.backward(grad)`` with a non-scalar grad): placeholder shape
            ``[k1, ..., kN]``, get_attr shape ``[m1, ..., mP, k1, ..., kN]``.

        Vulkan-only — gated by the outer ``_has_vulkan_input`` check on
        ``_chained``.
        """
        import re

        tangent_re = re.compile(r"^tangents?_\d+$")

        # 1. Collect unused tangent placeholders.
        unused_tangents: list = []
        for node in fx_g.graph.nodes:
            if node.op != "placeholder":
                continue
            if not tangent_re.match(node.name):
                continue
            if len(node.users) != 0:
                continue
            val = node.meta.get("val")
            if not isinstance(val, torch.Tensor):
                continue
            unused_tangents.append(node)

        if not unused_tangents:
            return fx_g

        # 2. Collect candidate get_attr nodes (null-storage or zero-stride
        #    broadcast on a vulkan tensor).
        def _is_null_storage(t: torch.Tensor) -> bool:
            try:
                return t.data_ptr() == 0
            except RuntimeError:
                return True

        def _is_zero_stride_broadcast(t: torch.Tensor) -> bool:
            try:
                strides = t.stride()
            except Exception:  # noqa: BLE001
                return False
            if not strides:
                return False
            if any(s != 0 for s in strides):
                return False
            return t.dim() > 0

        candidates: list = []  # (get_attr_node, attr_tensor)
        for node in fx_g.graph.nodes:
            if node.op != "get_attr":
                continue
            try:
                attr = getattr(fx_g, node.target, None)
            except Exception:  # noqa: BLE001
                attr = None
            if not isinstance(attr, torch.Tensor):
                continue
            if attr.device.type != "vulkan":
                continue
            if not (_is_null_storage(attr) or _is_zero_stride_broadcast(attr)):
                continue
            candidates.append((node, attr))

        if not candidates:
            return fx_g

        # 3. Compute a view-shape that lets tangent_shape broadcast to
        #    target_shape. Walks right-to-left greedily: each tangent dim
        #    either matches a target dim or is treated as size-1; extra
        #    target dims become size-1 in the view. Returns None if the
        #    tangent has unconsumed dims (no compatible broadcasting).
        #
        #    Examples:
        #      [] → [8, 16]              => [1, 1]
        #      [16] → [8, 16, 32]        => [1, 16, 1]   (sum(dim=[0,2]))
        #      [64] → [8, 64]            => [1, 64]      (sum(dim=0))
        #      [16, 32] → [8, 16, 32]    => [1, 16, 32]  (sum(dim=0))
        #      [4] → [4, 32]             => [4, 1]       (mean(dim=-1))
        #      [4, 8] → [4, 8, 16]       => [4, 8, 1]    (mean(dim=-1))
        def _compute_view_shape(t_shape, g_shape):
            if len(t_shape) > len(g_shape):
                return None
            view = [1] * len(g_shape)
            t_idx = len(t_shape) - 1
            for g_idx in range(len(g_shape) - 1, -1, -1):
                if t_idx < 0:
                    break
                ts = t_shape[t_idx]
                gs = g_shape[g_idx]
                if ts == gs or ts == 1:
                    view[g_idx] = ts
                    t_idx -= 1
                # Otherwise leave view[g_idx] = 1 (this target dim is broadcast)
            if t_idx >= 0:
                return None  # leftover tangent dims — incompatible
            return view

        modified = False
        used_candidates: set = set()
        for tangent in unused_tangents:
            t_val = tangent.meta["val"]
            t_shape = tuple(t_val.shape)
            t_dtype = t_val.dtype

            # Pick the first unused candidate whose dtype matches and whose
            # shape admits a view-shape for broadcasting from the tangent.
            picked = None
            picked_view = None
            for (gnode, attr) in candidates:
                if id(gnode) in used_candidates:
                    continue
                if attr.dtype != t_dtype:
                    continue
                view_shape = _compute_view_shape(t_shape, tuple(attr.shape))
                if view_shape is None:
                    continue
                picked = (gnode, attr)
                picked_view = view_shape
                break

            if picked is None:
                continue
            gnode, attr = picked
            used_candidates.add(id(gnode))
            target_shape = list(attr.shape)

            # Insert (optionally) view(tangent, view_shape) then
            # expand(..., target_shape). Skip view when t_shape already
            # equals view_shape (no reshape needed). When tangent is 0-dim
            # (`[]`), aten.expand handles broadcasting directly without a
            # view step.
            tv_val = tangent.meta.get("val")
            fm = None
            if isinstance(tv_val, _ft.FakeTensor):
                fm = tv_val.fake_mode

            # Insert view (if needed) and expand in topological order.
            # Each inserting_after() context pushes new nodes immediately
            # after the anchor, so we must anchor the SECOND insertion on
            # the FIRST node (not on `tangent`) — otherwise the expand
            # ends up positioned before its source `view`.
            #
            # IMPORTANT: mint FakeTensor metadata using the tangent's
            # actual device (preserving index=None vs index=0). PyTorch
            # treats ``torch.device("vulkan")`` (index=None) and
            # ``torch.device("vulkan:0")`` (index=0) as DIFFERENT devices
            # — the autograd engine cross-checks the gradient device
            # against the parameter device exactly, so a hardcoded
            # ``_vulkan_dev = torch.device("vulkan", 0)`` here produces
            # gradients on ``vulkan:0`` and trips
            # ``RuntimeError: Function X returned an invalid gradient ...
            #   expected device vulkan but got vulkan:0`` when the user
            # model lives on the un-indexed ``vulkan`` device.
            t_device = tv_val.device if isinstance(tv_val, torch.Tensor) else _vulkan_dev
            source = tangent
            anchor = tangent
            if list(t_shape) != picked_view and len(t_shape) > 0:
                with fx_g.graph.inserting_after(anchor):
                    view_node = fx_g.graph.call_function(
                        torch.ops.aten.view.default,
                        (tangent, list(picked_view)),
                    )
                if fm is not None:
                    view_node.meta["val"] = fm.from_tensor(
                        torch.empty(
                            list(picked_view),
                            dtype=t_dtype,
                            device=t_device,
                        ),
                        static_shapes=True,
                    )
                source = view_node
                anchor = view_node
            with fx_g.graph.inserting_after(anchor):
                expand_node = fx_g.graph.call_function(
                    torch.ops.aten.expand.default,
                    (source, target_shape),
                )
            # Borrow gnode meta so downstream FakeTensor / lifetime passes
            # see a vulkan tensor of the right shape/dtype.
            expand_node.meta = dict(gnode.meta)
            v = expand_node.meta.get("val")
            if not (
                isinstance(v, torch.Tensor) and v.device.type == "vulkan"
            ) and fm is not None:
                expand_node.meta["val"] = fm.from_tensor(
                    torch.empty(
                        target_shape,
                        dtype=t_dtype,
                        device=t_device,
                    ),
                    static_shapes=True,
                )
            gnode.replace_all_uses_with(expand_node)
            modified = True

        if modified:
            fx_g.graph.lint()
            fx_g.recompile()
        return fx_g

    def _rewrite_factory_meta_to_vulkan(fx_g):
        """S4.3 — extend factory-op device stamping to all _FACTORY_OPS.

        Two sub-cases are handled here:

        1. **Factory op still in the graph with device='meta'** (edge case
           where _stamp_factory_devices Stage 1 missed it): update the device
           kwarg to 'vulkan' and re-stamp node.meta['val'].

        2. **Constant-folded factory op → get_attr pointing to a CPU tensor**
           (the main S4.3 bug): torch.export lifts the concrete result of
           torch.zeros/ones/full/etc. as a graph-module constant. The joint
           graph then has a ``get_attr`` node whose ``node.meta['val']`` is a
           real CPU ``torch.Tensor`` (not a FakeTensor). Stage 1 of
           _stamp_factory_devices only rewrites ``call_function`` nodes, and
           Stage 2 calls _val_has_meta_tensor() which returns False for a
           real CPU tensor — so the constant is silently left on CPU. At .so
           runtime the tensor is allocated on the default CPU device instead
           of Vulkan.

        Fix for case 2: insert ``aten._to_copy.default(get_attr,
           device='vulkan')`` immediately after the get_attr node and update
           the downstream uses so the partitioner sees a coherent device.
           We only apply this to get_attr nodes whose target is in the graph
           module's state dict under an ``_exported_training`` prefix — those
           are the torch.export constant entries, not user model parameters
           or buffers.

        Vulkan-only — gated by the outer _has_vulkan_input check in _chained.
        """
        def _is_factory_constant_getattr(node, fx_g):
            if node.op != "get_attr":
                return False
            target = node.target
            if not isinstance(target, str) or not target.startswith("_exported_training"):
                return False
            try:
                attr = getattr(fx_g, target, None)
            except Exception:
                return False
            return isinstance(attr, torch.Tensor) and attr.device.type == "cpu"

        fm = None
        for _n in fx_g.graph.nodes:
            _v = _n.meta.get("val")
            if isinstance(_v, _ft.FakeTensor):
                fm = _v.fake_mode
                break

        modified = False
        for node in list(fx_g.graph.nodes):
            if node.op != "call_function":
                continue
            if node.target not in _FACTORY_OPS:
                continue
            # Case 1: factory op still in the graph, device kwarg is meta
            # (edge case — Stage 1 should normally catch this).
            kw_device = node.kwargs.get("device")
            if isinstance(kw_device, torch.device) and kw_device.type == "meta":
                new_kwargs = dict(node.kwargs)
                new_kwargs["device"] = _vulkan_dev
                node.kwargs = new_kwargs
                val = node.meta.get("val")
                if (
                    fm is not None
                    and isinstance(val, _ft.FakeTensor)
                    and val.device.type == "meta"
                ):
                    node.meta["val"] = _restamp_to_vulkan(val, fm)
                modified = True

        # Case 2: constant-folded factory op → get_attr → CPU tensor.
        if fm is not None:
            for node in list(fx_g.graph.nodes):
                if not _is_factory_constant_getattr(node, fx_g):
                    continue
                # Insert aten._to_copy.default(node, device='vulkan') right
                # after the get_attr.  This is the minimal, correct way to
                # move a concrete constant tensor to the Vulkan device in the
                # FX graph without touching any model parameter/buffer nodes.
                with fx_g.graph.inserting_after(node):
                    to_node = fx_g.graph.call_function(
                        torch.ops.aten._to_copy.default,
                        (node,),
                        {"device": _vulkan_dev, "dtype": None, "non_blocking": False},
                    )
                # Stamp the new node's val so the partitioner sees vulkan.
                src_val = node.meta.get("val")
                if isinstance(src_val, torch.Tensor):
                    to_node.meta["val"] = fm.from_tensor(
                        src_val.new_empty(src_val.shape, device=_vulkan_dev),
                        static_shapes=True,
                    )
                node.replace_all_uses_with(to_node)
                modified = True

        if modified:
            fx_g.graph.lint()
            fx_g.recompile()
        return fx_g

    def _chained(fx_g, joint_inputs):
        if callable(existing):
            fx_g = existing(fx_g, joint_inputs)
        if _has_vulkan_input(joint_inputs):
            # Order matters: replace empty(meta)→expand(tangent) BEFORE
            # the device-stamp pass so we don't end up with empty(vulkan)
            # zombies that read uninitialized memory at runtime.
            fx_g = _rewrite_empty_meta_to_tangent_expand(fx_g)
            # M-NEW.9 + M-AUDIT-PERF.1-followup: rewrite constant-folded
            # tangent get_attrs back to expand(tangent_placeholder, shape).
            # Must run after _rewrite_empty_meta_to_tangent_expand (which
            # also might leave tangents unused) but BEFORE device-stamp /
            # lifetime annotation (which expect a coherent graph).
            fx_g = _rewrite_constant_folded_tangent(fx_g)
            # S4.3: extend factory-op shim to all _FACTORY_OPS.
            # Runs after _rewrite_constant_folded_tangent (which may also
            # leave tangents unused) but BEFORE _stamp_factory_devices so
            # the get_attr → _to_copy(vulkan) rewrites are visible when
            # the device-stamp pass walks the graph.
            fx_g = _rewrite_factory_meta_to_vulkan(fx_g)
            fx_g = _stamp_factory_devices(fx_g)
            # PF.40: annotate node.meta["lifetime_class"] on the joint
            # graph. The annotation propagates into fw_module / bw_module
            # because the partitioner copies node meta. Consumed by
            # PF.41 (StepActivationPool) and PF.42 (step-end release hook).
            from torch_vulkan.inductor.lifetime import (
                annotate_lifetime_classes,
            )

            fx_g = annotate_lifetime_classes(fx_g, joint_inputs)
            # COMPILE.2: tag 0-d div nodes as must_be_in_forward so the
            # AOT partitioner doesn't split them into the backward
            # sub-graph (which would cause "Node was invalid" crashes
            # on conv + cross_entropy + backward models).
            from torch_vulkan.inductor.fx_passes.pre_aot_partition import (
                mark_0d_div_must_be_in_forward,
            )

            fx_g = mark_0d_div_must_be_in_forward(fx_g)
        return fx_g

    _chained._vulkan_partition_pass = True  # type: ignore[attr-defined]
    _fc.joint_custom_pass = _chained


def _patch_compile_fx_for_backward() -> None:
    try:
        from torch._inductor.codegen.common import register_backend_for_device
        from torch._inductor.custom_graph_pass import CustomGraphModulePass

        from ..scheduling import VulkanScheduling
        from ..wrapper import VulkanPythonWrapperCodegen

        class _Compatible(CustomGraphModulePass):
            _inner = _FixMetaDevicePass()

            def __call__(self, gm):
                self._inner.__call__(gm)

            def uuid(self):
                return self._inner.uuid()

        register_backend_for_device(
            "meta",
            VulkanScheduling,
            VulkanPythonWrapperCodegen,
            None,
            None,
            _Compatible(),
        )
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(
            "Registering meta→VulkanScheduling alias failed: %s", e
        )


def _rewrite_factory_meta_to_vulkan(
    gm: "torch.fx.GraphModule",
) -> "torch.fx.GraphModule":
    """Rewrite meta-device factory ops and CPU training constants to vulkan.

    Stage 1 — call_function nodes whose target is a factory op and whose
    device kwarg is torch.device('meta') are rewritten to vulkan:0.
    node.meta['val'] is restamped in lock-step when a FakeMode is in scope.

    Stage 2 — get_attr nodes whose target contains _exported_training and
    whose meta['val'] is on a non-vulkan device get an aten._to_copy.default
    node inserted immediately after them. All downstream users of the original
    get_attr are rewired to the copy node; the copy node retains the original
    get_attr as its first argument so the graph remains acyclic.
    """
    _vulkan_dev = torch.device("vulkan", 0)
    _ft = torch._subclasses.fake_tensor

    _FACTORY_OPS = frozenset((
        torch.ops.aten.empty.memory_format,
        torch.ops.aten.empty_strided.default,
        torch.ops.aten.zeros.default,
        torch.ops.aten.ones.default,
        torch.ops.aten.full.default,
        torch.ops.aten.empty_like.default,
        torch.ops.aten.zeros_like.default,
        torch.ops.aten.ones_like.default,
        torch.ops.aten.full_like.default,
    ))

    fm = None
    for _n in gm.graph.nodes:
        _v = _n.meta.get("val")
        if isinstance(_v, _ft.FakeTensor):
            fm = _v.fake_mode
            break

    structure_modified = False

    # Stage 1: factory-op device kwargs meta -> vulkan (no new nodes)
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in _FACTORY_OPS:
            continue
        kw_device = node.kwargs.get("device")
        if not (isinstance(kw_device, torch.device) and kw_device.type == "meta"):
            continue
        node.kwargs = {**node.kwargs, "device": _vulkan_dev}
        val = node.meta.get("val")
        if (
            fm is not None
            and isinstance(val, _ft.FakeTensor)
            and val.device.type == "meta"
        ):
            node.meta["val"] = _ft.FakeTensor.__new__(
                _ft.FakeTensor, fm, val, device=_vulkan_dev
            )

    # Stage 2: get_attr for _exported_training* CPU constants -> _to_copy
    for node in list(gm.graph.nodes):
        if node.op != "get_attr":
            continue
        target = node.target
        if not (isinstance(target, str) and "_exported_training" in target):
            continue
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor) and val.device.type == "vulkan":
            continue

        with gm.graph.inserting_after(node):
            copy_node = gm.graph.call_function(
                torch.ops.aten._to_copy.default,
                (node,),
                {"device": _vulkan_dev},
            )

        if fm is not None and isinstance(val, _ft.FakeTensor):
            # FakeTensor.__new__ requires elem.device == 'meta'
            _meta_elem = torch.empty(
                tuple(val.shape), dtype=val.dtype, device="meta"
            )
            copy_node.meta["val"] = _ft.FakeTensor.__new__(
                _ft.FakeTensor, fm, _meta_elem, device=_vulkan_dev
            )
        elif isinstance(val, torch.Tensor):
            copy_node.meta = dict(node.meta)

        node.replace_all_uses_with(
            copy_node,
            delete_user_cb=lambda user: user is not copy_node,
        )
        structure_modified = True

    if structure_modified:
        gm.graph.lint()
        gm.recompile()

    return gm


def _skip_misc_patterns_for_vulkan() -> None:
    try:
        from torch._inductor.fx_passes import misc_patterns as _mp

        _orig_init = getattr(_mp, "_misc_patterns_init", None)
        if _orig_init is None:
            return

        def _patched_misc_init(device):
            dev_type = getattr(device, "type", str(device))
            if "privateuseone" in dev_type or "vulkan" in dev_type:
                return
            return _orig_init(device)

        _mp._misc_patterns_init = _patched_misc_init
    except Exception:
        pass
