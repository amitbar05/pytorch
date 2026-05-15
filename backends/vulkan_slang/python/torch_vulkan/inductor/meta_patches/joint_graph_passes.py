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

    def _chained(fx_g, joint_inputs):
        if callable(existing):
            fx_g = existing(fx_g, joint_inputs)
        if _has_vulkan_input(joint_inputs):
            # Order matters: replace empty(meta)→expand(tangent) BEFORE
            # the device-stamp pass so we don't end up with empty(vulkan)
            # zombies that read uninitialized memory at runtime.
            fx_g = _rewrite_empty_meta_to_tangent_expand(fx_g)
            fx_g = _stamp_factory_devices(fx_g)
            # PF.40: annotate node.meta["lifetime_class"] on the joint
            # graph. The annotation propagates into fw_module / bw_module
            # because the partitioner copies node meta. Consumed by
            # PF.41 (StepActivationPool) and PF.42 (step-end release hook).
            from torch_vulkan.inductor.lifetime import (
                annotate_lifetime_classes,
            )

            fx_g = annotate_lifetime_classes(fx_g, joint_inputs)
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
