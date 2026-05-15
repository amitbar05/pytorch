"""BMM and matmul lowerings (PF.56, GAP 0)."""

from __future__ import annotations


def _register_bmm_lowering() -> None:
    """PF.56 — `aten.bmm.default` Vulkan lowering surface.

    `nn.MultiheadAttention` / `transformer_block` decompose into
    `aten.bmm.default` directly (no `torch_vulkan::*` shim). The actual
    lowering is installed by ``install_external_bmm`` in
    [`vulkan_template_caller.py`](./vulkan_template_caller.py) — it
    overrides Inductor's stock ``tuned_bmm`` with a Vulkan-aware version
    that adds Slang tiled-bmm callables as ``ExternKernelChoice``
    options alongside the default ``aten_bmm`` (which dispatches to the
    eager C++ ``vulkan_bmm`` shader). The autotuner picks the faster
    one. Vulkan f32 inputs go through one of the Slang tile shaders
    (`slang_mm.py.jinja` rendered with one of the `_pick_tile_configs`
    sets); non-Vulkan inputs fall through to the captured upstream
    ``tuned_bmm``.

    This function is the audit-visible companion: it ensures
    ``install_external_bmm`` has run by the time ``register()``
    completes, and ``aten.bmm.default`` appears in this file so
    ``audit_inductor_op_coverage.py``'s coverage scan picks it up.
    Idempotent — ``install_external_bmm`` early-returns on subsequent
    calls.

    Stage tag: ``BUG_ROOT="lowering"``.
    """
    # Trigger the install. ``__init__.py`` already calls this at backend
    # register time; the duplicate call is a no-op (guarded by
    # ``_bmm_installed``) but keeps this surface self-contained so a
    # caller that imports ``lowerings.register()`` directly (without
    # going through the inductor sub-package's full init) still gets
    # the bmm lowering wired. The lowering target is
    # ``aten.bmm.default`` (registered against ``aten.bmm`` because
    # OpOverload-level registration covers ``.default``).
    from ..vulkan_template_caller import install_external_bmm

    install_external_bmm()


def _register_mm_lowering() -> None:
    """TRAIN.1-F1 — Custom ``aten.mm`` lowering that bypasses
    ``unwrap_storage_for_input`` → ``realize_input`` for Vulkan tensors.

    Upstream Inductor's ``ExternKernel.unwrap_storage_for_input`` calls
    ``realize_input`` on every ``BaseView`` (non-``ReinterpretView``)
    input, which materializes the view into a contiguous buffer through
    three copy dispatches (``copy_as_strided → copy_copy → copy_as_strided``).
    Each ``aten.mm`` call pays this cost twice (once per operand), adding
    6 dispatches to every matmul.

    Our C++ ``vulkan_mm_out`` already handles non-contiguous inputs via
    the ``is_t_transposed`` fast-path — it recovers the physical storage
    from transposed views and passes stride flags through to the GPU
    compute shader.  The materialization is pure overhead.

    This lowering creates an ``ExternKernelOut`` directly (skipping
    ``tuned_mm``'s autotuning for now) and uses a thin subclass that
    overrides ``unwrap_storage_for_input`` to convert Vulkan
    ``BaseView`` inputs into ``ReinterpretView`` nodes instead of
    realizing them.  Non-Vulkan tensors fall through to the
    captured upstream ``tuned_mm``.

    Registered for ``aten.mm`` (which covers ``aten.mm.default`` via
    OpOverloadPacket registration).
    """
    import torch
    from torch._inductor import ir
    from torch._inductor import lowering as L
    from torch._inductor.lowering import register_lowering

    aten = torch.ops.aten
    _orig_mm = L.lowerings.get(aten.mm)

    # Thin subclass of ExternKernelOut that skips realize_input for
    # Vulkan BaseView inputs.  The override is minimal and surgical:
    # only the unwrap path for BaseView (non-ReinterpretView) nodes
    # on Vulkan devices is changed; everything else delegates to the
    # standard ExternKernel code path.
    class _VulkanMMOut(ir.ExternKernelOut):
        @classmethod
        def unwrap_storage_for_input(cls, x):
            if isinstance(x, ir.TensorBox):
                x = x.data
            if isinstance(x, ir.StorageBox):
                x = x.data
            if isinstance(x, ir.BaseView) and not isinstance(x, ir.ReinterpretView):
                try:
                    dev = x.get_device()
                    if dev is not None and dev.type == "vulkan":
                        # Vulkan matmul kernels handle non-contiguous
                        # inputs directly.  Convert the BaseView to a
                        # ReinterpretView so upstream codegen treats
                        # it as a layout-carrying node without
                        # triggering realize_input.
                        layout = x.get_layout()
                        unwrapped = x.unwrap_view()
                        return ir.ReinterpretView(data=unwrapped, layout=layout)
                except Exception:
                    pass
                # Non-Vulkan or error: fall through to standard realize.
                x = ir.ExternKernel.realize_input(x)
            if isinstance(x, ir.TensorBox):
                return cls.unwrap_storage_for_input(x)
            if isinstance(x, ir.TorchBindObject):
                return x
            assert isinstance(x, (ir.Buffer, ir.ReinterpretView)), type(x)
            return x

    @register_lowering(aten.mm, type_promotion_kind=None)
    def _vulkan_mm(tensor1, tensor2, *, layout=None):
        t1_dev = tensor1.get_device()
        if t1_dev is None or t1_dev.type != "vulkan":
            if _orig_mm is not None:
                return _orig_mm(tensor1, tensor2, layout=layout)
            return NotImplemented

        # OP.24: For int8 inputs, route through torch_vulkan::mm_int8 custom op.
        # The op is registered as a fallback kernel via make_fallback, which
        # means L.lowerings[torch.ops.torch_vulkan.mm_int8] exists and will
        # create the correct FallbackKernel IR node.  At runtime, the Vulkan
        # implementation dispatches through the Slang int8 tiled matmul template.
        # Output is float32 (int8×int8→int32 accumulation→float32).
        t1_dtype = tensor1.get_dtype()
        if t1_dtype == torch.int8 or t1_dtype == torch.uint8:
            from .mm_int8_op import _register_mm_int8_op

            _register_mm_int8_op()  # idempotent

            mm_int8_lowering = L.lowerings.get(torch.ops.torch_vulkan.mm_int8)
            if mm_int8_lowering is not None:
                return mm_int8_lowering(tensor1, tensor2, layout=layout)

            # Fallback: if for some reason the lowerings aren't registered,
            # fall through to the original mm path (will likely error).
            if _orig_mm is not None:
                return _orig_mm(tensor1, tensor2, layout=layout)
            return NotImplemented

        # For Vulkan tensors: use our custom kernel that skips
        # realize_input.  We construct the layout from the input
        # shapes (standard mm: M×K @ K×N → M×N).
        t1_size = tensor1.get_size()
        t2_size = tensor2.get_size()
        assert len(t1_size) == 2 and len(t2_size) == 2, "aten.mm expects 2D tensors"
        M = t1_size[0]
        N = t2_size[1]

        out_layout = ir.FixedLayout(
            device=t1_dev,
            dtype=tensor1.get_dtype(),
            size=[M, N],
            stride=[N, 1],
        )

        kernel = _VulkanMMOut(
            layout=out_layout,
            inputs=[tensor1, tensor2],
            python_kernel_name="torch.ops.aten.mm.out",
            op_overload=aten.mm.out,
        )
        return ir.TensorBox.create(kernel)


def _register_matmul_lowering() -> None:
    """GAP 0 — `aten.matmul.default` → `aten.bmm.default` shortcut for 3D.

    Without this, the upstream ``matmul`` decomposition creates
    ``expand → reshape → bmm → view`` IR nodes for 3D×3D inputs.
    The scheduler materializes the expand/reshape intermediates as
    constant-filled buffers (1.0f) instead of threading the original
    inputs through, producing silent wrong results (GAP 0).

    By routing ``aten.matmul`` directly to ``aten.bmm`` for matching
    3D inputs, we skip the problematic decomposition entirely.
    """
    import torch
    from torch._inductor import lowering as L
    from torch._inductor.lowering import register_lowering

    aten = torch.ops.aten

    _orig_matmul = L.lowerings.get(aten.matmul.default)

    @register_lowering(aten.matmul, type_promotion_kind=None)
    def _vulkan_matmul(tensor1, tensor2, *, layout=None):
        t1_device = tensor1.get_device()
        if t1_device.type != "vulkan":
            if _orig_matmul is not None:
                return _orig_matmul(tensor1, tensor2, layout=layout)
            return NotImplemented

        t1_sizes = list(tensor1.get_size())
        t2_sizes = list(tensor2.get_size())

        def _as_ints(sizes):
            try:
                return [int(s) for s in sizes]
            except Exception:
                return None

        s1 = _as_ints(t1_sizes)
        s2 = _as_ints(t2_sizes)

        if s1 is not None and s2 is not None and len(s1) >= 2 and len(s2) >= 2:
            if s1[-1] == s2[-2]:
                if len(s1) == 3 and len(s2) == 3:
                    return L.lowerings[aten.bmm.default](
                        tensor1, tensor2, layout=layout
                    )
                batch1 = 1
                for d in s1[:-2]:
                    batch1 *= d
                batch2 = 1
                for d in s2[:-2]:
                    batch2 *= d
                batch = max(batch1, batch2)
                t1_3d = L.lowerings[aten.reshape.default](
                    tensor1, [batch, s1[-2], s1[-1]]
                )
                t2_3d = L.lowerings[aten.reshape.default](
                    tensor2, [batch, s2[-2], s2[-1]]
                )
                result_3d = L.lowerings[aten.bmm.default](t1_3d, t2_3d, layout=layout)
                out_batch = list(t1_sizes[:-2])
                out_m = s1[-2]
                out_n = s2[-1]
                rs = out_batch + [out_m, out_n]
                return L.lowerings[aten.reshape.default](result_3d, rs)

        if _orig_matmul is not None:
            return _orig_matmul(tensor1, tensor2, layout=layout)
        return NotImplemented


def _register_matmul_backward() -> None:
    """T4.2 — Audit and document matmul backward routing via BWD_TEMPLATE_REGISTRY.

    ``aten.mm`` / ``aten.bmm`` / ``aten.addmm`` backward is decomposed by AOT
    Autograd during tracing into ``aten.mm`` calls with transposed operands
    (dA = dC @ B^T, dB = A^T @ dC).  Our forward template lowerings
    (``install_external_mm`` / ``install_external_bmm``) already handle these
    decomposed mm/bmm calls, so matmul backward routes through the template
    path by construction.

    This function audits the BWD_TEMPLATE_REGISTRY entries for matmul ops
    to ensure every forward template has a documented backward partner.
    At the exit gate, every ``aten.mm_backward`` / ``aten.bmm_backward`` /
    ``aten.addmm_backward`` call in the compiled graph decomposes into
    forward template dispatches — not extern ATEN dispatches.
    """
    import logging

    _log = logging.getLogger(__name__)

    from torch_vulkan.inductor.bwd_diff_dispatch import resolve_backward_kind

    matmul_ops = [
        "aten.mm.default",
        "aten.bmm.default",
        "aten.addmm.default",
    ]
    for op_name in matmul_ops:
        resolved = resolve_backward_kind(op_name)
        if resolved is None:
            _log.warning(
                "T4.2: No BWD_TEMPLATE_REGISTRY entry for %s; "
                "backward will fall through to default path.",
                op_name,
            )
        elif not resolved.is_template_jinja:
            _log.warning(
                "T4.2: BWD_TEMPLATE_REGISTRY entry for %s has kind=%s; "
                "expected TEMPLATE_JINJA.",
                op_name,
                resolved.kind,
            )
        else:
            _log.debug(
                "T4.2: %s backward routing confirmed → %s (template jinja)",
                op_name,
                resolved.fwd_key,
            )


def _register_mm_int8_lowering() -> None:
    """OP.24 — Register int8 matmul external callables for autotuning.

    Installs Slang int8 tiled-matmul callables into
    ``torch._inductor.config.external_matmul`` so that Inductor's
    ``tuned_mm`` lowering can benchmark them alongside the CPU fallback
    for int8×int8→float32 gemm.

    The ``_vulkan_mm`` lowering (registered by ``_register_mm_lowering``)
    detects int8 inputs and falls through to ``tuned_mm``, which picks
    the best available choice from ``external_matmul``.

    Idempotent — safe to call multiple times.
    """
    from ..templates.caller.gemm.install import install_external_mm_int8

    install_external_mm_int8()
