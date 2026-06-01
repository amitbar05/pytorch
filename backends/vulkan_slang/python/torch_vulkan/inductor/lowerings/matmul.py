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
    #
    # M17.1: For fp32 Vulkan tensors, ``codegen`` emits a direct call to
    # ``_slang_tile_mm`` instead of ``aten.mm.out``, routing through the
    # Slang tiled matmul and avoiding eager C++ ``vulkan_mm`` sub-dispatches.
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

        def __init__(
            self,
            layout,
            inputs,
            *,
            python_kernel_name=None,
            op_overload=None,
            tile_m=None,
            tile_n=None,
            tile_k=None,
            num_stages=None,
            m_per_thread=None,
            n_per_thread=None,
        ):
            super().__init__(
                layout=layout,
                inputs=inputs,
                python_kernel_name=python_kernel_name,
                op_overload=op_overload,
            )
            self._tile_m = tile_m
            self._tile_n = tile_n
            self._tile_k = tile_k
            self._num_stages = num_stages
            self._m_per_thread = m_per_thread
            self._n_per_thread = n_per_thread

        def codegen(self, wrapper):
            """Emit a call to ``_slang_tile_mm`` for the Slang tile path,
            or delegate to standard ``ExternKernelOut.codegen`` for the
            ``aten.mm.out`` fallback path.
            """
            if self._tile_m is not None:
                # M17.1: Slang tiled matmul path.
                wrapper.add_import_once(
                    "from torch_vulkan.inductor.vulkan_template_caller "
                    "import _slang_tile_mm"
                )
                input_names = [inp.codegen_reference() for inp in self.inputs]
                out_name = self.codegen_reference()
                self.codegen_comment(wrapper)
                wrapper.writeline(
                    f"_slang_tile_mm("
                    f"{self._tile_m}, {self._tile_n}, {self._tile_k}, "
                    f"{self._num_stages}, "
                    f"{input_names[0]}, {input_names[1]}, {out_name}, "
                    f"m_per_thread={self._m_per_thread}, "
                    f"n_per_thread={self._n_per_thread})"
                )
                self.codegen_size_asserts(wrapper)
            else:
                super().codegen(wrapper)

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
        #
        # Pre-realize any unrealized inputs (Pointwise / Reduction) before
        # constructing _VulkanMMOut.  This mirrors what Inductor's
        # mm_args() / realize_inputs() does for the standard tuned_mm path.
        # Without this, an unrealized Pointwise that feeds into mm (e.g. a
        # relu activation just before a Linear layer, lowered via
        # aten.matmul → view → aten.mm) would reach unwrap_storage as
        # TensorBox(StorageBox(Pointwise)) and fail the Buffer assertion.
        # The Vulkan-specific BaseView → ReinterpretView optimisation in
        # unwrap_storage_for_input only applies to *realized* view nodes;
        # unrealized computation nodes must be materialized at the lowering
        # site using realize_input so the scheduler can order them correctly.
        def _needs_realize(t):
            """Return True if t wraps an unrealized computation node."""
            inner = t
            if isinstance(inner, ir.TensorBox):
                inner = inner.data
            if isinstance(inner, ir.StorageBox):
                return not isinstance(inner.data, ir.Buffer)
            return False

        if _needs_realize(tensor1):
            tensor1 = ir.ExternKernel.realize_input(tensor1)
        if _needs_realize(tensor2):
            tensor2 = ir.ExternKernel.realize_input(tensor2)

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

        # M17.1: For fp32/fp16 Vulkan tensors, route through Slang tiled matmul
        # instead of eager C++ vulkan_mm.  The (8,8,8) 1-output-per-thread
        # tile is the most universal config — it fits a single wave on both
        # wave32 and wave64 hardware.
        # FP16.2 (2026-06-01): fp16 enabled — slang_mm.slang renders half
        # correctly (f16 storage, f32 accumulation).
        if t1_dtype in (torch.float32, torch.float16):
            kernel = _VulkanMMOut(
                layout=out_layout,
                inputs=[tensor1, tensor2],
                python_kernel_name="torch_vulkan.inductor.templates.caller.gemm.dispatch._slang_tile_mm",
                op_overload=None,
                tile_m=8,
                tile_n=8,
                tile_k=8,
                num_stages=1,
                m_per_thread=1,
                n_per_thread=1,
            )
            return ir.TensorBox.create(kernel)

        # Non-fp32/fp16 Vulkan tensors (e.g. bf16): fall through to aten.mm.out
        # path which dispatches to eager C++ vulkan_mm.
        kernel = _VulkanMMOut(
            layout=out_layout,
            inputs=[tensor1, tensor2],
            python_kernel_name="torch.ops.aten.mm.out",
            op_overload=aten.mm.out,
        )
        return ir.TensorBox.create(kernel)


def _register_linear_backward_decomposition() -> None:
    """M19.1 — decompose ``aten.linear_backward.default`` into mm + sum.

    M17.1-gap context (roadmap § 0.9): the C++ eager
    ``vulkan_linear_backward`` (csrc/ops/backward_ops.cpp) internally
    issues 8 sub-dispatches per Linear backward step (two reshape +
    contiguous on each of self/grad_output, plus mm / mm_ex /
    sum_dim). At the FX level this appears as a single
    ``aten.linear_backward`` extern node — opaque to Inductor's
    scheduler and unfusable.

    By installing a pure-aten decomposition into BOTH the AOT
    decomposition table (consulted during joint-graph tracing) and the
    Inductor decomposition table (consulted during the post-graph
    decomp pass), we trade the opaque extern for fully lowered
    primitives (``aten.mm`` × 2 + ``aten.sum.dim_IntList``). Each of
    those routes through our Slang tile / reduction kernels — fewer
    dispatches and fusable with surrounding pointwise ops.

    Decomposition mirrors ``torch._refs._refs.linear_backward``:
      grad_input  = grad_output @ weight                     # (B, in)
      grad_weight = grad_output.transpose(-2,-1) @ self_input # (out, in)
      grad_bias   = grad_output.flatten(end_dim=-2).sum(0)    # (out,)
    Handles 3D+ inputs by flattening leading dims into the batch axis.

    Historical regression risk (2026-05-17 M17.8.d, re-confirmed
    2026-05-18 M19.1): the decomposition lowers
    ``grad_weight = g2d.t() @ s2d`` via ``torch.mm`` which Inductor
    emits as ``aten.mm.default``. Upstream's ``tuned_mm`` (overload-
    specific) picks the ``aten_mm`` extern path, which delegates to
    the C++ ``vulkan_mm_out`` with a permuted-stride
    ``reinterpret_tensor`` view as the LHS. ``vulkan_mm_out`` detects
    the view via ``is_t_transposed`` and dispatches
    ``matmul_mm_tiled_fwd`` with ``transpose_a=true`` — a path that
    on RDNA1 wave64 returns only the first row of the expected
    output (rows 1..M-1 come back as zeros), giving a clear visual
    signature: ``vk_grad_weight[0, :] != 0`` but
    ``vk_grad_weight[1:, :] == 0``.

    Because that bug lives in the C++ ``vulkan_mm_dispatch`` /
    ``matmul_mm_tiled_fwd`` Slang shader (Group A / G files,
    out of this milestone's lane), M19.1 ships the decomposition
    function definition + dual-decomp-table installer but **keeps
    the call site (``__init__.py:_register_linear_backward_decomposition()``)
    commented out** until the mm tile bug is fixed. Tests for the
    parity gain are landed with ``xfail(strict=True,
    reason="M19.1 — gated on mm tile transpose-a fix")``. When the
    underlying mm path is corrected, the call site flips and the
    xfails strict-fail (visible signal to ratchet the gate).

    # 2026-05-18: defined + dual-decomp-table installer landed;
    # call site held until mm tile transpose-a bug fixed in csrc/
    # — see M19.1 in docs/10-inductor-backend.md § 0.6.2.
    """
    import torch
    from torch._decomp import decomposition_table as _aot_decomps
    from torch._inductor.decomposition import (
        decompositions as _ind_decomps,
        fast_random_decomps,
    )

    aten = torch.ops.aten

    def _linear_backward_decomp(self_input, grad_output, weight, output_mask):
        # Flatten any leading dims into a single batch axis so the mm/sum
        # are 2D and 1D respectively (matches the C++ kernel's reshape).
        # No ``.contiguous()`` on the reshape outputs — the Slang tile path
        # materializes non-contiguous inputs at dispatch time
        # (``_TRUST_INDUCTOR=False``), so the reshapes can stay as
        # zero-copy ``ReinterpretView`` nodes.
        out_features = weight.shape[0]
        in_features = weight.shape[1]
        g2d = grad_output.reshape(-1, out_features)
        s2d = self_input.reshape(-1, in_features)

        grad_input = grad_weight = grad_bias = None
        if output_mask[0]:
            gi_2d = torch.mm(g2d, weight)
            grad_input = gi_2d.reshape(self_input.shape)
        if output_mask[1]:
            # grad_weight = g2d.T @ s2d  → shape (out, in)
            # The transposed LHS reaches the mm lowering as an
            # Inductor ``ReinterpretView`` (Inductor's optimizer elides
            # the explicit ``.contiguous()`` after ``.t()``). That's
            # safe HERE because our ``aten.mm.default`` lowering
            # (``_vulkan_mm`` via the M19.1 dual-registration) routes
            # through the Slang tile path, which forces materialization
            # at the dispatch site (``_TRUST_INDUCTOR=False``). The C++
            # eager ``vulkan_mm_out`` ``is_t_transposed`` fast-path —
            # which produces zero on a permuted view — is bypassed.
            grad_weight = torch.mm(g2d.transpose(0, 1).contiguous(), s2d)
        if output_mask[2]:
            grad_bias = g2d.sum(dim=0)
        return grad_input, grad_weight, grad_bias

    # Install in BOTH decomp tables — AOTAutograd's table is consulted
    # during joint-graph tracing, Inductor's local table is consulted
    # during the post-graph decomp pass.  Belt-and-suspenders so the
    # decomposition definitely fires before the eager kernel is reached.
    _aot_decomps[aten.linear_backward.default] = _linear_backward_decomp
    _ind_decomps[aten.linear_backward.default] = _linear_backward_decomp
    # Clear the cached select_decomp_table result so the new entry takes
    # effect on the next compile (same pattern as OP.23).
    fast_random_decomps.cache_clear()


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
    from torch._inductor import ir
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
        ndim1 = len(t1_sizes)
        ndim2 = len(t2_sizes)

        def _as_ints(sizes):
            try:
                return [int(s) for s in sizes]
            except Exception:
                return None

        s1 = _as_ints(t1_sizes)
        s2 = _as_ints(t2_sizes)

        # ── M18.5: 3D+ × 2D fold ─────────────────────────────────────
        # MHA / Transformer / Linear-on-sequence emit ``matmul(x, w)``
        # where ``x`` is ``[..., K]`` and ``w`` is ``[K, N]``. The
        # leading dims may be symbolic (dynamic batch / seq). Fold the
        # leading dims of ``x`` into a single ``M`` row axis, route
        # through ``aten.mm``, and reshape the result back.
        #
        # This mirrors the ``should_fold`` branch of the upstream
        # ``aten.matmul`` decomposition
        # (``torch/_decomp/decompositions.py::matmul``) — by intercepting
        # here we skip the decomposition entirely and keep the symbolic
        # leading dims as-is, so dynamic-shape compiles don't bail when
        # ``_as_ints()`` returns ``None``.
        if ndim1 >= 2 and ndim2 == 2:
            # Reshape lhs ``[..., K]`` → ``[prod(leading), K]``. The
            # ``view`` lowering accepts mixed concrete/sympy sizes and
            # returns a TensorBox without materializing.
            view_lowering = L.lowerings[aten.view.default]
            K = t1_sizes[-1]
            N = t2_sizes[-1]
            leading = t1_sizes[:-1]
            from torch._inductor.utils import sympy_product

            # Pre-realize any unrealized computation (Pointwise / Reduction)
            # in tensor1 before creating the view.  The view lowering may
            # return the original TensorBox unchanged for same-shape views,
            # so realizing here ensures the mm lowering gets a realized buffer
            # instead of a raw Pointwise.  This is the canonical Inductor
            # pattern — ``mm_args()`` calls ``realize_inputs()`` for the same
            # reason.  Topological ordering is preserved because realize_input
            # fires before the mm node is registered in the graph.
            inner1 = tensor1
            if isinstance(inner1, ir.TensorBox):
                inner1 = inner1.data
            if isinstance(inner1, ir.StorageBox) and not isinstance(
                inner1.data, ir.Buffer
            ):
                ir.ExternKernel.realize_input(tensor1)

            M_flat = sympy_product(leading)
            t1_2d = view_lowering(tensor1, [M_flat, K])
            # ``aten.mm`` is our packet-level lowering (``_vulkan_mm``)
            # so this routes through the Slang tile / fp16 / int8 paths.
            mm_2d = L.lowerings[aten.mm](t1_2d, tensor2)
            out_shape = list(leading) + [N]
            return view_lowering(mm_2d, out_shape)

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
