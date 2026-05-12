"""Vulkan-specific Inductor wrapper codegen.

Subclasses `PythonWrapperCodegen` to emit Python kernel-call lines for the
vulkan device (which upstream's base class marks as "nyi"). For each kernel
we emit a plain function call — the runtime module takes the generated-kernel
function and args and dispatches the Slang-compiled SPIR-V to our Vulkan
runtime.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch._inductor.compile_fx
import torch._inductor.config
from torch._inductor import ir
from torch._inductor.codegen.wrapper import (
    PythonWrapperCodegen,
    SubgraphPythonWrapperCodegen,
)
from torch._inductor.virtualized import V


def _install_vulkan_skip_alignment_clone() -> None:
    """OP.5 fix — exclude Vulkan tensors from Inductor's alignment check.

    ``compile_fx.get_input_idxs_to_check`` flags every GPU input whose
    ``data_ptr() % 16 != 0`` for runtime cloning via
    ``align_inputs_from_check_idxs`` → ``copy_misaligned_inputs`` →
    ``clone_preserve_strides``. The clone path on a Vulkan tensor goes
    through PrivateUse1's ``aten::clone``, which (per the documented
    ``dispatch_copy_buffer`` 4B/elem bug) silently truncates int64 /
    float64 storage — turning ``[0, 1, 2, 3]`` into ``[0, 1, 0, 0]``
    on every int64 index buffer that flows through compiled graphs.

    Symptom: ``aten.index_put`` / ``embedding_dense_backward`` /
    ``embedding_bag_forward`` produce numerically wrong outputs even
    though the generated Slang kernel and binding emission are correct.
    Root cause is upstream Inductor speculatively cloning misaligned
    GPU inputs; Vulkan tensors carry synthetic data pointers (1, 2, 3,
    …) that *always* fail the ``% 16`` test, so every int64 index
    buffer triggers the truncating-clone bug.

    Fix: monkey-patch the alignment check to skip Vulkan tensors.
    Vulkan kernels read the underlying ``VkBuffer`` via descriptor sets
    that are device-aligned by construction; the host-side ``data_ptr``
    is a synthetic key into our buffer table, not a memory address, so
    "alignment" is meaningless for our backend. Idempotent.
    """
    _orig = torch._inductor.compile_fx.get_input_idxs_to_check
    if getattr(_orig, "_vulkan_skip_alignment_patched", False):
        return

    def _vulkan_aware_get_input_idxs_to_check(inputs, static_input_idxs):
        ids = list(_orig(inputs, static_input_idxs))
        return [
            i
            for i in ids
            if not (
                i < len(inputs)
                and isinstance(inputs[i], torch.Tensor)
                and inputs[i].device.type == "vulkan"
            )
        ]

    _vulkan_aware_get_input_idxs_to_check._vulkan_skip_alignment_patched = True  # type: ignore[attr-defined]
    torch._inductor.compile_fx.get_input_idxs_to_check = (
        _vulkan_aware_get_input_idxs_to_check
    )


_install_vulkan_skip_alignment_clone()


def _install_vulkan_pattern_matcher_tensor_attr_fix() -> None:
    """TR.14 fix — gracefully skip pattern_matcher replacements whose
    subgraph contains a Vulkan tensor with no backing buffer.

    Upstream's ``ReplacementPatternEntry.replace_with_graph`` builds a
    ``Replacer`` (an ``fx.Interpreter``) that walks the replacement
    subgraph and copies its nodes into the main graph. The
    ``get_attr`` branch of ``Replacer.run_node`` (in
    ``torch/_inductor/pattern_matcher.py``) only handles
    ``GraphModule`` attributes (HOP subgraphs); for plain tensor
    constants it raises::

        NotImplementedError(
            f"NYI: replacement_graph.{target} is not a graph module. Got {sub_gm}."
        )

    The f-string materializes the tensor via ``Tensor.__repr__`` →
    ``Tensor.tolist()``. On Vulkan, this crashes with
    ``RuntimeError: Vulkan tensor has no backing buffer`` whenever the
    replacement traced an op (e.g. ``torch.arange`` inside
    ``scatter_upon_const_tensor``) that ``make_fx`` lifted as
    ``self._tensor_constantN`` — those tensors carry FakeTensor
    metadata but no allocated ``VkBuffer`` (synthetic ``data_ptr=0``).

    Fix: monkey-patch ``ReplacementPatternEntry.replace_with_graph``
    so that when the replacement subgraph contains a ``get_attr`` to
    a Vulkan tensor with ``data_ptr=0``, we **skip the replacement**
    entirely and leave the original (un-replaced) FX nodes in the
    main graph. This unblocks compile (no crash) and falls back to
    the canonical lowering for the original op — which is the
    correct path on Vulkan since we can't faithfully reconstruct the
    constant's values from a missing buffer.

    The narrow contract: TR.14's ``RuntimeError: Vulkan tensor has
    no backing buffer`` no longer fires inside the pattern matcher.
    Upstream patterns that *can* produce a valid replacement (no
    Vulkan tensor constants, or tensor constants with real backing)
    still apply normally.

    Idempotent.
    """
    from torch._inductor import pattern_matcher as _pm

    cls = _pm.ReplacementPatternEntry
    if getattr(cls, "_vulkan_tensor_attr_patched", False):
        return

    _orig = cls.replace_with_graph

    def _replacement_has_unbacked_vulkan_tensor(replacement_graph) -> bool:
        """Return True iff the replacement subgraph contains a
        ``get_attr`` whose target resolves to a Vulkan tensor with
        no backing buffer (``data_ptr=0``).
        """
        if hasattr(replacement_graph, "graph"):
            rmod = replacement_graph
            rgraph = replacement_graph.graph
        else:
            rmod = getattr(replacement_graph, "owning_module", None)
            rgraph = replacement_graph
        if rmod is None:
            return False
        for n in rgraph.nodes:
            if n.op != "get_attr":
                continue
            val = getattr(rmod, n.target, None)
            if not isinstance(val, torch.Tensor):
                continue
            if isinstance(val, torch.fx.GraphModule):
                continue
            if val.device.type == "vulkan" and val.data_ptr() == 0:
                return True
        return False

    @staticmethod  # type: ignore[misc]
    def _patched_replace_with_graph(match, graph, replacement_graph, args):
        if _replacement_has_unbacked_vulkan_tensor(replacement_graph):
            # Skip this replacement — leave the original matched
            # nodes in the graph. The downstream lowering for the
            # original op is correct on Vulkan; the replacement
            # would have crashed at format-time and (even if forced
            # through) couldn't faithfully reproduce the lifted
            # constant's values.
            return None
        return _orig(match, graph, replacement_graph, args)

    cls.replace_with_graph = _patched_replace_with_graph
    cls._vulkan_tensor_attr_patched = True  # type: ignore[attr-defined]


_install_vulkan_pattern_matcher_tensor_attr_fix()


def _normalize_strides_to_row_major(
    shape: list[int] | tuple[int, ...],
    stride: list[int] | tuple[int, ...],
) -> list[int]:
    """Normalize strides to standard row-major (C-contiguous) order.

    vulkan_empty_strided always creates row-major contiguous buffers
    regardless of the requested strides. Inductor's kernel codegen uses
    the strides returned here to compute load/store memory offsets, so
    we must ensure they match the actual row-major buffer layout.

    For a tensor of shape (d0, d1, ..., dn), row-major strides are:
        stride[n-1] = 1
        stride[i] = product(shape[i+1:]) for i from n-2 down to 0
    """
    ndim = len(shape)
    if ndim == 0:
        return list(stride)
    row_major = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        row_major[i] = row_major[i + 1] * shape[i + 1]
    return row_major


def _trust_inductor() -> bool:
    # P5.11.a.1: default ON. The asserts are debug-only and the wrapper
    # codegen is mature enough that the per-buffer `assert_size_stride`
    # overhead is no longer a load-bearing safety net (CPU oracle covers
    # correctness; the elision saves ~150 µs/step on small workloads).
    # Opt-OUT preserved via `TORCH_VULKAN_TRUST_INDUCTOR=0` for engineers
    # debugging an Inductor codegen bug.
    return os.environ.get("TORCH_VULKAN_TRUST_INDUCTOR", "1") != "0"


# GPU.1 / GPU.2 / GPU.3 — Config-reading helpers cached at module load.
# These are called during codegen (not runtime dispatch) so the env var
# is captured once at graph-compile time.


def _batch_dispatch_enabled() -> bool:
    """GPU.1: batch dispatch submission."""
    return os.environ.get("TORCH_VULKAN_BATCH_DISPATCH", "1") != "0"


def _wrapper_fastpath_enabled() -> bool:
    """GPU.2: Python wrapper hot-path optimizations."""
    return os.environ.get("TORCH_VULKAN_WRAPPER_FASTPATH", "1") != "0"


def _profile_dispatches_enabled() -> bool:
    """GPU.3: per-dispatch profiling."""
    return os.environ.get("TORCH_VULKAN_PROFILE_DISPATCHES", "0") == "1"


# PF.41: lifetime_class names valid for the buffer pool. Mirrors
# `torch_vulkan.inductor.buffer_pool.LIFETIME_CLASSES` and
# `torch_vulkan.inductor.lifetime.LIFETIME_CLASSES`. Hard-coded here so
# the wrapper-codegen module stays import-free of the pool/FX subsystems.
_VALID_LIFETIME_CLASSES = frozenset(
    {"parameter", "gradient", "save_for_backward", "transient", "output", "scratch"}
)


def _lifetime_class_for_name(name: str) -> str:
    """Resolve the lifetime_class string for a buffer by name.

    PF.40 annotates joint-graph FX nodes with
    ``node.meta["lifetime_class"]``; the partitioner copies node meta
    into the per-side fw/bw modules, and Inductor's IR Buffer carries
    a back-pointer to its origin FX node via :meth:`get_origin_node`.
    Returns ``"transient"`` (the safest default for intra-step reuse)
    when the lookup misses for any reason — buffer not found, no
    origin node, no annotation, or an unrecognized class name.
    """
    try:
        buf = V.graph.try_get_buffer(name)
    except Exception:
        return "transient"
    if buf is None:
        return "transient"
    try:
        node = buf.get_origin_node()
    except Exception:
        return "transient"
    if node is None:
        return "transient"
    cls = node.meta.get("lifetime_class")
    if cls in _VALID_LIFETIME_CLASSES:
        return cls
    return "transient"


class VulkanPythonWrapperCodegen(PythonWrapperCodegen):
    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: "Optional[ir.GraphPartitionSignature]" = None,
    ):
        if is_subgraph:
            assert subgraph_name is not None
            assert parent_wrapper is not None
            return SubgraphPythonWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )
        return VulkanPythonWrapperCodegen()

    def write_header(self):
        super().write_header()
        self.header.splice(
            "from torch_vulkan import _empty_strided_vulkan as empty_strided_vulkan\n"
            "from torch_vulkan.inductor.buffer_pool import vulkan_pool_release\n"
            "from torch_vulkan.inductor.runtime import make_vulkan_kernel as _vk_make_kernel\n"
        )
        # GPU.1: Batch dispatch import when enabled.
        if _batch_dispatch_enabled():
            self.header.splice(
                "from torch_vulkan.inductor.runtime import DispatchBatcher\n"
            )
        # GPU.2/GPU.3: Fast-path profiling hooks.
        if _wrapper_fastpath_enabled() or _profile_dispatches_enabled():
            self.header.splice(
                "from torch_vulkan.inductor.runtime import _record_dispatch_time\n"
            )
        if _profile_dispatches_enabled():
            self.header.splice("from time import perf_counter_ns\n")
        # P5.11.a.1 (default ON; was P0.4 opt-IN): no-op the per-buffer
        # `assert_size_stride` / `assert_alignment` calls Inductor scatters
        # through the wrapper. The asserts are debug-only and add measurable
        # overhead per step on workloads with many small allocations (MLP
        # forward: 8 buffers/step, 452→302 µs/step). Defines no-op shims so
        # any residual call sites (e.g. from cached graphs) become harmless;
        # `codegen_deferred_input_asserts` skips emitting new ones.
        # Opt-OUT via `TORCH_VULKAN_TRUST_INDUCTOR=0`.
        if _trust_inductor():
            self.header.splice(
                "def assert_size_stride(*args, **kwargs):\n"
                "    return None\n"
                "def assert_alignment(*args, **kwargs):\n"
                "    return None\n"
            )

    def write_prefix(self) -> None:
        super().write_prefix()
        # GPU.1: Initialize the DispatchBatcher at function scope.
        if _batch_dispatch_enabled():
            self.writeline("_batcher = DispatchBatcher()")
            self.writeline("_batcher.__enter__()")

    def generate_end_graph(self):
        # GPU.1: Exit the DispatchBatcher before graph end — flushes all.
        if _batch_dispatch_enabled():
            self.writeline("_batcher.__exit__(None, None, None)")
        super().generate_end_graph()

    def generate_return(self, output_refs):
        # GPU.1: Flush batcher before return so kernels execute.
        if _batch_dispatch_enabled():
            self.wrapper_call.writeline("_batcher.__exit__(None, None, None)")
        super().generate_return(output_refs)

    def codegen_input_size_asserts(self) -> None:
        # P5.11.a.1: under default-ON, skip collecting per-input asserts
        # entirely so `codegen_deferred_input_asserts` has nothing to emit.
        if _trust_inductor():
            return
        super().codegen_input_size_asserts()

    def codegen_deferred_input_asserts(self, input_names) -> None:
        # P5.11.a.1: under default-ON, skip emitting per-input
        # `assert_size_stride` lines entirely (not just no-op the calls).
        # The structural contract test asserts zero call sites in the
        # emitted wrapper source; eliding here is what makes that lock hold.
        if _trust_inductor():
            return
        super().codegen_deferred_input_asserts(input_names)

    def generate_extern_kernel_alloc(self, node: ir.ExternKernelAlloc) -> None:
        """PF.33: Route extern-kernel (FallbackKernel) output allocations
        through the buffer pool instead of letting the C++ dispatcher
        allocate directly via ``VulkanAllocator``.

        Default upstream codegen emits::

            buf_N = torch.ops.aten.mm.default(arg0, arg1)

        which calls ``vulkan_empty`` / ``vulkan_empty_strided`` inside
        the C++ dispatcher, bypassing ``empty_strided_vulkan`` and the
        Python buffer pool entirely.  For large matmul / convolution
        outputs this is the dominant source of unpooled allocations in
        compiled training graphs.

        This override emits::

            buf_N = empty_strided_vulkan(size, stride, dtype, lifetime_class=...)
            torch.ops.aten.mm.out(arg0, arg1, out=buf_N)

        The pre-allocation hits the pool's ``vulkan_pool_acquire``;
        the ``out=`` variant writes into the caller-owned buffer instead
        of allocating a fresh one.  When the op has no ``out=`` variant
        the override falls back to upstream's default-allocation path.
        """
        if node.get_device() is None or node.get_device().type != "vulkan":
            return super().generate_extern_kernel_alloc(node)

        # NoneLayout (no return) and non-Layout outputs (MultiOutput,
        # mutation) must go through the default path — we can only
        # pre-allocate for single-tensor Layout outputs.
        layout = node.layout
        if not isinstance(layout, ir.Layout):
            return super().generate_extern_kernel_alloc(node)

        name = node.get_name()
        size = layout.size
        stride = layout.stride
        dtype = layout.dtype
        lifetime_class = _lifetime_class_for_name(name)

        # Emit pool-aware pre-allocation.
        size_tuple = self.codegen_python_shape_tuple(size)
        stride_tuple = self.codegen_python_shape_tuple(stride)
        self.writeline(
            f"{name} = empty_strided_vulkan("
            f"{size_tuple}, {stride_tuple}, {dtype}, "
            f"lifetime_class={lifetime_class!r})"
        )

        # Try to emit the ``out=`` variant so the op writes into the
        # pre-allocated buffer instead of allocating internally.
        python_kernel_name = node.python_kernel_name
        if python_kernel_name:
            # CP.9 / TRAIN.12: aten.rand / aten.randn .out variants
            # have a different signature than .default (they accept
            # only ``(size, *, out)``, not dtype/layout/device kwargs).
            # Emit the .default call directly; the PrivateUse1 dispatch
            # allocates internally. The buffer still gets pool-released
            # at end-of-life via make_buffer_free.
            if python_kernel_name in ("aten.rand.default", "aten.randn.default"):
                super().generate_extern_kernel_alloc(node)
                return

            # T4.12: aten.convolution* does not have a compatible
            # .out variant in the Vulkan dispatch path. Fall through to
            # the default-allocation path.
            if python_kernel_name.startswith("aten.convolution"):
                super().generate_extern_kernel_alloc(node)
                return
            if ".default" in python_kernel_name:
                # Standard aten op: replace .default with .out.
                out_kernel_name = python_kernel_name.replace(".default", ".out")
                args = [*node.codegen_args(), *node.codegen_kwargs(skip_out=True)]
                args.append(f"out={name}")
                node.codegen_comment(self)
                self.writeline(f"{out_kernel_name}({', '.join(args)})")
                if isinstance(node.layout, ir.Layout):
                    node.codegen_size_asserts(self)
                return
            else:
                # T6.3: Template-based extern kernels (mm, addmm, bmm,
                # flash attention, Philox, etc.) accept ``out=`` as a
                # keyword argument.  Pre-allocate via the pool and pass
                # the buffer so the template caller skips its internal
                # ``torch.empty`` fallback.
                args = [*node.codegen_args(), *node.codegen_kwargs(skip_out=True)]
                args.append(f"out={name}")
                node.codegen_comment(self)
                self.writeline(f"{python_kernel_name}({', '.join(args)})")
                if isinstance(node.layout, ir.Layout):
                    node.codegen_size_asserts(self)
                return

        # No python_kernel_name — fall back to upstream's default.
        # The internal allocation will bypass the pool for this buffer,
        # but the wrapper's ``make_buffer_free`` will still pool-release
        # it at end-of-life (recycling for the *next* step).
        super().generate_extern_kernel_alloc(node)

    def generate_extern_kernel_out(self, node: ir.ExternKernelOut) -> None:
        if node.get_device() is None or node.get_device().type != "vulkan":
            return super().generate_extern_kernel_out(node)
        super().generate_extern_kernel_out(node)

    def make_allocation(
        self, name, device, dtype, shape, stride, allocation_shape=None, is_pinned=False
    ):
        # Meta-device allocations leak in from AOT autograd's backward graph
        # when an IR node was originally vulkan but FakeTensor's view fast-path
        # dropped the device. Under our wrapper the only valid downstream
        # consumer is the vulkan runtime, so route them to empty_strided_vulkan.
        if (
            device is not None
            and device.type in ("vulkan", "meta")
            and not torch._inductor.config.test_configs.track_memory_lifecycle
        ):
            if allocation_shape is None:
                allocation_shape = shape

            codegen_shape_tuple = self.codegen_python_shape_tuple(shape)
            codegen_alloc_tuple = self.codegen_python_shape_tuple(allocation_shape)
            codegen_stride_tuple = self.codegen_python_shape_tuple(stride)
            # PF.41: thread the buffer's lifetime_class (from PF.40's FX
            # annotation) into the pool-acquire call so transient
            # intra-step reuse never collides with a still-live
            # save_for_backward buffer. Defaults to "transient" when the
            # IR buffer has no annotated origin (legacy graphs, meta-
            # device fallback path, allocations without an FX preimage).
            lifetime_class = _lifetime_class_for_name(name)
            out = (
                f"{name} = empty_strided_vulkan("
                f"{codegen_alloc_tuple}, "
                f"{codegen_stride_tuple}, "
                f"{dtype}, "
                f"lifetime_class={lifetime_class!r})"
            )
            if codegen_shape_tuple != codegen_alloc_tuple:
                out = (
                    out + f".as_strided({codegen_shape_tuple}, {codegen_stride_tuple})"
                )
            return out
        return super().make_allocation(
            name,
            device,
            dtype,
            shape,
            stride,
            allocation_shape=allocation_shape,
            is_pinned=is_pinned,
        )

    def make_buffer_free(self, buffer):
        # P0.4: route end-of-life buffers to the recycle pool instead of
        # `del`. The next step's empty_strided_vulkan call hits the pool
        # and skips the dispatcher round-trip (~17 us/call).
        # Skip vulkan-only routing for non-vulkan buffers (e.g. torchbind
        # objects mixed into a multi-device graph). Skip graph outputs:
        # the caller still holds the returned tensor; pool-releasing it
        # would re-vend a still-live buffer to the next step's allocation
        # and corrupt the prior return value.
        try:
            device = buffer.get_device()
        except (AttributeError, NotImplementedError):
            device = None
        if device is None or device.type != "vulkan":
            return super().make_buffer_free(buffer)
        # PF.54: multi-output extern kernels (e.g. ``aten.linear_backward``,
        # ``aten.native_layer_norm``) bind their result variable to a Python
        # tuple — children (``MultiOutput``) are unpacked via ``buf_N =
        # buf_M[i]`` and have their own pool-release lifecycle.
        # ``vulkan_pool_release`` keys on ``tensor.size()``/``stride()``,
        # which raises ``AttributeError`` on a tuple holder. Fall through to
        # upstream's plain ``del`` for any non-``Layout`` output spec.
        if not isinstance(buffer.get_output_spec(), ir.Layout):
            return super().make_buffer_free(buffer)
        name = buffer.get_name()
        # Inputs are owned by the caller; outputs are returned to the caller.
        # Pool-releasing either would re-vend a still-live tensor on the next
        # step and corrupt the caller's data.
        if name in V.graph.graph_inputs or name in V.graph.get_output_names():
            return super().make_buffer_free(buffer)
        # PF.41: keep the release-side lifetime_class in lockstep with
        # the acquire-side. Mismatched classes would land the released
        # tensor in one bucket while the next acquire looks in another,
        # silently degrading to perpetual misses (looks like a hit-rate
        # regression, not a correctness bug — exactly the kind of
        # silent failure floor-gate-then-ratchet is meant to surface).
        lifetime_class = _lifetime_class_for_name(name)
        return (
            f"vulkan_pool_release({name}, lifetime_class={lifetime_class!r}); "
            f"{name} = None"
        )

    def codegen_reinterpret_view(
        self,
        data,
        size,
        stride,
        offset,
        writeline,
        dtype=None,
    ) -> str:
        """T.12.B fix — materialize Vulkan graph-output reinterpret views
        via ``as_strided`` so ``.cpu()`` reads the right elements.

        Inductor emits ``reinterpret_tensor(t, size, stride, offset)`` to
        express a zero-copy stride/shape view of ``t``. Upstream's helper
        ``torch._C._dynamo.guards._reinterpret_tensor`` only rewrites
        tensor metadata — it never touches the underlying storage. That
        contract is fine for CUDA/CPU where every consumer (kernels,
        copies, ``.cpu()``) honors the new ``stride`` field when reading.

        On Vulkan the contract breaks specifically for *graph-output*
        non-contiguous views (e.g. ``torch.diagonal(x)`` lowering to
        ``reinterpret_tensor(x, (N,), (M+1,), 0)``). The runtime
        ``_copy_from`` (.cpu) path linearizes the source storage as if it
        were row-major contiguous (4 B/elem pack), ignoring metadata
        stride. For ``ii->i`` on a 4×4 matrix this produces the first row
        ``[0,1,2,3]`` instead of the diagonal ``[0,5,10,15]``.

        Fix: when this codegen call is emitting a *graph output* (called
        from ``get_output_refs`` / ``generate_return``) AND the view is
        non-contiguous, emit ``old.as_strided(size, stride[, offset])``.
        ``aten.as_strided`` on Vulkan dispatches the
        ``copy_as_strided_fwd`` Slang shader which allocates a fresh
        contiguous buffer and reads ``old`` with the user-supplied
        stride, producing a correctly materialized tensor that
        ``.cpu()`` then bulk-copies.

        Internal (non-output) reinterpret_tensor calls (e.g. addmm's
        weight transpose ``arg1_1.as_strided((K,N), (1,K))`` from a
        linear lowering) are *not* intercepted: their consumers
        (extern_kernels.addmm, mm, etc.) honor the metadata stride at
        the kernel-binding level. Materializing those would (a) add
        spurious dispatches and (b) lose the requested stride metadata
        because Vulkan ``as_strided`` returns a contiguous-strided
        result tensor.
        """
        try:
            device = data.get_device()
        except (AttributeError, NotImplementedError):
            device = None
        if device is None or device.type != "vulkan":
            return super().codegen_reinterpret_view(
                data, size, stride, offset, writeline, dtype=dtype
            )

        # Only intercept when this call is emitting a graph-output ref.
        # The flag is set by our ``generate_return`` override below.
        if not getattr(self, "_vulkan_in_generate_return", False):
            return super().codegen_reinterpret_view(
                data, size, stride, offset, writeline, dtype=dtype
            )

        # Determine the view's natural ("contiguous") strides for `size`.
        # If `stride` matches the natural strides AND `offset == 0`, the
        # underlying row-major buffer can be read directly — defer to
        # upstream which emits a metadata-only reinterpret_tensor.
        def _natural_strides(sz):
            out: list = []
            running: object = 1
            for s in reversed(list(sz)):
                out.append(running)
                # Multiply symbolically — `running` may be sympy.Expr or int.
                running = running * s
            out.reverse()
            return out

        # Size-1 dims have arbitrary stride upstream (often 0); normalize
        # both `stride` and the natural strides at those positions so
        # they don't spuriously force the materialization path.
        def _normalize(sz, st):
            return tuple(0 if int(s) == 1 else x for s, x in zip(sz, st))

        try:
            sz_list = list(size)
            st_list = list(stride)
            nat = _natural_strides(sz_list)
            try:
                int_sz = [int(s) for s in sz_list]
                normalized_requested = _normalize(int_sz, st_list)
                normalized_natural = _normalize(int_sz, nat)
                contiguous_strided = normalized_requested == normalized_natural and (
                    offset == 0 or str(offset) == "0"
                )
            except (TypeError, ValueError):
                contiguous_strided = tuple(st_list) == tuple(nat) and (
                    offset == 0 or str(offset) == "0"
                )
        except Exception:
            contiguous_strided = True  # fail safe — defer to upstream

        if contiguous_strided:
            return super().codegen_reinterpret_view(
                data, size, stride, offset, writeline, dtype=dtype
            )

        # Non-contiguous graph-output view on Vulkan → materialize.
        from torch._inductor.codegen.wrapper import codegen_reinterpret_view_helper

        _, _, _, d_dtype, collapsible = codegen_reinterpret_view_helper(data)
        name = data.get_name()
        base_dtype = d_dtype if collapsible else data.dtype

        s = self.codegen_python_shape_tuple(size)
        st = self.codegen_python_shape_tuple(stride)
        off = self.codegen_sizevar(offset)
        if off == "0":
            expr = f"{name}.as_strided({s}, {st})"
        else:
            expr = f"{name}.as_strided({s}, {st}, {off})"
        if dtype is not None and dtype != base_dtype:
            return f"aten.view.dtype({expr}, {dtype})"
        return expr

    def get_output_refs(self) -> list[str]:
        """T.12.B — wrap the cached ``get_output_refs`` so that
        ``codegen_reinterpret_view`` knows it's being called for graph
        outputs (not internal kernel args). Only the output codegen
        materializes non-contiguous Vulkan views via ``as_strided``;
        internal kernel arg sites keep the cheap zero-copy
        ``reinterpret_tensor`` path.
        """
        self._vulkan_in_generate_return = True
        try:
            return super().get_output_refs()
        finally:
            self._vulkan_in_generate_return = False

    def make_buffer_reuse(self, old, new, delete_old):
        # On the reuse path the storage is *aliased* to the new name, so
        # the old name's `del` must not pool-release (that would re-vend a
        # tensor still in use). Mirror upstream make_buffer_reuse but inline
        # a plain `del`.
        # PF.54 audit residual: tuple-holder buffers (MultiOutputLayout) do
        # not have a single ``size``/``stride``/``dtype``; defer to upstream
        # which only invokes us with single-tensor layouts in practice.
        if not isinstance(old.get_output_spec(), ir.Layout) or not isinstance(
            new.get_output_spec(), ir.Layout
        ):
            return super().make_buffer_reuse(old, new, delete_old)
        assert old.get_dtype() == new.get_dtype()
        old_name = old.get_name()
        new_name = new.get_name()
        del_line = ";"
        if old_name not in V.graph.get_output_names() and delete_old:
            del_line = f"; del {old_name}"
        if old.get_size() == new.get_size() and old.get_stride() == new.get_stride():
            return self.codegen_exact_buffer_reuse(old_name, new_name, del_line)
        reinterpret_view = self.codegen_reinterpret_view(
            old, new.get_size(), new.get_stride(), 0, self.wrapper_call.writeline
        )
        return (
            f"{self.declare}{new_name} = {reinterpret_view}{del_line}"
            f"{self.ending}  {self.comment} reuse"
        )

    def _generate_kernel_call_helper(
        self,
        kernel_name: str,
        call_args,
        *,
        device=None,
        triton=True,
        arg_types=None,
        raw_keys=None,
        raw_args=None,
        triton_meta=None,
        inductor_meta=None,
        graph_name: str = "",
        original_fxnode_name=None,
        current_stream_idx=None,
        **kwargs,
    ) -> None:
        # TR.13: newer Inductor builds (>= 2026-Q2) thread ``current_stream_idx``
        # through the kernel-call helper for CUDA-stream awareness. Vulkan has
        # no per-stream dispatch concept (we serialize on a single VkQueue),
        # so we accept and discard the kwarg. ``**kwargs`` absorbs any further
        # additions on upstream's side without breaking the override.
        if device is not None and device.type == "vulkan":
            call_str = ", ".join(call_args)
            # GPU.1: Batch dispatch via DispatchBatcher.
            if _batch_dispatch_enabled():
                self.writeline(f"_batcher.add({kernel_name}, {call_str})")
                return
            # GPU.3: Profiling with perf_counter_ns.
            if _profile_dispatches_enabled():
                self.writeline(f"_t0 = perf_counter_ns()")
                self.writeline(f"{kernel_name}({call_str})")
                self.writeline(
                    f"_record_dispatch_time({kernel_name!r}, "
                    f"(perf_counter_ns() - _t0) / 1000)"
                )
                return
            self.writeline(self.wrap_kernel_call(kernel_name, call_args))
            return
        # Forward only the kwargs the upstream parent actually accepts.
        super()._generate_kernel_call_helper(
            kernel_name,
            call_args,
            device=device,
            triton=triton,
            arg_types=arg_types,
            raw_keys=raw_keys,
            raw_args=raw_args,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            graph_name=graph_name,
            original_fxnode_name=original_fxnode_name,
        )
