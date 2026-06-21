"""Vulkan-specific Inductor wrapper codegen.

Subclasses `PythonWrapperCodegen` to emit Python kernel-call lines for the
vulkan device (which upstream's base class marks as "nyi"). For each kernel
we emit a plain function call — the runtime module takes the generated-kernel
function and args and dispatches the Slang-compiled SPIR-V to our Vulkan
runtime.

Module-level helpers and monkey-patches live in ``wrapper_helpers.py``
(M22.1 split to keep this file under the 800-line anti-goal #7 cap).
"""

from __future__ import annotations

from typing import Optional

import torch._inductor.config
from torch._inductor import ir
from torch._inductor.codegen.wrapper import (
    PythonWrapperCodegen,
    SubgraphPythonWrapperCodegen,
)
from torch._inductor.virtualized import V

from .fx_passes.alloc_alias_ir import apply_vulkan_ir_alias_pass
from .wrapper_helpers import (
    _batch_dispatch_enabled,
    _lifetime_class_for_name,
    _normalize_strides_to_row_major,
    _profile_dispatches_enabled,
    _trust_inductor,
    _wrapper_fastpath_enabled,
)


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

    def __init__(self):
        super().__init__()
        # M9.7: track lifetime_class per output buffer name for the stash.
        self._output_lifetime_map: dict[str, str] = {}

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
        # M9.7: per-function output stash for lookback recycling.
        # Populated at return-time; drained at the start of the next call.
        self.header.splice("_stashed_outputs = []\n")

    def write_prefix(self) -> None:
        super().write_prefix()
        # GPU.1: Initialize the DispatchBatcher at function scope.
        if _batch_dispatch_enabled():
            self.writeline("_batcher = DispatchBatcher()")
            self.writeline("_batcher.__enter__()")
        # M9.7: release outputs stashed by the previous call.
        # Drained before any new allocations so the pool can re-vend
        # same-sized output buffers immediately (lookback depth 1).
        # Manual 4-space indentation relative to function body —
        # the assembly wraps wrapper_call with 1 level of indent.
        self.writeline("if _stashed_outputs:")
        self.writeline("    for _t, _lt in _stashed_outputs:")
        self.writeline("        vulkan_pool_release(_t, lifetime_class=_lt)")
        self.writeline("    _stashed_outputs.clear()")

    def generate_end_graph(self):
        # GPU.1: Exit the DispatchBatcher before graph end — flushes all.
        if _batch_dispatch_enabled():
            self.writeline("_batcher.__exit__(None, None, None)")
        super().generate_end_graph()

    def run_wrapper_ir_passes(self, is_inference: bool) -> None:
        """M22.2: extend the upstream memory-planning pass with a Vulkan-specific
        IR-level alloc-alias pass.

        Upstream's ``memory_plan_reuse()`` already converts adjacent same-size
        pairs into ``ReuseLine``.  After that runs, ``apply_vulkan_ir_alias_pass``
        scans the remaining ``AllocateLine`` / ``FreeIfNotReusedLine`` entries and
        aliases pairs with matching Vulkan alloc keys that have non-overlapping
        lifetimes (the old buffer freed before the new one is allocated).
        """
        super().run_wrapper_ir_passes(is_inference)
        try:
            apply_vulkan_ir_alias_pass(self)
        except Exception:
            # Never let a cosmetic optimization break compilation.
            pass

    def generate_return(self, output_refs):
        # GPU.1: Flush batcher before return so kernels execute.
        if _batch_dispatch_enabled():
            self.wrapper_call.writeline("_batcher.__exit__(None, None, None)")
        # M9.7: stash outputs for next-call recycling (lookback depth 1).
        # Emit (tensor, lt) tuples so release reuses the correct pool key.
        # Skip gradient-class outputs: they transfer ownership to AccumulateGrad
        # (param.grad assignment). Stashing them keeps use_count >= 2, which
        # makes is_tensor_stealable() return false and forces clone_obey_contract
        # (new_empty_strided + copy_ = 1 Vulkan dispatch per parameter).
        # Gradient tensors are freed naturally when zero_grad() releases param.grad.
        if output_refs:
            stash_entries = []
            for ref in output_refs:
                base_name = ref.split(".")[0].split("(")[0].strip()
                lt = self._output_lifetime_map.get(base_name, "transient")
                if lt != "gradient":
                    stash_entries.append(f"({ref}, {lt!r})")
            if stash_entries:
                self.wrapper_call.writeline(
                    f"_stashed_outputs[:] = [{', '.join(stash_entries)}]"
                )
        super().generate_return(output_refs)

    # M22.2: The regex post-processor (M17.7) is superseded by the IR-level
    # alias pass in ``run_wrapper_ir_passes``.  ``generate()`` no longer needs
    # to touch the assembled source string.  The override is kept as a no-op
    # pass-through so the call chain is explicit.
    def generate(self, is_inference):
        return super().generate(is_inference)

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

    def _flush_batcher_before_direct_call(self) -> None:
        """M-NEW.12: emit ``_batcher._flush()`` before any direct (non-
        ``_batcher.add``) GPU-touching call.

        The wrapper queues Triton-style kernel dispatches via
        ``_batcher.add(kernel, ...)`` (see ``_generate_kernel_call_helper``)
        and only flushes at ``_batcher.__exit__``. But custom-op /
        template-caller lines such as ``torch.ops.torch_vulkan.foo(...)``
        and ``_slang_tile_conv2d(buf2, ...)`` are emitted as immediate
        function calls that dispatch synchronously. If a direct call
        reads a buffer that a queued kernel was supposed to populate,
        the read sees uninitialised (zero) data.

        Repro (pre-fix): SmallCNN's ``conv1 → gn1 → relu → maxpool →
        conv2 → gn2 → relu`` yields bias-only output at the second
        ``conv2d_gn_relu_fused`` because the preceding ``vulkan_kernel_0``
        (MaxPool2d) is queued and unflushed when ``conv2d_gn_relu_fused``
        runs. unique-value count: 32 (one per output channel = the bias
        per-channel value) instead of the expected per-spatial values.

        Idempotent on the no-pending path (``flush_current_if_active``
        no-ops if nothing is queued).
        """
        if _batch_dispatch_enabled():
            self.writeline("DispatchBatcher.flush_current_if_active()")

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

        # M-NEW.12: this is a direct dispatch; flush queued batcher work first.
        self._flush_batcher_before_direct_call()

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
        # M9.7: track lifetime class for output stash.
        self._output_lifetime_map[name] = lifetime_class

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
        # M-NEW.12: this is a direct dispatch (e.g. ``torch.ops.foo.out(...)``
        # or a template caller line emitted from a custom ``codegen()`` —
        # see ``lowerings/conv.py::_VulkanConv2dExternKernel``); flush any
        # queued ``_batcher.add`` dispatches first so the call reads
        # populated input buffers rather than zero-initialised ones.
        self._flush_batcher_before_direct_call()
        super().generate_extern_kernel_out(node)

    def generate_fallback_kernel(self, node) -> None:
        """M-NEW.12: flush the ``DispatchBatcher`` before any FallbackKernel
        line (custom ops like ``torch.ops.torch_vulkan.conv2d_gn_relu_fused``).

        These fall through ``ExternKernelAllocLine`` rather than
        ``generate_extern_kernel_alloc``, so the batcher-flush injection in
        ``generate_extern_kernel_alloc`` doesn't reach them. See the
        SmallCNN second-block trace: a ``conv2d_gn_relu_fused`` call
        immediately after a queued maxpool kernel reads from an
        un-flushed buffer.
        """
        # Determine if this node touches a vulkan device. FallbackKernel
        # nodes always have a device; non-vulkan goes straight to super().
        try:
            dev = node.get_device()
        except Exception:  # noqa: BLE001
            dev = None
        if dev is not None and dev.type == "vulkan":
            self._flush_batcher_before_direct_call()
        super().generate_fallback_kernel(node)

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
            # M9.7: track lifetime class for output stash.
            self._output_lifetime_map[name] = lifetime_class
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

    def _reuse_reads_donor(self, old_name, new):
        """True if the op producing ``new`` reads ``old_name`` — i.e. this is
        an in-place mutation reuse (the kernel reads the donor's data through
        the aliased output binding, ``in_out_ptr0``), not a reinterpret-view
        reuse.  For the in-place case the alias must be preserved or the kernel
        reads uninitialized memory (S2.0d-resid).  Best-effort: any failure
        falls back to ``False`` (treat as a view reuse, the prior behaviour)."""
        new_name = new.get_name()
        try:
            sched = getattr(V.graph, "scheduler", None)
            if sched is not None:
                sbuf = getattr(sched, "name_to_buf", {}).get(new_name)
                if sbuf is not None and getattr(sbuf, "defining_op", None):
                    for dep in sbuf.defining_op.read_writes.reads:
                        if dep.name == old_name:
                            return True
        except Exception:
            pass
        try:
            get = getattr(new, "get_read_names", None)
            if get is not None and old_name in get():
                return True
        except Exception:
            pass
        return False

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
        # S2.0/S2.1: a *graph output* must not be a different-shape
        # ``reinterpret_tensor`` alias of an internal donor.  On Vulkan this
        # aliasing is unsafe when the output is then written by an extern
        # dispatch: the runtime's per-buffer barrier/binding tracking keys on
        # the donor's storage, so the write-after-read hazard against the
        # donor's last reader is missed — observed as an ~80%-wrong conv
        # ``grad_bias`` returned gradient when a GroupNorm-stats buffer (stride
        # ``(4,1,16)``) was reinterpret-reused as the bias output (repro:
        # agent_space/probe_variants.py).  For an output buffer, allocate it
        # fresh and release the donor to the pool — allocate *before* freeing
        # so the pool can't re-hand the donor's buffer back (which would
        # re-introduce the alias).  Internal (non-output) reshape-reuse stays
        # on the cheap zero-copy path: its consumers are codegen kernels that
        # honour the view's stride, and the buffer-pool hit-rate depends on it.
        #
        # S2.0d-resid: the fresh-alloc is only correct for a *reinterpret-view*
        # reuse, where the producer of ``new`` fully overwrites it and never
        # reads the donor's old contents.  An *in-place mutation* reuse — where
        # the kernel reads ``old`` through the (aliased) output binding and
        # rewrites it in place (emitted as ``in_out_ptr0``) — REQUIRES the
        # alias: fresh-allocating ``new`` makes the kernel read uninitialized
        # memory.  This is exactly what corrupts a standalone GroupNorm's saved
        # ``rstd`` (``rstd = rsqrt(var+eps)`` computed in place over the welford
        # ``var`` buffer, then returned as a graph output) → garbage rstd →
        # ~100%-wrong ``gn.weight`` gradient (bias is correct: it doesn't use
        # rstd).  Detect the in-place case (``new`` reads ``old``) and keep it
        # on the alias path; the WAR hazard the fresh-alloc once guarded against
        # is now covered by the C++ read-tracking WAR barrier (S2.0d).
        if new_name in V.graph.get_output_names() and not self._reuse_reads_donor(
            old_name, new
        ):
            self.wrapper_call.writeline(
                self.make_allocation(
                    new_name,
                    new.get_device(),
                    new.get_dtype(),
                    new.get_size(),
                    new.get_stride(),
                )
            )
            if old_name not in V.graph.get_output_names() and delete_old:
                return self.make_buffer_free(old)
            return (
                f"{self.comment} S2.0: fresh alloc for output {new_name} "
                f"(donor {old_name} kept)"
            )
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
