"""Register the Vulkan backend with Inductor.

Import this module once (typically from `torch_vulkan.__init__`) to hook the
scheduler/wrapper/device-op-override classes into Inductor's backend registry.
After this runs, `torch.compile(fn, backend="inductor")` will route kernels
for `torch.device("vulkan")` tensors through our Slang codegen.

Supports pointwise, reduction (sum/prod/max/min/argmax/argmin/welford/any),
view ops, and extern kernels (mm/conv/attention fall back to eager dispatch
with pointwise epilogue fusion where possible).
"""

from __future__ import annotations

_registered = False


def _patch_aot_joint_trace(ctx_cls) -> None:
    """PF.1 / B1' frontend: convert meta-device tangents to vulkan FakeTensors
    BEFORE AOT autograd's joint graph trace runs.

    Without this, the autograd-codegen path inside ``torch.autograd.grad``
    (running under ``FunctionalTensorMode + FakeTensorMode + ProxyTorchDispatchMode``)
    sees a meta-device tangent, fails ``any_variable_defined(grads)`` in the
    generated ``SumBackward0_apply_functional`` (and equivalent for other
    backwards), and emits ``aten.empty.memory_format(..., device='meta')`` in
    place of ``grad.expand_symint(self.sym_sizes())``. The traced backward
    therefore loses every reference to ``tangents_*``, the partitioner sees
    a backward branch that's "constant w.r.t. inputs", and saves a static
    pre-computed (uninitialized) buffer — the canonical 0.5×/NaN gradient.

    Implementation: convert each meta-device tensor in the joint inputs by
    routing through ``FakeTensorMode.from_tensor`` on a real vulkan-device
    tensor. ``FakeTensor.__new__`` with a meta source is *not* sufficient —
    the resulting FakeTensor's expand() falls back to meta dispatch (its
    underlying TensorImpl points at the original meta storage), which
    cascades back into the same autograd-codegen failure. The
    ``from_tensor`` path constructs a fresh FakeTensor whose storage is
    correctly tagged vulkan and whose ``defined()`` returns True from C++
    autograd's perspective.
    """
    import logging

    _log = logging.getLogger(__name__)
    try:
        import torch
        import torch._functorch._aot_autograd.graph_capture as _gc
        from torch.utils._pytree import tree_map

        _orig_create_graph = _gc._create_graph
        _ft = torch._subclasses.fake_tensor
        _vulkan_dev = torch.device("vulkan", 0)

        def _fix_tensor(t):
            if not isinstance(t, torch.Tensor) or t.device.type != "meta":
                return t
            if isinstance(t, _ft.FakeTensor):
                fm = t.fake_mode
                # Route through a real vulkan tensor so ``from_tensor``
                # mints a FakeTensor whose underlying storage is properly
                # tagged vulkan. Going via FakeTensor.__new__ on a meta
                # source produces a FakeTensor whose expand() falls back
                # to meta dispatch (PF.1 root cause).
                real = torch.empty_strided(
                    list(t.shape),
                    list(t.stride()),
                    dtype=t.dtype,
                    device=_vulkan_dev,
                )
                return fm.from_tensor(real, static_shapes=True)
            return torch.empty_strided(
                t.shape,
                t.stride(),
                dtype=t.dtype,
                device=_vulkan_dev,
            )

        def _patched_create_graph(f, args, args_descs=None, **kwargs):
            fixed_args = tree_map(_fix_tensor, args)
            with ctx_cls():
                return _orig_create_graph(f, fixed_args, args_descs, **kwargs)

        _gc._create_graph = _patched_create_graph
        _log.info("Patched _create_graph to fix meta-device tangents to vulkan")
    except Exception as e:
        _log.warning("Patching _create_graph failed: %s", e)


def _patch_bw_compiler_devices(fix_pass_cls) -> None:
    import sys

    import torch as _torch
    import torch._inductor.compile_fx as _cfx

    _orig_bw = _cfx.compile_fx_backward

    import logging

    _log = logging.getLogger(__name__)

    if _orig_bw.__code__.co_name == "_new_bw_compiler":
        _log.warning("_patch_bw_compiler_devices: already patched, skipping")
        return

    _fixer = fix_pass_cls()

    def _new_bw_compiler(gm, example_inputs, **kwargs):
        _fixer(gm)
        return _orig_bw(gm, example_inputs, **kwargs)

    _cfx.compile_fx_backward = _new_bw_compiler
    _log.info("_patch_bw_compiler_devices: patched compile_fx_backward")


def _install_vulkan_gpu_types() -> None:
    """PF.30.h.1 — register Vulkan as a recognized GPU device type.

    Without this, Inductor's autotune harness treats Vulkan tensors as
    non-GPU and falls back to ``device_type="cuda"`` in
    ``GPUDeviceBenchmarkMixin.do_bench`` and ``benchmark_choice``. On a
    CUDA-less build that path raises ``"Torch not compiled with CUDA
    enabled"`` from ``torch.cuda.current_device()``, which surfaces as
    ``LoweringException`` and breaks every matmul autotune path.

    Append ``"vulkan"`` to ``torch._inductor.utils.GPU_TYPES`` so
    ``is_gpu("vulkan")`` returns True, routing ``do_bench`` through
    our ``VulkanInterface`` instead of the CUDA interface.

    Idempotent — safe to call multiple times.
    """
    import logging

    _log = logging.getLogger(__name__)
    try:
        from torch._inductor import utils as _ind_utils

        if "vulkan" not in _ind_utils.GPU_TYPES:
            _ind_utils.GPU_TYPES.append("vulkan")
            _log.info("Added 'vulkan' to torch._inductor.utils.GPU_TYPES")
    except Exception as e:
        _log.warning("Failed to append vulkan to GPU_TYPES: %s", e)


def _install_vulkan_scheduler_exemption() -> None:
    """PF.30.h.3 — exempt Vulkan from TritonMissing raise in Scheduler.

    Once ``is_gpu("vulkan")`` returns True, upstream's
    ``Scheduler.create_backend`` raises ``TritonMissing`` on every
    Vulkan compile because ``has_triton()`` is False and the device
    isn't ``"mps"``. Without this exemption, step 1 alone breaks every
    Vulkan compile.

    Idempotent — safe to call multiple times.
    """
    import logging

    _log = logging.getLogger(__name__)
    try:
        from torch._inductor import scheduler as _sched

        _orig = _sched.Scheduler.create_backend
        if getattr(_orig, "_vulkan_patched", False):
            return

        def _vulkan_aware_create_backend(self, device):
            import inspect

            from torch._inductor.codegen.common import get_scheduling_for_device
            from torch._inductor.exc import GPUTooOldForTriton, TritonMissing
            from torch._inductor.utils import is_gpu
            from torch._inductor.virtualized import V
            from torch.utils._triton import has_triton

            assert not is_gpu(device.type) or device.index is not None, (
                f"{device} should have been normalized in lowering"
            )
            V.graph.add_device_info(device)
            device_scheduling = get_scheduling_for_device(device.type)
            if device_scheduling is None:
                raise RuntimeError(f"Unsupported device type: {device.type}")
            if not has_triton():
                if (
                    device.type == "cuda"
                    and (
                        device_props := __import__("torch").cuda.get_device_properties(
                            device
                        )
                    ).major
                    < 7
                ):
                    raise GPUTooOldForTriton(device_props, inspect.currentframe())
                elif is_gpu(device.type) and device.type not in ("mps", "vulkan"):
                    raise TritonMissing(inspect.currentframe())
            return device_scheduling(self)

        _vulkan_aware_create_backend._vulkan_patched = True
        _sched.Scheduler.create_backend = _vulkan_aware_create_backend
        _log.info(
            "Patched Scheduler.create_backend to exempt vulkan from TritonMissing"
        )
    except Exception as e:
        _log.warning("Failed to patch Scheduler.create_backend: %s", e)


# M17.1: _install_vulkan_aten_only_autotune removed (2026-05-16).
# Slang tile mm/bmm/addmm correctness is now verified (max diff ~1e-5 vs CPU).
# The gate was already dead code (never called from _legacy_register).
# Use TORCH_VULKAN_DISABLE_SLANG_TILES=1 to disable Slang tiles if needed.


def _install_vulkan_cpu_timer_benchmark() -> None:
    """PF.30.h.2 — route Vulkan benchmark calls through the CPU timer.

    Upstream's ``Benchmarker.benchmark`` dispatches non-CPU devices to
    ``benchmark_gpu``, which ``TritonBenchmarker`` overrides via
    ``triton.testing.do_bench`` — but Triton isn't installed here.
    Without this patch every autotune choice raises
    ``ModuleNotFoundError: triton``, gets caught as
    ``NotImplementedError``, and timing is forced to inf →
    ``NoValidChoicesError``.

    Vulkan synchronize fires before/after via
    ``GPUDeviceBenchmarkMixin.do_bench``
    (``device_interface.synchronize()``), so the CPU timer reports
    honest wall-clock for the dispatched + flushed pipeline.

    Idempotent — safe to call multiple times.
    """
    import logging

    _log = logging.getLogger(__name__)
    try:
        import torch as _torch
        from torch._inductor.runtime import benchmarking as _bm

        _b = _bm.benchmarker
        if getattr(_b, "_vulkan_routed", False):
            return

        _orig_benchmark = _b.benchmark.__func__

        def _vulkan_benchmark(
            self, fn, fn_args=None, fn_kwargs=None, device=None, **kwargs
        ):
            inferred = device
            if inferred is None and (fn_args or fn_kwargs):
                inferred = self.infer_device(*(fn_args or ()), **(fn_kwargs or {}))
            if isinstance(inferred, str):
                inferred = _torch.device(inferred)
            if isinstance(inferred, _torch.device) and inferred.type == "vulkan":
                fn_args = fn_args or ()
                fn_kwargs = fn_kwargs or {}
                if not fn_args and not fn_kwargs:
                    callable_ = fn
                else:
                    callable_ = lambda: fn(*fn_args, **fn_kwargs)
                warmup = kwargs.pop(
                    "warmup",
                    _bm.inductor_config.inductor_default_autotune_warmup,
                )
                rep = kwargs.pop(
                    "rep",
                    _bm.inductor_config.inductor_default_autotune_rep,
                )
                return self.benchmark_cpu(callable_, warmup=warmup, rep=rep)
            return _orig_benchmark(
                self,
                fn,
                fn_args=fn_args,
                fn_kwargs=fn_kwargs,
                device=device,
                **kwargs,
            )

        import types

        _b.benchmark = types.MethodType(_vulkan_benchmark, _b)
        _b._vulkan_routed = True
        _log.info("Routed vulkan benchmark calls through CPU timer (PF.30.h.2)")
    except Exception as e:
        _log.warning("Failed to route Vulkan benchmark to CPU timer: %s", e)


# PF.30.h DRAFT reference — preserved for historical context.
# The above 4 install_* functions supersede the monolithic DRAFT.
# The DRAFT was shelved because it unmasked the GAP 0.1 wrapper-args bug,
# which was fixed by P5.11.a (2026-05-02). See docs/10-inductor-backend.md P1.1.
def _patch_register_vulkan_as_gpu_DRAFT_PF30H() -> None:
    """Archaic; superseded by _install_vulkan_gpu_types et al."""
    pass


def _namespace_inductor_cache() -> None:
    """Point the Inductor codecache at a vulkan-version-scoped subdirectory.

    Inductor's codecache normally lives at ``/tmp/torchinductor_$USER``. A
    backend-version swap (different git sha, slangc upgrade) silently reuses
    stale codegen otherwise, which is hard to debug. We override
    ``TORCHINDUCTOR_CACHE_DIR`` to a sibling path scoped by a short hash of
    the inductor module mtimes so a fresh ``pip install -e .`` gets fresh
    cache entries without manual ``rm -rf``. P5.7.

    Disable via ``TORCH_VULKAN_NO_CACHE_NS=1`` or by setting
    ``TORCHINDUCTOR_CACHE_DIR`` explicitly in the environment.
    """
    import getpass
    import os
    import tempfile

    if os.environ.get("TORCH_VULKAN_NO_CACHE_NS") == "1":
        return
    if os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        return  # respect user override
    sha = _backend_version_tag()
    if not sha:
        return
    try:
        user = getpass.getuser()
    except Exception:
        user = "default"
    cache_dir = os.path.join(
        tempfile.gettempdir(), f"torchinductor_{user}_vulkan_{sha}"
    )
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir


def _backend_version_tag() -> str:
    """Short identifier for the currently-installed torch_vulkan build.

    Cheap to compute (no subprocess on the hot path): hashes a few stable
    file paths' mtimes inside the inductor package so a developer's
    `pip install -e .` rebuild is reflected, but we don't pay a git
    invocation per Python startup.
    """
    import hashlib
    import os

    inductor_dir = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha1()
    for fname in (
        "kernel/main.py",
        "scheduling.py",
        "wrapper.py",
        "runtime.py",
        "overrides.py",
        "lowerings.py",
    ):
        path = os.path.join(inductor_dir, fname)
        try:
            st = os.stat(path)
        except OSError:
            continue
        h.update(f"{fname}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:12]


def _legacy_register() -> None:
    """Body of the historical scattered registration.

    Invoked by ``SlangVulkanBackend._register_with_inductor`` (see
    ``backend.py``). Subsequent PRs (Frontend / SlangIR / Runtime /
    Heuristics) migrate sections of this body into typed subsystem
    classes; once empty, the function and this comment go away.
    """
    global _registered
    if _registered:
        return

    _namespace_inductor_cache()

    # PF.30.h — install Vulkan autotune harness support before any
    # lowering or template registration triggers autotune code paths.
    # These 3 patches route the Inductor benchmarker through our
    # device interface instead of the CUDA/Triton paths. Slang tile
    # matmul is excluded at source (vulkan_template_caller._slang_tiles_enabled).
    _install_vulkan_gpu_types()
    _install_vulkan_scheduler_exemption()
    _install_vulkan_cpu_timer_benchmark()

    from torch._inductor.codegen.common import register_backend_for_device

    from . import meta_patches
    from .codegen import VulkanScheduling
    from .cpp_wrapper_gpu import VulkanCppWrapperGpu
    from .device_interface import register as _register_dev_iface
    from .device_op_overrides import register as _register_dev_ops
    from .wrapper import VulkanPythonWrapperCodegen

    _register_dev_iface()

    from .fx_passes import _make_vulkan_pass

    # DR.8 / T7.5: register VulkanCppWrapperGpu as the AOTI C++ wrapper.
    # When V.graph.aot_mode is True, Inductor selects this wrapper instead
    # of VulkanPythonWrapperCodegen, emitting C++ that calls the Vulkan AOTI
    # C ABI directly.
    register_backend_for_device(
        "vulkan",
        VulkanScheduling,
        VulkanPythonWrapperCodegen,
        VulkanCppWrapperGpu,
        None,
        _make_vulkan_pass(),
    )
    _register_dev_ops()
    meta_patches.apply()
    meta_patches._patch_compile_fx_for_backward()
    meta_patches._install_joint_partition_device_fix()
    _patch_aot_joint_trace(meta_patches._joint_trace_ctx)
    _patch_bw_compiler_devices(meta_patches._FixMetaDevicePass)

    # Enable Inductor's back-to-back GEMM fusion pass once at backend
    # registration. Used to be re-set per FX-graph inside _VulkanCustomPass,
    # which is harmless but wasteful — the config flag is global.
    from torch._inductor import config as _ic

    if hasattr(_ic, "b2b_gemm_pass"):
        _ic.b2b_gemm_pass = True

    # T5.4 Phase A: enable upstream combo_kernels to group sibling
    # pointwise ops into ForeachKernelSchedulerNodes, reducing dispatch
    # count on large pointwise chains.
    if hasattr(_ic, "combo_kernels"):
        _ic.combo_kernels = True
    # TRAIN.6-F1: Wave-uniform combo-kernel dispatch for reductions enabled.
    # VulkanComboKernel now uses a multi-dimensional grid where gid.y selects
    # the subkernel and gid.x is the subkernel's own workgroup ID. All threads
    # in a workgroup execute the same subkernel body, preserving wave
    # uniformity for reduction intrinsics (WaveActiveSum/WaveActiveMax).
    # Previously (TRAIN.6): combo_kernels_pointwise_only = True, which
    # excluded reduction kernels from combo fusion entirely.
    if hasattr(_ic, "combo_kernels_pointwise_only"):
        _ic.combo_kernels_pointwise_only = False

    from .vulkan_template_caller import (
        install_external_addmm,
        install_external_bmm,
        install_external_flash_attention,
        install_external_mm,
        install_external_optimizer,
        prewarm_matmul_templates,
    )

    install_external_mm()
    install_external_bmm()
    install_external_addmm()
    install_external_flash_attention()
    # T4.8: foreach optimizer template — registers
    # torch_vulkan::foreach_{sgd,sgd_momentum,adamw,lion}_step custom ops.
    # Re-enabled after M24 fix (list[float] schema, not float|list[float]).
    install_external_optimizer()

    from . import lowerings as _lowerings

    _lowerings.register()

    # Register fused-op custom_ops eagerly so the FX rewrites and any
    # downstream lowering have a stable OpOverload to reference.
    from .fx_passes import (
        _ensure_addmm_gelu_op_registered,
        _ensure_flash_attention_op_registered,
        _ensure_qkv_cat_op_registered,
        _ensure_scaled_bmm_op_registered,
        _ensure_swiglu_op_registered,
        register_eager_patch_custom_ops,
    )

    _ensure_scaled_bmm_op_registered()
    _ensure_swiglu_op_registered()
    _ensure_flash_attention_op_registered()
    _ensure_qkv_cat_op_registered()
    _ensure_addmm_gelu_op_registered()
    # PF.30.a/.b/.c — register the conv2d/conv1d/sdpa/max_pool2d
    # custom_op shims that the eager monkey-patches dispatch through
    # under torch.compile. Without this, the patches retain
    # @torch.compiler.disable and graph-break Dynamo at every call.
    register_eager_patch_custom_ops()

    # Fire off the slangc pre-warm in the background. The pool finishes
    # before the user's first compiled dispatch in the common case; if not,
    # the in-flight compile hits the same cache key and shares the result.
    # Disable with TORCH_VULKAN_NO_PREWARM=1.
    try:
        prewarm_matmul_templates(sync=False)
    except Exception:
        # Pre-warm is best-effort — never let it block backend registration.
        pass

    # M9.3: precompile `shaders/lib/*.slang` → `.slang-module` artifacts at
    # import time so the first user dispatch doesn't pay the cold cost.
    # The audit measured 9 s for the first 8 SmallCNN dispatches (~800 ms
    # each) — most of that is slangc parsing the lib imports on every
    # kernel compile. Pre-emitting the modules amortises that to import
    # time. Background thread, best-effort, opt-out via the same
    # TORCH_VULKAN_NO_PREWARM env var as the matmul prewarm.
    try:
        from .runtime import prewarm_shader_libs

        prewarm_shader_libs(sync=False)
    except Exception:
        pass

    _registered = True


def register() -> None:
    """Idempotently register the Vulkan Inductor backend.

    Public entry point. Routes through ``SlangVulkanBackend`` so a single
    class owns lifecycle of the four subsystems documented in the
    reorganization plan. The first cut is a no-op delegator — every
    behavior comes from ``_legacy_register``.
    """
    from .backend import SlangVulkanBackend

    SlangVulkanBackend.register()


# Auto-register on import.
register()
