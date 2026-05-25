"""Vulkan dispatch layer — JIT pipeline, kernel wrapping, dispatch FFI.

Provides the bridge between compiled SPIR-V and the C++ Vulkan runtime:
pybind dispatch entries, kernel wrapper factories (make_vulkan_kernel),
and AOTI model export.
"""

import hashlib
import os
import time
from typing import Optional

import torch

# ── Per-kernel stats ───────────────────────────────────────────────────

_KERNEL_STATS: dict[str, dict] = {}
# Maps each Inductor-generated kernel cache_key to the 12-char prefix of its
# SPIR-V SHA256. Populated whenever a kernel is built; lets observability
# tooling (e.g. `inductor_stats.summary()`) correlate per-kernel timing back
# to specific compiled binaries when debugging cache-miss / autotune churn.
_KERNEL_SPIRV_HASH: dict[str, str] = {}
# SPIR-V cache, keyed by the stable cache_key the generated wrapper already owns.


def _wrap_stats(key: str, inner):
    """Wrap a kernel callable to collect per-kernel timing + call count.

    The stats entry is looked up by key on every call rather than captured
    at wrap-time so that `reset_stats()` (which clears `_KERNEL_STATS`)
    correctly repopulates entries for already-wrapped kernels — otherwise
    cached kernels would write to a dangling dict that's no longer in
    `_KERNEL_STATS` and `get_stats()` would always report empty.

    M11.8: captures grid (wg_x, wg_y, wg_z) from dispatch args, and
    harvests VGPR / LDS / descriptor_count from reflection metadata.
    """

    def stats_kernel(*args):
        entry = _KERNEL_STATS.get(key)
        if entry is None:
            entry = {
                "call_count": 0,
                "total_us": 0.0,
                "last_args_len": 0,
                # M11.8: static metadata populated on first dispatch
                "wg_x": 0,
                "wg_y": 0,
                "wg_z": 0,
                "vgprs": 0,
                "lds_bytes": 0,
                "descriptor_count": 0,
            }
            _KERNEL_STATS[key] = entry
            # M11.8: seed static metadata from reflection on first touch
            _seed_kernel_meta(key, entry)
        t0 = time.perf_counter()
        inner(*args)
        entry["total_us"] += (time.perf_counter() - t0) * 1e6
        entry["call_count"] += 1
        entry["last_args_len"] = len(args)
        # M11.8: capture grid from dispatch args (last 3 positional args
        # are always wg_x, wg_y, wg_z in the Vulkan dispatch ABI).
        if len(args) >= 3:
            try:
                entry["wg_x"] = int(args[-3])
                entry["wg_y"] = int(args[-2])
                entry["wg_z"] = int(args[-1])
            except (TypeError, ValueError):
                pass

    return stats_kernel


def _seed_kernel_meta(key: str, entry: dict) -> None:
    """M11.8: Populate static per-kernel metadata from reflection cache.

    Harvests VGPR count, shared-memory bytes, and descriptor count from
    the SPIR-V reflection metrics cached during compilation.
    """
    from .slangc import _reflection_metrics_by_hash, _reflection_metrics_by_key

    spv_hash = _KERNEL_SPIRV_HASH.get(key, "")
    if spv_hash:
        refl = _reflection_metrics_by_hash.get(spv_hash)
        if refl is None:
            refl = _reflection_metrics_by_key.get(key)
        if refl is not None:
            entry["vgprs"] = int(refl.get("vgprs", 0) or 0)
            entry["lds_bytes"] = int(refl.get("shared_mem", 0) or 0)
            entry["descriptor_count"] = int(refl.get("descriptor_count", 0) or 0)


# ── Pre-resolved pybind entries ────────────────────────────────────────

# Pre-resolved pybind entries — avoids a package import on every dispatch.
_jit_dispatch = None
_jit_dispatch_cached = None
_jit_dispatch_cached_nopc = None
_jit_dispatch_indexed = None
_jit_dispatch_indexed_cached = None
_jit_dispatch_indexed_cached_nopc = None
_jit_pipeline = None
_descriptor_indexing_probe: Optional[bool] = None


def _get_jit_dispatch():
    global _jit_dispatch
    if _jit_dispatch is None:
        from torch_vulkan import _C as _c

        _jit_dispatch = _c._jit_dispatch
    return _jit_dispatch


def _get_jit_dispatch_indexed():
    """Resolve the descriptor-array variant of `_jit_dispatch` (N+1.5).

    Returns the C++ pybind entry that accepts a per-binding
    ``descriptor_counts: vector<uint32_t>``. Auto-falls-back to the flat
    path inside the C++ runtime when every count == 1, so callers may use
    it unconditionally — but Python codegen still prefers the cheaper
    flat ``_jit_dispatch`` when all counts are 1, to skip the extra
    pybind conversion on the hot path.
    """
    global _jit_dispatch_indexed
    if _jit_dispatch_indexed is None:
        from torch_vulkan import _C as _c

        _jit_dispatch_indexed = getattr(_c, "_jit_dispatch_indexed", None)
    return _jit_dispatch_indexed


def _descriptor_indexing_supported() -> bool:
    """Cached probe of `VK_EXT_descriptor_indexing` availability.

    Returns ``True`` when the C++ runtime reports the extension is
    enabled and the FFI shim is present. False on older builds (no
    ``_descriptor_indexing_enabled`` symbol) or when the device driver
    rejected the extension.
    """
    global _descriptor_indexing_probe
    if _descriptor_indexing_probe is not None:
        return _descriptor_indexing_probe
    try:
        from torch_vulkan import _C as _c

        probe = getattr(_c, "_descriptor_indexing_enabled", None)
        _descriptor_indexing_probe = bool(probe()) if probe is not None else False
    except Exception:
        _descriptor_indexing_probe = False
    return _descriptor_indexing_probe


def _get_jit_dispatch_cached():
    """Resolve the cached-pipeline fast-path dispatches (no key lookup per call).

    Returns (dispatch_with_pc, dispatch_no_pc, get_pipeline,
             dispatch_indexed_with_pc, dispatch_indexed_no_pc).
    Generated kernels pick the no-pc entry when n_pc=0 (the common pointwise
    case), avoiding a pybind bytes conversion on every dispatch.

    M9.5: indexed cached variants support descriptor-array kernels with
    pre-computed descriptor_counts — saves the per-dispatch key lookup.
    """
    global _jit_dispatch_cached, _jit_dispatch_cached_nopc, _jit_pipeline
    global _jit_dispatch_indexed_cached, _jit_dispatch_indexed_cached_nopc
    if _jit_dispatch_cached is None:
        from torch_vulkan import _C as _c

        _jit_dispatch_cached = _c._jit_dispatch_cached
        _jit_dispatch_cached_nopc = _c._jit_dispatch_cached_nopc
        _jit_pipeline = _c._jit_pipeline
        # M9.5: indexed cached variants (descriptor-array path).
        _jit_dispatch_indexed_cached = getattr(_c, "_jit_dispatch_indexed_cached", None)
        _jit_dispatch_indexed_cached_nopc = getattr(
            _c, "_jit_dispatch_indexed_cached_nopc", None
        )
    return (
        _jit_dispatch_cached,
        _jit_dispatch_cached_nopc,
        _jit_pipeline,
        _jit_dispatch_indexed_cached,
        _jit_dispatch_indexed_cached_nopc,
    )


# ── Validation ──────────────────────────────────────────────────────────


def _validate_no_null_storage(key: str, tensors: list[torch.Tensor]) -> bool:
    """PF.51 — fail-fast guard against null-storage (FakeTensor) leakage.

    A vulkan-tagged null-storage tensor (PF.13's ``make_vulkan_null``) reaches
    this layer only if an FX pass or wrapper-codegen step propagated a
    FakeTensor through to dispatch. The C++ layer would otherwise raise
    ``RuntimeError: Tensor has no backing Vulkan buffer`` with no
    indication of which arg was responsible. We surface the offender by
    name (and dispatch key) so the bug roots straight to the producing
    pipeline stage instead of to the runtime.
    """
    offenders: list[str] = []
    fake_count = 0
    vulkan_count = 0
    for i, t in enumerate(tensors):
        if t is None:
            continue
        # ``has_storage`` returns True for vulkan-null tensors (PF.13's
        # invariant — they carry a real Storage with a null DataPtr).
        # ``data_ptr() == 0`` is the canonical null-storage signal.
        try:
            dev_type = t.device.type
        except Exception:  # noqa: BLE001
            continue
        if dev_type not in ("vulkan", "privateuseone"):
            continue
        vulkan_count += 1
        try:
            ptr = t.data_ptr()
        except RuntimeError:
            # ``data_ptr`` raises on FakeTensor — tracing mode.
            fake_count += 1
            offenders.append(
                f"arg{i}: <FakeTensor> shape={list(t.shape)} dtype={t.dtype}"
            )
            continue
        if ptr == 0:
            # Zero-element tensors are intentional placeholders (e.g., dummy
            # c0/c_last slots for non-LSTM cells in the fused RNN kernel) —
            # not FakeTensor leakage.  Skip them rather than raising PF.51.
            if t.numel() == 0:
                continue
            offenders.append(
                f"arg{i}: shape={list(t.shape)} dtype={t.dtype} device={t.device}"
            )
    # If ALL Vulkan tensors are FakeTensors, we are in AOT Autograd tracing.
    # Skip the dispatch — outputs already have correct shapes.
    if fake_count > 0 and fake_count == vulkan_count:
        return True
    if offenders:
        raise RuntimeError(
            f"PF.51: vulkan-null-storage tensor reached dispatch '{key}' — "
            f"an FX pass or wrapper-codegen step is propagating a "
            f"FakeTensor through to the runtime. Offenders:\n  "
            + "\n  ".join(offenders)
        )
    return False


# ── Dispatch entries ────────────────────────────────────────────────────


def dispatch(
    key: str,
    spirv: bytes,
    tensors: list[torch.Tensor],
    wg_x: int,
    wg_y: int = 1,
    wg_z: int = 1,
    push_constants: bytes = b"",
    num_outputs: int = 1,
    spec_constants: list[tuple[int, int]] | None = None,
) -> None:
    """Dispatch a pre-compiled SPIR-V compute shader on the Vulkan stream.

    Thin wrapper around the C++ `_jit_dispatch` pybind entry. Requires that
    every tensor is on the vulkan device and already contiguous.

    CG.M15: ``spec_constants`` are (constant_id, value) pairs that override
    ``[[vk::constant_id]]`` defaults at pipeline-creation time.
    """
    from .slangc import _TRACE

    if _TRACE:
        import sys

        print(
            f"[vk-jit] key={key} tensors={len(tensors)} wg=({wg_x},{wg_y},{wg_z}) "
            f"pc_bytes={len(push_constants)} num_out={num_outputs}"
            f" spec={len(spec_constants) if spec_constants else 0}",
            file=sys.stderr,
            flush=True,
        )
    if _validate_no_null_storage(key, tensors):
        return  # tracing mode, skip actual dispatch
    _sc = list(spec_constants) if spec_constants else []
    _get_jit_dispatch()(
        key, spirv, tensors, wg_x, wg_y, wg_z, push_constants, num_outputs, _sc
    )


def dispatch_indexed(
    key: str,
    spirv: bytes,
    tensors: list[torch.Tensor],
    descriptor_counts: list[int],
    wg_x: int,
    wg_y: int = 1,
    wg_z: int = 1,
    push_constants: bytes = b"",
    num_outputs: int = 1,
) -> None:
    """N+1.5.a — descriptor-array variant of :func:`dispatch`.

    Routes through the C++ ``_jit_dispatch_indexed`` pybind entry, which
    writes Vulkan descriptor sets with ``descriptorCount`` taken from
    ``descriptor_counts`` (parallel to the binding order). When every
    count is ``1`` the C++ runtime auto-falls-back to the flat path —
    callers may still want to use :func:`dispatch` directly for that
    case to skip the extra pybind list conversion.

    Raises ``RuntimeError`` when any count > 1 but
    ``VK_EXT_descriptor_indexing`` is unavailable on the device.
    """
    if any(c > 1 for c in descriptor_counts):
        if not _descriptor_indexing_supported():
            raise RuntimeError(
                f"N+1.5: dispatch '{key}' uses a descriptor-array binding "
                f"(descriptor_counts={list(descriptor_counts)}) but the "
                f"Vulkan runtime reports VK_EXT_descriptor_indexing is "
                f"unavailable on this device."
            )
    indexed_fn = _get_jit_dispatch_indexed()
    if indexed_fn is None:
        raise RuntimeError(
            f"N+1.5: `_jit_dispatch_indexed` FFI symbol not present. "
            f"Rebuild the C++ extension."
        )
    from .slangc import _TRACE

    if _TRACE:
        import sys

        print(
            f"[vk-jit-idx] key={key} tensors={len(tensors)} "
            f"counts={list(descriptor_counts)} "
            f"wg=({wg_x},{wg_y},{wg_z}) pc_bytes={len(push_constants)} "
            f"num_out={num_outputs}",
            file=sys.stderr,
            flush=True,
        )
    if _validate_no_null_storage(key, tensors):
        return  # tracing mode, skip actual dispatch
    indexed_fn(
        key,
        spirv,
        tensors,
        list(int(c) for c in descriptor_counts),
        wg_x,
        wg_y,
        wg_z,
        push_constants,
        num_outputs,
    )


def compile_and_dispatch(
    src: str,
    tensors: list[torch.Tensor],
    wg_x: int,
    wg_y: int = 1,
    wg_z: int = 1,
    push_constants: bytes = b"",
    num_outputs: int = 1,
    entry: str = "computeMain",
    cache_key: str = "",
    spec_constants: list[tuple[int, int]] | None = None,
) -> None:
    """Compile Slang source (cached) and dispatch in one call.

    `cache_key` is required (used as both the SPIR-V cache key and the
    pipeline cache key). All in-tree callers supply one — the previous
    SHA1-of-SPIRV fallback was dead code.

    CG.M15: ``spec_constants`` are (constant_id, value) pairs for
    ``[[vk::constant_id]]`` overrides at pipeline-creation time.
    """
    if not cache_key:
        raise ValueError("compile_and_dispatch requires a non-empty cache_key")
    from .slangc import compile_slang_to_spirv

    spv = compile_slang_to_spirv(src, entry=entry, cache_key=cache_key)
    dispatch(
        cache_key,
        spv,
        tensors,
        wg_x,
        wg_y,
        wg_z,
        push_constants,
        num_outputs,
        spec_constants=spec_constants,
    )


# ── Kernel wrapper factories ────────────────────────────────────────────


def make_vulkan_kernel(
    src: str,
    key: str,
    n_buffers: int | None,
    pc_size_bytes: int,
    n_pc: int,
    n_outputs: int = 1,
    config_key: str | None = None,
):
    """Build a generated-kernel wrapper that dispatches via _jit_dispatch.

    Uses the raw dispatch path (not cached) because pipeline pre-creation
    requires passing SPIR-V to _jit_pipeline which the closure doesn't store.

    N+1.5.a: when slangc reflection reports any binding with
    ``descriptorCount > 1`` (i.e. an array binding such as
    ``RWStructuredBuffer<T> arr[N]``), the closure routes through the
    descriptor-array FFI ``_jit_dispatch_indexed`` so each array slot
    receives its own buffer. Flat layouts (every count == 1) keep the
    original ``_jit_dispatch`` fast-path to avoid the extra pybind
    list-conversion per call.

    DR.3: ``config_key`` is threaded through to the SPIR-V compilation
    path so harvested reflection metrics are cross-indexed under this
    structural key, enabling ``_get_actual_vgprs`` to find cached data
    on subsequent compiles.
    """
    import struct

    from .reflection import (
        _get_reflected_descriptor_counts_from_src,
        get_reflected_binding_count,
        get_reflected_descriptor_counts,
    )
    from .slangc import compile_slang_to_spirv

    # M9.4: Pre-allocate a bytearray per kernel for push constants.
    # struct.pack_into writes into the pre-allocated buffer in-place,
    # avoiding a Python bytes allocation on every dispatch.
    if n_pc:
        _pc_buf = bytearray(n_pc * 4)
        _pc_pack_into = struct.Struct(f"{n_pc}I").pack_into
    else:
        _pc_buf = None
        _pc_pack_into = None
    spv = compile_slang_to_spirv(src, cache_key=key, config_key=config_key)
    _KERNEL_SPIRV_HASH[key] = hashlib.sha256(spv).hexdigest()[:12]

    # ── D.3: Reflection-based buffer count ──
    # When n_buffers is None (or zero), derive the buffer count from
    # SPIR-V reflection so variable-arity kernels don't need a
    # hand-counted binding count.  The closure slices tensors from
    # ``args[:-(3+n_pc)]`` regardless, but callers can use this for
    # pre-dispatch validation or AOTI metadata.
    _n_buf = n_buffers
    if _n_buf is None or _n_buf == 0:
        _n_buf = get_reflected_binding_count(spv)
    # Cross-validate when both hand-count and reflection are available.
    if _n_buf is not None and _n_buf > 0 and n_buffers is not None and n_buffers > 0:
        if _n_buf != n_buffers:
            import warnings

            warnings.warn(
                f"D.3: kernel '{key}' hand-counted n_buffers={n_buffers} "
                f"but reflection reports {_n_buf} bindings. "
                f"Using reflection value."
            )
    # ── N+1.5.a: pick the dispatch FFI based on reflection ──
    # Reflection JSON is keyed by source-hash (see compile_slang_to_spirv);
    # the SPV-hash lookup will miss, so try source-hash first.
    descriptor_counts = _get_reflected_descriptor_counts_from_src(src)
    if descriptor_counts is None:
        descriptor_counts = get_reflected_descriptor_counts(spv)
    needs_indexed = bool(descriptor_counts) and any(c > 1 for c in descriptor_counts)
    if needs_indexed:
        if not _descriptor_indexing_supported():
            raise RuntimeError(
                f"N+1.5: kernel '{key}' uses a descriptor-array binding "
                f"(descriptor_counts={descriptor_counts}) but the Vulkan "
                f"runtime reports VK_EXT_descriptor_indexing is unavailable. "
                f"Either rebuild against a driver that exposes the extension, "
                f"or rewrite the shader to use only flat (descriptorCount=1) "
                f"bindings."
            )
        indexed_fn = _get_jit_dispatch_indexed()
        if indexed_fn is None:
            raise RuntimeError(
                f"N+1.5: kernel '{key}' needs `_jit_dispatch_indexed` but "
                f"the FFI symbol is missing. Rebuild the C++ extension."
            )
        # Freeze a tuple of uint32 for the closure (pybind picks up the
        # implicit conversion from list-of-int).
        dc = tuple(int(c) for c in descriptor_counts)
        if n_pc == 0:

            def kernel(*args):
                indexed_fn(
                    key,
                    spv,
                    list(args[:-3]),
                    dc,
                    args[-3],
                    args[-2],
                    args[-1],
                    b"",
                    n_outputs,
                )
        else:

            def kernel(*args):
                _pc_pack_into(_pc_buf, 0, *args[-(3 + n_pc) + 3 : -3])
                indexed_fn(
                    key,
                    spv,
                    list(args[: -(3 + n_pc)]),
                    dc,
                    args[-(3 + n_pc)],
                    args[-(3 + n_pc) + 1],
                    args[-(3 + n_pc) + 2],
                    bytes(_pc_buf),
                    n_outputs,
                )

        stats_enabled = os.environ.get("TORCH_VULKAN_INDUCTOR_STATS") == "1"
        return kernel if not stats_enabled else _wrap_stats(key, kernel)

    # ── flat path: every binding has descriptorCount == 1 ──
    dispatch_fn = _get_jit_dispatch()

    # Build closure that captures key + spv + pack function
    if n_pc == 0:

        def kernel(*args):
            dispatch_fn(
                key, spv, list(args[:-3]), args[-3], args[-2], args[-1], b"", n_outputs
            )

        stats_enabled = os.environ.get("TORCH_VULKAN_INDUCTOR_STATS") == "1"
        return kernel if not stats_enabled else _wrap_stats(key, kernel)

    # With push constants. The wrapper (`kernel/header.py:call_kernel`)
    # emits ordered_args as ``[bufs..., sizevars..., dyn_numels..., wg_x,
    # wg_y, wg_z]`` — sizevars / dynamic numels come BEFORE the wg dims.
    # D.2.a (2026-05-09): the original slice math `args[-(3+n_pc):-(3+n_pc)+3]`
    # mistakenly treated PCs as if they were appended after the wg dims,
    # so for n_pc>0 the wg dims were read from the PC region and the PC
    # bytes came back empty (`args[-n_pc:-3]` is empty when n_pc<=3).
    # Fix: wg dims always live in the last 3 slots; PCs occupy the
    # n_pc slots immediately before.
    def kernel(*args):
        _pc_pack_into(_pc_buf, 0, *args[-(3 + n_pc) : -3])
        dispatch_fn(
            key,
            spv,
            list(args[: -(3 + n_pc)]),
            args[-3],
            args[-2],
            args[-1],
            bytes(_pc_buf),
            n_outputs,
        )

    stats_enabled = os.environ.get("TORCH_VULKAN_INDUCTOR_STATS") == "1"
    return kernel if not stats_enabled else _wrap_stats(key, kernel)


def make_vulkan_kernel_via_aoti(
    src: str,
    key: str,
    n_buffers: int | None,
    pc_size_bytes: int,
    n_pc: int,
    n_outputs: int = 1,
):
    """PF.31 — same contract as ``make_vulkan_kernel`` but the dispatch
    closure routes through the C++ AOTI runtime ABI
    (``_aoti_make_kernel`` + ``_aoti_dispatch``) instead of the pybind
    JIT-pipeline path. The Python interpreter is still present to call the
    pybind wrappers — but the body of the closure is a single ABI call,
    matching what the AOTI-emitted C++ wrapper will do once PF.32 ships
    the SPV next to the `.so`. Used by the regression test that asserts
    a Python-free dispatch path.
    """
    import struct

    from torch_vulkan import _C as _c

    from .reflection import (
        _get_reflected_buffer_count_from_cache_key,
        get_reflected_binding_count,
    )
    from .slangc import compile_slang_to_spirv

    # M9.4: Pre-allocate bytearray for push constants in AOTI path.
    if n_pc:
        _pc_buf = bytearray(n_pc * 4)
        _pc_pack_into = struct.Struct(f"{n_pc}I").pack_into
    else:
        _pc_buf = None
        _pc_pack_into = None
    spv = compile_slang_to_spirv(src, cache_key=key)
    _KERNEL_SPIRV_HASH[key] = hashlib.sha256(spv).hexdigest()[:12]
    # D.3: When n_buffers is None, derive from SPIR-V reflection.
    _nb_aoti = n_buffers
    if _nb_aoti is None:
        _nb_aoti = get_reflected_binding_count(spv)
        if _nb_aoti is None:
            _nb_aoti = _get_reflected_buffer_count_from_cache_key(src)
        if _nb_aoti is None:
            raise RuntimeError(
                "D.3: Cannot determine buffer count for kernel"
                f" '{key}' - reflection unavailable."
            )

    handle = _c._aoti_make_kernel(spv, key, _nb_aoti, pc_size_bytes)
    _no = n_outputs

    if n_pc == 0:

        def kernel(*args):
            tensors = list(args[:-3])
            _c._aoti_dispatch(handle, tensors, args[-3], args[-2], args[-1], b"", _no)
    else:
        pc_start = -3 - n_pc

        def kernel(*args):
            tensors = list(args[:pc_start])
            _pc_pack_into(_pc_buf, 0, *args[pc_start:-3])
            _c._aoti_dispatch(
                handle, tensors, args[-3], args[-2], args[-1], bytes(_pc_buf), _no
            )

    kernel._aoti_handle = handle  # keep handle alive with the closure
    stats_enabled = os.environ.get("TORCH_VULKAN_INDUCTOR_STATS") == "1"
    if not stats_enabled:
        return kernel
    wrapped = _wrap_stats(key, kernel)
    wrapped._aoti_handle = handle
    return wrapped


# ── AOTI model export ───────────────────────────────────────────────────


def export_aoti_model(
    model: "torch.nn.Module",
    path: str,
    example_inputs: "tuple | None" = None,
) -> None:
    """Export a compiled model for AOTI deployment.

    Serializes compiled SPIR-V binaries, kernel metadata, buffer layouts,
    and dispatch order into a directory for later loading via
    ``_aoti_model_load``. The output is a directory containing:
        kernels.bin  — binary bundle of kernel SPIR-V + metadata
        metadata.json — human-readable dispatch order and buffer layouts

    The AOTI runtime can load this directory and execute all kernels
    without requiring the Python Inductor stack or slangc at runtime.

    Args:
        model: A ``torch.nn.Module`` compiled with the Vulkan Inductor backend.
        path: Output directory path. Created if it does not exist.
        example_inputs: Optional example inputs used to trigger tracing
                        if the model has not been pre-compiled.
    """
    import json
    import struct

    from torch_vulkan import _C as _c

    from .reflection import (
        _get_reflected_buffer_count_from_cache_key,
        get_reflected_binding_count,
    )
    from .slangc import _disk_cache_read, compile_slang_to_spirv

    os.makedirs(path, exist_ok=True)

    # Collect all compiled kernels referenced by the model's generated code.
    # The Inductor wrapper calls make_vulkan_kernel which populates
    # _KERNEL_SPIRV_HASH. We walk the compile cache entries for each key.
    kernels: "list[dict]" = []
    seen_keys: "set[str]" = set()

    for key, spv_hash in _KERNEL_SPIRV_HASH.items():
        if key in seen_keys:
            continue
        seen_keys.add(key)

        # Try to get SPIR-V from disk cache or in-memory compile cache
        spv = None
        # Disk cache first (always available if previously compiled)
        spv = _disk_cache_read(key)
        if spv is None:
            # Re-compile if not cached
            sc = getattr(_get_jit_dispatch, "_source_cache", None)
            if sc is not None and key in sc:
                src = sc[key]
                spv = compile_slang_to_spirv(src, cache_key=key)
        if spv is None:
            raise RuntimeError(
                f"export_aoti_model: cannot find SPIR-V for kernel '{key}'. "
                f"Run the model forward at least once to compile kernels."
            )

        # Determine n_buffers from SPIR-V reflection
        n_buf = get_reflected_binding_count(spv)
        if n_buf is None:
            n_buf = _get_reflected_buffer_count_from_cache_key("") or 0

        kernels.append(
            {
                "key": key,
                "spv": spv,
                "n_buffers": n_buf,
                "pc_size_bytes": 0,
                "spv_hash": spv_hash,
            }
        )

    if not kernels:
        raise RuntimeError(
            "export_aoti_model: no compiled kernels found. "
            "Run the model forward at least once to compile kernels."
        )

    # Write kernels.bin
    bin_path = os.path.join(path, "kernels.bin")
    with open(bin_path, "wb") as f:
        f.write(b"vk_aoti\n")
        f.write(struct.pack("<I", len(kernels)))
        for k in kernels:
            spv = k["spv"]
            spv_words = len(spv) // 4
            key_bytes = k["key"].encode("utf-8")
            f.write(
                struct.pack(
                    "<IIII",
                    spv_words,
                    k["n_buffers"],
                    k["pc_size_bytes"],
                    len(key_bytes),
                )
            )
            f.write(key_bytes)
            f.write(spv)

    # Write metadata.json
    meta_path = os.path.join(path, "metadata.json")
    meta = {
        "version": 1,
        "kernel_count": len(kernels),
        "kernels": [
            {
                "key": k["key"],
                "spv_hash": k["spv_hash"],
                "n_buffers": k["n_buffers"],
                "spv_size_bytes": len(k["spv"]),
            }
            for k in kernels
        ],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Verify: load via C++ AOTI runtime, then free
    model_handle = _c._aoti_model_load(path)
    try:
        import sys

        total_spv = sum(len(k["spv"]) for k in kernels)
        print(
            f"[vk-aoti] export → {path}  kernels={len(kernels)}  "
            f"spv_total={total_spv / 1024:.1f} KiB",
            file=sys.stderr,
            flush=True,
        )
    finally:
        _c._aoti_model_free(model_handle)
