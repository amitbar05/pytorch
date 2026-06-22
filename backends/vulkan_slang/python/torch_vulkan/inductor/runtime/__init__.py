"""Vulkan/Slang Inductor runtime package.

Re-exports all public symbols from submodules so existing imports from
``torch_vulkan.inductor.runtime`` continue to work unchanged.

Submodules:
  - ``slangc``:     slangc compilation, SPIR-V caching, reflection harvesting
  - ``reflection``: descriptor counts, binding reflection
  - ``dispatch``:   JIT dispatch, pipeline creation, kernel wrapping
  - ``batcher``:    DispatchBatcher for batch submission
  - ``profile``:    dispatch profiling and timing
"""

# ── Re-export from each submodule ───────────────────────────────────────

# slangc (largest module — compilation, caching, shader libs)
from .slangc import (  # noqa: F401
    SlangCompileTimeout,
    _ASYNC_COMPILE,
    _ASYNC_MAX_WORKERS,
    _ASYNC_POOL,
    _COMPILE_STATS,
    _DISK_CACHE_DIR,
    _INDUCTOR_STATS,
    _LT_SPEC_CONST_RE,
    _MAX_SLANGC_WORKERS,
    _NUMTHREADS_SRC_RE,
    _PARALLEL_COMPILE,
    _SHADERS_LIB_DIR,
    _SHADER_LIB_MODULE_CACHE_DIR,
    _SHADER_LIB_MODULE_STATS,
    _SLANGC,
    _SLANGC_TIMEOUT_S,
    _SLANG_BLANK_LINES_RE,
    _SLANG_BLOCK_COMMENT_RE,
    _SLANG_LINE_COMMENT_RE,
    _SLANG_TRAILING_WS_RE,
    _SPV_BASELINES_PATH,
    _TRACE,
    _analyze_slang_source_for_loop_depth,
    _analyze_spirv_binary,
    _build_specialize_const_args,
    _cache_by_hash,
    _cache_by_key,
    _cache_lock,
    _check_spv_regression,
    _compile_slang_batch_parallel,
    _compile_slang_to_spirv_inner,
    _default_max_workers,
    _device_subgroup_size_tag,
    _disk_cache_read,
    _disk_cache_write,
    _disk_metrics_path,
    _disk_metrics_read,
    _disk_metrics_write,
    _disk_reflection_path,
    _disk_reflection_read,
    _disk_reflection_write,
    _ensure_mm_tile_module,
    _ensure_shader_lib_modules,
    _extract_linktime_spec_constants,
    _get_async_pool,
    _resolve_slangc,
    _get_device_subgroup_size_tag,
    _harvest_reflection_metrics,
    _in_flight,
    _in_flight_lock,
    _invalidate_shader_lib_modules,
    _is_in_pool_worker,
    _load_spv_baselines,
    _mm_tile_module_available,
    _mm_tile_module_path,
    _normalize_slang_source,
    _parse_reflection_metrics,
    _pick_numthreads_from_reflection,
    _pool_local,
    _reflection_cache,
    _reflection_metrics_by_hash,
    _reflection_metrics_by_key,
    _reset_shader_lib_modules_ready,
    _reset_slangc_available_cache,
    _save_spv_baselines,
    _shader_lib_modules_lock,
    _shader_lib_sources,
    _slangc_available,
    _slangc_available_cache,
    _slangc_fingerprint,
    _slangc_fingerprint_cache,
    _slangc_modules_available,
    _slangc_supports_modules,
    _spv_baselines,
    _spv_baselines_loaded,
    _wrap_pool_worker,
    batch_compile_slang_to_spirv,
    cache_hit_rate,
    compile_slang_to_spirv,
    compile_slang_to_spirv_with_reflection,
    gc_spirv_cache,
    get_cached_metrics_for_key,
    get_reflection_metrics,
    parse_spec_constants,
    precompile_shader_libs,
    prewarm_compile,
    prewarm_shader_libs,
    reset_compile_stats,
    reset_reflection_baselines,
)

# reflection
from .reflection import (  # noqa: F401
    _binding_descriptor_count,
    _get_reflected_buffer_count_from_cache_key,
    _get_reflected_descriptor_counts_from_src,
    get_reflected_binding_count,
    get_reflected_descriptor_counts,
    get_reflection_json,
    reflection_layout,
)

# dispatch
from .dispatch import (  # noqa: F401
    _KERNEL_SPIRV_HASH,
    _KERNEL_STATS,
    _descriptor_indexing_probe,
    _descriptor_indexing_supported,
    _get_jit_dispatch,
    _get_jit_dispatch_cached,
    _get_jit_dispatch_indexed,
    _jit_dispatch,
    _jit_dispatch_cached,
    _jit_dispatch_cached_nopc,
    _jit_dispatch_indexed,
    _jit_dispatch_indexed_cached,
    _jit_dispatch_indexed_cached_nopc,
    _jit_pipeline,
    _seed_kernel_meta,
    _validate_no_null_storage,
    _wrap_stats,
    compile_and_dispatch,
    dispatch,
    dispatch_indexed,
    export_aoti_model,
    make_vulkan_kernel,
    make_vulkan_kernel_via_aoti,
)

# profile
from .profile import (  # noqa: F401
    _DISPATCH_TIMES,
    _DISPATCH_TIMES_MAX_SAMPLES,
    _record_dispatch_time,
    _reset_dispatch_times,
    dispatch_times,
)

# batcher
from .batcher import DispatchBatcher  # noqa: F401

# validation-as-codegen-check (M21.2)
from .validation_codegen import (  # noqa: F401
    ValidationResult,
    get_codegen_validation_mode,
    handle_validation_result,
    is_codegen_validation_enabled,
    validate_codegen_dispatch,
    validate_kernel_source,
)


# ── Cross-cutting helper (touches multiple submodules) ──────────────────


def reset_per_test_caches() -> None:
    """GAP 7.3 / PF.27.a.2 — clear in-memory caches that leak across tests.

    Called by the ``conftest.py`` per-test fixture so that dispatch-count
    and cold-compile-budget tests see deterministic cache state regardless
    of test ordering within a single xdist worker. Does *not* clear the
    on-disk SPIR-V cache (that's intentionally persistent across sessions).

    PF.27.a.2 extension (2026-05-02): also resets shader-lib module
    readiness, reflection cache, and async pool — globals that previously
    leaked worker-id-dependent state across tests within an xdist worker.

    PF.27.b/c (2026-05-29): also resets Philox RNG state so each test
    starts with fresh seed derivation and offset.
    """
    from .dispatch import _KERNEL_SPIRV_HASH
    from .profile import _DISPATCH_TIMES
    from .slangc import (
        _SHADER_LIB_MODULE_STATS,
        _cache_by_hash,
        _cache_by_key,
        _reflection_cache,
        reset_compile_stats,
    )

    _cache_by_key.clear()
    _cache_by_hash.clear()
    _KERNEL_SPIRV_HASH.clear()
    _reflection_cache.clear()
    _SHADER_LIB_MODULE_STATS["compiles"] = 0
    _SHADER_LIB_MODULE_STATS["cache_hits"] = 0
    reset_compile_stats()
    # _shader_lib_modules_ready is NOT reset here — the shader-lib .slang-module
    # files are on disk and persist across tests.  Tests that need a clean
    # shader-lib state call _reset_shader_lib_modules_ready() explicitly.
    # (M22a Stage 2 previously reset it here, which forced a full 16-module
    # recompile before every test that calls slangc, causing test timeouts.)
    _DISPATCH_TIMES.clear()

    # PF.27.b/c: reset Philox RNG state so each test starts fresh.
    try:
        from ..philox_state import reset_philox_state

        reset_philox_state()
    except Exception:
        pass


def __getattr__(name: str):
    """Dynamic lookup for mutable state that lives in submodules (M22a).

    `_shader_lib_modules_ready` is a bool in shader_lib that changes at
    runtime; a static `from .slangc import ...` binding would freeze the
    value at import time.  Python module `__getattr__` (PEP 562) intercepts
    attribute misses on the module object so tests that read
    `rt._shader_lib_modules_ready` see the live value.
    """
    if name == "_shader_lib_modules_ready":
        import torch_vulkan.inductor.runtime.shader_lib as _sl

        return _sl._shader_lib_modules_ready
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
