"""Inductor backend kill-switches, read from environment variables."""

import os
from typing import Optional

_NO_WG_TUNE = os.environ.get("TORCH_VULKAN_NO_WG_TUNE") == "1"
_NO_LOAD_HOIST = os.environ.get("TORCH_VULKAN_NO_LOAD_HOIST") == "1"
_NO_PACKED16 = os.environ.get("TORCH_VULKAN_NO_PACKED16") == "1"
_NO_VEC4_POINTWISE = os.environ.get("TORCH_VULKAN_NO_VEC4_POINTWISE") == "1"
_NO_EXTERN_EPILOGUE = os.environ.get("TORCH_VULKAN_NO_EXTERN_EPILOGUE") == "1"

# T5.2 — Positive preference for packed16 path (f16/bf16 pointwise).
# Default True: the codegen prefers the packed-uint16 path over the
# widen-compute-narrow fallback. Set TORCH_VULKAN_NO_PACKED16=1 to
# disable (no_packed16() returns True, prefer_packed16() returns False).
# These two are logically coupled: the kill-switch is the master off.
_prefer_packed16 = os.environ.get("TORCH_VULKAN_NO_PACKED16") != "1"

# M4 / P5.1 — Register-pressure-aware workgroup size selection.
# When enabled (default), _pick_threadgroup_size considers dtype size,
# welford detection, shared-memory usage, and estimated VGPR pressure
# to maintain ≥2 waves/CU occupancy on RDNA1 (VGPR ≤ 64 for wave64).
# Set TORCH_VULKAN_REGISTER_AWARE_WG=0 to revert to numel-only sizing.
_REGISTER_AWARE_WG = os.environ.get("TORCH_VULKAN_REGISTER_AWARE_WG") != "0"

# P5.6 — denormal flush policy. RDNA1 SPIR-V pipelines flush denormals to
# zero by default in some configurations. Users targeting deterministic
# numerics or training stability where small gradients matter want
# "preserve". The codegen does not yet emit `OpDecorate FPDenormalsPreserve`
# — this knob is recorded so a future kernel.py change can read it.
# Values: "flush" (default, current behavior) or "preserve".
_DENORMALS = os.environ.get("TORCH_VULKAN_DENORMALS", "flush").lower()
if _DENORMALS not in ("flush", "preserve"):
    _DENORMALS = "flush"

# P3.3/M13 — Slang reflection harvesting. When enabled (default 1),
# the slangc -dump-reflection pass runs after SPIR-V compilation to
# extract VGPR count, shared-memory usage, subgroup size, and loop-depth
# metrics. The metrics are cached alongside the SPIR-V and fed into
# workgroup-size selection (M4). Disable with TORCH_VULKAN_REFLECTION=0
# to fall back to heuristic estimates.
_REFLECTION_ENABLED = os.environ.get("TORCH_VULKAN_REFLECTION", "1") != "0"

# P1.1 / D.4 — Dynamic shapes support. When enabled (default 1/enabled),
# range-tree numels that are sympy Symbols (dynamic) are passed through
# the push-constant struct so the shader can adapt at runtime.  When
# disabled, all numels are assumed static and integer casts are used.

# D.4 exit gate (2026-05-09): dynamic-batch MLP fwd+bwd passes with
# correct gradients across B ∈ {1, 4, 16, 64}.  Pointwise dynamic shapes
# work; reductions (D.2.a) also compile.  Still experimental — set
# ``TORCH_VULKAN_DYNAMIC_SHAPES=0`` to disable if stability issues arise.
_DYNAMIC_SHAPES = os.environ.get("TORCH_VULKAN_DYNAMIC_SHAPES", "1") != "0"

# M17 — SPIR-V specialization constants. When enabled (default 1/enabled)
# and all range-tree numels are static, emit ``[[vk::constant_id(N)]]
# const uint xnumel;`` instead of push-constant struct members or
# ``static const uint`` module-scope declarations.  Slangc can then
# constant-fold loops and eliminate dead branches at SPIR-V emission time.
# Set TORCH_VULKAN_SPEC_CONSTANTS=0 to disable.
_SPEC_CONSTANTS = os.environ.get("TORCH_VULKAN_SPEC_CONSTANTS", "1") != "0"

# M22 — Dead code elimination pass. When enabled (default 1),
# the codegen scans generated Slang source, builds a use-def chain of
# CSE variable assignments, and strips assignments whose LHS is never
# referenced by any live computation or output store. This eliminates
# unused loads and computations that survive upstream DeferredLine
# pruning. Set TORCH_VULKAN_DCE=0 to disable for debugging.
_DCE_ENABLED = os.environ.get("TORCH_VULKAN_DCE", "1") != "0"

# P3.1 / M9 / CG.M14 — Slang ParameterBlock<KernelArgs> emission.
# When enabled (default 1), buffer bindings are declared as fields of a
# ``struct KernelArgs`` wrapped in ``ParameterBlock<KernelArgs> args;``
# instead of manual ``[[vk::binding(N)]]`` annotations. Slang
# auto-assigns binding indices; the SPIR-V reflection JSON contains
# the canonical mapping.  This unlocks auto-derived VkPipelineLayouts
# and future reflection improvements (M11 occupancy-aware codegen).
# Set ``TORCH_VULKAN_PARAMETER_BLOCK=0`` to disable.
_PARAMETER_BLOCK = os.environ.get("TORCH_VULKAN_PARAMETER_BLOCK", "1") == "1"

# DR.6 — Bank-conflict padding for groupshared arrays.
# RDNA1 LDS has 32 banks.  Without padding, strided access patterns
# (reduction tree stride, tile loads) land multiple threads on the same
# bank serializing access.  Adding +32 elements shifts the bank mapping
# for elements past offset 32, eliminating bank conflicts.
# Gate: TORCH_VULKAN_BANK_CONFLICT_PAD=1 enables +32 padding (default: 1
# on GPU, 0 on Lavapipe where LDS is smaller and has no bank conflicts).
_BANK_CONFLICT_PAD = os.environ.get("TORCH_VULKAN_BANK_CONFLICT_PAD", "1") == "1"

# DR.7 — Compile-time reflection routing.
# When enabled, the two-pass compilation flow peeks at SPIR-V reflection
# after Pass 1, adjusts numthreads based on VGPR count, and re-compiles
# with optimized workgroup sizing.  Default OFF until validated across
# all Vulkan devices.
# Set TORCH_VULKAN_REFLECTION_ROUTING=0 to disable.
_REFLECTION_ROUTING = os.environ.get("TORCH_VULKAN_REFLECTION_ROUTING", "1") != "0"

# DR.6 — Vec4/packed16 vectorization audit.
# When enabled (default 0), the codegen counts total loads/stores and
# how many were successfully vectorized to vec4/packed16 form, then
# logs a hit-rate summary at DEBUG level and stores metrics in inductor
# stats for later querying.
# Set TORCH_VULKAN_VEC4_AUDIT=1 to enable.
_VEC4_AUDIT = os.environ.get("TORCH_VULKAN_VEC4_AUDIT", "0") == "1"


# P3.2 / M14 — Link-time specialization for matmul tiles.
# When enabled, the tiled matmul body lives in a precompiled
# ``mm_tile.slang-module`` that is compiled ONCE per dtype.  Per-tile-config
# wrappers import it and resolve tile sizes at link time via
# ``extern static const int`` definitions — no per-config re-parse of the
# module body.  Collapses 112 slangc invocations (28 tiles x 2 stages x
# 2 dtypes) into 2 module compiles + thin-wrapper links.
# Opt-in (default 0) until stable across all Vulkan devices.
# Set ``TORCH_VULKAN_LINK_TIME_SPEC=1`` to enable.
_LINK_TIME_SPEC = os.environ.get("TORCH_VULKAN_LINK_TIME_SPEC", "0") == "1"

# GPU.1 — Batch dispatch submission. Disabled by default (2026-06-01).
# PERF.1 fix makes BATCH_DISPATCH=1 correct but adds setup/teardown
# overhead without batching benefit (batch mode exits on first flush).
# BATCH_DISPATCH=0 is 1.8x faster (385ms vs 676ms for MNISTNet batch=64).
# Set ``TORCH_VULKAN_BATCH_DISPATCH=1`` to enable (correct but slower).
_BATCH_DISPATCH = os.environ.get("TORCH_VULKAN_BATCH_DISPATCH", "0") != "0"

# GPU.2 — Python wrapper hot-path optimization. When enabled (default ON),
# the wrapper codegen caches attribute lookups at function scope, skips
# assert_size_stride checks for static-shape graphs, pre-allocates reusable
# tensor handle arrays, and uses fast-path pool acquire for the common case.
# Set ``TORCH_VULKAN_WRAPPER_FASTPATH=0`` to disable.
_WRAPPER_FASTPATH = os.environ.get("TORCH_VULKAN_WRAPPER_FASTPATH", "1") != "0"

# GPU.3 — Dispatch profiling instrumentation. When enabled, records
# per-dispatch wall-clock time (ns) and reports min/mean/max/σ via
# ``inductor_stats.dispatch_times()``.  Adds ~2 µs overhead per dispatch.
# Set ``TORCH_VULKAN_PROFILE_DISPATCHES=1`` to enable.
_PROFILE_DISPATCHES = os.environ.get("TORCH_VULKAN_PROFILE_DISPATCHES", "0") == "1"

# GPU.4 — Grid-aware workgroup sizing. When enabled (default ON),
# _pick_threadgroup_size reduces WG size for small grids to spread
# work across more CUs, improving GPU utilization.  For large grids
# the existing VGPR + loop-depth heuristics apply unchanged.
# Set ``TORCH_VULKAN_GRID_AWARE_WG=0`` to disable.
_GRID_AWARE_WG = os.environ.get("TORCH_VULKAN_GRID_AWARE_WG", "1") != "0"

# GPU.5 — Persistent pointwise kernel (micro-batching). When enabled
# (default ON for small numels), chains of small pointwise ops are
# emitted as a single grid-stride-loop kernel instead of N separate
# dispatches, eliminating Python→C++ round-trips between ops.
# Only active for numels < 4096.  Set ``TORCH_VULKAN_PERSISTENT_POINTWISE=0``
# to disable.
_PERSISTENT_POINTWISE = os.environ.get("TORCH_VULKAN_PERSISTENT_POINTWISE", "1") != "0"

# M11.5 — Wave-aligned workgroup size rounding.
# When enabled (default ON), the final WG size from _pick_threadgroup_size
# is rounded UP to the next multiple of the SIMD group size (64 for
# wave64/RDNA1).  Partial waves waste VGPR lanes; aligning eliminates
# the slang_validator.py M27 advisory.  Respects max_wg cap.
# Set ``TORCH_VULKAN_ROUND_WG_TO_WAVE=0`` to disable.
_ROUND_WG_TO_WAVE = os.environ.get("TORCH_VULKAN_ROUND_WG_TO_WAVE", "1") != "0"

# M11.7 — Occupancy gate. When enabled (default ON), the codegen
# estimates GPU occupancy after WG size selection and warns when
# occupancy falls below 50 %.  Set ``TORCH_VULKAN_STRICT_OCCUPANCY=1``
# to promote the warning to a hard error.  Set ``TORCH_VULKAN_OCCUPANCY_GATE=0``
# to disable the gate entirely.
_OCCUPANCY_GATE = os.environ.get("TORCH_VULKAN_OCCUPANCY_GATE", "1") != "0"
_STRICT_OCCUPANCY = os.environ.get("TORCH_VULKAN_STRICT_OCCUPANCY", "0") == "1"

# M11.3 — Register-tile pointwise: each thread processes 2-4 consecutive
# elements instead of 1, reducing thread count and improving ILP.
# Set to 2, 3, or 4 to enable; 0 disables.  Default 2: divisibility gate
# (numel % (wg*2)==0) and VGPR cap guard ineligible shapes safely.
# Opt out: TORCH_VULKAN_REGISTER_TILE=0.
_REGISTER_TILE = int(os.environ.get("TORCH_VULKAN_REGISTER_TILE", "2"))

# N+1.5.c — Descriptor indexing feature gate for >16 storage buffer bindings.
# When enabled (default) and VK_EXT_descriptor_indexing is available on the
# device, the binding cap is raised from ~60 to ≥256, unblocking aggressive
# combo-kernel fusion.  When disabled ("0"), the pre-indexing limit of 16
# effective bindings (60 device-limit minus 4 margin) is enforced.
# Set ``TORCH_VULKAN_DESCRIPTOR_INDEXING=0`` to force-disable.
_DESCRIPTOR_INDEXING = os.environ.get("TORCH_VULKAN_DESCRIPTOR_INDEXING", "1") != "0"

# N+1.5.c — Override for the maximum number of storage buffer bindings per
# kernel.  When unset, the scheduler probes the device limit and applies the
# descriptor-indexing-aware cap.  Set ``TORCH_VULKAN_MAX_STORAGE_BUFS=N`` to
# force a specific limit (useful for debugging fusion heuristics).
# Read lazily at call time via max_storage_bufs_override() so tests can set
# the env var after import.

# N+1.7 — Static shape specialization for fully-static kernels.
# When enabled (default ON) and ALL dimensions and numels are static
# (known sympy.Integer values), the push-constant struct is skipped
# entirely and all numels/sizevars are emitted as ``static const uint``
# module-scope declarations. Slangc can then constant-fold loop bounds
# and eliminate dead branches for zero push-constant updates.
# Set TORCH_VULKAN_STATIC_SPECIALIZATION=0 to disable.
_STATIC_SPECIALIZATION = (
    os.environ.get("TORCH_VULKAN_STATIC_SPECIALIZATION", "1") != "0"
)

# N+1.5.b — Parameter-array path for foreach optimizer template.
# When enabled, ``foreach_optimizer.py.jinja`` emits
# ``ParamSlot params[BATCH_SIZE]`` inside the ParameterBlock, with direct
# ``args.params[param_idx].p[idx]`` indexing in place of the
# ``switch (param_idx)`` cascade.  slangc emits a single binding per
# buffer-family with ``descriptorCount=BATCH_SIZE``; this requires
# (a) VK_EXT_descriptor_indexing on the device (probed by C++ Context),
# (b) ``_C._jit_dispatch_indexed`` Python wiring (N+1.5.a) that extracts
#     per-binding count from reflection and forwards it to the indexed
#     dispatch FFI.
# Opt-in (default 0) until both prerequisites land.  With the flag off,
# the template falls back to the per-buffer-family switch cascade that
# round-3 N+1.10 shipped — the working path through current Python+C++.
# Set ``TORCH_VULKAN_PARAMETER_ARRAY=1`` to enable.
_PARAMETER_ARRAY = os.environ.get("TORCH_VULKAN_PARAMETER_ARRAY", "0") == "1"


def no_wg_tune() -> bool:
    return _NO_WG_TUNE


def no_load_hoist() -> bool:
    return _NO_LOAD_HOIST


def no_packed16() -> bool:
    return _NO_PACKED16


def prefer_packed16() -> bool:
    """Positive preference for the packed16 codegen path.

    When True (default), f16/bf16 pointwise kernels use the packed-uint16
    path (two half values per uint32 word) instead of the
    widen-compute-narrow fallback.  Coupled with ``no_packed16()``:
    ``TORCH_VULKAN_NO_PACKED16=1`` disables both.
    """
    return _prefer_packed16


def no_vec4_pointwise() -> bool:
    """Kill-switch for the f32 contiguous-pointwise vec4 path.

    When False (default), single-axis f32 pointwise kernels with
    `numel % (max_threadgroup_size * 4) == 0` and trivial `xindex`-only
    indexing bind their I/O buffers as `StructuredBuffer<float4>` and
    each thread processes 4 elements per dispatch — quartering the
    workgroup count and halving global memory transactions.
    """
    return _NO_VEC4_POINTWISE


def no_extern_epilogue() -> bool:
    return _NO_EXTERN_EPILOGUE


def register_aware_wg() -> bool:
    """Whether _pick_threadgroup_size uses register-pressure heuristics (M4).

    When True (default), workgroup size keys on dtype, welford, shared-mem
    usage, and VGPR estimates to maintain ≥2 waves/CU occupancy.
    Set ``TORCH_VULKAN_REGISTER_AWARE_WG=0`` to disable.
    """
    return _REGISTER_AWARE_WG


def denormal_mode() -> str:
    """Return ``"flush"`` (default) or ``"preserve"``. P5.6."""
    return _DENORMALS


def dynamic_shapes() -> bool:
    """Whether dynamic-shape codegen is enabled (P1.1 / D.4).

    When True (default), range-tree numels that are sympy Symbols are
    emitted as push-constant struct members so the shader adapts at
    runtime.  D.4 exit gate: dynamic-batch MLP fwd+bwd verified across
    B ∈ {1, 4, 16, 64}.
    Set ``TORCH_VULKAN_DYNAMIC_SHAPES=0`` to disable.
    """
    return _DYNAMIC_SHAPES


def spec_constants() -> bool:
    """Whether SPIR-V specialization constants are used for static numels (M17).

    When True (default) and all range-tree numels are static, emit
    ``[[vk::constant_id(N)]] const uint xnumel;`` instead of push-constant
    struct members or module-scope ``static const uint`` declarations.
    Set ``TORCH_VULKAN_SPEC_CONSTANTS=0`` to disable.
    """
    return _SPEC_CONSTANTS


def reflection_enabled() -> bool:
    """Whether slangc reflection metrics are harvested (P3.3/M13).

    When True (default), the slangc -dump-reflection pass runs after
    SPIR-V compilation. The harvested VGPR count, shared-memory usage,
    subgroup size, and loop-depth feed into workgroup-size selection.
    Set ``TORCH_VULKAN_REFLECTION=0`` to disable and fall back to heuristics.
    """
    return _REFLECTION_ENABLED


def dce_enabled() -> bool:
    """Whether dead code elimination is active (M22).

    When True (default), the codegen scans generated Slang source,
    builds a use-def chain of CSE variable assignments, and strips
    assignments whose LHS is never referenced by any live computation
    or output store. Set ``TORCH_VULKAN_DCE=0`` to disable for debugging.
    """
    return _DCE_ENABLED


def parameter_block() -> bool:
    """Whether ParameterBlock<KernelArgs> emission is active (P3.1/M9).

    When True, buffer bindings use ``ParameterBlock<KernelArgs> args;``
    instead of ``[[vk::binding(N)]]`` annotations. Slang auto-assigns
    binding indices from the struct field order. Set
    ``TORCH_VULKAN_PARAMETER_BLOCK=0`` to disable. Default: on.
    """
    return _PARAMETER_BLOCK


def link_time_spec() -> bool:
    """Whether link-time specialization is active for matmul tiles (P3.2/M14).

    When True, the tiled matmul module (``mm_tile.slang-module``) is
    compiled ONCE per dtype, and per-tile-config wrappers resolve tile
    sizes at link time without re-parsing the module body.  Collapses
    112 slangc invocations into 2 module compiles + thin-wrapper links.
    Set ``TORCH_VULKAN_LINK_TIME_SPEC=1`` to enable.  Default: off.
    """
    return _LINK_TIME_SPEC


def parameter_array() -> bool:
    """Whether the foreach optimizer template uses the parameter-array
    path with descriptor-array bindings (N+1.5.b).

    When True, ``foreach_optimizer.py.jinja`` emits a single
    ``ParamSlot params[BATCH_SIZE]`` field inside the ParameterBlock,
    and the kernel indexes it directly as ``args.params[param_idx].p[i]``
    instead of using a per-buffer-family ``switch (param_idx)`` cascade.
    slangc emits one binding per buffer family with
    ``descriptorCount=BATCH_SIZE``; this requires VK_EXT_descriptor_indexing
    AND the N+1.5.a Python wiring of ``_C._jit_dispatch_indexed``.
    With the flag off (default), the template falls back to the
    switch-cascade layout that the existing flat ``_jit_dispatch`` path
    handles.  Set ``TORCH_VULKAN_PARAMETER_ARRAY=1`` to enable.
    """
    return _PARAMETER_ARRAY


def bank_conflict_pad() -> bool:
    """Whether groupshared arrays get +32 bank-conflict padding (DR.6).

    When True (default), ``groupshared`` array declarations in reduction
    kernels are padded by 32 elements to avoid RDNA1 LDS bank conflicts
    (32 banks, 4-byte bank width).  Disable with
    ``TORCH_VULKAN_BANK_CONFLICT_PAD=0`` for testing or on Lavapipe.
    """
    return _BANK_CONFLICT_PAD


def reflection_routing() -> bool:
    """Whether compile-time reflection routing is enabled (DR.7 / M11.1).

    When True (default), the two-pass compilation flow peeks at SPIR-V
    reflection after Pass 1, adjusts numthreads based on VGPR count, and
    re-compiles with optimized workgroup sizing.  This avoids cold-compile
    autotune cycles; halves cold-compile latency on first dispatch.
    Set ``TORCH_VULKAN_REFLECTION_ROUTING=0`` to disable.
    """
    return _REFLECTION_ROUTING


def vec4_audit_enabled() -> bool:
    """Whether the vec4/packed16 vectorization audit is active (DR.6).

    When True, the codegen counts total loads/stores and how many were
    successfully vectorized to vec4/packed16 form, logs a hit-rate
    summary at DEBUG level, and stores metrics in inductor stats.
    Set ``TORCH_VULKAN_VEC4_AUDIT=1`` to enable.
    """
    return _VEC4_AUDIT


def descriptor_indexing_enabled() -> bool:
    """Whether descriptor indexing is enabled (N+1.5.c).

    When True (default) and VK_EXT_descriptor_indexing is available on the
    device, the binding cap is raised to ≥256, unblocking aggressive
    combo-kernel fusion.  When False, the pre-indexing limit of 16
    effective bindings is enforced.

    Note: even when this returns True, the actual cap still depends on
    the device probe (``_C._descriptor_indexing_enabled()``).  Set
    ``TORCH_VULKAN_DESCRIPTOR_INDEXING=0`` to force-disable.
    """
    return _DESCRIPTOR_INDEXING


def static_specialization() -> bool:
    """Whether fully-static kernels skip push-constants entirely (N+1.7).

    When True (default) and ALL dimensions/numels are static
    (sympy.Integer), the kernel emits ``static const uint`` for all
    sizevars and numels — no push-constant struct, no ``pc.`` reads, no
    push-constant updates in the wrapper.  Slangc constant-folds loop
    bounds and eliminates dead branches.
    Set ``TORCH_VULKAN_STATIC_SPECIALIZATION=0`` to disable.
    """
    return _STATIC_SPECIALIZATION


def batch_dispatch() -> bool:
    """Whether batch dispatch submission is enabled (GPU.1).

    When True (default), the wrapper codegen collects kernel dispatches
    into a DispatchBatcher context manager that submits all dispatches
    in a single Python→C++ call to record them into one Vulkan command
    buffer.  Set ``TORCH_VULKAN_BATCH_DISPATCH=0`` to disable.
    """
    return _BATCH_DISPATCH


def wrapper_fastpath() -> bool:
    """Whether Python wrapper hot-path optimizations are enabled (GPU.2).

    When True (default), the wrapper caches attribute lookups at function
    scope, skips assert_size_stride for static-shape graphs, pre-allocates
    reusable tensor handle arrays, and uses fast-path pool acquire.
    Set ``TORCH_VULKAN_WRAPPER_FASTPATH=0`` to disable.
    """
    return _WRAPPER_FASTPATH


def profile_dispatches() -> bool:
    """Whether per-dispatch profiling is enabled (GPU.3).

    When True, records per-dispatch wall-clock time (ns) and makes it
    available via ``inductor_stats.dispatch_times()``.  Adds ~2 µs
    overhead per dispatch.
    Set ``TORCH_VULKAN_PROFILE_DISPATCHES=1`` to enable.
    """
    return _PROFILE_DISPATCHES


def grid_aware_wg() -> bool:
    """Whether grid-aware workgroup sizing is enabled (GPU.4).

    When True (default), _pick_threadgroup_size reduces WG size for
    small grids so more CUs have work, improving GPU utilization.
    Set ``TORCH_VULKAN_GRID_AWARE_WG=0`` to disable.
    """
    return _GRID_AWARE_WG


def persistent_pointwise() -> bool:
    """Whether persistent pointwise micro-batching is enabled (GPU.5).

    When True (default), chains of small pointwise ops (numel < 4096)
    are emitted as a single grid-stride-loop kernel instead of N
    separate dispatches.  Set ``TORCH_VULKAN_PERSISTENT_POINTWISE=0``
    to disable.
    """
    return _PERSISTENT_POINTWISE


def round_wg_to_wave() -> bool:
    """Whether WG sizes are rounded up to wave-size multiples (M11.5).

    When True (default), the final WG size from _pick_threadgroup_size
    is rounded UP to the next multiple of the SIMD group size (64 for
    wave64/RDNA1), capped at max_wg.  Eliminates the M27 advisory from
    slang_validator.py.  Set ``TORCH_VULKAN_ROUND_WG_TO_WAVE=0`` to disable.
    """
    return _ROUND_WG_TO_WAVE


def occupancy_gate() -> bool:
    """Whether the occupancy gate is active (M11.7).

    When True (default), the codegen estimates GPU occupancy after WG
    size selection and warns via ``trace_structured`` when occupancy
    falls below 50 %.  Set ``TORCH_VULKAN_OCCUPANCY_GATE=0`` to disable.
    """
    return _OCCUPANCY_GATE


def strict_occupancy() -> bool:
    """Whether low-occupancy warnings promote to hard errors (M11.7).

    When True, estimated occupancy below 50 % raises ``RuntimeError``
    instead of logging a warning.  Set ``TORCH_VULKAN_STRICT_OCCUPANCY=1``
    to enable.
    """
    return _STRICT_OCCUPANCY


def register_tile() -> int:
    """Return the register-tile size for pointwise kernels (M11.3).

    Returns 2, 3, or 4 when ``TORCH_VULKAN_REGISTER_TILE`` is set;
    returns 0 (disabled) by default.  When > 0, each thread processes
    this many consecutive elements in scalar pointwise kernels.
    """
    if _REGISTER_TILE in (2, 3, 4):
        return _REGISTER_TILE
    return 0


# DR.1+ — Aggressive fusion scheduling. When enabled (default ON),
# the fusion scheduler relaxes memory thresholds for pointwise-only
# chains, allows reduction+pointwise tail fusion when the reduction
# output has a single consumer, and skips materialization when all
# consumers of a buffer can be fused into the same kernel.
# Set ``TORCH_VULKAN_AGGRESSIVE_FUSION=0`` to disable.
_AGGRESSIVE_FUSION = os.environ.get("TORCH_VULKAN_AGGRESSIVE_FUSION", "1") != "0"


def aggressive_fusion() -> bool:
    """Whether aggressive fusion scheduling is enabled (DR.1+).

    When True (default), the fusion scheduler:
    - Relaxes memory thresholds for pointwise-only chains
    - Allows reduction+pointwise tail fusion for single-consumer reductions
    - Skips materialization when all consumers can be fused into one kernel
    Set ``TORCH_VULKAN_AGGRESSIVE_FUSION=0`` to disable.
    """
    return _AGGRESSIVE_FUSION


# C5 (2026-06-18): gate conv+GN+ReLU forward fusion.  The fused shader
# doesn't store the intermediate conv+bias output, which forces AOTAutograd
# to recompute it during backward (one extra slang_conv2d dispatch).
# Training workloads may prefer separate dispatches to avoid recomputation.
# Set ``TORCH_VULKAN_DISABLE_CONV_GN_FUSION=1`` to disable the fused path.
_DISABLE_CONV_GN_FUSION = os.environ.get("TORCH_VULKAN_DISABLE_CONV_GN_FUSION", "0") != "0"


def disable_conv_gn_fusion() -> bool:
    """Whether conv+GN+ReLU forward fusion is disabled (C5 gate).

    Default False (fusion enabled). Set TORCH_VULKAN_DISABLE_CONV_GN_FUSION=1
    to disable — trades 1 extra forward dispatch for eliminating 1 backward
    recomputation dispatch.
    """
    return _DISABLE_CONV_GN_FUSION


def max_storage_bufs_override() -> Optional[int]:
    """Override for max storage buffer bindings per kernel (N+1.5.c).

    Returns ``None`` when unset (the scheduler probes the device limit).
    When set to an integer via ``TORCH_VULKAN_MAX_STORAGE_BUFS=N``, the
    scheduler uses this value as the binding cap regardless of device
    limits or descriptor-indexing state.

    The env var is read lazily (at call time) so tests can set it after
    import.
    """
    env_val = os.environ.get("TORCH_VULKAN_MAX_STORAGE_BUFS")
    if env_val is None:
        return None
    try:
        return int(env_val)
    except ValueError:
        return None


# M22.3 — FX pattern firing-rate instrumentation.
# When enabled, each FX pattern match increments a per-name counter so
# users can see which rewrites actually fire and how often.  Zero overhead
# in production (default off).
# Set ``TORCH_VULKAN_PATTERN_STATS=1`` to enable.
_PATTERN_STATS = os.environ.get("TORCH_VULKAN_PATTERN_STATS", "0") == "1"


def pattern_stats_enabled() -> bool:
    """Whether FX pattern firing-rate counters are active (M22.3).

    When True, each pattern match via ``FxPatternRegistry.apply_all``
    increments a per-name counter.  Call ``dump_pattern_stats()`` from
    ``fx_passes/post_grad.py`` to emit a sorted table to stderr.
    Set ``TORCH_VULKAN_PATTERN_STATS=1`` to enable.
    """
    return _PATTERN_STATS
