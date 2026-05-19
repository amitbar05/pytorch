"""GEMM template installation and prewarming.

Registers Slang template matmul callables as Inductor external_matmul choices
and patches Inductor's tuned_addmm / tuned_bmm lowerings for Vulkan.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

from ....vulkan_template import (
    _MM_REGISTER_TILE_CONFIGS,
    _MM_TILE_CONFIGS,
)
from .classes import (
    _make_tile_addmm_fn,
    _make_tile_bmm_fn,
    _make_tile_mm_fn,
    _make_tile_mm_int8_fn,
)
from .dispatch import (
    _get_device_subgroup_size,
    _pick_register_tile_configs,
    _pick_tile_configs,
    _slang_tiles_enabled,
)
from .render import _render_mm_int8_slang, _render_mm_slang

# Module-level globals for install state
_installed = False
_bmm_installed = False
_addmm_installed = False

# M-NEW.1: ``ExternKernelChoice.__init__`` asserts ``not hasattr(
# extern_kernels, name)`` — the upstream API doesn't expose a public
# lookup method in PyTorch 2.11. Our tile callables have deterministic
# names (e.g. ``slang_addmm_8_8_8_s1_r1x1``), so the second ``aten.addmm``
# / ``aten.bmm`` lowered in the same compile would re-enter the
# constructor and crash with ``AssertionError: duplicate extern
# kernel: ...``. We cache the constructed singletons here keyed by
# ``fn.__name__``; the per-compile lowering looks them up first and
# only constructs once per (template, tile-config) combination.
#
# This is the dominant compile-mode blocker per Audit Agent 3
# (2026-05-18): every model with ≥2 ``Linear`` modules — MLP, ViT,
# Transformer block, Llama-MLP, Mixtral-MoE — hits it.
_EXTERN_CHOICE_CACHE: dict = {}


def _ensure_extern_choices(tile_fns, extern_kernel_choice_cls) -> None:
    """Pre-construct + cache ``ExternKernelChoice`` for each tile fn.

    Called at install time from ``install_external_bmm`` /
    ``install_external_addmm`` so the deterministic ``__name__``s are
    registered on ``torch._inductor.select_algorithm.extern_kernels``
    exactly once per process — including the path where a cached
    Inductor wrapper short-circuits ``_vulkan_tuned_addmm`` on a
    subsequent compile (the wrapper references
    ``extern_kernels.slang_addmm_*`` directly without going through
    our lowering).

    M19.1-followup (2026-05-20): also pre-populate upstream's
    ``torch._inductor.kernel.mm.lazy_register_extern_choice`` cache.
    Upstream's ``tuned_mm`` iterates ``inductor_config.external_matmul``
    and calls ``lazy_register_extern_choice(fn)`` for each entry. Without
    pre-population that triggers ``ExternKernelChoice(fn)`` — which fails
    the ``not hasattr(extern_kernels, name)`` assertion because we
    already registered the name here. Pre-poking the upstream cache
    with the SAME ExternKernelChoice instance avoids that path.
    """
    # Resolve the upstream ``extern_kernels`` namespace so we can also
    # short-circuit when the name is already registered there (e.g.
    # ``install_external_*`` re-entered after a partial init, or when
    # the same fn name appears across both bmm and addmm tile lists).
    from torch._inductor.select_algorithm import extern_kernels

    # Best-effort: also poke upstream's ``lazy_register_extern_choice``
    # process-wide cache. The function is ``@functools.cache``-decorated
    # — we directly populate its ``__wrapped__`` cache via ``cache_info``
    # / a re-implementation of ``cache_clear`` is not possible, but the
    # functools cache exposes ``__wrapped__`` and shares object identity
    # by ``functools.cache`` semantics. The cleanest pre-population path
    # is to call the function ourselves with the same fn — once
    # constructed, subsequent upstream calls hit the cache.
    try:
        from torch._inductor.kernel.mm import lazy_register_extern_choice as _upstream_lz
    except Exception:  # noqa: BLE001
        _upstream_lz = None

    for fn in tile_fns:
        if fn.__name__ in _EXTERN_CHOICE_CACHE:
            continue
        if hasattr(extern_kernels, fn.__name__):
            # Some other path constructed the choice already (rare).
            # Skip rather than re-trigger the duplicate assertion.
            continue
        # If upstream's lazy_register_extern_choice is available, use it
        # so the SAME instance ends up in BOTH caches (ours +
        # upstream's). Otherwise fall back to constructing directly.
        if _upstream_lz is not None:
            choice = _upstream_lz(fn)
        else:
            choice = extern_kernel_choice_cls(fn)
        _EXTERN_CHOICE_CACHE[fn.__name__] = choice


def install_external_mm() -> None:
    """Register Slang template mm callables as external matmul choices.

    Appends to ``torch._inductor.config.external_matmul`` so that Inductor's
    ``tuned_mm`` lowering benchmarks our Slang tiled-matmul templates
    alongside ``torch.mm`` (C++ Vulkan backend) and picks the fastest.

    For each tile config, registers both a single-stage (NUM_STAGES=1) and
    a double-buffered (NUM_STAGES=2) variant.  The pipelined variant overlaps
    shared-memory loads with compute, which helps on memory-bound problems.

    Safe to call multiple times — only installs once.
    """
    global _installed
    if _installed:
        return
    _installed = True

    tiles = _pick_tile_configs()
    reg_tiles = _pick_register_tile_configs()
    if not tiles and not reg_tiles:
        return

    from torch._inductor import config as inductor_config
    from torch._inductor.select_algorithm import ExternKernelChoice

    if _slang_tiles_enabled():
        mm_tile_fns: list = []
        for tm, tn, tk in tiles:
            mm_tile_fns.append(_make_tile_mm_fn(tm, tn, tk, num_stages=1))
            mm_tile_fns.append(_make_tile_mm_fn(tm, tn, tk, num_stages=2))
        for tm, tn, tk, mpt, npt in reg_tiles:
            mm_tile_fns.append(
                _make_tile_mm_fn(
                    tm, tn, tk, num_stages=1, m_per_thread=mpt, n_per_thread=npt
                )
            )
            mm_tile_fns.append(
                _make_tile_mm_fn(
                    tm, tn, tk, num_stages=2, m_per_thread=mpt, n_per_thread=npt
                )
            )
        # M-NEW.1: pre-construct ExternKernelChoices at install time so
        # the deterministic ``__name__``s are registered on
        # ``extern_kernels`` exactly once per process.  The upstream
        # ``lazy_register_extern_choice`` has ``@functools.cache`` which
        # catches repeat calls within one process, but pre-constructing
        # here provides defense-in-depth against codecache serialization
        # creating new callable objects.
        _ensure_extern_choices(mm_tile_fns, ExternKernelChoice)
        inductor_config.external_matmul.extend(mm_tile_fns)


def install_external_bmm() -> None:
    """Patch Inductor's tuned_bmm to include Slang tiled-bmm choices for Vulkan.

    Overrides the ``aten.bmm`` lowering with a Vulkan-aware version that adds
    our tiled Slang BMM callables as ExternKernelChoice options alongside the
    default ``aten_bmm`` (which dispatches to our C++ vulkan_bmm).  Inductor's
    autotuner benchmarks both and picks the faster one.

    Safe to call multiple times — only patches once.
    """
    global _bmm_installed
    if _bmm_installed:
        return
    _bmm_installed = True

    tiles = _pick_tile_configs()
    reg_tiles = _pick_register_tile_configs()
    if not tiles and not reg_tiles:
        return

    # Force registration of tuned_bmm before we capture + replace it.
    import torch._inductor.kernel.bmm as _bmm_mod  # noqa: F401
    from torch._inductor import lowering as _L
    from torch._inductor.kernel.mm_common import _is_static_problem, mm_args
    from torch._inductor.select_algorithm import (
        ExternKernelChoice,
        autotune_select_algorithm,
    )
    from torch._inductor.utils import use_aten_gemm_kernels

    aten = torch.ops.aten

    # Capture original after the forced import so lowerings[aten.bmm] is set.
    _orig_bmm = _L.lowerings.get(aten.bmm.default) or _L.lowerings.get(aten.bmm)

    bmm_tile_fns: list = []
    if _slang_tiles_enabled():
        bmm_tile_fns = [_make_tile_bmm_fn(tm, tn, tk) for tm, tn, tk in tiles]
        bmm_tile_fns += [
            _make_tile_bmm_fn(tm, tn, tk, m_per_thread=mpt, n_per_thread=npt)
            for tm, tn, tk, mpt, npt in reg_tiles
        ]
        # M-NEW.1: pre-construct ExternKernelChoices at install time so
        # the deterministic ``__name__``s are registered on
        # ``extern_kernels`` exactly once per process — even if a cached
        # Inductor wrapper short-circuits ``_vulkan_tuned_bmm`` on a
        # subsequent compile (the wrapper references
        # ``extern_kernels.slang_bmm_*`` directly).
        _ensure_extern_choices(bmm_tile_fns, ExternKernelChoice)

    @_L.register_lowering(aten.bmm, type_promotion_kind=None)
    def _vulkan_tuned_bmm(mat1, mat2, out_dtype=None, *, layout=None):
        if mat1.get_device().type != "vulkan":
            if _orig_bmm is not None:
                return _orig_bmm(mat1, mat2, out_dtype=out_dtype, layout=layout)
            return torch.bmm(mat1, mat2)

        from torch._inductor.kernel.bmm import aten_bmm
        from torch._inductor.kernel_inputs import MMKernelInputs

        m, n, k, layout_, mat1_, mat2_ = mm_args(
            mat1, mat2, layout=layout, out_dtype=out_dtype
        )
        kernel_inputs = MMKernelInputs([mat1_, mat2_], out_dtype=out_dtype)
        choices = []

        if use_aten_gemm_kernels():
            # M17.5: Only include aten_bmm as fallback when Slang tiles
            # are not available. For Vulkan fp32, Slang tiles are
            # preferred (single dispatch vs eager C++).
            if not bmm_tile_fns:
                choices.append(aten_bmm.bind(kernel_inputs.nodes(), layout_))

        if out_dtype is None:
            for fn in bmm_tile_fns:
                # M-NEW.1: choices are constructed once at install time
                # via ``_ensure_extern_choices``. Re-lookup so subsequent
                # ``aten.bmm`` lowerings reuse the singleton.
                choice = _EXTERN_CHOICE_CACHE.get(fn.__name__)
                if choice is None:
                    # Defensive: install was skipped (slang tiles
                    # disabled at install time, then re-enabled?) — fall
                    # back to constructing here so we don't KeyError.
                    choice = ExternKernelChoice(fn)
                    _EXTERN_CHOICE_CACHE[fn.__name__] = choice
                choices.append(choice.bind(kernel_inputs.nodes(), layout_))

        if not choices:
            choices.append(aten_bmm.bind(kernel_inputs.nodes(), layout_))

        # ``autotune_select_algorithm`` returns ``(node, autotune_info)`` under
        # normal conditions, but can return a bare TensorBox in certain config
        # combinations (e.g. when ``return_multi_template`` is True and
        # max_autotune/max_autotune_gemm are active — the MultiTemplateBuffer
        # path). Handle both cases so we always return a valid IR node.
        result = autotune_select_algorithm(
            "bmm", choices, kernel_inputs.nodes(), layout_
        )
        if isinstance(result, tuple):
            node, _ = result
        else:
            node = result
        return node


def install_external_addmm() -> None:
    """Patch Inductor's tuned_addmm to use fused Slang tiled addmm+bias for Vulkan.

    The fused shader computes ``out = a @ b + bias`` in a single dispatch,
    replacing the C++ backend's two-dispatch path (mm dispatch + bias-add dispatch).
    Only applied for Vulkan f32 tensors with alpha=beta=1.

    Safe to call multiple times — only patches once.
    """
    global _addmm_installed
    if _addmm_installed:
        return
    _addmm_installed = True

    tiles = _pick_tile_configs()
    if not tiles:
        return

    # Force registration of tuned_addmm before we capture + replace it.
    import torch._inductor.kernel.mm as _mm_mod  # noqa: F401
    from torch._inductor import lowering as _L
    from torch._inductor.kernel.mm_common import _is_static_problem, mm_args
    from torch._inductor.select_algorithm import (
        ExternKernelChoice,
        autotune_select_algorithm,
    )
    from torch._inductor.utils import use_aten_gemm_kernels

    aten = torch.ops.aten

    _orig_addmm = _L.lowerings.get(aten.addmm.default) or _L.lowerings.get(aten.addmm)

    reg_tiles = _pick_register_tile_configs()
    addmm_tile_fns: list = []
    if _slang_tiles_enabled():
        addmm_tile_fns = (
            [_make_tile_addmm_fn(tm, tn, tk, num_stages=1) for tm, tn, tk in tiles]
            + [_make_tile_addmm_fn(tm, tn, tk, num_stages=2) for tm, tn, tk in tiles]
            + [
                _make_tile_addmm_fn(
                    tm, tn, tk, num_stages=1, m_per_thread=mpt, n_per_thread=npt
                )
                for tm, tn, tk, mpt, npt in reg_tiles
            ]
            + [
                _make_tile_addmm_fn(
                    tm, tn, tk, num_stages=2, m_per_thread=mpt, n_per_thread=npt
                )
                for tm, tn, tk, mpt, npt in reg_tiles
            ]
        )
        # M-NEW.1: pre-construct ExternKernelChoices at install time so
        # the deterministic ``__name__``s are registered on
        # ``extern_kernels`` exactly once per process — even if a cached
        # Inductor wrapper short-circuits ``_vulkan_tuned_addmm`` on a
        # subsequent compile.
        _ensure_extern_choices(addmm_tile_fns, ExternKernelChoice)

    @_L.register_lowering(aten.addmm.default, type_promotion_kind=None)
    def _vulkan_tuned_addmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
        if (
            inp.get_device().type != "vulkan"
            or alpha != 1
            or beta != 1
            or mat1.get_dtype() not in (torch.float32, torch.float16)
        ):
            if _orig_addmm is not None:
                return _orig_addmm(
                    inp, mat1, mat2, alpha=alpha, beta=beta, layout=layout
                )
            return torch.addmm(inp, mat1, mat2, alpha=alpha, beta=beta)

        from torch._inductor.kernel.mm import aten_addmm
        from torch._inductor.kernel_inputs import MMKernelInputs

        # Use mm_args without the bias to avoid forced expansion to [M,N].
        # Our Slang tiled addmm reads bias as a 1-D vector; torch.addmm handles
        # broadcast internally so aten_addmm gets the original inp shape too.
        m, n, k, layout_, mat1_, mat2_ = mm_args(mat1, mat2, layout=layout)[:6]
        kernel_inputs = MMKernelInputs(
            [inp, mat1_, mat2_], scalars=dict(alpha=alpha, beta=beta)
        )
        choices = []

        if use_aten_gemm_kernels():
            # M17.5: Only include aten_addmm as fallback when Slang tiles
            # are not available. For Vulkan fp32, Slang tiles are
            # preferred (single fused dispatch vs 2 eager C++ dispatches).
            if not addmm_tile_fns:
                choices.append(aten_addmm.bind(kernel_inputs.nodes(), layout_))

        for fn in addmm_tile_fns:
            # M-NEW.1: choices are constructed once at install time via
            # ``_ensure_extern_choices``. Re-lookup so subsequent
            # ``aten.addmm`` lowerings reuse the singleton.
            choice = _EXTERN_CHOICE_CACHE.get(fn.__name__)
            if choice is None:
                # Defensive fallback (see bmm site above).
                choice = ExternKernelChoice(fn)
                _EXTERN_CHOICE_CACHE[fn.__name__] = choice
            choices.append(choice.bind(kernel_inputs.nodes(), layout_))

        if not choices:
            choices.append(aten_addmm.bind(kernel_inputs.nodes(), layout_))

        # ``autotune_select_algorithm`` returns ``(node, autotune_info)`` under
        # normal conditions, but can return a bare TensorBox in certain config
        # combinations (e.g. when ``return_multi_template`` is True and
        # max_autotune/max_autotune_gemm are active — the MultiTemplateBuffer
        # path). Handle both cases so we always return a valid IR node.
        # Returning the raw tuple makes Inductor's ``validate_ir`` walk into
        # the ``ExternKernelCaller`` payload and reject it as a non-supported
        # top-level IR node, which surfaces as
        # ``LoweringException: Found ExternKernelCaller`` for every
        # ``aten.addmm`` lowered through this path (every ``nn.Linear``).
        result = autotune_select_algorithm(
            "addmm", choices, kernel_inputs.nodes(), layout_
        )
        if isinstance(result, tuple):
            node, _ = result
        else:
            node = result
        return node


# Cache-key strings here MUST match the ones produced by `_slang_tile_mm`,
# `_slang_tile_addmm`, and `_slang_tile_bmm` above. If those formats change,
# update both sides.
def _collect_matmul_prewarm_specs() -> list[tuple[str, str]]:
    """Render the (cache_key, slang_src) pairs the runtime picks for the most
    common matmul shapes. Used to pre-populate the SPIR-V cache so the first
    user dispatch never blocks on a cold slangc invocation.
    """
    specs: list[tuple[str, str]] = []
    dtypes = ("float", "half")

    def _emit(tm, tn, tk, mpt, npt, dt):
        for ns in (1, 2):
            # N+1.11: _n111 prevents stale cache hits with old PC layout.
            specs.append(
                (
                    f"slang_mm_{tm}_{tn}_{tk}_s{ns}_r{mpt}x{npt}_{dt}_n111_a6",
                    _render_mm_slang(
                        tm,
                        tn,
                        tk,
                        dtype_a=dt,
                        dtype_b=dt,
                        dtype_c=dt,
                        dtype_acc="float",
                        num_stages=ns,
                        m_per_thread=mpt,
                        n_per_thread=npt,
                    ),
                )
            )
            specs.append(
                (
                    f"slang_addmm_{tm}_{tn}_{tk}_s{ns}_r{mpt}x{npt}_{dt}_n111_a6",
                    _render_mm_slang(
                        tm,
                        tn,
                        tk,
                        dtype_a=dt,
                        dtype_b=dt,
                        dtype_c=dt,
                        dtype_acc="float",
                        dtype_bias=dt,
                        num_stages=ns,
                        has_bias=True,
                        m_per_thread=mpt,
                        n_per_thread=npt,
                    ),
                )
            )
            specs.append(
                (
                    f"slang_addmm_epi_OpGELU_{tm}_{tn}_{tk}_s{ns}_r{mpt}x{npt}_{dt}_n111_a6",
                    _render_mm_slang(
                        tm,
                        tn,
                        tk,
                        dtype_a=dt,
                        dtype_b=dt,
                        dtype_c=dt,
                        dtype_acc="float",
                        dtype_bias=dt,
                        num_stages=ns,
                        has_bias=True,
                        epilogue_struct="OpGELU",
                        m_per_thread=mpt,
                        n_per_thread=npt,
                    ),
                )
            )
        specs.append(
            (
                f"slang_bmm_v2_{tm}_{tn}_{tk}_r{mpt}x{npt}_{dt}_n111_a6",
                _render_mm_slang(
                    tm,
                    tn,
                    tk,
                    dtype_a=dt,
                    dtype_b=dt,
                    dtype_c=dt,
                    dtype_acc="float",
                    m_per_thread=mpt,
                    n_per_thread=npt,
                    has_batch=True,
                ),
            )
        )

    for tm, tn, tk in _MM_TILE_CONFIGS:
        for dt in dtypes:
            _emit(tm, tn, tk, 1, 1, dt)
    for tm, tn, tk, mpt, npt in _MM_REGISTER_TILE_CONFIGS:
        for dt in dtypes:
            _emit(tm, tn, tk, mpt, npt, dt)
    return specs


def prewarm_matmul_templates(*, sync: bool = False) -> int:
    """Submit the standard mm/addmm/bmm × {f32, f16} × {1, 2 stages} tile
    configs to the slangc thread pool so the first user dispatch hits a
    populated SPIR-V cache. No-op when slangc is not available or when
    Slang tiles are disabled (``TORCH_VULKAN_DISABLE_SLANG_TILES=1``).

    N+1.6: When ``TORCH_VULKAN_ASYNC_COMPILE=1`` (the default), uses
    ``_compile_slang_batch_parallel`` with ``ThreadPoolExecutor`` to compile
    all missing tile configs in parallel.  Falls back to the legacy
    ``prewarm_compile`` fire-and-forget path when async compilation is
    disabled.

    Returns the number of source variants actually scheduled — entries that
    are already in memory or on disk are skipped.
    """
    if os.environ.get("TORCH_VULKAN_NO_PREWARM") == "1":
        return 0
    if not _slang_tiles_enabled():
        return 0
    from ....runtime import (
        _ASYNC_COMPILE,
        _compile_slang_batch_parallel,
        _slangc_available,
        prewarm_compile,
    )

    if not _slangc_available():
        return 0

    specs = _collect_matmul_prewarm_specs()

    # N+1.6: When async compilation is enabled, use the batch parallel path
    # which compiles all cache-miss specs in a dedicated ThreadPoolExecutor.
    # Falls back to prewarm_compile (fire-and-forget via global pool) when
    # async is explicitly disabled.
    if _ASYNC_COMPILE and sync:
        # Synchronous batch path: compile all missing configs in parallel
        # and block until complete.  Used by tests that need populated caches.
        try:
            _compile_slang_batch_parallel(specs)
        except RuntimeError:
            pass  # best-effort: individual failures are non-fatal
        return len(specs)

    if _ASYNC_COMPILE:
        # Fire-and-forget via batch parallel in a background daemon thread.
        import threading

        def _bg_prewarm():
            try:
                _compile_slang_batch_parallel(specs)
            except RuntimeError:
                pass

        t = threading.Thread(target=_bg_prewarm, daemon=True)
        t.start()
        return len(specs)

    return prewarm_compile(specs, sync=sync)


# ═══════════════════════════════════════════════════════════════════════════
# OP.24 — Int8 matmul installation
# ═══════════════════════════════════════════════════════════════════════════

_int8_installed = False

# Conservative tile configs for int8 — smaller tiles to keep LDS footprint
# manageable (int32 groupshared = 4× the bytes of float16/float32 for same
# element count). Register-tiling is used to amortize the unpack overhead.
#
# M-NEW.3.b: workgroup-thread comment is ``(WG_M*WG_N) threads`` where
# ``WG_M = TILE_M // M_PER_THREAD`` and ``WG_N = TILE_N // N_PER_THREAD``
# — same formula as the fp register-tile path (``_pick_register_tile_
# configs`` in dispatch.py). All entries below produce wave-aligned
# (multiple of 64) workgroups for RDNA1; the filter in
# ``install_external_mm_int8`` enforces this invariant for future
# additions.
_INT8_TILE_CONFIGS: list[tuple[int, int, int, int, int]] = [
    (32, 32, 16, 4, 4),  # WG=8*8=64 threads, 16 outputs/thread
    (16, 64, 16, 2, 4),  # WG=8*16=128 threads, 8 outputs/thread
    (64, 16, 16, 4, 2),  # WG=16*8=128 threads, 8 outputs/thread
    (16, 16, 16, 2, 2),  # WG=8*8=64 threads, 4 outputs/thread
]


def _int8_config_wg_threads(c: tuple[int, int, int, int, int]) -> int:
    """Compute the workgroup thread count for an int8 tile config.

    The int8 mm wrapper (``render._render_mm_int8_slang``) emits
    ``[numthreads(wg_n, wg_m, 1)]`` where::

        wg_m = tile_m // m_per_thread
        wg_n = tile_n // n_per_thread

    So the workgroup thread product is ``wg_m * wg_n``.
    """
    tile_m, tile_n, _tile_k, mpt, npt = c
    return (tile_m // mpt) * (tile_n // npt)


def _filter_int8_configs_wave_aligned(
    configs: list[tuple[int, int, int, int, int]],
    *,
    subgroup_size: int | None = None,
) -> list[tuple[int, int, int, int, int]]:
    """Drop int8 tile configs whose workgroup isn't wave-aligned.

    Mirrors the M-NEW.3 filter on the fp register-tile picker in
    ``_pick_register_tile_configs``: any kernel whose ``numthreads``
    product isn't a multiple of the device's subgroup size is rejected
    by the in-process M27 validator and wastes a slangc cold compile.

    On wave64 (RDNA1) we additionally enforce M17.1's "single-wave only"
    cap — the multi-wave ``GroupMemoryBarrierWithGroupSync()`` is broken
    in slangc 2026.5.2 + RADV, so wg-thread-count > 64 on RDNA1 is
    rejected too.

    The argument is taken as-is rather than reading
    ``_INT8_TILE_CONFIGS`` so callers (tests) can pass synthetic inputs.
    """
    sgs = subgroup_size if subgroup_size is not None else _get_device_subgroup_size()
    out: list[tuple[int, int, int, int, int]] = []
    for c in configs:
        n = _int8_config_wg_threads(c)
        if n <= 0:
            continue
        if n % sgs != 0:
            continue
        if sgs == 64 and n > sgs:
            # RDNA1 barrier bug — must stay single-wave.
            continue
        out.append(c)
    return out


def install_external_mm_int8() -> None:
    """Register Slang template int8 mm callables as external matmul choices.

    OP.24: Appends int8 matmul callables to
    ``torch._inductor.config.external_matmul`` so that Inductor's
    ``tuned_mm`` lowering can benchmark our Slang tiled-int8 templates
    alongside the CPU fallback path.

    Only active when Slang tiles are enabled (``TORCH_VULKAN_DISABLE_SLANG_TILES``
    is not set to ``1`` — same gate as other Slang tile matmul variants).

    Safe to call multiple times — only installs once.
    """
    global _int8_installed
    if _int8_installed:
        return
    _int8_installed = True

    if not _slang_tiles_enabled():
        return

    from torch._inductor import config as inductor_config
    from torch._inductor.select_algorithm import ExternKernelChoice

    # M-NEW.3.b: drop sub-wave / wave-misaligned configs so the M27
    # validator doesn't reject a slangc cold compile per such config.
    filtered = _filter_int8_configs_wave_aligned(_INT8_TILE_CONFIGS)
    int8_tile_fns: list = []
    for tm, tn, tk, mpt, npt in filtered:
        int8_tile_fns.append(
            _make_tile_mm_int8_fn(tm, tn, tk, m_per_thread=mpt, n_per_thread=npt)
        )
    # M-NEW.1: pre-construct ExternKernelChoices at install time
    # (same pattern as install_external_mm / bmm / addmm).
    _ensure_extern_choices(int8_tile_fns, ExternKernelChoice)
    inductor_config.external_matmul.extend(int8_tile_fns)


def _collect_int8_matmul_prewarm_specs() -> list[tuple[str, str]]:
    """Render the (cache_key, slang_src) pairs for int8 mm tile configs.

    Returns specs suitable for passing to the SPIR-V prewarm cache.

    M-NEW.3.b: matches the install path's wave-alignment filter so the
    prewarm cache only contains configs the autotuner will actually try.
    """
    specs: list[tuple[str, str]] = []
    filtered = _filter_int8_configs_wave_aligned(_INT8_TILE_CONFIGS)
    for tm, tn, tk, mpt, npt in filtered:
        cache_key = f"slang_mm_int8_{tm}_{tn}_{tk}_r{mpt}x{npt}_n111"
        src = _render_mm_int8_slang(tm, tn, tk, m_per_thread=mpt, n_per_thread=npt)
        specs.append((cache_key, src))
    return specs
