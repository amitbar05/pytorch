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
    _pick_register_tile_configs,
    _pick_tile_configs,
    _slang_tiles_enabled,
)
from .render import _render_mm_int8_slang, _render_mm_slang

# Module-level globals for install state
_installed = False
_bmm_installed = False
_addmm_installed = False


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

    if _slang_tiles_enabled():
        for tm, tn, tk in tiles:
            inductor_config.external_matmul.append(
                _make_tile_mm_fn(tm, tn, tk, num_stages=1)
            )
            inductor_config.external_matmul.append(
                _make_tile_mm_fn(tm, tn, tk, num_stages=2)
            )
        for tm, tn, tk, mpt, npt in reg_tiles:
            inductor_config.external_matmul.append(
                _make_tile_mm_fn(
                    tm, tn, tk, num_stages=1, m_per_thread=mpt, n_per_thread=npt
                )
            )
            inductor_config.external_matmul.append(
                _make_tile_mm_fn(
                    tm, tn, tk, num_stages=2, m_per_thread=mpt, n_per_thread=npt
                )
            )


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
                choices.append(
                    ExternKernelChoice(fn).bind(kernel_inputs.nodes(), layout_)
                )

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
            choices.append(ExternKernelChoice(fn).bind(kernel_inputs.nodes(), layout_))

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
                    f"slang_mm_{tm}_{tn}_{tk}_s{ns}_r{mpt}x{npt}_{dt}_n111",
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
                    f"slang_addmm_{tm}_{tn}_{tk}_s{ns}_r{mpt}x{npt}_{dt}_n111",
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
                    f"slang_addmm_epi_OpGELU_{tm}_{tn}_{tk}_s{ns}_r{mpt}x{npt}_{dt}_n111",
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
                f"slang_bmm_v2_{tm}_{tn}_{tk}_r{mpt}x{npt}_{dt}_n111",
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
_INT8_TILE_CONFIGS: list[tuple[int, int, int, int, int]] = [
    (32, 32, 16, 4, 4),  # 256 threads, 16 outputs/thread
    (16, 64, 16, 2, 4),  # 128 threads, 8 outputs/thread
    (64, 16, 16, 4, 2),  # 128 threads, 8 outputs/thread
    (16, 16, 16, 2, 2),  # 64 threads, 4 outputs/thread
]


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

    for tm, tn, tk, mpt, npt in _INT8_TILE_CONFIGS:
        inductor_config.external_matmul.append(
            _make_tile_mm_int8_fn(tm, tn, tk, m_per_thread=mpt, n_per_thread=npt)
        )


def _collect_int8_matmul_prewarm_specs() -> list[tuple[str, str]]:
    """Render the (cache_key, slang_src) pairs for int8 mm tile configs.

    Returns specs suitable for passing to the SPIR-V prewarm cache.
    """
    specs: list[tuple[str, str]] = []
    for tm, tn, tk, mpt, npt in _INT8_TILE_CONFIGS:
        cache_key = f"slang_mm_int8_{tm}_{tn}_{tk}_r{mpt}x{npt}_n111"
        src = _render_mm_int8_slang(tm, tn, tk, m_per_thread=mpt, n_per_thread=npt)
        specs.append((cache_key, src))
    return specs
