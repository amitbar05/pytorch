"""Vulkan Slang template matmul — ExternKernelChoice integration.

Provides callable functions that render the Slang tiled-matmul Jinja2 template,
compile it to SPIR-V, and dispatch via the Vulkan runtime. These are registered
as ``config.external_matmul`` entries so Inductor's ``tuned_mm`` / ``tuned_bmm``
lowering benchmarks them alongside ``torch.mm`` (C++ Vulkan backend).

The template lives in ``templates/slang_mm.py.jinja`` and supports compile-time
tile size selection (TILE_M, TILE_N, TILE_K) and optional epilogue fusion
(bias, OpReLU, OpGELU, OpSiLU, OpSigmoid, OpTanh, clamp, scale).

Usage::

    from torch_vulkan.inductor.vulkan_template_caller import install_external_mm

    install_external_mm()  # called once from register()
"""

from __future__ import annotations

import os
import struct
from typing import Any, Optional

import torch

from .buffer_pool import pool_acquire, pool_acquire_scratch, pool_release_scratch
from .vulkan_template import (
    _MM_REGISTER_TILE_CONFIGS,
    _MM_TILE_CONFIGS,
    _load_slang_template,
)

_tile_cache: dict[tuple, str] = {}
_installed = False
_bmm_installed = False
_addmm_installed = False
_optimizer_installed = False

# ── T4.1: Validated IPointwise struct names for epilogue dispatch ──────
# Every struct in this set must appear in shaders/lib/pointwise.slang as
# ``public struct OpX : IPointwise { static float apply(float x); };``
#
# CG.M10: The template shader's entry point is ``computeMain<Epilogue : IDifferentiable>``
# where ``Epilogue`` is a Slang type parameter resolved at SPIR-V compile time
# via the ``entry`` parameter of ``compile_and_dispatch``.  The Jinja2 template
# only controls whether the ``Epilogue::apply(...)`` call site is emitted; the
# concrete type name (e.g. ``OpGELU``) is NOT a Jinja2 variable — it is the
# Slang generic type argument.  (Anti-goal #6: no string-based template params.)
_VALID_IPOINTWISE_STRUCTS: frozenset[str] = frozenset(
    {
        "OpIdentity",
        "OpReLU",
        "OpSigmoid",
        "OpTanh",
        "OpGELU",
        "OpSiLU",
        "OpELU",
        "OpHardSwish",
        "OpHardSigmoid",
        "OpMish",
        "OpSoftplus",
        "OpLeakyReLU",
        "OpRelu6",
        "OpAbs",
        "OpNeg",
        "OpExp",
        "OpLog",
        "OpSqrt",
        "OpRsqrt",
        "OpReciprocal",
        "OpCos",
        "OpSin",
        "OpTan",
        "OpAtan",
        "OpCeil",
        "OpFloor",
        "OpRound",
        "OpSign",
        "OpLog2",
        "OpLog10",
        "OpLog1p",
        "OpExp2",
        "OpExpm1",
        "OpAcos",
        "OpAsin",
        "OpCosh",
        "OpSinh",
        "OpAsinh",
        "OpAcosh",
        "OpAtanh",
        "OpTrunc",
        "OpFrac",
        "OpLogicalNot",
        "OpBitwiseNot",
    }
)

# CG.M10 — Subset of IPointwise structs that also implement IDifferentiable.
# Only these can be used as epilogues for the mm template, since the template's
# entry point is ``computeMain<Epilogue : IDifferentiable>``.  Structs like
# OpCeil/OpFloor/OpRound/OpSign/OpTrunc/OpFrac/OpLogicalNot/OpBitwiseNot/OpRelu6
# are IPointwise-only (step/discrete functions are not differentiable).
_VALID_IDIFFERENTIABLE_STRUCTS: frozenset[str] = frozenset(
    {
        "OpIdentity",
        "OpReLU",
        "OpSigmoid",
        "OpTanh",
        "OpGELU",
        "OpSiLU",
        "OpELU",
        "OpHardSwish",
        "OpHardSigmoid",
        "OpMish",
        "OpSoftplus",
        "OpLeakyReLU",
        "OpAbs",
        "OpNeg",
        "OpExp",
        "OpLog",
        "OpSqrt",
        "OpRsqrt",
        "OpReciprocal",
        "OpCos",
        "OpSin",
        "OpTan",
        "OpAtan",
        "OpLog2",
        "OpLog10",
        "OpLog1p",
        "OpExp2",
        "OpExpm1",
        "OpAcos",
        "OpAsin",
        "OpCosh",
        "OpSinh",
        "OpAsinh",
        "OpAcosh",
        "OpAtanh",
    }
)


def _validate_epilogue_struct(name: str | None) -> str | None:
    """Validate and normalise an epilogue struct name.

    CG.M10: Only ``IDifferentiable`` structs are accepted (the mm template
    uses ``<Epilogue : IDifferentiable>``).  Returns the canonical struct
    name if valid, ``None`` if ``name`` is ``None`` (no epilogue), or raises
    ``ValueError`` if the name is not a known differentiable struct.
    """
    if name is None:
        return None
    if name in _VALID_IDIFFERENTIABLE_STRUCTS:
        return name
    if name in _VALID_IPOINTWISE_STRUCTS:
        raise ValueError(
            f"Epilogue struct '{name}' implements IPointwise but NOT IDifferentiable. "
            f"The mm template requires IDifferentiable for autodiff support. "
        )
    raise ValueError(
        f"Unknown IPointwise epilogue struct '{name}'. "
        f"Must be one of: {sorted(_VALID_IPOINTWISE_STRUCTS)}"
    )


def _render_mm_slang(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    dtype_a: str = "float",
    dtype_b: str = "float",
    dtype_c: str = "float",
    dtype_acc: str = "float",
    dtype_bias: str = "float",
    epilogue_struct: str | None = None,
    num_stages: int = 1,
    has_bias: bool = False,
    has_alpha: bool = False,
    has_beta: bool = False,
    has_scale: bool = False,
    has_clamp: bool = False,
    has_batch: bool = False,
    m_per_thread: int = 1,
    n_per_thread: int = 1,
    use_module: bool = False,
) -> str:
    """Render the slang_mm Jinja2 template.

    When ``use_module=True`` (P3.2 / M14), delegates to
    ``_render_mm_linktime_wrapper_slang`` to produce a thin wrapper that
    imports ``mm_tile.slang-module`` instead of inlining the full tile loop.

    `m_per_thread` / `n_per_thread` control register-tile depth. With both at 1
    the workgroup is `(tile_n, tile_m)` and each thread emits one output (legacy
    1-output-per-thread path). With either > 1 the workgroup shrinks to
    `(tile_n / n_per_thread, tile_m / m_per_thread)` and each thread accumulates
    an `(m_per_thread, n_per_thread)` register block — the standard
    register-tile pattern that lets us pick `tile_m * tile_n` > 1024 without
    blowing the threadgroup limit, and amortizes K-loop loads over many outputs.

    CG.M10: ``epilogue_struct`` is a validated ``IDifferentiable`` struct name
    from ``_VALID_IDIFFERENTIABLE_STRUCTS`` (e.g. ``"OpGELU"``).  When
    non-``None``, the Jinja2 template emits the ``Epilogue::apply(...)`` call
    site.  The concrete type is NOT a Jinja variable — it is a Slang generic
    type parameter on ``computeMain<Epilogue : IDifferentiable>``, resolved at
    SPIR-V compile time via the ``entry`` parameter of ``compile_and_dispatch``.
    (Anti-goal #6: no string-based template parameters.)
    """
    from jinja2 import Environment

    # P3.2 / M14: When use_module is True, delegate to the link-time
    # specialization path.  The heavy tile-loop body lives in
    # mm_tile.slang-module (compiled ONCE per dtype).
    if use_module:
        return _render_mm_linktime_wrapper_slang(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            m_per_thread=m_per_thread,
            n_per_thread=n_per_thread,
            num_stages=num_stages,
            has_bias=has_bias,
            epilogue_struct=epilogue_struct,
            dtype_a=dtype_a,
            dtype_b=dtype_b,
            dtype_c=dtype_c,
            dtype_acc=dtype_acc,
            dtype_bias=dtype_bias,
            has_alpha=has_alpha,
            has_beta=has_beta,
            has_scale=has_scale,
            has_clamp=has_clamp,
        )

    # Validate the struct name early — fail at render time, not at SPIR-V
    # compile time, so the error message points at the Python caller.
    epilogue_struct = _validate_epilogue_struct(epilogue_struct)
    has_epilogue = epilogue_struct is not None

    key = (
        tile_m,
        tile_n,
        tile_k,
        dtype_a,
        dtype_b,
        dtype_c,
        dtype_acc,
        dtype_bias,
        epilogue_struct,
        num_stages,
        has_bias,
        has_alpha,
        has_beta,
        has_scale,
        has_clamp,
        has_batch,
        m_per_thread,
        n_per_thread,
    )
    if key in _tile_cache:
        return _tile_cache[key]

    src = _load_slang_template("slang_mm")
    if not src:
        raise RuntimeError("slang_mm.py.jinja template not found")

    env = Environment()
    tmpl = env.from_string(src)
    rendered = tmpl.render(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        dtype_a=dtype_a,
        dtype_b=dtype_b,
        dtype_c=dtype_c,
        dtype_acc=dtype_acc,
        dtype_bias=dtype_bias,
        epilogue=has_epilogue,
        num_stages=num_stages,
        has_bias=has_bias,
        has_alpha=has_alpha,
        has_beta=has_beta,
        has_scale=has_scale,
        has_clamp=has_clamp,
        has_batch=has_batch,
        m_per_thread=m_per_thread,
        n_per_thread=n_per_thread,
    )
    _tile_cache[key] = rendered
    return rendered


def _check_workgroup_fits(
    tile_m: int,
    tile_n: int,
    m_per_thread: int = 1,
    n_per_thread: int = 1,
    max_wg: int = 1024,
) -> bool:
    """Return True iff the (tile_m, tile_n, m_per_thread, n_per_thread) shape's
    workgroup size fits the device's max_workgroup_invocations limit. Tiles that
    fail this would crash at pipeline creation; we filter them upfront."""
    wg_m = tile_m // m_per_thread
    wg_n = tile_n // n_per_thread
    return (
        wg_m * wg_n <= max_wg
        and wg_m * m_per_thread == tile_m
        and wg_n * n_per_thread == tile_n
    )


def _dtype_to_slang(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float"
    if dtype == torch.float16:
        return "half"
    if dtype == torch.bfloat16:
        return "uint"
    return "float"


_TRUST_INDUCTOR = os.environ.get("TORCH_VULKAN_TRUST_INDUCTOR") == "1"


def _reset_trust_inductor_cache() -> None:
    """Test hook — re-reads `TORCH_VULKAN_TRUST_INDUCTOR` from the environment.
    The flag is captured once at module import for hot-path use; tests that
    flip the env var mid-process can call this to refresh."""
    global _TRUST_INDUCTOR
    _TRUST_INDUCTOR = os.environ.get("TORCH_VULKAN_TRUST_INDUCTOR") == "1"


def _slang_tile_mm(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    num_stages: int,
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    src: str | None = None,
    cache_key: str | None = None,
    m_per_thread: int = 1,
    n_per_thread: int = 1,
    epilogue_struct: str | None = None,
) -> None:
    """Execute tiled matmul C = A @ B via Slang template shader.

    CG.M10: ``epilogue_struct`` is a validated ``IDifferentiable`` struct name
    (e.g. ``"OpGELU"``) used as the Slang generic type argument in the entry
    point name.  ``None`` (the default) resolves to ``OpIdentity`` — a no-op
    pass-through that trivially satisfies ``IDifferentiable``.
    """
    from .runtime import compile_and_dispatch

    M, K = a.shape
    _, N = b.shape

    epilogue_struct = _validate_epilogue_struct(epilogue_struct)
    epi_name = epilogue_struct if epilogue_struct is not None else "OpIdentity"

    if src is None or cache_key is None:
        dtype_s = _dtype_to_slang(a.dtype)
        src = _render_mm_slang(
            tile_m,
            tile_n,
            tile_k,
            dtype_a=dtype_s,
            dtype_b=dtype_s,
            dtype_c=dtype_s,
            dtype_acc="float",
            num_stages=num_stages,
            m_per_thread=m_per_thread,
            n_per_thread=n_per_thread,
            epilogue_struct=epilogue_struct,
        )
        # N+1.11: _n111 prevents stale cache hits with old PC layout.
        cache_key = (
            f"slang_mm_{tile_m}_{tile_n}_{tile_k}_s{num_stages}"
            f"_r{m_per_thread}x{n_per_thread}_{dtype_s}_n111"
        )
        if epilogue_struct is not None:
            cache_key += f"_epi_{epilogue_struct}"

    if not _TRUST_INDUCTOR:
        if not a.is_contiguous():
            a = a.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()
        if not out.is_contiguous():
            out = out.contiguous()

    # Pass actual strides so column-major operands (e.g. transposed weight
    # from im2col conv2d via reinterpret_tensor) compute correct offsets.
    # Mirrors _slang_tile_mm_backward which already uses actual strides.
    stride_a_m = a.stride(0)
    stride_a_k = a.stride(1) if a.dim() > 1 else 1
    stride_b_k = b.stride(0)
    stride_b_n = b.stride(1) if b.dim() > 1 else 1

    # N+1.11: PC layout now includes tile config fields after strides.
    # Order: M, N, K, stride_a_m, stride_a_k, stride_b_k, stride_b_n,
    #        stride_c_m, stride_c_n, tile_m, tile_n, tile_k,
    #        m_per_thread, n_per_thread
    pc = struct.pack(
        "14I",
        M,
        N,
        K,
        stride_a_m,
        stride_a_k,
        stride_b_k,
        stride_b_n,
        out.stride(0),
        out.stride(1),
        tile_m,
        tile_n,
        tile_k,
        m_per_thread,
        n_per_thread,
    )

    grid_x = (N + tile_n - 1) // tile_n
    grid_y = (M + tile_m - 1) // tile_m

    compile_and_dispatch(
        src,
        [a, b, out],
        grid_x,
        grid_y,
        1,
        push_constants=pc,
        num_outputs=1,
        entry=f"computeMain<{epi_name}>",
        cache_key=cache_key,
    )


def _slang_tile_addmm(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    num_stages: int,
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    src: str | None = None,
    cache_key: str | None = None,
    m_per_thread: int = 1,
    n_per_thread: int = 1,
    epilogue_struct: str | None = None,
) -> None:
    """Execute tiled addmm: out = a @ b + bias in one Slang dispatch.

    Buffer layout (has_bias=True): [a, b, bias, out] — output is last so
    num_outputs=1 correctly marks only 'out' as dirty.

    CG.M10: ``epilogue_struct`` is a validated ``IDifferentiable`` struct name;
    ``None`` resolves to ``OpIdentity`` (no-op pass-through) at the Slang
    generic type argument on ``computeMain<Epilogue : IDifferentiable>``.
    """
    from .runtime import compile_and_dispatch

    M, K = a.shape
    _, N = b.shape

    epilogue_struct = _validate_epilogue_struct(epilogue_struct)
    epi_name = epilogue_struct if epilogue_struct is not None else "OpIdentity"

    if src is None or cache_key is None:
        dtype_s = _dtype_to_slang(a.dtype)
        src = _render_mm_slang(
            tile_m,
            tile_n,
            tile_k,
            dtype_a=dtype_s,
            dtype_b=dtype_s,
            dtype_c=dtype_s,
            dtype_acc="float",
            dtype_bias=dtype_s,
            num_stages=num_stages,
            has_bias=True,
            m_per_thread=m_per_thread,
            n_per_thread=n_per_thread,
            epilogue_struct=epilogue_struct,
        )
        cache_key = (
            f"slang_addmm_{tile_m}_{tile_n}_{tile_k}_s{num_stages}"
            f"_r{m_per_thread}x{n_per_thread}_{dtype_s}_n111"
        )
        if epilogue_struct is not None:
            cache_key += f"_epi_{epilogue_struct}"

    if not _TRUST_INDUCTOR:
        if not a.is_contiguous():
            a = a.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()
        if not out.is_contiguous():
            out = out.contiguous()
    bias_1d = bias.view(-1) if bias.dim() > 1 else bias
    if not _TRUST_INDUCTOR and not bias_1d.is_contiguous():
        bias_1d = bias_1d.contiguous()

    # Pass actual strides so column-major operands compute correct offsets.
    stride_a_m = a.stride(0)
    stride_a_k = a.stride(1) if a.dim() > 1 else 1
    stride_b_k = b.stride(0)
    stride_b_n = b.stride(1) if b.dim() > 1 else 1
    stride_bias_n = 1

    # N+1.11: PC layout now includes tile config after stride_bias_n.
    # Order: M, N, K, stride_a_m, stride_a_k, stride_b_k, stride_b_n,
    #        stride_c_m, stride_c_n, stride_bias_n,
    #        tile_m, tile_n, tile_k, m_per_thread, n_per_thread
    pc = struct.pack(
        "15I",
        M,
        N,
        K,
        stride_a_m,
        stride_a_k,
        stride_b_k,
        stride_b_n,
        out.stride(0),
        out.stride(1),
        stride_bias_n,
        tile_m,
        tile_n,
        tile_k,
        m_per_thread,
        n_per_thread,
    )

    grid_x = (N + tile_n - 1) // tile_n
    grid_y = (M + tile_m - 1) // tile_m

    # Buffer order: [a, b, bias, out] — binding(2)=bias, binding(3)=c(out)
    # num_outputs=1 marks last buffer (out) as dirty.
    compile_and_dispatch(
        src,
        [a, b, bias_1d, out],
        grid_x,
        grid_y,
        1,
        push_constants=pc,
        num_outputs=1,
        entry=f"computeMain<{epi_name}>",
        cache_key=cache_key,
    )


def _slang_tile_addmm_gelu(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    num_stages: int,
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    src: str | None = None,
    cache_key: str | None = None,
    m_per_thread: int = 1,
    n_per_thread: int = 1,
) -> None:
    """Execute fused tiled addmm+gelu: out = gelu(a @ b + bias) in one Slang dispatch.

    PF.5 epilogue fusion. Buffer layout matches `_slang_tile_addmm` exactly
    ([a, b, bias, out]); only the rendered shader differs (has_epilogue=True).
    The epilogue struct is passed via the Slang generic type parameter on
    ``computeMain``, resolved at SPIR-V compile time.
    """
    from .runtime import compile_and_dispatch

    M, K = a.shape
    _, N = b.shape

    if src is None or cache_key is None:
        dtype_s = _dtype_to_slang(a.dtype)
        src = _render_mm_slang(
            tile_m,
            tile_n,
            tile_k,
            dtype_a=dtype_s,
            dtype_b=dtype_s,
            dtype_c=dtype_s,
            dtype_acc="float",
            dtype_bias=dtype_s,
            num_stages=num_stages,
            has_bias=True,
            epilogue_struct="OpGELU",
            m_per_thread=m_per_thread,
            n_per_thread=n_per_thread,
        )
        cache_key = (
            f"slang_addmm_epi_OpGELU_{tile_m}_{tile_n}_{tile_k}_s{num_stages}"
            f"_r{m_per_thread}x{n_per_thread}_{dtype_s}"
        )

    if not _TRUST_INDUCTOR:
        if not a.is_contiguous():
            a = a.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()
        if not out.is_contiguous():
            out = out.contiguous()
    bias_1d = bias.view(-1) if bias.dim() > 1 else bias
    if not _TRUST_INDUCTOR and not bias_1d.is_contiguous():
        bias_1d = bias_1d.contiguous()

    pc = struct.pack(
        "10I",
        M,
        N,
        K,
        a.stride(0),
        1,
        b.stride(0),
        1,
        out.stride(0),
        out.stride(1),
        1,
    )

    grid_x = (N + tile_n - 1) // tile_n
    grid_y = (M + tile_m - 1) // tile_m

    compile_and_dispatch(
        src,
        [a, b, bias_1d, out],
        grid_x,
        grid_y,
        1,
        push_constants=pc,
        num_outputs=1,
        entry="computeMain<OpGELU>",
        cache_key=cache_key,
    )


# ═══════════════════════════════════════════════════════════════════════════
# T4.13 — Consolidated GEMM caller
#
# `_SlangTileGEMM` is the parameterised picklable caller class shared by
# all four matmul variants (mm, addmm, bmm, addmm+gelu). The four historical
# classes below are thin subclasses that pin the relevant flag combination —
# they exist solely for backward-compat (pickle reduce target) and to keep
# the `_make_tile_*_fn` factory shortcuts and external imports stable. Each
# subclass keeps its original `__init__` signature and `__reduce__` tuple
# shape so previously pickled blobs remain bytes-equal.
#
# The four flag combinations:
#   _SlangTileMM         has_bias=False, has_batch=False, epilogue=None
#   _SlangTileAddMM      has_bias=True,  has_batch=False, epilogue=None
#   _SlangTileBMM        has_bias=False, has_batch=True,  epilogue=None
#   _SlangTileAddMMGelu  has_bias=True,  has_batch=False, epilogue="OpGELU"
# ═══════════════════════════════════════════════════════════════════════════


class _SlangTileGEMM:
    """Picklable callable for tiled Slang GEMM variants — Inductor's codecache
    pickles the external_matmul list as part of the cache key. Closures aren't
    picklable; a module-level class with instance state is.

    Caches the rendered Slang source + cache_key per tensor dtype on the
    instance so the per-dispatch path skips the Jinja-render dict lookup
    and cache_key f-string.

    `m_per_thread` / `n_per_thread` > 1 enable register-tiling: the workgroup is
    `(tile_n / n_per_thread, tile_m / m_per_thread)` and each thread holds an
    `(m_per_thread, n_per_thread)` register accumulator block. Required to use
    tile_m * tile_n > 1024 within RDNA1's max_workgroup_invocations limit.

    Variant flags:
        has_bias  — emit `out = a @ b + bias` (addmm path).
        has_batch — operate on 3D tensors `[B, M, K] @ [B, K, N]` (bmm path).
        epilogue  — Slang IDifferentiable struct name (e.g. `"OpGELU"`) applied
                    after bias-add. ``None`` resolves to ``OpIdentity``.

    Subclasses (`_SlangTileMM`, `_SlangTileAddMM`, `_SlangTileBMM`,
    `_SlangTileAddMMGelu`) pin specific flag combinations and keep their
    original ctor signatures for pickle backward-compat.
    """

    __slots__ = (
        "tile_m",
        "tile_n",
        "tile_k",
        "num_stages",
        "m_per_thread",
        "n_per_thread",
        "has_bias",
        "has_batch",
        "epilogue",
        "__name__",
        "_per_dtype",
    )

    # Subclasses override these to control the public class name (used in
    # cache keys / __name__ formatting) and the ``_SlangTileMM`` link-time
    # specialisation path. Defaults match the plain mm variant.
    _name_prefix: str = "slang_mm"
    _supports_link_time: bool = False

    def __init__(
        self,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        num_stages: int = 1,
        m_per_thread: int = 1,
        n_per_thread: int = 1,
        *,
        has_bias: bool = False,
        has_batch: bool = False,
        epilogue: str | None = None,
    ):
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.num_stages = num_stages
        self.m_per_thread = m_per_thread
        self.n_per_thread = n_per_thread
        self.has_bias = has_bias
        self.has_batch = has_batch
        self.epilogue = _validate_epilogue_struct(epilogue)
        self.__name__ = self._format_name()
        self._per_dtype: dict[str, tuple[str, str]] = {}

    def _format_name(self) -> str:
        return (
            f"{self._name_prefix}_{self.tile_m}_{self.tile_n}_{self.tile_k}"
            f"_s{self.num_stages}"
            f"_r{self.m_per_thread}x{self.n_per_thread}"
        )

    def _src_and_key(self, dtype_s: str) -> tuple[str, str]:
        cached = self._per_dtype.get(dtype_s)
        if cached is not None:
            return cached

        # Plain `_SlangTileMM` keeps the link-time specialisation hook for
        # P3.2/M14. The infrastructure is gated off today (slangc module
        # imports unstable) but the path is preserved for when it lands.
        if self._supports_link_time:
            from torch_vulkan.inductor.config import link_time_spec

            use_lt = link_time_spec()
            if use_lt:
                # P3.2/M14 TODO: Enable when slangc module imports are stable.
                # Currently slangc's import mechanism doesn't reliably export
                # public symbols from precompiled .slang-module files (returns
                # "declaration not accessible" for mm_tile:: symbols).
                use_lt = False
            if use_lt:
                src = _render_mm_linktime_wrapper_slang(
                    self.tile_m,
                    self.tile_n,
                    self.tile_k,
                    dtype_a=dtype_s,
                    dtype_b=dtype_s,
                    dtype_c=dtype_s,
                    dtype_acc="float",
                    num_stages=self.num_stages,
                    m_per_thread=self.m_per_thread,
                    n_per_thread=self.n_per_thread,
                )
                cache_key = (
                    f"mm_tile_lt_{self.tile_m}_{self.tile_n}_{self.tile_k}"
                    f"_s{self.num_stages}"
                    f"_r{self.m_per_thread}x{self.n_per_thread}_{dtype_s}"
                )
                cached = (src, cache_key)
                self._per_dtype[dtype_s] = cached
                return cached

        # Standard render path — covers mm, addmm, bmm, addmm+gelu.
        render_kwargs: dict[str, Any] = dict(
            dtype_a=dtype_s,
            dtype_b=dtype_s,
            dtype_c=dtype_s,
            dtype_acc="float",
            num_stages=self.num_stages,
            m_per_thread=self.m_per_thread,
            n_per_thread=self.n_per_thread,
        )
        if self.has_bias:
            render_kwargs["dtype_bias"] = dtype_s
            render_kwargs["has_bias"] = True
        if self.has_batch:
            render_kwargs["has_batch"] = True
        if self.epilogue is not None:
            render_kwargs["epilogue_struct"] = self.epilogue

        src = _render_mm_slang(self.tile_m, self.tile_n, self.tile_k, **render_kwargs)

        # DR.3 / N+1.12: Include subgroup_size and loop_depth in the
        # cache key so wave32 vs wave64 SPIR-V variants and kernels with
        # different nesting depths get distinct cache entries.
        _sgs = _get_device_subgroup_size()
        _sgs_tag = f"_sgs{_sgs}"

        # N+1.12: Structural loop-depth estimate for the tile kernel.
        # Base=3 (M-tile, N-tile, K-tile loops). Pipelining (+1) and
        # epilogue (+1) add nesting depth that affects VGPR allocation.
        _loop_depth = 3
        if self.num_stages > 1:
            _loop_depth += 1
        if self.epilogue is not None:
            _loop_depth += 1
        _ld_tag = f"_ld{_loop_depth}"

        # Cache key prefix mirrors the historical strings to keep SPIR-V
        # caches and autotune buckets distinct between variants.
        # N+1.11: _n111 tag prevents stale pre-N+1.11 cache hits (PC layout
        # now includes tile_m, tile_n, tile_k, m_per_thread, n_per_thread).
        if self.has_batch:
            cache_key = (
                f"slang_bmm_v2_{self.tile_m}_{self.tile_n}_{self.tile_k}"
                f"_r{self.m_per_thread}x{self.n_per_thread}_{dtype_s}"
                f"{_sgs_tag}{_ld_tag}_n111"
            )
        elif self.has_bias and self.epilogue == "OpGELU":
            cache_key = (
                f"slang_addmm_epi_OpGELU_{self.tile_m}_{self.tile_n}_{self.tile_k}"
                f"_s{self.num_stages}"
                f"_r{self.m_per_thread}x{self.n_per_thread}_{dtype_s}"
                f"{_sgs_tag}{_ld_tag}_n111"
            )
        elif self.has_bias:
            cache_key = (
                f"slang_addmm_{self.tile_m}_{self.tile_n}_{self.tile_k}"
                f"_s{self.num_stages}"
                f"_r{self.m_per_thread}x{self.n_per_thread}_{dtype_s}"
                f"{_sgs_tag}{_ld_tag}_n111"
            )
        else:
            cache_key = (
                f"slang_mm_{self.tile_m}_{self.tile_n}_{self.tile_k}"
                f"_s{self.num_stages}"
                f"_r{self.m_per_thread}x{self.n_per_thread}_{dtype_s}"
                f"{_sgs_tag}{_ld_tag}_n111"
            )
        if self.epilogue is not None and not (
            self.has_bias and self.epilogue == "OpGELU"
        ):
            # AddMMGelu already encodes the epilogue in its prefix; for plain
            # mm/addmm with a different epilogue, append `_epi_<Name>` —
            # matches the historical `_slang_tile_mm` / `_slang_tile_addmm`
            # cache-key suffix logic.
            cache_key += f"_epi_{self.epilogue}"

        cached = (src, cache_key)
        self._per_dtype[dtype_s] = cached
        return cached

    # __call__ dispatches to the matching low-level helper. The helpers
    # differ in push-constant layout and buffer ordering, so we pick by
    # flag rather than trying to merge them.
    def __call__(self, *args, **kwargs):
        if self.has_batch:
            return self._call_bmm(*args, **kwargs)
        if self.has_bias and self.epilogue == "OpGELU":
            return self._call_addmm_gelu(*args, **kwargs)
        if self.has_bias:
            return self._call_addmm(*args, **kwargs)
        return self._call_mm(*args, **kwargs)

    def _call_mm(
        self, a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        if out is None:
            out_size = (a.shape[0], b.shape[1])
            out = pool_acquire(out_size, a.dtype, a.device)
            if out is None:
                out = torch.empty(out_size, device=a.device, dtype=a.dtype)
        src, cache_key = self._src_and_key(_dtype_to_slang(a.dtype))
        _slang_tile_mm(
            self.tile_m,
            self.tile_n,
            self.tile_k,
            self.num_stages,
            a,
            b,
            out,
            src=src,
            cache_key=cache_key,
            m_per_thread=self.m_per_thread,
            n_per_thread=self.n_per_thread,
        )
        return out

    def _call_addmm(
        self,
        bias: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # PF.30.g: f16 reaches the Slang template via Slang's native `half`
        # storage. bf16 is held back until PF-D2.4 lands packed-uint storage
        # in the Jinja template (`_dtype_to_slang(bf16)` now correctly
        # returns `"uint"`).
        if alpha != 1 or beta != 1:
            return torch.addmm(bias, a, b, alpha=alpha, beta=beta)
        if a.dtype not in (torch.float32, torch.float16):
            return torch.addmm(bias, a, b, alpha=alpha, beta=beta)
        if out is None:
            out_size = (a.shape[0], b.shape[1])
            out = pool_acquire(out_size, a.dtype, a.device)
            if out is None:
                out = torch.empty(out_size, device=a.device, dtype=a.dtype)
        src, cache_key = self._src_and_key(_dtype_to_slang(a.dtype))
        _slang_tile_addmm(
            self.tile_m,
            self.tile_n,
            self.tile_k,
            self.num_stages,
            bias,
            a,
            b,
            out,
            src=src,
            cache_key=cache_key,
            m_per_thread=self.m_per_thread,
            n_per_thread=self.n_per_thread,
        )
        return out

    def _call_addmm_gelu(
        self,
        bias: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if out is None:
            out_size = (a.shape[0], b.shape[1])
            out = pool_acquire(out_size, a.dtype, a.device)
            if out is None:
                out = torch.empty(out_size, device=a.device, dtype=a.dtype)
        src, cache_key = self._src_and_key(_dtype_to_slang(a.dtype))
        _slang_tile_addmm_gelu(
            self.tile_m,
            self.tile_n,
            self.tile_k,
            self.num_stages,
            bias,
            a,
            b,
            out,
            src=src,
            cache_key=cache_key,
            m_per_thread=self.m_per_thread,
            n_per_thread=self.n_per_thread,
        )
        return out

    def _call_bmm(
        self, a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        if out is None:
            out_size = (a.shape[0], a.shape[1], b.shape[2])
            out = pool_acquire(out_size, a.dtype, a.device)
            if out is None:
                out = torch.empty(out_size, device=a.device, dtype=a.dtype)
        src, cache_key = self._src_and_key(_dtype_to_slang(a.dtype))
        _slang_tile_bmm(
            self.tile_m,
            self.tile_n,
            self.tile_k,
            a,
            b,
            out,
            src=src,
            cache_key=cache_key,
            m_per_thread=self.m_per_thread,
            n_per_thread=self.n_per_thread,
        )
        return out

    def __reduce__(self):
        # Default reduce — used by direct `_SlangTileGEMM(...)` instances.
        # Subclasses override to preserve their historical pickle shape
        # (bytes-equal for previously cached blobs).
        return (
            _SlangTileGEMM,
            (
                self.tile_m,
                self.tile_n,
                self.tile_k,
                self.num_stages,
                self.m_per_thread,
                self.n_per_thread,
            ),
            {
                "has_bias": self.has_bias,
                "has_batch": self.has_batch,
                "epilogue": self.epilogue,
            },
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Restore the variant flags injected by `__reduce__`'s state dict.
        self.has_bias = state.get("has_bias", False)
        self.has_batch = state.get("has_batch", False)
        self.epilogue = _validate_epilogue_struct(state.get("epilogue"))
        self.__name__ = self._format_name()


class _SlangTileMM(_SlangTileGEMM):
    """Picklable callable for tiled Slang mm. Pinned flags:
    ``has_bias=False, has_batch=False, epilogue=None``.

    Subclass of :class:`_SlangTileGEMM`. Kept as a named class so previously
    pickled `_SlangTileMM(...)` blobs remain bytes-equal across this refactor
    (the pickle reduce tuple is ``(_SlangTileMM, (tile_m, tile_n, tile_k,
    num_stages, m_per_thread, n_per_thread))`` exactly as before).
    """

    __slots__ = ()
    _name_prefix = "slang_mm"
    _supports_link_time = True

    def __init__(
        self,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        num_stages: int = 1,
        m_per_thread: int = 1,
        n_per_thread: int = 1,
    ):
        super().__init__(
            tile_m,
            tile_n,
            tile_k,
            num_stages,
            m_per_thread,
            n_per_thread,
            has_bias=False,
            has_batch=False,
            epilogue=None,
        )

    def __reduce__(self):
        return (
            _SlangTileMM,
            (
                self.tile_m,
                self.tile_n,
                self.tile_k,
                self.num_stages,
                self.m_per_thread,
                self.n_per_thread,
            ),
        )


class _SlangTileAddMM(_SlangTileGEMM):
    """Picklable addmm caller. Pinned flags:
    ``has_bias=True, has_batch=False, epilogue=None``.

    Falls back to ``torch.addmm`` for non-unit alpha/beta or unsupported
    dtypes; otherwise dispatches the fused tiled addmm shader. See
    :class:`_SlangTileMM` for register-tile semantics.
    """

    __slots__ = ()
    _name_prefix = "slang_addmm"

    def __init__(
        self,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        num_stages: int = 1,
        m_per_thread: int = 1,
        n_per_thread: int = 1,
    ):
        super().__init__(
            tile_m,
            tile_n,
            tile_k,
            num_stages,
            m_per_thread,
            n_per_thread,
            has_bias=True,
            has_batch=False,
            epilogue=None,
        )

    def __reduce__(self):
        return (
            _SlangTileAddMM,
            (
                self.tile_m,
                self.tile_n,
                self.tile_k,
                self.num_stages,
                self.m_per_thread,
                self.n_per_thread,
            ),
        )


class _SlangTileBMM(_SlangTileGEMM):
    """Picklable batched matmul caller. Pinned flags:
    ``has_bias=False, has_batch=True, epilogue=None``.

    Note: the historical ctor takes no ``num_stages`` (BMM hardcodes
    ``num_stages=1`` in its dispatch helper). This subclass preserves that
    signature for pickle bytes-equality.
    """

    __slots__ = ()
    _name_prefix = "slang_bmm_v2"

    def __init__(
        self,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        m_per_thread: int = 1,
        n_per_thread: int = 1,
    ):
        super().__init__(
            tile_m,
            tile_n,
            tile_k,
            1,  # num_stages — BMM hardcodes 1
            m_per_thread,
            n_per_thread,
            has_bias=False,
            has_batch=True,
            epilogue=None,
        )

    def _format_name(self) -> str:
        # BMM historically omits the `_s{num_stages}` suffix.
        return (
            f"{self._name_prefix}_{self.tile_m}_{self.tile_n}_{self.tile_k}"
            f"_r{self.m_per_thread}x{self.n_per_thread}"
        )

    def __reduce__(self):
        return (
            _SlangTileBMM,
            (
                self.tile_m,
                self.tile_n,
                self.tile_k,
                self.m_per_thread,
                self.n_per_thread,
            ),
        )


class _SlangTileAddMMGelu(_SlangTileGEMM):
    """Picklable callable for fused addmm+gelu — single Slang dispatch.
    Pinned flags: ``has_bias=True, has_batch=False, epilogue="OpGELU"``.

    Used by the ``torch_vulkan::addmm_gelu_fused`` custom op (PF.5).
    """

    __slots__ = ()
    _name_prefix = "slang_addmm_gelu"

    def __init__(
        self,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        num_stages: int = 1,
        m_per_thread: int = 1,
        n_per_thread: int = 1,
    ):
        super().__init__(
            tile_m,
            tile_n,
            tile_k,
            num_stages,
            m_per_thread,
            n_per_thread,
            has_bias=True,
            has_batch=False,
            epilogue="OpGELU",
        )

    def __reduce__(self):
        return (
            _SlangTileAddMMGelu,
            (
                self.tile_m,
                self.tile_n,
                self.tile_k,
                self.num_stages,
                self.m_per_thread,
                self.n_per_thread,
            ),
        )


_addmm_gelu_default_caller: Optional[_SlangTileAddMMGelu] = None


def _pick_addmm_gelu_tile(M: int, N: int, K: int) -> _SlangTileAddMMGelu:
    """Pick a tile config for the fused addmm+gelu dispatch.

    PF.5: keeps it simple — the canonical (32, 32, 32) tile fits any RDNA1
    workgroup limit, hits all MLP shapes (M=8, N∈{32,128}, K∈{64,128}) in
    a single dispatch, and reuses the prewarmed addmm SPIR-V cache size class.
    Larger workloads (transformer FFN, ResNet head) would benefit from
    register-tile autotuning — filed as follow-up below.
    """
    global _addmm_gelu_default_caller
    if _addmm_gelu_default_caller is None:
        _addmm_gelu_default_caller = _SlangTileAddMMGelu(
            32, 32, 32, num_stages=1, m_per_thread=1, n_per_thread=1
        )
    return _addmm_gelu_default_caller


def _slang_tile_bmm(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    src: str | None = None,
    cache_key: str | None = None,
    m_per_thread: int = 1,
    n_per_thread: int = 1,
    epilogue_struct: str | None = None,
) -> None:
    """Execute batched tiled matmul C[B,M,N] = A[B,M,K] @ B[B,K,N]."""
    from .runtime import compile_and_dispatch

    B, M, K = a.shape
    _, _, N = b.shape

    epilogue_struct = _validate_epilogue_struct(epilogue_struct)
    epi_name = epilogue_struct if epilogue_struct is not None else "OpIdentity"

    if src is None or cache_key is None:
        dtype_s = _dtype_to_slang(a.dtype)
        src = _render_mm_slang(
            tile_m,
            tile_n,
            tile_k,
            dtype_a=dtype_s,
            dtype_b=dtype_s,
            dtype_c=dtype_s,
            dtype_acc="float",
            m_per_thread=m_per_thread,
            n_per_thread=n_per_thread,
            epilogue_struct=epilogue_struct,
            has_batch=True,
        )
        # v2: cache key bumped because PC layout changed (added 3 batch
        # strides). Without this, stale SPIR-V from before the batch fix
        # would be reused with the new larger push-constant block.
        # N+1.11: _n111 prevents stale cache hits with old PC layout.
        cache_key = (
            f"slang_bmm_v2_{tile_m}_{tile_n}_{tile_k}"
            f"_r{m_per_thread}x{n_per_thread}_{dtype_s}_n111"
        )
        if epilogue_struct is not None:
            cache_key += f"_epi_{epilogue_struct}"

    if not _TRUST_INDUCTOR:
        if not a.is_contiguous():
            a = a.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()
        if not out.is_contiguous():
            out = out.contiguous()

    # N+1.11: PC layout includes batch strides then tile config fields.
    # Order: M, N, K, stride_a_m, stride_a_k, stride_b_k, stride_b_n,
    #        stride_c_m, stride_c_n, stride_a_b, stride_b_b, stride_c_b,
    #        tile_m, tile_n, tile_k, m_per_thread, n_per_thread
    pc = struct.pack(
        "17I",
        M,
        N,
        K,
        a.stride(1),
        a.stride(2),
        b.stride(1),
        b.stride(2),
        out.stride(1),
        out.stride(2),
        a.stride(0),
        b.stride(0),
        out.stride(0),
        tile_m,
        tile_n,
        tile_k,
        m_per_thread,
        n_per_thread,
    )

    grid_x = (N + tile_n - 1) // tile_n
    grid_y = (M + tile_m - 1) // tile_m
    grid_z = B

    compile_and_dispatch(
        src,
        [a, b, out],
        grid_x,
        grid_y,
        grid_z,
        push_constants=pc,
        num_outputs=1,
        entry=f"computeMain<{epi_name}>",
        cache_key=cache_key,
    )


def _make_tile_mm_fn(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    num_stages: int = 1,
    m_per_thread: int = 1,
    n_per_thread: int = 1,
):
    return _SlangTileMM(tile_m, tile_n, tile_k, num_stages, m_per_thread, n_per_thread)


def _make_tile_addmm_fn(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    num_stages: int = 1,
    m_per_thread: int = 1,
    n_per_thread: int = 1,
):
    return _SlangTileAddMM(
        tile_m, tile_n, tile_k, num_stages, m_per_thread, n_per_thread
    )


def _make_tile_bmm_fn(
    tile_m: int, tile_n: int, tile_k: int, m_per_thread: int = 1, n_per_thread: int = 1
):
    return _SlangTileBMM(tile_m, tile_n, tile_k, m_per_thread, n_per_thread)


# ═══════════════════════════════════════════════════════════════════════════
# Path 2 — Conv2d template wiring
# ═══════════════════════════════════════════════════════════════════════════

_conv2d_cache: dict[tuple, str] = {}


def _render_conv2d_slang(
    tile_w: int = 8,
    tile_h: int = 8,
    tile_c: int = 8,
    threads_w: int = 16,
    threads_h: int = 16,
    has_bias: bool = False,
) -> str:
    """Render the slang_conv2d Jinja2 template with tile configuration."""
    from jinja2 import Environment

    key = (tile_w, tile_h, tile_c, threads_w, threads_h, has_bias)
    if key in _conv2d_cache:
        return _conv2d_cache[key]

    src = _load_slang_template("slang_conv2d")
    if not src:
        raise RuntimeError("slang_conv2d.slang template not found")

    env = Environment()
    tmpl = env.from_string(src)
    rendered = tmpl.render(
        tile_w=tile_w,
        tile_h=tile_h,
        tile_c=tile_c,
        threads_w=threads_w,
        threads_h=threads_h,
        has_bias=has_bias,
    )
    _conv2d_cache[key] = rendered
    return rendered


def _slang_tile_conv2d(
    input_t: torch.Tensor,
    weight_t: torch.Tensor,
    out: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int = 1,
    bias: torch.Tensor | None = None,
    tile_w: int = 8,
    tile_h: int = 8,
    tile_c: int = 8,
    threads_w: int = 16,
    threads_h: int = 16,
) -> None:
    """Execute direct tiled conv2d via Slang template shader.

    Input:  [N, C_in, iH, iW]  (NCHW)
    Weight: [C_out, C_in, kH, kW]  (NCHW, groups=1)
    Output: [N, C_out, oH, oW]  (NCHW)
    """
    from .runtime import compile_and_dispatch

    N, C_in, iH, iW = input_t.shape
    C_out, C_in_w, kH, kW = weight_t.shape
    assert C_in == C_in_w, f"weight C_in mismatch: {C_in} vs {C_in_w}"
    assert groups == 1, "Only groups=1 supported"

    sH, sW = stride
    pH, pW = padding
    dH, dW = dilation

    oH = (iH + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    oW = (iW + 2 * pW - dW * (kW - 1) - 1) // sW + 1

    has_bias = bias is not None
    dtype_s = _dtype_to_slang(input_t.dtype)
    src = _render_conv2d_slang(
        tile_w=tile_w,
        tile_h=tile_h,
        tile_c=tile_c,
        threads_w=threads_w,
        threads_h=threads_h,
        has_bias=has_bias,
    )
    cache_key = (
        f"slang_conv2d_{tile_w}x{tile_h}x{tile_c}"
        f"_t{threads_w}x{threads_h}_{dtype_s}"
        f"{'_bias' if has_bias else ''}"
    )

    # Ensure contiguous for direct buffer access
    if not input_t.is_contiguous():
        input_t = input_t.contiguous()
    if not weight_t.is_contiguous():
        weight_t = weight_t.contiguous()

    # Pack push constants: 15 uint fields for no-bias, 17 for bias
    common_fields = (
        N,
        C_in,
        C_out,
        iH,
        iW,
        oH,
        oW,
        kH,
        kW,
        sH,
        sW,
        pH,
        pW,
        dH,
        dW,
        input_t.stride(0),
        input_t.stride(1),
        input_t.stride(2),
        input_t.stride(3),
        weight_t.stride(0),
        weight_t.stride(1),
        weight_t.stride(2),
        weight_t.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
    )
    if has_bias:
        bias_1d = bias.view(-1)
        pc = struct.pack(
            "29I",
            *common_fields,
            bias_1d.stride(0),
            0,  # _pad
        )
    else:
        pc = struct.pack("27I", *common_fields)

    grid_x = (oW + tile_w - 1) // tile_w
    grid_y = (oH + tile_h - 1) // tile_h
    tile_c_count = (C_out + tile_c - 1) // tile_c
    grid_z = N * tile_c_count

    buffers = [input_t, weight_t]
    if has_bias:
        buffers.append(bias.view(-1))
    buffers.append(out)

    compile_and_dispatch(
        src,
        buffers,
        grid_x,
        grid_y,
        grid_z,
        push_constants=pc,
        num_outputs=1,
        entry="computeMain",
        cache_key=cache_key,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CG.M6 — Conv2d backward template wiring
# ═══════════════════════════════════════════════════════════════════════════

_conv_bwd_cache: dict[tuple, str] = {}


def _render_conv_bwd_slang(
    tile_w: int = 8,
    tile_h: int = 8,
    tile_c: int = 8,
    threads_w: int = 16,
    threads_h: int = 16,
    has_bias: bool = False,
) -> str:
    """Render the slang_conv_bwd Jinja2 template with tile configuration."""
    from jinja2 import Environment

    key = (tile_w, tile_h, tile_c, threads_w, threads_h, has_bias)
    if key in _conv_bwd_cache:
        return _conv_bwd_cache[key]

    src = _load_slang_template("slang_conv_bwd")
    if not src:
        raise RuntimeError("slang_conv_bwd.py.jinja template not found")

    env = Environment()
    tmpl = env.from_string(src)
    rendered = tmpl.render(
        tile_w=tile_w,
        tile_h=tile_h,
        tile_c=tile_c,
        threads_w=threads_w,
        threads_h=threads_h,
        has_bias=has_bias,
    )
    _conv_bwd_cache[key] = rendered
    return rendered


def _slang_tile_conv2d_bwd(
    input_t: torch.Tensor,
    weight_t: torch.Tensor,
    grad_out: torch.Tensor,
    grad_input: torch.Tensor,
    grad_weight: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    bias: torch.Tensor | None = None,
    grad_bias: torch.Tensor | None = None,
    tile_w: int = 8,
    tile_h: int = 8,
    tile_c: int = 8,
    threads_w: int = 16,
    threads_h: int = 16,
) -> None:
    """Execute conv2d backward via Slang template shader.

    Computes dX (grad_input), dW (grad_weight), and optionally dB (grad_bias)
    in a single dispatch using ``bwd_diff(conv_inner_madd)``.

    FakeTensor-aware: during AOT Autograd tracing, all inputs are FakeTensors.
    We detect this and return early — the caller already allocated output tensors
    with correct shapes; no actual computation is needed during tracing.

    Args:
        input_t:    [N, C_in, iH, iW]  (NCHW) — saved forward input
        weight_t:   [C_out, C_in, kH, kW] — saved forward weight
        grad_out:   [N, C_out, oH, oW] — upstream gradient
        grad_input: [N, C_in, iH, iW] — output: input gradient (zero-initialized)
        grad_weight:[C_out, C_in, kH, kW] — output: weight gradient (zero-initialized)
        grad_bias:  [C_out] — output: bias gradient (zero-initialized), optional
    """
    # PF.51 guard: during AOT Autograd tracing, inputs are FakeTensors with
    # meta-device storage.  Skip the actual dispatch — the caller already
    # allocated output tensors with correct shapes.
    try:
        if input_t.untyped_storage().device.type == "meta":
            return  # tracing mode, outputs already allocated
    except Exception:
        pass  # real storage, proceed with dispatch

    from .runtime import compile_and_dispatch

    N, C_in, iH, iW = input_t.shape
    C_out, C_in_w, kH, kW = weight_t.shape
    assert C_in == C_in_w, f"weight C_in mismatch: {C_in} vs {C_in_w}"

    sH, sW = stride
    pH, pW = padding
    dH, dW = dilation

    oH = (iH + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    oW = (iW + 2 * pW - dW * (kW - 1) - 1) // sW + 1

    has_bias = bias is not None
    dtype_s = _dtype_to_slang(input_t.dtype)
    src = _render_conv_bwd_slang(
        tile_w=tile_w,
        tile_h=tile_h,
        tile_c=tile_c,
        threads_w=threads_w,
        threads_h=threads_h,
        has_bias=has_bias,
    )
    cache_key = (
        f"slang_conv_bwd_{tile_w}x{tile_h}x{tile_c}"
        f"_t{threads_w}x{threads_h}_{dtype_s}"
        f"{'_bias' if has_bias else ''}"
    )

    # Ensure contiguous for direct buffer access
    if not input_t.is_contiguous():
        input_t = input_t.contiguous()
    if not weight_t.is_contiguous():
        weight_t = weight_t.contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    # Pack push constants: 27 uint fields (no bias) or 28 (with bias)
    # Layout matches BwdPC in slang_conv_bwd.py.jinja:
    #   dims (15) + stride_in (4) + stride_w (4) + stride_go (4) = 27
    #   + _pad_bwd (1) with bias = 28
    common_fields = (
        N,
        C_in,
        C_out,
        iH,
        iW,
        oH,
        oW,
        kH,
        kW,
        sH,
        sW,
        pH,
        pW,
        dH,
        dW,
        input_t.stride(0),
        input_t.stride(1),
        input_t.stride(2),
        input_t.stride(3),
        weight_t.stride(0),
        weight_t.stride(1),
        weight_t.stride(2),
        weight_t.stride(3),
        grad_out.stride(0),
        grad_out.stride(1),
        grad_out.stride(2),
        grad_out.stride(3),
    )
    if has_bias:
        pc = struct.pack("28I", *common_fields, 0)  # _pad_bwd
    else:
        pc = struct.pack("27I", *common_fields)

    grid_x = (oW + tile_w - 1) // tile_w
    grid_y = (oH + tile_h - 1) // tile_h
    tile_c_count = (C_out + tile_c - 1) // tile_c
    grid_z = N * tile_c_count

    buffers = [input_t, weight_t, grad_out, grad_input, grad_weight]
    if has_bias:
        buffers.append(grad_bias.view(-1))

    compile_and_dispatch(
        src,
        buffers,
        grid_x,
        grid_y,
        grid_z,
        push_constants=pc,
        num_outputs=2 if not has_bias else 3,
        entry="computeMain",
        cache_key=cache_key,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Track 4.6 — Philox RNG template wiring
# ═══════════════════════════════════════════════════════════════════════════

_rng_installed = False

_PHILOX_RNG_CACHE: dict[tuple, str] = {}


def _render_philox_rng(
    output_dtype: str = "float",
    rng_mode: str = "uniform",
    fused_dropout: bool = False,
) -> str:
    """Render the philox_rng Jinja2 template with the given parameters.

    Args:
        output_dtype: Slang type string — ``"float"`` (f32), ``"half"`` (f16),
                      or ``"uint"`` (bf16).
        rng_mode: ``"uniform"`` (1 output per thread) or ``"normal"``
                  (Box-Muller, 2 outputs per thread).
        fused_dropout: If True, the shader applies a dropout mask + scale
                       after RNG and reads from a second input binding.
    """
    from jinja2 import Environment

    num_outputs = 2 if rng_mode == "normal" else 1
    key = (output_dtype, rng_mode, fused_dropout)
    if key in _PHILOX_RNG_CACHE:
        return _PHILOX_RNG_CACHE[key]

    src = _load_slang_template("philox_rng")
    if not src:
        raise RuntimeError("philox_rng.py.jinja template not found")

    env = Environment()
    tmpl = env.from_string(src)
    rendered = tmpl.render(
        output_dtype=output_dtype,
        rng_mode=rng_mode,
        fused_dropout=fused_dropout,
        num_outputs=num_outputs,
    )
    _PHILOX_RNG_CACHE[key] = rendered
    return rendered


def _dispatch_philox_rng(
    output_dtype: str,
    rng_mode: str,
    fused_dropout: bool,
    total_elements: int,
    seed_lo: int,
    seed_hi: int,
    offset: int,
    output: torch.Tensor,
    output2: torch.Tensor | None = None,
    input_tensor: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    cache_key: str = "",
) -> None:
    """Dispatch the Philox RNG template as a standalone compute shader.

    Args:
        output_dtype: Slang type for the output buffer(s).
        rng_mode: ``"uniform"`` or ``"normal"``.
        fused_dropout: Whether to apply dropout mask + scale.
        total_elements: Number of random samples to generate.
        seed_lo, seed_hi: 64-bit Philox seed split into two u32s.
        offset: Starting offset in the Philox counter sequence.
        output: Primary output tensor (must be pre-allocated).
        output2: Secondary output for ``rng_mode="normal"`` (Box-Muller).
        input_tensor: Input tensor for fused dropout mode.
        dropout_p: Dropout probability (only for fused_dropout).
        cache_key: Stable cache key for SPIR-V compilation caching.
    """
    from .runtime import compile_and_dispatch

    numel = total_elements
    threadgroup_size = 256
    grid_x = (numel + threadgroup_size - 1) // threadgroup_size

    # Build push constants: total_elements, seed_lo, seed_hi, offset, [dropout_p]
    # Layout matches the PC struct in philox_rng.py.jinja:
    #   uint total_elements; uint seed_lo; uint seed_hi; uint offset; [float dropout_p;]
    if fused_dropout:
        pc = struct.pack("4If", numel, seed_lo, seed_hi, offset, dropout_p)
    else:
        pc = struct.pack("4I", numel, seed_lo, seed_hi, offset)

    if not cache_key:
        cache_key = (
            f"slang_philox_{rng_mode}"
            f"{'_dropout' if fused_dropout else ''}"
            f"_{output_dtype}"
        )

    src = _render_philox_rng(
        output_dtype=output_dtype,
        rng_mode=rng_mode,
        fused_dropout=fused_dropout,
    )

    tensors: list[torch.Tensor] = [output]
    if rng_mode == "normal" and output2 is not None:
        tensors.append(output2)
    if fused_dropout and input_tensor is not None:
        tensors.append(input_tensor.contiguous())

    if not output.is_contiguous():
        output = output.contiguous()
        tensors[0] = output

    num_outs = len(tensors) - (1 if fused_dropout else 0)

    compile_and_dispatch(
        src,
        tensors,
        grid_x,
        1,
        1,
        push_constants=pc,
        num_outputs=num_outs,
        cache_key=cache_key,
    )


class _SlangPhiloxRNG:
    """Picklable callable for standalone Philox RNG dispatch.

    Caches the rendered Slang source + cache_key per dtype on the instance
    so the per-dispatch path skips the Jinja-render dict lookup.

    ``rng_mode`` selects the template variant:
    - ``"uniform"`` — one float32 output per element.
    - ``"normal"`` — two float32 outputs per element (Box-Muller).
    - ``"fused_dropout"`` — uniform RNG + dropout mask + scale in one pass.
    """

    __slots__ = ("rng_mode", "fused_dropout", "__name__", "_per_dtype")

    def __init__(self, rng_mode: str = "uniform", fused_dropout: bool = False):
        self.rng_mode = rng_mode
        self.fused_dropout = fused_dropout
        self.__name__ = f"slang_philox_{rng_mode}{'_dropout' if fused_dropout else ''}"
        self._per_dtype: dict[str, tuple[str, str, str]] = {}

    def _src_key_dtype(self, dtype_s: str) -> tuple[str, str, str]:
        """Return (src, cache_key, output_dtype_str) for the given dtype."""
        cached = self._per_dtype.get(dtype_s)
        if cached is not None:
            return cached

        output_dtype = dtype_s
        src = _render_philox_rng(
            output_dtype=output_dtype,
            rng_mode=self.rng_mode,
            fused_dropout=self.fused_dropout,
        )
        cache_key = (
            f"slang_philox_{self.rng_mode}"
            f"{'_dropout' if self.fused_dropout else ''}"
            f"_{dtype_s}"
        )
        result = (src, cache_key, output_dtype)
        self._per_dtype[dtype_s] = result
        return result

    def __call__(
        self,
        size: list[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        seed_lo: int = 0,
        seed_hi: int = 0,
        offset: int = 0,
        out: torch.Tensor | None = None,
        input_tensor: torch.Tensor | None = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        """Generate random values via the Philox template.

        Args:
            size: Output shape.
            dtype: Output dtype (f32, f16, bf16).
            device: Target Vulkan device.
            seed_lo, seed_hi: 64-bit Philox seed split into two u32s.
            offset: Starting counter offset.
            out: Pre-allocated output tensor (or None to allocate).
            input_tensor: Input for fused dropout mode.
            dropout_p: Dropout probability (fused_dropout only).

        Returns:
            Output tensor with random values.
        """
        if out is None:
            out = pool_acquire(tuple(size), dtype, device)
            if out is None:
                out = torch.empty(size, dtype=dtype, device=device)

        dtype_s = _dtype_to_slang(dtype)
        src, cache_key, output_dtype = self._src_key_dtype(dtype_s)

        # T6.7: ``output2`` is intermediate scratch — Philox normal-mode
        # produces 2 values per counter and we discard the second one.
        # Route it through the buffer pool's ``scratch`` bucket so the
        # storage is reused across training steps instead of leaking via
        # ``torch.empty`` per call.
        output2: torch.Tensor | None = None
        if self.rng_mode == "normal":
            output2 = pool_acquire_scratch(tuple(size), dtype, device)
            if output2 is None:
                output2 = torch.empty(size, dtype=dtype, device=device)

        total_elements = out.numel()

        _dispatch_philox_rng(
            output_dtype=output_dtype,
            rng_mode=self.rng_mode,
            fused_dropout=self.fused_dropout,
            total_elements=total_elements,
            seed_lo=seed_lo,
            seed_hi=seed_hi,
            offset=offset,
            output=out,
            output2=output2,
            input_tensor=input_tensor,
            dropout_p=dropout_p,
            cache_key=cache_key,
        )

        if self.rng_mode == "normal":
            # For the lowering path, just return the first output.
            # _philox_normal generates 2 values per counter; we consume both
            # but only return the first. Future: pack both as complex or
            # return a tuple for fused applications.
            # T6.7: release the discarded second-output scratch back to
            # the pool. The caller drops its local reference immediately.
            if output2 is not None:
                pool_release_scratch(output2)
                output2 = None
            return out
        return out

    def __reduce__(self):
        return (_SlangPhiloxRNG, (self.rng_mode, self.fused_dropout))


def _philox_seed_from_torch() -> tuple[int, int]:
    """Derive a deterministic Philox seed from the global PyTorch RNG state.

    Returns (seed_lo, seed_hi) as two unsigned 32-bit integers. Uses the
    CPU generator's current seed so ``torch.manual_seed(...)`` controls
    reproducibility across Vulkan compiled graphs.
    """
    import hashlib

    gen = torch.default_generator
    raw_state = gen.get_state()
    state_bytes = (
        raw_state.tobytes()
        if hasattr(raw_state, "tobytes")
        else str(raw_state).encode()
    )
    h = hashlib.sha256(state_bytes).digest()[:8]
    seed64 = int.from_bytes(h, "little")
    seed_lo = seed64 & 0xFFFFFFFF
    seed_hi = (seed64 >> 32) & 0xFFFFFFFF
    return seed_lo, seed_hi


def install_external_rng() -> None:
    """Register Vulkan Philox RNG template lowerings for ``aten.rand``,
    ``aten.randn``, ``aten.uniform``, and ``aten.native_dropout``.

    Analogous to ``install_external_mm()`` / ``install_external_addmm()``:
    intercepts the aten ops at the Inductor lowering level and routes them
    through the ``philox_rng.py.jinja`` template instead of the default
    ExternKernel fallback path.

    Safe to call multiple times — only installs once.
    """
    global _rng_installed
    if _rng_installed:
        return
    _rng_installed = True

    import torch
    from torch._inductor import config as inductor_config
    from torch._inductor import lowering as _L
    from torch._inductor.lowering import (
        fallback_handler,
        register_lowering,
    )

    aten = torch.ops.aten

    # Force fallback_random so aten.rand/randn reach our lowering instead
    # of being caught by replace_random.py's FX pass.
    if not inductor_config.fallback_random:
        inductor_config.fallback_random = True

    # Capture original fallbacks (from upstream lowering.py) so we can
    # delegate to them for non-Vulkan tensors.
    _orig_rand_lowering = _L.lowerings.get(aten.rand.default)
    _orig_randn_lowering = _L.lowerings.get(aten.randn.default)
    _orig_uniform_lowering = _L.lowerings.get(aten.uniform.default)
    _orig_dropout_lowering = _L.lowerings.get(aten.native_dropout.default)

    # Pre-build the three template callables.
    _uniform_rng = _SlangPhiloxRNG(rng_mode="uniform")
    _normal_rng = _SlangPhiloxRNG(rng_mode="normal")
    _dropout_rng = _SlangPhiloxRNG(rng_mode="uniform", fused_dropout=True)

    @register_lowering(aten.rand.default, type_promotion_kind=None)
    def _vulkan_rand(
        size,
        *,
        dtype=None,
        layout=None,
        device=None,
        pin_memory=None,
        generator=None,
    ):
        if generator is not None:
            return fallback_handler(aten.rand.generator)(
                size,
                dtype=dtype,
                layout=layout,
                device=device,
                pin_memory=pin_memory,
                generator=generator,
            )
        try:
            dev = device if device is not None else torch.device("cpu")
            if isinstance(dev, str):
                dev = torch.device(dev)
        except Exception:
            dev = torch.device("cpu")

        if dev.type != "vulkan":
            if _orig_rand_lowering is not None:
                return _orig_rand_lowering(
                    size,
                    dtype=dtype,
                    layout=layout,
                    device=device,
                    pin_memory=pin_memory,
                )
            return NotImplemented

        out_dtype = dtype if dtype is not None else torch.float32
        if out_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            if _orig_rand_lowering is not None:
                return _orig_rand_lowering(
                    size,
                    dtype=dtype,
                    layout=layout,
                    device=device,
                    pin_memory=pin_memory,
                )
            return NotImplemented

        # CP.9 / TRAIN.12: Defer to FallbackKernel so the PhiloxState
        # offset advances at runtime instead of being baked in at
        # trace time.  The PrivateUse1 dispatch in philox_dispatch.py
        # calls get_philox_state().advance(n) and passes the dynamic
        # offset to the Slang Philox template.
        import torch.utils._pytree as pytree
        from torch._inductor import ir

        return pytree.tree_map(
            TensorBox.create,
            ir.FallbackKernel.create(
                aten.rand.default,
                size,
                dtype=out_dtype,
                layout=layout,
                device=dev,
                pin_memory=pin_memory,
            ),
        )

    @register_lowering(aten.randn.default, type_promotion_kind=None)
    def _vulkan_randn(
        size,
        *,
        dtype=None,
        layout=None,
        device=None,
        pin_memory=None,
        generator=None,
    ):
        if generator is not None:
            return fallback_handler(aten.randn.generator)(
                size,
                dtype=dtype,
                layout=layout,
                device=device,
                pin_memory=pin_memory,
                generator=generator,
            )
        try:
            dev = device if device is not None else torch.device("cpu")
            if isinstance(dev, str):
                dev = torch.device(dev)
        except Exception:
            dev = torch.device("cpu")

        if dev.type != "vulkan":
            if _orig_randn_lowering is not None:
                return _orig_randn_lowering(
                    size,
                    dtype=dtype,
                    layout=layout,
                    device=device,
                    pin_memory=pin_memory,
                )
            return NotImplemented

        out_dtype = dtype if dtype is not None else torch.float32
        if out_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            if _orig_randn_lowering is not None:
                return _orig_randn_lowering(
                    size,
                    dtype=dtype,
                    layout=layout,
                    device=device,
                    pin_memory=pin_memory,
                )
            return NotImplemented

        # CP.9 / TRAIN.12: Defer to FallbackKernel (see _vulkan_rand).
        import torch.utils._pytree as pytree
        from torch._inductor import ir

        return pytree.tree_map(
            TensorBox.create,
            ir.FallbackKernel.create(
                aten.randn.default,
                size,
                dtype=out_dtype,
                layout=layout,
                device=dev,
                pin_memory=pin_memory,
            ),
        )

    @register_lowering(aten.uniform.default, type_promotion_kind=None)
    def _vulkan_uniform(self, from_=0, to=1, *, generator=None):
        if generator is not None:
            return fallback_handler(aten.uniform.default)(
                self,
                from_=from_,
                to=to,
                generator=generator,
            )
        try:
            dev = self.get_device()
        except Exception:
            dev = torch.device("cpu")

        if dev.type != "vulkan":
            if _orig_uniform_lowering is not None:
                return _orig_uniform_lowering(self, from_=from_, to=to)
            return NotImplemented

        out_dtype = self.get_dtype()
        if out_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            if _orig_uniform_lowering is not None:
                return _orig_uniform_lowering(self, from_=from_, to=to)
            return NotImplemented

        # CP.9 / TRAIN.12: Defer to FallbackKernel (see _vulkan_rand).
        # aten.uniform is in-place; the FallbackKernel handles the
        # copy-back via the PrivateUse1 dispatch which calls self.copy_.
        import torch.utils._pytree as pytree
        from torch._inductor import ir

        return pytree.tree_map(
            TensorBox.create,
            ir.FallbackKernel.create(
                aten.uniform.default,
                self,
                from_,
                to,
            ),
        )

    @register_lowering(aten.native_dropout.default, type_promotion_kind=None)
    def _vulkan_native_dropout(input_tensor, p, train):
        # During Inductor lowering, ``input_tensor`` is an ``ir.TensorBox`` —
        # not a ``torch.Tensor`` — so we cannot call runtime helpers like
        # ``_dropout_rng(...)`` (which would invoke ``.contiguous()`` and
        # ``torch.empty(...)`` on IR nodes). Instead, defer to the
        # ``FallbackKernel`` path, which emits an ``aten.native_dropout``
        # call resolved at runtime to our PrivateUse1 impl in
        # ``philox_dispatch.py:_vulkan_native_dropout`` (which dispatches the
        # fused Philox+dropout Slang shader). This mirrors upstream's
        # ``fallback_random=True`` lowering behaviour (see
        # ``torch/_inductor/lowering.py::native_dropout``).
        try:
            dev = input_tensor.get_device()
        except Exception:
            dev = torch.device("cpu")

        if dev.type != "vulkan":
            if _orig_dropout_lowering is not None:
                return _orig_dropout_lowering(input_tensor, p, train)
            return NotImplemented

        if not train:
            # Eval mode: return input unchanged, mask = all ones (built via
            # the upstream lowering for ``aten.ones`` so the mask is valid IR).
            from torch._inductor.lowering import lowerings as _L_lowerings

            ones_lowering = _L_lowerings.get(aten.ones.default)
            if ones_lowering is None:
                if _orig_dropout_lowering is not None:
                    return _orig_dropout_lowering(input_tensor, p, train)
                return NotImplemented
            mask = ones_lowering(
                list(input_tensor.get_size()),
                dtype=torch.bool,
                device=dev,
                layout=None,
                pin_memory=None,
            )
            return input_tensor, mask

        out_dtype = input_tensor.get_dtype()
        if out_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            if _orig_dropout_lowering is not None:
                return _orig_dropout_lowering(input_tensor, p, train)
            return NotImplemented

        # Defer to FallbackKernel — emits a runtime call to
        # ``aten.native_dropout.default`` which routes to our PrivateUse1
        # Vulkan impl that runs the fused Philox+dropout shader.
        import torch.utils._pytree as pytree
        from torch._inductor import ir
        from torch._inductor.ir import TensorBox

        return pytree.tree_map(
            TensorBox.create,
            ir.FallbackKernel.create(
                aten.native_dropout.default, input_tensor, p, train
            ),
        )

    # ── Pre-warm the Philox template variants ──────────────────────────
    from .runtime import _slangc_available, prewarm_compile

    if _slangc_available():
        rng_specs: list[tuple[str, str]] = []
        for dt in ("float", "half", "uint"):
            for mode, dropout in (
                ("uniform", False),
                ("normal", False),
                ("uniform", True),
            ):
                key = f"slang_philox_{mode}{'_dropout' if dropout else ''}_{dt}"
                src = _render_philox_rng(
                    output_dtype=dt,
                    rng_mode=mode,
                    fused_dropout=dropout,
                )
                rng_specs.append((key, src))
        prewarm_compile(rng_specs, sync=False)


def _slang_tiles_enabled() -> bool:
    """Check if Slang tile matmul shaders are enabled for autotune.

    Disabled by default — the Slang tile mm/bmm/addmm shaders produce
    incorrect forward output (max diff ~28 vs CPU). The ATEN choice
    (C++ vulkan_mm/vulkan_bmm via eager dispatch) is verified correct.
    Set ``TORCH_VULKAN_ENABLE_SLANG_TILES=1`` to re-enable for
    benchmarking / correctness auditing.
    """
    return os.environ.get("TORCH_VULKAN_ENABLE_SLANG_TILES") == "1"


def _get_device_subgroup_size() -> int:
    """Return the Vulkan device's subgroup size (32 or 64).

    N+1.12: Used by tile-config pickers to prefer wave64-friendly
    tile shapes on RDNA1 (subgroup_size=64) and wave32-friendly
    shapes on newer hardware (subgroup_size=32).

    Returns 64 on error / uninitialized.
    """
    try:
        from torch._dynamo.device_interface import get_interface_for_device

        iface = get_interface_for_device("vulkan")
        props = iface.Worker.get_device_properties()
        return props.subgroup_size
    except Exception:
        return 64


def _pick_tile_configs() -> list[tuple[int, int, int]]:
    """Select 1-output-per-thread tile configs based on env var or defaults.

    N+1.12: When subgroup_size=32 (wave32), prefer smaller tiles;
    when subgroup_size=64 (wave64, RDNA1), prefer larger tiles.
    """
    env = os.environ.get("TORCH_VULKAN_MM_TILES", "")
    if env:
        configs = []
        for part in env.split(","):
            part = part.strip()
            if part:
                try:
                    tm, tn, tk = (int(x) for x in part.split("x"))
                    configs.append((tm, tn, tk))
                except ValueError:
                    pass
        return configs if configs else _MM_TILE_CONFIGS

    # N+1.12: wave32 prefers smaller tiles, wave64 prefers all
    sgs = _get_device_subgroup_size()
    if sgs == 32:
        # Prefer tiles where max(tile_m, tile_n) <= 32 for wave32
        return [c for c in _MM_TILE_CONFIGS if max(c[0], c[1]) <= 32]
    return _MM_TILE_CONFIGS


def _pick_register_tile_configs() -> list[tuple[int, int, int, int, int]]:
    """Register-tiled tile configs: (tile_m, tile_n, tile_k, m_per_thread,
    n_per_thread). Each config covers a TILE_M*TILE_N output region with a
    workgroup of (TILE_M/M_PER_THREAD)*(TILE_N/N_PER_THREAD) threads, each
    holding M_PER_THREAD*N_PER_THREAD outputs in registers.

    N+1.12: When subgroup_size=32 (wave32), exclude register-heavy configs
    (m_per_thread * n_per_thread > 16) that would oversubscribe VGPRs.
    When subgroup_size=64 (wave64, RDNA1), keep all configs — wave64
    benefits from larger tile sizes that are multiples of 64.

    Set ``TORCH_VULKAN_NO_REGISTER_TILE=1`` to disable register-tile entries —
    Inductor's autotune will fall back to the (much smaller) legacy 1-output-
    per-thread set + aten_mm. Useful for bisecting register-tile correctness
    issues against the legacy path."""
    if os.environ.get("TORCH_VULKAN_NO_REGISTER_TILE") == "1":
        return []

    # N+1.12: wave32 filters out register-heavy configs
    sgs = _get_device_subgroup_size()
    if sgs == 32:
        return [
            c
            for c in _MM_REGISTER_TILE_CONFIGS
            if c[3] * c[4] <= 16  # m_per_thread * n_per_thread
        ]
    return _MM_REGISTER_TILE_CONFIGS


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
    Slang tiles are disabled (``TORCH_VULKAN_ENABLE_SLANG_TILES != 1``).

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
    from .runtime import (
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


# ═══════════════════════════════════════════════════════════════════════
# T4.7 — Flash attention template infrastructure
# ═══════════════════════════════════════════════════════════════════════

_flash_attention_cache: dict[tuple, str] = {}
_flash_attention_installed = False

# Flash attention autotune variants: (head_dim, num_stages).
# head_dim ∈ {32, 64, 128, 256}; num_stages ∈ {1, 2}.
# The template additionally varies is_causal, output_dtype, and head_layout
# at call time based on input tensors.
_FLASH_ATTENTION_VARIANTS: list[tuple[int, int]] = [
    (32, 1),
    (32, 2),
    (64, 1),
    (64, 2),
    (128, 1),
    (128, 2),
    (256, 1),
    (256, 2),
]


def _render_flash_attention(
    head_dim: int,
    head_layout: str = "bhsd",
    is_causal: bool = True,
    num_stages: int = 1,
    output_dtype: str = "float",
    BK: int = 64,
    BQ: int = 32,
) -> str:
    """Render the flash_attention Jinja2 template.

    Args:
        head_dim: D — attention head dimension (32, 64, 128, 256).
        head_layout: ``"bhsd"`` (head-major, default) or ``"bshd"`` (seq-major).
        is_causal: Whether to apply causal masking (k > q positions → -inf).
        num_stages: Pipeline depth for K/V tile prefetch (1 or 2).
        output_dtype: Slang type string — ``"float"`` (f32), ``"half"`` (f16),
                      or ``"uint"`` (bf16).
        BK: K/V tile size along the S (key) dimension.  Default 64 preserves
            pre-T4.10 behaviour.
        BQ: Q tile size along the N (query) dimension.  Default 32 preserves
            pre-T4.10 behaviour.

    T4.10: ``BK`` and ``BQ`` were previously hard-coded inside the Jinja
    template via ``{% set BK = 64 %}`` / ``{% set BQ = 32 %}``.  They are
    now caller-supplied so autotune can sweep tile configurations; the
    cache key includes them so different tile shapes hash to distinct
    SPIR-V cache entries.
    """
    from jinja2 import Environment

    key = (head_dim, head_layout, is_causal, num_stages, output_dtype, BK, BQ)
    if key in _flash_attention_cache:
        return _flash_attention_cache[key]

    src = _load_slang_template("flash_attention")
    if not src:
        raise RuntimeError("flash_attention.py.jinja template not found")

    env = Environment()
    tmpl = env.from_string(src)
    wg_size = min(head_dim, 256)
    rendered = tmpl.render(
        head_dim=head_dim,
        head_layout=head_layout,
        is_causal=is_causal,
        num_stages=num_stages,
        output_dtype=output_dtype,
        wg_size=wg_size,
        BQ=BQ,
        BK=BK,
    )
    _flash_attention_cache[key] = rendered
    return rendered


def _slang_tile_flash_attention(
    head_dim: int,
    is_causal: bool,
    output_dtype: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
    src: str | None = None,
    cache_key: str | None = None,
    num_stages: int = 1,
    BK: int = 64,
    BQ: int = 32,
) -> None:
    """Execute flash attention via the Slang template shader.

    Q: [B, H, N, D] (head-major, default) or [B, N, H, D] (seq-major).
    K/V: [B, KV_H, S, D] (head-major) or [B, S, KV_H, D] (seq-major).
    O: [B, H, N, D] — output follows Q layout.
    LSE: [B, H, N] — log-sum-exp per query position.

    Detects head_layout from Q strides: if stride(1) == N*D it's
    head-major (bhsd); if stride(1) == H*D it's seq-major (bshd).
    """
    from .runtime import compile_and_dispatch

    assert q.dim() == 4, f"Q must be 4-D, got shape {q.shape}"
    assert k.dim() == 4, f"K must be 4-D, got shape {k.shape}"
    assert v.dim() == 4, f"V must be 4-D, got shape {v.shape}"

    B, H, N, D = q.shape
    KV_H = k.shape[1]
    S = k.shape[2]

    # Detect head layout from Q strides.
    # head-major (bhsd): stride[1] = N*D, stride[2] = D
    # seq-major (bshd):  stride[1] = H*D, stride[2] = 1 (if contiguous) or D
    _hd = q.stride(1)
    _sd = q.stride(2)
    if _hd == N * D and _sd == D:
        head_layout = "bhsd"
    elif _hd == H * D:
        head_layout = "bshd"
    else:
        # Non-contiguous fallback: use contiguous copy.
        if not q.is_contiguous():
            q = q.contiguous()
            _hd = q.stride(1)
            _sd = q.stride(2)
        head_layout = "bhsd" if (_hd == N * D and _sd == D) else "bshd"

    if src is None or cache_key is None:
        src = _render_flash_attention(
            head_dim=D,
            head_layout=head_layout,
            is_causal=is_causal,
            num_stages=num_stages,
            output_dtype=output_dtype,
            BK=BK,
            BQ=BQ,
        )
        cache_key = (
            f"slang_flash_attention_D{D}_{head_layout}"
            f"{'_causal' if is_causal else ''}"
            f"_s{num_stages}_{output_dtype}"
            f"_BK{BK}_BQ{BQ}"
        )

    if not _TRUST_INDUCTOR:
        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()
        if not out.is_contiguous():
            out = out.contiguous()
        if not lse.is_contiguous():
            lse = lse.contiguous()

    # Push constants matching the PC struct in flash_attention.py.jinja:
    # 6 uints (B, H, KV_H, N, S, D) + 1 float (scale) + 2 uints (is_causal, q_layout)
    pc = struct.pack(
        "6IfII",
        B,
        H,
        KV_H,
        N,
        S,
        D,
        float(scale),
        int(is_causal),
        int(head_layout == "bshd"),
    )

    # T4.10: dispatch grid uses the caller-supplied BQ so tile-size
    # autotune correctly sizes the q-tile axis.  (BK does not appear in
    # the grid math — it parameterises the inner kv-loop, not the
    # workgroup sweep.)
    wg_size = min(D, 256)
    grid_x = B
    grid_y = H * ((N + BQ - 1) // BQ)
    grid_z = 1

    compile_and_dispatch(
        src,
        [q, k, v, out, lse],
        grid_x,
        grid_y,
        grid_z,
        push_constants=pc,
        num_outputs=2,
        cache_key=cache_key,
    )


class _SlangTileFlashAttention:
    """Picklable callable for Slang flash attention — Inductor's codecache
    pickles the external choices list as part of the cache key. Closures are
    not picklable; a module-level class with instance state is.

    Caches the rendered Slang source + cache_key per dtype on the instance
    so the per-dispatch path skips the Jinja-render dict lookup.
    """

    __slots__ = (
        "head_dim",
        "num_stages",
        "is_causal",
        "BK",
        "BQ",
        "__name__",
        "_per_dtype",
    )

    def __init__(
        self,
        head_dim: int,
        num_stages: int = 1,
        is_causal: bool = True,
        BK: int = 64,
        BQ: int = 32,
    ):
        self.head_dim = head_dim
        self.num_stages = num_stages
        self.is_causal = is_causal
        # T4.10: BK/BQ are tile sizes wired through to the Jinja template.
        # They participate in the SPIR-V cache key so different tile configs
        # produce distinct cached binaries.
        self.BK = BK
        self.BQ = BQ
        self.__name__ = (
            f"slang_flash_attention_D{head_dim}"
            f"{'_causal' if is_causal else ''}"
            f"_s{num_stages}_BK{BK}_BQ{BQ}"
        )
        self._per_dtype: dict[str, tuple[str, str]] = {}

    def _src_and_key(self, output_dtype: str) -> tuple[str, str]:
        cached = self._per_dtype.get(output_dtype)
        if cached is None:
            src = _render_flash_attention(
                head_dim=self.head_dim,
                head_layout="bhsd",
                is_causal=self.is_causal,
                num_stages=self.num_stages,
                output_dtype=output_dtype,
                BK=self.BK,
                BQ=self.BQ,
            )
            cache_key = (
                f"slang_flash_attention_D{self.head_dim}"
                f"_bhsd"
                f"{'_causal' if self.is_causal else ''}"
                f"_s{self.num_stages}_{output_dtype}"
                f"_BK{self.BK}_BQ{self.BQ}"
            )
            cached = (src, cache_key)
            self._per_dtype[output_dtype] = cached
        return cached

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        is_causal: bool = True,
        *,
        out: torch.Tensor | None = None,
        lse: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute flash attention via the template.

        Returns:
            Output tensor O (same shape as Q).
        """
        if out is None:
            out = pool_acquire(tuple(q.shape), q.dtype, q.device)
            if out is None:
                out = torch.empty_like(q)
        # T6.7: when ``lse`` is caller-supplied, treat it as caller-owned
        # (the bwd consumer needs it). When we allocate it ourselves, it
        # is internal scratch — only ``out`` is returned. Route the
        # scratch path through the pool's ``scratch`` bucket so the LSE
        # storage is reused across calls instead of leaking per dispatch.
        lse_scratch_owned = False
        if lse is None:
            B, H, N, _ = q.shape
            lse = pool_acquire_scratch((B, H, N), torch.float32, q.device)
            if lse is None:
                lse = torch.empty(B, H, N, dtype=torch.float32, device=q.device)
            lse_scratch_owned = True

        dtype_s = _dtype_to_slang(q.dtype)
        src, cache_key = self._src_and_key(dtype_s)
        _slang_tile_flash_attention(
            head_dim=q.shape[-1],
            is_causal=is_causal if is_causal is not None else self.is_causal,
            output_dtype=dtype_s,
            q=q,
            k=k,
            v=v,
            out=out,
            lse=lse,
            scale=scale,
            src=src,
            cache_key=cache_key,
            num_stages=self.num_stages,
            BK=self.BK,
            BQ=self.BQ,
        )
        if lse_scratch_owned:
            pool_release_scratch(lse)
            lse = None
        return out

    def __reduce__(self):
        # T4.10: include BK/BQ so pickled choice instances round-trip the
        # tile-size configuration.
        return (
            _SlangTileFlashAttention,
            (self.head_dim, self.num_stages, self.is_causal, self.BK, self.BQ),
        )


def _make_tile_flash_attention_fn(
    head_dim: int,
    num_stages: int = 1,
    is_causal: bool = True,
    BK: int = 64,
    BQ: int = 32,
) -> _SlangTileFlashAttention:
    return _SlangTileFlashAttention(head_dim, num_stages, is_causal, BK, BQ)


# ═══════════════════════════════════════════════════════════════════════
# CG.M7 — SDPA backward via flash_attention_bwd.py.jinja
# ═══════════════════════════════════════════════════════════════════════

_flash_attention_bwd_cache: dict[tuple, str] = {}


def _render_flash_attention_bwd(
    head_dim: int,
    head_layout: str = "bhsd",
    is_causal: bool = True,
    BK: int = 64,
    BQ: int = 32,
) -> str:
    """Render the flash_attention_bwd Jinja2 template.

    Args:
        head_dim: D — attention head dimension (32, 64, 128, 256).
        head_layout: ``"bhsd"`` (head-major, default) or ``"bshd"`` (seq-major).
        is_causal: Whether to apply causal masking.
        BK: K/V tile size along the S (key) dimension.
        BQ: Q tile size along the N (query) dimension.
    """
    from jinja2 import Environment

    key = (head_dim, head_layout, is_causal, BK, BQ)
    if key in _flash_attention_bwd_cache:
        return _flash_attention_bwd_cache[key]

    src = _load_slang_template("flash_attention_bwd")
    if not src:
        raise RuntimeError("flash_attention_bwd.py.jinja template not found")

    env = Environment()
    tmpl = env.from_string(src)
    wg_size = min(head_dim, 256)
    rendered = tmpl.render(
        head_dim=head_dim,
        head_layout=head_layout,
        is_causal=is_causal,
        wg_size=wg_size,
        BQ=BQ,
        BK=BK,
    )
    _flash_attention_bwd_cache[key] = rendered
    return rendered


def _dispatch_flash_attention_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lse: torch.Tensor,
    dO: torch.Tensor,
    dQ: torch.Tensor,
    dK: torch.Tensor,
    dV: torch.Tensor,
    scale: float,
    is_causal: bool = True,
    *,
    BK: int = 64,
    BQ: int = 32,
) -> None:
    """Execute SDPA backward via the CG.M7 flash_attention_bwd template.

    Computes dQ, dK, dV from saved Q, K, V, LSE and the output gradient dO.

    Args:
        q:     [B, H, N, D] — saved forward query
        k:     [B, KV_H, S, D] — saved forward key
        v:     [B, KV_H, S, D] — saved forward value
        lse:   [B, H, N] — log-sum-exp from forward
        dO:    [B, H, N, D] — gradient of loss w.r.t. output
        dQ:    [B, H, N, D] — output: gradient w.r.t. Q (pre-allocated)
        dK:    [B, KV_H, S, D] — output: gradient w.r.t. K (pre-allocated, zero-init)
        dV:    [B, KV_H, S, D] — output: gradient w.r.t. V (pre-allocated, zero-init)
        scale: 1/sqrt(D) or user-specified
        is_causal: whether causal masking was used in forward
    """
    from .runtime import compile_and_dispatch

    assert q.dim() == 4, f"Q must be 4-D, got shape {q.shape}"
    B, H, N, D = q.shape
    KV_H = k.shape[1]
    S = k.shape[2]

    # Detect head layout from Q strides.
    _hd = q.stride(1)
    _sd = q.stride(2)
    if _hd == N * D and _sd == D:
        head_layout = "bhsd"
    elif _hd == H * D:
        head_layout = "bshd"
    else:
        head_layout = "bhsd"  # default fallback

    src = _render_flash_attention_bwd(
        head_dim=D,
        head_layout=head_layout,
        is_causal=is_causal,
        BK=BK,
        BQ=BQ,
    )
    cache_key = (
        f"slang_flash_attention_bwd_D{D}_{head_layout}"
        f"{'causal' if is_causal else ''}"
        f"_BK{BK}_BQ{BQ}"
    )

    # Push constants matching BwdPC struct.
    pc = struct.pack(
        "6IfII",
        B,
        H,
        KV_H,
        N,
        S,
        D,
        float(scale),
        int(is_causal),
        int(head_layout == "bshd"),
    )

    wg_size = min(D, 256)
    grid_x = B
    grid_y = H * ((N + BQ - 1) // BQ)
    grid_z = 1

    compile_and_dispatch(
        src,
        [q, k, v, lse, dO, dQ, dK, dV],
        grid_x,
        grid_y,
        grid_z,
        push_constants=pc,
        num_outputs=3,
        cache_key=cache_key,
    )


class _SlangTileFlashAttentionBwd:
    """Picklable callable for CG.M7 SDPA backward dispatch.

    Caches the rendered Slang source + cache_key per configuration so the
    per-dispatch path skips the Jinja-render dict lookup.
    """

    __slots__ = (
        "head_dim",
        "is_causal",
        "BK",
        "BQ",
        "__name__",
        "_src",
        "_cache_key",
    )

    def __init__(
        self,
        head_dim: int,
        is_causal: bool = True,
        BK: int = 64,
        BQ: int = 32,
    ):
        self.head_dim = head_dim
        self.is_causal = is_causal
        self.BK = BK
        self.BQ = BQ
        self.__name__ = (
            f"slang_flash_attention_bwd_D{head_dim}"
            f"{'causal' if is_causal else ''}"
            f"_BK{BK}_BQ{BQ}"
        )
        self._src: str | None = None
        self._cache_key: str | None = None

    def _ensure_source(self) -> tuple[str, str]:
        if self._src is None:
            self._src = _render_flash_attention_bwd(
                head_dim=self.head_dim,
                head_layout="bhsd",
                is_causal=self.is_causal,
                BK=self.BK,
                BQ=self.BQ,
            )
            self._cache_key = (
                f"slang_flash_attention_bwd_D{self.head_dim}"
                f"_bhsd"
                f"{'causal' if self.is_causal else ''}"
                f"_BK{self.BK}_BQ{self.BQ}"
            )
        return self._src, self._cache_key

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        lse: torch.Tensor,
        dO: torch.Tensor,
        scale: float,
        is_causal: bool = True,
        *,
        dQ: torch.Tensor | None = None,
        dK: torch.Tensor | None = None,
        dV: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute SDPA backward via the CG.M7 template.

        Returns:
            (dQ, dK, dV) — gradients w.r.t. Q, K, V.
        """
        if dQ is None:
            dQ = torch.empty_like(q)
        if dK is None:
            dK = torch.zeros_like(k)
        if dV is None:
            dV = torch.zeros_like(v)

        self._ensure_source()
        _dispatch_flash_attention_bwd(
            q=q,
            k=k,
            v=v,
            lse=lse,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            scale=scale,
            is_causal=is_causal if is_causal is not None else self.is_causal,
            BK=self.BK,
            BQ=self.BQ,
        )
        return dQ, dK, dV

    def __reduce__(self):
        return (
            _SlangTileFlashAttentionBwd,
            (self.head_dim, self.is_causal, self.BK, self.BQ),
        )


def install_external_flash_attention() -> None:
    """Wire the Slang flash attention template into the SDPA lowering path.

    Registers a lowering for ``torch_vulkan::flash_attention_fused`` (the
    custom op that the SDPA FX pattern rewrites to) that dispatches through
    the flash_attention.py.jinja template instead of the C++ eager extern.

    The template produces both output O and LSE (log-sum-exp). LSE is stored
    as an extra output from the Inductor IR node so AOT Autograd can use it
    for the backward pass.

    Safe to call multiple times — only installs once.
    """
    global _flash_attention_installed
    if _flash_attention_installed:
        return
    _flash_attention_installed = True

    from torch._inductor import lowering as _L
    from torch._inductor.select_algorithm import (
        ExternKernelChoice,
        autotune_select_algorithm,
    )

    # Ensure the custom op is registered and import it.
    from .fx_passes.eager_patches import _ensure_flash_attention_op_registered

    flash_op = _ensure_flash_attention_op_registered()

    # Build autotune choices: each (head_dim, num_stages) pair.
    flash_tile_fns: list[_SlangTileFlashAttention] = [
        _make_tile_flash_attention_fn(hd, ns) for hd, ns in _FLASH_ATTENTION_VARIANTS
    ]

    @_L.register_lowering(flash_op, type_promotion_kind=None)
    def _vulkan_tuned_flash_attention(q, k, v, scale, is_causal, *, layout=None):
        dev = q.get_device()
        if dev is not None and dev.type != "vulkan":
            # Fall back to eager extern for non-Vulkan devices.
            import torch_vulkan

            return torch_vulkan.flash_attention(q, k, v, float(scale), bool(is_causal))

        from torch._inductor.kernel_inputs import MMKernelInputs
        from torch._inductor.select_algorithm import (
            ExternKernelChoice,
            autotune_select_algorithm,
        )

        # Use the eager C++ extern as a fallback choice.
        def _eager_flash(q, k, v, scale, is_causal, *, out=None):
            import torch_vulkan

            result = torch_vulkan.flash_attention(
                q, k, v, float(scale), bool(is_causal)
            )
            if out is not None:
                out.copy_(result)
                return out
            return result

        _eager_flash.__name__ = "aten_flash_attention"

        kernel_inputs = MMKernelInputs(
            [q, k, v], scalars={"scale": scale, "is_causal": is_causal}
        )
        tensor_nodes = kernel_inputs.nodes()
        choices = []

        # Add eager extern as first choice (verified correct).
        choices.append(ExternKernelChoice(_eager_flash).bind(tensor_nodes, layout))

        # Add template variants filtered by head_dim match.
        head_dim = q.get_size()[-1]
        for fn in flash_tile_fns:
            if fn.head_dim == head_dim:
                choices.append(ExternKernelChoice(fn).bind(tensor_nodes, layout))

        # Same defensive tuple-or-TensorBox unpack as ``_vulkan_tuned_addmm``;
        # see comment there.
        result = autotune_select_algorithm(
            "flash_attention", choices, tensor_nodes, layout
        )
        if isinstance(result, tuple):
            node, _ = result
        else:
            node = result
        return node

    # ── Pre-warm the flash attention template variants ──────────────
    from .runtime import _slangc_available, prewarm_compile

    if _slangc_available():
        fa_specs: list[tuple[str, str]] = []
        for dt in ("float", "half", "uint"):
            for hd, ns in _FLASH_ATTENTION_VARIANTS:
                for causal in (True, False):
                    key = (
                        f"slang_flash_attention_D{hd}_bhsd"
                        f"{'_causal' if causal else ''}"
                        f"_s{ns}_{dt}"
                    )
                    src = _render_flash_attention(
                        head_dim=hd,
                        head_layout="bhsd",
                        is_causal=causal,
                        num_stages=ns,
                        output_dtype=dt,
                    )
                    fa_specs.append((key, src))
        prewarm_compile(fa_specs, sync=False)


# ═══════════════════════════════════════════════════════════════════════
# T4.12 / OP.10 — FFT Stockham template infrastructure
# ═══════════════════════════════════════════════════════════════════════

# FFT sizes the template supports. Each workgroup computes one 1-D FFT
# using groupshared memory, so N must fit within the GPU's shared-memory
# budget. 2*N floats + N twiddle floats = 3*N*4 bytes. With 64 KB LDS:
#   3*N*4 ≤ 65536  →  N ≤ 5461
# We limit to N ≤ 2048 for safety (occupancy and subgroup pressure).
_FFT_STOCKHAM_SIZES: list[int] = [64, 128, 256, 512, 1024]

_fft_cache: dict[tuple, str] = {}
_fft_installed = False


def _render_fft_stockham(
    N: int,
    direction: str = "forward",
    normalized: bool = False,
    dtype: str = "float",
) -> str:
    """Render the fft_stockham Jinja2 template.

    Args:
        N: FFT size (must be a power of 2).
        direction: ``"forward"`` (exp(-2πi)) or ``"inverse"`` (exp(+2πi)).
        normalized: If True, scale output by 1/sqrt(N).
        dtype: Slang type string for storage (``"float"``).

    Returns:
        Rendered Slang source string ready for SPIR-V compilation.
    """
    from jinja2 import Environment

    if N & (N - 1) != 0:
        raise ValueError(f"FFT size N={N} must be a power of 2")

    half_N = N // 2
    log2_N = N.bit_length() - 1
    n_threads = min(half_N, 256)
    dir_sign = -1.0 if direction == "forward" else 1.0
    norm_scale: float | None = 1.0 / (N**0.5) if normalized else None

    key = (N, direction, normalized, dtype)
    if key in _fft_cache:
        return _fft_cache[key]

    src = _load_slang_template("fft_stockham")
    if not src:
        raise RuntimeError("fft_stockham.py.jinja template not found")

    env = Environment()
    tmpl = env.from_string(src)
    rendered = tmpl.render(
        N=N,
        half_N=half_N,
        log2_N=log2_N,
        n_threads=n_threads,
        dir_sign=dir_sign,
        norm_scale=norm_scale,
        dtype=dtype,
    )
    _fft_cache[key] = rendered
    return rendered


class _SlangTileFFT:
    """Picklable callable for FFT Stockham template dispatch.

    Each instance is configured for a specific N and direction.  The
    callable interface matches what ``ExternKernelChoice`` expects:
    ``fn(input, *, out=None) -> output``.

    The FFT is always applied along the last dimension (dim=-1).
    Input and output are interleaved complex as float pairs [re, im]
    stored in a 1-D or 2-D StructuredBuffer.
    """

    def __init__(self, N: int, *, direction: str = "forward", normalized: bool = False):
        if N & (N - 1) != 0:
            raise ValueError(f"_SlangTileFFT: N={N} must be a power of 2")
        self.N = N
        self.direction = direction
        self.normalized = normalized
        self._dtype_cache: dict[str, tuple[str, str]] = {}
        # Name must match what ExternKernelChoice uses for autotune logging.
        dir_label = "fwd" if direction == "forward" else "inv"
        norm_label = "_norm" if normalized else ""
        self.__name__ = f"slang_fft_stockham_{N}_{dir_label}{norm_label}"

    def _src_and_key(self, dtype_s: str) -> tuple[str, str]:
        cached = self._dtype_cache.get(dtype_s)
        if cached is not None:
            return cached

        src = _render_fft_stockham(
            self.N,
            direction=self.direction,
            normalized=self.normalized,
            dtype=dtype_s,
        )
        dir_label = "fwd" if self.direction == "forward" else "inv"
        norm_label = "_norm" if self.normalized else ""
        cache_key = f"slang_fft_{self.N}_{dir_label}{norm_label}_{dtype_s}"
        cached = (src, cache_key)
        self._dtype_cache[dtype_s] = cached
        return cached

    def __call__(
        self,
        x: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute FFT via Slang Stockham template.

        Args:
            x: Complex input tensor.  The FFT dimension must be the last
               dimension and must equal ``self.N``.  Shape ``(batch, N)`` or ``(N,)``.
            out: Optional output tensor.  If None, allocated internally.

        Returns:
            Complex output tensor, same shape as input.
        """
        return _dispatch_fft(
            x, out=out, N=self.N, direction=self.direction, normalized=self.normalized
        )


def _dispatch_fft(
    x: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    N: int,
    direction: str = "forward",
    normalized: bool = False,
) -> torch.Tensor:
    """Dispatch a single Stockham FFT kernel for complex-to-complex 1-D FFT.

    Args:
        x: Complex input. Must be contiguous and on Vulkan.
           Shape: ``(batch, N)`` or ``(N,)``.
        out: Pre-allocated output (same shape/dtype/device).
        N: FFT size (must match x.shape[-1]).
        direction: ``"forward"`` or ``"inverse"``.
        normalized: If True, scale by 1/sqrt(N).

    Returns:
        Complex output tensor.
    """
    from .runtime import compile_and_dispatch

    assert x.device.type == "vulkan", (
        f"_dispatch_fft: expected Vulkan tensor, got {x.device}"
    )
    assert x.is_complex(), "_dispatch_fft: input must be complex"
    assert x.shape[-1] == N, f"_dispatch_fft: N={N} but x.shape[-1]={x.shape[-1]}"

    x_real = torch.view_as_real(x.contiguous())
    batch = x.numel() // N

    dtype_s = _dtype_to_slang(x_real.dtype)
    src = _render_fft_stockham(
        N, direction=direction, normalized=normalized, dtype=dtype_s
    )
    dir_label = "fwd" if direction == "forward" else "inv"
    norm_label = "_norm" if normalized else ""
    cache_key = f"slang_fft_{N}_{dir_label}{norm_label}_{dtype_s}"

    if out is None:
        out_real = torch.empty_like(x_real)
    else:
        out_real = torch.view_as_real(out.contiguous())

    pc = struct.pack(
        "4I",
        batch,  # batch_in
        2 * N,  # in_stride
        2 * N,  # out_stride
        batch,  # batch_out
    )

    # One workgroup per batch element
    grid_x = batch

    compile_and_dispatch(
        src,
        [x_real, out_real],
        grid_x,
        1,
        1,
        push_constants=pc,
        num_outputs=1,
        cache_key=cache_key,
    )

    result = torch.view_as_complex(out_real.contiguous())
    return result.view_as(x)


def install_external_fft() -> None:
    """Register Stockham FFT template as an external FFT choice for Inductor.

    Registers lowerings for ``aten._fft_c2c``, ``aten._fft_r2c``, and
    ``aten._fft_c2r`` that route through the Stockham template for small
    power-of-2 sizes.  The eager C++ path (wired in Registration.cpp via
    OP.4-eager-followup) serves as the fallback for larger or non-power-of-2
    sizes.

    Safe to call multiple times — only installs once.
    """
    global _fft_installed
    if _fft_installed:
        return
    _fft_installed = True

    from torch._inductor import lowering as _L
    from torch._inductor.select_algorithm import ExternKernelChoice

    aten = torch.ops.aten

    # Build template callables for each supported N
    fft_c2c_fns: dict[int, _SlangTileFFT] = {}
    fft_c2c_inv_fns: dict[int, _SlangTileFFT] = {}
    for N in _FFT_STOCKHAM_SIZES:
        fft_c2c_fns[N] = _SlangTileFFT(N, direction="forward")
        fft_c2c_inv_fns[N] = _SlangTileFFT(N, direction="inverse")

    # ── _fft_c2c lowering ───────────────────────────────────────────
    # The underlying C++ implementation is already registered at
    # PrivateUse1 (OP.4-eager-followup), so the fallback works.
    # This lowering adds the Stockham template as a faster option
    # for power-of-2 sizes ≤ 1024.
    _fft_c2c_lowered = False

    def _ensure_fft_c2c_lowered():
        nonlocal _fft_c2c_lowered
        if _fft_c2c_lowered:
            return
        _fft_c2c_lowered = True

        orig = _L.lowerings.get(aten._fft_c2c.default)

        @_L.register_lowering(aten._fft_c2c, type_promotion_kind=None)
        def _vulkan_fft_c2c(x, dim, normalization, forward, *, layout=None):
            dev = x.get_device()
            if dev is not None and dev.type != "vulkan":
                if orig is not None:
                    return orig(x, dim, normalization, forward, layout=layout)
                return torch._fft_c2c(x, dim, normalization, forward)

            from torch._inductor.kernel_inputs import MMKernelInputs
            from torch._inductor.select_algorithm import (
                ExternKernelChoice,
                autotune_select_algorithm,
            )

            # Use the eager C++ extern as fallback.
            def _eager_fft_c2c(x, dim, normalization, forward, *, out=None):
                result = torch._fft_c2c(x, dim, normalization, forward)
                if out is not None:
                    out.copy_(result)
                    return out
                return result

            _eager_fft_c2c.__name__ = "aten_fft_c2c"

            kernel_inputs = MMKernelInputs(
                [x],
                scalars={
                    "dim": dim,
                    "normalization": normalization,
                    "forward": forward,
                },
            )
            tensor_nodes = kernel_inputs.nodes()
            choices = [ExternKernelChoice(_eager_fft_c2c).bind(tensor_nodes, layout)]

            # Add template choices for power-of-2 N that match.
            # Note: FakeTensor IR nodes carry symbolic shapes — we can't
            # query the runtime N here. Template matching by N is deferred
            # to autotune time (the extern kernel checks N at runtime).
            for N, fn in fft_c2c_fns.items():
                choices.append(ExternKernelChoice(fn).bind(tensor_nodes, layout))

            result = autotune_select_algorithm("fft_c2c", choices, tensor_nodes, layout)
            if isinstance(result, tuple):
                node, _ = result
            else:
                node = result
            return node

    _ensure_fft_c2c_lowered()


# ═══════════════════════════════════════════════════════════════════════
# T4.8 — Foreach optimizer template infrastructure
# ═══════════════════════════════════════════════════════════════════════

# Batch sizes the template can be rendered for.  The template unrolls
# over `batch_size` params per dispatch (each param gets its own binding
# slot).  We pre-select a few common sizes; callers pick the smallest
# size that fits their parameter count.
_OPTIMIZER_BATCH_SIZES = (1, 7, 15, 21, 32)


def _foreach_use_parameter_array() -> bool:
    """Whether the foreach optimizer template should emit the N+1.5.b
    ``ParamSlot params[BATCH_SIZE]`` array-of-structs path with
    descriptor-array bindings (descriptorCount=BATCH_SIZE per buffer
    family) instead of the round-3 flat-binding + switch-cascade layout.

    The new path requires:
      (a) VK_EXT_descriptor_indexing on the device
          (probed via ``_C._descriptor_indexing_enabled``);
      (b) The N+1.5.a Python wiring that extracts per-binding
          ``descriptorCount`` from reflection JSON and routes the
          dispatch through ``_C._jit_dispatch_indexed`` instead of the
          flat ``_C._jit_dispatch`` FFI.

    Until (b) lands, the master gate is the ``TORCH_VULKAN_PARAMETER_ARRAY=1``
    env var (default off via ``config.parameter_array()``).  We
    additionally probe (a) so flipping the env var on a device without
    descriptor indexing degrades gracefully back to the switch-cascade
    layout instead of failing pipeline creation.

    Caveat: when the env var is on AND the device supports descriptor
    indexing AND ``_C._jit_dispatch_indexed`` exists in the FFI but the
    runtime's ``dispatch()`` helper still routes through the flat
    ``_jit_dispatch`` (i.e., N+1.5.a is in flight but not finished),
    the dispatch will silently bind only the first buffer of each
    descriptor array.  This is documented in the template caveat at
    ``foreach_optimizer.py.jinja:14`` and is the user's responsibility
    when opting into the flag pre-N+1.5.a.
    """
    from . import config as _cfg

    if not _cfg.parameter_array():
        return False

    # Probe the C++ runtime side.  If the descriptor-indexing device
    # capability is missing or the indexed-dispatch FFI is absent,
    # silently fall back — flipping the flag on without C++ support
    # would break pipeline creation with a confusing binding-mismatch
    # error at first dispatch.
    try:
        import torch_vulkan as _tv

        ce = getattr(_tv, "_c_ext", None)
        if ce is None:
            return False
        probe = getattr(ce, "_descriptor_indexing_enabled", None)
        if probe is None or not probe():
            return False
        if not hasattr(ce, "_jit_dispatch_indexed"):
            return False
    except Exception:
        return False
    return True


def _render_foreach_optimizer_slang(
    algorithm: str,
    batch_size: int,
    output_dtype: str = "float",
    parameter_array: bool | None = None,
) -> str:
    """Render the foreach_optimizer.py.jinja template for a given
    (algorithm, batch_size, output_dtype) combination.

    The template is purely parameterized — no cache-key collision with
    mm because the source is keyed by the rendering parameters.

    ``parameter_array`` (N+1.5.b): if True, emit the
    ``ParamSlot params[BATCH_SIZE]`` array-of-structs path.  If None
    (default), consult the runtime via
    ``_foreach_use_parameter_array()`` so behaviour matches the active
    config flag + C++ capability without an extra plumbing argument at
    every call site.
    """
    from jinja2 import Environment

    source_template = _load_slang_template("foreach_optimizer")
    if not source_template:
        raise RuntimeError(
            "foreach_optimizer.py.jinja template not found — "
            "is the Vulkan Slang backend installed correctly?"
        )
    if parameter_array is None:
        parameter_array = _foreach_use_parameter_array()
    env = Environment()
    tmpl = env.from_string(source_template)
    return tmpl.render(
        algorithm=algorithm,
        batch_size=batch_size,
        output_dtype=output_dtype,
        parameter_array=parameter_array,
    )


def _foreach_cache_key(
    algorithm: str,
    batch_size: int,
    output_dtype: str,
    parameter_array: bool,
) -> str:
    """Build the SPIR-V / pipeline cache key for a foreach variant.

    The ``parameter_array`` layout produces a different SPIR-V
    descriptor layout (descriptorCount=N vs N flat bindings), so the
    cache key MUST distinguish the two.
    """
    suffix = "_pa" if parameter_array else ""
    return f"slang_foreach_{algorithm}_{batch_size}_{output_dtype}{suffix}"


def _slang_foreach_optimizer(
    algorithm: str,
    batch_size: int,
    output_dtype: str,
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    src: str | None = None,
    cache_key: str | None = None,
    momentum_bufs: list[torch.Tensor] | None = None,
    v_bufs: list[torch.Tensor] | None = None,
    lr: list[float] | None = None,
    weight_decay: list[float] | None = None,
    momentum: list[float] | None = None,
    beta2: list[float] | None = None,
    eps: list[float] | None = None,
) -> None:
    """Dispatch the foreach optimizer template for a batch of params.

    This is the low-level dispatch.  The caller is responsible for
    batching params into groups ≤ `batch_size` and providing the
    rendered source / cache_key.

    All per-param scalar lists must have length equal to `len(params)`.
    """
    from .runtime import compile_and_dispatch

    n_params = len(params)
    assert n_params <= batch_size, (
        f"_slang_foreach_optimizer: {n_params} params exceeds "
        f"rendered batch_size {batch_size}"
    )

    if src is None or cache_key is None:
        use_pa = _foreach_use_parameter_array()
        src = _render_foreach_optimizer_slang(
            algorithm, batch_size, output_dtype, parameter_array=use_pa
        )
        cache_key = _foreach_cache_key(algorithm, batch_size, output_dtype, use_pa)

    # Validate all params have the same shape and dtype.
    numel = params[0].numel()
    dtype = params[0].dtype
    for p in params:
        if p.numel() != numel:
            raise ValueError(
                f"foreach optimizer: all params must have same numel, "
                f"got {numel} vs {p.numel()}"
            )
        if p.dtype != dtype:
            raise ValueError(
                f"foreach optimizer: all params must have same dtype, "
                f"got {dtype} vs {p.dtype}"
            )

    # Build push constants.
    # Layout: uint n_params, uint _pad[3], then ParamConfig[n].
    pc_parts: list[bytes] = []
    pc_parts.append(struct.pack("4I", n_params, 0, 0, 0))

    default_lr = [0.01]
    default_wd = [0.0]
    default_momentum = [0.9]
    default_beta2 = [0.999]
    default_eps = [1e-8]

    lr_list = lr if lr is not None else default_lr * n_params
    wd_list = weight_decay if weight_decay is not None else default_wd * n_params
    mom_list = momentum if momentum is not None else default_momentum * n_params
    b2_list = beta2 if beta2 is not None else default_beta2 * n_params
    eps_list = eps if eps is not None else default_eps * n_params

    for i in range(n_params):
        pc_parts.append(struct.pack("If", params[i].numel(), lr_list[i]))
        pc_parts.append(struct.pack("f", wd_list[i]))
        if algorithm in ("sgd_momentum", "adamw", "lion"):
            pc_parts.append(struct.pack("f", mom_list[i]))
        if algorithm == "adamw":
            pc_parts.append(struct.pack("f", b2_list[i]))
            pc_parts.append(struct.pack("f", eps_list[i]))
        if algorithm == "lion":
            pc_parts.append(struct.pack("f", b2_list[i]))

    # Pad remaining ParamConfig slots for unused entries.
    for _ in range(n_params, batch_size):
        pc_parts.append(struct.pack("Iff", 0, 0.0, 0.0))
        if algorithm in ("sgd_momentum", "adamw", "lion"):
            pc_parts.append(struct.pack("f", 0.0))
        if algorithm == "adamw":
            pc_parts.append(struct.pack("f", 0.0))
            pc_parts.append(struct.pack("f", 0.0))
        if algorithm == "lion":
            pc_parts.append(struct.pack("f", 0.0))

    pc_bytes = b"".join(pc_parts)

    # Build buffer list: [param0, grad0, param1, grad1, ...].
    # Padding slots use a real-storage dummy (size 1, not 0) because the
    # runtime rejects null-storage tensors at dispatch (PF.51). The shader
    # bounds-checks `param_idx >= pc.n_params` so unused slots are never
    # read, but Vulkan still requires bound buffers to have valid storage.
    # T6.7: route the padding-dummy through the buffer-pool ``scratch``
    # bucket — the optimizer fires every step and a fresh ``torch.empty``
    # leaks one PrivateUse1 round-trip per call.
    dummy = pool_acquire_scratch((1,), dtype, params[0].device)
    if dummy is None:
        dummy = torch.empty(1, device=params[0].device, dtype=dtype)
    buffers: list[torch.Tensor] = []
    for i in range(n_params):
        buffers.append(params[i])
        buffers.append(grads[i])
    for _ in range(n_params, batch_size):
        buffers.append(dummy)  # param slot
        buffers.append(dummy)  # grad slot

    def _pad(lst: list[torch.Tensor] | None) -> list[torch.Tensor]:
        if lst is None:
            return [dummy] * batch_size
        return list(lst) + [dummy] * (batch_size - len(lst))

    # Add momentum / v buffers — always batch_size entries.
    if algorithm == "sgd_momentum":
        buffers.extend(_pad(momentum_bufs))
    elif algorithm == "adamw":
        buffers.extend(_pad(momentum_bufs))
        buffers.extend(_pad(v_bufs))
    elif algorithm == "lion":
        buffers.extend(_pad(momentum_bufs))

    # Grid: X = elements per param, Y = n_params.
    threadgroup_size = 256
    grid_x = (numel + threadgroup_size - 1) // threadgroup_size
    grid_y = n_params
    grid_z = 1

    # num_outputs: all param buffers + momentum/v buffers are mutated.
    num_outputs = n_params
    if algorithm == "sgd_momentum":
        num_outputs += n_params
    elif algorithm == "adamw":
        num_outputs += 2 * n_params
    elif algorithm == "lion":
        num_outputs += n_params

    compile_and_dispatch(
        src,
        buffers,
        grid_x,
        grid_y,
        grid_z,
        push_constants=pc_bytes,
        num_outputs=num_outputs,
        cache_key=cache_key,
    )
    # T6.7: dispatch is done; the padding ``dummy`` is dead. Drop the
    # buffer-list reference and return the storage to the scratch pool.
    buffers.clear()
    pool_release_scratch(dummy)
    dummy = None  # type: ignore[assignment]


class _SlangForeachOptimizer:
    """Picklable callable for a rendered foreach optimizer template.

    Each instance is bound to a specific (algorithm, batch_size,
    output_dtype) combination.  The `__call__` signature takes tensor
    lists and scalar lists and dispatches the template, automatically
    batching across multiple dispatches if there are more params than
    `batch_size`.

    Source is cached at module level per (algorithm, batch_size, output_dtype)
    so multiple instances with the same key share the rendered source.
    """

    __slots__ = (
        "algorithm",
        "batch_size",
        "output_dtype",
        "__name__",
        "_src",
        "_cache_key",
    )

    _src_cache: dict[tuple, tuple[str, str]] = {}

    def __init__(self, algorithm: str, batch_size: int, output_dtype: str = "float"):
        self.algorithm = algorithm
        self.batch_size = batch_size
        self.output_dtype = output_dtype
        self.__name__ = f"slang_foreach_{algorithm}_{batch_size}_{output_dtype}"
        self._src: str | None = None
        self._cache_key: str | None = None
        self._ensure_source()

    def _ensure_source(self) -> None:
        # N+1.5.b: include the parameter_array flag in the cache key so
        # the two layouts (flat + switch vs. array-of-structs) coexist
        # cleanly when the runtime flips the flag mid-process.
        use_pa = _foreach_use_parameter_array()
        key = (self.algorithm, self.batch_size, self.output_dtype, use_pa)
        cached = _SlangForeachOptimizer._src_cache.get(key)
        if cached is not None:
            self._src, self._cache_key = cached
            return
        self._src = _render_foreach_optimizer_slang(
            self.algorithm,
            self.batch_size,
            self.output_dtype,
            parameter_array=use_pa,
        )
        self._cache_key = _foreach_cache_key(
            self.algorithm, self.batch_size, self.output_dtype, use_pa
        )
        _SlangForeachOptimizer._src_cache[key] = (self._src, self._cache_key)

    def __call__(
        self,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        *,
        lr: list[float] | None = None,
        weight_decay: list[float] | None = None,
        momentum: list[float] | None = None,
        beta2: list[float] | None = None,
        eps: list[float] | None = None,
        momentum_bufs: list[torch.Tensor] | None = None,
        v_bufs: list[torch.Tensor] | None = None,
    ) -> None:
        """Dispatch the foreach optimizer template.

        If there are more params than `batch_size`, dispatches in
        multiple batches automatically.
        """
        n = len(params)
        bs = self.batch_size
        alg = self.algorithm
        dt = self.output_dtype

        for start in range(0, n, bs):
            end = min(start + bs, n)
            chunk_params = params[start:end]
            chunk_grads = grads[start:end]
            chunk_lr = lr[start:end] if lr is not None else None
            chunk_wd = weight_decay[start:end] if weight_decay is not None else None
            chunk_mom = momentum[start:end] if momentum is not None else None
            chunk_b2 = beta2[start:end] if beta2 is not None else None
            chunk_eps = eps[start:end] if eps is not None else None
            chunk_mom_bufs = (
                momentum_bufs[start:end] if momentum_bufs is not None else None
            )
            chunk_v_bufs = v_bufs[start:end] if v_bufs is not None else None

            _slang_foreach_optimizer(
                algorithm=alg,
                batch_size=bs,
                output_dtype=dt,
                params=list(chunk_params),
                grads=list(chunk_grads),
                src=self._src,
                cache_key=self._cache_key,
                momentum_bufs=chunk_mom_bufs,
                v_bufs=chunk_v_bufs,
                lr=chunk_lr,
                weight_decay=chunk_wd,
                momentum=chunk_mom,
                beta2=chunk_b2,
                eps=chunk_eps,
            )

    def __reduce__(self):
        return (
            _SlangForeachOptimizer,
            (self.algorithm, self.batch_size, self.output_dtype),
        )


# ── Batch size selection ──────────────────────────────────────────────

_foreach_default_callers: dict[str, _SlangForeachOptimizer] = {}


def _pick_foreach_optimizer_caller(
    algorithm: str,
    n_params: int,
    output_dtype: str = "float",
) -> _SlangForeachOptimizer:
    """Pick (or create) a _SlangForeachOptimizer instance with a
    batch_size large enough to cover `n_params`.

    Returns the smallest pre-defined batch_size ≥ n_params.
    """
    for bs in _OPTIMIZER_BATCH_SIZES:
        if bs >= n_params:
            key = f"{algorithm}_{bs}_{output_dtype}"
            caller = _foreach_default_callers.get(key)
            if caller is None:
                caller = _SlangForeachOptimizer(algorithm, bs, output_dtype)
                _foreach_default_callers[key] = caller
            return caller
    # Fallback: use the largest predefined size.
    bs = _OPTIMIZER_BATCH_SIZES[-1]
    key = f"{algorithm}_{bs}_{output_dtype}"
    caller = _foreach_default_callers.get(key)
    if caller is None:
        caller = _SlangForeachOptimizer(algorithm, bs, output_dtype)
        _foreach_default_callers[key] = caller
    return caller


# ── install_external_optimizer ─────────────────────────────────────────


def install_external_optimizer() -> None:
    """Pre-render the foreach optimizer template for the common
    (algorithm, batch_size, output_dtype) combinations, and register
    the custom ops for SGD, SGD+momentum, AdamW, Lion.

    The custom ops are registered via eager_patches; the FX pass
    (``_fuse_optimizer_step_to_foreach``) fuses optimizer-step
    patterns into these custom ops, which dispatch through the
    ``_SlangForeachOptimizer`` template.

    Safe to call multiple times — only installs once.
    """
    global _optimizer_installed
    if _optimizer_installed:
        return
    _optimizer_installed = True

    from .fx_passes.eager_patches import (
        _ensure_foreach_adamw_step_op_registered,
        _ensure_foreach_lion_step_op_registered,
        _ensure_foreach_sgd_momentum_step_op_registered,
        _ensure_foreach_sgd_step_op_registered,
    )

    _ensure_foreach_sgd_step_op_registered()
    _ensure_foreach_sgd_momentum_step_op_registered()
    _ensure_foreach_adamw_step_op_registered()
    _ensure_foreach_lion_step_op_registered()


def _collect_optimizer_prewarm_specs() -> list[tuple[str, str]]:
    """Render (cache_key, slang_src) pairs for the common optimizer
    template variants."""
    specs: list[tuple[str, str]] = []
    dtypes = ("float", "half")
    algorithms = ("sgd", "sgd_momentum", "adamw", "lion")
    use_pa = _foreach_use_parameter_array()
    for alg in algorithms:
        for bs in _OPTIMIZER_BATCH_SIZES:
            for dt in dtypes:
                key = _foreach_cache_key(alg, bs, dt, use_pa)
                src = _render_foreach_optimizer_slang(
                    alg, bs, dt, parameter_array=use_pa
                )
                specs.append((key, src))
    return specs


def prewarm_optimizer_templates(*, sync: bool = False) -> int:
    """Submit optimizer template variants to the slangc thread pool.

    No-op when slangc is not available.  With ``sync=False`` the call
    returns immediately and the cache is populated in the background.
    """
    if os.environ.get("TORCH_VULKAN_NO_PREWARM") == "1":
        return 0
    from .runtime import _slangc_available, prewarm_compile

    if not _slangc_available():
        return 0
    return prewarm_compile(_collect_optimizer_prewarm_specs(), sync=sync)


# ═══════════════════════════════════════════════════════════════════════════
# T4.5: Scatter / Gather / IndexPut template
# ═══════════════════════════════════════════════════════════════════════════

_SCATTER_CACHE: dict[tuple, str] = {}
_scatter_installed = False


def _render_scatter_atomic(
    operation: str,
    dtype: str = "float",
    index_dtype: str = "int",
) -> str:
    """Render the scatter_atomic Jinja2 template.

    Args:
        operation: One of:
            * ``"gather"`` / ``"scatter"`` / ``"scatter_add"`` /
              ``"index_put"`` / ``"index_put_accumulate"`` (T4.5).
            * ``"scatter_reduce_amax"`` / ``"scatter_reduce_amin"`` /
              ``"scatter_reduce_prod"`` / ``"scatter_reduce_mean"`` (T4.11).
        dtype: Slang element type for data buffers ("float", "half").
        index_dtype: Slang type for index buffer ("int", "int64_t").
    """
    from jinja2 import Environment

    valid_ops = {
        "gather",
        "scatter",
        "scatter_add",
        "index_put",
        "index_put_accumulate",
        # T4.11 — non-sum scatter_reduce modes.
        "scatter_reduce_amax",
        "scatter_reduce_amin",
        "scatter_reduce_prod",
        "scatter_reduce_mean",
    }
    if operation not in valid_ops:
        raise ValueError(
            f"Unknown scatter operation '{operation}'. Must be one of: {sorted(valid_ops)}"
        )

    key = (operation, dtype, index_dtype)
    if key in _SCATTER_CACHE:
        return _SCATTER_CACHE[key]

    src = _load_slang_template("scatter_atomic")
    if not src:
        raise RuntimeError("scatter_atomic.py.jinja template not found")

    env = Environment()
    tmpl = env.from_string(src)
    rendered = tmpl.render(
        operation=operation,
        dtype=dtype,
        index_dtype=index_dtype,
    )
    _SCATTER_CACHE[key] = rendered
    return rendered


def _dispatch_scatter_atomic(
    operation: str,
    numel: int,
    src_numel: int,
    out_numel: int,
    output: torch.Tensor,
    src: torch.Tensor,
    indices: torch.Tensor,
    dtype: str = "float",
    index_dtype: str = "int",
    cache_key: str = "",
    count_buffer: torch.Tensor | None = None,
) -> None:
    """Dispatch the scatter/gather/index_put template as a compute shader.

    Args:
        operation: ``"gather"``, ``"scatter"``, ``"scatter_add"``,
                   ``"index_put"``, ``"index_put_accumulate"`` or one of the
                   T4.11 reduce modes (``"scatter_reduce_amax"``,
                   ``"scatter_reduce_amin"``, ``"scatter_reduce_prod"``,
                   ``"scatter_reduce_mean"``).
        numel: Number of work items (= number of indices to process).
        src_numel: Element count of the source/values buffer.
        out_numel: Element count of the output buffer.
        output: Output tensor (must be pre-allocated).
        src: Source/values tensor.
        indices: Index tensor (int32 or int64).
        dtype: Slang type string for data buffers.
        index_dtype: Slang type string for the index buffer.
        cache_key: Stable cache key for SPIR-V compilation caching.
        count_buffer: Required for ``operation="scatter_reduce_mean"`` —
                      a uint32 tensor of length ``out_numel`` that is
                      atomically incremented per landed element so the
                      caller can divide for the mean.  Ignored for all
                      other operations.
    """
    import struct as _struct

    from .runtime import compile_and_dispatch

    threadgroup_size = 256
    grid_x = (numel + threadgroup_size - 1) // threadgroup_size

    # Push constants: numel, src_numel, out_numel
    pc = _struct.pack("3I", numel, src_numel, out_numel)

    if not cache_key:
        cache_key = f"slang_scatter_{operation}_{dtype}_{index_dtype}"

    src_rendered = _render_scatter_atomic(
        operation=operation,
        dtype=dtype,
        index_dtype=index_dtype,
    )

    # Ensure all tensors are contiguous before dispatch.
    # The shader uses flat (linear) indexing, so views with non-default
    # strides would cause silently-wrong results.
    out_contig = output.contiguous()
    src_contig = src.contiguous()
    idx_contig = indices.contiguous()

    # The C++ dispatch_shader marks the **last** num_outputs buffers as
    # outputs for dirty-buffer / barrier tracking.  Place the output(s)
    # last so the tracking is correct.  Mean-mode binds a second output
    # buffer (the per-target count) immediately after `out` to match the
    # KernelArgs struct field order in scatter_atomic.py.jinja.
    tensors: list[torch.Tensor] = [src_contig, idx_contig, out_contig]
    num_outputs = 1
    if operation == "scatter_reduce_mean":
        if count_buffer is None:
            raise ValueError(
                "scatter_reduce_mean requires a `count_buffer` (uint32 "
                "tensor of length out_numel) so the post-pass divide can "
                "compute the mean."
            )
        tensors.append(count_buffer.contiguous())
        num_outputs = 2

    # If the caller's output was non-contiguous we must copy the result
    # back into the original tensor after the dispatch completes.
    needs_copy_back = out_contig.data_ptr() != output.data_ptr()

    compile_and_dispatch(
        src_rendered,
        tensors,
        grid_x,
        1,
        1,
        push_constants=pc,
        num_outputs=num_outputs,
        cache_key=cache_key,
    )

    if needs_copy_back:
        output.copy_(out_contig)


def install_external_scatter() -> None:
    """Register Vulkan scatter/gather/index_put template lowerings.

    Intercepts ``aten.gather``, ``aten.scatter_add``, ``aten.scatter.src``,
    and ``aten.index_put`` at the Inductor lowering level and routes them
    through the ``scatter_atomic.py.jinja`` template instead of the default
    ExternKernel fallback path.

    Analogous to ``install_external_rng()`` for RNG ops.
    Safe to call multiple times — only installs once.
    """
    global _scatter_installed
    if _scatter_installed:
        return
    _scatter_installed = True

    # We don't replace the lowering — we rely on the existing Inductor
    # codegen for scatter/gather/index_put, which already works correctly
    # via indirect-indexing + atomic-add (see TestGatherScatterAdd,
    # TestIndexSelectAndScatterCodegen).  This install hook pre-warms the
    # template variants so that when the FxPatternRegistry (Track 4) or
    # template_registry routes a SCATTER-class op through the template
    # pipeline, the SPIR-V is already cached.
    #
    # Future (T4.5 follow-up): register custom lowerings that emit
    # the template directly for cases where Inductor stock codegen
    # falls short (e.g. multi-dimensional scatter with epilogue fusion).
    from .runtime import _slangc_available, prewarm_compile

    if not _slangc_available():
        return

    specs: list[tuple[str, str]] = []
    operations = (
        "gather",
        "scatter",
        "scatter_add",
        "index_put",
        "index_put_accumulate",
        # T4.11 — non-sum scatter_reduce modes.
        "scatter_reduce_amax",
        "scatter_reduce_amin",
        "scatter_reduce_prod",
        "scatter_reduce_mean",
    )
    for op in operations:
        for dt in ("float", "half"):
            for idt in ("int", "int64_t"):
                key = f"slang_scatter_{op}_{dt}_{idt}"
                src = _render_scatter_atomic(
                    operation=op,
                    dtype=dt,
                    index_dtype=idt,
                )
                specs.append((key, src))
    prewarm_compile(specs, sync=False)


# T2.8b: Removed dead `_render_mm_wrapper_slang` (~210L), `_USE_MM_TILED_MODULE`,
# and `_wrapper_cache` — all exclusively rendered/cached the now-deleted
# `mm_tiled.slang` module. The live link-time wrapper is below
# (`_render_mm_linktime_wrapper_slang`), which targets `mm_tile.slang-module`.


# ── P3.2 / M14: Link-time specialization via precompiled mm_tile module ──
# Cached per unique (tile_m, tile_n, tile_k, ..., dtype) tuple.
_linktime_wrapper_cache: dict[tuple, str] = {}


def _render_mm_linktime_wrapper_slang(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_per_thread: int = 1,
    n_per_thread: int = 1,
    num_stages: int = 1,
    has_bias: bool = False,
    epilogue_struct: str | None = None,
    dtype_a: str = "float",
    dtype_b: str = "float",
    dtype_c: str = "float",
    dtype_acc: str = "float",
    dtype_bias: str = "float",
    has_alpha: bool = False,
    has_beta: bool = False,
    has_scale: bool = False,
    has_clamp: bool = False,
) -> str:
    """Render a thin wrapper importing the precompiled mm_tile module.

    P3.2 / M14: The heavy template body lives in ``mm_tile.slang-module``
    (compiled once per dtype). This wrapper (~50 lines) defines the
    tile-size constants via ``static const int`` (link-time resolution
    of ``extern static const int`` in the module), push constants,
    buffer bindings, and the compute entry point.  Compilation is an
    order of magnitude faster because slangc only parses the wrapper and
    links against precompiled IR with constant specialization.

    Uses proper Slang link-time constant resolution:
    ``static const int TILE_M = <value>;`` before ``import mm_tile;``
    satisfies the module's ``extern static const int TILE_M;``
    declaration at link time.
    """
    key = (
        tile_m,
        tile_n,
        tile_k,
        m_per_thread,
        n_per_thread,
        num_stages,
        epilogue_struct,
        has_bias,
        has_alpha,
        has_beta,
        has_scale,
        has_clamp,
        dtype_a,
        dtype_b,
        dtype_c,
        dtype_acc,
        dtype_bias,
    )
    if key in _linktime_wrapper_cache:
        return _linktime_wrapper_cache[key]

    wg_m = tile_m // m_per_thread
    wg_n = tile_n // n_per_thread
    epilogue_type = epilogue_struct if epilogue_struct else "OpIdentity"
    out_binding_idx = 3 if has_bias else 2
    bias_binding_idx = 2

    lines: list[str] = []
    lines.append("// P3.2/M14 mm_tile link-time wrapper — auto-generated")
    lines.append(f"// TILE_M={tile_m} TILE_N={tile_n} TILE_K={tile_k}")
    lines.append(
        f"// M_PER_THREAD={m_per_thread} N_PER_THREAD={n_per_thread}"
        f" NUM_STAGES={num_stages}"
    )
    lines.append("")

    # Link-time specialization: define constants BEFORE import so Slang's
    # linker resolves the module's `extern static const int` declarations.
    lines.append(f"static const int TILE_M = {tile_m};")
    lines.append(f"static const int TILE_N = {tile_n};")
    lines.append(f"static const int TILE_K = {tile_k};")
    lines.append(f"static const int M_PER_THREAD = {m_per_thread};")
    lines.append(f"static const int N_PER_THREAD = {n_per_thread};")
    lines.append(f"#define NUM_STAGES {num_stages}")
    lines.append("")
    lines.append("// NUM_STAGES is a #define (not static const) because mm_tile.slang")
    lines.append("// uses it in #if guards for groupshared array sizing.")
    lines.append("import mm_tile;")
    lines.append("")

    # Push-constant struct
    lines.append("struct PC {")
    lines.append("    uint M;")
    lines.append("    uint N;")
    lines.append("    uint K;")
    lines.append("    uint stride_a_m;")
    lines.append("    uint stride_a_k;")
    lines.append("    uint stride_b_k;")
    lines.append("    uint stride_b_n;")
    lines.append("    uint stride_c_m;")
    lines.append("    uint stride_c_n;")
    if has_bias:
        lines.append("    uint stride_bias_n;")
    if has_alpha:
        lines.append("    float alpha;")
    if has_beta:
        lines.append("    float beta;")
    if has_scale:
        lines.append("    float scale;")
    if has_clamp:
        lines.append("    float clamp_min;")
        lines.append("    float clamp_max;")
    lines.append("};")
    lines.append("")
    lines.append("[[vk::push_constant]] PC pc;")
    lines.append("")

    # Buffer bindings
    lines.append(f"[[vk::binding(0)]] StructuredBuffer<{dtype_a}> a;")
    lines.append(f"[[vk::binding(1)]] StructuredBuffer<{dtype_b}> b;")
    if has_bias:
        lines.append(
            f"[[vk::binding({bias_binding_idx})]] StructuredBuffer<{dtype_bias}> bias;"
        )
    lines.append(f"[[vk::binding({out_binding_idx})]] RWStructuredBuffer<{dtype_c}> c;")
    lines.append("")

    # Entry point — delegates to mm_tile::computeTile<Epilogue>
    lines.append('[shader("compute")]')
    lines.append(f"[numthreads({wg_n}, {wg_m}, 1)]")
    lines.append("void computeMain(")
    lines.append("    uint3 gtid : SV_DispatchThreadID,")
    lines.append("    uint3 lid : SV_GroupThreadID,")
    lines.append("    uint3 gid : SV_GroupID)")
    lines.append("{")
    lines.append("    uint row_base = gid.y * (uint)TILE_M;")
    lines.append("    uint col_base = gid.x * (uint)TILE_N;")
    lines.append("")

    # Build the MM_PC struct for the module call
    lines.append("    mm_tile::MM_PC mm_pc;")
    lines.append("    mm_pc.M = pc.M;")
    lines.append("    mm_pc.N = pc.N;")
    lines.append("    mm_pc.K = pc.K;")
    lines.append("    mm_pc.stride_a_m = pc.stride_a_m;")
    lines.append("    mm_pc.stride_a_k = pc.stride_a_k;")
    lines.append("    mm_pc.stride_b_k = pc.stride_b_k;")
    lines.append("    mm_pc.stride_b_n = pc.stride_b_n;")
    lines.append("    mm_pc.stride_c_m = pc.stride_c_m;")
    lines.append("    mm_pc.stride_c_n = pc.stride_c_n;")
    lines.append("")

    # Call the module's computeTile function (handles load → mma → store).
    # The epilogue (bias, activation, clamp, etc.) is applied by the wrapper
    # AFTER the module's store to allow dtype-cast and alpha/beta blending.
    lines.append(f"    mm_tile::computeTile<{epilogue_type}>(")
    lines.append("        row_base, col_base, mm_pc, a, b, c, gid, lid);")
    lines.append("")

    # Post-module epilogue: alpha, beta, bias, clamp, scale
    # These are applied per-element AFTER the module's store_epilogue
    # to support operations not expressible as pure IPointwise.
    if has_alpha or has_beta or has_bias or has_scale or has_clamp:
        lines.append("    // Post-module epilogue adjustments")
        lines.append("    [unroll]")
        lines.append("    for (uint mi = 0; mi < (uint)M_PER_THREAD; mi++) {")
        lines.append("        uint row = row_base + lid.y * (uint)M_PER_THREAD + mi;")
        lines.append("        if (row >= pc.M) continue;")
        lines.append("        [unroll]")
        lines.append("        for (uint ni = 0; ni < (uint)N_PER_THREAD; ni++) {")
        lines.append(
            "            uint col = col_base + lid.x * (uint)N_PER_THREAD + ni;"
        )
        lines.append("            if (col >= pc.N) continue;")
        lines.append(
            f"            {dtype_acc} v = ({dtype_acc})"
            f"c[row * pc.stride_c_m + col * pc.stride_c_n];"
        )
        if has_alpha:
            lines.append("            v *= pc.alpha;")
        if has_bias:
            lines.append(f"            v += ({dtype_acc})bias[col * pc.stride_bias_n];")
        if has_beta:
            lines.append(
                f"            v = pc.beta * v +"
                f" (1.0 - pc.beta) * ({dtype_acc})"
                f"c[row * pc.stride_c_m + col * pc.stride_c_n];"
            )
        if has_scale:
            lines.append("            v *= pc.scale;")
        if has_clamp:
            lines.append(
                f"            v = clamp(v, ({dtype_acc})pc.clamp_min,"
                f" ({dtype_acc})pc.clamp_max);"
            )
        lines.append(
            f"            c[row * pc.stride_c_m + col * pc.stride_c_n] = ({dtype_c})v;"
        )
        lines.append("        }")
        lines.append("    }")

    lines.append("}")

    src = "\n".join(lines) + "\n"
    _linktime_wrapper_cache[key] = src
    return src


# ═══════════════════════════════════════════════════════════════════════════
# T4.2 — Matmul backward via forward template reuse
# ═══════════════════════════════════════════════════════════════════════════


def _render_mm_backward_slang(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    *,
    transpose_a: bool = False,
    transpose_b: bool = False,
    num_stages: int = 1,
    m_per_thread: int = 1,
    n_per_thread: int = 1,
) -> str:
    """Render the tiled matmul template for backward use.

    T4.2: Matmul backward computes dA = dC @ B^T and dB = A^T @ dC.
    Instead of transposing operands on the CPU (which requires a copy),
    this renders the forward mm template with the strides pre-configured
    for the transposed access pattern:

    - ``transpose_a=True`` → the A operand is read with stride_a_m=1
      (i.e. A is logically transposed: A^T has K rows and M columns)
    - ``transpose_b=True`` → the B operand is read with stride_b_k=1
      (i.e. B is logically transposed: B^T has N rows and K columns)

    The rendered shader is functionally identical to the forward template
    but the push-constant stride layout encodes the transposition, avoiding
    a host-side copy + contiguous call.

    Returns the Slang source string.
    """
    # Use the same _render_mm_slang with explicit dtype params.
    # Transposition is handled at dispatch time via the push-constant
    # strides; the template itself is identical.
    src = _render_mm_slang(
        tile_m,
        tile_n,
        tile_k,
        dtype_a="float",
        dtype_b="float",
        dtype_c="float",
        dtype_acc="float",
        num_stages=num_stages,
        m_per_thread=m_per_thread,
        n_per_thread=n_per_thread,
    )
    return src


def _slang_tile_mm_backward(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    *,
    transpose_a: bool = False,
    transpose_b: bool = False,
    num_stages: int = 1,
    m_per_thread: int = 1,
    n_per_thread: int = 1,
) -> None:
    """Execute tiled matmul for backward use with optional operand transposition.

    T4.2: When computing dA = dC @ B^T, we need ``transpose_b=True`` because
    B is logically transposed (read B^T as [N, K] via stride_b_k=stride(0),
    stride_b_n=stride(1)). When computing dB = A^T @ dC, we need
    ``transpose_a=True`` because A is logically transposed (read A^T as
    [K, M] via stride_a_m=stride(1), stride_a_k=stride(0)).

    The push-constant strides encode the transposition directly, avoiding
    a host-side ``.t().contiguous()`` call.  The grid and workgroup
    dimensions are identical to ``_slang_tile_mm``.

    Dimension extraction (M=rows of output, N=cols of output, K=reduction):
      - transpose_a=False → a[M, K]: M=a.shape[0], K=a.shape[1]
      - transpose_a=True  → a[K, M]: M=a.shape[1], K=a.shape[0]
      - transpose_b=False → b[K, N]: K=b.shape[0], N=b.shape[1]
      - transpose_b=True  → b[N, K]: K=b.shape[1], N=b.shape[0]
    """
    from .runtime import compile_and_dispatch

    # Extract M (output rows), N (output cols), K (reduction dim) from shapes.
    if transpose_a:
        K = a.shape[0]
        M = a.shape[1]
    else:
        M = a.shape[0]
        K = a.shape[1]
    if transpose_b:
        N = b.shape[0]
        K_b = b.shape[1]
    else:
        K_b = b.shape[0]
        N = b.shape[1]
    assert K == K_b, f"Reduction dim mismatch: K={K} vs K_b={K_b}"

    dtype_s = _dtype_to_slang(a.dtype)
    src = _render_mm_backward_slang(
        tile_m,
        tile_n,
        tile_k,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        num_stages=num_stages,
        m_per_thread=m_per_thread,
        n_per_thread=n_per_thread,
    )
    cache_key = (
        f"slang_mm_bwd_{tile_m}_{tile_n}_{tile_k}_s{num_stages}"
        f"_r{m_per_thread}x{n_per_thread}_{dtype_s}"
        f"{'_ta' if transpose_a else ''}{'_tb' if transpose_b else ''}"
    )

    if not _TRUST_INDUCTOR:
        if not a.is_contiguous():
            a = a.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()
        if not out.is_contiguous():
            out = out.contiguous()

    # Encode strides.  After contiguity is enforced above, stride(1)==1.
    # transpose_a=True: a[K,M] → logical A^T: rows along a.stride(1)=1,
    #   cols along a.stride(0)=M.  Strides swap vs forward.
    # transpose_b=True: b[N,K] → logical B^T: reduction along b.stride(0)=K,
    #   cols along b.stride(1)=1.  Strides identical to forward.
    # transpose_a=False: a[M,K] → stride_a_m=a.stride(0)=K, stride_a_k=a.stride(1)=1.
    if transpose_a:
        stride_a_m = a.stride(1)
        stride_a_k = a.stride(0)
    else:
        stride_a_m = a.stride(0)
        stride_a_k = a.stride(1)
    stride_b_k = b.stride(0)
    stride_b_n = b.stride(1)

    pc = struct.pack(
        "9I",
        M,
        N,
        K,
        stride_a_m,
        stride_a_k,
        stride_b_k,
        stride_b_n,
        out.stride(0),
        out.stride(1),
    )

    grid_x = (N + tile_n - 1) // tile_n
    grid_y = (M + tile_m - 1) // tile_m

    compile_and_dispatch(
        src,
        [a, b, out],
        grid_x,
        grid_y,
        1,
        push_constants=pc,
        num_outputs=1,
        entry="computeMain<OpIdentity>",
        cache_key=cache_key,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CG.M5 — Single-kernel matmul backward via [Differentiable] tile_inner_madd
# ═══════════════════════════════════════════════════════════════════════════


def _render_mm_bwd_slang(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    *,
    dtype_a: str = "float",
    dtype_b: str = "float",
    dtype_c: str = "float",
    dtype_acc: str = "float",
    has_batch: bool = False,
    m_per_thread: int = 1,
    n_per_thread: int = 1,
) -> str:
    """Render the CG.M5 slang_mm_bwd Jinja2 template.

    Produces a single-kernel backward that computes BOTH dA and dB in one
    dispatch by wrapping ``bwd_diff(tile_inner_madd)`` in a tiled K-loop.
    This replaces the 2-dispatch decomposition (dA = dC @ B^T, dB = A^T @ dC)
    with 1 fused dispatch.

    The template supports:
      - mm backward (has_batch=False): dA[M,K], dB[K,N]
      - bmm backward (has_batch=True): dA[B,M,K], dB[B,K,N]
      - Register tiling via m_per_thread / n_per_thread
    """
    from jinja2 import Environment

    key = (
        tile_m,
        tile_n,
        tile_k,
        dtype_a,
        dtype_b,
        dtype_c,
        dtype_acc,
        has_batch,
        m_per_thread,
        n_per_thread,
    )
    if key in _tile_cache:
        return _tile_cache[key]

    src = _load_slang_template("slang_mm_bwd")
    if not src:
        raise RuntimeError("slang_mm_bwd.py.jinja template not found")

    env = Environment()
    tmpl = env.from_string(src)
    rendered = tmpl.render(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        dtype_a=dtype_a,
        dtype_b=dtype_b,
        dtype_c=dtype_c,
        dtype_acc=dtype_acc,
        has_batch=has_batch,
        m_per_thread=m_per_thread,
        n_per_thread=n_per_thread,
    )
    _tile_cache[key] = rendered
    return rendered


def _slang_tile_mm_bwd(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    a: torch.Tensor,
    b: torch.Tensor,
    dC: torch.Tensor,
    dA: torch.Tensor,
    dB: torch.Tensor,
    *,
    m_per_thread: int = 1,
    n_per_thread: int = 1,
) -> None:
    """Execute the CG.M5 single-kernel matmul backward.

    Computes dA = dC @ B^T and dB = A^T @ dC in ONE dispatch by:
    1. Loading tiles of A (forward input), B (forward weight), and dC (grad_output)
    2. For each (m,n,k) element, calling bwd_diff(tile_inner_madd) to get
       dA contribution (= dC * B) and dB contribution (= dC * A)
    3. Aggregating dA and dB across the K dimension
    4. Storing both outputs

    This replaces two forward-template dispatches (dA = dC @ B^T, dB = A^T @ dC)
    with a single fused backward dispatch.

    Dimensions:
      - A: [M, K]  (forward input)
      - B: [K, N]  (forward weight)
      - dC: [M, N] (gradient of loss w.r.t. forward output)
      - dA: [M, K] (output, same shape as A)
      - dB: [K, N] (output, same shape as B)
    """
    from .runtime import compile_and_dispatch

    M, K_a = a.shape
    K_b, N = b.shape
    assert K_a == K_b, f"Reduction dim mismatch: K_a={K_a} vs K_b={K_b}"
    K = K_a
    assert dC.shape == (M, N), f"dC shape mismatch: {dC.shape} vs ({M}, {N})"
    assert dA.shape == (M, K), f"dA shape mismatch: {dA.shape} vs ({M}, {K})"
    assert dB.shape == (K, N), f"dB shape mismatch: {dB.shape} vs ({K}, {N})"

    dtype_s = _dtype_to_slang(a.dtype)
    src = _render_mm_bwd_slang(
        tile_m,
        tile_n,
        tile_k,
        dtype_a=dtype_s,
        dtype_b=dtype_s,
        dtype_c=dtype_s,
        dtype_acc=dtype_s,
        has_batch=False,
        m_per_thread=m_per_thread,
        n_per_thread=n_per_thread,
    )
    cache_key = (
        f"slang_mm_bwd_{tile_m}_{tile_n}_{tile_k}"
        f"_r{m_per_thread}x{n_per_thread}_{dtype_s}"
    )

    # Encode strides for all 5 buffers.
    # A: [M, K], B: [K, N], dC: [M, N], dA: [M, K], dB: [K, N]
    pc = struct.pack(
        "13I",
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        dC.stride(0),
        dC.stride(1),
        dA.stride(0),
        dA.stride(1),
        dB.stride(0),
        dB.stride(1),
    )

    grid_x = (N + tile_n - 1) // tile_n
    grid_y = (M + tile_m - 1) // tile_m

    compile_and_dispatch(
        src,
        [a, b, dC, dA, dB],
        grid_x,
        grid_y,
        1,
        push_constants=pc,
        num_outputs=2,
        entry="computeMain",
        cache_key=cache_key,
    )


def _slang_tile_bmm_bwd(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    a: torch.Tensor,
    b: torch.Tensor,
    dC: torch.Tensor,
    dA: torch.Tensor,
    dB: torch.Tensor,
    *,
    m_per_thread: int = 1,
    n_per_thread: int = 1,
) -> None:
    """Execute the CG.M5 single-kernel batched matmul backward (bmm).

    Same as _slang_tile_mm_bwd but for 3-D batched inputs.

    Dimensions:
      - A: [B, M, K]
      - B: [B, K, N]
      - dC: [B, M, N]
      - dA: [B, M, K]
      - dB: [B, K, N]
    """
    from .runtime import compile_and_dispatch

    B, M, K_a = a.shape
    B_b, K_b, N = b.shape
    assert B == B_b, f"Batch dim mismatch: {B} vs {B_b}"
    assert K_a == K_b, f"Reduction dim mismatch: K_a={K_a} vs K_b={K_b}"
    K = K_a
    assert dC.shape == (B, M, N), f"dC shape mismatch: {dC.shape} vs ({B}, {M}, {N})"
    assert dA.shape == (B, M, K), f"dA shape mismatch"
    assert dB.shape == (B, K, N), f"dB shape mismatch"

    dtype_s = _dtype_to_slang(a.dtype)
    src = _render_mm_bwd_slang(
        tile_m,
        tile_n,
        tile_k,
        dtype_a=dtype_s,
        dtype_b=dtype_s,
        dtype_c=dtype_s,
        dtype_acc=dtype_s,
        has_batch=True,
        m_per_thread=m_per_thread,
        n_per_thread=n_per_thread,
    )
    cache_key = (
        f"slang_bmm_bwd_{tile_m}_{tile_n}_{tile_k}"
        f"_r{m_per_thread}x{n_per_thread}_{dtype_s}"
    )

    # Encode strides for all 5 buffers + batch strides.
    pc = struct.pack(
        "18I",
        M,
        N,
        K,
        a.stride(1),
        a.stride(2),
        b.stride(1),
        b.stride(2),
        dC.stride(1),
        dC.stride(2),
        dA.stride(1),
        dA.stride(2),
        dB.stride(1),
        dB.stride(2),
        a.stride(0),
        b.stride(0),
        dC.stride(0),
        dA.stride(0),
        dB.stride(0),
    )

    grid_x = (N + tile_n - 1) // tile_n
    grid_y = (M + tile_m - 1) // tile_m
    grid_z = B

    compile_and_dispatch(
        src,
        [a, b, dC, dA, dB],
        grid_x,
        grid_y,
        grid_z,
        push_constants=pc,
        num_outputs=2,
        entry="computeMain",
        cache_key=cache_key,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CP.3 / OP.6 — RNN cell template infrastructure
# ═══════════════════════════════════════════════════════════════════════════

_rnn_cell_cache: dict[tuple, str] = {}
_rnn_installed = False

# Cell types the template supports.
_RNN_CELL_TYPES: tuple[str, ...] = ("lstm", "gru", "rnn_tanh", "rnn_relu")


def _render_rnn_cell(
    cell_type: str,
    hidden_size: int,
    input_size: int,
    dtype: str = "float",
) -> str:
    """Render the rnn_cell Jinja2 template.

    Args:
        cell_type: One of ``"lstm"``, ``"gru"``, ``"rnn_tanh"``, ``"rnn_relu"``.
        hidden_size: Number of hidden units.
        input_size: Input feature dimension.
        dtype: Slang type string — ``"float"`` (f32) or ``"half"`` (f16).

    Returns:
        Rendered Slang source string ready for SPIR-V compilation.
    """
    from jinja2 import Environment

    if cell_type not in _RNN_CELL_TYPES:
        raise ValueError(
            f"Unknown RNN cell_type '{cell_type}'. Must be one of: {_RNN_CELL_TYPES}"
        )

    key = (cell_type, hidden_size, input_size, dtype)
    if key in _rnn_cell_cache:
        return _rnn_cell_cache[key]

    src = _load_slang_template("rnn_cell")
    if not src:
        raise RuntimeError("rnn_cell.py.jinja template not found")

    env = Environment()
    tmpl = env.from_string(src)
    rendered = tmpl.render(
        cell_type=cell_type,
        hidden_size=hidden_size,
        input_size=input_size,
        dtype=dtype,
    )
    _rnn_cell_cache[key] = rendered
    return rendered


def _dispatch_rnn_cell(
    cell_type: str,
    hidden_size: int,
    input_size: int,
    batch_size: int,
    has_bias: bool,
    dtype: str,
    x_t: torch.Tensor,
    h_prev: torch.Tensor,
    c_prev: torch.Tensor | None,
    w_ih: torch.Tensor,
    w_hh: torch.Tensor,
    b_ih: torch.Tensor | None,
    b_hh: torch.Tensor | None,
    h_t: torch.Tensor,
    c_t: torch.Tensor | None,
    src: str | None = None,
    cache_key: str | None = None,
) -> None:
    """Dispatch a single RNN cell computation via the Slang template.

    Computes h_t (and c_t for LSTM) from x_t, h_prev, c_prev, weights, and biases.
    One dispatch processes ALL batch elements in parallel (grid_x = batch_size).

    Args:
        cell_type: ``"lstm"``, ``"gru"``, ``"rnn_tanh"``, or ``"rnn_relu"``.
        hidden_size: Number of hidden units.
        input_size: Input feature dimension.
        batch_size: Number of batch elements.
        has_bias: Whether bias tensors are provided.
        dtype: Slang type string.
        x_t: Input tensor [batch, input_size].
        h_prev: Previous hidden state [batch, hidden_size].
        c_prev: Previous cell state [batch, hidden_size] (LSTM only, else None).
        w_ih: Input-to-hidden weight [gate_size * hidden_size, input_size].
        w_hh: Hidden-to-hidden weight [gate_size * hidden_size, hidden_size].
        b_ih: Input-to-hidden bias [gate_size * hidden_size] (or None).
        b_hh: Hidden-to-hidden bias [gate_size * hidden_size] (or None).
        h_t: Output hidden state [batch, hidden_size].
        c_t: Output cell state [batch, hidden_size] (LSTM only, else None).
        src: Pre-rendered Slang source (rendered if None).
        cache_key: SPIR-V cache key (computed if None).
    """
    from .runtime import compile_and_dispatch

    is_lstm = cell_type == "lstm"

    if src is None or cache_key is None:
        src = _render_rnn_cell(
            cell_type=cell_type,
            hidden_size=hidden_size,
            input_size=input_size,
            dtype=dtype,
        )
        cache_key = f"slang_rnn_{cell_type}_h{hidden_size}_i{input_size}_{dtype}"

    # Push constants: hidden_size, input_size, stride_w_ih, stride_w_hh,
    # stride_x, stride_h, [stride_c (LSTM only)], has_bias
    stride_w_ih = input_size
    stride_w_hh = hidden_size
    stride_x = input_size
    stride_h = hidden_size
    stride_c = hidden_size

    if is_lstm:
        pc = struct.pack(
            "8I",
            hidden_size,
            input_size,
            stride_w_ih,
            stride_w_hh,
            stride_x,
            stride_h,
            stride_c,
            1 if has_bias else 0,
        )
    else:
        pc = struct.pack(
            "7I",
            hidden_size,
            input_size,
            stride_w_ih,
            stride_w_hh,
            stride_x,
            stride_h,
            1 if has_bias else 0,
        )

    # Ensure all inputs are contiguous.
    if not x_t.is_contiguous():
        x_t = x_t.contiguous()
    if not h_prev.is_contiguous():
        h_prev = h_prev.contiguous()
    if is_lstm and c_prev is not None and not c_prev.is_contiguous():
        c_prev = c_prev.contiguous()
    if not w_ih.is_contiguous():
        w_ih = w_ih.contiguous()
    if not w_hh.is_contiguous():
        w_hh = w_hh.contiguous()
    if not h_t.is_contiguous():
        h_t = h_t.contiguous()
    if is_lstm and c_t is not None and not c_t.is_contiguous():
        c_t = c_t.contiguous()

    # Build buffer list matching KernelArgs field order.
    # LSTM:  [x_t, h_prev, c_prev, w_ih, w_hh, b_ih, b_hh, h_t, c_t]
    # GRU/RNN: [x_t, h_prev, w_ih, w_hh, b_ih, b_hh, h_t]
    buffers: list[torch.Tensor] = [x_t, h_prev]
    if is_lstm:
        buffers.append(
            c_prev if c_prev is not None else torch.empty(0, device=x_t.device)
        )
    buffers.extend([w_ih, w_hh])
    # Bias tensors — use zero-sized placeholder if not provided.
    if has_bias and b_ih is not None:
        buffers.append(b_ih.contiguous() if not b_ih.is_contiguous() else b_ih)
    else:
        buffers.append(torch.empty(0, device=x_t.device))
    if has_bias and b_hh is not None:
        buffers.append(b_hh.contiguous() if not b_hh.is_contiguous() else b_hh)
    else:
        buffers.append(torch.empty(0, device=x_t.device))
    buffers.append(h_t)
    if is_lstm:
        buffers.append(c_t if c_t is not None else torch.empty(0, device=x_t.device))

    num_outputs = 2 if is_lstm else 1

    grid_x = batch_size
    grid_y = 1
    grid_z = 1

    compile_and_dispatch(
        src,
        buffers,
        grid_x,
        grid_y,
        grid_z,
        push_constants=pc,
        num_outputs=num_outputs,
        cache_key=cache_key,
    )


class _SlangTileRNN:
    """Picklable callable for RNN cell template dispatch.

    Each instance is configured for a specific cell_type.  The callable
    interface accepts the per-time-step tensors and returns the updated
    hidden/cell state.

    Caches the rendered Slang source per (dtype, hidden_size, input_size)
    tuple so repeated dispatches for the same cell skip the Jinja render.
    """

    __slots__ = ("cell_type", "__name__", "_per_spec")

    def __init__(self, cell_type: str):
        if cell_type not in _RNN_CELL_TYPES:
            raise ValueError(
                f"Unknown RNN cell_type '{cell_type}'. "
                f"Must be one of: {_RNN_CELL_TYPES}"
            )
        self.cell_type = cell_type
        self.__name__ = f"slang_rnn_{cell_type}"
        self._per_spec: dict[tuple, tuple[str, str]] = {}

    def _src_and_key(
        self, hidden_size: int, input_size: int, dtype: str
    ) -> tuple[str, str]:
        spec_key = (hidden_size, input_size, dtype)
        cached = self._per_spec.get(spec_key)
        if cached is not None:
            return cached

        src = _render_rnn_cell(
            cell_type=self.cell_type,
            hidden_size=hidden_size,
            input_size=input_size,
            dtype=dtype,
        )
        cache_key = f"slang_rnn_{self.cell_type}_h{hidden_size}_i{input_size}_{dtype}"
        cached = (src, cache_key)
        self._per_spec[spec_key] = cached
        return cached

    def __call__(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor | None,
        w_ih: torch.Tensor,
        w_hh: torch.Tensor,
        b_ih: torch.Tensor | None,
        b_hh: torch.Tensor | None,
        h_t: torch.Tensor,
        c_t: torch.Tensor | None,
    ) -> None:
        """Dispatch one RNN cell step.

        Args:
            x_t: Input [batch, input_size].
            h_prev: Previous hidden state [batch, hidden_size].
            c_prev: Previous cell state (LSTM only).
            w_ih: Input-to-hidden weight.
            w_hh: Hidden-to-hidden weight.
            b_ih: Input-to-hidden bias (or None).
            b_hh: Hidden-to-hidden bias (or None).
            h_t: Output hidden state [batch, hidden_size].
            c_t: Output cell state (LSTM only).
        """
        batch_size = x_t.shape[0]
        hidden_size = h_prev.shape[-1]
        input_size = x_t.shape[-1]
        dtype_s = _dtype_to_slang(x_t.dtype)
        has_bias = b_ih is not None and b_hh is not None

        src, cache_key = self._src_and_key(hidden_size, input_size, dtype_s)

        _dispatch_rnn_cell(
            cell_type=self.cell_type,
            hidden_size=hidden_size,
            input_size=input_size,
            batch_size=batch_size,
            has_bias=has_bias,
            dtype=dtype_s,
            x_t=x_t,
            h_prev=h_prev,
            c_prev=c_prev,
            w_ih=w_ih,
            w_hh=w_hh,
            b_ih=b_ih,
            b_hh=b_hh,
            h_t=h_t,
            c_t=c_t,
            src=src,
            cache_key=cache_key,
        )

    def __reduce__(self):
        return (_SlangTileRNN, (self.cell_type,))


# ═══════════════════════════════════════════════════════════════════════════
# T.10-fast — Fused multi-time-step RNN cell template infrastructure
# ═══════════════════════════════════════════════════════════════════════════

# Maximum hidden_size for the fused template (groupshared memory budget).
# 2 × float[1024] = 8 KB, well within the Vulkan minimum of 32 KB.
_FUSED_RNN_MAX_HIDDEN_SIZE = 1024

_rnn_cell_fused_cache: dict[tuple, str] = {}


def _render_rnn_cell_fused(
    cell_type: str,
    hidden_size: int,
    input_size: int,
    seq_len: int,
    dtype: str = "float",
) -> str:
    """Render the rnn_cell_fused Jinja2 template for multi-time-step dispatch.

    Args:
        cell_type: Currently only ``"lstm"`` is supported.
        hidden_size: Number of hidden units (≤ 1024).
        input_size: Input feature dimension.
        seq_len: Number of time steps to fuse into one kernel.
        dtype: Slang type string — ``"float"`` (f32) or ``"half"`` (f16).

    Returns:
        Rendered Slang source string ready for SPIR-V compilation.
    """
    from jinja2 import Environment

    if cell_type not in _RNN_CELL_TYPES:
        raise ValueError(
            f"Fused RNN template only supports {_RNN_CELL_TYPES}, got '{cell_type}'"
        )
    if hidden_size > _FUSED_RNN_MAX_HIDDEN_SIZE:
        raise ValueError(
            f"Fused RNN template requires hidden_size ≤ {_FUSED_RNN_MAX_HIDDEN_SIZE}, "
            f"got {hidden_size}"
        )

    key = (cell_type, hidden_size, dtype)
    if key in _rnn_cell_fused_cache:
        return _rnn_cell_fused_cache[key]

    src = _load_slang_template("rnn_cell_fused")
    if not src:
        raise RuntimeError("rnn_cell_fused.py.jinja template not found")

    env = Environment()
    tmpl = env.from_string(src)
    rendered = tmpl.render(
        cell_type=cell_type,
        hidden_size=hidden_size,
        input_size=input_size,
        seq_len=seq_len,
        dtype=dtype,
    )
    _rnn_cell_fused_cache[key] = rendered
    return rendered


def _dispatch_rnn_cell_fused(
    cell_type: str,
    hidden_size: int,
    input_size: int,
    seq_len: int,
    batch_size: int,
    has_bias: bool,
    dtype: str,
    x_seq: torch.Tensor,
    h0: torch.Tensor,
    c0: torch.Tensor | None,
    w_ih: torch.Tensor,
    w_hh: torch.Tensor,
    b_ih: torch.Tensor | None,
    b_hh: torch.Tensor | None,
    out_seq: torch.Tensor,
    h_last: torch.Tensor,
    c_last: torch.Tensor | None,
    src: str | None = None,
    cache_key: str | None = None,
) -> None:
    """Dispatch a fused multi-time-step RNN cell computation.

    Processes ALL ``seq_len`` time steps for ALL batch elements in ONE kernel
    dispatch.  One workgroup per batch element, internal loop over time steps.

    Args:
        cell_type: ``"lstm"``, ``"gru"``, ``"rnn_tanh"``, or ``"rnn_relu"``.
        hidden_size: Number of hidden units.
        input_size: Input feature dimension.
        seq_len: Number of time steps.
        batch_size: Number of batch elements.
        has_bias: Whether bias tensors are provided.
        dtype: Slang type string.
        x_seq: Input sequence [seq_len, batch, input_size].
        h0: Initial hidden state [batch, hidden_size].
        c0: Initial cell state [batch, hidden_size] (LSTM only, else None).
        w_ih: Input-to-hidden weight [gate_size*hidden_size, input_size].
        w_hh: Hidden-to-hidden weight [gate_size*hidden_size, hidden_size].
        b_ih: Input-to-hidden bias [gate_size*hidden_size] (or None).
        b_hh: Hidden-to-hidden bias [gate_size*hidden_size] (or None).
        out_seq: Output sequence [batch, seq_len, hidden_size] (pre-allocated).
        h_last: Final hidden state [batch, hidden_size] (pre-allocated).
        c_last: Final cell state [batch, hidden_size] (LSTM only, else None).
        src: Pre-rendered Slang source (rendered if None).
        cache_key: SPIR-V cache key (computed if None).
    """
    from .runtime import compile_and_dispatch

    is_lstm = cell_type == "lstm"

    if hidden_size > _FUSED_RNN_MAX_HIDDEN_SIZE:
        raise ValueError(
            f"Fused RNN template requires hidden_size ≤ {_FUSED_RNN_MAX_HIDDEN_SIZE}, "
            f"got {hidden_size}"
        )

    if src is None or cache_key is None:
        src = _render_rnn_cell_fused(
            cell_type=cell_type,
            hidden_size=hidden_size,
            input_size=input_size,
            seq_len=seq_len,
            dtype=dtype,
        )
        # Note: seq_len is a push constant, NOT embedded in the Slang source,
        # so the source is identical for all sequence lengths.  The cache key
        # tracks only source-varying parameters (hidden_size, input_size, dtype).
        cache_key = f"slang_rnn_fused_{cell_type}_h{hidden_size}_i{input_size}_{dtype}"

    # Push constants layout (11 uint32_t fields — same for all cell types).
    # PC: hidden_size, input_size, seq_len, stride_w_ih, stride_w_hh,
    #     stride_x_tbatch, stride_x_batch, stride_h_batch,
    #     stride_out_tbatch, stride_out_batch, has_bias
    stride_w_ih = input_size
    stride_w_hh = hidden_size
    stride_x_tbatch = batch_size * input_size
    stride_x_batch = input_size
    stride_h_batch = hidden_size
    stride_out_tbatch = hidden_size
    stride_out_batch = seq_len * hidden_size

    pc = struct.pack(
        "11I",
        hidden_size,
        input_size,
        seq_len,
        stride_w_ih,
        stride_w_hh,
        stride_x_tbatch,
        stride_x_batch,
        stride_h_batch,
        stride_out_tbatch,
        stride_out_batch,
        1 if has_bias else 0,
    )

    # Ensure all inputs are contiguous.
    if not x_seq.is_contiguous():
        x_seq = x_seq.contiguous()
    if not h0.is_contiguous():
        h0 = h0.contiguous()
    if is_lstm and c0 is not None and not c0.is_contiguous():
        c0 = c0.contiguous()
    if not w_ih.is_contiguous():
        w_ih = w_ih.contiguous()
    if not w_hh.is_contiguous():
        w_hh = w_hh.contiguous()
    if not out_seq.is_contiguous():
        out_seq = out_seq.contiguous()
    if not h_last.is_contiguous():
        h_last = h_last.contiguous()
    if is_lstm and c_last is not None and not c_last.is_contiguous():
        c_last = c_last.contiguous()

    # Build buffer list matching KernelArgs field order:
    # [x_seq, h0, c0_or_dummy, w_ih, w_hh, b_ih, b_hh, out_seq, h_last, c_last_or_dummy]
    # For non-LSTM cells, slot 2 (c0) and slot 9 (c_last) are unused placeholders.
    c0_buf: torch.Tensor
    c_last_buf: torch.Tensor
    if is_lstm and c0 is not None and c_last is not None:
        c0_buf = c0
        c_last_buf = c_last
    else:
        c0_buf = torch.empty(0, device=x_seq.device)
        c_last_buf = torch.empty(0, device=x_seq.device)

    buffers: list[torch.Tensor] = [x_seq, h0, c0_buf, w_ih, w_hh]
    if has_bias and b_ih is not None:
        buffers.append(b_ih.contiguous() if not b_ih.is_contiguous() else b_ih)
    else:
        buffers.append(torch.empty(0, device=x_seq.device))
    if has_bias and b_hh is not None:
        buffers.append(b_hh.contiguous() if not b_hh.is_contiguous() else b_hh)
    else:
        buffers.append(torch.empty(0, device=x_seq.device))
    buffers.extend([out_seq, h_last, c_last_buf])

    # num_outputs: out_seq and h_last are always written; c_last only for LSTM.
    num_outputs = 3 if is_lstm else 2

    grid_x = batch_size
    grid_y = 1
    grid_z = 1

    compile_and_dispatch(
        src,
        buffers,
        grid_x,
        grid_y,
        grid_z,
        push_constants=pc,
        num_outputs=num_outputs,
        cache_key=cache_key,
    )


def _can_use_fused_rnn_template(cell_type: str, hidden_size: int) -> bool:
    """Check whether the fused template can be used for these parameters.

    T.10-fast supports LSTM, GRU, RNN-tanh, and RNN-relu.
    """
    return cell_type in _RNN_CELL_TYPES and hidden_size <= _FUSED_RNN_MAX_HIDDEN_SIZE


class _SlangTileRNNFused:
    """Picklable callable for fused multi-time-step RNN cell dispatch.

    Each instance is configured for a specific cell_type.  Unlike
    :class:`_SlangTileRNN`, which dispatches once per time step, this
    callable processes the entire sequence in one kernel dispatch.

    Supports LSTM, GRU, RNN-tanh, and RNN-relu.

    Caches the rendered Slang source per (hidden_size, input_size, dtype) tuple.
    """

    __slots__ = ("cell_type", "__name__", "_per_spec")

    def __init__(self, cell_type: str):
        if cell_type not in _RNN_CELL_TYPES:
            raise ValueError(
                f"Fused RNN template only supports {_RNN_CELL_TYPES}, got '{cell_type}'"
            )
        self.cell_type = cell_type
        self.__name__ = f"slang_rnn_fused_{cell_type}"
        self._per_spec: dict[tuple, tuple[str, str]] = {}

    def _src_and_key(
        self, hidden_size: int, input_size: int, seq_len: int, dtype: str
    ) -> tuple[str, str]:
        spec_key = (hidden_size, input_size, dtype)
        cached = self._per_spec.get(spec_key)
        if cached is not None:
            return cached

        src = _render_rnn_cell_fused(
            cell_type=self.cell_type,
            hidden_size=hidden_size,
            input_size=input_size,
            seq_len=seq_len,
            dtype=dtype,
        )
        cache_key = (
            f"slang_rnn_fused_{self.cell_type}_h{hidden_size}_i{input_size}_{dtype}"
        )
        cached = (src, cache_key)
        self._per_spec[spec_key] = cached
        return cached

    def __call__(
        self,
        x_seq: torch.Tensor,
        h0: torch.Tensor,
        c0: torch.Tensor | None,
        w_ih: torch.Tensor,
        w_hh: torch.Tensor,
        b_ih: torch.Tensor | None,
        b_hh: torch.Tensor | None,
        out_seq: torch.Tensor,
        h_last: torch.Tensor,
        c_last: torch.Tensor | None,
    ) -> None:
        """Dispatch one fused multi-time-step RNN cell.

        Args:
            x_seq: Input sequence [seq_len, batch, input_size].
            h0: Initial hidden state [batch, hidden_size].
            c0: Initial cell state [batch, hidden_size] (LSTM only, else None).
            w_ih: Input-to-hidden weight.
            w_hh: Hidden-to-hidden weight.
            b_ih: Input-to-hidden bias (or None).
            b_hh: Hidden-to-hidden bias (or None).
            out_seq: Output buffer [batch, seq_len, hidden_size].
            h_last: Final hidden state buffer [batch, hidden_size].
            c_last: Final cell state buffer [batch, hidden_size] (LSTM only, else None).
        """
        batch_size = x_seq.shape[1]
        seq_len = x_seq.shape[0]
        hidden_size = h0.shape[-1]
        input_size = x_seq.shape[-1]
        dtype_s = _dtype_to_slang(x_seq.dtype)
        has_bias = b_ih is not None and b_hh is not None

        src, cache_key = self._src_and_key(hidden_size, input_size, seq_len, dtype_s)

        _dispatch_rnn_cell_fused(
            cell_type=self.cell_type,
            hidden_size=hidden_size,
            input_size=input_size,
            seq_len=seq_len,
            batch_size=batch_size,
            has_bias=has_bias,
            dtype=dtype_s,
            x_seq=x_seq,
            h0=h0,
            c0=c0,
            w_ih=w_ih,
            w_hh=w_hh,
            b_ih=b_ih,
            b_hh=b_hh,
            out_seq=out_seq,
            h_last=h_last,
            c_last=c_last,
            src=src,
            cache_key=cache_key,
        )

    def __reduce__(self):
        return (_SlangTileRNNFused, (self.cell_type,))


def install_external_rnn() -> None:
    """Register RNN cell template as the lowering route for RNN ops.

    Called from ``lowerings/__init__.py`` at backend init.  Replaces the
    CPU-roundtrip fallback path in ``lowerings/rnn.py`` with a Vulkan-native
    template dispatch that keeps data on-device.

    Safe to call multiple times — only installs once.
    """
    global _rnn_installed
    if _rnn_installed:
        return
    _rnn_installed = True

    # Pre-render all cell types for the two common dtypes at standard sizes
    # so the first dispatch doesn't block on Jinja rendering.
    from .runtime import _slangc_available, prewarm_compile

    if not _slangc_available():
        return

    rnn_specs: list[tuple[str, str]] = []
    for cell_type in _RNN_CELL_TYPES:
        for dt in ("float",):
            for hidden_size in (128, 256, 512):
                for input_size in (128, 256, 512):
                    cache_key = (
                        f"slang_rnn_{cell_type}_h{hidden_size}_i{input_size}_{dt}"
                    )
                    src = _render_rnn_cell(
                        cell_type=cell_type,
                        hidden_size=hidden_size,
                        input_size=input_size,
                        dtype=dt,
                    )
                    rnn_specs.append((cache_key, src))

    # T.10-fast: prewarm fused RNN templates for all cell types at common sizes.
    for dt in ("float",):
        for hidden_size in (128, 256, 512):
            for input_size in (128, 256, 512):
                for cell_type in _RNN_CELL_TYPES:
                    cache_key = (
                        f"slang_rnn_fused_{cell_type}_h{hidden_size}_i{input_size}_{dt}"
                    )
                    src = _render_rnn_cell_fused(
                        cell_type=cell_type,
                        hidden_size=hidden_size,
                        input_size=input_size,
                        seq_len=64,
                        dtype=dt,
                    )
                    rnn_specs.append((cache_key, src))

    prewarm_compile(rnn_specs, sync=False)
