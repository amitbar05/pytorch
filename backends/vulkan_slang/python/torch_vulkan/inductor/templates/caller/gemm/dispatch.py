"""GEMM dispatch logic.

Low-level Slang shader dispatch helpers for all matmul variants.
Includes tile-config pickers, device queries, and push-constant packing.
"""

from __future__ import annotations

import os
import struct
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

from ....vulkan_template import (
    _MM_REGISTER_TILE_CONFIGS,
    _MM_TILE_CONFIGS,
)
from ....vulkan_template_caller import (
    _dtype_to_slang,
    _validate_epilogue_struct,
)
from .render import _render_mm_int8_slang, _render_mm_slang

_TRUST_INDUCTOR = os.environ.get("TORCH_VULKAN_TRUST_INDUCTOR") == "1"


# ═══════════════════════════════════════════════════════════════════════════
# A.6 — Unified mm push-constant layout
# ═══════════════════════════════════════════════════════════════════════════
#
# A single monolithic PC struct serves every mm variant (mm, addmm, addmm+gelu,
# bmm, plus future alpha/beta/scale/clamp paths).  Fields are always present;
# the kernel reads bits from ``pc.flags`` to gate optional behaviour.
#
# Any change to this layout MUST be mirrored in the matching ``struct PC``
# block in ``templates/slang_mm.py.jinja`` (and the link-time wrapper in
# ``render.py``).
#
# Layout: 19 uint + 5 float = 96 bytes (well below the 128 B push-constant
# limit on RDNA1).
MM_FLAG_BIAS = 1
MM_FLAG_BATCH = 2
MM_FLAG_ALPHA = 4
MM_FLAG_BETA = 8
MM_FLAG_SCALE = 16
MM_FLAG_CLAMP = 32

# Format string for ``struct.pack`` — must match the Slang struct field order.
_MM_PC_FORMAT = "19I5f"


def _pack_mm_pc(
    M: int,
    N: int,
    K: int,
    stride_a_m: int,
    stride_a_k: int,
    stride_b_k: int,
    stride_b_n: int,
    stride_c_m: int,
    stride_c_n: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_per_thread: int,
    n_per_thread: int,
    *,
    stride_bias_n: int = 0,
    stride_a_b: int = 0,
    stride_b_b: int = 0,
    stride_c_b: int = 0,
    flags: int = 0,
    alpha: float = 0.0,
    beta: float = 0.0,
    scale: float = 0.0,
    clamp_min: float = 0.0,
    clamp_max: float = 0.0,
) -> bytes:
    """Pack the monolithic mm PC struct.

    Mirrors ``struct PC`` in ``templates/slang_mm.py.jinja``.  Optional
    fields default to 0 / 0.0 when their corresponding ``MM_FLAG_*`` bit is
    clear, so a single helper serves every mm dispatch (mm / addmm / bmm /
    addmm+gelu).
    """
    return struct.pack(
        _MM_PC_FORMAT,
        M,
        N,
        K,
        stride_a_m,
        stride_a_k,
        stride_b_k,
        stride_b_n,
        stride_c_m,
        stride_c_n,
        stride_bias_n,
        stride_a_b,
        stride_b_b,
        stride_c_b,
        tile_m,
        tile_n,
        tile_k,
        m_per_thread,
        n_per_thread,
        flags,
        alpha,
        beta,
        scale,
        clamp_min,
        clamp_max,
    )


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
    from ....runtime import compile_and_dispatch

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
            f"_r{m_per_thread}x{n_per_thread}_{dtype_s}_n111_a6"
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

    # A.6: Monolithic PC layout — all fields always present; flags=0 means
    # no bias / batch / alpha / etc. (plain mm).
    pc = _pack_mm_pc(
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
    from ....runtime import compile_and_dispatch

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
            f"_r{m_per_thread}x{n_per_thread}_{dtype_s}_n111_a6"
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

    # A.6: Monolithic PC layout — MM_FLAG_BIAS gates the bias-add branch.
    pc = _pack_mm_pc(
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
        stride_bias_n=stride_bias_n,
        flags=MM_FLAG_BIAS,
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
    from ....runtime import compile_and_dispatch

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
        # A.6: ``_a6`` tag — monolithic PC layout (closes the 10I undercount).
        cache_key = (
            f"slang_addmm_epi_OpGELU_{tile_m}_{tile_n}_{tile_k}_s{num_stages}"
            f"_r{m_per_thread}x{n_per_thread}_{dtype_s}_a6"
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

    # A.6: Pre-A.6 this path packed only 10 uints, leaving pc.tile_m /
    # pc.tile_n / pc.tile_k / pc.m_per_thread / pc.n_per_thread (the N+1.11
    # runtime tile fields) reading garbage on the GPU.  The unified packer
    # writes the full layout, closing that latent miscompute risk.
    pc = _pack_mm_pc(
        M,
        N,
        K,
        a.stride(0),
        1,
        b.stride(0),
        1,
        out.stride(0),
        out.stride(1),
        tile_m,
        tile_n,
        tile_k,
        m_per_thread,
        n_per_thread,
        stride_bias_n=1,
        flags=MM_FLAG_BIAS,
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
    from ....runtime import compile_and_dispatch

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
            f"_r{m_per_thread}x{n_per_thread}_{dtype_s}_n111_a6"
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

    # A.6: Monolithic PC layout — MM_FLAG_BATCH selects the batch path.
    pc = _pack_mm_pc(
        M,
        N,
        K,
        a.stride(1),
        a.stride(2),
        b.stride(1),
        b.stride(2),
        out.stride(1),
        out.stride(2),
        tile_m,
        tile_n,
        tile_k,
        m_per_thread,
        n_per_thread,
        stride_a_b=a.stride(0),
        stride_b_b=b.stride(0),
        stride_c_b=out.stride(0),
        flags=MM_FLAG_BATCH,
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


def _slang_tiles_enabled() -> bool:
    """Check if Slang tile matmul shaders are enabled for autotune.

    Enabled by default as of M17.1 (2026-05-16) — the three compounding
    blockers (E39999 numthreads, VUID-07988 binding mismatch, slangc 2026.5.2
    barrier + lid indexing bugs on wave64) are all resolved.  Forward matmul
    produces correct results: max diff ~1e-5 vs CPU across all tile configs.

    Set ``TORCH_VULKAN_DISABLE_SLANG_TILES=1`` to fall back to the ATEN
    (C++ vulkan_mm/vulkan_bmm) path for bisecting / debugging.
    """
    return os.environ.get("TORCH_VULKAN_DISABLE_SLANG_TILES") != "1"


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

    M-NEW.3: filter sub-wave workgroups (numthreads < subgroup_size)
    and non-wave-aligned shapes — the M27 in-process validator rejects
    those anyway, so emitting them only wastes ~10 s of slangc per
    config per cold compile.
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

    # N+1.12: wave32 prefers smaller tiles, wave64 prefers single-wave.
    # M17.1: On wave64 (RDNA1), restrict to tile configs where the workgroup
    # fits in a single wave (<= 64 threads).  Multi-wave workgroups have a
    # broken GroupMemoryBarrierWithGroupSync() in slangc 2026.5.2 -- the
    # barrier appears to use Subgroup scope instead of Workgroup scope,
    # so threads in different waves cannot see each other's LDS writes.
    sgs = _get_device_subgroup_size()
    if sgs == 64:
        # M17.1: cap WG at one wave (64 threads, barrier bug).
        # M-NEW.3: also require the WG to be a multiple of the wave size
        # (1×64, 2×64, …). For the 1-opt path that means WG ∈ {64} on
        # RDNA1, so the existing ``c[0] * c[1] <= 64`` filter is
        # effectively wave-aligned for the canonical 8×8 tile set.
        return [
            c
            for c in _MM_TILE_CONFIGS
            if c[0] * c[1] <= sgs and (c[0] * c[1]) % sgs == 0
        ]
    if sgs == 32:
        # Wave32: prefer max(TILE_M, TILE_N) <= 32 AND wave-aligned.
        return [
            c
            for c in _MM_TILE_CONFIGS
            if max(c[0], c[1]) <= 32 and (c[0] * c[1]) % sgs == 0
        ]
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

    M-NEW.3: filter out sub-wave / wave-misaligned workgroups. The
    M27 validator (``slang_validate/workgroup.py``) rejects any kernel
    whose ``numthreads`` product isn't a multiple of the wave size, so
    emitting (e.g.) a 32-thread WG on wave64 hardware wastes a slangc
    cold compile and is silently ignored by the autotuner. Audit Agent
    3 measured 4/14 (29 %) of post-M-NEW.1 addmm choices being
    rejected for this reason — all from
    ``(64, 32, 16, 8, 8)`` and ``(32, 64, 16, 8, 8)`` register tiles
    that produce 8×4=32 / 4×8=32 thread blocks on wave64.

    Set ``TORCH_VULKAN_NO_REGISTER_TILE=1`` to disable register-tile entries —
    Inductor's autotune will fall back to the (much smaller) legacy 1-output-
    per-thread set + aten_mm. Useful for bisecting register-tile correctness
    issues against the legacy path."""
    if os.environ.get("TORCH_VULKAN_NO_REGISTER_TILE") == "1":
        return []

    # N+1.12: wave32/wave64 filter register-tile configs.
    # M17.1: On wave64, only single-wave workgroups work (barrier bug).
    sgs = _get_device_subgroup_size()

    def _wg_threads(c: tuple[int, int, int, int, int]) -> int:
        return (c[0] // c[3]) * (c[1] // c[4])

    if sgs == 64:
        # M17.1 cap + M-NEW.3 wave alignment: WG must be exactly one
        # wave (64) on RDNA1 — barrier bug forbids multi-wave WGs, and
        # sub-wave WGs trigger M27 advisory rejection.
        return [
            c
            for c in _MM_REGISTER_TILE_CONFIGS
            if _wg_threads(c) == sgs
        ]
    if sgs == 32:
        # Wave32: register-heavy gate AND wave alignment.
        return [
            c
            for c in _MM_REGISTER_TILE_CONFIGS
            if c[3] * c[4] <= 16
            and _wg_threads(c) > 0
            and _wg_threads(c) % sgs == 0
        ]
    return _MM_REGISTER_TILE_CONFIGS


# ═══════════════════════════════════════════════════════════════════════════
# OP.24 — Int8 matmul dispatch
# ═══════════════════════════════════════════════════════════════════════════


def _slang_tile_mm_int8(
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
) -> None:
    """Execute tiled int8 matmul: out = A @ B (int8×int8→int32→float32).

    OP.24: A and B are ``torch.int8`` tensors packed as ``StructuredBuffer<uint>``
    (4×int8 per uint32 word). The shader unpacks at load time, accumulates
    in int32, and stores float32. Output ``out`` is ``torch.float32``.

    Args:
        tile_m, tile_n, tile_k: Tile dimensions.
        a: int8 weight tensor [M, K].
        b: int8 activation tensor [K, N].
        out: float32 output tensor [M, N] (pre-allocated).
        src: Pre-rendered Slang source (optional; renders if None).
        cache_key: SPIR-V cache key (optional; generated if None).
        m_per_thread, n_per_thread: Register-tile depth.
    """
    from ....runtime import compile_and_dispatch

    M, K = a.shape
    _, N = b.shape

    if src is None or cache_key is None:
        src = _render_mm_int8_slang(
            tile_m,
            tile_n,
            tile_k,
            m_per_thread=m_per_thread,
            n_per_thread=n_per_thread,
        )
        cache_key = (
            f"slang_mm_int8_{tile_m}_{tile_n}_{tile_k}"
            f"_r{m_per_thread}x{n_per_thread}_n111"
        )

    if not _TRUST_INDUCTOR:
        if not a.is_contiguous():
            a = a.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()
        if not out.is_contiguous():
            out = out.contiguous()

    # Pass actual element-level strides. The shader divides by 4 internally
    # to address uint32 words (4×int8 per word).
    stride_a_m = a.stride(0)
    stride_a_k = a.stride(1) if a.dim() > 1 else 1
    stride_b_k = b.stride(0)
    stride_b_n = b.stride(1) if b.dim() > 1 else 1

    # N+1.11 PC layout: M, N, K, stride_a_m, stride_a_k, stride_b_k,
    # stride_b_n, stride_c_m, stride_c_n
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
        entry="computeMain",
        cache_key=cache_key,
    )
