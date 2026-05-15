"""GEMM backward pass.

Backward-pass dispatch helpers for matmul operations.
Includes two backward strategies:
  - T4.2: Forward-template reuse with stride-based transposition
  - CG.M5: Single-kernel fused backward via [Differentiable] tile_inner_madd
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

from ....vulkan_template_caller import (
    _dtype_to_slang,
)
from .dispatch import _TRUST_INDUCTOR
from .render import (
    _render_mm_backward_slang,
    _render_mm_bwd_slang,
)

# ═══════════════════════════════════════════════════════════════════════════
# T4.2 — Matmul backward via forward template reuse
# ═══════════════════════════════════════════════════════════════════════════


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
    from ....runtime import compile_and_dispatch

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
    from ....runtime import compile_and_dispatch

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
    from ....runtime import compile_and_dispatch

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
