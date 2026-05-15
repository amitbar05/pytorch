"""FFT Stockham template callers.

Provides rendering, dispatch, and installation for the Stockham FFT Slang template.
"""

from __future__ import annotations

import os
import struct
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

from ...vulkan_template import _load_slang_template
from ...vulkan_template_caller import _dtype_to_slang



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
    from ...runtime import compile_and_dispatch

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
