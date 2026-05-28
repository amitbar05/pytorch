"""GEMM template classes.

Picklable callable classes for tiled Slang GEMM variants and their factory
functions.  Each class pins a specific flag combination for backward-compat
pickling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    pass

from ....buffer_pool import pool_acquire
from ....vulkan_template_caller import (
    _dtype_to_slang,
    _validate_epilogue_struct,
)
from .dispatch import (
    _get_device_subgroup_size,
    _slang_tile_addmm,
    _slang_tile_addmm_gelu,
    _slang_tile_bmm,
    _slang_tile_mm,
    _slang_tile_mm_int8,
)
from .render import (
    _render_mm_int8_slang,
    _render_mm_linktime_wrapper_slang,
    _render_mm_slang,
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
        has_bias  — emit ``out = a @ b + bias`` (addmm path).
        has_batch — operate on 3D tensors ``[B, M, K] @ [B, K, N]`` (bmm path).
        epilogue  — Slang IDifferentiable struct name (e.g. ``"OpGELU"``) applied
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

        src = _render_mm_slang(self.tile_m, self.tile_n, self.tile_k, use_module=False, **render_kwargs)

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
                f"{_sgs_tag}{_ld_tag}_n111_a6"
            )
        elif self.has_bias and self.epilogue == "OpGELU":
            cache_key = (
                f"slang_addmm_epi_OpGELU_{self.tile_m}_{self.tile_n}_{self.tile_k}"
                f"_s{self.num_stages}"
                f"_r{self.m_per_thread}x{self.n_per_thread}_{dtype_s}"
                f"{_sgs_tag}{_ld_tag}_n111_a6"
            )
        elif self.has_bias:
            cache_key = (
                f"slang_addmm_{self.tile_m}_{self.tile_n}_{self.tile_k}"
                f"_s{self.num_stages}"
                f"_r{self.m_per_thread}x{self.n_per_thread}_{dtype_s}"
                f"{_sgs_tag}{_ld_tag}_n111_a6"
            )
        else:
            cache_key = (
                f"slang_mm_{self.tile_m}_{self.tile_n}_{self.tile_k}"
                f"_s{self.num_stages}"
                f"_r{self.m_per_thread}x{self.n_per_thread}_{dtype_s}"
                f"{_sgs_tag}{_ld_tag}_n111_a6"
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
# OP.24 — Int8 matmul class
# ═══════════════════════════════════════════════════════════════════════════


class _SlangTileMMInt8:
    """Picklable callable for tiled Slang int8 matmul — inference only.

    OP.24: int8×int8 → int32 accumulation → float32 output. A and B are
    ``torch.int8`` tensors (packed 4× per uint32 in Vulkan storage);
    output is ``torch.float32``.

    Register-tile semantics: ``m_per_thread`` / ``n_per_thread`` > 1 enable
    register-tiling where each thread holds an ``(m_per_thread, n_per_thread)``
    accumulator block in int32 registers.
    """

    __slots__ = (
        "tile_m",
        "tile_n",
        "tile_k",
        "m_per_thread",
        "n_per_thread",
        "__name__",
        "_per_dtype",
    )

    def __init__(
        self,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        m_per_thread: int = 1,
        n_per_thread: int = 1,
    ):
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.m_per_thread = m_per_thread
        self.n_per_thread = n_per_thread
        self.__name__ = (
            f"slang_mm_int8_{tile_m}_{tile_n}_{tile_k}_r{m_per_thread}x{n_per_thread}"
        )
        self._per_dtype: dict[str, tuple[str, str]] = {}

    def _src_and_key(self, _dtype_s: str = "int8") -> tuple[str, str]:
        cached = self._per_dtype.get("int8")
        if cached is not None:
            return cached

        src = _render_mm_int8_slang(
            self.tile_m,
            self.tile_n,
            self.tile_k,
            m_per_thread=self.m_per_thread,
            n_per_thread=self.n_per_thread,
        )
        cache_key = (
            f"slang_mm_int8_{self.tile_m}_{self.tile_n}_{self.tile_k}"
            f"_r{self.m_per_thread}x{self.n_per_thread}_n111"
        )
        cached = (src, cache_key)
        self._per_dtype["int8"] = cached
        return cached

    def __call__(
        self, a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Execute int8 tiled matmul: out = a @ b.

        Args:
            a: int8 tensor [M, K].
            b: int8 tensor [K, N].
            out: float32 tensor [M, N] (optional; allocated if None).

        Returns:
            float32 tensor [M, N].
        """
        if out is None:
            out_size = (a.shape[0], b.shape[1])
            out = pool_acquire(out_size, torch.float32, a.device)
            if out is None:
                out = torch.empty(out_size, device=a.device, dtype=torch.float32)
        src, cache_key = self._src_and_key()
        _slang_tile_mm_int8(
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
        return (
            _SlangTileMMInt8,
            (
                self.tile_m,
                self.tile_n,
                self.tile_k,
                self.m_per_thread,
                self.n_per_thread,
            ),
        )


def _make_tile_mm_int8_fn(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_per_thread: int = 1,
    n_per_thread: int = 1,
):
    """Factory for picklable int8 matmul callables.

    Returns a :class:`_SlangTileMMInt8` instance suitable for appending to
    ``torch._inductor.config.external_matmul``.
    """
    return _SlangTileMMInt8(
        tile_m, tile_n, tile_k, m_per_thread=m_per_thread, n_per_thread=n_per_thread
    )
