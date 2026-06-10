"""Flash attention template callers.

Provides rendering, dispatch, and installation for the flash attention and
flash attention backward Slang templates.
"""

from __future__ import annotations

import os
import struct
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

from ...buffer_pool import pool_acquire, pool_acquire_scratch, pool_release_scratch
from ...vulkan_template import _load_slang_template
from ...vulkan_template_caller import _dtype_to_slang

_TRUST_INDUCTOR = os.environ.get("TORCH_VULKAN_TRUST_INDUCTOR") == "1"


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
        raise RuntimeError("flash_attention.slang template not found")

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
    from ...runtime import compile_and_dispatch

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

    # CG.M15: PC now includes bq, bk for runtime tile sizes.
    # 6 uints + 1 float + 2 uints + 2 uints = 10I + 1f
    pc = struct.pack(
        "6IfII2I",
        B,
        H,
        KV_H,
        N,
        S,
        D,
        float(scale),
        int(is_causal),
        int(head_layout == "bshd"),
        BQ,
        BK,
    )

    # T4.10: dispatch grid uses the caller-supplied BQ so tile-size
    # autotune correctly sizes the q-tile axis.  (BK does not appear in
    # the grid math — it parameterises the inner kv-loop, not the
    # workgroup sweep.)
    grid_x = B
    grid_y = H * ((N + BQ - 1) // BQ)
    grid_z = 1

    # CG.M15: spec_constants for [[vk::constant_id]] overrides.
    spec_constants = [
        (10, BQ),
        (11, BK),
        (12, D),
    ]

    compile_and_dispatch(
        src,
        [q, k, v, out, lse],
        grid_x,
        grid_y,
        grid_z,
        push_constants=pc,
        num_outputs=2,
        cache_key=cache_key,
        spec_constants=spec_constants,
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
# CG.M7 — SDPA backward via flash_attention_bwd.slang
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
        raise RuntimeError("flash_attention_bwd.slang template not found")

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
    from ...runtime import compile_and_dispatch

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
    # M20.3: BQ / BK / HEAD_DIM are spec constants (IDs 13-15) so the
    # SPIR-V hash collapses to (head_layout,) — is_causal is a runtime
    # push-constant flag and was never SPV-affecting in the backward.
    # The same SPIR-V module serves every (BQ, BK, head_dim) combo;
    # the tuple is applied as a pipeline spec-constant override below.
    cache_key = (
        f"slang_flash_attention_bwd_m20p3_{head_layout}"
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

    # M20.3: Vulkan spec-constant overrides for (BQ, BK, HEAD_DIM).
    spec_constants = [
        (13, BQ),
        (14, BK),
        (15, D),
    ]

    compile_and_dispatch(
        src,
        [q, k, v, lse, dO, dQ, dK, dV],
        grid_x,
        grid_y,
        grid_z,
        push_constants=pc,
        num_outputs=3,
        cache_key=cache_key,
        spec_constants=spec_constants,
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
            # M20.3: spec constants 13-15 carry (BQ, BK, HEAD_DIM); the
            # cache key collapses to head_layout only because every
            # other axis is either a runtime push-const (is_causal,
            # head_dim semantics) or a pipeline-spec override.
            self._cache_key = "slang_flash_attention_bwd_m20p3_bhsd"
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
    the flash_attention.slang template instead of the C++ eager extern.

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
    from ...fx_passes.eager_patches import _ensure_flash_attention_op_registered

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
    from ...runtime import _slangc_available, prewarm_compile

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
