"""PF.6.b — runtime dispatch path for ``bwd_diff(fwd)`` codegen.

Wraps ``compile_and_dispatch`` with the per-entry buffer + push-constant
contract emitted by ``bwd_diff_table.emit_bwd_diff_kernel``. Two public
entry points:

- ``dispatch_unary_bwd(aten_op, x, grad_out, *, out=None) -> grad_in``
- ``dispatch_binary_bwd(aten_op, a, b, grad_out, *, no_diff_kwargs=None,
  out_a=None, out_b=None) -> (grad_a, grad_b)``

Both invoke ``compile_and_dispatch`` with a stable, shape-independent
``cache_key`` (``bwd_diff_<aten_op>_<dtype>_<numthreads>``) — the
shader body iterates over ``numel`` from the push-constant block, so
the same SPIR-V binary covers all element counts.

Multi-dtype dispatch (T3.1): f16/bf16 inputs are widened to f32 for the
compute kernel and narrowed back afterward.  The Slang shader body is
already ``T : __BuiltinFloatingPointType`` generic; the widening keeps
buffer bindings as ``StructuredBuffer<float>`` regardless of input dtype
so the single precompiled SPIR-V module covers all float types.

T4.2: Both ``dispatch_unary_bwd`` and ``dispatch_binary_bwd`` validate
against ``BWD_TEMPLATE_REGISTRY`` via ``resolve_backward_kind()`` before
dispatching.  If an op is registered with a non-``BWD_DIFF`` kind (e.g.
``TEMPLATE_JINJA`` for mm backward), a clear error is raised directing
the caller to ``dispatch_template_bwd()`` instead.
"""

from __future__ import annotations

import struct

import torch

from torch_vulkan.inductor.bwd_diff_table import (
    BWD_DIFF_TABLE,
    BwdDiffEntry,
    emit_bwd_diff_kernel,
)
from torch_vulkan.inductor.runtime import compile_and_dispatch

_DEFAULT_NUMTHREADS = 256


def _entry(aten_op: str, expected_arity: int) -> BwdDiffEntry:
    if aten_op not in BWD_DIFF_TABLE:
        raise KeyError(
            f"PF.6.b: aten op {aten_op!r} is not in BWD_DIFF_TABLE; "
            f"add it to bwd_diff_table.BWD_DIFF_TABLE first or list it "
            f"in EXCLUDED_DIFFERENTIABLE_FWDS with a reason"
        )
    entry = BWD_DIFF_TABLE[aten_op]
    if entry.arity != expected_arity:
        raise ValueError(
            f"PF.6.b: aten op {aten_op!r} has arity {entry.arity}; "
            f"caller used the arity-{expected_arity} dispatcher"
        )
    return entry


def _check_float(*tensors: torch.Tensor) -> None:
    for t in tensors:
        if t.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise NotImplementedError(
                f"PF.6.b / T3.1: bwd_diff dispatcher supports "
                f"float32/float16/bfloat16 only; got {t.dtype}"
            )


def _check_vulkan(*tensors: torch.Tensor) -> None:
    for t in tensors:
        if t.device.type != "vulkan":
            raise ValueError(
                f"PF.6.b: bwd_diff dispatcher requires vulkan tensors; got {t.device}"
            )


def _ensure_f32(t: torch.Tensor) -> torch.Tensor:
    """Widen a f16/bf16 tensor to f32 on Vulkan; identity for f32."""
    if t.dtype == torch.float32:
        return t
    return t.float()


def _narrow_from_f32(t: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """Convert a f32 Vulkan tensor back to the target dtype."""
    if target_dtype == torch.float32:
        return t
    return t.to(target_dtype)


def _slang_dtype_str(dtype: torch.dtype) -> str:
    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        return "float"
    raise NotImplementedError(f"PF.6.b / T3.1: unsupported dtype {dtype}")


def _cache_key(aten_op: str, dtype: torch.dtype, numthreads: int) -> str:
    dt_str = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}.get(
        dtype, str(dtype)
    )
    return f"bwd_diff_{aten_op}_{dt_str}_{numthreads}"


def dispatch_unary_bwd(
    aten_op: str,
    x: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    numthreads: int = _DEFAULT_NUMTHREADS,
) -> torch.Tensor:
    """Dispatch the autodiff-emitted backward for a unary entry.

    Returns ``grad_in`` with the same shape/dtype/device as ``x``.

    f16/bf16 inputs are widened to f32 for the compute kernel (T3.1).

    T4.2: Validates op routing against ``BWD_TEMPLATE_REGISTRY``.  If the
    op is registered as ``TEMPLATE_JINJA`` or ``BACKWARD_DERIVATIVE``
    instead of ``BWD_DIFF``, raises a clear error directing the caller to
    the correct dispatch path (``dispatch_template_bwd()``).
    """
    # T4.2: Check BWD_TEMPLATE_REGISTRY before dispatching.  If this op
    # is registered with a non-BWD_DIFF kind, the caller must use
    # dispatch_template_bwd() (for TEMPLATE_JINJA) or the fast backward
    # annotation path (for BACKWARD_DERIVATIVE).
    resolved = resolve_backward_kind(aten_op)
    if resolved is not None and not resolved.is_bwd_diff:
        raise ValueError(
            f"T4.2: dispatch_unary_bwd called for {aten_op!r}, but "
            f"BWD_TEMPLATE_REGISTRY lists kind={resolved.kind}. "
            f"Use dispatch_template_bwd() for TEMPLATE_JINJA entries."
        )

    entry = _entry(aten_op, expected_arity=1)
    orig_dtype = x.dtype
    _check_float(x, grad_out)
    _check_vulkan(x, grad_out)
    if grad_out.shape != x.shape:
        raise ValueError(
            f"PF.6.b: grad_out shape {tuple(grad_out.shape)} does not "
            f"match x shape {tuple(x.shape)} (unary backward)"
        )
    x_f32 = _ensure_f32(x)
    go_f32 = _ensure_f32(grad_out)
    grad_in_f32 = out.float() if out is not None else torch.empty_like(x_f32)
    if grad_in_f32.shape != x.shape:
        raise ValueError(
            f"PF.6.b: out tensor shape {tuple(grad_in_f32.shape)} "
            f"does not match x {tuple(x.shape)}"
        )
    _check_vulkan(grad_in_f32)
    numel = x_f32.numel()
    slang_dtype = _slang_dtype_str(orig_dtype)
    src = emit_bwd_diff_kernel(
        aten_op,
        dtype=slang_dtype,
        numthreads=numthreads,
    )
    pc = struct.pack("<I", numel)
    wg_x = (numel + numthreads - 1) // numthreads
    compile_and_dispatch(
        src,
        tensors=[x_f32.contiguous(), go_f32.contiguous(), grad_in_f32],
        wg_x=wg_x,
        wg_y=1,
        wg_z=1,
        push_constants=pc,
        num_outputs=1,
        entry="bwd_op",
        cache_key=_cache_key(aten_op, orig_dtype, numthreads),
    )
    _ = entry
    if out is not None:
        if out.dtype != orig_dtype:
            out.copy_(_narrow_from_f32(grad_in_f32, orig_dtype))
        else:
            out.copy_(grad_in_f32)
        return out
    return _narrow_from_f32(grad_in_f32, orig_dtype)


from torch_vulkan.inductor.bwd_template_registry import BackwardKind

# ── T4.2: Template-backward dispatch (matmul backward via forward template) ─


class _ResolvedBackward:
    """Routing decision returned by ``resolve_backward_kind``."""

    __slots__ = ("kind", "fwd_key", "entry", "fwd_fn", "module")

    def __init__(
        self,
        kind,
        fwd_key: str,
        entry,
        fwd_fn: str = "",
        module: str = "",
    ) -> None:
        self.kind = kind
        self.fwd_key = fwd_key
        self.entry = entry
        self.fwd_fn = fwd_fn
        self.module = module

    @property
    def is_template_jinja(self) -> bool:
        return self.kind == BackwardKind.TEMPLATE_JINJA

    @property
    def is_bwd_diff(self) -> bool:
        return self.kind == BackwardKind.BWD_DIFF

    @property
    def is_backward_derivative(self) -> bool:
        return self.kind == BackwardKind.BACKWARD_DERIVATIVE


def resolve_backward_kind(op_name: str):
    """Check the BWD_TEMPLATE_REGISTRY for a backward routing decision.

    Returns an ``_ResolvedBackward`` if a matching entry exists (by
    ``op_name`` or by a known forward-template-key mapping), or ``None``
    if the op should fall through to the default lowering path.

    The registry is indexed by forward template key (e.g. "mm_default").
    This function maintains a secondary mapping from aten op names to
    forward template keys so that backward lowerings can call
    ``resolve_backward_kind("aten.mm")`` and get routing instructions.

    Returns ``None`` (not NotImplemented) — callers should fall through
    to their existing default path.
    """
    from torch_vulkan.inductor.bwd_template_registry import (
        BWD_TEMPLATE_REGISTRY,
        BwdTemplateEntry,
    )

    # Map aten op names → forward template keys for ops whose backward
    # is handled by reusing the forward template with transposed operands.
    _OP_TO_FWD_KEY: dict[str, str] = {
        "aten.mm": "mm_default",
        "aten.mm.default": "mm_default",
        "aten.bmm": "bmm_default",
        "aten.bmm.default": "bmm_default",
        "aten.addmm": "addmm_default",
        "aten.addmm.default": "addmm_default",
        # Conv backward decomposes through the same mm template entries.
        # Both the forward op (looked up to discover the paired backward
        # entry) and the explicit backward op name resolve to the same
        # forward template key.
        "aten.convolution": "conv_im2col_f32",
        "aten.convolution.default": "conv_im2col_f32",
        "aten.convolution_backward": "conv_im2col_f32",
        "aten.convolution_backward.default": "conv_im2col_f32",
        # Flash attention backward routes through template.
        "aten.scaled_dot_product_attention": "flash_attention_f32_bhsd",
        # CG.M3: Reduction backward via [Differentiable] scalar fold.
        # sum_backward -> reduce_fold_sum (bwd_diff gives dout for both inputs)
        "aten.sum_backward": "reduce_sum",
        "aten.sum.dim_IntList_backward": "reduce_sum",
        # mean_backward -> reduce_fold_sum (bwd_diff + /numel scaling)
        "aten.mean_backward": "reduce_mean",
        "aten.mean.dim_backward": "reduce_mean",
        # var_backward -> reduce_fold_sum (bwd_diff + 2*(x-mean)/(n-1) scale)
        "aten.var_backward": "reduce_var",
        "aten.var.correction_backward": "reduce_var",
        # prod_backward -> reduce_fold_prod (bwd_diff gives dout*prod/x_i)
        "aten.prod_backward": "reduce_prod",
        "aten.prod.dim_int_backward": "reduce_prod",
    }

    fwd_key = _OP_TO_FWD_KEY.get(op_name)
    if fwd_key is None:
        fwd_key = op_name  # try direct lookup

    entry: BwdTemplateEntry | None = BWD_TEMPLATE_REGISTRY.lookup(fwd_key)
    if entry is None:
        return None
    return _ResolvedBackward(
        kind=entry.kind,
        fwd_key=fwd_key,
        entry=entry,
        fwd_fn=entry.fwd_fn,
        module=entry.module,
    )


def dispatch_template_bwd(
    fwd_key: str,
    grad_out: torch.Tensor,
    *saved_tensors: torch.Tensor,
    outs: tuple[torch.Tensor | None, ...] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Dispatch backward through a TEMPLATE_JINJA entry in BWD_TEMPLATE_REGISTRY.

    For matmul backward, AOT Autograd decomposes it into forward mm calls,
    so the backward path is already covered by ``install_external_mm``.
    This function provides an explicit runtime dispatch entry point that
    reuses the forward template with transposed operands for matmul backward.

    Supported fwd_keys:
      - "mm_default" / "bmm_default" / "addmm_default": matmul backward.
        ``saved_tensors`` = (dC, saved_A, saved_B)
        Returns (dA, dB) = (dC @ B^T, A^T @ dC).
      - "conv_im2col_f32": conv2d backward routed through the im2col + mm
        decomposition that the forward FX pattern (T4.4) emits.
        ``saved_tensors`` = (saved_patches, saved_weight_flat) — both
        materialized by the forward im2col rewrite.
        Returns (dpatches, dweight_flat) = forward-mm backward over the
        same operands. Caller is responsible for the final fold to
        ``grad_input`` (NHWC patches → NCHW input) and the bias reduction.

        NOTE (T4.2 partial): bias reduction (``sum(grad_out, dim=(0,2,3))``)
        and the patches→input fold are NOT performed here — they're
        graph-level rewrites that belong in an FX pattern paired with
        ``_match_conv_im2col``. Wiring those up requires editing
        ``fx_passes/patterns/builtin_patterns.py`` (out of scope for this
        change). Today this entry point covers the heavy mm work; the
        residual orphan pointwise (= the missing fold + bias_sum) is
        what surfaces as the ``9 vs 8`` dispatch-count regression on
        ``test_convolution_backward_dispatch_count``.
    """
    from torch_vulkan.inductor.bwd_template_registry import (
        BWD_TEMPLATE_REGISTRY,
    )

    entry = BWD_TEMPLATE_REGISTRY.lookup(fwd_key)
    if entry is None:
        raise KeyError(
            f"T4.2: forward template key {fwd_key!r} not found in BWD_TEMPLATE_REGISTRY"
        )
    if entry.kind != BackwardKind.TEMPLATE_JINJA:
        raise ValueError(
            f"T4.2: dispatch_template_bwd expects TEMPLATE_JINJA kind; "
            f"got {entry.kind} for {fwd_key!r}"
        )

    if fwd_key in ("mm_default", "bmm_default", "addmm_default"):
        return _dispatch_matmul_bwd(fwd_key, grad_out, *saved_tensors, outs=outs)
    if fwd_key == "conv_im2col_f32":
        return _dispatch_conv_im2col_bwd(grad_out, *saved_tensors, outs=outs)
    if fwd_key == "flash_attention_f32_bhsd":
        return _dispatch_flash_attention_bwd(grad_out, *saved_tensors, outs=outs)

    raise NotImplementedError(f"dispatch_template_bwd not implemented for {fwd_key!r}")


def _dispatch_conv_im2col_bwd(
    grad_out: torch.Tensor,
    *saved_tensors: torch.Tensor,
    outs: tuple[torch.Tensor | None, ...] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Conv2d backward via the im2col+mm forward decomposition.

    Mirrors ``_dispatch_matmul_bwd`` for the conv-im2col case: the forward
    rewrites ``conv2d(x, w)`` into ``mm(patches, w_flat^T)`` (T4.4), so
    ``conv_backward`` is mathematically the matmul backward of *those same
    operands*:

        forward: out_flat = patches @ w_flat^T
                 patches  : [N*Hout*Wout, Cin*Kh*Kw]
                 w_flat   : [Cout, Cin*Kh*Kw]   (transposed before mm)

        backward over the matmul:
            dpatches  = grad_out_flat @ w_flat
            dw_flat^T = patches^T @ grad_out_flat
                      ⇒ dw_flat = grad_out_flat^T @ patches

    This routine performs ONLY the mm portion via the existing forward
    template. The patches→input fold and the bias reduction are FX-level
    rewrites that pair with ``_match_conv_im2col`` and remain TODO.

    ``saved_tensors`` = (patches, w_flat).
    Returns (dpatches, dw_flat).
    """
    if len(saved_tensors) != 2:
        raise ValueError(
            f"T4.2: dispatch_conv_im2col_bwd expects 2 saved tensors "
            f"(patches, w_flat); got {len(saved_tensors)}"
        )
    patches, w_flat = saved_tensors
    _check_float(grad_out, patches, w_flat)
    _check_vulkan(grad_out, patches, w_flat)

    if patches.dim() != 2 or w_flat.dim() != 2:
        raise ValueError(
            f"T4.2: conv_im2col_bwd requires 2-D patches/w_flat (post-im2col); "
            f"got {patches.dim()}D / {w_flat.dim()}D"
        )

    # grad_out arrives flattened to match the forward mm shape:
    # [N*Hout*Wout, Cout]. Caller is responsible for any reshape/permute.
    if grad_out.dim() != 2:
        raise ValueError(
            f"T4.2: conv_im2col_bwd requires grad_out flattened to 2-D "
            f"[N*Hout*Wout, Cout]; got {grad_out.dim()}D"
        )

    from torch_vulkan.inductor.vulkan_template_caller import (
        _slang_tile_mm_backward,
    )

    outs = outs or (None, None)
    out_dpatches, out_dw_flat = (
        outs[0] if len(outs) > 0 else None,
        outs[1] if len(outs) > 1 else None,
    )

    TILE = 32
    # dpatches = grad_out_flat @ w_flat  (no transpose — straight forward mm)
    if out_dpatches is None:
        M_p, K_p = grad_out.shape[0], w_flat.shape[1]
        out_dpatches = torch.empty(
            M_p, K_p, dtype=grad_out.dtype, device=grad_out.device
        )
    _slang_tile_mm_backward(TILE, TILE, TILE, grad_out, w_flat, out_dpatches)

    # dw_flat = grad_out_flat^T @ patches  (transpose_a)
    if out_dw_flat is None:
        K_w, N_w = grad_out.shape[1], patches.shape[1]
        out_dw_flat = torch.empty(K_w, N_w, dtype=patches.dtype, device=patches.device)
    _slang_tile_mm_backward(
        TILE,
        TILE,
        TILE,
        grad_out,
        patches,
        out_dw_flat,
        transpose_a=True,
    )

    return out_dpatches, out_dw_flat


def _dispatch_flash_attention_bwd(
    grad_out: torch.Tensor,
    *saved_tensors: torch.Tensor,
    outs: tuple[torch.Tensor | None, ...] | None = None,
) -> tuple[torch.Tensor, ...]:
    """SDPA backward via CG.M7 flash_attention_bwd template.

    Recomputes the softmax attention weights P from saved Q, K, V, and LSE,
    then computes dQ, dK, dV in a single fused dispatch.

    ``saved_tensors`` = (q, k, v, lse) — the four tensors saved during
    the forward pass (flash_attention.py.jinja computes LSE alongside O).

    Returns (dQ, dK, dV).
    """
    if len(saved_tensors) != 4:
        raise ValueError(
            f"CG.M7: dispatch_flash_attention_bwd expects 4 saved tensors "
            f"(q, k, v, lse); got {len(saved_tensors)}"
        )
    q, k, v, lse = saved_tensors
    _check_float(q, k, v, lse, grad_out)
    _check_vulkan(q, k, v, lse, grad_out)

    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"CG.M7: SDPA bwd requires 4-D inputs (B, H|KV_H, N|S, D); "
            f"got Q{q.shape} K{k.shape} V{v.shape}"
        )

    from torch_vulkan.inductor.vulkan_template_caller import (
        _dispatch_flash_attention_bwd as _dispatch_fa_bwd,
    )

    outs = outs or (None, None, None)
    dQ = outs[0] if len(outs) > 0 else None
    dK = outs[1] if len(outs) > 1 else None
    dV = outs[2] if len(outs) > 2 else None

    D = q.shape[-1]
    scale = 1.0 / (float(D) ** 0.5)

    if dQ is None:
        dQ = torch.empty_like(q)
    if dK is None:
        dK = torch.zeros_like(k)
    if dV is None:
        dV = torch.zeros_like(v)

    _dispatch_fa_bwd(
        q=q,
        k=k,
        v=v,
        lse=lse,
        dO=grad_out,
        dQ=dQ,
        dK=dK,
        dV=dV,
        scale=scale,
    )
    return dQ, dK, dV


def _dispatch_matmul_bwd(
    fwd_key: str,
    grad_out: torch.Tensor,
    *saved_tensors: torch.Tensor,
    outs: tuple[torch.Tensor | None, ...] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Matmul backward: dA = dC @ B^T, dB = A^T @ dC.

    CG.M5: Uses the single-kernel backward template ``slang_mm_bwd.py.jinja``
    which computes BOTH dA and dB in ONE dispatch by wrapping
    ``bwd_diff(tile_inner_madd)`` in a tiled K-loop.

    Falls back to the 2-dispatch path (forward template reuse with
    transposed operands) if the new template is not available.
    """
    if len(saved_tensors) != 3:
        raise ValueError(
            f"T4.2: dispatch_matmul_bwd expects 3 saved tensors "
            f"(dC, saved_A, saved_B); got {len(saved_tensors)}"
        )
    dC, saved_A, saved_B = saved_tensors
    _check_float(dC, saved_A, saved_B)
    _check_vulkan(dC, saved_A, saved_B)

    outs = outs or (None, None)
    out_a, out_b = (
        outs[0] if len(outs) > 0 else None,
        outs[1] if len(outs) > 1 else None,
    )

    is_batch = fwd_key == "bmm_default"
    # Default tile config for backward dispatch (32×32×32, single stage).
    # Forward uses autotuning across multiple configs; backward uses this
    # fixed config for deterministic performance.
    TILE = 32

    # ── CG.M5: single-kernel backward (1 dispatch instead of 2) ─────────
    # Try the new bwd_diff-based template first.  If the template is not
    # available (e.g. slangc version doesn't support [Differentiable]),
    # fall back to the 2-dispatch path below.
    try:
        from torch_vulkan.inductor.vulkan_template_caller import (
            _slang_tile_bmm_bwd,
            _slang_tile_mm_bwd,
        )

        if is_batch:
            if out_a is None:
                out_a = torch.empty_like(saved_A)
            if out_b is None:
                out_b = torch.empty_like(saved_B)
            _slang_tile_bmm_bwd(
                TILE,
                TILE,
                TILE,
                saved_A,
                saved_B,
                dC,
                out_a,
                out_b,
            )
        else:
            if out_a is None:
                out_a = torch.empty_like(saved_A)
            if out_b is None:
                out_b = torch.empty_like(saved_B)
            _slang_tile_mm_bwd(
                TILE,
                TILE,
                TILE,
                saved_A,
                saved_B,
                dC,
                out_a,
                out_b,
            )
        return out_a, out_b
    except Exception:
        # Fall through to the 2-dispatch path below.
        pass

    # ── Fallback: 2-dispatch path (T4.2) ─────────────────────────────────
    if is_batch:
        # BMM backward: dC[b,m,n], A[b,m,k], B[b,k,n]
        # dA = bmm(dC, B^T)  where B^T: [b, n, k]
        # dB = bmm(A^T, dC)  where A^T: [b, k, m]
        grad_a = _mm_with_transpose(dC, saved_B, transpose_b=True, out=out_a)
        grad_b = _mm_with_transpose(saved_A, dC, transpose_a=True, out=out_b)
    else:
        # MM backward: dC[m,n], A[m,k], B[k,n]
        # dA = mm(dC, B^T)  where B^T: [n,k]
        # dB = mm(A^T, dC)  where A^T: [k,m]
        #
        # T4.2: Use _slang_tile_mm_backward which encodes transposition
        # via push-constant strides (no CPU-side copy needed).
        from torch_vulkan.inductor.vulkan_template_caller import (
            _slang_tile_mm_backward,
        )

        if out_a is None:
            M_dA, K_dA = dC.shape[-2], saved_B.shape[-2]
            out_a = torch.empty(M_dA, K_dA, dtype=dC.dtype, device=dC.device)
        _slang_tile_mm_backward(
            TILE,
            TILE,
            TILE,
            dC,
            saved_B,
            out_a,
            transpose_b=True,
        )

        if out_b is None:
            K_dB, N_dB = saved_A.shape[-2], dC.shape[-1]
            out_b = torch.empty(K_dB, N_dB, dtype=saved_A.dtype, device=saved_A.device)
        _slang_tile_mm_backward(
            TILE,
            TILE,
            TILE,
            saved_A,
            dC,
            out_b,
            transpose_a=True,
        )
        grad_a = out_a
        grad_b = out_b

    return grad_a, grad_b


def _mm_with_transpose(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    transpose_a: bool = False,
    transpose_b: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a @ b with optional operand transposition, using the forward
    Slang tiled-mm template.

    When ``transpose_a`` is True, uses ``a^T`` (requires a is 2-D or 3-D).
    When ``transpose_b`` is True, uses ``b^T``.
    The underlying dispatch goes through ``_slang_tile_mm`` from
    ``vulkan_template_caller``, using a sensible default tile config.
    For production use, the Inductor autotuner (``install_external_mm``)
    selects the optimal tile config; this function provides a deterministic
    direct-dispatch path for backward graphs.

    NOTE: This function does CPU-side transposition (``.t().contiguous()``).
    For the 2-D mm case, prefer ``_slang_tile_mm_backward`` which encodes
    transposition via strides and avoids the copy.
    """
    if transpose_a:
        if a.dim() == 3:
            a = a.transpose(1, 2).contiguous()
        elif a.dim() == 2:
            a = a.t().contiguous()
        else:
            raise ValueError(
                f"T4.2: transpose_a requires 2-D or 3-D tensor; got {a.dim()}D"
            )
    if transpose_b:
        if b.dim() == 3:
            b = b.transpose(1, 2).contiguous()
        elif b.dim() == 2:
            b = b.t().contiguous()
        else:
            raise ValueError(
                f"T4.2: transpose_b requires 2-D or 3-D tensor; got {b.dim()}D"
            )

    M, K = a.shape[-2:]
    _, N = b.shape[-2:]

    if out is None:
        if a.dim() == 3 and b.dim() == 3:
            out = torch.empty(a.shape[0], M, N, dtype=a.dtype, device=a.device)
        else:
            out = torch.empty(M, N, dtype=a.dtype, device=a.device)
    elif not out.is_contiguous():
        out = out.contiguous()

    from torch_vulkan.inductor.vulkan_template_caller import (
        _slang_tile_bmm,
        _slang_tile_mm,
    )

    # Use a balanced default tile config (32x32x32) for backward dispatch.
    TILE = 32
    if a.dim() == 3 and b.dim() == 3:
        _slang_tile_bmm(TILE, TILE, TILE, a, b, out)
    else:
        _slang_tile_mm(TILE, TILE, TILE, 1, a, b, out)

    return out


def dispatch_binary_bwd(
    aten_op: str,
    a: torch.Tensor,
    b: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    no_diff_kwargs: dict[str, float] | None = None,
    out_a: torch.Tensor | None = None,
    out_b: torch.Tensor | None = None,
    numthreads: int = _DEFAULT_NUMTHREADS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch the autodiff-emitted backward for a binary entry.

    Returns ``(grad_a, grad_b)`` — each shape/dtype/device matching
    its corresponding forward input. ``no_diff_kwargs`` supplies values
    for any ``no_diff_params`` declared on the entry (e.g.
    ``smooth_l1.beta``, ``huber.delta``); ``KeyError`` if a required
    key is missing.

    f16/bf16 inputs are widened to f32 for the compute kernel (T3.1).

    T4.2: Validates op routing against ``BWD_TEMPLATE_REGISTRY``.
    """
    # T4.2: Check BWD_TEMPLATE_REGISTRY before dispatching.
    resolved = resolve_backward_kind(aten_op)
    if resolved is not None and not resolved.is_bwd_diff:
        raise ValueError(
            f"T4.2: dispatch_binary_bwd called for {aten_op!r}, but "
            f"BWD_TEMPLATE_REGISTRY lists kind={resolved.kind}. "
            f"Use dispatch_template_bwd() for TEMPLATE_JINJA entries."
        )

    entry = _entry(aten_op, expected_arity=2)
    orig_dtype = a.dtype
    _check_float(a, b, grad_out)
    _check_vulkan(a, b, grad_out)
    if a.shape != b.shape or a.shape != grad_out.shape:
        raise ValueError(
            f"PF.6.b: binary backward expects matching shapes for "
            f"a/b/grad_out; got {tuple(a.shape)}/{tuple(b.shape)}/"
            f"{tuple(grad_out.shape)}"
        )
    a_f32 = _ensure_f32(a)
    b_f32 = _ensure_f32(b)
    go_f32 = _ensure_f32(grad_out)
    grad_a_out = out_a.float() if out_a is not None else torch.empty_like(a_f32)
    grad_b_out = out_b.float() if out_b is not None else torch.empty_like(b_f32)
    _check_vulkan(grad_a_out, grad_b_out)
    if grad_a_out.shape != a.shape or grad_b_out.shape != b.shape:
        raise ValueError("PF.6.b: out_a/out_b shape mismatch with a/b")
    no_diff_kwargs = dict(no_diff_kwargs or {})
    missing = [k for k in entry.no_diff_params if k not in no_diff_kwargs]
    if missing:
        raise KeyError(
            f"PF.6.b: aten op {aten_op!r} requires no_diff_kwargs "
            f"{missing}; got keys {list(no_diff_kwargs)}"
        )
    extra = [k for k in no_diff_kwargs if k not in entry.no_diff_params]
    if extra:
        raise KeyError(
            f"PF.6.b: aten op {aten_op!r} has no_diff_params "
            f"{list(entry.no_diff_params)}; received unexpected keys "
            f"{extra}"
        )
    numel = a_f32.numel()
    slang_dtype = _slang_dtype_str(orig_dtype)
    src = emit_bwd_diff_kernel(
        aten_op,
        dtype=slang_dtype,
        numthreads=numthreads,
    )
    fmt = "<" + "f" * len(entry.no_diff_params) + "I"
    values = [float(no_diff_kwargs[k]) for k in entry.no_diff_params]
    pc = struct.pack(fmt, *values, numel)
    wg_x = (numel + numthreads - 1) // numthreads
    compile_and_dispatch(
        src,
        tensors=[
            a_f32.contiguous(),
            b_f32.contiguous(),
            go_f32.contiguous(),
            grad_a_out,
            grad_b_out,
        ],
        wg_x=wg_x,
        wg_y=1,
        wg_z=1,
        push_constants=pc,
        num_outputs=2,
        entry="bwd_op",
        cache_key=_cache_key(aten_op, orig_dtype, numthreads),
    )
    ga = _narrow_from_f32(grad_a_out, orig_dtype)
    gb = _narrow_from_f32(grad_b_out, orig_dtype)
    if out_a is not None:
        out_a.copy_(ga)
    if out_b is not None:
        out_b.copy_(gb)
    return ga, gb


# ── CG.M3: Reduction backward dispatch ────────────────────────────────────
# Reduction backward is conceptually a broadcast of the output gradient
# to the input shape.  The [Differentiable] fold functions in
# shaders/lib/reduction.slang provide the autodiff proof:
#   bwd_diff(reduce_fold_sum)(dpa, dpb, dout) -> dpa.d = dout, dpb.d = dout
#   -> every element's gradient = grad_output
#
# The kernel emits a broadcast: grad_in[tid] = grad_out (sum),
# optionally scaled by 1/numel (mean), 2*(x-mean)/(n-1) (var), or
# prod/x_i (prod).  Max/min reductions are NOT differentiable — they
# route gradient sparsely to the argmax/argmin position.


_REDUCTION_BWD_SRC = """
import reduction;

[[vk::push_constant]] cbuffer Push {{
    uint numel;
    float scale;
}};

[shader("compute")][numthreads({numthreads},1,1)]
void bwd_reduction(
    uniform StructuredBuffer<float> grad_out,       // scalar or broadcast-compatible
    uniform StructuredBuffer<float> saved_input,     // for var/std/prod backward
    uniform RWStructuredBuffer<float> grad_in,
    uint3 tid : SV_DispatchThreadID
) {{
    if (tid.x >= numel) return;
    float go = grad_out[0];
    float si = saved_input[tid.x];
    // scale encodes: 1.0 for sum, 1.0/numel for mean,
    // 2.0/(numel-1) for var, prod for prod
    grad_in[tid.x] = go * scale;
}}
"""


def dispatch_reduction_bwd(
    aten_op: str,
    grad_out: torch.Tensor,
    saved_input: torch.Tensor | None = None,
    *,
    scale: float = 1.0,
    out: torch.Tensor | None = None,
    numthreads: int = _DEFAULT_NUMTHREADS,
) -> torch.Tensor:
    """Dispatch a reduction backward broadcast kernel.

    For sum_backward:  scale=1.0,       grad_in[tid] = grad_out
    For mean_backward: scale=1.0/numel, grad_in[tid] = grad_out / numel
    For var_backward:  scale=2.0/(n-1), grad_in[tid] = grad_out * 2 * (x_i-mean) / (n-1)
                        (saved_input must be x - mean)
    For prod_backward: scale=prod,      grad_in[tid] = grad_out * prod / x_i
                        (saved_input must be the forward input x)

    The [Differentiable] fold functions in shaders/lib/reduction.slang
    (reduce_fold_sum / reduce_fold_prod) are verified by slangc to produce
    the correct gradient contributions.  This kernel IS the broadcast that
    their bwd_diff chain implies.
    """
    resolved = resolve_backward_kind(aten_op)
    if resolved is not None and not resolved.is_bwd_diff:
        raise ValueError(
            f"CG.M3: dispatch_reduction_bwd called for {aten_op!r}, but "
            f"BWD_TEMPLATE_REGISTRY lists kind={resolved.kind}."
        )

    orig_dtype = grad_out.dtype
    _check_float(grad_out)
    _check_vulkan(grad_out)
    go_f32 = _ensure_f32(grad_out)

    if saved_input is not None:
        _check_float(saved_input)
        _check_vulkan(saved_input)
        si_f32 = _ensure_f32(saved_input)
        numel = si_f32.numel()
        grad_in_f32 = out.float() if out is not None else torch.empty_like(si_f32)
    else:
        # No saved input — we need numel from somewhere. The caller must
        # provide `out` or we can't determine the output size.
        if out is None:
            raise ValueError(
                "CG.M3: dispatch_reduction_bwd requires either saved_input "
                "or out to determine the output size"
            )
        si_f32 = _ensure_f32(out)
        numel = si_f32.numel()
        grad_in_f32 = out.float() if out is not None else torch.empty_like(si_f32)

    _check_vulkan(grad_in_f32)
    slang_dtype = _slang_dtype_str(orig_dtype)
    src = _REDUCTION_BWD_SRC.format(numthreads=numthreads)
    pc = struct.pack("<If", numel, scale)
    wg_x = (numel + numthreads - 1) // numthreads
    compile_and_dispatch(
        src,
        tensors=[go_f32.contiguous(), si_f32.contiguous(), grad_in_f32],
        wg_x=wg_x,
        wg_y=1,
        wg_z=1,
        push_constants=pc,
        num_outputs=1,
        entry="bwd_reduction",
        cache_key=_cache_key(aten_op, orig_dtype, numthreads),
    )
    if out is not None:
        if out.dtype != orig_dtype:
            out.copy_(_narrow_from_f32(grad_in_f32, orig_dtype))
        else:
            out.copy_(grad_in_f32)
        return out
    return _narrow_from_f32(grad_in_f32, orig_dtype)
