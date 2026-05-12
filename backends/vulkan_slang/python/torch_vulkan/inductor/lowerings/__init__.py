"""Vulkan-specific Inductor op lowerings.

Registers lowerings for ops that Inductor would otherwise dispatch as
ExternKernels, preventing fusion with adjacent pointwise/reduction ops.

Each lowering checks if the input tensor is on a Vulkan device; if not, it
falls through to the original Inductor lowering (or the ExternKernel path).

Current lowerings:
  - aten.native_layer_norm: decomposed into mean/var/rsqrt/mul/add so
    Inductor's scheduler can fuse the normalization with subsequent ops
    (e.g. layer_norm + residual add -> single VulkanKernel dispatch).
  - aten.native_group_norm: same decomposition approach.
  - aten._softmax / aten._log_softmax: decomposed into max/sub/exp/sum/div
    so Inductor fuses these with the preceding linear (logits -> softmax -> CE).
  - Activation backward ops (sigmoid_backward, tanh_backward, silu_backward,
    gelu_backward): decomposed into pointwise primitives so they fuse with
    the surrounding chain. Without this, they fall through to extern aten.*
    dispatches and break fusion of the backward pass.
"""

from __future__ import annotations

_registered = False


def _is_vulkan(x) -> bool:
    """Return True if x is a Vulkan IR node."""
    try:
        return x.get_device().type == "vulkan"
    except Exception:
        return False


def _suppress_upstream_decomps() -> None:
    """Drop decompositions the upstream Inductor decomp table installs for ops
    where we want our own Vulkan lowering to fire instead.

    Inductor's `decompositions` dict is consulted by AOT autograd before the
    graph reaches Inductor's lowering layer. If a decomp for op X is present,
    Inductor never sees `aten.X.default` — it sees the post-decomp primitives
    — so our `register_lowering(aten.X)` is dead code. Removing the entry
    re-enables the lowering path.
    """
    import torch
    from torch._decomp import decomposition_table as _aot_decomps
    from torch._inductor.decomposition import decompositions, fast_random_decomps

    aten = torch.ops.aten
    ops_to_suppress = [
        aten.native_layer_norm_backward.default,
        aten.native_group_norm_backward.default,
        aten.native_batch_norm_backward.default,
        aten.native_batch_norm.default,
        # DR.1+: Suppress native_group_norm and native_layer_norm decompositions
        # so our dedicated Vulkan lowerings (norm.py) fire instead of the
        # upstream AOT autograd decomposition into view→native_batch_norm→view→
        # addcmul.  The upstream decomposition creates 5-8 dispatches with no
        # fusion-group annotations; our lowering produces IR nodes the scheduler
        # can fuse into 3-4 dispatches, and the op_class_conv_norm_activation FX
        # pass (priority 8) can annotate the full conv→norm→activation chain.
        aten.native_group_norm.default,
        aten.native_layer_norm.default,
        aten.embedding_dense_backward.default,
        aten._log_softmax_backward_data.default,
        aten._softmax_backward_data.default,
        aten.gelu_backward.default,
        aten.silu_backward.default,
        aten.sigmoid_backward.default,
        aten.tanh_backward.default,
        aten.elu_backward.default,
        aten.hardswish_backward.default,
        aten.hardsigmoid_backward.default,
        aten.softplus_backward.default,
        aten.mish_backward.default,
        # PF.26: Upstream decomp rewrites this as ``mask.type_as(grad) *
        # scale * grad`` which fuses into a single pointwise kernel whose
        # bool LOAD path assumes 1-uint-per-element storage.  External
        # eager-produced bool masks are byte-packed (4 bools / uint32), so
        # the fused load returns garbage (e.g. 65537 for ``[T,F,T,F]``).
        # Suppressing the decomp lets our lowering route the bool→float
        # cast through the C++ ``vulkan_to_copy`` kernel (which knows the
        # byte-packed layout) before fusing the two muls.
        aten.native_dropout_backward.default,
        # P12: suppress aten.clamp decomposition so our Vulkan lowering
        # fires instead of the upstream clamp → clamp_min → clamp_max
        # decomposition.
        aten.clamp.default,
        # OP.3: view-style copy ops whose stock decomp routes through
        # ``aten.narrow.default`` at the wrapper level on Vulkan, where the
        # storage_offset isn't propagated into the SSBO descriptor and the
        # output is silently zero-padded. Suppress so our Pointwise-based
        # lowerings in ``view.py`` fire instead.
        aten.narrow_copy.default,
        aten.repeat_interleave.self_int,
    ]
    if hasattr(aten, "relu_backward"):
        ops_to_suppress.append(aten.relu_backward.default)
    ops_to_suppress.append(aten.matmul.default)
    for op in ops_to_suppress:
        decompositions.pop(op, None)
    # ``aten.native_dropout_backward`` is also installed in the AOT decomp
    # table (``torch._decomp.decomposition_table``) which is consulted
    # before AOT autograd produces the FX graph that Inductor lowers.
    # Remove it there too so the op survives to our register_lowering.
    _aot_decomps.pop(aten.native_dropout_backward.default, None)
    # OP.3: same pop in the AOT decomp table for the view-style ops, plus
    # the fast_random_decomps cache that snapshots ``decompositions`` early.
    _aot_decomps.pop(aten.narrow_copy.default, None)
    _aot_decomps.pop(aten.repeat_interleave.self_int, None)
    _aot_decomps.pop(aten.repeat_interleave.self_Tensor, None)
    fast_random_decomps.cache_clear()


def _register_argsort_lowering() -> None:
    """Register a lowering for ``aten.argsort`` that decomposes it to
    ``aten.sort`` + index extraction, routed through the existing
    ``kernel/reduction.py:sort()`` → ``wg_bitonic_sort_float2`` codegen.

    Inductor's stock decomposition for ``aten.argsort`` decomposes it to
    ``aten.sort``, which our backend already handles via BackendFeature.SORT.
    However, that decomposition runs at the AOTAutograd level and can be
    suppressed when we suppress other decomps.  This lowering is a safety net:
    if argsort survives to the Inductor lowering phase, we explicitly pair
    each element with its index, sort the (key, idx) float2 pairs, and
    extract the sorted indices.
    """
    import torch
    from torch._inductor.lowering import register_lowering

    aten = torch.ops.aten

    # Decompose argsort → sort + index extraction.
    # argsort(self, dim=-1, descending=False) -> sort(self, dim, descending).indices
    @register_lowering(aten.argsort.default, type_promotion_kind=None)
    def _vulkan_argsort(x, dim=-1, descending=False, stable=False):
        return torch.sort(x, dim=dim, descending=descending).indices

    @register_lowering(aten.argsort.stable, type_promotion_kind=None)
    def _vulkan_argsort_stable(x, *, stable=True, dim=-1, descending=False):
        return torch.sort(x, dim=dim, descending=descending, stable=stable).indices


def register() -> None:
    global _registered
    if _registered:
        return
    _registered = True
    # GAP 4.1 / PF.6.b.iii: register all bwd_diff custom ops eagerly so they
    # are always available when a cached compiled backward graph calls them at
    # runtime (when the graph is loaded from cache, the lowering phase is
    # skipped, so lazy registration inside the lowerings would never fire).
    from .bwd_diff import (
        _BINARY_LOSS_BWD_DIFF_OPS,
        _UNARY_BWD_DIFF_OPS,
        _ensure_binary_loss_bwd_diff_op,
        _ensure_unary_bwd_diff_op,
    )

    for aten_op in _UNARY_BWD_DIFF_OPS:
        _ensure_unary_bwd_diff_op(aten_op)
    for aten_op in _BINARY_LOSS_BWD_DIFF_OPS:
        _ensure_binary_loss_bwd_diff_op(aten_op)
    _suppress_upstream_decomps()
    # T4.6: Register Philox RNG template lowerings for aten.rand/randn/uniform/dropout.
    from ..philox_dispatch import install as _install_philox_dispatch
    from ..vulkan_template_caller import (
        install_external_rng,
        install_external_scatter,
    )
    from ..vulkan_template_caller import (
        install_external_rnn as _install_external_rnn,
    )

    install_external_rng()
    install_external_scatter()
    _install_external_rnn()
    _install_philox_dispatch()
    # T4.8: Register foreach optimizer custom ops as fallback kernels
    # so Inductor can lower torch_vulkan::foreach_{sgd,sgd_momentum,adamw,lion}_step
    # nodes inserted by the _route_foreach_add_to_template FX pass.
    import torch
    from torch._inductor.lowering import make_fallback

    from .activation import (
        _register_clamp_lowerings,
        _register_pointwise_math_lowerings,
        _register_pow_scalar_lowering,
    )
    from .bool_mask import _register_bool_mask_read_lowering
    from .conv import _register_conv_and_pool_lowerings
    from .embedding import (
        _register_embedding_bag_forward,
        _register_embedding_dense_backward,
    )
    from .fft import _register_fft_lowerings
    from .loss import _register_loss_lowerings
    from .matmul import (
        _register_bmm_lowering,
        _register_matmul_backward,
        _register_matmul_lowering,
        _register_mm_lowering,
    )
    from .rng import _register_multinomial_lowering
    from .rnn import _register_rnn_fallbacks
    from .scatter import _register_scatter_family_lowerings
    from .searchsorted import _register_searchsorted_and_repeat_interleave_tensor

    make_fallback(torch.ops.torch_vulkan.foreach_sgd_step)
    make_fallback(torch.ops.torch_vulkan.foreach_sgd_momentum_step)
    make_fallback(torch.ops.torch_vulkan.foreach_adamw_step)
    make_fallback(torch.ops.torch_vulkan.foreach_lion_step)

    # TRAIN.7: autocast boundary ops — identity dtype casts that must not
    # break fusion.  aten._autocast_to_reduced_precision (fp32→fp16/bf16)
    # and aten._autocast_to_full_precision (fp16/bf16→fp32) are generated
    # by Dynamo when `torch.autocast("vulkan", dtype=torch.float16)` is
    # active.  Lower them as simple `to(dtype)` casts so Inductor's
    # pointwise scheduler can fuse through them.
    aten = torch.ops.aten
    from torch._inductor.lowering import register_lowering as _reg_low

    # `aten.to.dtype` — upstream Inductor exposes the `to_dtype` helper
    # function but does NOT register it as a lowering target.  Several
    # of our backward decomp / autocast paths call
    # `L.lowerings[aten.to.dtype](x, dtype)` directly (e.g.
    # `lowerings/activation.py:_native_dropout_backward` for bool→float
    # promotion, and the autocast lowerings below).  Register a thin
    # redirect so those callsites succeed without resorting to a
    # custom-op shim.  Routes to the upstream `to_dtype` helper which
    # uses `make_pointwise(_to_dtype, override_return_dtype=dtype)` —
    # exactly what `aten._to_copy` / `prims.convert_element_type` do.
    @_reg_low(aten.to.dtype, type_promotion_kind=None)
    def _vulkan_aten_to_dtype(
        x, dtype, *, non_blocking=False, copy=False, memory_format=None
    ):
        from torch._inductor.lowering import to_dtype as _to_dtype_helper

        return _to_dtype_helper(x, dtype, copy=copy)

    @_reg_low(aten._autocast_to_reduced_precision.default, type_promotion_kind=None)
    def _vulkan_autocast_to_reduced_precision(
        x, cuda_enabled, cpu_enabled, cuda_dtype, cpu_dtype
    ):
        # Ignore the cuda/cpu flags — we know we're on Vulkan, just
        # cast to the fast dtype (fp16 or bf16).  cpu_dtype holds the
        # autocast dtype when cuda_dtype is not the active one.
        from torch._inductor import lowering as L

        target_dtype = cuda_dtype if cuda_enabled else cpu_dtype
        return L.lowerings[aten.to.dtype](x, target_dtype)

    @_reg_low(aten._autocast_to_full_precision.default, type_promotion_kind=None)
    def _vulkan_autocast_to_full_precision(x, cuda_enabled, cpu_enabled):
        # Cast back to float32 — the standard "full precision" dtype.
        from torch._inductor import lowering as L

        return L.lowerings[aten.to.dtype](x, torch.float32)

    from .norm import (
        _register_batch_norm_backward,
        _register_batch_norm_forward,
        _register_group_norm,
        _register_group_norm_backward,
        _register_layer_norm,
        _register_layer_norm_backward,
    )
    from .softmax import _register_softmax, _register_softmax_backward
    from .view import _register_view_lowerings

    _register_view_lowerings()
    _register_conv_and_pool_lowerings()
    _register_bmm_lowering()
    _register_mm_lowering()
    _register_matmul_lowering()
    _register_matmul_backward()
    _register_layer_norm()
    _register_group_norm()
    _register_softmax()
    _register_batch_norm_forward()
    _register_clamp_lowerings()
    _register_pow_scalar_lowering()
    _register_pointwise_math_lowerings()
    # CG.M8: Register inline bwd_diff lowerings FIRST so they take priority
    # over the Python custom-op shim lowerings in bwd_lowerings.py.
    # Inductor's register_lowering is first-come-first-served.
    import torch
    from torch._inductor.lowering import register_lowering as _reg_low2

    from .bwd_diff_inline_lowering import (
        _register_inline_bwd_diff_lowerings,
    )

    aten2 = torch.ops.aten
    from torch._inductor import lowering as L2

    _register_inline_bwd_diff_lowerings(_reg_low2, L2, aten2)

    # TR.19: All backward lowerings are now consolidated in bwd_lowerings.py.
    # The _register_*_backward stubs below are no-ops retained for compat.
    from ..bwd_lowerings import register as _register_bwd_lowerings

    _register_bwd_lowerings()
    _register_softmax_backward()
    _register_layer_norm_backward()
    _register_group_norm_backward()
    _register_batch_norm_backward()
    _register_loss_lowerings()
    _register_embedding_dense_backward()
    _register_embedding_bag_forward()
    _register_fft_lowerings()
    _register_scatter_family_lowerings()
    # OP.1.d: bool-mask read (`x[mask]`) — CPU-roundtrip lowering for
    # the data-dependent-shape path that Inductor's stock lowering
    # mis-handles even after `aten::nonzero` (OP.1.a) is wired.
    _register_bool_mask_read_lowering()
    # N.1.b: ``aten.searchsorted`` (no Vulkan kernel) and
    # ``aten.repeat_interleave.Tensor`` (which needs cumsum +
    # searchsorted) routed through CPU — fastest unblock per the
    # N.1.b directive; N.1.b-fast tracks the GPU binary-search shader.
    _register_searchsorted_and_repeat_interleave_tensor()
    # OP.11: ``aten.multinomial`` → cumsum + Philox rand + searchsorted
    # inverse-CDF sampling.  replacement=True only for now.
    _register_multinomial_lowering()
    # OP.8: ``aten.argsort`` — decompose to ``aten.sort`` + index extraction.
    # BackendFeature.SORT is advertised; ``aten.sort`` routes through
    # ``kernel/reduction.py:sort()`` → ``wg_bitonic_sort_float2``.
    # This lowering pairs each element with its index, sorts the (key, idx)
    # pairs, and extracts the sorted indices.
    _register_argsort_lowering()
    # T.10: nn.LSTM / nn.GRU / nn.RNN compile coverage via fallback to
    # eager. Inductor's stock decomp lowers these to _thnn_fused_lstm_cell
    # (etc.), which has no Vulkan eager kernel; the fallback keeps the
    # high-level op intact and routes it through the regular eager path.
    _register_rnn_fallbacks()
    # T.10-bwd: RNN cell backward decompositions for LSTM/GRU into
    # Vulkan-supported primitives (sigmoid_backward, tanh_backward, etc.).
    from .rnn_bwd import register as _register_rnn_bwd

    _register_rnn_bwd()
