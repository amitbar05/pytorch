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
    from torch._inductor.decomposition import (
        decompositions,
        fast_random_decomps,
    )

    aten = torch.ops.aten
    ops_to_suppress = [
        aten.native_layer_norm_backward.default,
        # M-NEW.14 (2026-05-22): native_group_norm_backward MUST be suppressed.
        # The previous "PT.21 keep decomposed" rationale (M17.6 closeout) is
        # WRONG — the upstream decomp introduces a buffer-pool aliasing hazard
        # in two-stacked-block SmallCNN.  The compiled wrapper recycles
        # block-2 GN scratch buffers (buf5/buf6) into block-1 GN gradient
        # outputs (buf31/buf32) via `reinterpret_tensor`, producing the
        # exact-zero-grad fingerprint on `gn2.weight.grad` / `gn2.bias.grad`
        # and propagating NaN into `conv1.weight.grad`.  See
        # `agent_space/m_new_14_compiled_wrapper.py` lines 930-931 for the
        # smoking gun.  Suppressing the decomp lets the dedicated Vulkan
        # lowering at `bwd_lowerings.py::_register_group_norm_backward` fire
        # instead — it builds the same primitive chain but as a single IR
        # tree the scheduler can buffer-plan correctly.
        aten.native_group_norm_backward.default,
        aten.native_batch_norm_backward.default,
        aten.native_batch_norm.default,
        # MODEL.2 (2026-05-29): Suppress _native_batch_norm_legit decomps
        # so running_mean/running_var mutations go through the existing
        # native_batch_norm Vulkan lowering (norm.py:226-260) instead of
        # the upstream decomp that produces copy_ nodes which may not
        # propagate correctly across compiled graph invocations.
        aten._native_batch_norm_legit.default,
        aten._native_batch_norm_legit_functional.default,
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
        # TRAIN.1 (2026-05-27): Suppress upstream decompositions for loss
        # backward ops. Without these, AOT Autograd decomposes mse_loss_backward,
        # binary_cross_entropy_backward, smooth_l1_loss_backward, huber_loss_backward
        # into primitive ops (mul, sub, where, abs) before Inductor sees them.
        # Our bwd_diff Slang kernels in BWD_DIFF_TABLE + _BINARY_BWD_DIFF_LOWERING_OPS
        # never fire — dead code. Suppressing lets the fused single-dispatch
        # Slang backward execute instead of 4-6 dispatches of primitives.
        aten.mse_loss_backward.default,
        aten.binary_cross_entropy_backward.default,
        aten.smooth_l1_loss_backward.default,
        aten.huber_loss_backward.default,
        # PF.26: Upstream decomp rewrites this as ``mask.type_as(grad) *
        # scale * grad`` which fuses into a single pointwise kernel whose
        # bool LOAD path assumes 1-uint-per-element storage.  External
        # eager-produced bool masks are byte-packed (4 bools / uint32), so
        # the fused load returns garbage (e.g. 65537 for ``[T,F,T,F]``).
        # Suppressing the decomp lets our lowering route the bool→float
        # cast through the C++ ``vulkan_to_copy`` kernel (which knows the
        # byte-packed layout) before fusing the two muls.
        aten.native_dropout_backward.default,
        # P12 / M18.TB.1: suppress aten.clamp, clamp_min, clamp_max
        # decompositions.  The upstream torch._refs decomps form a cycle:
        # clamp → clamp_min + clamp_max → clamp.  Without suppressing all
        # three, any decomp that calls torch.clamp_min / torch.clamp_max
        # (e.g. our _hardtanh_via_clamp_min_max) triggers infinite recursion
        # under AOTAutograd tracing.  clamp / clamp_min / clamp_max all have
        # proper Meta kernels and upstream Inductor lowerings
        # (via register_pointwise / maximum / minimum), so suppressing the
        # decomps does not affect correctness.
        aten.clamp.default,
        aten.clamp_min.default,
        aten.clamp_max.default,
        # OP.3: view-style copy ops whose stock decomp routes through
        # ``aten.narrow.default`` at the wrapper level on Vulkan, where the
        # storage_offset isn't propagated into the SSBO descriptor and the
        # output is silently zero-padded. Suppress so our Pointwise-based
        # lowerings in ``view.py`` fire instead.
        aten.narrow_copy.default,
        aten.repeat_interleave.self_int,
        # OP.26: suppress aten.scaled_dot_product_attention decomposition
        # so our native lowering (attention.py) fires instead of AOT
        # decomposing it to _scaled_dot_product_attention_math.
        aten.scaled_dot_product_attention.default,
        # OP.23: suppress aten.lerp.Tensor decomposition so our lowering
        # (activation.py) fires instead of the upstream _lerp_tensor→addcmul
        # path which has scalar*TensorBox typing issues.
        aten.lerp.Tensor,
        # M17.3: suppress aten._adaptive_avg_pool2d decomposition so the
        # upstream Inductor lowering (which decomposes into avg_pool2d or
        # Pointwise-based adaptive pooling) fires instead of the AOTAutograd
        # decomp. The upstream lowering produces IR that our reduction
        # template can fuse with adjacent pointwise ops.
        aten._adaptive_avg_pool2d.default,
        # M22.15 (2026-05-24): suppress avg_pool2d_backward (upstream uses
        # ops.indirect_indexing → wrong SPIR-V) and both max_pool2d_with_indices
        # ops (forward uses indirect_indexing for wrong index output; backward
        # uses indirect_indexing + int64 loads which don't work on RDNA1 Vulkan;
        # FallbackKernel overrides for both in bwd_lowerings.py).
        aten.avg_pool2d_backward.default,
        aten.max_pool2d_with_indices.default,
        aten.max_pool2d_with_indices_backward.default,
        # A5 (2026-06-16): suppress aten.max_pool2d and aten.avg_pool2d forward
        # decomp so our FallbackKernel lowerings in bwd_lowerings.py fire instead
        # of the upstream Inductor lowerings which use ops.indirect_indexing
        # → wrong SPIR-V on Vulkan.
        aten.max_pool2d.default,
        aten.avg_pool2d.default,
        # TRAIN.4 (2026-05-27): suppress upstream nll_loss_backward decomposition
        # so our lowering in lowerings/loss.py can intercept it.
        aten.nll_loss_backward.default,
        # DECOMP.2 (2026-05-29): suppress aten.convolution_backward decomposition
        # so the registered custom lowering _vulkan_convolution_backward
        # (bwd_lowerings.py::_register_consolidated_backward_impls) fires directly
        # instead of being decomposed by AOT Autograd into mm+sum+fold primitives
        # (8-9 dispatches). The lowering produces 1 dispatch for a single conv bwd.
        # Gated by the lowering itself: groups=1, fp32, not transposed, 4D tensors.
        aten.convolution_backward.default,
    ]
    if hasattr(aten, "relu_backward"):
        ops_to_suppress.append(aten.relu_backward.default)
    ops_to_suppress.append(aten.matmul.default)
    # Guard ops that may not exist in all PyTorch versions
    for _attr in ("binary_cross_entropy_with_logits_backward", "l1_loss_backward"):
        if hasattr(aten, _attr):
            ops_to_suppress.append(getattr(aten, _attr).default)
    # OP.26: also pop from the AOT decomp table so AOT autograd does not
    # decompose aten.scaled_dot_product_attention before Inductor sees it.
    _aot_decomps.pop(aten.scaled_dot_product_attention.default, None)
    # DECOMP.2: also pop convolution_backward from AOT decomp.
    # The upstream decomp guard checks GPU_TYPES (excludes "vulkan"), but
    # the guard may not catch all paths; a defensive pop ensures our
    # registered _vulkan_convolution_backward lowering fires directly.
    _aot_decomps.pop(aten.convolution_backward.default, None)
    # V9.SUPPRESS.1: _softmax and _log_softmax are in the upstream
    # inductor_decompositions table (decomposition.py:79/94). The Vulkan
    # backend has native lowerings (lowerings/softmax.py) that decompose
    # into the same primitives (sub/exp/sum/div), but suppressing the
    # upstream decomp ensures our lowerings fire — giving the scheduler
    # a single fused softmax/log_softmax IR node for better fusion with
    # adjacent ops (e.g. log_softmax + gather for manual cross-entropy).
    ops_to_suppress.extend([
        aten._softmax.default,
        aten._log_softmax.default,
    ])
    for op in ops_to_suppress:
        decompositions.pop(op, None)
    # ``aten.native_dropout_backward`` is also installed in the AOT decomp
    # table (``torch._decomp.decomposition_table``) which is consulted
    # before AOT autograd produces the FX graph that Inductor lowers.
    # Remove it there too so the op survives to our register_lowering.
    _aot_decomps.pop(aten.native_dropout_backward.default, None)
    # TRAIN.1: also pop loss backward ops from the AOT decomp table.
    # Without this, AOTAutograd decomposes them before Inductor sees them,
    # even though we suppressed in the Inductor decompositions dict.
    for _op in [
        aten.mse_loss_backward.default,
        aten.binary_cross_entropy_backward.default,
        aten.smooth_l1_loss_backward.default,
        aten.huber_loss_backward.default,
        # TRAIN.4 (2026-05-27): nll_loss_backward — our lowering in
        # lowerings/loss.py decomposes into scatter + pointwise directly
        # at the Inductor IR level, avoiding the upstream decomposition
        # which may produce problematic IR nodes.
        aten.nll_loss_backward.default,
    ]:
        _aot_decomps.pop(_op, None)
    # Guard ops that may not exist in all PyTorch versions
    for _attr in ("binary_cross_entropy_with_logits_backward", "l1_loss_backward"):
        if hasattr(aten, _attr):
            _aot_decomps.pop(getattr(aten, _attr).default, None)
    # M18.TB.1 / P12: also pop clamp, clamp_min, clamp_max from the global
    # AOT decomp table. Without this, the upstream torch._refs decomps cause
    # infinite recursion under AOTAutograd tracing:
    #   clamp_min → torch.clamp(min=...) → clamp → clamp_min → ...
    # Both ``decompositions`` (already suppressed above for clamp) and
    # ``_aot_decomps`` must be clean. clamp_min / clamp_max have proper Meta
    # kernels, so FakeTensorProp can run them without storage access.
    _aot_decomps.pop(aten.clamp.default, None)
    _aot_decomps.pop(aten.clamp.Tensor, None)
    _aot_decomps.pop(aten.clamp_min.default, None)
    _aot_decomps.pop(aten.clamp_min.Tensor, None)
    _aot_decomps.pop(aten.clamp_max.default, None)
    _aot_decomps.pop(aten.clamp_max.Tensor, None)
    # OP.3: same pop in the AOT decomp table for the view-style ops, plus
    # the fast_random_decomps cache that snapshots ``decompositions`` early.
    _aot_decomps.pop(aten.narrow_copy.default, None)
    _aot_decomps.pop(aten.repeat_interleave.self_int, None)
    _aot_decomps.pop(aten.repeat_interleave.self_Tensor, None)
    # OP.23: also pop from AOT decomp table.
    _aot_decomps.pop(aten.lerp.Tensor, None)
    # M17.3: also pop adaptive_avg_pool2d from AOT decomp table.
    _aot_decomps.pop(aten._adaptive_avg_pool2d.default, None)
    # A5 (2026-06-16): also pop max_pool2d and avg_pool2d from AOT decomp table
    # so AOTAutograd does not decompose them before our FallbackKernel lowerings
    # (in bwd_lowerings.py) fire.
    _aot_decomps.pop(aten.max_pool2d.default, None)
    _aot_decomps.pop(aten.avg_pool2d.default, None)
    # M-NEW.14: also pop native_group_norm_backward from the AOT decomp
    # table so AOTAutograd / partitioner do not decompose it before
    # Inductor's lowering layer sees the op (see `ops_to_suppress`
    # comment above for the buffer-aliasing rationale).
    _aot_decomps.pop(aten.native_group_norm_backward.default, None)
    # M-NEW.15: same fix for native_batch_norm_backward — _patch_decompositions
    # was adding a shape-only proxy (_batch_norm_bwd) to the AOT decomp table,
    # causing AOTAutograd to decompose the op away before our Vulkan lowering
    # in bwd_lowerings.py fires. Pop it here so Inductor sees the raw op.
    _aot_decomps.pop(aten.native_batch_norm_backward.default, None)
    # MODEL.2 (2026-05-29): also pop _native_batch_norm_legit variants
    # from the AOT decomp table so AOTAutograd does not decompose them
    # into functional form with separate running stat updates.
    _aot_decomps.pop(aten._native_batch_norm_legit.default, None)
    _aot_decomps.pop(aten._native_batch_norm_legit_functional.default, None)
    # DECOMP.2 (2026-05-29): also pop convolution_backward from the AOT decomp
    # table so AOTAutograd does not decompose it into mm+sum+fold primitives
    # before Inductor's lowering layer sees the op. With this pop, the
    # registered _vulkan_convolution_backward lowering fires directly.
    _aot_decomps.pop(aten.convolution_backward.default, None)
    # M18.TB.1: Replace the upstream hardtanh decomp (hardtanh → clamp) with a
    # clamp_min/clamp_max based decomp that avoids aten.clamp entirely.  The
    # stock torch._refs decomp rewrites hardtanh as clamp(x, min, max), which
    # then appears in the backward graph as a recomputation node.  Inductor's
    # FakeTensorProp tries to run that clamp on Vulkan FakeTensors; PrivateUse1
    # has higher dispatch priority than Meta, so it hits the real Vulkan C++
    # clamp kernel which calls data_ptr() on a FakeTensor and crashes with
    # "Cannot access data pointer of Tensor".  clamp_min/clamp_max each have
    # proper Meta kernels and our Vulkan lowerings in activation.py, so
    # FakeTensorProp can execute them without touching real storage.
    def _hardtanh_via_clamp_min_max(self, min_val=-1.0, max_val=1.0):
        import torch as _torch
        return _torch.clamp_max(_torch.clamp_min(self, min_val), max_val)

    # Replace in BOTH tables: Inductor's `decompositions` (used by
    # AOTAutograd joint trace) and the global `_aot_decomps` (consulted by
    # non-Inductor AOT paths).
    decompositions[aten.hardtanh.default] = _hardtanh_via_clamp_min_max
    _aot_decomps[aten.hardtanh.default] = _hardtanh_via_clamp_min_max
    _aot_decomps[aten.hardtanh_.default] = (
        lambda self, min_val=-1.0, max_val=1.0:
        self.copy_(_hardtanh_via_clamp_min_max(self, min_val, max_val))
    )

    # M18.TB.1: inject hardtanh_backward into Inductor's decomp tables so
    # ``make_fx`` decomposes it to mask*grad during joint-graph tracing instead
    # of dispatching to the Vulkan PrivateUse1 ``hardtanh_backward`` C++ kernel
    # on FakeTensors.  There is no upstream Meta registration for
    # ``hardtanh_backward``, so without this the C++ eager kernel fires, returns
    # shape () instead of the input shape, and
    # ``HardtanhBackward0 returned invalid gradient [] vs [32]`` is raised.
    def _hardtanh_bwd_for_aot(grad_output, self, min_val, max_val):
        mask = (self > min_val) & (self < max_val)
        return grad_output * mask

    decompositions[aten.hardtanh_backward.default] = _hardtanh_bwd_for_aot
    _aot_decomps[aten.hardtanh_backward.default] = _hardtanh_bwd_for_aot

    # M18.HS.1: hardsigmoid stock AOT decomp goes to aten.clamp which triggers
    # FakeTensorProp crash (AutogradPrivateUse1 kernel for clamp has higher
    # dispatch priority than Python/__torch_dispatch__, so the real C++ Vulkan
    # clamp kernel fires on a FakeTensor and crashes with "Cannot access data
    # pointer").
    #
    # Fix (same pattern as M18.TB.1 hardtanh fix):
    #   1. Remove hardsigmoid from Inductor's decompositions dict so the stock
    #      clamp decomp never runs.  hardsigmoid has NO AutogradPrivateUse1
    #      kernel (only PrivateUse1), so FakeTensorMode._dispatch_impl reaches
    #      the fake_impl check and finds _unary_fake — no crash.
    #   2. Also pop from _aot_decomps to prevent the non-Inductor AOT path from
    #      inserting a clamp node into the graph.
    #   3. Inject hardsigmoid_backward as a primitive Python decomp (gt+lt+mask
    #      pattern, same as _hardtanh_bwd_for_aot).  Without this, the backward
    #      uses aten.hardsigmoid_backward with a frozen Vulkan constant as
    #      grad_output (null storage → zero), producing all-zero gradients.
    decompositions.pop(aten.hardsigmoid.default, None)
    decompositions.pop(aten.hardsigmoid_.default, None)
    _aot_decomps.pop(aten.hardsigmoid.default, None)
    _aot_decomps.pop(aten.hardsigmoid_.default, None)

    def _hardsigmoid_bwd_for_aot(grad_output, self):
        # hardsigmoid(x) = clamp(x/6+0.5, 0, 1)
        # d/dx = 1/6  if -3 < x < 3,  else 0
        mask = (self > -3.0) & (self < 3.0)
        return grad_output * mask.to(grad_output.dtype) * (1.0 / 6.0)

    decompositions[aten.hardsigmoid_backward.default] = _hardsigmoid_bwd_for_aot
    _aot_decomps[aten.hardsigmoid_backward.default] = _hardsigmoid_bwd_for_aot

    # TRAIN.8 (2026-05-27): Monkey-patch the upstream ``_nll_loss_forward``
    # function to use compile-time-constant total_weight = target.numel().
    # This prevents the AOTAutograd partitioner from marking the
    # div(sum_loss, total_weight) forward output as InvalidNode when target
    # is assigned to the backward partition.
    #
    # cross_entropy_loss is a CIA op → decomposed at C++ dispatch level into
    # _log_softmax + nll_loss_forward BEFORE AOT tracing. The nll_loss_forward
    # decomposition (from core_aten_decompositions) produces the problematic
    # div node. Our monkey-patch of _nll_loss_forward removes target from the
    # target-dependency chain.
    #
    # NOTE: Don't call patch_nll_loss_forward() here during _suppress_upstream_decomps
    # as it causes import hangs. The patch is applied later when the inductor
    # backend is actually initialized (see inductor/__init__.py).
    # Also suppress cross_entropy_loss from our decomp tables as a belt-and-suspenders
    # measure (won't help with CIA dispatch but prevents redundant decomp lookup).
    _aot_decomps.pop(aten.cross_entropy_loss.default, None)
    decompositions.pop(aten.cross_entropy_loss.default, None)

    # OP.23: Clear the fast_random_decomps cache so subsequent calls
    # to select_decomp_table() pick up our decomposition additions.
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
    # T4.8 / CODEGEN.1: Foreach optimizer step lowerings — replace the
    # FallbackKernel path with ExternKernelOut subclasses that emit direct
    # _slang_foreach_optimizer() calls during codegen.
    import torch
    from torch._inductor.lowering import make_fallback

    from .activation import (
        _register_clamp_lowerings,
        _register_pointwise_math_lowerings,
        _register_pow_scalar_lowering,
    )
    from .attention import _register_sdpa_lowering
    from .bool_mask import _register_bool_mask_read_lowering
    from .complex import _register_complex_lowerings
    from .conv import _register_conv_and_pool_lowerings
    from .embedding import (
        _register_embedding_bag_forward,
        _get_embedding_dense_backward_impl,
    )
    from .fft import _register_fft_lowerings
    from .loss import _register_loss_lowerings
    from .masking import _register_masking_lowerings
    from .matmul import (
        _register_bmm_lowering,
        _register_linear_backward_decomposition,
        _register_matmul_backward,
        _register_matmul_lowering,
        _register_mm_int8_lowering,
        _register_mm_lowering,
    )
    from .mm_int8_op import _register_mm_int8_op
    from .rng import _register_multinomial_lowering
    from .rnn import _register_rnn_fallbacks
    from .scatter import _register_scatter_family_lowerings
    from .searchsorted import _register_searchsorted_and_repeat_interleave_tensor

    from .optimizer_lowerings import _register_optimizer_lowerings

    _register_optimizer_lowerings()

    # TRAIN.2: GPU-only max_pool2d backward via scatter_add template.
    # The custom op is registered by fx_passes/eager/pool.py during
    # register_eager_patch_custom_ops(); register it as a known extern
    # kernel for Inductor codegen.
    if hasattr(torch.ops.torch_vulkan, "conv2d_backward"):
        from torch._inductor.lowering import (
            register_lowering as _conv_bwd_reg_low,
        )
        from torch_vulkan.inductor.lowerings.conv_backward import (
            _get_conv2d_backward_custom_op_lowering,
        )
        _conv_bwd_reg_low(torch.ops.torch_vulkan.conv2d_backward.default)(
            _get_conv2d_backward_custom_op_lowering()
        )
    # S3.5b: conv1d_backward_core — opaque non-autograd op taking 3-D tensors.
    # Registered by _ensure_conv1d_backward_core_op_registered() on first
    # backward compile.  make_fallback emits an extern_kernel node so Inductor
    # doesn't try to decompose or pattern-match it.
    if hasattr(torch.ops.torch_vulkan, "conv1d_backward_core"):
        make_fallback(torch.ops.torch_vulkan.conv1d_backward_core.default)
    if hasattr(torch.ops.torch_vulkan, "max_pool2d_scatter_bwd"):
        make_fallback(torch.ops.torch_vulkan.max_pool2d_scatter_bwd.default)

    # CODEGEN.2: avg_pool2d scatter backward custom op (overlapping pools).
    if hasattr(torch.ops.torch_vulkan, "avg_pool2d_scatter_bwd"):
        make_fallback(torch.ops.torch_vulkan.avg_pool2d_scatter_bwd.default)

    # S2.5: avg_pool2d forward custom op — emits torch_vulkan private op
    # instead of public aten.avg_pool2d (anti-goal #6 close-out).
    if hasattr(torch.ops.torch_vulkan, "avg_pool2d"):
        make_fallback(torch.ops.torch_vulkan.avg_pool2d.default)

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

    # M-CV.2: masked_fill.Scalar — decompose to where(mask, value, self) so
    # backward graphs that contain aten.masked_fill.Scalar (e.g.
    # maximum_backward / minimum_backward decompositions) compile correctly
    # instead of falling through to the Vulkan eager kernel which has no
    # backing-buffer for the output tensor.
    @_reg_low(aten.masked_fill.Scalar, type_promotion_kind=None)
    def _vulkan_masked_fill_scalar(x, mask, value):
        # masked_fill(self, mask, value): set elements to `value` where mask
        # is True, leave others unchanged.
        #
        # Decompose as where(mask, value_scalar, x).  The upstream Inductor
        # where() lowering (torch/_inductor/lowering.py:where) handles scalar
        # ``a``/``b`` arguments via ``constant_like(a)(b)``, which inlines the
        # constant directly into the pointwise IR without creating a full_like
        # broadcast that the scheduler might mis-represent.  Passing the
        # scalar directly avoids the stale-expand/index-bounds codegen bug that
        # was producing all-zeros for the "gt=False → keep original" case in
        # maximum_backward.
        from torch._inductor import lowering as L

        return L.lowerings[aten.where.self](mask, float(value), x)

    # M-CV.2: aten.expand broadcast-stride-0 fix.
    #
    # Root cause: when Inductor constant-folds ``tangents_1.expand([N])``
    # (where tangents_1=1.0 for sum().backward()), it creates a ConstantBuffer
    # backed by a *single-element* Vulkan tensor with stride=0 but shape=[N].
    # The Slang kernel then reads ``in_ptr[x0]`` for x0=0..N-1, but the
    # backing buffer only has 1 element (4 bytes), so reads at x0>0 are
    # out-of-bounds and return 0 in the Vulkan driver — producing all-zero
    # gradients for the second argument of maximum/atan2.
    #
    # Fix: when aten.expand produces a broadcast (output has more elements than
    # input), call `.realize()` to force a fully-allocated N-element contiguous
    # buffer instead of a stride-0 ExpandView.  Every ``in_ptr[x0]`` read is
    # then in-bounds.  Identity expands (same numel) are left unchanged.
    #
    # We save the upstream expand lowering before registering the override so we
    # can call it without recursing into ourselves.  Both ``aten.expand`` (the
    # overloaded namespace key) and ``aten.expand.default`` (the specific
    # overload) may appear in the backward graph — override both.
    from torch._inductor import lowering as _L_inner

    _upstream_expand_fn = _L_inner.lowerings.get(aten.expand)
    _upstream_expand_default_fn = _L_inner.lowerings.get(aten.expand.default)

    def _realize_if_broadcast(x, sizes, result):
        """Call result.realize() when the expand creates stride-0 dims."""
        try:
            x_size = getattr(x, "get_size", lambda: ())()
            x_numel = 1
            for s in x_size:
                x_numel *= int(s)
            out_numel = 1
            for s in sizes:
                s_int = int(s)
                if s_int > 0:
                    out_numel *= s_int
            if x_numel > 0 and x_numel != out_numel:
                result.realize()
        except Exception:
            pass  # Dynamic shapes or non-integer sizes — leave as-is.
        return result

    if _upstream_expand_fn is not None:
        @_reg_low(aten.expand, type_promotion_kind=None)
        def _vulkan_expand(x, sizes):
            return _realize_if_broadcast(x, sizes, _upstream_expand_fn(x, sizes))

    if _upstream_expand_default_fn is not None:
        @_reg_low(aten.expand.default, type_promotion_kind=None)
        def _vulkan_expand_default(x, sizes):
            return _realize_if_broadcast(x, sizes, _upstream_expand_default_fn(x, sizes))

    from .norm import (
        _register_batch_norm_backward,
        _register_batch_norm_forward,
        _register_group_norm,
        _register_group_norm_backward,
        _register_group_norm_fused,
        _register_layer_norm,
        _register_layer_norm_backward,
    )
    from .softmax import _register_softmax, _register_softmax_backward
    from .view import _register_view_lowerings

    _register_view_lowerings()
    _register_conv_and_pool_lowerings()
    # S1: register Slang tile choices for conv2d via autotune_select_algorithm.
    # Must run AFTER _register_conv_and_pool_lowerings() — it overrides the
    # lowering that function just registered.
    from ..templates.caller.conv_tile.install import install_external_conv

    install_external_conv()
    # M17.3: adaptive_avg_pool2d backward lowering
    from . import pool  # noqa: F811

    _register_bmm_lowering()
    _register_mm_lowering()
    _register_mm_int8_op()
    _register_mm_int8_lowering()
    _register_matmul_lowering()
    _register_matmul_backward()
    # M19.1: install pure-aten decomp for ``aten.linear_backward.default``.
    # Replaces the 8-dispatch C++ eager path with mm + sum primitives that
    # Inductor can schedule + fuse. Enabled by M22.13 — the underlying
    # mm tile transpose-a path is now corrected in
    # ``csrc/ops/matmul_ops.cpp::vulkan_addmm_out`` (detects
    # ``is_t_transposed(mat2)`` and routes to ``vulkan_linear`` which has
    # the validated ``is_t_transposed`` fast-path). See M19.1 docstring
    # in lowerings/matmul.py for the dispatch ratchet discussion.
    _register_linear_backward_decomposition()
    # OP.26: native SDPA lowering — routes aten.scaled_dot_product_attention
    # directly to the FlashAttention template, eliminating the symptom-fix
    # pattern matcher in fx_passes/patterns/sdpa.py (anti-goal #5).
    _register_sdpa_lowering()
    _register_layer_norm()
    # GN.1: Fused GN forward via standalone Slang shader (ExternKernelOut).
    # Replaces ~10 dispatch decomposition with 1 fused dispatch.
    # Currently DISABLED — runtime crash in group_norm.slang shader
    # when called via ExternKernelOut codegen during Inductor compilation.
    # When re-enabled, MUST be registered BEFORE _register_group_norm()
    # (Inductor's register_lowering is first-come-first-served).
    # _register_group_norm_fused()
    _register_group_norm()
    _register_softmax()
    _register_batch_norm_forward()
    _register_clamp_lowerings()
    _register_pow_scalar_lowering()
    _register_pointwise_math_lowerings()
    _register_complex_lowerings()
    # CG.M8: Register inline bwd_diff lowerings FIRST so they take priority
    # over the external-dispatch shims in bwd_lowerings.py.
    # register_lowering is first-come-first-served (get_overloads() filters
    # already-registered OpOverloadPacket overloads), so inline MUST go first.
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
    # bwd_lowerings is registered AFTER inline so it only claims ops that
    # the inline set does not cover (algebraic fallbacks: sigmoid, tanh, gelu,
    # mish, etc.).
    from ..bwd_lowerings import register as _register_bwd_lowerings

    _register_bwd_lowerings()
    _register_softmax_backward()
    _register_layer_norm_backward()
    _register_group_norm_backward()
    _register_batch_norm_backward()
    _register_loss_lowerings()
    # NOTE (anti-goal #3): _get_embedding_dense_backward_impl() no longer
    # registers — registration is in bwd_lowerings.py via _register_bwd_lowerings().
    _register_embedding_bag_forward()
    _register_fft_lowerings()
    # E2: tril/triu forward lowerings — backward reuses the same op
    # (see masking.py docstring), so this also closes the backward gap.
    _register_masking_lowerings()
    _register_scatter_family_lowerings()
    # M16.2: bool-mask read (`x[mask]`) — eager override (PrivateUse1)
    # decomposes to nonzero + index_select; Inductor lowering falls back
    # to eager for bool masks and delegates to upstream index_impl for int.
    _register_bool_mask_read_lowering()
    from .bool_mask import _register_index_tensor_lowering

    _register_index_tensor_lowering()
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
    # M19.6: foreach pointwise lowering coverage — validates upstream lowerings
    # for all 16 foreach element-wise ops and suppresses any stray AOT decomps.
    from .foreach_pointwise import register_foreach_pointwise_lowerings

    register_foreach_pointwise_lowerings()
