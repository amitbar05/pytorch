"""Python-level FakeTensor workarounds for view ops on Vulkan.

For each view op below we install a ``fake_impl`` via the internal
``FakeImplHolder.kernels`` list. FakeTensor's ``_dispatch_impl`` checks this
registry BEFORE the meta-in-TLS dispatch path, so the workaround takes
priority over our PrivateUse1 C++ kernel (which would otherwise try to
allocate real Vulkan storage and fail on FakeTensor inputs).

Split across sub-modules (anti-goal #7: no file > 800 lines):
  - shape_ops:      genuine shape-inference fake_impls
  - dtype_ops:      type-promotion / pointwise / reduction fake_impls
  - faketensor_hooks: FakeTensor infrastructure monkey-patches
  - op_registration: meta-decomp / PrivateUse1 / matmul / einsum / SDPA
  - autograd_registrations: AutogradPrivateUse1 pyimpls
  - joint_graph_passes: joint-graph device fix / lifetime annotation
  - decomposition_passes: decomposition overrides / pre-grad passes
"""

from __future__ import annotations

from typing import Callable

import torch
from torch._library.fake_impl import Kernel
from torch._library.simple_registry import singleton

# ── AutogradPrivateUse1 pyimpls ─────────────────────────────────────────────
from .autograd_registrations import (
    _register_activation_autograd_pyimpl,
    _register_permute_family_autograd_pyimpl,
    _register_view_symint_autograd_pyimpl,
)

# ── Decomposition / pre-grad passes ─────────────────────────────────────────
from .decomposition_passes import (
    _patch_decompositions,
    _patch_pre_grad_passes_for_conv_gn_relu_fusion,
    _patch_pre_grad_passes_for_optimizer_foreach,
    _patch_pre_grad_passes_for_relu_rewrite,
)

# ── Dtype / pointwise / reduction fake impls ────────────────────────────────
from .dtype_ops import (
    _amax_amin_fake,
    _argmax_argmin_fake,
    _binary_fake,
    _binary_no_alpha_fake,
    _binary_scalar_fake,
    _clamp_fake,
    _clamp_max_fake,
    _clamp_min_fake,
    _comparison_out_fake,
    _comparison_fake,
    _cross_entropy_loss_backward_fake,
    _elu_fake,
    _expand_backward_fake,
    _gelu_fake,
    _hardsigmoid_backward_fake,
    _hardswish_backward_fake,
    _hardtanh_backward_fake,
    _hardtanh_fake,
    _leaky_relu_fake,
    _linear_fake,
    _log_softmax_fake,
    _max_dim_fake,
    _mean_dim_fake,
    _min_dim_fake,
    _mish_backward_fake,
    _nll_loss_backward_fake,
    _permute_backward_fake,
    _repeat_backward_fake,
    _reshape_alias_fake,
    _select_backward_fake,
    _selu_backward_fake,
    _sigmoid_backward_fake,
    _slice_backward_fake,
    _softmax_fake,
    _softplus_backward_fake,
    _softplus_fake,
    _squeeze_backward_fake,
    _sum_dim_fake,
    _t_backward_fake,
    _tanh_backward_fake,
    _threshold_backward_fake,
    _transpose_backward_fake,
    _unary_fake,
    _unsqueeze_backward_fake,
    _view_backward_fake,
    _where_fake,
)

# ── FakeTensor infrastructure patches ───────────────────────────────────────
from .faketensor_hooks import (
    _patch_dynamo_clone_input_for_vulkan,
    _patch_fake_tensor_meta_conversion,
    _patch_fake_tensor_skip_const_fold_for_vulkan_null,
    _patch_fake_tensor_view_op_device,
    _patch_fx_graph_cache_reduce_tensor_for_vulkan,
    _patch_graph_lowering_get_attr_for_vulkan_null,
    _patch_tensor_deepcopy_for_vulkan,
)

# ── Joint graph / backward compilation passes ───────────────────────────────
from .joint_graph_passes import (
    _FixMetaDevicePass,
    _install_joint_partition_device_fix,
    _joint_trace_ctx,
    _patch_compile_fx_for_backward,
    _skip_misc_patterns_for_vulkan,
)

# ── Op registrations / decompositions ───────────────────────────────────────
from .op_registration import (
    _disable_bmm_to_mm_for_vulkan,
    _patch_einsum_proxy_decomp,
    _patch_proxy_call_matmul_decomp,
    _register_backward_meta_decomps,
    _register_logical_and_for_vulkan,
    _register_matmul_meta,
    _register_sdpa_meta,
)

# ── Shape-inference fake impls ──────────────────────────────────────────────
from .shape_ops import (
    _addmm_fake,
    _as_strided_fake,
    _as_strided_scatter_fake,
    _bmm_fake,
    _cat_fake,
    _chunk_fake,
    _clone_fake,
    _constant_pad_nd_fake,
    _contiguous_fake,
    _convolution_backward_overrideable_fake,
    _convolution_overrideable_fake,
    _copy__fake,
    _detach_fake,
    _diagonal_fake,
    _embedding_backward_fake,
    _embedding_fake,
    _empty_like_fake,
    _expand_fake,
    _fft_c2c_fake,
    _fft_c2r_fake,
    _fft_r2c_fake,
    _fill_scalar_fake,
    _flatten_using_ints_fake,
    _flip_fake,
    _gather_fake,
    _gather_fake_backward,
    _index_put_fake,
    _index_put_fake_impl,
    _index_select_fake,
    _linalg_svd_fake,
    _masked_fill_scalar_fake,
    _masked_scatter_fake,
    _mm_fake,
    _narrow_fake,
    _one_hot_fake,
    _permute_fake,
    _randperm_fake,
    _repeat_fake,
    _repeat_interleave_self_int_fake,
    _reshape_as_fake,
    _roll_fake,
    _scatter_add_fake,
    _scatter_reduce_fake,
    _scatter_src_fake,
    _scatter_value_fake,
    _sdpa_backward_fake,
    _select_int_fake,
    _slice_fake,
    _split_with_sizes_fake,
    _squeeze_dim_fake,
    _stack_fake,
    _sum_backward_fake,
    _t_fake,
    _tile_fake,
    _transpose_int_fake,
    _unflatten_int_fake,
    _unsqueeze_fake,
    _view_as_fake,
    _view_fake,
    _where_self_fake,
    _zero__fake,
)

# ── Registry ──────────────────────────────────────────────────────────────────
#
# G.1 audit (post): 147 entries — all classified (a), genuine FakeTensor
# shape-inference patches. The 15 dead backward-op fallbacks (meta decomps
# in op_registration.py win the dispatch race, or AOT decomp table in
# decomposition_passes.py handles shape inference) were removed; see
# agent_space/m15.2_audit_report.md §2 + G.1 followup.

_OP_IMPLS: dict[str, Callable] = {
    # View / shape ops
    "aten::as_strided": _as_strided_fake,
    "aten::permute": _permute_fake,
    "aten::transpose.int": _transpose_int_fake,
    "aten::t": _t_fake,
    "aten::view": _view_fake,
    "aten::_unsafe_view": _view_fake,
    "aten::reshape": _view_fake,
    # PF.55: composite-implicit-autograd shape ops whose C++
    # decomposition bypasses Python dispatch for the inner ``view`` call,
    # blowing up our PrivateUse1 C++ adapter on symbolic SymInt sizes.
    "aten::flatten.using_ints": _flatten_using_ints_fake,
    "aten::unflatten.int": _unflatten_int_fake,
    "aten::view_as": _view_as_fake,
    "aten::reshape_as": _reshape_as_fake,
    "aten::unsqueeze": _unsqueeze_fake,
    "aten::squeeze.dim": _squeeze_dim_fake,
    "aten::expand": _expand_fake,
    "aten::slice.Tensor": _slice_fake,
    "aten::constant_pad_nd": _constant_pad_nd_fake,
    "aten::as_strided_scatter": _as_strided_scatter_fake,
    "aten::masked_scatter": _masked_scatter_fake,
    # Memory / copy ops
    "aten::clone": _clone_fake,
    "aten::copy_": _copy__fake,
    "aten::contiguous": _contiguous_fake,
    "aten::empty_like": _empty_like_fake,
    "aten::fill_.Scalar": _fill_scalar_fake,
    "aten::zero_": _zero__fake,
    "aten::detach": _detach_fake,
    # Shape ops
    "aten::flip": _flip_fake,
    "aten::narrow": _narrow_fake,
    "aten::select.int": _select_int_fake,
    "aten::diagonal": _diagonal_fake,
    "aten::split_with_sizes": _split_with_sizes_fake,
    "aten::chunk": _chunk_fake,
    "aten::cat": _cat_fake,
    "aten::stack": _stack_fake,
    "aten::repeat": _repeat_fake,
    "aten::tile": _tile_fake,
    "aten::roll": _roll_fake,
    "aten::where.self": _where_self_fake,
    "aten::index_select": _index_select_fake,
    "aten::scatter_reduce.two": _scatter_reduce_fake,
    "aten::masked_fill_.Scalar": _masked_fill_scalar_fake,
    # BLAS
    "aten::mm": _mm_fake,
    "aten::bmm": _bmm_fake,
    "aten::addmm": _addmm_fake,
    "aten::linear": _linear_fake,
    # Pointwise unary
    "aten::neg": _unary_fake,
    "aten::abs": _unary_fake,
    "aten::exp": _unary_fake,
    "aten::log": _unary_fake,
    "aten::log2": _unary_fake,
    "aten::log10": _unary_fake,
    "aten::log1p": _unary_fake,
    "aten::sqrt": _unary_fake,
    "aten::rsqrt": _unary_fake,
    "aten::ceil": _unary_fake,
    "aten::floor": _unary_fake,
    "aten::round": _unary_fake,
    "aten::sign": _unary_fake,
    "aten::sgn": _unary_fake,
    "aten::reciprocal": _unary_fake,
    "aten::sin": _unary_fake,
    "aten::cos": _unary_fake,
    "aten::tan": _unary_fake,
    "aten::atan": _unary_fake,
    "aten::erf": _unary_fake,
    "aten::logical_not": _unary_fake,
    "aten::bitwise_not": _unary_fake,
    "aten::isnan": _unary_fake,
    "aten::isinf": _unary_fake,
    # Logical binary ops — return bool tensors.
    # pow_backward_exponent (PowBackward1) calls at::logical_and() to guard
    # the 0^0 edge case.  The real BackendCompilerFailed root cause was
    # _where_self_fake not broadcasting (see shape_ops.py fix), but the
    # fake_impls here ensure FakeTensorMode doesn't fall to the C++ Vulkan
    # dispatch for logical_and.
    "aten::logical_and.Tensor": _comparison_fake,
    "aten::logical_and.out": _comparison_out_fake,
    "aten::logical_or.Tensor": _comparison_fake,
    "aten::logical_or.out": _comparison_out_fake,
    # Activations (forward)
    "aten::relu": _unary_fake,
    "aten::sigmoid": _unary_fake,
    "aten::tanh": _unary_fake,
    "aten::silu": _unary_fake,
    "aten::selu": _unary_fake,
    "aten::mish": _unary_fake,
    "aten::hardswish": _unary_fake,
    "aten::hardsigmoid": _unary_fake,
    "aten::gelu": _gelu_fake,
    "aten::leaky_relu": _leaky_relu_fake,
    "aten::elu": _elu_fake,
    "aten::hardtanh": _hardtanh_fake,
    "aten::softplus": _softplus_fake,
    "aten::clamp": _clamp_fake,
    "aten::clamp_min": _clamp_min_fake,
    "aten::clamp_max": _clamp_max_fake,
    "aten::softmax.int": _softmax_fake,
    "aten::log_softmax.int": _log_softmax_fake,
    "aten::_softmax": _softmax_fake,
    "aten::_log_softmax": _log_softmax_fake,
    # Pointwise binary (Tensor)
    "aten::add.Tensor": _binary_fake,
    "aten::sub.Tensor": _binary_fake,
    "aten::mul.Tensor": _binary_no_alpha_fake,
    "aten::div.Tensor": _binary_no_alpha_fake,
    "aten::pow.Tensor_Tensor": _binary_no_alpha_fake,
    "aten::fmod.Tensor": _binary_no_alpha_fake,
    "aten::remainder.Tensor": _binary_no_alpha_fake,
    "aten::atan2": _binary_no_alpha_fake,
    "aten::maximum": _binary_no_alpha_fake,
    "aten::minimum": _binary_no_alpha_fake,
    # Pointwise binary (Scalar)
    "aten::add.Scalar": _binary_scalar_fake,
    "aten::sub.Scalar": _binary_scalar_fake,
    "aten::mul.Scalar": _binary_scalar_fake,
    "aten::div.Scalar": _binary_scalar_fake,
    "aten::pow.Tensor_Scalar": _binary_scalar_fake,
    # Comparisons
    "aten::eq.Tensor": _comparison_fake,
    "aten::ne.Tensor": _comparison_fake,
    "aten::lt.Tensor": _comparison_fake,
    "aten::gt.Tensor": _comparison_fake,
    "aten::le.Tensor": _comparison_fake,
    "aten::ge.Tensor": _comparison_fake,
    "aten::eq.Scalar": _comparison_fake,
    "aten::ne.Scalar": _comparison_fake,
    "aten::lt.Scalar": _comparison_fake,
    "aten::gt.Scalar": _comparison_fake,
    "aten::le.Scalar": _comparison_fake,
    "aten::ge.Scalar": _comparison_fake,
    # Reductions
    "aten::sum.dim_IntList": _sum_dim_fake,
    "aten::sum": _sum_dim_fake,
    "aten::mean.dim": _mean_dim_fake,
    "aten::mean": _mean_dim_fake,
    "aten::amax": _amax_amin_fake,
    "aten::amin": _amax_amin_fake,
    "aten::argmax": _argmax_argmin_fake,
    "aten::argmin": _argmax_argmin_fake,
    "aten::max.dim": _max_dim_fake,
    "aten::min.dim": _min_dim_fake,
    # Conv
    "aten::convolution_overrideable": _convolution_overrideable_fake,
    "aten::convolution_backward_overrideable": _convolution_backward_overrideable_fake,
    # Indexing
    "aten::gather": _gather_fake_backward,
    "aten::scatter_.src": _scatter_src_fake,
    "aten::scatter_.value": _scatter_value_fake,
    "aten::scatter_add_": _scatter_add_fake,
    "aten::index_put_": _index_put_fake,
    "aten::repeat_interleave.self_int": _repeat_interleave_self_int_fake,
    # Attention backward
    "aten::_scaled_dot_product_flash_attention_backward": _sdpa_backward_fake,
    # FFT
    "aten::_fft_r2c": _fft_r2c_fake,
    "aten::_fft_c2c": _fft_c2c_fake,
    "aten::_fft_c2r": _fft_c2r_fake,
    # SVD
    "aten::linalg_svd": _linalg_svd_fake,
    # Activation backward ops
    "aten::hardtanh_backward": _hardtanh_backward_fake,
    "aten::threshold_backward": _threshold_backward_fake,
    "aten::selu_backward": _selu_backward_fake,
    "aten::mish_backward": _mish_backward_fake,
    "aten::hardswish_backward": _hardswish_backward_fake,
    "aten::hardsigmoid_backward": _hardsigmoid_backward_fake,
    "aten::softplus_backward": _softplus_backward_fake,
    "aten::sigmoid_backward": _sigmoid_backward_fake,
    "aten::tanh_backward": _tanh_backward_fake,
    # Loss backward
    "aten::nll_loss_backward": _nll_loss_backward_fake,
    "aten::_cross_entropy_loss_backward": _cross_entropy_loss_backward_fake,
    # Reduction backward
    "aten::_sum_backward": _sum_backward_fake,
    # View ops needed by backward functionalization
    "aten::_reshape_alias": _reshape_alias_fake,
    "aten::select_backward": _select_backward_fake,
    "aten::slice_backward": _slice_backward_fake,
    "aten::unsqueeze_backward": _unsqueeze_backward_fake,
    "aten::squeeze_backward": _squeeze_backward_fake,
    "aten::t_backward": _t_backward_fake,
    "aten::transpose_backward": _transpose_backward_fake,
    "aten::view_backward": _view_backward_fake,
    "aten::permute_backward": _permute_backward_fake,
    "aten::expand_backward": _expand_backward_fake,
    "aten::repeat_backward": _repeat_backward_fake,
    # Embedding
    "aten::embedding": _embedding_fake,
    "aten::embedding_backward": _embedding_backward_fake,
    "aten::one_hot": _one_hot_fake,
    # Factory ops
    "aten::randperm.default": _randperm_fake,
}

_patched = False


def apply() -> None:
    global _patched
    if _patched:
        return

    for op_name, impl in _OP_IMPLS.items():
        try:
            entry = singleton.find(op_name)
            holder = entry.fake_impl
            holder.kernels = [Kernel(impl, __file__)]
        except Exception as e:  # pragma: no cover
            import logging

            logging.getLogger(__name__).warning(
                "Installing fake impl for %s failed: %s", op_name, e
            )

    # Register SDPA in meta_table so FakeTensorMode._dispatch_impl skips the
    # CompositeImplicitAutograd decompose() path and routes through the
    # op_registration meta decomposition (the _sdpa_fake fake_impl fallback
    # was deleted in the G.1 dead-fallback sweep).
    _register_sdpa_meta()
    _register_matmul_meta()
    _patch_proxy_call_matmul_decomp()
    _patch_einsum_proxy_decomp()
    _disable_bmm_to_mm_for_vulkan()
    # M-CV.2 Phase 2: register logical_and.{Tensor,out} for Vulkan eager.
    # PowBackward1's pow_backward_exponent calls at::logical_and() on Vulkan
    # tensors. The compiled backward lowers through overrides.py (&&), but
    # eager callers and any un-lowered path need the PrivateUse1 impl.
    _register_logical_and_for_vulkan()

    # PF.55 — Python AutogradPrivateUse1 impl for view/reshape/_unsafe_view
    # so dynamic-shape compile (`torch.compile(dynamic=True)`) doesn't blow
    # up our C++ ``vulkan_view_autograd_adapter``'s ``expect_int()`` call
    # on a symbolic SymInt size. Concrete sizes redispatch back to the C++
    # adapter to preserve zero-copy + autograd semantics.
    _register_view_symint_autograd_pyimpl()
    # T.12.A — Python AutogradPrivateUse1 impl for permute/transpose.int/t
    # so the FakeTensor result is a proper view aliasing ``self`` storage
    # (the C++ kernels return a fresh non-aliasing tensor, which makes
    # AOTAutograd lift the result as a frozen tensor constant and Inductor
    # constant-fold it to garbage uniform values).
    _register_permute_family_autograd_pyimpl()
    # relu ← clamp_min decomposition handles the Vulkan-aware backward
    # through clamp's working AutogradPrivateUse1 C++ adapter.

    # Override decompositions for activation backward ops that hit a PyTorch 2.11
    # FakeTensorMode bug: torch.where(cond, 0.0_scalar, tensor) returns shape []
    # instead of the tensor's shape. These decompositions avoid the scalar-first
    # where(cond, scalar, tensor) pattern.
    _patch_decompositions()

    # Inductor's _misc_patterns_init traces randperm_index_add_pattern with
    # tracing_mode="real". This calls randperm(device='vulkan:0') and then
    # __getitem__ on the result, which TorchFunctionMetadataMode dispatches
    # directly to C++ PrivateUse1, bypassing FakeTensorMode and our fake_impl.
    # Since randperm_index_add_pattern can never appear in a Vulkan FX graph
    # (we don't have aten::randperm on PrivateUse1), skip this init for Vulkan.
    _skip_misc_patterns_for_vulkan()

    # During Inductor's fake_tensor_prop on backward graphs, saved forward
    # tensors may arrive as plain `meta` device tensors (not FakeTensors).
    # This happens because AOT Autograd stores the saved inputs with device
    # metadata from the FakeTensor forward pass, and Inductor's backward
    # compilation propagates these as raw meta tensors rather than wrapping
    # them in the current FakeTensorMode's FakeTensors. The standard
    # validate_and_convert_non_fake_tensors rejects non-fake inputs unless
    # allow_non_fake_inputs=True. We patch it to auto-convert meta tensors
    # (which are shape-only and fully equivalent to FakeTensors) so the
    # backward graph compiles correctly under the inductor backend.
    _patch_fake_tensor_meta_conversion()
    # Always-on: Dynamo's `clone_input` calls `data_ptr()` on the input,
    # which crashes for FakeTensor inputs. Vulkan needs the same xla-style
    # `torch.clone(x)` fallback so attention-shaped graphs compile.
    _patch_dynamo_clone_input_for_vulkan()
    _patch_fake_tensor_view_op_device()

    # Skip _dispatch_impl's constant-fold path when a vulkan FakeTensor
    # holds a null-storage ``.constant`` — otherwise binary ops like
    # ``aten.add(<vulkan FakeTensor>, 1.0)`` during Inductor's FX trace
    # of conv graphs run the real C++ vulkan_add against an
    # un-allocated buffer and raise "Tensor has no backing Vulkan buffer".
    _patch_fake_tensor_skip_const_fold_for_vulkan_null()

    # PF.13.b.4 layered fix #2: AOT autograd's lazy-backward path
    # ``copy.deepcopy(bw_module)`` chokes on non-leaf Vulkan tensors
    # baked into the bw_module as saved-for-backward constants.
    _patch_tensor_deepcopy_for_vulkan()

    # PF.13.b.4 layered fix #4: FxGraphCachePickler._reduce_tensor calls
    # ``t.tolist()`` on Vulkan view tensors (e.g. k.transpose(-2,-1))
    # whose storage data pointer is invalid during AOTAutograd compilation.
    _patch_fx_graph_cache_reduce_tensor_for_vulkan()

    # M-CV.2: fix GraphLowering.get_attr for zero-stride Vulkan constants.
    # M-CV.2: GraphLowering.get_attr calls value.tolist() on Vulkan FakeTensors
    # that have no real backing storage (null data_ptr or zero strides).
    # This produces zeros → wrong gradients in maximum/atan2/etc backward.
    # Route null-storage Vulkan constants to add_tensor_constant() instead,
    # creating a ConstantBuffer that receives the real values at runtime.
    _patch_graph_lowering_get_attr_for_vulkan_null()

    # C1: rewrite ``aten.relu`` to ``where + gt + full_like`` BEFORE the
    # AOT joint trace, so ReluBackward0/threshold_backward never fires
    # against meta-cascaded saved outputs.
    _patch_pre_grad_passes_for_relu_rewrite()

    # T4.8: optimizer foreach step pattern matching on the pre-grad graph
    # catches in-place ``add_/mul_/addcdiv_/addcmul_`` sequences BEFORE
    # AOTAutograd functionalization decomposes them into triplets/doublets.
    _patch_pre_grad_passes_for_optimizer_foreach()

    # M17.2 Phase 2: fuse conv → group_norm → relu on the pre-grad graph
    # BEFORE AOTAutograd decomposition (native_group_norm is still intact).
    _patch_pre_grad_passes_for_conv_gn_relu_fusion()

    _patched = True
