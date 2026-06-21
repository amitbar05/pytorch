"""Eager custom-op registrations for fused FX-pattern targets.

This module is a thin re-export shim — the real implementations live in
the ``eager/`` sub-package (M15.1.e split).  All existing imports of the
form ``from torch_vulkan.inductor.fx_passes.eager_patches import X``
continue to work unchanged.
"""

from __future__ import annotations

from .eager import (
    _ensure_adaptive_avg_pool2d_op_registered,
    _ensure_avg_pool2d_op_registered,
    _ensure_addmm_gelu_op_registered,
    _ensure_conv1d_with_optional_bias_op_registered,
    _ensure_conv2d_with_optional_bias_op_registered,
    _ensure_flash_attention_op_registered,
    _ensure_foreach_adamw_step_op_registered,
    _ensure_foreach_lion_step_op_registered,
    _ensure_foreach_sgd_momentum_step_op_registered,
    _ensure_foreach_sgd_step_op_registered,
    _ensure_max_pool2d_op_registered,
    _ensure_qkv_cat_op_registered,
    _ensure_scaled_bmm_op_registered,
    _ensure_sdpa_with_optional_mask_op_registered,
    _ensure_swiglu_op_registered,
    register_eager_patch_custom_ops,
)

__all__ = [
    "_ensure_addmm_gelu_op_registered",
    "_ensure_adaptive_avg_pool2d_op_registered",
    "_ensure_avg_pool2d_op_registered",
    "_ensure_conv1d_with_optional_bias_op_registered",
    "_ensure_conv2d_with_optional_bias_op_registered",
    "_ensure_flash_attention_op_registered",
    "_ensure_foreach_adamw_step_op_registered",
    "_ensure_foreach_lion_step_op_registered",
    "_ensure_foreach_sgd_momentum_step_op_registered",
    "_ensure_foreach_sgd_step_op_registered",
    "_ensure_max_pool2d_op_registered",
    "_ensure_qkv_cat_op_registered",
    "_ensure_scaled_bmm_op_registered",
    "_ensure_sdpa_with_optional_mask_op_registered",
    "_ensure_swiglu_op_registered",
    "register_eager_patch_custom_ops",
]
