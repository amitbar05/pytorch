"""Eager custom-op registrations — sub-package.

Each submodule owns one op family; the public surface (all ``_ensure_*``
functions + ``register_eager_patch_custom_ops``) is re-exported here for
backward-compatible access from ``eager_patches.py``.
"""

from __future__ import annotations

from .addmm import (
    _ensure_addmm_gelu_op_registered,
    _ensure_scaled_bmm_op_registered,
)
from .conv import (
    _ensure_conv1d_with_optional_bias_op_registered,
    _ensure_conv2d_with_optional_bias_op_registered,
)
from .optimizer import (
    _ensure_foreach_adamw_step_op_registered,
    _ensure_foreach_lion_step_op_registered,
    _ensure_foreach_sgd_momentum_step_op_registered,
    _ensure_foreach_sgd_step_op_registered,
)
from .pool import (
    _ensure_adaptive_avg_pool2d_op_registered,
    _ensure_max_pool2d_op_registered,
)
from .qkv import _ensure_qkv_cat_op_registered
from .sdpa import (
    _ensure_flash_attention_op_registered,
    _ensure_sdpa_with_optional_mask_op_registered,
)
from .swiglu import _ensure_swiglu_op_registered


def register_eager_patch_custom_ops() -> None:
    """Register the conv2d / conv1d / sdpa / max_pool2d / adaptive_avg_pool2d
    custom_ops used by the eager monkey-patches in
    ``python/torch_vulkan/__init__.py`` (PF.30.a/.b/.d).

    Called once during backend init, before ``_register_optional_tensor_workarounds``
    swaps in the patched ``F.conv2d`` / ``F.conv1d`` /
    ``F.scaled_dot_product_attention`` / ``F.max_pool2d`` /
    ``F.adaptive_avg_pool2d``. Idempotent — each ``_ensure_*`` is a singleton.
    """
    _ensure_conv2d_with_optional_bias_op_registered()
    _ensure_conv1d_with_optional_bias_op_registered()
    _ensure_sdpa_with_optional_mask_op_registered()
    _ensure_max_pool2d_op_registered()
    _ensure_adaptive_avg_pool2d_op_registered()
    # T4.8 foreach optimizer custom ops — registered lazily by
    # install_external_optimizer() in vulkan_template_caller. They live in
    # this module (below) because eager_patches is the canonical home for
    # `torch_vulkan::*` custom_op factories.


__all__ = [
    "_ensure_addmm_gelu_op_registered",
    "_ensure_adaptive_avg_pool2d_op_registered",
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
