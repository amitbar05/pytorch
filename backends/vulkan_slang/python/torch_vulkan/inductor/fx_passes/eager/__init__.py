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
    _ensure_conv2d_backward_op_registered,
    _ensure_conv2d_gn_relu_fused_op_registered,
    _ensure_conv2d_relu_fused_op_registered,
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


# M-pipeline-1 (2026-05-18): module-level idempotency guard.  Even though
# every ``_ensure_*`` below is independently idempotent (each early-returns
# when the op already exists on ``torch.ops.torch_vulkan``), the lazy shim
# ``_ensure_patch_custom_ops`` inside ``_patched_conv2d``
# (``python/torch_vulkan/__init__.py``) calls
# ``register_eager_patch_custom_ops`` on every first ``F.conv2d`` invocation
# from a freshly-created closure (``_patch_custom_ops_done`` resets per
# ``_register_optional_tensor_workarounds`` invocation).  Tracking the
# "already done" state at this module scope guarantees that even when the
# closure flag is False, the heavy work runs at most once per process.
_REGISTER_DONE = False


def register_eager_patch_custom_ops() -> None:
    """Register the conv2d / conv1d / sdpa / max_pool2d / adaptive_avg_pool2d
    custom_ops used by the eager monkey-patches in
    ``python/torch_vulkan/__init__.py`` (PF.30.a/.b/.d).

    Called once during backend init, before ``_register_optional_tensor_workarounds``
    swaps in the patched ``F.conv2d`` / ``F.conv1d`` /
    ``F.scaled_dot_product_attention`` / ``F.max_pool2d`` /
    ``F.adaptive_avg_pool2d``. Idempotent — each ``_ensure_*`` is a singleton,
    and a module-level ``_REGISTER_DONE`` flag short-circuits subsequent
    invocations entirely (M-pipeline-1).
    """
    global _REGISTER_DONE
    if _REGISTER_DONE:
        return
    _ensure_conv2d_with_optional_bias_op_registered()
    _ensure_conv1d_with_optional_bias_op_registered()
    _ensure_sdpa_with_optional_mask_op_registered()
    _ensure_max_pool2d_op_registered()
    _ensure_adaptive_avg_pool2d_op_registered()
    # M17.2 Phase 1: conv+ReLU fused custom op
    _ensure_conv2d_relu_fused_op_registered()
    # M17.2 Phase 2: conv+GN+ReLU triple-fusion custom op
    _ensure_conv2d_gn_relu_fused_op_registered()
    # M17.8.d.2 / M18.2: opaque non-autograd conv2d_backward custom op so
    # AOTAutograd can preserve the backward as a single FX node rather than
    # tracing through ``torch.empty_like`` sub-ops (which the partitioner
    # collapses to literal zeros).
    _ensure_conv2d_backward_op_registered()
    # M18.8.b: install the enhanced conv→GN→ReLU fusion pass that matches
    # the Dynamo-emitted forms of the monkey-patched ``F.group_norm`` and
    # ``F.relu`` (the legacy fusion in meta_patches/decomposition_passes.py
    # only matches the post-AOT-decomp form ``aten.native_group_norm`` /
    # ``aten.relu`` and therefore never fires for the eager-Vulkan
    # ``nn.Sequential(Conv, GN, ReLU)`` topology).
    from ..post_grad import install_conv_patched_gn_relu_fusion

    install_conv_patched_gn_relu_fusion()
    # T4.8 foreach optimizer custom ops — registered lazily by
    # install_external_optimizer() in vulkan_template_caller. They live in
    # this module (below) because eager_patches is the canonical home for
    # `torch_vulkan::*` custom_op factories.
    _REGISTER_DONE = True


__all__ = [
    "_ensure_addmm_gelu_op_registered",
    "_ensure_adaptive_avg_pool2d_op_registered",
    "_ensure_conv1d_with_optional_bias_op_registered",
    "_ensure_conv2d_backward_op_registered",
    "_ensure_conv2d_gn_relu_fused_op_registered",
    "_ensure_conv2d_relu_fused_op_registered",
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
