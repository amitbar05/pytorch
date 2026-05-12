"""Built-in FX patterns for the Vulkan/Slang Inductor backend.

Each pattern lives in its own module; this file just imports them so
the registration side-effects fire. See ``fx_passes/patterns/<name>.py``
for the per-pattern match/rewrite logic.
"""

from __future__ import annotations

from . import (
    addmm_gelu,
    conv_im2col,
    matmul_epilogue,
    mm_add,
    op_class_fusion,
    qkv_cat,
    redundant_copy,
    relu_clamp_min,
    scaled_bmm,
    sdpa,
    swiglu,
)

__all__ = [
    "addmm_gelu",
    "swiglu",
    "scaled_bmm",
    "mm_add",
    "sdpa",
    "qkv_cat",
    "conv_im2col",
    "matmul_epilogue",
    "redundant_copy",
    "relu_clamp_min",
    "op_class_fusion",
]
