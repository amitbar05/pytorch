"""PF.6.b — bwd_diff dispatch package.

Re-exports all public symbols from the submodules: unary dispatch,
binary dispatch, and shared emission helpers (dtype widening, Vulkan
validation, backward-kind resolution, template-backward dispatch,
reduction backward broadcast).
"""

from torch_vulkan.inductor.bwd_diff.binary import dispatch_binary_bwd
from torch_vulkan.inductor.bwd_diff.emit_helpers import (
    _DEFAULT_NUMTHREADS,
    _cache_key,
    _check_float,
    _check_vulkan,
    _emit_reduction_bwd_src,
    _ensure_f32,
    _entry,
    _mm_with_transpose,
    _narrow_from_f32,
    _ResolvedBackward,
    _slang_dtype_str,
    dispatch_reduction_bwd,
    dispatch_template_bwd,
    resolve_backward_kind,
)
from torch_vulkan.inductor.bwd_diff.unary import dispatch_unary_bwd

__all__ = [
    "dispatch_unary_bwd",
    "dispatch_binary_bwd",
    "dispatch_template_bwd",
    "dispatch_reduction_bwd",
    "resolve_backward_kind",
    "_ResolvedBackward",
]
