"""PF.6.b — re-export shim for the bwd_diff dispatch package.

This file is retained so that all existing imports continue to work
unchanged.  The implementation lives in the ``bwd_diff/`` package.
"""

from torch_vulkan.inductor.bwd_diff import (
    _ResolvedBackward,
    dispatch_binary_bwd,
    dispatch_reduction_bwd,
    dispatch_template_bwd,
    dispatch_unary_bwd,
    resolve_backward_kind,
)

__all__ = [
    "dispatch_unary_bwd",
    "dispatch_binary_bwd",
    "dispatch_template_bwd",
    "dispatch_reduction_bwd",
    "resolve_backward_kind",
    "_ResolvedBackward",
]
