"""Combo-kernel fusion subpackage.

Submodules:
- ``body_rewriter`` — token-based Slang body rewriting (identifier renaming)
- ``binding_map`` — global binding-map construction across subkernels
- ``grid_builder`` — grid dimensions and numthreads computation
"""

from __future__ import annotations

from .binding_map import build_global_binding_map
from .body_rewriter import (
    _KEYWORDS,
    _NEVER_RENAME,
    _TYPE_KEYWORDS,
    _is_buffer_name,
    _is_local_to_rename,
    _rewrite_body,
    _Token,
    _tokenize,
)
from .grid_builder import (
    compute_grid_dims,
    compute_max_threadgroup_size,
    compute_max_workgroups,
)
