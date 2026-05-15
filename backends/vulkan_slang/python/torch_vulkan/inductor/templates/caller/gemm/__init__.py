"""GEMM template callers — re-export package.

This package was split from ``gemm.py`` into sub-modules (M15.1.k anti-goal #7).
Imports from ``templates.caller.gemm`` continue to work unchanged via re-exports.
"""

from .backward import (
    _render_mm_backward_slang,
    _render_mm_bwd_slang,
    _slang_tile_bmm_bwd,
    _slang_tile_mm_backward,
    _slang_tile_mm_bwd,
)
from .classes import (
    _make_tile_addmm_fn,
    _make_tile_bmm_fn,
    _make_tile_mm_fn,
    _make_tile_mm_int8_fn,
    _pick_addmm_gelu_tile,
    _SlangTileAddMM,
    _SlangTileAddMMGelu,
    _SlangTileBMM,
    _SlangTileGEMM,
    _SlangTileMM,
    _SlangTileMMInt8,
)
from .dispatch import (
    _check_workgroup_fits,
    _get_device_subgroup_size,
    _pick_register_tile_configs,
    _pick_tile_configs,
    _slang_tile_addmm,
    _slang_tile_addmm_gelu,
    _slang_tile_bmm,
    _slang_tile_mm,
    _slang_tile_mm_int8,
    _slang_tiles_enabled,
)
from .install import (
    _collect_int8_matmul_prewarm_specs,
    _collect_matmul_prewarm_specs,
    install_external_addmm,
    install_external_bmm,
    install_external_mm,
    install_external_mm_int8,
    prewarm_matmul_templates,
)
from .render import (
    _render_mm_int8_slang,
    _render_mm_linktime_wrapper_slang,
    _render_mm_slang,
)

__all__ = [
    "_check_workgroup_fits",
    "_collect_int8_matmul_prewarm_specs",
    "_collect_matmul_prewarm_specs",
    "_get_device_subgroup_size",
    "_make_tile_addmm_fn",
    "_make_tile_bmm_fn",
    "_make_tile_mm_fn",
    "_make_tile_mm_int8_fn",
    "_pick_addmm_gelu_tile",
    "_pick_register_tile_configs",
    "_pick_tile_configs",
    "_render_mm_backward_slang",
    "_render_mm_bwd_slang",
    "_render_mm_int8_slang",
    "_render_mm_linktime_wrapper_slang",
    "_render_mm_slang",
    "_slang_tile_addmm",
    "_slang_tile_addmm_gelu",
    "_slang_tile_bmm",
    "_slang_tile_bmm_bwd",
    "_slang_tile_mm",
    "_slang_tile_mm_backward",
    "_slang_tile_mm_bwd",
    "_slang_tile_mm_int8",
    "_slang_tiles_enabled",
    "_SlangTileAddMM",
    "_SlangTileAddMMGelu",
    "_SlangTileBMM",
    "_SlangTileGEMM",
    "_SlangTileMM",
    "_SlangTileMMInt8",
    "install_external_addmm",
    "install_external_bmm",
    "install_external_mm",
    "install_external_mm_int8",
    "prewarm_matmul_templates",
]
