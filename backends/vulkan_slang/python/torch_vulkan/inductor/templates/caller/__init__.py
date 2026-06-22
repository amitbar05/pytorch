"""Vulkan Slang template callers — sub-modules for each template family.

Each sub-module owns the rendering, dispatch, and installation logic for one
template family.  The parent ``vulkan_template_caller.py`` keeps the shared
registry and re-exports all public symbols for backward compatibility.
"""

from .conv import (
    _render_conv2d_slang,
    _render_conv_bwd_slang,
    _render_conv_gn_relu_slang,
    _slang_tile_conv2d,
    _slang_tile_conv2d_bwd,
    _slang_tile_conv2d_gn_relu,
)
from .fft import (
    _dispatch_fft,
    _render_fft_stockham,
    _SlangTileFFT,
    install_external_fft,
)
from .flash_attn import (
    _dispatch_flash_attention_bwd,
    _make_tile_flash_attention_fn,
    _render_flash_attention,
    _render_flash_attention_bwd,
    _slang_tile_flash_attention,
    _SlangTileFlashAttention,
    _SlangTileFlashAttentionBwd,
    install_external_flash_attention,
)
from .gemm import (
    _check_workgroup_fits,
    _collect_matmul_prewarm_specs,
    _get_device_subgroup_size,
    _make_tile_addmm_fn,
    _make_tile_bmm_fn,
    _make_tile_mm_fn,
    _pick_addmm_gelu_tile,
    _pick_register_tile_configs,
    _pick_tile_configs,
    _render_mm_backward_slang,
    _render_mm_bwd_slang,
    _render_mm_linktime_wrapper_slang,
    _render_mm_slang,
    _slang_tile_addmm,
    _slang_tile_addmm_gelu,
    _slang_tile_bmm,
    _slang_tile_bmm_bwd,
    _slang_tile_mm,
    _slang_tile_mm_backward,
    _slang_tile_mm_bwd,
    _slang_tiles_enabled,
    _SlangTileAddMM,
    _SlangTileAddMMGelu,
    _SlangTileBMM,
    _SlangTileGEMM,
    _SlangTileMM,
    install_external_addmm,
    install_external_bmm,
    install_external_mm,
    prewarm_matmul_templates,
)
from .optimizer import (
    _collect_optimizer_prewarm_specs,
    _foreach_cache_key,
    _foreach_use_parameter_array,
    _pick_foreach_optimizer_caller,
    _render_foreach_optimizer_slang,
    _slang_foreach_optimizer,
    _SlangForeachOptimizer,
    install_external_optimizer,
    prewarm_optimizer_templates,
)
from .rng import (
    _dispatch_philox_rng,
    _philox_seed_from_torch,
    _render_philox_rng,
    _SlangPhiloxRNG,
    install_external_rng,
)
from .rnn import (
    _can_use_fused_rnn_template,
    _dispatch_rnn_cell,
    _dispatch_rnn_cell_fused,
    _render_rnn_cell,
    _render_rnn_cell_fused,
    _SlangTileRNN,
    _SlangTileRNNFused,
    install_external_rnn,
)
from .scatter import (
    _dispatch_scatter_atomic,
    _render_scatter_atomic,
    install_external_scatter,
)
