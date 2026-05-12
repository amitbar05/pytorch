"""Functional FX passes — pattern-matching graph rewrites."""
from .matmul import _fuse_mm_add_to_addmm, _fuse_bmm_mul_to_scaled_bmm, _enable_b2b_gemm
from .attention import _fuse_sdpa_to_flash_attention, _fuse_qkv_linears
from .activation import _fuse_addmm_gelu, _fuse_silu_mul_to_swiglu
from .optimizer import _fuse_optimizer_step_to_foreach
from .utilities import _remove_redundant_copy, _topological_resort, _scaled_bmm_extern
