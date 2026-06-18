"""Vulkan template kernel infrastructure for Inductor.

Provides `VulkanTemplateKernel` (generates Slang template source via Jinja2)
and `SlangTemplate` (template registration analogous to `TritonTemplate`).

The matmul template lives in `templates/slang_mm.slang` and is parameterized
by tile sizes (TILE_M, TILE_N, TILE_K) with optional epilogue fusion (bias,
relu, gelu, silu, sigmoid, tanh, clamp, scale, dtype cast).

To integrate with Inductor's matmul lowering, register the template choices
via ``V.choices`` in a device-specific override of ``tuned_mm``.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

import torch
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.select_algorithm import KernelTemplate
from torch.utils._ordered_set import OrderedSet

from .kernel import VulkanKernel

_SLANG_TEMPLATES: dict[str, str] = {}

# ── N+1.8: Unwrap .slang-embedded Jinja2 markers ────────────────────────
# Convention:
#   /*{{*/ expr /*}}DEFAULT*/  →  {{ expr }}
#   /*{%*/ stmt /*%}*/         →  {% stmt %}
#   /*# comment #*/            →  {# comment #}
# The Slang-visible DEFAULT value after /*}}*/ makes the raw .slang file
# pass `slangc --syntax-check` with sensible defaults (tile_m=64, tile_k=16,
# m_per_thread=4, etc.). The loader strips the wrapper + default so Jinja2
# sees standard markers.
_SLANG_JINJA_EXPR_RE = re.compile(r"/\*\{\{\*/\s*(.*?)\s*/\*\}\}.*?\*/", re.DOTALL)
_SLANG_JINJA_STMT_RE = re.compile(r"/\*\{\%\*/\s*(.*?)\s*/\*\%\}\*/", re.DOTALL)
_SLANG_JINJA_COMMENT_RE = re.compile(r"/\*#\s*(.*?)\s*#\*/", re.DOTALL)


def _unwrap_slang_template(src: str) -> str:
    """Convert .slang-compatible Jinja2 markers to standard Jinja2 syntax.

    The .slang template wraps every Jinja2 directive in Slang block comments
    so that ``slangc --syntax-check`` can parse the raw file. This function
    reverses that wrapping before Jinja2 rendering.
    """
    # Block statements: /*{%*/ stmt /*%}*/ → {% stmt %}
    src = _SLANG_JINJA_STMT_RE.sub(r"{% \1 %}", src)
    # Inline expressions: /*{{*/ expr /*}}DEFAULT*/ → {{ expr }}
    src = _SLANG_JINJA_EXPR_RE.sub(r"{{ \1 }}", src)
    # Comments: /*# comment #*/ → {# comment #}
    src = _SLANG_JINJA_COMMENT_RE.sub(r"{# \1 #}", src)
    return src


def _load_slang_template(name: str) -> str:
    if name in _SLANG_TEMPLATES:
        return _SLANG_TEMPLATES[name]
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    # N+1.8: prefer `.slang` extension (valid Slang with Jinja2 in comments),
    # fall back to `.py.jinja` (legacy Python-string convention), then bare
    # `.jinja` for older callers.
    src = ""
    loaded_ext = ""
    for fname in (f"{name}.slang", f"{name}.py.jinja", f"{name}.jinja"):
        try:
            with open(os.path.join(template_dir, fname)) as f:
                src = f.read()
            loaded_ext = os.path.splitext(fname)[1]
            break
        except FileNotFoundError:
            continue
    if loaded_ext == ".slang":
        src = _unwrap_slang_template(src)
    _SLANG_TEMPLATES[name] = src
    return src


class VulkanTemplateKernel(VulkanKernel):
    """VulkanKernel subclass that generates Slang template source code.

    Used for matmul and other templates where the shader body comes from
    a Jinja2 template rather than the standard Inductor codegen path.
    """

    def __init__(
        self,
        *args,
        template_name: str = "",
        template_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.template_name = template_name
        self.template_kwargs = template_kwargs or {}

    def codegen_template_body(
        self,
        scheduling,
        template_node,
        epilogue_nodes,
        prologue_nodes,
        buf_name_to_prologue_group,
        prologue_preserves_zero_mask_fn,
        render,
    ) -> str:
        with self:
            if callable(render):
                src = render()
            else:
                src = ""
            from torch._inductor.select_algorithm import PartialRender

            if isinstance(src, PartialRender):
                src = src.finalize_all()
            return src

    def get_unfused_epilogues(self) -> list[Any]:
        return []


def _slang_mm_grid(M, N, K, meta):
    tile_m = meta.get("TILE_M", 64)
    tile_n = meta.get("TILE_N", 64)
    grid_x = (N + tile_n - 1) // tile_n
    grid_y = (M + tile_m - 1) // tile_m
    grid_z = 1
    return (grid_x, grid_y, grid_z)


def _slang_bmm_grid(M, N, K, meta):
    tile_m = meta.get("TILE_M", 64)
    tile_n = meta.get("TILE_N", 64)
    batch = meta.get("BATCH", 1)
    grid_x = (N + tile_n - 1) // tile_n
    grid_y = (M + tile_m - 1) // tile_m
    grid_z = batch
    return (grid_x, grid_y, grid_z)


class SlangTemplate(KernelTemplate):
    """Jinja2-based template for Vulkan Slang kernels.

    Analogous to ``TritonTemplate`` but generates Slang compute shaders
    compiled to SPIR-V instead of Triton PTX.
    """

    kernel_type = VulkanTemplateKernel

    def __init__(self, name: str, grid, source: str):
        super().__init__(name)
        self._grid_fn = grid
        self._source = source
        self._jinja_template = self._template_from_string(source)

    @property
    def uid(self) -> str:
        return f"slang::{self.name}"


# Tile configs: (TILE_M, TILE_N, TILE_K). The matmul template dispatches
# numthreads(TILE_M, TILE_N, 1) — each thread computes one output element in
# the 1-output-per-thread variant. Vulkan's max workgroup invocations on RDNA1
# is 1024, so TILE_M * TILE_N must stay ≤ 1024 or pipeline creation fails at
# dispatch time and the autotune silently falls back to aten_mm.
#
# Larger tiles (e.g. 64×64 = 4096 outputs/group) come through the register-tile
# path (`_MM_REGISTER_TILE_CONFIGS` below): a (WG_M, WG_N) workgroup where each
# thread accumulates an (M_PER_THREAD, N_PER_THREAD) sub-tile in registers.
# WG_M * WG_N must still fit the 1024 cap.
# M17.1: Restricted to single-wave workgroups (<= 64 threads on wave64)
# D1: Two-tier tile-config sets for Vulkan matmul autotune.
# Small default set (fast cold compile, 8 basic + 2 register = 20 variants
# with 2 num_stages each). Use TORCH_VULKAN_MM_TILES=expanded or
# TORCH_VULKAN_MAX_AUTOTUNE=2 to enable the larger sweep.
_MM_TILE_CONFIGS = [
    # Square 8×8 (WG=64): good general-purpose
    (8, 8, 8),
    (8, 8, 16),
    (8, 8, 32),
    (8, 8, 64),
]

# Expanded set: 16 basic tile configs covering square, tall-skinny, and
# short-wide shapes. Enables fine-grain K-tile exploration and aspect-ratio
# optimization. Activate with TORCH_VULKAN_MM_TILES=expanded.
_MM_TILE_CONFIGS_EXPANDED = [
    # Square 8×8 (WG=64): 8 K-tile variants
    (8, 8, 4),
    (8, 8, 8),
    (8, 8, 12),
    (8, 8, 16),
    (8, 8, 24),
    (8, 8, 32),
    (8, 8, 48),
    (8, 8, 64),
    # Tall-skinny 4×16 (WG=64): good for large-M/small-N matmuls
    (4, 16, 8),
    (4, 16, 16),
    (4, 16, 32),
    (4, 16, 64),
    # Short-wide 16×4 (WG=64): good for small-M/large-N matmuls
    (16, 4, 8),
    (16, 4, 16),
    (16, 4, 32),
    (16, 4, 64),
]

# Register-tiled configs: (TILE_M, TILE_N, TILE_K, M_PER_THREAD, N_PER_THREAD).
# WG = (TILE_M/M_PER_THREAD) × (TILE_N/N_PER_THREAD) = 64 on wave64.
# M17.1: Restricted to single-wave workgroups (barrier bug).
_MM_REGISTER_TILE_CONFIGS = [
    # (64, 64) tile: 8×8=64 threads, each holds 8×8=64 outputs
    (64, 64, 16, 8, 8),
]

_MM_REGISTER_TILE_CONFIGS_EXPANDED = [
    # (64, 64) tile: 8×8=64 threads, each holds 8×8=64 outputs  (64 VGPRs/thread)
    (64, 64, 16, 8, 8),
    (64, 64, 32, 8, 8),
    # (32, 32) tile: 8×8=64 threads, each holds 4×4=16 outputs  (16 VGPRs/thread)
    # Better occupancy than 64×64 — more wave slots, smaller tile per WG.
    (32, 32, 16, 4, 4),
    (32, 32, 32, 4, 4),
]

_slang_mm_template: Optional[SlangTemplate] = None
_slang_bmm_template: Optional[SlangTemplate] = None
_slang_conv2d_template: Optional[SlangTemplate] = None
_slang_philox_template: Optional[SlangTemplate] = None
_slang_flash_attention_template: Optional[SlangTemplate] = None
_slang_foreach_optimizer_template: Optional[SlangTemplate] = None


def _get_slang_mm_template():
    global _slang_mm_template
    if _slang_mm_template is not None:
        return _slang_mm_template
    source = _load_slang_template("slang_mm")
    if not source:
        return None
    _slang_mm_template = SlangTemplate(
        name="slang_mm",
        grid=_slang_mm_grid,
        source=source,
    )
    return _slang_mm_template


def _get_slang_bmm_template():
    global _slang_bmm_template
    if _slang_bmm_template is not None:
        return _slang_bmm_template
    source = _load_slang_template("slang_mm")
    if not source:
        return None
    _slang_bmm_template = SlangTemplate(
        name="slang_bmm",
        grid=_slang_bmm_grid,
        source=source,
    )
    return _slang_bmm_template


# ── Path 2: Conv2d template grid ───────────────────────────────────────


def _slang_conv2d_grid(M, N, K, meta):
    """Grid function for conv2d: (ceil(oW/TILE_W), ceil(oH/TILE_H), N*ceil(C_out/TILE_C)).

    The template maps gid.z to (batch, output_channel_tile).
    meta carries: N, C_out, oH, oW, TILE_W, TILE_H, TILE_C.
    """
    TILE_W = meta.get("TILE_W", 8)
    TILE_H = meta.get("TILE_H", 8)
    TILE_C = meta.get("TILE_C", 8)
    N_val = meta.get("N", 1)
    C_out = meta.get("C_out", 1)
    oH = meta.get("oH", 1)
    oW = meta.get("oW", 1)
    grid_x = (oW + TILE_W - 1) // TILE_W
    grid_y = (oH + TILE_H - 1) // TILE_H
    grid_z = N_val * ((C_out + TILE_C - 1) // TILE_C)
    return (grid_x, grid_y, grid_z)


def _get_slang_conv2d_template():
    global _slang_conv2d_template
    if _slang_conv2d_template is not None:
        return _slang_conv2d_template
    source = _load_slang_template("slang_conv2d")
    if not source:
        return None
    _slang_conv2d_template = SlangTemplate(
        name="slang_conv2d",
        grid=_slang_conv2d_grid,
        source=source,
    )
    return _slang_conv2d_template


# ── Track 4: Philox RNG template grid ──────────────────────────────────


def _philox_rng_grid(numel, meta):
    threadgroup_size = 256
    grid_x = (numel + threadgroup_size - 1) // threadgroup_size
    return (grid_x, 1, 1)


def _get_slang_philox_template():
    global _slang_philox_template
    if _slang_philox_template is not None:
        return _slang_philox_template
    source = _load_slang_template("philox_rng")
    if not source:
        return None
    _slang_philox_template = SlangTemplate(
        name="slang_philox_rng",
        grid=_philox_rng_grid,
        source=source,
    )
    return _slang_philox_template


# ── Track 4: Flash attention template grid ──────────────────────────────


def _flash_attention_grid(M, N, K, meta):
    batch = meta.get("BATCH", 1)
    heads = meta.get("HEADS", 1)
    seq_len = meta.get("SEQ_LEN", N)
    wg_size = min(meta.get("HEAD_DIM", 64), 256)
    BQ = 32
    grid_x = batch
    grid_y = heads * ((seq_len + BQ - 1) // BQ)
    grid_z = 1
    return (grid_x, grid_y, grid_z)


def _get_slang_flash_attention_template():
    global _slang_flash_attention_template
    if _slang_flash_attention_template is not None:
        return _slang_flash_attention_template
    source = _load_slang_template("flash_attention")
    if not source:
        return None
    _slang_flash_attention_template = SlangTemplate(
        name="slang_flash_attention",
        grid=_flash_attention_grid,
        source=source,
    )
    return _slang_flash_attention_template


# ── Track 4: Foreach optimizer template grid ────────────────────────────


def _foreach_optimizer_grid(numel, meta):
    batch_size = meta.get("BATCH_SIZE", 7)
    total_params = meta.get("TOTAL_PARAMS", batch_size)
    threadgroup_size = 256
    grid_x = (numel + threadgroup_size - 1) // threadgroup_size
    grid_y = min(total_params, batch_size)
    return (grid_x, grid_y, 1)


def _get_slang_foreach_optimizer_template():
    global _slang_foreach_optimizer_template
    if _slang_foreach_optimizer_template is not None:
        return _slang_foreach_optimizer_template
    source = _load_slang_template("foreach_optimizer")
    if not source:
        return None
    _slang_foreach_optimizer_template = SlangTemplate(
        name="slang_foreach_optimizer",
        grid=_foreach_optimizer_grid,
        source=source,
    )
    return _slang_foreach_optimizer_template
