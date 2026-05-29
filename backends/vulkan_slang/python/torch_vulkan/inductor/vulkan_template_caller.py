"""Vulkan Slang template callers — common registry and re-exports.

Template families have been split into ``templates/caller/*.py`` sub-modules.
This module retains the shared constants, utilities, and re-exports all public
symbols for backward compatibility.
"""

from __future__ import annotations

import os
import re
import struct
from typing import Any, Optional

import torch

from .buffer_pool import pool_acquire, pool_acquire_scratch, pool_release_scratch
from .vulkan_template import _load_slang_template

# ── T4.1: Auto-populated IPointwise / IDifferentiable struct sets ─────────
# M15.3.a: Parsed from ``shaders/lib/pointwise.slang`` at import time.
# Every ``public struct OpX : IPointwise`` (optionally with
# ``, IDifferentiable``) is discovered automatically — no manual sync
# required when adding a new op struct.
#
# CG.M10: The template shader's entry point is ``computeMain<Epilogue : IDifferentiable>``
# where ``Epilogue`` is a Slang type parameter resolved at SPIR-V compile time
# via the ``entry`` parameter of ``compile_and_dispatch``.  The Jinja2 template
# only controls whether the ``Epilogue::apply(...)`` call site is emitted; the
# concrete type name (e.g. ``OpGELU``) is NOT a Jinja2 variable — it is the
# Slang generic type argument.  (Anti-goal #6: no string-based template params.)

_POINTWISE_SLANG_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "shaders", "lib", "pointwise.slang"
    )
)

# Regex: ``public struct OpName : IPointwise[, IDifferentiable]``
_STRUCT_RE = re.compile(r"^public struct (\w+) : (.+)$")


def _parse_pointwise_structs() -> tuple[frozenset[str], frozenset[str]]:
    """Parse ``pointwise.slang`` and return ``(ipointwise, idiff)`` sets.

    Scans for ``public struct OpX : IPointwise`` declarations.
    Structs that also implement ``IDifferentiable`` are added to both sets;
    ``IPointwiseBinary`` / ``IComplexPointwise*`` structs are excluded.
    """
    ipointwise: set[str] = set()
    idifferentiable: set[str] = set()
    try:
        with open(_POINTWISE_SLANG_PATH) as f:
            for line in f:
                m = _STRUCT_RE.match(line.strip())
                if m is None:
                    continue
                name, interfaces = m.group(1), m.group(2)
                if "IPointwiseBinary" in interfaces or "IComplex" in interfaces:
                    continue
                if "IPointwise" not in interfaces:
                    continue
                ipointwise.add(name)
                if "IDifferentiable" in interfaces:
                    idifferentiable.add(name)
    except OSError:
        pass  # Keep empty sets; caller will see empty frozensets.
    return frozenset(ipointwise), frozenset(idifferentiable)


_VALID_IPOINTWISE_STRUCTS, _VALID_IDIFFERENTIABLE_STRUCTS = _parse_pointwise_structs()


def _validate_epilogue_struct(name: str | None) -> str | None:
    """Validate and normalise an epilogue struct name.

    CG.M10: Only ``IDifferentiable`` structs are accepted (the mm template
    uses ``<Epilogue : IDifferentiable>``).  Returns the canonical struct
    name if valid, ``None`` if ``name`` is ``None`` (no epilogue), or raises
    ``ValueError`` if the name is not a known differentiable struct.
    """
    if name is None:
        return None
    if name in _VALID_IDIFFERENTIABLE_STRUCTS:
        return name
    if name in _VALID_IPOINTWISE_STRUCTS:
        raise ValueError(
            f"Epilogue struct '{name}' implements IPointwise but NOT IDifferentiable. "
            f"The mm template requires IDifferentiable for autodiff support. "
        )
    raise ValueError(
        f"Unknown IPointwise epilogue struct '{name}'. "
        f"Must be one of: {sorted(_VALID_IPOINTWISE_STRUCTS)}"
    )


def _dtype_to_slang(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float"
    if dtype == torch.float16:
        return "half"
    if dtype == torch.bfloat16:
        return "uint"
    if dtype == torch.int8:
        return "uint"
    return "float"


_TRUST_INDUCTOR = os.environ.get("TORCH_VULKAN_TRUST_INDUCTOR") == "1"


def _reset_trust_inductor_cache() -> None:
    """Test hook — re-reads `TORCH_VULKAN_TRUST_INDUCTOR` from the environment.
    The flag is captured once at module import for hot-path use; tests that
    flip the env var mid-process can call this to refresh."""
    global _TRUST_INDUCTOR
    _TRUST_INDUCTOR = os.environ.get("TORCH_VULKAN_TRUST_INDUCTOR") == "1"


# ═══════════════════════════════════════════════════════════════════════════
# Re-exports from sub-modules (backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════

from .templates.caller.conv import (
    _render_conv2d_slang,
    _render_conv_bwd_slang,
    _slang_tile_conv2d,
    _slang_tile_conv2d_bwd,
)
from .templates.caller.conv3d import (
    _render_conv3d_slang,
    _render_conv3d_bwd_slang,
    _slang_tile_conv3d,
    _slang_tile_conv3d_bwd,
)
from .templates.caller.fft import (
    _dispatch_fft,
    _render_fft_stockham,
    _SlangTileFFT,
    install_external_fft,
)
from .templates.caller.flash_attn import (
    _dispatch_flash_attention_bwd,
    _make_tile_flash_attention_fn,
    _render_flash_attention,
    _render_flash_attention_bwd,
    _slang_tile_flash_attention,
    _SlangTileFlashAttention,
    _SlangTileFlashAttentionBwd,
    install_external_flash_attention,
)
from .templates.caller.gemm import (
    _check_workgroup_fits,
    _collect_int8_matmul_prewarm_specs,
    _collect_matmul_prewarm_specs,
    _get_device_subgroup_size,
    _make_tile_addmm_fn,
    _make_tile_bmm_fn,
    _make_tile_mm_fn,
    _make_tile_mm_int8_fn,
    _pick_addmm_gelu_tile,
    _pick_register_tile_configs,
    _pick_tile_configs,
    _render_mm_backward_slang,
    _render_mm_bwd_slang,
    _render_mm_int8_slang,
    _render_mm_linktime_wrapper_slang,
    _render_mm_slang,
    _slang_tile_addmm,
    _slang_tile_addmm_gelu,
    _slang_tile_bmm,
    _slang_tile_bmm_bwd,
    _slang_tile_mm,
    _slang_tile_mm_backward,
    _slang_tile_mm_bwd,
    _slang_tile_mm_int8,
    _slang_tiles_enabled,
    _SlangTileAddMM,
    _SlangTileAddMMGelu,
    _SlangTileBMM,
    _SlangTileGEMM,
    _SlangTileMM,
    _SlangTileMMInt8,
    install_external_addmm,
    install_external_bmm,
    install_external_mm,
    install_external_mm_int8,
    prewarm_matmul_templates,
)
from .templates.caller.optimizer import (
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
from .templates.caller.rng import (
    _dispatch_philox_rng,
    _philox_seed_from_torch,
    _render_philox_rng,
    _SlangPhiloxRNG,
    install_external_rng,
)
from .templates.caller.rnn import (
    _can_use_fused_rnn_template,
    _dispatch_rnn_cell,
    _dispatch_rnn_cell_fused,
    _render_rnn_cell,
    _render_rnn_cell_fused,
    _SlangTileRNN,
    _SlangTileRNNFused,
    install_external_rnn,
)
from .templates.caller.scatter import (
    _dispatch_scatter_atomic,
    _render_scatter_atomic,
    install_external_scatter,
)
