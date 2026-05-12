"""Vulkan-specific FX passes for Inductor optimization.

Split into sub-packages per the Track 1 codegen-refactor plan:
  joint_graph/  — passes operating on the fused forward+backward graph
  functional/   — pattern-matching graph rewrites (fusion passes)
  patterns/     — FX pattern registry (populated by Track 4)
  eager_patches.py  — custom_op registrations for fused-pattern targets
  post_grad.py       — post-grad passes (relu->clamp_min, prewarm, debug)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


# Re-export eagerly-registered op factories for backward-compat with
# ``__init__.py`` which imports them at the package level.
# Trigger registration of builtin patterns into FX_PATTERN_REGISTRY.
import torch_vulkan.inductor.fx_passes.patterns.builtin_patterns  # noqa: F401 — side-effect import

from .eager_patches import (
    _ensure_addmm_gelu_op_registered,
    _ensure_flash_attention_op_registered,
    _ensure_qkv_cat_op_registered,
    _ensure_scaled_bmm_op_registered,
    _ensure_swiglu_op_registered,
    register_eager_patch_custom_ops,
)

# Re-export functional passes for tests and external consumers.
from .functional import _fuse_optimizer_step_to_foreach, _fuse_qkv_linears
from .joint_graph import _materialize_implicit_tangents  # noqa: F401
from .patterns.registry import FX_PATTERN_REGISTRY
from .post_grad import (  # noqa: F401 — re-export for tests
    _VULKAN_OP_PREWARM_REGISTRY,
    _aten_target_name,
    prewarm_from_fx_graph,
)


def _make_vulkan_pass() -> object:
    from torch._inductor.custom_graph_pass import (
        CustomGraphModulePass,
        get_hash_for_files,
    )

    _file = __file__

    class _VulkanCustomPass(CustomGraphModulePass):
        """Composite FX pass: uses FxPatternRegistry for fusion patterns,
        with ad-hoc fallbacks for not-yet-registered passes."""

        def __call__(self, gm: "torch.fx.GraphModule") -> None:
            from .functional import (
                _fuse_optimizer_step_to_foreach,
            )
            from .joint_graph import _materialize_implicit_tangents
            from .post_grad import _maybe_dump_fx, prewarm_from_fx_graph

            _maybe_dump_fx(gm, "pre")
            _materialize_implicit_tangents(gm)

            # Track 4.3: registered patterns run in priority order
            # (relu→clamp_min, redundant_copy, plus fusion patterns).
            FX_PATTERN_REGISTRY.apply_all(gm)

            # Remaining ad-hoc passes not yet in registry:
            _fuse_optimizer_step_to_foreach(gm)
            try:
                prewarm_from_fx_graph(gm)
            except Exception:
                pass
            _maybe_dump_fx(gm, "post")

        def uuid(self) -> object:
            return get_hash_for_files((_file,))

    return _VulkanCustomPass()


def vulkan_custom_pass(gm: "torch.fx.GraphModule") -> "torch.fx.GraphModule":
    """Composite FX pass applied to the Inductor graph for Vulkan.

    Uses FxPatternRegistry for registered fusion patterns and ad-hoc
    calls for remaining passes not yet migrated to the registry.
    """
    from .functional import (
        _enable_b2b_gemm,
        _fuse_optimizer_step_to_foreach,
    )
    from .joint_graph import _materialize_implicit_tangents

    gm = _materialize_implicit_tangents(gm)
    FX_PATTERN_REGISTRY.apply_all(gm)
    gm = _fuse_optimizer_step_to_foreach(gm)
    _enable_b2b_gemm(gm)
    return gm
