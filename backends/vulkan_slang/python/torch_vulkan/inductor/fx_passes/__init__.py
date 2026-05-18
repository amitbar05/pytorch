"""Vulkan-specific FX passes for Inductor optimization.

Split into sub-packages per the Track 1 codegen-refactor plan:
  joint_graph/  — passes operating on the fused forward+backward graph
  functional/   — pattern-matching graph rewrites (fusion passes)
  patterns/     — FX pattern registry (populated by Track 4)
  eager_patches.py  — custom_op registrations for fused-pattern targets
  post_grad.py       — post-grad passes (relu->clamp_min, prewarm, debug)
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
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


# ─────────────────────────────────────────────────────────────────────
# M-pipeline-6 (2026-05-18): hash the WHOLE ``fx_passes/`` subtree, not
# just this file. The previous implementation passed ``(_file,)`` to
# ``get_hash_for_files`` — a change in ``fx_passes/eager/conv.py`` (e.g.
# the M-pipeline-1 idempotency fix that closed M18.8.b) or any sibling
# pass module left the pass UUID unchanged, so Inductor's cached
# ``compile_fx`` outputs would continue serving stale lowering results
# until the cache was manually nuked.
#
# Same shape as the M-pipeline-3 ``cfg_key`` invalidation bug — different
# cache key.
# ─────────────────────────────────────────────────────────────────────

_FX_PASSES_DIR: Path = Path(__file__).resolve().parent
_FX_PASS_UUID_DEBUG: bool = os.environ.get("TORCH_VULKAN_FX_PASS_UUID_DEBUG") == "1"


def _enumerate_fx_passes_files() -> tuple[str, ...]:
    """Return every ``*.py`` file under ``fx_passes/`` (recursive), sorted
    by absolute path for deterministic hashing.

    Excludes ``__pycache__/`` and any non-source artefacts. Symlinks are
    resolved so the same logical file under multiple paths hashes once.
    """
    files: set[str] = set()
    for p in _FX_PASSES_DIR.rglob("*.py"):
        # Skip caches / compiled artefacts
        if "__pycache__" in p.parts:
            continue
        try:
            files.add(str(p.resolve()))
        except OSError:
            # Broken symlink etc. — drop silently rather than crash
            # import.
            continue
    return tuple(sorted(files))


def _compute_fx_passes_subtree_uuid() -> bytes:
    """Hash every ``.py`` file under ``fx_passes/`` (recursive) into a
    single SHA-256 digest.

    The digest covers:
      1. the sorted relative-path list (so renames / additions /
         deletions change the digest even if total content is
         unchanged);
      2. each file's full byte content, in path-sorted order.

    Cached at module import via ``_FX_PASSES_SUBTREE_UUID`` below.
    Recomputable on demand via this function for tests that mutate
    files in-memory.
    """
    files = _enumerate_fx_passes_files()

    hasher = hashlib.sha256()
    # Cover the path list so add / delete / rename invalidates.
    # Use paths relative to ``fx_passes/`` so the digest is stable
    # across editable installs at different absolute prefixes.
    rel_paths = tuple(
        str(Path(f).relative_to(_FX_PASSES_DIR)) for f in files
    )
    hasher.update("|".join(rel_paths).encode("utf-8"))

    for path in files:
        try:
            with open(path, "rb") as fh:
                hasher.update(fh.read())
        except OSError:
            # File vanished between rglob and open — note the absence
            # in the digest so callers see a different UUID.
            hasher.update(b"<missing:" + path.encode("utf-8") + b">")

    return hasher.digest()


_FX_PASSES_SUBTREE_UUID: bytes = _compute_fx_passes_subtree_uuid()


if _FX_PASS_UUID_DEBUG:
    _files_at_import = _enumerate_fx_passes_files()
    print(
        f"[TORCH_VULKAN_FX_PASS_UUID_DEBUG] fx_passes subtree UUID = "
        f"{_FX_PASSES_SUBTREE_UUID.hex()[:16]}... "
        f"({len(_files_at_import)} files)",
        file=sys.stderr,
    )
    for _f in _files_at_import:
        try:
            _rel = Path(_f).relative_to(_FX_PASSES_DIR)
        except ValueError:
            _rel = Path(_f)
        print(
            f"[TORCH_VULKAN_FX_PASS_UUID_DEBUG]   - {_rel}", file=sys.stderr
        )


def _make_vulkan_pass() -> object:
    from torch._inductor.custom_graph_pass import (
        CustomGraphModulePass,
    )

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
            # M-pipeline-6: return the whole-fx_passes-subtree digest
            # rather than only this file's hash. The digest is cached
            # at module import in ``_FX_PASSES_SUBTREE_UUID`` so calling
            # ``uuid()`` per compile is O(1) — no rglob / file I/O.
            return _FX_PASSES_SUBTREE_UUID

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
