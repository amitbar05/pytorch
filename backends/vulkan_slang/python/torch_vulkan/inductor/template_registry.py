"""Template Registry — typed registry indexed by (op_class, dtype, shape_class).

Track 1.5 / Track 4:  Every forward template registered here MUST have a
paired backward template registered through ``bwd_template_registry.py``
(Track 3.8).  The registry is the single source of truth for which
pattern-matched templates are available; no ad-hoc dispatch in
``lowerings/`` or ``fx_passes/`` should route through a template without
an entry here.

C2 (shape bucketing, ROADMAP.md Pillar C):  ``shape_class`` used to be an
opaque caller-supplied object, so two tensors that are interchangeable from
the shader's point of view (same rank/dtype/layout/broadcast pattern, just
different concrete sizes) could land under different keys and trigger a
redundant slangc invocation.  ``canonical_shape_class`` below collapses
``(rank, dtype, layout_class, stride_class)`` into one canonical, hashable
key *before* template selection, so the SPIR-V cache key derived from it
(``cache_key_for``) is shared across same-class shapes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

from .codegen import OpClass


def _layout_class(shape: Sequence[int], strides: Sequence[int] | None) -> str:
    """Classify the memory layout of a tensor given its shape/strides.

    Returns one of ``"contiguous"``, ``"channels_last"``, or ``"strided"``.
    Scalars / 0-d and 1-d tensors are always ``"contiguous"`` (no layout
    ambiguity). When ``strides`` is unavailable, contiguity is assumed —
    callers that care about layout should always pass strides.
    """
    if strides is None or len(shape) <= 1:
        return "contiguous"

    contig = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        contig[i] = contig[i + 1] * max(shape[i + 1], 1)
    if list(strides) == contig:
        return "contiguous"

    if len(shape) == 4:
        n, c, h, w = shape
        cl = [c * h * w, 1, w * c, c]
        if list(strides) == cl:
            return "channels_last"

    return "strided"


def _stride_class(shape: Sequence[int], strides: Sequence[int] | None) -> str:
    """Classify broadcast/degenerate-stride patterns independent of layout.

    Returns ``"broadcast"`` if any non-size-1 dim carries a zero stride
    (expanded tensor), ``"degenerate"`` if every dim has size <= 1, else
    ``"dense"``.
    """
    if all(s <= 1 for s in shape):
        return "degenerate"
    if strides is not None and any(
        st == 0 and sz > 1 for sz, st in zip(shape, strides)
    ):
        return "broadcast"
    return "dense"


def canonical_shape_class(
    shape: Sequence[int],
    dtype: torch.dtype,
    strides: Sequence[int] | None = None,
) -> tuple[int, str, str, str]:
    """Canonicalize ``(rank, dtype, layout_class, stride_class)`` into a
    single hashable key (C2).

    Two tensors that reduce to the same canonical key are guaranteed to be
    served by the same compiled Slang template/SPIR-V module — the shader
    only depends on rank, element type, and layout/broadcast shape, not on
    the concrete sizes. Used both as ``TemplateKey.shape_class`` (template
    *selection*) and to build the slangc ``cache_key`` (SPIR-V *reuse*) via
    ``cache_key_for``.
    """
    rank = len(shape)
    return (
        rank,
        str(dtype),
        _layout_class(shape, strides),
        _stride_class(shape, strides),
    )


def cache_key_for(op_class: OpClass, shape_class: tuple[int, str, str, str]) -> str:
    """Derive a slangc ``cache_key`` string from an op class + canonical
    shape class (C2).

    This is the key handed to ``runtime.slangc.compile_slang_to_spirv``.
    Same canonical key ⇒ same string ⇒ in-memory/disk SPIR-V cache hit
    instead of a second slangc subprocess.
    """
    rank, dtype_s, layout, stride = shape_class
    return f"{op_class.name}_r{rank}_{dtype_s}_{layout}_{stride}"


@dataclass(frozen=True)
class TemplateKey:
    """Compound key for template dispatch.

    Attributes:
        op_class:    Heavy op-class tag (from ``OpClass`` enum).
        dtype:       PyTorch dtype of the primary input tensor.
        shape_class: Hashable shape descriptor, e.g. ``"square"``,
                     ``"tall_skinny"``, or a frozen tuple of symbolic sizes.
                     Prefer ``canonical_shape_class()`` to build this so
                     equivalent shapes collapse onto one key (C2).
    """
    op_class: OpClass
    dtype: torch.dtype
    shape_class: object = field(hash=True)


class TemplateRegistry:
    """Typed registry of generic Slang templates.

    Each entry maps a ``TemplateKey`` to a callable that instantiates
    the template for a specific graph node and returns an Inductor IR
    node (or a list of them for multi-output templates).

    Entry points:
      - ``register`` — add or replace a template callable.
      - ``lookup`` — find a matching template for a graph node.
      - ``keys`` — iterate all registered keys (for audit / coverage).
    """

    def __init__(self) -> None:
        self._entries: dict[TemplateKey, Callable[..., Any]] = {}

    def register(
        self, key: TemplateKey, fn: Callable[..., Any],
    ) -> None:
        """Register *fn* for *key*, overwriting any prior entry."""
        self._entries[key] = fn

    def lookup(self, key: TemplateKey) -> Callable[..., Any] | None:
        """Return the registered callable for *key*, or ``None``."""
        return self._entries.get(key)

    def keys(self):
        """Iterate over all registered ``TemplateKey`` entries."""
        return self._entries.keys()

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"<TemplateRegistry entries={len(self._entries)}>"


# Singleton registry used by Track 4's pattern matcher.
TEMPLATE_REGISTRY = TemplateRegistry()


def register_template(
    op_class: OpClass,
    dtype: torch.dtype,
    shape_class: object,
    fn: Callable[..., Any],
) -> None:
    """Convenience wrapper for single-entry registration."""
    TEMPLATE_REGISTRY.register(
        TemplateKey(op_class=op_class, dtype=dtype, shape_class=shape_class),
        fn,
    )


def lookup_template(
    op_class: OpClass,
    dtype: torch.dtype,
    shape_class: object,
) -> Callable[..., Any] | None:
    """Convenience wrapper for single-entry lookup."""
    return TEMPLATE_REGISTRY.lookup(
        TemplateKey(op_class=op_class, dtype=dtype, shape_class=shape_class),
    )


def lookup_template_bucketed(
    op_class: OpClass,
    shape: Sequence[int],
    dtype: torch.dtype,
    strides: Sequence[int] | None = None,
) -> Callable[..., Any] | None:
    """Look up a template by canonical shape bucket (C2).

    Equivalent to ``lookup_template(op_class, dtype, canonical_shape_class(...))``
    but spelled out so call sites at the lowering boundary (where raw
    shape/stride tuples are available, not yet a hand-picked ``shape_class``)
    don't need to import ``canonical_shape_class`` separately.
    """
    bucket = canonical_shape_class(shape, dtype, strides)
    return lookup_template(op_class, dtype, bucket)


def register_template_bucketed(
    op_class: OpClass,
    shape: Sequence[int],
    dtype: torch.dtype,
    fn: Callable[..., Any],
    strides: Sequence[int] | None = None,
) -> None:
    """Register a template under its canonical shape bucket (C2)."""
    bucket = canonical_shape_class(shape, dtype, strides)
    register_template(op_class, dtype, bucket, fn)
