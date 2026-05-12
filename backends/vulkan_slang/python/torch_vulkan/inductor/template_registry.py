"""Template Registry — typed registry indexed by (op_class, dtype, shape_class).

Track 1.5 / Track 4:  Every forward template registered here MUST have a
paired backward template registered through ``bwd_template_registry.py``
(Track 3.8).  The registry is the single source of truth for which
pattern-matched templates are available; no ad-hoc dispatch in
``lowerings/`` or ``fx_passes/`` should route through a template without
an entry here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .codegen import OpClass


@dataclass(frozen=True)
class TemplateKey:
    """Compound key for template dispatch.

    Attributes:
        op_class:    Heavy op-class tag (from ``OpClass`` enum).
        dtype:       PyTorch dtype of the primary input tensor.
        shape_class: Hashable shape descriptor, e.g. ``"square"``,
                     ``"tall_skinny"``, or a frozen tuple of symbolic sizes.
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
