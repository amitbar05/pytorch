"""FX pattern registry — populated by Track 4 (Pattern-Matched Template Dispatch).

Every ``FxPatternEntry`` maps a subgraph signature to a generic template plus a
``key_extractor`` that picks ``(op_class, dtype, shape_class)`` for dispatch.
New pattern entries are registered by dropping a module under this directory.

The registry replaces the ad-hoc graph-walking code in
``fx_passes/functional/*.py``. Each pattern entry provides:

* ``match_fn`` — scans the FX graph and yields ``(root_node, context)`` tuples.
* ``rewrite_fn`` — given a match context, replaces the subgraph with a fused node.
* ``template_fn`` — (optional) template callable for dispatch via ``TemplateRegistry``.

Usage (from ``fx_passes/__init__.py``)::

    from .patterns.registry import FX_PATTERN_REGISTRY

    for pattern in FX_PATTERN_REGISTRY.entries:
        gm = pattern.apply(gm)
"""
from __future__ import annotations

# Re-export registry classes for direct access.
from .registry import (
    FxPatternEntry,
    FxPatternRegistry,
    FX_PATTERN_REGISTRY,
    register_fx_pattern,
)

