"""FxPatternRegistry — typed registry of subgraph-to-template rewrite patterns.

T4.3: replaces ad-hoc graph-walking code in ``fx_passes/functional/*.py``
with a declarative registry. Each ``FxPatternEntry`` encodes:

* **match_fn** — yields ``(root_node, context)`` for every matching subgraph.
* **rewrite_fn** — given a match context, replaces the subgraph.
* **template_key_fn** — (optional) extracts a ``TemplateKey`` for dispatch
  via ``template_registry.TEMPLATE_REGISTRY``.

The registry is the single source of truth for which pattern→template
mappings exist.  No ad-hoc graph walks in lowerings or passes should
route through a template without an entry here.

Existing passes in ``functional/`` are preserved — this registry is
additive.  Migration from ad-hoc to registry entries is per-pattern
and tracked via Track 4.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch.fx import GraphModule, Node

    from ...codegen import OpClass
    from ...template_registry import TemplateKey


@dataclass(frozen=True)
class FxPatternEntry:
    """A single subgraph→template rewrite pattern.

    Attributes:
        name:         Human-readable identifier (matches Track 4 naming).
        match_fn:     ``(gm) -> Iterable[Match]`` that yields (root_node, context) tuples.
        rewrite_fn:   ``(gm, root_node, context) -> GraphModule`` — replaces
                      the matched subgraph with the fused node. Must call
                      ``gm.graph.lint()`` and ``gm.recompile()``.
        template_key_fn: Optional callable ``(context) -> TemplateKey | None``
                      to dispatch through ``TEMPLATE_REGISTRY``.
        priority:     Lower runs first (default 100). Within the same priority
                      order is registration order.
    """
    name: str
    match_fn: Callable[["GraphModule"], Iterable[tuple["Node", dict[str, Any]]]]
    rewrite_fn: Callable[["GraphModule", "Node", dict[str, Any]], "GraphModule"]
    template_key_fn: Optional[Callable[[dict[str, Any]], Optional["TemplateKey"]]] = None
    priority: int = 100

    def apply(self, gm: "GraphModule") -> "GraphModule":
        """Apply the pattern once: scan, match, rewrite if found.

        Returns the (possibly modified) graph module.  If no match is found,
        returns the original graph unchanged.

        When ``TORCH_VULKAN_PATTERN_STATS=1`` (M22.3), increments the
        per-name counter on every successful match.
        """
        for root_node, context in self.match_fn(gm):
            gm = self.rewrite_fn(gm, root_node, context)
            gm.graph.lint()
            gm.recompile()
            # M22.3 — count the match at the exact point it fires so we
            # don't need a fragile proxy (node-count delta fails for 1→1
            # rewrites like relu→clamp_min).
            try:
                from torch_vulkan.inductor.config import pattern_stats_enabled
                from torch_vulkan.inductor.fx_passes.post_grad import (
                    record_pattern_fire,
                )
                if pattern_stats_enabled():
                    record_pattern_fire(self.name)
            except ImportError:
                pass
            return gm
        return gm

    def apply_exhaustive(self, gm: "GraphModule") -> "GraphModule":
        """Apply the pattern repeatedly until no more matches are found."""
        changed = True
        while changed:
            prev = gm
            gm = self.apply(gm)
            changed = gm is not prev
        return gm

    def extract_key(self, context: dict[str, Any]) -> Optional["TemplateKey"]:
        """Extract a ``TemplateKey`` from a match context, if this pattern
        dispatches through the template registry."""
        if self.template_key_fn is None:
            return None
        return self.template_key_fn(context)

    def __repr__(self) -> str:
        return f"<FxPatternEntry name={self.name!r} priority={self.priority!r}>"


class FxPatternRegistry:
    """Ordered registry of ``FxPatternEntry`` instances.

    Entries are sorted by priority (lower runs first), then by registration order.
    """

    def __init__(self) -> None:
        self._entries: list[FxPatternEntry] = []

    def register(self, entry: FxPatternEntry) -> None:
        """Register *entry*, maintaining priority order."""
        self._entries.append(entry)
        self._entries.sort(key=lambda e: (e.priority, id(e)))

    @property
    def entries(self) -> list[FxPatternEntry]:
        """Return all registered entries in priority order."""
        return list(self._entries)

    def apply_all(self, gm: "GraphModule") -> "GraphModule":
        """Apply every registered pattern exhaustively to *gm*.

        Patterns are applied in priority order.  After each pattern exhausts
        its matches, the next pattern runs on the rewritten graph.

        Firing-rate counters (M22.3) are incremented inside
        ``FxPatternEntry.apply`` — no extra bookkeeping needed here.
        """
        for entry in self._entries:
            gm = entry.apply_exhaustive(gm)
        return gm

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"<FxPatternRegistry entries={len(self._entries)}>"


# Singleton registry used by Track 4's pattern matcher.
FX_PATTERN_REGISTRY = FxPatternRegistry()


def register_fx_pattern(
    name: str,
    match_fn: Callable[["GraphModule"], Iterable[tuple["Node", dict[str, Any]]]],
    rewrite_fn: Callable[["GraphModule", "Node", dict[str, Any]], "GraphModule"],
    *,
    template_key_fn: Optional[Callable[[dict[str, Any]], Optional["TemplateKey"]]] = None,
    priority: int = 100,
) -> FxPatternEntry:
    """Convenience function to create and register a ``FxPatternEntry``."""
    entry = FxPatternEntry(
        name=name,
        match_fn=match_fn,
        rewrite_fn=rewrite_fn,
        template_key_fn=template_key_fn,
        priority=priority,
    )
    FX_PATTERN_REGISTRY.register(entry)
    return entry
