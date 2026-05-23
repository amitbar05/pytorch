"""M22.2 — IR-level alloc→free→alloc aliasing pass.

Replaces the string-level regex post-processor in ``alloc_alias.py`` with a
structured scan of Inductor's ``self.lines`` list BEFORE text is emitted.
Each entry in ``self.lines`` is a ``MemoryPlanningLine`` subclass that carries
the actual ``BufferLike`` IR node — no regex needed.

The pass runs inside ``VulkanPythonWrapperCodegen.run_wrapper_ir_passes()``,
after Inductor's own ``memory_plan_reuse()`` has already converted obvious
adjacent-step reuse pairs into ``ReuseLine``.  We catch the remaining cases
where two Vulkan transient buffers share the same allocation footprint but
are too far apart (scheduler-node-wise) for Inductor's peak-memory gate to
allow reuse.

Semantics mirror the old regex pass exactly:

    old_buf  allocated, used, freed  (FreeIfNotReusedLine, is_reused=False)
    <gap>
    new_buf  allocated, used, freed  (AllocateLine  →  aliased)

→  After the pass:

    old_buf  allocated, used           (old FreeIfNotReusedLine.is_reused=True → no emit)
    <gap>
    new_buf  = old_buf  # aliased      (AllocateLine replaced by VulkanAliasAllocLine)
    <use new_buf>
    vulkan_pool_release(old_buf, ...)  (new_buf's FreeIfNotReusedLine replaced by redirect)

The Vulkan alias key is ``(dtype, alloc_shape_key, stride_key, lifetime_class,
has_view_suffix)``.  Two buffers are aliasable iff their keys match AND the
old buffer was freed before the new buffer was allocated (non-overlapping
lifetimes in the ``self.lines`` ordering).

The pass is intentionally conservative:
- Subgraph boundary lines (``EnterSubgraphLine`` / ``ExitSubgraphLine``) reset
  the free pool so aliases never cross subgraph boundaries.
- A given old buffer may only alias ONE new buffer (no double-aliasing).
- Graph inputs and outputs are never aliased.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from torch._inductor import ir
from torch._inductor.codegen.wrapper import (
    AllocateLine,
    FreeIfNotReusedLine,
    WrapperLine,
)
from torch._inductor.virtualized import V

from ..wrapper_helpers import _lifetime_class_for_name

if TYPE_CHECKING:
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen
    from torch._inductor.utils import IndentedBuffer


# ---------------------------------------------------------------------------
# Custom line types for the aliased alloc / free redirect
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class VulkanAliasAllocLine(WrapperLine):
    """Replaces an ``AllocateLine`` for a new buffer that aliases an old one.

    Emits: ``new_name = old_name  # aliased: same size/stride/dtype``
    """

    wrapper: PythonWrapperCodegen
    old_node: ir.Buffer
    new_node: ir.Buffer

    def codegen(self, code: IndentedBuffer) -> None:
        old_name = self.old_node.get_name()
        new_name = self.new_node.get_name()
        code.writeline(f"{new_name} = {old_name}  # aliased: same size/stride/dtype")


@dataclasses.dataclass
class VulkanAliasFreeRedirectLine(WrapperLine):
    """Replaces a ``FreeIfNotReusedLine`` for the aliased new buffer.

    Instead of ``vulkan_pool_release(new_name, ...); new_name = None``
    we emit ``vulkan_pool_release(old_name, ...); new_name = None`` so
    the pool eventually recycles the underlying VkBuffer under old_name's
    key (which is the same as new_name's key by construction).
    """

    wrapper: PythonWrapperCodegen
    old_node: ir.Buffer
    new_node: ir.Buffer

    def codegen(self, code: IndentedBuffer) -> None:
        old_name = self.old_node.get_name()
        new_name = self.new_node.get_name()
        lt = _lifetime_class_for_name(old_name)
        code.writeline(
            f"vulkan_pool_release({old_name}, lifetime_class={lt!r}); {new_name} = None"
        )


# ---------------------------------------------------------------------------
# Alias key helpers
# ---------------------------------------------------------------------------


def _alloc_alias_key(node: ir.Buffer) -> tuple | None:
    """Compute a Vulkan alias key for ``node``.

    Returns ``None`` when the node is not aliasable (no Layout, MultiOutput,
    or graph I/O).  Two buffers with identical keys have identical allocation
    footprints and can be aliased with non-overlapping lifetimes.

    The key is:
        (dtype, alloc_size_str, stride_str, lifetime_class, has_view_suffix)

    - ``alloc_size_str``: sympy-simplified storage size as a string
      (same granularity as Inductor's ``buffer_reuse_key``).
    - ``stride_str``: symbolic stride tuple string.
    - ``has_view_suffix``: True when ``allocation_shape != shape``, i.e., when
      the codegen would append ``.as_strided(shape, stride)``.
    - ``lifetime_class``: PF.41 bucket for pool keying.
    """
    try:
        spec = node.get_output_spec()
    except Exception:
        return None
    if not isinstance(spec, ir.Layout):
        return None

    try:
        device = node.get_device()
    except (AttributeError, NotImplementedError):
        device = None
    if device is None or device.type != "vulkan":
        return None

    name = node.get_name()
    # Never alias graph inputs or outputs.
    if name in V.graph.graph_inputs or name in V.graph.get_output_names():
        return None

    try:
        dtype = node.get_dtype()
        shape = tuple(node.get_size())
        stride = tuple(node.get_stride())
        alloc_shape = tuple(V.graph.get_allocation_size(node))
    except Exception:
        return None

    # Compute symbolic keys for comparison.
    try:
        alloc_size_str = str(
            V.graph.sizevars.simplify(V.graph.get_allocation_storage_size(node))
        )
        stride_key = str(
            tuple(str(V.graph.sizevars.simplify(s)) for s in stride)
        )
    except Exception:
        # Fallback: use repr for non-symbolic shapes.
        alloc_size_str = str(alloc_shape)
        stride_key = str(stride)

    has_view_suffix = alloc_shape != shape

    lt = _lifetime_class_for_name(name)

    return (dtype, alloc_size_str, stride_key, lt, has_view_suffix)


# ---------------------------------------------------------------------------
# Main pass entry point
# ---------------------------------------------------------------------------


def apply_vulkan_ir_alias_pass(wrapper: PythonWrapperCodegen) -> None:
    """Scan ``wrapper.lines`` and apply Vulkan-specific alloc aliasing.

    Called from ``VulkanPythonWrapperCodegen.run_wrapper_ir_passes()`` after
    Inductor's own memory planner has run.  Mutates ``wrapper.lines`` in place.
    """
    lines = wrapper.lines

    # --- Phase 1: collect surviving alloc / free events with their indices --
    # We only touch lines that are still AllocateLine / FreeIfNotReusedLine
    # (Inductor's planner converted reusable adjacent pairs to ReuseLine).

    # Map: buf_name → (line_index, AllocateLine)
    alloc_map: dict[str, tuple[int, AllocateLine]] = {}
    # List: [(line_index, FreeIfNotReusedLine)] in order of appearance
    free_events: list[tuple[int, FreeIfNotReusedLine]] = []

    for i, line in enumerate(lines):
        if isinstance(line, AllocateLine):
            node = line.node
            key = _alloc_alias_key(node)
            if key is not None:
                alloc_map[node.get_name()] = (i, line)
        elif isinstance(line, FreeIfNotReusedLine) and not line.is_reused:
            node = line.node
            if isinstance(node.get_output_spec(), ir.Layout):
                name = node.get_name()
                if name not in V.graph.graph_inputs:
                    free_events.append((i, line))

    if len(alloc_map) < 2 or not free_events:
        return  # nothing to alias

    # --- Phase 2: build alloc index ordering (sorted by line index) ---------
    sorted_allocs = sorted(alloc_map.items(), key=lambda kv: kv[1][0])

    # --- Phase 3: find aliasable (old_free, new_alloc) pairs ----------------
    # For each surviving AllocateLine, check whether a freed buffer of the
    # same key appeared earlier in the line list (free_idx < alloc_idx).

    # Track which old buffers are already acting as alias sources.
    already_aliasing: set[str] = set()
    # Result: new_name → (new_alloc_idx, old_node, old_free_line)
    alias_plan: dict[str, tuple[int, ir.Buffer, FreeIfNotReusedLine]] = {}

    for new_name, (new_alloc_idx, new_alloc_line) in sorted_allocs:
        if new_name in alias_plan:
            continue

        new_key = _alloc_alias_key(new_alloc_line.node)
        if new_key is None:
            continue

        best: tuple[str, int, FreeIfNotReusedLine] | None = None  # (old_name, free_idx, free_line)

        for free_idx, free_line in free_events:
            old_node = free_line.node
            old_name = old_node.get_name()

            if old_name == new_name:
                continue
            if old_name in already_aliasing:
                continue
            if free_idx >= new_alloc_idx:
                continue  # old buffer not freed before new allocation

            old_key = _alloc_alias_key(old_node)
            if old_key != new_key:
                continue

            # Prefer the most recently freed candidate.
            if best is None or free_idx > best[1]:
                best = (old_name, free_idx, free_line)

        if best is not None:
            old_name, old_free_idx, old_free_line = best
            old_alloc_info = alloc_map.get(old_name)
            if old_alloc_info is None:
                continue
            _old_alloc_idx, old_alloc_line = old_alloc_info
            alias_plan[new_name] = (new_alloc_idx, old_alloc_line.node, old_free_line)
            already_aliasing.add(old_name)

    if not alias_plan:
        return

    # --- Phase 4: mutate self.lines -----------------------------------------
    # Also need to find and replace the FreeIfNotReusedLine for each new_name.
    # Build a map: buf_name → (line_index, FreeIfNotReusedLine)
    new_free_map: dict[str, tuple[int, FreeIfNotReusedLine]] = {}
    for i, line in enumerate(lines):
        if isinstance(line, FreeIfNotReusedLine) and not line.is_reused:
            name = line.node.get_name()
            if name in alias_plan:
                new_free_map[name] = (i, line)

    for new_name, (new_alloc_idx, old_node, old_free_line) in alias_plan.items():
        # (a) Suppress the old buffer's pool-release.
        old_free_line.is_reused = True  # FreeIfNotReusedLine emits nothing when is_reused=True

        # (b) Replace the new buffer's AllocateLine with a VulkanAliasAllocLine.
        new_alloc_line = lines[new_alloc_idx]
        assert isinstance(new_alloc_line, AllocateLine)
        new_node = new_alloc_line.node
        lines[new_alloc_idx] = VulkanAliasAllocLine(
            wrapper=wrapper,
            old_node=old_node,
            new_node=new_node,
        )

        # (c) Replace the new buffer's FreeIfNotReusedLine with a redirect.
        if new_name in new_free_map:
            new_free_idx, _new_free_line = new_free_map[new_name]
            lines[new_free_idx] = VulkanAliasFreeRedirectLine(
                wrapper=wrapper,
                old_node=old_node,
                new_node=new_node,
            )
