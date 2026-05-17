"""M17.7 — Compile-time alloc→free→alloc aliasing pass.

Post-processes the generated Inductor wrapper source code string to
elide redundant ``empty_strided_vulkan`` calls. When two buffers have
the same size/stride/dtype and non-overlapping lifetimes (the first is
pool-released before the second is allocated), we alias the second to
the first and remove the second allocation entirely.

This is a **string-level** post-processing pass that operates on the
assembled wrapper source. It runs after the full codegen is assembled
but before the source is written to disk / compiled.

Transformation::

    buf0 = empty_strided_vulkan((16, 1024), (1024, 1), torch.float32,
                                lifetime_class='transient')
    ... use buf0 ...
    vulkan_pool_release(buf0, lifetime_class='transient'); buf0 = None
    ... unrelated dispatches ...
    buf5 = empty_strided_vulkan((16, 1024), (1024, 1), torch.float32,
                                lifetime_class='transient')
    ... use buf5 ...
    vulkan_pool_release(buf5, lifetime_class='transient'); buf5 = None

→::

    buf0 = empty_strided_vulkan((16, 1024), (1024, 1), torch.float32,
                                lifetime_class='transient')
    ... use buf0 ...
    ... unrelated dispatches ...
    buf5 = buf0  # aliased: same size/stride/dtype as buf0
    ... use buf5 ...
    vulkan_pool_release(buf0, lifetime_class='transient'); buf5 = None

The first buffer's release is removed (it's now kept alive) and the
second buffer's release is rewritten to release the aliased source.
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Regex patterns for parsing generated wrapper source lines
# ---------------------------------------------------------------------------

# Matches the full alloc line including the lifetime_class kwarg and any
# trailing .as_strided() view.
# Captures: (indent, name, size_tuple, stride_tuple, dtype, lifetime_class, rest)
# where `rest` is either empty or ".as_strided(shape_tuple, stride_tuple)"
_FULL_ALLOC_RE = re.compile(
    r"^(\s*)(\w+)\s*=\s*empty_strided_vulkan\("
    r"(\s*\([^)]*\)\s*),\s*"  # size tuple (allocation)
    r"(\s*\([^)]*\)\s*),\s*"  # stride tuple
    r"(\S+?),\s*"  # dtype
    r"lifetime_class=('[^']*')\s*"  # lifetime_class kwarg
    r"\)(.*)$"  # optional .as_strided(...) suffix or empty
)

# Extracts .as_strided(shape, stride) from the rest suffix.
# Captures: (view_shape, view_stride)
_AS_STRIDED_RE = re.compile(r"^\.as_strided\(\s*(\([^)]*\))\s*,\s*(\([^)]*\))\s*\)$")

# Matches: vulkan_pool_release(name, lifetime_class='...'); name = None
# Captures: (indent, name, lifetime_class)
_FREE_RE = re.compile(
    r"^(\s*)vulkan_pool_release\((\w+),\s*lifetime_class=('[^']*')\);\s*\2\s*=\s*None\s*$"
)


def _parse_tuple(tuple_str: str) -> str:
    """Normalize a tuple string for comparison — collapse whitespace."""
    return re.sub(r"\s+", "", tuple_str.strip())


def alias_alloc_free_pairs(source: str) -> str:
    """Post-process wrapper source to elide redundant alloc-free-alloc patterns.

    Args:
        source: The full generated Python wrapper source code as a string.

    Returns:
        The transformed source string with redundant allocations aliased.
    """
    lines = source.splitlines(keepends=True)

    # --- Phase 1: extract alloc and free events --------------------------
    # alloc_data stores all relevant info for matching and replacement.
    allocs: dict[str, tuple[int, str, str, str, str, str, str]] = {}
    #           name -> (line_idx, indent, size, stride, dtype, lt_class, rest)
    frees: list[tuple[int, str, str]] = []
    #           (line_idx, name, lt_class)

    for i, line in enumerate(lines):
        m = _FULL_ALLOC_RE.match(line)
        if m:
            indent, name, size_tup, stride_tup, dtype, lt_class, rest = m.groups()
            # Normalize the rest field: strip whitespace; empty string means
            # no .as_strided() suffix.
            rest_norm = rest.strip() if rest else ""
            allocs[name] = (
                i,
                indent,
                _parse_tuple(size_tup),
                _parse_tuple(stride_tup),
                dtype,
                lt_class,
                rest_norm,
            )
            continue

        m = _FREE_RE.match(line)
        if m:
            indent, name, lt_class = m.groups()
            frees.append((i, name, lt_class))

    if not frees or len(allocs) < 2:
        return source  # nothing to alias

    # --- Phase 2: build lifetime intervals -------------------------------
    # For each buffer, determine its [alloc_line, free_line] interval.
    # (Not strictly needed for the current matching logic which uses
    # line-index ordering directly, but kept for future extension.)
    lifetime: dict[str, tuple[int, int]] = {}
    for name, (alloc_idx, _, _, _, _, _, _) in allocs.items():
        free_idx = None
        for fi, fname, _ in frees:
            if fname == name and fi > alloc_idx:
                free_idx = fi
                break
        if free_idx is not None:
            lifetime[name] = (alloc_idx, free_idx)

    # --- Phase 3: find aliasable pairs -----------------------------------
    # For each freed buffer, find a later allocation with same
    # (size, stride, dtype, rest).  Aliasing rule:
    #   buf0's free < buf5's alloc → non-overlapping lifetimes.
    alias_map: dict[str, str] = {}
    #            new_name → old_name

    # Sort allocs by line number for deterministic processing.
    sorted_allocs = sorted(allocs.items(), key=lambda x: x[1][0])

    for new_name, (
        new_idx,
        new_indent,
        new_size,
        new_stride,
        new_dtype,
        new_lt,
        new_rest,
    ) in sorted_allocs:
        if new_name in alias_map:
            continue  # already aliased

        # Find the best candidate: a buffer that was freed BEFORE this
        # allocation and has matching (size, stride, dtype, rest).
        best_candidate: Optional[tuple[str, int]] = None  # (name, free_line)
        for old_free_idx, old_name, _ in frees:
            if old_name == new_name:
                continue
            if old_name in alias_map.values():
                # This old buffer is already acting as the source for
                # another alias — skip to avoid double-aliasing.
                continue
            old_alloc = allocs.get(old_name)
            if old_alloc is None:
                continue
            (
                _,
                old_indent,
                old_size,
                old_stride,
                old_dtype,
                old_lt,
                old_rest,
            ) = old_alloc

            # Must be freed before new allocation (non-overlapping lifetimes).
            if old_free_idx >= new_idx:
                continue

            # Must have matching allocation params AND matching view suffix.
            # (If old has .as_strided(X) and new has .as_strided(Y) with X≠Y,
            # aliasing buf5 = buf0 would give the wrong view.)
            if (
                old_size != new_size
                or old_stride != new_stride
                or old_dtype != new_dtype
                or old_rest != new_rest
            ):
                continue

            # Prefer the most recently freed candidate (closest to new alloc).
            if best_candidate is None or old_free_idx > best_candidate[1]:
                best_candidate = (old_name, old_free_idx)

        if best_candidate is not None:
            alias_map[new_name] = best_candidate[0]

    if not alias_map:
        return source

    # --- Phase 4: transform the source -----------------------------------
    # We need to:
    #  a) Remove the free line for the aliased-from buffer (buf0's release).
    #  b) Replace the alloc line for the aliased-to buffer with "buf5 = buf0".
    #  c) Rewrite the free line for the aliased-to buffer to release buf0.
    #
    # We work with line indices. Build a set of lines to delete, and
    # a dict of line replacements.

    # Collect which free lines to remove (the source buffer's release).
    free_lines_to_delete: set[int] = set()
    # Collect alloc line replacements: line_idx → (new_indent, new_name, old_name)
    alloc_replacements: dict[int, tuple[str, str, str]] = {}
    # Collect free line replacements: line_idx → new line text
    free_replacements: dict[int, str] = {}

    # Reverse alias map for lookup: old_name → list of new_names
    aliased_from: dict[str, list[str]] = {}
    for new_name, old_name in alias_map.items():
        aliased_from.setdefault(old_name, []).append(new_name)

    for new_name, old_name in alias_map.items():
        old_alloc = allocs[old_name]
        old_free_idx = next((fi for fi, fn, _ in frees if fn == old_name), None)
        new_alloc = allocs[new_name]
        new_free_idx = next((fi for fi, fn, _ in frees if fn == new_name), None)

        if old_free_idx is not None:
            free_lines_to_delete.add(old_free_idx)

        # Replace alloc: buf5 = empty_strided_vulkan(...) → buf5 = buf0
        alloc_replacements[new_alloc[0]] = (new_alloc[1], new_name, old_name)

        # Replace free: vulkan_pool_release(buf5, ...) → vulkan_pool_release(buf0, ...)
        if new_free_idx is not None:
            # Keep the original indent from the free line.
            free_line = lines[new_free_idx]
            fm = _FREE_RE.match(free_line)
            if fm:
                free_indent = fm.group(1)
                free_lt = fm.group(3)
                free_replacements[new_free_idx] = (
                    f"{free_indent}vulkan_pool_release({old_name}, "
                    f"lifetime_class={free_lt}); {new_name} = None\n"
                )

    # Build the output.
    result: list[str] = []
    for i, line in enumerate(lines):
        if i in free_lines_to_delete:
            # Add a comment noting the elision for debugging.
            # Find which buffer this free belonged to.
            freed_name = next((fn for fi, fn, _ in frees if fi == i), "?")
            indent = ""
            fm = _FREE_RE.match(line)
            if fm:
                indent = fm.group(1)
            result.append(
                f"{indent}# (pool-release elided: {freed_name} reused by alias)\n"
            )
            continue

        if i in alloc_replacements:
            indent, new_name, old_name = alloc_replacements[i]
            result.append(
                f"{indent}{new_name} = {old_name}  # aliased: same size/stride/dtype\n"
            )
            continue

        if i in free_replacements:
            result.append(free_replacements[i])
            continue

        result.append(line)

    return "".join(result)
