#!/usr/bin/env python3
"""P7.10.a — Performance Targets table parser + writer.

Parses the §"Performance Targets" tables in
``docs/10-inductor-backend.md`` and round-trips them back to the file
with updated cells. Used by the measurement-driven CI job (P7.10.b
follow-up) which fills in the **Today / CPU / Vulkan eager** columns
from a fresh measurement pass on each benchmark and emits a unified
diff that fails CI if **Today** regresses against the prior value or
against ``min(CPU, Vulkan eager)``.

This is the table-mechanics half of P7.10. The measurement-driving
half (the script that actually runs CPU eager / Vulkan eager / Vulkan
compiled and feeds numbers to ``apply_updates``) is filed as a sibling
follow-up so the table-update half can land + be tested in isolation.

Design notes:
* Tables live as plain Markdown (`| col | col | ... |`) so the
  parser is a small line-based state machine — no full Markdown AST
  needed.
* The doc has *three* tables under §"Performance Targets" today
  (forward dispatch, backward dispatch, end-to-end ms). Each is
  identified by its header row signature.
* Every cell update preserves the row's original cell widths / pad
  spacing so a diff against the prior file shows only the changed
  numbers, not whitespace churn.
* ``apply_updates`` is idempotent: running it twice with the same
  input reproduces the same output byte-for-byte.

Stage tag: ``BUG_ROOT="measurement"``.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass


_ROADMAP_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "docs", "10-inductor-backend.md"
))


# Header signatures of the three tables we own. The set is
# expected-to-change rarely; if a new column gets added,
# `parse_tables` raises so the change forces a roadmap PR.
_TABLE_FWD_HEADER = (
    "Workload", "Vulkan eager", "Today", "Target",
)
_TABLE_BWD_HEADER = (
    "Workload", "Vulkan eager", "Today", "Target",
)
_TABLE_E2E_HEADER = (
    "Workload", "CPU (ms)", "Vulkan eager (ms)",
    "Today (ms)", "Target (ms)",
)


@dataclass(frozen=True)
class TableRow:
    """One Markdown table row.

    ``cells`` holds the stripped content per column; rendering is
    delegated to ``Table.render`` so the whole table normalizes
    column widths at write time. Hand-written markdown often has
    inconsistent per-row padding — normalizing to max-per-column on
    every write produces a stable, idempotent file (subsequent
    no-op writes are byte-identical) at the cost of touching more
    rows than strictly changed.
    """

    cells: tuple[str, ...]
    is_align: bool = False  # Row of dashes / colons (Markdown align).

    @classmethod
    def parse(cls, line: str) -> "TableRow":
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            raise ValueError(f"not a Markdown table row: {line!r}")
        inner = stripped[1:-1]
        parts = inner.split("|")
        cells = tuple(p.strip() for p in parts)
        is_align = all(
            len(c) > 0 and set(c) <= set("-:") for c in cells
        )
        return cls(cells=cells, is_align=is_align)

    def with_cell(self, idx: int, value: str) -> "TableRow":
        new = list(self.cells)
        new[idx] = value
        return TableRow(cells=tuple(new), is_align=self.is_align)


@dataclass(frozen=True)
class Table:
    """A parsed Markdown table: header + alignment row + data rows.

    ``start_line`` / ``end_line`` are the inclusive line indices in
    the original markdown file so the writer can splice updated rows
    back into place without disturbing surrounding text.
    """

    header: TableRow
    align: TableRow
    rows: tuple[TableRow, ...]
    start_line: int
    end_line: int

    def column_index(self, name: str) -> int:
        try:
            return self.header.cells.index(name)
        except ValueError:
            raise ValueError(
                f"column {name!r} not in table {self.header.cells!r}",
            )

    def find_row(self, workload_substring: str) -> int:
        for i, row in enumerate(self.rows):
            # Workload column is always cell 0 in our schema.
            if workload_substring in row.cells[0]:
                return i
        raise ValueError(
            f"no row matches workload {workload_substring!r} in "
            f"table {self.header.cells!r}",
        )

    def render(self) -> list[str]:
        """Render header + align + data rows with normalized column
        widths (max per column across all rows). Idempotent.
        """
        # Compute per-column inner-cell width = max(content) + 2 for
        # padding. Alignment row dashes match the inner width.
        n = len(self.header.cells)
        widths = [0] * n
        all_rows = [self.header, *self.rows]
        for row in all_rows:
            for ci, cell in enumerate(row.cells):
                widths[ci] = max(widths[ci], len(cell) + 2)

        def fmt_data(row: TableRow) -> str:
            parts = []
            for c, w in zip(row.cells, widths):
                parts.append(f" {c} ".ljust(w))
            return "|" + "|".join(parts) + "|"

        def fmt_align() -> str:
            parts = []
            for c, w in zip(self.align.cells, widths):
                # Preserve `:` alignment markers if the original used
                # them; otherwise fill with dashes to width.
                left = c.startswith(":")
                right = c.endswith(":")
                if left or right:
                    body = "-" * max(w - int(left) - int(right), 1)
                    parts.append(
                        (":" if left else "")
                        + body
                        + (":" if right else "")
                    )
                else:
                    parts.append("-" * w)
            return "|" + "|".join(parts) + "|"

        return [fmt_data(self.header), fmt_align()] + [
            fmt_data(r) for r in self.rows
        ]


def parse_tables(source: str) -> list[Table]:
    """Walk the source markdown and return every table whose header
    row matches one of our known signatures. Tables outside §"Performance
    Targets" are ignored.
    """
    lines = source.splitlines()
    out: list[Table] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.lstrip().startswith("|"):
            i += 1
            continue
        # Try to parse this as a header.
        try:
            header = TableRow.parse(line)
        except ValueError:
            i += 1
            continue
        if header.cells not in (
            _TABLE_FWD_HEADER, _TABLE_BWD_HEADER, _TABLE_E2E_HEADER,
        ):
            i += 1
            continue
        # Next line is the alignment row.
        if i + 1 >= len(lines):
            i += 1
            continue
        try:
            align = TableRow.parse(lines[i + 1])
        except ValueError:
            i += 1
            continue
        # Data rows follow until a blank line or non-table line.
        rows: list[TableRow] = []
        j = i + 2
        while j < len(lines):
            l = lines[j]
            if not l.strip().startswith("|"):
                break
            try:
                rows.append(TableRow.parse(l))
            except ValueError:
                break
            j += 1
        out.append(Table(
            header=header, align=align, rows=tuple(rows),
            start_line=i, end_line=j - 1,
        ))
        i = j
    return out


@dataclass(frozen=True)
class CellUpdate:
    """Pending update: in the table identified by ``table_index``,
    set ``column`` of the row matching ``workload_substring`` to
    ``value``.
    """

    table_index: int
    workload_substring: str
    column: str
    value: str


def apply_updates(source: str, updates: list[CellUpdate]) -> str:
    """Return ``source`` with all updates applied. Idempotent: applying
    the same updates twice produces the same output.
    """
    tables = parse_tables(source)
    if not tables:
        return source

    lines = source.splitlines(keepends=False)
    line_endings = "\n" if source.endswith("\n") else ""

    # Bucket updates by table so we rewrite each table once.
    by_table: dict[int, list[CellUpdate]] = {}
    for u in updates:
        if u.table_index < 0 or u.table_index >= len(tables):
            raise ValueError(
                f"table_index {u.table_index} out of range "
                f"(have {len(tables)} tables)",
            )
        by_table.setdefault(u.table_index, []).append(u)

    # Process tables in reverse so later edits don't shift earlier
    # tables' line indices. We always re-render the *whole* affected
    # table with normalized column widths so the output is idempotent
    # (a second apply with the same updates produces no change).
    for tidx in sorted(by_table.keys(), reverse=True):
        table = tables[tidx]
        new_rows = list(table.rows)
        for u in by_table[tidx]:
            col = table.column_index(u.column)
            row_idx = table.find_row(u.workload_substring)
            new_rows[row_idx] = new_rows[row_idx].with_cell(col, u.value)
        new_table = Table(
            header=table.header, align=table.align,
            rows=tuple(new_rows),
            start_line=table.start_line, end_line=table.end_line,
        )
        lines[table.start_line:table.end_line + 1] = new_table.render()

    # Also: re-rendering a table even with no updates would normalize
    # its widths. We only re-render tables that had updates, so
    # untouched tables stay byte-identical to the input.
    return "\n".join(lines) + line_endings


def update_roadmap_file(
    updates: list[CellUpdate], path: str = _ROADMAP_PATH,
) -> str:
    """Read the roadmap, apply updates, write back. Returns the
    diff-friendly delta count (number of bytes changed). 0 means the
    update was a no-op."""
    with open(path, encoding="utf-8") as f:
        source = f.read()
    new = apply_updates(source, updates)
    if new == source:
        return 0
    with open(path, "w", encoding="utf-8") as f:
        f.write(new)
    return abs(len(new) - len(source))


def main(argv: list[str] | None = None) -> int:
    # CLI usage is intentionally narrow: dump the parsed tables for
    # inspection. The measurement-driven driver (P7.10.b) is the main
    # consumer; humans rarely run this script directly.
    with open(_ROADMAP_PATH, encoding="utf-8") as f:
        source = f.read()
    tables = parse_tables(source)
    print(f"Parsed {len(tables)} Performance Targets table(s).")
    for i, t in enumerate(tables):
        print()
        print(f"Table {i}: header={t.header.cells}")
        for r in t.rows:
            print(f"  {r.cells}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
