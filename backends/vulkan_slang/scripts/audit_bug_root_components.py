#!/usr/bin/env python3
"""PF.12 — Bug-rooting CI artifact.

Walks ``tests/test_inductor_regression.py`` and reports every test class
that carries a ``_BUG_ROOT_COMPONENT`` constant. The constant must name
a stage from the §"Inductor Pipeline Integration Map" in
``docs/10-inductor-backend.md``; an unrecognized stage tag is a CI
failure (a bug fix landed without root-component attribution).

Output is consumable by humans (CLI summary) and by the regression
suite (``audit_bug_root_components()`` returns structured stats so a CI
test can ratchet "every bug-fix carries a recognized stage tag" without
running the script manually).

The ``_BUG_ROOT_COMPONENT`` mechanism is the §"Bug-Rooting Protocol"
exit gate — a regression test for a real-world bug must trace the bug
to an exact pipeline stage before the fix lands.
"""
from __future__ import annotations

import ast
import os
import sys
from dataclasses import dataclass


_TESTS_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "tests", "test_inductor_regression.py"
))


# Stage tags recognized by this audit. Source of truth is PF.59's
# `audit_stage_tags.py:RECOGNIZED_STAGES` (3-bucket: canonical /
# sub_stages / meta_stages); we accept the union. Two registries
# previously diverged on canonical kebab-name forms (PF.12 short-form
# vs PF.59 long-form) and on sub-stage awareness (PF.12 didn't know
# about sub-stages, so e.g. `aoti-runtime` failed here while passing
# the PF.59 audit). PF.59 closeout consolidates: PF.59's registry is
# canonical, this file imports it. Short-form↔long-form canonical
# alignment is a follow-up (PF.59.b).
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from audit_stage_tags import all_recognized as _all_recognized

RECOGNIZED_STAGES: frozenset[str] = frozenset(_all_recognized())


@dataclass(frozen=True)
class BugRootEntry:
    test_class: str
    component: str
    line: int


def _extract_bug_root_entries(path: str) -> list[BugRootEntry]:
    """Walk the test file's AST; return every class-level
    ``_BUG_ROOT_COMPONENT = "<stage>"`` assignment.

    AST-based extraction (rather than a regex) so a string mention of
    ``_BUG_ROOT_COMPONENT`` inside a docstring, comment, or assertion
    message doesn't get counted as a tag.
    """
    with open(path, encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filename=path)
    entries: list[BugRootEntry] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            target_names = [
                t.id for t in stmt.targets if isinstance(t, ast.Name)
            ]
            if "_BUG_ROOT_COMPONENT" not in target_names:
                continue
            value = stmt.value
            if not (isinstance(value, ast.Constant)
                    and isinstance(value.value, str)):
                # Non-string literal — flag as unrecognized; the audit
                # gate will fail.
                entries.append(BugRootEntry(
                    test_class=node.name,
                    component=f"<non-string:{ast.dump(value)}>",
                    line=stmt.lineno,
                ))
                continue
            entries.append(BugRootEntry(
                test_class=node.name,
                component=value.value,
                line=stmt.lineno,
            ))
    return entries


@dataclass(frozen=True)
class AuditResult:
    entries: tuple[BugRootEntry, ...]
    unrecognized: tuple[BugRootEntry, ...]

    @property
    def is_clean(self) -> bool:
        return len(self.unrecognized) == 0


def audit_bug_root_components(
    path: str = _TESTS_PATH,
) -> AuditResult:
    """Programmatic entry point used by the regression suite.

    Returns the full set of entries plus the subset whose component tag
    is not in ``RECOGNIZED_STAGES``. A clean codebase yields
    ``unrecognized == ()``.
    """
    entries = tuple(_extract_bug_root_entries(path))
    unrecognized = tuple(
        e for e in entries if e.component not in RECOGNIZED_STAGES
    )
    return AuditResult(entries=entries, unrecognized=unrecognized)


def _format_summary(result: AuditResult) -> str:
    lines: list[str] = []
    lines.append(
        f"Bug-rooting audit: {len(result.entries)} test class(es) "
        f"carry _BUG_ROOT_COMPONENT.",
    )
    if result.entries:
        lines.append("")
        for e in result.entries:
            mark = "ok " if e.component in RECOGNIZED_STAGES else "BAD"
            lines.append(
                f"  [{mark}] {e.test_class:40s} component={e.component!r} "
                f"(line {e.line})",
            )
    if result.unrecognized:
        lines.append("")
        lines.append(
            f"FAIL: {len(result.unrecognized)} unrecognized stage tag(s). "
            f"Update RECOGNIZED_STAGES in {os.path.basename(__file__)} "
            f"or correct the test's _BUG_ROOT_COMPONENT.",
        )
    else:
        lines.append("")
        lines.append("OK: every _BUG_ROOT_COMPONENT names a recognized stage.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    result = audit_bug_root_components()
    print(_format_summary(result))
    return 0 if result.is_clean else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
