"""Regression test for the inductor op-coverage audit script.

Locks two M-RT.1 / M-RT.2 invariants in place:

  1. ``scripts/audit_inductor_op_coverage.py`` exits 0 from a clean
     ``backends/vulkan_slang/`` cwd.
  2. The audit's own ``Wrapper-emit import audit`` and ``Slangc smoke
     audit`` sub-summaries each report ``broken = 0`` — i.e. every
     canonical wrapper-codegen import resolves, and every canonical
     slangc smoke snippet compiles clean.

When this test goes red, the audit has caught a real regression (a
renamed Slang symbol, a stale ``import helpers;``, a slangc bug, an
inductor wrapper-codegen rename). Fix the underlying issue — do not
loosen this assertion.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest


# Repo root = backends/vulkan_slang/.
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_AUDIT_SCRIPT = _BACKEND_ROOT / "scripts" / "audit_inductor_op_coverage.py"


@pytest.fixture(scope="module")
def audit_run() -> subprocess.CompletedProcess[str]:
    """Run the audit once per module and reuse the output across asserts."""
    assert _AUDIT_SCRIPT.is_file(), f"audit script not found: {_AUDIT_SCRIPT}"
    return subprocess.run(
        [sys.executable, str(_AUDIT_SCRIPT)],
        cwd=str(_BACKEND_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_audit_exits_zero(audit_run: subprocess.CompletedProcess[str]) -> None:
    """The audit script must exit 0 — non-zero means a hard import/runtime
    error inside the script itself (which is the M-RT.1 / M-RT.2 surface)."""
    assert audit_run.returncode == 0, (
        f"audit script exited {audit_run.returncode}\n"
        f"--- stdout ---\n{audit_run.stdout}\n"
        f"--- stderr ---\n{audit_run.stderr}"
    )


def test_audit_no_broken_lines(audit_run: subprocess.CompletedProcess[str]) -> None:
    """Every ``broken = N`` / ``broken = N`` line in the audit output must
    report N == 0. The audit prints one such line per sub-summary
    (wrapper imports + slangc smokes); any non-zero N means a real
    regression that should fail this test rather than be silently
    accepted.
    """
    out = audit_run.stdout
    # Match `broken<spaces>=<spaces><digits>` to cover both `broken = N`
    # and `broken       = N` (the audit aligns the columns).
    matches = re.findall(r"broken\s*=\s*(\d+)", out)
    assert matches, (
        "audit output did not contain any `broken = N` lines — the audit "
        "format changed and this regression test needs updating.\n"
        f"--- stdout ---\n{out}"
    )
    nonzero = [n for n in matches if int(n) != 0]
    assert not nonzero, (
        f"audit reported broken={nonzero} — a wrapper import or slangc "
        f"smoke regressed.\n--- stdout ---\n{out}"
    )


def test_audit_no_fail_or_error_keywords(
    audit_run: subprocess.CompletedProcess[str],
) -> None:
    """Belt-and-braces: no ``FAIL`` / ``RuntimeError`` / ``Traceback``
    strings should leak into stdout. The script prints a structured
    summary; any of these markers means a sub-check raised through.
    """
    out = audit_run.stdout
    bad_markers = ("Traceback", "RuntimeError", "FAIL:", "ImportError")
    hits = [m for m in bad_markers if m in out]
    assert not hits, (
        f"audit stdout contained failure markers {hits} — investigate.\n"
        f"--- stdout ---\n{out}"
    )
