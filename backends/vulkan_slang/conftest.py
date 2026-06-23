"""PF.57 — session-level setup: resolve SLANGC before runtime.py imports.

When a fresh shell forgets to ``export SLANGC=...``, the audit-family
classes (``TestSlangcSmokeAudit``, ``TestPoolReleaseTupleOutput``) silently
degrade to RED — the slangc binary is on disk, just unreachable. This
conftest searches well-known build dirs at session start and exports
SLANGC so subsequent imports of ``torch_vulkan.inductor.runtime`` (which
captures ``_SLANGC`` at module-load time) see a real path.

Search order (first hit wins):
  1. Pre-existing ``SLANGC`` env var — never override an explicit choice.
  2. ``third_party/slang/build/slang-*/bin/slangc`` under repo or backend
     root, ranked by version (newest first → 2026.7.1 over 2026.5.2).
  3. ``/tmp/bin/slangc`` (the CLAUDE.md alt download path).
  4. ``shutil.which("slangc")`` for PATH installs.

Fail-soft: when nothing resolves, leave SLANGC unset so existing
``pytest.skip("slangc unavailable …")`` paths fire as before.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
from typing import Optional

_BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_BACKEND_ROOT, "..", ".."))


def _version_key(path: str) -> tuple[int, ...]:
    m = re.search(r"slang-(\d+)\.(\d+)\.(\d+)", path)
    return tuple(int(g) for g in m.groups()) if m else (0, 0, 0)


def _resolve_slangc() -> Optional[str]:
    candidate_globs = [
        os.path.join(_REPO_ROOT, "third_party/slang/build/slang-*/bin/slangc"),
        os.path.join(_BACKEND_ROOT, "third_party/slang/build/slang-*/bin/slangc"),
    ]
    hits: list[str] = []
    for pattern in candidate_globs:
        hits.extend(glob.glob(pattern))
    hits = [p for p in hits if os.access(p, os.X_OK)]
    hits.sort(key=_version_key, reverse=True)
    if hits:
        return hits[0]
    if os.access("/tmp/bin/slangc", os.X_OK):
        return "/tmp/bin/slangc"
    return shutil.which("slangc")


def pytest_configure(config):
    if not os.environ.get("SLANGC"):
        resolved = _resolve_slangc()
        if resolved:
            os.environ["SLANGC"] = resolved
            config._slangc_resolved = resolved


def pytest_report_header(config):
    val = getattr(config, "_slangc_resolved", None)
    msgs = []
    if val:
        msgs.append(f"SLANGC auto-resolved → {val}")
    elif not os.environ.get("SLANGC"):
        msgs.append("SLANGC unset (audit-family tests will skip)")
    if os.environ.get("TORCH_VULKAN_VUID_AS_ERROR") == "0":
        msgs.append("M-VAL.1: TORCH_VULKAN_VUID_AS_ERROR=0 (VUID-as-error DISABLED)")
    return "\n".join(msgs) if msgs else None


# ── M-VAL.1 (v7) — default-ON VUID-as-error pytest autouse fixture ──
#
# After M-VAL.3 closed the residual best-practices VUID backlog (zero
# VUIDs across 9 catalog models), this fixture is DEFAULT-ON: any
# Vulkan validation error that increments the counter between test
# start and test end fails the test.
#
# Opt-out:
#
#     TORCH_VULKAN_VUID_AS_ERROR=0 pytest tests/test_X.py::test_Y
#
# On boxes without the validation layer, the counter sticks at 0 and
# the fixture is a harmless no-op.
import pytest


@pytest.fixture(autouse=True)
def _mval1_vuid_as_error_fixture(request):
    if os.environ.get("TORCH_VULKAN_VUID_AS_ERROR") == "0":
        yield
        return

    try:
        from torch_vulkan import _c_ext  # type: ignore[attr-defined]
    except Exception:
        # Backend not loaded yet (e.g. unit-test that doesn't import
        # torch_vulkan). Nothing to assert against.
        yield
        return

    if not hasattr(_c_ext, "_validation_errors_count"):
        # C++ pre-M-VAL.1 build — counter not yet pybinded.
        yield
        return

    before = _c_ext._validation_errors_count()
    yield
    after = _c_ext._validation_errors_count()
    delta = after - before
    if delta > 0:
        pytest.fail(
            f"M-VAL.1: {delta} Vulkan VUID(s) emitted during "
            f"{request.node.nodeid} (counter {before} → {after}). "
            f"Inspect stderr for [Vulkan VUID] lines."
        )
