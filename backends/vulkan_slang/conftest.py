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
    if val:
        return f"SLANGC auto-resolved → {val}"
    if not os.environ.get("SLANGC"):
        return "SLANGC unset (audit-family tests will skip)"
    return None


