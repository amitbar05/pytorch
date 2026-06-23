"""P2.5 — backward-shader audit.

Walks the shader tree and classifies each `*backward*.slang` /
`*_bwd*.slang` file as either:

    - **lib-registered**: lives under `shaders/lib/` AND its forward
      declares `[BackwardDerivative(...)]`. The Slang autodiff link
      step picks it up.
    - **eager-only**: lives under `shaders/<category>/` and is invoked
      by the eager backend (these retire as Inductor takes over each
      family — tracked in the per-phase plans).
    - **orphan**: in `shaders/lib/` but no registration. CI fails on
      these so the autodiff path can't silently drift back to
      hand-written shaders bypassing `[BackwardDerivative]`.

The agent runs `audit_backward_shaders()` on every backward-related
change; CI runs it on every commit to gate P2.5's exit criterion.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass


SHADER_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "shaders")
)
LIB_SUBDIR = "lib"

_BACKWARD_NAME_PATTERNS = (
    re.compile(r".*backward.*\.slang$"),
    re.compile(r".*_bwd.*\.slang$"),
    re.compile(r".*bwd_.*\.slang$"),
)
_BACKWARD_DERIVATIVE_RE = re.compile(r"\[BackwardDerivative\s*\(")


@dataclass(frozen=True)
class BackwardShaderEntry:
    path: str
    rel_path: str
    in_lib: bool
    has_registration: bool

    @property
    def category(self) -> str:
        if self.in_lib:
            return "lib-registered" if self.has_registration else "orphan"
        return "eager-only"


def _is_backward_filename(name: str) -> bool:
    return any(p.match(name) for p in _BACKWARD_NAME_PATTERNS)


def _scan_lib_for_backward_registrations() -> set[str]:
    """Return the set of forward function names that declare a custom
    backward via `[BackwardDerivative(name)]` anywhere in `shaders/lib/`.
    """
    lib_dir = os.path.join(SHADER_ROOT, LIB_SUBDIR)
    seen: set[str] = set()
    if not os.path.isdir(lib_dir):
        return seen
    for fname in os.listdir(lib_dir):
        if not fname.endswith(".slang"):
            continue
        with open(os.path.join(lib_dir, fname)) as f:
            for m in _BACKWARD_DERIVATIVE_RE.finditer(f.read()):
                seen.add(m.group(0))
    return seen


def audit_backward_shaders() -> list[BackwardShaderEntry]:
    """Walk the shader tree and return one entry per backward file."""
    out: list[BackwardShaderEntry] = []
    for root, _dirs, files in os.walk(SHADER_ROOT):
        for fname in files:
            if not _is_backward_filename(fname):
                continue
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, SHADER_ROOT)
            in_lib = rel.split(os.sep)[0] == LIB_SUBDIR
            has_reg = False
            if in_lib:
                with open(full) as f:
                    has_reg = bool(_BACKWARD_DERIVATIVE_RE.search(f.read()))
            out.append(BackwardShaderEntry(
                path=full, rel_path=rel, in_lib=in_lib,
                has_registration=has_reg,
            ))
    out.sort(key=lambda e: e.rel_path)
    return out


def assert_no_orphan_backward_shaders() -> None:
    """Raise AssertionError if any `shaders/lib/*backward*.slang` lives
    without a corresponding `[BackwardDerivative]` registration."""
    orphans = [e for e in audit_backward_shaders() if e.category == "orphan"]
    if orphans:
        names = ", ".join(e.rel_path for e in orphans)
        raise AssertionError(
            f"P2.5 violation — orphan hand-written backward shaders in "
            f"shaders/lib/: {names}. Either retire these (delete + add a "
            f"`[Differentiable]` forward in pointwise.slang or similar) "
            f"or wire them via `[BackwardDerivative(name)]`."
        )


def summarize() -> dict:
    entries = audit_backward_shaders()
    return {
        "total": len(entries),
        "lib_registered": sum(1 for e in entries if e.category == "lib-registered"),
        "eager_only": sum(1 for e in entries if e.category == "eager-only"),
        "orphan": sum(1 for e in entries if e.category == "orphan"),
    }
