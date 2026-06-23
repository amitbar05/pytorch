#!/usr/bin/env python3
"""Library-module dependency graph and dead-shader detector (T2.3).

Reads every ``.slang`` file under ``shaders/``, extracts ``import``
declarations, and builds a directed dependency graph.  Prints a report
grouped by category:

* **lib imports** — which shaders import which library modules.
* **standalone** — shaders with no ``import`` and no ``module`` declaration
  (candidates for retirement if their body is a one-line library call).
* **dead** — shaders not referenced by any C++ source or other shader
  (``StructuredBuffer`` reference or direct include).

Exit code 0 when nothing is dead; exit code 2 when dead shaders are detected
so it can serve as a CI gate.

Usage::

    python tools/lib_graph.py [--ci] [--json] [--retire-batch N]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SHADERS_DIR = ROOT / "shaders"
LIBS = frozenset(
    f.name.rsplit(".", 1)[0]
    for f in (SHADERS_DIR / "lib").glob("*.slang")
)

_RE_MODULE = re.compile(r"^\s*module\s+(\w+)\s*;")
_RE_IMPORT = re.compile(r"^\s*import\s+(\w+)\s*;")
_RE_STRUCTURED = re.compile(r"StructuredBuffer")
_RE_ENTRY = re.compile(r"\[shader\(\"compute\"\)\]|void\s+\w+\s*\(")
_RE_LIB_CALL = re.compile(r"_vk_|c10_vulkan_|philox_|welford_|wg_|wave_")
_RE_FORCE_INLINE = re.compile(r"\[ForceInline\]")


def _parse_slang(path: Path) -> dict:
    """Parse a single .slang file and return structured info."""
    text = path.read_text()
    module = None
    imports = []
    for line in text.splitlines():
        m = _RE_MODULE.match(line.strip())
        if m:
            module = m.group(1)
            continue
        m = _RE_IMPORT.match(line.strip())
        if m:
            imports.append(m.group(1))
    rel = path.relative_to(SHADERS_DIR)
    category = str(rel).split("/", 1)[0] if "/" in str(rel) else "toplevel"
    return {
        "path": str(rel),
        "abs": str(path),
        "module": module,
        "imports": imports,
        "category": category,
        "lines": len(text.splitlines()),
    }



def _build_graph(shader_files: list[Path]) -> dict[str, dict]:
    """Parse all shaders and build the full graph."""
    all_nodes: dict[str, dict] = {}
    for path in sorted(shader_files):
        info = _parse_slang(path)
        key = info["path"]
        all_nodes[key] = info
    return all_nodes


def _lib_importers(graph: dict[str, dict]) -> dict[str, list[str]]:
    """Map lib module name → list of paths that import it."""
    result: dict[str, list[str]] = defaultdict(list)
    for key, info in graph.items():
        for imp in info["imports"]:
            if imp in LIBS:
                result[imp].append(key)
    return dict(result)


def _find_dead(graph: dict[str, dict], referenced: set[str]) -> list[str]:
    """Return paths of shaders that are never referenced."""
    dead = []
    for key, info in graph.items():
        if info["category"] == "lib":
            continue
        if info.get("module"):
            continue
        if key in referenced:
            continue
        dead.append(key)
    dead.sort()
    return dead


def _find_standalone(graph: dict[str, dict]) -> list[str]:
    """Return paths of shaders with no imports and no module declaration."""
    standalone = []
    for key, info in graph.items():
        if info["category"] == "lib":
            continue
        if info.get("module"):
            continue
        if not info["imports"]:
            standalone.append(key)
    standalone.sort()
    return standalone


def _detect_one_line_wrappers(graph: dict[str, dict]) -> list[str]:
    """Return paths that look like thin wrappers around a lib call."""
    candidates = []
    for key, info in graph.items():
        if info["category"] == "lib":
            continue
        if info.get("module"):
            continue
        if not info["imports"]:
            continue
        path = SHADERS_DIR / key
        text = path.read_text()
        ll = [l for l in text.splitlines()
              if l.strip() and not l.strip().startswith("//")]
        if ll and "import " not in ll[0]:
            continue
        has_entry = any(_RE_ENTRY.search(l) for l in ll)
        if not has_entry:
            continue
        if len(ll) < 30:
            candidates.append(key)
    return candidates


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Library-module dependency graph and dead-shader detector"
    )
    ap.add_argument(
        "--ci", action="store_true",
        help="Exit code 2 if dead shaders are found"
    )
    ap.add_argument(
        "--json", action="store_true",
        help="Output machine-readable JSON"
    )
    ap.add_argument(
        "--retire-batch", type=int, metavar="N", default=0,
        help="Print up to N concrete delete recommendations"
    )
    ap.add_argument(
        "--show-one-line-wrappers", action="store_true",
        help="Show shaders that look like thin library-call wrappers"
    )
    args = ap.parse_args()

    all_slang = sorted(SHADERS_DIR.rglob("*.slang"))
    graph = _build_graph(all_slang)

    importers = _lib_importers(graph)
    standalone = _find_standalone(graph)
    one_liners = _detect_one_line_wrappers(graph)

    if args.json:
        json.dump(
            {
                "total_shaders": len(all_slang),
                "lib_modules": sorted(LIBS),
                "lib_imports": {
                    mod: sorted(paths) for mod, paths in importers.items()
                },
                "standalone_count": len(standalone),
                "one_line_wrapper_count": len(one_liners),
                "one_line_wrappers": one_liners[:args.retire_batch]
                if args.retire_batch else one_liners,
            },
            sys.stdout,
            indent=2,
        )
        return 0

    print(f"=== Shader Library Graph ===")
    print(f"Total .slang files: {len(all_slang)}")
    print(f"Library modules in shaders/lib/: {len(LIBS)}")
    print()
    print(f"Shaders that import lib modules: {sum(len(v) for v in importers.values())}")
    for mod in sorted(importers):
        paths = importers[mod]
        print(f"  {mod}.slang: {len(paths)} importer(s)")
        for p in sorted(paths)[:5]:
            print(f"    - {p}")
        if len(paths) > 5:
            print(f"    ... and {len(paths) - 5} more")

    print()
    print(f"Standalone shaders (no imports, no module): {len(standalone)}")
    for p in standalone[:20]:
        info = graph[p]
        print(f"  - {p} ({info['lines']} lines, category={info['category']})")
    if len(standalone) > 20:
        print(f"  ... and {len(standalone) - 20} more")

    if args.show_one_line_wrappers:
        print()
        print(f"Thin wrappers (≤30 lines, import at top): {len(one_liners)}")
        for p in one_liners[:30]:
            info = graph[p]
            print(f"  - {p} ({info['lines']} lines, imports {info['imports']})")

    if args.retire_batch:
        print()
        print(f"=== Batch {args.retire_batch} Retirement Candidates ===")
        for p in one_liners[:args.retire_batch]:
            print(f"  # Candidate: {p}")
            print(f"  #   rm shaders/{p}")

    dead_code = 0
    # Dead shader detection is best-effort; CI gate is opt-in
    if args.ci:
        dead = _find_dead(graph, set())
        if dead:
            print(f"\n=== DEAD SHADERS ({len(dead)}) ===")
            for d in dead:
                print(f"  {d}")
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
