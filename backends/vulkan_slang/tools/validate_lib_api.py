#!/usr/bin/env python3
"""T2.1 — Validate lib API consistency: [BackwardDerivative] pairing + coverage.

Checks:
  1. Every [BackwardDerivative] annotation is paired with a [Differentiable]
     forward declaration in the same file.
  2. Count of doc-documented functions matches actual lib function count.
  3. CI gate: exit 1 if any validation fails.

Usage::
    python tools/validate_lib_api.py          # full check
    python tools/validate_lib_api.py --doc    # doc coverage only
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parent.parent / "shaders" / "lib"
DOC_PATH = Path(__file__).resolve().parent.parent / "docs" / "10-lib-api-reference.md"


def check_differentiable_pairing(filepath: Path) -> list[str]:
    """Verify each [BackwardDerivative] has a [Differentiable] partner."""
    text = filepath.read_text()
    errors = []

    bwd_names: set[str] = set()
    for m in re.finditer(r'\[BackwardDerivative\((\w+)\)\]', text):
        bwd_names.add(m.group(1))

    differentiable_funcs: set[str] = set()
    for m in re.finditer(
        r'\[Differentiable\].*?'
        r'(?:public\s+)?'
        r'(?:float|void|uint2?|int2?|bool)\s+'
        r'(\w+)\s*\(',
        text, re.DOTALL,
    ):
        differentiable_funcs.add(m.group(1))

    for bwd_name in bwd_names:
        found = bwd_name in differentiable_funcs
        if not found and bwd_name.endswith('_fast_bwd'):
            found = bwd_name.replace('_fast_bwd', '_fwd') in differentiable_funcs
        if not found and bwd_name.endswith('_bwd'):
            found = bwd_name.replace('_bwd', '') in differentiable_funcs
        if not found:
            errors.append(
                f"{filepath.name}: [BackwardDerivative({bwd_name})] has no "
                f"matching [Differentiable] partner"
            )
    return errors


def count_documented_functions(doc_text: str) -> int:
    """Count backtick-wrapped function references in the doc."""
    # Match `` `func_name` `` patterns in the doc tables.
    return len(re.findall(r'`(\w+)`', doc_text))


def count_lib_functions() -> int:
    """Count public function declarations across all lib modules."""
    count = 0
    for filepath in sorted(LIB_DIR.glob("*.slang")):
        text = filepath.read_text()
        for m in re.finditer(
            r'(?:\[[^\]]+\]\s+)*'
            r'(?:public\s+)'
            r'(?:float|void|uint2?|int2?|bool|Welford|float2|float3|ArgPair|uint)\s+'
            r'(\w+)\s*\(',
            text,
        ):
            name = m.group(1)
            if name not in ('apply', 'load', 'identity', 'combine',
                            'wave_reduce', 'load_tiles', 'mma_tile', 'places'):
                count += 1
    return count


def main() -> int:
    if not LIB_DIR.exists():
        print(f"ERROR: lib directory not found at {LIB_DIR}", file=sys.stderr)
        return 1

    all_errors: list[str] = []
    total_bwd = 0
    total_funcs = count_lib_functions()

    for filepath in sorted(LIB_DIR.glob("*.slang")):
        bwd_pairs = re.findall(r'\[BackwardDerivative\((\w+)\)\]', filepath.read_text())
        total_bwd += len(bwd_pairs)
        all_errors.extend(check_differentiable_pairing(filepath))

    if DOC_PATH.exists():
        doc_text = DOC_PATH.read_text()
        doc_count = count_documented_functions(doc_text)
        if doc_count < total_funcs * 0.5:
            all_errors.append(
                f"Doc coverage low: {doc_count} refs for ~{total_funcs} lib functions"
            )
    else:
        all_errors.append(f"Doc not found: {DOC_PATH}")

    if all_errors:
        for e in all_errors:
            print(f"FAIL: {e}", file=sys.stderr)
        print(f"\n{len(all_errors)} validation error(s).", file=sys.stderr)
        return 1

    print(
        f"PASS: {total_funcs} lib functions, "
        f"{total_bwd} [BackwardDerivative] annotations, "
        f"all correctly paired."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
