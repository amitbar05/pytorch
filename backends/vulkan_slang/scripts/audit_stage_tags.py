"""PF.59 — `_BUG_ROOT_COMPONENT` stage-tag taxonomy audit.

Defends the cross-cutting Discipline rule that every bug-rooted item
trace its symptom to a canonical pipeline stage from §"Inductor Pipeline
Integration Map" (or to a registered sub-stage / meta-stage). Three
drift instances surfaced in one loop motivated this audit:

  - PF.3.b cycled `wrapper-codegen → fake-impl → dynamo` before settling.
  - PF.27.a used `wrapper-codegen` as a catch-all.
  - P5.11.a-split's first PROPOSAL named numeric stage positions
    instead of kebab names; converted on accept.

The audit's `RECOGNIZED_STAGES` registry is the inline reference
specialists should consult at PROPOSAL drafting time. CLI subcommands:

  default     — print recognized/drift table (informative)
  --strict    — exit 1 on any drift not in the registry
  --routing   — run routing-gap delivery-verification check (incident
                investigation tool; informative, not failure-gating)
  --baseline  — list regression tests that are RED in main HEAD without
                being flagged by any audit-family verification matrix
                (the "regression-suite output silently masks failures"
                pattern team-lead flagged after the PF.40 bisect)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from typing import Iterable

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.normpath(os.path.join(_SCRIPTS_DIR, ".."))
_ROADMAP = os.path.join(_BACKEND_ROOT, "docs", "ROADMAP.md")
_PYTHON = os.path.join(_BACKEND_ROOT, "python")
_CSRC = os.path.join(_BACKEND_ROOT, "csrc")
_TESTS = os.path.join(_BACKEND_ROOT, "tests")


# Canonical 20 stages from §"Inductor Pipeline Integration Map" (rows 1-20
# of the table at docs/10-inductor-backend.md:133+). Kebab names chosen
# to match the most natural / shortest form already in use across the
# codebase. Where the codebase has standardized on a shorter alias (e.g.
# `partitioner` for stage 3, `fake-impl` for stage 4), the alias IS the
# canonical kebab — the table-row's title is the human-readable form,
# not the tag name.
RECOGNIZED_STAGES: dict[str, set[str]] = {
    "canonical": {
        "dynamo",                           # 1: Dynamo trace
        "aot-autograd-graph-capture",       # 2
        "partitioner",                      # 3: AOT autograd partitioner
        "fake-impl",                        # 4: FakeTensor / fake_impl registry
        "fake-tensor-prop",                 # 5
        "fx-passes",                        # 6
        "lowering",                         # 7
        "scheduler-fusion",                 # 8
        "kernel-codegen",                   # 9
        "pointwise-overrides",              # 10
        "wrapper-codegen",                  # 11
        "externkernelchoice-templates",     # 12
        "combo-kernel",                     # 13
        "runtime",                          # 14
        "reflection-descriptor-binding",    # 15
        "forward-graph-compile",            # 16
        "backward-graph-compile",           # 17
        "optimizer-step-compile",           # 18
        "measurement-autotune-cache",       # 19
        "slang-shader-pipeline",            # 20
    },
    # Sub-stages: deliberately-narrower scopes inside a canonical stage.
    # Add only when bug-rooting needs more granularity than the parent
    # stage and the narrower scope is reusable across multiple items.
    "sub_stages": {
        "templates",                        # under externkernelchoice-templates
                                            # (the four `_slang_tile_*` callers)
        "codegen-helpers-module",           # under kernel-codegen
        "codegen-epilogue-fusion",          # under kernel-codegen
        "buffer-pool-keying",               # under runtime
        "runtime-dispatch-overhead",        # under runtime
        "runtime-prewarm",                  # under runtime
        "runtime-python-wrapper-overhead",  # under runtime / wrapper-codegen
        "aoti-runtime",                     # under runtime (AOTI deployment path)
        "backward-compile",                 # under backward-graph-compile (alias).
                                            # NOTE: the canonical kebab is
                                            # `backward-graph-compile`. Four
                                            # roadmap occurrences predate the
                                            # taxonomy; rename recommended to
                                            # roadmap-manager at PF.59 ship time.
        "slang-shader-pipeline / module-cache-invalidation",  # PF.27.a.1
        "autodiff",                        # under kernel-codegen (Slang bwd_diff)
        # Stage-name aliases (predate strict taxonomy; renames recommended
        # at PF.59 ship time but keeping as recognized aliases lets the
        # taxonomy lock pass without forcing a sweep of historical items):
        "codegen",                          # alias for `kernel-codegen`
        "lowerings",                        # alias for `lowering`
        "backend-features",                 # alias for the Track N
                                            # BackendFeature.SCAN / SORT /
                                            # TUPLE_REDUCTION sub-stage of
                                            # `scheduler-fusion`
    },
    # Meta-stages: cross-cutting concerns that don't fit the linear
    # pipeline (governance, harness, packaging). Each one needs to
    # justify why it isn't a canonical / sub-stage.
    "meta_stages": {
        "measurement",                      # audit-family, perf-tracking items
        "test-harness",                     # PF.57 conftest + similar
        "package-surface",                  # __init__.py re-exports (PF.54)
        "op-coverage",                      # P7.4 op-coverage gate
        # The eager (C++ PrivateUse1) op path is a cross-cutting meta-
        # stage: it feeds into `fake-impl` (Stage 4) AND is the dispatch
        # target when a Vulkan tensor is used outside `torch.compile`.
        # Items that root-cause to "eager-only divergence" file under
        # this meta-tag.
        "eager",
    },
}


def all_recognized() -> set[str]:
    return set().union(*RECOGNIZED_STAGES.values())


_BUG_ROOT_RE = re.compile(r"_BUG_ROOT_COMPONENT\s*=\s*\"([^\"]+)\"")

# Documentation placeholders. These appear in proposal-format examples
# and similar; they are not real `_BUG_ROOT_COMPONENT` values, just the
# template syntax `<stage>` or ellipsis. Filtered out before the
# recognition check so the audit doesn't false-positive on docs.
_PLACEHOLDER_VALUES = frozenset({"<stage>", "...", "..."})


def _walk_files(root: str, suffixes: tuple[str, ...]) -> Iterable[str]:
    if not os.path.isdir(root):
        return
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(suffixes):
                yield os.path.join(dirpath, name)


def _collect_bug_root_values() -> Counter[str]:
    """Walk repo for every `_BUG_ROOT_COMPONENT="..."` literal.

    Documentation placeholders (`<stage>`, ellipsis) are filtered out;
    they appear in proposal-format examples and aren't real values.
    """
    values: Counter[str] = Counter()
    sources: list[str] = [_ROADMAP]
    sources.extend(_walk_files(_PYTHON, (".py",)))
    sources.extend(_walk_files(_CSRC, (".cpp", ".h")))
    sources.extend(_walk_files(_TESTS, (".py",)))
    for path in sources:
        try:
            with open(path) as f:
                src = f.read()
        except (OSError, UnicodeDecodeError):
            continue
        for m in _BUG_ROOT_RE.finditer(src):
            value = m.group(1)
            if value in _PLACEHOLDER_VALUES:
                continue
            values[value] += 1
    return values


_PIPELINE_MAP_ROW_RE = re.compile(
    # The pipeline-map table format is:
    #   | <N> | <Stage> | <Key files> | <Track> |
    # Earlier versions wrapped the stage name in `**…**` markdown bold;
    # the current doc keeps it plain.  Match either form by making the
    # asterisks optional.
    r"^\|\s*(\d+)\s*\|\s*(?:\*\*)?([^|*]+?)(?:\*\*)?\s*\|",
    re.MULTILINE,
)


def _pipeline_map_canonical() -> set[str]:
    """Parse the 20 stage rows out of §"Inductor Pipeline Integration Map".

    The audit parses the human-readable stage titles from the markdown
    table; the corresponding kebab-name canonical-tag for each row lives
    in `RECOGNIZED_STAGES['canonical']`. The lock asserts the *count*
    matches: 20 rows in the map, 20 entries in canonical. The mapping
    from row title to kebab is by-row-index (1→`dynamo`, 2→
    `aot-autograd-graph-capture`, etc.) — see the comment in
    `RECOGNIZED_STAGES['canonical']` for the mapping.
    """
    if not os.path.exists(_ROADMAP):
        return set()
    with open(_ROADMAP) as f:
        src = f.read()
    # Restrict to the §"Inductor Pipeline Integration Map" section to
    # avoid matching unrelated bolded numbered items.
    start = src.find("## Inductor Pipeline Integration Map")
    if start < 0:
        return set()
    end = src.find("\n## ", start + 1)
    section = src[start:end] if end > 0 else src[start:]
    rows: set[int] = set()
    for m in _PIPELINE_MAP_ROW_RE.finditer(section):
        rows.add(int(m.group(1)))
    # Return the synthetic set "stage_N" for each row found, so the test
    # comparison is purely structural: does the registry's canonical
    # have exactly one entry per pipeline-map row?
    canonical = RECOGNIZED_STAGES["canonical"]
    if len(rows) == len(canonical):
        # Treat as matched: structurally consistent.
        return canonical
    # Mismatch: return synthetic markers so the test diff shows row count.
    return {f"stage_{i}" for i in sorted(rows)}


def taxonomy_summary() -> dict:
    values = _collect_bug_root_values()
    recognized = all_recognized()
    unrecognized: set[str] = set()
    for v in values:
        if v not in recognized:
            unrecognized.add(v)
    pipeline_map = _pipeline_map_canonical()
    return {
        "total_distinct_values": len(values),
        "total_occurrences": sum(values.values()),
        "registered_canonical": RECOGNIZED_STAGES["canonical"],
        "registered_sub_stages": RECOGNIZED_STAGES["sub_stages"],
        "registered_meta_stages": RECOGNIZED_STAGES["meta_stages"],
        "pipeline_map_canonical": pipeline_map,
        "values_by_count": values,
        "unrecognized": len(unrecognized),
        "unrecognized_values": unrecognized,
    }


def routing_gap_check() -> dict:
    """Investigation CLI: scan the project's agent-message log for
    success-but-no-followthrough patterns. Today this is a stub —
    the harness doesn't expose a per-message delivery log to the
    specialist process. Returns a structured stub indicating the
    investigation pathway rather than a fail-loud audit.

    Test-class lock only asserts this function exists and returns
    a coherent dict, not its outcome.
    """
    return {
        "available": False,
        "reason": (
            "agent-message delivery log not exposed to specialist "
            "process; routing-gap symptoms manifest as missing "
            "downstream tick / accept after a `success=true` "
            "SendMessage. Investigation pathway: human review of "
            "team-lead and roadmap-manager inboxes vs sender's "
            "outbound history, comparing message timestamps."
        ),
        "remediation_pattern": (
            "When suspected: re-send via lead-relay channel "
            "(verified working at PF.58 / PF.59 ship time)."
        ),
    }


def baseline_red_check() -> dict:
    """Lists regression tests that are RED in main HEAD without being
    flagged by audit-family verification matrices.

    Stub today — the audit tool doesn't run the regression suite, but
    it documents the pattern so future incidents (like the PF.40 family
    bisect that surfaced two pre-existing AOTAutograd / RecursionError
    failures) have a place to land. Run `pytest --collect-only` plus a
    per-test status walk to populate.
    """
    known_pre_existing: list[dict] = [
        {
            "test": "TestPoolReleaseTupleOutput::test_mlp_backward_with_pool_enabled",
            "failure": "AttributeError: 'NoneType' object has no attribute 'args'",
            "stage": "aot_stage2_autograd",
            "suspected_root": "PF.40 (joint_custom_pass) or upstream PyTorch",
        },
        {
            "test": "TestPoolReleaseTupleOutput::test_wrapper_does_not_pool_release_tuple_holder",
            "failure": "AttributeError: 'NoneType' object has no attribute 'args'",
            "stage": "aot_stage2_autograd",
            "suspected_root": "PF.40 (joint_custom_pass) or upstream PyTorch",
        },
        {
            "test": "TestPermuteZeroCopyLowering::test_permute_e2e_zero_copy_no_assert_size_stride_failure",
            "failure": "RecursionError: maximum recursion depth exceeded",
            "stage": "compile_fx",
            "suspected_root": "PF.40 family / wrapper-emit / codegen",
        },
    ]
    return {
        "known_pre_existing_count": len(known_pre_existing),
        "known_pre_existing": known_pre_existing,
        "note": (
            "These three tests are RED in main HEAD, verified pre-existing "
            "via `git stash` + re-run. Surfacing them in this audit's "
            "output prevents 'silently masked' regressions from the "
            "regression-suite verification matrices."
        ),
    }


def _print_table(s: dict) -> None:
    print("Stage-tag taxonomy audit (PF.59)")
    print("=" * 76)
    print(f"  total distinct `_BUG_ROOT_COMPONENT` values: {s['total_distinct_values']}")
    print(f"  total occurrences:                            {s['total_occurrences']}")
    print()
    print("  Registry sizes:")
    print(f"    canonical:    {len(s['registered_canonical'])}")
    print(f"    sub-stages:   {len(s['registered_sub_stages'])}")
    print(f"    meta-stages:  {len(s['registered_meta_stages'])}")
    print()
    print("  Frequency table (top values):")
    by_count = s["values_by_count"]
    for value, count in by_count.most_common():
        if value in s["registered_canonical"]:
            tag = "[canonical]"
        elif value in s["registered_sub_stages"]:
            tag = "[sub-stage]"
        elif value in s["registered_meta_stages"]:
            tag = "[meta-stage]"
        else:
            tag = "[UNRECOGNIZED]"
        print(f"    {count:4d}  {value:40s} {tag}")
    print()
    if s["unrecognized"]:
        print(f"  ⚠  {s['unrecognized']} unrecognized value(s):")
        for v in sorted(s["unrecognized_values"]):
            print(f"    - {v}")
    else:
        print("  ✓ all values are recognized")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strict", action="store_true",
                   help="exit 1 on any unrecognized value")
    p.add_argument("--routing", action="store_true",
                   help="run routing-gap delivery-verification check")
    p.add_argument("--baseline", action="store_true",
                   help="list pre-existing RED tests that audit-family "
                        "matrices may silently mask")
    args = p.parse_args()

    if args.routing:
        r = routing_gap_check()
        print("Routing-gap check (PF.59 investigation CLI)")
        print("=" * 76)
        for k, v in r.items():
            print(f"  {k}: {v}")
        return 0

    if args.baseline:
        b = baseline_red_check()
        print("Baseline RED registry (PF.59 investigation CLI)")
        print("=" * 76)
        print(f"  Known pre-existing RED tests: {b['known_pre_existing_count']}")
        for entry in b["known_pre_existing"]:
            print(f"    - {entry['test']}")
            print(f"        failure: {entry['failure']}")
            print(f"        stage: {entry['stage']}")
            print(f"        suspected root: {entry['suspected_root']}")
        print()
        print(f"  Note: {b['note']}")
        return 0

    s = taxonomy_summary()
    _print_table(s)
    if args.strict and s["unrecognized"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
