#!/usr/bin/env python3
"""PF.19 — E2E dispatch coverage CI gate.

Walks ``tests/test_inductor_regression.py`` and reports every test
class that carries a ``_BUG_ROOT_COMPONENT`` constant, classifying
each as either:

* **e2e**: the class contains at least one method whose AST mentions
  ``_get_dispatch_count`` (the ``torch_vulkan._c_ext._get_dispatch_count``
  witness that proves the test path actually runs through real Vulkan
  dispatch);
* **structural**: the class only verifies at the IR / source / AST
  level (no real dispatch witness).

A bug-rooted regression that exists *only* as a structural test is a
verification debt: the test catches the symptom in the IR but won't
catch it in the runtime path. Some classes are legitimately
structural (the bug *is* in the IR layer), so the gate maintains an
explicit ``ALLOWED_STRUCTURAL`` allowlist enumerated below — every
entry must carry a comment explaining why the IR-level test is
sufficient. Anything outside the allowlist that lacks an e2e
dispatch witness fails CI.

This is the §"Bug-Rooting Protocol" companion to PF.12: PF.12 makes
sure every regression names its pipeline stage; PF.19 makes sure
every regression actually exercises that stage at runtime where it
can.
"""
from __future__ import annotations

import ast
import os
import sys
from dataclasses import dataclass


_TESTS_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "tests", "test_inductor_regression.py"
))


# Witness substrings that count as a real-dispatch assertion. Any
# method whose source mentions one of these is treated as an e2e
# verification path.
_E2E_WITNESSES: frozenset[str] = frozenset({
    "_get_dispatch_count",       # canonical Vulkan dispatch counter
    "_get_barrier_skip_count",   # barrier-skip counter (also a dispatch witness)
    "torch.compile(",            # compiles + runs the graph
    "torch_vulkan._c_ext._sync", # explicit sync = ran on device
})


# Test classes that are intentionally structural / IR-only. Each entry
# is a class name → one-line rationale. Adding a class here is a
# governance act — only do so if the bug really lives at the IR /
# source / AST layer where runtime dispatch wouldn't add coverage.
ALLOWED_STRUCTURAL: dict[str, str] = {
    # PF.12 itself — meta-test that audits the audit script. The
    # "stage" being tested is the audit infrastructure, not a runtime
    # path.
    "TestBugRootingComponentAudit":
        "PF.12 meta-audit: the gate IS the test, no runtime path to verify",
    # PF.10.a — slangc -target cpp compile gate. The pipeline stage
    # being verified is shader-source compilability for a *non-Vulkan*
    # target; the SPIR-V dispatch path is verified by other tests.
    "TestSlangCppTargetCompileGate":
        "PF.10.a: cpp-target compile is a non-Vulkan correctness gate",
    # P8.17 — heuristics cost-model skeleton. Pure-Python plan-cost
    # logic; "real dispatch" is unrelated to the seam being tested.
    "TestHeuristicsCostModelSkeleton":
        "P8.17: heuristics-engine plan logic, no dispatch surface",
    # PF.14 — thread-pool deadlock guard. Tests use a stub for
    # `_compile_slang_to_spirv_inner` so they verify the deadlock fix
    # without paying the slangc cold-compile cost.
    "TestNoReentrantSlangcDeadlock":
        "PF.14: structural deadlock-fix test, slangc stubbed for speed",
    # PF.18 — slangc timeout exception class. Verifies error surface;
    # injecting a real hung slangc is impractical.
    "TestSlangcSubprocessTimeoutSurface":
        "PF.18: timeout exception surface; subprocess monkey-patched",
    # PF.15 — cold-compile budget fixture itself.
    "TestSlangcColdCompileBudgetFixture":
        "PF.15: gate infrastructure; verifies the gate fires correctly",
    # PF.2 — Optimizer-step FX pass. Tests verify the FX rewrite
    # (15 add_ → 1 _foreach_add_, 7 addcdiv_ → 1 _foreach_addcdiv_,
    # mixed-alpha non-fuse, single-call non-fuse). End-to-end
    # _foreach_*_ → ComboKernel dispatch verification is gated on
    # PF.13 (C++ view-op fast-path fix) for real optimizer.step()
    # compile correctness. Promote out of allowlist when PF.13 lands.
    "TestOptimizerStepFXPass":
        "PF.2: FX-rewrite unit; e2e dispatch gated on PF.13",
    # PF.19 itself — the audit's own test class. Same pattern as PF.12:
    # the gate IS the test.
    "TestE2EDispatchCoverageAuditScript":
        "PF.19 meta-audit: the gate IS the test",
    # PF.16 — build-time SPV pre-population script. Tests the script's
    # public surface + JSON contract; the script is build-time tooling,
    # not a runtime path that needs a Vulkan dispatch witness.
    "TestPrecompileTestShadersScript":
        "PF.16: build-tooling test, no runtime dispatch surface",
    # P7.10.a — Performance Targets table parser/writer. Tests
    # markdown round-trip + idempotence; the table is a roadmap doc
    # artifact, not a runtime path.
    "TestPerfTargetsTableUpdate":
        "P7.10.a: doc-tool test, no runtime dispatch surface",
    # P8.14 — FX-time pre-warm pass. Tests verify the registry walk
    # + prewarm submit count. The pass is best-effort prewarming
    # ahead of dispatch; the dispatched kernels themselves are
    # exercised by other test classes (PF.4 e2e + activation tests).
    "TestFxTimePrewarmPass":
        "P8.14: FX-pass + registry-walk; submit-count is the contract",
    # PF.13 — C++ view-op fake-storage fix. Tests verify that view ops
    # (expand, permute, select, slice, cat) on FakeTensors return
    # vulkan-tagged null-storage tensors instead of meta-device tensors.
    # By construction the test exercises the FakeTensor path (no real
    # GPU dispatch); the e2e witness for the bw-gradient correctness
    # this unblocks lives in the partitioner regression tests, which
    # already carry _get_dispatch_count witnesses.
    "TestViewOpFakeStorage":
        "PF.13: FakeTensor metadata fix; runtime witness in partitioner suite",
    # PF.50 — MetaGuard convention sweep. Tests verify the C++ guards
    # (binary, unary, mm, sum, eq) return vulkan-tagged null-storage on
    # FakeTensor inputs instead of meta-device. By construction the
    # test exercises the FakeTensor path; the e2e witness for downstream
    # autograd correctness this unblocks lives in the partitioner /
    # backward suites, which already carry _get_dispatch_count witnesses.
    "TestMetaGuardConventionConsistency":
        "PF.50: FakeTensor metadata fix; runtime witness in backward suite",
    # PF.21 — `lib/atomics.slang` shared-module hoist. Tests verify the
    # module file exists, exports the public helpers, and that the three
    # eager atomic shaders import it. The end-to-end correctness witness
    # is the embedding-bwd duplicate-index test, which exercises the CAS
    # path eagerly (no inductor codegen, no dispatch_count). Future
    # PF.22 / PF.25 e2e dispatch witnesses will live in their own classes.
    "TestAtomicsLibModule":
        "PF.21: shader-module hoist; e2e witness in PF.22/PF.25 suites",
    # PF.27.a — Philox RNG helpers-module import alias test. Verifies
    # that the kernel codegen emits `import helpers;` and that the
    # Philox alias is present in helpers.slang. Structural (source
    # inspection), no SPIR-V dispatch.
    "TestPF27aHelpersPhiloxAliases":
        "PF.27.a: source-inspection of helpers import alias; no dispatch path",
    # PF.10.a parity — slangc -target cpp compile gate for the shader
    # library. Same rationale as TestSlangCppTargetCompileGate.
    "TestSlangCppTargetParity":
        "PF.10.a parity: cpp-target compile; no Vulkan dispatch",
    # PF.11 template contract — verifies the GELU epilogue template emits
    # the expected Slang source fragment. Source inspection only.
    "TestSlangMmTemplateGeluEpilogueContract":
        "PF.11: template source-inspection; no dispatch path",
    # PF.11 autodiff — tests the analytic GELU backward derivative formula.
    # Pure math correctness (CPU tensors), no Vulkan dispatch.
    "TestPF11GeluBackwardDerivative":
        "PF.11: analytic bwd formula check; CPU-only correctness test",
    # PF.11 autodiff — tests the analytic SiLU backward derivative formula.
    "TestPF11SiluBackwardDerivative":
        "PF.11: analytic bwd formula check; CPU-only correctness test",
    # PF.11 bwd-diff codegen — verifies the codegen table coverage and
    # source shape. Structural test against the generated table.
    "TestBwdDiffCodegen":
        "PF.11: bwd-diff codegen table structure; no dispatch path",
    # PF.11 coverage gate — verifies that all known backward ops appear
    # in the coverage table. Gate/audit test.
    "TestBwdDiffCoverageGate":
        "PF.11: coverage gate audit; no dispatch path",
    # PF.57 — relative path resolution for `-I` flags in slangc calls.
    # Subprocess test; verifies slangc arg construction.
    "TestSlangcRelativePathResolution":
        "PF.57: slangc -I path construction; subprocess test",
    # PF.11 backward numerics via bwd_diff dispatch — already verified
    # dispatch count in TestActivationBackward; this class verifies
    # numerical correctness only (CPU oracle comparison).
    "TestBwdDiffDispatchNumerics":
        "PF.11: backward numerical correctness vs CPU; dispatch in sibling class",
    # PF.21 hand-written backward shaders — count gate verifying the
    # expected number of bwd shaders. Structural count check only.
    "TestHandWrittenBackwardShaderCount":
        "PF.21: backward shader count gate; structural audit",
    # PF.31.a — AOTI C++ loader surface. Tests loader module import
    # and symbol presence. No runtime dispatch path.
    "TestAotiCppLoader":
        "PF.31.a: AOTI loader surface; no dispatch path",
    # PF.31.a — AOTI AOT compile MLP forward. Verifies AOTI compilation
    # pipeline produces a loadable .so. Structural compile test.
    "TestAotiAotCompileMlpFwd":
        "PF.31.a: AOTI compile pipeline; .so loading is the contract",
    # PF.31.a — AOTI .so load without torch_vulkan in PYTHONPATH.
    # Subprocess isolation test; no runtime dispatch.
    "TestAotiSoLoadsWithoutTorchVulkanPythonpath":
        "PF.31.a: AOTI .so isolation; subprocess test",
    # PF.41 — step-end release hook. Verifies the hook fires and releases
    # gradients. Structural lifecycle test; dispatch count in PF.42 suite.
    "TestStepEndReleaseHook":
        "PF.41: gradient release hook lifecycle; dispatch witness in PF.42",
    # PF.42 — gradient lifetime release on zero_grad. Verifies that
    # gradients are released after zero_grad(). Lifecycle test.
    "TestGradientLifetimeReleaseOnZeroGrad":
        "PF.42: gradient lifetime lifecycle; dispatch witness in pool suite",
    # PF.27.a.1 — slangc module-cache fingerprint invalidation. Tests
    # the cache-key fingerprint logic. Structural + subprocess test.
    "TestSlangcModuleCacheFingerprint":
        "PF.27.a.1: module-cache fingerprint; subprocess/structural test",
    # PF.61 — backward extern coverage CI gate. Verifies all known
    # backward extern ops have lowerings. Gate/audit test.
    "TestBackwardExternCoverageCiGate":
        "PF.61: backward extern coverage gate; audit test",
}


@dataclass(frozen=True)
class CoverageEntry:
    test_class: str
    component: str
    has_e2e_witness: bool
    method_count: int
    line: int


def _scan_class(node: ast.ClassDef, source: str) -> tuple[bool, int]:
    method_count = 0
    has_witness = False
    for stmt in node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_count += 1
            seg = ast.get_source_segment(source, stmt) or ""
            if any(w in seg for w in _E2E_WITNESSES):
                has_witness = True
    return has_witness, method_count


def _extract_entries(path: str) -> list[CoverageEntry]:
    with open(path, encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filename=path)
    entries: list[CoverageEntry] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        component: str | None = None
        comp_line: int = node.lineno
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if not any(
                isinstance(t, ast.Name) and t.id == "_BUG_ROOT_COMPONENT"
                for t in stmt.targets
            ):
                continue
            v = stmt.value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                component = v.value
                comp_line = stmt.lineno
                break
        if component is None:
            continue
        has_e2e, n_methods = _scan_class(node, source)
        entries.append(CoverageEntry(
            test_class=node.name,
            component=component,
            has_e2e_witness=has_e2e,
            method_count=n_methods,
            line=comp_line,
        ))
    return entries


@dataclass(frozen=True)
class AuditResult:
    entries: tuple[CoverageEntry, ...]
    missing_e2e: tuple[CoverageEntry, ...]

    @property
    def is_clean(self) -> bool:
        return len(self.missing_e2e) == 0


def audit_e2e_dispatch_coverage(
    path: str = _TESTS_PATH,
) -> AuditResult:
    entries = tuple(_extract_entries(path))
    missing = tuple(
        e for e in entries
        if not e.has_e2e_witness and e.test_class not in ALLOWED_STRUCTURAL
    )
    return AuditResult(entries=entries, missing_e2e=missing)


def _format_summary(result: AuditResult) -> str:
    lines: list[str] = []
    lines.append(
        f"E2E dispatch coverage audit: {len(result.entries)} bug-rooted "
        f"test class(es).",
    )
    if result.entries:
        lines.append("")
        for e in result.entries:
            tag = "e2e" if e.has_e2e_witness else (
                "struct-allow" if e.test_class in ALLOWED_STRUCTURAL
                else "struct-MISS"
            )
            lines.append(
                f"  [{tag:12s}] {e.test_class:40s} "
                f"component={e.component!r} "
                f"methods={e.method_count} (line {e.line})",
            )
    if result.missing_e2e:
        lines.append("")
        lines.append(
            f"FAIL: {len(result.missing_e2e)} bug-rooted test class(es) "
            f"have no e2e dispatch witness and are not in "
            f"ALLOWED_STRUCTURAL.",
        )
        lines.append(
            "Add a method that calls `_get_dispatch_count()` on a real "
            "compiled graph, or add the class to ALLOWED_STRUCTURAL "
            f"in {os.path.basename(__file__)} with a one-line rationale."
        )
    else:
        lines.append("")
        lines.append(
            "OK: every bug-rooted regression either runs through a "
            "real Vulkan dispatch or is in the structural-allowlist."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    result = audit_e2e_dispatch_coverage()
    print(_format_summary(result))
    return 0 if result.is_clean else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
