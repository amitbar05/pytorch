# Vulkan-Slang Inductor Backend — Full Pipeline Review & Overhauled Roadmap v15

> **v15 (2026-06-10)** — This document replaces `docs/14-inductor-backend.md`
> as the active roadmap. v14 is superseded. All v14 milestones are re-scoped
> under v15 pillars with revised effort estimates and precise file:line
> ownership based on an actual code-state audit performed on 2026-06-10.

---

# § 1 — Pipeline Review: End-to-End State

## 1.1 The compile pipeline (what actually runs)

```
torch.compile(model, backend="inductor", mode="max-autotune")
  │
  ├─ Dynamo FX capture
  │     └─ vulkan: fx_passes/eager/* (fused patterns pre-graph-break)
  │
  ├─ AOTAutograd (joint fwd+bwd graph)
  │     ├─ meta_patches/shape_ops.py          ← FakeTensor shape inference
  │     ├─ meta_patches/decomposition_passes.py ← decompose aten → vulkan-primitive
  │     └─ meta_patches/autograd_registrations.py
  │
  ├─ Post-grad passes
  │     ├─ fx_passes/post_grad.py             ← pattern matcher fusions
  │     ├─ fx_passes/patterns/*               ← conv+GN, mm+gelu, SDPA
  │     └─ fx_passes/alloc_alias.py           ← alloc elimination
  │
  ├─ GraphLowering + Scheduler
  │     ├─ kernel/template_registry.py        ← selects Slang templates
  │     ├─ scheduling.py                      ← fused vs. split decision
  │     └─ buffer_pool.py                     ← memory planning
  │
  ├─ Codegen
  │     ├─ codegen.py                         ← IR → Python wrapper src
  │     ├─ templates/caller/*.py              ← Per-kernel caller + args pack
  │     ├─ templates/*.slang                  ← Slang shader source
  │     └─ runtime/slangc.py                  ← slangc → SPIR-V
  │
  ├─ C++ wrapper compile (AOTI path)
  │     ├─ cpp_wrapper_gpu.py                 ← VulkanCppWrapperGpu
  │     ├─ csrc/aoti_runner_vulkan.cpp        ← AOTIModelContainerRunner
  │     ├─ csrc/aoti_shims.cpp                ← aoti_torch_*_vulkan symbols
  │     └─ csrc/aoti_preload.cpp              ← preload library (LD_PRELOAD)
  │
  └─ CompiledFxGraph / CompiledAOTI dispatch
        └─ runtime/dispatch.py + runtime/batcher.py
```

## 1.2 Verified working

| Layer | Status | Evidence |
|-------|--------|---------|
| Dynamo FX capture (Conv+GN+ReLU) | ✅ | `test_conv_gn_relu_fused_forward` PASS |
| AOTAutograd joint graph (5-step train) | ✅ | commit `675316e` — loss 1.511→1.500 |
| Post-grad combo fusion (fwd) | ✅ | `test_conv_gn_relu_fused_forward` PASS |
| Pointwise lowering + fusion | ✅ | `test_binary_ops_match_torch_eager` PASS |
| Slang conv2d (forward) | ✅ | `test_conv_gn_relu_slang_no_has_bias_jinja` PASS |
| Slang reductions (45 elementals) | ✅ | `test_bwd_diff_*` suite PASS |
| bwd_diff table (fwd→bwd) | ✅ | `test_conv_gn_relu_compile_backward_matches_cpu` PASS |
| SPIR-V codegen (slangc) | ✅ | md5sum verification in CI |
| Pre-slangc AST validator | ✅ | M-VAL.4 closed in v7 |
| VMA write-path (notify_host_write) | ✅ | `test_vma_regression.cpp` PASS |
| ThreadPoolExecutor atexit fix | ✅ | commit `916712e` |

## 1.3 Gaps (what's missing or broken)

### Gap 1 — AOTI wrapper does NOT link Vulkan runtime (AOTI.1)
**Current state**: The generated `.so` in `/tmp/torchinductor_amit/` links
`libtorch.so` + `libtorch_cpu.so` only. It does **not** link against the
`aoti_torch` Vulkan symbols or preload `aoti_preload.so`.

**What happens**: When `CompiledAOTI` loads the `.so` and calls
`aoti_torch_empty_strided_vulkan`, the symbol is unresolved. Either:
- `LD_PRELOAD=aoti_preload.so` is set at process level (not at `.so` load),
  and RTLD_LOCAL prevents the preload symbols from being visible to the
  child `dlopen`; or
- The `.so` link line omits `aoti_shims.o` / `aoti_preload.so` entirely.

**Missing piece**: `cpp_wrapper_gpu.py` (or the build step in
`compile_graph.py`) must emit a link line that includes the Vulkan AOTI
shim object file or preload library.

**File ownership**: `python/torch_vulkan/inductor/cpp_wrapper_gpu.py` (emit
`-Wl,--whole-archive` or equivalent), OR `python/torch_vulkan/inductor/compile_graph.py`
(write the link invocation).

### Gap 2 — AOTI run not verified on RDNA1 (AOTI.2)
**Blocker**: AOTI.1 must close first. Once linked, the wrapper must be
exercised with a Conv+GN fwd+bwd run and gradient values compared to CPU.

**Gradient correctness**: Conv weight, conv bias, GN weight, GN bias, and
input gradient all must be < 1e-4 from CPU eager baseline.

**File ownership**: New regression test in `tests/test_inductor_regression.py`
under class `TestAOTI_ConvGnTraining`.

### Gap 3 — atexit hang after AOTI runs (AOTI.3)
**Current state**: `timeout 120` → RC=124. The `ThreadPoolExecutor` fix
(commit `916712e`) addresses `_register()` import-time hang, but the
atexit teardown chain after a training run still blocks:
```
cleanup_runtimes() → DeviceRuntime::~DeviceRuntime() → vkDeviceWaitIdle()
  → async compile/slangc thread pools still alive
```

**Root cause**: `DeviceRuntime` destructor is synchronous. Any kernel
still queued (or any slangc subprocess still running) holds `vkDeviceWaitIdle`
indefinitely.

**Fix direction**: Replace the synchronous destructor drain with
non-blocking `shutdown(wait=False)` on all executors, plus a `vkQueueWaitIdle`
with timeout fallback. The `Context::shutdown` patch mentioned in v14 exists
but is incomplete — the teardown order needs to be: drain executors first,
then destroy DeviceRuntime.

**File ownership**: `csrc/backend/DeviceRuntime.cpp` (or wherever the
singleton lives), `python/torch_vulkan/inductor/runtime/common.py`
(pool shutdown ordering).

### Gap 4 — _register_device_module hang (AOTI.4)
**Current state**: `torch._register_device_module("vulkan", ...)` hangs
indefinitely. Binary search isolated to imports in
`meta_patches/shape_ops.py` and `meta_patches/decomposition_passes.py`.

**Mechanism**: At import time, these modules call `torch.ops.aten.*` to
register `fake_impl` decorators. During `torch._register_device_module`,
PyTorch is in the middle of setting up the dispatch/c10 dispatcher.
Calling `torch.ops` before registration completes triggers a deadlock on
`FakeTensorMode` initialization.

**Current mitigations**: `override getDeviceFromPtr` (commit `0ff39e43e`)
and container RAII (commit `7eaa80ce`) address runtime device-pointer
resolution, not the import-time deadlock.

**Fix direction**: Defer `fake_impl` registration to first-use, or guard
the `torch.ops.aten` calls with a `_register_complete` flag. The minimal
patch is a lazy-init wrapper around `meta_patches/__init__.py`.

**File ownership**: `python/torch_vulkan/inductor/meta_patches/__init__.py`,
`python/torch_vulkan/inductor/meta_patches/shape_ops.py`,
`python/torch_vulkan/inductor/meta_patches/decomposition_passes.py`.

### Gap 5 — SYNC.1: Batch dispatch still serial (SYNC.1)
**Current state**: `BATCH_DISPATCH=1` is 1.8× slower than `BATCH_DISPATCH=0`
(676 ms vs 385 ms for MNISTNet). Batch mode exits on first flush rather than
accumulating N kernels.

**Root cause**: The current batcher flushes whenever a kernel's output is
needed by the next node, rather than accumulating a full buffer of
independent kernels and flushing once.

**Fix direction**: Change the flush condition from "any dependency" to
"all dependencies in the current batch are resolved". This requires
tracking a ready-set per batch, not per kernel.

**File ownership**: `python/torch_vulkan/inductor/runtime/batcher.py`.

### Gap 6 — SYNC.2: VMA write-path regression (SYNC.2)
**Current state**: The `notify_host_write` fix (commit `2d438e4`) is in
`Allocator::allocate()`, but the recent MAPPED_BIT removal + unmap changes
(commits `fea31a5`, `a345ea7`) mean the full multi-kernel training path
has not been re-verified on RDNA1. The concern is: buffer is allocated
(write), then dispatched (read), then written again — does the HOST→COMPUTE
barrier fire correctly at each transition?

**File ownership**: `csrc/backend/Allocator.cpp`, `csrc/backend/VulkanBuffer.cpp`.

### Gap 7 — SLANG.1: Jinja `has_bias` still in slang_conv2d.slang (SLANG.1)
**Current state**: `slang_conv2d.slang` still has `#ifdef HAS_BIAS`
preprocessor branches. The Jinja caller (`templates/caller/conv.py`) passes
`HAS_BIAS` as a string-substituted define. M-SF.4 in v7 claimed this was
eliminated, but it was re-introduced or never fully migrated.

**Fix direction**: Replace `#ifdef HAS_BIAS` with a Slang `interface
IBias { bool has_bias; }` generic parameter on the kernel entry point.
The caller's `ParameterBlock` already has the `has_bias` field — wire it
to the interface.

**File ownership**: `python/torch_vulkan/inductor/templates/slang_conv2d.slang`,
`python/torch_vulkan/inductor/templates/caller/conv.py`.

### Gap 8 — SLANG.2: tril/triu/masked_fill/where missing from bwd_diff (SLANG.2)
**Current state**: `bwd_diff_table.py` + `bwd_diff/unary.py` + `bwd_diff/binary.py`
cover 45 elementals. `tril`, `triu`, `masked_fill`, and `where` (masked
select) are absent. These block sparse-attention and padding-mask models.

**Fix direction**: Add `[BackwardDerivative]` annotations in
`vk_elemental.slang` (or equivalent), register in `bwd_diff_table.py`,
and add C++ dispatch entries in `csrc/ops/`.

**File ownership**: `python/torch_vulkan/inductor/bwd_diff_table.py`,
`python/torch_vulkan/inductor/bwd_diff/unary.py` (or new `masked.py`),
`python/torch_vulkan/inductor/bwd_lowerings.py`.

### Gap 9 — SLANG.3: Spec-constant count validation missing (SLANG.3)
**Current state**: The pre-slangc AST validator (v7 M-VAL.4) checks brace
balance, binding contiguity, undefined identifiers, groupshared budget,
and numthreads product. It does NOT check that `[[vk::constant_id(N)]]`
spec-constant count matches the C++ `struct` pack size, or that
`[BackwardDerivative]` entry points have matching fwd/bwd signatures.

**Fix direction**: Add two checks to `python/torch_vulkan/inductor/slang_validator.py`:
1. Count `vk::constant_id` usages; compare to `ParameterBlock` field count.
2. For each `[BackwardDerivative]` entry, verify parameter count matches
   the forward entry it references.

**File ownership**: `python/torch_vulkan/inductor/slang_validator.py`.

### Gap 10 — LOWER.1: Fallback audit incomplete (LOWER.1)
**Current state**: `lowerings/__init__.py` has two active `make_fallback`
entries:
```python
make_fallback(torch.ops.torch_vulkan.max_pool2d_scatter_bwd.default)
make_fallback(torch.ops.torch_vulkan.avg_pool2d_scatter_bwd.default)
```
These are `_scatter_bwd` operations (tensor index scatter backward).
Neither has a Vulkan equivalent — they route to CPU aten. On the compile
path, this is a silent CPU call.

**Fix direction**: Add `# ratified-extern: no Vulkan equivalent for
pool2d_scatter_bwd; CPU fallback is irreducible` comments to both entries,
documenting the decision. No code change needed — just audit + document.

**File ownership**: `python/torch_vulkan/inductor/lowerings/__init__.py`
(around line 477).

### Gap 11 — LOWER.3: Conv backward TODO (LOWER.3)
**Current state**: `conv_backward.py:38` has a TODO:
> "a paired FX rewrite lifts it into the template path"

Currently `conv_backward` routes through `aten.convolution_backward` →
PrivateUse1 extern. The TODO is to add a pre-grad FX rewrite that replaces
`convolution_backward` with the Slang `bwd_diff(conv_inner_madd)` template
call directly.

**Why this matters**: Going through aten decomposition adds overhead and
loses gradient information precision. The template path gives exact bwd_diff
codegen with `[BackwardDerivative]` annotations already in `slang_conv2d.slang`.

**Fix direction**: Add a `PatternMatcher` pass in
`fx_passes/patterns/conv_backward_rewrite.py` that matches
`convolution_backward` → the `conv_bwd_diff` template node.

**File ownership**: `python/torch_vulkan/inductor/fx_passes/patterns/conv_backward_rewrite.py`
(new file), `python/torch_vulkan/inductor/lowerings/conv_backward.py`.

### Gap 12 — Regression suite gap (REG.1/2/3)
**Current state**: No v14-specific regression tests exist. The existing
`test_aoti_*` and `test_conv_gn_*` tests cover some of the same ground,
but there are no tests that specifically verify:
- `TestAOTI_WrapperLinksVulkanRuntime` (AOTI.1)
- `TestAOTI_ConvGnTraining` with grad parity (AOTI.2)
- `TestV14AOTI3_CleanExit` via subprocess + `timeout` (AOTI.3)
- `TestV14AOTI4_RegisterNoHang` via timed import (AOTI.4)
- `TestV14SYNC2_VmaWritePath` multi-kernel write-read cycle (SYNC.2)

**Fix direction**: Add these as new test classes in
`tests/test_inductor_regression.py`. Use `pytest.mark.skipif` for RDNA1-only
tests; use `subprocess.Popen` + `timeout` for clean-exit tests.

---

# § 2 — Overhauled v15 Roadmap

v15 restructures v14 into a cleaner dependency graph, re-estimates effort
based on the actual code-state audit, and adds explicit file:line ownership
for every item.

## v15 pillars

| # | Pillar | Goal | Effort |
|---|--------|------|--------|
| **V15-AOTI** | AOTI completeness | Generated `.so` resolves `aoti_torch_*_vulkan` at load time, runs Conv+GN fwd+bwd with clean exit. | 3 d |
| **V15-SYNC** | Synchronization & VMA | Non-blocking atexit teardown, validated multi-kernel VMA write path, batcher flush accumulation. | 1.5 d |
| **V15-LANG** | Slang language hygiene | Zero Jinja `#ifdef` in `.slang` files, bwd_diff covers masked ops, spec-constant validation complete. | 1.5 d |
| **V15-LOWER** | Lowering completeness | Fallback audit ratified, conv backward FX rewrite lands. | 1 d |
| **V15-REG** | Regression lock | Every milestone above has a regression test. | 0.5 d |

**Total v15 effort: 7.5 working days** (vs. v14's implicit 8.5 d across
15 items). Reduced because LOWER.2 is already done, and several v14 items
were over-estimated.

## v15 milestones (prioritized execution order)

### Priority 1 — LOWER.2 DONE ✅
`binary.py` lowering removed. Inductor pointwise handles elementwise binary
ops natively. No action needed.

### Priority 2 — LOWER.1 (0.25 d) → unblocks LOWER.3
**Action**: Add `# ratified-extern` comments to the two `make_fallback`
entries in `lowerings/__init__.py:477-481`. No code change.

**File**: `python/torch_vulkan/inductor/lowerings/__init__.py`

**Regression**: `TestV14LOWER1_FallbackAudit` — asserts that every
`make_fallback` entry in `__init__.py` has a `# ratified-extern` or
`# TODO` comment. This test fails if a new fallback is added without
documentation.

### Priority 3 — AOTI.4 (0.5 d) — import hang
**Action**: Wrap `meta_patches/__init__.py` in a lazy-init guard. The
`_OP_IMPLS` registration should not execute until `_register()` has
completed. Add a `_META_PATCHES_READY` flag checked by each module's
top-level import.

**Files**:
- `python/torch_vulkan/inductor/meta_patches/__init__.py` — add lazy guard
- `python/torch_vulkan/inductor/meta_patches/shape_ops.py` — guard `torch.ops.aten.*` calls
- `python/torch_vulkan/inductor/meta_patches/decomposition_passes.py` — same

**Regression**: `TestV14AOTI4_RegisterNoHang` — calls
`torch._register_device_module("vulkan")` with `timeout=5`, asserts no hang.

### Priority 4 — AOTI.1 (1 d) — link fix
**Action**: Modify `cpp_wrapper_gpu.py` (or the compile step in
`compile_graph.py`) to include `csrc/aoti_shims.o` in the wrapper link
line. The shim object provides `aoti_torch_empty_strided_vulkan`,
`aoti_torch_zeros_vulkan`, etc. as C symbols the wrapper can resolve
directly, eliminating the need for `LD_PRELOAD`.

**Alternative if linking the shim object is impractical**: Emit a
`dlopen("aoti_preload.so", RTLD_NOW|RTLD_GLOBAL)` call in the generated
wrapper source before any `aoti_torch_*` calls. This is more portable.

**Files**:
- `python/torch_vulkan/inductor/cpp_wrapper_gpu.py` — emit link args
- OR `python/torch_vulkan/inductor/compile_graph.py` — emit `dlopen` in wrapper src

**Regression**: `TestAOTI_WrapperLinksVulkanRuntime` — loads a minimal
wrapper `.so`, asserts `aoti_torch_empty_strided_vulkan` is resolvable.

### Priority 5 — AOTI.3 (0.5 d) — clean exit
**Action**: In `DeviceRuntime` destructor (or equivalent teardown path):
1. Call `executor.shutdown(wait=False)` on all compile/slangc pools.
2. Call `vkQueueWaitIdle` with 5-second timeout.
3. If timeout fires, log warning and proceed with destruction.

**Files**:
- `csrc/backend/DeviceRuntime.cpp` (or wherever singleton is)
- `python/torch_vulkan/inductor/runtime/common.py` — pool shutdown ordering

**Regression**: `TestV14AOTI3_CleanExit` — subprocess `python -c "..."` with
`timeout 60`, asserts `returncode == 0`.

### Priority 6 — SYNC.2 (0.25 d) — VMA write-path
**Action**: Verify the full multi-kernel training path: allocate buffer,
`copy_` from CPU, dispatch pointwise kernel reading it, dispatch second
kernel, read back, assert matches CPU. Test on RDNA1.

**Files**: `csrc/backend/Allocator.cpp`, `csrc/backend/VulkanBuffer.cpp`

**Regression**: `TestV14SYNC2_VmaWritePath` — RDNA1-only, skips on Lavapipe.

### Priority 7 — SYNC.1 (0.5 d) — batcher flush accumulation
**Action**: Change batcher flush condition from "any dependency" to
"batch is complete". Track a `ready_set` per batch rather than per kernel.
A kernel is added to the current batch if all its tensor dependencies are
already in the batch (or produced by kernels already in the batch).

**File**: `python/torch_vulkan/inductor/runtime/batcher.py`

**Regression**: `TestV14SYNC1_BatchPerf` — MNISTNet, asserts BATCH_DISPATCH=1
is ≤ 1.1× BATCH_DISPATCH=0 (was 1.8×; target is parity).

### Priority 8 — AOTI.2 (1 d) — wrapper E2E run
**Action**: After AOTI.1 links correctly, run a 5-step Conv+GN+SGD
training loop via the AOTI path. Compare all gradients (conv.w, conv.b,
gn.w, gn.b, input grad) to CPU baseline. Threshold: max diff < 1e-4.

**Prerequisites**: AOTI.1 must pass.

**Regression**: `TestAOTI_ConvGnTraining` — 3 sub-tests:
forward parity, backward parity, 5-step loss monotonic decrease.

### Priority 9 — SLANG.1 (0.5 d) — ParameterBlock has_bias
**Action**: Replace `#ifdef HAS_BIAS` in `slang_conv2d.slang` with a
Slang `interface IBias { bool has_bias; }` generic. The kernel entry point
becomes `shader conv2d<IBias>(...)`. The caller's `ParameterBlock` passes
`has_bias` via the interface.

**Files**:
- `python/torch_vulkan/inductor/templates/slang_conv2d.slang`
- `python/torch_vulkan/inductor/templates/caller/conv.py`

**Regression**: `TestV14SLANG1_ParameterBlockConv` — compiles `slang_conv2d.slang`,
asserts no `#ifdef`/`#endif` preprocessor directives in the shader.

### Priority 10 — SLANG.2 (0.5 d) — tril/triu/masked_fill/where bwd_diff
**Action**: Add `[BackwardDerivative]` for:
- `tril` / `triu` — mask is zero or one; derivative is zero where mask is zero
- `masked_fill` — derivative is zero where mask is false, 1 where true
- `where` (masked select) — derivative routes to the selected branch

Register in `bwd_diff_table.py`, add entries in `bwd_lowerings.py`.

**Files**:
- `python/torch_vulkan/inductor/bwd_diff_table.py`
- `python/torch_vulkan/inductor/bwd_lowerings.py`
- `python/torch_vulkan/inductor/bwd_diff/` (new `masked.py` or extend `unary.py`)

**Regression**: `TestV14SLANG2_TrilBackward` — tril + matmul + sum backward
gradient matches CPU.

### Priority 11 — SLANG.3 (0.5 d) — spec-constant validation
**Action**: Add two AST validator checks in `slang_validator.py`:
1. Count `[[vk::constant_id(N)]]` occurrences; assert ≤ max valid N (15).
2. For each `[BackwardDerivative]` entry, find its forward counterpart
   and assert parameter count + types match.

**File**: `python/torch_vulkan/inductor/slang_validator.py`

**Regression**: `TestV14SLANG3_SpecConstValidator` — feeds a shader with
mismatched spec-constant count to validator, asserts `RuntimeError`.

### Priority 12 — LOWER.3 (1 d) — conv backward FX rewrite
**Action**: Add a `PatternMatcher` pass in
`fx_passes/patterns/conv_backward_rewrite.py` that matches the
`aten.convolution_backward` subgraph pattern and replaces it with a call
to the `conv_bwd_diff` Slang template node. This bypasses the aten
decomposition entirely.

**Files**:
- `python/torch_vulkan/inductor/fx_passes/patterns/conv_backward_rewrite.py` (new)
- `python/torch_vulkan/inductor/lowerings/conv_backward.py` (add template path)

**Prerequisites**: LOWER.1 audit complete.

**Regression**: `TestV14LOWER3_ConvBwdTemplate` — asserts that a conv backward
graph node's `name` field contains `conv_bwd_diff` (template path), not
`convolution_backward` (aten path).

### Priority 13 — REG.1/2/3 (0.5 d) — regression lock
**Action**: After each parent milestone passes, add its regression test to
`tests/test_inductor_regression.py` (if not already added in the milestone
itself). Run the full suite on RDNA1 and assert zero failures.

---

# § 3 — Dependency Graph (v15)

```
LOWER.1 (audit fallbacks) ──→ LOWER.3 (conv bwd FX rewrite)
AOTI.4 (import hang) ──────────────────────────┐
                                               ├──→ AOTI.1 (link fix) ──→ AOTI.2 (E2E run) ──→ REG.1
AOTI.3 (clean exit) ──────────────────────────┘         └──→ REG.2 (via AOTI.3)
SYNC.2 (VMA write) ──────────────────→ REG.3
SYNC.1 (batcher flush)
SLANG.1 / SLANG.2 / SLANG.3  ── independent, parallel with AOTI/SYNC
```

Parallelism:
- AOTI.4 + AOTI.1 can be worked simultaneously (AOTI.4 unblocks nothing
  downstream of AOTI.1, but they touch different files).
- SLANG.1/2/3 + SYNC.1 + LOWER.1 run in parallel with AOTI.4.
- LOWER.3 starts after LOWER.1 (1-day overlap).
- AOTI.2 starts after AOTI.1 (1-day overlap).
- REG tests written alongside their parent milestones.

---

# § 4 — File:line ownership map

| File (relative to backends/vulkan_slang/) | Lines | Owner milestone |
|---|---|---|
| `python/torch_vulkan/inductor/lowerings/__init__.py` | 477-481 | LOWER.1 |
| `python/torch_vulkan/inductor/meta_patches/__init__.py` | 1-end | AOTI.4 |
| `python/torch_vulkan/inductor/meta_patches/shape_ops.py` | 1-end | AOTI.4 |
| `python/torch_vulkan/inductor/meta_patches/decomposition_passes.py` | 1-end | AOTI.4 |
| `python/torch_vulkan/inductor/cpp_wrapper_gpu.py` | 1-300 | AOTI.1 |
| `python/torch_vulkan/inductor/compile_graph.py` | 1-400 | AOTI.1 |
| `csrc/backend/DeviceRuntime.cpp` | 1-end | AOTI.3 |
| `python/torch_vulkan/inductor/runtime/common.py` | 200-379 | AOTI.3 |
| `csrc/backend/Allocator.cpp` | 1-end | SYNC.2 |
| `csrc/backend/VulkanBuffer.cpp` | 1-end | SYNC.2 |
| `python/torch_vulkan/inductor/runtime/batcher.py` | 1-end | SYNC.1 |
| `python/torch_vulkan/inductor/templates/slang_conv2d.slang` | 1-200 | SLANG.1 |
| `python/torch_vulkan/inductor/templates/caller/conv.py` | 1-300 | SLANG.1 |
| `python/torch_vulkan/inductor/bwd_diff_table.py` | 1-end | SLANG.2 |
| `python/torch_vulkan/inductor/bwd_lowerings.py` | 1-end | SLANG.2 |
| `python/torch_vulkan/inductor/bwd_diff/unary.py` | 1-end | SLANG.2 |
| `python/torch_vulkan/inductor/slang_validator.py` | 1-end | SLANG.3 |
| `python/torch_vulkan/inductor/fx_passes/patterns/conv_backward_rewrite.py` | new | LOWER.3 |
| `python/torch_vulkan/inductor/lowerings/conv_backward.py` | 461 lines | LOWER.3 |
| `tests/test_inductor_regression.py` | 62,686 lines | REG.1/2/3 |

---

# § 5 — Anti-goals (carried forward, unchanged)

1. No new model-specific `.slang` files — templates only.
2. No new `aten.<op>_backward` lowerings — backward routes through
   `bwd_diff_table.py` → Slang `[BackwardDerivative]`.
3. No symptom-fixes in `meta_patches/` — if a fix needs a new primitive,
   file it as a v15 milestone.
4. No CPU fallbacks on the compile path.
5. No file in `python/torch_vulkan/inductor/` exceeds 800 lines.
6. No Jinja string substitution for kernel parameters that Slang
   ParameterBlock + generics can handle.

---

# § 6 — Disciplines (carried forward, unchanged)

1. Every milestone names a regression test in `tests/test_inductor_regression.py`.
2. Correctness before performance. Gradient parity with CPU is the exit criterion.
3. Floor-gate-then-ratchet: land `xfail(strict=True)` first, then flip.
4. Items that turn out wrong get removed, not annotated.
5. One commit per milestone. Title format `vulkan: V15-<ID> — <why>`.

---

# § 7 — v14 → v15 delta (what changed)

| v14 item | v15 status | Change |
|---|---|---|
| AOTI.1 | V15-AOTI.1 | Re-scoped: link via `dlopen` or shim object, not just LD_PRELOAD |
| AOTI.2 | V15-AOTI.2 | Unchanged |
| AOTI.3 | V15-AOTI.3 | Root cause clarified: DeviceRuntime destructor drain order |
| AOTI.4 | V15-AOTI.4 | Root cause clarified: import-time `torch.ops.aten` deadlock |
| SYNC.1 | V15-SYNC.1 | Root cause clarified: flush condition is per-kernel, not per-batch |
| SYNC.2 | V15-SYNC.2 | Re-verify after MAPPED_BIT removal |
| SLANG.1 | V15-SLANG.1 | Confirmed: `#ifdef HAS_BIAS` still present in slang_conv2d.slang |
| SLANG.2 | V15-SLANG.2 | Unchanged |
| SLANG.3 | V15-SLANG.3 | Unchanged |
| LOWER.1 | V15-LOWER.1 | Only 2 active fallbacks; simple ratify + comment |
| LOWER.2 | **DONE** | binary.py already removed |
| LOWER.3 | V15-LOWER.3 | Unchanged |
| REG.1/2/3 | V15-REG.1/2/3 | Unchanged |

**Removed from v14** (no longer needed):
- LOWER.2 — already done.

**Added to v15** (gaps discovered during audit):
- Explicit file:line ownership table (§4).
- AOTI.1 alternative path (`dlopen` in wrapper source) for portability.
- LOWER.1 narrowed to 2 specific fallbacks (not "audit all").

---

# § 8 — Execution order (v15)

```
Day 1:
  AM — LOWER.1 (0.25 d) + SLANG.1 (0.5 d) [parallel]
  AM — AOTI.4 (0.5 d) [parallel]
  PM — AOTI.1 (0.5 d of 1 d) [after AOTI.4 start]
  PM — SLANG.2 (0.5 d) [parallel]

Day 2:
  AM — AOTI.1 finish (0.5 d remaining)
  AM — SYNC.2 (0.25 d) + SLANG.3 (0.5 d) [parallel]
  PM — SYNC.1 (0.5 d of 0.5 d) + LOWER.3 prep (0.5 d) [parallel]
  PM — AOTI.3 (0.5 d)

Day 3:
  AM — AOTI.2 (1 d) [after AOTI.1]
  AM — LOWER.3 (0.5 d of 1 d) [after LOWER.1]
  PM — LOWER.3 finish (0.5 d)
  PM — SLANG.1 finish if not done Day 1

Day 4:
  AM — REG.1 + REG.2 + REG.3 write + verify
  AM — Full regression suite run on RDNA1
  PM — v15 closeout, mark all milestones ✅
```

---

# § 9 — Hardware notes (unchanged)

- **GPU**: AMD Radeon RX 5600 XT (NAVI10/RDNA1) at `/dev/dri/renderD128`.
  Always verify on real hardware; software Vulkan is diagnostic-only.
- **VMA flags**: `HOST_ACCESS_RANDOM_BIT | MAPPED_BIT` for HostVisible buffers
  (re-evaluated after recent unmap removal; see SYNC.2).
- **Wave size**: 64 (RDNA1).

---

*v15 created 2026-06-10. Replaces v14 as active plan. v7-v14 remain
closed and are reference-only.*
