# Vulkan-Slang Inductor Backend — Full Pipeline Review & Overhauled Roadmap v15

> **v15 (2026-06-10)** — replaces `docs/14-inductor-backend.md` as the active
> roadmap. v14 is superseded. Based on a line-by-line code audit of the
> actual working tree (commit `916712e` + 7 uncommitted modifications).

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
  │     ├─ csrc/aoti_shims.cpp                ← aoti_torch_*_vulkan C symbols
  │     └─ csrc/aoti_preload.cpp              ← preload shared library
  │
  └─ CompiledFxGraph / CompiledAOTI dispatch
        └─ runtime/dispatch.py + runtime/batcher.py
```

## 1.2 Verified working (audit evidence)

| Layer | Status | Evidence |
|-------|--------|---------|
| Dynamo FX capture (Conv+GN+ReLU) | ✅ | `test_conv_gn_relu_fused_forward` PASS |
| AOTAutograd joint graph (5-step train) | ✅ | commit `675316e` — loss 1.511→1.500 |
| Post-grad combo fusion (fwd) | ✅ | `test_conv_gn_relu_fused_forward` PASS |
| Pointwise lowering + fusion | ✅ | `test_binary_ops_match_torch_eager` PASS |
| Slang conv2d forward | ✅ | `slang_conv2d.slang` — no `#ifdef`, bias in ParameterBlock |
| Slang reductions (45 elementals) | ✅ | `test_bwd_diff_*` suite PASS |
| bwd_diff table (fwd→bwd) | ✅ | `test_conv_gn_relu_compile_backward_matches_cpu` PASS |
| SPIR-V codegen (slangc) | ✅ | md5sum verification in CI |
| Pre-slangc AST validator | ✅ | M-VAL.4 closed in v7 |
| VMA write-path (notify_host_write) | ✅ | `test_vma_regression.cpp` PASS |
| ThreadPoolExecutor atexit fix | ✅ | commit `916712e` |

## 1.3 Gaps (what's missing or broken)

### Gap 1 — AOTI wrapper does NOT link Vulkan runtime (AOTI.1)
**Current state**: The generated `.so` at `/tmp/torchinductor_amit_<sha>/` is
built by upstream `CppWrapperCpu`. It links `libtorch.so` + `libtorch_cpu.so`
only. It does **not** include `csrc/backend/aoti_shims.o` (provides
`aoti_torch_empty_strided_vulkan`, `aoti_torch_zeros_vulkan`,
`aoti_torch_ones_vulkan`, `aoti_torch_full_vulkan`,
`aoti_torch_as_strided_vulkan`, `aoti_torch_delete`,
`aoti_torch_vulkan_mm_out` as C symbols).

**What happens**: When `CompiledAOTI` loads the `.so` and calls
`aoti_torch_empty_strided_vulkan`, the symbol is unresolved because:
1. The `.so` link line does not include `csrc/backend/aoti_shims.o`.
2. `csrc/aoti_preload.cpp` exists as a shared library but is never injected
   via `LD_PRELOAD` into the wrapper's `dlopen` context.
3. The `aoti_torch_*` symbols ARE present in the main `_C.cpython-312...so`
   (confirmed via `nm -D`), but the wrapper `.so` has no `DT_NEEDED` entry
   for it and RTLD_LOCAL prevents symbol visibility across `dlopen`.

**Missing piece**: `VulkanCppWrapperGpu` (in `cpp_wrapper_gpu.py`) extends
`CppWrapperCpu` but does NOT override the build step to add
`csrc/backend/aoti_shims.o` to the extension's `extra_objects`.

**File ownership**: `python/torch_vulkan/inductor/cpp_wrapper_gpu.py` (add
`extra_objects=["csrc/backend/aoti_shims.o"]`), AND
`python/torch_vulkan/inductor/__init__.py` (where `VulkanCppWrapperGpu` is
registered at line 809).

**AOT training prerequisite**: Until AOTI.1 is fixed, the AOTI path cannot
run even a forward pass. The `.so` loads but crashes on the first
`aoti_torch_empty_strided_vulkan` call with `undefined symbol`.

### Gap 2 — AOTI run not verified on RDNA1 (AOTI.2)
**Blocker**: AOTI.1 must close first. Once linked, the wrapper must be
exercised with a Conv+GN fwd+bwd run and gradient values compared to CPU.

**Gradient correctness**: Conv weight, conv bias, GN weight, GN bias, and
input gradient all must be < 1e-4 from CPU eager baseline.

**File ownership**: New regression test in `tests/test_inductor_regression.py`
under class `TestAOTI_ConvGnTraining`.

**AOT training coverage today**: `test_aoti_three_layer_mlp_no_bias` (forward
only, no training) and `test_aoti_conv2d_sdpa_model` (xfail). No existing
test runs Conv+GN backward through AOTI. `test_conv_gn_relu_compile_backward_matches_cpu`
uses `torch.compile` directly (Python-wrapper path), not AOTI C++ wrapper.

### Gap 3 — atexit hang after AOTI runs (AOTI.3)
**Current state**: `timeout 120` → RC=124. The `ThreadPoolExecutor` fix
(commit `916712e`) addresses `_register()` import-time hang, but the
atexit teardown chain after a training run still blocks:
```
cleanup_runtimes() → DeviceRuntime::~DeviceRuntime() → vkDeviceWaitIdle()
  → async compile/slangc thread pools still alive
```

**Root cause**: `DeviceRuntime` destructor is synchronous. Any kernel still
queued (or slangc subprocess still running) holds `vkDeviceWaitIdle`
indefinitely.

**Fix direction**: Replace synchronous destructor drain with
non-blocking `shutdown(wait=False)` on all executors, plus
`vkQueueWaitIdle` with 5-second timeout. Teardown order: drain executors
first, then destroy DeviceRuntime.

**File ownership**: `csrc/backend/DeviceRuntime.cpp`,
`python/torch_vulkan/inductor/runtime/common.py`.

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
the `torch.ops.aten` calls with a `_register_complete` flag. Minimal patch:
lazy-init wrapper around `meta_patches/__init__.py`.

**File ownership**: `python/torch_vulkan/inductor/meta_patches/__init__.py`,
`python/torch_vulkan/inductor/meta_patches/shape_ops.py`,
`python/torch_vulkan/inductor/meta_patches/decomposition_passes.py`.

### Gap 5 — SYNC.1: Batch dispatch still serial (SYNC.1)
**Current state**: `BATCH_DISPATCH=1` is 1.8× slower than `BATCH_DISPATCH=0`
(676 ms vs 385 ms for MNISTNet). Batch mode exits on first flush rather than
accumulating N kernels.

**Root cause**: The current batcher flushes whenever a kernel's output is
needed by the next node, rather than accumulating a full buffer.

**Fix direction**: Change the flush condition from "any dependency" to
"all dependencies in the current batch are resolved". Track a `ready_set`
per batch.

**File ownership**: `python/torch_vulkan/inductor/runtime/batcher.py`.

### Gap 6 — SYNC.2: VMA write-path regression (SYNC.2)
**Current state**: The `notify_host_write` fix (commit `2d438e4`) is in
`Allocator::allocate()`, but the recent MAPPED_BIT removal + unmap changes
(commits `fea31a5`, `a345ea7`) mean the full multi-kernel training path
has not been re-verified on RDNA1.

**File ownership**: `csrc/backend/Allocator.cpp`, `csrc/backend/VulkanBuffer.cpp`.

### Gap 7 — SLANG.1: `slang_conv_bwd.py.jinja` still loaded for backward conv
**Current state**: Forward `slang_conv2d.slang` is clean — no `#ifdef`, bias
in ParameterBlock, runtime-gated by `stride_bias != 0` (M-SF.4 landed).

**However**: The backward conv shader is still loaded from
`slang_conv_bwd.py.jinja` (Python-string Jinja2 template, 225 lines).
`slang_conv_bwd.slang` exists (11,882 bytes) but is NEVER loaded — the
caller at `conv.py:448-470` passes `has_bias` as a Jinja variable to
`.py.jinja`. This is the actual "Jinja for kernel parameters" anti-goal
violation.

**Fix direction**:
1. Migrate `slang_conv_bwd.py.jinja` content into `slang_conv_bwd.slang`
   using Slang `interface IBias { bool has_bias; }` generic on entry point.
2. Update `conv.py:457` to load `.slang` (already preferred by
   `_load_slang_template`, but error message says `.py.jinja`).
3. Delete `slang_conv_bwd.py.jinja`.

**File ownership**: `python/torch_vulkan/inductor/templates/slang_conv_bwd.slang`,
`python/torch_vulkan/inductor/templates/caller/conv.py`.

**Regression**: `TestV14SLANG1_ParameterBlockConv` — both fwd and bwd
`slang_conv*.slang` compile with no `.py.jinja` fallback.

### Gap 7b — Remaining 10 `.py.jinja` templates (SLANG.1b)
**Current state**: 11 `.py.jinja` files remain. Priority order for migration:

1. `persistent_pointwise.py.jinja` (139 L — simplest)
2. `philox_rng.py.jinja` (130 L — simple RNG)
3. `scatter_atomic.py.jinja` (176 L)
4. `fft_stockham.py.jinja` (160 L)
5. `rnn_cell.py.jinja` + `rnn_cell_fused.py.jinja` + `rnn_cell_bwd.py.jinja`
6. `flash_attention.py.jinja` + `flash_attention_bwd.py.jinja`
7. `foreach_optimizer.py.jinja` (250 L — algorithm branching, hardest)

**Note**: Several `.slang` files already have Jinja-in-comments
(`/*{%*/ ... /*%}*/`) for algorithm branching — convert to Slang `interface`
generics during migration.

**Regression**: `TestV14SLANG1b_NoPyJinja` — asserts zero `.py.jinja`
files remain in `templates/`.

### Gap 8 — SLANG.2: tril/triu/masked_fill/where missing from bwd_diff
**Current state**: `bwd_diff_table.py` covers ~45 elementals (activations,
trig, exp/log, binary pointwise, losses). `tril`, `triu`, `masked_fill`,
and `where` (masked select) are absent. These block sparse-attention and
padding-mask models.

**Fix direction**: Add `[BackwardDerivative]` annotations in the elemental
Slang library, register in `bwd_diff_table.py`. No new C++ dispatch needed
— `bwd_diff` codegen emits the shader directly.

**File ownership**: `python/torch_vulkan/inductor/bwd_diff_table.py`,
`python/torch_vulkan/inductor/bwd_lowerings.py`,
`python/torch_vulkan/inductor/bwd_diff/unary.py` (or new `masked.py`).

### Gap 9 — SLANG.3: Spec-constant + bwd_diff signature validation
**Current state**: The pre-slangc AST validator (`slang_validate/` sub-package,
8 passes, 670 LOC) checks brace balance, binding contiguity, undefined
identifiers, groupshared budget, and numthreads product. Missing:
1. `[[vk::constant_id(N)]]` count vs. `ParameterBlock` field count match.
2. `[BackwardDerivative]` entry points have matching fwd/bwd signatures.

**Fix direction**: Add two passes:
- `slang_validate/spec_constants.py` — count + range check.
- `slang_validate/bwd_diff_scan.py` — already exists (67 LOC) but only
  scans for `[BackwardDerivative]` annotations, does not verify signature
  matching. Extend it.

**File ownership**: `python/torch_vulkan/inductor/slang_validate/bwd_diff_scan.py`,
new `python/torch_vulkan/inductor/slang_validate/spec_constants.py`.

### Gap 10 — LOWER.1: Fallback audit (2 active entries)
**Current state**: `lowerings/__init__.py:477-481` has two active
`make_fallback` entries:
```python
make_fallback(torch.ops.torch_vulkan.max_pool2d_scatter_bwd.default)
make_fallback(torch.ops.torch_vulkan.avg_pool2d_scatter_bwd.default)
```
These are `_scatter_bwd` ops — no Vulkan equivalent. Routes to CPU aten on
compile path.

**Fix direction**: Add `# ratified-extern: no Vulkan equivalent for
pool2d_scatter_bwd; CPU fallback is irreducible` comments.

**File ownership**: `python/torch_vulkan/inductor/lowerings/__init__.py`.

### Gap 11 — LOWER.3: Conv backward FX rewrite
**Current state**: `conv_backward.py` has a TODO at line 38:
> "a paired FX rewrite lifts it into the template path"

Currently `conv_backward` routes through `aten.convolution_backward` →
PrivateUse1 extern. The fix is a pre-grad FX rewrite that replaces
`convolution_backward` with the Slang `bwd_diff(conv_inner_madd)` template
call, bypassing aten decomposition.

**Fix direction**: Add `PatternMatcher` pass at
`fx_passes/patterns/conv_backward_rewrite.py`.

**File ownership**: new `python/torch_vulkan/inductor/fx_passes/patterns/conv_backward_rewrite.py`,
`python/torch_vulkan/inductor/lowerings/conv_backward.py`.

### Gap 12 — Regression suite gap (REG.1/2/3)
No v15-specific regression tests exist for the AOTI/SYNC/SLANG milestones
listed above. Existing `test_aoti_*` and `test_conv_gn_*` tests cover
adjacent ground but miss the specific failure modes.

---

# § 2 — Overhauled v15 Roadmap

## v15 pillars

| # | Pillar | Goal | Effort |
|---|--------|------|--------|
| **V15-AOTI** | AOTI completeness | Generated `.so` resolves `aoti_torch_*_vulkan` at load time, runs Conv+GN fwd+bwd with clean exit. | 3 d |
| **V15-SYNC** | Synchronization & VMA | Non-blocking atexit teardown, validated multi-kernel VMA write path, batcher flush accumulation. | 1.5 d |
| **V15-LANG** | Slang language hygiene | Zero `.py.jinja` templates remaining, bwd_diff covers masked ops, spec-constant + bwd_diff signature validation complete. | 2 d |
| **V15-LOWER** | Lowering completeness | Fallback audit ratified, conv backward FX rewrite lands. | 1 d |
| **V15-REG** | Regression lock | Every milestone above has a regression test. | 0.5 d |

**Total: 8 working days.**

## v15 milestones (prioritized execution order)

### Priority 1 — LOWER.2 DONE ✅
`binary.py` lowering removed. Inductor pointwise handles elementwise binary
ops natively. No action needed.

### Priority 2 — LOWER.1 (0.25 d) → unblocks LOWER.3
**Action**: Add `# ratified-extern` comments to the two `make_fallback`
entries in `lowerings/__init__.py:477-481`.

**Regression**: `TestV14LOWER1_FallbackAudit` — asserts every `make_fallback`
entry has a `# ratified-extern` or `# TODO` comment.

### Priority 3 — AOTI.4 (0.5 d) — import hang
**Action**: Wrap `meta_patches/__init__.py` in a lazy-init guard. Add a
`_META_PATCHES_READY` flag; each module's top-level import checks it before
calling `torch.ops.aten.*`.

**Regression**: `TestV14AOTI4_RegisterNoHang` — calls
`torch._register_device_module("vulkan")` with `timeout=5`, asserts no hang.

### Priority 4 — AOTI.1 (1 d) — link fix
**Action**: Add `extra_objects=["csrc/backend/aoti_shims.o"]` to the
`CppExtension` that `VulkanCppWrapperGpu` builds. The shim object provides
all `aoti_torch_*_vulkan` C symbols.

**File**: `python/torch_vulkan/inductor/cpp_wrapper_gpu.py` (where the
`CppExtension` is constructed), `python/torch_vulkan/inductor/__init__.py:809`
(where `VulkanCppWrapperGpu` is registered).

**Regression**: `TestAOTI_WrapperLinksVulkanRuntime` — loads a minimal
wrapper `.so`, asserts `aoti_torch_empty_strided_vulkan` resolves.

### Priority 5 — AOTI.3 (0.5 d) — clean exit
**Action**: In `DeviceRuntime` destructor: `shutdown(wait=False)` on all
executors, then `vkQueueWaitIdle` with 5-second timeout, then proceed.

**File**: `csrc/backend/DeviceRuntime.cpp`,
`python/torch_vulkan/inductor/runtime/common.py`.

**Regression**: `TestV14AOTI3_CleanExit` — subprocess `python -c "..."` with
`timeout 60`, asserts `returncode == 0`.

### Priority 6 — SYNC.2 (0.25 d) — VMA write-path
**Action**: Verify full multi-kernel training path on RDNA1: allocate,
`copy_` from CPU, dispatch pointwise, dispatch second, read back, match CPU.

**Regression**: `TestV14SYNC2_VmaWritePath` — RDNA1-only, skips Lavapipe.

### Priority 7 — SYNC.1 (0.5 d) — batcher flush accumulation
**Action**: Track `ready_set` per batch; flush only when batch is complete.

**Regression**: `TestV14SYNC1_BatchPerf` — MNISTNet, BATCH_DISPATCH=1 ≤ 1.1×
BATCH_DISPATCH=0.

### Priority 8 — AOTI.2 (1 d) — wrapper E2E run (AFTER AOTI.1)
**Action**: 5-step Conv+GN+SGD training via AOTI path. All gradients < 1e-4
from CPU baseline.

**Prerequisites**: AOTI.1 must pass.

**Regression**: `TestAOTI_ConvGnTraining` — forward parity, backward parity,
5-step loss monotonic decrease.

### Priority 9 — SLANG.1 (0.5 d) — migrate slang_conv_bwd.py.jinja → .slang
**Action**:
1. Migrate `slang_conv_bwd.py.jinja` → `slang_conv_bwd.slang` with Slang
   `interface IBias { bool has_bias; }` generic.
2. Update `conv.py:457` to load `.slang`, remove `has_bias` from Jinja render.
3. Delete `slang_conv_bwd.py.jinja`.

**Regression**: `TestV14SLANG1_ParameterBlockConv` — both fwd and bwd
`slang_conv*.slang` compile, no `.py.jinja` fallback loaded.

### Priority 9b — SLANG.1b (1 d) — migrate remaining 10 .py.jinja → .slang
Order: persistent_pointwise, philox_rng, scatter_atomic, fft_stockham,
rnn_cell×3, flash_attention×2, foreach_optimizer.

**Regression**: `TestV14SLANG1b_NoPyJinja` — zero `.py.jinja` files in
`templates/`.

### Priority 10 — SLANG.2 (0.5 d) — tril/triu/masked_fill/where bwd_diff
**Action**: Add `aten.tril_backward`, `aten.triu_backward`,
`aten.masked_fill_backward`, `aten.where_backward` to `bwd_diff_table.py`
with appropriate `[BackwardDerivative]` Slang annotations.

**Regression**: `TestV14SLANG2_TrilBackward` — tril + matmul + sum bwd
gradient matches CPU.

### Priority 11 — SLANG.3 (0.5 d) — spec-constant + bwd_diff signature validation
**Action**: Extend `slang_validate/bwd_diff_scan.py` to verify parameter
count/types match between `[BackwardDerivative]` entry and its forward
counterpart. Add `slang_validate/spec_constants.py` for constant_id count
vs. ParameterBlock field count.

**Regression**: `TestV14SLANG3_SpecConstValidator` — feeds mismatched
shader, asserts `RuntimeError`.

### Priority 12 — LOWER.3 (1 d) — conv backward FX rewrite
**Action**: Add `PatternMatcher` pass in
`fx_passes/patterns/conv_backward_rewrite.py` that matches
`aten.convolution_backward` → `conv_bwd_diff` template node.

**Prerequisites**: LOWER.1 audit complete.

**Regression**: `TestV14LOWER3_ConvBwdTemplate` — graph node name contains
`conv_bwd_diff` (template path), not `convolution_backward` (aten path).

### Priority 13 — REG.1/2/3 (0.5 d) — regression lock
Add all parent milestone tests to `tests/test_inductor_regression.py`.
Run full suite on RDNA1.

---

# § 3 — Dependency Graph

```
LOWER.1 ──→ LOWER.3
AOTI.4 ──────────────────────────┐
                                 ├──→ AOTI.1 ──→ AOTI.2 ──→ REG.1
AOTI.3 ──────────────────────────┘         └──→ REG.2 (via AOTI.3)
SYNC.2 ──────────────────→ REG.3
SYNC.1
SLANG.1 → SLANG.1b → SLANG.2 → SLANG.3  (serial within LANG pillar)
```

Parallelism:
- AOTI.4 + AOTI.1 can run simultaneously (different files, AOTI.4 unblocks
  nothing downstream of AOTI.1 but both are needed before AOTI.2).
- SLANG.1/1b/2/3 are serial within the pillar but run in parallel with
  AOTI/SYNC/LOWER workstreams.
- LOWER.3 starts after LOWER.1 (1-day overlap).

---

# § 4 — File:line ownership map

| File | Lines | Milestone |
|------|-------|-----------|
| `python/torch_vulkan/inductor/lowerings/__init__.py` | 477-481 | LOWER.1 |
| `python/torch_vulkan/inductor/meta_patches/__init__.py` | 1-end | AOTI.4 |
| `python/torch_vulkan/inductor/meta_patches/shape_ops.py` | 1-end | AOTI.4 |
| `python/torch_vulkan/inductor/meta_patches/decomposition_passes.py` | 1-end | AOTI.4 |
| `python/torch_vulkan/inductor/cpp_wrapper_gpu.py` | 76-170 | AOTI.1 |
| `python/torch_vulkan/inductor/__init__.py` | 809 | AOTI.1 |
| `csrc/backend/DeviceRuntime.cpp` | 1-end | AOTI.3 |
| `python/torch_vulkan/inductor/runtime/common.py` | 200-379 | AOTI.3 |
| `csrc/backend/Allocator.cpp` | 1-end | SYNC.2 |
| `csrc/backend/VulkanBuffer.cpp` | 1-end | SYNC.2 |
| `python/torch_vulkan/inductor/runtime/batcher.py` | 1-end | SYNC.1 |
| `python/torch_vulkan/inductor/templates/slang_conv_bwd.slang` | 1-230 | SLANG.1 |
| `python/torch_vulkan/inductor/templates/caller/conv.py` | 448-470 | SLANG.1 |
| `python/torch_vulkan/inductor/templates/*.py.jinja` | 10 files | SLANG.1b |
| `python/torch_vulkan/inductor/bwd_diff_table.py` | 1-end | SLANG.2 |
| `python/torch_vulkan/inductor/bwd_lowerings.py` | 1-end | SLANG.2 |
| `python/torch_vulkan/inductor/slang_validate/bwd_diff_scan.py` | 1-67 | SLANG.3 |
| `python/torch_vulkan/inductor/slang_validate/spec_constants.py` | new | SLANG.3 |
| `python/torch_vulkan/inductor/fx_passes/patterns/conv_backward_rewrite.py` | new | LOWER.3 |
| `python/torch_vulkan/inductor/lowerings/conv_backward.py` | 461 lines | LOWER.3 |
| `tests/test_inductor_regression.py` | 62,686 lines | REG |

---

# § 5 — Slang Smart-Features Audit Summary

## 5.1 What's already smart (no action needed)

| Feature | Status | Where |
|---------|--------|-------|
| ParameterBlock<KernelArgs> on all active templates | ✅ | All `slang_*.slang` files |
| Spec constants `[[vk::constant_id]]` for tile sizes | ✅ | `slang_conv2d.slang:30-34`, `slang_conv_bwd.slang:40-44`, `slang_mm.slang` |
| `[Differentiable]` on all elemental fwd shaders | ✅ | `shaders/lib/*.slang` |
| `[BackwardDerivative]` on 45 elementals | ✅ | `bwd_diff_table.py` |
| Runtime stride_bias gating (no `#ifdef`) | ✅ | `slang_conv2d.slang:207-208` |
| Pre-slangc AST validator (8 passes) | ✅ | `slang_validate/` sub-package |

## 5.2 What needs work

| Issue | Severity | Location |
|-------|----------|----------|
| `slang_conv_bwd.py.jinja` loaded instead of `.slang` | P0 — anti-goal violation | `conv.py:448-470` |
| 10 remaining `.py.jinja` templates | P1 — anti-goal | `templates/*.py.jinja` |
| `tril`/`triu`/`masked_fill`/`where` missing from bwd_diff | P1 — blocks models | `bwd_diff_table.py` |
| Spec-constant count not validated | P2 — correctness safety | `slang_validate/` |
| bwd_diff signature matching not validated | P2 — correctness safety | `slang_validate/bwd_diff_scan.py` |

## 5.3 Conv model AOT-training capability

The conv model **CAN be trained** through `torch.compile(backend="inductor")`
(Python-wrapper path, verified commit `675316e`). It **CANNOT yet be trained**
through the AOTI C++ wrapper path because:

1. **AOTI.1 blocks everything**: The `.so` is missing `aoti_shims.o` in its
   link line. Even a single `aoti_torch_empty_strided_vulkan` call fails
   with `undefined symbol`.

2. **AOTI.2 unverified**: Once AOTI.1 is fixed, the AOTI wrapper has never
   been exercised with a Conv+GN backward pass.

3. **AOTI.3 blocks CI**: Even if AOTI.1+2 work, the process hangs at exit
   (`timeout 120` → RC=124), making automated training impossible.

**Minimal path to AOT training**:
1. Fix AOTI.1 (link `aoti_shims.o`).
2. Fix AOTI.3 (non-blocking teardown).
3. Run `TestAOTI_ConvGnTraining` with 5-step Conv+GN+SGD loop.
4. Verify all 5 gradient tensors < 1e-4 from CPU baseline.

---

# § 6 — Anti-goals (carried forward)

1. No new model-specific `.slang` files — templates only.
2. No new `aten.<op>_backward` lowerings — backward routes through
   `bwd_diff_table.py` → Slang `[BackwardDerivative]`.
3. No symptom-fixes in `meta_patches/`.
4. No CPU fallbacks on the compile path.
5. No file in `python/torch_vulkan/inductor/` exceeds 800 lines.
6. No Jinja string substitution for kernel parameters that Slang
   ParameterBlock + generics + interfaces can handle.

---

# § 7 — Disciplines (carried forward)

1. Every milestone names a regression test in `tests/test_inductor_regression.py`.
2. Correctness before performance. Gradient parity with CPU is the exit criterion.
3. Floor-gate-then-ratchet: land `xfail(strict=True)` first, then flip.
4. Items that turn out wrong get removed, not annotated.
5. One commit per milestone. Title format `vulkan: V15-<ID> — <why>`.

---

# § 8 — v14 → v15 delta

| v14 item | v15 status | Change |
|---|---|---|
| AOTI.1 | V15-AOTI.1 | Root cause pinpointed: `aoti_shims.o` missing from `extra_objects` |
| AOTI.2 | V15-AOTI.2 | Clarified: zero existing AOTI Conv+GN backward tests |
| AOTI.3 | V15-AOTI.3 | Root cause: DeviceRuntime destructor drain order |
| AOTI.4 | V15-AOTI.4 | Root cause: import-time `torch.ops.aten` deadlock |
| SYNC.1 | V15-SYNC.1 | Root cause: flush condition per-kernel vs per-batch |
| SYNC.2 | V15-SYNC.2 | Re-verify after MAPPED_BIT removal |
| SLANG.1 | V15-SLANG.1 | **Corrected**: forward is clean; backward `slang_conv_bwd.py.jinja` is the violation |
| SLANG.2 | V15-SLANG.2 | Unchanged |
| SLANG.3 | V15-SLANG.3 | Extended: bwd_diff signature matching |
| LOWER.1 | V15-LOWER.1 | Narrowed to 2 specific fallbacks |
| LOWER.2 | **DONE** | binary.py already removed |
| LOWER.3 | V15-LOWER.3 | Unchanged |
| REG.1/2/3 | V15-REG.1/2/3 | Unchanged |

**New in v15**: SLANG.1b (migrate remaining 10 `.py.jinja` templates).

---

*v15 created 2026-06-10. Replaces v14 as active plan. v7-v14 remain
closed and are reference-only.*
