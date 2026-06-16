# Vulkan-Slang Inductor Backend — Consolidated Roadmap

> **Canonical, single-source roadmap (created 2026-06-15).** Supersedes and
> replaces the entire numbered series `docs/10/14/15/16-inductor-backend.md`
> and `docs/codegen-optimization-roadmap.md`. Those files are deleted; their
> still-open items are folded in below. Closed-milestone history lives in
> `docs/10-inductor-backend-history.md` and `docs/archive/`.
>
> Built from a fresh ground-truth audit (3 parallel code audits + GPU smoke
> run) of the working tree at commit `60541e0e1e8`.

---

## Mission

Ship a fully optimizing, **training-grade** `torch.compile(backend="inductor")`
backend on Vulkan/Slang that supports **any** PyTorch model. Every kernel is
auto-generated from Slang templates → SPIR-V. No per-model `.slang` files. No
per-model `csrc/ops/*.cpp`. No CPU fallbacks on the compile path. No Python at
deployment (AOTI `.so`).

The four durable goals (carried from the v7→v16 pillar history):

| Pillar | Goal |
|---|---|
| **Codegen-only** | No `extern_kernels.X` to aten / eager Vulkan inside compiled wrappers; no `if device != vulkan: aten` branches on the compile path. |
| **Slang-smart** | `ParameterBlock` + generics + `interface`s + spec-constants + `[BackwardDerivative]` + reflection metadata. Jinja only for spec-constant numeric tunables. |
| **Validation-driven** | Vulkan validation layer mandatory in tests (`TORCH_VULKAN_VUID_AS_ERROR=1`); a VUID during autotune rejects the candidate; a VUID on a landed kernel fails the test. |
| **Profile-and-warmup** | `torch_vulkan.prepare_device(level, timeout_s)` once at process start pays the cold cost up front. |

---

## § 1 — Current State (ground truth, 2026-06-15)

The backend **trains real models end-to-end on the GPU today** via the
eager-PrivateUse1 and `torch.compile` paths (MNIST CNN/MLP/ResNet/Transformer,
Conv+GN, AMP fp16/bf16 — see `tests/test_mnist_training.py`,
`tests/test_v9_e2e_conv_training.py`). The render-group hardware blocker that
gated AOTI is cleared (`amit ∈ render`). The work that remains is (a) closing
the **AOTI Python-less deployment** path, (b) finishing the **Slang-smart
interface migrations**, (c) the **performance** layer that makes correct
batching actually faster, and (d) the long tail of **op coverage**.

### Status scorecard

Legend: ✅ done · 🟡 partial · ⛔ open · 🔬 needs re-verification

**AOT / deployment (AOTI)**
| Item | State | Evidence |
|---|---|---|
| Link `aoti_shims.o` into wrapper `.so` (14 `aoti_torch_*` symbols) | ✅ | `setup.py:133`, `cpp_wrapper_gpu.py` |
| Import does not hang (`meta_patches` lazy) | ✅ | `meta_patches/__init__.py` |
| Clean process exit (`shutdown(wait=False)` at atexit) | ✅ | `runtime/common.py:373-379` |
| **Conv+GN+pool+linear training E2E + grad parity** (`torch.compile` path) | ✅ | `TestAOTITrainingE2E` (`test_inductor_regression.py:62689`) — **FIXED 2026-06-16 (M23.2)**. Root cause: `gw_box.realize()` in `_get_conv_backward_lowering_impl` (`conv_backward.py:332`) caused the scheduler to discard the ExternKernelOut's weight-gradient writes. Fix: removed `.realize()` calls (matching the sibling `_get_conv2d_backward_custom_op_lowering` pattern at line 430-433). Both `test_conv_gn_relu_grad_match_cpu` and `test_conv_compile_backward_matches_cpu` now PASS on GPU. |
| AOTI `.so` fwd+bwd+optimizer full step, SPIR-V cache reuse across loads | ⛔ | not exercised end-to-end; ABI shim symbols present (`test_aoti_extern.py`) but no full-training `.so` test |

**Training correctness (the M19–M23 / FP16 line, recently active)**
| Item | State | Evidence |
|---|---|---|
| Linear-backward decomposition (mm+mm+sum, no eager extern) | ✅ | M-CG.4 / M19.1 |
| grad_weight / grad_bias zero-init allocator + `copy_` path | ✅ | commit `60541e0e1e8` (M23.1) |
| Conv backward fp16→fp32 upcast in template caller | ✅ | commit `3444718dd33` (FP16.1) |
| Conv backward still routes through `aten.convolution_backward` (ratified extern) | 🟡 | re-eval toward `bwd_diff(conv_inner_madd)` paired FX rewrite (`conv_backward.py:38` TODO) |

**Slang-smart codegen (LANG)**
| Item | State | Evidence |
|---|---|---|
| Zero `.py.jinja` files; `.slang` is canonical | ✅ | `find templates -name '*.py.jinja'` empty |
| `conv_bwd` `has_bias` de-Jinja → runtime `stride_grad_bias` gate | ✅ | `slang_conv_bwd.slang:85-88,264` |
| `ParameterBlock<KernelArgs>` on all active templates | ✅ | 17/17 templates |
| Spec-constants for tile/wg tunables (11 templates) | ✅ | `[[vk::constant_id(N)]]` |
| `[Differentiable]` / `[BackwardDerivative]` on bwd entry points | ✅ | conv/mm/reduction bwd shaders |
| Subgroup (`WaveActive*`) reduction ops behind `IWaveReduction` | ✅ | `shaders/lib/reduction.slang:59-170` |
| `flash_attention*` wg_size/BQ/BK as spec-constants | ✅ | committed `03c00dd6176`; `is_causal`/`head_layout` remain Jinja (defensible: code-structural) |
| **`foreach_optimizer` algorithm → Slang interface** | ⛔ | still `{% if algorithm %}` + Jinja buffer-array `for` loops |
| **`rnn_cell*` direction → interface; cell_type (lstm/gru) Jinja** | ⛔ | `rnn_cell.slang:35-47` Jinja `cell_type` + `direction` branches (cell_type was a gap in v16) |
| **AST validator: spec-constant pass** | ✅ | `slang_validate/spec_constants.py` (189 L) |
| **AST validator: bwd_diff signature matching** | 🟡 | `bwd_diff_scan.py` pairs `[Differentiable]`↔`[BackwardDerivative]` but does not match param count/types |

**Performance (PERF)**
| Item | State | Evidence |
|---|---|---|
| Batcher `ready_set` per-batch flush (correctness) | ✅ | `runtime/batcher.py:55-152` |
| **Batch dispatch is correct but 1.8× slower** → default OFF | ⛔ | `config.py:119-123`; the payoff needs compile/exec overlap (below) |
| **Async-compile double-buffer overlap** (exec kernel N while compiling N+2) | 🟡 | async pool exists (`slangc.py:544`), overlap not wired into flush path |
| **Shape bucketing** in template registry (canonicalize → cache SPIR-V) | ⛔ | `template_registry.py` has a `shape_class` field only; no canonicalization |
| **Persistent kernel routing** for large reductions (numel>65536) | 🟡 | `persistent_pointwise.slang` exists; not wired in `bwd_diff_table.py` |

**Autotune (TUNE)**
| Item | State | Evidence |
|---|---|---|
| Hardware probe / level-2 autotune sweep infra | ✅ | `hardware_probe.py`, `autotune.py` |
| **Inductor `tuned_mm`/`tuned_conv`/`tuned_flash_attention` choice hooks** | ⛔ | `VulkanTemplateKernel` choices not registered into `V.choices` |

**Op coverage** (last full audit 2026-05-09: 45% native / 28% decomposed / 24% extern / 2% missing / 1% wrong; improved since)
| Item | State |
|---|---|
| Active `make_fallback` entries: `max_pool2d_scatter_bwd`, `avg_pool2d_scatter_bwd` | 🟡 replace with Slang codegen or ratify |
| Missing ops: `sort`, `bucketize`, `multinomial`, sparse (csr/coo), eager FFT | ⛔ |
| `tril`/`triu`/`masked_fill`/`where` backward via `bwd_diff` (sparse-attn / padding masks) | ⛔ |
| Reflection introspection (VGPR/LDS → numthreads): parsed, partially used | 🟡 |

---

## § 2 — Forward Roadmap (open work, prioritized)

Ordering principle: **correctness/deployment before performance before
coverage breadth.** Each item names its regression test (Discipline #1).

### Pillar A — AOT deployment (highest priority)

#### A1 — Fix conv grad_weight mismatch in compiled Conv+GN training ✅ (M23.2)
**FIXED 2026-06-16.** Root cause: `gw_box.realize()` + `gb_box.realize()` in
`_get_conv_backward_lowering_impl` (`conv_backward.py:328-343`) caused the
Inductor scheduler to treat the pre-allocated grad_weight/grad_bias buffers as
already-finalized, discarding the ExternKernelOut's mutation writes (zero
gradients / sign-flipped weight grads manifesting as `rel_diff≈1.97`). The
sibling path `_get_conv2d_backward_custom_op_lowering` (line 430-433) already
had a comment explicitly warning against this pattern — the fix removes the
`.realize()` calls to match. Verified: `test_conv_gn_relu_grad_match_cpu` and
`test_conv_compile_backward_matches_cpu` both PASS on GPU (`--gpu`).

#### A2 — AOTI full training step (fwd + bwd + optimizer) in one `.so` ⛔
Extend A1 to include the optimizer update inside the AOTI package; verify the
SPIR-V cache is reused across a second `aoti_load_package` with zero recompiles.
- **Files**: `cpp_wrapper_gpu.py`, `runtime/slangc.py` (cache key), `csrc/backend/`
- **Exit**: `TestAOTI_FullStep` — 2 loads, second load recompiles 0 shaders.

#### A3 — Conv backward paired FX rewrite → `bwd_diff(conv_inner_madd)` 🟡
Remove the last ratified extern on the training path. Pre-grad FX pass replaces
`aten.convolution_backward` with the Slang `[BackwardDerivative]` conv template
(closes anti-goal #3 for conv).
- **Files**: `lowerings/conv_backward.py:38`, `fx_passes/post_grad.py`
- **Exit**: `TestConvBwdNoExtern` — compiled conv-bwd graph has no `convolution_backward` extern node; grad parity holds.

### Pillar B — Slang-smart codegen

#### B1 — `foreach_optimizer` algorithm → `interface IOptimizerAlgorithm` ⛔
Replace `{% if algorithm == "adamw" %}` and the Jinja buffer-array `for` loops
with a single module: algorithm chosen by spec-constant `ALGORITHM_ID` (0=SGD,
1=AdamW, 2=Lion); buffer arrays via runtime-indexed descriptor array. 3 SPIR-V
variants → 1.
- **Files**: `templates/foreach_optimizer.slang`, `caller/optimizer.py`
- **Exit**: `TestOptimizerInterface` — one compile, dispatch SGD/AdamW/Lion by spec-const, parity vs eager.

#### B2 — `rnn_cell*` direction + cell_type → interface/spec-const ⛔
`direction` → runtime gate on a spec-constant. `cell_type` (lstm/gru/rnn_tanh/
rnn_relu) is currently a Jinja structural branch **not tracked in v16** — fold
into an `IRnnCell` interface (gate `has_cell_state` / `gate_size` by spec-const).
Covers `rnn_cell.slang`, `rnn_cell_bwd.slang`, `rnn_cell_fused.slang`.
- **Files**: `templates/rnn_cell*.slang`, `caller/rnn.py`
- **Exit**: `TestRnnInterface` — LSTM + GRU + bidirectional from one module set, parity vs eager.

#### B3 — AST validator: bwd_diff signature matching 🟡
Extend `bwd_diff_scan.py` to extract forward/backward parameter lists and assert
matching count/types (not just presence of the annotation).
- **Files**: `slang_validate/bwd_diff_scan.py`
- **Exit**: `TestValidatorBwdSignature` — mismatched-arity shader raises `RuntimeError` pre-slangc.

### Pillar C — Performance (the batching payoff)

#### C1 — Async-compile / dispatch overlap → make `BATCH_DISPATCH=1` win ⛔
M12 made batched dispatch *correct* but it is 1.8× slower (`385ms→676ms`
MNISTNet) because setup/teardown is serial. Wire the existing async slangc pool
into a double-buffer: execute kernel N while compiling N+2. Target: batched ≤
1.1× unbatched, then flip the default to ON.
- **Files**: `runtime/batcher.py`, `runtime/slangc.py`, `csrc/backend/DeviceRuntime.cpp`, `config.py:123`
- **Exit**: `TestBatchPerf` — MNISTNet batched overhead ≤ 10%; default flips to ON.

#### C2 — Shape bucketing in template registry ⛔
Canonicalize `(rank, dtype, layout_class, stride_class)` before template
selection; cache compiled SPIR-V by the canonical key so same-class shapes never
re-invoke slangc.
- **Files**: `kernel/template_registry.py`
- **Exit**: `TestShapeBucketing` — two same-class shapes ⇒ one slangc invocation.

#### C3 — Persistent-kernel routing for large reductions 🟡
Route reductions with `numel > 65536` to `persistent_pointwise.slang` (loop over
chunks in one workgroup) from `bwd_diff_table.py`.
- **Files**: `bwd_diff_table.py`, `templates/persistent_pointwise.slang`
- **Exit**: `TestPersistentReduction` — large `sum`/`mean` parity + dispatch-count drop.

### Pillar D — Autotune

#### D1 — Wire Slang templates into Inductor autotune ⛔
Register `VulkanTemplateKernel` choices into `V.choices` via device-specific
`tuned_mm` / `tuned_conv` / `tuned_flash_attention` overrides; 3–5 tile configs
each; benchmark on RDNA1; reject any candidate that emits a VUID (validation-
driven, Pillar goal). Cache the winner.
- **Files**: `kernel/template_registry.py`, new `tuned_*` hooks
- **Exit**: `TestAutotuneMM` — best-of-N tile config chosen and cached; VUID candidate rejected.

### Pillar E — Op coverage (breadth, ongoing)

#### E1 — Eliminate the 2 pooling-bwd `make_fallback`s ⛔
Replace `max_pool2d_scatter_bwd` / `avg_pool2d_scatter_bwd` with Slang
`scatter_atomic` codegen, or ratify with an upstream-reason comment.
- **Files**: `lowerings/__init__.py:477,481`, `templates/scatter_atomic.slang`

#### E2 — Masking backward set for attention/padding ⛔
Add `[BackwardDerivative]` + `bwd_diff_table` entries for `tril`/`triu`/
`masked_fill`/`where`. Unblocks sparse-attention and padding-mask models.

#### E3 — Missing-op decomposition or codegen ⛔
`sort`, `bucketize`, `multinomial`, eager FFT, sparse (csr/coo) — decompose to
existing primitives where possible; otherwise file per-op sub-items.

### Pillar F — Regression lock (continuous)

#### F1 — Consolidate milestone tests under stable names, run full GPU suite ⛔
Every A–E item lands a named test in `tests/test_inductor_regression.py`. No
`agent_space/` script as sole verification (Discipline #1).

---

## § 3 — Dependency graph

```
A1 (verify AOTI E2E) ──→ A2 (full-step .so) ──→ F1 (lock)
   └─ A3 (conv-bwd FX rewrite, independent, closes anti-goal #3)

B1 (foreach interface) ┐
B2 (rnn interface)     ├─ independent, parallel with A/C/D
B3 (validator)         ┘

C1 (overlap) ──→ flip BATCH_DISPATCH default ──→ C2 (shape bucketing)
C3 (persistent reductions) ← independent

D1 (autotune) ← independent; consumes C2's canonical keys when present

E1/E2/E3 (coverage) ← independent, breadth work
```

Parallel streams: **A** (deployment) is the critical path to "Python-less
training"; **B** and **E** are independent codegen-quality work; **C**/**D** are
the performance layer and can run in parallel once A1 confirms the path is green.

---

## § 4 — Anti-goals (durable)

1. No new model-specific `.slang` files — templates only.
2. No new `aten.<op>_backward` lowerings — backward routes through
   `bwd_diff_table.py` → Slang `bwd_diff()` / `[BackwardDerivative]`.
3. No hand-tuned shader that isn't auto-generated.
4. No symptom-fixes in `meta_patches/` that paper over a missing primitive —
   file the primitive as a roadmap item instead.
5. No string-based/Jinja template parameters for anything Slang `interface`
   generics + spec-constants + `ParameterBlock` can express. Jinja is allowed
   only for spec-constant numeric tunables and genuinely code-structural
   branches (e.g. `is_causal`).
6. No CPU fallbacks on the compile path.
7. No file in `python/torch_vulkan/inductor/` exceeds 800 lines.

## § 5 — Discipline (durable)

1. Every roadmap item names a regression test in `tests/test_inductor_regression.py`.
2. Correctness before performance. Gradient parity with CPU is the exit criterion.
3. Floor-gate-then-ratchet: land `xfail(strict=True)` first, then flip.
4. Items that turn out wrong get removed, not annotated.
5. One commit per milestone: `vulkan: <Item> — short why`.
6. Validation-driven: `TORCH_VULKAN_VUID_AS_ERROR=1` in tests; a VUID is a failure.

---

## Inductor Pipeline Integration Map

The 20 canonical pipeline stages. Bug-rooting tags every fixed item to one of
these (or a registered sub/meta-stage); the taxonomy is enforced by
`scripts/audit_stage_tags.py` + `tests/test_inductor_regression.py`. The
human-readable title is the row; the canonical kebab tag is in the last column.

| # | Stage | Where it lives | Canonical tag |
|---|-------|----------------|---------------|
| 1 | Dynamo trace | `torch/_dynamo` FX capture | `dynamo` |
| 2 | AOTAutograd graph capture | joint fwd+bwd capture | `aot-autograd-graph-capture` |
| 3 | AOTAutograd partitioner | fwd/bwd split | `partitioner` |
| 4 | FakeTensor / fake_impl registry | `meta_patches/`, op meta-kernels | `fake-impl` |
| 5 | FakeTensor propagation | shape/dtype prop | `fake-tensor-prop` |
| 6 | FX passes (pre/post-grad) | `fx_passes/` | `fx-passes` |
| 7 | Lowering | `lowerings/` | `lowering` |
| 8 | Scheduler fusion | `scheduling.py`, `combo_kernel/` | `scheduler-fusion` |
| 9 | Kernel codegen | `kernel/` | `kernel-codegen` |
| 10 | Pointwise OpsHandler overrides | `kernel/pointwise.py` | `pointwise-overrides` |
| 11 | Wrapper codegen | `wrapper.py`, `cpp_wrapper_gpu.py` | `wrapper-codegen` |
| 12 | ExternKernelChoice templates | `vulkan_template*.py`, `templates/caller/` | `externkernelchoice-templates` |
| 13 | Combo kernel | `combo_kernel/` | `combo-kernel` |
| 14 | Runtime dispatch | `runtime/`, `buffer_pool.py` | `runtime` |
| 15 | Reflection / descriptor binding | `runtime/reflection*.py` | `reflection-descriptor-binding` |
| 16 | Forward graph compile | compile-path fwd | `forward-graph-compile` |
| 17 | Backward graph compile | `bwd_diff/`, `bwd_diff_table.py` | `backward-graph-compile` |
| 18 | Optimizer-step compile | `foreach_optimizer` path | `optimizer-step-compile` |
| 19 | Measurement / autotune cache | `hardware_probe.py`, `autotune.py` | `measurement-autotune-cache` |
| 20 | Slang shader pipeline (slangc → SPIR-V) | `runtime/slangc.py`, `shaders/` | `slang-shader-pipeline` |

---

## § 6 — History & reference

- **Closed-milestone history** (v6.x → v16, M18–M23, FP16): `docs/10-inductor-backend-history.md`, `docs/archive/`.
- **Pipeline / API reference**: `docs/how-to-compile-and-codegen.md`,
  `docs/inductor-pipeline-analysis.md`, `docs/10-lib-api-reference.md`.
- **Companion CLAUDE.md**: `backends/vulkan_slang/CLAUDE.md` (build/test/env knobs/file ownership).

*This file is the single canonical roadmap. Do not fork a new numbered version —
edit this doc in place: mark items ✅ as they close, add new sub-items under the
right pillar.*
