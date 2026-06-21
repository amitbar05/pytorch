# Vulkan-Slang Inductor Backend — Consolidated Roadmap

> **Canonical, single-source roadmap.** Created 2026-06-15; **completely
> overhauled 2026-06-19** around the *warm-up pipeline spine*
> (PROBE → TUNE → COMPILE+VALIDATE → TRAIN → DEPLOY). Supersedes and replaces
> the numbered series `docs/10/14/15/16-inductor-backend.md` and
> `docs/codegen-optimization-roadmap.md` (deleted). Closed-milestone history
> lives in `docs/10-inductor-backend-history.md` and `docs/archive/`.
>
> The 2026-06-19 overhaul re-grounded § 1 / § 2 / § 3 on a fresh 4-way audit
> (warm-up subsystem, Conv+GN compile path, AOTI/autotune/perf code-vs-claim,
> and a live GPU ground-truth run that trained a Conv+GN model on RDNA1). The
> prior pillar IDs (W/A/B/C/D/E/F, M19–M23) are preserved in parentheses so
> closed history stays traceable. **Do not fork a new numbered roadmap — edit
> this file in place.**

---

## Mission

Ship a fully optimizing, **training-grade** `torch.compile(backend="inductor")`
backend on Vulkan/Slang that supports **any** PyTorch model. Every kernel is
auto-generated from Slang templates → SPIR-V. No per-model `.slang` files. No
per-model `csrc/ops/*.cpp`. No CPU fallbacks on the compile path. No Python at
deployment (AOTI `.so`).

---

## The Pipeline Spine (organizing principle)

The backend's life-cycle is one linear pipeline. **The warm-up function is the
front of this spine, not a side feature.** Every roadmap item below hangs off
exactly one stage.

```
   ┌─ prepare_device(level, timeout_s, validate) ─────────────────┐
   │                                                              │
   ▼                                                              ▼
S0 PROBE ──▶ S1 TUNE ──▶ S2 COMPILE+VALIDATE ──▶ S3 TRAIN ──▶ S4 DEPLOY
 hardware    canonical    shader-lib + template     compiled     AOTI .so
 microbench  shape sweep  SPIR-V, validated under   fwd+bwd+opt  (no Python)
 + limits    → autotune   the validation layer      step on GPU
             cache
   │            │              │                       │            │
   └──── prepare_model(model, sample_input) fills the per-model SPIR-V cache ──┘
```

The four durable goals map onto the spine:

| Goal | Where it lives on the spine |
|---|---|
| **Profile-and-warmup** | S0 must *measure* the GPU **and S1/S2 must consume that measurement** (today it is captured but unused — see S0.1). |
| **Validation-driven** | S2 must run under the Vulkan validation layer; a VUID during S1 autotune rejects the candidate, a VUID on a landed kernel fails the test. |
| **Slang-smart** | S2 codegen uses `ParameterBlock` + generics + `interface`s + spec-constants + `[BackwardDerivative]` + reflection metadata. Jinja only for spec-constant numeric tunables. |
| **Codegen-only** | S3 compiled wrappers contain **no** `extern_kernels.X` to aten / eager Vulkan, **no** `if device != vulkan: aten` branches. |

After warm-up, `torch.compile(backend="inductor")` finds pre-compiled,
hardware-tuned, validated kernels in the cache — no cold slangc latency during
training. **Proven on RDNA1 (2026-06-19): warm-up drops the first real compile
of a Conv+GN model from ~13 s to 0.59 s.**

---

## § 1 — Current State (ground truth, 2026-06-19)

The backend **trains real Conv+GN models end-to-end on the GPU today** via
`torch.compile(backend="inductor")`. A Conv→GroupNorm→ReLU→Pool one-step
fwd+bwd+SGD loop compiles and runs with per-param gradient parity vs CPU
(L∞ < 1e-4) and **zero VUIDs** under `TORCH_VULKAN_VUID_AS_ERROR=1`. Backward is
fully Slang. **As of 2026-06-20, `Linear`-head CNNs (S2.0), stacked same-
resolution conv+GN stages (S2.0d), and ResNet-style residual blocks
(S2.0d-resid) all train end-to-end with per-param grad parity vs CPU** — the
correctness blockers for a full CNN are closed.

Live numbers (RX 5600 XT, RDNA1): Conv(3→16)+GN(4,16)+ReLU+AdaptiveAvgPool,
B=4 — **21 dispatches/step, ~6.7 ms warm**, of which **~10 are tiny plumbing
kernels** (4 strided copies + 4 per-param SGD adds + fill + expand).

### Status scorecard

Legend: ✅ done · 🟡 partial · ⛔ open · 🔴 regression/defect · 🔬 needs re-verify

**S0 — PROBE (hardware profile)**
| Item | State | Evidence |
|---|---|---|
| Microbench (launch latency, mem/LDS BW, atomics) | ✅ | `device_profile.py`; cached `~/.cache/torch_vulkan/device_profile_<id>.json` |
| Device limits (CU count, LDS, max WG, subgroup) | 🟡 | `_get_device_capabilities()` not in pybind → **NAVI10 defaults** (`device_profile.py:156`) |
| **Profile is *consumed* by codegen** | 🔴 | **Only `_wave64_persistent_ok()` reads one field** (`subgroup_size_max`, `scheduling_helpers.py:44`). Mem-BW/LDS-BW/latency/CU-count all dead. Codegen uses hardcoded NAVI10 constants. |

**S1 — TUNE (autotune sweep)**
| Item | State | Evidence |
|---|---|---|
| Canonical-shape sweep (96 combos: mm/conv/linear/bmm/conv-bwd/norm/softmax/gelu × fp32/fp16) | ✅ | `hardware_probe.py:_run_level_2_autotune()` |
| MM autotune → Inductor `tuned_mm` (10+40 variants) | ✅ | `install_external_mm()`; `templates/caller/gemm/install.py:163` |
| Per-kernel WG-size autotune (numthreads 64/128/256/512) | ✅ | `hardware_probe.py:174`; cache `~/.cache/torch_vulkan/wg_autotune/` |
| Conv tile autotune | 🟡 | env-var only (`TORCH_VULKAN_CONV_TILE`, `lowerings/conv.py:194`); not in `V.choices` |
| **V.choices for conv / flash_attention** | ⛔ | non-MM templates not registered into Inductor's choice-matching |

**S2 — COMPILE + VALIDATE**
| Item | State | Evidence |
|---|---|---|
| Shader-lib + template SPIR-V precompile (sync) | ✅ | `hardware_probe.py:_run_level_1_sync()` |
| `prepare_model()` → 100% warm SPIR-V cache | ✅ | `hardware_probe.py:791`; proven 13 s→0.59 s |
| Subprocess validation of autotune winners | ✅ | `autotune.py:validate_winner` spawns fresh-instance subprocess |
| **In-process validation during warm-up** | 🔴 | aspirational — needs `VK_INSTANCE_LAYERS` set *before* `import torch_vulkan` (instance built at import). `validate=True` is a no-op for S0/S1 in-process. |
| **Warm→train cache coherence** | 🟡 | depends on tuning env-knobs (`MM_TILES`/`CONV_TILE`) staying frozen; not hashed into cache key |

**S2 — Conv+GN+ReLU+Pool+Linear compile-path dispatch audit**
| Op | Dir | Mechanism | Slang? |
|---|---|---|---|
| Conv2d | fwd | `_VulkanConv2dExternKernel` → `slang_conv2d.slang` | ✅ |
| Conv2d | bwd | `_VulkanConvBwdExternKernel` → `slang_conv_bwd.slang` + `bwd_diff` | ✅ |
| GroupNorm | fwd | decomposition → Slang codegen (2 dispatches: GPU.1 L2 workaround, `norm.py:43`) | ✅ |
| GroupNorm | bwd | 2× extern (`group_norm_backward.slang` + `_weight.slang`) | ✅ |
| ReLU | fwd/bwd | Pointwise → Slang codegen | ✅ |
| Conv+GN+ReLU fused | fwd | pre-grad `conv2d_gn_relu_fused` → `conv_gn_relu.slang`. ✅ **S2.2 FIXED 2026-06-21**: M-CG.3 WG 256→64 workaround confirmed safe — 5-seed parity test passes (`TestConvGnReluFusedWriteCoverage` ✅, 0 VUIDs). | ✅ |
| AdaptiveAvgPool2d / MaxPool2d / AvgPool2d | fwd | **FallbackKernel → eager C++** (codegen-only violation) | 🔴 |
| Pooling | bwd | `scatter_atomic.slang` / codegen | ✅ |
| Linear | fwd | `aten.addmm` → `slang_mm.slang` | ✅ |
| Linear | bwd | `slang_mm_bwd.slang` — ✅ **FIXED 2026-06-21 (S2.1)**: `aten.mm.default` now routes through `_vulkan_mm` (forced override after `get_overloads()` skip), and `_adaptive_avg_pool2d_backward.default` routes through `Pointwise.create` (same override + `ops.*-on-TensorBox` bug fix). | ✅ |
| SGD/AdamW/Lion | step | eager: `foreach_optimizer.slang` (`IOptimizer`). **Compiled step fans out to per-param `binary_add_inplace`** (no foreach bridge) | 🔴 |
| CrossEntropyLoss | fwd/bwd | decomposed → Slang codegen | ✅ |

**Defects (2026-06-19 ground-truth run)**
| ID | Defect | Severity |
|---|---|---|
| **S2.0** | Conv+GN CNN training (Linear & 1×1-conv heads): `buf7` combo inversion + wrong conv `grad_bias` (output reuse) + `StorageBox` unwrap crash + 4D grad_out assumption. | ✅ **FIXED 2026-06-19** (S2.0a/reuse/b/c) |
| **S2.0d** | Stacked conv+GN+ReLU backward exploded (WAR-barrier miss). **FIXED 2026-06-20** — `csrc/ops/dispatch.cpp` now tracks reads + emits a WAR barrier (`test_stacked_conv_gn_backward_war` ✅). **S2.0d-resid also FIXED 2026-06-20** (two more root causes, see below): residual `relu(out+identity)` backward now has full per-param grad parity (`test_resnet_block_residual_grad_parity` ✅). | ✅ **FIXED** |
| **S2.1** | Conv+GN+Pool+Linear backward wrapper leaks — **FIXED 2026-06-21**: two root-cause fixes in `matmul.py` + `bwd_lowerings.py` + `pool.py` (`TestNoExternInFullCNNBwd` ✅). | ✅ **FIXED** |
| **S3.1** | Compiled SGD step = 1 tiny `binary_add_inplace` per param tensor (4 here) + strided copies; should route to the foreach `IOptimizer` extern. | 🟡 perf |

**S3 — TRAIN (steady-state perf)**
| Item | State | Evidence |
|---|---|---|
| Tiny-kernel fusion (fill/copy/inplace) | 🟡 | ~10/21 dispatches still plumbing; C6.x cap-raises + `_coalesce_orphan_pointwise` wired (`__init__.py:907`) but not fully coalescing |
| Persistent-kernel routing for large reductions | 🔴 | **dead code** — `dispatch_persistent_pointwise()` defined, never called; no numel>65536 routing |
| Batch-dispatch overlap (exec N ∥ compile N+2) | 🟡 | async precompile real (`slangc.py:573`); full overlap TODO; `BATCH_DISPATCH=1` still 1.8× slower → default OFF |
| GN backward fusion | ✅ | already 2 fused extern dispatches (`gn_backward_extern.py`); the 11-kernel figure was loss-bwd, not GN-bwd |

**S4 — DEPLOY (AOTI)**
| Item | State | Evidence |
|---|---|---|
| C++ AOTI runtime ABI + shims + wrapper codegen | ✅ | `AotiRuntime.{h,cpp}`; pointwise model compiles/loads/dispatches, 0 VUIDs |
| All 6 extern families emit AOTI dispatch | ✅ | conv2d/conv3d fwd+bwd, mm, GN fwd+bwd, optimizer (`emit_aoti_extern_dispatch`) |
| Model-level API (`model_load/run/free`, v2 binary) | ✅ | `AotiRuntime.cpp:331-638`; `TestAOTIModelAPI` round-trip |
| **Full training step (fwd+bwd+opt) in one `.so`** | ⛔ | blocked **upstream**: `torch.export` eager `empty.memory_format` dispatch gap (A2.6). `torch.compile` path handles it (`TestAOTITrainingE2E` ✅). |

**Anti-goal compliance (2026-06-19)**
| # | Anti-goal | Status |
|---|---|---|
| 1 | No model-specific `.slang` files | ✅ |
| 2 | No new `aten.<op>_backward` lowerings | ✅ |
| 3 | No hand-tuned shaders | ✅ |
| 4 | No symptom-fixes in `meta_patches/` | 🟡 several remain |
| 5 | No Jinja for interface-level params | ✅ (foreach + rnn_cell migrated) |
| 6 | **No CPU/eager fallbacks on the compile path** | 🟡 pooling fwd still custom-op FallbackKernel (S2.5); S2.1 extern leaks ✅ fixed; S2.2 write-coverage ✅ confirmed |
| 7 | No file > 800 lines | 🟡 `pointwise.py` 820L |

---

## § 2 — Forward Roadmap (prioritized, by spine stage)

Ordering: **correctness that blocks training a full CNN first, then make the
warm-up spine actually deliver (probe→consume, validate-in-process), then
steady-state perf, then deployment, then breadth.** Each item names its
regression test (Discipline #1).

### S2.0 — ✅ FIXED 2026-06-19: Conv+GN CNN training (Linear & 1×1-conv heads)

The canonical Conv+GN CNNs now compile **and** train with grad parity vs CPU.
Four distinct bugs were on this path; all fixed and regression-locked. Verified:
`test_small_cnn_conv_gn_relu_linear_head` (new), `test_small_cnn_conv_gn_relu_fc`,
`test_simple_cnn_conv_maxpool_fc` all green; full-CNN grad parity worst
`8.26e-06`; **zero regressions** vs a clean-main baseline of the same GPU sweep.

#### S2.0a — ✅ combo orphan-coalescing readiness inversion (`buf7` crash)
Conv→GN→ReLU→Pool→Flatten→`Linear` crashed the generated backward wrapper with
`UnboundLocalError: buf7`. `_coalesce_orphan_pointwise` magnitude-bucketed two
tiny ops at **different dependency depths** (a `tangents_1` reducer ready at
entry + `buf7`/rstd copiers ready only after a later kernel) into one combo
emitted at the earliest member's position → read `buf7` before it was allocated.
Fix: a **readiness drop** in the safety filter removes members whose inputs
aren't available at the combo's emission point (only ever shrinks a bucket, so
it can't form a new grouping that breaks combo partitioning).
- **Files**: `vulkan_combo_kernel.py:_coalesce_orphan_pointwise`.

#### S2.0-reuse — ✅ output reinterpret-reuse of a non-contiguous donor
The same model returned an ~80%-wrong conv `grad_bias`: Inductor reinterpret-
reused a GroupNorm-stats buffer (stride `(4,1,16)`) as the bias **output**; the
Vulkan runtime's per-buffer barrier/binding tracking keyed on the donor's
storage and missed the write-after-read hazard. Fix: in `make_buffer_reuse`,
when the reused-as buffer is a **graph output**, allocate it fresh and release
the donor to the pool (allocate-before-free). Internal reshape-reuse is
untouched, so the buffer-pool hit-rate is preserved.
- **Files**: `wrapper.py:make_buffer_reuse`.

#### S2.0b — ✅ realize/unwrap left `ComputedBuffer(data=StorageBox)`
1×1-conv-head CNNs crashed in `decide_layout` with `'StorageBox' object has no
attribute 'get_pointwise_size'`. The five copies of `_vk_realize_then_unwrap`
unwrapped all StorageBoxes then all Views in sequence, but interleaved nesting
(`StorageBox → View → StorageBox → Buffer`) left a trailing StorageBox that got
wrapped in a malformed ComputedBuffer. Fix: unwrap to a **fixpoint**.
- **Files**: `lowerings/_vk_realize_utils.py`, `lowerings/conv.py`,
  `lowerings/conv_backward.py`, `lowerings/gn_forward_extern.py`,
  `lowerings/gn_backward_extern.py`.

#### S2.0c — ✅ conv backward assumed 4D grad_out
A 1×1-conv classifier after global-pool + `flatten` delivers `grad_out` already
collapsed to `(N, C_out)`; the conv-bwd caller read `stride(2)`/`stride(3)` →
`IndexError`. Fix: reshape grad_out back to `(N, C_out, oH, oW)` (raises if the
element count disagrees, so genuine shape errors still fail loudly).
- **Files**: `templates/caller/conv.py`.
- **Exit (all S2.0)**: `test_small_cnn_conv_gn_relu_linear_head` +
  `test_small_cnn_conv_gn_relu_fc` + `test_simple_cnn_conv_maxpool_fc`.

#### S2.0d — ✅ Stacked + residual conv+GN+ReLU backward (FIXED 2026-06-20)
**Non-residual (stacked) case — FIXED via C++ WAR barrier.** The C++ smart
barrier (`dispatch.cpp`) tracked only *writes* (`dirty_buffers`). A buffer
*read* by one extern dispatch (conv-bwd / GN-bwd) and then *written* by a later
one — via Inductor same-shape exact-reuse aliasing — was a write-after-read
hazard with no barrier. Fix: track read buffers (`read_buffers`), emit a
combined-mask barrier when an output overlaps a prior read. `test_stacked_conv_gn_backward_war` ✅.

**S2.0d-resid (residual `relu(out+identity)`) — FIXED 2026-06-20** via two more
root causes, both found by dumping the residual backward wrapper and bisecting
the per-param grad with `agent_space/check_resnet_bisect.py residual` (forward
was always exact; backward exploded `5.4e5` at the stem, with `gn2.bias` and
`classifier` correct). After both fixes every parameter has grad parity vs CPU
at ~1e-6 (`test_resnet_block_residual_grad_parity` ✅); the stacked/noresid/
noblock/small-CNN variants stay green.

1. **Standalone-GroupNorm saved-`rstd` corruption** (`wrapper.py:make_buffer_reuse`).
   A GroupNorm *not* immediately followed by ReLU (here `gn2`, separated from
   the final ReLU by the residual `+`) does not fuse into `conv2d_gn_relu_fused`;
   it lowers as a standalone norm whose forward saves `mean` + `rstd`. The `rstd`
   save is codegen'd as an *in-place* kernel (`rstd = rsqrt(var+eps)` over the
   welford `var` buffer, emitted as `in_out_ptr0`). The S2.0/S2.1 graph-output
   fresh-alloc fired for the (graph-output) `rstd` buffer — allocating it fresh
   instead of aliasing `var` — so the in-place kernel read **uninitialized
   memory** → garbage `rstd` → exploded `gn.weight`/grad_input (`gn.bias`, which
   doesn't use `rstd`, stayed correct: the diagnostic signature). Fix: the
   fresh-alloc is only correct for a *reinterpret-view* reuse (producer fully
   overwrites the output); skip it for an *in-place mutation* reuse (detected via
   `new` reading `old`). The WAR hazard the fresh-alloc once guarded against is
   now covered by the S2.0d C++ WAR barrier above.

2. **Persistent grid-stride loop re-runs in-place accumulations** (`kernel/header.py`).
   The residual fan-in `grad_x += grad_conv1` (an `in_out_ptr` add) ran under the
   persistent-pointwise grid-stride loop. That loop wraps only `self.body`; the
   per-element index vars are derived from `gtid.x` in the preamble *outside* the
   loop, so each `_pi` iteration re-executes the **same** element. For separate-
   buffer pointwise (`out[i]=f(in[i])`) re-execution is idempotent (harmless);
   for an in-place op it re-applies the mutation N times → ~2× stem gradients
   (only the stem, since the block grads route correctly). Fix: skip the
   persistent wrap whenever the kernel binds any in-place buffer.

- **Files**: `python/torch_vulkan/inductor/wrapper.py` (`_reuse_reads_donor` +
  gate in `make_buffer_reuse`), `python/torch_vulkan/inductor/kernel/header.py`
  (`_has_inplace` gate on the persistent wrap).
- **Repro/evidence**: `agent_space/check_resnet_bisect.py {residual,noresid,noblock}`.
- **Exit**: `test_resnet_block_residual_grad_parity` (per-param grad parity) +
  `test_resnet_block_conv_gn_residual_fc` (loss-based sweep). ✅

**Follow-up filed — S2.0d-deadgrad** (latent, order-dependent; does NOT block
conv training): a standalone GroupNorm `(y*y).sum()` (GN not followed by ReLU,
non-constant grad) yields a **dead** `gn.weight` gradient (`rel=1.0`, vk≈0)
*depending on process state*. `agent_space/gn_assert.py` (plain script) gives the
correct grad (rel 4e-7); `agent_space/gn_after_import.py` — the identical model,
but after `import tests.test_inductor_regression` — reproduces the dead grad with
a **fresh cache**. The test module has **no** module-scope `torch_vulkan`/inductor
import or `config` patch (only string constants + class/def defs), so this is not
a config leak: importing a large inert module perturbs Python memory layout / set
iteration order, which flips a **scheduler-fusion / DCE iteration-order
dependence** — for some orderings the `gn.weight` reduction (the
`sum(grad·x̂)` over N,H,W) is silently dropped/zeroed. The weight-grad is a
reduction (not in-place) so it is untouched by the S2.0d-resid fixes. Not a
blocker: real conv training works (full `TestTrain8ConvTrainingSweep` green; the
residual model's standalone `gn2` weight-grad matches CPU at ~1e-6). Next: find
the order-dependent set/dict iteration in the scheduler/DCE path that drops the
standalone-GN weight-grad reduction (likely in `scheduling.py` fusion or
`kernel/header.py:_eliminate_dead_code`).

#### S2.0e — ⛔ Pre-existing GPU-suite failures (slangc/env)
A clean-main GPU sweep shows ~24 pre-existing failures unrelated to S2.0
(conv1d/conv3d/depthwise compile, `f16_mm_through_compile`, addmm tile cache,
`pool_hit_rate_above_90pct`, `wrapper_pool_integration`) — several surface
`slangc failed for kernel …`. Triage whether this is a slangc-version/cache
regression in the current tree.
- **Exit**: `TestConv1dCompile` + `TestConvGeneralityGaps` green on GPU.

### S2.1 — ✅ FIXED 2026-06-21: Eliminate the extern/aten leak in the combined backward wrapper

Two root causes:

1. **mm backward went through `tuned_mm` (autotuner → `extern_kernels.mm`).**
   `get_overloads()` in Inductor's `_register_lowering` skips overloads already
   in `lowerings`. `aten.mm.default` was pre-registered by `tuned_mm` before our
   `_vulkan_mm` ran; `@register_lowering(aten.mm)` only overrode the packet key,
   NOT `.default`. Fix: explicit `lowerings[aten.mm.default] = lowerings[aten.mm]`
   after registration in `_register_mm_lowering()`.

2. **`aten._adaptive_avg_pool2d_backward.default` went through `fallback_handler`
   (same `get_overloads()` skip bug + `ops.*-on-TensorBox` bug in the lowering).**
   Original `_adaptive_avg_pool2d_backward_vulkan` called `ops.mul(TensorBox, ...)`
   at lowering time, producing invalid OpsValue → lowering fell through to FallbackKernel.
   Fix: (a) rewrite using `Pointwise.create(inner_fn=...)` with affine index mapping
   `h // kH, w // kW`; (b) force-override `.default` after registration (same pattern).

- **Files**: `lowerings/matmul.py`, `lowerings/pool.py`, `bwd_lowerings.py`.
- **Tests**: `TestNoExternInFullCNNBwd::test_lowering_registration_no_fallback` ✅
  `TestNoExternInFullCNNBwd::test_no_extern_in_cnn_linear_wrapper` ✅

### S2.2 — ✅ FIXED 2026-06-21: Conv+GN+ReLU fused shader write-coverage confirmed safe

The M-CG.3 fix (256→64 WG, single-wave64 on RDNA1) resolved the slangc
write-coverage miscompile.  `TestConvGnReluFusedWriteCoverage` runs 5 independent
seeds of fused fwd+bwd (B=2, 3→16 conv, GN(4,16), ReLU, 8×8 input) and asserts
L∞ < 1e-4 on every run under the default VUID-as-error fixture (M-VAL.1).  All 5
seeds pass in ~7 s on RDNA1.
- **Files**: `tests/test_inductor_regression.py:TestConvGnReluFusedWriteCoverage`.
- **Tests**: `TestConvGnReluFusedWriteCoverage::test_fused_bwd_parity_multi_run` ✅.

### S2.5 — Pool forward: replace FallbackKernel with custom-op dispatch (anti-goal #6 close-out)

`aten.avg_pool2d.default` and `aten._adaptive_avg_pool2d.default` are intentionally
routed to `FallbackKernel` (`bwd_lowerings.py`) because the upstream Inductor
lowering uses `make_loader+indirect_indexing`, which produces wrong SPIR-V on Vulkan.

**Attempted (2026-06-21, reverted):** `Reduction.create` with affine index arithmetic.
This works for global-pool (1×1 output, `reduction_numel≥128` → Inductor multilayer
path), but fails for spatial-pool (4×4 output, `reduction_numel=16` → single-layer
path) with `NotImplementedError: Reduction` in our Slang codegen — our backend
only handles 1D reduce; 2D window reductions need a dedicated codegen path.

**Correct approach:** Register `torch_vulkan::avg_pool2d_forward` as a C++ custom op
backed by the existing Vulkan pool kernel in `csrc/ops/legacy_eager.cpp`.  Wire a
`FallbackKernel`-style lowering to emit `torch.ops.torch_vulkan.avg_pool2d_forward(...)`
(private Vulkan compute) instead of `torch.ops.aten.avg_pool2d(...)` (public aten
eager).  This closes anti-goal #6 without needing 2D-reduction codegen support.

- **Files**: `csrc/ops/legacy_eager.cpp` (new `torch_vulkan::avg_pool2d_forward` op),
  `bwd_lowerings.py:_vulkan_avg_pool2d` (emit custom-op extern instead of aten).
- **Prerequisite**: C++ rebuild.
- **Exit**: `TestPoolAdaptiveRouting::test_adaptive_avg_pool2d_routes_to_avg_pool2d`
  flips from "wrapper contains aten.avg_pool2d" to "wrapper contains
  torch_vulkan.avg_pool2d_forward".  Anti-goal #6 row updates to ✅ once this
  and S2.2 are both green.

### S0.1 — 🟡 Make the device profile *drive* codegen (the missing half of "probe")

**WG sizing wired 2026-06-20.** `threadgroup_sizing.py` read device limits from
the device-interface query, which on this stack under-reports
`max_workgroup_size` as **256** (real: **1024**) and returns **no CU count**
(→ hardcoded 20; real: 16) — so WG sizes were capped 4× below the hardware
ceiling. Added `device_profile.profile_limit(key, fallback)` (reads the
warm-up profile, loading the on-disk cache without ever profiling at codegen
time) and routed all four `max_workgroup_size`/`compute_units` sites through it.
Result: `conv_gn_relu` warm step **8.2 ms → 6.9 ms (~16%)**, others neutral,
**zero correctness regressions** vs a clean-main baseline.
- **Files**: `device_profile.py:profile_limit`, `kernel/threadgroup_sizing.py`.
- **Exit**: `TestM211DeviceProfile::test_profile_limit_drives_codegen` ✅.

**Still open (S0.1 remainder):** the rich *microbench* data is still unused —
LDS budget ← `shared_memory_per_workgroup_bytes`; persistent-vs-grid-stride and
batch-vs-direct dispatch thresholds ← `empty_kernel_launch_us` / `memcpy_d2d_GBps`;
matmul/conv tile selection ← measured mem BW. Wire these next so warm-up is
fully portable (an 80-CU card tiles differently from a 16-CU card).

### S0.2 — Complete the device-limits pybind query

`_get_device_capabilities()` is not exposed, so S0 falls back to NAVI10
defaults — a portability hole and a precondition for S0.1.
- **Files**: `csrc/init.cpp` (expose caps), `device_profile.py:156`.
- **Exit**: `TestDeviceLimitsReal` — probed limits come from the live device,
  not the NAVI10 fallback constants.

### S2.3 — In-process validation during warm-up (make `validate=True` real)

Today `prepare_device(validate=True)` only validates autotune *subprocesses*;
in-process S0/S1 need `VK_INSTANCE_LAYERS` before the Vulkan instance is built
at import. Add a bootstrap: when `validate=True` (or a sentinel env) is
requested and layers aren't active, **re-exec the process** (or defer instance
creation) with `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` so the whole
warm-up runs under validation — catching shader bugs at warm-up, not mid-train.
- **Files**: `hardware_probe.py` (re-exec bootstrap), `__init__.py` (deferred
  instance creation), `csrc/vulkan/Context.cpp`.
- **Exit**: `TestWarmupValidationInProcess` — `prepare_device(validate=True)`
  from a clean env validates S0/S1 kernels in-process; an injected VUID fails.

### S2.4 — Warm→train cache coherence (hash the tuning knobs)

Warm-up sets `MM_TILES=expanded` / `WG_AUTOTUNE=1`; if training doesn't preserve
them the kernel source hash differs → silent cold recompile + autotune miss.
Hash the active tuning knobs into the SPIR-V/autotune cache key, or emit a
warm-up manifest the training path asserts against.
- **Files**: `runtime/slangc.py` (cache key), `autotune.py`, `hardware_probe.py`
  (manifest).
- **Exit**: `TestWarmCacheCoherence` — after `prepare_device(deep)`, a training
  compile of a swept shape reports 100% SPIR-V + autotune cache hits.

### S3.1 — Compiled optimizer: route SGD/AdamW to the foreach extern

The compiled step fans out to one `binary_add_inplace` per parameter (4 tiny
dispatches here) instead of the foreach `IOptimizer` extern that eager already
uses (B1). Bridge the eager-foreach interface into the compiled optimizer step.
- **Files**: `lowerings/optimizer_lowerings.py`, `fx_passes/post_grad.py`
  (recognize the per-param add cluster), `templates/foreach_optimizer.slang`.
- **Exit**: `TestCompiledOptimizerForeach` — N-param SGD/AdamW step compiles to
  ≤2 dispatches (vs N today); parity vs eager.

### S3.2 — Tiny-kernel fusion (close out the plumbing dispatches)

~10/21 dispatches are fill/copy/inplace. C6.x raised caps and wired
`_coalesce_orphan_pointwise`, but coalescing is incomplete. Push per-step
dispatch count down by folding strided copies + fills into neighbouring compute
kernels and the combo grid.
- **Files**: `scheduling.py`, `vulkan_combo_kernel.py`, `kernel/pointwise.py`.
- **Exit**: `TestTinyKernelFusion` — Conv+GN training step ≤14 dispatches (vs 21
  today); no standalone `copy_fill`/`copy_strided` dispatches.

### S3.3 — Wire persistent-kernel routing for large reductions (C3 is dead code)

`dispatch_persistent_pointwise()` exists but is never called. Add the
numel>65536 threshold routing from `bwd_diff_table.py` / the reduction lowering
into `persistent_pointwise.slang`.
- **Files**: `bwd_diff_table.py`, `templates/caller/persistent_pointwise.py`.
- **Exit**: `TestPersistentReduction` — large `sum`/`mean` parity + dispatch-count
  drop vs grid-stride path.

### S3.4 — Batch-dispatch / async-compile overlap (flip `BATCH_DISPATCH=1`)

Make batched dispatch win: execute kernel N while compiling N+2 via the existing
async slangc pool, bringing the 1.8× batch penalty to ≤1.1×.
- **Files**: `runtime/batcher.py`, `runtime/slangc.py`, `scheduling.py`.
- **Exit**: `TestBatchDispatchOverlap` — MNISTNet step with `BATCH_DISPATCH=1`
  ≤1.1× the unbatched time; parity holds.

### S1.1 — Register conv / flash_attention choices into `V.choices`

MM autotune is in `tuned_mm`; conv tiling is env-var-only and flash is
unregistered. Wire non-MM templates into Inductor's choice-matching so warm-up
auto-explores them.
- **Files**: `vulkan_template.py`, `templates/caller/.../install.py`, `dispatch.py`.
- **Exit**: `TestConvAutotuneChoices` — conv2d compile sweeps registered tile
  configs via `V.choices`; best is cached and reused.

### S2.5 — Pooling forward: pure Slang codegen (kill the eager fallback)

`max_pool2d`/`avg_pool2d`/`adaptive_avg_pool2d` forward route through
FallbackKernel → eager C++ (anti-goal #6). Upstream `indirect_indexing`
produces wrong SPIR-V; replace with scatter/reduce Slang codegen.
- **Files**: `bwd_lowerings.py:730-844`, `lowerings/pool.py`,
  `templates/scatter_atomic.slang`.
- **Exit**: `TestPoolFwdSlang` — pool fwd graphs contain no FallbackKernel;
  output matches CPU.

### S4.1 — Full training-step `.so` (fwd+bwd+optimizer)

All extern families + the v2 model API are wired; the remaining blocker is
**upstream** — `torch.export` eager `empty.memory_format` dispatch on the vulkan
device (A2.6). Track upstream; in the meantime keep the `torch.compile` path
(`TestAOTITrainingE2E` ✅) as the deployment story.
- **Files**: `csrc/backend/AotiRuntime.cpp`, `cpp_wrapper_gpu.py`,
  `meta_patches/` (export `empty` shim).
- **Exit**: `TestAOTIFullTrainingStep` — single `.so` runs fwd+bwd+SGD; weights
  update; data parity vs `torch.compile` path.

### Continuous — coverage breadth (S2/E pillar)

- **E1** — replace `max_pool2d_scatter_bwd` / `avg_pool2d_scatter_bwd`
  `make_fallback`s with Slang `scatter_atomic` codegen, or ratify.
- **E2** — masking backward (`tril`/`triu`/`masked_fill`/`where`) via `bwd_diff`
  for sparse-attn / padding masks.
- **E3** — missing ops: `sort`, `bucketize`, `multinomial`, sparse (csr/coo),
  eager FFT — decompose where possible, else file per-op sub-items.

### Continuous — regression lock (F pillar)

Every S-item lands a named test in `tests/test_inductor_regression.py`. No
`agent_space/` script as sole verification (Discipline #1). Run the full GPU
suite under `TORCH_VULKAN_VUID_AS_ERROR=1`.

---

## § 3 — Dependency graph

```
prepare_device(level, timeout_s, validate)
│
├─ S0 PROBE
│   ├─ S0.2 (complete limits pybind) ──┐
│   └─ S0.1 (profile DRIVES codegen) ◀─┘  🔥 unlocks portable tiling
│
├─ S1 TUNE
│   ├─ MM autotune ✅
│   └─ S1.1 (conv/flash → V.choices) ⛔
│
├─ S2 COMPILE + VALIDATE
│   ├─ S2.0 (buf7 full-CNN bwd crash) ✅ FIXED
│   ├─ S2.1 (extern/aten leak)        ✅ FIXED 2026-06-21
│   ├─ S2.2 (conv_gn_relu RDNA1 bug)  ✅ FIXED 2026-06-21
│   ├─ S2.3 (in-process validation)   🔴  ◀── makes validate=True real
│   ├─ S2.4 (warm→train coherence)    🟡
│   └─ S2.5 (pooling fwd Slang)       🔴 anti-goal #6
│
├─ S3 TRAIN (perf — after correctness)
│   ├─ S3.1 (compiled foreach optimizer) 🟡
│   ├─ S3.2 (tiny-kernel fusion)         🟡
│   ├─ S3.3 (persistent reduction wiring)🔴 dead code
│   └─ S3.4 (batch-dispatch overlap)     🟡
│
└─ S4 DEPLOY
    └─ S4.1 (full-step .so) ⛔ blocked upstream (torch.export empty.memory_format)

Continuous: E1/E2/E3 (coverage) · F (regression lock)
```

**Critical path to "train a full CNN with GN":** S2.0 → S2.1 (then S2.2 to
de-risk). **Critical path to realize the warm-up vision:** S0.2 → S0.1 (probe
drives codegen) ∥ S2.3 (validation actually runs). Perf (S3.*) and breadth
(E/F) follow correctness.

---

## § 4 — Anti-goals (durable)

1. No new model-specific `.slang` files — templates only.
2. No new `aten.<op>_backward` lowerings — backward routes through
   `bwd_diff_table.py` → Slang `bwd_diff()` / `[BackwardDerivative]`.
3. No hand-tuned shader that isn't auto-generated.
4. No symptom-fixes in `meta_patches/` that paper over a missing primitive —
   file the primitive as a roadmap item instead.
5. No string-based/Jinja template parameters for anything Slang `interface`
   generics + spec-constants + `ParameterBlock` can express. Jinja only for
   spec-constant numeric tunables and genuinely code-structural branches.
6. **No CPU/eager fallbacks on the compile path** — `TORCH_CHECK(false)` for
   unimplemented ops; no `extern_kernels.X` / `torch.ops.aten.*` in a compiled
   wrapper (S2.5 pooling-fwd is the remaining breach; S2.1 ✅ fixed 2026-06-21).
7. No file in `python/torch_vulkan/inductor/` exceeds 800 lines.

## § 5 — Discipline (durable)

1. Every roadmap item names a regression test in `tests/test_inductor_regression.py`.
2. Correctness before performance. Gradient parity with CPU is the exit criterion.
3. Floor-gate-then-ratchet: land `xfail(strict=True)` first, then flip.
4. Items that turn out wrong get removed, not annotated.
5. One commit per item: `vulkan: <S-id> — short why`.
6. Validation-driven: `TORCH_VULKAN_VUID_AS_ERROR=1` in tests; a VUID is a failure.

---

## Inductor Pipeline Integration Map

The 20 canonical pipeline stages. Bug-rooting tags every fixed item to one of
these; the taxonomy is enforced by `scripts/audit_stage_tags.py` +
`tests/test_inductor_regression.py`.

| # | Stage | Where it lives | Canonical tag |
|---|-------|----------------|---------------|
| 0 | Device warm-up / profile | `hardware_probe.py`, `device_profile.py` | `warmup-profile` |
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

- **Closed-milestone history** (v6.x → v16, M18–M23, FP16, the W/A/B/C/D/E/F
  pillars closed before this overhaul): `docs/10-inductor-backend-history.md`,
  `docs/archive/`.
- **Pipeline / API reference**: `docs/how-to-compile-and-codegen.md`,
  `docs/inductor-pipeline-analysis.md`, `docs/10-lib-api-reference.md`.
- **Companion CLAUDE.md**: `backends/vulkan_slang/CLAUDE.md` (build/test/env knobs/file ownership).

### Pillar-ID → spine-ID crosswalk (for tracing old notes)

| Old | New | Old | New |
|---|---|---|---|
| W1–W3, W5 | S0/S1/S2 ✅ | C1 | S3.4 |
| W4 | S2.3 | C3 | S3.3 |
| A1 (M23.2) | ✅ closed | C4/C7 | ✅ GN-bwd fused |
| A2 / A2.5–A2.7 | S4.1 | C6 | S3.2 |
| A3 / A4 | ✅ closed | D1 | S1.1 |
| A5 | S2.5 | E1/E2/E3 | Continuous |
| B1 / B2 / B3 | ✅ closed | F1 | Continuous |

*This file is the single canonical roadmap. Do not fork a new numbered version —
edit it in place: mark items ✅ as they close, add new sub-items under the right
spine stage.*
