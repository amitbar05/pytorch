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

## § 1 — Current State (ground truth, 2026-06-16)

The backend **trains real models end-to-end on the GPU today** via the
`torch.compile(backend="inductor")` path. The Conv+GN+ReLU+Pool+Linear+
CrossEntropyLoss+SGD training loop compiles and runs with per-param gradient
parity vs CPU (rel_diff < 5e-3). Backward path is fully Slang. Forward path
is ~80% Slang (conv, Linear, GN, ReLU, CrossEntropy) with pooling forward
using FallbackKernel (eager C++). Optimizer uses Slang `IOptimizer` interface.

### Status scorecard

Legend: ✅ done · 🟡 partial · ⛔ open · 🔬 needs re-verification

**Warm-up / device profiling (pillar W)**
| Item | State | Evidence |
|---|---|---|
| W1: Hardware microbenchmark (launch latency, mem/LDS BW, atomics, limits) | ✅ | `device_profile.load_or_profile()`; cached at `~/.cache/torch_vulkan/` |
| W2: Shader-lib + matmul template SPIR-V precompile | ✅ | `hardware_probe.py:_run_level_1_sync()`; on-disk SPIR-V cache |
| W3: Canonical-shape autotune sweep (mm + conv2d shapes × dtypes) | ✅ | `_run_level_2_autotune()`; populates `~/.cache/torch_vulkan/autotune/` |
| W4: Vulkan validation layer during warm-up (catch bugs at warm-up, not training) | ⛔ | Warm-up runs without validation by default; need `VK_INSTANCE_LAYERS` opt-in |
| W5: Per-model warm-up (`prepare_model(model, sample_input)` → 100% SPIR-V cache) | ⛔ | No model-targeted warm-up yet; only canonical shapes |

**AOT / deployment (AOTI)**
| Item | State | Evidence |
|---|---|---|
| C++ AOTI Runtime ABI (`AotiRuntime.{h,cpp}` — make/dispatch/destroy + T7.4) | ✅ | 8 symbols exported; tested via `TestAotiCppLoader` (5/6 pass, 1 fixed this session) |
| Link `aoti_shims.o` into wrapper `.so` (14 `aoti_torch_*` symbols) | ✅ | `setup.py:133`, `cpp_wrapper_gpu.py` |
| C++ wrapper codegen registered (`VulkanCppWrapperGpu` in `device_codegens`) | ✅ | `__init__.py:828-835`; codegen verified (T7.2 tests pass) |
| Import does not hang (`meta_patches` lazy) | ✅ | `meta_patches/__init__.py` |
| Clean process exit (`shutdown(wait=False)` at atexit) | ✅ | `runtime/common.py:373-379` |
| **Conv+GN+pool+linear training E2E + grad parity** (`torch.compile` path) | ✅ | `TestAOTITrainingE2E` — **FIXED 2026-06-16 (M23.2)** |
| **PF.60**: RecursionError in tensor_str during AOTI compile | ✅ | Monkey-patch works; AOTI C++ wrapper compiles successfully |
| **PF.30.e**: FunctionalTensor view ops crash | ✅ | Null-storage guard catches FakeTensors; AOTI compile passes |
| **AOTI C++ wrapper codegen**: Slang→SPIR-V + emit AOTI dispatch ABI | ✅ | **FIXED 2026-06-16** — `.so` compiles, links 3 AOTI symbols, 0 VUIDs |
| **AOTI runner dispatch**: AOTIModelContainerRunnerCpu.run() | ✅ | **FIXED 2026-06-17** — vulkan.h static inline shadow resolved; both tensors on vulkan:0 |
| AOTI `.so` fwd+bwd+optimizer full step, data correctness | 🟡 | Pointwise fwd verified; extern-kernel codegen blocker (A2.5) — conv/GN/Linear emit Python syntax in C++ wrapper |
| Model-level AOTI API (`model_load/run/free`) | 🟡 | Stub implementation: single-kernel dispatch, no per-kernel buffer layouts |

**Training correctness (the M19–M23 / FP16 line, recently active)**
| Item | State | Evidence |
|---|---|---|
| Linear-backward decomposition (mm+mm+sum, no eager extern) | ✅ | M-CG.4 / M19.1 |
| grad_weight / grad_bias zero-init allocator + `copy_` path | ✅ | commit `60541e0e1e8` (M23.1) |
| Conv backward fp16→fp32 upcast in template caller | ✅ | commit `3444718dd33` (FP16.1) |
| Conv backward routes through `_VulkanConvBwdExternKernel` → `_slang_tile_conv2d_bwd` (Slang `bwd_diff(conv_inner_madd)`) | ✅ | A3 ratified 2026-06-16 — `TestConvBwdNoExtern` |

**Compile-path dispatch audit — Conv+GN+ReLU+Pool+Linear training (2026-06-16)**
| Op | Direction | Mechanism | Slang? |
|---|---|---|---|
| Conv2d | fwd | `_vulkan_aten_convolution` → `_VulkanConv2dExternKernel` → `slang_conv2d.slang` | ✅ Slang (A4) |
| Conv2d | bwd | `_VulkanConvBwdExternKernel` → `_slang_tile_conv2d_bwd` → `slang_conv_bwd.slang` + `bwd_diff` | ✅ Slang |
| GroupNorm | fwd | Inductor decomposition (var_mean/reduction/pointwise) → Slang codegen | ✅ Slang |
| GroupNorm | bwd | ExternKernelOut → `group_norm_backward.slang` / `_weight.slang` | ✅ Slang |
| ReLU | fwd | Pointwise → Slang codegen | ✅ Slang |
| ReLU | bwd | Pointwise → Slang codegen | ✅ Slang |
| AdaptiveAvgPool2d | fwd | FallbackKernel → eager C++ Vulkan | 🟡 FallbackKernel (A5) |
| AdaptiveAvgPool2d | bwd | Inductor decomposition (broadcast+scale+slice) → Slang codegen | ✅ Slang |
| MaxPool2d | fwd | FallbackKernel → eager C++ Vulkan | 🟡 FallbackKernel (A5) |
| MaxPool2d | bwd | `torch_vulkan::max_pool2d_scatter_bwd` → `scatter_atomic.slang` | ✅ Slang |
| AvgPool2d | fwd | FallbackKernel → eager C++ Vulkan | 🟡 FallbackKernel (A5) |
| AvgPool2d | bwd | Codegen (non-overlapping) / `scatter_atomic.slang` (overlapping) | ✅ Slang |
| Linear | fwd | `aten.addmm` → `slang_mm.slang` template | ✅ Slang |
| Linear | bwd | `slang_mm_bwd.slang` template | ✅ Slang |
| SGD/AdamW/Lion | step | `foreach_optimizer.slang` with Slang `IOptimizer` interface (B1) | ✅ Slang |
| CrossEntropyLoss | fwd/bwd | Decomposed → pointwise/reduction → Slang codegen | ✅ Slang |
| Conv+GN+ReLU fused | fwd | Pre-grad fusion → `torch_vulkan::conv2d_gn_relu_fused` → `conv_gn_relu.slang` ⚠️ | 🟡 known RDNA1 bugs |

**Gap summary**: Pooling forward (max/avg/adaptive) uses FallbackKernel → eager C++ Vulkan (A5 partial). The fused conv_gn_relu forward shader has known write-coverage bugs on RDNA1. Backward path is fully Slang. Optimizer uses Slang interface (B1 ✅).

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
| **`foreach_optimizer` algorithm → Slang interface** | ✅ | `templates/foreach_optimizer.slang` uses `IOptimizer` + generics `<Algo : IOptimizer>`; entry point via `computeMain<AdamWImpl>` |
| **`rnn_cell*` direction → interface; cell_type (lstm/gru) Jinja** | ⛔ | `rnn_cell.slang:35-47` Jinja `cell_type` + `direction` branches (cell_type was a gap in v16) |
| **AST validator: spec-constant pass** | ✅ | `slang_validate/spec_constants.py` (189 L) |
| **AST validator: bwd_diff signature matching** | ✅ | `bwd_diff_scan.py` validates param count/types: DifferentialPair mapping, no_diff preservation, output grad check (B3) |

**Performance (PERF)**
| Item | State | Evidence |
|---|---|---|
| Batcher `ready_set` per-batch flush (correctness) | ✅ | `runtime/batcher.py:55-152` |
| **Batch dispatch is correct but 1.8× slower** → default OFF | ⛔ | `config.py:119-123`; the payoff needs compile/exec overlap (below) |
| **Async-compile double-buffer overlap** (exec kernel N while compiling N+2) | 🟡 | async pool exists (`slangc.py:544`), overlap not wired into flush path |
| **Shape bucketing** in template registry (canonicalize → cache SPIR-V) | ⛔ | `template_registry.py` has a `shape_class` field only; no canonicalization |
| **Persistent kernel routing** for large reductions (numel>65536) | 🟡 | `persistent_pointwise.slang` exists; not wired in `bwd_diff_table.py` |
| **GN backward kernel fusion** (11 tiny kernels → 1-2 fused) | ⛔ | Profiled 2026-06-18; 11 dispatches for sub/mul/sum/fill/expand/pow; ~2-3ms overhead |
| **Conv backward fwd-recomputation elimination** | 🟡 | C5.1: added `TORCH_VULKAN_DISABLE_CONV_GN_FUSION` gate for experimentation; warm-rerun may hang |
| **Tiny-kernel fusion** (fill/copy/inplace 13 dispensers, 45% of total) | 🟡 | C6.1 rnumel cap↑ + C6.2 removed 5 redundant .zero_() dispatches in GN/conv3d bwd |
| **GN backward Slang-extern** (single shader vs 11 pointwise ops) | ⛔ | Single-dispatch GN backward like conv_gn_relu fwd; eliminates intermediates |

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

**Anti-goal compliance (2026-06-16)**
| # | Anti-goal | Status | Notes |
|---|---|---|---|
| 1 | No model-specific `.slang` files | ✅ | All templates; no per-model files |
| 2 | No new `aten.<op>_backward` lowerings | ✅ | All bwd through bwd_diff_table.py or ExternKernelOut |
| 3 | No hand-tuned shaders | ✅ | All auto-generated from templates |
| 4 | No symptom-fixes in meta_patches | 🟡 | Several remain — see meta_patches/ audit |
| 5 | No string-based/Jinja for interface-level params | ✅ | foreach_optimizer uses IOptimizer; rnn_cell remaining (B2) |
| 6 | No CPU fallbacks on compile path | ✅ | TORCH_CHECK(false) for unimplemented ops |
| 7 | No file > 800 lines in python/torch_vulkan/inductor/ | 🟡 | pointwise.py at 820L; most others in 700-800 band |

---

## § 2 — Forward Roadmap (open work, prioritized)

Ordering principle: **warm-up before compile, correctness/deployment before
performance before coverage breadth.** Each item names its regression test
(Discipline #1).

### Pillar W — Warm-up pipeline (pre-compile prerequisite)

The warm-up phase runs once at process start, **before** any `torch.compile`
call. It profiles the specific GPU hardware, compiles and caches optimized
SPIR-V for canonical op shapes, and validates all shaders against the Vulkan
validation layer. After warm-up, `torch.compile(backend="inductor")` finds
pre-compiled, hardware-validated, autotuned kernels in the cache — no cold
slangc latency during training.

This pillar formalizes the **Profile-and-warmup** durable goal (line 29) as
an explicit first-class pipeline stage: **warm-up → compile (AOT) → train**.

The canonical entry point is `torch_vulkan.prepare_device(level, timeout_s)`.
Levels: `"quick"` (~5 s, microbench only), `"medium"` (~30 s warm / minutes
cold, +shader-lib +matmul SPIR-V), `"deep"` (~3 min warm / up to 15 min cold,
+canonical-shape autotune sweep over mm + conv2d shapes × dtypes).

#### W1 — Hardware microbenchmark ✅
`device_profile.load_or_profile()` captures launch latency, mem BW, LDS BW,
atomic throughput, device limits. Cached at `~/.cache/torch_vulkan/device_profile_<id>.json`.
- **Files**: `hardware_probe.py:_run_level_0()`, `device_profile.py`
- **Exit**: `TestWarmupMicrobench` — profile loads from cache on second call.

#### W2 — Shader-lib + matmul template SPIR-V precompile ✅
Synchronous precompilation of shader library modules (`shaders/lib/*.slang`)
and matmul template SPIR-V. Populates the on-disk SPIR-V cache so the first
compiled wrapper doesn't pay slangc cold-compile latency.
- **Files**: `hardware_probe.py:_run_level_1_sync()`
- **Exit**: `TestWarmupShaderLib` — SPIR-V cache populated; second compile
  reuses cached modules.

#### W3 — Canonical-shape autotune sweep ✅
Runs `a @ b` and `F.conv2d(x, w, b)` through `torch.compile(backend="inductor")`
at a grid of shapes × dtypes (fp32, fp16) to populate the per-kernel WG-size
autotune cache (`~/.cache/torch_vulkan/autotune/*.json`).
- **Files**: `hardware_probe.py:_run_level_2_autotune()`
- **Exit**: `TestWarmupAutotune` — WG-size cache populated for canonical shapes;
  subsequent compile finds cached winners.

#### W4 — Vulkan validation during warm-up ⛔
The warm-up phase should run with `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`
+ `TORCH_VULKAN_VUID_AS_ERROR=1` so any shader bugs are caught at warm-up time,
not mid-training. Currently validation is opt-in and separate from the warm-up
pipeline — the warm-up runs without validation by default.
- **Files**: `hardware_probe.py`, `runtime/common.py` (validation layer wiring)
- **Exit**: `TestWarmupValidation` — `prepare_device(level="deep")` with
  validation enabled; any VUID from precompiled/autotuned shaders fails the
  warm-up call. This gates the compile phase: no validated warm-up → no training.

#### W5 — Per-model warm-up (shape-specific precompile) ⛔
Extend warm-up to accept a model + sample input, trace the model's specific
ops through the Inductor pipeline (fwd+bwd+optimizer), compile and cache all
resulting SPIR-V. Turns the generic canonical-shape sweep into a model-targeted
warm-up that guarantees 100% SPIR-V cache hits for that model's training loop.
- **Files**: `hardware_probe.py` (new `prepare_model(model, sample_input, …)`
  entry point)
- **Exit**: `TestWarmupModel` — `prepare_model(model, sample_input)` populates
  all SPIR-V needed for that model's training; subsequent `torch.compile(model)`
  finds 100% cache hits, zero slangc invocations.

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

**2026-06-16 pipeline audit status.** The AOTI pipeline has four layers, each
at a different readiness state:

| Layer | Component | State | Gap |
|---|---|---|---|
| L1 | C++ AOTI Runtime ABI (`AotiRuntime.{h,cpp}`) | ✅ | 8 symbols exported. `make_kernel`/`dispatch`/`destroy` + 4 T7.4 specializations + model-level stub |
| L2 | AOTI shim symbols (`aoti_shims.o` via `extra_objects`) | ✅ | `empty_strided_vulkan`, `zeros_vulkan`, `mm_out`, `delete` — 14 symbols linked into `_C.so` |
|| L3 | C++ wrapper codegen (`VulkanCppWrapperGpu`) | ✅ | Registered. SPIR-V embedding + dispatch ABI + tensor handle passing all work. Verified: pointwise model compiles, loads, dispatches. |
|| L4 | End-to-end `.so` compile+load+dispatch via runner | ✅ | **FIXED 2026-06-17.** vulkan.h had a static inline `aoti_torch_empty_strided_vulkan` that shadowed the real `_C.so` implementation with a CPU fallback (wrong signature). Changed to `extern "C"` declaration. Verified: AOTIModelContainerRunnerCpu.run() dispatches with both tensors on vulkan:0, VUID=0. |

**Blocker chain for L4:**

1. **PF.60 — RecursionError in tensor_str during AOTI compile** ✅
   **RESOLVED 2026-06-16.** Monkey-patch works; AOTI compile proceeds without error.
   - **Verified**: Pointwise model AOTI compile succeeds, C++ wrapper generates correctly.

2. **PF.30.e — FunctionalTensor view ops crash during AOTI fake-trace** ✅
   **RESOLVED 2026-06-16.** Null-storage guards in shape_ops catch FakeTensors.
   - **Verified**: AOTI compile passes; no view-op crashes during fake-tensor propagation.

3. **AOTI C++ wrapper codegen** ✅ **FIXED 2026-06-16**
   Two scheduling.py changes + two cpp_wrapper_gpu.py changes resolved the
   Slang-source-in-C++ compile failure:
   1. `define_kernel`/`define_combo_kernel` detect `V.graph.aot_mode` and skip
      Python `_vk_make_kernel()` emission
   2. Python `//` → `/` (C++ integer division avoids comment interpretation)
   3. Bare `min(` → `std::min(` (C++ namespace qualification)
   4. SPIR-V compiled from Slang source at AOTI codegen time, embedded as
      `static const uint32_t` arrays
   **Verified**: Pointwise model AOTI `.so` compiles, links 3 `torch_vulkan_aoti_*`
   symbols, zero VUIDs. `aot_load` API doesn't support Vulkan device yet
   (separate issue — uses `torch._export.aot_load` which checks known devices).

4. **AOTI allocator — vulkan.h static inline shadow** ✅ **FIXED 2026-06-17**
   vulkan.h defined `aoti_torch_empty_strided_vulkan` as a `static inline`
   function forwarding to `aoti_torch_empty_strided` with a signature
   mismatch (passed `device_idx` where `device_type` expected → fell to CPU).
   This shadowed the real `extern "C"` implementation in `_C.so`. Changed to
   `extern "C"` declaration. Verified: AOTIModelContainerRunnerCpu.run()
   dispatches with both tensors device=vulkan:0, VUID=0.

5. **AOTI extern-kernel codegen: Python syntax in C++ wrapper** ⛔
   **DISCOVERED 2026-06-17.** The conv2d/GN/Linear extern-kernel lowerings
   have custom `codegen` methods that emit Python-style function calls
   (e.g., `_slang_tile_conv2d(arg4_1, conv_weight, buf1, stride=(1, 1), ...)`).
   During AOTI compilation, this Python syntax leaks verbatim into the C++
   wrapper, causing C++ compiler errors (undeclared identifiers, Python
   tuple literals). Pointwise/reduction kernels don't have this problem —
   they route through `scheduling.py:define_kernel` which detects
   `V.graph.aot_mode` and skips Python emission.
   - **Fix needed**: Each extern-kernel `codegen` method must detect
     `aot_mode` and emit C++ AOTI dispatch calls with pre-compiled SPIR-V
     embedded as `static const uint32_t` arrays. SPIR-V can be compiled
     at codegen time from the template + static shape metadata. After
     this fix, simple conv models will compile to loadable `.so` files.
   - **Files**: `lowerings/conv.py:248-290` (codegen), `lowerings/conv_backward.py`,
     `lowerings/gn_forward_extern.py`, `lowerings/gn_backward_extern.py`,
     `lowerings/matmul.py`
   - **Exit**: `TestAOTIConvGNForward` — AOTI `.so` compiles and produces
     output matching eager forward.
   - **Prerequisite fix**: `_flush_batcher_before_direct_call` no-op added to
     `VulkanCppWrapperGpu` (line 95-97 of `cpp_wrapper_gpu.py`).

6. **AOTI so-load without torch_vulkan on PYTHONPATH** 🟡
   `TestAotiSoLoadsWithoutTorchVulkanPythonpath` — prerequisites (PF.60,
   PF.30.e, AOTI compile) are now resolved. Test needs re-evaluation.

7. **Model-level AOTI API is a stub** 🟡
   `AotiRuntime.h` declares `torch_vulkan_aoti_model_load/run/free`.
   Implementation does simplified single-kernel dispatch — no per-kernel
   buffer layouts, no intermediate tensor management.
   The old `"rc=" in str(exc.value)` assertion failed because the pybind wrapper
   now surfaces the C-side human-readable error (`"empty SPIR-V"`). Fixed to
   `"empty SPIR-V" in str(exc.value)`. The underlying C ABI error contract is
   working correctly — the test just needed the assertion updated.

8. **Full training step .so (fwd + bwd + optimizer)** ⛔
   Gated on item 5 (extern-kernel codegen). Once conv/matmul/GN extern kernels
   emit proper C++ AOTI dispatch, the forward-only `.so` compiles and dispatches.
   The next step is multi-graph: fwd+bwd+optimizer compiled into a single `.so`
   with correct buffer lifetime management across subgraphs. The `torch.compile`
   path already handles this correctly (verified: `TestAOTITrainingE2E` ✅).

#### A3 — Conv backward paired FX rewrite → `bwd_diff(conv_inner_madd)` ✅ (ratified 2026-06-16)
The `aten.convolution_backward` lowering intercepts at Inductor lowering time
and creates `_VulkanConvBwdExternKernel`, which codegens `_slang_tile_conv2d_bwd`.
The Slang shader (`slang_conv_bwd.slang`) uses `bwd_diff(conv_inner_madd)`
internally. The current ExternKernelOut → _slang_tile_conv2d_bwd →
bwd_diff(conv_inner_madd) path IS the paired FX rewrite — the aten node is
intercepted at lowering time and the Slang shader already uses bwd_diff
internally. Ratified rather than doing a pre-grad FX rewrite because the
Inductor-level interception achieves the same end state (Slang template
dispatch, no decomposition into individual aten.mm calls) with less complexity.
- **Files**: `lowerings/conv_backward.py`, `templates/slang_conv_bwd.slang`
- **Exit**: `TestConvBwdNoExtern` — compiled conv-bwd wrapper contains `_slang_tile_conv2d_bwd`; gradient parity vs CPU holds.

#### A4 — Conv2d forward: replace eager `extern_kernels.convolution` with Slang template ✅
**DONE 2026-06-16.** `_vulkan_aten_convolution` lowering registered in
`lowerings/conv.py:441-458`. Intercepts `aten.convolution.default` for groups==1,
4D, Vulkan device, and delegates to `_vulkan_conv2d_with_optional_bias` →
`_VulkanConv2dExternKernel` → `_slang_tile_conv2d` → `slang_conv2d.slang`.
Conv fwd now uses Slang template with `ParameterBlock<KernelArgs>`, spec-constants,
and epilogue generics.
- **Note**: The pre-grad `_fuse_conv_patched_gn_relu` pass takes precedence for
  Conv+GN+ReLU chains, replacing them with the fused `conv2d_gn_relu_fused` op
  before lowering. A4 applies to conv-only or conv+ReLU models.
- **Exit**: conv fwd graphs use `_slang_tile_conv2d` (no `extern_kernels.convolution`).

#### A5 — Pooling forward: replace eager aten with Slang codegen 🟡 (partial)
**PARTIAL 2026-06-16.** FallbackKernel lowerings registered for
`aten.max_pool2d.default` and `aten.avg_pool2d.default` in
`bwd_lowerings.py:730-797`. Both are suppressed from upstream Inductor
decomposition (ops_to_suppress + AOT decomp pop). The FallbackKernel routes
through the C++ Vulkan kernel, avoiding upstream `indirect_indexing` which
produces wrong SPIR-V. This is a stepping stone — the next step is pure Slang
codegen for pooling forward (e.g., scatter_atomic-based or reshape+reduce).
`aten.adaptive_avg_pool2d` already has a decomposition that delegates to
`aten.avg_pool2d` (now FallbackKernel).
- **Files**: `bwd_lowerings.py:730-797`, `lowerings/__init__.py:160-165,251-255`
- **Exit**: `TestPoolFwdSlang` — compiled pool fwd graphs dispatch through FallbackKernel; output matches CPU.

### Pillar B — Slang-smart codegen

#### B1 — `foreach_optimizer` algorithm → `interface IOptimizerAlgorithm` ✅ (ratified 2026-06-16)
SLANG INTERFACE DONE. Template uses `interface IOptimizer` with generic
`<Algo : IOptimizer>` compile-time parameter. Concrete types: `SGDImpl`,
`SGD MomentumImpl`, `AdamWImpl`, `LionImpl`. Entry point selected via
`computeMain<AdamWImpl>`. One SPIR-V module per algorithm type (vs
3 Jinja variants before), but the module is reused across all batch sizes
(1/7/15/21/32). The parameter-buffer Jinja `for i in range(batch_size)` loop
is structural (declares buffer bindings), defensible under anti-goal #6.
- **Files**: `templates/foreach_optimizer.slang`, `caller/optimizer.py`
- **Exit**: `TestOptimizerInterface` — one compile, dispatch SGD/AdamW/Lion by spec-const, parity vs eager.

#### B2 — `rnn_cell*` direction + cell_type → interface/spec-const ⛔
`direction` → runtime gate on a spec-constant. `cell_type` (lstm/gru/rnn_tanh/
rnn_relu) is currently a Jinja structural branch **not tracked in v16** — fold
into an `IRnnCell` interface (gate `has_cell_state` / `gate_size` by spec-const).
Covers `rnn_cell.slang`, `rnn_cell_bwd.slang`, `rnn_cell_fused.slang`.
- **Files**: `templates/rnn_cell*.slang`, `caller/rnn.py`
- **Exit**: `TestRnnInterface` — LSTM + GRU + bidirectional from one module set, parity vs eager.

#### B3 — AST validator: bwd_diff signature matching ✅
**DONE 2026-06-16.** `bwd_diff_scan.py` extended with `validate_bwd_diff_signatures()`
that extracts forward/backward parameter lists and validates:
1. Param count: backward must have len(forward_params) + 1 (output gradient)
2. Per-param: differentiable T → inout DifferentialPair<T>; no_diff → no_diff
3. Output gradient: last backward param must not be inout/no_diff/DifferentialPair
Integrated into `validate_slang_source()` pipeline. Runs pre-slangc.
- **Files**: `slang_validate/bwd_diff_scan.py:68-254`, `slang_validate/__init__.py`

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

#### C4 — GN backward kernel fusion ⛔
**PROFILED 2026-06-18.** GN backward is decomposed into **11 individual kernels**
(sub, mul, 3×sum, fill, expand, fill, pow, mul_scalar, mul). Each is a tiny
pointwise/reduction dispatch. These should be fused into 1-2 kernels (like
`conv_gn_relu_fused` handles forward). Eliminating 10 dispatches per backward
pass would recover ~2-3ms per step.
- **Files**: `bwd_lowerings_norm.py`, `bwd_diff_table.py`, `combo_kernel/`
- **Exit**: `TestGNBwdFused` — GN backward uses ≤3 dispatches (vs 11 today);
  gradient parity vs CPU holds.

#### C5 — Conv backward forward-recomputation elimination 🟡 **C5.1 PARTIAL 2026-06-18**
The conv backward path re-runs `slang_conv2d` (conv forward) as part of gradient
computation because the fused `conv_gn_relu` shader doesn't store the intermediate
conv+bias output. This adds one heavy dispatch per backward pass.

**C5.1 (2026-06-18)**: Added `TORCH_VULKAN_DISABLE_CONV_GN_FUSION` config gate
(default 0 = fusion enabled). When set to 1, the pre-grad fusion pass skips
conv+GN+ReLU fusion, using separate `slang_conv2d` + GN dispatches. This
eliminates the backward recomputation but trades 1 extra forward dispatch.
Note: warm re-runs with fusion disabled may hang (scheduler deadlock TBD).
- **Files**: `config.py:539-554`, `fx_passes/post_grad.py:724-726`
- **Exit**: `TestConvBwdNoRecompute` — backward path contains zero conv-fwd
  dispatches; output unchanged.

#### C6 — Tiny-kernel fusion (fill, copy, inplace-add) 🟡 **C6.1 PARTIAL 2026-06-18**
**PROFILED 2026-06-18.** Each training step includes **13 tiny dispatches**
for fill (5×), copy_strided (4×), and binary_add_inplace (4×), nearly all with
`wg=(1,1,1)`. These account for 45% of dispatch count but <5% of FLOPs.
Fusing them into the surrounding compute kernels would eliminate ~4ms of
dispatch overhead per step.

**C6.1 (2026-06-18)**: Raised persistent-pointwise reduction fusion cap from
1024 → 8192 (`scheduling.py:283`). This enables the persistent pointwise
kernel to fuse pointwise + reduction chains (e.g., loss backward `sum` +
element-wise ops) into fewer dispatches. Verified gradient parity passes.

**C6.2 (2026-06-18)**: Removed 5 redundant `.zero_()` calls across
GN backward and conv3d backward (`gn_backward_extern.py:143,240,242`,
`conv3d_backward.py:74,75,80`). Each emitted a separate `copy_fill_fwd`
GPU dispatch. The M23.1 allocator already zero-initializes buffers.
Also switched conv3d from `aten.full` (alloc+fill dispatch) to
`empty.memory_format` (zero-init by allocator, no dispatch).
Remaining: ~8 micro-copy/inplace dispatches for gradient accumulation.
- **Files**: `gn_backward_extern.py`, `conv3d_backward.py`, `scheduling.py:279-291`
- **Exit**: `TestTinyKernelFusion` — per-step dispatch count ≤20 (vs 30 today);
  no standalone `copy_fill_fwd`/`copy_strided_copy_fwd` dispatches.

#### C7 — GN backward Slang-extern rewrite ✅ **ALREADY DONE**
The GN backward is already implemented as 2 fused Slang dispatches via
`_VulkanGNBwdInputExternKernel` and `_VulkanGNBwdWeightExternKernel`
(`lowerings/gn_backward_extern.py`). The 11 pointwise dispatches observed
in profiling come from the LOSS backward (sub+mul+sum+fill+expand+pow+...),
not from GN backward itself. The GN backward is well-optimized.
- **Files**: `lowerings/gn_backward_extern.py`, `bwd_lowerings_norm.py:110-258`

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
- **Sub-item E1.1 (M23.2-spinoff)**: Fixed avg_pool2d_backward `DonatedBuffer`→
  `OpsValue` crash when conv output is reused as pool input. The codegen path
  (`avg_pool2d_backward_codegen`) built `ops.mul/reshape/expand` chains that
  produced bare `OpsValue` nodes the lowering framework can't wrap when the
  input is a DonatedBuffer. Fix: detect DonatedBuffer and route through the
  scatter_bwd fallback instead (`bwd_lowerings.py:631-651`). Verified:
  `test_avg_pool2d_backward_grad_parity` now PASSES on GPU.
- **Files**: `lowerings/__init__.py:477,481`, `templates/scatter_atomic.slang`, `bwd_lowerings.py:616-651`

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
WARM-UP PIPELINE (pre-compile — runs once before any torch.compile)
├─ W1 (microbench)      ✅
├─ W2 (shader-lib precompile) ✅
├─ W3 (autotune sweep)  ✅
├─ W4 (validation during warm-up) ⛔
└─ W5 (per-model warm-up) ⛔
      │
      │  After warm-up, all downstream pillars find hot caches:
      ▼
A1 (conv-bwd grad fix) ✅ ──→ A2 (full-step .so) ⛔
   │                            ├─ A2.1 (PF.60: verify tensor_str fix) 🟡
   │                            ├─ A2.2 (PF.30.e: FunctionalTensor view ops) ⛔
   │                            ├─ A2.3 (AOTI backward codegen) ⛔
   │                            ├─ A2.4 (Training step .so compile+load) ⛔
   │                            ├─ A2.5 (PYTHONPATH-clear subprocess load) 🟡
   │                            ├─ A2.6 (SPIR-V cache reuse) ⛔
   │                            └─ A2.7 (Model-level API: real scheduling) 🟡
   ├─ A3 (conv-bwd FX rewrite) ✅
   ├─ A4 (conv fwd eager→Slang) ✅
   └─ A5 (pooling fwd) 🟡 — FallbackKernel done, pure Slang TBD

B1 (foreach interface) ✅
B2 (rnn interface)     ⛔
B3 (validator)         ✅

C1 (overlap) ──→ flip BATCH_DISPATCH default ──→ C2 (shape bucketing) ⛔
C3 (persistent reductions) 🟡

D1 (autotune) ⛔

E1 (pooling-bwd fallbacks) 🟡
E2 (masking backward) ⛔
E3 (missing ops)     ⛔
```

Parallel streams: **W4** (validation during warm-up) gates the entire
compile path — catch shader bugs at warm-up, not mid-training. **W5**
(per-model warm-up) guarantees zero cold slangc during training.
**A2.1** (PF.60 verify) is the lowest-hanging fruit — one test run
without xfail. **A5-pure** (Slang codegen for pooling fwd) and **E1**
(pooling-bwd Slang) are the remaining codegen gaps. **B2** cleans up
the last Jinja template. **C**/**D** are performance/autotune;
**E2/E3** are op coverage breadth.

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

- **Closed-milestone history** (v6.x → v16, M18–M23, FP16): `docs/10-inductor-backend-history.md`, `docs/archive/`.
- **Pipeline / API reference**: `docs/how-to-compile-and-codegen.md`,
  `docs/inductor-pipeline-analysis.md`, `docs/10-lib-api-reference.md`.
- **Companion CLAUDE.md**: `backends/vulkan_slang/CLAUDE.md` (build/test/env knobs/file ownership).

*This file is the single canonical roadmap. Do not fork a new numbered version —
edit this doc in place: mark items ✅ as they close, add new sub-items under the
right pillar.*
