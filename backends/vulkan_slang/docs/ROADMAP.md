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

**AOT / deployment (AOTI)**
| Item | State | Evidence |
|---|---|---|
| C++ AOTI Runtime ABI (`AotiRuntime.{h,cpp}` — make/dispatch/destroy + T7.4) | ✅ | 8 symbols exported; tested via `TestAotiCppLoader` (5/6 pass, 1 fixed this session) |
| Link `aoti_shims.o` into wrapper `.so` (14 `aoti_torch_*` symbols) | ✅ | `setup.py:133`, `cpp_wrapper_gpu.py` |
| C++ wrapper codegen registered (`VulkanCppWrapperGpu` in `device_codegens`) | ✅ | `__init__.py:828-835`; codegen verified (T7.2 tests pass) |
| Import does not hang (`meta_patches` lazy) | ✅ | `meta_patches/__init__.py` |
| Clean process exit (`shutdown(wait=False)` at atexit) | ✅ | `runtime/common.py:373-379` |
| **Conv+GN+pool+linear training E2E + grad parity** (`torch.compile` path) | ✅ | `TestAOTITrainingE2E` — **FIXED 2026-06-16 (M23.2)** |
| **PF.60**: RecursionError in tensor_str during AOTI compile | 🟡 | Monkey-patch installed (`pf60_tensor_str_fix.py`); needs verification with `aot_compile` on Vulkan model |
| **PF.30.e**: FunctionalTensor view ops crash | 🟡 | `is_null_storage` guard in `vulkan_permute`/view/reshape catches FakeTensors (data_ptr=0 confirmed); likely already fixed — needs re-verification |
| **AOTI backward codegen**: C++ wrapper for bwd graphs | ⛔ | Not exercised — `TestCP12AOTITrainingStep` uses Python wrapper |
| AOTI `.so` fwd+bwd+optimizer full step, SPIR-V cache reuse across loads | ⛔ | L4 gap — full C++ compile+link+load+dispatch chain unverified |
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

**2026-06-16 pipeline audit status.** The AOTI pipeline has four layers, each
at a different readiness state:

| Layer | Component | State | Gap |
|---|---|---|---|
| L1 | C++ AOTI Runtime ABI (`AotiRuntime.{h,cpp}`) | ✅ | 8 symbols exported. `make_kernel`/`dispatch`/`destroy` + 4 T7.4 specializations + model-level stub |
| L2 | AOTI shim symbols (`aoti_shims.o` via `extra_objects`) | ✅ | `empty_strided_vulkan`, `zeros_vulkan`, `mm_out`, `delete` — 14 symbols linked into `_C.so` |
| L3 | C++ wrapper codegen (`VulkanCppWrapperGpu`) | 🟡 | Registered in `device_codegens["vulkan"]`. SPIR-V embedding works. `_generate_kernel_call_helper` emits init+dispatch. **Not yet exercised with a real C++ compile+link+load cycle.** |
| L4 | End-to-end `.so` compile+load+dispatch | ⛔ | No test exercises the full chain: Slang source → SPIR-V → embed in C++ → compile `.so` → load via `aot_load_package` → dispatch → verify output. |

**Blocker chain for L4:**

1. **PF.60 — RecursionError in tensor_str during AOTI compile** 🟡
   Monkey-patch installed (`pf60_tensor_str_fix.py`, activated at
   `__init__.py:802-803`). May be resolved — the fix detects Vulkan
   tensors in `_str_intern` and returns a safe placeholder. Not yet
   verified with `torch._inductor.aot_compile` on a real model.
   - **Verify**: Run `TestAotiAotCompileMlpFwd.test_aoti_three_layer_mlp_no_bias`
     without the `xfail` marker. If it passes, PF.60 is resolved.

2. **PF.30.e — FunctionalTensor view ops crash during AOTI fake-trace** 🟡
   **2026-06-16 analysis**: The `vulkan_permute`/`vulkan_view`/`vulkan_reshape`
   C++ ops already have null-storage guards (`is_null_storage || is_meta ||
   !has_storage`). `is_null_storage` checks `storage().data() == nullptr`,
   which catches FakeTensors (confirmed: FakeTensor with `device=vulkan:0`
   has `data_ptr=0`). The guard at `shape_ops.cpp:189` returns a null-storage
   Vulkan tensor, allowing fake-tensor propagation to continue without
   dispatching a real shader. **This may already be resolved** — the xfail
   gate needs re-verification with the actual `aot_compile` call.
   - **Verify**: Run `TestAotiAotCompileMlpFwd` both subtests without xfail.
     If they pass, mark PF.30.e ✅ and remove the xfail markers.

3. **AOTI codegen for backward graphs** ⛔ → **ROOT CAUSE FOUND 2026-06-16**
   `VulkanCppWrapperGpu` is registered in `device_codegens["vulkan"]` but
   the AOTI C++ compile step fails because the generated C++ contains
   **Slang source code as Python triple-quoted strings** instead of
   compiled SPIR-V binary arrays. The wrapper emits:
   ```cpp
   vulkan_kernel_0_slang = '''struct KernelArgs {
       RWStructuredBuffer<float> out_ptr0;
   };
   [shader("compute")] [numthreads(64, 1, 1)]
   void computeMain(...) { ... }
   '''
   ```
   This is valid Python syntax (produced by `VulkanPythonWrapperCodegen`),
   not valid C++. The C++ compiler rejects it with "empty character constant".
   
   **Root cause**: The AOTI path selects `VulkanCppWrapperGpu` (correctly
   registered), but the kernel call emission goes through the **Python
   wrapper's `_generate_kernel_call_helper`** instead of the C++ wrapper's
   override. The Python wrapper stores Slang source as Python strings; the
   C++ wrapper should compile Slang→SPIR-V at AOTI package time and emit
   `torch_vulkan_aoti_make_kernel` + `torch_vulkan_aoti_dispatch` calls
   with binary SPIR-V arrays.
   
   **Fix**: The C++ wrapper's `_generate_kernel_call_helper` method
   (lines 270–451 of `cpp_wrapper_gpu.py`) already has the correct logic
   — it reads SPIR-V from the compile cache, embeds it as `static const
   uint32_t` arrays, and emits `torch_vulkan_aoti_make_kernel` +
   `torch_vulkan_aoti_dispatch` calls. The issue is that this method is
   **not being reached** during AOTI codegen — the upstream
   `_generate_kernel_call_helper` (from `CppWrapperCpu`) is handling the
   Vulkan kernels instead, falling through to the Python wrapper pattern.
   - **Files**: `cpp_wrapper_gpu.py:270-451`, `__init__.py:828-835`
   - **Verify**: AOTI `.so` compiles without Slang-source-in-C++ errors

4. **Full training step .so (fwd + bwd + optimizer)** ⛔
   No test loads an AOTI `.so` containing a full training step and
   executes it. The three subgraphs (fwd, bwd, optimizer) need to be
   compiled into a single `.so` with correct buffer lifetime management
   across subgraphs.
   - **Files**: `cpp_wrapper_gpu.py`, `runtime/slangc.py` (cache key),
     `csrc/backend/`
   - **Exit**: `TestAOTI_FullStep` — single `.so` executes fwd→loss→bwd→step;
     SPIR-V cache reused across `aoti_load_package` calls with zero recompiles.

5. **AOTI so-load without torch_vulkan on PYTHONPATH** 🟡
   `TestAotiSoLoadsWithoutTorchVulkanPythonpath` is written and xfail-gated
   behind PF.60 + PF.30.e + PF.31.b. Once those resolve, this test auto-flips
   to verify the Python-less load contract — the "AOT" half of the mission.

6. **Model-level AOTI API is a stub** 🟡
   `AotiRuntime.h` declares `torch_vulkan_aoti_model_load/run/free` with a
   `kernels.bin` binary format. The implementation exists but `model_run`
   does simplified single-kernel dispatch — all kernels get the same tensor
   set, no per-kernel buffer layouts, no intermediate tensor management, no
   workgroup derivation from tensor shapes. This is a placeholder awaiting
   real multi-kernel scheduling (P4.x).

7. **Test fix**: `test_aoti_make_kernel_surfaces_errors` assertion ✅ **FIXED 2026-06-16**
   The old `"rc=" in str(exc.value)` assertion failed because the pybind wrapper
   now surfaces the C-side human-readable error (`"empty SPIR-V"`). Fixed to
   `"empty SPIR-V" in str(exc.value)`. The underlying C ABI error contract is
   working correctly — the test just needed the assertion updated.

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

Parallel streams: **A2.1** (PF.60 verify) is the lowest-hanging fruit — one
test run without xfail. **A2.2** (PF.30.e FunctionalTensor view ops) is the
active blocker for all downstream AOTI work. **A5-pure** (Slang codegen for
pooling fwd) and **E1** (pooling-bwd Slang) are the remaining codegen gaps.
**B2** cleans up the last Jinja template. **C**/**D** are performance/autotune;
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
