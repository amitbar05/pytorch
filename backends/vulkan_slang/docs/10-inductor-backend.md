# Vulkan-Slang Inductor Backend Roadmap v6.1

> **v5 North Star achieved (2026-05-10).** SmallCNN trains end-to-end.
> 16 waves, 57 items shipped. v5 completion details archived in
> [10-inductor-backend-history.md](10-inductor-backend-history.md).
>
> **v6 focuses on the 4 remaining blockers and production hardening.**
> **v6.1 (2026-05-13) adds M9–M12** based on a 4-track audit (codegen quality,
> GPU utilization, Slang feature exploitation, profiling). See § 0.5 below
> for the full audit summary.
>
> **Last updated: 2026-05-13.**
>
> **Live state:** 39,182 lines of test code, 66 e2e model tests, 4 strict xfails.
> Feature flags ON: spec_constants, descriptor_indexing, static_specialization,
> bank_conflict_pad, dynamic_shapes, batch_dispatch, wrapper_fastpath,
> grid_aware_wg, persistent_pointwise.

---

## 0. What Remains (v6)

### The 4 Remaining Blockers

| # | Item | Blocker | Actionable? |
|---|------|---------|-------------|
| 1 | **T4.12** Conv1d/3D/depthwise/transposed conv | Template generality work | ✅ **Yes — implement now** |
| 2 | **N+1.9** Link-time tile specialization | slangc upstream bug (E30600) | ❌ Monitor slangc releases |
| 3 | **T7.2** Full .so subprocess load | C++ build infrastructure | ❌ Needs build system |
| 4 | **Track CI** GPU hardware | No CI runner with Vulkan GPU | ❌ Needs hardware |

### v6 Milestones

| # | Milestone | Key Deliverable | Effort |
|---|-----------|----------------|--------|
| **M6** | **Conv generality (T4.12)** | Conv1d/3D/depthwise/transposed conv via template parameterization | 1-2w |
| **M7** | **Production hardening** | Link-time spec (when slangc ships), AOTI .so packaging, CI gate | gated |
| **M8** | **Model zoo expansion** | More real-world models compiling end-to-end | ongoing |
| **M9** | **Host-overhead reduction** | Buffer pool fix, fence batching, prewarm-on-import — close the 96% host/kernel gap | 1w |
| **M10** | **Anti-goal cleanup** | Split monoliths (5786L+3902L); ✅ anti-goal #6 closed (M10.4 / CG.M10, mm epilogue generic) | 1-2w |
| **M11** | **Occupancy-aware codegen** | Reflection-routed WG sizing (DR.7), subgroup reductions, LDS bank-padding rigour | 1-2w |
| **M12** | **Reduction backward via autodiff (CG.M3)** | `[Differentiable]` on 8 reduction ops; auto-generate backward, retire hand-written | 1w |

---

## 0.5. Audit Findings (2026-05-13)

Four-track audit synthesised from parallel sub-agents. Numbers verified by
probe scripts under `agent_space/probe_*.py` and `agent_space/vk_validation_sweep*.py`.

### 0.5.1 Headline numbers

| Probe | Result | Reaction |
|-------|--------|----------|
| MLP train warm step | **75 µs kernel / 1.63 ms wall** | 96 % host overhead — feeds M9 |
| SmallCNN train warm step | **191 µs kernel / 43.9 ms wall** | 230× host/kernel — feeds M9 |
| SmallCNN cold compile | **9.0 s for 8 dispatches** (8 slangc, 800 ms ea) | Prewarm-on-import — feeds M9 |
| MLP buffer pool, 10 steps | **0 / 50 hits (0 %)**, 20 releases, peak 8 | Pool key bug — M9 P0 |
| GN + ReLU + GlobalAvg | **2 kernels (target: 1)** | Reduction-boundary fusion gap — M9 |
| Transformer (d=64, h=4, s=32) compile | **`UnboundLocalError: buf10`** in wrapper | combo-batcher bug — file + M9 |
| Validation sweep (10 paths) | **0 VUIDs** (post 4 fixes earlier in 2026-05-13) | Clean |
| Validation sweep best-practices | **0 hints** (post CommandPool fix) | Clean |
| CPU eager MLP / Vulkan | 0.033 ms vs 1.63 ms = **49× slower** | Host overhead, not kernels |
| CPU eager SmallCNN / Vulkan | 0.634 ms vs 43.9 ms = **69× slower** | Host overhead, not kernels |

### 0.5.2 Anti-goal violations confirmed

| Anti-goal | State | Where | Fix milestone |
|-----------|-------|-------|---------------|
| #6 — no string template params | ✅ **FIXED 2026-05-13** | mm template now uses `computeMain<Epilogue : IDifferentiable>` Slang generic in `templates/slang_mm.{slang,py.jinja}`; verified by `tests/test_cgm10_idifferentiable.py` (11 tests, all passing) | M10.4 / CG.M10 — closed |
| #7 — files ≤ 800 L | **VIOLATED 4×** | `vulkan_template_caller.py` 5786 L, `meta_patches.py` 3902 L, `kernel/pointwise.py` 1555 L, `fx_passes/eager_patches.py` 1159 L | M10.1-3 |
| #3 — no `aten.*_backward` lowerings | **PARTIAL** | Reduction bwd: 6 `[Differentiable]` in `lib/reduction.slang`; `tests/test_cgm3_reduction_backward.py` has 13 tests / 11 xfailed — code present, dispatch wiring incomplete | M12 |
| #2 — `csrc/ops/model_ops.cpp` = 0 L | **VIOLATED** | 885 L remain | Track-4 close (existing) |

### 0.5.3 Slang feature exploitation (updated 2026-05-13 post-CG.M10)

| Feature | Score | Top blocker |
|---------|-------|-------------|
| Generics `<T : Float>` / `<Op : I…>` | 70 % | mm now uses `<Epilogue : IDifferentiable>`; conv/SDPA templates still need same treatment (CG.M5-M7) |
| Interfaces | 80 % | mm/reduction shipped; `IConvBias`, `INormAffine` defined but not yet consumed by templates |
| `[Differentiable]` + `bwd_diff()` | 75 % | Reduction (6/8 ops `[Differentiable]`, dispatch wiring partial — M12), conv (CG.M6 in `tests/test_cgm6_*`), SDPA (CG.M7 in `tests/test_cgm7_*`) |
| `[BackwardDerivative(fast_bwd)]` | 38 % | CG.M9 audit live; PF.11 benchmark not wired |
| `ParameterBlock<T>` | 100 % | shipped, locked by `TestTemplateParameterBlockInvariant` |
| Reflection metadata | 35 % | VGPR/LDS parsed but never consumed at codegen (DR.7 dormant) |
| Link-time specialisation | 15 % | slangc upstream E30600 |
| Capabilities `[require(…)]` | 78 % | No systematic audit for non-RDNA1 targets |
| `[SpecializationConstant]` | n/a | Vulkan `[[vk::constant_id]]` preferred — correct by accident |

### 0.5.4 Vulkan-spec spec hygiene (closed 2026-05-13)

| VUID / hint | Was | Fix |
|-------------|-----|-----|
| `VUID-VkDeviceCreateInfo-pNext-02830` | pNext chain mixed `Vulkan12Features` + legacy `DescriptorIndexingFeatures` | Collapsed to Vulkan 1.2 struct (`Context.cpp`) |
| `VUID-VkDescriptorSetLayoutCreateInfo-flags-03000` | per-binding `UPDATE_AFTER_BIND_BIT` w/o matching layout flag | Set layout flag unconditionally when desc-indexing on (`Pipeline.cpp`) |
| `VUID-VkShaderModuleCreateInfo-pCode-08740` | SPIR-V `Int64` used, `shaderInt64` never enabled | Added cap query + enable (`Context.{h,cpp}`) |
| `BestPractices-vkCreateCommandPool-command-buffer-reset` | `RESET_COMMAND_BUFFER_BIT` set, unused | Dropped flag (`CommandBuffer.cpp`) |
| **non-fatal SPIR-V validation: `OpULessThan` operand class** | Surfaces during SmallCNN cold compile | File — needs codegen-side fix (probably emitting signed compare with unsigned operand) |

### 0.5.5 New punch list (work into M9-M12)

**M9 — Host-overhead reduction (1w, P0 for perf)**

- [ ] **M9.1** Buffer-pool 0 % hit rate root-cause (the 50/0 result over 10 MLP steps). Acquire/release count mismatch (50 vs 20) — most outputs escape. Inspect `_key()` and `vulkan_pool_release` flow; ensure outputs released after step. (1-2d)
- [ ] **M9.2** Deferred command-buffer batching: stop per-dispatch `submit_and_wait`; submit 4-8 dispatches per `vkQueueSubmit`. (`Stream.cpp` has a skeleton.) Target: -5 to -10 ms / SmallCNN step. (2-3d)
- [ ] **M9.3** Prewarm-on-import: shader-lib precompile runs at first `torch_vulkan` import, not first dispatch. Eliminates the 9 s cold step on SmallCNN. (0.5d)
- [ ] **M9.4** Push-constant in-place updates: pre-allocate bytearray per kernel; update fields, don't `bytes(pc_data)` per dispatch. (1d)
- [ ] **M9.5** Cached `_jit_dispatch_indexed`: codegen prefers indexed variant when any binding has count > 1. (1d)
- [ ] **M9.6** Adaptive `_PER_KEY_CAP` in `buffer_pool.py` (scratch=8, transient=6, save_for_backward=4). (1d)
- [ ] **M9.7** Pool non-extern Inductor outputs (currently only extern-kernel outputs are pooled). (1-2d)
- [ ] **M9.8** Reduction-boundary fusion: GN + ReLU + GlobalAvg should fuse into 1 kernel, not 2. (2-3d)
- [ ] **M9.9** Transformer combo-batcher `UnboundLocalError: buf10` — `vulkan_combo_kernel` emits buf reference before assignment. Repro: `agent_space/probe_transformer.py`. (1-2d)

**M10 — Anti-goal cleanup (1-2w, debt reduction)**

- [ ] **M10.1** Split `vulkan_template_caller.py` (5786 L → 4-5 files of ≤ 800 L each), one per template family (gemm, scatter, optimizer, flash_attn, rng). (1-2d)
- [ ] **M10.2** Split `meta_patches.py` (3902 L) into `meta_patches/{shape_ops,dtype_ops,faketensor_hooks}.py`. Audit for stale patches now that Track-1 codegen is clean. (1d)
- [ ] **M10.3** Split `kernel/pointwise.py` (1555 L): extract `PointwiseLoadMixin` + `PointwiseVec4Mixin`. (1.5d)
- [x] **M10.4** **CG.M10** — promoted mm epilogue from `{{ epilogue }}::apply()` Jinja string interp to Slang `<Epilogue : IDifferentiable>` generic. ✅ **Shipped 2026-05-13** (`templates/slang_mm.{slang,py.jinja}` line 219/222; `tests/test_cgm10_idifferentiable.py` 11 tests passing). Closes anti-goal #6.
- [ ] **M10.5** Lift `_VALID_IPOINTWISE_STRUCTS` frozenset to auto-parse `lib/pointwise.slang` at startup; drop manual sync. (0.5d)
- [ ] **M10.6** Remove redundant outer cast in `kernel/pointwise.py:68-81` int8 load dispatch (`((float)((int(…) << 24) >> 24))` → `((int(…) << 24) >> 24)`). (0.5d)
- [ ] **M10.7** Audit & remove stale TODO gate at `vulkan_template_caller.py:754` (P3.2/M14 dead flag). (0.5d)
- [ ] **M10.8** Extract pickle/repr boilerplate from `_SlangTile{MM,AddMM,BMM}` into a common base. (1d)

**M11 — Occupancy-aware codegen (1-2w, throughput)**

- [ ] **M11.1** **DR.7 wire-up**: feed reflection VGPR/LDS into `_pick_threadgroup_size_*` instead of leaving `estimate_occupancy()` as a debug-only tool. Default flag → on. Target: +10-20 % on reduction/normalisation kernels. (2-3d)
- [ ] **M11.2** Subgroup reductions for WG ≤ wave64: emit `WaveActiveSum`/`WaveActiveMax` instead of LDS reduce. (2d)
- [ ] **M11.3** Register-tile pointwise: load+compute+store unrolled ×2-4. (3d)
- [ ] **M11.4** Persistent-mode WG autotune: scale WG size by `numel / CU_count`; today the persistent path uses a fixed WG. (1-2d)
- [ ] **M11.5** Round non-multiple-of-64 WG sizes up to next multiple on RDNA1 (auto-fix `slang_validator.py` advisory). (0.5d)
- [ ] **M11.6** LDS bank-padding rigour: auto-pad WG-shared arrays > 1 KB to nearest power of 2 to avoid stride-1 bank conflicts. (1-2d)
- [ ] **M11.7** Occupancy gate in `codegen.py`: warn (or `--strict` fail) if estimated occupancy < 50 %. (1-2d)
- [ ] **M11.8** Extend `_KERNEL_STATS` to capture grid, WG, VGPR, LDS, descriptor count (populated from reflection). (1-2d)

**M12 — Reduction backward via autodiff (CG.M3, ~1w remaining)**

- [~] **M12.1** Partial: 6 `[Differentiable]` annotations in `shaders/lib/reduction.slang` (sum/mean/var fold paths shipped). Still need: prod, max, min, argmax, argmin coverage. (1-2d)
- [ ] **M12.2** Route `aten.{sum,mean,prod,…}_backward` through `bwd_diff_table.py` instead of hand-rolled lowering. `tests/test_cgm3_reduction_backward.py` has 13 tests / 11 xfailed — dispatch wiring is the gate. Closes anti-goal #3 for the reduction class. (2d)
- [ ] **M12.3** Retire any `aten.*_backward` reduction shaders still living outside `lib/`. (1d)

### 0.5.6 Items downgraded / re-classified

- **CG.M3** (reduction backward via `[Differentiable]`) — promoted to M12 (was loose in CLAUDE.md "next").
- **DR.7** (reflection-routed WG) — promoted to M11.1; previously code-present-but-dormant.
- **N+1.9** (link-time spec) — stays gated on slangc E30600; no change.
- **CG.M4–M7** (norm/matmul/conv/SDPA via `[Differentiable]`) — stay in CLAUDE.md tactical list; M12 (reduction) is the natural first hop.

---

## 1. M6 — Conv Generality (T4.12) — ✅ Phase 1 SHIPPED

### Current State
- ✅ Conv2d fwd+bwd: full support via `slang_conv2d.slang` + CG.M6 bwd template
- ✅ Dilation > 1: supported via im2col decomposition
- ✅ **Conv1d: fwd+bwd via reshape to Conv2d (M6 Phase 1 — DONE 2026-05-11)**
- ✅ Conv3d KD=1: shipped. Grouped conv (arbitrary groups>1): per-group decomposition
- ✅ Depthwise (groups>1): per-channel fwd+bwd shipped
- ✅ Transposed conv: lowering registered (aten.conv_transpose2d.input)

### Plan (updated 2026-05-10)

**Phase 1 — Conv1d support ✅ DONE (2026-05-11):**
- Conv1d lowered by reshaping [N,C,L] → [N,C,L,1], dispatching to Conv2d, squeezing back
- Automatically inherits backward from Conv2d's CG.M6 bwd template
- Stride, padding, dilation, groups all handled via the 2D parameter expansion
- Lowering lives in `lowerings/conv.py` (`_vulkan_conv1d_with_optional_bias`)
- Tests in `TestConv1dCompile`: fwd correctness, backward, grouped, causal conv1d
- Unblocks: Whisper encoder, Mamba causal conv1d, audio models

**Phase 2 — Depthwise conv (groups=C):**
- Depthwise is Conv2d where each input channel has its own kernel
- Can be lowered to group Conv2d with groups=C
- The existing im2col path should handle this with correct groups parameter

**Phase 3 — Conv3d:**
- Conv3d can be lowered to Conv2d by merging spatial dims
- Or implement directly by extending the 2D template

**Phase 4 — Transposed conv:**
- Transposed conv = backward of regular conv
- Can reuse CG.M6's conv backward template with swapped roles

### Files
- `lowerings/conv.py` — add reshape-based lowering for Conv1d
- `templates/slang_conv2d.py.jinja` — extend for groups>1
- `tests/test_inductor_regression.py` — flip Conv1d/3D/depthwise xfails
- `tests/test_e2e_models.py` — enable Whisper/Mamba conv1d tests

---

## 2. M7 — Production Hardening (gated)

### N+1.9 Link-time tile spec
- **Blocker:** slangc `E30600` cross-module generic specialization bug
- **When fixed:** 112 slangc invocations → 2 per matmul family, 10× compile time reduction
- **Code ready:** `vulkan_template_caller.py:670-678` has the `use_lt` gate
- **Action:** Monitor https://github.com/shader-slang/slang/releases

### T7.2 Full AOTI .so deployment
- **Blocker:** C++ build infrastructure for .so packaging
- **Code ready:** `cpp_wrapper_gpu.py` emits C++ with embedded SPIR-V
- **Action:** Integrate with PyTorch's AOTI build system

### Track CI
- **Blocker:** No CI runner with Vulkan GPU
- **Action:** Set up self-hosted runner with RDNA1 GPU

---

## 3. M8 — Model Zoo Expansion (ongoing)

### Currently covered (85 tests — Llama-3 block added)
ViT encoder, Llama MLP, Llama-3 full block, Mixtral MoE, Stable Diffusion (RMSNorm+RoPE+SwiGLU), UNet block, Whisper full encoder, Mamba-2 selective scan,
MiniGPT, ResNet, Transformer block, Qwen3.5 GatedDeltaNet, SmallCNN training

### Candidates for expansion
| Model | Key ops | Blocker |
|-------|---------|---------|
| Whisper full encoder | Conv1d + attention | ✅ ADDED (full pipeline) |
| Mamba/Mamba2 | Causal conv1d | ✅ ADDED |
| Stable Diffusion UNet | Conv2d + GroupNorm + attention + upsample | ✅ ADDED (attention+upsample, GN xfail) |
| Llama-3 full | RoPE + RMSNorm + SwiGLU | ✅ ADDED (8 tests) |
| Mixtral MoE | Top-k routing + experts | ✅ ADDED (6 tests) |

---

## 4. Heuristic Improvements (2026-05-11)

| Feature | Gate | Impact |
|---------|------|--------|
| Aggressive fusion | `TORCH_VULKAN_AGGRESSIVE_FUSION=1` | Relaxed memory threshold, reduction+pointwise tails, multi-consumer fusion |
| Persistent kernel v2 | `TORCH_VULKAN_PERSISTENT_POINTWISE=1` | Per-thread work + op-count scaling, cap raised to 16384 |
| Grid-aware WG v2 | `TORCH_VULKAN_GRID_AWARE_WG=1` | Targets num_cus × waves_per_cu wave slots |
| Dispatch ratchet | `TestDispatchCountRatchet` | MLP fwd ≤8, SmallCNN train ≤25 |

---

## 5. GPU Utilization (Wave 16, all active by default)

| Feature | Gate | Impact |
|---------|------|--------|
| Batch dispatch | `TORCH_VULKAN_BATCH_DISPATCH=1` | Single `vkQueueSubmit` per graph |
| Wrapper fast-path | `TORCH_VULKAN_WRAPPER_FASTPATH=1` | Cached imports, skipped validation |
| Dispatch profiling | `TORCH_VULKAN_PROFILE_DISPATCHES=1` | Per-kernel timing stats |
| Grid-aware WG | `TORCH_VULKAN_GRID_AWARE_WG=1` | Smaller WGs for small grids |
| Persistent pointwise | `TORCH_VULKAN_PERSISTENT_POINTWISE=1` | Grid-stride loop fusion |
| Occupancy estimator | `estimate_occupancy()` | VGPR/LDS/thread bottleneck report |

---

## 5. Reference Files

| Concern | Primary file(s) |
|---------|----------------|
| Backend registration | `python/torch_vulkan/inductor/__init__.py` |
| Scheduler / fusion | `python/torch_vulkan/inductor/scheduling.py` |
| Kernel codegen | `python/torch_vulkan/inductor/kernel/` |
| Lowerings | `python/torch_vulkan/inductor/lowerings/` |
| FX passes | `python/torch_vulkan/inductor/fx_passes/` |
| Runtime / slangc | `python/torch_vulkan/inductor/runtime.py` |
| bwd_diff dispatch | `python/torch_vulkan/inductor/bwd_diff_dispatch.py` |
| bwd_diff table | `python/torch_vulkan/inductor/bwd_diff_table.py` |
| Templates | `python/torch_vulkan/inductor/templates/` |
| C++ AOTI runtime | `csrc/backend/AotiRuntime.cpp` |
| Slang lib modules | `shaders/lib/` |
| Regression tests | `tests/test_inductor_regression.py` |
| E2E model tests | `tests/test_e2e_models.py` |

---

## 6. Building, Testing, Profiling

### Build
```bash
python -m pip install --no-build-isolation -v -e .
```

### Regression suite
```bash
python -m pytest backends/vulkan_slang/tests/test_inductor_regression.py -x -q
```

### E2E model tests
```bash
python -m pytest backends/vulkan_slang/tests/test_e2e_models.py -x -q
```

### Useful environment knobs
```bash
TORCH_VULKAN_DYNAMIC_SHAPES=1      # Variable-batch support (default ON)
TORCH_VULKAN_BATCH_DISPATCH=1      # Batch dispatch (default ON)
TORCH_VULKAN_PERSISTENT_POINTWISE=1 # Persistent kernels (default ON)
TORCH_VULKAN_GRID_AWARE_WG=1       # Grid-aware WG sizing (default ON)
TORCH_VULKAN_PROFILE_DISPATCHES=1  # Dispatch timing
TORCH_VULKAN_SPEC_CONSTANTS=1      # Spec constants (default ON)
TORCH_VULKAN_DESCRIPTOR_INDEXING=1 # >16 bindings (default ON)
TORCH_VULKAN_BANK_CONFLICT_PAD=1   # LDS bank padding (default ON)
TORCH_VULKAN_STATIC_SPECIALIZATION=1 # Static const (default ON)
TORCH_VULKAN_ASYNC_COMPILE=1       # Parallel slangc (default ON)
```
