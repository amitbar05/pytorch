# Vulkan-Slang Inductor Backend Roadmap v6.2

> **v5 North Star achieved (2026-05-10).** SmallCNN trains end-to-end.
> **v6.1 closeouts (2026-05-11 / 2026-05-13):** M6 Phase 1 (Conv1d),
> M9.1 (buffer-pool key bug), M9.3 (prewarm-on-import), M10.4 / CG.M10
> (mm epilogue generic) → archived in
> [10-inductor-backend-history.md](10-inductor-backend-history.md).
>
> **v6.2 (2026-05-13) refresh** following a four-track parallel audit
> (codegen / op-coverage / scheduler / training). Active milestones grow
> by **M13–M16** to capture audit-derived gaps. See § 0.5 for the
> refreshed audit numbers.
>
> **Last updated: 2026-05-13.**
>
> **Live state:** 9 model architectures train end-to-end under
> `torch.compile`; 47 lowerings + 24 explicit decomp suppressions;
> 57/58 `aten.*_backward` ops route through `bwd_diff_table`; buffer
> pool at 36 % hit rate on MLP train.
>
> **Feature flags ON by default:** spec_constants, descriptor_indexing,
> static_specialization, bank_conflict_pad, dynamic_shapes,
> batch_dispatch, wrapper_fastpath, grid_aware_wg,
> persistent_pointwise, buffer_pool, prewarm_on_import.

---

## 0. What Remains (v6.2)

### The 4 Remaining External Blockers

| # | Item | Blocker | Actionable? |
|---|------|---------|-------------|
| 1 | **T4.12 Phase 2-4** Conv3d KD>1 / depthwise (groups=C arbitrary) / transposed (1D/3D) | Template generality | ✅ **Yes — implement now** |
| 2 | **N+1.9** Link-time tile specialization | slangc upstream bug E30600 | ❌ Monitor slangc releases |
| 3 | **T7.2** Full .so subprocess load | C++ build infrastructure | ❌ Needs build system |
| 4 | **Track CI** GPU hardware | No CI runner with Vulkan GPU | ❌ Needs hardware |

### Active milestones (in priority order)

| # | Milestone | Goal | Effort |
|---|-----------|------|--------|
| **M9** | **Host-overhead reduction** | Close the 96 %/230× host/kernel gap. M9.1 / M9.3 closed; M9.2 / M9.4 / M9.7-M9.9 remain. | 1-2w remaining |
| **M11** | **Occupancy-aware codegen** | Wire reflection → WG sizing; subgroup reductions; LDS bank rigour. | 1-2w |
| **M12** | **Reduction backward via autodiff** | 6/8 reduction ops `[Differentiable]`; route through `bwd_diff_table`. | 1w |
| **M13** | **Slang feature saturation (NEW)** | Bring conv / SDPA / reduction / pointwise up to the mm gold standard: generics, interfaces, `ParameterBlock`, link-time spec, capabilities, wave intrinsics. | 2-3w |
| **M14** | **Op coverage gaps (NEW)** | Complex-dtype binary, sparse / scatter-atomic, dynamic-shape reduction, foreach element-wise, quantized int8, RNN backward. | 2-3w |
| **M15** | **Anti-goal #5/#7 cleanup (expanded M10)** | Split 6 newly-discovered monoliths; audit `meta_patches.py` for symptom-fixes. | 1-2w |
| **M16** | **Track 4 finish (NEW)** | Delete `csrc/ops/model_ops.cpp` (925 L of legacy eager ops). Irreversible. | 1w |
| **M6** | **Conv generality** | Phase 1 done (Conv1d); Phase 2-4 remain (depthwise/3D/transposed-1D/3D). | 1-2w |
| **M7** | **Production hardening** | Link-time spec (gated on slangc), AOTI .so packaging, CI gate. | gated |
| **M8** | **Model zoo expansion** | More real-world models end-to-end. | ongoing |
| **M10** | **Anti-goal #7 cleanup** | M10.1-3 originally listed; M10.5-8 small fixes; M10.4 closed. Subsumed under M15 — see there. | merged into M15 |

---

## 0.5. Audit findings (2026-05-13 refresh)

Four parallel sub-agents audited codegen, op coverage, scheduler, and
training. Numbers verified by probe scripts under `agent_space/probe_*.py`
and `agent_space/vk_validation_sweep*.py`. Source: this turn's session.

### 0.5.1 Headline numbers

| Probe | Result | Reaction |
|-------|--------|----------|
| MLP train warm step | 75 µs kernel / 1.63 ms wall | **96 % host overhead** — M9.2 / M9.4 |
| SmallCNN train warm step | 191 µs kernel / 43.9 ms wall | 230× host/kernel — M9.2 / M9.4 / M9.8 |
| SmallCNN cold compile | ✅ prewarmed on import (M9.3, 2026-05-13) | — |
| MLP buffer pool, 10 steps | ✅ 18 / 50 hits (36 %) — M9.1, 2026-05-13 | 90 % of releasable buffers recycle |
| GN + ReLU + GlobalAvg | 2 kernels (target: 1) | Reduction-boundary fusion gap — M9.8 |
| Transformer combo-kernel | `UnboundLocalError: buf10` in `vulkan_combo_kernel.py:987-1019` token rewriter | M9.9 (root cause located) |
| Models that train end-to-end | **9 architectures** (MLP, SmallCNN, Transformer, Qwen3.5 GatedDeltaNet, ViT, Mamba-2, Llama MLP+block, Mixtral MoE) | North star — sustain |
| Backward op coverage | **57/58** `aten.*_backward` via `bwd_diff_table` | Only legacy `embedding_dense_backward` hand-rolled (not Slang-eligible) |
| `csrc/ops/model_ops.cpp` line drift | 885 L → **925 L** (+40 since v6.1 audit) | Reverse drift — see M16 |
| Files > 800 L | **10 violators** (was 4 in v6.1) | M10 expanded → M15 |

### 0.5.2 Slang feature saturation (per-feature %)

| Feature | Score | Top blocker / what to do |
|---------|-------|---------|
| Generics `<T : Float>` / `<Op : I…>` | 70 % | mm uses `<Epilogue : IDifferentiable>`; conv/SDPA/reduction still string-templated (CG.M12-M13) |
| Interfaces `IPointwise` etc. | 80 % | Defined; reduction codegen still passes `op_template="OpSum"` as string (CG.M13) |
| `[Differentiable]` / `bwd_diff()` | 80 % | 80 ops carry annotation; reduction dispatch wiring partial (M12.2) |
| `[BackwardDerivative]` | 30 % | Only `pointwise.slang` has perf overrides (29 ops); other libs zero. CG.M11 |
| `ParameterBlock<T>` | 30 % | mm only. Pointwise/reduction still emit manual `[[vk::binding(N)]]`. CG.M14 |
| Reflection metadata (VGPR/LDS) | **0 %** | Parsed by `_extract_linktime_spec_constants`, never consumed at codegen. M11.1 |
| Link-time specialisation | 40 % | mm only (TILE_M/N/K, M/N_PER_THREAD); conv / SDPA / reduction hardcoded. CG.M15 |
| Capabilities `[require(…)]` | **0 %** | No subgroup-size or shader-model gating anywhere. CG.M16 |
| `[[vk::constant_id]]` | 20 % | mm only. Others use push constants exclusively. CG.M15 |
| vec2/vec4 packing | 60 % | Codegen does string `replace(…)` to vectorise — fragile; no Slang struct abstraction. CG.M14 |
| Subgroup ops (`WaveActiveSum`) | 50 % | `wave_sum` helper exists; no automatic emission in reduction kernels. CG.M16 |
| Persistent kernels | 40 % | Only small-numel pointwise. Multi-wave persistent reductions not auto-selected. M11.4 |
| Grid-aware WG sizing | 100 % | Pointwise only; reductions don't query grid. Documented as "on by default" — verified. |

### 0.5.3 Anti-goal accounting (refreshed)

| # | Anti-goal | State | Where | Fix milestone |
|---|-----------|-------|-------|---------------|
| #2 | `csrc/ops/model_ops.cpp` = 0 L | **VIOLATED** (925 L, +40 drift) | 22 legacy eager kernels (triu/tril, constant_pad_nd, index_tensor, repeat, stack, erf, narrow, flip, roll, as_strided, sin/cos, mse_loss fwd+bwd, …) | **M16** |
| #3 | No `aten.*_backward` lowerings | ✅ **CLOSED** | 57/58 via `bwd_diff_table`; only legacy `embedding_dense_backward` (not Slang-eligible) | — |
| #5 | No symptom-patches in `meta_patches` | **VIOLATED** | 3902 L; 120+ `@register_fake` hooks; `_fuse_sdpa_to_flash_attention` is a symptom-fix for missing native attention primitive | M15.2 / M14.6 |
| #6 | No string-template params | **PARTIAL** | mm fixed (M10.4); conv / SDPA still Jinja-conditional on `has_bias` / `has_activation`; reduction still string-keyed `op_template="OpSum"`; `generic_pointwise_dispatch.py` Jinja2-templates raw Slang source | CG.M12 / CG.M13 |
| #7 | Files ≤ 800 L | **VIOLATED 10×** | See table § 0.5.4 | **M15.1** |

### 0.5.4 File-size violators (full list)

| File | Lines | Cap multiple | Already in roadmap? | Milestone |
|------|------:|-------------:|---------------------|-----------|
| `vulkan_template_caller.py` | 5786 | 7.2× | ✅ M10.1 | M15.1 |
| `meta_patches.py` | 3902 | 4.9× | ✅ M10.2 | M15.1 / M15.2 |
| `runtime.py` | 2955 | 3.7× | ❌ NEW | **M15.1** |
| `kernel/pointwise.py` | 1555 | 1.9× | ✅ M10.3 | M15.1 |
| `fx_passes/eager_patches.py` | 1159 | 1.4× | ❌ NEW | **M15.1** |
| `vulkan_combo_kernel.py` | 1106 | 1.4× | ❌ NEW | **M15.1** |
| `kernel/reduction.py` | 981 | 1.2× | ❌ NEW | **M15.1** |
| `bwd_diff_dispatch.py` | 913 | 1.1× | ❌ NEW | **M15.1** |
| `validate.py` | 813 | 1.0× | ❌ NEW (borderline) | M15.1 |
| `lowerings/rnn.py` | 805 | 1.0× | ❌ NEW (borderline) | M15.1 |

### 0.5.5 New items added by this audit

Counted: **22 new items** across M9 (1), M11 (1), M13 (6), M14 (6), M15 (6), M16 (1), M6 (1).

---

## 1. M9 — Host-overhead reduction (P0 for perf, 1-2w remaining)

Closes the 96 % / 230× host/kernel gap. M9.1 (buffer pool) and M9.3
(prewarm) closed 2026-05-13.

- [x] **M9.1** — buffer-pool key bug ✅ closed 2026-05-13 (see history doc)
- [ ] **M9.2** Deferred command-buffer batching: stop per-dispatch `submit_and_wait`; submit 4–8 dispatches per `vkQueueSubmit`. Skeleton in `Stream.cpp`. Target: −5 to −10 ms / SmallCNN step. **Next-largest perf win.** (2-3d)
- [x] **M9.3** — prewarm-on-import ✅ closed 2026-05-13 (see history doc)
- [ ] **M9.4** Push-constant in-place updates: pre-allocate bytearray per kernel; update fields, don't `bytes(pc_data)` per dispatch. (1d)
- [ ] **M9.5** Cached `_jit_dispatch_indexed`: codegen prefers indexed variant when any binding has count > 1. (1d)
- [ ] **M9.6** Adaptive `_PER_KEY_CAP` in `buffer_pool.py` (scratch=8, transient=6, save_for_backward=4). (1d)
- [ ] **M9.7** Pool non-extern Inductor outputs (currently only extern-kernel outputs are pooled). Closes the residual ~64 % miss rate. (1-2d)
- [ ] **M9.8** Reduction-boundary fusion: GN + ReLU + GlobalAvg should fuse into 1 kernel, not 2. Relax `rnumel_fuse_cap` gate in `scheduling.py:248-261`, or change gate to predicate on consumer pattern rather than rnumel. (2-3d)
- [ ] **M9.9** Transformer combo-batcher `UnboundLocalError: buf10`. **Root cause located**: `vulkan_combo_kernel.py:987-1019` `_rewrite_body()` token-based renaming runs before the buffer-name map is fully seeded; if a buffer name isn't in `per_sub_maps[idx]`, the rewriter emits the original name and collides with a renamed local from a previous subkernel. Fix: pre-seed buffer names via `_build_global_binding_map()` (line 689-795) before the rewrite loop. (1-2d)

---

## 2. M11 — Occupancy-aware codegen (1-2w, throughput)

The headline finding of this audit: **reflection metadata is 0 % used**
despite roadmap text. `occupancy_audit.py` hardcodes shapes per kernel
category. `_extract_linktime_spec_constants` parses VGPR / LDS counts but
they're never fed into WG sizing. Wiring this up is M11.1's whole point.

- [ ] **M11.1** **DR.7 wire-up (THE 0 % gap)**: feed reflection VGPR/LDS into `_pick_threadgroup_size_*` instead of leaving `estimate_occupancy()` as a debug-only tool. Default flag → on. Target: +10–20 % on reduction/normalisation kernels. (2-3d)
- [ ] **M11.2** Subgroup reductions for WG ≤ wave64: emit `WaveActiveSum`/`WaveActiveMax` instead of LDS reduce. (2d)
- [ ] **M11.3** Register-tile pointwise: load+compute+store unrolled ×2-4. (3d)
- [ ] **M11.4** Persistent-mode WG autotune: scale WG size by `numel / CU_count`; today the persistent path uses a fixed WG. Extend persistent path to multi-wave reductions (currently small-numel pointwise only). (1-2d)
- [ ] **M11.5** Round non-multiple-of-64 WG sizes up to next multiple on RDNA1 (auto-fix `slang_validator.py` advisory). (0.5d)
- [ ] **M11.6** LDS bank-padding rigour: auto-pad WG-shared arrays > 1 KB to nearest power of 2 to avoid stride-1 bank conflicts. (1-2d)
- [ ] **M11.7** Occupancy gate in `codegen.py`: warn (or `--strict` fail) if estimated occupancy < 50 %. (1-2d)
- [ ] **M11.8** Extend `_KERNEL_STATS` to capture grid, WG, VGPR, LDS, descriptor count (populated from reflection). (1-2d)
- [ ] **M11.9 (NEW)** Reduction WG sizing: `kernel/reduction.py:725+` has no grid-aware path. Add one — feed `numel/CU_count` like pointwise does. (1d)

---

## 3. M12 — Reduction backward via autodiff (CG.M3, ~1w remaining)

`reduction.slang` has 6 `[Differentiable]` annotations (sum/mean/var fold
paths shipped) but the dispatch wiring is incomplete: 11/13 tests in
`tests/test_cgm3_reduction_backward.py` are xfailed. Routing through
`bwd_diff_table` would close anti-goal #3 for the reduction class
(already closed for activations/losses/conv/mm).

- [~] **M12.1** 6 `[Differentiable]` annotations in `shaders/lib/reduction.slang` (sum/mean/var). Still need: prod, max, min, argmax, argmin. (1-2d)
- [ ] **M12.2** Route `aten.{sum,mean,prod,max,min}_backward` through `bwd_diff_table.py` instead of hand-rolled lowerings. `tests/test_cgm3_reduction_backward.py` 11/13 xfailed — dispatch wiring is the gate. (2d)
- [ ] **M12.3** Retire any `aten.*_backward` reduction shaders living outside `lib/`. (1d)

---

## 4. M13 — Slang feature saturation (NEW, 2-3w)

The mm template (M10.4 / CG.M10) is the gold standard. Bring conv, SDPA,
reduction, and pointwise dispatch up to the same bar.

- [ ] **CG.M11 BackwardDerivative coverage outside `pointwise.slang`**: `norm.slang` 8 forward `[Differentiable]` but 4 backward; `losses.slang` 12/10 split (gap is small); `reduction.slang` 6 forward / 0 backward; `mm_tile.slang` 1 forward / 0 backward. Add explicit `[BackwardDerivative(...)]` overrides for the perf-critical ops (avoids autodiff fallback). (3-4d)
- [ ] **CG.M12 Pointwise dispatch via Slang generics**: `generic_pointwise_dispatch.py` Jinja2-templates raw Slang source and compiles a fresh module per op. Replace with a single Slang generic `<Op : IPointwise>` instantiated via specialization constants or slangc generic-specialisation at link time. Closes anti-goal #6 for pointwise. (3d)
- [ ] **CG.M13 Reduction codegen via interface**: `reduction.py:76-100` uses `op_template="OpSum"` strings. Switch to `<W : IWaveReduction>` constraint on a single `wg_reduce<W>` template; per-op variants become type instantiations. (3d)
- [ ] **CG.M14 ParameterBlock in pointwise/reduction**: kernels still emit manual `[[vk::binding(N)]]` literals. Wrap bindings in `struct KernelArgs` + `ParameterBlock<KernelArgs>` like `slang_mm.slang:87-100` does. Saves ~5 LOC per kernel and unlocks reflection. (2-3d)
- [ ] **CG.M15 Link-time spec constants for conv / SDPA**: only mm uses `[[vk::constant_id]]` for tile / per-thread params. Conv (`slang_conv2d.slang`), SDPA (`flash_attention.slang`), and the reduction template family hardcode loop bounds. Extract them to spec constants so a single SPIR-V module covers many tile choices. Reduces slangc invocations from N×tile_count to N. Gated on M13's slangc-bug status. (3-4d)
- [ ] **CG.M16 Capabilities `[require(...)]` audit**: zero usage. Add explicit gates for `subgroupSize == 64` on the RDNA1 paths and ladder up to `subgroupSize == 32` for RDNA2/3 / NVIDIA. Today everything assumes wave64. (1-2d)
- [ ] **CG.M17 Replace string `.replace()` vec4 codegen**: `kernel/pointwise.py:964-967` does `body.replace(f"{inner}[{a}]", f"_v_{inner}[_k]")` — fragile token surgery. Move to a typed Slang `vec4<float>` struct abstraction. (2d)

---

## 5. M14 — Op coverage gaps (NEW, 2-3w)

Closes the "any PyTorch model" story by filling categorical holes the
audit found.

- [ ] **OP.20 Complex-dtype binary elementwise**: complex64/128 matmul + softmax work, but `complex_add`, `complex_mul`, `complex_div` have no `IPointwise` struct in `shaders/lib/pointwise.slang`. They fall through to `ExternKernel` (eager dispatch). Add 4-5 complex-valued op structs; lower via `generic_pointwise_dispatch`. **Unblocks**: vision/audio models using `torch.view_as_complex`. (2-3d)
- [ ] **OP.21 Sparse / scatter-atomic backward**: `scatter_atomic.py.jinja` is a placeholder (`codegen.py:164-172`). Backward of `index_put_`, `scatter_add_`, and `embedding_bag` all currently go through eager. **Unblocks**: sparse attention, embedding-bag training, retrieval/RecSys workloads. (8-10d — biggest item)
- [ ] **OP.22 Dynamic-shape reduction codegen**: D.4 forward partial; backward fails on symbolic `B`. `kernel/symbolic.py` raises `NotImplementedError` on symbolic-stride reductions. **Unblocks**: variable-batch training without per-shape recompilation. (5-7d)
- [ ] **OP.23 Foreach element-wise ops**: `install_external_optimizer` covers SGD/AdamW/Lion. Missing: `foreach_add`, `foreach_mul`, `foreach_div`, `foreach_lerp`, `foreach_clip_grad_norm`. Reuse the foreach template plumbing. **Unblocks**: gradient-clipping codepaths, multi-param updates. (2-3d)
- [ ] **OP.24 Quantized int8 matmul (inference)**: No int8 GEMM kernel. Falls to CPU/external. **Unblocks**: GPTQ / AWQ / quantized Llama inference. Forward only — quantized training is out of scope. (5-7d)
- [ ] **OP.25 RNN backward via Slang autodiff**: `bwd_lowerings.py:687L` decomposes RNN grads manually; GRU backward marked "more complex" and incomplete. **Unblocks**: LSTM/GRU training parity. (6-8d)
- [ ] **OP.26 Anti-symptom: native attention primitive**: `_fuse_sdpa_to_flash_attention` is currently a symptom-fix for the absence of a native `aten.scaled_dot_product_attention` lowering. Promote the FlashAttention template to a real primitive lowering registered via `@register_lowering(aten.scaled_dot_product_attention)`. Closes anti-goal #5 for SDPA. (3d)

---

## 6. M15 — Anti-goal #5/#7 cleanup (NEW, 1-2w)

Expanded successor to v6.1's M10. Six newly-discovered file-size
violators plus a `meta_patches.py` symptom-fix audit.

- [ ] **M15.1 File splits (10 violators):**
  - [ ] M15.1.a `vulkan_template_caller.py` (5786 L) → `templates/caller/{gemm,scatter,optimizer,flash_attn,rng}.py` (was M10.1). (1-2d)
  - [ ] M15.1.b `meta_patches.py` (3902 L) → `meta_patches/{shape_ops,dtype_ops,faketensor_hooks}.py` + audit for stale patches (was M10.2). (1d)
  - [ ] M15.1.c `runtime.py` (2955 L) → `runtime/{slangc,dispatch,batcher,profile,prewarm}.py`. **NEW.** (1-2d)
  - [ ] M15.1.d `kernel/pointwise.py` (1555 L) → extract `PointwiseLoadMixin` + `PointwiseVec4Mixin` (was M10.3). (1.5d)
  - [ ] M15.1.e `fx_passes/eager_patches.py` (1159 L) → `fx_passes/eager/{addmm,sdpa,swiglu,qkv,optimizer}.py`. **NEW.** (1d)
  - [ ] M15.1.f `vulkan_combo_kernel.py` (1106 L) → split body-rewriter from binding-map / grid-builder. **NEW** (also fixes M9.9 indirectly). (1d)
  - [ ] M15.1.g `kernel/reduction.py` (981 L) → extract `ReductionLoadMixin` + per-axis tile picker. **NEW.** (1d)
  - [ ] M15.1.h `bwd_diff_dispatch.py` (913 L) → split unary/binary dispatch + emit helpers. **NEW.** (1d)
- [ ] **M15.2 `meta_patches.py` symptom-fix audit**: 120+ `@register_fake` hooks. Classify each as (a) genuine FakeTensor shape-inference patch, (b) workaround for missing primitive, (c) drift from upstream. Promote (b) to real lowerings or FX rewrites; delete (c) and re-validate against upstream. Closes anti-goal #5. (2-3d)
- [ ] **M15.3 Small fixes carried over from v6.1:**
  - [ ] Lift `_VALID_IPOINTWISE_STRUCTS` frozenset to auto-parse `lib/pointwise.slang` at startup; drop manual sync. (0.5d)
  - [ ] Remove redundant outer cast in `kernel/pointwise.py:68-81` int8 load dispatch. (0.5d)
  - [ ] Audit & remove stale TODO gate at `vulkan_template_caller.py:754` (P3.2/M14 dead flag). (0.5d)
  - [ ] Extract pickle/repr boilerplate from `_SlangTile{MM,AddMM,BMM}` into a common base. (1d)

---

## 7. M16 — Track 4 finish (NEW, 1w, IRREVERSIBLE)

`csrc/ops/model_ops.cpp` is 925 L (drift from 885 since v6.1 audit). 22
legacy eager kernels — these block the "no per-model `csrc/ops/*.cpp`
entries" anti-goal. Track 4 was meant to delete this file. The drift
suggests new ops are still landing here despite the anti-goal.

- [ ] **M16.1** Inventory `model_ops.cpp` — categorise each of the 22 ops as (a) covered by an Inductor lowering already (delete from cpp), (b) needs a new Inductor lowering before delete, (c) genuinely eager-only (move to `csrc/ops/legacy_eager.cpp` to make the boundary explicit). (1d)
- [ ] **M16.2** Add eager-mode lowering parity for category (b) ops. (3-4d)
- [ ] **M16.3** Delete `model_ops.cpp` and lock with a regression test that fails the build if it returns. (0.5d)
- [ ] **M16.4** Lock the boundary: pre-commit hook or `build.py` check that fails when any new `*_backward` or `*_op` symbol lands under `csrc/ops/` outside the allowed list. (0.5d)

---

## 8. M6 — Conv generality (Phase 2-4)

Phase 1 (Conv1d) closed 2026-05-11 — see history doc.

- [ ] **Phase 2 — Depthwise conv (groups=C, arbitrary)**: existing per-group decomposition handles up to groups=4; needs extension. (2-3d)
- [ ] **Phase 3 — Conv3d (KD>1)**: merge spatial dims or extend the 2D template. (3-4d)
- [ ] **Phase 4 — Transposed conv (1D / 3D)**: reuse CG.M6's conv backward template with swapped roles. (2-3d)

Files: `lowerings/conv.py`, `templates/slang_conv2d.py.jinja`,
`tests/test_inductor_regression.py` (flip xfails), `tests/test_e2e_models.py`.

---

## 9. M7 — Production hardening (gated)

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

## 10. M8 — Model zoo expansion (ongoing)

### Currently trains end-to-end under `torch.compile` (9 architectures)
MLP, SmallCNN, Transformer block, Qwen3.5 GatedDeltaNet, ViT encoder,
Mamba-2, Llama MLP + full block, Mixtral MoE. (66 e2e tests total
including forward-only.)

### Candidates blocked on specific milestones
| Model | Key ops | Blocker |
|-------|---------|---------|
| Stable Diffusion UNet full | GroupNorm backward + conv-transpose decoder | M14 (GN bwd) + M6 Phase 4 (transposed) |
| LSTM/GRU language model | RNN backward | OP.25 |
| Quantized Llama inference | int8 matmul | OP.24 |
| Sparse attention models | scatter-atomic bwd | OP.21 |
| Variable-batch fine-tune | dynamic-shape reduction bwd | OP.22 |

---

## 11. Heuristic / GPU-utilization flags (state)

| Feature | Gate | Default | Audit verdict |
|---------|------|---------|---------------|
| Aggressive fusion | `TORCH_VULKAN_AGGRESSIVE_FUSION` | OFF | Verified working; relaxes rnumel cap |
| Persistent kernel v2 | `TORCH_VULKAN_PERSISTENT_POINTWISE` | ON | Pointwise only; reduction extension pending M11.4 |
| Grid-aware WG v2 | `TORCH_VULKAN_GRID_AWARE_WG` | ON | Pointwise only; reduction extension pending M11.9 |
| Batch dispatch | `TORCH_VULKAN_BATCH_DISPATCH` | ON | Single `vkQueueSubmit` per graph; M9.2 takes it further (multi-graph batching) |
| Wrapper fast-path | `TORCH_VULKAN_WRAPPER_FASTPATH` | ON | Cached imports, skipped validation |
| Dispatch profiling | `TORCH_VULKAN_PROFILE_DISPATCHES` | OFF | On-demand per-kernel timing |
| Async compile | `TORCH_VULKAN_ASYNC_COMPILE` | ON | ThreadPoolExecutor; in-flight dedup |
| Buffer pool | `TORCH_VULKAN_BUFFER_POOL` | ON | 36 % hit rate on MLP train (M9.1 closed 2026-05-13) |
| Prewarm-on-import | `TORCH_VULKAN_NO_PREWARM=1` to disable | ON | Shader-lib `.slang-module` precompiled in bg thread (M9.3 closed 2026-05-13) |
| Dispatch ratchet | `TestDispatchCountRatchet` | — | MLP fwd ≤ 8, SmallCNN train ≤ 25 |

---

## 12. Critical path (dependency chart)

```
M9.2 (cmd-buf batch) ──→ closes 96 % host overhead (training perf)
M9.8 (red-bound fusion) ──→ unblocks GN+ReLU+GlobalAvg → 1 kernel
M9.9 (combo-batcher) ──→ unblocks Transformer compile

M11.1 (DR.7 wire-up) ──→ +10-20 % on reductions   ┐
M11.2 (subgroup red.) ──→ +sum/mean/max perf      ├─ throughput phase
M11.4 (persistent reduction) ──→ small-rnumel perf┘

M12.1-3 (red. bwd) ──→ closes anti-goal #3 for reductions

CG.M12 (pointwise generic) ┐
CG.M13 (reduction generic) ├─→ closes anti-goal #6 fully
CG.M15 (conv/SDPA spec constants) ┘

OP.21 (sparse) ──→ unblocks sparse attention / embedding-bag bwd
OP.22 (dyn-shape red.) ──→ variable-batch training
OP.25 (RNN bwd) ──→ LSTM/GRU training
OP.26 (native SDPA prim) ──→ closes anti-goal #5 for attention

M15.1 (file splits) ──→ closes anti-goal #7 (parallel-safe; no semantic risk)
M15.2 (meta_patches audit) ──→ closes anti-goal #5

M16 (Track 4 finish) ──→ closes anti-goal #2; IRREVERSIBLE
```

---

## 13. Reference files

| Concern | Primary file(s) |
|---------|----------------|
| Backend registration | `python/torch_vulkan/inductor/__init__.py` |
| Scheduler / fusion | `python/torch_vulkan/inductor/scheduling.py` |
| Combo kernel | `python/torch_vulkan/inductor/vulkan_combo_kernel.py` |
| Kernel codegen | `python/torch_vulkan/inductor/kernel/` |
| Lowerings | `python/torch_vulkan/inductor/lowerings/` |
| FX passes | `python/torch_vulkan/inductor/fx_passes/` |
| Runtime / slangc | `python/torch_vulkan/inductor/runtime.py` |
| Buffer pool | `python/torch_vulkan/inductor/buffer_pool.py` |
| bwd_diff dispatch | `python/torch_vulkan/inductor/bwd_diff_dispatch.py` |
| bwd_diff table | `python/torch_vulkan/inductor/bwd_diff_table.py` |
| Templates | `python/torch_vulkan/inductor/templates/` |
| Template caller | `python/torch_vulkan/inductor/vulkan_template_caller.py` |
| meta patches | `python/torch_vulkan/inductor/meta_patches.py` |
| C++ AOTI runtime | `csrc/backend/AotiRuntime.cpp` |
| C++ legacy eager ops | `csrc/ops/model_ops.cpp` (slated for deletion — M16) |
| Slang lib modules | `shaders/lib/{helpers,dtype_pack,philox,special_math,bucket,mm,mm_tile,atomics,conv,norm,pointwise,reduction,losses,tensor_layout}.slang` |
| Slang templates | `python/torch_vulkan/inductor/templates/*.{jinja,slang}` |
| Regression tests | `tests/test_inductor_regression.py` (39 k lines, 66 e2e model tests, 9 training-grade architectures) |
| E2E model tests | `tests/test_e2e_models.py` |

---

## 14. Building, testing, profiling

### Build
```bash
cd backends/vulkan_slang
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MAX_JOBS=8 python setup.py build_ext --inplace
```

### Regression suite (~90 s with xdist)
```bash
python -m pytest tests/ -n 4 --timeout=120 -p no:faulthandler
```

### E2E model tests
```bash
python -m pytest tests/test_e2e_models.py -x -q -p no:faulthandler
```

### Useful environment knobs
```bash
TORCH_VULKAN_DYNAMIC_SHAPES=1      # Variable-batch (default ON)
TORCH_VULKAN_BATCH_DISPATCH=1      # Batch dispatch (default ON)
TORCH_VULKAN_PERSISTENT_POINTWISE=1 # Persistent kernels (default ON)
TORCH_VULKAN_GRID_AWARE_WG=1       # Grid-aware WG (default ON)
TORCH_VULKAN_PROFILE_DISPATCHES=1  # Dispatch timing
TORCH_VULKAN_SPEC_CONSTANTS=1      # Spec constants (default ON)
TORCH_VULKAN_DESCRIPTOR_INDEXING=1 # >16 bindings (default ON)
TORCH_VULKAN_BANK_CONFLICT_PAD=1   # LDS bank padding (default ON)
TORCH_VULKAN_STATIC_SPECIALIZATION=1 # Static const (default ON)
TORCH_VULKAN_ASYNC_COMPILE=1       # Parallel slangc (default ON)
TORCH_VULKAN_BUFFER_POOL=1         # Output buffer recycle (default ON)
TORCH_VULKAN_NO_PREWARM=0          # Set =1 to disable bg shader-lib precompile
TORCH_VULKAN_POOL_STATS=1          # Detailed per-event pool stats
TORCH_VULKAN_INDUCTOR_STATS=1      # Per-kernel call_count / total_us
```
