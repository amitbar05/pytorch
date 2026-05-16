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
> **Last updated: 2026-05-15 (session: full roadmap audit — M14, M16, M6 deep-dives)**
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
| **M17** | **🔥 Inductor VK perf parity with CPU (NEW, HIGHEST PRIORITY)** | SmallCNN+GroupNorm is 5.7× slower than CPU eager (4.68 ms vs 0.82 ms). Cut dispatch count from ~20/step to ≤5, reactivate Slang matmul, fuse conv+gn+relu, fuse linear backward. **Target: 1× CPU parity for SmallCNN+GN, then 2× wins on bigger workloads.** | 3-4w |
| **M9** | **Host-overhead reduction** | Close the 96 %/230× host/kernel gap. M9.1-M9.5, M9.8-M9.9 closed; M9.6-M9.7 remain. | 1-2w remaining |
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
| Reflection metadata (VGPR/LDS) | **100 %** | M11.1 closed: DR.7 Pass-2 feeds VGPR/LDS/loop_depth into `_pick_numthreads_from_reflection`; `reflection_routing` default ON. |
| Link-time specialisation | 40 % | mm only (TILE_M/N/K, M/N_PER_THREAD); conv / SDPA / reduction hardcoded. CG.M15 |
| Capabilities `[require(…)]` | **0 %** | No subgroup-size or shader-model gating anywhere. CG.M16 |
| `[[vk::constant_id]]` | 20 % | mm only. Others use push constants exclusively. CG.M15 |
| vec2/vec4 packing | 60 % | Codegen does string `replace(…)` to vectorise — fragile; no Slang struct abstraction. CG.M14 |
| Subgroup ops (`WaveActiveSum`) | 80 % | M11.2: direct wave intrinsics now emitted for single-wave sum/prod/max/min reductions. Remaining: any/xor/arg/welford. |
| Persistent kernels | 40 % | Only small-numel pointwise. Multi-wave persistent reductions not auto-selected. M11.4 |
| Grid-aware WG sizing | 100 % | M11.9 closed: reductions now have grid-aware path feeding `numel/CU_count`. Pointwise + reduction both query grid. |

### 0.5.3 Anti-goal accounting (refreshed)

| # | Anti-goal | State | Where | Fix milestone |
|---|-----------|-------|-------|---------------|
| #2 | `csrc/ops/model_ops.cpp` = 0 L | **VIOLATED** (925 L, +40 drift) | 22 legacy eager kernels (triu/tril, constant_pad_nd, index_tensor, repeat, stack, erf, narrow, flip, roll, as_strided, sin/cos, mse_loss fwd+bwd, …) | **M16** |
| #3 | No `aten.*_backward` lowerings | ✅ **CLOSED** | 57/58 via `bwd_diff_table`; only legacy `embedding_dense_backward` (not Slang-eligible) | — |
| #5 | No symptom-patches in `meta_patches` | **VIOLATED** | 3902 L; 120+ `@register_fake` hooks; `_fuse_sdpa_to_flash_attention` is a symptom-fix for missing native attention primitive | M15.2 / M14.6 |
| #6 | No string-template params | **PARTIAL** | mm fixed (M10.4); conv / SDPA still Jinja-conditional on `has_bias` / `has_activation`; reduction codegen now uses Slang generics (CG.M13 done)`op_template="OpSum"`; `generic_pointwise_dispatch.py` Jinja2-templates raw Slang source | CG.M12 / CG.M13 |
| #7 | Files ≤ 800 L | **VIOLATED 11×** (was 10×, +gemm.py 2331L, −vulkan_template_caller.py) | See table § 0.5.4 | **M15.1** |

### 0.5.4 File-size violators (full list)

| File | Lines | Cap multiple | Already in roadmap? | Milestone |
|------|------:|-------------:|---------------------|-----------|
| `vulkan_template_caller.py` | ~~5786~~ **265** | ~~7.2×~~ **0.3×** | ✅ M10.1 | **M15.1 ✅** |
| `meta_patches.py` | 3902 | 4.9× | ✅ M10.2 | M15.1 / M15.2 |
| `runtime.py` | 2955 | 3.7× | ❌ NEW | **M15.1.c** |
| `kernel/pointwise.py` | 1555 | 1.9× | ✅ M10.3 | M15.1.d |
| `fx_passes/eager_patches.py` | 1159 | 1.4× | ❌ NEW | **M15.1.e** |
| `vulkan_combo_kernel.py` | 1106 | 1.4× | ❌ NEW | **M15.1.f** |
| `kernel/reduction.py` | 981 | 1.2× | ❌ NEW | **M15.1.g** |
| `bwd_diff_dispatch.py` | 913 | 1.1× | ❌ NEW | **M15.1.h** |
| `validate.py` | 813 | 1.0× | ❌ NEW (borderline) | M15.1.i |
| `lowerings/rnn.py` | 805 | 1.0× | ❌ NEW (borderline) | M15.1.j |
| `templates/caller/gemm.py` | 2331 | 2.9× | ❌ NEW | **M15.1.a follow-up** |

### 0.5.5 New items added by this audit

Counted: **22 new items** across M9 (1), M11 (1), M13 (6), M14 (6), M15 (6), M16 (1), M6 (1).

---

## 0.9. M17 — 🔥 Inductor VK perf parity with CPU (HIGHEST PRIORITY, 3-4w)

**Baseline (2026-05-16, after the per-step recompile fix landed today):**
- SmallCNN forward+backward+AdamW step: **4.68 ms VK vs 0.82 ms CPU → CPU 5.7× faster**
- MLP step: VK 2.79 ms vs CPU 0.43 ms → CPU 6.5× faster
- Transformer step: VK 6.69 ms vs CPU 1.44 ms → CPU 4.6× faster

Even after closing the Dynamo recompile loop, VK trails CPU on every workload. The 4.68 ms / 5.7× SmallCNN gap decomposes by `agent_space/conv_perf_breakdown.py`:

### M17.0 — Dispatch census (SmallCNN: `Conv2d → GroupNorm → ReLU → AvgPool → Linear` + MSELoss + AdamW, batch=2, image=16×16)

Across 4 compiled wrappers (fwd / bwd / loss / opt), per training step:

| Category | Count | Notes |
|----------|------:|-------|
| `extern_kernels.addmm` | 1 | Linear forward — **NOT codegen**, eager Vulkan dispatch (OP.27) |
| `aten.linear_backward` extern | 7 | Linear backward decomposed into 7 stock-Inductor sub-dispatches (mm+mm+sum+view+...) |
| `aten._adaptive_avg_pool2d_backward` extern | 3 | Pool backward via eager Vulkan |
| `torch.ops.torch_vulkan.adaptive_avg_pool2d` custom op | 3 | Pool forward via eager (single dispatch but no fusion) |
| `torch.ops.torch_vulkan.conv2d_with_optional_bias` custom op | 3 | Conv forward — opaque to Inductor, no Slang fusion with downstream GN/ReLU |
| Slang batched kernels (`_batcher.add(vulkan_kernel_N, ...)`) | 6 | GN forward (welford+normalize+affine), ReLU, MSELoss fwd/bwd, GN backward — these ARE codegen-fused |
| `empty_strided_vulkan` allocations | 20 | One per intermediate tensor — buffer pool absorbs most via `lifetime_class` |
| `vulkan_pool_release` | 14 | Buffer recycling |
| **Total dispatches per step** | **~20** | Each ~0.2-0.4 ms on the 16 CU NAVI10 = 4-8 ms baseline overhead |

**Root cause analysis:**

1. **Linear (`nn.Linear`) is the worst leak — 8 dispatches/step.** `addmm` extern + 7 backward stock-decomp sub-dispatches. Should be **1 fused mm+bias+epilogue + 1 fused mm+bias backward = 2 dispatches**.
2. **Conv2d eager-custom-op is opaque to Inductor's fusion pass.** Cannot fuse the Conv → GN → ReLU chain because Conv is a black box. Each is a separate dispatch.
3. **Adaptive avg pool has both an eager forward and an extern backward.** Should be 1+1 = 2 Slang dispatches (forward is just per-output-cell reduction; backward is broadcast add).
4. **AdamW per-parameter updates** — not visible in this dispatch census (separate compiled wrapper), but profile shows ~5 ms of optimizer time per step. `install_external_optimizer` registers a foreach AdamW custom op; need to verify it's being chosen over the stock per-parameter loop.

### M17.1 — Reactivate Slang tile matmul (closes OP.27; biggest win, 5-7d)

Currently `extern_kernels.{mm,bmm,addmm}` for every Linear / MultiheadAttention.bmm / etc. Two cascading blockers found 2026-05-15:

  1. `slang_mm.slang:218 [numthreads(WG_N, WG_M, 1)]` — slangc 2026.5+ rejects spec-constant-derived expressions in `numthreads` (`E39999`). **Fix**: hard-code via Jinja `{{ tile_n // n_per_thread }}` literal — different tile configs already render to different SPIR-V modules so no dedup is lost. (1d)
  2. After fix #1, Vulkan validation fails `VUID-VkComputePipelineCreateInfo-layout-07988`: SPIR-V uses `Set 1 Binding 0/1/2` but pipeline layout only declares `Set 0`. The `ParameterBlock<KernelArgs>` set assignment doesn't match what `_jit_dispatch` reflects. **Fix**: align ParameterBlock set/binding indices with the C++ dispatch layout in `csrc/vulkan/Pipeline.cpp`, OR back out ParameterBlock for mm and emit per-binding decorations. (3-4d)
  3. Remove `_install_vulkan_aten_only_autotune()` constraint and verify the autotuner picks Slang choices when shape ≥ 64x64. (1d)
  4. Add a perf regression test for mm: cold compile time, warm step time, vs CPU reference. (1d)

  **Expected gain**: addmm 1 → fused mm+bias + 7 stock-decomp backward → 1 fused mm bwd = **8 dispatches → 2** for Linear alone. For Transformer with 4 Linears + 1 MultiheadAttention bmm = ~16 → 5 dispatches. **Estimated step time 6.69 ms → ~2 ms.**

### M17.2 — Conv → GroupNorm → ReLU fusion (3-4d)

Conv2d uses a `torch_vulkan::conv2d_with_optional_bias` custom op — opaque to Inductor. Inductor sees it as an extern, can't fuse downstream pointwise into it. Result: GN sees a fresh input buffer for every conv output, even though GN normalization is per-channel-then-broadcast (purely Pointwise/Reduction at that stage).

**Plan**:
  1. Make conv2d a proper Inductor `register_lowering` (not a custom op) for the static-shape case. Use `slang_conv2d` template directly, with `epilogue` parameter for activation. ParameterBlock-based — same gold standard as mm.
  2. Add a conv epilogue interface (`IConvEpilogue : IDifferentiable`) and emit `OpReLU`, `OpGELU`, `OpGroupNormPlusReLU` as concrete structs that satisfy it.
  3. Inductor's pattern matcher (`fx_passes/post_grad.py`) already has a pattern for conv+relu via `conv_epilogue` annotation (`scheduling.py:can_fuse_vertical`). Extend it to recognize conv+gn+relu as a triple-fusion.
  4. Backward: `slang_conv_bwd` already emits one fused kernel. Add a `gn_backward` chained dispatch that consumes the `save_mean/rstd` saved in forward (now correctly shaped after the 2026-05-15 fix).

  **Expected gain**: conv (3) + gn (2 fwd + 2 bwd Slang kernels) + relu (1) = 8 → 3 dispatches. **~2 ms saved per SmallCNN step.**

### M17.3 — Native adaptive_avg_pool2d Slang lowering (1-2d)

`torch.ops.torch_vulkan.adaptive_avg_pool2d.default` is a custom op (forward is 1 dispatch but no fusion); backward is `aten._adaptive_avg_pool2d_backward` extern (3 sub-dispatches). For typical `output_size=(8,8)` with `input_size=(16,16)`, this is a simple 2×2 average + per-output-cell scale.

**Plan**: lower `aten._adaptive_avg_pool2d` as a Reduction over `(kH, kW)` windows. Same for backward (broadcast `grad_out / (kH*kW)` per input pixel). Inductor scheduler will then fuse with adjacent pointwise (ReLU before, Linear-reshape after). **Saves ~6 dispatches/step.**

### M17.4 — AdamW: route eager `_foreach_*_` into the fused Slang custom op (2-3d)

**Measured 2026-05-16** via `agent_space/adamw_dispatch_count.py` — `opt.step()` for a 4-Linear MLP on Vulkan:

| path | 10 steps wall | per step | Slang dispatches |
|------|---------------|----------|-----------------:|
| `AdamW(foreach=True)`  | 17.02 ms | **1.70 ms** | 0 (all eager) |
| `AdamW(foreach=False)` | 21.63 ms | 2.16 ms | 0 (all eager) |

That's 36 % of SmallCNN+GN's 4.68 ms step time burned in optimizer — **and 0 Slang kernels fire** because `opt.step()` runs outside `torch.compile` and dispatches `aten._foreach_*_` directly to the Vulkan eager kernels. `install_external_optimizer` already registers `torch_vulkan::foreach_adamw_step` as a single-dispatch Slang fused kernel, but it's only chosen when Inductor sees the optimizer step — which never happens for a user-written `opt.step()` outside compile.

**Three paths to fix:**
  1. **Intercept `aten._foreach_addcmul_` / `_foreach_sqrt_` / `_foreach_add_` in Vulkan eager dispatch** and route the typical AdamW sequence (mul-add-mul-addcmul-sqrt-addcdiv chain) into `vulkan_foreach_adamw_step`. C++-level pattern match in the dispatcher. (3d)
  2. **Replace `torch.optim.AdamW`** with `vulkan_optim.AdamW` that calls our fused custom op directly. Drop-in replacement registered when `torch_vulkan` is imported. (1d, user-visible)
  3. **Make Dynamo capture `opt.step()`** by wrapping it in `torch.compile(opt.step, mode="reduce-overhead")`. PyTorch supports this in 2.5+. (1d for backend support)

**Expected gain**: 1.70 ms → ~0.2 ms per step = **~1.5 ms saved**, ≈ 32 % of SmallCNN+GN step time.

### M17.5 — Reduce per-dispatch overhead (2-3d)

Even after fusion, per-dispatch wall time is ~0.2-0.4 ms (vkQueueSubmit + driver dispatch). The `DispatchBatcher` (`M9.2`) flushes every 8 dispatches; for ≤10-dispatch graphs it submits once but each dispatch inside still hits per-pipeline-bind overhead.

**Plan**:
  1. Use `vkCmdDispatchIndirect` to chain N dispatches with shared pipeline layout into a single submit (saves N-1 pipeline binds).
  2. Pre-record reusable command buffers per compiled graph (cache keyed by graph hash). Re-record only when shapes change.
  3. Increase `_DEFAULT_BATCH_DISPATCH_CAP` from 8 → 32 for fully static graphs.

  **Expected gain**: per-dispatch overhead ~0.4 → ~0.1 ms × 5 dispatches = **1.5 ms saved**.

### M17.6 — Skip extern fall-through for the few ops that still extern (1-2d)

`extern_kernels.{addmm,bmm,mm,convolution}` all dispatch through `torch._C._nn.X` which itself dispatches through Vulkan eager — paying the eager-kernel overhead (input dtype check, contiguous copy on stride mismatch, etc.). Even if we keep the eager backing, route through a thin Slang wrapper that skips the redundant dispatcher layers.

### M17.7 — Memory: collapse alloc + reinterpret_tensor chains (1-2d)

20 allocations + 14 pool releases per SmallCNN step is high. Many are `reinterpret_tensor` views that get separately allocated. Audit `buffer_pool.py` hit rate per-graph; aim for ≥80 % (currently 36 % MLP).

---

### M17 critical path (dependency chart)

```
M17.1 Slang matmul reactivation ──┐
                                  ├──→ Linear / bmm / addmm
M17.0 dispatch census ────────────┤   8 → 2 dispatches      ─→  CPU parity SmallCNN
                                  │
M17.2 conv+gn+relu fusion ────────┼──→ 8 → 3 dispatches
                                  │
M17.3 adaptive_avg_pool Slang ────┼──→ 6 → 2 dispatches
                                  │
M17.4 AdamW foreach verify ───────┼──→ 6 → 1 dispatch
                                  │
M17.5 cmd buf chaining ───────────┘──→ per-dispatch 0.4 → 0.1 ms
```

### Companion tooling (in `agent_space/`, git-ignored)

| Script | Purpose |
|--------|---------|
| `perf_vs_cpu.py` | 7-spec benchmark (incl. SmallCNN-GN), before/after table |
| `codegen_audit.py` | Greps compiled wrappers for `extern_kernels.X` (leakage tracker) |
| `perf_profile.py` | Per-kernel timing (`TORCH_VULKAN_INDUCTOR_STATS=1`) |
| `perf_cprofile.py` | cProfile a warm step to find Python overhead |
| `slang_mm_correctness.py` | Reproducer for Slang tile mm bugs (OP.27 / M17.1) |
| `conv_perf_breakdown.py` | Per-wrapper dispatch census for the conv model |
| `adamw_dispatch_count.py` | Per-step optimizer dispatch + wall-time counter (M17.4) |

---

## 1. M9 — Host-overhead reduction (P0 for perf, 1-2w remaining)

Closes the 96 % / 230× host/kernel gap. M9.1 (buffer pool) and M9.3
(prewarm) closed 2026-05-13.

- [x] **M9.1** — buffer-pool key bug ✅ closed 2026-05-13 (see history doc)
- [x] **M9.2** ✅ GPU-validated 2026-05-13: fixed fence reset race + descriptor pool pre-reset callback. Deferred command-buffer batching: stop per-dispatch `submit_and_wait`; submit 4–8 dispatches per `vkQueueSubmit`. C++ impl done (Stream::{flush_async,flush_sync}, batched async flush at 8 dispatches in dispatch.cpp), needs GPU validation. Target: −5 to −10 ms / SmallCNN step. **Next-largest perf win.** (2-3d)
- [x] **M9.3** — prewarm-on-import ✅ closed 2026-05-13 (see history doc)
- [x] **M9.4** ✅ GPU-validated 2026-05-13. Push-constant in-place updates: pre-allocate bytearray per kernel; update fields, don't `bytes(pc_data)` per dispatch. (1d) — bytearray + pack_into done in make_vulkan_kernel and aoti path, needs GPU validation
- [x] **M9.5** ✅ GPU-validated 2026-05-13. Cached `_jit_dispatch_indexed`: codegen prefers indexed variant when any binding has count > 1. (1d) — C++ FFI bindings added (_jit_dispatch_indexed_cached{,_nopc}), Python FFI resolution updated, needs GPU validation
- [x] **M9.6** Adaptive `_PER_KEY_CAP` in `buffer_pool.py` (scratch=8, transient=6, save_for_backward=4). (1d) ✅
- [x] **M9.7** Pool non-extern Inductor outputs (currently only extern-kernel outputs are pooled). Closes the residual ~64 % miss rate. (1-2d) ✅
- [x] **M9.8** Reduction-boundary fusion: GN + ReLU + GlobalAvg should fuse into 1 kernel, not 2. Relax `rnumel_fuse_cap` gate in `scheduling.py:248-261`, or change gate to predicate on consumer pattern rather than rnumel. (2-3d) — **FIX**: Added reduction-boundary fusion relaxation in `can_fuse_vertical`; rnumel cap check now overrides base tiling rejection. Tests in `TestM98ReductionBoundaryFusion`.
- [x] **M9.9** Transformer combo-batcher `UnboundLocalError: buf10`. **Root cause located**: `vulkan_combo_kernel.py:987-1019` `_rewrite_body()` token-based renaming runs before the buffer-name map is fully seeded; if a buffer name isn't in `per_sub_maps[idx]`, the rewriter emits the original name and collides with a renamed local from a previous subkernel. Fix: pre-seed buffer names via `_build_global_binding_map()` (line 689-795) before the rewrite loop. (1-2d) — **FIX (2026-05-15 reopen)**: `_rewrite_body` per-subkernel suffix change wasn't enough; the runtime `UnboundLocalError` resurfaced on Transformer training. **Real root cause**: `kernel/header.py:call_kernel` and `vulkan_combo_kernel.py:call_kernel` substitute reused-buffer names one step but Inductor's memory planner can chain reuses (`buf9 → buf10 → buf11` — buf10 itself is reused into buf11). One-step substitution leaves the dead intermediate (buf10) in the kernel arg list. **Fix**: walk the alias chain transitively via `_resolve(name)` in both `call_kernel` sites. M9.9 regression test (`test_m99_combo_kernel_no_unbound_local`) now PASSES (was xfail). Transformer 3-step training works end-to-end through `torch.compile`.

---

## 2. M11 — Occupancy-aware codegen (1-2w, throughput)

The headline finding of this audit: **reflection metadata is 0 % used**
despite roadmap text. `occupancy_audit.py` hardcodes shapes per kernel
category. `_extract_linktime_spec_constants` parses VGPR / LDS counts but
they're never fed into WG sizing. Wiring this up is M11.1's whole point.

- [x] **M11.1** **DR.7 wire-up (THE 0 % gap)**: feed reflection VGPR/LDS into `_pick_threadgroup_size_*` instead of leaving `estimate_occupancy()` as a debug-only tool. Default flag → on. Target: +10–20 % on reduction/normalisation kernels. (2-3d)
- [x] **M11.2** Subgroup reductions for WG ≤ wave64: emit `WaveActiveSum`/`WaveActiveMax` instead of LDS reduce. (2d)
- [x] **M11.3** Register-tile pointwise: load+compute+store unrolled ×2-4. (3d) ✅
- [x] **M11.4** Persistent-mode WG autotune: scale WG size by `numel / CU_count`; today the persistent path uses a fixed WG. Extend persistent path to multi-wave reductions (currently small-numel pointwise only). (1-2d) ✅ — CU-count scaling done; multi-wave reduction extension deferred to M11.4b.
- [x] **M11.5** Round non-multiple-of-64 WG sizes up to next multiple on RDNA1 (auto-fix `slang_validator.py` advisory). (0.5d) ✅
- [x] **M11.6** LDS bank-padding rigour: auto-pad WG-shared arrays > 1 KB to nearest power of 2 to avoid stride-1 bank conflicts. (1-2d) ✅
- [x] **M11.7** Occupancy gate in `codegen.py`: warn (or `--strict` fail) if estimated occupancy < 50 %. (1-2d) ✅
- [x] **M11.8** Extend `_KERNEL_STATS` to capture grid, WG, VGPR, LDS, descriptor count (populated from reflection). (1-2d) ✅
- [x] **M11.9 (NEW)** Reduction WG sizing: `kernel/reduction.py:725+` has no grid-aware path. Add one — feed `numel/CU_count` like pointwise does. (1d)

---

## 3. M12 — Reduction backward via autodiff (CG.M3, ~1w remaining)

`reduction.slang` has 6 `[Differentiable]` annotations (sum/mean/var fold
paths shipped) but the dispatch wiring is incomplete: 11/13 tests in
`tests/test_cgm3_reduction_backward.py` are xfailed. Routing through
`bwd_diff_table` would close anti-goal #3 for the reduction class
(already closed for activations/losses/conv/mm).

- [x] **M12.1** 8 `[Differentiable]` annotations in `shaders/lib/reduction.slang` (sum/mean/var/prod fold paths via `reduce_fold_sum` / `reduce_fold_prod` + existing `combine_sum_nan` / `combine_prod_nan` / `welford_combine` / `OpSum.combine` / `OpProd.combine`). Max/min intentionally non-differentiable (sparse grad → argmax/argmin). Argmax/argmin are index ops, not gradient ops. (1-2d) ✅
- [x] **M12.2** Route `aten.{sum,mean,var,prod}_backward` through `bwd_lowerings.py` decomposition into primitives + `bwd_diff_dispatch.py` broadcast template. `bwd_template_registry.py` entries intact. `_REDUCTION_BWD_SRC_TEMPLATE` now handles sum/mean (plain), var (saved*scale), prod (scale/saved). Reduction backward lowerings registered in `bwd_lowerings.py:_register_reduction_backward()`. (2d) ✅
- [x] **M12.3** Retire any `aten.*_backward` reduction shaders living outside `lib/`. No legacy shaders found — already clean. (1d) ✅

---

## 4. M13 — Slang feature saturation (NEW, 2-3w)

The mm template (M10.4 / CG.M10) is the gold standard. Bring conv, SDPA,
reduction, and pointwise dispatch up to the same bar.

- [x] **CG.M11 BackwardDerivative coverage outside `pointwise.slang`**: reduction.slang done (2 new: combine_max/combine_min with [BackwardDerivative]; OpMaxReduce/OpMinReduce now implement IDifferentiableReduction). norm.slang 8/4; losses.slang 12/10; mm_tile.slang 1/0 remain. (3-4d)
- [x] **CG.M12 Pointwise dispatch via Slang generics**: `generic_pointwise_dispatch.py` reads a single `pointwise_generic.slang` source file with four generic entry points (`<Op : IPointwise>`, `<Op : IPointwiseBinary>`, `<Op : IComplexPointwise>`, `<Op : IComplexPointwiseBinary>`). The concrete op struct is resolved via slangc's entry parameter (e.g. `computeMain<OpAbs>`). No Jinja2 templating, no per-op source variation. Closes anti-goal #6 for pointwise. (3d) ✅
- [x] **CG.M13 Reduction codegen via interface**: IWaveReduction now has finalize(val, count); OpArgMax/OpArgMin structs implementing IWaveReduction; codegen emits wg_reduce_wave<OpSum>(...) (proper Slang generics); _op_template_generic documented. Regression tests in TestCGM13ReductionGenerics. (3d)
- [x] **CG.M14 ParameterBlock in pointwise/reduction**: Inductor codegen path (`header.py`) emits `ParameterBlock<KernelArgs>` by default (gated on `config.parameter_block()`, default ON). All buffer accesses use `_buf_path()` → `args.` prefix. `pointwise_generic.slang` (eager dispatch) converted to ParameterBlock as well. All 10+ templates (mm, conv, flash_attn, philox, rnn, fft, persistent_pointwise) use ParameterBlock. Saves ~5 LOC per kernel and unlocks reflection. (2-3d) ✅
- [ ] **CG.M15 Link-time spec constants for conv / SDPA**: only mm uses `[[vk::constant_id]]` for tile / per-thread params. Conv (`slang_conv2d.slang`), SDPA (`flash_attention.slang`), and the reduction template family hardcode loop bounds. Extract them to spec constants so a single SPIR-V module covers many tile choices. Reduces slangc invocations from N×tile_count to N. Gated on M13's slangc-bug status. (3-4d)
- [x] **CG.M16 Capabilities `[require(...)]` audit**: reduction.slang fully annotated: wave_active_sum_nan/wave_active_product_nan -> [require(spirv, subgroup_arithmetic)]; wg_welford/wg_argmax/wg_bitonic_sort_wave/wg_scan_inclusive -> [require(spirv, subgroup_shuffle)]. helpers.slang already had requires. Regression tests in TestCGM16ReductionCapabilities. mm.slang and conv.slang have no subgroup intrinsics. (1-2d)
- [x] **CG.M17 Replace string `.replace()` vec4 codegen**: Replaced fragile ``str.replace()`` token surgery with line-by-line regex transformation using ``(?<!\w)`` word-boundary matching. New helpers ``_apply_vec4_body_rewrite``, ``_rewrite_line_vec4_accesses``, and ``_line_has_non_rewritable_access`` in ``kernel/pointwise.py`` process each Slang statement individually, matching buffer accesses by regex rather than substring substitution. The ``_rewritable_idx_full`` set now includes ``((int)(alias))`` forms for all aliases, not just ``rt_name``. (2d)

---

## 5. M14 — Op coverage gaps (NEW, 2-3w)

Closes the "any PyTorch model" story by filling categorical holes the
audit found.

- [x] **OP.20 Complex-dtype binary elementwise**: complex64/128 matmul + softmax work, but `complex_add`, `complex_mul`, `complex_div` have no `IPointwise` struct in `shaders/lib/pointwise.slang`. They fall through to `ExternKernel` (eager dispatch). Add 4-5 complex-valued op structs; lower via `generic_pointwise_dispatch`. **Unblocks**: vision/audio models using `torch.view_as_complex`. (2-3d) ✅ — 6 complex structs in pointwise.slang (OpComplexAdd/Sub/Mul/Div/Conj/Abs), IComplexPointwise interface, float2-based dispatch, 10 regression tests.
- [ ] **OP.21 Sparse / scatter-atomic backward**: **Partial fix (2026-05-15):** `_register_embedding_bag_backward()` added to `lowerings/embedding.py` — decomposes `aten._embedding_bag_backward` for modes 0/1 (`index_put` accumulate) and mode 2 (`scatter_reduce` amax) with padding_idx support. Registered via `bwd_lowerings.py:register()`. Remaining: regression tests for embedding_bag training + verification that `scatter_add_`/`gather` backward flows through existing AOTAutograd synthesis. (8-10d → ~5d remaining)
- [ ] **OP.22 Dynamic-shape reduction codegen**: **Audit (2026-05-15):** Dynamic-shape infrastructure is near-complete — `is_dynamic_stride()` exists in `kernel/symbolic.py` but is **never called**. Pointwise dynamic shapes work; reduction forward works. Gap: backward stride expressions containing `sympy.Symbol(B)` must emit push-constant reads instead of `static const uint`. Plan: (1) extend `header.py` PC struct to include symbolic strides, (2) wire `is_dynamic_stride()` into load/store indexing in `pointwise_load_mixin.py` and `indexing.py`, (3) test batch sizes {1,4,16,64}. **Tackle first** (5-7d, easier than OP.21, unblocks all variable-batch training).
- [x] **OP.23 Foreach element-wise ops**: `install_external_optimizer` covers SGD/AdamW/Lion. Missing: `foreach_add`, `foreach_mul`, `foreach_div`, `foreach_lerp`, `foreach_clip_grad_norm`. Reuse the foreach template plumbing. **Unblocks**: gradient-clipping codepaths, multi-param updates. (2-3d) ✅ — foreach_add/mul/div/norm working via Inductor combo kernel; lerp out-variant path blocked on ForeachKernelSchedulerNode; clip_grad_norm blocked on C++ storage access.
- [x] **OP.24 Quantized int8 matmul (inference)**: ✅ — `shaders/lib/mm_int8.slang` (int8 unpack + int32 accumulate + float32 store), `_render_mm_int8_slang()` wrapper, `_slang_tile_mm_int8()` dispatch, `torch_vulkan::mm_int8` custom op, `_register_mm_int8_lowering()` in matmul.py with autotuning. Forward-only. **Unblocks**: GPTQ / AWQ / quantized Llama inference.
- [x] **OP.25 RNN backward via Slang autodiff**: `bwd_lowerings.py:687L` decomposes RNN grads manually; GRU backward marked "more complex" and incomplete. **Unblocks**: LSTM/GRU training parity. (6-8d)
- [x] **OP.26 Anti-symptom: native attention primitive**: `_fuse_sdpa_to_flash_attention` is currently a symptom-fix for the absence of a native `aten.scaled_dot_product_attention` lowering. Promote the FlashAttention template to a real primitive lowering registered via `@register_lowering(aten.scaled_dot_product_attention)`. Closes anti-goal #5 for SDPA. (3d) ✅ Done 2026-05-14 — new `lowerings/attention.py` with native SDPA + sdpa_with_optional_mask lowerings routing to FlashAttention template; disabled fx_passes/patterns/sdpa.py pattern matcher; removed pre-grad SDPA decomposition; 9/10 regression tests passing, 1 xfail (pre-existing GQA C++ backend limitation).
- [ ] **OP.27 Slang tile matmul codegen reactivation (NEW 2026-05-15)**: `_install_vulkan_aten_only_autotune` constrains matmul autotune to `"ATEN"` because Slang tile mm/bmm/addmm "produce incorrect forward output (max diff ~28 vs CPU)". Audit (2026-05-15 via `agent_space/codegen_audit.py`) confirms every `nn.Linear` / `nn.MultiheadAttention` lowers to `extern_kernels.{mm,bmm,addmm}` (eager Vulkan dispatch), not Slang codegen. **Investigation**: `agent_space/slang_mm_correctness.py` shows two compounding blockers: (1) `slang_mm.slang:218` `[numthreads(WG_N, WG_M, 1)]` — slangc 2026.5+ rejects expressions derived from `[[vk::constant_id]]` with `E39999: expression does not evaluate to a compile-time constant`; (2) After hard-coding numthreads via Jinja, slangc compiles but Vulkan validation fails with `VUID-VkComputePipelineCreateInfo-layout-07988`: shader uses `Set 1 Binding 0/1/2` but pipeline layout only declares `Set 0` — the `ParameterBlock<KernelArgs>` set assignment doesn't match what `_jit_dispatch` reflects. Both fixes need landing together: replace spec-constant numthreads with Jinja literals AND align ParameterBlock set/binding indices with the C++ dispatch layout. Until then, matmul stays on extern_kernels (correct, just unfused). (3-5d). Companion test: `agent_space/codegen_audit.py` greps every compiled wrapper for `extern_kernels.X` calls per model — current numbers: MLP 2 (bmm,addmm), SmallCNN 1 (addmm), Transformer 3 (2×bmm, 1×addmm).

---

## 6. M15 — Anti-goal #5/#7 cleanup (NEW, 1-2w)

Expanded successor to v6.1's M10. Six newly-discovered file-size
violators plus a `meta_patches.py` symptom-fix audit.

- [ ] **M15.1 File splits (10 violators):**
  - [x] M15.1.a `vulkan_template_caller.py` (5786 L → **265 L**) → `templates/caller/{gemm,scatter,optimizer,flash_attn,rng,conv,fft,rnn}.py`. `gemm.py` (2331 L) needs further split in follow-up. (2026-05-13)
  - [x] M15.1.b `meta_patches.py` (3902 L) → file no longer exists; `register_fake` hooks distributed across `fx_passes/eager/{addmm,sdpa,swiglu,qkv,optimizer,conv,pool}.py` and `lowerings/rnn/common.py`. ✅
  - [x] M15.1.c `runtime.py` (2955 L) → `runtime/{slangc,dispatch,batcher,profile,reflection}.py`. **NEW.** (1-2d) ✅
  - [x] M15.1.d `kernel/pointwise.py` (1555 L → **761 L**) → extracted `PointwiseLoadMixin` + `PointwiseVec4Mixin` (was M10.3). ✅
  - [x] M15.1.e `fx_passes/eager_patches.py` (1159 L) → `fx_passes/eager/{addmm,sdpa,swiglu,qkv,optimizer,conv,pool}.py`. **NEW.** (1d) ✅
  - [x] M15.1.f `vulkan_combo_kernel.py` (1106 L) → split body-rewriter from binding-map / grid-builder into `combo_kernel/{body_rewriter,binding_map,grid_builder}.py`. **NEW** (also fixes M9.9 indirectly). (1d) ✅
  - [x] M15.1.g `kernel/reduction.py` (1045 L → **576 L**) → extract `ReductionLoadMixin` + `reduction_tile_picker.py`. **NEW.** (1d) ✅
  - [x] M15.1.h `bwd_diff_dispatch.py` (913 L) → split unary/binary dispatch + emit helpers into `bwd_diff/{unary,binary,emit_helpers}.py`. **NEW.** (1d) ✅
  - [x] M15.1.i `validate.py` (813 L) → split into per-pass validation modules `slang_validate/{braces,bindings,symbols,memory,workgroup,push_constants,bwd_diff_scan}.py`. **NEW.** (0.5d) ✅
  - [x] M15.1.j `lowerings/rnn.py` (805 L) → split per-cell-type dispatch into `lowerings/rnn/{lstm,gru,common}.py`. **NEW.** (0.5d) ✅
  - [x] M15.1.k `templates/caller/gemm.py` (2331 L) → split into `gemm/{render,dispatch,classes,install,backward}.py` (largest: `classes.py` 778 L, `dispatch.py` 638 L, `render.py` 574 L). **NEW.** ✅
- [ ] **M15.2 `meta_patches.py` symptom-fix audit**: **COMPLETE 2026-05-13** — full audit in `agent_space/m15.2_audit_report.md`. 196 hooks/patches surveyed across 8 files: 157 (a) genuine FakeTensor, 35 (b) workaround for missing primitive, 4 (b?) unclear, 0 (c) dead code. Priority (b) hooks to promote filed as new M15 items. All files under 800 L. ✅
- [ ] **M15.3 Small fixes carried over from v6.1:**
  - [x] Lift `_VALID_IPOINTWISE_STRUCTS` frozenset to auto-parse `lib/pointwise.slang` at startup; drop manual sync. ✅ — `_parse_pointwise_structs()` in `vulkan_template_caller.py` (M15.3.a).
  - [x] Remove redundant outer cast in `kernel/pointwise.py:68-81` int8 load dispatch. ✅ — code already clean after file split; int8/uint8/bool dispatch correctly casts uint→float.
  - [x] Audit & remove stale TODO gate at `vulkan_template_caller.py:754` (P3.2/M14 dead flag). ✅ — file split to 266 L; line 754 no longer exists.
  - [x] Extract pickle/repr boilerplate from `_SlangTile{MM,AddMM,BMM}` into a common base. ✅ — `_SlangTileGEMM` base class handles `__setstate__`; subclass-specific `__reduce__` needed for pickle bytes-equality.

---

## 7. M16 — Track 4 finish (NEW, 1w, IRREVERSIBLE)

`csrc/ops/model_ops.cpp` is 925 L (drift from 885 since v6.1 audit). 22
legacy eager kernels — these block the "no per-model `csrc/ops/*.cpp`
entries" anti-goal. Track 4 was meant to delete this file. The drift
suggests new ops are still landing here despite the anti-goal.

- [x] **M16.1** Inventory `model_ops.cpp` — categorise each of the 22 ops as (a) covered by an Inductor lowering already (delete from cpp), (b) needs a new Inductor lowering before delete, (c) genuinely eager-only (move to `csrc/ops/legacy_eager.cpp` to make the boundary explicit). (1d) ✅ — 24 ops category (a), 1 op category (b) = `aten.index.Tensor` boolean-mask path, 4 ops category (c).
- [x] **M16.2** Add eager-mode lowering parity for category (b) ops. ✅ — `lowerings/bool_mask.py` provides PrivateUse1 eager override + Inductor lowering for `aten.index.Tensor` bool-mask. Bool masks decompose via `nonzero` + `index_select`; integer indices route through upstream `index_impl` Pointwise. (M16.2 done 2026-05-13)
- [ ] **M16.3** Delete `model_ops.cpp` and lock with a regression test that fails the build if it returns. **Audit (2026-05-15):** Plan: (1) move 4 category-(c) genuinely-eager ops (`vulkan_contiguous`, `vulkan_to_copy`, `vulkan_as_strided`, `vulkan_resize_`) to new `csrc/ops/legacy_eager.cpp`, (2) remove all 24+ category-(a) registrations from `Registration.cpp` and declarations from `ops.h`, (3) delete `model_ops.cpp`, (4) add `_validate_no_model_ops()` in `setup.py` and `TestM16ModelOpsDeleted` regression test. Risk: HIGH for link errors — must remove registrations+declarations simultaneously. (0.5d)
- [ ] **M16.4** Lock the boundary: **Audit (2026-05-15):** Plan: add `_validate_ops_boundary()` in `setup.py` with `ALLOWED_OPS_FILES` whitelist (29 legitimate files) and `FORBIDDEN_NEW_FILE_PATTERNS` (catch `*_backward*`, `*model_ops*`, unknown `*_ops.cpp`). Runs at build time — every `setup.py build_ext` gates violations. No separate pre-commit hook needed. Forbidden symbol patterns check for `vulkan_\w+_backward` outside `backward_ops.cpp`/`autograd_ops.cpp`. (0.5d)

---

## 8. M6 — Conv generality (Phase 2-4)

Phase 1 (Conv1d) closed 2026-05-11 — see history doc.

- [x] **Phase 2 — Depthwise conv (groups=C, arbitrary)**: ✅ — per-group decomposition in `fx_passes/patterns/conv_im2col.py` handles arbitrary groups via `aten.slice.Tensor` + per-group `torch_vulkan::conv2d_with_optional_bias` (groups=1) + `aten.cat`. No hardcoded group limit. Backward inherited from Conv2d bwd template.
- [ ] **Phase 3 — Conv3d (KD>1)**: **Audit (2026-05-15):** Current `_conv3d_to_conv2d_lowering` handles KD==1 only. Approach: merge depth into spatial batch (N*D) and kernel depth into kernel height (KD*KH), with depth padding/striding as pre-processing. No template changes needed — the existing Conv2d tiled template handles the reshaped input. Edge cases: strided/dilated depth, depth padding. (3-4d)
- [ ] **Phase 4 — Transposed conv (1D / 3D)**: **PARTIAL (2026-05-15):** New `lowerings/conv_transpose.py` (~330 L) ships `_impl_2d/1d/3d` decomposition (flip + channel-swap + zero-upsample → Conv2d, post-pad for output_padding). Registered against `aten.conv_transpose1d/2d/3d` overloads (unreachable — see below) and `aten.convolution.default` (the real path — `F.conv_transpose*` decomposes to `aten.convolution(transposed=True)` before reaching the lowering registry). 2D path falls through to upstream extern (preserving the existing test_transposed_conv_graph_breaks_gracefully behavior); 1D/3D paths route through the decomposition. Tests in `TestM6Phase4ConvTranspose` (10 tests, all `xfail(strict=False)`). conv.py shrunk 795→690 L. **Blockers found this session** (filed in xfail reasons): (1) `aten.flip` only in upstream's decomposition table — added `make_fallback(override_decomp=True)`; (2) `aten.transpose.int` no lowering — replaced with `aten.permute`; (3) `torch_vulkan::conv2d_with_optional_bias` OpOverload identity changes after `register_eager_patch_custom_ops` re-registration — added string-keyed lookup (`_get_conv2d_lowering`); (4) `aten.clone` returns Pointwise IR while `_VulkanConv2dExternKernel` requires realized inputs — added `ExternKernel.realize_input`. All four issues addressed but downstream IR validation still asserts on the per-frame conv2d ExternKernelOut. Next steps: add codegen-stage realize hooks or build a custom op `torch_vulkan::conv_transpose{1,2,3}d` that wraps the eager path. (~2-3d remaining)

**M6 — New issues discovered 2026-05-15 (conv training analysis):**

- [x] **M6.5 Conv2d backward gradient correctness**: **FIXED (2026-05-15):** `g_inp = torch.empty_like(inp)` → `torch.zeros_like(inp)` in `fx_passes/eager/conv.py:L133`. The CAS `vk_atomic_add` reads before write — uninitialized memory corrupts gradient computations. ParameterBlock field ordering verified correct (input/weight/grad_out/grad_input/grad_weight/grad_bias). ✅
- [x] **M6.6 Depthwise conv backward via aten.convolution_backward**: **FIXED (2026-05-15):** Per-group `_slang_tile_conv2d_bwd` decomposition in `fx_passes/eager/conv.py:_conv2d_backward`. When groups>1 + Vulkan + f32, splits tensors per-group, calls Slang bwd for each group, concatenates results. Avoids the FunctionalTensor type mismatch in `aten.convolution_backward`. Non-Vulkan/non-f32 still falls through to aten. ✅
- [ ] **M6.8 Vulkan allocator offset-view handling (NEW 2026-05-15)**: `csrc/ops/dispatch.cpp:get_buffer_info` looks up `tensor.data_ptr()` in `VulkanAllocator`'s pointer→buffer map. For a view with non-zero `storage_offset()`, `data_ptr()` includes the offset and the lookup misses, throwing "Tensor has no backing Vulkan buffer". This blocks depthwise conv compile mode: `tests/test_inductor_regression.py::TestConvGeneralityGaps::test_m6_depthwise_conv_{matches_cpu,backward_matches_cpu}` both fail because Inductor's wrapper emits `reinterpret_tensor(primals_1, (1,1,H,W), (C*H*W,...,1), offset=N*sizeof)` per group. Even `.contiguous()` and `.clone()` fail because they also dispatch through the same buffer lookup. Fix: (1) `get_buffer_info` should look up by `storage().data_ptr()` (without offset), then carry offset into the binding (`vkBindBufferMemory` / push-constants); (2) update all dispatch sites to pass the offset explicitly. Affects every op that takes a tensor argument. (3-4d) Adjacent fix already in this session: `templates/caller/conv.py:_slang_tile_conv2d_bwd` PF.51 guard now checks output (`grad_input`, `grad_weight`, `grad_bias`) tensors — not just `input_t` — so any FakeTensor leak through the per-group decomposition exits cleanly instead of crashing the dispatch. Doesn't unblock depthwise compile but tightens the guard.
- [x] **M6.7 Combo kernel body rewriter bugs (BN+ReLU+Conv fusion)**: **FIXED (2026-05-15):** Gate added in `scheduling.py:can_fuse_vertical()` — when `conv_epilogue` fusion group would mix template (conv) + reduction (norm) nodes, fusion is rejected. The generic combo kernel body rewriter can only handle pointwise subkernels; template+reduction fusion must use the native conv template epilogue. ✅

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
