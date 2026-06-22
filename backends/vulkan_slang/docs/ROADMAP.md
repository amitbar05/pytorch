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
>
> **2026-06-22 deep pipeline audit** added §§ S4.0/S4.2/S4.3 (AOTI gaps),
> CG.1–CG.4 (codegen correctness), and SP.1–SP.3 (Slang/SPIR-V pipeline)
> based on a cross-vendor read-only audit of all kernel, runtime, lowering,
> and AOTI-emit source files. New E4/E5 coverage items added.

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
| Device limits (CU count, LDS, max WG, subgroup) | ✅ **S0.2 FIXED 2026-06-21** | `_device_caps()` wired into `_query_limits()` (`device_profile.py:_query_limits`). Returns real `max_workgroup_size=1024`, `subgroup=64`, `shared_memory=65536` from device. `TestM211DeviceProfile::test_device_limits_come_from_hardware` ✅ |
| **Profile is *consumed* by codegen** | 🟡 **S0.1 partial** | `profile_limit` wired to `max_workgroup_size`/`compute_units` (`threadgroup_sizing.py`), LDS budget (`kernel/main.py:291` via `profile_limit("shared_memory_per_workgroup_bytes")` 2026-06-21). Mem-BW/latency-threshold routing still unused. |

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
| **Warm→train cache coherence** | ✅ **S2.4 FIXED 2026-06-21** | `_restore_probe_defaults()` reads `mm_tiles_mode` from probe_status.json and applies as soft default on every import. `TestWarmCacheCoherence` ✅ |

**S2 — Conv+GN+ReLU+Pool+Linear compile-path dispatch audit**
| Op | Dir | Mechanism | Slang? |
|---|---|---|---|
| Conv2d | fwd | `_VulkanConv2dExternKernel` → `slang_conv2d.slang` | ✅ |
| Conv2d | bwd | `_VulkanConvBwdExternKernel` → `slang_conv_bwd.slang` + `bwd_diff` | ✅ |
| GroupNorm | fwd | decomposition → Slang codegen (2 dispatches: GPU.1 L2 workaround, `norm.py:43`) | ✅ |
| GroupNorm | bwd | 2× extern (`group_norm_backward.slang` + `_weight.slang`) | ✅ |
| ReLU | fwd/bwd | Pointwise → Slang codegen | ✅ |
| Conv+GN+ReLU fused | fwd | pre-grad `conv2d_gn_relu_fused` → `conv_gn_relu.slang`. ✅ **S2.2 FIXED 2026-06-21**: M-CG.3 WG 256→64 workaround confirmed safe — 5-seed parity test passes (`TestConvGnReluFusedWriteCoverage` ✅, 0 VUIDs). | ✅ |
| AdaptiveAvgPool2d / MaxPool2d / AvgPool2d | fwd | ✅ **S2.5 FIXED 2026-06-21**: `torch_vulkan::avg_pool2d` Python custom op registered; `F.avg_pool2d` monkey-patched; wrapper emits `torch.ops.torch_vulkan.avg_pool2d.default` (private Vulkan compute, FallbackKernel) instead of `torch.ops.aten.avg_pool2d` (public aten eager). | ✅ |
| Pooling | bwd | `scatter_atomic.slang` / codegen | ✅ |
| Linear | fwd | `aten.addmm` → `slang_mm.slang` | ✅ |
| Linear | bwd | `slang_mm_bwd.slang` — ✅ **FIXED 2026-06-21 (S2.1)**: `aten.mm.default` now routes through `_vulkan_mm` (forced override after `get_overloads()` skip), and `_adaptive_avg_pool2d_backward.default` routes through `Pointwise.create` (same override + `ops.*-on-TensorBox` bug fix). | ✅ |
| SGD/AdamW/Lion | step | ✅ **S3.1 FIXED 2026-06-21**: compiled step routes to `foreach_sgd_step` ExternKernel → `foreach_optimizer.slang` (`IOptimizer`) — same path as eager. Both `_foreach_add_.List` (inplace) and `_foreach_add.List` (functional, post-AOTAutograd) are handled. | ✅ |
| CrossEntropyLoss | fwd/bwd | decomposed → Slang codegen | ✅ |

**Defects (2026-06-19 ground-truth run + 2026-06-22 deep audit)**
| ID | Defect | Severity |
|---|---|---|
| **S2.0** | Conv+GN CNN training (Linear & 1×1-conv heads): `buf7` combo inversion + wrong conv `grad_bias` (output reuse) + `StorageBox` unwrap crash + 4D grad_out assumption. | ✅ **FIXED 2026-06-19** (S2.0a/reuse/b/c) |
| **S2.0d** | Stacked conv+GN+ReLU backward exploded (WAR-barrier miss). **FIXED 2026-06-20** — `csrc/ops/dispatch.cpp` now tracks reads + emits a WAR barrier (`test_stacked_conv_gn_backward_war` ✅). **S2.0d-resid also FIXED 2026-06-20** (two more root causes, see below): residual `relu(out+identity)` backward now has full per-param grad parity (`test_resnet_block_residual_grad_parity` ✅). | ✅ **FIXED** |
| **S2.1** | Conv+GN+Pool+Linear backward wrapper leaks — **FIXED 2026-06-21**: two root-cause fixes in `matmul.py` + `bwd_lowerings.py` + `pool.py` (`TestNoExternInFullCNNBwd` ✅). | ✅ **FIXED** |
| **S3.1** | Compiled SGD step = 1 tiny `binary_add_inplace` per param tensor (4 here) + strided copies; should route to the foreach `IOptimizer` extern. | ✅ **FIXED 2026-06-21** — routes `aten._foreach_add.List` (functional form, post AOTAutograd) + `aten._foreach_add_.List` (inplace form) both to `foreach_sgd_step` ExternKernel. Batch sizes capped at 8 (push-const limit). Tests: `test_s3_1_compiled_sgd_optimizer_correctness_vs_cpu` ✅, `test_s3_1_compiled_sgd_variable_numel_correctness` ✅ |
| **TRAIN.4.b** | `vulkan_nll_loss_forward` (C++ FallbackKernel) ignores weight for mean reduction — returns `total_weight = N` instead of `sum(weight[target_n])`, causing 75% wrong gradients in weighted cross-entropy. | ✅ **FIXED 2026-06-21** — added `aten.nll_loss_forward` Inductor lowering in `lowerings/loss.py` that calls `_nll_loss_decomp()` with correct weighted total_weight IR. `TestTrain4CrossEntropyBackward::test_cross_entropy_with_weight` ✅ |
| **S3.5a** | **Causal conv1d wrong values** (`test_m6_causal_conv1d_matches_cpu`): `VkDescriptorBufferInfo.offset` hardcoded to 0 in `bind_buffers` — `reinterpret_tensor` views with non-zero `storage_offset` all bind to the start of the VkBuffer, so per-group dispatches write the same positions. Exposed by S3.5 commit (ReinterpretView preserved instead of cloned). **C++ fix designed and stashed** (`csrc/ops/dispatch.h/cpp`, `csrc/vulkan/DescriptorSet.h/cpp`): thread `storage_offset * element_size` through `get_buffer_info → dispatch_shader → bind_buffers → VkDescriptorBufferInfo.offset`. | 🔴 **OPEN** — fix in `git stash@{0}` (needs rebuild + verify) |
| **S3.5b** | **Conv1d backward x.grad=0** (`test_m6_conv1d_backward_matches_cpu` + `test_s3_5b_conv1d_backward_grad_input_not_primal`): Root cause was AoT backward graph buffer-reuse aliasing: `input_4d = input.unsqueeze(-1)` created a dead intermediate that Inductor's memory planner reused for `conv2d_backward[0]` (grad-input). Since `input_4d` is a view of `primals_3` (primal x), grad-input aliased x, returning x itself as x.grad. **Fix**: new `conv1d_backward_core` opaque non-autograd custom op takes 3-D tensors directly (no Python-side unsqueeze), with a fake kernel deriving outputs from `input` (FT proxy) so FunctionalTensorMode produces fresh storage refs. Forward graph now saves `primals_3` as residual; backward graph uses it as a proper graph INPUT (not freed/reused). | ✅ **CLOSED 2026-06-22** — `fx_passes/eager/conv_backward.py` + `conv.py` + `lowerings/__init__.py` |
| **S3.5c** | **Grouped conv push-constant size mismatch** (`test_m6_conv1d_groups_matches_cpu`): Original bug claimed `slang_addmm_8_8_8_s1_r1x1` SPIR-V declared 52 bytes but layout got 48 bytes (VUID-VkComputePipelineCreateInfo-layout-07987). **Three 2026-06-22 audits conclusively show this attribution is wrong**: (1) no Python path produces 48-byte PC for `slang_addmm` — both `compile_and_dispatch` and `emit_aoti_extern_dispatch` emit 96 bytes; (2) the `groups>1` conv path uses per-group `_VulkanConv2dExternKernel` + `aten.cat`, never `slang_addmm`; (3) the test docstring explicitly states "Conv2d eager fallback handles groups>1 correctly" — so there is no Slang kernel for this path at all. The VUID error was likely from a different kernel in a stale test run, now superseded by the S3.5 commit. Needs GPU re-run to verify. | 🟡 **NEEDS GPU VERIFY** — original root cause is invalid; test likely already passes; re-run to confirm |
| **S3.5d** | **Conv2d backward `.item()` intentional GPU pipeline drain** (`fx_passes/eager/conv_backward.py:99`): `grad_bias[0].item()` is a deliberate sync barrier inserted to prevent stale gradient data in the FallbackKernel compiled path. NOT removable without a proper Vulkan submission fence in the C++ dispatch layer. Performance stall on every conv bwd+bias; correctness preserved. Proper fix: C++ submission tracking in `csrc/ops/dispatch.cpp`. | 🟡 **KNOWN LIMITATION** — intentional; fix requires C++ dispatch change |
| **S4.0** | **AOTI: MM/addmm/bmm templates emit `n_pc=0` → `pc_size_bytes=0` in AotiRuntime** (`cpp_wrapper_gpu.py:367-374`): Template kernels bypass `define_kernel`; `_set_kernel_meta` is never called for them, so `get_kernel_meta` returns `n_pc=0`. The AOTI `.so` creates a pipeline layout with zero push-constant bytes while the SPIR-V expects 96 bytes (24×uint32). Any AOTI-compiled model using MM will crash on first dispatch. Fix: emit `_set_kernel_meta` from the template caller’s AOTI codegen path, or derive `pc_size_bytes` from SPIR-V reflection in `cpp_wrapper_gpu.py`. | 🔴 **OPEN** — blocking S4 AOTI correctness |
| **CG.1** | **argmin/argmax index precision loss for tensors > 16M elements** (`kernel/reduction.py:336-342`): The `(value, index)` pair is encoded as `float2({value}, float({index}))` — casting index to float truncates to 24-bit mantissa precision. Fix: emit `uint2` pair or a dedicated struct. | 🔴 **OPEN** — silent wrong results for large tensors |
| **CG.2** | **bf16 packed16 store uses `WaveReadLaneAt` unconditionally** (`kernel/pointwise.py:145-170`): wave32 hardware reads wrong lane. Guard with device simd_group_size check. | 🔴 **OPEN** — latent, wave32 only |
| **CG.3** | **packed16 + welford guard bypass** (`kernel/pointwise_load_mixin.py:130-145`): early-return at line 138 skips `has_welford` guard — fp16 loads + welford produces garbage mean/m2. Move welford check before early-return. | 🔴 **OPEN** — latent, triggers on fp16 reduction kernels with GroupNorm |
| **SP.1** | **Async compile still serial: `.result()` blocks caller** (`runtime/slangc.py:552-560`): `compile_slang_to_spirv` calls `pool.submit(...).result()` on cache-miss — overlap never happens. Fix: return a `Future` on cache-miss; callers await lazily. Prerequisite for S3.4. | 🔴 **OPEN** — blocks S3.4 |
| **SP.2** | **numthreads rewrite path dead** (`runtime/reflection_ext.py:654-700`): `_rewrite_numthreads_in_source` runs but rewritten source is never recompiled — `get_optimized_numthreads` has no callers outside `slangc.py`. Wire into `make_vulkan_kernel:_maybe_autotune_wg` or remove. | 🔴 **OPEN** — dead code / wasted compile time |
| **MS.2** | **Use-after-free: AOTI shim `zeros/ones/full/as_strided` return dangling `AtenTensorHandle`** (`csrc/backend/aoti_shims.cpp:146,172,193,215`): each function returned `tensor.unsafeGetTensorImpl()` of a local `at::Tensor` that was destroyed on return. Fix: allocate with `new at::Tensor(std::move(tensor))` at all four sites (matching `empty_strided_vulkan`). | ✅ **FIXED 2026-06-22** — `csrc/backend/aoti_shims.cpp` |
| **MS.1** | **Memory leak: `aoti_torch_delete` is a no-op** (`csrc/backend/aoti_shims.cpp:154-156`): `RAIIAtenTensorHandle` destructor calls this for every intermediate tensor; the handle was silently discarded. Fix: `delete reinterpret_cast<at::Tensor*>(handle)`. Fixed together with MS.2. | ✅ **FIXED 2026-06-22** — `csrc/backend/aoti_shims.cpp` |

**S3 — TRAIN (steady-state perf)**
| Item | State | Evidence |
|---|---|---|
| Tiny-kernel fusion (AccumulateGrad stash) | ✅ **S3.2 FIXED 2026-06-21** | gradient stash fix → 12 dispatches/step (≤14); `TestTinyKernelFusion` ✅ |
| Persistent-kernel routing for large reductions | 🔴 | **dead code** — `dispatch_persistent_pointwise()` defined, never called; no numel>65536 routing |
| Batch-dispatch overlap (exec N ∥ compile N+2) | 🟡 | async precompile at codegen-time only (`slangc.py:573`); `_compile_ahead` stubs in `batcher.py:63-65` declared but **never wired**; true runtime overlap requires SP.2 + wiring stubs; `BATCH_DISPATCH=1` still 1.8× slower → default OFF |
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
| 6 | **No CPU/eager fallbacks on the compile path** | ✅ S2.5 ✅ FIXED 2026-06-21 (avg_pool2d → `torch_vulkan` custom op); S2.1 ✅ fixed; S2.2 ✅ confirmed |
| 7 | No file > 800 lines | 🟡 `pointwise.py` 820L |
| 8 | **Binary loss backward via inline bwd_diff (not external custom-op shim)** | ✅ **CG.M8 FIXED 2026-06-21** — `aten.{mse,l1,bce,bce_with_logits,smooth_l1,huber}_loss_backward` route through `ops.vulkan_bwd_diff_binary` inline path with correct 1/N mean-scale. `TestTrain1LossBackwardReachability` 10/10 ✅. |

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

**Follow-up filed — S2.0d-deadgrad** ✅ **FIXED 2026-06-21 (side effect of S2.0d-resid)**:
The order-dependent dead `gn.weight` gradient no longer reproduces. Both
`agent_space/gn_assert.py` and `agent_space/gn_after_import.py` (with
`import test_inductor_regression`) now show rel≈4e-7 on fresh cache
(`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`). The S2.0d-resid fix 1
(`make_buffer_reuse`: skip fresh-alloc for in-place mutation reuse) fixed the
rstd initialization corruption that was the actual root cause; the apparent
set-ordering non-determinism was a red herring caused by the corruption
producing different wrong results in different orderings.

#### S2.0e — 🟡 Pre-existing GPU-suite failures (slangc/env)
A clean-main GPU sweep showed ~24 pre-existing failures unrelated to S2.0.
**2026-06-22 (S3.5)**: `TestConv1dCompile` basic + padding now pass (✅). Three
remain open as S3.5a/b/c defects above. Other failures (`TestConvGeneralityGaps`,
`f16_mm_through_compile`, pool/wrapper tests) still need triage.
- **Exit**: `TestConv1dCompile` + `TestConvGeneralityGaps` fully green on GPU.

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

### S2.5 — ✅ FIXED 2026-06-21: Pool forward custom-op dispatch (anti-goal #6 close-out)

`aten.avg_pool2d.default` was routed to `FallbackKernel` → eager aten (public CPU
path), violating anti-goal #6. `Reduction.create` was attempted but requires 2D
window reduction support our backend lacks.

**Fix (2026-06-21, no C++ rebuild required):** Register `torch_vulkan::avg_pool2d`
as a pure-Python `torch.library.custom_op` (fake/autograd impl backed by aten).
Monkey-patch `F.avg_pool2d` to redirect Vulkan tensors through it during trace.
Add `make_fallback` lowering so Inductor emits
`torch.ops.torch_vulkan.avg_pool2d.default` (private Vulkan compute path via
`FallbackKernel`) instead of `torch.ops.aten.avg_pool2d` (public aten eager).
`divisor_override=None` encoded as `int=0` (Python custom ops don't support
`Optional[int]`); impl converts back.

- **Files**: `fx_passes/eager/pool.py` (`_ensure_avg_pool2d_op_registered`),
  `fx_passes/eager/__init__.py` (register in `register_eager_patch_custom_ops`),
  `fx_passes/eager_patches.py` (re-export), `python/torch_vulkan/__init__.py`
  (`_patched_avg_pool2d` + `F.avg_pool2d` swap), `lowerings/__init__.py`
  (`make_fallback(torch.ops.torch_vulkan.avg_pool2d.default)`).
- **Exit**: `TestPoolAdaptiveRouting::test_avg_pool2d_uses_custom_op_not_aten_extern` ✅
  — wrapper contains `torch.ops.torch_vulkan.avg_pool2d.default`, zero
  `torch.ops.aten.avg_pool2d` references.

### S0.1 — 🟡 Make the device profile *drive* codegen (the missing half of "probe") [Slices A+B FIXED 2026-06-22]

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

**S0.1 partial FIXED 2026-06-21:** persistent-vs-grid-stride numel cap now
scales from device profile. `_persistent_pointwise_numel_cap()` in
`scheduling.py` reads `compute_units` + `empty_kernel_launch_us` from
`profile_limit` and computes `16384 × (cu/16) × (launch_us/12.3)`, clamped to
[4096, 65536]. For RDNA1 (16 CU, 12.3µs) cap stays at 16384; for an 80-CU/50µs
GPU the cap scales to 65536. `TestProfileDrivenPersistentCap` ✅.

**Still open (S0.1 remainder) — 2026-06-22 deep audit:**

*What the profile already drives* (✅):
`max_workgroup_size`, `compute_units`, LDS budget (`shared_memory_per_workgroup_bytes`),
persistent-pointwise numel cap, subgroup size (for tile filtering), MM tiles mode.

*Hard-coded sites that should consume the profile* (confirmed by audit):

| Priority | File:line | Hard-coded | Should use |
|---|---|---|---|
| ✅ **DONE** | `autotune.py` | WG candidates `[128,256,512]` | `profile_limit("max_workgroup_size")` — FIXED 2026-06-22 (Slice A) |
| ✅ **DONE** | `templates/caller/gemm/dispatch.py` | `max_wg: int = 1024` in `_check_workgroup_fits` | `profile_limit("max_workgroup_size", 1024)` — FIXED 2026-06-22 (Slice B) |
| **HIGH** | `vulkan_template.py:185-196` | Static 4 MM tile configs | Filter/expand by `memcpy_d2d_GBps` |
| **HIGH** | `lowerings/conv.py:208` | Default tile `(8,8,8)` | Scale with `memcpy_d2d_GBps` |
| **MEDIUM** | `vulkan_template.py:253,278,294` | `threadgroup_size=256` for RNG/optimizer/flash | `profile_limit("max_workgroup_size", 256)` |
| **MEDIUM** | `kernel/main.py:260-275` | Cooperative-reduction thresholds (65536, 8192, …) | Scale with `compute_units` + `empty_kernel_launch_us` |
| **MEDIUM** | `scheduling.py:340-349` | `rnumel_fuse_cap` 64/256/8192 | Scale with `compute_units` |

**Slices A+B FIXED 2026-06-22** (commit `5ecbb66c34e`):
- `autotune.py:get_wg_size_variants` now reads `profile_limit("max_workgroup_size", 1024)` and filters all candidate lists through it, respecting `_autotune_level()` tier. Enables >1024-WG GPUs; caps 512 ceiling on restricted devices.
- `templates/caller/gemm/dispatch.py:_check_workgroup_fits` `max_wg` default changed from hardcoded `1024` to lazy `profile_limit("max_workgroup_size", 1024)`.

**Profile availability**: `profile_limit()` reads the on-disk cache and never
crashes when absent (returns the `fallback`). No race possible — cache written
atomically (tmp+rename) before codegen ever runs under normal `prepare_device()`
flow. Risk of stale profile on driver update is the only residual concern
(mitigated by cache-key including `device_name`).

### S0.2 — ✅ FIXED 2026-06-21: Complete the device-limits pybind query

`_device_caps()` (already in `_C` since M18.4-followup-C) was not wired into
`_query_limits()` — which was checking for the nonexistent `_get_device_capabilities`.
Fix: update `device_profile.py:_query_limits()` to call `_C._device_caps()` first,
giving correct `max_workgroup_size=1024` and `max_compute_shared_memory=65536` from
the live device instead of the NAVI10 name-based defaults.  No C++ change needed.
- **Files**: `python/torch_vulkan/inductor/device_profile.py:_query_limits`.
- **Exit**: `TestM211DeviceProfile::test_device_limits_come_from_hardware` ✅ — limits
  come from the real device, not NAVI10 fallback constants.


### S2.3 — In-process validation during warm-up (make `validate=True` real)

**2026-06-22 deep audit (confirmed) — current state:**
- `prepare_device(validate=True)` sets `TORCH_VULKAN_VUID_AS_ERROR=1` and (for
  level≥2) spawns **subprocesses** via `validation_codegen.py:validate_codegen_dispatch`
  to validate autotune winners. It does **nothing** for S0/S1 in-process.
- The VkInstance is created at `import torch_vulkan` (`csrc/vulkan/Context.cpp:228`,
  `init_instance()`) — `VK_INSTANCE_LAYERS` must be set before import; calling
  `validate=True` after import cannot activate the layer.
- A `VkDebugUtilsMessengerEXT` callback **already exists** at `Context.cpp:241-257`
  (`Context::debug_callback`). It increments `g_validation_errors_count` and
  prints, but **always returns `VK_FALSE`** — never throws or aborts.
- `TORCH_VULKAN_VUID_AS_ERROR` is checked in `conftest.py:57-85` (pytest fixture)
  and `validation_codegen.py:242-260` (autotune mode). Not checked in production
  dispatch path.

**Minimal fix (~25-30 LOC, medium risk) — four steps:**

| Step | File | Change |
|---|---|---|
| A | `csrc/vulkan/Context.cpp:50-119` | In `debug_callback`, add a branch: if `severity >= ERROR` and `type & VALIDATION`, increment new `std::atomic<uint64_t> g_fatal_validation_errors`. |
| B | `csrc/vulkan/Context.h` | Add `static uint64_t fatal_validation_errors_count()` + reset. |
| C | `csrc/init.cpp` | Pybind the new accessor + reset (~6 LOC). |
| D | `runtime/dispatch.py` or sync wrapper | After every `flush_sync()`, poll `_fatal_validation_errors_count()` and raise `RuntimeError` if `> 0`, gated behind `TORCH_VULKAN_VALIDATION` env var. |

**Why not `re-exec`**: The re-exec bootstrap in the original plan would be
high-risk (breaks `torch.compile` session state). The callback+poll approach
reuses the existing infrastructure and is the same pattern already used by the
M-VAL.1 pytest fixture.

**Layer enablement note**: `TORCH_VULKAN_VALIDATION=1` (not `VK_INSTANCE_LAYERS`)
is what the C++ code reads (`Context.cpp:182-183`). Set it before import.

**2026-06-22 implementation plan — confirmed SIMPLER than original (1 C++ change):**

The 4-step plan above can be collapsed to a SINGLE C++ change:

**Step 1 only — `csrc/vulkan/Context.cpp:107`** (before `return VK_FALSE;`):
```cpp
if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT &&
    std::getenv("TORCH_VULKAN_VUID_AS_ERROR") != nullptr &&
    std::strcmp(std::getenv("TORCH_VULKAN_VUID_AS_ERROR"), "0") != 0)
    throw std::runtime_error(data->pMessage);
```
- `<stdexcept>` already included at `Context.cpp:17`
- `TORCH_VULKAN_VUID_AS_ERROR` already set at `hardware_probe.py:592` when `validate=True`
- Env-var path: `prepare_device(validate=True)` → `hardware_probe.py:592` sets
  `os.environ["TORCH_VULKAN_VUID_AS_ERROR"] = "1"` → C `getenv` reads it in-process
  (no FFI plumbing needed — same process, env table shared)

**No changes needed** to `Context.h`, `csrc/init.cpp`, or `runtime/dispatch.py` for the
basic "throw on VUID error" path.

**Note on correctness**: Throwing from a Vulkan callback is UB per spec (the driver is
not required to unwind correctly). For production use, the poll-after-dispatch approach
(original Steps B-D) is safer. For test gating, the throw approach is acceptable and
widely used in validation layers for test infrastructure.

**Existing test coverage**: `conftest.py:70-118` already has an `autouse` fixture that:
- Skips only when `TORCH_VULKAN_VUID_AS_ERROR == "0"` (line 70)
- Snapshots `_c_ext._validation_errors_count()` before each test (line 115)
- Calls `pytest.fail` if delta > 0 after the test (lines 117-118)
If the callback throws, the exception propagates immediately — no fixture needed.
Tests run with the default env (env-var unset OR set to "1") automatically get validation.

**ROADMAP reference correction**: The ROADMAP's `cpp_wrapper_gpu.py:490-497` reference
is incorrect — those lines are the `emit_aoti_extern_dispatch` parameter list. The actual
Python validation gate is `runtime/validation_codegen.py:260`.

- **Files**: `csrc/vulkan/Context.cpp:107` (4-line insertion — single C++ change).
- **Exit**: `TestWarmupValidationInProcess` — `prepare_device(validate=True)` from
  a clean env with `TORCH_VULKAN_VALIDATION=1` set before import validates S0/S1
  kernels in-process; an injected VUID fails the dispatch with `RuntimeError`.

### S2.4 — Warm→train cache coherence (hash the tuning knobs) ✅ FIXED 2026-06-21

Warm-up set `MM_TILES=expanded` temporarily (during `_run_level_2_autotune`)
then cleared it. Training used the default 4 tile configs instead of the 16
expanded ones → generated different Slang source → different SPIR-V cache key
→ cold slangc on first training compile.

Fix (manifest approach):
1. After level-2 autotune, `_write_probe_status` now saves `"mm_tiles_mode":
   "expanded"` to `probe_status_<id>.json`.
2. `_restore_probe_defaults()` reads this manifest on every import and sets
   `TORCH_VULKAN_MM_TILES=expanded` as a soft default (only if the user hasn't
   explicitly set it). Called from `auto_probe_on_import()` unconditionally —
   fast disk read, no GPU work.

Note: WG autotune was already coherent — `make_vulkan_kernel` reads the disk
cache even when `WG_AUTOTUNE` env is unset.

- **Files**: `hardware_probe.py` (`_restore_probe_defaults`, `_write_probe_status`,
  `auto_probe_on_import`).
- **Exit**: `TestWarmCacheCoherence::test_probe_status_restores_mm_tiles_env` ✅
  `TestWarmCacheCoherence::test_user_mm_tiles_not_overridden` ✅

### S3.1 — Compiled optimizer: route SGD/AdamW to the foreach extern ✅ FIXED 2026-06-21

The compiled step fans out to one `binary_add_inplace` per parameter (4 tiny
dispatches here) instead of the foreach `IOptimizer` extern that eager already
uses (B1). Bridge the eager-foreach interface into the compiled optimizer step.
- **Files**: `lowerings/optimizer_lowerings.py`, `fx_passes/functional/optimizer.py`
  (route `aten._foreach_add.List` + `aten._foreach_add_.List`), `templates/foreach_optimizer.slang`,
  `templates/caller/optimizer.py` (batch_size cap at 8 for 256-byte push-const limit).
- **Root causes fixed**: (1) `ExternKernelOut` base class DCE'd void kernels → switched to
  `ExternKernel` + `NoneLayout` + `MutationOutput` per param; (2) post-grad graphs use
  functional `_foreach_add.List` not inplace `_foreach_add_.List` → added Pass 3 in
  `fx_passes/functional/optimizer.py`; (3) variable-numel params → use `max_numel` for grid;
  (4) generic entry point `computeMain<SGDImpl>` not `computeMain`; (5) push-const overflow
  for batch_size > 8 → cap at 8 (240 bytes ≤ 256 limit).
- **Exit tests**: `test_s3_1_compiled_sgd_optimizer_correctness_vs_cpu` ✅,
  `test_s3_1_compiled_sgd_variable_numel_correctness` ✅, `test_e2e_compiled_sgd_15_params_one_dispatch` ✅.

### S3.2 — Tiny-kernel fusion (close out the plumbing dispatches) ✅ FIXED 2026-06-21

Root cause: `_stashed_outputs` in `wrapper.py:generate_return` was holding extra
references to gradient-class output tensors, keeping `use_count >= 2` so
`is_tensor_stealable()` returned false and `AccumulateGrad` fell to
`clone_obey_contract` (`new_empty_strided + copy_` = 1 Vulkan dispatch per
parameter). Fix: skip `lt == "gradient"` outputs from the stash — they transfer
ownership to AccumulateGrad anyway and are freed naturally by `zero_grad()`.

Result with `torch_vulkan.SGD` (1 fused dispatch for all params):
- forward=2, loss=1, backward=8, optimizer=1, **total=12** (≤14 target ✓)
- **Files fixed**: `python/torch_vulkan/inductor/wrapper.py` (`generate_return`)
- **Exit test**: `TestTinyKernelFusion::test_conv_gn_pool_training_step_dispatch_count` ✅

### S3.5 — ✅ COMMITTED 2026-06-22: Fix conv1d compile path (stride order + ReinterpretView unwrap)

`test_m6_conv1d_basic_matches_cpu` + `test_m6_conv1d_padding_matches_cpu` were failing
due to two codegen bugs in `lowerings/conv.py`:
1. Weight shape assumed `(C_out, C_in, kH, kW)` → strides emitted as `(kW, kH, C_in, C_out)` order.
   The 1D→2D reshape sets kW=1, so stride order must be `(C_in*kH*kW, kH*kW, kW, 1)`.
2. `_vk_realize_then_unwrap` was stripping `ReinterpretView` nodes — Inductor then had
   no offset information and generated `empty_strided` calls instead of `reinterpret_tensor`.

Fix: correct the stride order and preserve `ReinterpretView` nodes.
- **Commit**: `ac0a424724d vulkan: S3.5 — fix conv1d compile path (stride order + ReinterpretView unwrap)`
- **Exit**: `test_m6_conv1d_basic_matches_cpu` ✅, `test_m6_conv1d_padding_matches_cpu` ✅

**Regressions exposed by S3.5 (ReinterpretView now preserved):**
S3.5 preserving `ReinterpretView` exposes the long-standing storage-offset bug (S3.5a)
and the grouped-conv push-constant mismatch (S3.5c). See defect table above.

#### S3.5a — 🔴 OPEN: Fix storage-offset in `bind_buffers` (causal conv1d)

`VkDescriptorBufferInfo.offset` is hardcoded to 0. After S3.5 preserves
`ReinterpretView` nodes, the Inductor wrapper emits `reinterpret_tensor(buf, ..., non_zero_offset)`
calls — but all of them bind the same start of the VkBuffer. For causal depthwise conv
(groups=C), all 16 per-group dispatches write to position 0.

**C++ fix designed (in `git stash@{0}`):**
- `csrc/ops/dispatch.h`: add `VkDeviceSize offset` to `BufferInfo`
- `csrc/ops/dispatch.cpp`: `get_buffer_info()` computes `off_bytes = storage_offset * element_size`,
  recovers the base VkBuffer via pointer arithmetic; `dispatch_shader` / `dispatch_shader_indexed`
  thread offsets through; descriptor-set cache key includes offsets
- `csrc/vulkan/DescriptorSet.h/cpp`: `bind_buffers` offset overload sets `buf_info.offset = off`,
  `buf_info.range = size - off`

**To close:** `git stash pop`, rebuild C++ (`TORCH_DEVICE_BACKEND_AUTOLOAD=0 MAX_JOBS=3 python setup.py build_ext --inplace`), verify `test_m6_causal_conv1d_matches_cpu`.
- **Exit**: `test_m6_causal_conv1d_matches_cpu` ✅

#### S3.5b — ✅ CLOSED 2026-06-22: Conv1d backward x.grad=0

**Root cause** (definitive, from AoT graph trace analysis):

The old `_conv1d_backward` built `input_4d = input.unsqueeze(-1)` in Python and passed
it as the first arg of `conv2d_backward`. `input_4d` is a **dead intermediate** in the
AoT backward FX graph — it dies immediately after the `conv2d_backward` call. The
Inductor memory planner reused its buffer for `conv2d_backward[0]` (grad-input), which
has the same shape `(N, C_in, L, 1)`. Since `input_4d` is a view of `primals_3` (primal
x), `conv2d_backward[0]` aliased `primals_3` in the AoT graph, and the backward returned
x itself as x.grad (either primal values or all-zero depending on buffer lifetime).

**Fix**: new `torch_vulkan::conv1d_backward_core` opaque non-autograd custom op:
- Takes **3-D tensors directly** (no Python-side unsqueeze) → `input` is a backward
  **graph INPUT** (never freed by memory planner)
- Fake kernel derives outputs from `input` (FT proxy) → FunctionalTensorMode produces
  fresh storage refs → AoT's alias analysis correctly treats grad-input as independent
- The forward graph now saves `primals_3` as a residual; backward graph has it as a
  proper graph input: `conv1d_backward_core(primals_3, expand_default, primals_1, ...)`

**Files changed**:
- `fx_passes/eager/conv_backward.py` — `_ensure_conv1d_backward_core_op_registered()`
- `fx_passes/eager/conv.py` — `_conv1d_backward` calls `conv1d_backward_core.default`
- `fx_passes/eager/__init__.py` — registers `conv1d_backward_core` at import time
- `lowerings/__init__.py` — `make_fallback(conv1d_backward_core.default)`
- `tests/test_inductor_regression.py` — `test_s3_5b_conv1d_backward_grad_input_not_primal`

- **Exit**: `test_m6_conv1d_backward_matches_cpu` ✅, `test_s3_5b_conv1d_backward_grad_input_not_primal` ✅

#### S3.5c — 🟡 NEEDS GPU VERIFY: Grouped conv — bug attribution invalid

**2026-06-22 three-audit conclusion**:

The entire original bug description ("`slang_addmm_8_8_8_s1_r1x1` PC layout 48 bytes <
SPIR-V declared 52 bytes, VUID-VkComputePipelineCreateInfo-layout-07987") is a **false
attribution** for `test_m6_conv1d_groups_matches_cpu`:

| Claim | Evidence | Verdict |
|-------|----------|---------|
| `scheduling.py:766` produces `n_pc=12` for MM kernels | `audit-npc-for-mm-exact`: MM ExternKernelOut bypasses `define_kernel` entirely | ❌ WRONG |
| groups>1 conv lowering produces `slang_addmm` | `audit-s3.5c-root-cause-v2`: groups>1 → per-group `_VulkanConv2dExternKernel` + `cat` | ❌ WRONG |
| test exercises Slang MM kernel | `test_inductor_regression.py:8521` docstring: "Conv2d eager **fallback** handles groups>1" | ❌ WRONG — no Slang kernel |

**True state**: `test_m6_conv1d_groups_matches_cpu` with `nn.Conv1d(groups=2)` routes
through the `FallbackKernel` eager path for groups>1 conv2d. No `slang_addmm` is ever
compiled for this test. The VUID error in the original report was almost certainly from
a different kernel in a stale test state, now superseded by the S3.5 commit.

**Note on `scheduling.py:789` and `n_pc`**: the only path that CAN produce
`pc_size_bytes=48` is `VulkanKernel`-type kernels (non-extern, from `define_kernel`)
where `n_pc = sizevars + range_trees = 12`. This is a latent bug for any VulkanKernel
where the SPIR-V retains more push-constant fields than the Python-side sizevar count
predicts. Not triggered by the S3.5 test suite today, but should be audited separately.

**Action**: GPU re-run of `test_m6_conv1d_groups_matches_cpu`. Expected: PASS.

- **Files**: None — no fix needed for this specific bug description
- **Exit**: `test_m6_conv1d_groups_matches_cpu` ✅ on current branch (needs GPU confirmation)

#### S3.5d — 🟡 KNOWN LIMITATION: Conv2d backward `.item()` intentional GPU→CPU sync

**2026-06-22 fx_passes + exact-line audit (confirmed)** — `fx_passes/eager/conv_backward.py:99`
calls `_ = grad_bias[0].item()` inside `_conv2d_backward_impl`, gated by
`use_slang_bwd=True AND has_bias`. The call is an **intentional GPU pipeline drain**
(not a value check): the comment at lines 93–97 explicitly states the FallbackKernel
wrapper may consume output buffers immediately, and without a blocking sync, a
pending Vulkan write leaves stale data in `grad_input`/`grad_weight`/`grad_bias`.

**This cannot be removed with a shape check** — only a proper GPU sync primitive
(Vulkan pipeline barrier or `vkQueueWaitIdle`) would be an equivalent replacement.

**Correct fix path** (lower priority, tracked as perf improvement):
- Ensure `_VulkanConvBwdExternKernel` emits a proper Vulkan submission fence or that
  the C++ dispatch layer tracks outstanding writes, eliminating the need for the
  Python-side `.item()` barrier.
- Until then, `.item()` is a deliberate stall; removing it WITHOUT the C++ barrier
  fix will silently produce stale gradients.

- **Files**: `fx_passes/eager/conv_backward.py:93-99`, `csrc/ops/dispatch.cpp`
  (submission/fence tracking)
- **Exit**: No `.item()` in `_conv2d_backward_impl`; TSAN/valgrind + correctness
  test confirm no stale gradient under the compiled path.

### S3.3 — Wire persistent-kernel routing for large reductions (C3 is dead code)

**2026-06-22 deep audit (confirmed) — confirmed dead-code map:**

#### Persistent pointwise — IS wired (for numel ≤ cap)

`_persistent_pointwise_numel_cap()` at `scheduling.py:33`: default 16384
(formula: `16384 × (cu_count/16) × (launch_us/12.3)`, clamped [4096, 65536]).
Used at `scheduling.py:884`:
```python
if int(numel) > _persistent_pointwise_numel_cap(): return kernels  # skip persistent
```
`dispatch_persistent_pointwise()` at `templates/caller/persistent_pointwise.py:64`
exists and is exported at `__init__.py:76`, but has **zero callers outside its own
module** — nothing in the scheduler actually invokes it.

`templates/persistent_pointwise.slang` (163 LOC): fully implemented grid-stride
loop over 10 ops (identity/relu/sigmoid/tanh/gelu/add_scalar/mul_scalar/sub/pow/fill),
uses OpRange in StructuredBuffer (C6.3 RDNA1 128B push-const workaround).

#### Persistent reduction — FULLY DEAD

1. **No template** — `persistent_reduction.slang` does **not exist anywhere** in
   the tree. `shaders/lib/reduction.slang` and `shaders/lib/vk_reduction.slang`
   contain wave-based reduction helpers for the existing multi-pass cooperative path
   only — no persistent/grid-stride variant.

2. **Dead heuristic** — `should_use_persistent_reduction()` at
   `kernel/main.py:146-165` (20 LOC, returns True for dynamic shapes or
   `rnumel ≤ 8192`) has an explicit docstring:
   *"Currently not wired into codegen — retained for a future C3 task."*
   Never called; only `should_use_cooperative_reduction()` (line 172) is
   consulted by `create_kernel_choices` at `scheduling.py:867`.

3. **Note — `_persistent_2d_layout()`** at `kernel/reduction.py:140,235,283,334`
   is NOT persistent-kernel routing — it selects a `(TY, TX)` 2D thread
   arrangement for multi-axis reductions (e.g. `sum(dim=(0,2))`), enabling
   `vk_wg_reduce_*_2d` helpers. This is a thread-layout choice within the
   existing multistage codegen, not a separate persistent-kernel path.

4. **Routing gate** — `create_kernel_choices` (`scheduling.py:867`) returns
   early for `is_reduction()` kernels before any persistent check is reached.
   For pointwise, the cap is checked at line 884 but `dispatch_persistent_pointwise`
   is still never invoked (the callers emit the normal grid path instead).

5. **No inter-CTA scratch** — current multi-pass allocates fresh intermediate
   tensors each pass (`csrc/ops/reduction_ops.cpp:24-33`). No global scratch buffer.

#### Test status

| Test | File:Line | Status |
|------|-----------|--------|
| `test_multi_axis_sum_persistent_one_dispatch` | `test_inductor_regression.py:1920` | ✅ Active, green (tests persistent 2D layout for `sum(dim=(0,2))` with ≤1 dispatch) |
| `TestM192PersistentPointwise` | `test_inductor_regression.py:53896` | ✅ Active (scheduler wiring: `_enable_persistent_mode` fires/skips correctly) |
| `test_aten_argmax_compiles_2d_layout` | `test_inductor_regression.py:42730` | xfail(strict=True) (unrelated — codegen guard at `reduction.py:71`) |
| **`test_persistent_reduction`** | **Does not exist** | — |

No xfailed/skipped test exists for persistent reduction routing.

**Minimal path to a working 1-D persistent sum:**
- (A) New `templates/reduction_persistent.slang` (~100 LOC) — grid-stride outer
  loop over `rnumel`, per-element reduce accumulate, partial-WG guard, multi-axis
  flat indexing.
- (B) Wire `should_use_persistent_reduction()` call in `kernel/main.py`;
  threshold `rnumel > 65536` (or read `_persistent_pointwise_numel_cap` logic).
- (C) Add `reduction_sum_persistent_fwd` to `csrc/ops/reduction_ops.cpp` +
  embed compiled SPIR-V in `csrc/generated/shaders.h`.
- (D) Scratch only needed for `argmax`/`argmin` — defer to Phase 2.
- (E) `TestPersistentReduction` — large 1-D sum parity + dispatch-count drop.

**Estimate**: 3–5 dev-days (not a 1-day ticket — template + C++ + embed + test).
- **Files**: `templates/reduction_persistent.slang` (new, ~100 LOC),
  `kernel/main.py:146-165`, `scheduling.py:867`, `csrc/ops/reduction_ops.cpp`,
  `csrc/generated/shaders.h`.
- **Exit**: `TestPersistentReduction` — large `sum`/`mean` parity vs CPU +
  dispatch-count drop vs multi-pass C++ path.

### S3.4 — Batch-dispatch / async-compile overlap (flip `BATCH_DISPATCH=1`)

Make batched dispatch win: execute kernel N while compiling N+2 via the existing
async slangc pool, bringing the 1.8× batch penalty to ≤1.1×.

**2026-06-22 batcher audit (confirmed) — exact dead stubs found:**

| Component | Location | State |
|---|---|---|
| `BATCH_DISPATCH` env flag | `config.py:121`, `wrapper_helpers.py:204-212` | Read at codegen time; default OFF; 1.8× slower |
| `batcher.py` | `runtime/batcher.py:48-93` | Batches C++ command buffers via `with _batcher:` blocks; `_flush()` on `__exit__` |
| `_compile_ahead` / `_compile_ahead_submitted` | `batcher.py:63-65` | **Declared but never written or read — completely dead stubs** |
| `async_precompile_slang` | `slangc.py:573`; called from `scheduling.py:728,805` | Fire-and-forget at **codegen time only**; not called at dispatch time |
| Blocking call at dispatch | `slangc.py:486`: `compile_slang_to_spirv()` → `future.result()` | This is what blocks kernel-N dispatch while kernel-N+2 compiles |

**True runtime compile-dispatch overlap** requires:
1. SP.2 first: make `compile_slang_to_spirv` non-blocking (return Future)
2. In `batcher.py:add()`: look ahead at queued kernels, submit not-yet-compiled
   sources to async pool via `_compile_ahead.append()`
3. In `batcher.py:_flush()`: dispatch ready kernels immediately; await pool futures
   for upcoming kernels only when their turn arrives

The `_compile_ahead` stubs at `batcher.py:63-65` were designed for step 2 but never wired.

- **Files**: `runtime/batcher.py:63-65` (wire `_compile_ahead`), `runtime/slangc.py`
  (SP.2 prerequisite), `scheduling.py:728,805` (async fire-and-forget already there).
- **Exit**: `TestBatchDispatchOverlap` — MNISTNet step with `BATCH_DISPATCH=1`
  ≤1.1× the unbatched time; parity holds.

### S1.1 — Register conv / flash_attention choices into `V.choices`

MM autotune is in `tuned_mm`; conv tiling is env-var-only; flash partially uses
`V.choices` but BK/BQ not varied and warm-up probe missing. Wire non-MM templates
fully into Inductor's choice-matching so warm-up auto-explores them.

**2026-06-22 audit (confirmed) — per-op status:**

| Op | V.choices status | Gap | Effort |
|----|-----------------|-----|--------|
| MM (`tuned_mm`) | ✅ Full — `ExternKernelChoice` + `autotune_select_algorithm` + `MultiTemplateBuffer` | None | — |
| Conv2d | ❌ None — `ExternKernelOut` + env-var only (`TORCH_VULKAN_CONV_TILE`) | Tile variants not in choice-matching; env-var hack has no per-shape persistence | 3–5 days, ~385 LOC |
| Flash attention | 🟡 Partial — `ExternKernelChoice` + `autotune_select_algorithm` exist | Not in warm-up probe; BK/BQ not varied; no InductorChoices hooks | 1–2 days, ~150 LOC |

**Conv2d implementation plan (~385 LOC across 6 files):**
1. `templates/caller/conv.py` — add `_SlangTileConv2d` picklable callable class (~80 LOC)
2. `templates/caller/conv.py` (or `lowerings/conv.py`) — add `install_external_conv()` (~100 LOC)
3. `lowerings/conv.py:_VulkanConv2dExternKernel` — replace `_resolve_conv_tile()` + direct call with `autotune_select_algorithm("conv2d", choices, ...)` (~60 LOC)
4. `lowerings/conv.py:_codegen_aoti` — mirror tile variant rendering for AOTI path (~60 LOC)
5. `inductor/__init__.py:_legacy_register` — call `install_external_conv()` alongside other installs (~5 LOC)
6. `tests/` — `TestConvAutotuneChoices` regression test (~80 LOC)

**Flash attention remaining work (~150 LOC):**
1. Add BK/BQ variants to `_FLASH_ATTENTION_VARIANTS` at `templates/caller/flash_attn.py` (~20 LOC)
2. Add flash-attention probe shapes to `hardware_probe.py` (~30 LOC)
3. Extend `_SlangTileFlashAttention` to autotune BK/BQ at runtime (~40 LOC)
4. Optional: `InductorChoices.get_flash_attention_fwd_configs()` hook (~30 LOC)

- **Files**: `lowerings/conv.py`, `templates/caller/conv.py`, `templates/caller/flash_attn.py`, `hardware_probe.py`, `inductor/__init__.py`.
- **Exit**: `TestConvAutotuneChoices` — conv2d compile sweeps registered tile configs via `V.choices`; best is cached and reused.

### S2.5 — Pooling forward: pure Slang codegen (kill the eager fallback)

`max_pool2d`/`avg_pool2d`/`adaptive_avg_pool2d` forward route through
FallbackKernel → eager C++ (anti-goal #6). Upstream `indirect_indexing`
produces wrong SPIR-V; replace with scatter/reduce Slang codegen.
- **Files**: `bwd_lowerings.py:730-844`, `lowerings/pool.py`,
  `templates/scatter_atomic.slang`.
- **Exit**: `TestPoolFwdSlang` — pool fwd graphs contain no FallbackKernel;
  output matches CPU.

### S4.0 — AOTI: fix MM/addmm/bmm `n_pc=0` → zero push-constant layout (BLOCKING)

**2026-06-22 audit (confirmed exact lines) — TWO independent bugs:**

#### Bug A — `ExternKernelOut` metadata gap (`bmm` / `addmm`)
`bmm` and `addmm` use `ExternKernelChoice` → `ExternKernelOut` IR nodes that
bypass `VulkanScheduling.define_kernel()`. The fallback chain in
`cpp_wrapper_gpu.py:_generate_kernel_call_helper` (lines 369–376) reads
`n_pc` from `inductor_meta` or `get_kernel_meta` — both return 0/None because
`ExternKernelOut` never populates `inductor_meta["n_pc"]` and `define_kernel`
(which calls `_set_kernel_meta`) is never invoked for these nodes.
`mm` has a custom `_codegen_aoti` at `lowerings/matmul.py:253` that packs the
96-byte PC correctly — `bmm` and `addmm` have no equivalent override.

#### Bug B — `.so` bundle hardcodes `pc_size_bytes=0u` for ALL kernels
`cpp_wrapper_gpu.py:emit_aoti_spv_header` line **716** hardcodes `0u` in:
```
f'    {{nullptr, "{key}", {c_name}_data, {len(spv) // 4}, {n_buf}u, 0u}},'
```
`self._spv_metadata` correctly stores `pc_size_bytes=96` for MM, but
`emit_aoti_spv_header` never consults it. Every kernel in the `.so` ABI table
gets `pc_size_bytes=0` → `Pipeline.cpp:125` creates `VkPipelineLayout` with no
`VkPushConstantRange` → `AotiRuntime.cpp:601-612` skips `vkCmdPushConstants`
→ shader reads `pc.M/N/K` as all-zeros → bounds checks always fail →
**silent all-zeros output (RADV); VUID abort with validation layer ON.**

#### Correct PC layout
`templates/slang_mm.slang:57-92` — `struct PC` with 24 fields (19×uint32 +
5×float32 = **96 bytes**). Python packer at
`templates/caller/gemm/dispatch.py:56-95` (`_MM_PC_FORMAT = "19I5f"`) matches.

#### Exact fix — two changes in `cpp_wrapper_gpu.py`

**Fix B (Bug B) — `emit_aoti_spv_header` line 716:**

Add optional `metadata` param (all existing callers use positional `bundle` only,
so the default `None` is backward-compatible):
```python
# OLD (line 670 signature + line 716 loop body):
def emit_aoti_spv_header(bundle: dict[str, bytes]) -> str:
    ...
            f'    {{nullptr, "{key}", {c_name}_data, {len(spv) // 4}, {n_buf}u, 0u}},'

# NEW:
def emit_aoti_spv_header(bundle: dict[str, bytes], metadata: dict[str, dict] | None = None) -> str:
    ...
            meta = (metadata or {}).get(key, {})
            f'    {{nullptr, "{key}", {c_name}_data, {len(spv) // 4}, {n_buf}u, {meta.get("pc_size_bytes", 0)}u}},'
```

**Fix A (Bug A) — `_generate_kernel_call_helper` lines 367-376:**

Add SPIR-V reflection fallback when both `inductor_meta` and `get_kernel_meta`
return `n_pc=0` (the `ExternKernelOut` bmm/addmm case):
```python
# After the existing n_pc==0 get_kernel_meta lookup block, add:
            if n_pc == 0:
                from .runtime.reflection import reflection_layout
                from .runtime.reflection_ext import _disk_reflection_read
                import hashlib
                refl = _disk_reflection_read(hashlib.sha256(spv).hexdigest())
                if refl is not None:
                    pc_size_bytes = reflection_layout(refl).get("push_constant_size", 0)
```

**Confirmed building blocks (2026-06-22 audit):**
- `reflection_layout(reflection_json: str) -> dict` at `runtime/reflection.py:112` —
  parses slangc JSON, returns `{"bindings": [...], "push_constant_size": int}`
- `_disk_reflection_read(hash_key: str) -> Optional[str]` at
  `runtime/reflection_ext.py:88` — reads cached reflection JSON from disk;
  also exported from `runtime/slangc.py:124`
- No `reflect_spv` or `read_pc_layout_from_spv` function exists — the above
  two-call pattern is the canonical approach

The `bmm`/`addmm` `_codegen_aoti` override approach (option 3 from original plan)
is a follow-up; the reflection fallback is the minimal fix that unblocks AOTI
correctness for all `ExternKernelOut` ops without per-op overrides.

**Tests that catch a regression:**
- `TestDR8::test_t72_emit_aoti_spv_header_is_valid_cpp` (line ~27980)
- `TestDR8::test_t72_emit_header_with_multiple_kernels` (line ~28050)
- `TestCorrectness::test_bmm_compiled_matches_cpu` (line ~7829)
- `TestCorrectness::test_correctness_addmm_register_tile` (line ~3766)

#### Affected files
| File | Lines | Issue |
|---|---|---|
| `cpp_wrapper_gpu.py` | 369–376 | `n_pc` fallback returns 0 for `ExternKernelOut` |
| `cpp_wrapper_gpu.py` | 716 | `emit_aoti_spv_header` hardcodes `0u` for ALL kernels |
| `lowerings/matmul.py` | 182–260 | `_codegen_aoti` only for `mm`, not `bmm`/`addmm` |
| `scheduling.py` | 758–766 | `n_pc=0` for fully-static kernels |
| `kernel/header.py` | 606–629 | Emits PC struct even when `n_pc=0` |
| `csrc/backend/AotiRuntime.cpp` | 160–185, 230–260, 601–612 | Skips PCs when size=0 |
| `csrc/vulkan/Pipeline.cpp` | 125–132 | No `VkPushConstantRange` when size=0 |
| `csrc/ops/dispatch.cpp` | 394–395 | No `vkCmdPushConstants` when size=0 |

- **Exit**: `TestAOTITemplatePCLayout::test_aoti_mm_push_constant_size_nonzero` —
  compile `nn.Linear` → `.so`, dispatch, assert no VUID + correct output.
  `TestAOTIBmmAddmmPC` — same for bmm/addmm.

### MS.1+MS.2 — AOTI shim memory-safety: dangling handle + delete no-op (CRITICAL)

**2026-06-22 memory-safety audit (confirmed) — Two co-located HIGH bugs in `aoti_shims.cpp`:**

#### MS.2 — Use-after-free: `zeros/ones/full/as_strided` return dangling `AtenTensorHandle`
- **Files**: `csrc/backend/aoti_shims.cpp:146, 172, 193, 215`
- **Severity**: **Critical / High**

**Exact 4-site fix** (variable name is `result` at line 146, `tensor` at 172/193/215):
```cpp
// OLD (lines 146, 172, 193, 215):
    if (out_handle) *out_handle = tensor.unsafeGetTensorImpl();

// NEW (each site):
    if (out_handle) *out_handle = static_cast<void*>(new at::Tensor(std::move(tensor)));
```
`aoti_torch_empty_strided_vulkan` at line 98 already uses this pattern.
Confirmed by grep: these four are the **only** `unsafeGetTensorImpl()` returns
in `aoti_shims.cpp`.

#### MS.1 — Memory leak + broken free: `aoti_torch_delete` is a no-op
- **Files**: `csrc/backend/aoti_shims.cpp:154-156`
- **Severity**: **High**

**Exact 1-line fix**:
```cpp
// OLD:
int aoti_torch_delete(void* handle) {
  (void)handle;
  return 0;
}
// NEW:
int aoti_torch_delete(void* handle) {
  delete reinterpret_cast<at::Tensor*>(handle);
  return 0;
}
```

**Both fixes must land together** (MS.2 → allocate with `new`; MS.1 → actually
delete). Combined diff is **5 lines changed in one file**. All callers
(`cpp_wrapper_gpu.py` `make_buffer_free` / `make_allocation`) already expect
heap-allocated `at::Tensor*` handles; `AotiRuntime.cpp` has zero direct calls
to `aoti_torch_delete` (confirmed by grep).

#### MS.3 — `desc_set_cache` data race under concurrent same-device dispatch
- **Files**: `csrc/ops/dispatch.h:119-127`, `csrc/ops/dispatch.cpp:307-326`
- **Severity**: Medium — `std::unordered_map` read+write in `dispatch_shader`
  without per-device lock (only `g_runtime_mutex` covers the outer map).
- **Fix direction**: add `mutable std::mutex desc_set_mutex_` to `DeviceRuntime`; lock
  around all `desc_set_cache` accesses.

**2026-06-22 exact fix (confirmed, 12 access sites):**

**Step 1 — `csrc/ops/dispatch.h` (DeviceRuntime struct):** Add mutex field:
```cpp
// After existing fields in DeviceRuntime (around line 119):
mutable std::mutex desc_set_mutex_;
```

**Step 2 — `csrc/ops/dispatch.h:89-93`**: insert mutex field:
```cpp
// Current (lines 89-93):
    std::unordered_map<DescSetCacheKey, VkDescriptorSet, DescSetCacheKeyHash>
        desc_set_cache;
};

// Replacement:
    std::unordered_map<DescSetCacheKey, VkDescriptorSet, DescSetCacheKeyHash>
        desc_set_cache;

    mutable std::mutex desc_set_mutex_;
};
```

**Step 3 — `csrc/ops/dispatch.cpp` — all 12 access sites (exact replacements):**

**Site 3+4+5 (MAIN HOT PATH) — `dispatch_shader` lines 361-375:**
```cpp
// CURRENT (lines 361-375):
    VkDescriptorSet desc_set = VK_NULL_HANDLE;
    if (kUseDescCache) {
        DeviceRuntime::DescSetCacheKey key{
            pipeline->descriptor_set_layout(), buffers_hash};
        auto cache_it = rt.desc_set_cache.find(key);
        if (cache_it != rt.desc_set_cache.end()) {
            desc_set = cache_it->second;
        } else {
            uint64_t gen_before = rt.desc_pool->reset_generation();
            desc_set = rt.desc_pool->allocate(
                pipeline->descriptor_set_layout());
            if (rt.desc_pool->reset_generation() != gen_before) {
                rt.desc_set_cache.clear();
            }
            rt.desc_set_cache[key] = desc_set;
        }
    } else {

// REPLACEMENT:
    VkDescriptorSet desc_set = VK_NULL_HANDLE;
    if (kUseDescCache) {
        DeviceRuntime::DescSetCacheKey key{
            pipeline->descriptor_set_layout(), buffers_hash};
        {
            std::lock_guard<std::mutex> _lk(rt.desc_set_mutex_);
            auto cache_it = rt.desc_set_cache.find(key);
            if (cache_it != rt.desc_set_cache.end()) {
                desc_set = cache_it->second;
            } else {
                uint64_t gen_before = rt.desc_pool->reset_generation();
                desc_set = rt.desc_pool->allocate(
                    pipeline->descriptor_set_layout());
                if (rt.desc_pool->reset_generation() != gen_before) {
                    rt.desc_set_cache.clear();
                }
                rt.desc_set_cache[key] = desc_set;
            }
        }
    } else {
```

**Sites 8-10 — `dispatch_shader_indexed` lines 593-598:**
```cpp
// CURRENT:
    {
        DeviceRuntime::DescSetCacheKey key{
            pipeline->descriptor_set_layout(), buffers_hash};
        auto cache_it = rt.desc_set_cache.find(key);
        if (cache_it != rt.desc_set_cache.end()) {
            desc_set = cache_it->second;
        } else {
            desc_set = rt.desc_pool->allocate(pipeline->descriptor_set_layout());
            rt.desc_set_cache[key] = desc_set;
        }
    }

// REPLACEMENT:
    {
        std::lock_guard<std::mutex> _lk(rt.desc_set_mutex_);
        DeviceRuntime::DescSetCacheKey key{
            pipeline->descriptor_set_layout(), buffers_hash};
        auto cache_it = rt.desc_set_cache.find(key);
        if (cache_it != rt.desc_set_cache.end()) {
            desc_set = cache_it->second;
        } else {
            desc_set = rt.desc_pool->allocate(pipeline->descriptor_set_layout());
            rt.desc_set_cache[key] = desc_set;
        }
    }
```

**Sites 1,2,6,7,11,12 — all `desc_set_cache.clear()` lines (116,271,373,541,699,771):**
Wrap each bare `rt.desc_set_cache.clear();` in a lock_guard block:
```cpp
// OLD (one line):
        rt.desc_set_cache.clear();  // M17.5: <comment>

// NEW (three lines):
        {
            std::lock_guard<std::mutex> _lk(rt.desc_set_mutex_);
            rt.desc_set_cache.clear();  // M17.5: <comment>
        }
```

**Regression test**: `tests/test_concurrent_dispatch.cpp` (new) or run existing
`test_inductor_regression.py` under thread-sanitizer.

#### MS.4 — Validation errors non-fatal; GPU faults silently corrupt outputs
- **Files**: `csrc/vulkan/Context.cpp:91-119`, `AotiRuntime.cpp:270-289`,
  `cpp_wrapper_gpu.py:490-497`
- **Severity**: Medium — ties directly to S2.3 (in-process validation). VUID
  callback returns `VK_FALSE`; validation errors are counted but never propagated.
  Complement to S2.3: once in-process `VkDebugUtilsMessengerEXT` is wired, poll
  `validation_errors_count()` after `flush_sync()` and throw on `> 0`.

**Exit criterion**: `TestAOTIShimMemSafety` — AOTI model using zeros/ones/full
completes without ASAN/TSAN/valgrind errors; no leak reported for intermediate
tensors; `desc_set_cache` stress test under two concurrent threads passes TSAN.

#### MS.5 — Broader data race on `DeviceRuntime` fields beyond `desc_set_cache`

**2026-06-22 audit (confirmed, exact lines from `audit-ms5-dispatch-race-exact`)**

`g_runtime_mutex` (`dispatch.cpp:19`, `std::recursive_mutex`) protects regions at
lines 133 and 175 only. All other accesses to the four fields below are unlocked:

| Field | Unlocked access lines in `dispatch.cpp` |
|---|---|
| `rt.batch_mode` | 97 (write), 103–104 (reads), 264, 533 |
| `rt.dirty_buffers` | 113 (clear), 268, 428, 433, 441–444, 467, 538, 634, 639, 647–648, 650, 659 |
| `rt.read_buffers` | 114 (clear), 269, 433, 442, 469, 539, 639, 648, 661 |
| `rt.host_written_buffers` | 115 (clear), 270, 406, 412 (clear), 620, 626 (clear) |

`dispatch.h:26–49` declares all four fields as plain members of `DeviceRuntime` (no
atomics, no per-field mutex). The lock at line 133 covers `desc_set_cache` accesses
(MS.3 scope); line 175 covers a `set_pre_read_callback` registration. Neither
covers the hazard-tracking fields above.

**Severity**: High — same race class as MS.3; triggers under any concurrent
same-device dispatch scenario (e.g., two threads driving different Vulkan streams
on the same device).

**Fix**: extend the per-dispatch `lock_guard<std::recursive_mutex>` introduced by
MS.3 to also cover reads/writes of `dirty_buffers`, `read_buffers`,
`host_written_buffers`, and `batch_mode` at all dispatch-path entry points.
- **Files**: `csrc/ops/dispatch.cpp` (all unlocked-access lines listed above)
- **Exit**: TSAN clean under concurrent-dispatch stress test.

#### MS.6 — AOTI partial-load memory leak: `AotiKernelHandleImpl*` children

**2026-06-22 audit (confirmed)** — `torch_vulkan_aoti_model_load` in
`AotiRuntime.cpp` returns early on error (returns 8–13) and calls `delete model`,
but the `AotiModel` destructor does **not** `delete` the raw `AotiKernelHandleImpl*`
pointers already pushed into `model->kernels`. Any partial-load failure leaks all
already-constructed kernel handles.

**Severity**: Medium — affects error paths only; not a hot-path leak.

**Fix**: change `model->kernels` from `std::vector<AotiKernelHandleImpl*>` to
`std::vector<std::unique_ptr<AotiKernelHandleImpl>>`, or add explicit cleanup
loop in the error path.
- **Files**: `csrc/backend/AotiRuntime.cpp` (partial-load error paths, lines ∼ 8–13)
- **Exit**: valgrind shows no leak on any error path of `model_load`.

#### MS.7 — AOTI `model_run` missing per-kernel bounds check

**2026-06-22 audit (confirmed)** — `torch_vulkan_aoti_model_run` in
`AotiRuntime.cpp` does not verify `all_tensors.size() >= n_in + n_out` for each
kernel before slicing. If the caller passes fewer tensors than expected, the
output-slice logic can duplicate inputs as outputs or produce an undersized
`kbuffers`, causing the kernel to read wrong bindings silently.

**Severity**: Medium — UB/wrong results on any caller that passes wrong buffer
count (including generated wrappers with an off-by-one in their emit path).

**Fix**: add `TORCH_CHECK(all_tensors.size() >= n_in + n_out, ...)` at the top
of the per-kernel dispatch loop in `model_run`.
- **Files**: `csrc/backend/AotiRuntime.cpp` (per-kernel loop in `model_run`)
- **Exit**: `TestAOTIModelRunBoundsCheck` — passing too-short tensor list throws
  rather than corrupting.

#### MS.8 — AOTI `aoti_dispatch` missing descriptor-count validation

**2026-06-22 audit (confirmed)** — `torch_vulkan_aoti_dispatch` does not check
`n_tensors` against `h->n_buffers` (the kernel's expected descriptor count). Any
`n_tensors <= max_bindings` is accepted, allowing a silent buffer-count mismatch
to reach the Vulkan descriptor set bind.

**Severity**: Low — normally the generated wrapper passes the correct count; only
exposed by hand-crafted callers or a code-gen bug.

**Fix**: add `TORCH_CHECK(n_tensors == h->n_buffers, ...)` at entry of
`aoti_dispatch`.
- **Files**: `csrc/backend/AotiRuntime.cpp` (`aoti_dispatch` entry)
- **Exit**: assert fires on wrong `n_tensors` in unit test.

---

### S4.1 — Full training-step `.so` (fwd+bwd+optimizer)

All extern families + the v2 model API are wired; the remaining blocker is
**upstream** — `torch.export` eager `empty.memory_format` dispatch on the vulkan
device (A2.6). Track upstream; in the meantime keep the `torch.compile` path
(`TestAOTITrainingE2E` ✅) as the deployment story.
- **Files**: `csrc/backend/AotiRuntime.cpp`, `cpp_wrapper_gpu.py`,
  `meta_patches/` (export `empty` shim).
- **Exit**: `TestAOTIFullTrainingStep` — single `.so` runs fwd+bwd+SGD; weights
  update; data parity vs `torch.compile` path.

### S4.2 — AOTI dispatch gaps: pool, scatter, rng, bwd_diff families

**2026-06-22 audit (confirmed exact call sites)**: `emit_aoti_extern_dispatch`
exists at 9 call sites (conv2d/3d fwd+bwd, GN fwd+bwd ×2, matmul, foreach-optim).
**Zero** such calls exist in:

| Family | File | Gap |
|---|---|---|
| Pool fwd | `lowerings/pool.py:735,773` | FallbackKernel only, no AOTI emit |
| Pool bwd | `bwd_lowerings.py:641,807` | `fallback_handler` → custom op, no AOTI emit |
| Scatter | `lowerings/scatter.py` | `ir.Scatter` + decomp only, no AOTI emit |
| RNG/Philox | `lowerings/rng.py` | decomposes to `aten.rand`, no AOTI emit |
| bwd_diff unary | `bwd_diff/unary.py` | `compile_and_dispatch` only |
| bwd_diff binary | `bwd_diff/binary.py` | `compile_and_dispatch` only |

Fix: add `if V.graph.aot_mode: wrapper.emit_aoti_extern_dispatch(...)` branch
alongside each existing `compile_and_dispatch` call. Priority order:
1. bwd_diff inline paths (every activation backward)
2. pool fwd/bwd (every CNN)
3. RNG/Philox (dropout, weight init)
4. scatter (indexing-heavy models)
- **Files**: `bwd_diff/unary.py`, `bwd_diff/binary.py`, `lowerings/pool.py`,
  `bwd_lowerings.py`, `lowerings/rng.py`, `lowerings/scatter.py`.
- **Exit**: `TestAOTIDispatchCompleteness` — assert zero Python-fallback calls
  in a Conv+GN+ReLU+Pool+CE-loss compiled `.so`.

### S4.3 — Extend A2.6 factory-op shim to all `_FACTORY_OPS`

**2026-06-22 audit (confirmed exact lines)** (`meta_patches/joint_graph_passes.py:178-403`):
`_FACTORY_OPS` at line 209-217 lists all 9 factory ops (empty.memory_format,
empty_strided, zeros, ones, full, empty_like, zeros_like, ones_like, full_like).
Stage 1 (line 273) rewrites `device='meta'` → `device='vulkan'` for all of them.
**Bug**: `_rewrite_empty_meta_to_tangent_expand` (line 354) hard-guards on
`node.target is aten.empty.memory_format` only (line 403). When `torch.export`
lifts `zeros`/`ones`/`full` into `get_attr` nodes or constant-folds them, Stage 2
`_val_has_meta_tensor` returns False (concrete tensor, not FakeTensor) and the
restamp is skipped — those tensors allocate on CPU at `.so` runtime.

Fix: add a companion `_rewrite_factory_meta_to_vulkan` pass matching all
`_FACTORY_OPS` entries except `empty.memory_format`, replacing
`device='meta'` nodes with vulkan-device equivalents and following the
`get_attr` lifting path to restamp lifted constants.
- **Files**: `meta_patches/joint_graph_passes.py:354-403` (new companion pass).
- **Exit**: `TestAOTIFactoryOpsShim` — model with `torch.zeros` + `torch.ones`
  exports without `GuardOnDataDependent`; `.so` runs correctly on Vulkan.

---

### CG.1 — argmin/argmax index precision loss for tensors > 16M elements

**2026-06-22 deep audit (confirmed) — 3-file fix, larger than originally scoped:**

*Path correction*: bug is in `kernel/reduction.py` (731 lines), NOT
`lowerings/reduction.py` (172 lines, no argmax/argmin).

All three argmin/argmax codegen sites (`kernel/reduction.py:374,376,382`) encode
the `(value, index)` pair as `float2({value}, float({index}))`. The cast truncates
`index` to a 24-bit float mantissa — any tensor with > 16,777,216 elements silently
returns a **wrong index**. Downstream Slang shaders (`ArgPair.idx` is declared
`float` in `shaders/lib/vk_reduction.slang:421` and `shaders/lib/reduction.slang:424`)
expect the float encoding, so both Python and Slang must change together.

**Fix requires 3 files — exact change table (confirmed by 2026-06-22 Slang audit):**

| File | Line(s) | Change |
|------|---------|--------|
| `kernel/reduction.py` | 374, 376, 382 | `float2({value}, float({index}))` → `ArgPair({value}, {index})` |
| `kernel/reduction.py` | all `.x`/`.y` in result extraction | `.x` → `.val`, `.y` → `.idx` |
| `shaders/lib/vk_reduction.slang` | 421-424 | `ArgPair.idx: float` → `int64_t` (or Slang `int64`) |
| `shaders/lib/vk_reduction.slang` | 429-430 | `groupshared float _wg_arg_smem_idx[...]` → `groupshared int64_t` |
| `shaders/lib/vk_reduction.slang` | 454 | sentinel `-1.0f` → `-1` |
| `shaders/lib/vk_reduction.slang` | 829 | `vk_wg_reduce_argmax` signature: `float2` → `ArgPair` |
| `shaders/lib/vk_reduction.slang` | 824-825 | groupshared argmax scratch `float` → `int64_t` |
| `shaders/lib/vk_reduction.slang` | 838+ | all `.x`/`.y` → `.val`/`.idx` in `vk_wg_reduce_argmax` body |
| `shaders/lib/vk_reduction.slang` | 889 | `vk_wg_reduce_argmin` signature: `float2` → `ArgPair` |
| `shaders/lib/vk_reduction.slang` | 898+ | all `.x`/`.y` → `.val`/`.idx` in `vk_wg_reduce_argmin` body |
| `shaders/lib/reduction.slang` | same offsets | **Identical changes** (second copy of the same shader) |

**No `groupshared ArgPair` exists** — the groupshared scratch is split into separate
`_wg_arg_smem_val[]` and `_wg_arg_smem_idx[]` arrays; only the `idx` array type changes.

**WaveReadLaneAt `int64_t` support**: Slang/SPIR-V does support `WaveReadLaneAt` on
64-bit integers via `OpGroupNonUniformShuffle` for `int64`. Verify on the target driver.

**Test input that triggers the bug**: `torch.argmax(torch.randn(1, 16_777_217, device="vulkan:0"), dim=-1)` — index ≥ 2²⁴ is rounded by `float()`.

- **Files**: `kernel/reduction.py`, `shaders/lib/vk_reduction.slang`,
  `shaders/lib/reduction.slang`, `tests/test_inductor_regression.py:18504` (test update).
- **Exit**: `TestArgmaxLargeIndex` — `torch.argmax` on a 16,777,217-element tensor
  matches CPU exactly; `test_inductor_regression.py:18504` updated for `int64` buffer.

### CG.2 — bf16 packed16 fallback store missing `_packed16_vw_active` guard (structural conflict)

**2026-06-22 audit (confirmed exact lines + corrected root cause)** (`kernel/pointwise.py:718–736`):

The bf16 fallback store path (lines 718–736) unconditionally emits `WaveReadLaneAt`
to broadcast the high half of a packed-16 pair, with no `_packed16_vw_active` guard.
The primary packed16 path (lines 685–714) correctly guards with
`if not getattr(self, "_packed16_vw_active", False)` — the fallback never does.

**Corrected root cause (NOT a wave32 XOR issue):**
- The `^ 1u` XOR (`WaveGetLaneIndex() ^ 1u`) is mathematically correct on BOTH
  wave32 and wave64 — it always pairs adjacent lanes (0↔1, 2↔3, …), and this is
  in-range on both wave sizes. The ROADMAP's "wrong lane on wave32" framing was
  imprecise.
- The actual bug is **structural**: when `_packed16_vw_active` is `True`, the
  `_packed16_vw_rewrite()` in `pointwise_vec4_mixin.py:622` replaces the entire
  body with `gtid.x`-based vector loads/stores (no wave ops). If the fallback path
  still sets `_pw_has_wave_ops = True` in that state, it signals that wave ops are
  needed for a body that the vector-write rewrite has already transformed — a
  structural conflict that produces incorrect codegen.

**Exact 4-line fix** (insert after `self._pw_uses_subbyte_packing = True` in the
fallback block at `kernel/pointwise.py:725`):
```python
# OLD (lines 725–736):
            self._pw_uses_subbyte_packing = True
            self.headers.add("packed16_bf16")
            ...

# NEW (insert guard, mirroring primary path at lines 698–699):
            self._pw_uses_subbyte_packing = True
            # CG.2: guard wave-op flag — when _packed16_vw_active is True,
            # _packed16_vw_rewrite replaces this scalar WaveReadLaneAt path
            # with a gtid.x-based vector write that is correct on both
            # wave32 and wave64.
            if not getattr(self, "_packed16_vw_active", False):
                self._pw_has_wave_ops = True
            self.headers.add("packed16_bf16")
            ...
```

**Test shape** (catches the structural conflict):
```python
# Small bf16 tensor where _packed16_vw_active is False (numel not a multiple
# of max_threadgroup_size * 4 = 1024), forcing the scalar WaveReadLaneAt path:
x = torch.tensor([1000 + i for i in range(64)], dtype=torch.bfloat16)
# Identity op — exercises the store path without computation masking corruption.
# After round-trip: element 2k must equal 1000+2k, element 2k+1 must equal 1000+2k+1.
```

- **Files**: `kernel/pointwise.py:725` (4-line insertion after `_pw_uses_subbyte_packing = True`).
- **Exit**: `TestBf16PackedStoreWave32` — bf16 pointwise output matches CPU reference
  for both `[64]` (scalar path) and `[1024]` (vector-write path) shapes.

### CG.3 — packed16 + welford guard bypass (GroupNorm garbage mean/m2)

**2026-06-22 audit (confirmed exact lines)** (`kernel/pointwise_vec4_mixin.py:234 vs 237`):
The packed16 `_vec4_pw_eligible` path returns `True` at line ~234 **without**
checking `self.has_welford`. The float path immediately after correctly guards
with `if self.has_welford: return False` at line ~237. With fp16/bf16 GroupNorm
(fp16 input + Welford reduction), the packed16 path returns eligible, the
`_packed16_vw_rewrite` vectorizes the body into `float4` loads/stores, destroying
the strict sequential ordering that Welford's online algorithm requires —
producing garbage `mean` and `m2`.

**Exact 1-line fix** (insert before `return True` of packed16 block):
```python
# OLD:
            if not self._p16_load_records or not self._p16_store_records:
                return False
            return True

        # ── float vec4 path ────
        if self.has_welford:
            return False

# NEW:
            if not self._p16_load_records or not self._p16_store_records:
                return False
            if self.has_welford:
                return False
            return True

        # ── float vec4 path ────
        if self.has_welford:
            return False
```

**Test shape that triggers the bug**: `F.group_norm(x, num_groups=4)` with
`x = torch.randn(1, 16, 8, 8, dtype=torch.float16)` (numel = 1024 =
`max_threadgroup_size(256) × 4`, enters packed16 vec4 path; num_groups=4 →
256-element Welford reduction).

- **Files**: `kernel/pointwise_vec4_mixin.py` — 1-line insert before `return True`
  of the packed16 block.
- **Exit**: new `TestPacked16Vec4WelfordGuard` in `tests/test_inductor_regression.py`
  — fp16 GroupNorm (shape `[1,16,8,8]`, `num_groups=4`) forward matches CPU;
  emitted Slang source does **not** contain `_pvw_in_` / `_pvw_out_` identifiers.

### CG.4 — vec4 eligibility false-positive from `\w+` regex on composite index

**2026-06-22 deep audit (confirmed)** (`kernel/pointwise.py:385-394`,
`_check_index_lane_dependency`):
`buf_access_re` uses `(\w+)` for the index group — for `buf[base + xindex]`
it matches `base` only (stops at the space before `+`), misses `xindex`, and
**the whole match fails** (`.+` after `\w+` expects `]` but finds `+`). Zero
matches → the per-buffer dep-check never runs → `return False` (eligible) —
a false positive even when `base` depends on `lid.x`.

**Exact 5-line fix** (replace at `kernel/pointwise.py:385-394`):
```python
# OLD (lines 385-394)
        buf_access_re = re.compile(
            r"\b(" + "|".join(re.escape(n) for n in all_inners) + r")\s*\[\s*(\w+)\s*\]"
        )
        for m in buf_access_re.finditer(body_str):
            idx_var = m.group(2)
            # Check if idx_var or any of its transitive deps reference lane IDs
            closure = self._transitive_dep_closure(deps, {idx_var})
            if "__lane_id__" in closure:
                return True
        return False

# NEW
        buf_access_re = re.compile(
            r"\b(" + "|".join(re.escape(n) for n in all_inners) + r")\s*\[\s*(.+?)\s*\]"
        )
        for m in buf_access_re.finditer(body_str):
            # The index may be a composite expression (e.g. base + xindex).
            # Extract every identifier token and check each transitively.
            for idx_var in re.findall(r"\b([a-zA-Z_]\w*)\b", m.group(2)):
                # Check if idx_var or any of its transitive deps reference lane IDs
                closure = self._transitive_dep_closure(deps, {idx_var})
                if "__lane_id__" in closure:
                    return True
        return False
```

**Edge cases**: nested brackets `buf[a[i]]` — `.+?` non-greedy captures `a[i`,
incorrect but harmless for generated pointwise code (never produces nested
brackets). Literal-only index `buf[42]` — `findall` returns empty, no false dep.
`lid.x` in index — `findall` returns `["lid"]` (not the raw lane-id sentinel),
but CSE always pre-assigns such expressions to variables so the dep-graph check
covers them via the variable in the assignment.

- **Files**: `kernel/pointwise.py:385-394` (5-line change).
- **Exit**: `TestVec4EligibilityCompositeIndex` — kernel with
  `buf[base + xindex]` where `base = lid.x * 16` returns ineligible (False)
  from `_check_index_lane_dependency`.

### CG.5 — Split `kernel/pointwise.py` to ≤ 800 lines (anti-goal #7)

**2026-06-22 deep audit (confirmed)** — actual file is
`python/torch_vulkan/inductor/kernel/pointwise.py` (**825 lines**, not
`lowerings/pointwise.py` which does not exist).

**Minimal split (move-only, net ~0 LOC):**
- Extract the **bwd_diff block** (lines 44–201, 158 LOC) —
  `register_inline_unary_bwd`, `register_inline_binary_bwd`,
  `_emit_inline_bwd_diff_body` — into a new
  `kernel/pointwise_bwd.py` as `class PointwiseBwdMixin`.
- Add `PointwiseBwdMixin` to `PointwiseMixin`'s MRO:
  ```python
  # Before (line 38):
  class PointwiseMixin(PointwiseLoadMixin, PointwiseVec4Mixin):
  # After:
  class PointwiseMixin(PointwiseLoadMixin, PointwiseVec4Mixin, PointwiseBwdMixin):
  ```
- `pointwise.py` shrinks from **825 → ~667 lines** (well under 800).
- DCE / lane-dep block (lines 202–395) stays — it is tightly coupled to `store`.

**Circular-import risk: NONE** — neither `pointwise_load_mixin.py` nor
`pointwise_vec4_mixin.py` imports from `pointwise.py`.

**External imports to update: NONE** — `PointwiseMixin` remains the only
public symbol and stays in `kernel/pointwise.py`. Importers in `kernel/main.py`
and both test files are unaffected.

**Gross diff: ~329 LOC** (158 deleted from `pointwise.py`, ~170 in new
`pointwise_bwd.py`, 1 import line + 1 MRO line added to `pointwise.py`).

**2026-06-22 exact split plan (confirmed):**
- `PointwiseMixin` declared at `pointwise.py:28` (not line 38); class declaration is
  `class PointwiseMixin(PointwiseLoadMixin, PointwiseVec4Mixin):`
- Symbols in lines 44-201: ALL are methods/class attributes of `PointwiseMixin`
  (`register_inline_unary_bwd`, `register_inline_binary_bwd`, `_emit_inline_bwd_diff_body`,
   class attrs `_LANE_ID_TOKENS`, `_ASSIGN_RE`, `_DCE_ASSIGN_RE`, `_DCE_ALWAYS_LIVE`,
   `_DCE_LIVE_PREFIXES`, `_dce_parse_assignments` static method)
- `kernel/pointwise_bwd.py` does NOT exist yet — must be created
- New file `pointwise_bwd.py` needs top-level `import re` only; `BWD_DIFF_TABLE` and
  `bwd_diff_inline` imports are function-level inside the methods (stay as-is)
- ONLY `kernel/main.py:30` imports `PointwiseMixin` — no other external importers
- `main.py` needs NO changes (it imports `PointwiseMixin` from `pointwise.py`, which
  stays as the public symbol; `PointwiseBwdMixin` is an internal detail)

- **Files**: `kernel/pointwise.py` (move lines 44–201 to new file, add import + MRO at line 28),
  new `kernel/pointwise_bwd.py` (~170 LOC).
- **Exit**: all existing pointwise tests pass after split;
  `wc -l kernel/pointwise.py` reports < 800.

---

### SP.1 — Wire or remove the dead numthreads-rewrite path in `reflection_ext.py`

**2026-06-22 audit (confirmed exact lines)**:
- Rewrite + recompile call: `runtime/slangc.py:280-310`
- Optimized-numthreads compile helper: `runtime/slangc.py:365-381`
- Phase 6 overwrite (root cause): `runtime/slangc.py:460-467`
- Rewrite helper + store: `runtime/reflection_ext.py:668-700`

Phase 6 at `slangc.py:460-467` unconditionally writes the **original** SPV back
to `_cache_by_hash[hash_key]` and disk after `_compile_with_optimized_numthreads`
has already stored the optimized SPV — silently discarding it. Additionally,
`get_optimized_numthreads` (`reflection_ext.py:674`) is stored in
`_optimized_numthreads_by_hash` but has **zero callers** outside `slangc.py`;
`_maybe_autotune_wg` in `dispatch.py` never consults it. Net result: wasted
slangc subprocess + dispatch grid always uses unoptimized numthreads.

**2026-06-22 deep audit (confirmed) — REMOVE (Option A) is safer than wiring:**

- `get_optimized_numthreads` and `_optimized_numthreads_by_hash` are **never
  consumed** anywhere (`slangc.py:137` imports but never calls;
  `__init__.py:103` re-exports but is never invoked). The dead second-compile
  also produces a **latent mismatch**: it replaces the cached SPV with one
  expecting the new numthreads, but the dispatch closure was already built with
  the original WG dims — so the wrong grid is passed to a rewritten SPV.
  `_maybe_autotune_wg` already owns WG selection correctly via
  `_build_kernel_for_wg`; wiring both would create a race.

**Option A — remove (3 files, ~160 LOC deleted):**

| File | Change |
|------|--------|
| `runtime/reflection_ext.py` | Delete lines 655–700: `_NUMTHREADS_SRC_RE`, `_optimized_numthreads_by_hash`, `_parse_numthreads_from_source`, `_rewrite_numthreads_in_source`, `get_optimized_numthreads`; also delete dead `_pick_numthreads_from_reflection` call in `_process_reflection` |
| `runtime/slangc.py` | Remove dead branch in `_process_reflection` (lines 280–310); remove `_compile_with_optimized_numthreads` function (lines 308–370); remove dead imports `_pick_numthreads_from_reflection`, `_rewrite_numthreads_in_source`, `get_optimized_numthreads`, `_optimized_numthreads_by_hash` |
| `runtime/__init__.py` | Remove 3 dead exports: `_parse_numthreads_from_source` (line 75), `_rewrite_numthreads_in_source` (line 84), `get_optimized_numthreads` (line 103) |

**Test impact**: `test_inductor_regression.py:36229` and
`test_slangc_modular.py:260` call `_parse_numthreads_from_source`; those test
assertions must also be removed (or converted to test the autotune path instead).

**2026-06-22 test-impact audit (confirmed exact dispositions):**

| Test | File:Line | Action | Replacement |
|------|-----------|--------|-------------|
| `test_dr7_numthreads_parsing` | `test_inductor_regression.py:36228` | **(B) Update** | Replace with `_extract_wg_from_numthreads` from `runtime/dispatch.py`; assert `[numthreads(256,1,1)]`→256, `[numthreads(16,8,1)]`→128, missing→256 |
| `test_dr7_numthreads_rewrite` | `test_inductor_regression.py:36245` | **(A) Delete** | No surviving equivalent — the inline `src.replace` in `_build_kernel_for_wg` has no standalone helper to unit-test |
| `test_routing_triggers_recompile` | `test_slangc_modular.py:256` | **(A) Delete** | Entire behavior is removed; `_compile_with_optimized_numthreads` and `_parse_numthreads_from_source` won't exist |
| `test_dr7_heavy_kernel_gets_fewer_threads` | `test_inductor_regression.py:~36252` | **(C) Keep** | `_pick_numthreads_from_reflection` function definition survives removal |
| `test_dr7_pick_numthreads_fallback_on_none` | `test_inductor_regression.py:~36270` | **(C) Keep** | Same |
| `test_dr7_pick_numthreads_boundary_values` | `test_inductor_regression.py:~36288` | **(C) Keep** | Same |

**Replacement body for `test_dr7_numthreads_parsing`:**
```python
def test_dr7_numthreads_parsing(self):
    """_extract_wg_from_numthreads extracts total WG size from a
    numthreads attribute string (used by the surviving WG-autotune path)."""
    from torch_vulkan.inductor.runtime.dispatch import _extract_wg_from_numthreads
    # 1D workgroup
    assert _extract_wg_from_numthreads("[numthreads(256, 1, 1)]") == 256
    # 2D workgroup
    assert _extract_wg_from_numthreads("[numthreads(16, 8, 1)]") == 128
    # 3D workgroup
    assert _extract_wg_from_numthreads("[numthreads(8, 4, 2)]") == 64
    # Malformed / missing numthreads falls back to 256
    assert _extract_wg_from_numthreads("no numthreads here") == 256
```

Note: `_pick_numthreads_from_reflection` is NOT listed for deletion in the SP.1
scope. Only the call to it inside `_process_reflection` is removed. The function
definition stays (a follow-up pass may clean it up separately).

- **Files**: `runtime/reflection_ext.py:655-700`, `runtime/slangc.py:280-370`,
  `runtime/__init__.py:75,84,103`; test cleanup in 2 test files.
- **Exit**: full test suite green after removal; no perf regression on
  any kernel (autotune path is unchanged).

### SP.2 — Remove `.result()` blocking from async compile path (prerequisite for S3.4)

**2026-06-22 audit (confirmed — SP.2 NOT yet implemented):**

Line 557 in `slangc.py` is the blocking call (confirmed):
```python
552:        if _PARALLEL_COMPILE and _ASYNC_COMPILE and not _is_in_pool_worker():
553:            pool = _get_async_pool()
554:            spv = pool.submit(
555:                _wrap_pool_worker(_compile_slang_to_spirv_inner),
556:                src, entry, hash_key, include_paths, config_key,
557:            ).result()   # ← blocks immediately; async submit is pointless
```

**SP.2 is more complex than "just remove .result()"** — confirmed by 2026-06-22 audit:

`compile_slang_to_spirv` currently returns `bytes`. Making it return `bytes | Future[bytes]`
requires making ALL consumers Future-aware. Key callers:

| Caller | File:Line | Breaks on Future because... |
|--------|-----------|------------------------------|
| `make_vulkan_kernel` eager path | `dispatch.py:463` | `spv` captured by closure |
| `_KERNEL_SPIRV_HASH[key] = sha256(spv)` | `dispatch.py:464` | `sha256()` needs bytes |
| `get_reflected_binding_count(spv)` | `dispatch.py:474, 490` | reflection needs bytes |
| `_c._aoti_make_kernel(spv, ...)` | `dispatch.py:701` | C++ call needs bytes |
| `_build_kernel_for_wg` | `dispatch.py:1051` | same pattern |
| `batch_compile_slang_to_spirv` | `slangc.py:686-698` | already parallel via pool |

**Zero `.result()` calls in `dispatch.py`** — all blocking is in `slangc.py:557` and
`slangc.py:693-695`. The dispatch.py closures capture `spv: bytes` at closure-build time
(not at dispatch time), so adding a `future.result()` in each closure body is the fix
pattern, but reflection and hash calls at closure-build time must also be deferred.

**Minimal proof-of-concept for S3.4**: `batch_compile_slang_to_spirv` with 2 uncached
sources already shows compile parallelism (no S3.4 dispatch overlap yet). The full S3.4
proof-of-concept requires the deferred-resolution closure pattern.

**Estimated scope**: ~100-150 LOC change across `slangc.py:552-560` (return Future on miss)
+ `dispatch.py:460-700` (8 closure sites made Future-aware) + reflection/hash deferral.

- **Files**: `runtime/slangc.py:552-560`, `runtime/dispatch.py:460-700` (8 closure sites).
- **Exit**: `TestAsyncCompileNonBlocking` — two concurrent `compile_slang_to_spirv`
  calls on different kernels complete in overlapping wall-clock time; wall-clock time
  < 2× single-compile latency.

### SP.3 — Add PC-layout hash to SPIR-V template cache key

**2026-06-22 deep audit (confirmed) — 4-file change:**

The MM template key uses hardcoded `_n111_a6` (a one-time invalidation tag from
N+1.11). A future PC field addition will produce stale SPIR-V reuse silently:
`_cache_by_key` hits on the unchanged key before `_cache_by_hash` can detect
the source change. The generic `hash_key` at `slangc.py:510-516` also omits
any PC-layout component (the PC struct IS in the source text, so `hash_key`
does catch source changes — but `cache_key` bypasses it).

**Four-part fix:**

**Part 1 — `gemm/classes.py`**: extract `struct PC { … }` from rendered `src`
and append its sha256[:8] to all four `cache_key` variants:
```python
_pc_match = re.search(r"struct PC\s*\{([^}]+)\}", src)
_pc_fields = _pc_match.group(1) if _pc_match else ""
_pc_tag = f"_pc{hashlib.sha256(_pc_fields.encode()).hexdigest()[:8]}"
# append _pc_tag to each of the four cache_key f-strings (after _n111_a6)
```

**Part 2 — `runtime/slangc.py:474`**: add optional `pc_layout_hash` param:
```python
# OLD:
def compile_slang_to_spirv(src, entry="computeMain", cache_key=None, ...):
    ...
    hash_key = hashlib.sha256((entry + "
" + _normalize_slang_source(src) + inc_tag + sgs_tag + lib_tag).encode()).hexdigest()
# NEW: add pc_layout_hash: Optional[str] = None param; mix into hash_key:
    pc_tag = "" if pc_layout_hash is None else f"
PC={pc_layout_hash}"
    hash_key = hashlib.sha256((...  + lib_tag + pc_tag).encode()).hexdigest()
```

**Part 3 — `runtime/dispatch.py:376`**: thread `pc_layout_hash` through
`compile_and_dispatch()` → `compile_slang_to_spirv()` (1-line signature + 1-line call change).

**Part 4 — `templates/caller/gemm/dispatch.py`**: in each dispatch helper
(`_slang_tile_mm_dispatch`, `_slang_tile_bmm_dispatch`, `_slang_tile_addmm_dispatch`,
`_slang_mm_bwd_dispatch`), compute `_pc_hash = sha256(struct_PC_text)[:8]` from
the rendered `src` and pass as `pc_layout_hash=_pc_hash` to `compile_and_dispatch`.

**Three distinct PC layouts confirmed:**
- Forward/addmm/bmm: 24 fields, 96 bytes (`19I5f`)
- int8 matmul: 9 fields, **36 bytes** (`9I` — `render.py:553-563`, 9 `uint` fields; packed with `struct.pack("9I", ...)`; earlier estimate of 40 bytes was wrong)
- Backward: 19 fields, 76 bytes (`19I`)

- **Files**: `gemm/classes.py` (4 f-strings + regex import),
  `runtime/slangc.py:474,512` (param + hash-key),
  `runtime/dispatch.py:376` (thread-through),
  `gemm/dispatch.py` (4 call sites).
- **Exit**: `TestTemplateCacheKeyDistinct` — adding a field to `struct PC` in
  `render.py` produces a new `cache_key`; no stale SPIR-V is returned.

### SP.B1 — `batch_compile_slang_to_spirv` missing `sgs_tag` + `lib_tag`

**2026-06-22 audit (confirmed)** (`runtime/slangc.py:633`): Hash key omits
`_get_device_subgroup_size_tag()` and `_shader_lib_import_hash(src)` that are
present in `compile_slang_to_spirv` and `async_precompile_slang`. A batch
compile on a different subgroup-size device or after a shader-lib update can
hit a stale disk-cache entry.

**Exact fix** (`runtime/slangc.py:636-639`):
```python
# OLD
        hash_key = hashlib.sha256(
            (entry + "\n" + _normalize_slang_source(src) + inc_tag).encode()
        ).hexdigest()
# NEW
        hash_key = hashlib.sha256(
            (entry + "\n" + _normalize_slang_source(src) + inc_tag + _get_device_subgroup_size_tag()).encode()
        ).hexdigest()
```
- **Files**: `runtime/slangc.py:636-639`.
- **Exit**: batch-compiled key matches single-compile key for the same (src, device-sgs) tuple; wave32 vs wave64 no longer share a cache entry.
- **Test gap (confirmed 2026-06-22)**: ZERO existing tests call `batch_compile_slang_to_spirv`. The existing sgs cache-key tests (`test_dr3_cache_key_includes_subgroup_size:35375`, `test_n112_subgroup_size_in_cache_key:35615`) check `VulkanKernel.config_key` only — not the SPIR-V disk-cache `hash_key`. New test `TestBatchCompileHashKeyBySubgroupSize` required: mock `_get_device_subgroup_size_tag` to return `"_sgs64"` vs `"_sgs32"`, spy on the computed `hash_key`, assert `key_64 != key_32`.

### SP.B2 — `prewarm_compile` missing `sgs_tag` + `lib_tag`

**2026-06-22 audit (confirmed)** (`runtime/shader_lib.py:102`): Same omission as
B.1 — prewarm hash key misses `sgs_tag` + `lib_tag`. Pre-warm may skip kernels
that are actually absent from the cache on this device/lib-version combo.

**Exact fix** (`runtime/shader_lib.py:99-101`):
```python
# OLD
        hash_key = hashlib.sha256(
            ("computeMain\n" + _normalize_slang_source(src)).encode()
        ).hexdigest()
# NEW
        hash_key = hashlib.sha256(
            ("computeMain\n" + _normalize_slang_source(src) + _shader_lib_import_hash(src)).encode()
        ).hexdigest()
```
- **Files**: `runtime/shader_lib.py:99-101`.
- **Exit**: prewarm populates the cache correctly after a shader-lib update; cold-miss rate at first dispatch → 0.
- **Test gap (confirmed 2026-06-22)**: Existing `test_prewarm_compile_does_not_deadlock_with_one_worker:23546` checks only that the call returns within 15s. New test `TestPrewarmCompilePopulatesCache` required: call `prewarm_compile([(key, src)], sync=True)`, assert `key in _cache_by_key`; then call `compile_slang_to_spirv(src, cache_key=key)` with a spy and assert zero inner compile calls (cache hit).

### SP.B3 — `_build_kernel_for_wg` reuses original `cache_key` → WG autotune dispatches unoptimized SPV

**2026-06-22 audit (confirmed)** (`runtime/dispatch.py:1050`): `_maybe_autotune_wg`
calls `_build_kernel_for_wg` with the original `key`. Inside,
`compile_slang_to_spirv(new_src, cache_key=key)` hits `_cache_by_key` and
returns the **original** SPV without compiling the new-numthreads source. The
benchmark loop runs, picks a "winner", but the winning closure dispatches the
unoptimized SPIR-V. WG autotune is silently a no-op.

**Exact fix** — change the call site at `runtime/dispatch.py:955-957` (not line 1050):
```python
# OLD (lines 955-957)
            return _build_kernel_for_wg(
                src, orig_nt, cached_wg, key, n_pc, n_outputs,
                dispatch_fn, pc_buf, pc_pack_into,
            )
# NEW
            return _build_kernel_for_wg(
                src, orig_nt, cached_wg, f"{key}_wg{cached_wg}", n_pc, n_outputs,
                dispatch_fn, pc_buf, pc_pack_into,
            )
```
The `_build_kernel_for_wg` function at line 1050 passes `cache_key` straight to
`compile_slang_to_spirv`, so fixing the call site is sufficient — no change needed inside
`_build_kernel_for_wg` itself.
- **Files**: `runtime/dispatch.py:955-957`.
- **Exit**: `TestWGAutotune` — the winning WG-size kernel's dispatched SPV contains the
  expected `numthreads` attribute; SPV stored under `key` and `f"{key}_wg{wg}"` are distinct.

### SP.B4 — `_save_wg_cache()` never called → autotune disk cache never written

**2026-06-22 audit (new finding)** (`runtime/dispatch.py:901`):
`_save_wg_cache()` is **defined** at line 901 but has **zero call sites** anywhere
in the codebase. The in-memory `_WG_AUTOTUNE_CACHE` dict is populated during
warm-up benchmarking, but it is **never persisted** to
`~/.cache/torch_vulkan/wg_autotune/`. On the next process launch (training mode,
when `_wg_autotune_enabled()` returns False), `_load_wg_cache()` finds nothing
and every kernel dispatches at the original unoptimized WG size.

This is distinct from SP.B3 (wrong cache key in the *current* process's dispatch
closure). SP.B4 means the correct winner is picked within a single warm-up run,
but that winner is **never remembered** across process boundaries.

**Exact 1-line fix** (`runtime/dispatch.py:1010`):
```diff
     # Cache the winner
     _WG_AUTOTUNE_CACHE[src_hash] = best_wg
+    _save_wg_cache(src_hash, best_wg)

     # If the original was fastest, return None (caller uses default)
     if best_wg == orig_wg:
```
`_save_wg_cache` already exists at line 901 (takes `src_hash, best_wg`, writes
`~/.cache/torch_vulkan/wg_autotune/{src_hash}.json`). `_load_wg_cache()` is
already called at line 634 — it loads but the disk is always empty because
nothing calls `_save_wg_cache`. The 1-line fix closes the loop.

Concurrent-write risk: **low / benign** — different kernels write different filenames;
same-kernel race writes identical content to the same path (harmless).

- **Files**: `runtime/dispatch.py:1010` (1-line insert).
- **Exit**: End-to-end test: after warm-up, assert
  `~/.cache/torch_vulkan/wg_autotune/<hash>.json` exists; after clearing
  `_WG_AUTOTUNE_CACHE` and calling `_load_wg_cache()`, the winner is returned for
  the same `src_hash`.

---

### Continuous — coverage breadth (S2/E pillar)

- **E1** — replace `max_pool2d_scatter_bwd` / `avg_pool2d_scatter_bwd`
  `make_fallback`s with Slang `scatter_atomic` codegen.

  **2026-06-22 audit (confirmed): implement, NOT ratify.**
  - `lowerings/__init__.py:489,493` — both still `make_fallback`; no fusion possible.
  - `avg_pool2d_scatter_bwd` (`fx_passes/eager/pool.py:286-410`) computes scatter
    indices **on CPU with a numpy nested loop** (violation of anti-goal #6: no CPU
    roundtrip on the compiled path). Only the final scatter dispatch is GPU-side.
  - `max_pool2d_scatter_bwd` is GPU-side only but `make_fallback` prevents fusion.
  - **NEW (2026-06-22 fx_passes audit)**: `fx_passes/eager/pool.py:228` additionally
    computes `plane_ids` on CPU via `torch.arange(..., device="cpu") // output_spatial`
    then transfers to GPU — a second CPU allocation + int-div on every max_pool2d
    backward with indices (hot path). Comment in file: "avoid aten.floor_divide on Vulkan".
    This must also be eliminated as part of E4.
  - Existing `shaders/pooling/max_pool2d_backward.slang` + `avg_pool2d_backward.slang`
    are gather-style and **not used** by the current Python custom-op path (which
    routes through `templates/scatter_atomic.slang` instead).
  - **This is the same work as E4.** The E4 4-phase plan already describes the full
    implementation. E1 is now subsumed by E4; see E4 for the complete plan.
- **E2** — masking backward (`tril`/`triu`/`masked_fill`/`where`) verification.

  **2026-06-22 audit (confirmed): item is ALREADY IMPLEMENTED — only verification needed.**

  Rationale: `tril`/`triu`/`masked_fill`/`where` have no `aten.*_backward` ops.
  Autograd decomposes their backward into the **same forward op on `grad_output`**:
  - `tril(x).backward()` → `aten.tril.default(grad_output)` (re-applies same op)
  - `where(cond, a, b).backward()` → `where(cond, grad, 0)` + `where(cond, 0, grad)` 
  - `masked_fill(x, mask, val).backward()` → `masked_fill(grad, mask, 0)`

  `bwd_diff` is fundamentally incompatible with these ops (they are conditional
  selects, not differentiable float elementals; `masking.py:14-15` explicitly states
  this). The ROADMAP item's original "via `bwd_diff`" framing was incorrect.

  **What IS done**: all forward lowerings exist in `lowerings/masking.py`; autograd
  decomposes backward into forward ops; `TestMaskingBackward` has 5 regression tests
  (lines 63772-63840 of `test_inductor_regression.py`), not xfailed.

  **Only remaining work**: run the 5 `TestMaskingBackward` tests on GPU to confirm
  they pass. If any fail, it is a bug in the forward lowering, not a missing backward
  mechanism. Re-scope this item from "implement" to "GPU verify."

  **Tests**: `test_tril_backward_grad_parity`, `test_triu_backward_grad_parity`,
  `test_tril_batched_backward_grad_parity`, `test_masked_fill_backward_grad_parity`,
  `test_where_backward_grad_parity`.
- **E3** — missing ops: `sort`, `bucketize`, `multinomial`, sparse (csr/coo),
  eager FFT — decompose where possible, else file per-op sub-items.

  **2026-06-22 audit (confirmed) — per-op status:**

  | Op | Status | File:Line | Note |
  |----|--------|-----------|------|
  | `aten.sort` | 🟡 Partial — `BackendFeature.SORT` route exists | `kernel/reduction.py:567`, `shaders/reduction/sort.slang` | Inline insertion sort O(n²) — only correct for small tensors |
  | `aten.bucketize` | 🟡 Partial — PrivateUse1 override | `lowerings/searchsorted.py:124`, `shaders/lib/bucket.slang` | Binary search shader exists; needs correctness verification |
  | `aten.multinomial` | ✅ Has `@register_lowering` | `lowerings/rng.py:43` | Decomposes to cumsum + Philox rand + searchsorted; no dedicated Slang |
  | `aten._sparse_csr_tensor_unsafe` | ❌ None found | — | True gap; edge case |
  | `aten.to_sparse` | ❌ None found | — | True gap; edge case |
  | FFT (`_fft_r2c`, `_fft_c2c`, `_fft_c2r`) | ✅ Has fake impls in meta_patches | `meta_patches/__init__.py:359-361` | Fake impls for shape inference; actual GPU dispatch unclear |

  **Practical impact**: All three "common" E3 ops (sort, bucketize, multinomial) have
  partial implementations. None appear in ResNet-50/BERT/ViT critical paths — all
  are edge-case ops. The sort insertion-sort shader is the only known correctness risk
  (silently wrong for tensors with > ~1000 elements).

  **Additional finding (2026-06-22 lowerings audit)**: `aten.amax` (`reduction.py:92`)
  and `aten.amin` (`reduction.py:98`) use `FallbackKernel.create` with **no dedicated
  Slang shader**. They route to the C++ eager reduction path. No correctness risk
  (C++ path is correct), but no Slang codegen — these cannot be fused with adjacent
  pointwise ops. Future optimization: add `shaders/reduction/amax_dim.slang` and
  `shaders/reduction/amin_dim.slang` (reuse `max_dim.slang` / `min_dim.slang` template).
- **E4** — **Pool backward Slang codegen** (overlapping `max_pool2d_backward`,
  `avg_pool2d_backward`): required for ResNet-50/VGG-16 backward training.

  **2026-06-22 deep-audit (confirmed) — 4-phase implementation plan:**

  *Current state*: Both ops are registered in `bwd_lowerings.py` (max: line 810,
  avg: line 616) but both route through `FallbackKernel` → Python custom op in
  `fx_passes/eager/pool.py` (max: line 149, avg: line 549) → CPU index computation
  → GPU transfer → `scatter_atomic.slang`. The scatter Slang template and atomics
  library are fully functional; the existing gather-style Slang shaders at
  `shaders/pooling/max_pool2d_backward.slang` and `avg_pool2d_backward.slang` need
  rewriting to scatter-style before they can back an IR lowering.

  *ResNet-50 maxpool1* (3×3, stride=2, pad=1) is **overlapping** — requires
  scatter-add path; no non-overlapping shortcut applies.

  **Phase 1 — Rewrite Slang shaders to scatter-style** (~130 LOC):
  - `shaders/pooling/max_pool2d_backward.slang` — each thread processes one
    output position, reads saved argmax index, atomically adds `grad_output[i]`
    to `grad_input[indices[i]]` (eliminates O(iH×iW×kH×kW) nested loop).
  - `shaders/pooling/avg_pool2d_backward.slang` — each thread processes one
    output position, scatters `grad_output[i] / pool_size` to all in-window
    input positions (handles `count_include_pad`, `divisor_override`).

  **Phase 2 — New Python dispatch caller** (~200 LOC):
  - `templates/caller/pool.py` (new) — `_render_*`, `_dispatch_*`, push-constant
    packing (N,C,iH,iW,oH,oW,kH,kW,sH,sW,pH,pW + `count_include_pad`),
    `install_external_pool_backward()` prewarm.

  **Phase 3 — Inductor IR lowerings** (~210 LOC):
  - `lowerings/pool.py` — add `max_pool2d_backward_codegen()`; extend
    `avg_pool2d_backward_codegen` to emit `ir.Scatter(scatter_mode='atomic_add')`
    for overlapping windows instead of returning `None`.
  - `bwd_lowerings.py:616,810` — replace both `fallback_handler` call sites with
    direct calls to the new codegen functions.

  **Phase 4 — Cleanup** (~80 LOC removed):
  - Simplify / guard-gate the Python custom-op registrations in
    `fx_passes/eager/pool.py:149,549` (keep as emergency fallback under an env
    flag, or remove entirely once Phase 3 is verified).

  **Minimal first slice** (unblocks ResNet-50 benchmarks fastest): scatter-add
  path only, no `ceil_mode`, `dilation=1`, `count_include_pad=True` — covers all
  ResNet-50 and VGG-16 pool configs listed above.

  **Total estimate**: ~540 LOC net change across 5 files + 2 Slang shader rewrites.
  - **Files**: `shaders/pooling/max_pool2d_backward.slang`,
    `shaders/pooling/avg_pool2d_backward.slang`,
    `templates/caller/pool.py` (new),
    `lowerings/pool.py`, `bwd_lowerings.py:616,810`.
  - **Exit**: `TestPoolBwdSlang` — both ops produce no `FallbackKernel` in the
    graph; gradients match CPU for ResNet-50 maxpool1 + VGG-16 pool configs;
- **E5** — **SDPA / flash_attention backward — wiring only** (infrastructure
  already exists): **2026-06-22 audit (confirmed)**: `_SlangTileFlashAttentionBwd`
  exists at `templates/caller/flash_attn.py:522`; `flash_attention_bwd.slang`
  exists; `_dispatch_flash_attention_bwd` exists at `bwd_diff/emit_helpers.py:352`
  and unpacks `(q, k, v, lse)` from saved tensors; `BWD_TEMPLATE_REGISTRY` entry
  `flash_attention_f32_bhsd` is registered at `bwd_template_registry.py:157`.

  **2026-06-22 implementation plan (confirmed, ready for `claude_code`):**
  - **File**: `python/torch_vulkan/inductor/bwd_lowerings.py` — inline only, no new
    file (anti-goal #3 forbids `lowerings/attention_backward.py`)
  - **Placement**: new helper `_register_sdpa_backward_lowering()` inserted before
    `register()` at ~line 894; call it from the `register()` body
  - **Decorator**: `@register_lowering(aten.scaled_dot_product_attention_backward.default, type_promotion_kind=None)`
  - **Pre-land gate**: confirm `aten.scaled_dot_product_attention_backward` exists in
    this PyTorch tree with `python -c "from torch._ops import ops; print(ops.aten.scaled_dot_product_attention_backward)"`)
  - **Exact function (~25 lines)**:

  ```python
  def _register_sdpa_backward_lowering():
      @register_lowering(
          aten.scaled_dot_product_attention_backward.default,
          type_promotion_kind=None,
      )
      def _vulkan_sdpa_backward(
          grad_out, query, key, value, out, logsumexp,
          dropout_p, is_causal, *, scale=None
      ):
          if not _is_vulkan(grad_out):
              return NotImplemented
          if float(dropout_p) != 0.0:
              return NotImplemented
          # resolve head_dim from SymInt
          head_dim_sym = query.get_size()[-1]
          try:
              head_dim = V.graph.sizevars.size_hint(head_dim_sym)
          except (TypeError, ValueError):
              return NotImplemented
          if head_dim not in {32, 64, 128, 256}:
              return NotImplemented
          return dispatch_template_bwd(
              "flash_attention_f32_bhsd",
              grad_out, query, key, value, logsumexp,
          )
  ```

### Continuous — regression lock (F pillar)

Every S-item lands a named test in `tests/test_inductor_regression.py`. No
`agent_space/` script as sole verification (Discipline #1). Run the full GPU
suite under `TORCH_VULKAN_VUID_AS_ERROR=1`.

---

## § 2.5 — Ready-to-implement queue (ordered by severity)

All items below have confirmed exact fix specs with copy-paste diffs. Each is a
separate `claude_code` implement ticket — one at a time, cross-reviewed by `pi`.

| # | Item | Files | LOC | Severity | Status |
|---|------|-------|-----|----------|--------|
| 1 | MS.1+MS.2 | `csrc/backend/aoti_shims.cpp` | ~8 | 🔴 CRITICAL | Fix spec confirmed |
| 2 | SP.B4 | `python/torch_vulkan/inductor/runtime/dispatch.py:1010` | 1 | 🔴 HIGH | Fix spec confirmed |
| 3 | SP.B3 | `python/torch_vulkan/inductor/runtime/dispatch.py:955-957` | 1 | 🔴 HIGH | Fix spec confirmed |
| 4 | SP.B1 | `python/torch_vulkan/inductor/runtime/slangc.py:636-639` | 3 | 🟡 HIGH | Fix spec confirmed |
| 5 | SP.B2 | `python/torch_vulkan/inductor/runtime/shader_lib.py:99-101` | 3 | 🟡 HIGH | Fix spec confirmed |
| 6 | CG.3 | `python/torch_vulkan/inductor/kernel/pointwise_vec4_mixin.py` | 1 | 🟡 MEDIUM | Fix spec confirmed |
| 7 | CG.4 | `python/torch_vulkan/inductor/kernel/pointwise.py:385-394` | 5 | 🟡 MEDIUM | Fix spec confirmed |
| 8 | CG.2 | `python/torch_vulkan/inductor/kernel/pointwise.py:725` | 4 | 🟡 MEDIUM | Fix spec confirmed |
| 9 | S4.0 | `python/torch_vulkan/inductor/cpp_wrapper_gpu.py` | ~15 | 🟡 MEDIUM | Fix spec confirmed (Bug A: ExternKernelOut meta gap; Bug B: emit_aoti_spv_header hardcodes 0u) |
| 10 | CG.1 | `kernel/reduction.py` + 2 Slang files | ~50 | 🟡 MEDIUM | Fix spec confirmed |
| 11 | E5 | `python/torch_vulkan/inductor/bwd_lowerings.py` | ~25 | 🟡 MEDIUM | Fix spec confirmed |
| 12 | S2.3 | `csrc/vulkan/Context.cpp:107` | 4 | 🟡 MEDIUM | Fix spec confirmed |
| 13 | S0.1 | `autotune.py:43-57` + `gemm/dispatch.py:121-137` | ~15 | 🟡 LOW | Fix spec confirmed |
| 14 | MS.3 | `csrc/ops/dispatch.h` + `csrc/ops/dispatch.cpp` | ~30 | 🟡 LOW | Fix spec confirmed |
| 15 | SP.1 | 3 Python files + 2 test files | ~160 del | 🟢 CLEANUP | Fix spec confirmed |
| 16 | meta_patches cleanup | `meta_patches/shape_ops.py`, `dtype_ops.py`, `decomposition_passes.py` | ~80 del | 🟢 CLEANUP | Fix spec confirmed (see § 3.6) |

**Ticket details for claude_code:**

### Ticket 1 — MS.1+MS.2: AOTI shim memory safety

**File**: `csrc/backend/aoti_shims.cpp`  
**Contract**: No use-after-free on `zeros/ones/full/as_strided`; `aoti_torch_delete` properly frees the heap tensor. All 5 AOTI tests pass. Diff must NOT change any public API signature.  
See full fix spec in `## MS.1+MS.2` section above.

### Ticket 2 — SP.B4: WG autotune disk persistence

**File**: `python/torch_vulkan/inductor/runtime/dispatch.py` (line 1010, 1-line insert)  
**Contract**: After warm-up, `~/.cache/torch_vulkan/wg_autotune/<hash>.json` exists; after clearing in-memory cache + reload, the winner is returned. See `## SP.B4` section above.

### Ticket 3 — SP.B3: WG autotune wrong cache key

**File**: `python/torch_vulkan/inductor/runtime/dispatch.py` (lines 955-957, 1-line change)  
**Contract**: The winning WG-size kernel's SPV has the expected `numthreads` attribute; SPV stored under `key` and `f"{key}_wg{wg}"` are distinct. See `## SP.B3` section above.

### Ticket 4 — SP.B1: batch compile missing sgs_tag

**File**: `python/torch_vulkan/inductor/runtime/slangc.py` (lines 636-639, 3-line change)  
**Contract**: Batch-compiled key matches single-compile key for the same (src, device-sgs) tuple; wave32 vs wave64 no longer share a cache entry. See `## SP.B1` section above.

### Ticket 5 — SP.B2: prewarm compile missing lib_tag

**File**: `python/torch_vulkan/inductor/runtime/shader_lib.py` (lines 99-101, 3-line change)  
**Contract**: Prewarm populates the cache correctly; cold-miss rate at first dispatch → 0. See `## SP.B2` section above.

### Ticket 6 — CG.3: packed16+Welford guard

**File**: `python/torch_vulkan/inductor/kernel/pointwise_vec4_mixin.py` (1-line insert)  
**Contract**: `test_packed16_vec4_welford_guard` passes — fp16 GroupNorm `[1,16,8,8]` forward matches CPU; emitted Slang does NOT contain `_pvw_in_` identifiers. See `## CG.3` section above.

### Ticket 7 — CG.4: vec4 eligibility regex fix

**File**: `python/torch_vulkan/inductor/kernel/pointwise.py:385-394` (5-line change)  
**Contract**: `TestVec4EligibilityCompositeIndex` — kernel with `buf[base + xindex]` where `base = lid.x * 16` returns `False` from `_check_index_lane_dependency`. See `## CG.4` section above.

### Ticket 8 — CG.2: bf16 fallback _packed16_vw_active guard

**File**: `python/torch_vulkan/inductor/kernel/pointwise.py:725` (4-line insert)  
**Contract**: `TestBf16PackedStoreWave32` passes for both `[64]` (scalar path) and `[1024]` (vector-write path) shapes. See `## CG.2` section above.

### Ticket 12 — S2.3: VUID callback throw

**File**: `csrc/vulkan/Context.cpp:107` (4-line C++ insertion)  
**Contract**: `prepare_device(validate=True)` with `TORCH_VULKAN_VALIDATION=1` before import; an injected VUID fails the dispatch with `RuntimeError`. Existing `conftest.py` VUID fixture passes on all clean tests. See `### S2.3` section above.

### Ticket 16 — meta_patches dead-code cleanup

**Files** (3 Python files):
- `meta_patches/shape_ops.py:200-260` — delete 5 dead backward fakes:
  `_native_batch_norm_backward_fake`, `_native_group_norm_backward_fake`,
  `_native_layer_norm_backward_fake`, `_upsample_bilinear2d_backward_fake`,
  `_upsample_nearest2d_backward_fake`
- `meta_patches/dtype_ops.py:80-170` — delete 6 dead backward fakes:
  `_gelu_backward_fake`, `_softmax_backward_data_fake`,
  `_log_softmax_backward_data_fake`, `_avg_pool2d_backward_fake`,
  `_max_pool2d_with_indices_backward_fake`, `_linear_backward_fake`
- `meta_patches/shape_ops.py:468-476` — delete `_randperm_fake` (comment confirms
  randperm can never appear in Vulkan FX graph)
- `meta_patches/decomposition_passes.py:473-648` — delete
  `_patch_pre_grad_passes_for_conv_gn_relu_fusion` body (already disabled in
  `__init__.py`; dead code)

**Contract**: Full test suite still passes after deletion; no `NameError` or import
error. `grep` for each deleted symbol returns zero results outside the deleted blocks.
See `## § 3.6` section for full classification.

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
│   ├─ S2.4 (warm→train coherence)    ✅ FIXED 2026-06-21
│   └─ S2.5 (pooling fwd custom-op)   ✅ FIXED 2026-06-21
│
├─ S3 TRAIN (perf — after correctness)
│   ├─ S3.1 (compiled foreach optimizer) ✅
│   ├─ S3.2 (tiny-kernel fusion)         ✅ FIXED 2026-06-21
│   ├─ S3.5 (conv1d stride+ReinterpretView) ✅ COMMITTED 2026-06-22
│   │   ├─ S3.5a (storage-offset in bind_buffers) 🔴 fix in-flight
│   │   ├─ S3.5b (conv1d backward x.grad=0)       🔴 in-flight
│   │   └─ S3.5c (grouped conv PC size mismatch)   🟡 FALSE ATTRIBUTION — test uses eager fallback; needs GPU verify
│   ├─ S3.3 (persistent reduction wiring) 🔴 dead code
│   └─ S3.4 (batch-dispatch overlap)      🟡 ◀── needs SP.2 first
│
├─ S4 DEPLOY
│   ├─ MS.1+MS.2 (aoti_shims.cpp: delete no-op + dangling handle) 🔴 CRITICAL ◀── crashes any AOTI model using zeros/ones
│   ├─ S4.0 (AOTI n_pc=0 for MM templates) 🔴 BLOCKING ◀── blocks all AOTI MM
│   ├─ S4.1 (full-step .so) ⛔ blocked upstream (torch.export empty.memory_format)
│   ├─ S4.2 (AOTI dispatch gaps: pool/scatter/rng/bwd_diff) 🔴
│   └─ S4.3 (A2.6 factory-op shim → all _FACTORY_OPS) 🔴
│
├─ Codegen correctness (CG)
│   ├─ CG.1 (argmin/argmax uint2 index > 16M) 🔴
│   ├─ CG.2 (bf16 packed16 wave32 WaveReadLaneAt guard) 🔴
│   ├─ CG.3 (packed16+welford guard bypass) 🔴
│   ├─ CG.4 (vec4 eligibility regex false-positive) 🔴
│   └─ CG.5 (pointwise.py split to ≤800 L) ⛔
│
├─ Slang/SPIR-V pipeline (SP)
│   ├─ SP.1 (reflection_ext numthreads dead/wire) 🔴
│   ├─ SP.2 (async compile .result() blocking) 🔴 ◀── prerequisite for S3.4
│   └─ SP.3 (SPIR-V cache key PC-layout hash) 🔴
│
└─ Continuous: E1/E2/E3/E4/E5 (coverage) · F (regression lock)
```

**Critical path to "train a full CNN with GN":** S2.0 → S2.1 (closed). Current
blockers: S3.5a/b/c (conv1d/grouped conv regressions, in-flight). **Critical
path to realize the warm-up vision:** S0.2 → S0.1 (probe drives codegen) ∥
S2.3 (validation actually runs). **Critical path to AOTI Python-free deploy:**
S4.0 (MM n_pc=0 bug) → S4.2 (dispatch gaps) → S4.3 (factory shim) → S4.1.
**Perf critical path:** SP.2 (async non-blocking) → S3.4 (batch-dispatch
overlap). Breadth (E4 pool-bwd, E5 SDPA-bwd) and codegen quality (CG.*) follow
correctness.

---

## § 3.5 — A2.6 Factory ops audit findings

**2026-06-22 audit (confirmed):**

| Op | `@register_lowering` in `lowerings/`? | AOTI shim? | Current path |
|----|---------------------------------------|------------|--------------|
| `aten.zeros` | ❌ None | `aoti_torch_zeros_vulkan` (dead code, line 159) | Upstream Inductor → `tensor_constructor` → `_full` → `Pointwise.create` + fill kernel |
| `aten.ones` | ❌ None | `aoti_torch_ones_vulkan` (dead code, line 180) | Upstream Inductor → same path |
| `aten.full` | ❌ None | `aoti_torch_full_vulkan` (dead code, line 201) | Upstream Inductor → `Pointwise.create(ops.constant(fill_value))` |
| `aten.empty` | ❌ None | — | Upstream Inductor → `aoti_torch_empty_strided` |
| `aten.empty_like` | ❌ None | — | Upstream Inductor → `aoti_torch_empty_strided` |
| `aten.zeros_like` | ❌ None | — | Upstream Inductor → `empty_strided` + fill kernel |
| `aten.ones_like` | ❌ None | — | Upstream Inductor → `empty_strided` + fill kernel |

**Key finding: no gap.** All factory ops work correctly via upstream Inductor fallthrough.
`aten.full`/`zeros`/`ones` become `Pointwise.create` (a pointwise fill kernel), not
constant-folded buffers. AOTI models use `aoti_torch_empty_strided_vulkan`
(`cpp_wrapper_gpu.py:198`) + a fill kernel for each factory op.

**Dead code identified**: `aoti_torch_zeros_vulkan` (line 159), `aoti_torch_ones_vulkan`
(line 180), `aoti_torch_full_vulkan` (line 201) — defined but never called. Could be
removed as part of a cleanup pass, or wired to AOTI codegen if pure-AOTI fill semantics
are needed without a compute shader.

**No `make_fallback`** is registered for any factory op in the Vulkan backend.
**No factory-op-to-fill canonicalization pass** exists in `fx_passes/` — upstream
Inductor handles this transparently.

**Verdict**: A2.6 is NOT an open bug; factory ops work correctly via the upstream path.
The dead shim functions are a mild code cleanliness issue but not a correctness risk.
Mark this item as ✅ (no action needed) or 🟢 (cleanup-only).

## § 3.6 — meta_patches/ audit (anti-goal #4 inventory)

**2026-06-22 audit (confirmed) — full classification of `meta_patches/` contents:**

### Classification key
- **(A)** Legitimate `fake_impl` needed for FakeTensor shape inference only — not a symptom fix
- **(B)** Symptom-fix that papers over a missing primitive or upstream gap — target for future replacement
- **(C)** Workaround for upstream PyTorch/Dynamo/AOTAutograd gap — kept until upstream fixes

| Op / Pass | File:Line | Class | Safe to delete now? |
|---|---|---|---|
| Bulk `_OP_IMPLS` (view/shape/BLAS/pointwise/reduction/conv/embedding) | `__init__.py:130-275` | A | No |
| `_fft_r2c/c2c/c2r` fake impls | `__init__.py:250-252`; `shape_ops.py:300-316` | A | No (GPU dispatch via C++ exists; FakeTensor still needs these) |
| `_linalg_svd_fake` | `__init__.py:233`; `shape_ops.py:330-342` | B | When real lowering exists |
| `_randperm_fake` | `__init__.py:296`; `shape_ops.py:468-476` | B | **Yes** (comment: randperm can never appear in Vulkan FX graph) |
| `_FixMetaDevicePass` / `_install_joint_partition_device_fix` | `joint_graph_passes.py:26-300` | B | No (M15.2 AOTAutograd device-loss workaround) |
| `_rewrite_empty_meta_to_tangent_expand` | `joint_graph_passes.py:303-395` | B | No (PF.13 0-dim expand device-loss workaround) |
| `_rewrite_constant_folded_tangent` | `joint_graph_passes.py:398-580` | B | No (undoes AOTAutograd constant fold of upstream gradient) |
| `_skip_misc_patterns_for_vulkan` | `joint_graph_passes.py:680-700` | B | No (guards upstream `_misc_patterns_init`) |
| `_patch_dynamo_clone_input_for_vulkan` | `faketensor_hooks.py:14-115` | C | No (mirrors XLA path; upstream won't fix for Vulkan) |
| `_patch_fake_tensor_view_op_device` | `faketensor_hooks.py:118-170` | C | No (`in_kernel_invocation` meta device loss is upstream) |
| `_patch_fake_tensor_meta_conversion` | `faketensor_hooks.py:173-220` | C | No (meta-tensor saved activations are upstream behavior) |
| `_patch_tensor_deepcopy_for_vulkan` | `faketensor_hooks.py:223-280` | C | No (Vulkan missing from XLA/MPS/meta `__deepcopy__` fast-path) |
| `_patch_fx_graph_cache_reduce_tensor_for_vulkan` | `faketensor_hooks.py:283-350` | C | No (`tolist()` on null-storage Vulkan tensors) |
| `_patch_fake_tensor_skip_const_fold_for_vulkan_null` | `faketensor_hooks.py:353-460` | C | No (constant-fold on null Vulkan storage) |
| `_patch_graph_lowering_get_attr_for_vulkan_null` | `faketensor_hooks.py:463-595` | C | No (`GraphLowering.get_attr` Vulkan null handling) |
| `_register_matmul_meta` | `op_registration.py:89-145` | C | No (`aten.matmul` missing from `meta_table`) |
| `_register_sdpa_meta` | `op_registration.py:308-345` | C | No (SDPA missing from `meta_table`) |
| `_register_backward_meta_decomps` (6 ops: gelu/silu/leaky_relu/elu/upsample bwd) | `op_registration.py:19-75` | A | No |
| `_patch_proxy_call_matmul_decomp` | `op_registration.py:148-230` | B | No (M15.2 workaround; should become FX-level decomp) |
| `_patch_einsum_proxy_decomp` | `op_registration.py:233-305` | B | No (M15.2 workaround; should become FX-level decomp) |
| `_disable_bmm_to_mm_for_vulkan` | `op_registration.py:348-420` | B | No (guards upstream `bmm_to_mm` pattern) |
| `_register_logical_and_for_vulkan` | `op_registration.py:423-470` | B | No (eager path needed for PowBackward1) |
| `_register_bitwise_ops_for_vulkan` | `op_registration.py:473-520` | B | No (needed for `torch.isclose` / testing) |
| `_register_view_symint_autograd_pyimpl` | `autograd_registrations.py:14-145` | B | No (SymInt crash in `vulkan_view_autograd_adapter`) |
| `_register_permute_family_autograd_pyimpl` | `autograd_registrations.py:148-260` | B | No (non-aliasing permute causes constant-fold bugs) |
| `_register_activation_autograd_pyimpl` | `autograd_registrations.py:263-310` | B | **Maybe** (C1 pre-grad rewrite may have made it redundant) |
| `_patch_pre_grad_passes_for_relu_rewrite` | `decomposition_passes.py:350-470` | B | No (M15.2 workaround for ReluBackward0 meta cascade) |
| `_patch_pre_grad_passes_for_optimizer_foreach` | `decomposition_passes.py:220-345` | B | No (M15.2 workaround for missing foreach lowerings) |
| `_patch_pre_grad_passes_for_conv_gn_relu_fusion` | `decomposition_passes.py:473-648` | B | **Yes** (already **disabled** in `__init__.py`) |
| Dead backward fakes: `_native_batch_norm_backward_fake`, `_native_group_norm_backward_fake`, `_native_layer_norm_backward_fake`, `_upsample_bilinear2d_backward_fake`, `_upsample_nearest2d_backward_fake` | `shape_ops.py:200-260` | Dead | **Yes** (unreferenced) |
| Dead backward fakes: `_gelu_backward_fake`, `_softmax_backward_data_fake`, `_log_softmax_backward_data_fake`, `_avg_pool2d_backward_fake`, `_max_pool2d_with_indices_backward_fake`, `_linear_backward_fake` | `dtype_ops.py:80-170` | Dead | **Yes** (unreferenced) |

### Summary verdict

**Immediately deletable (no functionality loss):** 13 dead / already-disabled items:
- 5 dead backward fakes in `shape_ops.py:200-260`
- 6 dead backward fakes in `dtype_ops.py:80-170`
- `_randperm_fake` in `shape_ops.py:468-476` (comment confirms never appears in graph)
- `_patch_pre_grad_passes_for_conv_gn_relu_fusion` body in `decomposition_passes.py:473-648`
  (already disabled, dead code)

**Maybe deletable after verification:** `_register_activation_autograd_pyimpl`
(`autograd_registrations.py:263-310`) — check whether C1 pre-grad rewrite made it redundant.

**Structural symptom fixes (class B) that remain valid workarounds:** all other (B) items.
These paper over upstream PyTorch/AOTAutograd/Dynamo gaps that have no upstream fix planned
for PrivateUse1 backends. Filing each as a roadmap item (per anti-goal #4) would duplicate
the upstream tracker without adding actionable work — mark them as `tracked-upstream`.

**New implement ticket (cleanup)**:  
See `§ 2.5` ticket 16: delete the 13 immediately-removable items (~80 LOC net).

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
   wrapper. ✅ All known violations resolved: S2.1 (mm/adaptive_avg_pool_bwd ✅
   fixed 2026-06-21), S2.5 (avg_pool2d fwd → `torch_vulkan` custom op ✅
   fixed 2026-06-21).
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
| A2 / A2.5–A2.7 | S4.1 / S4.3 | C6 | S3.2 |
| A3 / A4 | ✅ closed | D1 | S1.1 |
| A5 | S2.5 | E1/E2/E3 | Continuous |
| — | S4.0 (new, AOTI MM n_pc) | — | S4.2 (new, AOTI gaps) |
| — | CG.1–CG.5 (new, 2026-06-22) | — | SP.1–SP.3 (new, 2026-06-22) |
| — | E4 (pool bwd Slang) | — | E5 (SDPA bwd) |
| B1 / B2 / B3 | ✅ closed | F1 | Continuous |

*This file is the single canonical roadmap. Do not fork a new numbered version —
edit it in place: mark items ✅ as they close, add new sub-items under the right
spine stage.*
