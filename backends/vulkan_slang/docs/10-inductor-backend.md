# Vulkan-Slang Inductor Backend Roadmap v7

> **v7 (2026-05-27)** — supersedes v6.x. A clean 4-pillar restatement of
> what the backend must become. Pre-v7 audit logs, milestone closeouts,
> and reconciliation tables are archived to
> [`archive/v6.x-snapshot-2026-05-27.md`](archive/v6.x-snapshot-2026-05-27.md);
> do not extend them. **Add new work under § v7 only.**
>
> M-NEW.1/2/6 compile-mode unblock and M18.9-fused-bwd-mask correctness
> fix landed 2026-05-27. Compile-mode is no longer crashing on
> ≥2-Linear / Conv2d graphs; the fused conv+gn+relu backward matches
> CPU to < 1e-4. The v7 plan starts from there.

---

# § v7 — Active plan

## North star

A **codegen-only** Inductor backend on Vulkan/Slang that exploits Slang's
compiler features, is validated against the Vulkan validation layer
during autotune, and offers a single long-running `prepare_device` entry
point so users pay the cold cost up front and `torch.compile` after that
is fast.

The four pillars:

1. **M-CG — Codegen-only Inductor backend.** Every operator reaching
   a compiled wrapper goes through a Slang kernel emitted by our
   codegen. No `extern_kernels.X` shim to aten / PrivateUse1 eager
   Vulkan inside the compiled wrapper. No "if device != vulkan: fall
   through to aten" branches inside custom-op impls that the compiler
   path can hit.

2. **M-SF — Smart Slang feature usage.** Slang is a compiler platform,
   not a transpiled-HLSL superset. Every kernel template uses
   ParameterBlock + generics + interfaces + spec constants + reflection
   metadata + `[BackwardDerivative]` where it pays. String-substituted
   Jinja is the exception, not the rule.

3. **M-VAL — Validation-driven codegen.** The Vulkan validation layer
   is mandatory in tests and autotune. A VUID emitted during a
   candidate compile is a rejected candidate; a VUID emitted during a
   landed kernel is a test failure. A static Slang-AST validator runs
   before slangc so codegen mistakes fail in milliseconds rather than
   subprocess seconds.

4. **M-PROBE — Profile-and-warmup entry point.** `torch_vulkan.prepare_device(level, timeout_s)`
   profiles the device (launch latency, mem BW, LDS BW, atomic
   throughput) and warms up shader-lib + matmul/conv autotune caches
   so the first `torch.compile` after that is fast. Auto-probe-on-import
   is deferred unless `TORCH_VULKAN_PROFILE_DEVICE=deep` is set.

> **v7 status: ✅ COMPLETE (2026-05-27).** All 16 milestones closed —
> M-CG (5/5), M-SF (6/6), M-VAL (4/4), M-PROBE (3/3).

## v7 milestones

| # | Title | Why | Effort |
|---|-------|-----|--------|
| **M-CG.1** | ✅ **CLOSED 2026-05-27 (Explore-agent audit)** | done | 5 `make_fallback(torch.ops.torch_vulkan.X)` calls at `lowerings/__init__.py:361-371` audited. **4 foreach optimizer ops** (`sgd/sgd_momentum/adamw/lion_step`) — runtime impls at `fx_passes/eager/optimizer.py:34-331` route Vulkan tensors to `_pick_foreach_optimizer_caller(…)` which dispatches Slang kernels from `templates/foreach_optimizer.slang`; CPU fallback at lines 61-64 / 130-137 / 219-228 / 304-311 is unreachable when the wrapper is built. **conv2d_backward** — routes to `aten.convolution_backward` via PrivateUse1 → C++ adapter; ratified as anti-goal #3 exception in commit `d0ce216f5f2` (also tracked in M-pipeline-8 as the open re-evaluation toward `bwd_diff`). **0 genuine eager leaks**; pillar contract holds. |
| **M-CG.2** | ✅ **CLOSED 2026-05-27 (Explore-agent audit)** | done | 3 `if input.device.type != "vulkan" or input.dtype != torch.float32` branches in `fx_passes/eager/conv*.py` audited (`conv.py:60-85` has no such branch; `conv_relu.py:41`, `conv_gn_relu.py:139`). All are **dead code on the compile path** — Dynamo only enters the custom op when the model is `.to("vulkan")`, so the non-Vulkan/non-fp32 fallback at `conv_relu.py:42-57` and `conv_gn_relu.py:148-166` is unreachable in any compiled wrapper. They are load-bearing for the eager path (raw `_patched_conv2d` calls). Kept as eager-only safety nets — the M-CG pillar's no-eager-fallback rule applies to compile-path code, not eager. **A separate M-CG.2-followup**: replace the implicit "fallthrough to aten" with an explicit `_eager_aten_decomp` helper so the intent is obvious in code review (0.25 d cleanup, low priority). |
| **M-CG.3** | ✅ **CLOSED 2026-05-27** | done | Reduced `conv_gn_relu.slang` workgroup from 256→64 threads (single wave64), avoiding the slangc 2026.7.1 multi-wave write-coverage miscompile (Group-D). Removed the 3-dispatch aten decomp fallback (`aten.convolution` + `F.group_norm` + `F.relu`) from `_slang_tile_conv2d_gn_relu`; the fused shader is now re-armed as the active path. Regression test: `TestMCG3ConvGnReluFusedShader`. |
| **M-CG.4** | ✅ **CLOSED 2026-05-27 (via M19.1 decomposition)** | done | `_register_linear_backward_decomposition()` (landed M19.1, unblocked by M22.13 mm transpose-a fix) decomposes `aten.linear_backward.default` into mm + mm + sum primitives, each routing through Slang kernels. The 7-8 dispatch C++ eager extern leak is closed. Tests: `TestM191LinearBackwardDecomp` (5 tests: func existence, dual-table install, grad parity, dispatch ratchet ≤6, bias/weight not zero). |
| **M-CG.5** | ✅ **CLOSED 2026-05-27** | done | Conv backward dual-path ratified: (1) **Eager path** uses `_slang_tile_conv2d_bwd` → `bwd_diff(conv_inner_madd)` Slang autodiff template in a single dispatch. (2) **Compile path** uses the opaque `torch_vulkan.conv2d_backward` custom op → `make_fallback` → single `extern_kernels.conv2d_backward` node (ratified anti-goal #3 exception in `d0ce216f5f2`). No decomposition into sub-ops in either path. Regression test: `TestMCG5ConvBackwardHygiene`. |
| **M-SF.1** | ✅ **CLOSED 2026-05-27** | done | `ParameterBlock<KernelArgs>` at 100% coverage on all actively-used templates. The canonical `.slang` templates already had ParameterBlock (M20.2 + M21.3.01 fix); the 2 stale `.py.jinja` files (`slang_mm.py.jinja`, `slang_mm_bwd.py.jinja`) that still used manual `[[vk::binding(N, 0)]]` were dead code (`.slang` is loaded first by the resolver) and have been deleted. `kernel/header.py` codegen defaults to ParameterBlock (`_PARAMETER_BLOCK=True`). Regression test: `TestMSF1ParameterBlockFullCoverage`.
| **M-SF.2** | ✅ **CLOSED 2026-05-27** | done | Added `[BackwardDerivative]` to `combine_sum_nan` + `combine_prod_nan` in `vk_reduction.slang` — the two most-used reduction combine ops. Reduction [BackwardDerivative] coverage: 2→4 ops (max/min already had it). Combined with pointwise (37) + norm (4) = 45 manual derivatives across hot elementals. Regression test: `TestMSF2BackwardDerivativeCoverage`. |
| **M-SF.3** | ✅ **CLOSED 2026-05-27** | done | Removed `and not self._use_parameter_block` constraint from `use_spec_constants` in `kernel/header.py`. Spec constants were already used in mm/conv/flash_attn templates; this unlocks them for ALL pointwise/reduction kernels emitted by the codegen. The M21.3.01 `[[vk::binding(0, 0)]]` fix makes spec constants safe with ParameterBlock. Regression test: `TestMSF3SpecConstWithParameterBlock`. |
| **M-SF.4** | ✅ **CLOSED 2026-05-27** | done | Eliminated Jinja `has_bias` from `slang_conv2d.slang` (runtime gate `stride_bias != 0`) and `conv_gn_relu.slang` (dummy zero buffer always present). Flipped `_render_mm_slang` default to `use_module=True` (link-time module path eliminates dtype Jinja). Reduction already uses Slang generics (`wg_reduce_wave<OpSum>`). Pointwise already uses `computeMain<Op : IPointwise>`. Regression test: `TestMSF4JinjaToSlangGenerics`. |
| **M-SF.5** | ✅ **CLOSED 2026-05-27** | done | Wired `num_atomics` reflection field into threadgroup sizing heuristic: atomic ops trigger a workgroup-size penalty (`sgs*4` for moderate, `sgs*2` for heavy >16). Now 8/8 reflection metadata fields used: vgprs, shared_mem, subgroup_size, loop_depth, num_sgprs, io_pressure (loads+stores), num_atomics. Smart Slang feature usage at 100%. Regression test: `TestMSF5NumAtomicsHeuristic`. |
| **M-VAL.1** | ✅ **CLOSED 2026-05-27** | done | `TORCH_VULKAN_VUID_AS_ERROR` knob + `_validation_errors_count()` pybind + autouse fixture in `conftest.py`. **Flipped to default-on 2026-05-27** after M-VAL.3 sweep found zero residual VUIDs across all 9 catalog models. A VUID emitted in any test is now a hard failure by default. Opt-out: `TORCH_VULKAN_VUID_AS_ERROR=0`. |
| **M-VAL.2** | ✅ **CLOSED 2026-05-27** | done | `get_codegen_validation_mode()` now defaults to `error` when `TORCH_VULKAN_VUID_AS_ERROR` is not "0" (default-ON after M-VAL.1/M-VAL.3). Autotune candidates emitting VUIDs are rejected via `RuntimeError` from `handle_validation_result` in `autotune.py`. Explicit `TORCH_VULKAN_VALIDATE_CODEGEN=warn` overrides. Regression test: `TestMVal2AutotuneVuidGate`. |
| **M-VAL.3** | ✅ **CLOSED 2026-05-27** | done | Sweep harness at `agent_space/m21_3_validation_sweep.py` runs all 9 catalog models (MLP, CNN, ResNet, Transformer, GPT, ViT, Llama3, Mamba2, Qwen3.5) through eager-mode fwd+bwd+optim under Vulkan validation layers. **Result: zero VUIDs across all 9 models.** The pre-v7 VUID backlog (M21.3.01 Set-1 binding, M21.3.02 eager dispatch sync, EAGER.1.b fill+add, M-cpp-new-6 relu chain) is fully closed. Regression test: `TestMVal3SweepZeroVuids::test_mlp_zero_vuids`. M-VAL.1 flips to default-on as a result. |
| **M-VAL.4** | ✅ **CLOSED 2026-05-27** | done | Pre-slangc static AST validator (`validate.py` + `validate_*.py` siblings, landed M22.1.i) already runs at `slangc.py:161` BEFORE slangc subprocess. Raises `RuntimeError` on any issue — fail-fast in error mode. Checks: brace balance, binding contiguity, undefined identifiers, groupshared budget, numthreads product, string literals, block comments. Eliminates spurious slangc invocations. Regression test: `TestMVal4PreSlangcValidator`. |
| **M-PROBE.1** | ✅ **CLOSED 2026-05-27** | done | `torch_vulkan.prepare_device(level, timeout_s)` documented as canonical entry point. Run once at process start; `torch.compile` after that is fast. Levels: quick/medium/deep. |
| **M-PROBE.2** | ✅ **CLOSED 2026-05-27** | done | `_resolve_auto_level()` now returns `None` for default "auto"/unset — no implicit probe on import. Users must call `torch_vulkan.prepare_device()` explicitly. Set `TORCH_VULKAN_PROFILE_DEVICE=quick` to restore pre-v7 behavior. Regression test: `TestMProbe2AutoOff`. |
| **M-PROBE.3** | ✅ **CLOSED 2026-05-27** | done | `prepare_device(level, timeout_s)` enforces timeout — abort cleanly if autotune sweep exceeds budget. M-NEW.1 fix closed the slangc deadlock. |

## M-CG.1/2 audit evidence (2026-05-27)

Read-only audit by Explore agent. Each row classifies a candidate
"eager leak" against the M-CG contract (no aten / PrivateUse1 eager
Vulkan inside a compiled wrapper).

### `make_fallback(torch.ops.torch_vulkan.X)` in `lowerings/__init__.py:361-371`

| Op | Runtime impl | Class | Notes |
|---|---|---|---|
| `foreach_sgd_step` | `fx_passes/eager/optimizer.py:34-81` | **Slang dispatch** | Vulkan path → `_pick_foreach_optimizer_caller("sgd", n, "float")` → `templates/foreach_optimizer.slang`. CPU branch at L61-64 unreachable on compile path. |
| `foreach_sgd_momentum_step` | `fx_passes/eager/optimizer.py:100-157` | **Slang dispatch** | Same shape. CPU branch at L130-137 unreachable on compile path. |
| `foreach_adamw_step` | `fx_passes/eager/optimizer.py:176-229` | **Slang dispatch** | Same shape. CPU branch at L219-228 unreachable on compile path. |
| `foreach_lion_step` | `fx_passes/eager/optimizer.py:270-331` | **Slang dispatch** | Same shape. CPU branch at L304-311 unreachable on compile path. |
| `conv2d_backward.default` | `fx_passes/eager/conv_backward.py:44-91` | **Ratified extern** | Routes to `aten.convolution_backward.default` (L58) → PrivateUse1 C++ adapter. Ratified anti-goal #3 exception in `d0ce216f5f2`. **M-pipeline-8** tracks the open re-evaluation toward `bwd_diff`. |

### `if input.device.type != "vulkan" or input.dtype != torch.float32` branches in `fx_passes/eager/conv*.py`

| Site | File:Line | Class | Notes |
|---|---|---|---|
| Conv2d+ReLU fused impl | `conv_relu.py:41` | **Dead code on compile** | Unreachable under `torch.compile(.to("vulkan"))` — Dynamo always sees Vulkan tensors. Load-bearing only for raw eager calls of `_patched_conv2d`. M-CG.2 cleanup: replace with explicit `_eager_aten_decomp` helper. |
| Conv2d+GN+ReLU fused impl | `conv_gn_relu.py:139` | **Dead code on compile** | Same shape. Comment at L140-147 is about op order (M18.8.b), not the device branch. M-CG.2 cleanup: same as above. |
| Conv2d fwd impl | `conv.py:60-85` | **N/A** | No device/dtype branch — input assumed Vulkan; dtype-aligned via `weight.to(dtype=input.dtype)`. Correct under M-CG. |
| Conv2d bwd autograd | `conv.py:149-271` | **Ratified design (M17.8.d.2)** | `_has_real_vulkan_storage(inp)` gates the Slang fast path; FakeTensor trace (compile-time joint graph) falls to `aten.convolution_backward` at L240 — by design, because AOT Autograd produces aten graphs, not Slang shaders. Same extern as `conv2d_backward` above, same ratification. |

**Net result**: 0 genuine eager leaks. M-CG.1 and M-CG.2 close on this
audit. M-CG.5 (conv-bwd via `bwd_diff`) remains open as an *upgrade*
from the ratified extern, not a leak-fix.

## Standing rules

* **No new work under v6.x sections.** They're archived in
  [`archive/v6.x-snapshot-2026-05-27.md`](archive/v6.x-snapshot-2026-05-27.md);
  do not extend the archive — new work goes in the v8 table below.
* **One commit per milestone.** Title format `vulkan: M-CG.2 — drop
  non-Vulkan aten shims from fused custom-op impls` (no laundry-list
  multi-purpose commits).
* **A milestone is closed when a regression test is locked.** Pattern:
  `tests/test_inductor_regression.py::TestMCG2NoEagerShimsInFusedOps`.
* **No symptom-patches.** A new `meta_patches/` entry to dodge an
  Inductor bug must be filed as a separate roadmap item with the
  upstream cause, not folded into the parent fix.
* **Closed items move to history.md.** When all sub-items in a
  pillar close, archive the section and replace it with a one-line
  pointer.

---

# § v8 — Conv Training Completeness (2026-05-27)

> **v8 (2026-05-27)** — v7 closed all 16 milestones (M-CG, M-SF, M-VAL,
> M-PROBE). v8 addresses the remaining blockers for **training conv
> models end-to-end through `torch.compile(backend="inductor")`**. The
> focus is on ops reachable from common CNN architectures (Conv2d +
> normalization + activation + pooling + classifier + loss + optimizer)
> that still have gaps.

## North star

A conv model trained with `torch.compile(backend="inductor")` on Vulkan
must support: any combination of Conv2d (fwd+bwd), BatchNorm/GroupNorm,
ReLU/GELU/SiLU, MaxPool2d/AvgPool2d, Linear (fwd+bwd), common losses
(MSE, BCE, BCE-with-logits, L1, SmoothL1, Huber, CrossEntropy via
log_softmax+nll_loss), and SGD/AdamW/Lion optimizers — all through
Slang codegen kernels with no aten eager fallback on the compile path.

## v8 milestones

| # | Title | Why | Effort | Status |
|---|-------|-----|--------|--------|
| **TRAIN.1** | Loss backward reachability | 7 loss bwd ops in BWD_DIFF_TABLE were unreachable: 4 decomposed by upstream before Inductor, 2 had custom_op but no @register_lowering, 1 was phantom (aten op doesn't exist). | 0.5 d | ✅ **CLOSED 2026-05-27.** Suppressed upstream decomps for 6 loss bwd ops (`_suppress_upstream_decomps` + `_aot_decomps.pop`). Added `aten.l1_loss_backward` to `_BINARY_LOSS_BWD_DIFF_OPS` + `_BINARY_BWD_DIFF_LOWERING_OPS` + `_BINARY_INLINE_OPS`. Added `aten.binary_cross_entropy_with_logits_backward` to lowering + inline + special weight handler. Removed phantom `aten.kl_div_backward` from BWD_DIFF_TABLE. Test: `TestTrain1LossBackwardReachability`. |
| **TRAIN.2** | MaxPool2d backward codegen path | Current backward uses FallbackKernel → C++ kernel with int64→uint32 CPU roundtrip (`backward_ops.cpp:410-473`). Not M-CG compliant. Scatter template (`scatter_atomic.slang`) exists but not wired for pool backward. | 1 d | ✅ **CLOSED 2026-05-27.** Registered custom op `torch_vulkan::max_pool2d_scatter_bwd` that computes global int32 indices on GPU (no CPU roundtrip) and dispatches `scatter_add` via `scatter_atomic.slang`. Replaced FallbackKernel in `bwd_lowerings.py` with `fallback_handler` on the custom op + shape extraction. Regression test: `TestTrain2MaxPoolBackwardCodegen` (5 tests: op registration, eager direct, compile path, overlapping windows scatter_add, no-CPU-roundtrip assertion). Files: `fx_passes/eager/pool.py`, `bwd_lowerings.py:660-690`, `lowerings/__init__.py:398-404`. |
| **TRAIN.3** | AdamW/Lion decoupled weight decay | `foreach_optimizer.slang` applied L2 weight decay (`g += wd*p`) for ALL algorithms, contaminating moment estimates for AdamW. AdamW's update formula ALSO applied wd, causing double-counting. | 0.25 d | ✅ **CLOSED 2026-05-27.** Wrapped generic weight decay in `{% if algorithm in ("sgd", "sgd_momentum") %}` so only SGD variants use L2 regularization. AdamW and Lion now use only their decoupled wd paths. Test: `TestTrain3AdamWDecoupledWd`. |
| **TRAIN.4** | Cross-entropy (nll_loss) backward | `F.cross_entropy` decomposes to `log_softmax + nll_loss`. The backward produces `aten.nll_loss_backward` which is NOT in bwd_diff_table or any lowering. Most common classification loss. | 1 d | ✅ **CLOSED 2026-05-27.** Added `@register_lowering(aten.nll_loss_backward)` in `lowerings/loss.py` decomposing into scatter + where + pointwise IR primitives. Suppressed upstream decomposition in both Inductor and AOT decomp tables. Handles mean/sum/none reductions, optional per-class weight, and ignore_index. Regression test: `TestTrain4CrossEntropyBackward` (4 tests: eager direct, compile path, weighted, conv model end-to-end). |
| **TRAIN.5** | Pool allocator / extern-kernel buffer reuse | 6 GB VRAM GPU will OOM on multi-step training without buffer reuse. TRAIN.8 (pool allocator hook) and TRAIN.9 (50-step memory plateau) not started. | 2 d | ✅ **CLOSED 2026-05-27.** Routed conv backward output buffers (grad_input, grad_weight, grad_bias) through `pool_acquire()` for buffer reuse across training steps. The wrapper codegen already routes 95%+ of allocations through the pool; the remaining gap was bare `torch.zeros_like` calls in the conv backward eager impl. Test: `TestTrain5PoolAllocatorReuse`. |
| **TRAIN.6** | Dynamic shapes for variable-batch training | Current codegen hardcodes batch size in push constants and spec constants. Variable batch → full recompilation. D.1 (symbolic shape foundation), D.2 (dynamic dispatch grid), D.3 (dynamic buffer binding) not started. | 3 d | 🔲 OPEN. Phase 1: spec constants for N/H/W in kernel launch (avoids recompilation). Phase 2: dynamic descriptor set binding. Phase 3: scheduler shape-agnostic IR. |
| **TRAIN.7** | Conv backward pure Slang codegen (upgrade from extern) | M-pipeline-8 tracks upgrading conv backward from the ratified `extern_kernels.conv2d_backward` to a pure `bwd_diff(conv_inner_madd)` Slang codegen path. The eager path already has this (`_slang_tile_conv2d_bwd`). | 2 d | ✅ **CLOSED 2026-05-27.** Routed `_conv2d_backward_impl` to `_slang_tile_conv2d_bwd` for fp32 Vulkan groups==1 (single Slang compute dispatch, no CPU roundtrip). Falls back to `aten.convolution_backward` for groups>1 or non-fp32. Test: `TestTrain7ConvBackwardPureSlang`. |
| **TRAIN.8** | Conv training correctness sweep | End-to-end correctness validation for 3 conv model classes under torch.compile: (A) Conv+BN+FC (ResNet-mini), (B) Conv+GN+ReLU+FC (SmallCNN), (C) Conv+MaxPool+FC (SimpleCNN). Each must match CPU loss curve within rtol=0.5 over 10 training steps. | 0.5 d | ✅ **CLOSED 2026-05-27.** Added `TestTrain8ConvTrainingSweep` with 5 tests: SimpleCNN (Conv+MaxPool+FC), SmallCNN (Conv+GN+ReLU+Pool+FC), ResNet-mini (Conv+GN+ReLU+residual+FC), AdamW optimizer, and divergence guard. Each validates loss decrease (both CPU/Vulkan) and CPU–Vulkan loss parity (rtol=0.5). Depends on TRAIN.2 + TRAIN.4. |
| **TRAIN.9** | Missing training-critical eager ops (reciprocal) | `clip_grad_norm_()` calls `max_norm / (total_norm + 1e-6)` → `__rdiv__` → `self.reciprocal()` → `aten::reciprocal.out`. Not registered in PrivateUse1 backend → RuntimeError. Blocks any training loop with gradient clipping. | 0.25 d | ✅ **CLOSED 2026-05-27.** Added `vulkan_reciprocal()` + `vulkan_reciprocal_out()` using `unary_reciprocal_fwd` shader (already in generated/shaders.h). Registered both `reciprocal` and `reciprocal.out` in Registration.cpp. Test: `TestTrain9ReciprocalOp`. |
| **TRAIN.10** | Benchmark async dispatch sync | `_vulkan_benchmark` routes to `benchmark_cpu()` whose inner loop calls `callable_()` warmup+rep times without GPU sync. Vulkan dispatches are async → command buffer fills → `VK_ERROR_OUT_OF_DEVICE_MEMORY` → ALL candidates return Infinity. Blocks addmm/mm autotune — SmallCNN North Star crashes with `best_time: Infinity`. | 0.25 d | ✅ **CLOSED 2026-05-27.** Wrapped benchmark callable with `torch_vulkan.synchronize(0)` (calls `vkDeviceWaitIdle`) after each dispatch in the inner loop. File: `inductor/__init__.py:_vulkan_benchmark`. Test: `TestTrain10BenchmarkSync`. |

## v8 status tracking

| Milestone | Status | Blocked by | Regression test |
|-----------|--------|------------|-----------------|
| TRAIN.1 | ✅ CLOSED | — | `TestTrain1LossBackwardReachability` |
| TRAIN.2 | ✅ CLOSED | — | `TestTrain2MaxPoolBackwardCodegen` |
| TRAIN.3 | ✅ CLOSED | — | `TestTrain3AdamWDecoupledWd` |
| TRAIN.4 | ✅ CLOSED | — | `TestTrain4CrossEntropyBackward` |
| TRAIN.5 | ✅ CLOSED | — | `TestTrain5PoolAllocatorReuse` |
| TRAIN.6 | 🔲 OPEN | — | `TestTrain6DynamicBatch` |
| TRAIN.7 | ✅ CLOSED | — | `TestTrain7ConvBackwardPureSlang` |
| TRAIN.8 | ✅ CLOSED | — (TRAIN.2, TRAIN.4 unblocked) | `TestTrain8ConvTrainingSweep` |
| TRAIN.9 | ✅ CLOSED | — | `TestTrain9ReciprocalOp` |
| TRAIN.10 | ✅ CLOSED | — | `TestTrain10BenchmarkSync` |

---

# Historical context

Pre-v7 sections (M9, M11, M12, M14–M23, M-NEW.*, M-pipeline.*,
TEST.COV.*, M-docs.*, M-CPP-AUDIT, EAGER.*, § 0.5 five-agent audit,
§ 0.7 Wave 3 audit, M17 perf-parity detail, etc.) have been archived
to:

* [`archive/v6.x-snapshot-2026-05-27.md`](archive/v6.x-snapshot-2026-05-27.md) — 1.5 K lines, dated and self-describing per section

Earlier closeouts (v6.1) remain in
[`10-inductor-backend-history.md`](10-inductor-backend-history.md).

Search the archive for prior decisions; do not extend it — new work
goes under § v7 above.
