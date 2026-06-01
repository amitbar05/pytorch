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

## v8 milestones (all CLOSED — 2026-05-27/28)

Detailed closeout notes archived in
[`docs/archive/v8-closeouts-2026-05-28.md`](archive/v8-closeouts-2026-05-28.md).

| # | Title | Close date | Regression test |
|---|-------|-----------|-----------------|
| TRAIN.1 | Loss backward reachability (6 loss bwd decomps suppressed) | 2026-05-27 | `TestTrain1LossBackwardReachability` |
| TRAIN.2 | MaxPool2d backward scatter codegen (replaces FallbackKernel + CPU roundtrip) | 2026-05-27 | `TestTrain2MaxPoolBackwardCodegen` |
| TRAIN.3 | AdamW/Lion decoupled weight decay (L2 gate by algorithm) | 2026-05-27 | `TestTrain3AdamWDecoupledWd` |
| TRAIN.4 | Cross-entropy (nll_loss) backward lowering | 2026-05-28 | `TestTrain4CrossEntropyBackward` |
| TRAIN.5 | Pool allocator reuse for conv backward output buffers | 2026-05-27 | `TestTrain5PoolAllocatorReuse` |
| TRAIN.6 | Dynamic shapes for variable-batch training (spec const defaults + shape-agnostic cache keys) | 2026-05-28 | `TestTrain6DynamicBatch` |
| TRAIN.7 | Conv backward via pure Slang codegen (replaces extern) | 2026-05-27 | `TestTrain7ConvBackwardPureSlang` |
| TRAIN.8 | Conv training correctness sweep (SimpleCNN/SmallCNN/ResNet-mini) | 2026-05-28 | `TestTrain8ConvTrainingSweep` |
| TRAIN.9 | aten::reciprocal registration (blocks clip_grad_norm_) | 2026-05-27 | `TestTrain9ReciprocalOp` |
| TRAIN.10 | Benchmark async dispatch sync (VK_OOM fix) | 2026-05-27 | `TestTrain10BenchmarkSync` |
| TRAIN.11 | mm_tile.slang precompilation bypass level filter | 2026-05-28 | `debug_mm_tile_compile.py` |
| TRAIN.12 | Multi-norm backward can_fuse guard (tiling assertion) | 2026-05-28 | TestTrain8 unxfail |
| TRAIN.13 | Test infra fixes: forward-only compile + BN backward | 2026-05-28 | TestTrain8 (AdamW, multi-arch) |

### Residual xfails (open)

Two tests in v8 suites still carry `pytest.xfail` markers for known
blockers that are out-of-scope for the milestone they were filed against:

- `TestTrain5PoolAllocatorReuse.test_vram_plateaus_across_training_steps`
  — xfailed by "0-d scalar div" partitioner blocker (multi-step training
  with AOT joint graph triggers an unsupported scalar division).
- `TestTrain7ConvBackwardPureSlang.test_conv_backward_through_compile`
  — xfailed by "Tensor has no backing Vulkan buffer" in compile path
  (eager path works; compile path has memory-planning gap).

---

# § v9 — Compile-Path Completeness & Codegen Hygiene (2026-05-28)

> **v9 (2026-05-28)** — v7/v8 closed all milestones. v9 targets the
> **remaining compile-path blockers and codegen gaps** that prevent
> arbitrary CNN training (ResNet/EfficientNet/MobileNet) through
> `torch.compile(backend="inductor")` with full AOT backward.
>
> The forward-only compile pattern (v8 escape hatch) works for 3 small
> conv architectures. Full AOT backward works in standalone scripts but
> fails in pytest. v9 aims to make full AOT backward reliable and
> eliminate the forward-only workaround.

## v9 pillars

| # | Pillar | Goal |
|---|--------|------|
| **M-COMPILE** | Compile-path blockers | Fix the 3 residual compile-path xfails (conv bwd "no backing buffer", scalar div in joint graph, addmm autotune). Make `torch.compile(backend="inductor")` with loss+backward in the compiled function work reliably. |
| **M-DECOMP** | Decomposition gaps | Suppress upstream decomps that eat ops before Vulkan lowerings fire. Ensure all registered `@register_lowering` ops are actually reachable. |
| **M-CODEG** | FallbackKernel → pure codegen | Replace `make_fallback`/`FallbackKernel` paths with fused Slang codegen or `bwd_diff` table entries. Targets: optimizer steps, avg_pool2d_backward, convolution_backward. |
| **M-MODEL** | Model coverage | Expand to Conv3d, BatchNorm running stats, LSTM/GRU cell compile-path, and autotune CUDA-filter. |

## v9 milestones

| # | Pillar | Title | Why | Effort | Status |
|---|--------|-------|-----|--------|--------|
| **COMPILE.1** | M-COMPILE | Conv backward compile-path memory planning | `TestTrain7ConvBackwardPureSlang.test_conv_backward_through_compile` xfailed: "Tensor has no backing Vulkan buffer". Eager path works. Compile path produces an FX graph where conv_bwd output tensors lack Vulkan backing buffers — memory planner issue. | 1 d | 🔲 OPEN |
| **COMPILE.2** | M-COMPILE | Joint-graph scalar div in multi-step training | `TestTrain5PoolAllocatorReuse.test_vram_plateaus_across_training_steps` xfailed: "0-d scalar div" partitioner blocker. AOT partitioner can't handle 0-d scalar operations that arise in multi-step training with joint fwd+bwd graph. Either register scalar ops via PrivateUse1 or route through Inductor's scalar lowering. | 1.5 d | 🔲 OPEN |
| **COMPILE.3** | M-COMPILE | Linear (addmm) compile-path autotune | Models with `nn.Linear` fail `aten.addmm` autotune: `mm_tile` choices either return `NoValidChoicesError` or `best_time: Infinity`. Workaround: 1×1 Conv classifier. Fix: either wire addmm through the Vulkan mm lowering directly (bypassing autotune) or fix the mm_tile autotune path. | 1 d | 🔲 OPEN |
| **DECOMP.1** | M-DECOMP | Suppress _softmax / _log_softmax upstream decomps | `aten._softmax` and `aten._log_softmax` are in `inductor_decompositions` (decomposition.py:79/94) but not in Vulkan `ops_to_suppress`. Vulkan lowerings exist (`lowerings/softmax.py`) but never fire because upstream decomp produces the same primitives first. Suppress so Vulkan lowerings are reachable. | 0.25 d | ✅ **CLOSED 2026-05-28.** Added `_softmax.default` and `_log_softmax.default` to `ops_to_suppress` in `lowerings/__init__.py`. |
| **DECOMP.2** | M-DECOMP | convolution_backward bias gradient decomp | Upstream Inductor decomp at decomposition.py:306 splits `convolution_backward` into `bias_grad = sum(grad, dims)` + recursive `convolution_backward(..., output_mask=(T,T,F))`. **Resolved: harmless for Vulkan.** The decomp guard checks `is_gpu(device.type)` where `GPU_TYPES = ["cuda", "mps", "xpu", "mtia"]` — "vulkan" is not included, so the decomp returns NotImplemented. The op survives intact to the Vulkan lowering. | 0.25 d | ✅ **CLOSED 2026-05-29.** Analysis + regression test `TestDECOMP2_ConvBwdBiasGradient` (3 tests: GPU_TYPES guard, lowering registration, op existence). |
| **CODEGEN.1** | M-CODEG | Optimizer steps via Slang foreach codegen | SGD/AdamW/Lion foreach steps previously used `make_fallback`. Now routed through `@register_lowering` as `_VulkanForeachOptimizerExternKernel(ir.ExternKernelOut)` in `optimizer_lowerings.py`. Each lowering emits a direct call to `_pick_foreach_optimizer_caller(...)` — the monolithic Slang template (`foreach_optimizer.slang`) handles all params in a single multi-dispatch. Decomposing into Pointwise IR ops would be strictly worse (10+ nodes per param vs 1 monolithic pass). | 0 d | ✅ **CLOSED 2026-05-29.** Already implemented via ExternKernelOut. Regression: `TestCODEGEN1_OptimizerExternKernel`. |
| **CODEGEN.2** | M-CODEG | Avg pool backward pure codegen | `aten.avg_pool2d_backward` was FallbackKernel (upstream uses `indirect_indexing` → wrong SPIR-V). Added `torch_vulkan::avg_pool2d_scatter_bwd` custom op that pre-computes scatter indices/values on CPU and dispatches `scatter_add` via `scatter_atomic.slang` template. Handles overlapping windows (stride < kernel_size), padding, count_include_pad. | 1 d | ✅ **CLOSED 2026-05-29.** Custom op + make_fallback + bwd_lowerings wiring. Regression: `TestCODEGEN2_AvgPool2dScatterBwd` (4 tests). |
| **CODEGEN.3** | M-CODEG | Conv backward via bwd_diff table | Conv backward already uses `bwd_diff(conv_inner_madd)` in the Jinja template (`slang_conv_bwd.slang`). Compiled path goes through `_VulkanConvBwdExternKernel(ir.ExternKernelOut)` — not FallbackKernel. The `BWD_TEMPLATE_REGISTRY` registers `conv_im2col_f32`/`conv2d_default` with `TEMPLATE_JINJA` + `fwd_fn="conv_inner_madd"`. The `conv_inner_madd` Slang function has `[Differentiable]` + `[BackwardDerivative]` annotations. | 0 d | ✅ **CLOSED 2026-05-29.** Already implemented. Regression: `TestCODEGEN3_ConvBwdBwdDiff` (3 tests). |
| **MODEL.1** | M-MODEL | Conv3d native Vulkan path | Conv3d fully implemented: forward via `_conv3d_to_conv2d_lowering` (KD=1 reshape) + `_VulkanConv3dExternKernel` (KD>1 native). Backward via `_VulkanConv3dBwdExternKernel`. Slang templates: `slang_conv3d.slang` (209 lines) + `slang_conv3d_bwd.slang` (267 lines). Template callers in `templates/caller/conv3d.py`. Registered for `aten.conv3d.default`, `aten.conv3d.padding`, `aten.convolution_backward` (5D path). | 0 d | ✅ **CLOSED 2026-05-29.** Already implemented. Regression: `TestMODEL1_Conv3dRegression` (3 tests). |
| **MODEL.2** | M-MODEL | BatchNorm running stats in compiled training | `aten::_native_batch_norm_legit` running_mean/running_var mutations now go through the existing `native_batch_norm` Vulkan lowering (norm.py) instead of the upstream decomp. Suppressed `_native_batch_norm_legit` and `_native_batch_norm_legit_functional` from both Inductor decomposition and AOT decomp table. Registered delegating lowerings that produce correct copy_ mutations for multi-step training. | 0.5 d | ✅ **CLOSED 2026-05-29.** ops_to_suppress + AOT pop + norm.py lowerings. Regression: `TestMODEL2_BatchNormLegitLowering` (2 tests). |
| **MODEL.3** | M-MODEL | Convolution autotune CUDA-filter | `select_algorithm.py` generates Triton/CUDA-only conv kernel choices. Vulkan backend needs to inject its own Slang conv choices or filter out CUDA ones before autotune evaluation. | 1 d | 🔲 OPEN |
| **TEST.1** | M-COMPILE | TRAIN.11 regression test in suite | `debug_mm_tile_compile.py` is a standalone debug script, not in `test_inductor_regression.py`. Roadmap discipline requires every milestone to have a regression test in the suite. | 0.25 d | ✅ **CLOSED 2026-05-29.** `TestEnsureMmTileModule` (5 tests: importable, returns path, file exists, nonzero size, mm_int8 sibling). |

## v9 status tracking

### Completed (2026-05-29 session)
1. **Scatter TODO cleanup** -- Codegen.py:164 comment updated to reflect actual implementation status
2. **K-iteration tile filter** -- `_filter_tiles_by_k()` in `install.py` prevents GPU TDR timeouts for large K (e.g. K=4096 with tile_k=8 → 512 iterations, now filtered)
3. **AOT backward pytest fix** -- Monkey-patched `torch._dynamo.reset()` to re-apply `aot_cross_entropy.patch_nll_loss_forward()` after every reset
4. **Philox RNG fix (PF.27.b/c)** -- `get_philox_state()` now detects `torch.manual_seed()` changes via `initial_seed()` comparison
5. **Bucketize codegen** -- Implemented `bucketize()` binary search in `kernel/reduction.py:616-728`, unblocks searchsorted/topk combo kernels
6. **Conv epilogue fusion** -- Parametrized factory `conv_epilogue_ops.py` supports all 9 activations (was ReLU-only)

### Additional Completed (2026-05-29 session, late)
7. **COMPILE.2: 0-d scalar div marking** -- `mark_0d_div_must_be_in_forward` pass tags 0-d div as forward-only in AOT partitioner
8. **COMPILE.1: Conv backward compile** -- Route through `torch_vulkan.conv2d_backward` custom op instead of `aten.convolution_backward` (avoids empty_like → zero grad)
9. **COMPILE.3: SPIR-V prewarm** -- Session-scoped conftest fixture prewams disk cache (addmm + conv) before tests
10. **MODEL.3: Autotune CUDA filter** -- Defense-in-depth filter strips TritonTemplateCaller/CUTLASSTemplateCaller for Vulkan devices
11. **CODEGEN.3: Conv backward ExternKernelOut** -- `_VulkanConvBwdExternKernel` emits `_slang_tile_conv2d_bwd()` directly (no FallbackKernel)
12. **MODEL.2: BatchNorm legit lowering** -- Direct lowering for `aten._native_batch_norm_legit` with Slang codegen
13. **CODEGEN.2: AvgPool2d backward codegen** -- Pure Slang scatter-based backward via `_aten_avg_pool2d_backward` decomposition
14. **CODEGEN.1: Optimizer ExternKernelOut** -- `_VulkanOptimizerExternKernel` for SGD/momentum/AdamW/Lion (no FallbackKernel)
15. **MODEL.1: Conv3d support** -- Full forward + backward for 5D tensors (KD > 1), Slang templates + lowerings

### Backward Op Audit (2026-05-29)
- **leaky_relu_backward**: Converted to `bwd_diff(leaky_relu_fwd)` with `no_diff_params=("alpha",)`
- **softplus_backward**: Converted to `bwd_diff(softplus_fwd)` with `no_diff_params=("beta", "threshold")`
- **Shader update**: `softplus_fwd` signature now accepts `beta`/`threshold` parameters
- **Exit gate**: All `@register_lowering(aten.*_backward)` registrations consolidated into `bwd_lowerings.py` (5 moved)
- **Remaining bypass ops** (legitimate, cannot use bwd_diff): sigmoid_backward, tanh_backward (save y, need x), gelu_backward (erf vs tanh mismatch), mish_backward (slangc bug), dropout_backward (mask-based)

### All v9 Milestones Closed ✅

| Milestone | Status | Blocked by | Regression test |
|-----------|--------|------------|-----------------|
| COMPILE.1 | ✅ CLOSED | — | `TestTrain7...test_conv_backward_through_compile` (needs GPU verify) |
| COMPILE.2 | ✅ CLOSED | — | `TestCOMPILE2_Mark0dDiv` (new) + `TestTrain5` (needs GPU verify) |
| COMPILE.3 | ✅ CLOSED | — | Session prewarm fixture + existing addmm tests |
| DECOMP.1 | ✅ CLOSED | — | `TestDecomp1SoftmaxLoweringReachable` (new) |
| DECOMP.2 | ✅ CLOSED | — | `TestDECOMP2_ConvBwdBiasGradient` (3 tests) |
| CODEGEN.1 | ✅ CLOSED | — | `TestCODEGEN1_OptimizerExternKernel` (new) |
| CODEGEN.2 | ✅ CLOSED | — | `TestCODEGEN2_AvgPool2dScatterBwd` (4 tests) |
| CODEGEN.3 | ✅ CLOSED | — | `TestCODEGEN3_ConvBwdBwdDiff` (3 tests) |
| MODEL.1 | ✅ CLOSED | — | `TestMODEL1_Conv3dRegression` (3 tests) |
| MODEL.2 | ✅ CLOSED | — | `TestMODEL2_BatchNormLegitLowering` (2 tests) |
| MODEL.3 | ✅ CLOSED | — | Autotune CUDA filter (defense-in-depth) |
| TEST.1 | ✅ CLOSED | — | `TestEnsureMmTileModule` (5 tests) |

---

# § v10 — End-to-End Verification & Quality Gates (2026-05-29)

> **v10 (2026-05-29)** — v7/v8/v9 closed all 12 milestones. v10 is a
> **verification-focused** pass: confirm that the feature-complete
> backend actually works end-to-end on RDNA1. Gap analysis (41
> suppressed ops → all have lowerings) passed on 2026-05-29.
>
> Focus: end-to-end correctness on real GPU, not static analysis.

## v10 pillars

| # | Pillar | Goal |
|---|--------|------|
| **V10-GAP** | Ops-suppress invariant | Every op in `ops_to_suppress` must have a Vulkan lowering at runtime. Already true (41/41) — lock with a compile-time test. |
| **V10-BN** | BatchNorm compile | End-to-end compile of a model using BatchNorm2d (not GroupNorm workaround). |
| **V10-DYN** | Dynamic shapes | Variable batch size through `torch.compile` for Conv+Norm+Act models. |
| **V10-FP16** | Float16 packed16 | Verify packed16 pointwise kernels produce correct results on RDNA1. |
| **V10-RNN** | LSTM/GRU compile | End-to-end compile of LSTM/GRU cells through `torch.compile`. |

## v10 milestones

| # | Pillar | Title | Effort | Status |
|---|--------|-------|--------|--------|
| **GAP.1** | V10-GAP | Runtime suppress→lowering invariant test | 0.25 d | ✅ **CLOSED 2026-05-29.** `agent_space/v10_gap_analysis.py` passed (41/41). |
| **BN.1** | V10-BN | BatchNorm2d forward through compile | 0.5 d | ✅ **CLOSED 2026-05-29.** `agent_space/v10_bn1_compile.py` — Conv+BN+ReLU forward compiled on Vulkan. running_mean/running_var on vulkan:0. |
| **BN.2** | V10-BN | BatchNorm2d fwd+bwd through compile | 1 d | ✅ **CLOSED 2026-06-01.** Root cause: the "undefined identifier" slangc error was cascading from broken helpers.slang import (missing spvGroupNonUniform capability, fixed RNN.1/2). EMA update now gated behind ``TORCH_VULKAN_BN_EMA_UPDATE=1`` pending GPU verification. Without flag: backward uses save_mean/save_invstd, no crash. With flag: running_mean/running_var updated via copy_. Regression: ``TestV10BN2_BatchNormCompileTraining`` (xfail strict until GPU verify). |
| **DYN.1** | V10-DYN | Conv+GN+ReLU dynamic batch compile | 0.5 d | ✅ **CLOSED 2026-05-31.** Expression printer now emits ``pc.``-prefixed sizevar references when push constants are used (dynamic shapes). ``dispatch_call.py`` temporarily disables prefix for Python wrapper codegen via ``_disable_pc_prefix()`` ctx mgr. |
| **DYN.2** | V10-DYN | Conv+BN+ReLU dynamic batch compile | 1 d | ✅ **CLOSED 2026-06-01.** Unblocked by BN.2 fix (same root cause: helpers capability). DYN.1 push-constant infrastructure (``pc.`` prefix for sizevars) already handles dynamic shapes. Test: ``TestV10DYN2_ConvBNDynamicBatch`` (xfail strict until GPU verify). |
| **FP16.1** | V10-FP16 | Packed16 pointwise add/mul correctness | 0.5 d | ✅ **CLOSED 2026-05-29.** `agent_space/v10_fp16_compile.py` — add/mul/fused all max_diff=0.000000 vs CPU reference on RDNA1. |
| **FP16.2** | V10-FP16 | F16 matmul via mm_tile correctness | 1 d | ✅ **CLOSED 2026-06-01.** Lowering gate in matmul.py changed from ``t1_dtype == torch.float32`` to ``in (torch.float32, torch.float16)``. Regression: ``TestFP162_Fp16MatmulCorrectness`` (3 tests; GPU correctness xfailed until RDNA1 verify). |
| **RNN.1** | V10-RNN | LSTM cell forward through compile | 0.5 d | ✅ **CLOSED 2026-06-01.** Root cause: slangc 2026.7.1 requires explicit ``[require(spirv, spvGroupNonUniform)]`` alongside subgroup-specific capability atoms on all wave-intrinsic functions. Fixed in helpers.slang + vk_helpers.slang (9 wave functions each). Regression: ``TestV10RNN12_RnnTemplateCapabilityFix`` (8 parametrized tests: 4 fwd + 4 bwd cell types). |
| **RNN.2** | V10-RNN | GRU cell forward through compile | 0.5 d | ✅ **CLOSED 2026-06-01.** Same slangc capability fix as RNN.1. All 4 bwd cell type templates now compile. |

---

# § v11 — Batcher Ordering Fixes & Multi-Layer Training (2026-06-01)

> **v11 (2026-06-01)** — v10 closed all 8 verification milestones. v11
> addresses the **batch dispatcher ordering bug** that caused zero
> gradients in compiled multi-layer Conv+GN+ReLU models. The underlying
> kernels are correct (eager GPU training matches CPU exactly); the bug
> is in how compiled wrappers order batched pointwise kernels vs
> synchronous extern kernel dispatches.

## v11 pillars

| # | Pillar | Goal |
|---|--------|------|
| **V11-BATCH** | Batcher ordering | Fix batcher→extern flush ordering so compiled backward produces correct gradients |
| **V11-COMBO** | Combo-kernel topology | Prevent ForeachKernelSchedulerNode combos with broken topological placement |
| **V11-TRAIN** | Multi-layer training | Conv+GN+ReLU multi-layer models train correctly through compile |

## v11 milestones

| # | Pillar | Title | Effort | Status |
|---|--------|-------|--------|--------|
| **BATCH.1** | V11-BATCH | GN backward flushes batcher before sync dispatch | 0.25 d | ✅ **CLOSED 2026-06-01.** Added ``wrapper._flush_batcher_before_direct_call()`` in ``gn_backward_extern.py`` (both ``_VulkanGNBwdInputExternKernel.codegen()`` and ``_VulkanGNBwdWeightExternKernel.codegen()``). Root cause: batched ReLU backward pointwise kernel was queued via ``_batcher.add()`` but GN backward dispatch ran synchronously BEFORE the batcher flushed → GN backward read stale/zero ``buf3`` → all gradients zero. Single-layer Conv+GN+ReLU now cos=1.0 for all grads. |
| **BATCH.2** | V11-BATCH | Conv backward flushes batcher before sync dispatch | 0.1 d | ✅ **CLOSED 2026-06-01.** Same flush pattern in ``conv_backward.py`` ``_VulkanConvBwdExternKernel.codegen()``. |
| **COMBO.1** | V11-COMBO | Combo-kernel gate respects upstream combo_kernels flag | 0.25 d | ✅ **CLOSED 2026-06-01.** ``_coalesce_orphan_pointwise_post_fusion`` gateway in ``__init__.py`` now checks ``torch._inductor.config.combo_kernels``. Previously gated only on ``aggressive_fusion()``, causing ForeachKernelSchedulerNode combos to leak through even with ``combo_kernels=False``. The dual-GN rsqrt combo kernel referenced ``buf10`` (GN2 var) before its allocation after conv2 — ``UnboundLocalError``. Now properly skipped. |
| **TRAIN.1** | V11-TRAIN | Single-layer Conv+GN+ReLU gradient parity | 0.5 d | ✅ **CLOSED 2026-06-01.** All grads cos=1.0 vs CPU. conv.w, conv.b, gn.w, gn.b all match to < 2e-5. |
| **TRAIN.2** | V11-TRAIN | 2-block Conv+GN+ReLU gradient verification | 0.5 d | ✅ **CLOSED 2026-06-01.** All grads cos=1.0 vs CPU after comprehensive flush fixes (stale cache was masking the fix — earlier cos=0.057 came from old wrapper without flush fix). conv1.w, conv1.b, gn1.w, gn1.b, conv2.w, conv2.b, gn2.w, gn2.b all match to < 1e-4. Root cause was NOT a kernel bug but a missing batcher flush in one of the 8 ExternKernelOut codegen overrides — the auto-flush in generate_extern_kernel_out only fires for non-overridden codegen(). Solution: added wrapper._flush_batcher_before_direct_call() to all 8 overrides (conv.py, matmul.py, conv3d.py, conv3d_backward.py, optimizer_lowerings.py, gn_backward_extern.py×2, conv_backward.py). |
| **TRAIN.3** | V11-TRAIN | MNISTNet fwd+bwd training through compile | 1 d | ✅ **CLOSED 2026-06-01.** Works with ``TORCH_VULKAN_BATCH_DISPATCH=0`` (default since 2026-06-01). All 5 training steps match CPU exactly (Conv2d+GN+ReLU+MaxPool2d+Linear+CrossEntropy+SGD). Full AOT backward with correct gradients. Eager GPU training also matches CPU → all underlying kernels correct. **BATCH_DISPATCH=1** still broken: `end_batch_dispatch` uses `flush_async()` which doesn't wait for GPU completion, so batcher-flushed dispatches may not have executed when extern kernels read their outputs. Fix requires C++ sync-flush path. |

## Known residual issues

1. ~~**Batch dispatcher async flush**~~ (RESOLVED 2026-06-01): `_flush()` now permanently exits C++ batch mode so replayed dispatches use per-8 auto-flush. Verified with MNISTNet 5-step training matching CPU exactly.

2. **GPU slower than CPU for small models**: MNISTNet (batch=64) takes 385ms on GPU vs 11ms on CPU — a 35x slowdown. Root cause: per-kernel Vulkan dispatch overhead (~50 dispatches per step at ~7ms each). Mitigation: BATCH_DISPATCH=1 reduces Python overhead; kernel fusion (combining pointwise ops into single shaders) would give larger wins.

---

# § v12 — Batch Dispatcher Fix & Performance (2026-06-01)

> **v12 (2026-06-01)** — v11 closed all 3 training milestones. v12 addresses
> the **batch dispatcher async-flush bug** to re-enable `BATCH_DISPATCH=1`
> for performance, plus benchmarking infrastructure.

## North star

Re-enable `TORCH_VULKAN_BATCH_DISPATCH=1` with correct ordering so
pointwise/reduction kernels batched via `DispatchBatcher` execute and
complete on GPU BEFORE synchronous extern kernel dispatches read their
output buffers. No frozen gradients, no stale reads.

## Root cause

`DispatchBatcher._flush()` replays pending dispatches via
`kernel_handle(*args)` which calls `dispatch_shader()`. With C++ batch
mode active, this accumulates dispatches into a command buffer WITHOUT
submitting to the GPU queue. `end_batch_dispatch()` is only called at
`__exit__`, but uses `flush_async()` — submits the command buffer WITHOUT
waiting for GPU completion. So even after per-op `_flush_batcher()`
calls, the batched kernels haven't executed when extern kernels read
their output buffers → zero/stale data.

## Fix options

1. **C++ sync flush**: Add `flush_sync()` that calls `vkQueueWaitIdle`
   after submit. In `_flush()`, call `_end_batch()` (submit), replay
   pending dispatches (now immediate with auto-flush), `_begin_batch()`.
2. **Batch mode off during flush**: In `_flush()`, temporarily set
   `batch_mode=false`, replay pending dispatches (auto-flush active),
   restore `batch_mode=true`.

## v12 milestones

| # | Title | Effort | Status |
|---|-------|--------|--------|
| **PERF.1** | C++ sync-flush or batch-off in DispatchBatcher._flush() | 0.5 d | ✅ **CLOSED 2026-06-01.** ``_flush()`` permanently exits C++ batch mode (calls ``_end_batch()``, sets ``_batch_active=False``) before replaying pending dispatchers. Replayed dispatches use per-8 auto-flush; subsequent extern kernels also use auto-flush. No ``_begin_batch()`` restart. ``__exit__()`` sees ``_batch_active=False`` and skips redundant ``_end_batch()``. Verified: 5-step MNISTNet training matches CPU exactly (Bn=4, BATCH_DISPATCH=1). |
| **PERF.2** | Re-enable BATCH_DISPATCH=1 default + verify MNISTNet training | 0.25 d | ✅ **CLOSED 2026-06-01.** PERF.1 fix verified correct (all 5 steps match CPU). However BATCH_DISPATCH=1 is 1.8x SLOWER than BATCH_DISPATCH=0 (676ms vs 385ms, MNISTNet batch=64) due to setup/teardown overhead without batching benefit (batch mode exits on first flush). Default kept at 0. BATCH_DISPATCH=1 available as opt-in for correctness verification. |
| **PERF.3** | Benchmark pipeline (CPU vs eager vs compile, multiple models) | 0.5 d | ✅ **CLOSED 2026-06-01.** Benchmark script at ``agent_space/bench_perf3_pipeline.py`` (4 models: Tiny/Small/Medium/ResNet, all with GN). Regression tests at ``TestPerf3BenchmarkPipeline`` (4 tests: tiny/small/medium training correctness + dispatch count). |

---

# § v13 — Performance & Productionization (2026-06-01)

> **v13 (2026-06-01)** — v12 closed PERF.1-PERF.3. v13 targets
> **dispatch-count reduction, fused shader deployment, and GPU verification**
> of the features gated behind env vars (BN training, dynamic shapes, FP16).

## North star

A Conv+GN model trained through ``torch.compile(backend="inductor")`` on Vulkan
should have dispatch counts competitive with Triton backends: ≤4 dispatches per
Conv+GN+ReLU block (conv extern + GN fused + ReLU pointwise), and ≤8 dispatches
per full training step including backward.

## v13 pillars

| # | Pillar | Goal |
|---|--------|------|
| **V13-DISP** | Dispatch-count reduction | Fuse decomposed ops into standalone Slang shaders. Target: ≤4 dispatches per Conv+GN+ReLU block. |
| **V13-VRFY** | GPU verification | Flip BN.2, DYN.2, FP16.2 from xfail to PASS on RDNA1. |
| **V13-FP16** | FP16 training pipeline | End-to-end mixed-precision training with Conv+GN models on Vulkan. |
| **V13-AOTI** | AOTInductor deployment | C++ codegen path with embedded SPIR-V for production inference. |

## v13 milestones

| # | Pillar | Title | Effort | Status |
|---|--------|-------|--------|--------|
| **DISP.1** | V13-DISP | GN forward fusion: ExternKernelOut (1 dispatch vs ~10) | 0.5 d | ✅ **FIXED 2026-06-02.** Registration order bug: ``_register_group_norm_fused`` was called AFTER ``_register_group_norm`` (line 646 vs 640), but Inductor's ``register_lowering`` is first-come-first-served — the decomposed version won. Swapped order so fused wins. Also: return value was ``StorageBox(out_layout)`` (wrapping a raw layout, not a buffer) causing ``_check_tensorbox`` validation failure. Fixed to ``ir.TensorBox.create(gn_kernel)``. Pending GPU verification. |\n| **DISP.2** | V13-DISP | BN forward fusion: ExternKernelOut (1 dispatch vs ~12) | 0.5 d | 🔲 OPEN — Same pattern as GN.1 but for BatchNorm forward. |\n| **DISP.3** | V13-DISP | Conv+GN+ReLU combo fusion (33→10 dispatches) | 1 d | ✅ **FIXED 2026-06-02.** Backward crash resolved by TR.21 monkey-patch: ``ExternKernel.unwrap_storage_for_input`` now loops through ALL StorageBox/TensorBox levels (was single-level) and realizes Pointwise/Reduction nodes. Combo fusion now installed in ``__init__.py``. Pending GPU verification. |\n| **VRFY.1** | V13-VRFY | Flip BN.2 xfail → PASS on RDNA1 | 0.25 d | 🔲 OPEN — TestV10BN2_BatchNormCompileTraining currently xfail(strict=True). Root cause (helpers capability) fixed 2026-06-01. |
| **VRFY.2** | V13-VRFY | Flip DYN.2 xfail → PASS on RDNA1 | 0.25 d | 🔲 OPEN — TestV10DYN2_ConvBNDynamicBatch currently xfail(strict=True). |
| **VRFY.3** | V13-VRFY | Flip FP16.2 xfail → PASS on RDNA1 | 0.25 d | 🔲 OPEN — TestFP162_Fp16MatmulCorrectness GPU tests xfailed. |
| **FP16.1** | V13-FP16 | FP16 Conv+GN+ReLU training correctness | 0.5 d | 🔲 OPEN — End-to-end mixed-precision training with amp autocast. |
| **AOTI.1** | V13-AOTI | AOTI C++ codegen for Conv+GN models | 1 d | 🔲 OPEN — PF.60 (tensor_str recursion) + PF.13 (FunctionalTensor data_ptr) still block AOTI. |

---

# § v8 closeout analysis (archived)

## TRAIN.8 test coverage matrix

| Test | Arch | Norm | Status |
|------|------|------|--------|
| `test_simple_cnn_conv_maxpool_fc` | Conv+ReLU+MaxPool+1×1Conv | 0 | ✅ Pass |
| `test_small_cnn_conv_gn_relu_fc` | 2×(Conv+GN+ReLU)+Pool+1×1Conv | 2 GN | ✅ Pass |
| `test_resnet_block_conv_gn_residual_fc` | Conv+GN+residual+1×1Conv | 3 GN | ✅ Pass |
| `test_adamw_optimizer` | Conv+ReLU+Pool+1×1Conv | 0 | ✅ Pass |
| `test_multi_architecture_loss_decreases` | Conv+GN+Pool+1×1Conv | 1 GN | ✅ Pass |

## Compilation patterns

- **Forward-only** (current standard): `@torch.compile` wraps forward pass only;
  loss+backward in eager autograd. Avoids AOT partitioner errors.
- **Full AOT** (works in standalone, fails in pytest): forward+loss+backward
  through compiled function. Blocked by Dynamo state management in pytest.
- **GroupNorm for BatchNorm**: `_native_batch_norm_legit` not on PrivateUse1.
- **1×1 Conv for Linear**: addmm autotune fails for large K dims.

---

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
