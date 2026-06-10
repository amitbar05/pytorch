# Vulkan-Slang Inductor Backend — Slang-Smart AOT Roadmap v16

> **v16 (2026-06-10)** — replaces `docs/15-inductor-backend.md`. Based on
> exhaustive template-loader analysis: every `.slang` file was diffed against
> its `.py.jinja` counterpart, `_load_slang_template` resolution order was
> verified, and every `env.from_string(src)` call site was audited.

---

# § 1 — Slang Usage Analysis (Ground Truth)

## 1.1 The `_load_slang_template` resolution rule

```python
# vulkan_template.py:68
for fname in (f"{name}.slang", f"{name}.py.jinja", f"{name}.jinja"):
    ...  # first match wins
if loaded_ext == ".slang":
    src = _unwrap_slang_template(src)  # strip /*{{*/ /*}}*/ wrappers
```

**`.slang` always wins when it exists.** The `/*{{*/ expr /*}}DEFAULT*/`
convention keeps the raw file passing `slangc --syntax-check` standalone.
`_unwrap_slang_template` converts those wrappers to standard `{{ expr }}`
before Jinja2 sees them.

## 1.2 Per-template actual source of truth

| Template | `.slang` exists? | `.py.jinja` exists? | Caller error msg says `.py.jinja`? | Caller renders Jinja vars? | Source of truth |
|---|---|---|---|---|---|
| `slang_conv2d` | ✅ 9,143B | ❌ | ❌ ("slang_conv2d.slang") | tile sizes only | `.slang` ✅ clean |
| `slang_conv3d` | ✅ 7,372B | ❌ | ❌ | tile sizes only | `.slang` ✅ clean |
| `slang_mm` | ✅ 15,934B | ❌ | ❌ | tile sizes + epilogue flags | `.slang` ✅ clean |
| `slang_mm_bwd` | ✅ 11,883B | ❌ | ❌ | tile sizes only | `.slang` ✅ clean |
| `conv_gn_relu` | ✅ 11,762B | ❌ | ❌ | none (empty render) | `.slang` ✅ clean |
| `slang_conv_bwd` | ✅ 11,882B | ✅ 9,146B | ❌ ("slang_conv_bwd.slang") | `has_bias` via `{% if %}` | `.slang` ⚠️ has_bias not yet Slang interface |
| `fft_stockham` | ✅ 6,540B | ✅ 6,080B | ❌ ("fft_stockham.py.jinja") | `N, half_N, log2_N, n_threads, dir_sign, norm_scale` | `.slang` (but caller msg stale) |
| `flash_attention` | ✅ 9,935B | ✅ 7,502B | ❌ ("flash_attention.py.jinja") | `head_dim, head_layout, is_causal, num_stages, wg_size, output_dtype, num_queries_per_block` | `.slang` (caller msg stale) |
| `flash_attention_bwd` | ✅ 18,594B | ✅ 12,680B | ❌ ("flash_attention_bwd.py.jinja") | `head_dim, head_layout, is_causal, wg_size, BQ, BK, num_stages, causal_diagonal_offset` | `.slang` (caller msg stale) |
| `foreach_optimizer` | ✅ 10,891B | ✅ 9,488B | ❌ ("foreach_optimizer.py.jinja") | `algorithm, batch_size, output_dtype, parameter_array` | `.slang` (caller msg stale) |
| `philox_rng` | ✅ 4,886B | ✅ 3,898B | ❌ ("philox_rng.py.jinja") | `output_dtype, rng_mode, fused_dropout, num_outputs` | `.slang` (caller msg stale) |
| `scatter_atomic` | ✅ 9,230B | ✅ 5,526B | ❌ ("scatter_atomic.py.jinja") | `dim_size, block_size, elem_type` | `.slang` (caller msg stale) |
| `rnn_cell` | ✅ 9,275B | ✅ 5,734B | ❌ ("rnn_cell.py.jinja") | `hidden_size, input_size, has_bias, direction` | `.slang` (caller msg stale) |
| `rnn_cell_bwd` | ✅ 13,934B | ✅ 8,442B | ❌ ("rnn_cell.py.jinja" typo?) | `hidden_size, input_size, has_bias, direction, num_layers` | `.slang` (caller msg stale) |
| `rnn_cell_fused` | ❌ | ✅ 8,972B | ❌ | `hidden_size, input_size, has_bias, direction, num_steps` | **`.py.jinja` only — NOT migrated** |
| `persistent_pointwise` | ❌ | ✅ 4,944B | ❌ | `num_threads, elem_type, num_outputs` | **`.py.jinja` only — NOT migrated** |

## 1.3 What is already Slang-smart

| Smart feature | Templates that have it | Templates that don't |
|---|---|---|
| `ParameterBlock<KernelArgs>` push-constant struct | ALL 15 `.slang` files | — |
| `[[vk::constant_id(N)]]` spec-constants for tile sizes | All except `foreach_optimizer`, `persistent_pointwise` | `foreach_optimizer` (uses `{{ batch_size }}` as array bound), `persistent_pointwise` |
| `[Differentiable]` on fwd entry points | All active templates | — |
| `[BackwardDerivative]` on bwd entry points | All bwd templates | — |
| Runtime gating (`stride_bias != 0`) instead of `#ifdef` | `slang_conv2d` ✅ | `slang_conv_bwd` ❌ (uses `{% if has_bias %}`) |
| Slang `interface` generics for structural parameters | `slang_conv2d` ✅ | All others ❌ |
| Jinja only for spec-constant tile tunables | `slang_conv2d`, `slang_conv3d`, `slang_mm`, `slang_mm_bwd` ✅ | `foreach_optimizer` (algorithm), `flash_attention*` (wg_size), `rnn_cell*` (direction) |
| Pre-slangc AST validator (8 passes, 670 LOC) | ✅ | Missing: spec-constant count, bwd_diff signature matching |

## 1.4 What needs work (prioritized)

### Already smart — no code change needed (but error messages are stale)
These 12 templates all load from `.slang` today. The callers pass
`tmpl.render(vars...)` for spec-constant tile sizes (legitimate use of
Jinja for autotune). The `.py.jinja` files are dead weight — they exist
on disk but are never read. **Only the error messages in callers say
`.py.jinja`** — misleading but functionally harmless.

**Cleanup**: Delete the 10 stale `.py.jinja` files, update error messages
to say `.slang`. No shader logic changes.

### P0 — slang_conv_bwd: `has_bias` is Jinja, not Slang interface
`slang_conv_bwd.slang:86-103` has:
```slang
/*{%*/ if has_bias /*%}*/
    RWStructuredBuffer<uint> grad_bias;
/*{%*/ endif /*%}*/
```
`slang_conv2d.slang` eliminated this pattern — bias is always in
`ParameterBlock`, gated at runtime by `stride_bias != 0`. The backward
shader should do the same: always include `grad_bias` in the struct,
accumulate only when `args.stride_bias > 0`.

**Impact**: Eliminates the last `{% if %}` structural branch in the conv
pipeline. The `.slang` file then becomes fully structural (no compile-time
branching on has_bias).

### P1 — `foreach_optimizer`: algorithm is Jinja, not Slang interface
`foreach_optimizer.slang:50-86` selects between SGD/AdamW/Lion via
`{% if algorithm == "adamw" %}` Jinja branches. This creates 3 separate
SPIR-V modules. Slang `interface IOptimizer { ... }` generic would let
the same module serve all three, with algorithm selected by a single
spec-constant `ALGORITHM_ID` (0=SGD, 1=AdamW, 2=Lion).

**Impact**: Reduces shader variants from 3→1. All three algorithms share
the same kernel structure (param_buffer, grad_buffer, state_buffer);
only the update math differs.

### P2 — `flash_attention*`: wg_size/head_layout/causal is Jinja
`flash_attention.slang` and `flash_attention_bwd.slang` each have 6-8
Jinja variables (`wg_size`, `head_layout`, `is_causal`, `num_stages`,
`output_dtype`, `BQ`, `BK`). These produce N×M shader variants. Some
are legitimately runtime (is_causal, num_stages) but wg_size and head_layout
could be spec-constants.

**Impact**: Medium. Reduces variant count but not correctness.

### P2 — `rnn_cell*`: direction/num_layers is Jinja
`rnn_cell.slang`, `rnn_cell_bwd.slang`, `rnn_cell_fused.py.jinja` have
`{% if direction == "bidirectional" %}` branches. These are compile-time
decisions (model architecture is fixed at trace time), so Jinja is
defensible here — but Slang interfaces would be cleaner.

### P3 — `persistent_pointwise` and `rnn_cell_fused`: no `.slang` exists
These two are still pure `.py.jinja` templates. They need `.slang`
counterparts created before they can participate in any smart-feature
migration.

---

# § 2 — AOT Backend: What "Smart and Optimized" Means

The current AOTI path (`CompiledAOTI` → `AOTIModelContainerRunner`) is
architecturally sound but has 3 blocking correctness gaps and several
performance gaps:

## 2.1 Correctness blockers (must fix before AOT trains)

| Blocker | Root cause | Fix |
|---|---|---|
| `undefined symbol: aoti_torch_empty_strided_vulkan` | `VulkanCppWrapperGpu` missing `extra_objects=["csrc/backend/aoti_shims.o"]` | Add to `CppExtension` in `cpp_wrapper_gpu.py` |
| `timeout 120 → RC=124` at exit | `DeviceRuntime` destructor waits on slangc thread pools | `shutdown(wait=False)` + `vkQueueWaitIdle` timeout |
| `_register_device_module` hangs forever | `meta_patches/shape_ops.py` calls `torch.ops.aten.*` at import time during FakeTensorMode init | Lazy-init guard |

## 2.2 Performance gaps (fix after correctness)

| Gap | Current | Target | Fix |
|---|---|---|---|
| One SPIR-V module per (dtype, has_bias, algorithm) config | N variants per template | 1 variant + spec-constants | Slang `interface` generics |
| Batcher flushes on first dependency | 1.8× slower than unbatch | Parity (1.0×) | `ready_set` per batch |
| No pipeline overlap (compile serialized) | slangc runs after dispatch | Overlap compile + dispatch | Double-buffer + async compile queue |
| Recompilation on every shape change | No shape-class bucketing | Cache by (shape_class, dtype, layout) | Shape canonicalization in `template_registry.py` |
| No subgroup ops | scalar reductions only | `subgroupReduce` for 4-8× speedup | `vk::subgroup` decorations in `shaders/lib/reduce.slang` |
| No auto-tune for Slang templates | Manual tile sizes | Inductor `tuned_config` + `autotune` | Register `VulkanTemplateKernel` choices in `tuned_mm`/`tuned_conv` |

---

# § 3 — Overhauled v16 Roadmap

## v16 pillars

| # | Pillar | Goal | Effort |
|---|--------|------|--------|
| **V16-AOTI** | AOTI correctness | `.so` links, runs Conv+GN fwd+bwd, clean exit. | 2.5 d |
| **V16-LANG** | Slang smart features | Zero `.py.jinja` files, all templates use Slang interfaces for structural params, validator complete. | 2 d |
| **V16-PERF** | AOT performance | Batcher parallel, pipeline overlap, shape bucketing, subgroup ops. | 2 d |
| **V16-TUNE** | Autotune | Inductor autotune wired to Slang templates. | 1.5 d |
| **V16-REG** | Regression lock | Full AOT Conv+GN training test suite, gradient parity, perf baselines. | 0.5 d |

**Total: 8.5 working days.**

## v16 milestones (execution order)

### M1 — AOTI.1 (0.5 d) — link aoti_shims.o into wrapper .so
**Action**: In `cpp_wrapper_gpu.py`, override the `CppExtension` build to
include `csrc/backend/aoti_shims.o` in `extra_objects`. The shim object
provides `aoti_torch_empty_strided_vulkan`, `aoti_torch_zeros_vulkan`,
`aoti_torch_ones_vulkan`, `aoti_torch_full_vulkan`,
`aoti_torch_as_strided_vulkan`, `aoti_torch_delete`,
`aoti_torch_vulkan_mm_out` as C symbols.

**File**: `python/torch_vulkan/inductor/cpp_wrapper_gpu.py:76-170`

**Verification**: `TestAOTI_WrapperLinksVulkanRuntime` — load minimal wrapper
`.so`, assert `aoti_torch_empty_strided_vulkan` resolves via `ctypes.CDLL`.

### M2 — AOTI.4 (0.5 d) — fix import hang
**Action**: Wrap `meta_patches/__init__.py` in lazy-init. Add
`_META_PATCHES_READY = False` at module level; each submodule's top-level
`torch.ops.aten.*` call guarded by `if _META_PATCHES_READY:`. Set flag to
`True` after `_register()` completes.

**Files**: `meta_patches/__init__.py`, `meta_patches/shape_ops.py`,
`meta_patches/decomposition_passes.py`

**Verification**: `TestV16AOTI4_RegisterNoHang` — timed import (5s timeout).

### M3 — AOTI.3 (0.5 d) — clean exit
**Action**: In `DeviceRuntime` destructor:
1. `executor.shutdown(wait=False)` on all compile/slangc pools.
2. `vkQueueWaitIdle` with 5-second timeout.
3. Log warning and proceed on timeout.

**Files**: `csrc/backend/DeviceRuntime.cpp`,
`python/torch_vulkan/inductor/runtime/common.py:200-379`

**Verification**: `TestV16AOTI3_CleanExit` — subprocess with `timeout 60`,
assert `returncode == 0`.

### M4 — AOTI.2 (1 d) — Conv+GN AOT training E2E (AFTER M1+M2+M3)
**Action**: 5-step Conv+GN+SGD training loop via AOTI C++ wrapper path.
Compare all 5 gradient tensors to CPU eager baseline. Threshold: max diff
< 1e-4.

**Prerequisites**: M1 (link fix), M2 (no import hang), M3 (clean exit).

**Regression**: `TestAOTI_ConvGnTraining` — forward parity, backward parity,
5-step loss monotonic decrease.

### M5 — LANG.1 (0.5 d) — eliminate last Jinja structural branch (has_bias)
**Action**: In `slang_conv_bwd.slang:86-103`:
1. Remove `{% if has_bias %}` / `{% endif %}`.
2. Always include `grad_bias` in `KernelArgs` push-constant struct.
3. Gate accumulation at runtime: `if (args.stride_bias > 0) { vk_atomic_add(...) }`.
4. Remove `has_bias` from `tmpl.render()` in `conv.py:459-464`.

**Files**: `templates/slang_conv_bwd.slang`, `templates/caller/conv.py`

**Verification**: `TestV16LANG1_ConvBwdNoJinja` — compile both fwd and bwd
with has_bias=True and has_bias=False, assert gradients match CPU.

### M6 — LANG.2 (0.25 d) — delete stale .py.jinja files + fix error messages
**Action**: Delete these 10 files (they are never loaded — `.slang` always
wins):
```
templates/fft_stockham.py.jinja
templates/flash_attention.py.jinja
templates/flash_attention_bwd.py.jinja
templates/foreach_optimizer.py.jinja
templates/philox_rng.py.jinja
templates/rnn_cell.py.jinja
templates/rnn_cell_bwd.py.jinja
templates/scatter_atomic.py.jinja
templates/slang_conv_bwd.py.jinja
```

Update error messages in callers:
- `caller/conv.py:457` — "slang_conv_bwd.py.jinja" → "slang_conv_bwd.slang"
- `caller/conv3d.py:220` — "slang_conv3d_bwd.py.jinja" → "slang_conv3d_bwd.slang"
- `caller/fft.py:67` — "fft_stockham.py.jinja" → "fft_stockham.slang"
- `caller/flash_attn.py:85` — "flash_attention.py.jinja" → "flash_attention.slang"
- `caller/flash_attn.py:402` — "flash_attention_bwd.py.jinja" → "flash_attention_bwd.slang"
- `caller/optimizer.py:78` — "foreach_optimizer.py.jinja" → "foreach_optimizer.slang"
- `caller/rng.py:52` — "philox_rng.py.jinja" → "philox_rng.slang"
- `caller/rnn.py:59` — "rnn_cell.py.jinja" → "rnn_cell.slang"
- `caller/rnn.py:360` — "rnn_cell_fused.py.jinja" → "rnn_cell_fused.slang" (after M7)
- `caller/scatter.py:64` — "scatter_atomic.py.jinja" → "scatter_atomic.slang"

**Verification**: `TestV16LANG2_NoPyJinja` — asserts `find templates/ -name
"*.py.jinja"` returns empty.

### M7 — LANG.3 (0.5 d) — migrate rnn_cell_fused + persistent_pointwise to .slang
**Action**: Create `rnn_cell_fused.slang` from `rnn_cell_fused.py.jinja`
using `/*{{*/ var /*}}*/` convention for spec-constant defaults. Create
`persistent_pointwise.slang` from `persistent_pointwise.py.jinja` same way.

Update callers to load `.slang` (already handled by `_load_slang_template`
preference — just fix error messages).

**Files**: new `templates/rnn_cell_fused.slang`, new
`templates/persistent_pointwise.slang`, `caller/rnn.py`, `caller/scatter.py`

**Verification**: `TestV16LANG3_PointwiseAndRnnFused` — runs a small RNN
and pointwise reduction, compares output to eager.

### M8 — LANG.4 (0.5 d) — Slang interface for foreach_optimizer algorithm
**Action**: Add `interface IOptimizerAlgorithm` to
`foreach_optimizer.slang`:
```slang
interface IOptimizerAlgorithm {
    uint algorithm_id;  // 0=SGD, 1=AdamW, 2=Lion
}
```
Replace all `{% if algorithm == "adamw" %}` Jinja branches with a switch
on `algorithm_id`. The caller passes `algorithm_id` as a
`[[vk::constant_id]]` spec-constant instead of a Jinja variable.

**Files**: `templates/foreach_optimizer.slang`, `caller/optimizer.py`

**Verification**: `TestV16LANG4_OptimizerInterface` — compile once, dispatch
SGD/AdamW/Lion by setting different spec-constant values, compare to eager.

### M9 — LANG.5 (0.25 d) — Slang interface for flash_attention wg_size
**Action**: Add `interface IFlashAttnConfig` with `uint wg_size`,
`uint BQ`, `uint BK`, `uint num_stages`. Replace Jinja `wg_size` variable
with spec-constants. Keep `is_causal` and `head_layout` as Jinja (they
control code structure, not just numeric values).

**Files**: `templates/flash_attention.slang`,
`templates/flash_attention_bwd.slang`, `caller/flash_attn.py`

**Verification**: `TestV16LANG5_FlashAttnSpecConst` — same SPIR-V module
dispatched with 2 different wg_size values.

### M10 — LANG.6 (0.25 d) — Slang interface for rnn_cell direction
**Action**: Add `interface IRnnDirection { uint is_bidirectional; }` to
`rnn_cell.slang`, `rnn_cell_bwd.slang`. Replace `{% if direction ==
"bidirectional" %}` with runtime branch on `args.is_bidirectional`.

**Files**: `templates/rnn_cell.slang`, `templates/rnn_cell_bwd.slang`,
`caller/rnn.py`

### M11 — LANG.7 (0.25 d) — extend AST validator
**Action**: Add `slang_validate/spec_constants.py` pass: count
`[[vk::constant_id(N)]]` usages, assert N is in valid range (0-63) and
count matches `ParameterBlock` field count. Extend
`slang_validate/bwd_diff_scan.py` to verify `[BackwardDerivative]`
entry points have matching parameter count/types with their forward
counterparts.

**Files**: `slang_validate/spec_constants.py` (new),
`slang_validate/bwd_diff_scan.py`

**Verification**: `TestV16LANG7_Validator` — feed shader with mismatched
spec-constant count, assert `RuntimeError`.

### M12 — PERF.1 (0.5 d) — batcher flush accumulation
**Action**: Replace per-kernel flush with per-batch flush. Track
`ready_set: set[str]` per batch; a kernel is added to the current batch
if all its tensor dependencies are already in the batch (or produced by
kernels already in the batch). Flush only when no more kernels can be
added.

**File**: `python/torch_vulkan/inductor/runtime/batcher.py`

**Target**: `BATCH_DISPATCH=1` ≤ 1.1× `BATCH_DISPATCH=0` (was 1.8×).

**Verification**: `TestV16PERF1_BatchPerf` — MNISTNet, assert batch dispatch
overhead ≤ 10%.

### M13 — PERF.2 (0.5 d) — async compile pipeline overlap
**Action**: Add a double-buffer compile queue: while kernel N is executing
on GPU, compile kernel N+2 in a background thread. Requires:
1. Pre-allocate SPIR-V module slots (max 8 per graph).
2. `slangc` runs in a dedicated `ThreadPoolExecutor(max_workers=2)`.
3. Dispatch uses `vkCmdDispatch` with already-compiled pipeline; if not
   ready, falls back to synchronous compile (rare cold path).

**Files**: `runtime/batcher.py`, `runtime/slangc.py`,
`csrc/backend/DeviceRuntime.cpp`

**Target**: 15-20% wall-time reduction on graphs with >10 kernels.

### M14 — PERF.3 (0.5 d) — shape bucketing in template registry
**Action**: Canonicalize tensor shapes to (rank, dtype, layout_class,
stride_class) before template selection. Cache compiled templates keyed by
this canonical shape. Same shape class reuses cached SPIR-V module.

**File**: `kernel/template_registry.py`

**Target**: Eliminate redundant `slangc` invocations for same-class shapes.

### M15 — PERF.4 (0.5 d) — subgroup reduction ops
**Action**: Add `subgroupReduce(add)`, `subgroupReduce(mul)`,
`subgroupReduce(min/max)` to `shaders/lib/reduce.slang`. Use `[[vk::subgroup]]`
decorations. Wire up in `bwd_diff_table.py` for sum/loss backward.

**File**: `python/torch_vulkan/inductor/templates/shaders/lib/reduce.slang` (verify path)

**Target**: 4-8× speedup on reduction-heavy ops (sum, mean, norm).

### M16 — TUNE.1 (1 d) — Inductor autotune for Slang templates
**Action**: Register `VulkanTemplateKernel` choices for conv2d, mm, and
flash_attention in the Inductor `tuned_*` hooks. Provide 3-5 tile-size
configs per kernel (small/medium/large). Autotune benchmarks on RDNA1
and caches best config.

**Files**: `kernel/template_registry.py`, add `tuned_conv2d`,
`tuned_mm`, `tuned_flash_attention` hooks.

### M17 — TUNE.2 (0.5 d) — persistent kernel for large reductions
**Action**: For reductions where `numel > 65536`, switch to a persistent
thread model (loop over chunks in the same workgroup). Requires
`persistent_pointwise.slang` template (already in LANG.3).

**Files**: `templates/persistent_pointwise.slang`, `bwd_diff_table.py`

### M18 — REG (0.5 d) — regression lock
Add all milestone tests to `tests/test_inductor_regression.py`. Run full
suite on RDNA1. Assert:
- `TestAOTI_ConvGnTraining` — 5-step gradients < 1e-4
- `TestAOTI_CleanExit` — subprocess returns 0
- `TestV16LANG2_NoPyJinja` — zero `.py.jinja` files
- `TestV16LANG1_ConvBwdNoJinja` — has_bias compile + gradient parity
- `TestV16PERF1_BatchPerf` — batch overhead ≤ 10%
- `TestV16LANG4_OptimizerInterface` — SGD/AdamW/Lion via spec-consts

---

# § 4 — Dependency Graph (v16)

```
AOTI pillar:
  M1 (link aoti_shims.o)
    → M2 (import hang)
    → M3 (atexit hang)
    → M4 (E2E training test)

LANG pillar (DONE: M5, M6):
  M5 ✅ conv_bwd de-Jinja (stride_grad_bias runtime gate)
    → M6 ✅ delete 9 stale .py.jinja files + fix error messages
      → M7 (rnn_cell_fused + persistent_pointwise .slang)
        → M8 (foreach_optimizer interface)
          → M9 (flash_attention interface)
            → M10 (rnn_cell direction interface)
              → M11 (AST validator extension)

M7-M11 can proceed in parallel with AOTI M1-M4 (independent code paths).

Performance pillar (parallel):
  M12 (batcher ready_set) ──→ M13 (async compile double-buffer) ──→ M14 (shape bucketing)
  M15 (subgroup ops) ← after M12
  M16 (autotune) ← independent, parallel with M12-M15
  M17 (persistent kernel) ← after M7
```

Parallel streams:
- **AOTI correctness**: M1, M2, M3 in parallel → M4 → M18
- **Slang smart** (M5-M6 ✅): M7/M8/M9/M10/M11 (M8-M10 can parallelize)
- **Performance**: M12 → M13 → M14 → M15 in series; M16 parallel with M12; M17 after M7
- **Regression**: M18 after all correctness milestones pass

---

# § 5 — File:line Ownership Map (v16)

| File | Lines | Milestone |
|------|-------|-----------|
| `python/torch_vulkan/inductor/cpp_wrapper_gpu.py` | 76-170 | M1 |
| `python/torch_vulkan/inductor/__init__.py` | 809 | M1 |
| `python/torch_vulkan/inductor/meta_patches/__init__.py` | 1-end | M2 |
| `python/torch_vulkan/inductor/meta_patches/shape_ops.py` | 1-end | M2 |
| `python/torch_vulkan/inductor/meta_patches/decomposition_passes.py` | 1-end | M2 |
| `csrc/backend/DeviceRuntime.cpp` | 1-end | M3 |
| `python/torch_vulkan/inductor/runtime/common.py` | 200-379 | M3 |
| `tests/test_inductor_regression.py` | 62,686 lines | M4, M18 |
| `python/torch_vulkan/inductor/templates/slang_conv_bwd.slang` | 86-103 | M5 |
| `python/torch_vulkan/inductor/templates/caller/conv.py` | 457-464 | M5, M6 |
| `python/torch_vulkan/inductor/templates/caller/conv3d.py` | 220 | M6 |
| `python/torch_vulkan/inductor/templates/caller/fft.py` | 67 | M6 |
| `python/torch_vulkan/inductor/templates/caller/flash_attn.py` | 85, 402 | M6 |
| `python/torch_vulkan/inductor/templates/caller/optimizer.py` | 78 | M6 |
| `python/torch_vulkan/inductor/templates/caller/rng.py` | 52 | M6 |
| `python/torch_vulkan/inductor/templates/caller/rnn.py` | 59, 360 | M6 |
| `python/torch_vulkan/inductor/templates/caller/scatter.py` | 64 | M6 |
| `templates/*.py.jinja` (10 files) | all | M6 (delete) |
| `templates/rnn_cell_fused.slang` | new | M7 |
| `templates/persistent_pointwise.slang` | new | M7 |
| `templates/foreach_optimizer.slang` | 50-86 | M8 |
| `templates/flash_attention.slang` | 1-80 | M9 |
| `templates/flash_attention_bwd.slang` | 1-100 | M9 |
| `templates/rnn_cell.slang` | 1-60 | M10 |
| `templates/rnn_cell_bwd.slang` | 1-80 | M10 |
| `slang_validate/spec_constants.py` | new | M11 |
| `slang_validate/bwd_diff_scan.py` | 1-67 | M11 |
| `python/torch_vulkan/inductor/runtime/batcher.py` | 1-end | M12, M13 |
| `python/torch_vulkan/inductor/runtime/slangc.py` | 1-end | M13 |
| `python/torch_vulkan/inductor/kernel/template_registry.py` | 1-end | M14, M16 |
| `templates/shaders/lib/reduce.slang` | verify path | M15 |
| `python/torch_vulkan/inductor/bwd_diff_table.py` | 1-end | M17 |
| `python/torch_vulkan/inductor/templates/persistent_pointwise.slang` | new | M17 |

---

# § 6 — Anti-goals (unchanged)

1. No new model-specific `.slang` files — templates only.
2. No new `aten.<op>_backward` lowerings — backward routes through
   `bwd_diff_table.py` → Slang `[BackwardDerivative]`.
3. No symptom-fixes in `meta_patches/`.
4. No CPU fallbacks on the compile path.
5. No file in `python/torch_vulkan/inductor/` exceeds 800 lines.
6. No Jinja string substitution for parameters that Slang interfaces +
   spec-constants + ParameterBlock can handle.

---

*v16 created 2026-06-10. Replaces v15 as active plan. v7-v15 remain
closed and reference-only.*
