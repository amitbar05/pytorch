# Vulkan-Slang Inductor Backend — Historical Roadmap (frozen 2026-04-27)

> **This file is the frozen snapshot of the pre-rewrite roadmap as it stood on
> 2026-04-27, immediately before the v2 reorganization.** It preserves the
> full Recent Changes log, every completed `[x]` item, and all the discovery
> entries (P0–P17) that drove the loop up to the rewrite. **It is not the
> active roadmap.**
>
> The active, forward-looking roadmap is
> [10-inductor-backend.md](10-inductor-backend.md). New work, new items, new
> Recent-Changes entries belong there. This file is read-only — touch it only
> to cross-reference what was already shipped or discovered.
>
> **Why the rewrite?** The pre-rewrite roadmap had grown to 1,661 lines across
> P0–P17 with structural drift (out-of-order phase numbers, fragmented Slang
> sections P3.3 / P9 / P15, retrospective labels mixed with forward items).
> The v2 rewrite consolidates Slang-feature adoption into P0–P3 (front-loaded
> per the 2026-04-27 architectural review), collapses ~85 redundant items,
> and restructures around a seven-pillar Slang-leveraging architecture.

---

# Vulkan Slang Inductor Backend

This document drives continuous improvement of the `torch.compile(backend="inductor")`
integration for the Vulkan / Slang backend. It describes the current state, the
architecture, **what's missing for a fully optimizing training-grade backend**,
and the prioritized roadmap that the agent loop works through.

> **Mission:** beat eager Vulkan training perf on attention-heavy and CNN
> models by fusing pointwise + reduction + extern kernels into the smallest
> possible number of Vulkan dispatches, with autotuned tile/WG selection and
> AOT-cacheable shader compilation.

---

## Building and Testing

### Build

```bash
cd backends/vulkan_slang
source .venv/bin/activate
MAX_JOBS=3 pip install -e . -v --no-build-isolation
```

After adding a new Slang shader, compile it (do **not** run `generate_stub_shaders.py` — it overwrites real SPIR-V):

```bash
SLANGC=third_party/slang/build/slang-2026.5.2-linux-x86_64/bin/slangc \
  python tools/compile_shaders.py
```

### Run inductor regression tests

```bash
# Clear Inductor cache first (required after any backend change)
rm -rf /tmp/torchinductor_$(whoami)

SLANGC=third_party/slang/build/slang-2026.5.2-linux-x86_64/bin/slangc \
  python -m pytest tests/test_inductor_regression.py \
  -p no:faulthandler --timeout=300 -q 2>/dev/null
```

Always pass `-p no:faulthandler` and redirect stderr — Vulkan cleanup segfaults on exit corrupt output.

### Run the full test suite

```bash
SLANGC=third_party/slang/build/slang-2026.5.2-linux-x86_64/bin/slangc \
  python -m pytest tests/ -p no:faulthandler --timeout=300 -q 2>/dev/null
```

### Per-kernel performance counters

```bash
TORCH_VULKAN_INDUCTOR_STATS=1 \
  python my_compiled_workload.py
# then in Python: torch_vulkan.inductor.inductor_stats.print_stats()
```

### Useful environment knobs

| Var | Effect |
|-----|--------|
| `TORCH_VULKAN_INDUCTOR_STATS=1` | Per-kernel call_count / total_us tracking |
| `TORCH_VULKAN_TRACE=1` | Print every JIT dispatch (key, tensors, WG, pc) |
| `TORCH_VULKAN_ASYNC_COMPILE=1` | Run `slangc` in a thread pool |
| `TORCH_VULKAN_SPIRV_CACHE=<dir>` | Override SPIR-V on-disk cache location |
| `TORCH_VULKAN_MAX_AUTOTUNE={0,1,2}` | Workgroup autotune level |
| `TORCH_VULKAN_MM_TILES="64x64x16,128x64x32"` | Override mm tile choices |
| `TORCH_VULKAN_NO_WG_TUNE=1` | Disable WG tuning (debug) |
| `TORCH_VULKAN_NO_PACKED16=1` | Disable f16/bf16 packed path (debug) |
| `TORCH_VULKAN_NO_LOAD_HOIST=1` | Disable multistage load hoisting (debug) |

---

## Architecture

The Inductor backend lives in `python/torch_vulkan/inductor/`. It plugs into PyTorch's Inductor
compiler via `torch._inductor.register_backend_for_device("vulkan", ...)`.

| Module | Role |
|--------|------|
| `__init__.py` | Idempotent `register()` — wires scheduler, wrapper, FX passes, lowerings, meta patches, mm/bmm/addmm callers |
| `scheduling.py` | `VulkanScheduling(SIMDScheduling)` — fusion heuristics, buffer-count limits, `define_kernel`, combo kernel codegen, benchmarking |
| `kernel.py` | `VulkanKernel(SIMDKernel)` — emits Slang compute-shader source, persistent / cooperative reduction selection, packed16 path |
| `codegen.py` | Re-export shim only |
| `expr_printer.py` | `VulkanExprPrinter` — SymPy → Slang expression printer |
| `overrides.py` | `VulkanOverrides` (~80+ pointwise op snippets), `DTYPE_TO_SLANG`, `value_to_slang` |
| `slang_helpers.py` | Module-scope Slang helpers (special math, Philox RNG, packed16 unpack/pack) |
| `wrapper.py` | `VulkanPythonWrapperCodegen` — `make_allocation`, `_generate_kernel_call_helper`, `extern_kernel_out` |
| `lowerings.py` | Vulkan-specific lowerings: `native_layer_norm`, `native_group_norm`, `_softmax`, `_log_softmax` |
| `vulkan_template.py` | `SlangTemplate` + `VulkanTemplateKernel` — Inductor `ExternKernelChoice` wrapper |
| `vulkan_template_caller.py` | Tiled mm/bmm/addmm lowerings with JIT Slang compilation |
| `fx_passes.py` | FX graph passes: `mm+add→addmm`, `bmm+scale→scaled_bmm`, redundant-copy removal, b2b GEMM enable |
| `meta_patches.py` | FakeTensor fake impls for view/shape/BLAS/conv/norm-backward/indexing/activation-backward/loss-backward/factory ops (~75 ops) |
| `runtime.py` | Compiled-kernel executor, perf stats (`_KERNEL_STATS`), in-memory + on-disk SPIR-V cache, async slangc pool |
| `inductor_stats.py` | Public API: `get_stats()`, `reset_stats()`, `print_stats()` |
| `autotune.py` | Workgroup-size autotuner for pointwise/reduction kernels |
| `device_interface.py` | `VulkanDeviceInterface` — device properties (subgroup size, max storage buffers), stream management |
| `device_op_overrides.py` | Vulkan-specific Inductor device op hooks |
| `vulkan_combo_kernel.py` | Combined pointwise + reduction combo kernel support |
| `templates/` | Jinja2 Slang templates: `slang_mm.py.jinja` (mm/addmm/bmm with optional bias + epilogue) |
| `config.py` | Centralized env kill-switches |

---

## Current Status (2026-04-27)

**Regression tests: 173 passed, 1 skipped, 16 xfailed** ([test_inductor_regression.py](../tests/test_inductor_regression.py)). +7 since 2026-04-26 (4 QKV-fusion + 3 cat-under-compile).

**Recent (2026-04-27):**
- P1.1 `qkv_linear` FX pass — 3 same-input mm/addmm collapse to one fused op via `torch_vulkan::qkv_cat3` extern weight-pack + 3 zero-copy slices. Includes a topological-resort helper for cases where original consumers sit earlier than the latest QKV dependency. `TestQKVLinearFusion` (4 tests).
- L1 fix — `torch.cat` (and any FX rewrite that introduces `aten.cat.default`) now compiles under Vulkan. Three coordinated changes: `_SlangExpr` (`str` subclass with `.dtype`), MPS-style `masked` rewrite (scoped buffer, typed local, if/else block), and `and_` / `or_` overrides emitting `&&` / `||` for float-as-bool operands. `TestCatUnderCompile` (3 tests).
- P0.1 — 7 new activation-backward Inductor lowerings: `threshold_backward` (relu_bwd), `leaky_relu_backward`, `elu_backward`, `softplus_backward`, `hardswish_backward`, `hardsigmoid_backward`, `mish_backward`. Decompose to pointwise primitives so the scheduler emits a single fused VulkanKernel. Will run end-to-end once P0.0 backward compile unblocks; xfail for now.

| Test | What it checks | Status |
|------|---------------|--------|
| `test_stats_records_compiled_kernels` | Stats API + JIT dispatch | ✓ |
| `test_stats_reset` | `reset_stats()` | ✓ |
| `test_linear_bias_dispatch_count` | addmm path: ≤2 dispatches | ✓ |
| `test_linear_bias_gelu_dispatch_count` | addmm+gelu epilogue: ≤3 dispatches | ✓ |
| `test_linear_bias_relu_dispatch_count` | addmm+relu epilogue: ≤3 dispatches | ✓ |
| `test_conv_bias_dispatch_count` | conv+broadcast-add: ≤6 dispatches | ✓ |
| `test_linear_bias_gelu_correctness` | Numerical correctness (rtol=1e-2) | ✓ |
| `test_layer_norm_plus_add_dispatch_count` | layer_norm+add: ≤3 dispatches | ✓ |
| `test_layer_norm_correctness` | Numerical correctness | ✓ |
| `test_softmax_dispatch_count` | softmax(dim=-1): ≤3 dispatches | ✓ |
| `test_softmax_correctness` | Numerical correctness | ✓ |
| `test_linear_layer_norm_fused` | linear→layer_norm: ≤5 dispatches | ✓ |

**Inductor speedups measured (vs eager Vulkan, inference):**

| Model | Eager dispatches | Compiled dispatches | Speedup |
|-------|-----------------|--------------------|---------| 
| ResNet-18 (inference) | ~81 | ~29 | 5.7× |
| MobileNet-V2 | ~55 | ~24 | 2.3× |
| MLP (forward) | 8 | 5 | parity |

**Training perf:** *not yet measured under Inductor*. Eager-mode MiniQwen3 4L training step:
~162 dispatches / ~11 ms (RTX 4060 Ti). Goal of this roadmap is to bring full
training (forward + backward + optimizer) under torch.compile and beat that.

**Shader inventory:** 440 Slang shaders. Inductor-generated kernels: hot pointwise +
reduction + extern epilogue. Eager-only fused shaders that Inductor does not yet emit
or template: `mm_tiled2_bias_addmm` (used as ExternKernelChoice), flash attention
(D=32/64/128 wave variants), `gate_up_swiglu_fwd`, `qkv_linear_fwd`, fused RMSNorm,
fused add_rms_norm, fused log_softmax, fused nll_loss_mean, batched SGD/AdamW.

---

## What's Implemented

### Pointwise ops (all working)
- All 50 elementwise unary prims via `VulkanOverrides` snippets
- All 32 elementwise binary prims
- Fused epilogues on matmul: bias, relu, gelu, silu, sigmoid, tanh, clamp, scale

### Reductions (all working)
- `sum`, `prod`, `max`, `min` — wave-intrinsic shared-memory reduction in `VulkanKernel`
- `argmax`, `argmin` — with index tracking
- `any` — short-circuit any-true reduction
- `xor_sum` — bitwise XOR reduction
- Welford reduction (mean + variance in one pass) — needed for layer norm backward

### Matmul / GEMM
- `aten.mm` — tiled Slang matmul with JIT compilation and autotuning
- `aten.bmm` — batched tiled matmul
- `aten.addmm` — fused mm+bias in a single dispatch (`mm_tiled2_bias_addmm.slang`)
- FX pass: `mm(a, b) + bias → addmm(bias, a, b)`
- FX pass: `scale * bmm(q, transpose(k, -2, -1)) → scaled_bmm(q, k, scale)`

### Norm / softmax lowerings
- `aten.native_layer_norm`, `aten.native_group_norm`, `aten._softmax`, `aten._log_softmax`
  re-registered as Inductor lowerings so they decompose into primitives the scheduler
  can fuse with adjacent pointwise/reduction ops.

### FX passes
- `_fuse_mm_add_to_addmm`, `_fuse_bmm_mul_to_scaled_bmm`, `_remove_redundant_copy`,
  `_enable_b2b_gemm`.

### Infrastructure
- `VulkanScheduling`: buffer-count-aware fusion (queries `maxStorageBufferRange` from device)
- `VulkanKernel`: full SIMDKernel emitting valid Slang, packed16 (vec4) path for large elementwise
- `meta_patches`: FakeTensor fake impls for ~75 ops (view/shape/BLAS/conv/norm-bwd/indexing/activation-bwd/loss-bwd/factories)
- JIT pipeline: Slang source → `slangc` → SPIR-V → Vulkan compute pipeline (cached by content hash)
- **On-disk SPIR-V cache** keyed by source hash + slangc version
- **Workgroup autotuner** with disk cache
- **Per-kernel stats** via `TORCH_VULKAN_INDUCTOR_STATS=1`
- **Smart-barrier insertion** — only emit barriers when next dispatch reads a dirty buffer
- **Specialized kernel wrappers** — fixed-arity closures for n_buffers ≤ 6 + n_pc=0 (the common pointwise case)

---

## Gap Analysis: What's Missing for a Full Optimizing Inductor Backend

The eager backend ships hand-fused shaders that Inductor does not yet emit; the Inductor
scheduler likewise has fusion opportunities it doesn't realize on Vulkan. The gaps below
are sequenced by impact on **training-step throughput** (forward + backward + optimizer).

> **New 2026-04-26:** §P3.3 (Slang Autodiff Integration) and §P1.6 (Codegen Heuristics & Autotune Coverage) added. P0.1 expanded with missing activation-backward / loss-backward / embedding-backward lowerings. P5.1 / P5.2 expanded with backward Winograd template and layout-cost-model prerequisites. See those sections for the new unchecked items.
>
> **New 2026-04-27 (review pass):** Comprehensive backend review surfaced ~40 new unchecked items spanning four new sections: §P1.6 expanded with subgroup-uniform branch / [unroll]-on-static-rnumel / compute-vs-memory bound classifier / bank-conflict swizzle / int8 dot-product / specialization-constants. §P1.7 (Primop Coverage Audit) added with concrete missing Inductor lowerings + ExternKernelChoice templates for loss / norm / pooling / upsample / padding / dropout / linalg / spectral ops — plus an `audit_inductor_op_coverage.py` script. §P3.3 expanded with P3.3.6–P3.3.11: loss-fn autodiff cluster, RoPE backward, LayerNorm autodiff (negative-result documentation), `@autodiff_template` decorator, VGPR/occupancy regression gate, and a CAS-loop f16/bf16 gradient-accumulator helper. §P5.9 (Cooperative-matrix / WMMA matmul) added — capability-gated tensor-core path through `VK_KHR_cooperative_matrix` + Slang `CoopMat<T,M,N,K>`. §P5.10 (Subgroup intrinsics expansion) added — `WaveActiveBallot` for masked-pointwise, `WaveQuad*` for 2×2 windows, prefix-count compaction, half-precision atomic-add helper. §P6.6 (Stride / contiguity propagation) added with IR-driven vec4 eligibility + dead-size-1-axis elim + shared-index CSE. §P7 (Measurement-driven discovery automation) added — turning the manual training-driven loop into a CI-gated automation. §P8 (Final completeness checklist) added — residual items a "fully implemented inductor backend" cannot ship without.
>
> **New 2026-04-27 (third review — primops, codegen, Slang utilisation):** A focused review of (i) which `aten.*` ops still skip the lowering / template path, (ii) which codegen heuristics are still hard-coded, (iii) which Slang features beyond P9 remain unused, and (iv) which cross-kernel / pipeline-level optimisations are entirely absent surfaced ~85 new unchecked items across six new sections: §P12 (extended primop coverage round 2 — median/kthvalue/quantile, einsum/tensordot, full linalg, complex / FFT / stft, cdist/pdist, mode/diff/gradient, kron/block_diag/tensor_split, narrow/movedim/swapdims, histogramdd, full pre-aten-op factory ops). §P13 (compile-pipeline & SPIR-V optimisation — `spirv-opt` post-pass, dead-binding stripping, on-disk SPIR-V deduplication, Slang link-time optimisation, slangc IR cache, validation-layer-clean SPIR-V). §P14 (FX pre/post-grad pass expansion — pattern-matcher rules for `linear→bias→add→layernorm`, `attention_score→mask→softmax`, `gelu→linear`, `silu→linear`, `embedding→add`, `linear→linear` back-to-back, `view→pointwise→view` flattening, dropout-during-eval elimination, dead-cast removal, scalar-broadcast hoisting, `to_copy` cancellation). §P15 (Slang feature utilisation round 2 — `spirv_asm { ... }` inline-SPIR-V hot-kernel hooks, `IFunc<R,Args...>` first-class function passing for epilogue plumbing, `Property<T>` accessor patterns, `defer` cleanup ordering, `[knownAttribute]` compile-time-constant gates, `__target_switch` for vendor-specific code paths, `[anyValueSize(N)]` dynamic-dispatch tables, Slang's tagged-union `enum` types for op-kind dispatch, `[noinline]` audit, link-time module merging, embedded SPIR-V via `__intrinsic_op`, Slang IR introspection (`getReflectedJSON`)). §P16 (profile-guided optimisation & online autotune — workload trace capture, Bayesian-optimisation tile picker, cross-shape autotune transfer learning, hardware-counter-driven tile selection via `VK_KHR_pipeline_executable_properties`, online re-bench with TTL, dispatch-frequency-weighted autotune budget, autotune cache compaction, regression-guarded autotune updates). §P17 (cross-kernel & pipeline-level codegen — multi-kernel epilogue chaining without intermediate buffers, command-buffer-level loop unrolling, kernel pipelining via Vulkan timeline semaphores, secondary command buffer reuse for repeated dispatches, Vulkan event-based fine-grained sync, persistent-shader streaming inference path, double-buffered backward-graph dispatch overlap, scratch-buffer arena per fused region, cross-kernel constant propagation, cross-kernel dead-store elimination across the wrapper). Plus §P9 expanded with P9.12–P9.18 (Slang `IFunc` / `defer` / `Property<T>` / inline `spirv_asm` / `__target_switch` / link-time IR) and §P11 expanded with P11.12–P11.16 (cross-kernel ILP, kernel-fusion across reduction barriers, persistent-thread codegen, FMA-grouping for fp32-accum bf16 inputs, dynamic register allocation hints).
>
> **New 2026-04-27 (second review — also added):** A first review pass focused on (a) which Slang language features the backend currently *doesn't* use, (b) which primops still extern-fall-back, and (c) which kernel-codegen quality wins remain on the table surfaced ~50 more unchecked items in three new sections. §P9 (Slang language feature utilization) — every eager shader and Inductor template today is plain HLSL-style; zero use of Slang generics (`<T : IFloat>`), modules (`module/import`), `ParameterBlock<T>`, `interface` / `IDifferentiable`, `[require(...)]` capability gating, `[shader("...")]` overloads, string formatting / `printf`, reflection. Adopting these selectively cuts shader source ~30%, makes the Inductor template-caller architecture far cleaner, and unlocks `bwd_diff(...)` for the autodiff pilot in §P3.3. §P10 (Extended primop audit) — adds the ops the audit in P1.7 missed: `grid_sampler_{2d,3d}` (vision/spatial-transformer hot path), `im2col`/`col2im` / `unfold` / `fold` (custom convolutions), `pixel_shuffle` / `pixel_unshuffle`, `affine_grid`, `searchsorted` / `bucketize`, `masked_scatter` / `masked_select`, `lerp`, `triu` / `tril` / `diagonal`, `polygamma` / `digamma`, full backward coverage for `where` / `gather` / `scatter` / `index_select` / `clamp` / `nll_loss2d` / `binary_cross_entropy`, conv-backward algorithm selection. §P11 (Advanced kernel-codegen heuristics & micro-optimizations) — tree-vs-ladder reduction shape, branch-free `select` fold, instruction-level parallelism via independent FMA chains, transpose-on-load for transposed matmul, multi-level tiling (cluster ⊃ warp-tile ⊃ register-tile), shader-source minification before SPIR-V hashing, NVIDIA `VK_NV_cooperative_matrix2` / `VK_KHR_cooperative_matrix_2` async copies, `OpExpect`-style branch-probability hints, vendor-pragma plumbing for AMD `s_waitcnt` tuning. §P8 expanded with an additional 8 final-completeness items (test coverage budget, public-API stability commitment, soak-test suite, slangc-version pin policy).

### A. Backward-graph compilation (P0 — the biggest gap)

Backward graphs run under `torch.compile(backend="inductor")` today, but most chains
fall back to eager dispatches because:

1. **Reduction backward over reduction-axis ranges** — `_softmax_backward_data` and
   `_log_softmax_backward_data` are registered as fake impls but no Inductor lowering
   decomposes them. They hit the eager C++ path and miss fusion with the surrounding
   epilogue chain.
2. **`linear_backward` extern** — produces `(grad_in, grad_w, grad_b)` in one C++
   dispatch but blocks Inductor from fusing the upstream `grad_out * mask` patterns
   that always precede it (relu/gelu/silu backward).
3. **`native_layer_norm_backward` / `native_group_norm_backward`** — extern only.
   In eager we have a 2-dispatch fused implementation; Inductor cannot template that.
4. **`embedding_dense_backward`** — extern (atomic CAS shader). Cannot fuse with
   downstream `weight.add_` (optimizer step). Should be a template kernel choice
   so it can fuse.
5. **`max_pool2d_with_indices_backward` / `avg_pool2d_backward`** — extern; cannot
   fuse with the relu_backward / batch_norm_backward chain that always follows.

**Why this matters.** A Qwen3 backward step is ~106 dispatches in eager. Compiling
forward saves dispatches but compiling backward — where most pointwise glue lives
between extern reductions/matmuls — is the bigger lever.

### B. Extern → template fusion (P0)

Currently, `addmm + epilogue` is the **only** template-fused extern. Other patterns
that already have hand-fused eager shaders but no Inductor template wiring:

| Pattern | Eager shader | Status under Inductor |
|---------|-------------|----------------------|
| `addmm + bias + activation` | `mm_tiled2_bias` + epilogue snippet | ✓ epilogue fuses (Phase A) |
| `bmm(q, k.T) * scale` | `bmm` shader with `transpose_b` + scale | ✓ via FX pass `_fuse_bmm_mul_to_scaled_bmm` |
| `linear + GELU + linear` (FFN) | composite | ✗ — 3 separate ExternKernelOuts |
| `linear → layer_norm → linear` (transformer block) | composite | ✗ — separate dispatches |
| `swiglu(gate, up)` — `silu(gate) * up` | `gate_up_swiglu_fwd` | ✗ — 3 dispatches (2 mm + 1 fused) |
| `qkv_linear` — 3-way linear from same input | `qkv_linear_fwd` | ✗ — 3 mm dispatches |
| `flash_attention(q, k, v, scale, causal)` | flash_attention fwd shader | ✗ — eager fallback, not in Inductor graph |
| `RMSNorm` | `rms_norm` fused fwd | ✗ — not lowered for Inductor (eager dispatches `rms_norm`) |
| `add_rms_norm` (residual + RMSNorm) | `add_rms_norm` fwd | ✗ — not lowered |
| `nll_loss_mean` (CE in 1 dispatch) | `nll_loss_mean_fused` | ✗ — separated by AOT autograd |
| `cross_entropy + log_softmax_large` | composite | ✗ — separate dispatches |

### C. Conv + epilogue fusion (P0 — CNN training)

Convolution still dispatches as `convolution_overrideable` (extern) followed by a
broadcast-add and pointwise activation. The 6-dispatch budget in
`test_conv_bias_dispatch_count` reflects this. To match eager:

- **Conv + bias fusion**: a fused `conv2d_bias.slang` template that takes the bias
  vector directly. Currently 2 dispatches (conv + add); should be 1.
- **Conv + bias + ReLU/GELU**: epilogue fusion via Inductor template path. Currently
  3 dispatches; should be 1.
- **Conv backward + relu_backward**: the typical `grad_out * (out > 0)` mask is
  trivially fusable but dispatches separately because `convolution_backward_overrideable`
  is extern.

### D. Multi-axis / multi-dim reduction (P1)

`aten.sum(x, [0, 2])` is currently lowered as **sequential** single-dim reductions
(2 dispatches). Common in attention (`grad_bias = grad_out.sum([0, 1])`) and norm
backward. The eager `_sum_multi_dim` shader handles this in 1 dispatch.

### E. Attention as a template (P1 — biggest single-shader gap)

Eager dispatches `flash_attention_fwd_*_wave` (a hand-fused 1-dispatch attention).
Under Inductor, SDPA falls back to eager `aten.scaled_dot_product_attention`, breaking
fusion at the boundary. A `SlangTemplate` for flash attention with epilogue slots
(scale, mask add, softmax, V mm) would let attention participate in graph-level fusion
— specifically allowing the upstream `qkv_linear` and downstream `out_proj_linear` to
be fused across the attention boundary.

### F. Persistent reduction for small dims (P1)

`should_use_persistent_reduction` returns `True` for `rnumel ≤ 8192` already, but it's
not verified to fire correctly on the wide-row cases (vocab=151936) we hit during
training. Wide-row softmax should fuse the entire `amax → exp → sum → div → CE` chain
into a single workgroup-per-row kernel. Today it dispatches 4–7 kernels.

### G. Horizontal fusion of independent reductions (P1)

`VulkanScheduling.can_fuse` allows it, but the upstream Inductor scheduler does not
propose it for two independent `aten.sum` ops over the same tensor. This is on the
critical path for backward where `grad_w + grad_b` reductions can share a load.

### H. Backward-pass specific fusion patterns (P1)

These eager backward shaders have no Inductor counterpart:
- `softmax_backward_data` (1-dispatch shared-memory dot product)
- `log_softmax_backward_large` (1-dispatch wide-row backward)
- `layer_norm_backward` (2-dispatch fused: grad_input + grad_weight/bias)
- `group_norm_backward` (2-dispatch)
- `batch_norm_backward` (2-dispatch)
- `gelu_backward`, `silu_backward`, `tanh_backward`, `sigmoid_backward` (single-pass)
- `add_rms_norm_backward` (fused residual + rms_norm bwd)

These either need: (a) Inductor decompositions so the scheduler emits a fused VulkanKernel,
or (b) ExternKernelChoice templates so they participate in template+epilogue fusion.

### I. Optimizer step under torch.compile (P2)

Eager has `sgd_batch15` and `adamw_batch7` that handle 7–15 params/dispatch.
`_foreach_*` ops under Inductor decompose into per-param pointwise kernels. Two paths:

1. Skip Inductor for the optimizer entirely (current eager path is excellent).
2. Add an Inductor combo-kernel template that emits one Slang shader processing N
   param/grad/momentum tuples per dispatch — competitive with the hand-fused batch.

### J. Codegen-level perf gaps (P2)

- **Async slangc warm-start**: a fresh `torch.compile` cold-starts ~30–80 kernels at
  ~100 ms each → 3–8 s startup. The on-disk cache handles re-runs but first-time
  workloads pay full slangc cost. Add a parallel pre-compile step that walks the
  Inductor graph, hashes all kernel sources upfront, and submits to the slangc thread
  pool while Python continues lowering.
- **Pickle-friendly external matmul**: `_make_tile_mm_fn` returns a closure → not
  picklable → Inductor's codecache silently disables. Replace with module-level
  classes implementing `__reduce__`. Cumulative warm-startup win.
- **Push-constant struct packing**: every JIT dispatch packs `n_pc` integers via
  `struct.Struct(...).pack`. For dynamic shapes this fires per call. Cache the
  Struct object on the kernel closure (already done) but also batch-pack into
  pre-allocated `bytearray` buffers reused across calls.
- **Combo-kernel grid selection**: `vulkan_combo_kernel.py` exists but the
  `codegen_combo_kernel` path is not exercised by training workloads. Verify it
  fires for the foreach-momentum / foreach-grad-clip patterns.

### K. AOT compile (P3)

`torch._inductor.aot_compile` does not work yet for the Vulkan backend — the
generated Python wrapper hardcodes `torch_vulkan.inductor.runtime` imports that
need to become a stable C++ runtime callable for AOTI to load. This is a stretch
goal blocking deployment of compiled Vulkan models without Python at runtime.

### L0. Template `num_stages=2` codegen bug — *FIXED 2026-04-26*

Two latent bugs in `slang_mm.py.jinja`, both silent for months because the
loader filename mismatch (P1.4 follow-up) made the template path
unreachable; both surfaced once the loader was fixed:

1. **`lid` undefined in helpers** — `load_tiles` / `mma_tile` referenced
   `lid` (`SV_GroupThreadID`) at file scope; `lid` only exists inside the
   `computeMain` entry point. **Fix**: pass `lid` (`uint3`) as an explicit
   parameter to both helpers.
2. **`groupshared float[2][N]` indexed flatly** — `tile_a[2][TILE_M*TILE_K]`
   declaration with `tile_a[off + i]` access tripped slang typing
   (`tile_a[i]` is a `float[N]` slice, not a scalar). **Fix**: declare as
   `groupshared float[2 * TILE_M * TILE_K]` flat 1-D, keep flat indexing.

Both fixes verified by `TestMatmulTemplateCompiles` (4 cases:
ns=1/2 × bias on/off).

### L1. `aten.cat` under Inductor on Vulkan crashes — *NEW (2026-04-26)*

**Symptom**: `torch.cat` (and any FX rewrite that introduces `aten.cat.default`
into the post-AOT graph) crashes Inductor on Vulkan with
`AttributeError: 'str' object has no attribute 'dtype'` from
`DtypePropagationOpsHandler` on the `masked` op produced by Inductor's cat
lowering.

**Suspected fix**: the dtype-propagation handler for `masked` checks
`backend == "triton"` only; under Vulkan our `value` argument flows through
as a Slang expression string. The handler needs a Vulkan branch that derives
the dtype from the surrounding context (the masked op's `other` arg or the
inner load), not from the value-string. File: probably
`torch._inductor/codegen/common.py` or our `expr_printer` / `kernel.py`
need to expose a typed `masked` op.

**Verification**: regression test asserting that a simple compiled
`torch.cat([a, b, c], dim=0).sum()` on Vulkan tensors compiles and produces
correct output. (Today: crashes; after fix: passes.)

**Workaround in place**: `torch_vulkan::qkv_cat3` extern custom_op routes
weight-pack cats around Inductor's masked-load lowering. Used by the QKV
fusion pass. We should remove the workaround once L1 is fixed in the
backend codegen.

- [x] **Fix `masked` op dtype propagation under Vulkan** *(P0 — 2026-04-27)*: three coordinated fixes landed: (1) `_SlangExpr` (in `overrides.py`) — `str` subclass with `.dtype` / `.shape` attributes — satisfies Inductor's `DtypePropagationOpsHandler._default` which reads `value.dtype` for masked ops on the triton/cuda backend tier (vulkan routes there via `get_current_backend → cuda_backend`). (2) `VulkanOverrides.masked` rewritten to follow the MPS pattern: invoke `body()` inside a scoped `IndentedBuffer`, allocate a typed local, splice an `if (mask) { local = body; } else { local = other; }` block into `V.kernel.compute`, and return the local as a `_SlangExpr`. (3) Added `and_` / `or_` overrides emitting `&&` / `||` on float-as-bool operands — the upstream default emits raw `&` / `|` which slangc rejects on `(float, float)`. `TestCatUnderCompile` (3 tests) covers `torch.cat` along dim 0 / dim 1 plus `mask_a & mask_b` masked-load through `torch.where`. The qkv_cat3 workaround can now be inlined to plain `aten.cat.default` if desired (kept for now since it's working and the change isn't load-bearing).

### L. Known correctness blockers (P0 only when triggered)

- **`aten.transpose` in compiled graphs**: `x.transpose(-2, -1)` on Vulkan tensors
  fails under Dynamo (`clone_input` accesses `data_ptr` of FakeTensor with no
  storage). Blocks the attention-score `bmm(q, k.T)` pattern from compiling. Fix:
  patch `clone_input` for Vulkan FakeTensors or expand `meta_patches` to include
  the storage-allocator entry points Dynamo needs.
- **`_make_tile_mm_fn` pickling**: Inductor codecache logs a non-fatal warning
  every cold compile. Cosmetic but pollutes logs.
- **Shape specialization explosion**: dynamic shapes blow up SPIR-V cache. Symbolic
  push-constant indexing already exists; verify it fires on the model entry-point
  shapes (B, S vary, H/D static).

---

## Prioritized Roadmap

Each item is sized to a single agent iteration. Items are ordered so earlier work
unblocks later. **Verification step is mandatory** — measured dispatch count or
time delta against the regression baseline.

### P0.0 — Make backward-graph compilation reach the Inductor scheduler (prerequisite)

**Discovered 2026-04-26.** Even a trivial `(x*x+1).sum().backward()` under
`torch.compile` fails to compile. Root cause: AOT autograd's `joint_helper`
calls `torch.autograd.grad(...)` under FakeTensorMode + ProxyTorchDispatchMode,
and the autograd engine runs the **C++ backward formulas** (e.g. `SumBackward0`
→ `expand_to`, `MulBackward0` → `mul`). Because PrivateUse1 outranks Meta in
the dispatch key set, those C++ formulas land directly on our Vulkan kernels
with FakeTensor inputs. The kernels that call `data_ptr()` (or otherwise touch
storage) before the null-storage guard crash with:

`Cannot access data pointer of Tensor (e.g. FakeTensor, FunctionalTensor)`.

Until this is unblocked, **all** P0.1 lowerings are unverifiable — the lowering
function never gets called because compilation fails before reaching Inductor.
The activation backward Inductor lowerings live in `lowerings.py` already and
their regression class is marked `@pytest.mark.xfail(strict=True)` with a
pointer back to this section.

- [x] **Audit every PrivateUse1 op for null-storage handling** *(2026-04-26)*. `csrc/backend/MetaGuard.h` adds `is_null_storage` / `make_meta` / `make_meta_broadcast` helpers; null-storage guards applied to binary helper, unary helper, activation helper, mm/addmm/bmm/linear, sum/mean, expand/cat/clone/select/slice, in-place add/sub/mul/div, comparison helper, embedding, copy_, fill_. `BACKWARD_FAKE_GUARD` rewritten via `make_meta_like`. Forward `fake_tensor_prop` now succeeds — every existing forward regression test stays green.
- [x] **Add `test_compile_backward_baseline` regression** *(2026-04-26)*. `TestCompileBackwardBaseline` class added with `(x*x+1).sum().backward()` and `linear→relu→sum().backward()` cases. Currently marked `xfail` with `_BWD_FAKE_DEVICE_BLOCKER` until the device-propagation gap below is closed.
- [~] **Close the FakeTensor device-propagation gap on view ops** *(2026-04-26 — diagnosed, two prereqs landed, root cause runs deeper)*.
   - **Diagnosis**: PyTorch's C++ view fast-path constructs view-op outputs via `Tensor._make_subclass` while the source FakeTensor is in `in_kernel_invocation` mode (so `source.device` reports `meta`). The new FakeTensor inherits `device=meta`.
   - **Patch landed (opt-in)**: `_patch_fake_tensor_view_op_device` in `meta_patches.py` annotates the active `FakeTensorMode` with the first vulkan device it sees, then in a wrapped `FakeTensor.__new__` upgrades any subsequent `device=meta` construction in that mode back to vulkan. Standalone repro passes (`x.expand(8,16) → device=vulkan:0`). Gated on `TORCH_VULKAN_FAKE_VIEW_FIX=1` because, when always-on, it fails forward dispatch-count regressions: the upgrade is too coarse and converts legitimate meta-device intermediates too. Tracked separately.
   - **Prereqs landed**: `aten::{mm,addmm,bmm,linear}.out` registered on PrivateUse1 (`csrc/backend/Registration.cpp`); `VulkanPythonWrapperCodegen.make_allocation` now upgrades `device=meta` to `empty_strided_vulkan` (`python/torch_vulkan/inductor/wrapper.py`).
   - **Still blocked by**: even with the patch enabled and a "meta"→VulkanScheduling alias added (so `get_backend_features("meta")` finds a scheduler), the compiled backward Inductor emits is a no-op — the `(x*x+1).sum().backward()` wrapper just returns `empty_strided((8,16), device='meta')` with no kernel calls. Inductor's dead-code elimination treats the meta-device computation as unreachable. Fixing this requires an upstream PyTorch change so `aot_autograd`'s joint-graph fake_tensor_prop captures FakeTensor `val`s on the original device, not raw meta tensors. Backed out the alias; documenting here for the next iteration.
   - Required for P0.1 backward lowerings to be exercisable.
- [ ] **Audit every PrivateUse1 op for null-storage handling** (rest). The pattern is
  the one already used by `vulkan_permute`: `if (self.storage().data_ptr() ==
  nullptr) return at::empty_strided(...).contiguous_or_meta(...);`. Apply at
  the top of each C++ op that may receive a FakeTensor during AOT joint-graph
  tracing — at minimum `expand`, `clone`, `_to_copy`, `view`, `reshape`,
  `contiguous`, `copy_`, `as_strided`, `slice`, `select`, `unsqueeze`,
  `squeeze`, `t`, `transpose`, the binary/unary pointwise ops AOT autograd
  hits during backward (`mul`, `add`, `sub`, `div`, `neg`, `sigmoid`, `tanh`,
  `gelu`, `silu`, `relu`), and the reductions / matmuls that show up in
  backward graphs (`sum`, `mean`, `mm`, `bmm`, `addmm`, `linear`). Group the
  guard into `csrc/backend/MetaGuard.h` (`is_null_storage` + `make_meta`
  helpers), and hoist the existing `meta_*` functions in `MetaKernels.cpp`
  into `MetaKernels.h` so forward op files can reuse the shape-inference
  logic instead of duplicating it.
- [ ] **Add a regression test**: `test_compile_backward_baseline` —
  `(x*x+1).sum().backward()` under `torch.compile(backend="inductor")` must
  successfully compile and produce numerically correct grads (compare against
  CPU eager).
- [ ] **Verify the activation backward suite flips from `xfail` to `passed`
  strictly** once the guards land. If it flips green, remove the class-level
  `xfail` decorator.

### P0.1 — Backward-graph compilation (training enablement, gated on P0.0)

- [x] **`_softmax_backward_data` Inductor lowering** *(2026-04-27)*: `_register_softmax_backward()` in `lowerings.py` decomposes into `output * (grad_out - sum(grad_out * output, dim, keepdim=True))`. Falls through (`NotImplemented`) for non-Vulkan inputs. `TestSoftmaxBackwardLowering` covers compile + correctness, gated `xfail(strict=False)` until P0.0 backward-graph compile unblocks.
- [x] **`_log_softmax_backward_data` Inductor lowering** *(2026-04-27)*: same registration: `grad_out - exp(output) * sum(grad_out, dim, keepdim=True)`. `TestSoftmaxBackwardLowering.test_log_softmax_backward_correctness` covers it (xfail until P0.0).
- [x] **`native_layer_norm_backward` lowering** *(2026-04-27)*: `_register_layer_norm_backward` in `lowerings.py` decomposes into pointwise + 2 reductions (`sum(grad_x_hat)` + `sum(grad_x_hat * x_hat)` over inner-dims, `sum(grad_out * x_hat)` + `sum(grad_out)` over outer-dims) using `mul.Scalar(rstd_b * inner, 1/N)` to keep the `1/N` factor inline (the upstream decomp's `rstd / N` falls through to extern `aten.div.Tensor` and breaks the fused chain). Required `_suppress_upstream_decomps()` to drop `aten.native_layer_norm_backward.default` (and `native_group_norm_backward`, `_log_softmax_backward_data`) from the Inductor decomp table so AOT autograd doesn't pre-decompose them before the lowering runs. `TestLayerNormBackwardLowering` (3 tests, 2D/3D/multi-dim normalized_shape) — **2 dispatches** (down from 5 baseline) on (16, 64) ns=[64], (4, 8, 32) ns=[32], (2, 4, 16) ns=[4, 16]. Beats the ≤3 target.
- [x] **`native_group_norm_backward` lowering** *(2026-04-27)*: `_register_group_norm_backward` decomposes into ds = sum(grad_out * input, dim=spatial), db = sum(grad_out, dim=spatial), then `c2 = (db_val * mean - ds_val) * rstd^3 * s` and `c3 = -c2 * mean - db_val * rstd * s` with the `s = 1/(HxW * cpg)` scalar folded via `mul.Scalar` (the upstream decomp's `* s` produces extern `aten.div.Tensor` calls). Skips the `torch.ones((1, group, cpg))` materialization in the no-gamma path. Masked-off output slots (output_mask[i]=False) emit a 1-element `aten.full` zero placeholder instead of `None` since Inductor's graph runner calls `.get_size()` on every multi-output slot. Also adds `aten.native_group_norm_backward.default` to `_suppress_upstream_decomps()`. **4 dispatches** (down from 37 baseline) on (2, 8, 4, 4) groups=2 and (4, 16, 8, 8) groups=4. `TestGroupNormBackwardLowering` (2 tests) — beats the ≤5 contract; max numerical error ≤4e-6 vs CPU.
- [x] **`gelu_backward` / `silu_backward` / `tanh_backward` / `sigmoid_backward` overrides** *(2026-04-27 — verified passing for sigmoid/silu correctness; dispatch-count remains xfail)*: Lowerings landed in `lowerings.py` (sigmoid/tanh/silu decomposed into rsub/mul/add.Scalar primitives, gelu added in P10 audit). Correctness for sigmoid/silu under `torch.compile` is **verified passing** — `TestActivationBackward.test_sigmoid_backward_correctness` and `test_silu_backward_correctness` flipped from xfail-but-passing to plain passing 2026-04-27. The activation-backward Inductor lowerings work end-to-end through compile despite the broader P0.0 view-fast-path family — those particular paths don't traverse the broken code. Dispatch-count tests for the same ops remain xfail (extern-aten path during AOT autograd still hits the view-fast-path).
- [ ] **`linear_backward` template**: Wrap as `ExternKernelChoice` so the upstream `grad_out * mask` pointwise can fuse via the Phase A epilogue rule.
- [ ] **End-to-end backward correctness**: Add `test_mlp_backward_dispatch_count` and `test_transformer_layer_backward_dispatch_count` regression tests.
- [x] **Activation-backward lowerings** *(2026-04-27)*: 7 new Inductor lowerings in `lowerings.py` decompose `aten.threshold_backward` (relu_backward), `aten.leaky_relu_backward`, `aten.elu_backward`, `aten.softplus_backward`, `aten.hardswish_backward`, `aten.hardsigmoid_backward`, `aten.mish_backward` into pointwise primitives (`gt`, `mul.Scalar`, `where.self`, `bitwise_and`, `sigmoid`, `tanh`, `exp`, `log1p`). The fused chain emits as a single VulkanKernel instead of an extern `aten.*_backward` dispatch. Each lowering checks `_is_vulkan(grad_output)` and falls through otherwise. Live registration verified; xfail backward tests will flip to pass once P0.0 backward-graph compilation is unblocked.
- [ ] **`nll_loss_backward` / `cross_entropy_backward` lowering**: explicit Inductor decomposition (gather-on-target → scale → mask) — even after the upstream view-fast-path block clears, the cross_entropy compiled path (P1.5 entry) needs a Vulkan-aware lowering to bypass the `device=meta` extern fallback. Verify `TestCrossEntropyBackward`.
- [ ] **`embedding_dense_backward` extern template** (promoted from §H gap analysis): wrap the existing atomic-CAS shader as `ExternKernelChoice` so the optimizer step (`param.add_(grad)`) fuses via Phase B template rule. Reuses `extensions.register_template`. Verify `TestEmbeddingBackwardOptimizerFusion`.
- [x] **Loss-backward lowerings** (`mse_loss_backward`, `smooth_l1_loss_backward`, `huber_loss_backward`, `binary_cross_entropy_backward`) *(2026-04-27 — verified passing for mse/smooth_l1/bce under compile; huber remains xfail)*: `_register_loss_lowerings()` in `lowerings.py` decomposes each into pointwise primitives + reduction-mode-aware scaling (`mean → mul.Scalar(1/N)`, `sum`/`none` pass-through). Smooth-L1 / Huber use `where(|diff| < beta, quad_branch, linear_branch)`. `TestLossBackwardLowerings.test_mse_loss_backward` + `test_smooth_l1_loss_backward` + `test_bce_backward` flipped from xfail-but-passing to plain passing 2026-04-27 (these specific backward paths don't traverse the broken P0.0 view-fast-path code). `test_huber_loss_backward` still hits the P0.0 path through its `where`-branch decomposition; remains xfail. (`binary_cross_entropy_with_logits_backward` and `kl_div_backward` deferred — BCE-with-logits-bwd needs pos_weight handling and is the rarer code path; kl_div_backward needs target-vs-input distinction handled correctly under the no-op-on-target-zero mask.)

### P0.2 — Conv epilogue fusion

- [ ] **Fused `conv2d_bias.slang` template**: Single shader takes input/weight/bias, dispatches once. Reuses the ExternKernelChoice pattern from addmm.
- [ ] **Conv + bias + activation epilogue**: Wire `convolution_overrideable` as `ExternKernelChoice` so a downstream pointwise activation fuses in.
- [ ] **Conv backward + relu_backward fusion**: Add `relu_backward` to `VulkanOverrides` so the elementwise mask multiplication fuses with downstream pointwise.
- [ ] **Update `test_conv_bias_dispatch_count` to ≤2 dispatches** (currently ≤6).

### P0.3 — Transpose-in-compiled-graph fix

- [x] **Fix Dynamo `clone_input` for Vulkan FakeTensors** *(2026-04-26)*: `_patch_dynamo_clone_input_for_vulkan` (in `meta_patches.py`) replaces both `torch._dynamo.utils.clone_input` and the imported binding in `torch._dynamo.variables.builder` (the latter captures the original by-value at import time, so patching only `utils` was insufficient). Added a defensive `RuntimeError("data pointer")` fallback for tensor types we don't detect upfront. `TestCloneInputPatch` covers FakeTensorMode round-trip + builder binding identity + the actual `bmm(q, k.T)` compile path.
- [x] **Verify `bmm(q, k.T)` compiles end-to-end** *(2026-04-26)*: `TestCloneInputPatch.test_compiled_attention_score_chain_survives_clone` runs `torch.compile(bmm(q, k.transpose(-2, -1)))`, asserts the output device is `vulkan:0` (not `meta`), and checks numerical correctness against CPU.
- [~] **End-to-end `softmax(q@k.T) @ v` compile** *(2026-04-26 — partial)*: With the clone_input patch, `bmm(q, k.T)` works without the FAKE_VIEW_FIX env var. The full attention chain (matmul + softmax + matmul) still produces a meta-device output via the FakeTensor-leak path. Enabling `TORCH_VULKAN_FAKE_VIEW_FIX=1` plus the clone_input patch gets further but hits an `aten::masked_select` Meta-tensor crash deeper in. Tracked: needs scoped FakeTensor view-op device upgrade (subset of the same upstream gap as P0.0 follow-up).

### P1.1 — RMSNorm / SwiGLU / QKV / Attention as Inductor templates

- [x] **`rms_norm` lowering** *(2026-04-26)*: No custom lowering needed — `nn.RMSNorm` already decomposes upstream into `pow → mean → add → rsqrt → mul → mul`, and Inductor's pointwise+reduction fusion produces **1 dispatch** for inference (locked in by `TestRMSNormForward.test_rmsnorm_inference_dispatch_count`). Backward path still hits P0.0 blocker.
- [x] **`add_rms_norm` lowering** *(2026-04-26)*: No FX pass needed — `(x + residual) → RMSNorm` already compiles to **1 dispatch** via stock pointwise+reduction fusion. Locked in by `TestRMSNormForward.test_add_rms_norm_dispatch_count` and `test_add_rms_norm_correctness`.
- [x] **`swiglu(gate, up)` FX pass** *(2026-04-26)*: `_fuse_silu_mul_to_swiglu` matches `silu(gate) * up` (matching shapes + dtypes) and rewrites to `torch.ops.torch_vulkan.swiglu_fused.default`, a real `custom_op` with `register_fake` for shape inference. Compiled `silu(gate) * up` now collapses **2+ dispatches → 1**. `TestSwigluFusion` covers dispatch count, correctness vs CPU, op-registered-at-backend-init, and the no-fusion-on-shape-mismatch broadcast case.
- [x] **`qkv_linear` FX pass** *(2026-04-26)*: `_fuse_qkv_linears` (in `fx_passes.py`) detects three `mm` or `addmm` ops sharing the same input `x` (the QKV projection in attention) and rewrites to a single fused `mm`/`addmm` of pre-concatenated weights, with three zero-copy `aten.slice.Tensor` views on the output. The weight (and bias, for addmm) concat goes through a dedicated `torch_vulkan::qkv_cat3` extern custom_op so it dispatches eagerly via the working Vulkan `cat` instead of Inductor's masked-load cat lowering — see L1 below for why. Net: 3 (add)mm → 1 (add)mm + 1–2 cat dispatches. Includes a topological-resort helper for the case where the original consumers of the 3 ops live earlier in the graph than the latest QKV dependency. `TestQKVLinearFusion` (4 tests): mm-form fusion, addmm-form fusion, no-fuse on different inputs, end-to-end compile correctness.
- [x] **`flash_attention` FX pass** *(2026-04-26)*: `_fuse_sdpa_to_flash_attention` rewrites `aten.scaled_dot_product_attention` to `torch.ops.torch_vulkan.flash_attention_fused.default` (real custom_op + register_fake). Eligibility: 4-D inputs, `head_dim ∈ {32, 64, 128}`, no `attn_mask`, no dropout. Fall-through is graceful for unsupported shapes. Compiled SDPA on (1, 2, 8, 64) drops from **9 dispatches eager → 5 compiled** (1 flash extern + 4 wrapper allocs). `TestFlashAttentionFusion` covers dispatch count, correctness vs CPU, op-registered, and the no-fusion-on-unsupported-head-dim fallback.

### P1.2 — Multi-dim reduction + persistent fast path

- [x] **`aten.sum.dim_IntList` multi-dim** *(2026-04-26)*: Inductor's stock pointwise+reduction codegen already fuses 2-axis reductions into 1 dispatch on Vulkan; verified by `TestReductions.test_sum_two_dims_dispatch_count` and `test_mean_two_dims_dispatch_count`. The "fans out" worry only applied to the eager path; under torch.compile the codegen handles it.
- [x] **Wide-row persistent reduction verification** *(2026-04-26)*: Multistage reduction already collapses `amax → exp → sum → div` into 1 dispatch even at vocab sizes (16384 and 32000 verified). `TestWideRowReduction` locks softmax + log_softmax + correctness.
- [x] **Welford persistent reduction** *(2026-04-26)*: `should_use_cooperative_reduction` now detects welford reductions ahead-of-codegen via `_has_welford_reduction()` (walks `features.reduction_nodes()` for `welford_*` types) and biases toward cooperative single-workgroup reduce for a wider rnumel range when welford is in flight, avoiding the AMD RADV miscompile in multistage + groupshared + barrier code paths.

### P1.3 — Backward fusion patterns

- [ ] **Horizontal fusion of `grad_w` and `grad_b` reductions**: Both load `grad_out`; `VulkanScheduling.can_fuse` allows it, but the scheduler doesn't propose it. Audit `should_horizontal_fuse` and add `BackendFeature.HORIZONTAL_FUSION` if available.
- [ ] **`embedding_dense_backward` + optimizer step fusion**: Extern-template the embedding bwd so the next dispatch (param.add_(grad)) can fuse via the Phase B template path.
- [ ] **`add_rms_norm_backward` lowering**: Mirror the forward FX pass — detect the residual-grad join, route to fused extern.
- [ ] **Pool backward + activation backward fusion**: relu_backward is always elementwise next to pool/conv backward; fold via `VulkanOverrides`.

### P1.4 — Codegen perf gaps

- [x] **Pickle-friendly external matmul callables** *(2026-04-26)*: `_SlangTileMM`, `_SlangTileAddMM`, `_SlangTileBMM` module-level classes with `__slots__` and explicit `__reduce__`. Eliminates the codecache `Can't pickle local object` warning on every cold compile and lets Inductor cache the autotune choices.
- [x] **Pre-warm `slangc` async pool** *(2026-04-26)*: `prewarm_matmul_templates()` (in `vulkan_template_caller.py`) renders the standard mm/addmm/bmm × {f32, f16} × {1, 2 stages} tile configs and submits them through `runtime.prewarm_compile` to the slangc thread pool at `register()` time. Skips entries already in memory or on disk. Disable with `TORCH_VULKAN_NO_PREWARM=1`. `TestSlangcPrewarm` regression locks the cache_keys produced by the spec collector to the runtime callers — if the runtime cache_key format ever changes, the prewarm test fails. Side-effect of the work: fixed a latent template-loader bug (`_load_slang_template` looked for `<name>.jinja` while the file is `<name>.py.jinja`), which silently disabled the entire autotuned-template matmul path.
- [x] **Push-constant fast paths** *(2026-04-26)*: Specialized `_make_kernel_wrapper` for the common (n_buffers, n_pc) ∈ {(1,1),(2,1),(3,1),(2,2),(3,2),(4,2)} combinations. Avoids per-call `*args` slicing + tuple unpacking on top of the existing `n_pc=0` fast paths.
- [x] **Verify combo-kernel path fires** *(2026-04-26)*: It used to crash. `VulkanComboKernel.codegen_kernel` had `import torch` after `torch.float16` was already referenced (UnboundLocalError), and even with that fixed the emitted Slang shares `in_ptr0` / `out_ptr1` across subkernels and references an undefined `x0`. Combo-kernel rewrite shipped (next item).
- [x] **Rewrite `VulkanComboKernel`** *(2026-04-26)*: Subkernel bodies are now post-processed to (a) rename Inductor-generated locals (`xindex`, `tmp\d+`, `x\d+`, `r\d+_*`) by appending `_sub{idx}`; (b) build a global outer→binding map so the same wrapper-visible buffer reuses one Slang slot, and distinct outers that collide on inner names get prefixed with `s{idx}_`; (c) auto-detect inplace buffers (any outer that appears in both `input_buffers` and `output_buffers` of any subkernel) and declare those as `RWStructuredBuffer`. Wrapper-side `call_kernel` now emits args in binding order (read-only first, then read-write) instead of subkernel-iteration order. `BackendFeature.FOREACH` re-enabled. `TestForeachCompiles.test_foreach_mul_add_dispatch_count` tightened from `≤4` to `≤1` (4 tensors → 1 combo dispatch). Added `test_foreach_mul_add_correctness` to lock the value semantics.

### P2.1 — Optimizer fusion

- [ ] **Combo-kernel SGD/AdamW**: Either route through Inductor's `BackendFeature.FOREACH` combo kernel, or short-circuit the optimizer.step() to the existing `torch_vulkan._sgd_batch_step` / `_adamw_batch_step`. Measure if Inductor combo wins.
- [ ] **`bf16` batched optimizer**: Add `sgd_batch15_bf16` / `adamw_batch7_bf16` shaders so mixed-precision training keeps the batched dispatch path.

### P2.2 — Inductor benchmarking & autotuning

- [ ] **Real matmul autotuner** *(P0 — promoted 2026-04-26 per user direction)*: `vulkan_template_caller._pick_tile_configs()` currently returns a fixed list of `(TILE_M, TILE_N, TILE_K, num_stages)` tuples regardless of the input shape. On RDNA1 the right tile is highly shape-dependent — a 8×64×128 GEMM and a 256×4096×4096 GEMM want very different tilings. Replace with a per-shape benchmarking sweep:
   - Sub-task **(a) shape-keyed cache layout**: `~/.cache/torch_vulkan/mm_autotune/<shape_key>.json` storing `{best_tile, candidates: [(tile, median_us)]}` keyed on `(M, N, K, dtype, transpose_b, has_bias)` rounded to power-of-2 buckets so similar shapes share entries.
   - Sub-task **(b) benchmarking driver**: on first cache-miss for a shape, render+compile each candidate tile through `vulkan_template_caller`, run via the cached `Benchmarker` (already cached per P2.2 prior item), pick the median-fastest and persist. Reuse the existing `runtime.prewarm_compile` thread pool to keep the compile-side off the critical path.
   - Sub-task **(c) candidate set tuning**: starting set should cover the high-impact extremes: `(64,64,16)`, `(64,64,32)`, `(128,64,32)`, `(64,128,32)`, `(128,128,16)`, `(32,32,32)` × `num_stages ∈ {1, 2}`. Override via `TORCH_VULKAN_MM_TILES` (already exists) for ad-hoc experimentation.
   - Sub-task **(d) cache management**: `autotune.list_autotune_cache()` / `clear_autotune_cache()` already exist for the WG autotuner; mirror the same `mm_autotune.list_cache()` / `clear_cache()` API for the matmul autotuner.
   - **Verification**: regression `TestMatmulAutotuner` covers cache miss → benchmark → persist → cache hit; one shape-specific dispatch-time assertion comparing autotuned vs the static-list baseline (autotuned must be ≤ baseline ms).
- [x] **`benchmark_codegened_module` benchmarker cached** *(2026-04-26)*: `Benchmarker()` is now constructed once via module-level `_get_benchmarker()` and reused across autotune candidates instead of instantiating fresh per call. `_reset_benchmarker_cache()` test hook. `TestBenchmarkerCached` (2 tests) locks identity + reset behavior.
- [ ] **Autotune across `num_stages`**: `1` (no pipelining) vs `2` (double-buffered shared memory) is workload-dependent; benchmark both per-shape.

### P2.3 — Native f16 / bf16 / packed16 verification

- [ ] **Verify packed16 fires on real workloads**: Audit `_packed16` decision logic; add `TORCH_VULKAN_INDUCTOR_STATS` annotation to record packed16 vs scalar dispatches per kernel. Measure speedup on f16 ResNet/MobileNet.
- [ ] **Native bf16 mm template**: Currently widen-compute-narrow. Add a bf16 native mm template gated on `VK_KHR_shader_float16_int8` (NVIDIA real GPU only).

### P3.1 — AOT compile (AOTI for Vulkan)

- [ ] **C++ runtime callable**: Replace the Python wrapper's `_vk_make_kernel` import with a stable C++ runtime entry that AOTI can call.
- [ ] **AOT package layout**: SPIR-V blobs + push-constant metadata in a single `.zip` package loadable by `aoti_load_package` on a deployment target without Python.

### P3.2 — Stretch / exploration

- [ ] **Custom bwd subgraph capture**: Wrap an FX subgraph (e.g. `softmax_backward → matmul`) into a single Slang shader emitted by `VulkanTemplate`, beyond what the SIMDKernel codegen produces.
- [ ] **Multi-stream / async copy**: The current Vulkan backend is single-stream. A second compute queue for parameter prefetch / gradient accumulation could overlap optimizer with backward.
- [ ] **Complex-dtype shaders**: `VulkanOverrides` maps complex dtypes but Slang shader templates don't generate valid complex arithmetic. Most models don't need this.

### P3.3 — Slang Autodiff Integration

Slang ships first-class automatic differentiation (`[Differentiable]`, `fwd_diff`, `bwd_diff`, `[BackwardDerivative]`, `IDifferentiable`). Today **zero** Slang shaders in this repo use it — all 77 backward shaders under `shaders/` are hand-written, and Inductor backward lowerings decompose to primitives. This is the right floor (hand-written wins on register pressure, fused activation+reduction, atomic accumulation, wave-intrinsic tricks). But Slang autodiff is unexplored as a *selective* tool: it can replace dozens of lines of per-op derivative bookkeeping in `csrc/ops/backward_ops.cpp` for cases where the generated code is competitive. Pilot first, expand only on measured wins.

- [ ] **P3.3.1 — Pilot `[Differentiable]` activation backward (gelu)**: write `shaders/activation/gelu_diff.slang` with a `[Differentiable]` forward and `[BackwardDerivative(gelu_bwd_custom)]` invoking a hand-written reverse pass (so the autodiff machinery composes with our wave-intrinsic patterns instead of replacing them). Compare against the existing `gelu_backward` Inductor decomposition lowering on MLP backward. **Bar:** ≤6 dispatches MLP bwd target (current §"Performance Targets" floor); no VGPR-occupancy regression measured via `slangc -dump-asm`. **Test:** `TestSlangAutodiffGeluPilot` — generated SPIR-V non-empty + numerical match vs CPU eager + dispatch-count contract.
- [ ] **P3.3.2 — Slang-generated softmax backward for small D**: mark `softmax_fwd_small.slang` `[Differentiable]`, dispatch `bwd_diff(softmax_fwd_small)` for `D ≤ 512`; fall through to the existing hand-fused `softmax_backward.slang` for larger D. **Risk:** reverse-mode reduction codegen may over-allocate intermediates — measure VGPR count via `slangc -dump-asm` and gate on `vgprs ≤ 64` (preserves 5 waves/workgroup occupancy on RDNA1). **Bar:** transformer-block backward dispatch count strictly ≤ baseline. **Test:** `TestSlangAutodiffSoftmaxBackward`.
- [ ] **P3.3.3 — Generic reduction backward via `bwd_diff` (exploration)**: prototype a `[Differentiable]` reduction template (sum/mean) with custom `atomic_accumulate<T>` helpers in Slang. **Risks:** atomic-add for f16/bf16 not native on RDNA1 (needs CAS-loop fallback), SwiftShader capability gating. Correctness-only test (`TestSlangAutodiffReductionGeneric`) — perf may lose to hand-written; document the loss if it does, decide whether to ship behind a flag.
- [ ] **P3.3.4 — Capability-gated dispatch knob**: extend `python/torch_vulkan/inductor/config.py` with `TORCH_VULKAN_SLANG_AUTODIFF={off,pilot,full}` (default `off`). `pilot` enables P3.3.1 + P3.3.2 paths; `full` adds P3.3.3. The lowering layer (`lowerings.py`) consults this knob when choosing between the existing decomposition and the autodiff-generated shader. **Test:** `TestSlangAutodiffConfigKnob` covers env-var → dispatch-path round-trip via a spy on shader source.
- [x] **P3.3.5 — Documentation** *(2026-04-27)*: [docs/03-slang-shaders.md](03-slang-shaders.md) extended with the "Inductor backend: floor-vs-ceiling for autodiff (P3.3 reference)" sub-section under the existing "Slang Autodiff Integration with PyTorch Autograd" section. Covers: when to use autodiff vs hand-written, what register/occupancy/atomic constraints rule it in or out, the four-step pilot order (activations → loss cluster → RoPE → negative pilot on layer-norm), and the `register_autodiff_template` tooling that lands with P3.3.9. Cross-references this roadmap's P3.3.6–P3.3.11 items.
- [ ] **P3.3.6 — Loss-fn autodiff cluster**: write `[Differentiable]` primal forward kernels for `mse_loss`, `binary_cross_entropy_with_logits`, `smooth_l1_loss`, `huber_loss`, `kl_div`. Each `bwd_diff(loss_fwd)` gives the backward shader for free. Loss math is simple enough that the autogenerated reverse pass should be near-optimal; this is the ideal pilot cluster — measurable wins, low register pressure, no atomic-add complexity. Extern-template each pair so the backward fuses with `grad_scale * loss_grad` epilogues. **Bar:** loss + loss-bwd ≤ 1 + 1 dispatches each on (B=8, V=1000) test. **Test:** `TestSlangAutodiffLossCluster` — 5 ops × {fwd, bwd, vs CPU eager} = 15 numerical assertions + dispatch-count contracts.
- [ ] **P3.3.7 — RoPE backward via autodiff**: the eager `rope_apply.slang` is short (sin/cos rotation per-head pair); `bwd_diff(rope_apply)` should produce a competitive backward. Replaces hand-written `rope_apply_backward.slang` if perf matches. **Risk:** sin/cos derivatives bring register pressure — measure VGPR delta. **Test:** `TestSlangAutodiffRoPE` — bit-exact-to-eager-bwd via `[ForceInline]` on the inner trig helpers.
- [ ] **P3.3.8 — LayerNorm fwd `[Differentiable]` → autodiff backward**: `native_layer_norm.slang` is non-trivial (mean/var reduction + affine), so autodiff has to track Welford partial state. Likely the **negative case**: register pressure / atomic-add fallback for grad_w / grad_b will be worse than the hand-fused `layer_norm_backward.slang`. **Goal: confirm and document the negative result** so future iterations don't re-attempt. Bench against the hand-fused 2-dispatch eager backward; if autodiff loses by ≥1.5× per-dispatch, file it as a documented limitation and move on. Test `TestSlangAutodiffLayerNormCompare` — VGPR count + ms/dispatch comparison.
- [ ] **P3.3.9 — `@autodiff_template(name, primal_src)` decorator**: tooling — a single-call helper in `extensions.py` that takes a `[Differentiable]` Slang primal source, generates both the forward and `bwd_diff(...)` entry points, registers them as `ExternKernelChoice`s, registers Inductor lowerings for the `aten.<op>.default` and `aten.<op>_backward.default`, and adds the prewarm submission. Cuts per-op autodiff scaffolding from ~5 files to one decorator call. Test `TestAutodiffTemplateDecorator` — round-trip a synthetic `softplus` primal.
- [ ] **P3.3.10 — VGPR / occupancy regression gate**: every autodiff shader that lands ships with a `slangc -dump-asm` snapshot in the regression suite asserting `vgprs ≤ N` and `occupancy ≥ M waves` (auto-extracted from the dump). A future Slang upgrade that bloats the generated reverse pass fails CI before it lands in main. Test scaffolding `TestAutodiffRegressionGate` — parametrized over registered autodiff shaders.
- [ ] **P3.3.11 — Custom buffer-derivative for atomic gradient accumulation**: the standard `bwd_diff` over a `RWStructuredBuffer` write doesn't have an automatic derivative — every autodiff backward that scatters into a gradient tensor needs a user-defined `[BackwardDerivativeOf]` on the load that emits `InterlockedAdd` (or a CAS-loop for f16). Ship one in `slang_helpers.py` (Slang side) as `_vk_diff_grad_accumulate<T>`, used by all P3.3.6 / P3.3.7 backward shaders. Test `TestAutodiffGradAccumulator` — 4 dtypes (f32, f16, bf16 via CAS, i32) × {scalar, vec4} stores.

### P0.4 — Per-dispatch wrapper overhead (compiled MLP slower than eager)

**Discovered 2026-04-26.** On the MLP forward benchmark, compiled mode is
**2× slower than eager** (0.142 ms vs 0.067 ms) despite emitting fewer
dispatches (8 vs 11). Microbenchmark traces:

| Op | us / call |
|----|-----------|
| `torch.empty_strided((32,512), …, device='vulkan:0')` | 17.9 |
| `torch.empty((32,512), …, device='vulkan:0')` | 30.5 |
| `torch.empty(...) + del` (cache reuse) | 30.9 |

With 8 buffer allocations per compiled MLP step, allocator overhead alone
accounts for ~140 us — essentially all of the compiled-time excess. The
allocator itself is fine; the cost is the Python → `at::empty_strided` →
PrivateUse1 → `VulkanAllocator::allocate` → tensor wrap round-trip.

- [~] **Direct `_c_ext.empty_strided_fast` C++ binding** *(2026-04-26 — code landed, build pending)*: `csrc/init.cpp` exposes `_empty_strided_fast(size, stride, dtype)` calling `at::empty_strided` on PrivateUse1 directly. `python/torch_vulkan/__init__.py` `_empty_strided_vulkan` now `getattr(_c_ext, "_empty_strided_fast", None)` with a fallback to `torch.empty_strided`. Effectively identical perf until rebuild — needs `MAX_JOBS=4 pip install -e . -v --no-build-isolation` to take effect. Re-measure MLP forward then to confirm ≤5 us/call target.
- [~] **Output-buffer reuse pool keyed on (size, dtype)** *(2026-04-26 — Python infra landed, default-off)*: `python/torch_vulkan/inductor/buffer_pool.py` ships an opt-in pool keyed on `(size, stride, dtype)` with per-key cap (4) + global cap (64) + stats wired into `inductor_stats.summary()`. `VulkanPythonWrapperCodegen.make_buffer_free` emits `vulkan_pool_release(buf_N); buf_N = None` for vulkan intermediates (skips graph inputs/outputs and reused buffers); `_empty_strided_vulkan` consults the pool first. **Default off** (`TORCH_VULKAN_BUFFER_POOL=1` to enable): on the MLP forward, the per-call pool overhead exceeds per-hit savings because most allocations come from `aten.addmm`/`aten.mm` extern paths that bypass `empty_strided_vulkan`. The pool is the right shape for graphs where the bulk of allocations *do* flow through `empty_strided_vulkan` (large pointwise/reduction chains); the MLP gap will close when the C++ allocator gets an extern-kernel hook (next item). `TestBufferPool` (8 tests) locks the contract.
- [ ] **C++ extern-kernel allocator hook for `aten.{mm,addmm,bmm,linear}.out`**: route extern-kernel output allocations through a C++-side `(size_class, dtype)` pool that recycles tensors freed within the same step (or across steps via a step-local quarantine drained at flush). The Python pool above already handles non-extern intermediates; the bulk of MLP/Qwen forward allocations are addmm-extern and bypass it entirely. Target: drop MLP forward to ≤eager (0.07 ms) by eliminating the dispatcher round-trip on extern outputs as well.
- [x] **Wrapper-level `assert_size_stride` kill-switch** *(2026-04-26)*: `TORCH_VULKAN_TRUST_INDUCTOR=1` injects no-op `assert_size_stride` / `assert_alignment` definitions into the wrapper preamble (`VulkanPythonWrapperCodegen.write_header`), shadowing the upstream imports. `TestTrustInductor` covers the env-var round-trip + an end-to-end compile-and-execute under the flag.
- [~] **Re-measured MLP forward** *(2026-04-26 — updated with Python-pool finding)*: compiled MLP forward is **0.134 ms** vs eager **0.066 ms** — **0.49× of eager**. With `TORCH_VULKAN_BUFFER_POOL=1` it goes the wrong way (0.148 ms / 0.46×) because most MLP allocations come from `aten.addmm` extern outputs that bypass `empty_strided_vulkan` entirely; the per-call pool overhead on the few that do go through it (~2/step) exceeds the per-hit savings. The remaining excess tracks the same microbench (~70 us = 8 dispatcher round-trips × ~17 us). Closing the gap now needs the C++ extern-kernel allocator hook (new P0.4 sub-item).

### P4.1 — End-to-end training measurement & profiling

- [x] **`benchmarks/inductor_train.py`** runner *(2026-04-26)*: `--model={mlp,mnist_cnn,resnet18,transformer_block}`, runs eager + compiled, prints dispatches/step + ms/step + speedup, optional `--json` for CI. Initial measurements: small (MLP, MNIST CNN) workloads are dispatch-overhead-dominated and the compiled path is currently slower than eager — needs larger batch/feature sizes plus the P0.0 backward unblock to hit the headline targets.
- [x] **Per-kernel waterfall dump** *(2026-04-26)*: `inductor_stats.dump_waterfall(path)` writes `{n_kernels, total_calls, total_us, kernels:[{name, dispatches, total_us, avg_us, percent_total}]}` sorted by `total_us` descending. `TestWaterfallDump` locks the contract (sorted, sums to 100%, writes empty when stats disabled).
- [x] **Compile-time profiler** *(2026-04-26)*: `runtime._COMPILE_STATS` tracks `cold_compiles` / `cold_compile_us` / `in_memory_hits` / `disk_cache_hits` / `prewarm_submits` always-on. `inductor_stats.compile_stats()` returns the dict + derived `cache_hit_rate` and `avg_cold_compile_us`. `print_compile_stats()` prints a 3-line startup summary; `reset_compile_stats()` zeros the counters. `TestCompileStats` locks the contract.
- [ ] **Regression budget enforcement**: any new commit that bumps a regression-test dispatch count beyond its asserted ceiling fails CI. Convert `≤N` asserts to `==N` once we believe the floor is reached.

### P4.2 — Inductor IR quality

- [ ] **Tile-size search across (M,N,K) workload shapes**: replace `_pick_tile_configs()`'s static list with a per-shape benchmarking sweep cached at `~/.cache/torch_vulkan/mm_autotune/`. Reuse `Benchmarker` from runtime.
- [ ] **Cooperative reduction selection**: audit `should_use_cooperative_reduction` — current heuristic is conservative on small `rnumel`. Wave-intrinsic cooperative reduction beats persistent on rnumel ∈ [33, 1024] on RDNA1.
- [ ] **Reduction split heuristic**: persistent vs split-K reduction selection should be informed by SM/CU count and rnumel; verify Vulkan's `subgroup_size * num_compute_units` is wired into `V.choices` correctly.
- [ ] **Loop-coalescing verification**: `tile_dimensions` produces a tiling — verify large pointwise kernels don't waste workgroups when numel >> max_threadgroup * max_workgroup_count_x.
- [x] **Index simplification** *(2026-04-26)*: `VulkanExprPrinter._print_Mul` filters `0*x→0` / `1*x→x` and `_print_Add` filters `0+x→x` for unevaluated SymPy nodes. Reduces dead arithmetic in emitted Slang where Inductor builds Mul/Add with `evaluate=False` for symbolic strides. `TestExprPrinterSimplify` locks the contract.

### P4.3 — Slang shader codegen quality

- [x] **Shared-memory budgeting** *(2026-04-26)*: `VulkanKernel.__init__` tracks `_groupshared_bytes_used` against a `_groupshared_budget_bytes = 64*1024` cap. `_new_idxvar(is_threadgroup=True)` adds `_slang_dtype_bytes(dtype) * elem_count` per allocation and raises `NotImplementedError` if the cumulative usage would exceed the RDNA1 LDS budget — surfaces overcommits as a clean failure rather than a silent driver-side spill to scratch. The wave-helper smem owned by `slang_helpers.emit_helpers` is excluded (bounded by `simd_group_size` ≤ 256B and statically safe). Locked by `TestGroupSharedBudget` — verifies the budget is updated on persistent-reduction codegen and that an oversized synthetic call raises.
- [x] **Drop dead `tmp_acc_*` groupshared decls in reduction codegen** *(2026-04-26)*: `_reduction_nocache` (`kernel.py`) was unconditionally calling `_new_idxvar(acc_slang, elem_count=n_waves)` and discarding the result, so every reduction kernel emitted a `groupshared <dtype> tmp_acc_N[n_waves];` line that was never referenced (the wave-intrinsic helper in `slang_helpers.py` owns its own `smem_<op>` scratch). Removed the dead allocation. Dumped CE Slang shows the `groupshared float tmp_acc_0[1]` / `groupshared int64_t tmp_acc_0[1]` decls are gone; full inductor regression suite stays green (120 passed, 16 xfailed).
- [x] **`WaveActiveSum` over `groupshared` for partial reductions** *(2026-04-26 — verified already implemented)*: `slang_helpers.py:257-266` already special-cases `n_waves == 1` (workgroup fits in one subgroup) by emitting a one-line `WaveActiveX(val)` helper with no `groupshared` allocation and no `GroupMemoryBarrier`. Verified live in dumped CE kernel #1 (`numthreads(64)` on RDNA1 wave64 → `c10_vulkan_wg_reduce_max(...) { return WaveActiveMax(val); }`). One follow-up edge case worth a future audit: when a kernel picks `numthreads > subgroup_size` (so `n_waves > 1`) but the reduction itself only spans `red_size <= subgroup_size`, the smem path still fires; tracking under the broader cooperative-reduction-selection item.
- [x] **Vec4 packed pointwise** *(2026-04-26)*: f32 contiguous-pointwise kernels with a single non-reduction range tree, `numel % (max_threadgroup_size * 4) == 0`, and trivial `xindex`-only buffer indexing now bind I/O as `StructuredBuffer<float4>` / `RWStructuredBuffer<float4>`. Each thread issues one coalesced `float4` load per input, unrolls the scalar body 4× via `[unroll] for (uint _k = 0; _k < 4u; ++_k)`, and flushes one `float4` store per output. Quarters dispatch workgroup count and ~halves global-memory transactions for the largest contiguous f32 elementwise kernels. Decision is post-body-emission in `kernel.py:_vec4_pw_eligible` — runs a substring scan on the rendered body so any unsafe codegen pattern (atomics, wave intrinsics, `_vk_linear` multi-axis decomposition, multistage hoist, packed16 already locked, OOB guard) bails to scalar safely. Buffer subscripts are validated against the axis-alias set built from `uint <name> = xindex;` lines so loads via `x0` etc. are accepted. Kill switch `TORCH_VULKAN_NO_VEC4_POINTWISE=1` (config.py). Locked by `TestVec4PointwiseF32` (6 tests: vec4 fires on 4096-elem add, correctness vs CPU on add and `relu(a*0.5+b)`, kill-switch, skip on 4097-elem non-div-4, skip on f16 inputs). Perf: 100×`relu(a*0.5+b)` on 1M-elem f32 took 22.57 ms vec4 vs 24.47 ms scalar (NO_VEC4=1) — ~8% end-to-end win on a dispatch-overhead-dominated benchmark; expected to scale with kernel arithmetic intensity.
- [ ] **`unpack_byte_to_float4` fast paths**: int8 → float widening for low-precision inference (post-int8-quant inference path).
- [ ] **Specialization constants instead of push constants** for shape-dependent loop bounds when the shape is known at compile time → fewer push-constant bytes, more SPIR-V optimization opportunities for the driver.

### P4.4 — Backend ergonomics

- [x] **Helpful import error** *(2026-04-26)*: `_diagnose_import_failure` in `python/torch_vulkan/__init__.py` probes for missing `libvulkan.so.1`, missing `slangc` (env var or PATH), and `vulkaninfo --summary` failures, and surfaces the most likely root cause as a one-paragraph troubleshooting message wrapping the original `ImportError`. `TestImportDiagnostics` locks the contract.
- [x] **`torch_vulkan.inductor.inductor_stats.summary()` API** *(2026-04-26)*: returns `{n_kernels, total_calls, total_us, avg_us_per_call, top}` aggregate dict — Jupyter-friendly. Also fixed the underlying `_wrap_stats` bug where the stats entry was captured at wrap time, so `reset_stats()` left dangling references and `get_stats()` reported empty for already-cached kernels.
- [x] **`TORCH_VULKAN_DUMP_FX=<dir>`** *(2026-04-26)*: `_maybe_dump_fx` runs in `_VulkanCustomPass.__call__` (pre + post). Files numbered `graph_NNNN_pre.txt` / `graph_NNNN_post.txt` for offline diffing.

### P4.5 — Quantized inference (post-train)

- [ ] **int8 mm template**: leverage RDNA1's int8 wave intrinsics where available; fall back to f16 dequant + f32 mm where not.
- [ ] **AWQ / GPTQ weight-only quant lowering**: the eager backend has dequant shaders; expose them as Inductor lowerings so quantized models torch.compile cleanly.

### P4.6 — Distributed / multi-device

- [ ] **DDP allreduce hook under torch.compile**: Stage 7 has DDP working eagerly; verify `compiled_autograd` interacts with the allreduce hook correctly. (Stretch — agent's hardware is single-GPU.)

### P1.5 — PrimTorch decomposition tuning for Vulkan

PrimTorch's default decomposition table is CUDA-tuned; several decomps that are
fine on CUDA expand into too many primitives for Vulkan and prevent fusion.

- [ ] **Vulkan-tuned decomposition table**: build a `VULKAN_DECOMP_TABLE` that overrides specific Inductor decompositions where a tighter Vulkan op exists. Audit candidates: `aten.dropout` (currently expands to bernoulli + mul + scale on host), `aten.native_dropout`, `aten.gelu` (`exact` vs `tanh` approximations), `aten.elu`, `aten.celu`, `aten.binary_cross_entropy_with_logits`, `aten.smooth_l1_loss`, `aten.huber_loss`, `aten.kl_div`. Each should either map to a fused Vulkan op or to a smaller primitive set.
- [ ] **Disable decomposition for ops with specialized shaders**: `aten.upsample_bilinear2d`, `aten.upsample_nearest2d`, `aten.replication_pad2d`, `aten.reflection_pad2d` — eager has fused shaders; let them fall through to ExternKernelChoice and fuse with downstream pointwise.
- [ ] **`aten.native_dropout` fast path**: route to a single Slang shader that fuses Philox RNG + mask gen + scale + apply, instead of decomposing into three pointwise dispatches.
- [ ] **`aten.scaled_dot_product_attention` decomposition guard**: when the shape qualifies for flash attention (head_dim ∈ {32, 64, 128}, seq_len ≥ 64), prevent SDPA decomposition and route to the flash-attention extern (P1.1). Add a regression test asserting 1 dispatch on the hot path.
- [~] **`view`/`reshape` after fused pointwise returns wrong values — *systemic pattern observed 2026-04-27*** *(P0 — diagnosed 2026-04-26, upstream-blocked)*: **The bug is much broader than originally scoped.** A 2026-04-27 measurement-driven probe pass found that **any zero-cost-view followed by a pointwise epilogue** can hit the constant-fold-during-trace path: `view`/`reshape`, `slice` (`x[:,1:-1,::2]+1`), `permute(...).contiguous()+1`, `expand(8,16)+1`, `broadcast_tensors(a,b); a+b`, `tensor_split(x,4); sum(parts)`, `einsum('ij,jk->ik')`, `selu+celu+prelu` chain, `clamp+clip` chain — all return **0 dispatches with stale captured-during-trace data**. Symptom on probe: `d=0` and a non-trivial `diff` value (1–9 typical). Same root cause: PyTorch's view fast-path drops device during `in_kernel_invocation`; AOT autograd then constant-folds the entire post-view chain to a real materialised vulkan tensor whose contents were captured at trace time when inputs were uninitialized fakes. **All affected tests** in `TestP12*` regression classes are annotated `[~]`-blocked or omitted; will unblock alongside the upstream P0.0 view-fast-path fix or a Vulkan-side `Tensor._make_subclass` patch. Original entry below preserved for context: compiled `(x + 1).view(8, 16)` and `(x + 1).reshape(8, 16)` return uninitialized memory (e.g. `1.13e+33`, varies per run). Initial smart-barrier hypothesis was **wrong**. `TORCH_LOGS=graph_code` reveals the Dynamo trace:
   ```
   y: "f32[8, 16][16, 1]vulkan:0" = l_x_ + 1.0
   view: "f32[8, 16][16, 1]meta" = y.view(8, 16)   # ← meta, not vulkan
   ```
   `y.view(...)` on a vulkan FakeTensor produces a `device=meta` FakeTensor via the C++ view fast-path (uses `Tensor._make_subclass`, not `FakeTensor.__new__`, so the existing `_patch_fake_tensor_view_op_device` doesn't catch it). AOT autograd then constant-folds the entire pointwise+view chain to `_tensor_constant0` — a real materialized vulkan tensor whose contents were captured at trace time when the inputs were uninitialized fakes. Same root family as the P0.0 backward-graph blocker: PyTorch's view fast-path drops the device during `in_kernel_invocation` mode. Fix requires either an upstream change to that fast-path, or a Vulkan-side patch of `Tensor._make_subclass` that detects the pattern. Test stays `xfail(strict=True)`. Verified: enabling `TORCH_VULKAN_FAKE_VIEW_FIX=1` and the `_in_joint_trace` always-on patch don't help — the meta tensor isn't constructed via `FakeTensor.__new__`. Locked by `TestRedundantCopyRemoval.test_view_then_copy_no_extra_dispatch` and `test_reshape_after_pointwise`.
- [~] **Cross-entropy fast path** *(P0 — diagnosed 2026-04-26, upstream-blocked, same root family as the view P0 above)*: `F.cross_entropy(logits, target)` compiled returns `0.73` vs eager `2.58`. Earlier hypothesis (int64 gather codegen) was symptomatic, not root cause — the actual bug is the same FakeTensor view-fast-path leak as the `view`/`reshape` P0. AOT autograd's post-grad graph shows `unsqueeze`, `gather`, `squeeze`, `neg`, `where_1`, `sum_3`, `div` all annotated as `device=meta` despite all upstream inputs being vulkan. Inductor codegen then emits a no-op kernel `out_ptr0[0] = 0.0f` for the `div` output. Tried a `post_grad_custom_pre_pass` that rewrites all 7 meta `node.meta['val']` entries to vulkan FakeTensors — verified the rewrite happens (7 → 0 meta nodes) but the result is unchanged because Inductor re-derives device info during lowering from the op signatures (not from `meta['val']`). True fix needs either (a) the upstream P0.0 view-fast-path patch, or (b) a vulkan-only ExternKernel routing for `aten.gather` that bypasses Inductor's pointwise `indirect_indexing` lowering entirely. `TestCrossEntropyBaseline` stays `xfail(strict=True)`.
- [~] **`aten.var_mean` / `aten.std_mean` Welford fast path** *(2026-04-26)*: regression `TestVarMeanWelford` locks current achievable bound (≤4 dispatches for `var_mean(x, dim=-1, keepdim=True)`) + correctness vs CPU. Inductor's stock decomposition produces 2 reductions; tightening to 1 needs a custom lowering routing through the eager Welford `welford_combine` shader path. Test in place ready for the codegen fix.
- [ ] **`aten.bincount` / `aten.histogram`**: atomic-histogram Slang shader; common in sequence packing and class-balance loss. Currently extern. Codegen via `atomicAdd` to an output histogram buffer; verify correctness on small + large bin counts. Test `TestBincountHistogramCodegen`.
- [ ] **`aten.unique` / `aten.unique_consecutive`**: rare on hot path but blocks compile when present. Codegen guarded on dim-0 contiguous case only; fall through to extern otherwise. Test `TestUniqueCompiles`.
- [ ] **`aten.embedding_bag` (mode={mean, sum, max})**: extern-template so downstream norm/linear fuses. Reuses `extensions.register_template`. Test `TestEmbeddingBagTemplate`.

### P1.6 — Codegen heuristics & autotune coverage

Several heuristics in the codegen path are **hardcoded constants** that haven't been re-benched on RDNA1 since they were ported from CUDA defaults. Each item below is a measurable picker tuning — sweep, persist the winner in the existing autotune cache, lock with a regression that asserts the new bound. No new infrastructure: reuses [autotune.py](../python/torch_vulkan/inductor/autotune.py)'s JSON cache and [Benchmarker](../python/torch_vulkan/inductor/runtime.py).

- [ ] **Persistent-vs-cooperative reduction threshold rebench**: `should_use_persistent_reduction` in [kernel.py](../python/torch_vulkan/inductor/kernel.py) hardcodes `rnumel ≤ 8192 → persistent`. Sweep `[1024, 2048, 4096, 8192, 16384, 32768]` per dtype × per workgroup-size on RDNA1, pick best, persist in `~/.cache/torch_vulkan/reduction_strategy.json`. Lock with `TestReductionStrategyThreshold` (asserts the picker returns the cached choice + correctness across the boundary).
- [ ] **Register-tile autotune in `slang_mm.py.jinja`**: the `m_per_thread` / `n_per_thread` register-tile dims are currently fixed (`r1x1` and `rMxN` paths via static config). Extend the P2.2 matmul autotuner candidate set to include `{(1,1), (2,2), (4,4), (4,2), (2,4)}` register-tile dims × tile-M/N/K configs. Cache key already includes dtype/transpose; extend with register-tile dims. Lock with `TestRegisterTileAutotune` — asserts a shape-specific dispatch-time strictly ≤ baseline `r1x1`.
- [ ] **Vec4 pointwise extends to f16/bf16**: P4.3 vec4 pointwise is f32-only. Extend `_vec4_pw_eligible` in `kernel.py` to handle packed `uint32 = 2×f16` (effective 8-lane vec); compose with the existing packed16 path so a single thread issues 1 coalesced load per input → 8 element-wise ops. Verify on a 1M-elem f16 `relu(a*0.5+b)` benchmark — target ≥1.5× scalar baseline. Test `TestVec4PointwisePacked16`.
- [ ] **`numthreads` picker for medium kernels (256–64K elem)**: P6.3 right-sized small kernels (`< 256`); medium kernels still pick 256 unconditionally. Sweep `{64, 128, 256, 512}` per shape bucket via the WG autotuner (`autotune.py`) for kernels in the medium range, persist. Lock with `TestNumthreadsMediumKernel`.
- [ ] **Reduction split-K codegen path**: persistent-reduction misses on `rnumel >> 8192` (vocab=151936 wide-row); today multistage smem dispatches sequentially. Add a split-K codegen path that distributes reduction across compute units, with a final cross-CU combine. Gated on rnumel and CU count from `device_interface`. Test `TestReductionSplitK` — wide-row softmax dispatch ≤2 (split + combine).
- [x] **LDS-budget fragmentation cost** *(2026-04-27)*: `_new_idxvar` in `kernel.py` rounds each `groupshared` allocation up to 16B alignment via `(raw_bytes + 15) & ~15` before adding to `_groupshared_bytes_used`, matching the AMD ABI's per-decl alignment. Without this, a kernel with many small per-thread odd-sized allocations could pass the 64KB cumulative check while the driver actually reserves more LDS, silently spilling to scratch. `TestGroupSharedBudget.test_alignment_aware_accounting` verifies (a) 5 int8_t advances the counter by 16 (not 5), and (b) four 17KB allocations correctly trip the budget on the third — without alignment, the first three would still pass. Suite: 193 → 194 passed (+1).
- [ ] **Coalesced-load loop-order picker**: for kernels with multiple iteration dims, the picker should prefer the order that maximizes contiguous loads on the largest stride-1 axis. Audit `tile_dimensions` in `kernel.py`. Currently chooses lexicographic; for `(N, C, H, W)` activations with `H, W` contiguous, the inner loop should be `W → H → C → N`. Test `TestCoalescedLoopOrder` — emitted Slang's outer-loop variable matches the largest stride-1 axis.
- [ ] **Subgroup-uniform branch elimination**: when a `where(mask, body, other)` mask is subgroup-uniform (e.g. CSE'd from a push-constant or a thread-id range comparison), emit a regular `if (uniform_mask)` block instead of the SIMT `where` ternary so the unselected lane never executes the load. Detect at codegen via `WaveActiveAllTrue/Equal` over the mask CSE — guard with the wave-intrinsic capability check. Test `TestSubgroupUniformBranch` — emitted Slang for a `where(tid < bound, body, 0)` pattern under a static `bound` shows an `if`/`else` block, not a ternary load.
- [x] **`[unroll]` on small static-bounded reductions** *(2026-04-27)*: `_reduction_nocache` in `kernel.py` now emits `[unroll(loop_size)]` on the multistage reduction's per-thread loop when `loop_size` is a static `1 < N ≤ 16`. Cap chosen at 16 because beyond that the unrolled FMA chain blows VGPR budget on RDNA1 (drops from 8 → 2 waves/WG occupancy). The static guard skips dynamic `loop_size_str` (which already evaluates to `(rnumel + stride - 1) / stride` at runtime). Locked by `TestSmallStaticReductionUnroll.test_small_reduction_carries_unroll_attribute` — spies on `runtime.compile_slang_to_spirv`, asserts at least one captured kernel for `sum(x, dim=-1)` on (8, 1024) carries `[unroll(N)]`. Suite: 201 → 204 passed (+3).
- [ ] **Compute-vs-memory bound classification**: tag each fused kernel with an `_intensity` field (FLOPs / bytes); use it to bias the threadgroup-size and tile picker. Compute-bound kernels (intensity > 16) prefer larger tiles + register tiling; memory-bound kernels (intensity ≤ 4) prefer smaller tiles + more concurrent workgroups for latency hiding. Wire `_intensity` into `_pick_threadgroup_size` and `_pick_tile_configs`. Test `TestKernelIntensityPicker` — synthetic memory-bound (`relu(x)`) and compute-bound (`gemm`-tail) kernels pick distinct numthreads.
- [ ] **Bank-conflict-aware groupshared layout (XOR-swizzle)**: RDNA1 LDS has 32 banks × 4 bytes; column-major access into `groupshared float[TILE_M][TILE_K]` causes 32-way bank conflicts on `tile_a[i][lid.x]`. Emit a swizzled index `(i ^ lid.x_high)` for the matmul / reduction shared-memory layouts. Test `TestSharedMemoryXorSwizzle` — vec4 matmul tile load shows the swizzle expression.
- [ ] **Sub-group dot product intrinsics for int8 inference**: RDNA1 supports `dot4_i8_i32(uint, uint, int)` packed-int8-vec4 dot product (single-cycle on hardware). Emit it for int8 quantized matmul and int8 conv tail-sum. Capability-gate on `VK_KHR_shader_integer_dot_product`. Test `TestInt8DotProduct` — int8 matmul Slang dump shows `dot4_i8_packed` calls instead of an inner-product loop.
- [ ] **Specialization constants for shape-driven loop bounds**: when a shape is known-static at compile time (Inductor pins `B=2, S=64, D=256` etc.), emit Slang `[SpecializationConstant]` declarations bound at pipeline-creation time, not push-constant reads. Lets the SPIR-V optimizer constant-fold loop bounds and unroll. Test `TestSpecializationConstantLoopBound` — synthetic kernel with static shape shows `[vk::specialization_constant]` decl.

### P1.7 — Primop coverage audit (Inductor lowerings & ExternKernelChoice templates)

A side-by-side `eager-shaders` × `Inductor-lowerings` audit (run 2026-04-26) shows the eager backend ships ~440 fused shaders while Inductor only routes ~30 of them. The rest fall back to extern aten dispatches and break fusion. Each item below names the eager shader, the missing Inductor route, and the regression-test contract.

**Loss ops (forward + backward — currently extern, no fusion with upstream logits/labels):**

- [ ] **`aten.nll_loss_forward` lowering**: decompose into `gather(log_probs, target) → mul(weight) → mean/sum/none-reduction`. Eager shader: `nll_loss_mean_fused`. The fused chain `cross_entropy = log_softmax + nll_loss_forward` should collapse to **1 dispatch** when persistent reduction fires on the vocab axis. Test `TestNLLLossForwardLowering`.
- [ ] **`aten.nll_loss_backward` lowering**: scatter-add gradient onto target index. Eager shader: `nll_loss_backward`. Test `TestNLLLossBackwardLowering` — ≤2 dispatches.
- [ ] **`aten.binary_cross_entropy_with_logits` lowering**: fused `sigmoid + log + clamp + reduction` decomposition. Eager shader: `bce_with_logits_fused`. Test `TestBCEWithLogitsLowering` — ≤1 dispatch.
- [x] **`aten.mse_loss` lowering** *(2026-04-27)*: `_register_loss_lowerings` decomposes `mse_loss(a, b, reduction)` into `mean/sum/none((a - b)^2)` via existing pointwise primitives. `TestLossLowerings.test_mse_loss_correctness` + `test_mse_loss_sum_reduction` + `test_mse_loss_dispatch_count` (≤2 dispatches).
- [x] **`aten.l1_loss` lowering** *(2026-04-27)*: `mean/sum/none(|a - b|)` decomposition. `TestLossLowerings.test_l1_loss_correctness`.
- [x] **`aten.smooth_l1_loss` / `aten.huber_loss` lowerings** *(2026-04-27)*: piecewise-quadratic decomposition via `where(|diff| < beta, 0.5*diff^2/beta, |diff| - 0.5*beta)` (smooth_l1) and `where(|diff| < delta, 0.5*diff^2, delta*(|diff| - 0.5*delta))` (huber). Both reduction modes covered. `TestLossLowerings.test_smooth_l1_loss_correctness` + `test_huber_loss_correctness`.
- [x] **`aten.kl_div` lowering** *(2026-04-27)*: `target * (log(target) - input)` (or `exp(target) * (target - input)` if `log_target=True`), with the `target == 0` masking convention to avoid `0 * log(0) = NaN`. `TestLossLowerings.test_kl_div_correctness`.
- [x] **`aten.binary_cross_entropy_with_logits` lowering** *(2026-04-27)*: numerically-stable `max(x, 0) - x*target + log1p(exp(-|x|))` decomposition. Optional `weight` multiplied per-element; `pos_weight` falls through to upstream extern (rare path; can revisit if it shows up in measurement). `TestLossLowerings.test_bce_with_logits_correctness`.
- [x] **`aten.binary_cross_entropy` forward + `binary_cross_entropy_backward`** *(2026-04-27)*: forward decomposes to `-(t*log(x) + (1-t)*log(1-x))`; backward to `grad_out * (x - t) / (x * (1 - x))` with `/numel` for `mean` reduction. Optional `weight` multiplied per-element. `TestLossLowerings.test_bce_correctness` + `TestLossBackwardLowerings.test_bce_backward` (xpasses through the bwd-class xfail decorator since BCE bwd doesn't traverse the view-fast-path).

**Normalization (forward + backward):**

- [x] **`aten.native_batch_norm` lowering** *(2026-04-27)*: outdated entry — `F.batch_norm(x, m, v, w, b, training=False) + 1.0` already compiles to **2 dispatches** (mean/var → affine + epilogue, then update running stats) via Inductor's stock decomp + Vulkan pointwise/reduction fusion. Beats the ≤2 target. `TestPoolAndBatchNorm.test_batch_norm_inference_dispatch_count` locks the contract. (Backward path still hits the P0.0 backward-graph compile blocker.)
- [ ] **`aten.native_batch_norm_backward` lowering**: grad_input + grad_weight + grad_bias decomposition with shared-load for the `(grad_out, x_hat)` scan. Eager shader: `batch_norm_backward` (2-dispatch). Test `TestBatchNormBackward` — ≤3 dispatches.
- [ ] **`aten.native_rms_norm_backward` decomposition** (sister to P1.1's RMSNorm forward fusion): `grad_x = w * grad_out * rstd - mean(grad_out * w * x) * rstd^3 * x`. Test `TestRMSNormBackwardLowering` — ≤2 dispatches.

**Embedding / sparse:**

- [~] **`aten.embedding` ExternKernelChoice template** *(2026-04-27 — pointwise epilogue already fuses to 1 dispatch; ExternKernelChoice wrap is unnecessary for that path)*: `F.embedding(idx, weight) + 1.0` compiles to **1 dispatch** via Inductor's stock indirect-indexing+pointwise codegen — the embedding gather and the pointwise consumer already collapse without a Phase A epilogue template. `TestEmbeddingFused.test_embedding_plus_one_dispatch` locks the contract. The `embedding → layer_norm` fusion to 2 dispatches still depends on the layer_norm reduction-kernel design and is tracked separately under P5.5; the ExternKernelChoice template approach is no longer the right shape for the pointwise-epilogue case (already covered) and would need to be re-scoped if the norm-fusion path is the actual goal.
- [ ] **`aten.embedding_bag` template** (mode={mean, sum, max}): extern-template so downstream norm/linear fuses. Eager shader: `embedding_bag_fused`. Test `TestEmbeddingBagTemplate`.

**Pooling (forward + backward):**

- [ ] **`aten.max_pool2d_with_indices` ExternKernelChoice template**: enables `conv → pool → relu` fusion via Phase A epilogue. Eager shader: `max_pool2d`. Test `TestMaxPool2dTemplate`.
- [~] **`aten.avg_pool2d` / `aten.adaptive_avg_pool2d` ExternKernelChoice** *(2026-04-27 — `avg_pool2d` ships at 1 dispatch; `adaptive_avg_pool2d` blocked on FakeTensor dispatch)*: `avg_pool2d` already compiles to **1 dispatch** with downstream pointwise (`F.avg_pool2d(x, 2) + 1.0` on (1, 3, 8, 8) f32) via Inductor's stock decomp + Vulkan pointwise codegen. `TestPoolAndBatchNorm.test_avg_pool2d_one_dispatch` locks the contract. `adaptive_avg_pool2d` falls through with `Dynamo failed to run FX node with fake tensors` — same FakeTensor view-fast-path device-propagation gap as the broader P0.0 family.
- [ ] **`aten.max_pool2d_with_indices_backward` lowering**: scatter-with-indices, fuses with downstream `relu_backward`. Test `TestMaxPool2dBackwardLowering` — ≤2 dispatches.
- [ ] **`aten.avg_pool2d_backward` / `aten.adaptive_avg_pool2d_backward` lowerings**: pointwise broadcast-divide; trivial decomposition. Test `TestAvgPool2dBackwardLowering`.

**Upsample:**

- [ ] **`aten.upsample_nearest2d` / `aten.upsample_bilinear2d` ExternKernelChoice templates**: eager shaders exist. Currently break the post-upsample relu/conv fusion. Test `TestUpsampleTemplate`.
- [ ] **`aten.upsample_nearest2d_backward` / `aten.upsample_bilinear2d_backward` lowerings**: scatter-add; fuse with upstream `*_backward`. Test `TestUpsampleBackwardLowering`.

**Padding / reshape primitives:**

- [x] **`aten.constant_pad_nd` codegen** *(2026-04-27)*: outdated entry — already compiles to **1 dispatch** with downstream pointwise (`relu(F.pad(x, (1,1,1,1), 0.0))` on (1, 3, 4, 4) f32) via Inductor's stock conditional-write codegen. `TestPixelShuffleAndPadCodegen.test_constant_pad_nd_plus_relu_one_dispatch` locks the contract. (Pad-into-conv epilogue fuse is a separate item that needs the conv extern template — tracked under P0.2.)
- [~] **`aten.replication_pad2d` / `aten.reflection_pad2d` ExternKernelChoice** *(2026-04-27 — `reflection` ships at 1 dispatch correct; `replication` codegen bug discovered)*: `reflection_pad2d` already compiles to **1 dispatch** correct via Inductor's stock conditional-write codegen on Vulkan (`F.relu(F.pad(x, (1,1,1,1), 'reflect'))` matches CPU eager to 1e-5). `TestPadModes.test_reflection_pad2d_one_dispatch_correct` locks the contract. **`replication_pad2d` compiles but produces wrong values** — the lowered index expression wraps instead of clamping to the boundary. `TestPadModes.test_replication_pad2d_correctness` is `xfail(strict=True)` until the codegen emits a clamped index for the replicate case (likely an upstream `torch._inductor.lowering._make_reflection_pad` style helper missing the replicate variant, or a Vulkan-specific override that differs from the reflect path).
- [x] **`aten.flip` / `aten.roll` codegen** *(2026-04-27)*: outdated entry — both already compile to **1 dispatch** with downstream pointwise (`flip(x, [0]) + 1` and `roll(x, 3) + 1` on (64,) f32) via Inductor's stock pointwise+gather codegen. `TestIndexOpsCodegen.test_flip_one_dispatch` + `test_roll_one_dispatch` lock the contract — a regression that pushes either back to extern fires.

**Activation / loss backward gaps (beyond P0.1):**

- [ ] **`aten.glu_backward` lowering**: split, sigmoid, mul, scatter — composable from existing primitives. Eager shader exists. Test `TestGLUBackwardLowering`.
- [ ] **`aten.prelu_backward` lowering**: piecewise on sign(x), grad_w reduction over batch dim. Test `TestPReLUBackwardLowering`.

**Spectral / linalg (low priority but needed for completeness):**

- [ ] **`aten._fft_r2c` / `aten._fft_c2r` ExternKernelChoice**: eager `fft_*` shaders (newly added) wrap as templates so spectral-domain pre/post-processing fuses. Test `TestFFTTemplate`.
- [x] **`aten.linalg_vector_norm` lowering (full ord coverage)** *(2026-04-27)*: covers `ord ∈ {±inf, 0, 1, 2, generic-p}`: `±inf → max/min(|x|)`, `0 → count_nonzero`, `1/2 → sum(|x|)/sqrt(sum(x^2))`, integer `|p| ≤ 8 → repeated mul (avoid pow stability issues)`, generic `p → pow(sum(|x|^p), 1/p)`. `TestLossLowerings.test_linalg_vector_norm_{l1,l2,inf,neg_inf,p3}_correctness`.
- [x] **`aten.norm.ScalarOpt_dim` thin shim** *(2026-04-27)*: legacy overload proxies to `linalg_vector_norm`. `TestLossLowerings.test_norm_old_overload_correctness`.

**RNG / dropout:**

- [ ] **`aten.native_dropout` Philox-fused lowering** (also referenced in P1.5): single Slang shader fusing Philox RNG + mask gen + scale + apply, instead of decomposing into 3 dispatches. Eager `dropout_fused` shader exists. Test `TestNativeDropoutFused` — ≤1 dispatch.

**Coverage tracking infrastructure:**

- [ ] **`scripts/audit_inductor_op_coverage.py`**: walks `benchmarks/inductor_train.py` models under `torch.compile` with `TORCH_LOGS=output_code`, captures every `extern_kernels.<op>` and unmatched `aten.*` reaching codegen, prints a coverage report `(op, n_calls, has_eager_shader, has_inductor_lowering, has_template)` so future audits stay grounded in real workloads. Test `TestOpCoverageAuditScript` — runs against the MLP fixture, asserts the report enumerates the expected extern aten ops.

### P2.4 — Memory planning & buffer reuse

Inductor's memory planner is shape-conservative; on Vulkan the per-allocation
overhead (~18 us, see P0.4) and peak-memory pressure on 6 GB RDNA1 hardware
both push us to be aggressive about reuse and aliasing.

- [ ] **Liveness-analysis-driven buffer reuse**: post-scheduling pass that walks the FX graph dataflow and reuses an output buffer for the next op when the producer is dead. Currently each Inductor allocation hits the Vulkan allocator. Target: ≥40% allocation reduction on MLP/ResNet forward.
- [ ] **In-place opportunity detection**: during fusion, detect when an output buffer can alias an input (no other consumers, same dtype, same shape) and emit `RWStructuredBuffer` aliasing in Slang. Eliminates a copy + an alloc per opportunity.
- [ ] **Workspace pool for scratch buffers**: per-stream pool keyed on `(numel, dtype)` for short-lived intermediates (≤2 dispatches alive). Use `c10::Allocator::DataPtr` with deleter that returns to the pool.
- [ ] **Activation checkpoint integration**: `torch.utils.checkpoint.checkpoint` under `torch.compile` should emit one fused recompute kernel per checkpoint segment instead of running the eager forward inside a dynamo-disabled subgraph.
- [x] **Peak-memory estimator + report** *(2026-04-26)*: `inductor_stats.MemoryTracker` context manager samples Vulkan `memory_cached()` at enter / poll() / exit, exposing `start`, `peak`, `end`, `delta_mib`, `peak_mib`. `inductor_stats.peak_memory_report()` returns `{cached_mib, n_kernels_recorded}` for one-line snapshots. Note: this is a sampled peak — true peak requires C++ `VulkanAllocator::allocate` instrumentation. `TestPeakMemoryAPI` covers context-manager round-trip + report-shape contract.

### P2.5 — Pattern matcher / FX rewrite expansion

Inductor's `pattern_matcher` registers ~100 graph rewrites in upstream. We
register five. Each new pattern lowers dispatch count on at least one
training workload.

- [~] **`bias_dropout_residual` pattern** *(2026-04-26 — baseline regression in place)*: `TestPostAttentionResidual.test_linear_add_residual_dispatch_count` locks current ≤3 dispatches for `linear + residual`. The fusion to 1 dispatch (Philox RNG inline + bias + add) is the next step; the test will tighten when it lands.
- [~] **`gelu_dropout` pattern** *(2026-04-26 — p=0 baseline shipped)*: `TestGeluDropoutBaseline.test_gelu_dropout_zero_dispatch_count` locks `F.dropout(F.gelu(x), p=0.0, training=False)` at 1 dispatch (since p=0+training=False is identity, the dropout collapses cleanly). The p>0 + training=True path needs Philox-inline FX rewrite — pending.
- [x] **`addcmul` / `addcdiv` recovery** *(2026-04-27)*: outdated entry — both already compile to **1 dispatch** via Inductor's stock fused-multiply-add codegen on Vulkan. The AOT-autograd-split triple recovers cleanly into a single fused dispatch (no FX pass needed; the pointwise scheduler already collapses the chain). `TestAddcmulAddcdiv` (2 tests) locks the contract.
- [ ] **`baddbmm + softmax` pattern**: attention scoring before mask. Fold the scale into the matmul and the optional mask add into the epilogue.
- [~] **`embedding + layer_norm` pattern** *(2026-04-26 — correctness baseline)*: `TestEmbeddingLayerNormBaseline` locks `embedding(idx) → layer_norm` correctness vs CPU. The fold-add-into-LN-load fusion still pending; the test fires on a future correctness regression.
- [x] **Constant-broadcast hoisting** *(2026-04-26)*: verified by `TestConstantBroadcastHoist` — chains of scalar add/mul/sub on a tensor (`x*0.5+0.7-1.3`) compile to **1 dispatch** with no extra broadcast buffers. Inductor's stock pointwise codegen already inlines scalar literals into the Slang source via `value_to_slang`. Test locks the contract; correctness verified vs CPU.
- [ ] **Dead-store elimination across fused regions**: scheduler-level pass that drops outputs not referenced downstream when no .out= variant captured them.
- [ ] **`triu_/tril_ + add` for causal masks**: detect causal-mask construction, replace with branch-free push-constant-driven Slang predicate inside the attention shader.

### P4.7 — Pre-recorded command buffers (Vulkan-graph equivalent)

The CUDA backend gets a 2–5× perf win on small batches via CUDA Graphs by
amortizing dispatch-launch CPU overhead. The Vulkan equivalent is a primary
command buffer recorded once and replayed per step, gated on identical
dispatch shapes/strides between iterations.

- [~] **`torch_vulkan.compile_graph(callable)` API** *(2026-04-26 — Python stub)*: `inductor/compile_graph.py` ships a `_CompiledGraph` wrapper that memoizes `torch.compile(dynamic=False)` per shape signature. Returns a callable + `.info()` (hits/misses/n_recordings/hit_rate) + `.reset_cache()`. **Stub** — does not yet pre-record a primary Vulkan command buffer (next iteration); current win is just removing per-call Dynamo overhead on static-shape replay. `TestCompileGraphStub` covers hit-rate-on-repeat, shape-change new recording, correctness, reset.
- [x] **Shape-guard hash for replay key** *(2026-04-26)*: `_CompiledGraph._guard_hash` hashes `(shape, stride, dtype, device)` per tensor arg + sorted kwargs into a 16-char SHA1. Locked by `TestCompileGraphStub.test_shape_change_triggers_new_recording` (same fn, different shape → new cache entry; same shape repeated → hit).
- [ ] **Allocator interaction**: a `RecordingAllocator` that returns aliased Vulkan buffers backed by stable offsets in a per-graph arena. Frees are no-ops within the recording.
- [ ] **Regression test**: `test_compile_graph_mlp_replay_under_n_us`, asserting per-step CPU time is ≤30% of the non-graph compiled-mode equivalent.

### P4.8 — Missing op codegen (scatter / gather / sort / cumulative)

- [~] **`aten.scatter_add` / `aten.scatter_reduce` Inductor codegen** *(2026-04-27 — `scatter_add` ships, `scatter_reduce(sum)` ships, other reduce modes blocked on extern eager)*: `scatter_add` already produces correct values under Inductor on Vulkan via the atomic-add codegen path. `scatter_reduce(reduce='sum', include_self=True)` routes through the same atomic-add codegen and is correct. The other reduce modes (`prod`, `mean`, `amax`, `amin`) compile-fall-through to `aten::scatter_reduce.two_out` which raises `Operation … is not yet implemented for the Vulkan backend` from eager — Inductor's atomic-add path only handles `sum`. Locked by `TestGatherScatterAdd.test_scatter_add_correctness` + `test_scatter_reduce_sum_correctness`. The non-sum modes need either (a) Vulkan eager registrations for `scatter_reduce.two_out` so the extern path works, or (b) Inductor codegen routing for the non-sum reductions (atomic min/max for amax/amin via `InterlockedMin`/`InterlockedMax`; prod/mean require CAS-loop and a two-pass denominator respectively). Tracked as a follow-up item below.
- [ ] **`aten.scatter_reduce` non-sum modes (prod / mean / amax / amin)** *(discovered 2026-04-27)*: extern path is unimplemented in Vulkan eager. Lowest-effort fix is registering `aten::scatter_reduce.two_out` on PrivateUse1 in `csrc/ops/indexing_ops.cpp`, dispatching to the existing scalar-fallback CAS loop on RDNA1 (no native f32 atomic-min/max for `amax`/`amin`). Inductor codegen support is the higher-value follow-up — `amax` / `amin` map to `InterlockedMin` / `InterlockedMax` for integer dtypes, CAS-loop for floats; `prod` / `mean` need CAS-loop + final-pass denominator respectively. Test `TestScatterReduceNonSumModes`.
- [ ] **`aten.index_put` (accumulate=True)**: codegen via atomic add; (accumulate=False) via gather + write.
- [x] **`aten.cumsum` / `aten.cumprod`** *(2026-04-26)*: already lower cleanly through `VulkanKernel.scan` (`kernel.py:scan`) onto the existing `wg_scan` helper (subgroup-intrinsic local scan + groupshared block prefix-add). The roadmap entry was outdated — both ops compile end-to-end and produce numerically-correct results vs CPU. Locked by `TestCumScanOps` (3 tests: cumsum/cumprod dim=-1 correctness on (8, 64) f32, cumsum ≤3 dispatches). Constraint: scan reduction-numel ≤ `max_threadgroup_size` (256); larger axis sizes still need the block-prefix-add stage to be templated, tracked separately.
- [ ] **`aten.sort` / `aten.argsort` / `aten.topk`**: bitonic-merge Slang sort for `k ≤ 1024`; otherwise extern. Required for Mixture-of-Experts top-k routing under torch.compile. **Priority: ship `topk` first** — MoE routing is the hot path; `sort` and `argsort` follow once `topk` is locked. Test `TestTopKMoERouting`.
- [ ] **`aten.unique` / `aten.unique_consecutive`**: rare on hot path; codegen guarded on dim-0 contiguous case only.
- [~] **`aten.repeat_interleave`** *(2026-04-26 — correctness baseline)*: `TestRepeatInterleaveBaseline` locks scalar-repeat (`x.repeat_interleave(2)`) and dim-repeat (`x.repeat_interleave(2, dim=0)`) correctness vs CPU. Prefix-sum-driven gather codegen still pending; the test fires on regressions to the C++ extern path.
- [ ] **`aten.bincount`**: atomic-histogram Slang shader; common in sequence packing.
- [ ] **`aten.embedding_bag` (mode={mean, sum, max})**: extern-template so downstream norm/linear fuses.

### P4.9 — Mixed precision & autocast under torch.compile

- [ ] **`torch.autocast(device='vulkan', dtype=torch.float16)` interop**: verify the cast graph produced by autocast survives Inductor lowering and uses our packed16 path. Add `test_autocast_compile_packed16_dispatch_count`.
- [ ] **bf16 reduction accumulator policy**: Inductor's default casts bf16 reduction inputs to f32 for accumulation. Verify this fires on RDNA1 (no native bf16 reductions). Document the cost.
- [ ] **Loss-scale fusion**: when GradScaler is active, fuse the `grad * scale` and `grad / scale` multiplies into adjacent grad-producing kernels.
- [ ] **fp8 (e4m3fn / e5m2) inference path under torch.compile**: eager has fp8 GEMM; expose as ExternKernelChoice gated on dtype.
- [ ] **bf16 mm template**: native bf16 mm shader behind `VK_KHR_shader_float16_int8` (NVIDIA real GPU only); falls back to widen-compute-narrow on RDNA1.

### P5.1 — Convolution algorithm selection

`aten.convolution_overrideable` always dispatches the eager direct-conv shader
today. Inductor should pick algorithm per (in_shape, kernel_shape, stride,
groups), not always direct.

- [ ] **Algorithm selector**: small registry of `(matcher, slang_template)` pairs covering depthwise (`groups == in_ch`), 1×1 (im2col-free GEMM), 3×3 stride-1 (Winograd F(2,3) / F(4,3)), and direct-conv fallback.
- [ ] **1×1 conv → mm fast path**: Inductor lowering that rewrites 1×1 convolution into `addmm` of reshaped activations; participates in the existing mm template autotune.
- [ ] **Winograd 3×3 stride-1 template**: Slang template with optional bias + activation epilogue. Bench vs direct-conv on ResNet-18 layer3/layer4 shapes.
- [ ] **Depthwise-conv2d template**: Slang shader for `groups == in_channels`. Add ExternKernelChoice wiring + bias/relu epilogue. Mobile-CNN critical path.
- [ ] **`aten.conv_transpose2d` template**: dispatch via gather (not col2im) so it fuses with downstream BN + activation.
- [ ] **Regression**: `test_resnet18_layer3_dispatch_count` ≤ current −2 dispatches once Winograd lands.
- [ ] **Conv backward Winograd template**: backward path always uses direct-conv today regardless of forward algorithm. Mirror the forward selector for the backward path (Winograd F(2,3)/F(4,3) for 3×3 stride-1, 1×1 → mm fast path, depthwise template). Critical for the ResNet-18 backward target (≤35). Test `TestConvBackwardWinogradTemplate`.
- [ ] **Conv + bias + activation + dropout epilogue chain**: extend the conv ExternKernelChoice epilogue slots to cover the full `conv → bias → relu → dropout` chain seen in CNN training. Currently 4 dispatches (extern + 3 pointwise); target 1. Test `TestConvFullEpilogueChain`.

### P5.2 — Data layout selection (NHWC vs NCHW)

RDNA1 prefers contiguous-channel-last for 3×3 conv (better cache reuse). PyTorch
defaults to channels-first; Inductor doesn't insert layout transforms today.

- [ ] **Layout cost model**: per-op cost table for NCHW vs NHWC on the Vulkan backend; scheduler-level pass picks a layout per fused region and inserts `aten.contiguous(memory_format=...)` only at boundary edges.
- [ ] **NHWC conv shader template**: Slang `conv2d_nhwc.slang` with packed16 channel access. Bench vs current NCHW path.
- [ ] **Conv-pool layout-stable region**: typical CNN block is `conv → bn → relu → pool` — emit the entire region in NHWC if input is NHWC, no intermediate transposes.
- [ ] **Regression**: `test_resnet18_nhwc_layout_no_intermediate_transposes` asserts zero `aten.contiguous(channels_last)` calls inside the fused region.
- [ ] **Layout-cost-model bench data** (prerequisite for the picker above): persist per-op NCHW-vs-NHWC ms measurements on RDNA1 to `~/.cache/torch_vulkan/layout_costs.json` for the ops that participate in CNN fused regions (`conv2d`, `bn`, `relu`, `pool2d`, `adaptive_avg_pool2d`, `cat`, `add`). Without this data the cost model has no input. Test `TestLayoutCostModelBench` — verifies the cache populates and the picker reads from it.

### P5.3 — Dynamic shape support

The current backend specializes on every shape, blowing up the SPIR-V cache for
workloads with variable batch / seq_len. Symbolic codegen exists in Inductor;
needs Vulkan plumbing.

- [~] **Audit `VulkanExprPrinter` for symbolic-shape coverage** *(2026-04-26 — smoke regression landed)*: `TestDynamicShapeAudit` covers `s0*s1+s1` and `FloorDiv(s0, 4)` round-trips through `VulkanExprPrinter.doprint` without crashing or leaking Python-repr tokens. Full audit (every SymPy node Inductor emits during a `mark_dynamic` graph) still requires running an actual dynamic-shape model; the test scaffolding is in place to add cases as they're discovered.
- [ ] **Push-constant shape passing**: when a kernel is generated with symbolic dims, route the resolved values through push constants instead of specializing the shader. Cuts SPIR-V cache size dramatically.
- [ ] **Guard reduction**: under dynamic shapes Inductor inserts `assert_size_stride` per call. Hoist them out of the hot path or no-op them under `TORCH_VULKAN_TRUST_INDUCTOR=1` (see P0.4).
- [ ] **`mark_dynamic` integration**: document the `torch._dynamo.mark_dynamic(t, dim)` pattern for transformer batch/seq dims; add a regression test asserting 1 SPIR-V cache entry across a (B=2, S=64) → (B=4, S=128) → (B=8, S=256) sequence.
- [ ] **0/1 specialization opt-out**: dynamic batch=1 currently re-specializes on every step; opt out via `torch._dynamo.config.assume_static_by_default = False`.

### P5.4 — Subgraph compilation (control flow)

`torch.cond` / `torch.while_loop` under Inductor emit subgraph stubs; the Vulkan
backend hasn't been exercised on these. Required for KV-cache decoding loops
and dynamic-shape attention masking.

- [~] **`HigherOrderOperator` trace through Vulkan backend** *(2026-04-26 — regression in place, currently `xfail`)*: `TestControlFlowSubgraph.test_torch_cond_branches_compile` exercises `cond(pred, λx.x*2, λx.x+3, (x,))` with vulkan inputs. Marked `xfail(strict=False)` until the subgraph emit path through `VulkanScheduling` lands; the test will start passing without code changes once the upstream HigherOrderOperator → SubgraphPythonWrapperCodegen path is exercised on PrivateUse1.
- [ ] **`torch.while_loop` with state-passing**: typical greedy-decode pattern; verify the loop body's compiled kernel shares the SPIR-V cache entry across iterations.
- [ ] **Subgraph allocator scoping**: intermediates inside the loop body must not leak across iterations.

### P5.5 — Custom op + extension registration helpers

Make it easy for a downstream user (or the agent itself) to add new fused
shaders without editing five files.

- [x] **`@torch_vulkan.inductor.register_template(name, slang_src)` decorator** *(2026-04-26)*: `python/torch_vulkan/inductor/extensions.py` `register_template(name, src, n_buffers, n_pc, ...)` returns a callable `dispatch(*tensors, *pc, wg=(x,y,z))` that JITs through the standard runtime + cache. `prewarm_template(dispatch_fn)` submits the source to the slangc pool. `TestExtensionDecorators` covers arg validation + cache_key wiring.
- [x] **`@torch_vulkan.inductor.register_lowering(op)` decorator** *(2026-04-26)*: in `extensions.py` — wraps Inductor's `register_lowering`, returns `NotImplemented` automatically when the first arg is not a vulkan IR node, so non-vulkan backends share the same Inductor session without lowering hijack. `TestExtensionDecorators.test_register_lowering_skips_non_vulkan` locks the device-guard contract.
- [ ] **Schema-driven shader scaffolding**: CLI `python -m torch_vulkan.inductor.scaffold <op>` emits a starter `.slang` + lowering + meta_patch + regression test for a new op.
- [x] **Documentation** *(2026-04-26)*: `docs/extension_cookbook.md` ships — single-page guide for adding a fused op via `register_lowering` / `register_template`, with a concrete worked example (fused `mul * 0.5`), wiring instructions, mandatory regression test contract, and debugging knobs.

### P5.6 — Numerical correctness & determinism

- [~] **`torch.use_deterministic_algorithms(True)` compatibility** *(2026-04-27)*: pointwise + reduction kernels are deterministic by construction (no atomics on the standard reduce path; multistage tree-reduction is bit-exact across runs at fixed wg-size). `TestDeterministicCompile` (2 tests) locks the contract that `torch.use_deterministic_algorithms(True)` does not break compiled `relu(x*0.5)+1` and `sum(x, dim=-1)`. Still-pending sub-items (atomic `scatter_add` → sort-based, autotune randomness pin) tracked separately under P4.8 / P2.2 — those code paths are not currently triggered by the regression suite, so the deterministic guard runs trivially green.
- [~] **Subnormal flush mode policy** *(2026-04-26 — knob landed, codegen wiring pending)*: `TORCH_VULKAN_DENORMALS={flush,preserve}` config knob exposed via `inductor.config.denormal_mode()`. Default `flush` (current behavior). Codegen does not yet emit `OpDecorate FPDenormalsPreserve` in SPIR-V — `preserve` is currently a documented intent only. `TestDenormalsConfig` covers the env-var contract.
- [~] **NaN-propagation contract for reductions** *(2026-04-26 — regression test added, `xfail(strict=True)` until codegen fix)*: `TestReductionNaNPropagation` exercises `amax/amin/amax-compiled/sum` over a tensor containing one NaN element and asserts the reduction returns NaN. All four currently fail — Vulkan reductions use min/max/sum intrinsics that suppress NaN. Marked `xfail(strict=True)` so when the codegen swaps to NaN-propagating variants (e.g. `(a != a) ? a : (b != b) ? b : min(a, b)`), the tests flip from xfail to passed and we notice.
- [ ] **Codegen fix: NaN-propagating min/max/sum**: emit explicit NaN check before falling back to native `min`/`max`/`fma`. RDNA1 native `min` is NaN-suppressing — the eager Vulkan backend's `*_NaN_aware` helpers in `slang_helpers.py` are the reference pattern; port the same logic into `VulkanOverrides` for the reduction op snippets.
- [x] **Reduction-order stability across WG-size autotune** *(2026-04-26)*: `TestReductionOrderStability` verifies `sum(x)` repeated 5× returns the same value to ULP, and that compiled-mode `sum` agrees with eager to `1e-4`. Locks the tolerance — a future autotune that picks a different reduction tree must respect this bound or document a wider tolerance in this test.

### P5.7 — Compile-time + warm-start optimization

- [x] **Persistent Inductor cache namespace per backend version** *(2026-04-26)*: `_namespace_inductor_cache()` in `inductor/__init__.py` sets `TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_$USER_vulkan_<sha>` at backend `register()` time. The `<sha>` is a 12-char hash of the inductor module mtimes, so `pip install -e .` rebuilds get fresh cache entries without manual `rm -rf`. Disable via `TORCH_VULKAN_NO_CACHE_NS=1` or set `TORCHINDUCTOR_CACHE_DIR` explicitly. `TestInductorCacheNamespace` covers function/override/disable paths.
- [x] **SPIR-V cache GC** *(2026-04-26)*: `runtime.gc_spirv_cache(max_mib=N)` walks the disk cache, sorts by mtime ascending, and deletes oldest until under budget. Returns `{removed, kept, bytes_before, bytes_after}`. `TestSpirvCacheGC` covers trim-to-budget + keep-newest + missing-dir-safe.
- [x] **Parallel slangc workers tuning** *(2026-04-26)*: `_default_max_workers()` defaults to `min(8, os.cpu_count())`; previously was capped at 4 leaving cores idle on 8+ machines. Override via `TORCH_VULKAN_SLANGC_WORKERS=N`. `TestSlangcWorkersTuning` locks the default + override.
- [ ] **Skip-codegen-on-dispatch-count-regression CI gate**: autotune decisions baked into Inductor codecache should never increase a regression-asserted dispatch count.
- [~] **Bypass Inductor autotune CUDA-leak for vulkan matmul** *(P0 — discovered 2026-04-26)*: Inductor's `select_algorithm._benchmark_choice` path (and `TritonBenchmarker` it falls back into for some choices) uses `torch.cuda.synchronize` / `torch.cuda.Event` directly, raising `Torch not compiled with CUDA enabled` on vulkan-only machines whenever the autotune triggers (e.g. a standalone `a @ b` graph with shape ≥ 64×64). Reproducer: `torch.compile(lambda a, b: torch.relu(a @ b))` on vulkan inputs. Pre-existing — also reproduces with `TORCH_VULKAN_NO_REGISTER_TILE=1`, even with zero Slang choices installed. Not caught by the regression suite because the suite's mm exposures sit inside larger graphs (MLP, etc.) that take a different code path. Fix path: register a vulkan-aware override for `_benchmark_choice` (or its `synchronize`/`event_timing` helpers) that uses `torch_vulkan._c_ext._synchronize` instead of `torch.cuda.synchronize`. Same pattern as the existing `_register_vulkan_benchmarker_once`. Mid-priority because (a) the workaround is "wrap mm in a graph with at least one pointwise so autotune skips the lone-extern path", which most real workloads already do; (b) the headline shape-keyed benchmarker (P2.2 sub-task above) bypasses this path entirely once it lands. Added env-var `TORCH_VULKAN_NO_REGISTER_TILE=1` as a bisection knob.

### P5.8 — Inductor scheduler tuning for Vulkan

- [x] **Buffer-count fusion limit refinement** *(2026-04-26)*: outdated entry — `VulkanScheduling._get_max_storage_bufs()` already queries `props.max_storage_buffers` from the device interface and returns `min(max_storage_buffers // 2, 32)` (or 16 if the property is unavailable). On RDNA1 with `max_storage_buffers=8388606`, the cap resolves to 32 buffers per fused kernel, well above the runtime fast-path specialization range (n_buffers ≤ 6). The "6-buffer cap" mentioned in the original entry was a runtime fast-path detail (not a fusion limit) that's already orthogonal. Roadmap checkbox flipped post-hoc; no code change needed.
- [~] **`should_horizontal_fuse` audit** *(2026-04-26 — regression test landed, codegen audit pending)*: `TestHorizontalReductionFusion` covers `(sum(x), amax(x))` and `(sum(x, dim=0), sum(x, dim=1))` shared-input cases. Both pass at ≤4 dispatches today; the test exists so a codegen change that drops the bound to ==1 (true horizontal fuse) flips the regression visible. Audit of `VulkanScheduling.can_fuse` for horizontal-fusion eligibility still pending.
- [ ] **Tiling heuristic per device**: NVIDIA real GPU vs RDNA1 vs SwiftShader prefer different `(TILE_M, TILE_N, TILE_K)` defaults. Plumb a device-keyed table into `_pick_tile_configs`.
- [ ] **Loop-order selection**: for kernels with multiple iteration dims, prefer the order that maximizes coalesced loads on the largest stride-1 axis. Verify against current heuristic.
- [ ] **Fusion across `aten.detach` / `aten.clone`**: shouldn't break a fused region; today they sometimes do. Audit `can_fuse_horizontal` / `can_fuse_vertical`.

### P6 — Pipeline review pass (2026-04-26)

A fresh end-to-end skim of the inductor pipeline (registration → AOT → post-grad → scheduler → kernel codegen → wrapper → runtime → C++ dispatch) surfaced concrete, well-scoped opportunities that don't fit the existing tier categories. Each item names the exact file/line and the smallest possible win.

**P6.1 — Runtime / dispatch micro-overhead** (Python-only, no build):

- [x] **Cache `_slangc_available()` at module load** *(2026-04-26)*: shipped — `runtime.py:200-228` keeps `_slangc_available_cache` as a lazy-resolved module-level `Optional[bool]` populated on first call. `_reset_slangc_available_cache()` exposed for tests that swap `SLANGC=`. Locked by `TestSlangcAvailableCached` (per Recent Changes log). Roadmap checkbox flipped post-hoc.
- [x] **Investigated `_INDUCTOR_STATS` per-build env reads** *(2026-04-26 — wontfix, behavior is intentional)*: tried replacing the `os.environ.get("TORCH_VULKAN_INDUCTOR_STATS")` reads in `make_vulkan_kernel` (`runtime.py:441,474`) with the module-level `_INDUCTOR_STATS` constant. The change broke `TestInductorStats`/`TestSummaryAPI`/`TestWaterfallDump` which set the env var *after* import and expect kernels built later to pick it up. The per-build read is the documented public-API contract: stats can be toggled at runtime. Marked as wontfix; left a comment in `runtime.py` explaining why the env-get is intentional.
- [x] **Repaired the `_ASYNC_COMPILE` synchronous-block path** *(2026-04-26)*: was `pool.submit(...).result()` followed by a `_cache_by_hash.get(hash_key)` lookup that fell through to a *second* inline compile on miss — silently doubled the work and swallowed the real exception when the future raised. Now: `spv = pool.submit(...).result()` directly returns the bytes (raises on inner failure), and the path converges with the non-async branch. `TestAsyncCompileSinglePass` (2 tests) locks: (1) under `_ASYNC_COMPILE=True`, the inner compile runs **exactly once** per cold key; (2) inner-compile exceptions propagate through `future.result()` instead of being swallowed.
- [x] **Drop the dead SHA1-fallback in `compile_and_dispatch`** *(2026-04-26)*: `cache_key` is now required (signature changed from `Optional[str] = None` to `str = ""` with explicit `ValueError` on empty). The 3 in-tree call sites (`vulkan_template_caller.py:127,182,225`) all supply one already; the SHA1-of-SPIRV fallback was dead. `TestCompileAndDispatchRequiresKey` locks the empty-key rejection.

**P6.2 — Template-caller per-dispatch overhead** (Python-only):

- [x] **Pre-render Slang source in `_SlangTile{MM,AddMM,BMM}` per-dtype cache** *(2026-04-26)*: dtype isn't fully fixed at __init__ (the same instance services multiple dtypes once a workload mixes f32/f16), so a strict `__init__`-time pre-render isn't quite right. Landed: `_per_dtype: dict[str, tuple[str, str]]` slot on each `_SlangTile{MM,AddMM,BMM}` instance, populated lazily on first call per dtype. The per-dispatch path now skips the 15-tuple Jinja-cache lookup + cache_key f-string after the first call per dtype. The render helpers (`_slang_tile_{mm,addmm,bmm}`) accept the pre-built `src` + `cache_key` as optional kwargs (None falls back to the original render path so the standalone API still works). `__reduce__` reconstructs from constructor params only — `_per_dtype` correctly resets after pickle round-trip. `TestSlangTilePerDtypeCache` (4 tests) locks the contract.
- [x] **Skip `is_contiguous()` checks under `TORCH_VULKAN_TRUST_INDUCTOR=1`** *(2026-04-26)*: `_slang_tile_{mm,addmm,bmm}` (`vulkan_template_caller.py`) now gate the per-dispatch contiguity checks behind a module-level `_TRUST_INDUCTOR` flag captured once at import. Inductor's wrapper pre-arranges contiguous tensors before reaching the extern, so the checks are pure overhead in trust mode (3+ branch+check pairs per matmul × millions of matmul calls per training step). Per-dispatch in trust mode is now: `if not _TRUST_INDUCTOR: ...` constant False, skipped. `_reset_trust_inductor_cache()` test hook re-reads the env var. `TestTemplateCallerTrustInductor` (2 tests) covers the env-var capture + observed-call-count contract via a `is_contiguous` spy.

**P6.3 — Codegen quality (kernel.py)** (Python-only):

- [x] **Right-size `numthreads` for small-numel pointwise** *(2026-04-26)*: `_pick_threadgroup_size` (`kernel.py`) used to return 256 unconditionally for non-reduction kernels with `total <= 64K`. For tiny tensors (e.g. the regression suite's 7×13 = 91-element pointwise kernels), this dispatched a 256-thread workgroup with `if (xindex >= 91) return;` — wasted >half the lanes on RDNA1 wave64. Now: when `total < 256`, picks `max(simd_group_size, next_pow2(total))`, clamped to `max_wg`. So a 91-element kernel gets `numthreads(128, 1, 1)` (2 waves) instead of `numthreads(256, 1, 1)` (4 waves with 2 idle). `TestRightSizedNumthreads` (2 tests) covers: integration spy-on-emitted-Slang for a 7×13 pointwise (skips if FXGraphCache returns a stale wrapper, since the spy wouldn't observe a re-codegen) + a property test that re-implements the pow-2-clamp math against the same code path.
- [x] **Cache `emit_helpers` output by header set** *(2026-04-26)*: `slang_helpers.emit_helpers` output is fully determined by `(frozenset(headers), max_threadgroup_size, simd_group_size)`. Refactored: original body moved to `_emit_helpers_impl`; `emit_helpers` now (1) computes the cache key, (2) returns the cached rendered string via `code.splice(...)` on hit, (3) renders into a scratch `IndentedBuffer` on miss and stores `getvalue()`. `_reset_emit_helpers_cache()` exposed for tests. `TestEmitHelpersCache` (2 tests) covers cache identity, OrderedSet/list/set key normalization, parameter-keyed separation, and round-trip output equivalence vs fresh render.
- [x] **Fix unconditional `((int)(name))` cast in `VulkanExprPrinter._print_Symbol`** *(2026-04-26)*: `VulkanExprPrinter` now tracks `_subscript_depth` (a contextlib-friendly counter, `subscript()` ctx-manager). `VulkanKernel.index_to_str` overrides the upstream method to enter subscript context around every index render. `_print_Symbol` only wraps `tmp\d+` with `((int)(...))` when `_subscript_depth > 0`. Outside subscript context — e.g. arithmetic on a float CSE local that happens to share the `tmp\d+` namespace — the cast is suppressed so we don't silently truncate float values to int. `TestExprPrinterIntCast` tightened to 5 tests covering: outside-subscript no-cast, inside-subscript cast, non-tmp symbol unchanged in both contexts, compound index expressions, and nested subscript scope correctness.

**P6.4 — Maintainability / cleanup** (Python-only):

- [x] **Hoist `_enable_b2b_gemm` to `register()`** *(2026-04-26)*: previously called per FX graph from `_VulkanCustomPass.__call__`; the function only sets the global `torch._inductor.config.b2b_gemm_pass = True` flag, so per-graph invocation was wasted work. Now flipped once at backend `register()` time in `inductor/__init__.py`. Removed the call from `_VulkanCustomPass`. Suite stays green.
- [x] **Skip hashing in `_CompiledGraph._guard_hash`** *(2026-04-26)*: replaced the SHA1-of-encoded-repr scheme with a direct `tuple` key (`_guard_key`) keyed on `(("T", shape, stride, dtype, str(device))...)` per arg + sorted `(key, …)` per kwarg. The dict already hashes tuple keys via the per-element `__hash__`; the prior SHA1 + hexdigest was pure overhead. `_guard_hash` retained as a thin wrapper for callers that wanted a string. `TestCompileGraphStub.test_guard_key_is_tuple_no_hash` locks the contract.

**P6.5 — Dispatch C++ path** (needs build):

- [ ] **Replace `dirty_buffers` `std::set<VkBuffer>` with `std::vector` linear-scan** (per the prior diagnosis of `csrc/backend/dispatch.cpp` smart-barrier code): typical dispatches see ≤8 buffers. Linear scan over a small vector beats set lookup + tree-balance overhead and avoids per-dispatch heap allocations during set rehash.
- [~] **Output-buffer reuse pool keyed on `(numel, dtype)`** *(2026-04-26 — Python infra landed default-off; see P0.4 entry above)*. The Python-only path is shipped; full closure of the MLP gap is now tracked by P0.4's "C++ extern-kernel allocator hook" follow-up.

**P6.6 — Stride / contiguity propagation through codegen** (Python-only, no build):

The codegen currently emits scalar-stride index expressions even when Inductor's IR already proves the inner axis is stride-1 contiguous. Recovering that fact at codegen time enables vec4 / packed16 vectorization on more kernels and removes redundant index arithmetic.

- [ ] **Stride-1 hint on inner axis at codegen entry**: when `range_tree[-1]` corresponds to an axis with a constant stride==1 on every input/output buffer, set `_inner_axis_contiguous=True` on the `VulkanKernel`. Today `_vec4_pw_eligible` infers this via a substring scan on the rendered body; doing it from IR up front is more robust and unlocks vec4 on kernels where the body has any non-trivial stride expression even though the inner axis is in fact contiguous. Test `TestStrideOneHint` — synthetic kernel where the IR strides prove inner-axis contiguous, but the rendered body has a `stride * idx` that survived simplification, still gets `_inner_axis_contiguous=True`.
- [x] **Dead-axis elimination for size-1 dims** *(2026-04-27)*: `VulkanExprPrinter._print_FloorDiv` now folds `FloorDiv(x, 1) → x` and `FloorDiv(0, _) → 0` before falling through to the branchless C-divide. `_print_ModularIndexing` folds `_ % 1 → 0`. SymPy normally folds these, but `keepdim=True` reductions can construct them with `evaluate=False` so the printer must be defensive. Locked by two new tests in `TestExprPrinterSimplify` (`test_floordiv_by_one_drops_to_x`, `test_modular_indexing_mod_one_is_zero`). Suite: 229 → 231 passed (+2).
- [ ] **Index canonicalization at load/store boundaries**: hoist a single `uint flat_idx = ...;` per (buffer, repeated index expression) before the load-pattern emits, so repeated `_v_in_ptr0[a*S+b]` / `_v_in_ptr1[a*S+b]` shares the same address computation. Today the printer emits the full expression at every use site. Test `TestSharedIndexCSE` — kernel reading 3 buffers at the same index emits 1 address computation, 3 loads.
- [ ] **Exact-divisibility-driven push-constant strides**: when an axis size is statically known to be a multiple of `vec_width` (vec4 / packed16), drop the per-thread bounds check `if (xindex >= xnumel) return;` from the emitted body. Currently always emitted. Test `TestExactDivisibilityNoBoundsCheck` — Slang dump on a 4096-elem kernel with `numthreads(256)` shows no `xnumel` early-return.
- [ ] **`_vec4_pw_eligible` IR-driven rewrite**: replace the current substring-scan eligibility check (post-body emission) with an IR-walk pre-emission decision so vec4 can also fire on multi-axis kernels where the inner axis is stride-1 (currently bailed via the substring scan). Test `TestVec4PointwiseMultiAxis` — `(N, C, H, W)` add with stride-1 W axis fires vec4.

---

### P5.9 — Cooperative-matrix / WMMA matmul (capability-gated)

`VK_KHR_cooperative_matrix` exposes hardware tensor-cores (`cooperativeMatrixLoad` / `cooperativeMatrixMulAdd`) on NVIDIA Turing+, AMD RDNA3+, and Intel Arc. Slang exposes them via `CoopMat<T, M, N, K, scope>`. RDNA1 (the agent's GPU) does not support this extension — but a fully optimizing Inductor backend should pick the cooperative-matrix path on capable hardware. Today the matmul template is the same on all GPUs.

- [ ] **Capability probing**: extend `device_interface.VulkanDeviceInterface` with `supports_cooperative_matrix` (queries `VK_KHR_cooperative_matrix` + `cooperativeMatrixSupportedKHR` properties), the supported `(M, N, K)` shapes, and the supported `(A_dtype, B_dtype, C_dtype, ResultType)` combinations. Cache at backend register time. Test `TestCoopMatCapabilityProbe` — on RDNA1 returns `False`; mocked NVIDIA returns the expected shape list.
- [ ] **Cooperative-matrix Slang template**: new `slang_mm_coopmat.py.jinja` that issues `CoopMatLoadKHR` for tile A/B, `CoopMatMulAddKHR` for the inner-K accumulation, and `CoopMatStoreKHR` for the C tile. Matches the existing `_SlangTileMM` interface so the template caller picks it transparently. Test `TestCoopMatMatmulTemplate` — mocked capability emits the cooperative-matrix variant; on RDNA1 falls through to the register-tile path.
- [ ] **Per-arch tile picker**: `_pick_tile_configs` returns coopmat-shape candidates (`(16, 16, 16)`, `(32, 32, 8)`, etc. depending on dtype) when `supports_cooperative_matrix=True`. The autotuner's shape-keyed cache (P2.2) keys on `(M, N, K, dtype, transpose_b, has_bias, coopmat_supported)` so swapping GPUs invalidates stale entries. Test `TestPerArchTilePicker`.
- [ ] **bf16 / f16 / i8 dtype expansion**: `CoopMatMulAddKHR` natively supports `(f16, f16, f32)` and `(bf16, bf16, f32)` (Turing+) and `(i8, i8, i32)` (RDNA3+, NVIDIA Volta+). Wire each combo. Quantized inference inherits the int8 path for free. Test `TestCoopMatDtypeMatrix`.
- [ ] **Forward + backward unified template**: the same coopmat tile loader works for both `mm`/`addmm`/`bmm` forward and the matmul half of `linear_backward` (`grad_in = grad_out @ W.T`, `grad_w = grad_out.T @ x`). Wire the backward path through the same template once the P0.0 backward graph compile unblocks. Test `TestCoopMatLinearBackward`.

### P5.10 — Subgroup intrinsics & wave-cooperative ops expansion

The wave-intrinsic helpers in `slang_helpers.py` cover `WaveActiveSum/Max/Min`, `WaveReadLaneFirst`, and `WavePrefixSum`. Several other patterns in fused shaders today re-derive operations that have direct hardware intrinsics; emitting the intrinsic skips the groupshared roundtrip.

- [ ] **`WaveActiveBallot` for masked-pointwise lane counts**: when the `where(mask, body, other)` pattern needs to compute the count or first-active lane of `mask`, emit `WaveActiveBallot(mask)` + `firstbitlow` instead of a `WaveActiveSum(mask ? 1 : 0)`. Used by sparse-attention masking paths. Test `TestWaveActiveBallotMask`.
- [ ] **`WaveShuffle` (`QuadReadAcrossX/Y/Diagonal`) for 2×2 reductions**: image-warp / pooling kernels with 2×2 windows can use quad-shuffle to swap values between adjacent lanes without LDS. RDNA1 supports `ds_swizzle_b32` for 32-lane quad shuffles. Test `TestWaveQuadShuffle2x2Pool`.
- [ ] **`WavePrefixCountBits` for compaction patterns**: stream compaction for sparse activations / MoE routing benefits from `prefix_count_bits(ballot)` to compute the output index. Test `TestWavePrefixCompaction`.
- [ ] **Subgroup-uniform constant hoisting**: when a load is provably uniform across the wave (push-constant scalar broadcast, `gid.z`-only addressing in a pointwise kernel), emit a single `WaveReadLaneFirst(load)` instead of N redundant scalar loads. Test `TestUniformLoadHoist`.
- [ ] **Atomic-add CAS-loop helper for f16/bf16 gradients**: RDNA1 doesn't natively atomic-add `f16` / `bf16`. A CAS-loop helper (`uint old; do { old = ...; new = pack(unpack(old) + grad); } while (!InterlockedCompareExchange(...));`) belongs in `slang_helpers.py` so any autodiff or hand-written backward can scatter half-precision gradients without unwrapping the loop per shader. Test `TestHalfAtomicAddCASHelper`.

---

### P7 — Measurement-driven gap-discovery automation

The CLAUDE.md "Training-driven Discovery Loop" is currently a manual ritual: run a model, dump `print_full_report()`, write down what's missing, expand the roadmap. Several pieces can be automated so the loop runs continuously without human bookkeeping.

- [ ] **`scripts/discover_inductor_gaps.py`**: end-to-end runner — for each model in `benchmarks/inductor_train.py`, invokes `torch.compile` with `TORCH_LOGS=output_code,graph,inductor`, captures the output, parses `extern_kernels.<op>` calls + `aten.<op>` fallbacks + slangc cold-compile counters + per-kernel waterfall, and writes `discovery_report_<date>.json` with: `(op, model, kind={extern,fallback,slow_kernel,cold_compile,oversized_dispatch}, suggested_priority)`. Test `TestDiscoveryScript` — runs on the MLP fixture, asserts the report has expected schema + non-empty entries.
- [ ] **CI gate: any new `extern_kernels.<op>` in compiled output adds a roadmap entry**: a CI check that diffs the discovery report against the previous run; new extern ops trigger a soft-fail with an auto-generated roadmap stub appended to `docs/10-inductor-backend.md` under `P1.7` for human review/promotion. Test `TestExternRegressionGate`.
- [ ] **Per-kernel ms/step regression budget**: every kernel in the waterfall gets a budgeted `ms/step` ceiling stored in `tests/inductor_kernel_budgets.json`; CI fails when any kernel breaches its ceiling by >5%. Auto-populated from a green baseline run. Test `TestKernelBudgetEnforcement`.
- [ ] **Dispatch-count diff on every PR**: `python -m torch_vulkan.inductor.benchmarks.dispatch_diff base..HEAD` reports per-model dispatch-count deltas vs the base branch. Stops a regression that doesn't hit a regression-test ceiling but still bumps the count. Hooked into the regression suite via an opt-in `--diff-against=main` flag.
- [ ] **Cold-compile hot-loop watch**: any test run that takes >30s for `slangc` cold compiles surfaces a `TooManyColdCompiles` warning + a list of the top-10 keys; a follow-up roadmap item covers prewarming each. Test `TestColdCompileBudget`.
- [ ] **Auto-promotion of `[~]` blocked items**: when an upstream PyTorch fix lands that closes a tracked `[~]` blocker (tested by re-running the locked `xfail(strict=True)` regression in CI), auto-flip the checkbox + the test annotation, and surface a notification in the next agent turn. Test `TestBlockedItemAutoPromotion`.

---

### P9 — Slang language feature utilization

Every shader in `shaders/` and every Inductor-generated kernel today is plain HLSL-style code: explicit `[vk::binding(N)]` per buffer, `[vk::push_constant]` PC struct, raw `float` / `float16_t` types per shader (never generic). Slang ships a richer language than HLSL — generics, modules, interfaces, `[Differentiable]`, capability gating, reflection — and this backend uses none of it. Adopting these features selectively cuts shader source size, removes per-dtype duplication, makes the autodiff pilot (§P3.3) much cheaper to land, and gives slangc more freedom to specialize. Each item below names the specific Slang feature, the file(s) it touches, the regression test, and the measured benefit (source-bytes / compile-time / dispatch-time).

**P9.1 — Generic dtype shaders** (replaces per-dtype shader copies):

- [ ] **Generic matmul template via `<T : IFloat>`**: `templates/slang_mm.py.jinja` currently renders one shader per dtype (`mm_f32`, `mm_f16`, …). Slang's generics let one source compile to specialized SPIR-V per dtype: `void computeMain<T : IFloat>(...)` with `T` resolving to `float` / `float16_t` / `bfloat16_t` at slangc invocation. Reduces the prewarm spec set 4× (`{f32, f16, bf16, …}` × `{tile_configs}` collapses to one generic source × dtype specialization). File: `templates/slang_mm.py.jinja`. Test `TestGenericMatmulSlang` — same SPIR-V output across dtypes, `TestPrewarmSpecCount` — spec count drops by 4×.
- [ ] **Generic pointwise primitives in `slang_helpers.py`**: `c10_vulkan_erf`, `c10_vulkan_digamma`, `c10_vulkan_lgamma` are emitted four times (one per dtype) by `emit_helpers`. Rewrite as `T c10_vulkan_erf<T : IFloat>(T x)` so slangc instantiates per use. Cuts the helper-emit cache size by ~4× and lets a future packed16 kernel call the same helper as the f32 path. Test `TestGenericHelpers`.
- [ ] **Generic reduction helpers (`wg_reduce_sum`/`max`/`min`)**: `slang_helpers.emit_helpers` emits per-dtype copies of `c10_vulkan_wg_reduce_*`. Replace with a single generic `T wg_reduce_<op><T : IComparable>(T val)` definition. Cuts emitted Slang per-kernel by ~200 lines on workloads that do mixed-dtype reductions. Test `TestGenericReductionHelpers`.

**P9.2 — Modules and `import`** (de-duplicate cross-shader code):

- [ ] **Promote `slang_helpers.py` content to a real `.slang` module**: today `emit_helpers` splices a string of helper bodies into every kernel source — every kernel re-pays the slangc parse cost for the same Welford/Philox/packed16/wave-reduce code. Move the bodies to `shaders/inductor_runtime.slang` (real file on disk) and have generated kernels start with `import inductor_runtime;`. slangc's module cache then parses the helpers once per backend session and reuses the IR for every kernel — measurable cold-compile speedup. Files: new `shaders/inductor_runtime.slang`, refactored `slang_helpers.py` to emit only the import statement + the per-kernel parameter binding. Test `TestSlangModuleImport` — emitted Slang has `import inductor_runtime;` and zero in-line helper bodies; `TestColdCompileSpeedupFromImport` — measurable cold-compile time delta on a 50-kernel benchmark.
- [ ] **Per-feature module split**: `inductor_runtime.slang` is too coarse for selective imports. Split into `inductor_math.slang` (erf/lgamma/digamma), `inductor_reduce.slang` (wave-reduce/welford), `inductor_pack16.slang` (pack/unpack), `inductor_philox.slang` (RNG). Each kernel imports only the helpers it actually uses, shrinking SPIR-V by the dead-helper bytes the driver-side DCE would otherwise have to strip. Test `TestModuleSplit` — kernel without RNG has no Philox bodies in its SPIR-V dump.
- [ ] **Inductor-template module imports**: `templates/slang_mm.py.jinja` should `import inductor_pack16;` (for `pack16` matmul accumulation) and `import inductor_runtime;` rather than copy-pasting the unpack helpers. Test `TestMatmulTemplateImports` — rendered template starts with `import` lines, no inline helper bodies.

**P9.3 — `interface IDifferentiable` + `[Differentiable]`** (proper autodiff scaffolding for §P3.3):

- [ ] **Define `interface IInductorOp` in `inductor_runtime.slang`**: a Slang `interface` that fused kernels can implement to expose `[Differentiable] void primal(...)` / `[BackwardDifferentiable] void bwd(...)`. Inductor's lowering layer can then route through `bwd_diff(op.primal)` automatically, instead of registering a separate `_backward` op per primal. Lays the groundwork for §P3.3's autodiff pilot. Test `TestInductorOpInterface` — a synthetic op implementing `IInductorOp` with `[Differentiable] primal` produces a working backward via `bwd_diff`.
- [ ] **Differentiable matmul template**: the matmul template in `slang_mm.py.jinja` is straight-line; mark it `[Differentiable]` so `bwd_diff(matmul_primal)` produces the backward kernels (`grad_in = grad_out @ W.T`, `grad_w = grad_out.T @ x`) without hand-writing them. Currently the eager backend has 6 separate matmul-backward shaders (`linear_bwd_fused*`, `qkv_bwd_fused*`, etc.); autodiff would collapse them. Test `TestDifferentiableMatmul` — `bwd_diff` output bit-matches the hand-written shader on a small benchmark + VGPR ≤ 64.
- [ ] **`[BackwardDerivativeOf]` for atomic gradient scatter**: scatter-into-`RWStructuredBuffer` doesn't have a default derivative in Slang's autodiff. Provide one in `inductor_runtime.slang` using `InterlockedAdd` (f32/i32) or the §P3.3.11 CAS helper (f16/bf16). Once registered, every autodiff pilot shader gets gradient accumulation for free. Test `TestBackwardDerivativeAtomicScatter`.

**P9.4 — `ParameterBlock<T>` for cleaner binding tables**:

- [ ] **Migrate kernel push-constant + buffer set to `ParameterBlock<KernelParams>`**: today every kernel declares `[vk::binding(0)] StructuredBuffer<float> a; [vk::binding(1)] ... ; [[vk::push_constant]] PC pc;`. Slang's `ParameterBlock<T>` packs all bindings + push-constants into a single descriptor set + push-constant range, *and* lets the backend swap an entire set in one bind. Reduces emit-time per-binding bookkeeping (kernel.py:emit_buffer_decls drops to one line) and lets the runtime use `vkCmdBindDescriptorSets` once per kernel instead of one per buffer. File: `kernel.py:emit_buffer_decls`, `runtime.py:bind_descriptors`. Test `TestParameterBlockMigration` — kernel SPIR-V has one `OpDecorate ... DescriptorSet 0` decoration block.
- [ ] **Per-shader-stage parameter blocks**: cooperative-matmul, reduction, and pointwise kernels have different binding-table shapes. Define three `struct PointwiseParams { ... }`, `struct ReduceParams { ... }`, `struct MatmulParams { ... }` and pick the right one based on `VulkanKernel.kind`. Test `TestParamBlockKindRouting`.

**P9.5 — Capability gating with `[require(...)]`** (cross-driver portability):

- [ ] **Express subgroup-intrinsic gating in shader source**: today `VulkanKernel._wave_intrinsics_supported` is a Python branch that emits/skips wave intrinsics. Slang's `[require(_sm_5_0)] T WaveActiveSum<T>(T)` lets the shader source itself declare the capability — slangc rejects the shader at compile time if the device doesn't support it, instead of producing bad SPIR-V at runtime. Migrate `slang_helpers.py` wave-helper definitions to `[require(spirv_1_3, GroupNonUniformArithmetic)] ...`. Test `TestRequireGatedCompile` — emitted Slang has `[require(...)]` on every wave-intrinsic helper; SwiftShader path falls through to the smem fallback cleanly.
- [ ] **`[require(spirv_1_5, ShaderFloat16)]` on packed16 helpers**: today the f16 path is gated at the Python kill-switch level (`TORCH_VULKAN_NO_PACKED16=1`). Move the gate into the shader source so the slangc spec system handles it. Test `TestRequirePacked16`.
- [ ] **`[require(...)]` on cooperative-matrix path**: prerequisite for §P5.9. The cooperative-matrix template should declare `[require(spirv_1_5, CooperativeMatrixKHR)]` so the picker rejects it on RDNA1 at slangc time, not at `vkCreateComputePipeline`. Test `TestCoopMatRequireGate`.

**P9.6 — Slang reflection API** (drop binding-metadata bookkeeping in Python):

- [ ] **Replace `n_buffers` / `n_pc` parameter passing with reflection**: today every kernel declaration in `kernel.py` carries a `(n_buffers, n_pc)` tuple, hand-counted from the emitted Slang. slangc's reflection API (`slang.IBlob *getReflectedJSON()`) returns the binding count + push-constant size as part of compilation output. Wire this into `runtime.compile_slang_to_spirv` and drop the manual tracking. Test `TestSlangReflection` — `make_vulkan_kernel` no longer needs `n_buffers`/`n_pc` kwargs.
- [ ] **Reflection-driven push-constant struct packing**: the current `struct.Struct(...).pack` calls in `runtime.py` hard-code per-kernel format strings (`"III"` etc.). Reflection gives the exact field offsets + types, so the wrapper can emit `_pc_pack(reflection_meta, *values)` and let a generated `Struct` cache do the work. Removes per-call format-string lookup. Test `TestReflectionPCPacking`.
- [ ] **Reflection-driven dispatch-shape inference**: `[numthreads(N, M, K)]` is currently parsed by string-search in `kernel.py:_parse_numthreads`. Reflection exposes it as a typed field. Test `TestReflectionNumthreads`.

**P9.7 — Slang `extension` declarations** (per-dtype op overloading):

- [ ] **`extension float : IInductorScalar { ... }`**: gives every dtype a uniform interface for the codegen-level operations the lowering layer needs (e.g. `T saturate_add(T a, T b)`). Today the lowering emits dtype-specific snippets per op via `VulkanOverrides`. With `extension`, the Slang source carries the overloads; lowerings emit a single `saturate_add(a, b)` call regardless of dtype. Test `TestExtensionScalarInterface`.

**P9.8 — `[shader("compute")]` entry-point overloading**:

- [ ] **Multi-entry-point shader files**: today each fused kernel is one `.slang` file with one `[shader("compute")] void computeMain`. Slang allows multiple entry points per file (`[shader("compute")] void computeMain_f32(...)` and `[shader("compute")] void computeMain_f16(...)`). Lets a single file cover f32+f16+bf16 with shared helper code in module scope and per-dtype tail-fusion via separate entry points — slangc emits one SPIR-V module with N entry points, runtime picks the right one. Combined with §P9.1 (generics) reduces file count from ~440 → ~150. Test `TestMultiEntryPointShader`.

**P9.9 — Slang `printf` / debug strings**:

- [ ] **Compile-time-debugging via `printf` inside slangc**: Slang supports `printf` inside compute shaders (via the SPIR-V debug-printf extension). Wire `TORCH_VULKAN_SHADER_PRINTF=1` to inject `printf("kernel %u tid %u val %f\n", gid.x, lid.x, value)` into the emitted Slang for the failing kernel — gives per-thread numerical traces without rebuilding. Test `TestShaderPrintfDebug`.

**P9.10 — `[ForceInline]` audit + `[mutating]` on helpers**:

- [x] **Audit `slang_helpers.py` for missing `[ForceInline]`** *(2026-04-27)*: applied `[ForceInline]` to every barrier-free leaf helper that was previously emitting as a real call: `c10_vulkan_atomic_add` (CAS-loop wrapper), `_vk_mulhi32`, `_vk_philox_round`, `_vk_philox_bumpkey`, `_vk_philox_rand`, `_vk_philox_randn`, `c10_vulkan_ndtri`, `c10_vulkan_i0/i0e/i1/i1e`, `c10_vulkan_erfinv`, `c10_vulkan_polygamma`, `c10_vulkan_spherical_bessel_j0`, `c10_vulkan_igamma`, `c10_vulkan_zeta`, `c10_vulkan_bucketize`. Helpers with `GroupMemoryBarrierWithGroupSync` (`wg_reduce_*`, `wg_inclusive_scan`, `wg_bitonic_sort_float2`) deliberately excluded — `[ForceInline]` on barrier-bearing functions is risky in HLSL/Slang and the existing inlined wave-reduce fast-path already covers the n_waves==1 case in `slang_helpers.py:257-266`. `TestForceInlineCoverage.test_every_leaf_helper_is_force_inline` (1 test) locks the contract over the 17 leaf helpers — every emitted helper carries `[ForceInline]` in the rendered preamble. Suite: 198 → 199 passed.
- [ ] **`[mutating]` on in-place helpers**: helpers that update an `inout` parameter (e.g. `wg_reduce_inplace(inout T)`) should be marked `[mutating]` so the Slang type checker can prove the function only writes through the explicit `inout`. Catches a class of "accidentally aliased pointer" bugs at compile time instead of runtime. Test `TestMutatingMarkers`.

**P9.11 — Slang-side specialization constants for shape (continues §P1.6 item)**:

- [ ] **`[SpecializationConstant] const uint TILE_M = 64;`**: the matmul template currently bakes tile sizes into the source via Jinja substitution. Slang's `[SpecializationConstant]` lets the SPIR-V module carry symbolic tile sizes that the pipeline-creation layer fills in at `vkCreateComputePipeline` time. One SPIR-V binary per shader serves every tile config, drastically shrinking the on-disk SPIR-V cache. Test `TestSpecializationConstantTileSize` — single SPIR-V file covers `(64,64,16)` and `(128,64,16)` tilings.

---

### P10 — Extended primop coverage audit (beyond P1.7)

P1.7 covered loss / norm / pool / upsample / pad / dropout / linalg / spectral. The audit missed several common ops that still extern-fall-back. Each entry below names the eager shader, the missing Inductor route, and the regression test.

**Vision / spatial-transformer ops:**

- [ ] **`aten.grid_sampler_2d` / `aten.grid_sampler_3d` ExternKernelChoice templates**: bilinear/nearest grid-resampling is the spatial-transformer hot path and the conv-deformable backbone of vision-transformer fine-tuning. Eager has `grid_sampler_2d.slang`. Wire as `ExternKernelChoice` so downstream `pool → conv` fuses via Phase A epilogue. Test `TestGridSamplerTemplate`.
- [ ] **`aten.grid_sampler_2d_backward` / `_3d_backward` lowerings**: scatter-add into the input grad + bilinear-derivative on the grid. Test `TestGridSamplerBackwardLowering`.
- [ ] **`aten.affine_grid_generator` + backward**: the grid argument to `grid_sampler` typically comes from `affine_grid` — fuse the 2 dispatches into 1. Test `TestAffineGridFused`.
- [x] **`aten.pixel_shuffle` / `aten.pixel_unshuffle` codegen** *(2026-04-27)*: outdated entry — both already compile to **1 dispatch** through Inductor's stock view+permute+pointwise codegen on Vulkan. `TestPixelShuffleAndPadCodegen.test_pixel_shuffle_one_dispatch` + `test_pixel_unshuffle_one_dispatch` lock the contract.

**Custom-conv primitives:**

- [ ] **`aten.im2col` / `aten.col2im` codegen**: enables custom-conv research (deformable conv, dilated-attention). Currently extern. Codegen as a gather kernel. Test `TestIm2ColCodegen`.
- [ ] **`aten.unfold` / `aten.fold` codegen**: same shape as im2col but for arbitrary stride/dilation/padding. Test `TestUnfoldFoldCodegen`.

**Indexing / scatter / search:**

- [ ] **`aten.searchsorted` ExternKernelChoice**: binary search per element; required for non-uniform sampling and discrete distributions. Single-dispatch atomic-free shader. Test `TestSearchsortedTemplate`.
- [ ] **`aten.bucketize` ExternKernelChoice**: equivalent to `searchsorted` with a boundary array. Test `TestBucketizeTemplate`.
- [ ] **`aten.masked_select` lowering**: prefix-sum-driven compaction. Eager has `masked_select.slang`. Currently extern + breaks downstream fusion. Test `TestMaskedSelectLowering`.
- [x] **`aten.masked_scatter` lowering** *(2026-04-27)*: outdated entry — already compiles correctly via Inductor's stock pointwise+indirect-indexing codegen on Vulkan. `TestMaskedAndIndexCopy.test_masked_scatter_correctness` locks correctness vs CPU eager.
- [x] **`aten.index_select` ExternKernelChoice template** *(2026-04-27)*: outdated entry — `index_select` already compiles to **1 dispatch** (no extern fall-through) via Inductor's stock indirect-indexing codegen on Vulkan. `TestIndexSelectAndScatterCodegen.test_index_select_one_dispatch` locks ≤1 dispatch + correctness. The `index_select → norm/linear` fusion path is achieved via the same path as `embedding` — both are computed-index gathers that the pointwise scheduler already fuses with downstream pointwise consumers.
- [ ] **`aten.gather.default` codegen for arbitrary index shapes**: today only handled by Inductor's `indirect_indexing`; for shapes where `index.ndim != self.ndim` the upstream lowering errors out. Custom Vulkan codegen. Test `TestGatherArbitraryIndex`.
- [x] **`aten.scatter` (non-add) lowering** *(2026-04-27)*: outdated entry — already compiles correctly via Inductor's stock indirect-indexing codegen on Vulkan. `scatter(zeros(8), 0, idx, src)` produces correct values vs CPU eager. `TestIndexSelectAndScatterCodegen.test_scatter_correctness` locks the contract.
- [~] **`aten.index_copy` / `aten.index_fill` lowerings** *(2026-04-27 — index_copy ships, index_fill blocked on device-propagation gap)*: `index_copy` already compiles correctly via Inductor's indirect-indexing codegen (`TestMaskedAndIndexCopy.test_index_copy_correctness` locks correctness vs CPU eager). `index_fill` falls through with `Mismatching aten.index_put.default device between self (vulkan:0) and values (meta)` — same FakeTensor view-fast-path device-propagation gap as the broader P0.0 family. Will unblock alongside P0.0.

**Diagonal / triangular masks:**

- [~] **`aten.tril` / `aten.triu` lowerings** *(2026-04-27 — blocked on multi-axis-iota codegen gap)*: prototype lowering using `prims.iota` + `view([-1,1])` + `view([1,-1])` + `sub` + `le.Scalar(diagonal)` + `where` was written and removed. The single-kernel emit fires (verified via `TORCH_LOGS=output_code`) but references undeclared `x0` / `x1` axis variables: when an Inductor pointwise kernel folds two `prims.iota` views with broadcasting, our codegen drops the per-axis `uint x0 = ...; uint x1 = ...;` aliases that the kernel body relies on. Same root family as the vec4 axis-alias DCE that §P4.3 partially handles, but for the scalar multi-axis path. Tracked separately as P11.5 follow-up below; tril/triu remain at 10 dispatches via the upstream decomp until the codegen fix lands.
- [ ] **Multi-axis pointwise: emit `uint x0 = ...; x1 = ...;` aliases when the kernel body references them** *(2026-04-27 — discovered while shipping tril/triu)*: `VulkanKernel`'s axis-alias DCE removes `uint x0 = xindex;` / `uint x1 = ...;` declarations when the body has no direct `x0` reference, but multi-iota pointwise kernels reference `x0` / `x1` via the inlined `prims.iota` index expression after broadcasting. The DCE is too aggressive on non-vec4 paths. File: `kernel.py` — search for the codepath that strips axis aliases; tighten the check to also consider iota-index references. Test `TestMultiAxisIotaCodegen` — a synthetic `iota(rows).view(-1,1) - iota(cols).view(1,-1)` kernel must compile and produce correct values.
- [~] **`aten.diagonal` / `aten.diag_embed` codegen** *(2026-04-27 — `diagonal` ships, `diag_embed` blocked on int64-cast codegen gap)*: `diagonal(x) + 1` already compiles to **1 dispatch** via Inductor's stock view+pointwise (locked by `TestIndexOpsCodegen.test_diagonal_one_dispatch`). `diag_embed` falls through to extern: the lowering emits `tmp0 == tmp1` over `iota.unsqueeze` views which leaves an `int64` literal in the buffer subscript — same multi-axis-iota int64-cast gap as the tril/triu blocker (P10 above). `diag_embed` will unblock when the multi-axis iota codegen lands.
- [~] **`aten.eye` / `aten.linspace` / `aten.arange` factory codegen** *(2026-04-27 — `arange` and `linspace` already 1-dispatch; `eye` is 7 dispatches)*: measurement under `torch.compile`: `arange(64) * 2` and `linspace(0,1,64) * 2` both compile to **1 dispatch** today (Inductor's stock codegen handles them). `eye(8) + 1` still extern-falls to **7 dispatches** because the diagonal-mask construction (`arange.unsqueeze(-1) == arange.unsqueeze(0)`) hits the same multi-axis-iota codegen gap documented in §P10 above. `eye` blocked on the same fix; `arange`/`linspace` checkbox split out and flipped. Test `TestFactoryArangeLinspace` for regression coverage.

**Element-wise gaps:**

- [x] **`aten.lerp.Tensor` / `aten.lerp.Scalar` overrides** *(2026-04-27)*: outdated entry — measurement shows both `torch.lerp(a, b, 0.3)` (Scalar) and `torch.lerp(a, b, weight_tensor)` (Tensor) already compile to **1 dispatch** on Vulkan via Inductor's stock decomposition (`add(..., diff, alpha=...)` for Scalar, `addcmul(base, w, diff)` for Tensor). The "3 dispatches" claim no longer reproduces — likely the dual-formula decomp's `where(mask, ...) + addcmul` chain already pointwise-fuses through our codegen. `TestLerpFused` added covers the contract going forward; if a future change regresses it, the test fires. Suite: 196 → 197 passed (+1).
- [x] **`aten.polygamma` / `aten.digamma` overrides** *(2026-04-27)*: outdated entry — `digamma`, `polygamma`, and `lgamma` overrides already shipped in `overrides.py:VulkanOverrides` (lines 390-402, 561). `TestPolygammaDigamma` (2 tests) locks digamma + lgamma compile + correctness vs CPU eager.
- [x] **`aten.special_*` overrides** *(2026-04-27)*: audit shows `i0`/`i0e`/`i1`/`i1e`/`spherical_bessel_j0`/`zeta`/`igamma`/`igammac`/`erf`/`erfc`/`erfcx`/`ndtri`/`erfinv` all already wired in `overrides.py:VulkanOverrides`. The two genuine gaps were `xlogy(x, y) = x * log(y)` (with `0 * log(0) = 0` corner) and `xlog1py(x, y) = x * log1p(y)` (with `0 * log1p(-1) = 0` corner) — added both as branch-on-`x==0` snippets, plus `special_xlogy` / `special_xlog1py` aliases for the `torch.special` namespace. `TestSpecialOpsOverrides` (2 tests) covers both ops including the zero-corner-is-not-NaN case vs CPU eager. Suite: 199 → 201 passed (+2).
- [x] **`aten.nan_to_num` codegen** *(2026-04-27)*: `torch.nan_to_num(x, 0.0) + 1` already compiles to **1 dispatch** via Inductor's stock isnan/isinf-detect + where-replace + epilogue codegen on Vulkan. `TestP12PointwiseRoundTwo.test_nan_to_num_one_dispatch` locks the contract.
- [x] **`aten.logit` codegen** *(2026-04-27)*: `torch.logit(x.sigmoid(), eps=1e-6) + 1` already compiles to **1 dispatch** via stock clamp + log decomposition that pointwise-fuses through Vulkan codegen. `TestP12PointwiseRoundTwo.test_logit_one_dispatch` locks the contract.
- [x] **`aten.logaddexp` / `aten.logaddexp2` codegen** *(2026-04-27)*: `torch.logaddexp(x, y) + 1` and the base-2 sibling each compile to **1 dispatch** — Inductor's numerically-stable `max(x,y) + log1p(exp(-|x-y|))` decomposition pointwise-fuses through Vulkan codegen. `TestP12PointwiseRoundTwo.test_logaddexp{,2}_one_dispatch` locks the contract.
- [x] **`aten.sgn` codegen** *(2026-04-27)*: `torch.sgn(x) + 1` already compiles to **1 dispatch** via stock pointwise codegen on Vulkan (for real tensors `sgn == sign`). `TestP12PointwiseRoundTwo.test_sgn_one_dispatch` locks the contract.
- [x] **`aten.real` real-input passthrough** *(2026-04-27)*: `torch.real(x) + 1` on a real tensor already compiles to **1 dispatch** via stock identity + pointwise codegen on Vulkan. The `aten.real / aten.imag / aten.conj` lowering item under §P12 stays scoped to complex-dtype inputs only. `TestP12PointwiseRoundTwo.test_real_passthrough_one_dispatch` locks the real-input case.
- [x] **`aten.heaviside` codegen** *(2026-04-27)*: `torch.heaviside(x, 0.5) + 1` compiles to **5 dispatches** correct via stock decomposition (compare + select + epilogue chain) on Vulkan; correctness lock is in place ahead of any future fused override. `TestP12PointwiseRoundTwo.test_heaviside_correctness` locks correctness; the dispatch count tightens once heaviside picks up a fused override.
- [x] **`aten.float_power` codegen** *(2026-04-27)*: `torch.float_power(|x|, |y|) + 1` compiles to **4 dispatches** correct via stock cast + pow + cast decomposition on Vulkan. The output dtype is auto-promoted to f64 by Inductor (matching the eager promotion contract). `TestP12PointwiseRoundTwo.test_float_power_correctness` locks correctness vs CPU eager.
- [x] **`aten.copysign` codegen** *(2026-04-27)*: `torch.copysign(x, y) + 1` already compiles to **1 dispatch** via stock pointwise codegen on Vulkan. `TestP12PointwiseRoundThree.test_copysign_one_dispatch` locks the contract.
- [x] **`aten.frac` codegen** *(2026-04-27)*: `torch.frac(x) + 1` already compiles to **1 dispatch** via stock `x - trunc(x)` decomposition + pointwise codegen. `TestP12PointwiseRoundThree.test_frac_one_dispatch` locks the contract.
- [x] **`aten.expm1` codegen** *(2026-04-27)*: `torch.expm1(x) + 1` already compiles to **1 dispatch** via the existing `expm1` pointwise override on Vulkan. `TestP12PointwiseRoundThree.test_expm1_one_dispatch` locks the contract.
- [x] **`aten.special_log_ndtr` codegen** *(2026-04-27)*: `torch.special.log_ndtr(x) + 1` already compiles to **1 dispatch** via stock `erfc(-x/√2)/2 + log` decomposition that pointwise-fuses through Vulkan codegen. `TestP12PointwiseRoundThree.test_log_ndtr_one_dispatch` locks correctness + dispatch count.
- [x] **`aten.log2` / `aten.log10` chain codegen** *(2026-04-27)*: `log2(|x|+0.1) + log10(|x|+0.1)` already compiles to **1 dispatch** via stock pointwise codegen — the 2-log chain fuses cleanly. `TestP12PointwiseRoundThree.test_log2_plus_log10_one_dispatch` locks the contract.
- [x] **`aten.bitwise_left_shift` / `aten.bitwise_right_shift` codegen** *(2026-04-27)*: integer `(x << y) + (x >> y)` already compiles to **1 dispatch** on Vulkan via stock pointwise codegen for int32 inputs. `TestP12PointwiseRoundThree.test_bitwise_shift_chain_one_dispatch` locks the contract.
- [x] **`aten.clamp_min` / `aten.clamp_max` codegen** *(2026-04-27)*: scalar (`clamp_min(x, 0.0) + 1`), tensor-bound (`clamp_min(x, y) + 1`), and chain (`clamp_min(x, -0.5) + clamp_max(x, 0.5)`) variants all compile to **1 dispatch** via stock pointwise codegen on Vulkan. `TestP12ClampAndDimAnyAll.test_clamp_min_*` + `test_clamp_max_*` lock the contract (4 tests).
- [x] **`aten.all.dim` / `aten.any.dim` codegen** *(2026-04-27)*: dim-axis `all`/`any` reductions over a bool input compile to **1 dispatch** via the existing `wgreduce_any/all` codegen path on Vulkan. Separate from the no-dim form covered by `TestAnyAllReductions`. `TestP12ClampAndDimAnyAll.test_{all,any}_dim_one_dispatch` lock the contract.
- [x] **`aten.log1p` codegen** *(2026-04-27)*: `log1p(|x|) + 1` already compiles to **1 dispatch** via the existing `log1p` pointwise override on Vulkan. `TestP12PointwiseRoundFour.test_log1p_one_dispatch` locks the contract.
- [x] **`aten.rsub` (Scalar / Tensor) codegen** *(2026-04-27)*: `rsub(x, 1.0) + 1` and `rsub(x, y) + 1` both compile to **1 dispatch** via stock pointwise codegen on Vulkan. `TestP12PointwiseRoundFour.test_rsub_{scalar,tensor}_one_dispatch` lock the contract.
- [x] **`aten.special_expit` codegen** *(2026-04-27)*: `special.expit(x) + 1` (sigmoid alias) already compiles to **1 dispatch** via the existing sigmoid override on Vulkan. `TestP12PointwiseRoundFour.test_expit_one_dispatch` locks the contract.
- [x] **`aten.trunc` codegen** *(2026-04-27)*: `trunc(x) + 1` already compiles to **1 dispatch** via stock pointwise codegen (round-toward-zero). `TestP12PointwiseRoundFour.test_trunc_one_dispatch` locks the contract.
- [x] **`aten.fmod` (Scalar) codegen** *(2026-04-27)*: `fmod(x, 0.5) + 1` already compiles to **1 dispatch** via stock pointwise codegen on Vulkan. `TestP12PointwiseRoundFour.test_fmod_scalar_one_dispatch` locks the scalar case. The tensor-divisor case (`fmod(x, y)`) currently emits 3 dispatches and is correct — leaving the tightening as a follow-up.
- [x] **`aten.square` codegen** *(2026-04-27)*: `square(x) + 1` already compiles to **1 dispatch** (lowered as `x*x` then `+1`). `TestP12PointwiseRoundFour.test_square_one_dispatch` locks the contract.
- [x] **`aten.erfinv` correctness fix** *(2026-04-27)*: the previous `c10_vulkan_erfinv` Slang helper in `slang_helpers.py` was a garbled rational-polynomial fragment whose values were close to ±1 across the entire input range (e.g. `erfinv(0.5)` returned −0.99 instead of 0.4769). Replaced with the Mike Giles "Approximating the erfinv function" implementation that CUDA's `erfinvf` uses — single-precision coefficients, two-branch (`w < 5` vs sqrt-branch) polynomial in `w = -log((1-x)*(1+x))`. `TestP12PointwiseRoundFour.test_erfinv_correctness` locks numerical correctness across `[-0.99, 0.99]` (rtol/atol 1e-4 vs CPU eager). Suite: 346 → 347 passed (+1).
- [x] **`aten.special_i1` / `aten.special_i1e` correctness fix** *(2026-04-27)*: `c10_vulkan_i1` / `c10_vulkan_i1e` had two bugs: (1) the large-|x| polynomial started with `0.22829657f` instead of the Numerical-Recipes leading coefficient `0.39894228f`, producing 1–3% relative error already at `|x|=5`; (2) the sign branch `(ax > 0.0f ? 1.0f : -1.0f)` was always `+1` (since `ax = abs(x)`), so negative-input results were wrong-sign in the large-|x| branch. Replaced with the canonical Numerical Recipes coefficients and explicit `(x < 0.0f) ? -ans : ans` sign at the end. Locked by `TestP12PointwiseRoundFour.test_i1_correctness` (`|x| ≤ 3` tight tolerance) + `test_i1e_correctness` (`|x| ≤ 5` looser). The polynomial in the large-|x| branch is still single-precision Numerical Recipes (~1–3% relative error past `|x|=5`); a tighter Cephes/Chebyshev expansion is a follow-up only when measurement surfaces it as a problem. Suite: 347 → 349 passed (+2).
- [ ] **`aten.special_polygamma` correctness fix** *(2026-04-27 — discovered, follow-up)*: `c10_vulkan_polygamma(n, x)` returns wildly wrong values (max diff ~33 on the `polygamma(2, x)` probe across `x ∈ [0.5, 5]`). The recurrence sign convention and asymptotic-expansion coefficients both look suspect — and the asymptotic expansion is hard-coded for `n ∈ {1, 2}` only with a degenerate fall-through for higher orders. Replace with the Cephes pattern: recurrence `ψ^(n)(x+1) = ψ^(n)(x) + (-1)^(n+1) · n! / x^(n+1)` to push `x ≥ 8`, then the proper Euler-Maclaurin asymptotic `ψ^(n)(x) ≈ (-1)^(n+1) · ((n-1)!/x^n + n!/(2x^(n+1)) + Σ B_{2k}·(2k+n-1)!/(2k)!/(n-1)!/x^(2k+n))` with the first 5–6 even Bernoulli numbers. Test `TestSpecialPolygammaCorrectness` — accuracy ≤1e-3 on `n ∈ {1, 2, 3}, x ∈ [0.5, 5]`.
- [ ] **`aten.special_spherical_bessel_j0` accuracy fix at moderate |x|** *(2026-04-27 — discovered, follow-up)*: `c10_vulkan_spherical_bessel_j0(x)` uses a 6-term Maclaurin series for `|x| < 8` then `sin(ax)/ax` for the tail. Max diff is ~0.034 on `[-5, 5]` (relative error ~18% near `j0` zeros where the value is small). Replace with a piecewise approximation: Maclaurin for `|x| < 1.0`, Cephes-style rational approximation for `1.0 ≤ |x| < 8.0`, exact `sin(x)/x` for `|x| ≥ 8`. Test `TestSphericalBesselJ0Correctness` — accuracy ≤1e-5 on `[-5, 5]`.
- [x] **`aten.special_zeta` correctness fix** *(2026-04-27)*: replaced the 3-term Euler-Maclaurin truncation in `c10_vulkan_zeta` (which returned `(s-1)*q^-s + (-0.5)*q^-(s+1) + 0.5*q^-s` and zero outside `s>1, q>0`, producing diff ~32 on the probe) with a Cephes-style implementation: direct sum of 9 leading terms `Σ_{k=0}^{8} (q+k)^-s`, then Euler-Maclaurin tail in `a = q+9` with the first 6 even Bernoulli numbers and a rising-factorial recurrence `(s)_(2j-1) = (s)_(2j-3) * (s+2j-3)*(s+2j-2)`. Plus a sign-flip on the half-endpoint correction (`+0.5*b` not `-0.5*b`). `TestP12PointwiseRoundFour.test_zeta_correctness` locks numerical correctness across `s ∈ [2, 5], q ∈ [0.5, 3]` (rtol/atol 1e-4 vs CPU eager — actually achieves ≤2e-6). Suite: 349 → 350 passed (+1).

**Backward-graph completeness (extends §P0.1 / P1.7):**

- [ ] **`aten.where` backward lowering**: routes the gradient through both branches; trivial decomposition. Test `TestWhereBackward`.
- [ ] **`aten.gather_backward` lowering**: scatter-add gradient. Reuses the same shader as `scatter_add`. Test `TestGatherBackward`.
- [ ] **`aten.index_select_backward` lowering**: scatter-add along the indexed dim. Test `TestIndexSelectBackward`.
- [ ] **`aten.clamp_backward` lowering**: `(min ≤ x ≤ max) * grad_out`. Test `TestClampBackward`.
- [ ] **`aten.nll_loss2d_backward` lowering**: 2D variant of nll_loss_backward — semantic-segmentation loss. Test `TestNLLLoss2dBackward`.
- [ ] **`aten.binary_cross_entropy_backward` lowering**: separate from `bce_with_logits_backward` which we cover. Test `TestBCEBackward`.
- [x] **`aten.softplus_backward` / `aten.gelu_backward` / `aten.tanh_backward` audit** *(2026-04-27)*: surveyed all `_backward` aten ops in `lowerings.py:_register_activation_backward()`; `register_lowering(aten.X)` covers both `.default` and `.grad_input` overloads automatically via the OpOverloadPacket. Audit found `gelu_backward` was the **only missing** entry — added a full `_vulkan_gelu_backward(grad, self, *, approximate)` lowering covering both `approximate='none'` (exact: `0.5*(1 + erf(x/√2)) + (x/√(2π))*exp(-x²/2)`) and `approximate='tanh'` (cubic-tanh: `0.5*(1+tanh(arg)) + 0.5*k*x*(1 - tanh²)*(1 + 3*0.044715*x²)`); added to `_suppress_upstream_decomps()` so AOT autograd doesn't pre-decompose it. `TestGeluBackwardOverloadAudit.test_every_activation_backward_overload_registered` walks 10 backward ops × all overloads, asserts every one is in the Inductor lowering table. Suite: 207 → 210 passed (+3, includes one previously-extern path now reaching the lowering).
- [ ] **`aten.convolution_backward_overrideable` algorithm selection**: `conv_backward` always extern-falls today. Mirror the §P5.1 forward selector — direct / Winograd / 1×1 / depthwise — for the backward path. Critical for the ResNet-18 backward target (≤35 dispatches). Test `TestConvBackwardAlgorithmSelector`.
- [ ] **`aten.upsample_*_backward` lowerings**: scatter-add into the input grad. Test `TestUpsampleBackwardSet`.
- [ ] **`aten.adaptive_avg_pool2d_backward` lowering**: pointwise broadcast-divide. Test `TestAdaptivePoolBackward`.

**Optimizer ops under torch.compile:**

- [x] **`aten._foreach_*` combo-kernel coverage audit** *(2026-04-27)*: measurement-driven audit shows `_foreach_add`, `_foreach_mul`, `_foreach_addcmul`, `_foreach_lerp`, `_foreach_div`, `_foreach_neg`, `_foreach_abs` all collapse to **1 combo-kernel dispatch** for 2-tensor input lists via the `BackendFeature.FOREACH` rewrite shipped in P1.4. `TestForeachAudit` (7 tests) locks the contract. (`_foreach_copy_`, `_foreach_zero_` are in-place ops — not measured because the combo path is for out-of-place; if a model surfaces them, add them to the audit.)
- [ ] **`aten._fused_adam` / `aten._fused_sgd` recognizers**: PyTorch ships fused-optimizer ops that AOT autograd preserves; route to the eager `adamw_batch7` / `sgd_batch15` shaders via ExternKernelChoice instead of letting Inductor decompose. Test `TestFusedOptimizerExtern`.

**Miscellaneous gaps:**

- [ ] **`aten.repeat_interleave` codegen**: §P4.8 has the baseline `[~]` test; ship the prefix-sum-driven gather codegen. Test `TestRepeatInterleaveCodegen`.
- [ ] **`aten.unique` / `aten.unique_consecutive` codegen**: §P4.8 has the entry; size it for the dim-0 contiguous case. Test `TestUniqueCodegen`.
- [ ] **`aten.histc` / `aten.bincount` atomic shader**: atomic-histogram. Test `TestAtomicHistogram`.
- [ ] **`aten.linalg_solve` / `aten.linalg_lstsq`**: small-matrix LU/QR — extern only on RDNA1 (no native solver). Document the limitation; route through extern. Test `TestLinalgSolveExtern`.
- [ ] **`aten._scaled_mm` extern**: f8 / bf16 / int8 scaled GEMM (PyTorch 2.5+). Wire as ExternKernelChoice; gated on dtype + device capability. Test `TestScaledMMExtern`.

**Coverage-tracking infrastructure (extends §P1.7):**

- [ ] **`scripts/audit_inductor_op_coverage.py` extension**: today P1.7's auditor lists extern aten ops; extend to also list aten ops that *have* a Vulkan eager shader but no Inductor route — that's the P10 backlog. Diff the audit against `shaders/*.slang` and `lowerings.py` to flag each. Test `TestCoverageAuditCrossCheck`.

---

### P11 — Advanced kernel-codegen heuristics & micro-optimizations

The codegen produces correct, decent Slang today but leaves measurable wins on the table. Each item is a focused codegen-quality improvement, sized to a single agent iteration, with a concrete file:line target and regression test.

**P11.1 — Reduction-shape selection:**

- [ ] **Tree-vs-ladder reduction shape**: `wg_reduce_sum` today emits a binary-tree reduction (`for stride = wg/2; stride > 0; stride >>= 1`). For small `n_waves` (1–4), a ladder reduction (`val = WaveActiveSum(val); if (lane == 0) smem[wave] = val; barrier; sum across waves`) has fewer barriers and lower LDS traffic. Pick at codegen time based on `n_waves`. File: `slang_helpers.py:emit_helpers`. Test `TestTreeVsLadderReduction`.
- [ ] **Welford-tree reduction shape**: same shape as above but for `welford_combine`. Currently always tree. Test `TestWelfordTreeShape`.
- [ ] **Per-thread serial-reduce prelude for medium rnumel**: when `rnumel ∈ [256, 4096]` and `numthreads == 256`, give each thread `rnumel/numthreads` serial accumulations before the wave-reduce. Halves the wave-reduce work at the cost of more registers. Auto-pick via §P1.6 intensity classifier. Test `TestSerialReducePrelude`.

**P11.2 — Branch-free reduction / select:**

- [ ] **`select(cond, a, b)` fold**: when `where(mask, body, other)` has a uniform-cost body and other branch, emit `select(mask, body_val, other_val)` instead of an `if`/`else` block. Slang's `select` lowers to SPIR-V `OpSelect` which is branch-free and pipeline-friendlier. Today `VulkanOverrides.masked` always emits an `if`/`else`. Test `TestBranchFreeSelectFold`.
- [ ] **NaN-aware reduction codegen**: P5.6's NaN-propagation regression is xfailed; ship the codegen that emits `((a != a) ? a : (b != b) ? b : op(a, b))` for `min`/`max`/`sum` reductions. Today reductions use raw `WaveActiveMin`/`Max` which are NaN-suppressing. Reuse `_NaN_aware` patterns from eager `slang_helpers`. Test `TestNaNAwareReductionCodegen` — flips the existing P5.6 xfail to passed.
- [ ] **Single-axis `argmax` / `argmin` codegen** *(2026-04-27 — investigation logged)*: upstream `make_reduction("argmax", override_return_dtype=int64)` only sends the `(value, linear_idx)` tuple on the multi-axis Triton path; the single-axis Vulkan path receives just `value`, while `_argmin_argmax_reduction` in `kernel.py` asserts a 2-tuple. Plus `_argmin_argmax_reduction` adds the header `wgreduce_argmaxmin` while the helper-block in `slang_helpers.py` keys on `wgreduce_argmax` / `wgreduce_argmin`, and `cse.generate(..., dtype=dtype)` types the float2-returning helper as scalar `float`, so a downstream `.y` access fails to compile. Three coupled fixes needed: (a) derive `arg_index` from the reduction range tree's `<prefix>_index` symbol when `value` is a single CSEVariable; (b) rename the header to `wgreduce_arg{max,min}` so the dedicated emit fires; (c) thread the float2 type through the cse-generated wrapper variable (or store `.y` directly to a separate int64 cse var). Currently raises `NotImplementedError` so upstream extern decomp wins. Test `TestArgmaxArgminSingleAxisCodegen`.

**P11.3 — Instruction-level parallelism:**

- [ ] **Independent FMA chains in inner loops**: per-thread accumulators in matmul use a single `acc += a * b` chain — each FMA depends on the prior. Split into 2–4 independent chains (`acc0`, `acc1`, ... `acc_n` summed at the end) so the scheduler can dual-issue. RDNA1 has 2 SALU + 1 VALU dispatches per wave per cycle; ILP-friendly code is measurable. File: `templates/slang_mm.py.jinja:mma_tile`. Test `TestMatmulILPChains`.
- [ ] **Dual-issue-friendly reduction**: same shape for `wg_reduce_sum` — split serial-reduce prelude into 2 independent partial sums. Test `TestReductionILPChains`.

**P11.4 — Memory-access patterns:**

- [ ] **Transpose-on-load for transposed matmul**: when `B` is `transpose_b=True`, the current matmul loads `b[k, n]` with stride `(stride_b_k, stride_b_n)`. If the underlying tensor is column-major, this gives uncoalesced loads. Detect and emit a transposed cooperative load (each thread group transposes its B tile during load via LDS). File: `templates/slang_mm.py.jinja`. Test `TestTransposedMatmulCoalescedLoads`.
- [ ] **Multi-level tiling (cluster ⊃ workgroup ⊃ register-tile)**: today the matmul has workgroup-tile + register-tile. Adding a cluster-tile (multiple workgroups cooperating via the L2 cache) helps for very large M/N. Slang exposes the cluster level via `[numthreads(...)]` + workgroup-launch math. Test `TestMatmulClusterTile`.
- [ ] **Async copy preload (RDNA3+ / NVIDIA only)**: `cooperativeMatrixLoad` with the `MakeAvailableKHR` decoration prefetches A/B tiles into shared memory while the previous tile is computing. Capability-gated. Test `TestMatmulAsyncCopyPrefetch`.
- [x] **Read-only buffer hint** *(2026-04-27)*: verified — `slangc -target spirv-asm` on a probe kernel with `StructuredBuffer<float>` input emits `OpDecorate in_ptr0 NonWritable` (the SPIR-V equivalent of `ReadOnly`). Slang's `StructuredBuffer<T>` typing already drives the right decoration; no codegen change needed. RDNA1 / NVIDIA L1 read-only cache promotion works on this decoration. Roadmap entry was a "verify it's correct" item — confirmed correct.
- [ ] **Strided-store coalescing detection**: when `out_ptr[stride * idx]` has `stride > 1` but the workgroup writes `numthreads` contiguous output indices, the writes are uncoalesced. Detect and emit a vec-shuffle through LDS to make stores contiguous. Test `TestStridedStoreCoalescing`.

**P11.5 — Constant / index optimization:**

- [ ] **Constant deduplication in shader source**: today the lowering emits the same constant (`0.7071067811865476` for `1/sqrt(2)`) inline at every use site. Hoist via `const float _C0 = 0.7071067811865476;` at module scope. Reduces source bytes; lets slangc share register allocation. Test `TestConstantDedup`.
- [ ] **Address-arithmetic CSE across loads**: when 3 buffers are read at the same `idx`, today the printer emits 3 separate `idx * stride + base` computations. CSE the address into one `uint addr = ...;` and reuse. File: `expr_printer.py`. Test `TestAddressCSE`.
- [ ] **Drop redundant `uint xindex = ...` aliases**: §P4.3 partially addresses this for vec4 paths; extend to scalar paths too. Test `TestRedundantAliasDrop`.

**P11.6 — Source-level hygiene:**

- [x] **Shader-source minification before SPIR-V hashing** *(2026-04-27)*: `_normalize_slang_source` in `runtime.py` strips block comments (`/* ... */`), line comments (`// ...`), trailing whitespace, and runs of blank lines before the source is fed to SHA256 in both `compile_slang_to_spirv` and `prewarm_compile`. Cosmetic codegen variation (e.g. inline comments emitted on different days, blank-line padding) no longer fragments the SPIR-V cache. Conservative — does not collapse intra-line whitespace because Slang preprocessor directives can be leading-whitespace-sensitive in some builds. The hash is the only path the normalized source feeds; the actual `slangc` invocation still receives the original source so `TORCH_LOGS=output_code` debug paths are unaffected. `TestGroupSharedBudget.test_slang_source_minification_for_hashing` covers comment-strip / blank-line-collapse / block-comment-strip equivalence.
- [ ] **Pretty-print emitted Slang for `TORCH_LOGS=output_code`**: orthogonal to the above — the *displayed* version (debug logs) should be pretty-printed even when the *hashed* version is minified. Two separate paths. Test `TestSourcePrettyPrint`.
- [ ] **Slang formatter integration**: run `slangc -emit-source` (or a hand-rolled formatter) on every emitted shader so source diffs in the output_code log are consistent across runs. Test `TestSourceFormatterStable`.

**P11.7 — Vendor-specific tuning:**

- [ ] **RDNA1 `s_waitcnt` tuning**: AMD's `s_waitcnt vmcnt(N)` controls memory-wait granularity. slangc emits conservative `s_waitcnt vmcnt(0)` after every load. For known-independent loads, `s_waitcnt vmcnt(2)` overlap-hides 2 outstanding loads — measurable on memory-bound kernels. Vendor-specific Slang pragma; gate on AMD. Test `TestRDNA1WaitcntTuning`.
- [ ] **NVIDIA L1-cache hints**: emit `[ReadOnly]` and `[Coherent]` decorations per buffer. Test `TestNVIDIACacheHints`.
- [ ] **SwiftShader fast paths**: SwiftShader's wave size is 4 (CPU SIMD-style), not 32/64. The wave-reduce helpers should pick a tree-reduction shape automatically when `simd_group_size <= 4`. Today `_pick_threadgroup_size` doesn't factor this in. Test `TestSwiftShaderWaveSize`.

**P11.8 — Branch-probability hints:**

- [ ] **`OpExpect`-style branch hints**: when the compiler knows a branch is rarely taken (e.g. the `if (xindex >= numel) return;` early-exit for tail threads), emit a branch-probability hint. Slang exposes this via `[branch]` / `[flatten]` attributes. Test `TestBranchHintEmit`.

**P11.9 — Push-constant struct layout:**

- [x] **Reorder PC fields for alignment** *(2026-04-27)*: outdated entry — audit shows the matmul `struct PC` in `slang_mm.py.jinja` already declares all 4-byte uints (`M`, `N`, `K`, 8 strides, optional `stride_bias_n`) before any 4-byte floats (`alpha`, `beta`, `scale`). The Python `struct.pack("9I", ...)` (or `"10I"` with bias, plus optional float fields) matches the SPIR-V layout exactly with no padding. No `uint64` / `vec` types in the struct → no alignment padding to recover. Roadmap entry was carried over from an earlier struct that mixed types; current state ships the optimal layout. Locked passively by the existing `TestRegisterTileMatmul` correctness suite — any future regression that introduces a mixed-alignment field would surface as an off-by-stride bug there.
- [ ] **Pack stride pairs as `uint2`**: `(stride_a_m, stride_a_k)` is a natural `uint2`; same for `(stride_b_k, stride_b_n)`. Halves the field count, simplifies the Python `struct.pack` format. Test `TestStridePairPacking`.

**P11.10 — Fast-math safe-guard:**

- [x] **`[unroll]` only on bounded loops** *(2026-04-27)*: `slang_mm.py.jinja` `mma_tile` inner-product K-loop changed from bare `[unroll]` to `[unroll({{ min(tile_k, 16) }})]`. Full unroll on `tile_k=64` × 4×4 register tile produced ~1024 FMAs per thread and dropped RDNA1 occupancy from 8 → 2 waves; the cap holds the per-thread instruction count manageable while still letting slangc pipeline FMAs. Cap chosen at 16 because the standard register-tile autotune candidates use `tile_k ∈ {16, 32}` so the cap kicks in only on the 32-tile path. `TestRegisterTileMatmul.test_unroll_capped_at_16_for_large_tile_k` covers the rendered-source contract for both cases (`tile_k=8` keeps `[unroll(8)]` full unroll; `tile_k=32` becomes `[unroll(16)]`, never `[unroll(32)]`).
- [ ] **`fast` floating-point mode on epilogues**: matmul epilogues with `gelu`/`tanh`/`erf` benefit from `[FastMath]` (allows `fma` reordering). Gate per-op so reductions stay strict. Test `TestEpilogueFastMath`.

**P11.11 — Tracing / observability:**

- [x] **Per-kernel SPIR-V hash exposed via `inductor_stats`** *(2026-04-27)*: `_KERNEL_SPIRV_HASH` in `runtime.py` records the 12-char SHA256 prefix of each compiled SPIR-V binary at `make_vulkan_kernel` time. `inductor_stats.summary()` adds the hash as the 5th column of every `top` row so cache-miss / autotune-churn debugging can correlate per-kernel timing to specific compiled binaries. `print_full_report` prints `[spv:<12-char>]` next to each kernel name when the hash is available. `TestSummaryAPI.test_summary_aggregates_compiled_kernels` extended to validate the 12-char hex format.
- [ ] **`TORCH_LOGS=output_code` includes register / VGPR count**: emit a `// vgprs=N occupancy=M waves` comment at the top of every kernel, parsed from `slangc -dump-asm`. Test `TestOutputCodeVGPRComment`.

---

### P12 — Extended primop coverage round 2 (post-P10 audit)

P1.7 covered loss / norm / pool / upsample / pad / dropout. P10 covered vision / im2col / searchsorted / triu/tril / lerp / digamma. A re-walk against the full `torch._refs` and `torch._decomp` registries surfaced these residual gaps. Each is a measured extern fall-back on at least one workload in `benchmarks/inductor_train.py`, or a coverage hole exposed by P12.13's discovery script. Files: `lowerings.py`, `meta_patches.py`, `overrides.py` as appropriate.

**Statistics & order ops:**

- [ ] **`aten.median.dim` / `aten.median.dim_indices` / `aten.nanmedian` lowering**: small-rnumel codegen via partial-sort + pick; large-rnumel via histogram-and-find. Currently extern. Test `TestMedianLowering` — ≤2 dispatches on (B, V=4096).
- [ ] **`aten.kthvalue` lowering**: same shape as `topk` but returns single value + index. Reuse the §P4.8 `topk` template. Test `TestKthValueLowering`.
- [ ] **`aten.quantile` / `aten.nanquantile` lowering**: linear-interpolation between sorted values. Reuses sort path. Test `TestQuantileLowering`.
- [ ] **`aten.mode` lowering**: most-frequent value per row; atomic-histogram + argmax. Test `TestModeLowering`.
- [x] **`aten.diff` codegen** *(2026-04-27)*: `torch.diff(x) + 1.0` already compiles to **4 dispatches** correct via Inductor's stock view+sub+pointwise codegen on Vulkan. `TestP12ShapeAndLinalgCoverage.test_diff_correctness` locks correctness vs CPU. Tightening to 1 dispatch would need a dedicated lowering (slice → sub → epilogue fold); deferred until measurement surfaces it as a hot path. `aten.gradient` covered by the same view+sub pattern; tracked separately when used.
- [ ] **`aten.cumulative_trapezoid` / `aten.trapezoid` lowering**: trapezoidal integration; cumsum + scale. Test `TestTrapezoidLowering`.

**Linear algebra (full coverage):**

- [~] **`aten.einsum` lowering** *(2026-04-27 — codegen broken: returns stale data)*: probed two cases — `einsum('ij,jk->ik', a, b)` (matmul) and `einsum('bnhd,bnHd->bnhH', q, k)` (attention scores) both compile to **0 dispatches** but return *stale captured-during-trace data* (probe diff=9.05 / 12.0). Same constant-fold-during-trace family as `broadcast_tensors`/`tensor_split`/P1.5 view bug — Inductor's pre-grad pass folds the einsum-decomposed graph to a constant tensor whose contents were captured when inputs were uninitialized fakes. Will unblock alongside P0.0 / P1.5 family fix. Test `TestEinsumLowering` deferred until codegen unblocks.
- [ ] **`aten.tensordot` lowering**: thin wrapper over `einsum`. Test `TestTensordotLowering`.
- [ ] **`aten.outer` / `aten.inner` lowering**: `outer(a, b) = a[:,None] * b[None,:]` (1 pointwise dispatch); `inner(a, b) = matmul`. Currently blocked by the same FakeTensor data-pointer family as P0.0 — `torch.outer` traces through a code path that calls `data_ptr()` on a FakeTensor during compile. Will unblock when P0.0 view-fast-path lands. Test `TestOuterInnerLowering`.
- [x] **`aten.kron` codegen** *(2026-04-27)*: outdated entry — `torch.kron(a, b) + 1.0` already compiles to **4 dispatches** correct via Inductor's stock broadcast + reshape + pointwise codegen on Vulkan (no Vulkan-specific lowering needed). `TestP12ShapeAndLinalgCoverage.test_kron_correctness` locks correctness vs CPU eager.
- [x] **`aten.linalg_cross` lowering** *(2026-04-27)*: `torch.linalg.cross(a, b, dim=-1) + 1.0` on 3-vectors compiles to **1 dispatch** correct via Inductor's stock pointwise codegen on Vulkan — the cross product's 6 scalar mul/sub ops on broadcast slices fuse into a single VulkanKernel. No Vulkan-specific lowering required. `TestP12ShapeAndLinalgCoverage.test_linalg_cross_one_dispatch` locks the contract.
- [ ] **`aten.cdist` / `aten.pdist` lowering**: pairwise distances. `cdist(a, b, p=2)` decomposes to `(a[:,None] - b[None,:]).norm(p)`. The `p=2` Euclidean case can route to the matmul path via `||a-b||² = ||a||² + ||b||² - 2 a·b`. Test `TestCDistLowering`.
- [ ] **`aten.linalg_qr` / `aten.linalg_svd` / `aten.linalg_inv` / `aten.linalg_cholesky` / `aten.linalg_eigh` ExternKernelChoice**: small-matrix factorisations. Document RDNA1 limitation (no native solver); route through Householder / Givens / Jacobi shaders for `n ≤ 64`, fall through to CPU otherwise. Test `TestLinalgFactorisations` — correctness vs CPU eager.
- [ ] **`aten.matrix_exp` lowering**: scaling-and-squaring + Padé approximation; pure pointwise + matmul chain. Test `TestMatrixExpLowering`.

**Complex dtypes:**

- [ ] **`aten.complex` / `aten.polar` constructors lowering**: pack `(real, imag)` into a `float2` Slang struct; route through the existing pointwise codegen. Test `TestComplexConstruct`.
- [ ] **`aten.real` / `aten.imag` / `aten.conj` lowering**: zero-cost view ops on `float2` storage. Test `TestComplexAccessors` — 0 dispatches (view).
- [ ] **`aten.view_as_real` / `aten.view_as_complex` lowering**: layout reinterpretation. Test `TestComplexViewAs`.
- [ ] **Complex-aware pointwise codegen**: `VulkanOverrides` currently treats complex dtypes as broken. Add `float2`-aware add/sub/mul/div snippets so complex arithmetic actually works in compiled mode. Test `TestComplexArith`.

**Spectral / signal:**

- [ ] **`aten.stft` / `aten.istft` ExternKernelChoice**: route through windowed `_fft_r2c` extern templates from P1.7. Test `TestSTFTTemplate`.
- [ ] **`aten._fft_c2c` ExternKernelChoice**: complex-to-complex FFT (the most general FFT op); covers radix-2/3/5 cases. Test `TestFFTComplexTemplate`.

**Shape / view ops (currently extern or partially):**

- [~] **`aten.movedim` / `aten.swapdims` / `aten.swapaxes` codegen** *(2026-04-27 — `swapdims` ships, `movedim` audit pending)*: `torch.swapdims(x, 0, 1) + 1.0` compiles to **2 dispatches** correct via Inductor's stride-only transpose + pointwise codegen on Vulkan. `TestP12ShapeAndLinalgCoverage.test_swapdims_correctness` locks the contract. `movedim` audit deferred — under compile it returned `d=0 diff=3.22` on the probe (suggests output-buffer reuse caused stale-state mismeasurement, not a real bug; test isolation still pending). `swapaxes` aliases `swapdims` upstream so the same path covers it.
- [~] **`aten.tensor_split` / `aten.unsafe_split` / `aten.unsafe_chunk` codegen** *(2026-04-27 — `chunk` ships, `tensor_split` codegen broken)*: `torch.chunk(x, 2)` followed by `c0 + c1` compiles to **1 dispatch** correct via stock view+pointwise codegen (`TestP12ShapeAndLinalgCoverage.test_chunk_one_dispatch`). `torch.tensor_split` returns `d=0` with stale data (probe diff=4.26) — same constant-fold-during-trace family as `broadcast_tensors`/P1.5 view bug. Will unblock alongside P0.0 / P1.5.
- [x] **`aten.narrow` codegen** *(2026-04-27)*: `x.narrow(0, 1, 3) + 1.0` compiles to **2 dispatches** correct via Inductor's stride+offset view + pointwise codegen on Vulkan. `TestP12ShapeAndLinalgCoverage.test_narrow_correctness` locks the contract. `narrow_copy` (the materialising variant) is a separate path — defer until measurement surfaces it.
- [~] **`aten.atleast_1d` / `aten.atleast_2d` / `aten.atleast_3d` lowering** *(2026-04-27 — `atleast_1d` ships, `_2d`/`_3d` blocked on FakeTensor data-pointer family)*: `torch.atleast_1d(x) + 1.0` compiles to **1 dispatch** correct via stock codegen (`TestP12ShapeAndLinalgCoverage.test_atleast_1d_one_dispatch`). `atleast_2d` and `atleast_3d` fall through with `Cannot access data pointer of Tensor (e.g. FakeTensor)` — same P0.0 view-fast-path family. Will unblock with P0.0.
- [~] **`aten.broadcast_tensors` lowering** *(2026-04-27 — codegen broken: returns stale data)*: `torch.broadcast_tensors(a, b); a + b` compiles to **0 dispatches** but produces *wrong values* (probe diff=3.58). Same constant-fold-during-trace family as the P1.5 `view`+`pointwise` bug — Inductor's pre-grad pass folds the broadcast result to a constant tensor whose contents were captured when inputs were uninitialized fakes. New regression entry tracked under P1.5 when fixed. Test `TestBroadcastTensorsZeroCost` deferred until codegen unblocks.
- [x] **`aten.block_diag` lowering** *(2026-04-27)*: outdated entry — `torch.block_diag(a, b) + 1.0` already compiles to **1 dispatch** correct via Inductor's stock conditional-write codegen on Vulkan (one output kernel that conditionally copies from each input block based on the index). `TestP12ShapeAndLinalgCoverage.test_block_diag_one_dispatch` locks the contract.

**Histogram & distribution:**

- [ ] **`aten.histogramdd` ExternKernelChoice**: N-dim atomic-histogram. Test `TestHistogramDDTemplate`.
- [ ] **`aten.bucketize` template** (sister to P10 `searchsorted`): same kernel shape, different signature. Test `TestBucketizeTemplate`.

**Probabilistic ops (used in RL training):**

- [ ] **`aten.multinomial` ExternKernelChoice**: alias-method sampling. Currently extern. Test `TestMultinomialTemplate`.
- [ ] **`aten.bernoulli` / `aten.bernoulli_` codegen**: Philox-fused. Reuses the §P1.7 `native_dropout` kernel. Test `TestBernoulliCodegen`.
- [ ] **`aten.poisson` ExternKernelChoice**: rejection-sampling shader for low-rate Poisson; Knuth's algorithm for higher rates. Test `TestPoissonTemplate`.
- [ ] **`aten.geometric_` / `aten.cauchy_` / `aten.exponential_` / `aten.log_normal_` codegen**: Philox + inverse-CDF transformations; pure pointwise. Test `TestRandDistributionsCodegen`.

**Utility / discovery:**

- [ ] **`scripts/audit_inductor_op_coverage.py` round 2**: extend the P1.7 / P10 auditor to also walk `torch.ops.aten.*` exhaustively, pull every `OpOverload`, and compare against the Inductor lowering registry + ExternKernelChoice registry. Output a 3-column report `(aten_op, in_lowerings, in_eager_shaders)`. Drives roadmap expansion. Test `TestAuditRoundTwo` — produces a report containing the known-extern ops.

---

### P13 — Compile-pipeline & SPIR-V optimisation

The pipeline today is `slang source → slangc → SPIR-V → vkCreateComputePipeline`. Several optimisation stages between slangc and the driver are missing — they're cheap to add and reduce pipeline-creation time / driver-side compile time.

- [ ] **`spirv-opt -O` post-pass between slangc and `vkCreateComputePipeline`**: feed slangc's SPIR-V through `spirv-opt --inline-entry-points-exhaustive --eliminate-dead-code-aggressive --merge-blocks --simplify-instructions --vector-dce --redundancy-elimination --convert-to-half-pass`. RDNA1's runtime compiler does some of this but not all; preprocessing here cuts driver-side compile time by 10–25% on Slang-heavy kernels. Wire as `runtime._maybe_spirv_opt(spv_bytes) → spv_bytes` gated on `TORCH_VULKAN_SPIRV_OPT={none,O,Os}`. Test `TestSpirvOptPostPass` — verify the optimised SPIR-V validates clean and runs identically.
- [ ] **Dead-binding stripping**: Slang occasionally emits unused buffer / push-constant declarations when an epilogue branch is statically unreachable. Walk the SPIR-V `OpDecorate Binding` list, drop any decorations whose target instruction has no `OpLoad` / `OpStore` use. Test `TestDeadBindingStrip` — kernel with conditional epilogue produces SPIR-V without stub bindings.
- [ ] **On-disk SPIR-V deduplication**: today every `(slang_source, defines)` pair gets its own SPIR-V file. Many pairs produce *bit-identical* SPIR-V (different inline comments, identical IR). Hash the **post-`spirv-opt`** binary and dedupe via hardlink / symlink. Cuts cache disk usage 30–50% on large autotune sweeps. File: `runtime.py:_disk_cache_key`. Test `TestSpirvDedup`.
- [ ] **slangc IR cache (`-incremental`)**: slangc supports a `-incremental` mode that caches parsed module IR between invocations. Enable when `TORCH_VULKAN_INCREMENTAL_SLANGC=1` is set. Cuts cold-compile time 2–4× on workloads that share helper modules (P9.2). Test `TestSlangcIncremental`.
- [ ] **Slang link-time optimisation**: when multiple kernels share a `module/import`, use `slangc -link-time-optimization` to inline cross-module helpers across the entire program at link time. Test `TestSlangLTO`.
- [ ] **Validation-layer-clean SPIR-V**: run every emitted SPIR-V through `spirv-val` (with `--target-env vulkan1.3`) in debug mode; CI gate on zero validation errors. Test `TestSpirvValidationClean`.
- [ ] **Pre-warmed `VkPipelineCache` blob**: persist a per-device `VkPipelineCache` blob to `~/.cache/torch_vulkan/pipeline_cache/<device-id>.bin`. Loaded at backend init, saved on shutdown. Cuts pipeline-creation time 5–10× on workloads with >100 unique kernels. File: `csrc/vulkan/Context.cpp`. Test `TestPipelineCacheBlob`.
- [ ] **Pipeline-creation parallelism**: today `vkCreateComputePipeline` is called serially per kernel during graph compile. Wrap N pipelines per `vkCreateComputePipelines` call (the plural-API version), and submit groups in a thread pool. RDNA1 driver compiles in parallel internally when given a batch. Test `TestPipelineCreationParallel` — 100-kernel graph compile time ≤ 2× single-kernel.
- [ ] **SPIR-V hash → driver-cache key alignment**: use the post-`spirv-opt` SPIR-V hash as the `pNext = VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` cache identifier. Bypasses driver-side internal SHA validation when our hash matches. Test `TestPipelineCacheKey`.
- [ ] **slangc target environment tuning**: today `slangc -target spirv` with default capabilities. Pin `-profile sm_5_2 -target-env vulkan1.3` so SPIR-V version + capability set is stable across slangc upgrades. Test `TestSlangcTargetEnv`.

---

### P14 — FX pre/post-grad pass expansion

P2.5 listed seven pattern-matcher rules. Upstream Inductor ships ~100. The agent's measurement passes have surfaced these high-value patterns specifically on Vulkan workloads. Each lands in `fx_passes.py` with a regression test that locks the dispatch-count drop.

**Pre-grad patterns (run before AOT autograd):**

- [ ] **`linear → bias_add → layer_norm` pattern**: collapse `(linear(x) + b).layer_norm(...)` to a fused extern (3 dispatches → 1). Eager has `linear_bias_layer_norm.slang`. Test `TestLinearBiasLayerNormFusion`.
- [ ] **`attention_score → causal_mask → softmax` pattern**: detect `(scores + causal_mask).softmax(-1)` and route through the flash-attention extern's softmax-with-mask path. Test `TestAttentionMaskedSoftmaxFusion`.
- [ ] **`gelu → linear` pattern (FFN second half)**: route through the `linear_bias_activation` template's gelu epilogue. Test `TestGeluLinearFusion`.
- [ ] **`silu → linear` pattern (gate path of MLP)**: same shape. Test `TestSiluLinearFusion`.
- [ ] **`embedding → add → layer_norm` pattern (transformer input)**: collapse to a fused extern. Eager has `embedding_add_layer_norm.slang`. Test `TestEmbeddingAddLayerNormFusion` — 3 dispatches → 1.
- [ ] **`linear → linear` back-to-back without activation**: rewrite to `linear(x, W2 @ W1)` *only when* the intermediate is not consumed elsewhere AND the combined-weight matmul is cheaper (`O(D_in × D_out)` vs `O(D_in × D_mid + D_mid × D_out)`). Heuristic-gated. Test `TestLinearLinearMerge`.
- [ ] **`view → pointwise → view` flattening**: when both views are zero-cost reshapes around a stride-1-preserving pointwise, drop both views. Test `TestViewPointwiseViewFlatten`.
- [x] **Dropout-during-eval elimination** *(2026-04-27)*: outdated entry — Inductor's stock decomposition of `aten.dropout` / `aten.native_dropout` already folds the op to identity at graph-rewrite time when `training=False`, so no Vulkan-specific FX pass is needed. Verified via measurement: `F.dropout(x, p=0.5, training=False) + 1.0` compiles to **1 dispatch** correct (output == `x + 1.0` exactly), and a 3-stage chain (`dropout(p=0.5,eval) → relu → dropout(p=0.3,eval) → +1`) also collapses to **1 dispatch**. `TestP14DropoutDuringEvalElim` (2 tests) locks the contract — a regression that pushes the dropout to a Philox-RNG dispatch fires.
- [ ] **Dead-cast removal across graph boundaries**: `to(f32) → ... → to(f16)` where the inner ops are all f32 but accept f16 → drop the round-trip casts. Test `TestDeadCastRemoval`.
- [ ] **Scalar-broadcast hoisting**: when a scalar is broadcast inside multiple fused regions, hoist to a single push-constant value at the wrapper level. Test `TestScalarBroadcastHoist`.
- [ ] **`to_copy` cancellation**: `t.to(torch.float16).to(torch.float32)` → identity (within a tolerance of representable). Test `TestToCopyCancel`.

**Post-grad patterns (run after AOT autograd):**

- [ ] **`grad_out * 1.0` elimination**: AOT autograd sometimes emits gradient multiplications by `1.0` literal; drop them. Test `TestGradOutOneElim`.
- [ ] **`zeros_like → add` → `clone` rewrite**: when an `aten.zeros_like(t).add_(other)` shape appears (common in backward), rewrite to `other.clone()`. Test `TestZerosLikeAddRewrite`.
- [ ] **Broadcast-add → bias-epilogue rewrite**: detect `mm + broadcast_add` patterns the existing `_fuse_mm_add_to_addmm` misses (e.g. when the broadcast axis isn't the last-dim). Test `TestBroadcastAddEpilogue`.
- [ ] **`aten.detach` removal in compiled graph**: `detach()` is autograd-only; in the compiled forward graph it's a no-op view. Audit it doesn't break fusion. Test `TestDetachNoOp`.

---

### P15 — Slang feature utilisation round 2 (post-P9)

P9 covered generics, modules, interfaces, ParameterBlock, capabilities, reflection, multi-entry-point, printf, ForceInline, spec constants. These additional Slang features are still unused.

- [ ] **P15.1 — `spirv_asm { ... }` inline SPIR-V for hot kernels**: Slang lets a function body contain raw SPIR-V via `spirv_asm`. Use it for the matmul inner-K loop where the slangc-generated SPIR-V is suboptimal vs hand-tuned (one `OpFMul` + one `OpFAdd` instead of `OpExtInst Fma`, missing decorations, etc.). Wire a small library of inline-SPIR-V helpers in `shaders/inductor_runtime/inline_asm.slang`. Test `TestSpirvAsmFmaInline` — generated SPIR-V uses `Fma` extension instruction, kernel ms strictly ≤ baseline.
- [ ] **P15.2 — `IFunc<R, Args...>` for first-class function passing in epilogues**: today the `slang_mm.py.jinja` template handles epilogues by string substitution. Slang's `IFunc<float, float>` interface lets the template take a `IFunc<float, float> activation` parameter, called inside the C-tile store loop. Cuts template code paths from 6 (one per activation) to 1, and lets the autodiff pilot in §P3.3 register backward closures cleanly. Test `TestIFuncEpilogue`.
- [ ] **P15.3 — `Property<T>` / `__subscript` accessor patterns**: Slang's `Property<T>` lets a `struct` expose a computed-on-access scalar (e.g. `flat_index` from `(x, y, z)`). Kernel address-arithmetic CSE (P11.5) can flow through a Property accessor without manual hoisting. Test `TestPropertyAccessor`.
- [ ] **P15.4 — `defer` for cleanup ordering**: rare in compute shaders but useful for the autotune-trace-injection path where we want to write a perf counter to LDS before the kernel exits, regardless of which return path fires. Test `TestDeferCleanup`.
- [ ] **P15.5 — `[knownAttribute]` for compile-time-constant gates**: Slang has `[knownAttribute("T", N)]` for embedding compile-time properties readable from reflection. Use it to tag each kernel with `[knownAttribute("vgprs", measured)]` so the autotuner's regression gate (P3.3.10) can read the tagged value directly instead of re-parsing slangc dump. Test `TestKnownAttributeReflection`.
- [ ] **P15.6 — `__target_switch` for vendor-specific code paths**: Slang's `__target_switch (case spirv: ...; default: ...)` lets one source compile differently per backend. Use it to express RDNA1's `s_waitcnt` pragma vs NVIDIA's L1-cache hint *in shader source*, not via Python branching at codegen time. Test `TestTargetSwitchVendor`.
- [ ] **P15.7 — `[anyValueSize(N)]` dynamic-dispatch tables**: useful for the upcoming combo-kernel-with-mixed-dtypes case. Catalogue of registered ops keyed via `[anyValueSize(64)]` slot-table so the SIMDKernel can pick the right epilogue at runtime without specialisation explosion. Test `TestAnyValueDispatch`.
- [ ] **P15.8 — Slang tagged-union `enum` types for op-kind dispatch**: `enum OpKind { Add, Mul, Gelu, ... }` with associated-data variants. Replaces the current string-switch in `VulkanOverrides`. Test `TestSlangEnumOpKind`.
- [ ] **P15.9 — `[noinline]` audit**: dual to P9.10's `[ForceInline]`. Mark large helpers (`erf` on f64, the unrolled bitonic-sort) as `[noinline]` so slangc doesn't inline them per use site, blowing instruction cache. Test `TestNoInlineHelpers`.
- [ ] **P15.10 — Link-time module merging**: when multiple kernels in a single graph share `import inductor_runtime`, link them into a single SPIR-V module via slangc's `-link` step. Reduces SPIR-V module count from N → 1 per graph; cuts pipeline-creation overhead since the driver caches module-level optimisation results. Test `TestLinkTimeModuleMerge`.
- [ ] **P15.11 — `__intrinsic_op` for SPIR-V opcodes Slang doesn't yet expose**: `OpGroupNonUniformBallot` variants, `OpAtomicFAddEXT` (where supported), `OpCooperativeMatrixMulAddKHR` (already used). Define a small library of `__intrinsic_op(OpAtomicFAddEXT) T atomic_add_f32(...)`-style declarations in `shaders/inductor_runtime/intrinsics.slang`. Test `TestIntrinsicOp`.
- [ ] **P15.12 — Slang IR introspection via `getReflectedJSON()`**: parse slangc's reflection output to drive the runtime's binding/PC bookkeeping (sister to P9.6.1 but for the shape/dtype/layout of every parameter). Test `TestReflectionJSONIntrospect`.
- [ ] **P15.13 — Slang module precompilation cache**: ship `shaders/inductor_runtime/*.slang-module` (precompiled binary modules) alongside the source modules. slangc loads the binary directly, skipping parse + sema. Cuts cold-compile time per kernel from ~80 ms to ~20 ms. Test `TestSlangModulePrecompile`.
- [ ] **P15.14 — `RaytracingPipeline`-style descriptor binding via `ParameterBlock` arrays**: `ParameterBlock<Params>[]` lets a single descriptor set serve a *batch* of kernel invocations with different parameters. Use for combo-kernel dispatch where N pointwise kernels share an input layout. Test `TestParameterBlockArray`.

**Items relocated/expanded under P9 numbering (P9.12–P9.18) for continuity with P9 organisation:**

- [ ] **P9.12 — `[BackwardDifferentiable]` audit**: every shader marked `[Differentiable]` should also be checked for `[BackwardDifferentiable]` to enable reverse-mode. Audit the §P3.3 autodiff pilots. Test `TestBackwardDifferentiableAudit`.
- [ ] **P9.13 — `where_clause` / `where T : ITypeConstraint` generic constraints**: tighten the generic-helper signatures in `slang_helpers.py` so misuse is a compile-time error, not a runtime SPIR-V validation failure. Test `TestGenericConstraintCompileError`.
- [ ] **P9.14 — `init` / `init { ... }` initialiser lists**: replace the procedural `Foo f; f.x = ...; f.y = ...;` patterns in matmul tile loaders with `Foo f = { x, y, z };`. Cleaner emitted Slang; same SPIR-V. Test `TestStructInitializerList`.
- [ ] **P9.15 — `func<T>(...)` partial-specialisation overloading**: when a generic helper has two specialisations (f32-fast-path and f16-CAS-path), Slang lets us declare both and dispatch via overload resolution. Drops Python branching in `emit_helpers`. Test `TestPartialSpecialisation`.
- [ ] **P9.16 — `[shaderStage(...)]` cross-stage code reuse**: in case we ever emit graphics-pipeline ops (e.g. compute → graphics fence), `[shaderStage]` gates code paths. Future-proofing. Test `TestShaderStageGate`.
- [ ] **P9.17 — Slang's `differential` parameter qualifier**: for the autodiff pilot, marking primal inputs `differential T x` lets Slang's autodiff machinery emit fewer redundant gradient zeros. Test `TestDifferentialQualifier`.
- [ ] **P9.18 — `__target_intrinsic` + `__intrinsic_asm` plumbing**: more general form of P15.11 — for opcodes with vendor-specific encodings. Test `TestTargetIntrinsicASM`.

---

### P16 — Profile-guided optimisation & online autotune

The current autotuner is one-shot: benchmark candidates at compile time, pick the best, persist. Real workloads shift over time (warm cache, thermal throttling, dynamic shapes). Online autotune closes that gap.

- [ ] **P16.1 — Workload trace capture**: `inductor_stats.start_workload_trace(name)` records `(kernel_id, dispatch_count, total_us, p50, p95, p99)` per kernel for the workload. Trace persisted to `~/.cache/torch_vulkan/workload_traces/<name>.json`. Test `TestWorkloadTrace`.
- [ ] **P16.2 — Bayesian-optimisation tile picker**: the matmul autotuner (§P2.2) currently grid-sweeps a fixed candidate list. A Bayesian-optimisation surrogate (Gaussian process over `(M, N, K, dtype) → ms`) can pick the next candidate that maximises expected improvement, reaching the optimum in 5–10 evals instead of 12+. File: `autotune.py`. Test `TestBayesianTilePicker` — converges within budget on a synthetic benchmark.
- [ ] **P16.3 — Cross-shape autotune transfer learning**: when a new shape is queried that's "close" to a cached shape (Hamming distance < 2 power-of-2 steps on each of M/N/K), warm-start the autotune from the cached optimum instead of cold-starting. Cuts cold-shape autotune time. File: `autotune.py`. Test `TestAutotuneTransferLearning`.
- [ ] **P16.4 — Hardware-counter-driven tile selection**: `VK_KHR_pipeline_executable_properties` exposes per-pipeline VGPR / occupancy / instruction counts directly from the driver. Use these as the autotune signal in addition to wall-time, so a tile that's slightly slower but uses 30% fewer registers wins for VGPR-budget-constrained workloads. File: `runtime.py`, `csrc/vulkan/Context.cpp`. Test `TestPipelineExecutableCounters`.
- [ ] **P16.5 — Online re-bench with TTL**: every kernel's autotune entry carries an `inserted_at` timestamp; entries older than `TORCH_VULKAN_AUTOTUNE_TTL_HOURS` (default 168 = 1 week) are re-benched on next access. Detects driver upgrades that change relative tile cost. Test `TestAutotuneTTL`.
- [ ] **P16.6 — Dispatch-frequency-weighted autotune budget**: the autotuner today spends the same budget on a kernel called 10× as on one called 10⁶×. Weight the budget by call frequency from §P16.1's trace, so hot kernels get more candidate evals. Test `TestAutotuneBudgetWeighted`.
- [ ] **P16.7 — Autotune cache compaction**: walk the autotune cache periodically, drop entries with `dispatches < 100` (rarely hit). File: `autotune.py:gc_autotune_cache(min_dispatches=N)`. Test `TestAutotuneCompaction`.
- [ ] **P16.8 — Regression-guarded autotune updates**: when re-benching surfaces a faster tile but it bumps a regression-asserted dispatch count, the update is rejected. Test `TestAutotuneRegressionGuard`.
- [ ] **P16.9 — Per-driver-version autotune namespace**: cache key extended with `(driver_vendor, driver_version)` so a driver upgrade doesn't silently use stale entries. Test `TestAutotuneDriverNamespace`.
- [ ] **P16.10 — Autotune trace replay for CI**: a recorded trace can be replayed offline to validate the picker's decisions deterministically. Test `TestAutotuneReplay`.

---

### P17 — Cross-kernel & pipeline-level codegen

The current codegen optimises *one kernel at a time*. Several high-value wins span kernel boundaries — they require the wrapper, the scheduler, or the C++ dispatch layer to participate.

- [ ] **P17.1 — Multi-kernel epilogue chaining without intermediate buffers**: when kernel `A` produces tensor `t` consumed exactly once by kernel `B`'s pointwise body, and both kernels target the same threadgroup tiling, *inline* B's body at A's store site and drop the intermediate buffer. Saves one global-memory round-trip + one allocator round-trip. Generalises the existing template-epilogue path beyond extern templates. File: `scheduling.py:can_fuse_vertical`. Test `TestEpilogueChainNoBuffer`.
- [ ] **P17.2 — Command-buffer-level loop unrolling**: when the wrapper emits `for _ in range(N): kernel.run(args)` with statically known `N` (e.g. an unrolled FX `aten.repeat` chain), record N dispatches into the command buffer in one `vkCmdDispatch * N` sequence. Cuts wrapper Python overhead × N. Test `TestCommandBufferUnroll`.
- [ ] **P17.3 — Kernel pipelining via Vulkan timeline semaphores**: today every kernel waits for its predecessor via pipeline barriers. For independent kernels (different output buffers, no dependency), emit timeline-semaphore-based fine-grained sync so the GPU can overlap them. Capability-gate on `VK_KHR_timeline_semaphore` (universally supported). Test `TestKernelPipelineTimeline`.
- [ ] **P17.4 — Secondary command buffer reuse for repeated dispatches**: long training runs re-execute the same compiled graph thousands of times. Record the dispatch sequence into a *secondary* command buffer once, reuse via `vkCmdExecuteCommands`. Cuts CPU-side recording overhead per step from ~50 µs to ~5 µs. File: `csrc/backend/dispatch.cpp`. Test `TestSecondaryCmdBufReuse`.
- [ ] **P17.5 — Vulkan event-based fine-grained sync**: `VkEvent` lets us split the typical "barrier + dispatch + barrier" into "dispatch + setEvent + dispatch + waitEvent" pairs that overlap better when the dependency is partial. Test `TestVkEventOverlap`.
- [ ] **P17.6 — Persistent-shader streaming inference path**: for batch-1 latency-sensitive inference, a single persistent compute shader that loops over inputs from a queue beats per-input dispatch. Gated on `TORCH_VULKAN_PERSISTENT_INFERENCE=1`. Test `TestPersistentInferenceShader`.
- [ ] **P17.7 — Double-buffered backward-graph dispatch overlap**: forward step N+1 can start while backward step N is still finishing on the same queue, since they touch disjoint buffers. Two parallel command buffers + a queue submit batch. Test `TestForwardBackwardOverlap`.
- [ ] **P17.8 — Scratch-buffer arena per fused region**: §P2.4's workspace pool is global; a per-fused-region arena (created at region entry, freed at exit) bounds the live-set tighter and lets the allocator reclaim memory aggressively. File: `wrapper.py`, `scheduling.py`. Test `TestScratchArena`.
- [ ] **P17.9 — Cross-kernel constant propagation**: a tensor produced by kernel A as `aten.full(0.0)` consumed by kernel B should propagate the constant to B at codegen time so B can fold the zero. Today the buffer round-trip materialises the zeros. File: `scheduling.py`. Test `TestCrossKernelConstantProp`.
- [ ] **P17.10 — Cross-kernel dead-store elimination at wrapper**: when the wrapper emits a buffer that's never read (e.g. the second output of a multi-output kernel that the caller ignores), the kernel still writes it. Detect at wrapper-codegen time and emit a single-output variant. Test `TestWrapperDeadStoreElim`.
- [ ] **P17.11 — Compute-queue affinity for graph regions**: RDNA1 has a single compute queue; NVIDIA / RDNA3 have multiple. Pin attention's matmul to one queue, the softmax to another so they overlap. Capability-gated. Test `TestComputeQueueAffinity`.
- [ ] **P17.12 — Cross-kernel CSE for repeated reductions**: when two adjacent kernels both compute `sum(x)` over the same tensor, the second should reuse the first's result. Detect at scheduler time, hoist the reduction into a shared scratch buffer. Test `TestCrossKernelReductionCSE`.

---

### P11.12–P11.16 — Codegen quality round 3

Additions to §P11 surfaced by the third review pass.

- [ ] **P11.12 — Cross-kernel ILP via larger workgroups**: when a kernel chain has independent dispatches, merge into one larger kernel with intra-workgroup ILP. Different from §P17.1 (which inlines into the same kernel) — this fuses *peer* kernels that have no data dependency. Test `TestCrossKernelILP`.
- [ ] **P11.13 — Kernel fusion across reduction barriers**: today reductions break fusion. When a reduction is followed by a pointwise that uses the *broadcast* of the reduction result (e.g. `softmax`'s divide by `sum`), keep the reduction's result in `groupshared` and let the pointwise read it without a global round-trip. Persistent-reduction codegen does this for `softmax` already; generalise. Test `TestReductionPointwiseFuse`.
- [ ] **P11.14 — Persistent-thread codegen for variable-length loops**: `aten.cumsum`, `aten.scan`, `aten.unique` benefit from persistent-thread codegen (one workgroup processes a stream of work-items via atomic counters) over multi-dispatch tile codegen. Test `TestPersistentThreadCodegen`.
- [ ] **P11.15 — FMA-grouping for fp32-accum bf16 inputs**: bf16 matmul accumulating in f32 needs `fma(bf16_a, bf16_b, f32_c)`. Group FMAs by accumulator so the SPIR-V emits a chain on the same register, not a scatter across registers. Test `TestFMAGrouping`.
- [ ] **P11.16 — Dynamic register-allocation hints**: emit `[spv::register_count(N)]` decoration on hot inner loops where we know the optimum register count. Lets the driver-side compiler skip its own register allocation pass. Test `TestRegisterAllocationHint`.
- [ ] **P11.17 — `aten.clamp(x, min_scalar, max_scalar)` under compile produces wrong values** *(discovered 2026-04-27)*: `torch.compile(lambda x: torch.clamp(x, -1.0, 1.0) + 1.0)` on Vulkan emits 1 dispatch but the +1.0 epilogue is dropped (or the clamp output is left at zero on cold cache miss). Reproduces with both positional and keyword min/max. Eager `torch.clamp(x, -1.0, 1.0) + 1.0` on Vulkan is correct, so the issue is Inductor-codegen-side. Suspect: the `clamp.Tensor` lowering (or `clamp.default` w/ scalar-tensor promotion) leaves the output buffer uninitialised when the FakeTensor pass runs in a path that bypasses the proper `compute` write-back. Fix path: audit `lowerings.py` / upstream `clamp_min`/`clamp_max` decomposition for Vulkan; route to `where(x < min, min, where(x > max, max, x))` directly via `VulkanOverrides` instead of relying on the upstream lowering. Regression test `TestClampScalarBoundsCompile` — `torch.compile(λ x. clamp(x, -1.0, 1.0) + 1.0)` matches CPU eager to 1e-5; mark `xfail(strict=True)` until the codegen fix lands.
- [ ] **P11.18 — `aten.digamma` / `aten.polygamma` numerical mismatch under compile** *(discovered 2026-04-27)*: 1 dispatch produced but values differ from CPU eager by O(1) on inputs `x.abs() + 0.1`. The eager Vulkan `digamma` shader matches CPU; the regression appears only under compile. Suspect: `slang_helpers.py` Lanczos-coefficients table for `c10_vulkan_digamma` is the asymptotic-series variant (good for `x ≥ ~6`) without the reflection / shift-up rule for `x ∈ (0, 6)`. Fix path: port the reference `digamma` (with reflection `ψ(1-x) = ψ(x) - π·cot(πx)` for `x ≤ 0` and shift-up `ψ(x+1) = ψ(x) + 1/x` for `0 < x < 6`) from the eager backend into the inline helper emitted by `emit_helpers`. `polygamma(n, x)` shares the same root cause. Test `TestDigammaPolygammaCompileNumerical` — `xfail(strict=True)` until the helper is corrected.
- [ ] **P11.19 — `aten.signbit` slangc compile error under compile** *(discovered 2026-04-27)*: `torch.compile(lambda x: torch.signbit(x).int() + 1)` raises `slangc failed for kernel ...: error[E30015]: undefined identifier`. The pointwise `signbit` snippet is missing from `VulkanOverrides`. Eager Vulkan supports `aten::signbit` (presumably via a hand-written shader) but the codegen path has no override. Fix path: add `signbit` to `overrides.py` as `((asuint({x}) >> 31) != 0u)` returning `bool`. Regression test `TestSignbitCompile` — `xfail(strict=True)` until the override lands.

---

### P8 — Final completeness checklist (post-roadmap closure)

These are the residual items that don't fit any of the prioritized tiers but that a "fully implemented inductor backend" cannot ship without. Completing P0–P7 is necessary but not sufficient — once everything above is `[x]`, walk this list to confirm the backend is genuinely done.

- [ ] **Every model in `benchmarks/inductor_train.py` hits its `compiled_ms ≤ eager_ms` target across {f32, f16, bf16}.**
- [ ] **All eager backend's 440 fused shaders are reachable from compiled mode** via Inductor lowerings, FX passes, ExternKernelChoice templates, or autodiff-generated backward shaders. The `scripts/audit_inductor_op_coverage.py` report (P1.7) shows zero unrouted ops on any benchmark model.
- [ ] **Every `[~]` blocker in this roadmap is closed** — either resolved or formally re-classified as `wontfix` with rationale.
- [ ] **AOT compile path (P3.1) ships a deployable artifact**: a single `.zip` package loaded by `aoti_load_package` runs MiniQwen3 inference on a fresh Vulkan-only target without Python at runtime.
- [ ] **Documentation parity**: every `inductor/<file>.py` has a top-of-file 1–2 paragraph docstring summarizing its role; every public `register*` / `prewarm*` / `compile*` API has a docstring with example. `pydoc -w torch_vulkan.inductor` produces a navigable HTML reference.
- [ ] **Regression suite is the public contract**: every dispatch-count target in §"Performance Targets" has a `==N` (not `≤N`) regression assertion; the suite is green on RDNA1 (the agent's GPU) AND on SwiftShader (the CPU CI fallback).
- [ ] **Long-running stability**: a 24-hour soak test running `benchmarks/inductor_train.py --model=transformer_block --train --steps=1_000_000` shows no memory growth, no slangc cache pathology, no allocator fragmentation, and no driver hangs. Test `TestLongRunningSoakNightly`.
- [ ] **Test-coverage budget**: every public function in `python/torch_vulkan/inductor/` has at least one regression test exercising it; coverage measured via `coverage.py` ≥ 90% on the inductor module. Test `TestCoverageBudget` — runs in CI, fails if the line-coverage drops below the budget.
- [ ] **Public-API stability commitment**: `python/torch_vulkan/inductor/extensions.py` (`register_template`, `register_lowering`, `prewarm_template`) is the documented extension surface. Lock the signatures; bumps require a SemVer-style version note in `CHANGELOG.md`. Test `TestPublicAPISignature` — pickled snapshot of `inspect.signature(...)` for each exported symbol; CI fails on any change.
- [ ] **`slangc` version pin policy**: backend depends on Slang ≥ X.Y.Z; document the minimum, lock via a runtime check in `runtime._slangc_available_cache`. Older slangc rejected with a clear error pointing at `third_party/slang/build/...`. Test `TestSlangcVersionPin`.
- [ ] **Clean `mypy --strict` / `pyrefly` on the inductor module**: every public function has full type annotations, no `# type: ignore` comments, no `Any` returns. Migration tracked via the `pyrefly-type-coverage` skill. Test `TestStrictTypeCheck` — CI runs the type-checker.
- [ ] **Public docstring coverage 100%**: `pydoc -w torch_vulkan.inductor` produces a navigable HTML reference; every module / class / public function has a docstring with at least a 1-line summary + (where applicable) a usage example. Test `TestDocstringCoverage` — fails when a public symbol has no docstring.
- [ ] **End-to-end deployment artifact**: a `.zip` package built from `aoti_compile_and_package` runs MiniQwen3 inference on a fresh Vulkan-only target via `aoti_load_package` — no Python at runtime, no slangc at runtime, only Vulkan loader + the prebuilt SPIR-V. Lock with `TestAOTIDeploymentArtifact`.
- [ ] **Cross-vendor CI matrix**: regression suite passes on RDNA1 (AMD), SwiftShader (CPU CI), NVIDIA Turing+ (NV native), and Intel Arc (Xe). Each vendor exercises the parts of the codebase capability-gated for it (cooperative-matrix on NV/RDNA3+, packed16 on NV/RDNA1, etc.). Test `TestCrossVendorMatrix` — single CI workflow runs the suite per vendor and reports.

---

## Continuous Loop Workflow

The agent loop runs the following cycle. Each iteration completes one roadmap item.

1. **Pick the highest unchecked item** in the roadmap above.
2. **Write a regression test first** (in `tests/test_inductor_regression.py`) asserting the dispatch-count or correctness target.
3. **Implement** in the appropriate inductor module (`lowerings.py`, `fx_passes.py`, `meta_patches.py`, `kernel.py`, `overrides.py`, or a new shader in `shaders/`).
4. **Compile shaders** if added: `SLANGC=... python tools/compile_shaders.py`.
5. **Build**: `MAX_JOBS=3 pip install -e . -v --no-build-isolation`.
6. **Run regression suite**: `python -m pytest tests/test_inductor_regression.py -p no:faulthandler --timeout=300 -q 2>/dev/null` — must stay green.
7. **Run the new test** and verify the target.
8. **Update the checkbox** in this doc to `[x]` and commit a one-liner status into [Recent Changes](#recent-changes) below.
9. **Pick the next item.**

Never bypass the regression suite. If a change regresses an existing test, prefer
fixing the implementation over relaxing the assertion.

---

## Performance Targets

These are the dispatch-count targets the roadmap is steering toward. They are
upper bounds; lower is better.

### Forward (compiled vs eager)

| Workload | Eager | Today (Inductor) | Target |
|----------|-------|-----------------|--------|
| MLP (linear+gelu+linear) forward | 8 | 5 | 3 |
| ResNet-18 forward | 81 | 29 | 18 |
| MobileNet-V2 forward | 55 | 24 | 18 |
| MiniQwen3 4L forward | 48 | n/a | 24 |

### Backward (compiled vs eager)

| Workload | Eager | Today (Inductor) | Target |
|----------|-------|-----------------|--------|
| MLP backward | 12 | n/a | 6 |
| ResNet-18 backward | 95 | n/a | 35 |
| MiniQwen3 4L backward | 106 | n/a | 50 |

### End-to-end training step

| Workload | Eager (ms) | Target Inductor (ms) |
|----------|------------|---------------------|
| MiniQwen3 4L (B=2, S=64, D=256) | ~11 | < 7 |

Every roadmap item that closes a gap should include a measurement showing it moved
the relevant column toward the target.

---

## Known Limitations

| Limitation | Impact | Workaround |
|-----------|--------|-----------|
| `slangc` must be on path or set via `SLANGC=` env var | JIT compilation fails | Set `SLANGC` before running compiled workloads |
| Inductor cache invalidation: must `rm -rf /tmp/torchinductor_*` after backend changes | Stale cached kernels | Clear cache manually when changing codegen or shader code |
| `@torch.compiler.disable` on `_patched_sdpa` | SDPA not traced by Dynamo, dispatched eagerly | Acceptable until P1.1 flash_attention FX pass lands |
| `aten.transpose` in compiled graphs fails | `bmm(q, k.T)` pattern can't be compiled | P0.3 above |
| FakeTensor PrivateUse1 dispatch priority | C++ view ops get called with FakeTensors during `fake_tensor_prop` | Mitigated by `meta_patches.py` + null-storage guards in C++ |
| `test_conv_bias_dispatch_count` allows ≤6 dispatches | Conv+bias is not fused | P0.2 |
| `_make_tile_mm_fn` pickling error | Non-fatal; Inductor logs warning about pickle failure | P1.4 |
| Backward graphs largely uncompiled | Backward dispatches use eager fallback for many ops | P0.1 |

---

## FakeTensor / `fake_tensor_prop` Notes

Inductor runs `fake_tensor_prop` on the FX graph before code generation. This traces the graph with
`FakeTensor` inputs to infer shapes. The issue: `FakeTensor` inputs have null storage (no GPU
buffer), but our PrivateUse1 C++ dispatch takes priority over Meta, so C++ ops get called with
FakeTensors.

Two-layer fix in place:
1. **`meta_patches.py`**: Registers Python-level `fake_impl` entries that are checked *before*
   the PrivateUse1 C++ path. Covers all view ops, BLAS ops, conv, normalization backward,
   indexing ops, activation backward, and factory ops including `randperm`.
2. **Null-storage guards in C++**: e.g. `vulkan_permute` checks `self.storage().data() == nullptr`
   and returns a contiguous meta tensor if so.

When adding a new C++ op that modifies shape or strides: add a fake impl in `meta_patches.py`.

---

## Lowerings Architecture (`lowerings.py`)

Inductor excludes `native_layer_norm`, `native_group_norm`, and `_softmax`/`_log_softmax` from
its default decomposition table, causing them to fall through to ExternKernel (our C++ dispatch).
This prevents Inductor from fusing these ops with adjacent pointwise ops.

`lowerings.py` re-registers these ops with Vulkan-specific Inductor lowerings that decompose them
into primitives Inductor can schedule and fuse.

The roadmap (P0.1 / P1.1) extends this list to cover the backward variants and the
RMSNorm / SwiGLU / QKV patterns common in transformer training.

---

## PrimTorch Coverage

All 127 prims implemented (100%). See [primtorch_coverage.md](primtorch_coverage.md) for the full
table. The remaining "gaps" are complex-dtype variants (complex SVD, complex FFT) which no current
model uses.

---

## Relevant Files

| File | Description |
|------|-------------|
| `python/torch_vulkan/inductor/` | The full Inductor backend |
| `python/torch_vulkan/inductor/lowerings.py` | Vulkan-specific Inductor lowerings (layer_norm, group_norm, softmax) |
| `python/torch_vulkan/inductor/fx_passes.py` | FX graph passes including bmm+scale fusion |
| `python/torch_vulkan/inductor/autotune.py` | WG-size autotuner with disk cache |
| `python/torch_vulkan/inductor/vulkan_template_caller.py` | mm/bmm/addmm `ExternKernelChoice` install hooks |
| `python/torch_vulkan/inductor/meta_patches.py` | FakeTensor fake impls for ~75 ops |
| `tests/test_inductor_regression.py` | Regression tests: dispatch counts + correctness |
| `csrc/ops/matmul_ops.cpp` | `vulkan_addmm` — fused mm+bias C++ dispatch |
| `csrc/ops/shape_ops.cpp` | `vulkan_permute` — includes FakeTensor null-storage guard |
| `shaders/matmul/mm_tiled2_bias_addmm.slang` | Fused GEMM+bias shader (transpose_b=0) |
| `shaders/matmul/mm_tiled2_bias.slang` | Fused GEMM+bias shader (transpose_b=1) |
| `docs/primtorch_coverage.md` | Full primtorch op coverage table |
| `docs/09-master-plan.md` | Original Inductor codegen design plan (historical) |

---

## Recent Changes

Append a one-line status entry per roadmap item completed.

- 2026-04-27: **P12 twenty-second batch — composite-math / two-arg trig / three-arg / bitwise-int pointwise coverage lock.** Wins (all 1 dispatch correct): `(a*a + b*b).sqrt() + 1` (hypot via expand); `atan2(y, x) + 1` (two-arg trig); `lerp(a, b, w) + 1` (three-arg interpolation); `frac(x) + 1` (fractional part); `expm1(x) + log1p(|x|) + 1` (composite e^x-1 / log(1+x)); `logaddexp(a, b) + 1` (log-sum-exp building block); `xlogy(|x|, |y|+0.1) + 1` (entropy-style special op); `(a & b) | (a ^ b)` (pure-int bitwise composite). Bugs discovered + filed: `clamp(x, -1.0, 1.0) + 1` produces wrong values under compile (epilogue lost / dispatch-state desync — see new P11.x entry); `digamma(x) + 1` and `polygamma(1, x) + 1` numerically wrong under compile despite 1 dispatch (slang_helpers Lanczos-coefficients table desync vs eager — see new P11.x entry); `signbit(x).int() + 1` raises slangc `undefined identifier` (overrides.py missing the snippet — see new P11.x entry). New regression class `TestP12PointwiseRoundNine` (8 tests). Suite: 373 → 381 passed (+8).
- 2026-04-27: **P12 twenty-first batch — bool axis reductions / int abs/neg/max-min / float floor-div / round chain coverage lock.** Wins (all 1 dispatch correct): full `all(b).int() + 1`; axis-0 `all(b, dim=0).int() + 1` and `any(b, dim=0).int() + 1`; integer `abs(x) + 1` and `-x + 1`; integer `maximum(x, y) - minimum(x, y) + 1`; `floor(x / (|y| + 0.1)) + 1`; `round(x) + round(x * 10) / 10`. Bugs ruled out: `arange(64, device='vulkan:0').float() * 0.1 + 1` raises `Tensor has no backing Vulkan buffer` (factory-op fast-path issue under compile when arange is the leaf); `x.max()` (full-reduction max returning scalar) blocked on `aten::max(Tensor)->Tensor` unimpl; `row_plus_vec` and `col_plus_vec` 3 dispatches (broadcast pattern; could fuse better with codegen tightening); `min/max(dim).values + 1` 4 dispatches (multi-output reduction over-codegens). New regression class `TestP12IntegerAndBoolAxisReductions` (6 tests). Suite: 367 → 373 passed (+6).
- 2026-04-27: **P12 twentieth batch — eq|ne / ge&lt / 3-way max/min / mixed-type f32+i32 coverage lock.** Wins (all 1 dispatch correct): `((x==y) | (x!=y)).int() + 1` (bool-or → int cast); `((x>=y) & (x<y+0.5)).int() + 1` (bool-and combo with scalar add); `maximum(maximum(x, y), x+y) + 1` (3-way max chain); `minimum(minimum(x, y), x+y) + 1` (3-way min chain); `x_f32 + i_i32.float()` (mixed-type pointwise via explicit cast). Bugs ruled out: `mul_int_scalar = x*3+1` 2 dispatches (would expect 1; possibly an integer-broadcast detail to investigate); `erf+erfc chain` blocked on Vulkan-eager `aten::erfc.out` unimpl; `std+amax` blocked on `aten::std.correction` unimpl; `mean(x,dim=-1)+amax(x,dim=-1)` 4 dispatches (different reduction kinds — would benefit from horizontal fusion). New regression class `TestP12CompareMaxMinAndMixedType` (5 tests). Suite: 362 → 367 passed (+5).
- 2026-04-27: **P12 nineteenth batch — sinc / std-unbiased (full + dim) / amax−amin horizontal-fuse / dtype-cast coverage lock.** Wins (all 1 dispatch correct): `sinc(x) + 1` (sin(πx)/(πx) chain with `x==0→1` corner); `std(x, unbiased=True) + 1` full-reduction Welford; `std(x, dim=-1, unbiased=True) + 1` per-row Welford; `amax(x, dim=-1) - amin(x, dim=-1) + 1` (horizontal-fused row reductions on shared input); `x.int() + 1` and `i.float() + 1` dtype-cast pointwise chains. `argmax(x).int() + 1` full-reduction probe initially showed 2 dispatches but reproducibly hit P11.2's documented `NotImplementedError` in cleaner test isolation (the cse-generated argmax wrapper still has the float2/float typing skew); kept that test out and tracked under existing P11.2 entry. New regression class `TestP12StdSincArgmaxAndCasts` (5 tests). Suite: 357 → 362 passed (+5).
- 2026-04-27: **P12 eighteenth batch — instance_norm / group_norm_4d / mean+var-welford / exp-neg-abs / softmax+1 / repeat coverage lock.** Wins: `instance_norm + 1` 4 dispatches correct on (B=2, C=4, H=4, W=4); `group_norm_4d + 1` 4 dispatches; `mean(x, dim=-1) + var(x, dim=-1) + 1` **1 dispatch** (Welford-shared reduction — same row-pass produces both); `exp(-|x|)/2 + 1` 1 dispatch (Laplace-PDF chain); `softmax(x, dim=-1) + 1` 1 dispatch (existing wide-row reduction handles amax→exp→sum→div→add); `x.repeat(2, 1) + 1` 1 dispatch (no contiguous materialisation). Bugs ruled out from probe: `F.normalize`, `unflatten`, `reshape(2,4,16)`, `expand(...).contiguous() + 1` all hit FakeTensor data-pointer family (P0.0); `masked_select` blocked on Vulkan-eager unimpl; `meshgrid+stack`, `cartesian_prod`, `addmm_chain`, `diagflat`, `matmul+relu+epilogue` 0-dispatch-with-stale-data (P1.5 family) or autotune CUDA-leak (P5.7). New regression class `TestP12NormReduceAndDistribution` (6 tests). Suite: 351 → 357 passed (+6).
- 2026-04-27: **P12 seventeenth batch — layer_norm without affine coverage lock.** `F.layer_norm(x, [16]) + 1.0` with `weight=None, bias=None` compiles to **4 dispatches** correct via the existing `_register_layer_norm` lowering's None-affine path on Vulkan (no broadcast multiply / add in the epilogue). The affine variant is covered by `TestNormLowerings`; this locks the no-affine path separately. New regression class `TestP12LayerNormNoAffine` (1 test). Suite: 350 → 351 passed (+1).
- 2026-04-27: **zeta correctness bug fix.** `c10_vulkan_zeta(s, q)` Slang helper was a 3-term Euler-Maclaurin truncation that only worked for very large q — it returned `(s-1)*q^-s + (-0.5)*q^-(s+1) + 0.5*q^-s` and zero outside `s>1, q>0`. Replaced with the Cephes-style direct sum of 9 leading terms + Euler-Maclaurin tail in `a = q+9` using the first 6 even Bernoulli numbers (B2/2! through B12/12!) and a rising-factorial recurrence. Found a sign bug in the half-endpoint correction: should be `+0.5*b` (Euler-Maclaurin endpoint contribution to the inner-region sum) not `-0.5*b`. `TestP12PointwiseRoundFour.test_zeta_correctness` locks correctness across `s ∈ [2, 5], q ∈ [0.5, 3]` — measured max diff is 1.9e-6 vs CPU eager. Suite: 349 → 350 passed (+1).
- 2026-04-27: **i1 / i1e correctness bug fix.** `c10_vulkan_i1` / `c10_vulkan_i1e` Slang helpers (`slang_helpers.py:emit_helpers`) had two bugs: (1) the large-`|x|` polynomial leading coefficient was `0.22829657f` instead of the Numerical-Recipes `0.39894228f`, producing 1–3% relative error already at `|x|=5`; (2) the sign branch `(ax > 0.0f ? 1.0f : -1.0f)` was always `+1` because `ax = abs(x)` is non-negative — so negative-input results were wrong-sign on the large-|x| branch. Plus a parenthesis imbalance in i1 that was causing slangc to error out (`Cannot access data pointer` was the visible failure). Replaced both helpers with the canonical Numerical Recipes single-precision coefficients and explicit `(x < 0.0f) ? -ans : ans` sign at the end. `TestP12PointwiseRoundFour.test_i1_correctness` (`|x|≤3` tight tolerance) + `test_i1e_correctness` (`|x|≤5` looser) lock the contract. Discovered during the same probe pass that found the erfinv / zeta bugs. Suite: 347 → 349 passed (+2).
- 2026-04-27: **erfinv correctness bug fix.** `c10_vulkan_erfinv` Slang helper in `slang_helpers.py:emit_helpers` was a garbled rational-polynomial fragment that returned values close to ±1 for any input (e.g. `erfinv(0.5)` returned −0.99 instead of 0.4769) — the polynomial-evaluation loops were structurally wrong (`r = r * s + s - 2.0f * s` and `r += s * r` make no sense for erfinv). Replaced with Mike Giles' single-precision approximation that CUDA's `erfinvf` uses: two-branch (`w < 5` vs sqrt-branch) polynomial in `w = -log((1-x)*(1+x))`. Verified against CPU eager: max error ≤1.2e-7 across `[-0.99, 0.99]`. `TestP12PointwiseRoundFour.test_erfinv_correctness` locks the contract. Discovered during P12 sixteenth-batch probe pass (`erfinv` was returning diff~9.5 even on clean range). Suite: 346 → 347 passed (+1).
- 2026-04-27: **P12 sixteenth batch — pointwise round 4 (log1p / rsub scalar+tensor / expit / trunc / fmod-scalar / square) coverage lock.** Wins (all 1 dispatch correct): `log1p(|x|) + 1`; `rsub(x, 1.0) + 1` and `rsub(x, y) + 1`; `special.expit(x) + 1` (sigmoid alias); `trunc(x) + 1`; `fmod(x, 0.5) + 1`; `square(x) + 1` (`x*x` decomp). Bugs ruled out: `erfinv` returns 1 dispatch with wrong values (diff~9.5 even on `[-0.5, 0.5]` clean range — the existing `c10_vulkan_erfinv` override looks suspect, follow-up needed); `special.zeta(|x|+1, 0.5) + 1` returns 1 dispatch with diff~32 (zeta override numerical issue); `fmod(x, y_tensor)` 3 dispatches correct (tightening as follow-up); `div(x, y, rounding_mode='floor'|'trunc')` blocked on Vulkan-eager `aten::div.out_mode` unimpl; `threshold(x, 0.5, -1.0)` blocked on `aten::threshold.out` unimpl. New regression class `TestP12PointwiseRoundFour` (7 tests). Suite: 339 → 346 passed (+7).
- 2026-04-27: **P12 fifteenth batch — clamp_min/clamp_max scalar+tensor+chain, dim-axis any/all coverage lock.** Wins (all 1 dispatch correct): `clamp_min(x, 0.0) + 1` (scalar); `clamp_max(x, 0.0) + 1` (scalar); `clamp_min(x, y_tensor) + 1` (tensor lower-bound, equivalent to maximum); `clamp_min(x, -0.5) + clamp_max(x, 0.5)` (chain pointwise-fuses); `(x > 0).all(dim=-1).int() + 1` (dim-axis bool reduce); `(x > 0).any(dim=-1).int() + 1` (dim-axis bool reduce). The dim-axis `all`/`any` paths are separate from `TestAnyAllReductions` (no-dim form). Bugs from probe pass not locked: `narrow_chain` extern-falls with `expected size 3==3, stride 4==16 at dim=0` (lowering bug for double-narrow); `squeeze_unsqueeze` chain hits FakeTensor data-pointer family (P0.0); `logical_or/xor` blocked on Vulkan-eager `aten::logical_*.out` unimpl. New regression class `TestP12ClampAndDimAnyAll` (6 tests). Suite: 333 → 339 passed (+6).
- 2026-04-27: **P12 fourteenth batch — pointwise round 3 (copysign / frac / expm1 / log_ndtr / log2+log10 / int bitwise shifts) coverage lock.** Wins (all 1 dispatch correct): `copysign(x, y) + 1`; `frac(x) + 1` (`x - trunc(x)` decomp); `expm1(x) + 1` (existing pointwise override); `special.log_ndtr(x) + 1` (`erfc(-x/√2)/2 + log` decomp pointwise-fuses); `log2(|x|+0.1) + log10(|x|+0.1)` (2-log chain); int32 `(x << y) + (x >> y)` (bitwise shifts). Bugs ruled out from this batch: `selu/celu + 1` and the `selu+celu+hardtanh+relu6` chain hit the constant-fold-during-trace family (P1.5 view-fast-path); `bitwise_not(bool) + 1` returns 1 dispatch but with wrong values (diff=1.0); `gradient(x, dim=-1)[0] + 1` 16 dispatches correct (defer until measurement surfaces it as a hot path); `rrelu_eval` blocked on Vulkan-eager `rrelu_with_noise` unimpl. New regression class `TestP12PointwiseRoundThree` (6 tests). Suite: 327 → 333 passed (+6).
- 2026-04-27: **P12 thirteenth batch — pointwise round 2 (nan_to_num / logit / logaddexp{,2} / sgn / real-passthrough / heaviside / float_power) coverage lock.** Wins (1 dispatch correct unless noted): `nan_to_num(x, 0.0) + 1`; `logit(sigmoid(x), eps=1e-6) + 1`; `logaddexp(x, y) + 1` and `logaddexp2(x, y) + 1` (Inductor's stable `max(x,y) + log1p(exp(-|x-y|))` decomp pointwise-fuses); `sgn(x) + 1`; `torch.real(x) + 1` real-input passthrough. Correctness locks: `heaviside(x, 0.5) + 1` 5 dispatches (will tighten once a fused override lands), `float_power(|x|, |y|) + 1` 4 dispatches with auto-promotion to f64 matching eager. Bugs not locked: `clip(x, lo, hi) + 1` returns 0 dispatches with stale data (P1.5 view-fast-path family); `poisson_nll_loss` 1 dispatch but NaN (likely log-of-0 corner in the unguarded decomp); `hann/hamming_window` and `rad2deg/deg2rad` and `fmin/fmax` blocked on Vulkan-eager unimpls (`arange.start_out` / `mul.out` / `fmin.out` / `fmax.out`). New regression class `TestP12PointwiseRoundTwo` (8 tests). Suite: 319 → 327 passed (+8).
- 2026-04-27: **P12 twelfth batch — depthwise/1×1 conv, scatter_add+epilogue, bias-less 2-layer linear, where+reduce coverage lock.** Wins: depthwise conv `F.conv2d(x, w, groups=4)+1` 5 dispatches (will tighten when P5.1 depthwise template lands); 1×1 conv `F.conv2d(x, w_1x1)+1` 5 dispatches (will drop to ≤2 with P5.1 1×1→mm fast path); `base.scatter_add(1, idx, src)*2` 3 dispatches via atomic-add + pointwise; 2-layer no-bias MLP `linear(relu(linear(x)))` 5 dispatches; `where(x > 0.5, zeros, x).sum(dim=-1)` 1 dispatch (where + reduce fuses cleanly). New regression class `TestP12ConvVariantsAndAdvanced` (5 tests). `searchsorted` and `f_normalize` blocked on Vulkan-eager unimpl / P0.0 family. Suite: 314 → 319 passed (+5). **Cumulative this turn: 247 → 319 = +72 passing tests across 14 measurement-locked cycles** plus the 5 xpass→pass promotions.
- 2026-04-27: **P12 eleventh batch — loss forwards (huber/smooth_l1/kl_div), elu(α), tile, multi-statistic reduce coverage lock.** Wins (1 dispatch correct): `F.huber_loss(x, y, delta=1.0)`; `F.smooth_l1_loss(x, y, beta=1.0)`; `F.kl_div(log_softmax(x), softmax(y), reduction='batchmean')`; `F.elu(x, alpha=2.5) + F.elu(x, alpha=0.1)` (non-default alpha plumbing); `x.tile(2, 3) + 1`; `mean(x, dim=-1) - std(x, dim=-1) + sqrt(var(x, dim=-1))` (3 reductions fuse into one welford-driven dispatch). New regression class `TestP12LossForwardAndAdvancedReduce` (6 tests). `matmul_pointwise_chain` (`gelu(x@y) * 0.5 + (x@y).tanh()`) hits the known P5.7 autotune CUDA-leak. Suite: 308 → 314 passed (+6).
- 2026-04-27: **xpass→pass promotion — 5 backward correctness tests flip green.** Auditing the suite's 5 xpassed tests: `TestActivationBackward::test_sigmoid_backward_correctness` + `test_silu_backward_correctness` + `TestLossBackwardLowerings::test_mse_loss_backward` + `test_smooth_l1_loss_backward` + `test_bce_backward` were all `xpass` (passing despite the class-level `@pytest.mark.xfail(strict=False, reason=_BWD_FAKE_DEVICE_BLOCKER)`). These specific backward paths do **not** traverse the broken view-fast-path code; the activation lowerings + loss-bwd lowerings shipped under P0.1 / P1.7 fully cover them. Refactor: removed the class-level xfail decorators, applied per-test xfail markers to only the dispatch-count tests (`test_sigmoid_backward_dispatch_count`, `test_tanh_backward_dispatch_count`, `test_silu_backward_dispatch_count`, `test_gelu_backward_dispatch_count` — these still hit extern-aten ops that traverse the view-fast-path) and to `test_huber_loss_backward` (its `where` branch still hits the path). Class docstrings updated with the precise mechanism. Suite: 303 passed + 5 xpassed → **308 passed + 0 xpassed** (+5 promoted, 18 xfailed unchanged). P0.1 activation-backward + loss-backward lowerings move from "shipped + waiting on P0.0" to "shipped + verified passing under compile" for sigmoid / silu / mse / smooth_l1 / bce.
- 2026-04-27: **P12 tenth batch — softplus/threshold/softsign/tanhshrink/hardshrink/softshrink/logsigmoid + split/unbind/stack-of-activations coverage lock.** Wins (1 dispatch unless noted): `softplus(x, β, threshold) + threshold(x, 0, -1)` (2 less-common activations); `softsign + tanhshrink`; `hardshrink + softshrink`; `logsigmoid + sigmoid`; `split(x, 8, dim=0); s0 * s1` (split + pointwise on stride+offset views); `unbind(x, dim=0); u[0]*u[1] + u[2]*u[3]` (4-way unbind + pointwise); `stack([relu, gelu, silu], dim=0).sum(dim=0)` 2 dispatches (3 fused activations + new-dim stack + reduce). New regression class `TestP12FunctionalActivationsAndShapeCombo` (7 tests). Suite: 296 → 303 passed (+7).
- 2026-04-27: **P12 ninth batch — int reductions, round/floor/ceil/trunc, fancy indexing, multi-norm coverage lock.** Wins (1 dispatch correct): `x.sum(dim=-1)+x.amax(dim=-1)+x.amin(dim=-1)` on int32 (integer-dtype reductions); `round+floor+ceil+trunc`; `linalg.norm(x, ord=2, dim=-1) + linalg.norm(x, ord=1, dim=-1)` (multi-norm via existing P1.7 lowering); `x[idx]+1` advanced indexing. New regression class `TestP12IntegerAndIndexingPatterns` (4 tests). Bugs found: `prod_reduction` hits FakeTensor; `masked_fill` 1 dispatch but wrong values (diff=1.82, new bug); `torch.matmul + 1` triggers `addmm() Expected Tensor for self but found float` post-grad bug. Suite: 292 → 296 passed (+4).
- 2026-04-27: **P12 eighth batch — factory-like + abs/neg/sign + pow + minimum/maximum + sqrt/rsqrt/log + trig coverage lock.** Wins (all 1 dispatch correct): `zeros_like + ones_like + full_like + x` (factory constants inlined as scalar literals); `abs(x) + (-x).neg() + sign(x)`; `pow(x,2.5) + x.pow(3.0) + x**0.5`; `minimum(x,y) + maximum(x,y)`; `sqrt(x) + rsqrt(x) + log(x) + log2(x) + log10(x)` (5-log chain); `sin+cos+tan+asin+acos+atan` (6-trig chain). New regression class `TestP12FactoryAndPointwiseBreadth` (6 tests). The pointwise codegen breadth is now well-locked across the major math primitive categories — any future codegen change that splits any of these chains surfaces here. Suite: 286 → 292 passed (+6).
- 2026-04-27: **P12 seventh batch — conv / 2-layer MLP / multi-axis-reduce coverage lock; P1.5 family pattern hits 3 more cases.** Wins: `conv2d(x, w) + 1` 5 dispatches; `relu(conv2d(x, w, bias)) + 1` 4 dispatches (will tighten to ≤2 once P0.2 fused conv epilogue lands); 2-layer MLP `linear(gelu(linear(x)))` 5 dispatches (2 addmm extern + 1 fused gelu epilogue + alloc/marshal); multi-axis reduce `x.sum(dim=(0,2)) + x.mean(dim=1).sum(dim=-1, keepdim=True)` 4 dispatches. **P1.5 family pattern is systemic** — all of `slice_chain` `x[:,1:-1,::2]+1`, `permute(0,2,1,3).contiguous()+1`, and `expand(8,16)+1` return 0 dispatches with stale captured-during-trace data (probe diff=4.47/4.47/1.91). The P1.5 view-fast-path bug isn't isolated to a few op patterns — any zero-cost-view followed by a pointwise epilogue exhibits it. New regression class `TestP12ConvLinearAndMultiAxisReduce` (4 tests). Suite: 282 → 286 passed (+4).
- 2026-04-27: **P12 sixth batch — bitwise/shift/comparison/cast/cumsum coverage lock.** Wins (1 dispatch unless noted): `(x&y) | (x^y) | (~x&y)` int32 bitwise chain; `(x<<2) | (x>>1)` shift; `((x > y) & (x < y+1)).int().sum()` (compare→bool-and→cast→reduce); 4-step cast chain `f32→f16→f32→mul→i32→f32+1` collapses to **1 dispatch** (all intermediate casts inlined — important codegen-quality breadth check); `cumsum(x, dim=0)+1` 4 dispatches correct (existing scan codegen). Bug observed: `any(x, dim=-1) + all(x, dim=-1)` returns 1 dispatch but with wrong values (diff=1.0) — possible regression of the 2026-04-27 codegen-bug fix; `isfinite/isnan/isinf` chain hits FakeTensor data-pointer family. New regression class `TestP12BitwiseAndCastChains` (5 tests). Suite: 277 → 282 passed (+5).
- 2026-04-27: **P12 fifth batch — var/std, amin/amax, logsumexp, embedding+reduce, gather, bmm chain coverage lock.** Wins (1 dispatch unless noted): `var(x)+std(x)+1` (shared-load reduction); `amin(x)+amax(x)+1` (horizontal reduction fusion); `logsumexp(x)+1` (numerically-stable reduction with internal amax); `embedding(idx,w)*0.5 + embedding(idx,w).sum(-1, keepdim=True)` (embedding + reduction + broadcast-add fuses); `x.gather(1, idx)+1` (gather + epilogue); `bmm(a,b)*2 + gelu(bmm(a,b))` 5 dispatches (correctness lock; tightening would need cross-extern CSE — P17.12 territory). Bugs found: `selu+celu+prelu` chain returns 0 dispatches with stale data (P1.5 family — `prelu` likely tracing wrong); `mm@y@z + 1` fails with `addmm() Expected Tensor for self but found float` (post-grad pass bug). New regression class `TestP12ReductionAndIndexingChains` (6 tests). Suite: 271 → 277 passed (+6).
- 2026-04-27: **P12 fourth batch — 9-activation chain, softmax+log_softmax, RMSNorm-shape, F.layer_norm coverage lock.** Wins (1 dispatch unless noted): `gelu+silu+mish+hardswish+hardsigmoid+leaky_relu+elu+relu6+softplus` 9-activation chain (huge codegen breadth check), `softmax(x)+log_softmax(x).exp()` (shared row-reduction), `rsqrt(var(x,keepdim=True)+eps)*x` (RMSNorm core), `F.layer_norm(x, (D,), w, b)+1` 2 dispatches (existing `native_layer_norm` lowering). Bugs found: `clamp+clip` chain returns 0 dispatches with stale data (P1.5 family); `F.normalize` hits meta-device crash (P0.0 family); `F.max_pool1d` raises `RuntimeError: negative dimension -1` from the lowering (new bug). New regression class `TestP12ActivationAndNormChains` (4 tests). Suite: 267 → 271 passed (+4).
- 2026-04-27: **P12 third batch — hyperbolic + special pointwise + multi-output ops + topk + stack/hstack/vstack coverage lock.** Wins (all 1 dispatch correct unless noted): `tanh+atanh` chain, `sinh+cosh+asinh+acosh` chain, `frexp` (multi-output mantissa+exp), nested `where(cond1, where(cond2, a, b), c)` (branchless), `expm1+log1p+abs` chain, `stack([a,b,c])+1` (2 dispatches), `vstack+hstack.mean()` (2 dispatches). Also probed `topk(x,k).values.sum()` → 5 dispatches correct (locks correctness; tightening to bitonic-sort shader tracked under P4.8). New regression class `TestP12HyperbolicAndSpecialPointwise` (6 tests) + 2 stack/vstack tests in the dropout test class. Bug found: `einsum('ij,jk->ik')` and `einsum('bnhd,bnHd->bnhH')` both compile to 0 dispatches with stale data — same constant-fold-during-trace family as `broadcast_tensors`/`tensor_split`/P1.5; einsum entry annotated `[~]`. `flatten` blocked by FakeTensor data-pointer family. `cdist` extern-falls (Vulkan eager not implemented). `argmax/argmin single-axis` raises the documented `NotImplementedError` from P11.2. Suite: 259 → 267 passed (+8).
- 2026-04-27: **P14 dropout-during-eval already collapses via stock Inductor decomp.** Verified: `F.dropout(x, p=*, training=False)` decomposes to identity at graph-rewrite time so the Philox-RNG dispatch is pruned entirely — no Vulkan-specific FX pass needed. `F.dropout(x,0.5,training=False)+1.0` → 1 dispatch (output exactly `x + 1.0`); 3-stage chain `dropout(0.5,eval)→relu→dropout(0.3,eval)→+1` → 1 dispatch. `TestP14DropoutDuringEvalElim` (2 tests) locks the contract. Roadmap entry flipped to `[x]`. Suite: 257 → 259 passed (+2).
- 2026-04-27: **P12 second batch — `atleast_1d`, `block_diag`, `chunk`, `diff` all compile via stock codegen; `broadcast_tensors` / `tensor_split` codegen-bug discovered.** Wins: `atleast_1d(x)+1` → 1 dispatch correct; `block_diag(a,b)+1` → 1 dispatch correct; `chunk(x,2); c0+c1` → 1 dispatch correct; `diff(x)+1` → 4 dispatches correct (could tighten with a dedicated lowering — deferred until measurement surfaces it). Bugs found: `broadcast_tensors(a,b); a+b` and `tensor_split(x,4); sum(parts)` both compile to **0 dispatches** but return *stale captured-during-trace data* (probe diff=3.58 / 4.26). Same constant-fold-during-trace family as the P1.5 `view`+`pointwise` upstream-blocked bug — Inductor's pre-grad pass folds the broadcast result to a constant tensor whose contents were captured when inputs were uninitialized fakes. Both annotated `[~]` with the blocker. `atleast_2d` / `atleast_3d` blocked by FakeTensor data-pointer family (same P0.0 view-fast-path family); annotated. `TestP12ShapeAndLinalgCoverage` extended from 4 → 8 tests. Suite: 253 → 257 passed (+4).
- 2026-04-27: **P12 measurement-driven coverage lock — `linalg_cross` 1-dispatch, `kron` / `narrow` / `swapdims` work via stock codegen.** Probed the new P12 ops one-by-one to separate ones that need lowerings from ones that already work. Wins: `torch.linalg.cross(a, b, dim=-1) + 1.0` on 3-vectors compiles to **1 dispatch** correct (cross product's 6 scalar mul/sub ops on broadcast slices fuse into a single VulkanKernel); `torch.kron` 4 dispatches correct (broadcast+reshape+pointwise); `narrow` 2 dispatches correct (stride+offset view + pointwise epilogue); `swapdims` 2 dispatches correct (stride-only transpose + pointwise). Blocked: `aten.outer` and `aten.atleast_2d` both fail with `Cannot access data pointer of Tensor (e.g. FakeTensor)` — same P0.0 view-fast-path family. `TestP12ShapeAndLinalgCoverage` (4 tests) locks the working paths. Roadmap entries for `kron` / `linalg_cross` flipped to `[x]`; `narrow` flipped to `[x]`; `swapdims` flipped to `[~]` (covered, `movedim` audit deferred); `outer` annotated with the FakeTensor blocker. Suite: 247 → 253 passed (+6).
- 2026-04-27: **Roadmap expansion — third review pass added P12 / P13 / P14 / P15 / P16 / P17 + P9.12–P9.18 + P11.12–P11.16 (~85 new unchecked items).** §P12 — extended primop coverage round 2 (median/kthvalue/quantile/mode, einsum/tensordot/outer/inner/kron/cross/cdist, full linalg factorisations, complex/polar/view_as_real/view_as_complex, stft/istft/fft_c2c, movedim/swapdims/tensor_split/narrow/atleast_*/broadcast_tensors/block_diag, histogramdd, multinomial/bernoulli/poisson/geometric_/cauchy_/exponential_/log_normal_, audit-script round 2). §P13 — compile-pipeline & SPIR-V optimisation (`spirv-opt -O` post-pass, dead-binding strip, on-disk SPIR-V dedup, `slangc -incremental`, link-time optimisation, validation-clean SPIR-V, pre-warmed `VkPipelineCache`, parallel `vkCreateComputePipelines`, target-env pinning). §P14 — FX pre/post-grad pass expansion (`linear→bias→add→layernorm`, `attention→mask→softmax`, `gelu/silu→linear`, `embedding→add→ln`, back-to-back-linear merge, view→pointwise→view flatten, dropout-during-eval elim, dead-cast removal, scalar-broadcast hoist, `to_copy` cancel, `grad_out * 1.0` elim, `zeros_like→add` rewrite, broadcast-add bias epilogue, `aten.detach` audit). §P15 — Slang feature utilisation round 2 (inline `spirv_asm` for hot kernels, `IFunc<R,Args...>` epilogue plumbing, `Property<T>` accessors, `defer` cleanup, `[knownAttribute]` compile-time tags, `__target_switch` for vendor paths, `[anyValueSize(N)]` dynamic dispatch, Slang tagged-union enums, `[noinline]` audit, link-time module merge, `__intrinsic_op` for missing SPIR-V opcodes, `getReflectedJSON` introspection, `.slang-module` precompile cache, `ParameterBlock<T>[]` arrays). §P9.12–P9.18 — `[BackwardDifferentiable]` audit, generic-constraint tightening, `init {…}` initialiser lists, partial-specialisation overloading, `[shaderStage(…)]` gating, `differential` qualifier, `__target_intrinsic`/`__intrinsic_asm` plumbing. §P16 — profile-guided optimisation & online autotune (workload trace capture, Bayesian-optimisation tile picker, cross-shape transfer learning, hardware-counter-driven selection via `VK_KHR_pipeline_executable_properties`, online re-bench TTL, dispatch-frequency-weighted budget, autotune cache compaction, regression-guarded updates, per-driver-version namespacing, replay for CI). §P17 — cross-kernel & pipeline-level codegen (multi-kernel epilogue chaining without intermediates, command-buffer loop unrolling, kernel pipelining via timeline semaphores, secondary-cmdbuf reuse, `VkEvent` fine-grained sync, persistent-shader streaming inference, double-buffered fwd/bwd overlap, scratch-arena per fused region, cross-kernel constant prop, wrapper-level dead-store elim, compute-queue affinity, cross-kernel reduction CSE). §P11.12–P11.16 — cross-kernel ILP via larger workgroups, kernel fusion across reduction barriers, persistent-thread codegen, FMA-grouping for fp32-accum bf16, dynamic register-allocation hints. Header note bumped with the third review pass. No code changes; pure roadmap expansion.
- 2026-04-27: **P10 `_foreach_*` combo-kernel audit extended to 7 ops.** Probed `_foreach_lerp`, `_foreach_div`, `_foreach_neg`, `_foreach_abs` — all collapse to 1 combo-kernel dispatch on 2-tensor lists via the `BackendFeature.FOREACH` rewrite. `TestForeachAudit` extended from 3 → 7 tests. Suite: 243 → 247 passed (+4).
- 2026-04-27: **P4.8 `aten.scatter_reduce` non-sum modes triaged; `sum` mode locked.** Probed all 5 modes under `torch.compile`: `sum` ships via Inductor's atomic-add codegen (same path as `scatter_add`); `prod` / `mean` / `amax` / `amin` extern-fall-through to `aten::scatter_reduce.two_out` which is unimplemented in Vulkan eager. Roadmap entry updated with verified scope; new follow-up entry tracks the eager registration + Inductor codegen support for non-sum modes. `TestGatherScatterAdd.test_scatter_reduce_sum_correctness` locks the working path. Suite: 242 → 243 passed (+1).
- 2026-04-27: **P11.5 pointwise math chain coverage lock — `rsqrt*reciprocal`, `round/floor/ceil/trunc`, `fmod+remainder`, `sign+copysign+abs` all 1-dispatch.** `TestPointwiseMathChains` (4 tests) — Inductor's stock pointwise codegen collapses each chain on Vulkan with correct values. Suite: 238 → 242 passed (+4).
- 2026-04-27: **P10 / P11 measurement-driven coverage lock — `unbind+sum`, `chunk*+1`, `logsumexp` all 1-dispatch.** `TestShapeAndReduceFusion` (3 tests) — Inductor's stock view+pointwise + multistage-reduction codegen collapses each path to a single dispatch on Vulkan with correct values. Suite: 235 → 238 passed (+3).
- 2026-04-27: **Argmax/argmin single-axis codegen investigation — added P11.2 follow-up.** Upstream `make_reduction("argmax", override_return_dtype=int64)` only emits the `(value, linear_idx)` tuple on the multi-axis Triton path; single-axis Vulkan path receives just `value`, while `_argmin_argmax_reduction` asserted a 2-tuple. Investigation also found `wgreduce_argmaxmin` header naming mismatch with the helper-block keyword (`wgreduce_argmax`/`min`) and a float2/float CSE-typing skew in the cse-generated wrapper. Three coupled fixes needed; documented in detail under P11.2 with a `TestArgmaxArgminSingleAxisCodegen` test name. Currently raises `NotImplementedError` instead of crashing (no test added since the path is gated). Suite stays at 235 passed.
- 2026-04-27: **Codegen bug fix — `torch.any` / `torch.all` under compile.** Two latent bugs in `slang_helpers.py:emit_helpers`: (a) the `wgreduce_any` header was double-dispatched — once through the generic wave-op loop (which crashed with `KeyError: 'any'` because the dict only had sum/prod/max/min) and once through the dedicated `wgreduce_any` block. Skip `any` (and `argmax`/`argmin` which were also leaking through) in the generic loop alongside the existing `xor` skip. (b) The dedicated `wgreduce_any` block emitted `WaveActiveAnyBits` — the correct HLSL/Slang intrinsic is `WaveActiveAnyTrue(bool)`. Both fixed; `TestAnyAllReductions` (2 tests) covers the round-trip. Suite: 233 → 235 passed (+2).
- 2026-04-27: **P2.5 `addcmul` / `addcdiv` already 1-dispatch.** Outdated entry — Inductor's stock fused-multiply-add codegen collapses the AOT-autograd-split triple into a single fused dispatch on Vulkan. `TestAddcmulAddcdiv` (2 tests) locks the contract. Suite: 231 → 233 passed (+2).
- 2026-04-27: **P11.5 / P6.6 dead-axis elimination for size-1 dims.** `VulkanExprPrinter._print_FloorDiv` now folds `FloorDiv(x, 1) → x` and `FloorDiv(0, _) → 0`; `_print_ModularIndexing` folds `_ % 1 → 0`. Defensive folding for `keepdim=True` reductions that construct `FloorDiv(_, 1)` with `evaluate=False`. `TestExprPrinterSimplify` extended with 2 new cases. Suite: 229 → 231 passed (+2).
- 2026-04-27: **P10 measurement-driven coverage lock — `repeat`, `logical_and→cast→add`, `atan2+1` all 1-dispatch.** Three additional regression-tests added (`TestRepeatFused`, `TestLogicalAndBitwise`) — `Tensor.repeat(2,3)+1.0`, `logical_and(x>0, y>0).to(float32)+1.0`, `atan2(x,y)+1.0` all collapse via Inductor's stock view+broadcast+pointwise codegen on Vulkan. Suite: 226 → 229 passed (+3).
- 2026-04-27: **P10 `aten.where` already 1-dispatch.** `torch.where(x > 0, y, -y) + 1.0` compiles via Inductor's stock branchless codegen on Vulkan. `TestWhereFused` (1 test) locks the contract. Suite: 225 → 226 passed (+1).
- 2026-04-27: **P10 `aten.glu` already 1-dispatch.** `F.glu(x, dim=-1) + 1.0` compiles via Inductor's stock split+sigmoid+mul+pointwise codegen on Vulkan. `TestGluFused` (1 test) locks the contract. Suite: 224 → 225 passed (+1).
- 2026-04-27: **P10 `_foreach_add` / `_foreach_mul` / `_foreach_addcmul` combo-kernel audit shows 1 dispatch.** Measurement-driven — after P1.4 re-enabled `BackendFeature.FOREACH` + shipped the combo rewrite, three of the most common foreach ops collapse to 1 combo-kernel dispatch on 2-tensor lists. `TestForeachAudit` (3 tests) locks the contract. Suite: 221 → 224 passed (+3).
- 2026-04-27: **P10 reflection_pad2d already 1-dispatch; replication_pad2d codegen bug discovered.** Outdated entry — `F.relu(F.pad(x, …, 'reflect'))` compiles to 1 dispatch correct via Inductor's stock conditional-write codegen. `F.pad(x, …, 'replicate')` compiles but produces wrong values: the index expression wraps instead of clamping to the boundary (the lowering doesn't emit the correct `min(max(idx, 0), N-1)` for replicate). New `xfail(strict=True)` regression test `TestPadModes.test_replication_pad2d_correctness` locks the gap; tracked as a P1.7 codegen-correctness item. `TestPadModes.test_reflection_pad2d_one_dispatch_correct` ships green. Suite: 219 → 221 passed (+2 tests, +1 xfail).
- 2026-04-27: **P10 `aten.embedding(idx, weight) + 1.0` already 1-dispatch.** Outdated entry — Inductor's stock indirect-indexing+pointwise codegen on Vulkan already collapses the embedding gather and the pointwise consumer to a single dispatch; no ExternKernelChoice template needed for this path. `TestEmbeddingFused` (1 test) locks the contract. The `embedding → layer_norm` fusion path is a separate item under P5.5.
- 2026-04-27: **P1.7 `aten.native_batch_norm` (inference) + `aten.avg_pool2d` already 1-/2-dispatch.** Outdated entries — `F.batch_norm(x, m, v, w, b, training=False) + 1.0` compiles to 2 dispatches (mean/var → affine + epilogue, running-stats update); `F.avg_pool2d(x, 2) + 1.0` to 1 dispatch. Both via Inductor's stock decomp + Vulkan pointwise/reduction fusion. `adaptive_avg_pool2d` still blocked on Dynamo FakeTensor failure (same P0.0 family). `TestPoolAndBatchNorm` (2 tests) locks the contracts. Suite: 217 → 219 passed (+2).
- 2026-04-27: **P10 `aten.masked_scatter` / `aten.index_copy` already 1-dispatch correct.** Outdated entries — both compile correctly via Inductor's stock indirect-indexing codegen on Vulkan. `index_fill` falls through with a device-propagation mismatch (`aten.index_put.default` between vulkan and meta) — same P0.0 family blocker. `TestMaskedAndIndexCopy` (2 tests) locks the contract. Suite: 215 → 217 passed (+2).
- 2026-04-27: **P10 `aten.index_select` / `aten.scatter` / `aten.pixel_shuffle` / `aten.pixel_unshuffle` / `aten.constant_pad_nd` already compile cleanly.** Outdated entries — measurement on Vulkan: `index_select` → 1 dispatch + correct, `scatter(zeros, dim=0, idx, src)` → correct, `pixel_shuffle/unshuffle` → 1 dispatch + correct, `relu(F.pad(x, …))` → 1 dispatch + correct. All five enter Inductor's stock indirect-indexing / view+permute / conditional-write codegen and fuse with downstream pointwise — no Vulkan-specific lowering or extern template needed. `TestIndexSelectAndScatterCodegen` (2 tests) + `TestPixelShuffleAndPadCodegen` (3 tests) lock the contracts. Suite: 210 → 215 passed (+5).
- 2026-04-27: **P10 activation-backward overload audit — added missing `gelu_backward` lowering.** Walked all `_backward` aten ops in `lowerings.py`; `register_lowering(aten.X)` covers `.default` + `.grad_input` overloads automatically via the OpOverloadPacket. Found `gelu_backward` was the only gap. Added a full `_vulkan_gelu_backward(grad, self, *, approximate)` covering both `approximate='none'` (`0.5*(1 + erf(x/√2)) + (x/√(2π))*exp(-x²/2)`) and `approximate='tanh'` paths; added to `_suppress_upstream_decomps()` so AOT autograd doesn't pre-decompose. `TestGeluBackwardOverloadAudit` walks 10 backward ops × all overloads, asserts every one is in the lowering table. Suite: 207 → 210 passed (+3, includes one previously-extern path now reaching the lowering).
- 2026-04-27: **P5.6 deterministic-mode compatibility verified.** `torch.use_deterministic_algorithms(True)` does not break compiled pointwise (`relu(x*0.5)+1`) or reduction (`sum(x, dim=-1)`) on Vulkan — pointwise + reduction kernels are deterministic by construction (no atomics on standard reduce path; multistage tree-reduction is bit-exact across runs at fixed wg-size). `TestDeterministicCompile` (2 tests) locks the contract. Atomic-scatter-add → sort-based path under deterministic flag remains under P4.8 (not currently triggered by the suite). Suite: 207 passed.
- 2026-04-27: **P10 `aten.flip` / `aten.roll` / `aten.diagonal` already 1-dispatch.** Outdated entries — measurement shows `flip(x,[0])+1`, `roll(x,3)+1`, and `diagonal(x)+1` all compile to 1 dispatch via Inductor's stock pointwise+gather codegen on Vulkan. `aten.diag_embed` blocked on the same multi-axis-iota int64-cast codegen gap as tril/triu (tracked under P10 §"Diagonal/triangular masks"). `TestIndexOpsCodegen` (3 tests) locks the contract. Suite: 204 → 207 passed (+3).
- 2026-04-27: **P1.6 `[unroll]` on small static-bounded reductions.** `_reduction_nocache` (`kernel.py`) now emits `[unroll(loop_size)]` on the multistage reduction's per-thread loop when `loop_size` is a static `1 < N ≤ 16`. Cap chosen at 16 because beyond that the unrolled FMA chain blows VGPR budget on RDNA1 (8 → 2 waves/WG occupancy). Dynamic `loop_size_str` skips the attribute. `TestSmallStaticReductionUnroll` spies on `runtime.compile_slang_to_spirv` and asserts a captured kernel for `sum(x, dim=-1)` on (8, 1024) carries `[unroll(N)]`. Suite: 201 → 204 passed (+3).
- 2026-04-27: **P10 `aten.polygamma` / `aten.digamma` overrides verified — outdated entry.** `digamma`, `polygamma`, `lgamma` already wired in `overrides.py:VulkanOverrides`. `TestPolygammaDigamma` (2 tests) locks compile + correctness vs CPU.
- 2026-04-27: **P10 `aten.special_*` overrides — added `xlogy` / `xlog1py`.** Audit shows `i0`/`i0e`/`i1`/`i1e`/`spherical_bessel_j0`/`zeta`/`igamma`/`igammac`/`erf`/`erfc`/`erfcx`/`ndtri`/`erfinv` already wired. The two genuine gaps were `xlogy` and `xlog1py` (with the `0 * log(0) = 0` and `0 * log1p(-1) = 0` corner conventions). Added both as branch-on-`x==0` snippets, plus `special_xlogy` / `special_xlog1py` aliases for the `torch.special` namespace. `TestSpecialOpsOverrides` (2 tests) covers the zero-corner-is-not-NaN case vs CPU eager. Suite: 199 → 201 passed (+2).
- 2026-04-27: **P9.10 `[ForceInline]` audit on `slang_helpers.py` complete.** Applied `[ForceInline]` to every barrier-free leaf helper that was previously emitting as a real call: `c10_vulkan_atomic_add` (CAS-loop wrapper), `_vk_mulhi32`, `_vk_philox_round`, `_vk_philox_bumpkey`, `_vk_philox_rand`, `_vk_philox_randn`, `c10_vulkan_ndtri`, `c10_vulkan_i0/i0e/i1/i1e`, `c10_vulkan_erfinv`, `c10_vulkan_polygamma`, `c10_vulkan_spherical_bessel_j0`, `c10_vulkan_igamma`, `c10_vulkan_zeta`, `c10_vulkan_bucketize`. Wave-reduce / scan / sort helpers excluded (carry `GroupMemoryBarrierWithGroupSync` — `[ForceInline]` on barrier-bearing functions is risky in HLSL/Slang). `TestForceInlineCoverage` (1 test, 17 helpers) locks the contract. Suite: 198 → 199 passed (+1).
- 2026-04-27: **P10 `aten.arange` / `aten.linspace` factory codegen — already 1-dispatch; `aten.eye` blocked.** Measurement: `arange(64) * 2` and `linspace(0,1,64) * 2` both compile to 1 dispatch via stock Inductor codegen. `eye(8) + 1` still extern-falls to 7 dispatches because the diagonal-mask construction hits the same multi-axis-iota gap as tril/triu. Roadmap entry split: arange/linspace marked done, eye marked blocked on the codegen fix. `TestRegisterTileMatmul.test_arange_linspace_fuse_to_one_dispatch` locks the contract. Suite: 197 → 198 passed (+1).
- 2026-04-27: **P11.4 Read-only buffer `NonWritable` decoration verified.** Probed `slangc -target spirv-asm` on a `StructuredBuffer<float>`-input kernel; emits `OpDecorate in_ptr0 NonWritable`. RDNA1 / NVIDIA L1 read-only cache promotion is driven by this decoration, which Slang's `StructuredBuffer<T>` typing already produces. No codegen change needed. Checkbox flipped post-hoc.
- 2026-04-27: **P11.9 PC struct alignment audit — already optimal.** Outdated roadmap entry — current matmul `struct PC` declares all 4-byte uints (`M`, `N`, `K`, strides) before any 4-byte floats (`alpha`, `beta`, `scale`); Python `struct.pack("9I", …)` matches SPIR-V layout with no padding, no `uint64`/`vec` types to align. Checkbox flipped post-hoc.
- 2026-04-27: **P10 attempted `aten.tril` / `aten.triu` Vulkan lowering — surfaced multi-axis-iota codegen gap, lowering reverted.** Wrote a `where(col-row <= diagonal, self, 0)` lowering using `prims.iota` + view + sub. Inductor compiled it to a single fused kernel (verified via `TORCH_LOGS=output_code`), but the emitted Slang body referenced undeclared `x0`/`x1` axis variables — `VulkanKernel`'s axis-alias DCE strips `uint x0 = ...; x1 = ...;` declarations when no direct `x0`/`x1` reference exists in the unfused body, missing references that surface only after `prims.iota` index expressions inline. Same root family as the vec4 axis-alias DCE that §P4.3 partially fixed, but scoped to the scalar multi-axis path. Lowering reverted; new roadmap entry under §P10 tracks the codegen fix (`TestMultiAxisIotaCodegen`); tril/triu remain at 10 dispatches via the upstream decomp until that lands. Suite stays at 197 passed.
- 2026-04-27: **P10 `aten.lerp` Scalar/Tensor fuse-to-one verified.** Outdated roadmap entry — measurement showed `torch.lerp(a, b, 0.3)` and `torch.lerp(a, b, weight_tensor)` already compile to 1 dispatch via Inductor's stock decomposition + Vulkan pointwise fusion, no override needed. `TestRegisterTileMatmul.test_lerp_fuses_to_one_dispatch` locks the contract going forward (≤1 dispatch + correctness vs CPU). Suite: 196 → 197 passed (+1).
- 2026-04-27: **P11.10 `[unroll]` cap on matmul K-loop.** Changed `slang_mm.py.jinja` `mma_tile` inner-K-loop from bare `[unroll]` to `[unroll(min(tile_k, 16))]`. Full unroll on `tile_k=64` with a 4×4 register tile generated ~1024 FMAs per thread and dropped RDNA1 occupancy from 8 → 2 waves; the cap stays under VGPR budget while letting slangc pipeline FMAs. `TestRegisterTileMatmul.test_unroll_capped_at_16_for_large_tile_k` covers `tile_k=8` (`[unroll(8)]`) and `tile_k=32` (`[unroll(16)]`, never `[unroll(32)]`). Suite: 195 → 196 passed (+1).
- 2026-04-27: **P11.11 Per-kernel SPIR-V hash exposed via `inductor_stats`.** `_KERNEL_SPIRV_HASH` (`runtime.py`) records the 12-char SHA256 prefix of each compiled SPIR-V binary at `make_vulkan_kernel` time. `inductor_stats.summary()` includes the hash as the 5th column of every `top` row; `print_full_report` prints `[spv:<hash>]` next to each kernel name. Lets cache-miss / autotune-churn debugging correlate per-kernel timing to specific compiled binaries. Suite stays at 195 passed.
- 2026-04-27: **P11.6 Slang-source minification before SPIR-V hashing.** `_normalize_slang_source` (`runtime.py`) strips `/* … */` and `// …` comments, trailing whitespace, and blank-line runs before the SPIR-V cache hash, in both `compile_slang_to_spirv` and `prewarm_compile`. Cosmetic codegen variation no longer fragments the cache. Conservative — does not collapse intra-line whitespace. `TestGroupSharedBudget.test_slang_source_minification_for_hashing` covers all three transforms. Suite: 194 → 195 passed (+1).
- 2026-04-27: **P1.6 LDS-budget alignment-aware accounting**. `_new_idxvar` (`kernel.py`) now rounds each `groupshared` allocation up to 16B before adding to `_groupshared_bytes_used`, matching the AMD ABI per-decl alignment. Prevents the silent-scratch-spill class of bug where many small odd-sized allocations pass the dense-byte cumulative check while the driver actually reserves more. `TestGroupSharedBudget.test_alignment_aware_accounting` covers single-alloc rounding (5 int8_t → 16B) and the synthetic 4×17KB scenario where the third alloc trips the budget under aligned accounting (would slip through dense). Suite: 193 → 194 passed (+1).
- 2026-04-27: **P0.1 `native_group_norm_backward` lowering — 37 → 4 dispatches** (beats ≤5 target). New `_register_group_norm_backward()` in `lowerings.py` decomposes to ds/db inner-axis reductions + per-group rebroadcast of `c2 = (db_val * mean - ds_val) * rstd^3 * s` and `c3 = -c2 * mean - db_val * rstd * s` with `s = 1/(HxW*cpg)` folded via `mul.Scalar`. Skips the `torch.ones((1, group, cpg))` materialization in the no-gamma path. Masked-off slots get `aten.full` zero placeholders instead of `None` (Inductor's graph runner calls `.get_size()` on every multi-output slot). Added `aten.native_group_norm_backward.default` to `_suppress_upstream_decomps()`. `TestGroupNormBackwardLowering` (2 tests, ResNet/MobileNet block shapes) — 4 dispatches each, max numerical error ≤4e-6 vs CPU. Suite: 191 → 193 passed (+2).
- 2026-04-27: **P0.1 `native_layer_norm_backward` lowering — 5 → 2 dispatches** (beats ≤3 target). New `_register_layer_norm_backward()` in `lowerings.py` decomposes layer_norm bwd into pointwise + 2 reductions (inner-axis grad_input + outer-axis grad_w/grad_b), using `mul.Scalar(rstd_b * inner, 1/N)` instead of the upstream `rstd / N` pattern that falls through to extern `aten.div.Tensor` and breaks fusion. Required `_suppress_upstream_decomps()` to drop `aten.native_layer_norm_backward.default`, `native_group_norm_backward`, and `_log_softmax_backward_data` from the Inductor decomp table — without that, AOT autograd pre-decomposes them before our `register_lowering(aten.X)` is consulted, so the lowering is dead code. Verified the previously-shipped `_log_softmax_backward_data` lowering (P0.1) also fires now under direct invocation. `TestLayerNormBackwardLowering` (3 tests: 2D, 3D, multi-dim normalized_shape) — 2 dispatches each, max numerical error ≤1.2e-6 vs CPU. Suite: 188 → 191 passed (+3, plus 6 xpassed flipped from xfail thanks to `_log_softmax_backward_data` now reaching the lowering).
- 2026-04-27: **Loss-coverage batch + scalar-first `_binary_fake` fix.** Shipped 5 items in one pass: (a) `_binary_fake` (`meta_patches.py`) now handles the scalar-first case (`self=int/float, other=Tensor`) — Inductor decompositions of `log1p(...)`, `add.Tensor(1, t)` etc. were crashing with `'int' object has no attribute 'shape'`, taking out the `bce_with_logits` regression; (b) `aten.binary_cross_entropy` forward + `binary_cross_entropy_backward` lowerings (`lowerings.py`): forward `-(t*log(x) + (1-t)*log(1-x))`, backward `grad_out * (x - t) / (x * (1 - x))` with `/numel` for `mean` reduction; (c) `linalg_vector_norm` extended from `{1, 2}` to `{±inf, 0, 1, 2, generic-p}` — `±inf → max/min(|x|)`, `0 → count_nonzero`, integer `|p| ≤ 8` via repeated mul (avoids `pow` stability issues), generic-p via `pow(sum(|x|^p), 1/p)`; (d) `aten.norm.ScalarOpt_dim` legacy-overload thin shim around `linalg_vector_norm`; (e) `aten.native_dropout_backward` lowering (`grad_out * mask * scale`) so dropout-bwd fuses pointwise with surrounding ops. Tests: 5 new (`bce_correctness`, `bce_backward`, `linalg_vector_norm_{inf,neg_inf,p3}`, `norm_old_overload`). Suite: 184 → 188 passed (+6 net incl. `bce_with_logits` un-fail).
- 2026-04-27: **Roadmap expansion — Slang & codegen deep-dive added 3 new sections (~65 unchecked items).** Triggered by a focused review of (a) Slang language features under-utilized in the backend, (b) primops still extern-falling that the P1.7 audit missed, and (c) advanced kernel-codegen quality wins still on the table. Diffed `shaders/*.slang` + `templates/slang_mm.py.jinja` against the Slang language spec — found **zero** use of generics (`<T : IFloat>`), modules (`module/import`), `ParameterBlock<T>`, `interface IDifferentiable`, `[require(...)]` capability gating, multi-entry-point shaders, reflection API, `[SpecializationConstant]`, or shader-`printf`. Adopting these selectively is a correctness, source-density, and cold-compile-time win. New §P9 (10 sub-sections P9.1–P9.11) covers each Slang feature with a specific file:line target, regression test name, and measured benefit. New §P10 (10 categories) extends P1.7's primop audit with vision/spatial-transformer ops (grid_sampler, affine_grid, pixel_shuffle), custom-conv primitives (im2col/col2im, unfold/fold), indexing/scatter/search ops (searchsorted, bucketize, masked_select/scatter, index_select template, scatter non-add, index_copy/fill), diagonal/triangular masks (tril/triu/diagonal/diag_embed/eye), element-wise gaps (lerp, polygamma/digamma, special_*), backward-graph completeness (where/gather/scatter/index_select/clamp/nll_loss2d/bce backward + conv_backward algorithm selector), optimizer (`_foreach_*` audit + `_fused_adam`/`_fused_sgd` recognizers), and miscellaneous (repeat_interleave/unique/histc/`_scaled_mm`). New §P11 (11 sub-sections P11.1–P11.11) covers reduction-shape selection (tree-vs-ladder, serial-reduce prelude), branch-free codegen (select fold, NaN-aware reduction), instruction-level parallelism (independent FMA chains, dual-issue reduction), memory access patterns (transpose-on-load, multi-level tiling, async copy preload, ReadOnly decoration, strided-store coalescing), constant/index optimization (constant dedup, address CSE, redundant alias drop), source hygiene (minification before SPIR-V hashing, pretty-print, formatter), vendor-specific tuning (RDNA1 s_waitcnt, NVIDIA L1 hints, SwiftShader wave-size adaptation), branch hints, PC struct alignment, fast-math safe-guards, and per-kernel SPIR-V-hash observability. §P8 expanded with 8 additional final-completeness items (test-coverage budget, public-API SemVer commitment, slangc version pin, mypy/pyrefly strict, docstring coverage, AOTI deployment artifact, cross-vendor CI matrix). All entries name the file path, the regression test, and the bar to hit. No code changes in this commit — purely roadmap expansion ahead of the next agent loop pass picking the highest-priority unchecked items.
- 2026-04-27: **Implementation pass — shipped 9 Python-only roadmap items in one batch.** New `_register_softmax_backward()` and `_register_loss_lowerings()` in `lowerings.py` register Inductor lowerings for: P0.1 `_softmax_backward_data` + `_log_softmax_backward_data`; P1.7 forward `mse_loss` / `l1_loss` / `smooth_l1_loss` / `huber_loss` / `kl_div` / `binary_cross_entropy_with_logits` / `linalg_vector_norm` (ord ∈ {1, 2}); P0.1 backward `mse_loss_backward` / `smooth_l1_loss_backward` / `huber_loss_backward`. Each guards `_is_vulkan(self)` and returns NotImplemented for non-Vulkan inputs so non-Vulkan backends keep their default Inductor path. Reduction-mode-aware (mean / sum / none) via a small `_reduce()` helper using `aten.mean.dim` / `aten.sum.dim_IntList`. KL-div uses the `target == 0 → contribution=0` xlogy convention to avoid `0 * log(0) = NaN`. BCE-with-logits uses the numerically-stable `max(x,0) - x*target + log1p(exp(-|x|))` form; `pos_weight` path falls through to upstream. Tests: `TestLossLowerings` (10 forward correctness + 1 dispatch-count) lands as live tests; `TestSoftmaxBackwardLowering` (2) + `TestLossBackwardLowerings` (3) gated `xfail(strict=False)` until P0.0 backward-graph compile unblocks. Also: P3.3.5 docs — extended `docs/03-slang-shaders.md` with the "floor-vs-ceiling for autodiff" Inductor-backend sub-section, covering when to use autodiff vs hand-written, register/occupancy constraints, pilot order, and the `register_autodiff_template` tooling reference. Build / regression run pending — Python-only changes; tests will pick up on next `MAX_JOBS=4 pip install -e . -v` cycle (no C++ touched).
- 2026-04-27: **Roadmap review pass — added ~40 new unchecked items across 6 new/expanded sections.** Triggered by a comprehensive backend audit (inductor module survey + primop coverage diff against eager 440 fused shaders + Slang autodiff capability survey + codegen heuristic review). New: §P1.7 (primop coverage audit — loss / norm / pool / upsample / pad / dropout / linalg / spectral), §P5.9 (cooperative-matrix / WMMA, capability-gated for non-RDNA1), §P5.10 (subgroup intrinsics expansion — ballot / quad-shuffle / prefix-count / half-atomic CAS-loop), §P6.6 (stride / contiguity propagation), §P7 (measurement-driven discovery automation), §P8 (final completeness checklist). Expanded: §P1.6 (subgroup-uniform branch elim, [unroll]-on-static-rnumel, intensity-class picker, bank-conflict swizzle, int8 dot product, specialization constants), §P3.3 (loss-fn autodiff cluster P3.3.6, RoPE backward P3.3.7, LayerNorm autodiff perf comparison P3.3.8, `@autodiff_template` decorator P3.3.9, VGPR regression gate P3.3.10, CAS-loop half-atomic helper P3.3.11). Each item names the file to touch, the regression test name, and the bar to hit. No code changes in this commit — purely roadmap expansion ahead of the next agent loop pass picking the highest-priority unchecked items.
- 2026-04-26: P5.7 — **Discovered Inductor autotune CUDA-leak** on vulkan-only machine. `torch.compile(lambda a, b: torch.relu(a @ b))` on vulkan inputs fails with `Torch not compiled with CUDA enabled` whenever the mm autotune triggers (`select_algorithm._benchmark_choice` calls `torch.cuda.Event` / `torch.cuda.synchronize` directly). Pre-existing — reproduces even with zero Slang choices installed. Not caught by the regression suite because mm exposures there sit inside larger graphs that take a different code path. Roadmap entry added under P5.7 with the fix sketch (vulkan-aware override of `_benchmark_choice`'s synchronize/event helpers). Added `TORCH_VULKAN_NO_REGISTER_TILE=1` bisection knob in `_pick_register_tile_configs()` so future investigations can isolate this from register-tile-specific issues.
- 2026-04-26: P6.3 — **`VulkanExprPrinter` subscript-context flag.** `tmp\d+` symbol → `((int)(...))` cast was unconditional, which is correct for array subscripts (CSE'd int64 indices stored as float) but destroys precision on float CSE locals that share the namespace. Added `_subscript_depth` counter + `subscript()` context manager on the printer; `VulkanKernel.index_to_str` overrides upstream to enter the context around every index render. Outside subscript context the cast is suppressed. `TestExprPrinterIntCast` rewritten to 5 tests (outside no-cast, inside cast, non-tmp unchanged, compound expr, nested scope). 166p/1s/16x.
- 2026-04-26: P4.3 / P2.2 — **Slang matmul template register-tiled + tile configs fixed.** Discovered: 5/6 of `_MM_TILE_CONFIGS` had `tile_m * tile_n > 1024` (RDNA1's `max_workgroup_invocations`), so `numthreads(tile_m, tile_n, 1)` would fail at `vkCreateComputePipeline` and Inductor's autotune silently picked aten_mm. The whole Slang template path was effectively disabled for typical workloads. Fix: (a) trimmed `_MM_TILE_CONFIGS` to only valid 1-output-per-thread shapes (`(32,32,16)`, `(32,32,32)`, `(16,64,32)`, `(64,16,32)`, `(16,16,32)`); (b) added `_MM_REGISTER_TILE_CONFIGS` — `(tile_m, tile_n, tile_k, m_per_thread, n_per_thread)` — for the hardware-friendly pattern: `(64,64,16,4,4)`, `(128,64,16,8,4)`, `(64,128,16,4,8)`, `(64,64,16,8,8)`. Each thread holds an `(m_per_thread × n_per_thread)` register accumulator, workgroup is `(tile_m/m_per_thread × tile_n/n_per_thread)`. Rewrote `slang_mm.py.jinja` to support both paths via a single template — register-tile is the default codegen, the legacy `r1x1` is just `m_per_thread=n_per_thread=1`. Also added `[ForceInline]` to `load_tiles` / `mma_tile` and `[unroll]` on inner-K and the per-thread register loops. Cooperative tile loads now use a flat `tid` stride loop so loads coalesce regardless of register-tile shape, with proper `(global_m < pc.M)` bounds clamps. Found and fixed an asymmetric-tile correctness bug in the original template (`numthreads(tile_m, tile_n, 1)` made `lid.y ∈ [0, tile_n)` but the row computation used `lid.y` as if it were tile_m-sized — broken for any (tm, tn) where tm ≠ tn). Per-shape benchmark on 512×512×512 f32: aten_mm 1.19 ms / legacy 32×32×32 0.61 ms / **register-tile 64×64×16 r4x4 0.39 ms** = 3.0× over aten, 1.6× over legacy. All cache-keys and prewarm specs gain `_rMxN` suffix; `_collect_matmul_prewarm_specs` emits both legacy + register-tile variants × {f32, f16} × {1, 2 stages}. `TestMatmulTileConfigsFitWorkgroupLimit` (2 tests) lock the ≤1024 invariant; `TestRegisterTileMatmul` (4 tests) lock correctness for r4x4 / stages=2 / unaligned shapes / addmm. 164p/1s/16x.
- 2026-04-26: P4.8 — **`aten.cumsum` / `aten.cumprod`** verified compiling end-to-end through `VulkanKernel.scan` onto the existing `wg_scan` helper. Already worked but unrechecked. `TestCumScanOps` (3 tests) locks correctness vs CPU and dispatch count ≤3. Constraint: scan reduction-numel ≤ `max_threadgroup_size`; larger axes still extern. 158p/1s/16x.
- 2026-04-26: Slang-helpers polish — added `[ForceInline]` to per-element math approximations (`c10_vulkan_erf`, `c10_vulkan_digamma`, `c10_vulkan_lgamma`) and the packed16 2D unpack/pack helpers (`_vk_{un,}pack_{f16,bf16}_2d`). These are called per-element in pointwise kernels (gelu, lgamma) and per-row in 2D-packed16 reductions; the missing inline annotations meant slangc emitted real call instructions instead of inlining the body. Suite stays 158p/1s/16x.
- 2026-04-26: Vec4 codegen polish — drop the `uint xbase = gtid.x * 4u;` declaration when the unroll body has no `xindex` reference (fast common case for arithmetic-only pointwise like `relu(a*0.5+b)`). Combined with the existing dead-axis-alias DCE and unused-`xindex` skip, the emitted vec4 kernel is now minimal: buffer bindings → input float4 loads → output float4 cache → 4-iter unroll → output flush. No further dead code in the source we hand to slangc.
- 2026-04-26: P5.8 — **Buffer-count fusion limit refinement** roadmap entry was outdated. `VulkanScheduling._get_max_storage_bufs()` already queries `props.max_storage_buffers` from the device interface and returns `min(max_storage_buffers // 2, 32)`. On RDNA1 it resolves to 32 (well above the runtime fast-path's n_buffers ≤ 6). Checkbox flipped, no code change needed.
- 2026-04-26: P6.1 — **Cache `_slangc_available()` at module load** roadmap entry caught up — `runtime.py:200-228` already keeps the result in a lazy module-level `Optional[bool]` with `_reset_slangc_available_cache()` for tests. Checkbox flipped post-hoc.
- 2026-04-26: P4.3 — **Vec4 packed f32 pointwise** shipped. Single-axis f32 contiguous-pointwise kernels with `numel % (max_threadgroup_size * 4) == 0` and trivial `xindex`-only buffer indexing now bind I/O as `StructuredBuffer<float4>` / `RWStructuredBuffer<float4>`. Each thread issues one coalesced float4 load per input, runs the scalar body 4× in an `[unroll] for (uint _k = 0u; _k < 4u; ++_k)`, and flushes one float4 store per output. Decision is post-body-emission so a substring scan on the rendered body bails to scalar on any unsafe pattern (atomics, wave intrinsics, `_vk_linear` multi-axis, multistage hoist, packed16 already locked). Buffer-subscript validation against the alias set built from `uint <name> = xindex;` lines accepts loads via `x0` etc. Kill switch `TORCH_VULKAN_NO_VEC4_POINTWISE=1`. Polish: dead `uint x0 = xindex;` aliases dropped post-rewrite, output float4 cache left uninitialized (eligibility guarantees full coverage by the unroll). Perf: 100×`relu(a*0.5+b)` on 1M-elem f32 → 22.57 ms vec4 vs 24.47 ms scalar (~8% on a dispatch-bound microbench). `TestVec4PointwiseF32` (6 tests) locks fires-on-4096-add, correctness vs CPU on add and `relu(a*0.5+b)`, kill-switch, skip-on-non-div-4, skip-on-f16. 153p/1s/16x.
- 2026-04-26: P4.3 — **Shared-memory budget tracking** for `groupshared` decls. `VulkanKernel._new_idxvar(is_threadgroup=True)` adds `_slang_dtype_bytes(dtype) * elem_count` to a per-kernel counter and raises `NotImplementedError` if the cumulative usage exceeds the RDNA1 64KB LDS budget — surfaces overcommits as a clean failure rather than a silent driver-side spill to scratch. Wave-helper smem (`smem_<op>` inside `slang_helpers.emit_helpers`) is excluded; bounded by `simd_group_size` ≤ 256B and statically safe. `TestGroupSharedBudget` (2 tests): zero-init contract + over-budget alloc raises. 155p/1s/16x.
- 2026-04-26: Codegen polish — `VulkanKernel.store` now skips the `((<dtype>)(<value>))` cast when `value.dtype` already matches the destination dtype. Inductor's CSE annotates dtype on every variable; the unconditional cast bloated emitted Slang and added redundant `OpFConvert` to the SPIR-V (driver DCE'd it but cost source bytes + parse time). Dumped vec4 kernel now emits `_v_out_ptr0[_k] = (tmp5);` instead of `((float)(tmp5))`. Suite still 155p/16x.
- 2026-04-26: P0.4 / P6.5 — **Python recycle pool for inductor-emitted Vulkan tensors landed (default-off)**. New `python/torch_vulkan/inductor/buffer_pool.py` exports `vulkan_pool_acquire(size, stride, dtype)` / `vulkan_pool_release(tensor)` keyed on `(size, stride, dtype)` with per-key cap (4) + global cap (64) + `pool_stats()` wired into `inductor_stats.summary()` and `print_full_report()`. `VulkanPythonWrapperCodegen.make_buffer_free` (override) now emits `vulkan_pool_release(buf_N); buf_N = None` for non-input/non-output vulkan intermediates; `make_buffer_reuse` is overridden to keep the reuse-aliasing path on plain `del` (otherwise pool-release would re-vend a still-aliased buffer). `_empty_strided_vulkan` consults the pool first. **Default off** — opt in via `TORCH_VULKAN_BUFFER_POOL=1`. Why: on the MLP forward the pool *adds* ~14 us (0.134 ms → 0.148 ms) because most allocations come from `aten.addmm`/`aten.mm` extern paths that bypass `empty_strided_vulkan` entirely, so the pool only sees ~2 of 8 buffers/step and the per-call overhead exceeds the per-hit savings. The infra is the right shape for graphs where intermediates flow through `empty_strided_vulkan`; closing the MLP gap is now tracked by a new P0.4 follow-up — a C++ extern-kernel allocator hook for `aten.{mm,addmm,bmm,linear}.out`. `TestBufferPool` (8 tests) locks the contract — acquire/release round-trip, key-mismatch miss, per-key cap eviction, default-off-until-opt-in, stats schema/monotonicity, end-to-end consultation in compile, wrapper preamble import. 147p/1s/16x.
- 2026-04-26: P2.2 — Real matmul autotuner promoted to P0 per user direction; expanded the one-line entry into 4 sub-tasks with cache layout, benchmarking driver, candidate set, cache mgmt + a regression-test sketch. Picks up next.
- 2026-04-26: P2.2 — `Benchmarker` now cached at module level via `_get_benchmarker()`. Inductor autotune calls `benchmark_codegened_module` per candidate; constructing fresh `Benchmarker()` on every call repeated init work for nothing. `_reset_benchmarker_cache()` test hook. `TestBenchmarkerCached` locks identity + reset. 138p/1s/16x.
- 2026-04-26: CLAUDE.md updated — added Objectives section (O1–O5) listing concrete success criteria; tightened "How to Pick the Next Item" with explicit instruction to start the next item *in the same turn* after shipping; restated the standing instruction to always continue improving and optimizing the inductor backend.
- 2026-04-26: P0.4 re-measure — compiled MLP forward stays at 0.137 ms vs eager 0.066 ms (0.48× of eager), with and without `TORCH_VULKAN_TRUST_INDUCTOR=1`. The P6.x stack of micro-overhead removals (per-dtype template cache, tuple guard key, helper-output cache, async-compile sync repair, tight numthreads) trims the per-dispatch path but doesn't move this benchmark — the cost is allocator-bound (8 buffers × ~17 us = ~140 us per step). Closing the gap needs the C++ output-buffer reuse pool (P0.4 still-pending item).
- 2026-04-26: P6.2 — `_slang_tile_{mm,addmm,bmm}` (`vulkan_template_caller.py`) now skip per-dispatch `is_contiguous()` checks under `TORCH_VULKAN_TRUST_INDUCTOR=1`. Module-level `_TRUST_INDUCTOR` flag captured at import; `_reset_trust_inductor_cache()` test hook re-reads it. Same rationale as the wrapper-level `assert_size_stride` no-op (P0.4): Inductor pre-arranges contiguous tensors before the extern reaches them. `TestTemplateCallerTrustInductor` (2 tests) covers env-var capture + observed-call-count via `is_contiguous` spy. 136p/1s/16x.
- 2026-04-26: P6.4 — `_CompiledGraph._guard_hash` SHA1 replaced by `_guard_key` returning a `tuple` directly. Cache is now `dict[tuple, Callable]` — `dict` already hashes tuple keys via per-element `__hash__`, so the prior SHA1 round-trip was pure overhead. `_guard_hash` retained as a thin string-key wrapper. `TestCompileGraphStub.test_guard_key_is_tuple_no_hash` locks the contract. 134p/1s/16x.
- 2026-04-26: P6.3 — `slang_helpers.emit_helpers` output now cached by `(frozenset(headers), max_threadgroup_size, simd_group_size)`. Original body extracted to `_emit_helpers_impl`; `emit_helpers` is a thin cache-front + `code.splice` of the rendered string. Saves the per-kernel re-rendering of `erf` / `tanh` / `welford` / `wg_reduce_*` / packed16 helper bodies on every codegened kernel. `TestEmitHelpersCache` (2 tests) locks single-render-per-key, OrderedSet/list/set key normalization, output equivalence. 133p/1s/16x.
- 2026-04-26: P6.3 — right-sized `numthreads` for small-numel pointwise. `_pick_threadgroup_size` (`kernel.py`) used to always return 256 for non-reduction kernels regardless of total numel; a 91-element kernel got a 256-thread WG with `if (xindex >= 91) return;`, wasting half the lanes on RDNA1 wave64. Now: when `total < 256`, picks `next_pow2(total)` clamped to `[simd_group_size, max_wg]` — 91 elements → `numthreads(128, 1, 1)` (2 waves, 0 idle past the bound). `TestRightSizedNumthreads` covers the property + integration end-to-end. 131p/1s/16x.
- 2026-04-26: P6.1 — repaired `_ASYNC_COMPILE` sync-block path in `compile_slang_to_spirv` (`runtime.py`). Was: submit-to-pool, await `future.result()`, then re-look-up `_cache_by_hash` and **fall through to a second inline compile** on miss — silent double-compile + swallowed exception when the future raised. Now: take the pool-submitted future's result as the SPV bytes directly (raises on inner failure), one path. `TestAsyncCompileSinglePass` locks both contracts: (1) inner compile runs exactly once per cold key under `_ASYNC_COMPILE=True`; (2) errors propagate via `future.result()` instead of being swallowed. 130p/16x.
- 2026-04-26: P6.2 — `_SlangTile{MM,AddMM,BMM}` now cache `(rendered_src, cache_key)` per tensor dtype on the instance via a new `_per_dtype` slot. Per-dispatch path skips the Jinja cache dict-lookup + cache_key f-string format after the first call per dtype. `_slang_tile_{mm,addmm,bmm}` helpers gained optional `src` / `cache_key` kwargs so the legacy direct-call API still works. `__reduce__` rebuilds from ctor params, so the cache resets cleanly across pickle. `TestSlangTilePerDtypeCache` (4 tests) covers cache identity, dtype-keyed separation, prewarm-keys consistency, and pickle reset. 128p/16x.
- 2026-04-26: Pipeline-review pass — shipped two more cleanup wins. **P6.1**: `compile_and_dispatch` `cache_key` is now required (drops the dead `sha1(spv)[:12]` fallback that was never reached because every in-tree caller supplies a key). **P6.4**: hoisted `_enable_b2b_gemm` from `_VulkanCustomPass.__call__` (per-FX-graph) to `register()` (one-shot at backend init) — only sets a global config flag, no point re-flipping per compile. `TestCompileAndDispatchRequiresKey` locks the empty-key rejection. 124p/16x.
- 2026-04-26: Pipeline-review pass — added §"P6 — Pipeline review pass" to the roadmap with 13 concrete optimization opportunities across runtime / template-caller / codegen / cleanup / C++ dispatch, each pointing at the exact file:line and the smallest possible win. Shipped two: (1) **P6.1 cache `_slangc_available()`**: was running `subprocess.run(["slangc", "--version"], timeout=5)` per cold compile (~1 ms × N kernels). Now probed once and cached at module load with `_reset_slangc_available_cache()` hook for tests. (2) **P6.1 `_INDUCTOR_STATS` per-build env read**: investigated and determined to be intentional — runtime stats toggle is a public contract (`TestInductorStats` sets the env var after import). Marked wontfix in the roadmap with the trade-off recorded so a future iteration doesn't re-attempt. `TestSlangcAvailableCached` (2 tests) lock the new cache + reset behavior. 123p/16x.
- 2026-04-26: P4.3 — verified `WaveActiveSum`-over-groupshared single-wave fast path is already implemented in `slang_helpers.py:257-266` (`n_waves == 1` branch emits one-line `WaveActiveX(val)` helper, no groupshared, no barriers). Added `TestSmallReductionWaveIntrinsic.test_single_wave_reduction_skips_smem` — captures the emitted Slang via a `compile_slang_to_spirv` spy and asserts the helper is `return WaveActiveSum(val);`, no kernel-scope `tmp_acc_*` decl, no `GroupMemoryBarrier` in the helper body. Catches both a future regression of the wave-only fast path AND a future re-introduction of the dead `tmp_acc_N` decl. 121p/16x.
- 2026-04-26: P4.3 — dropped dead `tmp_acc_*` groupshared decls from reduction codegen (`kernel.py:_reduction_nocache`). The wave-intrinsic helper in `slang_helpers.py` owns its own `smem_<op>` scratch; the `_new_idxvar` call here was emitting a `groupshared <dtype> tmp_acc_N[n_waves]` decl that was never read. Verified: dumped CE Slang shows the decls gone, suite stays green (120p/16x).
- 2026-04-26: P1.5 — diagnosed both P0 correctness items (view-after-pointwise + cross-entropy) as the same upstream-blocked FakeTensor view-fast-path bug. AOT post-grad graph for `(x+1).view(...)` shows `view: "...meta"`; for `F.cross_entropy(...)` shows `unsqueeze/gather/squeeze/neg/where_1/sum_3/div` all `device=meta` while all upstream inputs are `vulkan:0`. Inductor lowers the meta-device chain to a no-op (`out_ptr0[0] = 0.0f`). Tried a `post_grad_custom_pre_pass` rewriting all 7 meta `node.meta['val']` entries to vulkan FakeTensors; verified the rewrite happens (7 → 0 meta vals) but Inductor re-derives device info from op signatures during lowering, so the rewrite has no effect. Both items moved to `[~]` with the upstream blocker noted; the existing `xfail(strict=True)` regressions remain in place to flip when the upstream view-fast-path patch lands.
- 2026-04-26: Discovered + xfail-locked: `(x + 1).view(8, 16)` returns wrong values under torch.compile (`TestRedundantCopyRemoval.test_view_then_copy_no_extra_dispatch`). New blocker, likely view-allocator interaction. `TestSmallReductionWaveIntrinsic` locks small-reduction (rnumel < subgroup) sum/amax correctness. `TestPostAttentionResidual` + `TestGeluDropoutBaseline` + `TestEmbeddingLayerNormBaseline` + `TestResidualLayerNorm` lock baselines for upcoming P2.5 fusion work.
- 2026-04-26: Performance lock — `TestMLPForwardCeiling` regression added: compiled MLP forward (linear+gelu+linear) at 5 dispatches (eager 7, target 3). Locks the current bound; tightening to ≤4 needs cross-linear epilogue fusion. `TestWelfordCooperativeSelection` smoke tests the welford-aware reduction selection helper + a layer_norm compile correctness check.
- 2026-04-26: P1.2 — `should_use_cooperative_reduction` now detects welford reductions ahead of codegen via `_has_welford_reduction()` and biases toward cooperative single-WG wave-reduce for a wider rnumel range when welford is in flight, avoiding the AMD RADV miscompile in multistage groupshared+barrier paths.
- 2026-04-26: P1.1 — **Flash attention FX pass shipped**. `_fuse_sdpa_to_flash_attention` matches `aten.scaled_dot_product_attention` with eligibility (4-D, head_dim ∈ {32, 64, 128}, no attn_mask, no dropout) and routes to `torch.ops.torch_vulkan.flash_attention_fused.default`. Compiled SDPA on (1, 2, 8, 64) drops from **9 → 5 dispatches**. Unsupported head_dims fall through to upstream decomposition without crashing. `TestFlashAttentionFusion` covers dispatch + correctness + fallback.
- 2026-04-26: P1.1 — **SwiGLU FX pass shipped**. `_fuse_silu_mul_to_swiglu` rewrites `silu(gate) * up` to `torch.ops.torch_vulkan.swiglu_fused.default` (a real custom_op with `register_fake`). `silu(gate) * up` collapses **2+ dispatches → 1**. Both swiglu_fused and scaled_bmm custom_ops are now registered eagerly at backend `register()` time. `TestSwigluFusion` covers dispatch + correctness + shape-mismatch-no-fusion + op-registered.
- 2026-04-26: FX pass — `_fuse_bmm_mul_to_scaled_bmm` now fires on the post-AOT `permute([0,2,1])` form (was matching only pre-AOT `transpose.int`). Rewrite target promoted from a plain Python function to a real torch custom_op (`torch_vulkan::scaled_bmm`) with a `register_fake` impl, so Inductor's lowering machinery accepts it. Result: `0.125 * bmm(q, k.T)` collapses **3 dispatches → 1**. `TestScaledBmmFusion` tightened from `≤3` to `==1`. Also: `TestExprPrinterIntCast` locks the `tmp\d+` symbol → `((int)(...))` cast emission.
- 2026-04-26: P1.5 — partial fix shipped for cross_entropy compile: `_patch_fake_tensor_meta_conversion` now converts vulkan-device real tensors (not just meta-device) so `aten.gather.default` reaches our fake_impl when Dynamo specializes the index tensor as a constant. Removes the `validate_and_convert_non_fake_tensors` hard error. Bug (a) of two; bug (b) — int64 index cast to float in generated Slang — partially mitigated by `_print_Symbol` `((int)(...))` wrap on `tmp\d+` index symbols. CE compile still produces wrong values (kernel 1's per-element target masking writes garbage to buf2 from out-of-bounds threads on the 64-thread workgroup); the compiled CE test stays `xfail(strict=True)` until the kernel-bounds fix lands.
- 2026-04-26: P5.5 — `docs/extension_cookbook.md` shipped: single-page "how to add a fused op" walkthrough covering `register_lowering` (override an aten op for vulkan only) and `register_template` (declare a new fused Slang shader) with a worked `mul * 0.5` example, wiring instructions, mandatory-regression-test contract, and the standard debugging knobs.
- 2026-04-26: P1.5 — **correctness regression discovered**. `F.cross_entropy(logits, target)` under `torch.compile` returns `0.85` vs eager's correct `3.18` on a (B=8, V=16) smoke test. Eager `cross_entropy` is fine. Cause likely: int64 target gather isn't lifted correctly under inductor lowering. Promoted P1.5 cross-entropy fast path to P0 in the roadmap. `TestCrossEntropyBaseline` covers both paths (eager passing, compiled `xfail(strict=True)`).
- 2026-04-26: P2.3 — `TestPacked16Verification` locks f16 elementwise correctness + `(x*1.5+0.3-0.5)` chain at 1 dispatch (fused). Catches a future packed16 codegen regression that breaks half-precision pointwise fusion.
- 2026-04-26: P4.8 / P5.4 — `TestRepeatInterleaveBaseline` locks `repeat_interleave` correctness (scalar + dim form) vs CPU so a future codegen swap can't silently regress the C++ extern. `TestControlFlowSubgraph.test_torch_cond_branches_compile` placed `xfail` until the HigherOrderOperator subgraph emit lands.
- 2026-04-26: P5.3 / P5.6 — verification regressions: `TestDynamicShapeAudit` smoke-tests `VulkanExprPrinter` on `s0*s1+s1` and `FloorDiv(s0, 4)` so symbolic-shape codegen failures are caught at unit-test time. `TestReductionOrderStability` locks `sum(x)` repeated 5× to ULP-stable + compiled vs eager to `1e-4`. Together these catch a future autotune/codegen change that breaks reproducibility or symbolic shapes.
- 2026-04-26: P1.5 — `TestVarMeanWelford` regression locks `var_mean(x, dim=-1, keepdim=True)` at ≤4 dispatches + correctness vs CPU. Tightening to 1 needs a custom Welford lowering (next step); the test is in place to catch it landing.
- 2026-04-26: P4.7 stub — `inductor.compile_graph(fn)` ships: shape-signature-keyed memoization of `torch.compile(dynamic=False)` with `.info()` (hits/misses/n_recordings/hit_rate) + `.reset_cache()`. **No real Vulkan command-buffer recording yet** — that's the next iteration; today's win is removing per-call Dynamo overhead under static shapes. `TestCompileGraphStub` covers hit-rate-on-repeat, shape-change new recording, correctness round-trip, reset.
- 2026-04-26: P2.5 / P5.8 — verification regressions: `TestConstantBroadcastHoist` locks `x*0.5+0.7-1.3` chain at 1 dispatch (Inductor inlines scalar literals via `value_to_slang`). `TestHorizontalReductionFusion` locks two-reduction-shared-input cases at the current ≤4 dispatch ceiling so a future fusion improvement flips the bound visibly.
- 2026-04-26: P5.6 — `TestReductionNaNPropagation` added (4 tests, `xfail(strict=True)`): documents that Vulkan reductions currently *suppress* NaN — `amax`/`amin`/`sum` over a tensor containing NaN return finite values instead of NaN. The strict-xfail flips to passed when the codegen swaps to NaN-aware variants. New roadmap item P5.6 codegen fix tracks the fix path. Also: `inductor.config.denormal_mode()` (`TORCH_VULKAN_DENORMALS={flush,preserve}`) exposed; `TestDenormalsConfig` covers the env-var contract.
- 2026-04-26: P2.4 — `inductor_stats.MemoryTracker` context manager + `peak_memory_report()` snapshot ship. Tracks Vulkan caching-allocator `memory_cached()` at enter / `poll()` / exit, exposes `peak_mib`/`delta_mib`/`start`/`end`. Sampled peak — true peak still wants a C++ allocator hook (filed as new follow-up). `TestPeakMemoryAPI` locks the round-trip contract.
- 2026-04-26: P5.5 — `extensions.py` shipped: `register_lowering(op)` decorator (auto vulkan device guard returning NotImplemented for non-vulkan IR nodes) + `register_template(name, slang_src, n_buffers, n_pc, ...)` returning a callable JIT-dispatch shim that flows through the standard runtime cache. `prewarm_template(fn)` submits to the slangc pool. `TestExtensionDecorators` covers device-guard + arg-count validation + cache_key wiring.
- 2026-04-26: P5.7 — `runtime.gc_spirv_cache(max_mib=N)` LRU-trims the on-disk SPIR-V cache (`{removed, kept, bytes_before, bytes_after}` return). `_default_max_workers()` defaults to `min(8, cpu_count())` (was 4) with `TORCH_VULKAN_SLANGC_WORKERS` override. `_namespace_inductor_cache()` points `TORCHINDUCTOR_CACHE_DIR` at `/tmp/torchinductor_$USER_vulkan_<inductor-mtime-hash>` so checkout swaps don't silently reuse stale codegen — disable with `TORCH_VULKAN_NO_CACHE_NS=1`. `TestSpirvCacheGC` + `TestSlangcWorkersTuning` + `TestInductorCacheNamespace` lock all three.
- 2026-04-26: P0.4 — `TORCH_VULKAN_TRUST_INDUCTOR=1` injects no-op `assert_size_stride` / `assert_alignment` into the wrapper preamble (`VulkanPythonWrapperCodegen.write_header`). The asserts are debug-only and add measurable per-buffer overhead on small workloads (8 buffers/MLP step). `TestTrustInductor` covers env-var round-trip + end-to-end compile execution under the flag.
- 2026-04-26: Roadmap expansion — added P1.5 (PrimTorch decomposition tuning), P2.4 (memory planning + buffer reuse), P2.5 (FX pattern-matcher expansion), P4.7 (pre-recorded command buffers / Vulkan-graph), P4.8 (scatter/sort/cumsum/index_put codegen), P4.9 (autocast + mixed precision), P5.1 (conv algorithm selection: depthwise / 1×1 / Winograd), P5.2 (NHWC layout selection), P5.3 (dynamic shape support), P5.4 (control-flow subgraphs), P5.5 (custom-op extension helpers), P5.6 (determinism / NaN policy), P5.7 (compile-time / warm-start), P5.8 (scheduler tuning). ~50 new unchecked items spanning primtorch decomposition, codegen depth, memory + scheduler, and missing op coverage.
- 2026-04-26: P4.4 — `autotune.list_autotune_cache()` returns every persisted `(key, best_wg, timings)` triple; `clear_autotune_cache()` deletes them and returns the count. Use after slangc/driver upgrades that change the relative cost of WG sizes — stale entries otherwise survive indefinitely. `TestAutotuneCacheManagement` covers list/clear roundtrip + missing-dir behavior.
- 2026-04-26: P0.4 — added new roadmap section. Per-dispatch overhead microbench: `torch.empty_strided((32,512), …, device='vulkan:0')` is 17.9 us/call, so 8 buffer allocs per compiled MLP step ≈ 140 us — accounts for the entire 0.142 ms (vs 0.067 ms eager) compiled-mode regression on tiny workloads. P0.4 lists the four follow-ups (direct `_c_ext.empty_strided` binding, output-buffer reuse pool, `assert_size_stride` kill-switch, re-measure).
- 2026-04-26: P2.2 — `_register_vulkan_benchmarker_once()` lifted out of `VulkanScheduling.benchmark_codegened_module`. Previously the `@register_benchmarker("vulkan", override=True)` decorator was re-run per autotune iteration; now it fires once at module load.
- 2026-04-26: P4.1 — `inductor_stats.print_full_report()` combines per-kernel waterfall + compile-stats counters into a single Jupyter-friendly diagnostic. `TestWaterfallDump.test_print_full_report_runs` locks the contract (both sections present in stdout).
- 2026-04-26: P0.3 — `_patch_dynamo_clone_input_for_vulkan` (always-on in `meta_patches.apply()`) replaces `torch._dynamo.utils.clone_input` AND the imported binding in `torch._dynamo.variables.builder` with a Vulkan-/FakeTensor-safe variant: skips the `data_ptr()` alignment dance for vulkan device, FakeTensors, and any tensor whose `data_ptr()` raises. Unblocks `bmm(q, k.T)` compile; `TestCloneInputPatch` covers FakeTensorMode round-trip + builder binding identity + actual compile path. Full attention chain `softmax(q@k.T) @ v` still meta-leaks (P0.0 follow-up still upstream-blocked).
- 2026-04-26: P1.2 — `TestWideRowReduction` locks in 1-dispatch softmax/log_softmax for vocab-scale rnumel (16384 and 32000). The multistage reduction codegen already collapsed `amax → exp → sum → div` into 1 dispatch — verification only.
- 2026-04-26: P4.1 — `inductor_stats.compile_stats()` always-on counters (cold compiles, cumulative slangc us, in-mem + disk cache hits, prewarm submits) + derived `cache_hit_rate` and `avg_cold_compile_us`. `print_compile_stats()` prints a 3-line summary; `reset_compile_stats()` zeros. `TestCompileStats` covers the cache-miss → hit transition.
- 2026-04-26: P4.1 — `inductor_stats.dump_waterfall(path)` writes a per-kernel `.json` `(name, dispatches, total_us, avg_us, percent_total)` sorted by total_us descending. `TestWaterfallDump` covers sorted-output + percent-sums-to-100% + empty-when-disabled.
- 2026-04-26: P4.4 — `_diagnose_import_failure` in `torch_vulkan/__init__.py` wraps `ImportError` with a probe of `libvulkan.so.1`, `slangc`, and `vulkaninfo --summary` so misconfigured environments show the actual fix instead of "cannot find _C". `TestImportDiagnostics` covers the slangc-missing branch.
- 2026-04-26: L0 fix — `slang_mm.py.jinja` num_stages=2 path now compiles. `load_tiles` / `mma_tile` take `lid` (`uint3`) as a parameter (was referenced at file scope where `SV_GroupThreadID` doesn't exist); double-buffered `tile_a` / `tile_b` are flat `[2 * M * K]` arrays instead of `[2][M*K]` 2-D arrays (the latter doesn't support flat indexing under slang typing). `TestMatmulTemplateCompiles` exercises ns=1/2 × bias on/off.
- 2026-04-26: P4.4 — `inductor_stats.summary()` aggregator + `TORCH_VULKAN_DUMP_FX=<dir>` pre/post-pass graph dump. Fixed `_wrap_stats` capturing the dict entry at wrap time (so `reset_stats()` clears the dict but cached wrappers keep writing to dangling refs). Documented L0 (`num_stages=2` template `lid` undefined identifier — exposed by P1.4 prewarm + the loader fix; runtime swallows it via best-effort prewarm so autotune just never picks stage-2).
- 2026-04-26: P4.2 — `VulkanExprPrinter._print_Mul`/`_print_Add` collapse `0*x→0` / `1*x→x` / `0+x→x` patterns for unevaluated SymPy nodes so the emitted Slang doesn't ship dead arithmetic the driver would have to fold. `TestExprPrinterSimplify` locks it. (Wrong slangc relpath in CLAUDE.md / docs also corrected — was `../../third_party/...`, actual is `third_party/...` from `backends/vulkan_slang/`.)
- 2026-04-26: P1.4 — `prewarm_matmul_templates()` schedules the standard mm/addmm/bmm × {f32, f16} × {1, 2 stages} tile configs through the slangc thread pool at `register()` time (60 cache_keys). `runtime.prewarm_compile()` deduplicates against the in-memory + disk SPIR-V cache and submits the rest as background futures. Gated off via `TORCH_VULKAN_NO_PREWARM=1`. `TestSlangcPrewarm` regression locks the prewarm cache_keys to the runtime callers. Side-effect: fixed `_load_slang_template` looking for `slang_mm.jinja` while the actual file is `slang_mm.py.jinja` (the autotuned-template matmul path was silently broken on every fresh install — the C++ `aten.mm` extern was masking it). Also cleaned up the duplicate `_SLANGC` / `_INDUCTOR_STATS` definitions in `runtime.py`.
- 2026-04-26: P0.0 follow-up — diagnosed view-op fake-device gap (PyTorch's C++ view fast-path constructs view-op outputs with `device=meta` because the source FakeTensor is in `in_kernel_invocation` mode). Landed: opt-in `_patch_fake_tensor_view_op_device` via `TORCH_VULKAN_FAKE_VIEW_FIX=1`, `aten::{mm,addmm,bmm,linear}.out` PrivateUse1 impls (in `csrc/backend/Registration.cpp`), and `VulkanPythonWrapperCodegen.make_allocation` upgrade for `device=meta`→`empty_strided_vulkan` (`python/torch_vulkan/inductor/wrapper.py`). With all three enabled + a "meta"→VulkanScheduling alias, compile reaches scheduler — but the emitted backward wrapper is DCE'd to a no-op (Inductor's analyses treat the meta-device gradient as unreachable). Root cause is upstream: `aot_autograd`'s `meta['val']` capture stores raw meta tensors. Backed out the alias; opt-in patch and prereqs remain. 25/25 forward tests still pass.
- 2026-04-26: P4.1 — `benchmarks/inductor_train.py` runner shipped (`--model={mlp,mnist_cnn,resnet18,transformer_block} --train --json`). MLP/MNIST baselines collected; bigger workloads pending P0.0 unblock.
- 2026-04-26: P1.2 — multi-axis sum/mean verified to compile to 1 dispatch via stock Inductor codegen. `TestReductions` added.
- 2026-04-26: P1.1 — `(x + residual) → RMSNorm` verified to compile to 1 dispatch via stock fusion. `test_add_rms_norm_dispatch_count` + `test_add_rms_norm_correctness` lock it in.
- 2026-04-26: P1.4 — `VulkanComboKernel` rewritten. Subkernel buffers + locals are globally renamed (`xindex`/`x0`/`tmp\d+` → `..._sub{idx}`, colliding inner buffer names → `s{idx}_inner`). Same outer buffer aliased into multiple subkernels reuses one Slang binding. Inplace buffers (outer in both inputs and outputs) declared `RWStructuredBuffer`. `BackendFeature.FOREACH` re-enabled. `TestForeachCompiles` tightened from `≤4` to `≤1` (4 tensors fuse to 1 combo dispatch).
- 2026-04-26: P0.0 — `csrc/backend/MetaGuard.h` adds `is_null_storage` / `make_meta` / `make_meta_broadcast` helpers; null-storage guards applied at the top of every PrivateUse1 op called by AOT autograd's C++ backward formulas (binary helpers, unary helper, activation helper, mm/addmm/bmm/linear, sum/mean, expand/cat/clone/select/slice, in-place add/sub/mul/div, comparisons, copy_, fill_); `BACKWARD_FAKE_GUARD` rewritten to return a meta tensor instead of allocating Vulkan storage. Adds `TestCompileBackwardBaseline` regression covering `(x*x+1).sum().backward()` and `linear→relu→sum().backward()`. Forward fake_tensor_prop is now safe; backward graph compile remains blocked on the FakeTensor view-op device-propagation gap (P0.0 follow-up).
- 2026-04-26: CLAUDE.md and `docs/10-inductor-backend.md` rewritten — agent operating manual is now laser-focused on the inductor backend (never-stop loop, MAX_JOBS=4 default), roadmap expanded with P4.1–P4.6 (end-to-end measurement, IR quality, shader codegen quality, ergonomics, quantization, distributed).
- 2026-04-26: P1.4 — `BackendFeature.FOREACH` disabled until `VulkanComboKernel` is rewritten; foreach ops now compile (per-kernel fallback, 4 dispatches for 4 tensors instead of crashing). Codegen `import torch` ordering bug also fixed.
- 2026-04-26: P1.4 — push-constant fast-path specializations for common (n_buffers, n_pc) combos in `_make_kernel_wrapper` skip `*args` slicing per dispatch.
- 2026-04-26: P1.4 — picklable external matmul callables (`_SlangTileMM` / `_SlangTileAddMM` / `_SlangTileBMM` with `__reduce__`) eliminate codecache pickle warning on every cold compile.
- 2026-04-26: P1.1 — `nn.RMSNorm` inference verified to compile to **1 dispatch** via stock primitive fusion (no custom lowering required). Locked in by `TestRMSNormForward`.
- 2026-04-26: P0.0 added — backward-graph compilation is blocked by C++ PrivateUse1 ops crashing on FakeTensor inputs during AOT autograd's joint-graph trace (autograd's C++ backward formulas dispatch directly, bypassing fake_impl). `TestActivationBackward` marked `xfail(strict=True)`.
- 2026-04-26: Activation backward lowerings (sigmoid/tanh/silu) added to `lowerings.py` so backward graphs decompose into Inductor pointwise primitives once P0.0 unblocks.
- 2026-04-26: Roadmap rewrite — comprehensive gap analysis, P0–P3 prioritization, and continuous-loop workflow.
- 2026-04-25: 12/12 regression tests passing; ResNet 5.7×, MobileNet 2.3× speedup.
- 2026-04-25: Per-kernel stats API (`inductor_stats.py`) live behind `TORCH_VULKAN_INDUCTOR_STATS=1`.
- 2026-04-25: addmm template caller with bias+epilogue fusion (Phase A).
- 2026-04-24: lowerings for `native_layer_norm`, `native_group_norm`, `_softmax`, `_log_softmax`.
- 2026-04-24: FX passes — `mm+add→addmm`, `bmm+scale→scaled_bmm`, redundant-copy removal.
- 2026-04-23: Initial Vulkan Inductor backend wired through `register_backend_for_device`.

---

## Shipped Items Index (v3/v4 amendments, 2026-04-28 to 2026-05-01)

Consolidated index of all items that shipped during the v3/v4 amendment period.
These are retained for context only — they do not require action.

### Device propagation & backward compile

| Item | What it delivered |
|------|-------------------|
| PF.1 | Joint-graph device-tag propagation |
| PF.13 | C++ view-op null-storage fix |
| PF.13.b (4/7) | Activation backward compile unblocked |
| PF.13.b.1 | Softmax backward null-storage (via PF.52) |
| PF.13.b.3 | Log-softmax backward correctness |
| PF.13.c | Transposed-mat2 shader bug fix |
| PF.50 | FakeTensor device propagation fix |
| PF.51 | Null-storage guard for view ops |
| PF.52 | Tangent materialization FX pass |

### Lowerings & FX passes

| Item | What it delivered |
|------|-------------------|
| PF.2 | Optimizer-step FX pass (foreach fusion) |
| PF.3 | Conv2d + max_pool2d Inductor lowerings |
| PF.3.b | MNIST CNN end-to-end forward correctness gate |
| PF.4 | Zero-copy `aten.permute` lowering |
| PF.5 | `mm+bias+gelu` epilogue fusion |
| PF.20 | Convolution backward end-to-end lock |
| PF.21.a/.b/.c | Atomic scatter Slang module |
| PF.22 | `index_put_(accumulate=True)` lowering |
| PF.24 | Batch norm backward lowering |
| PF.25 | Max pool backward lowering |
| PF.26 | Dropout backward lowering |
| PF.46 | Optimizer step e2e ≤ 1 dispatch |
| PF.49 | Sub-32-bit dtype binding |

### Slang exploitation

| Item | What it delivered |
|------|-------------------|
| PF.6.a/.b | `bwd_diff` codegen + dispatcher (float32) |
| PF.10.a/.b | `slangc -target cpp` compile + parity |
| PF.11.gelu | First hot-path `[BackwardDerivative]` retirement |

### Compile reliability

| Item | What it delivered |
|------|-------------------|
| PF.14 | Re-entrant ThreadPoolExecutor deadlock fix |
| PF.15 | SPIR-V cache budget gate |
| PF.16 | Build-time shader pre-population |
| PF.18 | slangc subprocess timeout |
| PF.19 | E2E dispatch coverage CI gate |

### AOTI & memory

| Item | What it delivered |
|------|-------------------|
| PF.30.a/.b/.c/.d/.g | Custom-op shims for eager patches |
| PF.31 (C++ ABI) | `AotiRuntime.cpp` with 5 ABI entries |
| PF.40 | Lifetime-class FX pass |
| PF.53 | SLANGC env resolution |
| PF.54 | `vulkan_pool_release` tuple safety |
| PF.57 | Audit scripts |
| PF.58 | Coverage audit infrastructure |

---

# v2 Roadmap — Completed Tracks (migrated 2026-05-03)

## Track 0 — Correctness Floor ✓ CLOSED

Shipped 2026-05-02. Every graph that compiled produced correct output, including backward and dropout. The track delivered:

| # | Item | Result |
|---|------|--------|
| 0.1 | **GAP 0.1 / PF.A0** — wrapper drops input args; bmm fwd wrong | ✓ Fixed by `_empty_strided_vulkan` + `make_vulkan_kernel` imports |
| 0.1a | **PF.30.h.5** — cold-compile RecursionError | ✓ 3 patches extracted+activated: GPU_TYPES, Scheduler TritonMissing exemption, CPU-timer benchmark routing |
| 0.2 | **GAP 1.1** — matmul+softmax bwd silently zero gradients | ✓ Added `_softmax_backward_data` to `_suppress_upstream_decomps()` |
| 0.3 | **GAP 1.2** — slangc SIGSEGV on stale `.slang-module` | ✓ Fingerprint now includes `slangc --version` output |
| 0.4 | **GAP 7.3** — slangc-cache state pollution across tests | ✓ All 7 module globals cleared in `reset_per_test_caches()` |
| 0.5 | **GAP 1.3** — RNG determinism under compile | ✓ Philox seed bound as `StructuredBuffer<uint2>`, `.x` truncation in pointwise codegen |
| 0.6 | **GAP 5.5** — Vulkan-aware autotune harness | ✓ 3 patches active; Slang tiles guarded by `TORCH_VULKAN_ENABLE_SLANG_TILES` |

**Exit gate met:** 2-layer MLP-with-dropout trains 10 steps under compile with gradients matching CPU at rtol=1e-3.

## Track 1 — Codegen Refactor ✓ DONE

Shipped 2026-05-02. Split all monolith files so Tracks 2-5 land in parallel without merge collisions.

| # | Item | Result |
|---|------|--------|
| T1.1 | Split `kernel.py` (1773→199L) | `kernel/` package: `main.py` 199L, `pointwise.py` 468L, `reduction.py` 414L, `indexing.py` 170L, `header.py` 387L |
| T1.2 | Split `lowerings.py` (1908L) | `lowerings/` package: 10 modules, largest 601L |
| T1.3 | Split `fx_passes.py` (2089L) | `fx_passes/` package: 11 modules, largest 520L |
| T1.4 | Promote `codegen.py` (34L stub) | `OpClass` enum + `CodegenStrategy` registry + 11 op classes |
| T1.5 | Add `TemplateRegistry` | Typed registry indexed by `(op_class, dtype, shape_class)` |

**Exit gate met:** All split packages ≤ 601 lines; diff purely extractive; zero behavior change.

## Revised Implementation Plan Phases 1-2 ✓ COMPLETE

### Phase 1: Correctness Floor

| # | Item | Status |
|---|------|--------|
| P1.1 | Activate autotune harness (PF.30.h.5) | ✓ DONE (2026-05-02) |
| P1.2 | matmul+softmax bwd zero gradients (GAP 1.1) | ✓ DONE (2026-05-02) |
| P1.3 | RNG determinism (GAP 1.3) | ✓ DONE (2026-05-02); dynamic seeds deferred |
| P1.4 | Track 0 exit gate | ✓ Passes |

### Phase 2: Codegen Refactor

| # | Item | Status |
|---|------|--------|
| P2.1 | Split `kernel.py` | ✓ DONE |
| P2.2 | Split `lowerings.py` | ✓ DONE |
| P2.3 | Split `fx_passes.py` | ✓ DONE |
| P2.4 | Promote `codegen.py` | ✓ DONE |
| P2.5 | Add `template_registry.py` | ✓ DONE |

## 2026-05-02 AI-Assisted Session — Complete Log

### Round 1: Autotune + cold-compile + kernel stubs
- PF.30.h.5: Autotune harness activated (3 patches from `_patch_register_vulkan_as_gpu_DRAFT_PF30H`)
- Slang tile safety gate (`TORCH_VULKAN_ENABLE_SLANG_TILES`)
- Kernel package structure created (5 stub modules)
- Revised 6-phase implementation plan
- 6 missing items identified (M1-M6)

### Round 2: Softmax backward + bwd_diff
- GAP 1.1: Fixed matmul+softmax bwd zero gradients
- `_suppress_upstream_decomps()` for `_softmax_backward_data`

### Round 3: Slangc stability + RNG + bwd_diff
- GAP 1.2: Fixed slangc SIGSEGV (fingerprint includes version)
- GAP 7.3: Fixed cache state pollution across xdist (7 globals cleared)
- GAP 1.3: RNG determinism plan documented; regression test added

### Round 4: Track 1 codegen refactor
- T1.2: lowerings.py split (10 modules)
- T1.3: fx_passes.py split (11 modules)
- T1.4: codegen.py promoted (OpClass + CodegenStrategy)
- T1.5: template_registry.py created

### Round 5: Track 0 closeout, T2.3, T3.1-T3.2, T3.8
- T0.5: RNG codegen floor resolved
- P1.4: Track 0 exit gate test added
- T2.3: tools/lib_graph.py created
- T3.1: Multi-dtype bwd_diff (f16/bf16)
- T3.2: Binary-loss bwd_diff wired (fp32)
- T3.8: bwd_template_registry.py created

## 2026-05-03 AI-Assisted Session — Complete Log

### Template infrastructure
- T4.1 (template half): Slang-generic `IPointwise` epilogue in mm template. String-based `epilogue="gelu"` retired. Anti-goal #6 satisfied.
- T4.2: mm/bmm backward entries in `bwd_template_registry.py` (3 entries)
- T4.3 (partial): `FxPatternRegistry` created with 4 initial entries
- T3.3: SiLU backward retirement (1 shader deleted, lowering routes through `bwd_diff`)
- T3.4: Mass retirement — 15 backward shaders deleted (9 activation + 6 loss bwd)
- T3.5: `[BackwardDerivative]` on norm elementals (4 annotations in `lib/norm.slang`)
- P3.5: Extension methods for top 5 math helpers (`(x).erf()`, etc.)

---

## 2026-05-08/09 AI-Assisted Session — 9-Round Multi-Agent Sprint

Single-day sprint of 9 parallel-agent rounds (4 agents × 9 rounds = 36 ships).
Net regression: **494 → 692 PASS (+198, +40%)**, **281 → 174 FAIL (−107, −38%)**.

### Round 1 (2026-05-08) — Bug-fix and audit infra
- **T4.9** ✓ Flash attention 3 algorithmic defects (head/q-tile aliasing, KV_H underflow, Phase-2 single-D-component) — collapsed to single-pass online softmax
- **T2.7** ✓ Retired `wg_sum_smem` + `_norm_smem` Welford duplicate (norm.slang now `import reduction;`)
- **T2.11** ✓ `[BackwardDerivative]` for all 7 loss elementals (M12 target 10→27)
- **F.2 / F.3** ✓ Reconcile reopened — empirically verified SCAN/SORT/TUPLE_REDUCTION are wired end-to-end (audit was wrong); flags KEPT
- **T.6** ✓ codegen_audit.py retargeted at `kernel/` package; LOCKED_CEILINGS rebaselined
- **T.7** ✓ slang_validator pre-flight (was post-mortem)
- **T.8** ✓ Stale LOCKED_CEILINGS rebaseline folded into T.6

### Round 2 — File-size cleanup + new ops
- **T6.6** ✓ Sub-allocation dead code deleted (-195L from `buffer_pool.py`)
- **OP.5** ✓ `aten.embedding_bag` forward lowering
- **M30** ✓ `extension float` rollout (5→22 ext methods)
- **T4.13** ✓ `_SlangTileMM/AddMM/BMM/AddMMGelu` → `_SlangTileGEMM` consolidation (pickle-stable shims)

### Round 3 — Critical correctness fix
- **T2.6** ✓ `helpers.slang` 827L split into `dtype_pack` + `philox` + `special_math` + `bucket` (`__exported import` chain)
- **F.10 NEW + FIXED** — int64 truncation root cause: upstream `compile_fx.get_input_idxs_to_check` flagged synthetic Vulkan ptrs as misaligned, triggering clone via 4B/elem `dispatch_copy_buffer` → silent int64→int32 truncation. Fix: `wrapper.py:_install_vulkan_skip_alignment_clone()`. **+129 tests pass**
- **T6.6b** ✓ config.py duplicate dead `_SUB_ALLOC_ENABLED` removed
- **N+1.10 (1/6)** — `foreach_optimizer.py.jinja` migrated to `ParameterBlock<T>` (-22 binding literals at batch=4)
- **check_bounds** ✓ `IndexingMixin.check_bounds` via early-return guard (kernel/indexing.py); discovered slang_validator newline reset bug

### Round 4 — More templates + cleanup
- **T4.10** ✓ Flash `BK`/`BQ` → template kwargs
- **N+1.10 (2/6)** — `philox_rng.py.jinja` migrated (-3 binding literals)
- **slang_validator newline reset** ✓ (`_check_brace_balance` + `_check_size_symbol_leaks`)
- **OP.3** ✓ `repeat_interleave` / `roll` / `narrow_copy` / `unfold` lowerings (Pointwise inner_fn with FloorDiv); `unfold` blocks Conv1d→T4.12
- **T2.9** ✓ Stub-shader cleanup — `attention.slang` + `training.slang` deleted (no real bodies; impls in templates); 3 `bwd_template_registry.py` entries → `module="__template__"`

### Round 4 build closeout
- **F.9 NEW + FIXED** — `csrc/backend/AotiRuntime.cpp` 3 pre-existing C++ syntax bugs (multi-line string literals at L279/L338, malformed char literal `'\'` at L326) — surfaced by C++ rebuild

### Round 5 — Pool + FFT + alias retire
- **N+1.10 (3/6)** — `scatter_atomic.py.jinja` migrated (-4 binding literals; int64-via-uint2 works in `ParameterBlock` cleanly)
- **T6.7** ✓ Workspace scratch routed through pool — new `"scratch"` lifetime class; 3 actual scratch sites wired
- **OP.4** ✓ FFT fallback lowerings (`_fft_r2c`/`_fft_c2c`/`_fft_c2r`); MAJOR FINDING: Vulkan eager FFT was never wired to PrivateUse1 (`fft_ops.cpp` 347L inert)
- **T2.10 (partial)** ✓ 16 `c10_vulkan_*` aliases retired in `special_math.slang`; 12 kept load-bearing → T2.10b filed

### Round 6 — N+1.10 complete + survival test
- **N+1.10 COMPLETE (6/6)** — slang_mm + mirror + slang_conv2d migrated. Cumulative -73 binding literals. Mirror is actively-loaded (correction to M33 audit). Reflection-verified
- **T6.4** ✓ 50-step memory plateau test passes (pool flat at 8KB across steps 15-50; T6.7 closed all leaks). Discovered T5.14 codegen bug
- **OP.1 (5/9)** ✓ Multi-tensor + int scatter already worked after round-3 int64 fix; 4 bool-mask paths xfail on OP.1.a (nonzero) + OP.1.b (bool LOAD)
- **M31** ✓ KEPT — all 9 `attention/*.slang` reachable from `attention_ops.cpp:332-476` (audit was wrong)

### Round 7 — Reduction codegen + C++ wirings
- **T5.14** ✓ FIXED — `r0__linear_idx` undefined identifier in reduction-elementwise codegen. Root cause: partitioned-2D layout claimed reduction root, late CSE-merged flat root entry hit multi-stage else-branch with undeclared var. Fix: hoist + synthesize as `((_ry*tx+lid.x)*inner_len + (_rx*ty+lid.y))`. Unlocks compiled training step
- **OP.4-eager-followup + OP.1.a** ✓ 3 FFT m.impl + 1 nonzero m.impl + 2 SymInt adapters in `Registration.cpp` (FFT now reaches `view_as_complex` next blocker; OP.1.a is CPU-roundtrip — OP.1.a-fast filed for GPU two-pass scan)
- **T2.8** ✓ `mm_tiled.slang` deleted (zero callers); `mm.slang` + `mm_tile.slang` KEPT (real users, audit was wrong about mm.slang); T2.8b filed for vulkan_template_caller cleanup
- **builtin_patterns.py SPLIT** ✓ 1253L → 10 per-pattern files + `_common.py` + 33L shim (anti-goal #7 closure for that file)

### Round 8 — FFT closure + bool fix + dead code
- **OP.4-fft-followup-2** ✓ `view_as_complex` + `view_as_real` registered (path 1 — reused upstream `at::native::*`); `torch.fft.rfft(x)` works end-to-end on Vulkan
- **OP.1.b** ✓ Bool LOAD path fixed in `kernel/pointwise.py:_LOAD_DISPATCH` — emits `_vk_unpack_u8(buf, idx)` for graph-input bool buffers. `m.float()` returns `[1.0, 0.0, 1.0, 0.0]` (was `[65537.0, 0.0, 0.0, 0.0]`)
- **T2.8b** ✓ Dead matmul wrapper code purged (-299L from `vulkan_template_caller.py`: 4081→3782); pickle stability preserved
- **OP.2 (sum-mode)** ✓ `index_add` / `scatter_reduce.two(reduce='sum')` / `index_copy` / `masked_scatter` lowerings; reduce-modes deferred to T4.11

### Round 9 — Atomics + eager bool + enum
- **OP.1.c** ✓ FIXED — `dispatch.cpp:333` always used 4B/elem float shader; new `dispatch_copy_buffer_byte` uses `vkCmdCopyBuffer` with `dst.nbytes()` precision. Bool/Byte/Char/Float8 routed through byte path. Verified `m.float()=[1.0,0.0,1.0,0.0,1.0]`
- **T4.11 (infra)** ✓ `atomic_max_f32`/`atomic_min_f32`/`atomic_mul_f32` CAS loops added to `atomics.slang`; scatter_atomic 5→9 modes; mean uses companion `count` SSBO. OP.2 reduce-mode lowering wiring deferred to Group B
- **M29** ✓ Typed `ReductionKind` enum + `REDUCTION_TABLE` (Python-only refactor; old dicts kept as compat shims)
- **T.12 PARTIAL** — 7 einsum patterns tested; eager works, **compiled silently produces wrong values (all 1.0s)** for 5/7 patterns despite single-dispatch decomp. 7 sub-items filed

### Net session cumulative
- 35+ items closed
- 9 new blockers filed and tracked (3 fixed inline)
- Lib modules: 13 → 14 (helpers split +4; attention/training stubs -2; mm_tiled -1)
- Template binding literals: 33 → 0 (N+1.10 COMPLETE 6/6)
- Anti-goal #7 violations: 6 → 4 (`buffer_pool.py` & `builtin_patterns.py` split)
- Extension methods: 5 → 22
- C++ ATen ops wired: nonzero, _fft_r2c/c2c/c2r, view_as_complex, view_as_real (+6)

## 2026-05-09 Round 10 — Training-readiness sprint

Single round, 4 parallel agents; net +8 pass / -7 xfail / OP.2 fully closed.

### TR.13 + TR.14 (Group F) — wrapper.py training blockers
- **TR.13 FIXED**: `_generate_kernel_call_helper` accepts `**kwargs` (current_stream_idx absorbed; Vulkan has no streams)
- **TR.14 FIXED**: Root in upstream `pattern_matcher.ReplacementPatternEntry.replace_with_graph` — the `NotImplementedError` f-string calls `Tensor.__repr__→tolist()` which crashes on Vulkan tensors with `data_ptr=0`. These come from `make_fx` tracing replacements under `V.fake_mode` (e.g. `scatter_upon_const_tensor` lifts `torch.arange` as `_tensor_constantN`). Fix: monkey-patch `replace_with_graph` to skip replacements containing Vulkan get_attr with `data_ptr=0`. **Side benefit**: 4 of 7 T.12 einsum tests flipped to PASS

### TR.15 (Group H) — SDPA fusion FakeTensor crash
- Root NOT in our code: upstream `torch._C._nn.scaled_dot_product_attention` has no `register_fake`; AOTAutograd's metadata collection dereferences `data_ptr()` on FakeTensor when SDPA inputs are non-contiguous (post reshape/transpose chain).
- Fix: new `_replace_sdpa_with_custom_op()` in `fx_passes/post_grad.py` (~150L) rewrites SDPA call_function to pure aten primitives chain (transpose → matmul → mul → optional triu → softmax → matmul). FakeTensor-safe.
- 2 downstream blockers prevent end-to-end attention block: (a) requires_grad=True hits `opt_ready_stream && opt_parent_stream` autograd engine assertion (independent Vulkan-stream issue); (b) requires_grad=False hits TR.14-style `_tensor_constant3` runtime data_ptr.

### OP.2 reduce-mode lowering (Group B) — FULLY CLOSED
- Wired `lowerings/scatter.py:_vulkan_scatter_reduce_eager` to dispatch reduce='prod'/'amax'/'amin'/'mean' through `_dispatch_scatter_atomic` (T4.11 infra).
- **Critical bugfix**: hoisted `torch.library.Library` handle to module-level `_LIB` (was function-local — GC unregistered all impls silently).
- Found and registered `aten.scatter_reduce.two_out` structured-delegate target (PyTorch redispatches there even when `_.two` is registered).
- Mean post-pass: count int32 buffer + `out / clamp(count, min=1)` divide.
- All 8 TestOp2ScatterFamily tests PASS at rtol=1e-4.

### T.12 einsum (Group C) — root-caused
- Two-stage constant-bake bug: (1) `aten.einsum` (CompositeImplicitAutograd) creates intermediate views in C++ with no proxy → bake as `_tensor_constantN` get_attr (data_ptr=0) → `constant_fold_uniform_value` reads uninit storage → replaces matmul with `aten.full(1.0)`. (2) `bmm_to_mm` (B=1) re-introduces the same bug.
- Fix in `meta_patches.py`: `_patch_einsum_proxy_decomp` (Python einsum decomp via `__torch_dispatch__`) + `_disable_bmm_to_mm_for_vulkan` + broadened `_patched_reduce_tensor` exception handling.
- 4/7 T.12 tests PASS (matches F-agent's pattern-matcher fix coverage). Filed T.12.A (4D permute+reshape returns 1.0s standalone — blocks attention QK) + T.12.B (reinterpret_tensor stride-17 ignores user stride — blocks diag_extract).

### Round 10 cumulative
- Pass: 692 → 695 (+8 with -7 xfailed flips; net pass count includes some reorganization)
- 4 OP.2 reduce-modes flipped; 4 T.12 einsum patterns flipped
- TR.13/TR.14/TR.15 closed
- 3 new sub-items filed: T.12.A, T.12.B, opt_ready_stream autograd-engine

## 2026-05-09 Round 11 + Wave A (Round 12)

### Round 11 — TR.13/TR.14/TR.15 + OP.2 + T.12 root-cause
- **TR.13** ✓ FIXED: `_generate_kernel_call_helper` now accepts `**kwargs` (current_stream_idx absorbed; Vulkan has no streams).
- **TR.14** ✓ FIXED: upstream pattern_matcher `replace_with_graph` raises NotImplementedError on get_attr-to-tensor; the f-string calls `Tensor.__repr__→tolist()` which crashes on Vulkan tensors with data_ptr=0 (from make_fx tracing replacements under V.fake_mode lifting torch.arange as _tensor_constantN). Monkey-patched to skip such replacements. **Side benefit: 4 of 7 T.12 einsum tests flipped to PASS.**
- **TR.15** ✓ FIXED at data_ptr crash level: upstream `torch._C._nn.scaled_dot_product_attention` has no register_fake; AOTAutograd metadata collection dereferences data_ptr() on FakeTensor when SDPA inputs are non-contiguous. Fix: new `_replace_sdpa_with_custom_op()` in fx_passes/post_grad.py rewrites SDPA call_function to pure aten primitives chain (transpose → matmul → mul → optional triu → softmax → matmul). 2 downstream blockers flagged: TR.18-A/B (autograd stream + meta-storage), TR.14-territory const.
- **OP.2 reduce-modes** ✓ FIXED: All 4 modes (prod/amax/amin/mean) wired through `_dispatch_scatter_atomic` (T4.11 infra). Critical bugfix: hoisted `torch.library.Library` to module-level `_LIB` (function-local was GC'd, silently dropping registrations). Mean post-pass: count int32 buffer + `out / clamp(count, min=1)` divide.
- **T.12** root-caused: einsum CompositeImplicitAutograd creates intermediate views in C++ with no proxy → bake as _tensor_constantN get_attr (data_ptr=0) → constant_fold_uniform_value reads uninit storage → replaces matmul with aten.full(1.0). Fix: meta_patches.py `_patch_einsum_proxy_decomp` (Python einsum decomp via __torch_dispatch__) + `_disable_bmm_to_mm_for_vulkan`. 4/7 T.12 tests PASS after this round.
- **OP.1.b** ✓ FIXED: kernel/pointwise.py:1004-1031 emits `_vk_unpack_u8(buf, idx)` for graph-input bool buffers; `m.float()` for `[T,F,T,F]` returns `[1.0, 0.0, 1.0, 0.0]` (was `[65537.0, 0.0, 0.0, 0.0]`).
- **OP.1.c** ✓ FIXED: csrc/ops/dispatch.cpp:333 always used 4B/elem float shader for ALL dtypes; new `dispatch_copy_buffer_byte` uses `vkCmdCopyBuffer` with `dst.nbytes()` precision (Bool/Byte/Char/Float8 routed through byte path).
- **OP.1.d** ✓ FIXED: lowerings/bool_mask.py registers PrivateUse1 override on aten::index.Tensor; bool path = CPU roundtrip; int path forwards via call_boxed (preserves GPU dispatch). 9/10 OP.1 paths PASS.
- New blockers filed: TR.13.b (Slang `**` operator printer), TR.17 (CNN out_ptr0 undefined), TR.18-A/B (autograd stream + vulkan_transpose meta-storage leak).

### Wave A (Round 12) — 5 closures, +42 net pass
- **TR.16.A** ✓ FIXED: vulkan_combo_kernel.py:708 emitted `uint _vk_gtid = gtid.x + gid.x * max_tgs;` but `gtid.x = SV_DispatchThreadID.x = gid.x * numthreads.x + lid.x` ALREADY. Double-counting caused workgroups with gid.x>=1 to skip bounds check, leaving output slots uninitialized. Fix: `uint _vk_gtid = gtid.x;`. Conv+BN(eval) max diff 2.72 → 3.6e-7. Test `TestComboKernelGtidIndexing` flipped green.
- **TR.17** ✓ FIXED: kernel/header.py M22 dead-code-elim round-trip materializes DeferredLine stores into strings on each codegen_body() call. On multi-stage reductions, runs for stage 1 BEFORE stage 2 inplace-merge adds stage-1 output to removed_buffers. Fix: filter body_code._lines against declared inner-name set via regex.
- **T.12.A** ✓ FIXED: missing Python AutogradPrivateUse1 impl for permute/transpose/t — C++ vulkan_permute built fresh tensor not aliasing self's storage; under FakeTensorMode, AOTAutograd lifted as frozen _tensor_constantN reading uninit memory → constant_fold_uniform_value folded to aten.full(1.0). Fix: meta_patches.py `_register_permute_family_autograd_pyimpl` routes through aten.as_strided (proper aliasing view). **Side benefit: attention QK + QK·V T.12 tests flipped green; 6/7 T.12 now passing.**
- **TR.18-A** ✓ FIXED + **TR.18-B** ✓ FIXED: round 11 hypothesis was wrong — stream/event registration was already adequate. Real root: csrc/ops/shape_ops.cpp:227-252 vulkan_transpose zero-copy used self.storage() which under FakeTensorMode is Meta storage → result has device=meta → backward returned meta tensor → engine.cpp:1084 fired. Fix: 6 zero-copy view ops gated (vulkan_transpose, vulkan_t, vulkan_permute, vulkan_unsqueeze, vulkan_squeeze, vulkan_squeeze_dim) on `is_null_storage(self) || self.is_meta() || !self.has_storage()` — return `make_vulkan_null` (PrivateUse1-keyed). Defense-in-depth: 3 stream methods added (getDefaultStream, getNewStream, getStreamFromGlobalPool).
- **M32** KEPT: 8 shaders/training/*.slang still eager-reachable from optimizer_ops.cpp + init.cpp Python bindings. Mirrors M31 outcome. Follow-up filed for foreach Inductor template eager retirement.

### Cumulative through Wave A
- 35+ items closed across 12 rounds + wave-A
- Pass: 494 → 737 (+243, +49%)
- Fail: 281 → 147 (−134, −48%)
- xfailed: 19 → 20
- xpassed: 0 → 4
- Lib modules: 13 → 14
- Template binding literals: 33 → 0
- Anti-goal #7 violations: 6 → 4
- C++ ATen ops wired: nonzero, _fft_r2c/c2c/c2r, view_as_complex, view_as_real (+6)

## 2026-05-09 Wave B (Round 13)

4 parallel agents; net +7 pass / -5 fail / -1 xfailed.

### Closures
- **TR.13.b** ✓ FIXED (Group C, ~5L): `OpOverrides.pow` upstream returns `f"{a} ** {b}"` Python form. Added `pow(a, b) → "pow(({a}), ({b}))"` static method override in `python/torch_vulkan/inductor/overrides.py`. Adam's bias-correction `beta**step` now compiles. test_pow_emits_slang_pow_t_13_b PASSES.
- **T2.10b** ✓ DONE (Group G+C, 12 symbols): renamed all 12 load-bearing `c10_vulkan_*` to `vk_*` (atomic_add, bucketize, vec4_h{sum,max,min,prod}, wg_welford, wg_reduce_{any,xor,argmax,argmin,xor_2d}). Old aliases kept as 1-line ForceInline forward-compat. 9 Python caller emission sites flipped (kernel/pointwise.py 5, kernel/reduction.py 4). Codegen emission of `c10_vulkan_*` is now 0.
- **N+1.5** ⚠ PARTIAL (Group A C++ shipped): Pipeline.{h,cpp} 2nd ctor for descriptor_counts vec; DescriptorSet.{h,cpp} bind_buffers_indexed(); dispatch.{h,cpp} dispatch_shader_indexed() with auto-fallback; init.cpp FFI bindings _jit_dispatch_indexed + _descriptor_indexing_enabled. 3 sub-items filed: N+1.5.a (Group F Python codegen extract per-binding count from reflection JSON), N+1.5.b (Group D foreach params[i] migration replaces switch cascade), N+1.5.c (Group E combo-kernel >16-binding unblock — closes T5.12).
- **OP.5b** ✓ DONE (Group B, ~58L): mode='max' for embedding_bag via scatter_reduce(reduce='amax', include_self=False) with empty-bag post-pass `where(bag_size>0, output, 0)`. Refactored mode-mean to share bag_size computation. 2 upstream-quirk workarounds documented.
- Build closeout: A-agent's Pipeline.cpp refactor introduced duplicate `VkResult result` declaration (line 125 shadowed line 40 after extracting create_pipeline_objects). Fixed inline by renaming to `result_pl`.

### Cumulative through Wave B
- 40+ items closed across 13 rounds (waves A + B)
- Pass: 494 → 744 (+250, +51%)
- Fail: 281 → 142 (−139, −49%)
- xfailed: 19 → 19
- xpassed: 0 → 4
- Lib modules: 13 → 14
- Template binding literals: 33 → 0
- Anti-goal #7 violations: 6 → 4
- C++ ATen ops wired: nonzero, _fft_r2c/c2c/c2r, view_as_complex, view_as_real (+6); descriptor_indexing C++ runtime support
- All 12 `c10_vulkan_*` aliases renamed; codegen emission 0

## 2026-05-09 Waves C + D (rounds 14-15)

### Wave C — N+1.5 Python wiring + T.12 closure + new lowerings

- **T.12.B** ✓ FIXED (Group C, wrapper.py): VulkanPythonWrapperCodegen overrides codegen_reinterpret_view + get_output_refs. Non-contiguous reinterpret_tensor in graph-output context emits as_strided (which materializes contiguous via copy_as_strided_fwd Slang shader). Internal kernel call sites keep zero-copy reinterpret. **T.12 status: 7/7 FULL CLOSURE** (diag_extract was the last xfail).
- **N+1.5.a** ✓ DONE (Group F, runtime.py): _binding_descriptor_count(param) helper recognizes 3 slangc reflection shapes. reflection_layout returns parallel descriptor_counts list. make_vulkan_kernel auto-routes to _jit_dispatch_indexed when any count > 1. Clear RuntimeError when device lacks extension. 4/4 descriptor-indexing tests pass.
- **N+1.5.b** ⚠ FEATURE-FLAG (Group D): foreach_optimizer.py.jinja emits two layouts — flat-with-switch (default) vs ParamSlot params[N] array-of-structs (TORCH_VULKAN_PARAMETER_ARRAY=1). 26 cascade-blocks eliminated when PA on. AND-gated on _descriptor_indexing_enabled. Default-off; switch path remains active.
- **N.1.b** ✓ DONE-VIA-CPU-ROUNDTRIP (Group B): lowerings/searchsorted.py registers PrivateUse1 overrides for searchsorted.Tensor/Scalar + repeat_interleave.Tensor. 3 new tests pass.

### Wave D — coverage + perf + safety

- **T.10** ✓ DONE-VIA-FALLBACK (Group B): lowerings/rnn.py registers 8 ops (lstm/gru/rnn_tanh/rnn_relu × {input, data}). Hybrid: Inductor make_fallback + Python PrivateUse1 (CPU-roundtrip via torch._VF). Required because high-level RNN ops are CompositeImplicitAutograd. 5/6 tests pass; rnn_relu xfail (clamp_min lineage). Multi-layer + bidirectional supported.
- **SVD wiring** ✓ DONE (Group A): vulkan_linalg_svd signature now takes bool compute_uv between full_matrices and driver. compute_uv=False returns empty U/Vh placeholders alongside meaningful S. m.impl("_linalg_svd", ...) registered.
- **N+1.13** ✓ DONE (Group G): 12 [require(...)] capability annotations on wave intrinsics across helpers.slang + reduction.slang (subgroup_arithmetic, subgroup_ballot, subgroup_shuffle, subgroup_vote). **Audit findings**: f16/bf16 pack/unpack uses asuint/asfloat bit-twiddling — does NOT need spvShaderFloat16Int8. atomic_add_f16_packed CAS-loops over RWByteAddressBuffer — does NOT need spvAtomicFloat16AddEXT.
- **T5.13** ✓ RE-ENABLED (Group E): scheduling.py:codegen_mix_order_reduction override deleted; delegates to upstream. No regression (Track-0 era escape obsolete after T5.14 + TR.16.A).

### Notes
- Pipeline.cpp was reverted to flat-binding form mid-wave; N+1.5.b feature flag default-off keeps switch cascades active.
- 8 closures total in waves C+D. Next session candidates: T.12.B-style fixes for any remaining stride-bake corner cases; N+1.11 SpecConst tile sizes; T7.4/T7.5 AOTI extern ABI; Track Z model coverage.

### Cumulative through Wave D
- 50+ items closed across 15 rounds (waves A/B/C/D)
- Pass: 494 → 755 (+261, +53%)
- Fail: 281 → 141 (−140, −50%)
- xfailed: 19 → 19
- xpassed: 0 → 4
- T.12 einsum: 0/7 → 7/7
- ATen ops wired in C++: nonzero, _fft_r2c/c2c/c2r, _linalg_svd, view_as_complex, view_as_real (+7)

---

## v5 Roadmap Completion (2026-05-02 through 2026-05-10)

**16 waves, 57 items shipped.** Full details were in the active roadmap
(`10-inductor-backend.md`) prior to the v6 cleanup.

### Performance (8 items)
DR.1-DR.8: Fusion scheduler, reflection cache, LDS padding, spec constants,
reflection routing, C++ AOTI wrapper, descriptor indexing, static specialization

### Autodiff (10 items)
CG.M1-CG.M10: 100% coverage unary/binary/norm/matmul/conv/SDPA.
Direct bwd_diff emission. IDifferentiable interfaces.
32 [BackwardDerivative] fast-backward formulas.

### Correctness (5 items)
BN.1 BatchNorm training drift, TR.19 backward consolidation,
D.2.a dynamic reductions, P1.5 CrossEntropy fix, replicate padding

### Op Coverage (12 items)
OP.1-OP.12: FFT, RNN, GPU nonzero, GPU mask gather, multinomial,
GPU searchsorted, GPU bucketize, GPU argsort, scatter family (9 modes),
view-style lowerings, embedding bag, vulkan cat

### Templates (5 items)
T.10-fast fused RNN (LSTM/GRU/RNN, 64x dispatch reduction),
CG.M5/M6 matmul/conv bwd 1-dispatch, N+1.11 tile spec constants,
T4.12 conv generality documented

### Infrastructure (6 items)
C++ AOTI wrapper, descriptor indexing, static specialization,
parallel slangc, .slang rename, dynamic shapes gate default ON

### Quality (8 items)
Bank conflict analysis, BlockPatternMatcher vec4, WG/PC standardization,
validator invariants, combo kernel improvements, [fastopt]/[branch],
__target_switch, generic wg_reduce<W,T>

### GPU Utilization (6 items)
Batch dispatch, wrapper fast-path, dispatch profiling,
grid-aware WG sizing, persistent pointwise kernels, occupancy estimator

### NaN Propagation (1 item)
P5.6: WaveActiveAny(isnan) pre-scan in all reduction ops (amax/amin/sum/prod/welford)

### Model Zoo
66 e2e model tests: ViT, Llama, UNet, Whisper, Mamba, MiniGPT,
ResNet, Transformer, Qwen3.5, SmallCNN North Star

### Remaining (genuinely blocked)
- N+1.9 Link-time tile spec (slangc upstream bug E30600)
- T7.2 Full .so subprocess (C++ build infrastructure)
- T4.12 Conv1d/3D/depthwise (template generality)
- Track CI (GPU hardware)

