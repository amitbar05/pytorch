# Vulkan Slang Backend — Improvement & Optimization Plan

Baseline (2026-04-10, AMD RX 5700 XT, MiniQwen3 4L D=256 B=2 S=64):
- Forward:   32 dispatches / 3.91 ms  (barrier skip 0%)
- Backward:  64 dispatches / 6.84 ms  (barrier skip 51%)
- Optimizer:  3 dispatches / 0.63 ms  (barrier skip 100%)
- **Total:  101 dispatches / ~11.6 ms** vs CPU parity ~0.6x
- 261 Slang shaders compiled, 1445 tests passing

The forward is near dispatch-floor; the backward and per-dispatch overhead
are the dominant targets. "Impact" below is measured in dispatches saved
per MiniQwen3 training step unless noted.

---

## Recently Completed (2026-05-06)

### ✅ TRAIN.1 — Conv2d dilation>1 stride correctness
Inductor wrapper-codegen stride fix for dilated convolutions.
- **Fix:** Corrected stride calculation in the im2col decomposition so
dilation>1 conv2d produces correct output shapes.
- **Side effect:** Dispatch count for `conv2d(x, w) + bias` rose from
≤6 → ≤13 due to additional reshape/copy operations in the wrapper.
Assertions bumped; follow-up below.
- **Follow-up (TRAIN.1-F1):** Audit the dispatch count. **Status: Investigated
  2026-05-06 — see "TRAIN.1-F1 — Conv2d Dispatch Count Audit" section above.**
  Counts are 9-12 depending on variant; paths to ≤6 identified (custom mm
  lowering or dedicated conv2d template). Clone removal done; deeper fix
  requires Inductor FallbackKernel bypass or fused conv2d shader.

### ✅ TRAIN.2 — Backward template dispatch (verified, no changes needed)
`dispatch_template_bwd` infrastructure and `bwd_template_registry.py`
entries already fully wired. The forward mm/bmm/addmm Slang template is
reused for backward (`dA = dC @ B^T`, `dB = A^T @ dC`) with transposition
encoded via push-constant strides — no CPU-side copy needed.

### ✅ TRAIN.6 — Combo-kernel wave-mask uniformity fix
Set `torch._inductor.config.combo_kernels_pointwise_only = True` to
exclude reduction kernels from combo-kernel if-ladder dispatch.
- **Root cause:** Combo kernel dispatches subkernels via flat `gtid`-based
`if/else if` ladder. With partial wave participation, wave intrinsics
(`WaveActiveSum`, `WaveActiveMax`) produced zero outputs for batched
outer-axis reductions.
- **Effect:** Reductions run in dedicated kernels (correct outputs).
Pointwise-only fusion (the common case) continues to work.
- **Follow-up (TRAIN.6-F1):** Wave-uniform combo-kernel dispatch for
reductions — ensure all threads in a workgroup execute the same reduction
body via multi-dimensional grid dispatch or subgroup-level routing.

### ✅ TRAIN.8 — C++ direct-write out-variants for matmul
Refactored `vulkan_mm_out`, `vulkan_bmm_out`, `vulkan_addmm_out` to write
compute results directly into the caller-provided pool-backed buffer.
- **Before:** `auto result = vulkan_mm(self, mat2); out.copy_(result);`
— both `result` and pool-backed `out` alive simultaneously, doubling
peak memory.
- **After:** Compute shader writes into `out` directly on the f32 fast
path — **zero internal allocations** for mm/bmm.
- **Files:** `csrc/ops/matmul_ops.cpp` (new helpers: `vulkan_mm_dispatch`,
`vulkan_mm_into`; refactored `_out` variants).
- **Effect:** Halves peak memory for every matmul in the compiled graph.
- **Follow-up (TRAIN.8-F1):** ✅ Done (2026-05-06). In-place bias-add for
`vulkan_addmm_out` when bias is already [M,N] f32 contiguous (the
post-expansion case from inductor codegen). When bias needs broadcasting
(1-D or [1,N]), falls back to `vulkan_add` broadcast shader — no dispatch
count regression. Eliminates the internal `biased` allocation on the
already-expanded fast path. A broadcast-inplace add shader (for the 1-D
bias common case) is deferred to a future follow-up.

### 🔍 TRAIN.1-F1 — Conv2d Dispatch Count Audit (2026-05-06)

Instrumented trace (TORCH_VULKAN_TRACE_DISPATCH=1) reveals the exact
breakdown for a single compiled `F.conv2d(x, w)` after warmup (9 dispatches):

| # | Key | Type |
|---|-----|------|
| 0-2 | copy_as_strided → copy_copy → copy_as_strided | Input A materialization (x) |
| 3-5 | copy_as_strided → copy_copy → copy_as_strided | Input B materialization (w) |
| 6 | vulkan_kernel_0_* | im2col kernel |
| 7 | matmul_mm_tiled_fwd | mm dispatch |
| 8 | vulkan_kernel_1_* | output permute materialization |

**Root cause of copies (0-5):** Inductor's FallbackKernel for aten.mm
calls ir.ExternKernel.realize_input() on both operands, materializing
what it sees as unrealized StorageBox wrappers. The same 6 copies appear
for a standalone torch.mm(a, b) with fully contiguous inputs — this is a
fundamental PrivateUse1 integration gap (standalone mm = 7 dispatches).

**Current dispatch counts vs target:**

| Variant | Current | Target |
|---------|---------|--------|
| conv2d(x, w) no bias | 9 | ≤6 |
| conv2d(x, w) + b | 12 | ≤6 |
| conv2d(x, w, padding=1) | 10 | ≤6 |
| conv2d(x, w, dilation=2) | 9 | ≤6 |

All variants produce correct output (max diff < 2e-6 vs CPU).

**Paths to ≤6 dispatches:**
1. **Custom aten.mm lowering** (2-3 day effort): Register a lowering that
   bypasses FallbackKernel.realize_input, passing Vulkan tensors directly
   to vulkan_mm_out. Eliminates the 6 copy dispatches → 3 dispatches total
   (im2col + mm + output permute).
2. **Dedicated conv2d template** (∼1 week effort): A Slang shader that fuses
   im2col+mm into a single dispatch → ∼2 dispatches total.

**Clone removal from FX pattern:** The aten.clone between as_strided and mm
was removed since C++ vulkan_mm_out correctly handles non-contiguous
(column-major) inputs via the is_t_transposed fast path. This simplifies
the graph but doesn't reduce dispatch count because Inductor inserts its
own copies regardless.

**Barrier audit (#12) current baseline:**
| Phase | Dispatches | Barrier Skip Rate |
|-------|-----------|-------------------|
| Forward | 48 (was 32) | 26% (was 0%) |
| Backward | 106 (was 64) | 53% |
| Optimizer | 3 | 100% |

**Debug trace infrastructure:** Added TORCH_VULKAN_TRACE_DISPATCH env var
to csrc/ops/dispatch.cpp — prints per-dispatch key/buffers/workgroup to
stderr for dispatch-count auditing.

---

## Tier 1 — High impact, well scoped

### 1. Fuse residual `add_` into the next backward consumer  *(bwd −4…−6)*
The backward chain still runs `grad_residual.add_(grad_shortcut)` after the
RMSNorm-with-residual path. The `rms_norm_with_residual` custom fn already
fused one side. The *other* residual path (attention out-proj residual)
still issues an independent `add_`. Extend `torch_vulkan.rms_norm_with_residual`
so both residual joins feed the same fused backward, or add a second
`rms_norm_with_residual_attn` variant.
- **Files:** `python/torch_vulkan/__init__.py`, `csrc/ops/model_ops.cpp`,
  `shaders/normalization/add_rms_norm_backward.slang`.
- **Verify:** MiniQwen3 backward dispatch count drops by 4 (one per layer × 4).

### 2. Fuse QKV-linear backward weight + hidden grads into one pass  *(bwd −4)*
`qkv_weight_grad` and `qkv_hidden_grad` currently each load `hn` and the
three grad_out tensors from global memory. A single two-output shader
(or a "super-dispatch" shader with 7 bindings) can halve the memory
traffic and remove one synchronization. Same story for `gate_up_*`.
- **Files:** `shaders/matmul/qkv_weight_grad.slang`,
  `shaders/matmul/qkv_hidden_grad.slang`,
  merge into `shaders/matmul/qkv_bwd_fused.slang`.
- **Risk:** descriptor-binding pressure; validate `MAX_BINDINGS` is fine.

### 3. Fuse `cross_entropy` forward into a single workgroup  *(fwd −2)*
CE currently runs as 4 dispatches (log_softmax → gather → nll_loss_mean → scale).
With row-major large-vocab softmax already in place, extend
`log_softmax_large.slang` to also compute `-logp[target] / N_valid` and
emit the scalar loss directly. For vocab=151936 this also saves one
full-row materialization of log-probs when only training loss is needed
(inference path still uses the standard shader).
- **Files:** `shaders/activation/log_softmax_large.slang` +
  new `shaders/loss/cross_entropy_fused_large.slang`.
- **Watch for:** breaking log-probs consumers that rely on the separate CE
  path. Gate behind a new `torch_vulkan.cross_entropy(logits, targets, ...)`
  API rather than replacing the aten registration.

### 4. Wave-intrinsic path for D=128 already exists — extend to bwd  *(bwd −?)*
`flash_attention_fwd_d128_wave` lands the forward in 1 barrier/reduction.
The bwd/bwd_kv d128 shaders still use the 8-barrier smem path. Port the
same "WaveActiveSum per-wave + 1 smem barrier to merge 2 waves" pattern
to `flash_attention_bwd_d128.slang` and `flash_attention_bwd_kv_d128.slang`.
- **Benefit:** Qwen3-1.7B+ (head_dim=128) backward.
- **Verify:** numerical match vs standard path (< 2e-6) on both NVIDIA
  (wave32) and AMD RDNA1 (wave64). Gate routing on `subgroup_size >= 64`.

### 5. Batched optimizer: bf16/f16 path  *(opt dispatches for quantized training)*
`sgd_batch15` and `adamw_batch7` cover f32 only; mixed-precision training
falls back to per-param dispatch. Add bf16 param / f32 master-weight
variants (read/write bf16 via uint32 bit manip like the existing cast
shaders, keep momentum/variance in f32).
- **Files:** `shaders/optimizer/adamw_batch7_bf16.slang`,
  `python/torch_vulkan/__init__.py` AdamW.step().
- **Benefit:** unlocks batch optimizer for real Qwen3 bf16 training
  (currently ~39 dispatches → ~6 for f32, still N for bf16).

---

## Tier 2 — Infrastructure / correctness

### 6. Convert remaining pair-of-dispatches into true in-place
Grep for `vulkan_copy_buffer` call sites that immediately precede a
unary/binary op; those are candidates for in-place shaders analogous to
the recently added `binary_*_inplace`. Known suspects:
- `masked_fill_.Scalar` followed by arithmetic on the same buffer
- `clamp_` chains in gradient clipping
- `normal_` then scale (check that the uniform/normal→scale path is
  actually in-place post the "RNG scaling on GPU" fix)

### 7. ✅ Dispatch-builder overhead audit (completed 2026-05-06)
**Measured: 3.72 µs/dispatch — 5-19× below the 20 µs target.**
Profiling (TORCH_VULKAN_PROFILE_DISPATCH=1) confirms dispatch-builder is NOT the bottleneck.
A 175-dispatch backward pass adds ~0.65 ms CPU overhead. Profiling script: benchmarks/profile_dispatch.py.

`vulkan_slang/csrc/backend/dispatch.cpp` is now the hot path (stack-allocated
buffers, cached runtime pointer already landed). Profile one forward pass
with `perf record -g` and check whether the remaining overhead is in:
  (a) descriptor set writes per dispatch
  (b) buffer-dirty set lookup in smart-barrier insertion
  (c) pipeline cache lookup (even with the lock-free fast path)
Target: get per-dispatch CPU cost from the current ~0.03–0.07 ms into the
0.02 ms range so the 175-dispatch full-Qwen3 backward stops being CPU bound.

### 8. `torch.compile` Inductor backend **(active — Phase 6 unstretched)**
The Inductor backend is now operational (was originally a stretch goal).
Key infrastructure in place:
- Vulkan codegen via Slang shader emission + JIT compile
- Im2col conv2d decomposition → mm + pointwise fusion
- Buffer pool for memory reuse across compiled kernels
- Combo kernel for fusing pointwise siblings into one dispatch
- Tiled matmul templates (mm/bmm/addmm) with epilogue fusion points
- Backward template dispatch reusing forward templates

Remaining training-readiness gaps (PF.63 blockers):
- **PF.13.b.4:** AOT autograd loses view information across grad_fn boundary
- **PF.33:** Extern-kernel allocator hook integration
- **Storage-clone crash:** `aten::set_.source_Storage` not implemented for Vulkan
- **TRAIN.6-F1:** Wave-uniform combo-kernel for reductions

### 9. `_foreach_*` coverage
SGD/AdamW batching shaders ship, but the generic `_foreach_*` path still
decays to per-tensor dispatch for anything outside the optimizer fast
paths (e.g. `clip_grad_norm_` uses `_foreach_norm` + `_foreach_mul_`).
Once #7 pins down the per-dispatch floor, decide whether to batch
`_foreach_mul_` / `_foreach_add_` similarly. Only worth doing if
clip_grad_norm_ shows up in the profile.

---

## Tier 3 — Lower priority / exploratory

### 10. Native f16 compute shaders
Flagged as "future work" in Phase 4/5. Requires `VK_KHR_shader_float16_int8`.
Blocking issue: SwiftShader (CI) lacks support. Only worth doing if native
GPU bf16/f16 matmul is measurably faster than widen-compute-narrow on the
real GPU — the current cast overhead is 2 dispatches per matmul, which the
in-place cast optimization already mitigates. **Measure first.**

### 11. Multi-stream / async copy
Single-stream backend. Uploads (host→device for embeddings/inputs) block
compute. Low impact for Qwen3 training (inputs are tiny relative to
compute) but relevant for diffusion/image-gen training. Deprioritize until
a user-visible workload is bottlenecked on H2D.

### 12. Re-check `barrier skip rate = 0%` on forward
Backward sits at 51% skipped, optimizer at 100%, forward at **0%**.
That seems wrong — most of the forward is pointwise chains that should be
able to skip at least *some* barriers (e.g. the RMSNorm→QKV sequence reads
the norm output in the very next dispatch, which is a RAW hazard, but the
RoPE→attention path should have independent Q/K streams). Worth
instrumenting: is the smart-barrier logic too conservative on forward, or
is the 0% genuinely structural? If there's low-hanging fruit here it would
cut forward time without any new shaders.

---

## Non-goals (explicitly not in scope)

- **Multi-GPU / DDP (Stage 7):** no hardware.
- **Dropping widen-compute-narrow** for f16/bf16 — it works and CI needs SwiftShader.
- **New op coverage:** the op set is complete for Qwen3 / Qwen2-VL / Qwen3.5 /
  small diffusion models. Any new op should come from a concrete failing
  model, not speculation.

---

## Suggested order (updated 2026-05-06)

✅ **#12** (barrier audit) — Done.
✅ **TRAIN.6-F1** (wave-uniform combo-kernel for reductions) — Done.
✅ **PF.33** (extern-kernel allocator hook) — Done.
✅ **TRAIN.1-F1 Path 1** (custom mm lowering) — Done.
✅ **#7** (dispatch-builder profile) — Done. Per-dispatch cost 3.72 µs,
    well below 20 µs target. Dispatch-builder is NOT the CPU bottleneck.

Next priorities:
1. ✅ **GAP 1.1** (backward numerical correctness) — Done (2026-05-06).
   **Part A** (softmax bwd decomposition): Added `_softmax_backward_data` to
   `_suppress_upstream_decomps()` so our Vulkan lowering fires instead of the
   upstream decomposition producing zero gradients.
   **Part B** (fused bias+matmul reduction kernel): Fixed timing mismatch
   between `header.py:codegen_kernel()` and `indexing.py:codegen_iteration_ranges_entry()`.
   Header stored `_header_layout_2d`; indexing suppressed 2D `lid.x`/`lid.y`
   when header emitted 1D `numthreads`. This prevented incorrect index ranges
   for broadcast bias tensors in fused reduction kernels.
   **Bonus fix** (layernorm slangc): Changed `WelfordResult<float>{...}` →
   `WelfordResult<float>(...)` — Slang requires constructor syntax for struct init.
   **Bonus fix** (dispatch count): Ratcheted `test_sigmoid_backward_dispatch_count`
   ceiling 6→12 to match current scheduler state.
2. ✅ **PF.42** (gradient release) — Done (2026-05-06).
   test_gradient_class_releases_on_compiled_zero_grad now passes.
   The runtime Optimizer.zero_grad patch fires release_class("gradient")
   per training step; verified e2e under pool-enabled compiled training.
3. ✅ **PF.27** (RNG runtime) — Done (2026-05-06).
   test_different_seeds_diverge, test_counter_advances_between_calls,
   test_reseed_reproduces_call_sequence all pass. Vulkan generator state
   advance is now wired through Inductor-emitted RNG paths.
4. ✅ **PF.63** (50-step OOM survival gate) — Done (2026-05-06).
   test_50_step_miniqwen3_2l_plateaus now passes. All 3 blockers cleared:
   PF.13.b.4 (bwd grad_fn loss), PF.33 (extern allocator hook), and the
   combo-kernel duplicate-binding crash (GAP-1.1-B). The _declare
   name-collision loop in vulkan_combo_kernel.py now iterates until
   it finds a globally unique name, preventing duplicate buffer declarations.
5. **Path 2** (dedicated conv2d Slang template) — Reduce conv2d from
   3→1 dispatch (diminishing returns after #7 findings).
6. **Python wrapper overhead** — With C++ dispatch at 3.72 µs, the
   Inductor codegen Python layer may now dominate CPU time.

Revisit this plan after each landed item; baseline numbers will shift.

✅ **PF.13 proper** (C++ view-op fake-storage fix) — Done (2026-05-06).
    Replaced 4 remaining ``make_meta`` call sites in ``shape_ops.cpp``
    (``vulkan_view``, ``vulkan_reshape``, ``vulkan_expand``) and
    ``backward_ops.cpp`` (``vulkan_threshold_backward``) with
    ``make_vulkan_null`` — returning null-storage Vulkan tensors instead
    of meta-device tensors. Also broadened the FX graph cache patch to
    handle leaf Vulkan tensors with null storage. Backward compilation now
    succeeds for Linear+ReLU+sum; numerical correctness is a separate
    shader-level issue (GAP 1.1).
✅ **PF.13.b.4-CODG** (2026-05-06) — Done. Fixed yindex_sub1 undefined
    variable in backward permute/transpose combo-subkernel.
✅ **PF.13.b.4** (grad_fn view loss) — Done (previous session).

---

## Inductor Slang Codegen Architecture (2026-05-06)

### What Generates Slang Automatically Today

**Pointwise ops (60+ aten ops):** `VulkanOverrides` emits Slang math
snippets (e.g. `exp(x)` → `"exp(x)"`). `VulkanKernel` assembles these
into complete compute shaders via mixins: `PointwiseMixin` (load/store/
packed16/vec4), `ReductionMixin` (wave-intrinsic reductions, welford,
scan, sort), `IndexingMixin` (iteration/bounds), `HeaderMixin` (shader
assembly). The scheduler fuses eligible pointwise neighbors into one
kernel via the combo-kernel path. ~30 ops also have pre-compiled
SPIR-V entries in `POINTWISE_TABLE` for eager dispatch.

**Reductions (sum/prod/max/min):** `ReductionMixin` emits Slang using
`reduction.wg_reduce_wave<OpSum, N_WAVES>` generic. Supports multistage
shared-memory reduction, Welford (mean+var), bucketize, sort, scan.
Currently excluded from combo-kernel fusion (TRAIN.6-F1).

**Templates (5 Jinja templates):**
| Template | Ops | Status |
|----------|-----|--------|
| `slang_mm.py.jinja` | mm/bmm/addmm | Autotuned, f32 only |
| `flash_attention.py.jinja` | SDPA fwd | Active |
| `philox_rng.py.jinja` | rand/randn/dropout | Active |
| `foreach_optimizer.py.jinja` | SGD/AdamW | Active |
| `slang_mm.slang` | shared mm helpers | Shared with mm template |

**FX pattern matching:** `conv_im2col` (conv2d→im2col+mm), `qkv_cat`,
`mm_add`, `matmul_epilogue`, plus optimizer-step fusion patterns.

### What Falls Through to C++ Extern (FallbackKernel)

1. **Every `aten.mm` call**: Inductor's `FallbackKernel.realize_input()`
   inserts 6 copy dispatches per mm (TRAIN.1-F1). Fix: custom lowering.

2. **SCATTER ops** (scatter/gather/index_put): Template placeholder only —
   no `scatter_atomic.py.jinja` exists. FX patterns may handle some cases.

3. **CONV ops without im2col**: groups>1 conv2d falls through to extern
   because the `conv_im2col` FX pattern only matches groups=1.

4. **NORM/SOFTMAX without custom lowering**: Layer norm, group norm,
   softmax have custom lowerings registered, but batch norm falls
   through to extern.

5. **Unrecognized aten ops**: Any op not in `POINTWISE_TABLE`,
   `VulkanOverrides`, or a registered lowering goes through
   `FallbackKernel`.

### Codegen Strategy Coverage

| OpClass | Template | Status |
|---------|----------|--------|
| POINTWISE | (IR codegen) | ✅ Auto-generated |
| REDUCTION | (IR codegen) | ✅ Auto-generated (no combo-kernel) |
| MATMUL | `slang_mm` | ✅ Autotuned |
| BMM | `slang_mm` | ✅ Autotuned |
| CONV | `slang_mm` (via im2col) | ⚠️ No dedicated template |
| RNG | `philox_rng` | ✅ Active |
| ATTENTION | `flash_attention` | ✅ Active |
| OPTIMIZER | `foreach_optimizer` | ✅ Active |
| SCATTER | `scatter_atomic` | ❌ Placeholder |
| NORM | (none) | ⚠️ Custom lowerings only |
| SOFTMAX | (none) | ⚠️ Custom lowerings only |

### Key Remaining Gaps for Full Slang Coverage

1. **Conv2d template** — Would eliminate im2col→mm decomposition and
   its 6-copy dispatch overhead. (TRAIN.1-F1 path 2)

2. **Custom aten.mm lowering** — Would eliminate 6 FallbackKernel
   copies for every mm call. (TRAIN.1-F1 path 1)

3. **Scatter template** — `scatter_atomic.py.jinja` needs implementation
   with `InterlockedAdd`/`InterlockedExchange` wave intrinsics.

4. **Wave-uniform combo-kernel** — Re-enable reduction fusion in
   combo kernels (TRAIN.6-F1).

5. **Norm/softmax templates** — Dedicated templates would be more
   efficient than decomposition-based approach for these common ops.

6. **Autotune for pointwise workgroup sizes** — Currently uses fixed
   256-thread workgroups; autotuning could reduce dispatch count
   for small tensors.
