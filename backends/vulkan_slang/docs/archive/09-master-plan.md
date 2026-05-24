# 09 — Master Plan: Inductor Backend Codegen

## 1. Context

The Vulkan/Slang backend is feature-complete on the eager side (1864 tests,
413 shaders, 122/127 primtorch ops, zero CPU fallbacks). The full Inductor
hook surface — `VulkanScheduling`, `VulkanPythonWrapperCodegen`,
`VulkanInterface`, `VulkanDeviceOpOverrides`, FakeTensor meta patches,
SPIR-V + `VkPipelineCache` disk caches — is wired up and 46
`test_stage8_compile.py` tests pass. What remains is moving Inductor from
"parity with eager" to "beats eager":

1. Backward graphs fall back to eager under `torch.compile`, so
   training-step fusion is unreachable.
2. fp16/bf16 generated kernels widen-compute-narrow inside Slang; the
   eager side already ships 16+ `*_packed16.slang` shaders that pack two
   halves per `uint32_t` and reduce dispatches 10–50% on CNN models.
3. Extern kernels (mm/conv/attention) end fusion — `linear + bias + gelu`
   becomes 3 dispatches instead of 1, despite fused eager-path kernels
   existing (`mm_tiled2_bias`, `batch_norm_gelu`, `batch_norm_relu`,
   `add_relu`, `add_rms_norm`) and an FX pass `vulkan_fuse` already doing
   this at the graph level.
4. [codegen.py](../python/torch_vulkan/inductor/codegen.py) is 1446 lines
   and multiple upcoming workstreams want to touch non-overlapping regions.

This plan sequences those fixes alongside codegen refactor, meta-kernel
completeness, multi-axis reduction fusion, perf counters, and two stretch
workstreams (AOTI + content-addressable SPIR-V).

## 2. Workstreams

Workstreams are listed in execution order (see §3 for the ordering
rationale). Each calls out concrete files, line ranges, and eager-path
utilities to reuse.

---

### W1. Codegen refactor / split

**Motivation.** [codegen.py](../python/torch_vulkan/inductor/codegen.py) is
1446 lines vs. upstream `torch/_inductor/codegen/mps.py` at ~1200 lines —
but the MPS codegen is already split across 4 sibling files. Several later
workstreams each want to touch non-overlapping regions of `codegen.py`;
split first to keep subsequent diffs clean.

**Tasks.**
- Split [python/torch_vulkan/inductor/codegen.py](../python/torch_vulkan/inductor/codegen.py) into:
  - `expr_printer.py` — `VulkanExprPrinter` (current lines 86–153, ~70 lines)
  - `overrides.py` — `VulkanOverrides` + `DTYPE_TO_SLANG` + `value_to_slang` (lines 59–440)
  - `kernel.py` — `VulkanKernel` (lines 447–~1370)
  - `scheduling.py` — `VulkanScheduling` (lines 1378–1446) plus the extern-epilogue extensions from W5
- Keep `codegen.py` as a thin re-export shim so the import in
  [__init__.py:26](../python/torch_vulkan/inductor/__init__.py#L26)
  (`from .codegen import VulkanScheduling`) still resolves.
- Move `_emit_helpers` (line 986) into `slang_helpers.py` — purely strings
  + assembly. Makes the W4 packed16 helpers drop-in.
- Consolidate env kill-switches (`TORCH_VULKAN_NO_WG_TUNE`,
  `TORCH_VULKAN_NO_LOAD_HOIST`, plus the new `TORCH_VULKAN_NO_PACKED16`
  and `TORCH_VULKAN_NO_EXTERN_EPILOGUE`) into
  `python/torch_vulkan/inductor/config.py`.
- Pure refactor — behaviour-preserving. Full `tests/test_stage8_compile.py`
  + `agent_space/test_inductor_broad.py` + `agent_space/test_inductor_ops_sweep.py`
  pass identically before/after.

**Files touched.** New: `expr_printer.py`, `overrides.py`, `kernel.py`,
`scheduling.py`, `slang_helpers.py`, `config.py`. Modified: `codegen.py`
(reduced to shim).

**Reuse.** `torch/_inductor/codegen/mps.py` is the layout reference.

---

### W2. FakeTensor meta-kernel completeness

**Motivation.**
[meta_patches.py:169](../python/torch_vulkan/inductor/meta_patches.py#L169)
installs Python fake-impls for only 14 ops (`_OP_IMPLS`). Under
`torch.compile` any unregistered op hits `RuntimeError: Cannot access data
pointer of Tensor (e.g. FakeTensor ...)` — exactly the whack-a-mole failure
mode historically seen on batch_norm / layer_norm / conv / SDPA / pool ops.

**Tasks.**
- Enumerate missing ops: flip
  [tests/test_stage8_compile.py](../tests/test_stage8_compile.py) to
  `backend="inductor"` (see W3) and collect every `NotImplementedError` /
  FakeTensor data-pointer error. Also run
  [tests/test_prims_ops.py](../tests/test_prims_ops.py) under compile.
- Add fake-impls to
  [meta_patches.py::_OP_IMPLS](../python/torch_vulkan/inductor/meta_patches.py#L169)
  for high-priority candidates: `scatter`, `scatter_add_`, `gather`,
  `index_put_`, `repeat_interleave`, `constant_pad_nd`,
  `convolution_backward_overrideable`, `native_batch_norm_backward`,
  `native_layer_norm_backward`, `_scaled_dot_product_flash_attention_backward`,
  `as_strided_scatter`, `masked_scatter`, `upsample_bilinear2d_backward`,
  `upsample_nearest2d_backward`, `_fft_r2c`, `_fft_c2c`, `_fft_c2r`,
  `linalg_svd`.
- Meta impls for fused custom ops (`torch_vulkan.swiglu`,
  `torch_vulkan.rms_norm`, `torch_vulkan.flash_attention`,
  `torch_vulkan.cross_entropy`, `torch_vulkan.add_rms_norm`,
  `torch_vulkan.fused_linear`, `torch_vulkan.batch_norm_gelu`,
  `torch_vulkan.batch_norm_relu`, `torch_vulkan.add_relu`,
  `torch_vulkan.scaled_bmm`) so users composing them with `torch.compile`
  don't break.
- Prefer C++ meta impls in
  [csrc/backend/MetaKernels.cpp](../csrc/backend/MetaKernels.cpp) where
  cheaper than Python — that file already has meta impls for `empty_like`,
  `zero_`, `fill_.Scalar`, `fill_.Tensor`, `set_.source_*` — extend the
  same pattern.
- Add `tests/test_inductor_meta_kernels.py`: instantiates each op under
  `FakeTensorMode`, checks shape/dtype match eager.

**Files touched.**
[python/torch_vulkan/inductor/meta_patches.py](../python/torch_vulkan/inductor/meta_patches.py),
[csrc/backend/MetaKernels.cpp](../csrc/backend/MetaKernels.cpp),
new `tests/test_inductor_meta_kernels.py`.

**Reuse.** `at::native::*_meta` helpers already exist for most standard
ops — call into them rather than re-implementing shape inference.

---

### W3. Backward fusion under `torch.compile`

**Motivation.** Training-step fusion is the biggest unclaimed ROI target.
CLAUDE.md currently ends benchmark discussion with "Backward + optimizer
still run eager since torch.compile backward on PrivateUse1 is blocked."
Root cause per CLAUDE.md: `DispatchKey::Undefined` hitting
`_refs._reshape_view_helper` during AOT Autograd functionalization +
view-helper pass.

**Tasks.**
- **Audit the exact failure.** Run
  `agent_space/bench_inductor_training_full.py` with
  `TORCH_LOGS="+aot,+dynamo,+inductor"` and
  `torch._dynamo.config.verbose=True`. Capture the first
  `DispatchKey::Undefined` raise + the op stack.
- **Add `CompositeExplicitAutograd` kernels** in
  [csrc/backend/MetaKernels.cpp](../csrc/backend/MetaKernels.cpp) for
  every op that AOT Autograd calls on undefined tensors during
  view-helper lowering. The pattern exists: `MetaKernels.cpp` already
  registers `set_.source_Storage_storage_offset` /
  `set_.source_Tensor_storage_offset` at `CompositeExplicitAutograd` for
  this reason. Likely additions (verify from trace): `_reshape_alias`,
  `_unsafe_view`, `view`, `expand`, `squeeze`, `unsqueeze` — all
  SymInt-aware via
  `set_sizes_and_strides(SymIntArrayRef, SymIntArrayRef, SymInt)`.
- **Prune `AutogradPrivateUse1` registrations** in
  [Registration.cpp:1736](../csrc/backend/Registration.cpp#L1736)
  (`TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m)`). The comment at
  line 1738 already notes these "... break torch.compile (FakeTensor
  dispatch ...)". Keep only the genuinely-needed subset — `max_pool2d`,
  `convolution`, `sdpa`, `prelu`, `selu`, `clamp`, `batch_norm_train`,
  `rms_norm`, fused SwiGLU family. Every unnecessary autograd
  registration is an opaque fusion boundary for AOT Autograd.
- **Backward-helper Meta kernels.** Every op in
  [csrc/ops/backward_ops.cpp](../csrc/ops/backward_ops.cpp) needs a
  matching `TORCH_LIBRARY_IMPL(aten, Meta, m)` block in `MetaKernels.cpp`.
  The existing `zero_` / `fill_` Meta registrations are the template.
- **Flip tests to `backend="inductor"`**. All 46
  [tests/test_stage8_compile.py](../tests/test_stage8_compile.py) tests
  currently use `backend="eager"` (see lines 121, 136, 147, …). Add
  `@requires_inductor_backward` marker; flip pure-forward tests to
  `backend="inductor"` and training-step tests to `backend="aot_eager"`
  first (validates meta kernels) then `backend="inductor"` once
  fused-backward works.

**Files touched.**
[csrc/backend/MetaKernels.cpp](../csrc/backend/MetaKernels.cpp),
[csrc/backend/Registration.cpp](../csrc/backend/Registration.cpp) (1736+),
[csrc/ops/backward_ops.cpp](../csrc/ops/backward_ops.cpp),
[tests/test_stage8_compile.py](../tests/test_stage8_compile.py),
[python/torch_vulkan/inductor/meta_patches.py](../python/torch_vulkan/inductor/meta_patches.py)
(expand `_OP_IMPLS` from W2 as new failures surface).

**Reuse.** The `set_.source_*_storage_offset` registration pattern in
`MetaKernels.cpp` is the template. The `torch_vulkan.cross_entropy` +
fused CE backward pair is proof that fused backward kernels work once the
path opens.

---

### W4. fp16/bf16 packed16 codegen

**Motivation.**
[codegen.py:66-70](../python/torch_vulkan/inductor/codegen.py#L66) maps
`torch.half → "float16_t"` and `torch.bfloat16 → "float"` (widened to f32
in-shader). Generated Slang operates on one half value per thread, pays a
cast round-trip, and wastes half the memory bandwidth. The eager path has
**16+ `*_packed16.slang` shaders** (`binary_add_broadcast_packed16`,
`reduction_sum_dim_strided_half`, `activation_log_softmax_half`,
`activation_softmax_half`, `binary_scalar_packed16`,
`conv_conv2d_fwd_dw_half`, `copy_permute_half`, `copy_select_half`,
`copy_expand_half`, `layer_norm_half`, `batch_norm_train`
dtype-parameterized, `sigmoid_backward_packed16`, etc.) that pack two
halves per `uint32_t` and delivered 10–50% dispatch reductions in eager.

**Tasks.**
- Add a **"packed16 mode" flag** to `VulkanKernel`. Eligibility: every
  iteration axis's innermost numel is even, every load/store index is a
  function of the innermost axis, dtype ∈ {f16, bf16}, no indirect
  indexing, no multistage reduction.
- **Module-scope pack/unpack helpers.** Copy the Slang bodies of the eager
  `unpack_lo`/`unpack_hi`/`pack_halves` functions (from
  `shaders/binary/add_packed16.slang` and
  `shaders/normalization/layer_norm_half.slang`) into `_emit_helpers`
  verbatim. They handle NaN/Inf/sign edge cases (CLAUDE.md documents the
  correct f16_to_f32 sign fix + f32_to_f16 NaN preservation + bf16 RNE).
- **`load()` path.** When packed16 is active, emit
  `float tmp = f16_from_word_lane(buf[word_idx], lane_idx);`.
- **`store()` path.** Two-thread cooperative pack + conditional write:
  `if ((lid.x & 1) == 0) buf[word_idx] = pack_halves(val_even, val_odd_from_wave_shuffle);`.
  Use `WaveReadLaneAt` for the cross-lane odd value in flat pointwise
  kernels; use a small `groupshared` staging array for multi-axis kernels.
- **bf16 native.** Extend to `uint16_t`-packed bf16 with IEEE 754 RNE cast
  (mirror `shaders/casts/f32_to_bf16.slang`).
- **Safety.** Disable packed16 when `self.multistage_reduction_entry` is
  set — shared-memory + pack-guard composition is fragile for v1. Re-enable
  after W6 lands.
- **Kill-switch.** `TORCH_VULKAN_NO_PACKED16=1` forces the old path (A/B,
  bug isolation).

**Files touched.** Post-W1 split: `overrides.py` (dtype map), `kernel.py`
(packed16 flag, load/store, codegen_kernel buffer type), `slang_helpers.py`
(pack/unpack helpers).

**Reuse.** Every `*_packed16.slang` in `shaders/` is already-validated
Slang whose bit tricks translate 1:1 to inline helpers. The eager test
suite ([tests/test_dtype_support.py](../tests/test_dtype_support.py),
[tests/test_math_correctness.py](../tests/test_math_correctness.py))
already exercises correctness of the packed16 shaders — if codegen'd
kernels match bitwise, correctness is inherited.

---

### W5. Extern-kernel epilogue fusion (mm / conv / attention + pointwise tail)

**Motivation.**
[codegen.py:17](../python/torch_vulkan/inductor/codegen.py#L17) explicitly:
"Extern kernels (matmul, conv, attention) fall back to the eager Vulkan
dispatch and remain outside Inductor's fused kernel graph for now." Cost:
`linear(x) + bias → gelu` → 3 extern calls instead of one fused kernel.
The eager side already has `mm_tiled2_bias`, `batch_norm_gelu`,
`batch_norm_relu`, `add_relu`, `add_rms_norm`, `FusedLinearBias`, plus the
`vulkan_fuse` FX backend (`_fuse_bn_gelu_in_graph`,
`_fuse_bn_relu_in_graph`, `_fuse_add_relu_in_graph`) — Inductor does not
see any of them.

**Tasks.**
- In `VulkanScheduling` (override `fuse` or `can_fuse_vertical`), detect
  the `ExternKernel → pointwise chain → write` pattern: single-output
  extern call feeding exactly one consumer buffer that is only read by
  subsequent pointwise ops, terminating in a store.
- **Phase A: extern + epilogue as 2 dispatches.** Extern kernel stays
  separate; add a second fused pointwise kernel that reads its output.
  Collapses N extern-tail pointwise dispatches into 1.
- Teach `VulkanPythonWrapperCodegen._generate_kernel_call_helper`
  ([wrapper.py:78](../python/torch_vulkan/inductor/wrapper.py#L78)) to
  emit the extern-epilogue pair as a single wrapper call preserving
  input→output ordering.
- **Phase B: route mm+bias+activation to fused eager kernel.** For the
  common pattern, call the eager `mm_tiled2_bias_fwd` shader already in
  [csrc/ops/matmul_ops.cpp](../csrc/ops/matmul_ops.cpp). The FX backend
  `_fuse_bn_gelu_in_graph` is the direct template.
- New module `python/torch_vulkan/inductor/extern_epilogue.py` for the
  pattern matcher + wrapper.

**Files touched.** Post-W1 split: `scheduling.py`,
[wrapper.py:78-107](../python/torch_vulkan/inductor/wrapper.py#L78).
New: `python/torch_vulkan/inductor/extern_epilogue.py`, test class
`tests/test_stage8_compile.py::TestExternEpilogueFusion`.

**Reuse.** `vulkan_fuse` FX backend in
[python/torch_vulkan/__init__.py](../python/torch_vulkan/__init__.py)
(search `_fuse_bn_gelu_in_graph`) is the working eager-path analog. Eager
fused kernels to call into: `mm_tiled2_bias`, `batch_norm_gelu`,
`batch_norm_relu`, `add_relu`, `add_rms_norm`.

---

### W6. Multi-axis reduction fusion + adaptive workgroup tuning

**Motivation.** `_pick_threadgroup_size` at
[codegen.py:492-514](../python/torch_vulkan/inductor/codegen.py#L492) only
tunes by reduction numel and only shrinks. It doesn't grow for wide
pointwise on large tensors (where 512-thread WGs win GMEM throughput),
doesn't consider dtype (packed16 halves the effective element count — a
128-element row is 64 `uint32`s, changing the optimal WG size), and
doesn't consider register pressure from the load-hoist cache at line 639
(the hard-coded 64 can spill for wide rnumel).

Multi-axis reductions — `sum(dim=(0, 2))`, `layer_norm(dim=(-2, -1))` —
are sequenced as separate reduction kernels by the current
`SIMDScheduling`. When an axis is small and total fits in a workgroup, a
single persistent kernel with a 2D thread layout beats two sequential
reductions.

**Tasks.**
- **Persistent multi-axis reduction.** In `_pick_threadgroup_size`, extend
  the decision logic: when `len(reduction_axes) == 2` and
  `prod(rnumels) <= 256`, pick a 2D (`THREADS_Y`, `THREADS_X`) layout
  matching axis shape. Emit `numthreads(X, Y, 1)` at `codegen_kernel`;
  drive `gtid.y` = inner reduction index, `gtid.x` = outer reduction
  index in one persistent workgroup.
- **2D wg_reduce helper.** Add
  `c10_vulkan_wg_reduce_{op}_2d(float val, uint2 tid, uint2 size)`: one
  `WaveActiveSum` across Y first (threads with `tid.x == 0` hold
  partials), stage into `groupshared float[Y]`, `WaveActiveSum` across
  them in wave 0. Keep the "1 barrier when n_waves>1" optimization
  (current helpers at codegen.py:1058-1086).
- **Shape-aware WG picker.** For pointwise, read `self.numels`. Small
  total (< 4K): stay at 256 (dispatch latency dominates). Huge (> 256K):
  bump to 512. Gate on `TORCH_VULKAN_NO_WG_TUNE` (already exists at line
  496). Pass `effective_numel = numel // 2` when W4 packed16 is active.
- **Dynamic load-hoist cache threshold.** Replace the hard-coded `64` at
  [codegen.py:639](../python/torch_vulkan/inductor/codegen.py#L639)
  (`if loop_size > 64`) with a register-pressure-derived value: 64 for
  wave64, 128 for wave32. Read from `self.simd_group_size` (already a
  class attribute at line 454).

**Files touched.** Post-W1 split: `kernel.py`, `slang_helpers.py`.

**Reuse.** `shaders/rnn/lstm_cell_fwd.slang` and
`shaders/normalization/batch_norm_train.slang` both use 2D thread
layouts — closest eager-path analogues for `groupshared` + `WaveActive*`
in 2D. The 256-thread wave-intrinsic helpers (codegen.py:1070-1086)
translate 1:1 to 2D with a product-of-waves sizing.

---

### W7. Per-kernel perf counters + regression baselines

**Motivation.** CLAUDE.md documents rich per-op perf counters on the eager
path (`_get_dispatch_count`, `_get_flush_count`,
`_get_preread_flush_count`, `_get_barrier_count`,
`_get_barrier_skip_count`). Inductor-generated kernels route through
`_jit_dispatch_cached` / `_jit_dispatch_cached_nopc`
([runtime.py:72-85](../python/torch_vulkan/inductor/runtime.py#L72)) via
the same `dispatch_shader_cached` path so totals are captured — but
**per-kernel** stats are not surfaced. Regression catching ("did my
refactor slow softmax at shape [64,16384]?") is a manual-stopwatch
exercise today.

**Tasks.**
- **Inductor-kernel registry.** `runtime.py` module-level
  `dict[key, InductorKernelStats]` where stats track `call_count`,
  `total_dispatch_us`, `last_shape_seen`, optionally per-shape buckets.
  Increment inside the specialised wrappers in `make_vulkan_kernel`
  (line 205-273) under `TORCH_VULKAN_INDUCTOR_STATS=1`.
- **Python API.** `torch_vulkan.inductor_stats.get_stats() -> dict` +
  `reset_stats()`.
- **Golden-number regression tests.** Commit
  `tests/inductor_bench_baseline.json` with per-shape-per-kernel dispatch
  counts from a known-good run. New
  `tests/test_stage8_compile_regression.py` runs the sweep from
  `agent_space/test_inductor_ops_sweep.py`, compares dispatch counts
  (not wall time — thermally noisy) to baseline with configurable
  tolerance. Fails on upward regression.
- **Bench integration.** Wire into
  `agent_space/bench_inductor_training_full.py` — dump stats after each
  benchmark step so the dispatch bottleneck is visible.

**Files touched.**
[python/torch_vulkan/inductor/runtime.py](../python/torch_vulkan/inductor/runtime.py),
new `python/torch_vulkan/inductor_stats.py`, new
`tests/inductor_bench_baseline.json`, new
`tests/test_stage8_compile_regression.py`,
`agent_space/bench_inductor_training_full.py`.

**Reuse.** The C++ perf counter API is the blueprint. Parallels
`torch.profiler` but lightweight — no profiler context required.

---

### W8 (stretch). `cpp_wrapper` / AOTI integration

**Motivation.**
[__init__.py:37-38](../python/torch_vulkan/inductor/__init__.py#L37)
passes `cpp_wrapper=None` and `fx_wrapper=None`. This means
`torch._export.aot_compile` cannot produce a standalone `.so` for Vulkan
models, blocking deployment scenarios without a Python interpreter.

**Tasks.**
- Sketch `VulkanCppWrapperCodegen` emitting C++ kernel-call glue analogous
  to `VulkanPythonWrapperCodegen._generate_kernel_call_helper`. Emitted
  calls resolve to `torch_vulkan::_C::_jit_dispatch_cached(...)` via
  existing pybind entries.
- `csrc/aoti_runtime.cpp` exports a C-ABI `vulkan_kernel_dispatch(...)`
  for the generated C++ to link against. Runtime is already half-ready:
  `_jit_dispatch_cached` takes pipeline pointer + tensor list + bytes —
  AOTI shim is glue, not new kernel infra.
- Smoke test `tests/test_aoti_vulkan.py`: compile a two-op forward, dump
  `.so` to tempdir, reload via `torch._export.aot_load`, compare to eager.

**Files touched.** New `python/torch_vulkan/inductor/cpp_wrapper.py`,
modified
[python/torch_vulkan/inductor/__init__.py:33-39](../python/torch_vulkan/inductor/__init__.py#L33),
new `csrc/aoti_runtime.cpp`, new `tests/test_aoti_vulkan.py`.

**Reuse.** `torch/_inductor/codegen/cpp_wrapper_gpu.py` is the upstream
reference. The `aoti-debug` skill is available when runtime glue fails
(expected pitfalls: device mismatch, constant loading, symbol visibility).

---

### W9 (stretch). Content-addressable SPIR-V cache keys

**Motivation.**
[runtime.py:112](../python/torch_vulkan/inductor/runtime.py#L112)
computes `hash_key = sha256(entry + "\n" + src)`. Any whitespace change —
e.g. Inductor renumbers `tmpN` because a preceding node was added —
invalidates the cache. Small codegen refactors also churn cached entries.

**Tasks.**
- Add a pre-hash canonicalizer in `runtime.py`: (a) strip single-line
  comments, (b) collapse whitespace, (c) rename `tmpN` variables in
  source-order so order-independent programs hash the same.
- Optional gate: `TORCH_VULKAN_SPIRV_CANONICAL_CACHE=1` so regressions
  are isolable.

**Files touched.**
[runtime.py:99-159](../python/torch_vulkan/inductor/runtime.py#L99)
(`compile_slang_to_spirv`).

**Reuse.** Triton does this trick. Low risk — worst case is cache miss +
recompile, never a correctness issue.

---

### W10. Primtorch coverage doc hygiene

**Motivation.**
[docs/primtorch_coverage.md:228-234](primtorch_coverage.md) marks
`fft_r2c`, `fft_c2r`, `fft_c2c` as ⚠ stubs but
[csrc/ops/fft_ops.cpp](../csrc/ops/fft_ops.cpp) exists,
[csrc/ops/ops.h](../csrc/ops/ops.h) exports the entry points,
[tests/test_fft_svd.py](../tests/test_fft_svd.py) has 47 passing tests,
and CLAUDE.md shows FFT ships with ~24 compiled shaders. Same for
`linalg_svd` (row 261, marked 0/1 but actually registered via
`prims::svd`).

**Tasks.**
- Run `pytest tests/test_fft_svd.py --timeout=300 -p no:faulthandler 2>/dev/null`
  to confirm passing status.
- Flip those rows to ✓, bump summary line 265 from 122/127 to the verified
  count (likely 126/127 or 127/127).

**Files touched.** [docs/primtorch_coverage.md](primtorch_coverage.md)
only. No code changes.

---

## 3. Priority ordering

1. **W1 — Codegen refactor** (1–2 days). Behaviour-preserving, unblocks
   clean diffs in W3, W4, W6.
2. **W2 — Meta-kernel audit** (1 day). Must run before W3 flips tests to
   `backend="inductor"`, otherwise W3 chases FakeTensor errors belonging
   to W2.
3. **W3 — Backward fusion under torch.compile** (3–5 days, highest ROI).
   Every subsequent workstream benefits from seeing backward kernels in
   the graph.
4. **W4 — fp16/bf16 packed16 codegen** (3–4 days, high ROI). Directly
   mirrors the eager-path packed16 win that produced 10–50% dispatch
   reductions.
5. **W5 — Extern-kernel epilogue fusion** (4–6 days, high ROI on CNNs).
   Depends on W3 for backward-side fusion, on W4 for half-precision
   epilogue correctness.
6. **W6 — Multi-axis reduction + WG tuning** (2–3 days).
   LayerNorm/GroupNorm + LN+activation chains benefit most. Deliver
   after W5 so they compose.
7. **W7 — Perf counters + regression baselines** (1–2 days). Last so
   golden numbers reflect the post-optimization state; otherwise baseline
   goes stale within a week.
8. **W8 — AOTI / cpp_wrapper** (stretch). Only if deployment is a
   near-term need.
9. **W9 — Content-addressable SPIR-V cache** (stretch). Quality-of-life;
   helps agent iteration speed.
10. **W10 — Primtorch coverage doc hygiene** (30 min). Independent; any
    time.

## 4. Verification

Common loop:
`bash tools/rebuild.sh && pytest tests/test_stage8_compile.py --timeout=300 -p no:faulthandler 2>/dev/null`.
Follow with `agent_space/` scripts for wall-clock.

### W1 (refactor)
Pure-refactor safety: diff the summary line of
`pytest tests/test_stage8_compile.py tests/test_prims_ops.py tests/test_stage4_training.py --timeout=300 -p no:faulthandler 2>/dev/null`
before and after. Must be identical.

### W2 (meta kernels)
`pytest tests/test_inductor_meta_kernels.py -x -vv` — every op's meta impl
returns a FakeTensor of correct shape/dtype. Full
`tests/test_stage8_compile.py` under `backend="inductor"` after W3 — zero
`RuntimeError: Cannot access data pointer`.

### W3 (backward fusion)
- `pytest tests/test_stage8_compile.py -k "training" -vv` — all
  training-step tests pass with `backend="inductor"`.
- `python agent_space/bench_inductor_training_full.py` — backward
  dispatch counts from `_get_dispatch_count()` drop vs eager baseline.
- Golden gradient parity: two 3-layer MLPs + one small ResNet, eager vs
  compiled, max-elementwise-grad-delta < 1e-5 fp32, < 1e-3 fp16.

### W4 (packed16)
- `pytest tests/test_dtype_support.py` under compile — same tolerances as
  eager (1 ULP RNE for bf16).
- `agent_space/bench_inductor_ops_sweep.py` on fp16 pointwise chains —
  dispatch count halves where eager-packed16 wins.
- `TORCH_VULKAN_NO_PACKED16=1` — dispatch counts regress back to
  baseline, confirming new path fires.

### W5 (extern epilogue)
- New `tests/test_stage8_compile.py::TestExternEpilogueFusion`
  (`linear + bias + gelu`, `conv2d + bias + relu6`, `sdpa + scale_add`)
  — dispatch counts drop, outputs within eager tolerance.
- `agent_space/bench_inductor_vs_eager.py` on ResNet: current "9% faster
  than eager" grows.

### W6 (multi-axis + WG tuning)
- LayerNorm `[64, 4096]` fwd+bwd — dispatch count drops (multi-axis
  fusion) vs current sequential. f32 and (post-W4) packed16.
- `agent_space/bench_inductor_hotpath.py` on `softmax [16, 4096]` +
  `softmax [4096, 32]` — latter gets an additional speedup beyond the
  current 60→38 µs because the shape-aware picker selects a tighter
  64-thread workgroup.

### W7 (counters + baselines)
- `python -c "import torch_vulkan; ...; print(torch_vulkan.inductor_stats.get_stats())"`
  after a bench — non-empty dict with reasonable counts.
- `pytest tests/test_stage8_compile_regression.py` — passes against
  committed baseline; artificially slow one kernel, test fails.

### W8 (AOTI)
`tests/test_aoti_vulkan.py` — compile, dump `.so`, reload, compare to
eager.

### W9 (canonical cache)
Add a whitespace-only change to an Inductor kernel; with flag ON, observe
cache HIT (no slangc subprocess in `TORCH_VULKAN_TRACE=1`). Without it,
MISS.

### W10 (primtorch doc)
`pytest tests/test_fft_svd.py` passes → flip doc rows to ✓, bump summary.

## 5. Out of scope (per user direction)

- Specific perf targets / golden-number wall times.
- Stage 7 DDP / multi-GPU (hardware unavailable per CLAUDE.md).
- New shader authoring for eager-only paths. Only new shaders introduced
  by this plan are reused module-scope Slang helpers for packed16 and 2D
  wg-reduce — both in Inductor codegen, not standalone `.slang` files.
- Eager-path op-code refactor (duplication cleanup, consolidation of 40+
  backward ops) — separate concern.

## 6. Critical files

- [python/torch_vulkan/inductor/codegen.py](../python/torch_vulkan/inductor/codegen.py) (primary target, splits in W1)
- [python/torch_vulkan/inductor/runtime.py](../python/torch_vulkan/inductor/runtime.py)
- [python/torch_vulkan/inductor/meta_patches.py](../python/torch_vulkan/inductor/meta_patches.py)
- [python/torch_vulkan/inductor/wrapper.py](../python/torch_vulkan/inductor/wrapper.py)
- [python/torch_vulkan/inductor/__init__.py](../python/torch_vulkan/inductor/__init__.py)
- [csrc/backend/MetaKernels.cpp](../csrc/backend/MetaKernels.cpp)
- [csrc/backend/Registration.cpp](../csrc/backend/Registration.cpp)
- [csrc/ops/backward_ops.cpp](../csrc/ops/backward_ops.cpp)
- [tests/test_stage8_compile.py](../tests/test_stage8_compile.py)
- [docs/primtorch_coverage.md](primtorch_coverage.md)
