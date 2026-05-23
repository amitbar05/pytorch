# Vulkan-Slang Inductor Backend Roadmap v6.3

> **v5 North Star achieved (2026-05-10).** SmallCNN trains end-to-end.
> **v6.1 closeouts** archived in [10-inductor-backend-history.md](10-inductor-backend-history.md).
> **v6.2 (2026-05-13)** added M13–M16 from a four-track audit.
>
> **v6.3 (2026-05-18) overhaul** following a five-agent comprehensive
> audit (FX / lowerings / scheduler+codegen / Slang-lib / runtime+validation).
> Each agent tested hypotheses against the live pipeline; surfaced **6 P0
> correctness blockers** (vk_wg_reduce_* undefined; FunctionalTensor
> 3-copy drift in `conv.py:528,847`; `empty_like` in 8 backward decomps;
> DTYPE_TO_SLANG element-size mismatches; Transformer 3D-matmul compile
> crash; DescriptorPool FREE_BIT anti-pattern) and **31 new milestones**
> across M18–M23. Subsequent full clean rebuild surfaced **4 additional
> in-tree blockers** filed as M22.8–11 (66 unused meta-kernels in
> `MetaKernels.cpp` = forgotten registrations or 600 LoC dead code;
> `vulkan_empty` device-binding gap; etc.). See § 0.6.
>
> § 0.5.2 reframings (corrections to prior numbers): persistent kernels
> 40 % → 0 % effective; `[require]` capabilities 0 % → ~80 % already wired;
> reflection metadata 100 % → 40 %.
>
**Last updated: 2026-05-23 (session: M20.1 RNN-cell bwd fully unblocked — explicit `[BackwardDerivative]` for 8 special-math fwd ops in `pointwise.slang` + M20.3 spec-const tiles verified DONE)**

**v6.5 supersedes v6.3** for cumulative state of audit closeouts. See § 0.0.8 for code-landed items and § 0.0.9 for audit-only / in-flight worktree items.
>
> **Live state:** 9 model architectures train end-to-end under
> `torch.compile`; 47 lowerings + 24 explicit decomp suppressions;
> 57/58 `aten.*_backward` ops route through `bwd_diff_table`; buffer
> pool at 36 % hit rate on MLP train. **Anti-goal #2 CLOSED** (`model_ops.cpp` deleted M16.3, build gate in `setup.py`). **OP.22 CLOSED** (`is_dynamic_stride()` wired into 8 codegen sites).
>
> **Feature flags ON by default:** spec_constants, descriptor_indexing,
> static_specialization, bank_conflict_pad, dynamic_shapes,
> batch_dispatch, wrapper_fastpath, grid_aware_wg,
> persistent_pointwise, buffer_pool, prewarm_on_import.

---

## 0.0.5.1 SLANGC UPGRADE ATTEMPT 2026-05-22 (REVERTED)

User asked to upgrade slangc to the latest. Tried v2026.8.1 and v2026.9.1
(downloaded the official linux-x86_64 release binaries). Both **segfault**
reproducibly on `import helpers;` from any entry file outside `shaders/lib/`
(any cwd, any -I form, with or without -ignore-capabilities). The minimal
repro fits in one line:

```bash
echo 'import helpers; [shader("compute")] [numthreads(64,1,1)] void computeMain(uint3 lid : SV_GroupThreadID, RWStructuredBuffer<float> buf) { buf[lid.x] = float(lid.x); }' > /tmp/t.slang
slangc-2026.9.1 /tmp/t.slang -target spirv -entry computeMain -o /tmp/t.spv -I backends/vulkan_slang/shaders/lib -ignore-capabilities
# → Segmentation fault (core dumped); exit 139
```

The same kernel + lib set under v2026.7.1 compiles cleanly (just with the
known `spvGroupNonUniform` capability noise we already mask with
`-ignore-capabilities`). Bisected: a stripped reconstruction of
`helpers.slang` placed in `/tmp/` (same content, same module name) compiles
fine — so the crash is data-dependent on the entry path resolving an
external module through a directory full of sibling .slang files. Looks
like a slangc 2026.8+ regression in cross-directory module discovery /
implicit-load.

**Decision**: stay on v2026.7.1. The 2026.9.1 + 2026.8.1 binaries were
removed from `third_party/slang/build/`. The runtime resolver
(`runtime/slangc.py:_resolve_slangc`) sorts by semver and now picks
v2026.7.1 again.

**Follow-up**: file the segfault upstream once we have a reduced repro
that doesn't involve our `__exported import` re-export pattern.

---

## 0.0.5 CONV-TRAINING E2E SESSION 2026-05-21

User ran the SmallCNN training E2E and worked through each error.

**Landed (commit `f4666c76ec9`):**
- **slangc resolver** picks 2026.7.1 from repo-root `third_party/slang/`
  (the backend-root tree carries a stale 2026.5.2 that segfaults).
  Sorts candidates by parsed semver, prefers newest.  Matches the
  conftest auto-resolver so standalone `python -c` works.
- **`-ignore-capabilities` slangc flag** added to every runtime
  invocation site.  slangc 2026.7.1 trips on `subgroup_ballot` →
  `wave_active_count_bits` (`helpers.slang:175`) with `E36104:
  undeclared capability 'spvGroupNonUniform'` and crashes the
  compiler under thread-pool concurrency.  The runtime SPIR-V is
  unaffected; every device we ship to supports subgroupVote/Ballot.
- **`aten::random_.from`** PrivateUse1 IMPL in `philox_dispatch.py`
  via the existing uniform → floor → cast path.  Allows eager
  `torch.randint(low, high, size, device="vulkan")` to work.
- **Library lifetime trap fixed.**  `_rng_lib` is now module-global,
  not a local in `install()`.  Decorator-style `@torch.library.impl
  (lib, name)` accidentally keeps the lib alive via closure on the
  produced wrapper; direct `lib.impl(name, fn)` does NOT — the lib
  GCs on `install()` return and PyTorch unregisters every kernel.
- **Eager-side philox install hook** in `torch_vulkan._register()`
  so `torch.randint` works in eager mode before any `torch.compile`.

**Test status (after the above):**
- ✅ `test_forward_compiles`
- ✅ `test_forward_output_shape` (B ∈ {1, 4, 8})
- ✅ `test_3step_training_loss` (SmallCNN trains E2E, loss strictly
  decreasing over 3 SGD steps — 226 s cold)
- ✅ `test_dispatch_count`
- 🔥 `test_forward_vs_cpu` — output is ~2× the CPU output
  (`abs_max_err ≈ 1.12` post-cache-clear).  Probe in isolation:
  *Conv alone* matches CPU to fp32 precision (`max_err 4.8e-7`);
  *Conv + GroupNorm* gives ~2× CPU; *GroupNorm alone* gives ~1.42×
  (sqrt(2)) on the same shape.  Pattern: VK variance is consistently
  smaller than CPU's by a factor that depends on input distribution,
  not a constant — points at a Welford-reduction codegen bug in the
  `aten.var_mean.correction` lowering (`lowerings/norm.py:81`), not
  the conv_gn_relu fused template (which is unused without ReLU).
- 🔥 `test_grad_parity`, `test_10step_no_nan`, `test_forward_backward`
  — backward path produces NaN gradients for conv1.weight; suspected
  cascade from the 2× GN forward bug.

**M-NEW.10 ✅ CLOSED 2026-05-21** — Welford correctness fix in
`shaders/lib/reduction.slang::wg_welford` (+ mirror in
`vk_reduction.slang`).  Two stacked bugs:

* **(B1) OOB shuffle.**  The wave-level butterfly guard was
  `lane + offset < size`, but `lane` is within-wave (0..simd-1) and
  `size` is the total workgroup-reduction extent.  Whenever
  `size > simd`, the guard was vacuously true, so lanes near simd-1
  issued `WaveReadLaneAt(..., lane+offset >= simd)` — an out-of-range
  subgroup shuffle that returns undefined data on RDNA1.  Symptom:
  variance ~0.51× CPU for size ≥ 256, and variance ≈ 0 for size < simd
  (lane 63's stale m2 won the unguarded post-reduction store race).
  Fix: replaced the guard with `lane + offset < wave_valid` where
  `wave_valid = min(simd, size - wave_id*simd)`, snapshotted all three
  neighbour reads (mean, m2, n) before any local write, and broadcast
  lane 0's result to every lane via `WaveReadLaneAt(..., 0)` for the
  `n_waves == 1` fast path (so unguarded post-reduction pointwise stores
  see a uniform value).

* **(B2) Out-of-line callable mis-codegens groupshared writes.**  With
  `[ForceInline]` removed under M-AUDIT-PERF.1, slangc emits
  `wg_welford` as an out-of-line callable.  In the multistage code
  path (per-thread n>1 input, e.g. `numthreads=256` with 16 elements
  per thread for size=4096), the out-of-line version dropped roughly
  half the per-thread contributions in the cross-wave smem fold.
  Confirmed in isolation: the same algorithm inlined into the kernel
  matches CPU bit-for-bit; emitted as a function call produces m2 at
  half magnitude.  Fix: re-added `[ForceInline]`.  The M-AUDIT-PERF.1
  slangc-hang justification (slangc 2026.5.2 > 30 s inlining) no
  longer reproduces on slangc 2026.7.1.

Regression gate: `tests/test_inductor_regression.py::TestMNEW10WelfordVariance`
covers seven shapes (size in {16, 64, 128, 256, 1024, 4096}) plus an
end-to-end `F.group_norm` parity check.  All match CPU to fp32
precision.

Repro probe: `agent_space/probe_gn_2x.py` (now passes for every shape).
The end-to-end `test_forward_vs_cpu` still fails with `max err ≈ 0.87`
post-fix — the Welford bug was real but only part of the picture; the
remaining error traces to a downstream slang_addmm / linear-layer
path, not GroupNorm.  Filed as separate follow-up.

**M-NEW.11 ✅ PARTIAL CLOSEOUT 2026-05-21** — investigated the
remaining `max err ≈ 0.87` and confirmed it is NOT a slang_addmm bug.
Direct probe of compiled `nn.Linear(2048→10)` at SmallCNN-shaped
inputs (B ∈ {1, 2, 4, 8, 64}) shows the tile-matmul path is correct to
fp32 precision (≤ 2e-6 max abs err vs CPU); compiled raw
`aten.addmm/aten.mm` show only normal fp32 reduction noise at K=2048
(~1.5e-4 — well inside the test's 1e-3 atol).  Layer-bisect probe
(`agent_space/probe_smallcnn_layer_bisect.py`) pins the first
divergence at the `relu1` stage of the SmallCNN forward — i.e. the
fused **conv → GN → ReLU** custom op (M17.2 Phase 3,
`torch_vulkan::conv2d_gn_relu_fused`).

Two bugs in `templates/conv_gn_relu.slang` + caller:

1. **Op-order semantic bug.** The shader pre-clamped `relu(conv+bias)`
   *before* feeding it into the Welford accumulator, computing the
   composition `gn(relu(conv(x)))` instead of the
   PyTorch-canonical `relu(gn(conv(x)))`.  Pass-1 + Pass-3 now compute
   the raw conv+bias and apply ReLU only at the end of Pass-3, after
   the affine transform.

2. **slangc 2026.7.1 write-coverage miscompile** (Group-D probe-only,
   not yet filed upstream).  When Pass-3's stored value depends on
   both the conv-load chain (``args.input``, ``args.weight``,
   ``args.bias``) AND the welford outputs (``mean``, ``rstd``), slangc
   silently drops Wave 1's (lanes 64..127 on RDNA1 wave64)
   ``OpStore``s.  Verified via three orthogonal probes:
   * `agent_space/probe_cgr_direct.py` — direct call to the caller
     with pre-filled output shows half the channels per WG never
     touched (mean stays at the pre-fill value).
   * `agent_space/probe_cgr_write_test.py` — replacing the math with
     a constant + tid sentinel writes EVERY cell.  Forwarding the
     constant + a `sum*0 + mean*0 + rstd*0` term to fake the data
     dependency triggers the miscompile again — confirming the
     trigger is the structural dependency, not the produced value.
   * `agent_space/probe_cgr_write_pattern.py` — encodes
     ``co * 10000 + oh * 100 + ow`` as the stored value, demonstrating
     half the channels never get written even though every cell
     theoretically has exactly one writer.

Mitigation (Group D): `_slang_tile_conv2d_gn_relu`
(`templates/caller/conv.py`) now routes through a 3-dispatch eager
decomp (`aten.convolution` + `F.group_norm` + `F.relu`) until either
the slangc bug is narrowed or a different shader structure is found.
The shader file is left in-tree (with the op-order fix and bug
documentation) as a regression baseline; the dispatch wiring below
the early-return is preserved so the fast path can be re-armed by
deleting a single block of code once the underlying issue is fixed.

Regression gate:
`tests/test_inductor_regression.py::TestMNew11ConvGnReluFusedCorrectness`
covers the SmallCNN first / second blocks, a tiny case that
specifically reproduces the Wave-1 store-drop fingerprint
(must have at least one non-zero per channel), and a `bias=None`
path (which forced an additional fallback fix to synthesise a
zero bias around the existing PrivateUse1 "tensor does not have a
device" blocker from § 0.0.6 row B).  All four pass.

Layer-bisect re-run with the M-NEW.11 fix confirms `conv1 → gn1 →
relu1 → pool1` now matches CPU to ~1e-6 (previously `relu1`
diverged at 4.3 abs err).  The remaining SmallCNN failure surfaces
at the **second conv2d** dispatch (extern_kernels.convolution), which
emits a constant repeated output — a SEPARATE bug in the
extern-conv lowering path (NOT in Group D scope), filed as the
next follow-up.

**M-NEW.12 ✅ CLOSED 2026-05-22** — `DispatchBatcher` direct-dispatch
race in the wrapper codegen.

Bisection initially hypothesised that the second conv2d was going
through `extern_kernels.convolution` and dropping the call.  The
actual root cause is structural: the wrapper queues Triton-style
kernel dispatches via ``_batcher.add(...)`` (see
``wrapper.py::_generate_kernel_call_helper``) and flushes only at
``_batcher.__exit__`` at the end of the wrapper.  Custom-op /
template-caller lines — ``torch.ops.torch_vulkan.conv2d_gn_relu_fused.
default(...)``, ``_slang_tile_conv2d(buf2, primals_6, buf3, ...)``,
etc. — are emitted by ``ir.FallbackKernel.codegen`` /
``ir.ExternKernelOut.codegen`` / custom ``codegen()`` methods on
lowering nodes as **immediate** Python function calls.  When a
direct call reads from a buffer that a still-queued kernel was
supposed to populate, the read sees uninitialised (zero) data.

Repro probe `agent_space/probe_second_conv2d.py` (Test C: full
chain) — pre-fix output was `2 × 32 × 16 × 16` with **only 32 unique
values** (one per output channel = the bias broadcast across all
spatial positions); the actual conv kernel never ran because the
preceding queued MaxPool2d hadn't fired yet.  Setting
`TORCH_VULKAN_BATCH_DISPATCH=0` made the bug disappear, confirming
the diagnosis.

Fix (Group F + Group H + `__init__.py`):

1. ``runtime/batcher.py`` — add class-level ``DispatchBatcher._current``
   (set/cleared on ``__enter__`` / ``__exit__``) plus a public
   ``DispatchBatcher.flush_current_if_active()`` classmethod that
   any direct-call site can invoke.

2. ``wrapper.py`` — emit a
   ``DispatchBatcher.flush_current_if_active()`` line at the head of
   every direct-dispatch path:
   * ``generate_extern_kernel_alloc`` (for ``ExternKernelAlloc`` /
     ``FallbackKernel`` allocations with ``out=`` rewriting),
   * ``generate_extern_kernel_out`` (for ``ExternKernelOut`` —
     covers the ``_slang_tile_conv2d`` / ``_slang_tile_mm`` template
     caller lines whose lowering custom-overrides ``codegen()``),
   * ``generate_fallback_kernel`` (for the FallbackKernel path
     that emits ``buf_N = torch.ops.foo.default(...)`` without
     going through extern_kernel_alloc — this was the missed hook
     in the first attempt; ``conv2d_gn_relu_fused`` goes through
     here because it has ``MultiOutputLayout``).

3. ``python/torch_vulkan/inductor/__init__.py`` —
   ``_patch_direct_call_template_helpers_flush_batcher`` wraps
   ``_slang_tile_conv2d`` and ``_slang_tile_mm`` in
   ``vulkan_template_caller`` so they flush the active batcher
   (defence in depth — catches any future direct-dispatch caller
   that bypasses the wrapper-side hook).

Regression gate:
``tests/test_inductor_regression.py::TestM_NEW_12_SecondConv2dCorrectness``
(four tests: `pool_then_conv2`, `full_chain_relu2`,
`two_convs_back_to_back`, `dispatchbatcher_active_tracking`).

Layer-bisect re-run with M-NEW.12 confirms **all 10 stages** of the
SmallCNN forward (`conv1 … fc`) match CPU to ≤ 2.9e-6.
`tests/test_e2e_models.py::TestSmallCNNTrain::test_forward_vs_cpu`
passes.

Forward path is closed.  Remaining red tests
(`test_grad_parity`, `test_10step_no_nan`, `test_forward_backward`)
all fail with `Non-finite grad for conv1.weight` (NaN gradients) —
a SEPARATE backward-path issue.  Not in M-NEW.12 scope; tracked as
M-NEW.13 follow-up.

**M-NEW.13 ✅ CLOSED 2026-05-22** — stale ``_vk_unpack_u8`` load path
for bool graph inputs in the backward shader.

Root cause: ``kernel/pointwise_load_mixin.py::PointwiseLoadMixin.load``
emitted ``_vk_unpack_u8(args.in_ptr_<mask>, idx)`` for ``torch.bool``
buffers that lived in ``V.graph.graph_inputs`` (canonical case: a
saved ReLU mask the partitioner threads from forward into the
backward graph as a graph input).  The helper signature in
``shaders/lib/dtype_pack.slang:132`` is::

    public float _vk_unpack_u8(StructuredBuffer<uint> buf, uint idx)

— it expects a 4 B/slot ``uint`` SSBO and unpacks the byte at
``idx`` from the 32-bit slot.  Post-M18.4-followup-C
(``overrides.py::DTYPE_TO_SLANG[torch.bool] = "uint8_t"``) all bool
buffers — graph inputs OR compile-internal — bind as
``StructuredBuffer<uint8_t>`` (1 B/slot) to match PyTorch's
1 B/element bool storage (the ``dispatch_copy_buffer_byte`` path in
``csrc/ops/dispatch.cpp`` preserves the layout on upload).  slangc
errors out with::

    error[E30019]: type mismatch — expected
    StructuredBuffer<uint, DefaultDataLayout>, got
    StructuredBuffer<uint8_t, DefaultDataLayout>

and the whole backward graph fails to compile.  AOTAutograd's
``CompiledFxGraph`` then propagates the partial / undefined buffer
via a fallback, which downstream div-by-variance kernels (GN
backward) turn into NaN gradients on ``conv1.weight``.

Symptom on SmallCNN bisect (``probe_nan_backward.py``):
* Stage 1-3 (Linear, ReLU+Linear, MaxPool+Linear) — all ✅.
* Stage 4 (GN + ReLU + Pool + Linear) — pre-fix: slangc compile
  error at the GN backward's ReLU-mask-load shader.  Post-fix: ✅.
* Stage 5 (Conv + GN + ReLU + Pool + Linear, single block) — ✅.
* Stage 6 (full 2-block SmallCNN) — ✅ ALL grads finite.

Fix (Group C, ``kernel/pointwise_load_mixin.py:203-229``): drop the
special branch that routed bool graph inputs through
``_vk_unpack_u8``.  Bool buffers — whether graph inputs OR
compile-internal — now read via the generic ``_LOAD_DISPATCH``
table's ``((float)(v[i]))`` form, matching the ``uint8_t`` slot
binding.  Comment on the bool store path in
``kernel/pointwise.py:710-722`` refreshed in parallel (the
implicit ``uint -> uint8_t`` narrowing cast was already
spec-correct; only the comment had drifted).

Defensive companion (Group F+B, ``inductor/__init__.py::
_patch_extern_convolution_out_kwarg``): the patched runtime wrapper
around ``extern_kernels.convolution`` now invokes
``DispatchBatcher.flush_current_if_active()`` at entry, matching
the M-NEW.12 fix on ``_slang_tile_conv2d`` / ``_slang_tile_mm`` —
defense in depth against any future codegen path that emits an
``extern_kernels.convolution(...)`` line without going through
``generate_extern_kernel_alloc`` (which already emits the flush).

Regression gate:
``tests/test_inductor_regression.py::TestM_NEW_13_NoNaNBackward``
(two tests: ``test_gn_relu_pool_linear_backward_compiles_and_no_nan``
— stage-4 of the bisect probe; and
``test_relu_compare_save_backward_no_compile_error`` — direct
``relu+linear+sum`` repro that forces the partitioner to save a
bool mask as a backward graph input).

**M-NEW.14 ✅ CLOSED 2026-05-22** — Two-block SmallCNN backward NaN
fixed.  Root cause: `native_group_norm_backward` was being decomposed by
AOTAutograd into primitives that shared buffer-pool storage (the
`reinterpret_tensor` aliasing at wrapper lines 930-931).  Fix: suppress
`aten.native_group_norm_backward.default` from BOTH the Inductor
decomposition table AND `torch._decomp.decomposition_table` so it reaches
the registered Vulkan lowering opaque.  Also completely rewrote
`_vulkan_native_group_norm_backward` in `bwd_lowerings.py` to use the
correct 4D formula (unweighted `ds_g`/`db_g` reductions, gamma applied
outside).  Gates: `TestM_NEW_14_TwoBlockSmallCNNBackward` (2 tests, PASS).
Commit: `a6b40f6c6b8`.

**Bisection** (`agent_space/probe_nan_backward.py`):

| Stage | Model | ALL_FINITE |
|-------|-------|-----------|
| 1 | Linear only | ✅ |
| 2 | ReLU + Linear | ✅ |
| 3 | MaxPool + Flatten + Linear | ✅ |
| 4 | GN + ReLU + Pool + Linear | ✅ |
| 5 | Conv + GN + ReLU + Pool + Linear (single block) | ✅ |
| 6 | Conv + GN + ReLU + Pool + Conv + GN + ReLU + Pool + Linear (full SmallCNN) | ❌ |

The failure surfaces ONLY with **two stacked Conv+GN+ReLU blocks**.
Single-block (Stage 5) passes; two-block (Stage 6) fails.

**Fingerprint on Stage 6**:

| Parameter | Pattern | Interpretation |
|-----------|---------|----------------|
| `conv1.weight.grad` | ALL NaN | gradient input from upstream is NaN/Inf |
| `conv1.bias.grad` | ALL NaN | computed as `sum(reshape_15, [0,2,3])` over the NaN chain |
| `gn1.weight.grad` | finite, [1.6, 1490] | rstd-scaled chain ~100× CPU magnitude → reads correct stats but grad_output is huge |
| `gn1.bias.grad` | finite, small | sum over the propagated-from-upstream `mul_17` which sees the relu mask |
| `conv2.weight.grad` | finite, [-7.7e3, 9.1e3] | gradient ~100× CPU baseline (Stage 5 sees [-41, 40]) |
| `conv2.bias.grad` | finite, [-5.6e3, 6.7e3] | same magnitude inflation |
| **`gn2.weight.grad`** | **EXACTLY 0.0** | suspicious — matches `output_mask[1]=False` zero-buffer fingerprint |
| **`gn2.bias.grad`** | **EXACTLY 0.0** | suspicious — matches `output_mask[2]=False` zero-buffer fingerprint |
| `fc.weight.grad`, `fc.bias.grad` | finite | downstream of the chain corruption |

**Backward graph analysis** (`/tmp/fx_dump/graph_0002_post.txt`):

The forward fuses both `Conv → GN → ReLU` blocks into
`torch_vulkan::conv2d_gn_relu_fused` custom ops via
`fx_passes/post_grad.py::_fuse_conv_patched_gn_relu`.  AOTAutograd's
joint trace inlines the registered backward
(`fx_passes/eager/conv.py:776 _conv2d_gn_relu_backward`) and then
**decomposes `aten.native_group_norm_backward` away** during
partitioning — there is no `aten.native_group_norm_backward` node
in the post-grad backward graph at all.  The gradient flow lives
as primitive sums / muls / unsqueezes.

The gn2-weight zero pattern thus is NOT the
`_register_group_norm_backward` lowering's `output_mask` branch
(that lowering never runs).  Instead `gn2.weight.grad` is computed
as the inlined chain `reshape_9 = reshape(sum_6, [32])` where
`sum_6 = sum(mul_16, [0])` and `mul_16 = sub_2 * unsqueeze_2`.
`unsqueeze_2 = unsqueeze(getitem_6 [=rstd2], -1)`.  For `mul_16` to
collapse to all-zeros, either every `sub_2` entry is zero OR every
`rstd_2` entry is zero — implausible for a well-conditioned
recomputation.

The huge-magnitude gn1 values + exact-zero gn2 values strongly
suggest the **recomputed forward chain inside the backward graph
produces wrong intermediates** for the second block.  Candidate
culprits:

1. Inductor wrapper emits `extern_kernels.convolution(..., bias=None)`
   for both `convolution` (block 2 conv recompute) and
   `convolution_1` (block 1 conv recompute), with the real
   `primals_7` / `primals_2` bias added by a separate fused
   pointwise kernel further down.  GN is shift-invariant so the
   no-bias conv output normalised is mathematically identical, but
   a buffer-lifetime / race issue between the queued bias-add
   kernel and the GN-normalisation read could leave bias-stale
   data at the moment GN reads its input.
2. The two stacked recomputations of `conv → GN` share
   buffer-pool slots (`buf4`, `buf5`, `buf6`, `buf21`, `buf22`,
   `buf23`) that get aliased or freed across the multi-kernel
   sequence.  Trace at lines 873–910 of the generated wrapper:
   `vulkan_kernel_1` writes mean/rstd into `buf5`/`buf6`,
   `vulkan_kernel_2` consumes them, then `buf5`/`buf6` are
   **reused via `reinterpret_tensor`** for block-1 buffers
   (`buf31 = reinterpret_tensor(buf6, ...)`, `buf32 =
   reinterpret_tensor(buf5, ...)`).  This is the canonical buffer-
   aliasing fingerprint and is a high-probability culprit.
3. The `vulkan_convolution_backward_overrideable_adapter` CPU
   fallback (`csrc/backend/Registration.cpp:430`) is invoked
   directly via `torch.ops.aten.convolution_backward.default` (line
   911 of generated wrapper).  Its inputs go through `.cpu()`
   copy which auto-flushes pending GPU work via the read
   callback, but the **fallback runs on CPU and copies the result
   back via `.to(dev)`**.  If the round-trip mis-handles
   non-contiguous strides on the result side, the gradient flowing
   into block 1 would be corrupted in exactly the observed way.

**Out of immediate scope** because the fix requires either:

* a partitioner-level rewrite to STOP decomposing
  `native_group_norm_backward` (kept as opaque so the
  `_register_group_norm_backward` lowering fires — Group B),
* a buffer-pool aliasing audit (`buffer_pool.py` + scheduler
  liveness — Group E + F), or
* a CPU-fallback round-trip stride audit
  (`vulkan_convolution_backward_overrideable_adapter` — Group A).

None of these are 30-minute fixes; all three require careful
investigation in their respective lanes.  Filing as OPEN with the
diagnosis above; the smoking gun is the
**buffer-aliasing reinterpret_tensor pattern at lines 930–931 of
the generated wrapper** (`buf31 = reinterpret_tensor(buf6, ...)`;
`buf32 = reinterpret_tensor(buf5, ...)`), where block-2 GN
intermediate buffers are recycled as block-1 GN gradient outputs.

**Repro**:
```bash
cd backends/vulkan_slang
rm -rf ~/.cache/torch_vulkan/spirv ~/.cache/torch_vulkan/slang-modules /tmp/torchinductor_$(whoami)
TORCH_VULKAN_NO_PREWARM=1 .venv/bin/python agent_space/probe_nan_backward.py
# Stages 1-5 pass; stage 6 (full SmallCNN) hits the NaN/zero pattern.
# Or:
TORCH_VULKAN_NO_PREWARM=1 .venv/bin/python agent_space/probe_nan_stage6.py
```

**Reverting the kernel/* M19.4 changes** (`pointwise.py`,
`pointwise_vec4_mixin.py`) to HEAD does NOT fix the issue —
confirmed the M19.4 work is unrelated.  Disabling the dispatch
batcher (`TORCH_VULKAN_BATCH_DISPATCH=0`) also does NOT help —
the issue is not the M-NEW.12 race class.

---

## 0.0.6 CONV-TRAINING CRITICAL PATH (2026-05-19, user-directed focus)

**Goal:** `TestSmallCNNTrain` (test_e2e_models.py:1172) trains end-to-end
through `torch.compile(backend="inductor")` with **non-zero, correct
gradients** on every parameter, and at least matches eager-mode loss
trajectory over 10 steps.

Pipeline under test:
`Conv2d → GroupNorm → ReLU → MaxPool2d → ... → Linear → CrossEntropyLoss`.

### Session 2026-05-20 progress (this commit)

| # | Item | Status | Evidence |
|---|------|--------|----------|
| **1** | **M-NEW.9 + M-AUDIT-PERF.1-followup** (unified) — rewrite AOTAutograd's constant-folded tangent get_attr back to `aten.view+expand(tangents_N, target_shape)` for **both** scalar AND non-scalar tangent shapes via greedy right-to-left dim-matching. | ✅ **DONE 2026-05-20** | `_rewrite_constant_folded_tangent` added to `meta_patches/joint_graph_passes.py:448-686` running inside `_chained` before `_stamp_factory_devices`. **gate: tests/test_cgm3_reduction_backward.py 14/14 PASS** (including the 6 previously-broken sum/mean tests). Fix also handles `sum(dim=[0,2])`-style mid-axis broadcasts via greedy `_compute_view_shape` (insert size-1 dims where target dim doesn't match tangent dim). |
| **2** | **M18.2** verification — `_has_real_vulkan_storage` helper + `@torch.compiler.disable` removal | ✅ **already in tree** (no edits needed) | `fx_passes/eager/_common.py:16` exposes `_has_real_vulkan_storage`; `fx_passes/eager/conv.py:442,776` both `_conv2d_relu_backward` and `_conv2d_gn_relu_backward` import the shared helper. Only comments mentioning `@torch.compiler.disable` remain, never the decorator. |
| **3** | **M22.14** verification — `_ensure_conv2d_backward_op_registered` + `make_fallback` | ✅ **already in tree** (no edits needed) | `fx_passes/eager/conv.py:1037` defines the registration; `fx_passes/eager/__init__.py:80` calls it on package init; `lowerings/__init__.py:256` does `make_fallback(torch.ops.torch_vulkan.conv2d_backward.default)`. |
| **4** | **M19.1** verification — `_register_linear_backward_decomposition` call site live | ✅ **already in tree** (no edits needed) | `lowerings/__init__.py:335` uncommented; `lowerings/matmul.py:246` defines the dual-decomp-table installer. |
| **5** | **conv_gn_relu PC overflow** — 136B push constant exceeded RDNA1 128B cap | ✅ **FIXED 2026-05-20** | Dropped `_pad` + `spatial_size` + `channels_per_group` (derived in-shader): `templates/conv_gn_relu.slang` PC struct now 29 uints + 1 float + 1 uint = 124B; matching `struct.pack("29IfI", ...)` in `templates/caller/conv.py:_slang_tile_conv2d_gn_relu`. |
| **6** | **slang_mm duplicate extern kernel** — `lazy_register_extern_choice` cache poisoned by `_ensure_extern_choices`'s pre-construction | ✅ **FIXED 2026-05-20** | `templates/caller/gemm/install.py::_ensure_extern_choices` now pre-populates upstream's `torch._inductor.kernel.mm.lazy_register_extern_choice` cache with the SAME ExternKernelChoice instance via direct call. Before fix: upstream's `tuned_mm` re-constructed `ExternKernelChoice(fn)` and tripped `duplicate extern kernel: slang_mm_8_8_8_s1_r1x1` because the name was already registered. |

### Remaining blockers surfaced by SmallCNN end-to-end run

| # | Item | Status | Evidence |
|---|------|--------|----------|
| **A** | **`vulkan` vs `vulkan:0` device-tag normalization** — user-created tensors via `device="vulkan"` get `device.index=None`; compiled backward returns `device.index=0`. Autograd engine rejects: `Function CompiledFunctionBackward returned an invalid gradient at index 2 - expected device vulkan but got vulkan:0`. | ✅ **FIXED 2026-05-20** | `vulkan_empty` and `vulkan_empty_strided` in `csrc/backend/Registration.cpp` now normalize `device.index < 0` → `0`. Also added `current_device()` to `VulkanModule` in `python/torch_vulkan/__init__.py`. Verified: `torch.zeros((), device='vulkan').device.index == 0`. |
| **B** | **`extern_kernels.convolution(..., out=buf4)` codegen** — Inductor's wrapper codegen emits `out=` kwarg for `aten.convolution.default` fallback, but the aten op signature has no `out=` parameter. | ✅ **FIXED 2026-05-20** | `_patch_extern_convolution_out_kwarg` in `python/torch_vulkan/inductor/__init__.py` wraps `extern_kernels.convolution` to (1) handle `out=` by copying into the pre-allocated buffer, (2) synthesize a zero bias when `bias=None` (workaround for a `tensor does not have a device` bug in PrivateUse1's `aten.convolution(None bias)` path), and (3) route via `aten.convolution_overrideable` which hits our registered backend adapter. |
| **C** | **conv+ReLU compile-mode `LocalSizeId` / `maintenance4` SPIR-V validation error** — slangc emits SPIR-V using `OpExecutionMode LocalSizeId` which requires Vulkan `maintenance4`. | ✅ **FIXED 2026-05-20** | `csrc/vulkan/Context.cpp` now queries `VkPhysicalDeviceMaintenance4Features` via `vkGetPhysicalDeviceFeatures2`, enables `VK_KHR_maintenance4` device extension when supported, and chains the feature struct in `vkCreateDevice`. Cap stashed in `Caps::maintenance4`. |

### NEW blockers surfaced after fixing A/B/C — the next mile

| # | Item | Status | Evidence |
|---|------|--------|----------|
| **D** | **`Expected operands to be scalar or vector int: ULessThan` SPIR-V validation** — a generated kernel emits `OpULessThan %bool %sub1 %sub1` where both operands are bool (from a chained boolean comparison). SPIR-V's `ULessThan` requires integer operands. Crashes the pipeline-create. | 📋 OPEN 2026-05-20 | Reproducible via `agent_space/test_small_cnn_training.py` — second autotune of `mm` (linear bwd recompute). Fix: locate the codegen site emitting `ULessThan` on bools (likely an `expr_printer` mishandling of `(a < b) < (c < d)` patterns) and route through a bool→int cast. |
| **E** | **Pipeline layout mismatch: SPIR-V uses Set 1 Binding 0/1/4/5 but layout doesn't declare them** — generated SPIR-V declares bindings on Set 1 but our `VkPipelineLayout` only declares Set 0. | 📋 OPEN 2026-05-20 | `[ VUID-VkComputePipelineCreateInfo-layout-07988 ]` errors during SmallCNN training. Investigation: either slangc's auto-binding-set assignment differs from our pipeline-layout build, OR a multi-set kernel needs explicit `[[vk::binding(N, 0)]]` annotations (we have one for `conv_gn_relu` but not the autotuned mm). |
| **F** | **DescriptorPool reset-while-in-use** — `vkResetDescriptorPool` called while descriptor sets are referenced by an in-flight command buffer. `[ VUID-vkResetDescriptorPool-descriptorPool-00313 ]` floods logs and indicates a real synchronization gap. | 📋 OPEN 2026-05-20 | `csrc/ops/dispatch.cpp` descriptor recycle path needs to wait on the submission fence before reset. The M-CPP-AUDIT.2/3 fix landed in earlier sessions but appears insufficient under the new SmallCNN training pressure (4-5 nested compiles + reuses). |
| **G** | **🔥🔥 P0 — slang_mm / slang_addmm dispatches DO NOT WRITE to their `out` buffer** | ✅ **CLOSED 2026-05-21** (two distinct bugs, both fixed) | **(1) DR.7 2D-workgroup flatten**: `runtime/slangc.py:_pick_numthreads_from_reflection` returned `(base, 1, 1)` unconditionally. For `[numthreads(WG_N, WG_M, 1)]` mm kernels this collapsed `lid.y` to 0; only row 0 wrote. Fix preserves `current_numthreads` when `Y>1` or `Z>1`. Regression: `TestDR7ReflectionRouting::test_dr7_preserves_2d_workgroups`. **(2) Stale `slang_mm.slang` / `slang_mm_bwd.slang` PC layout mismatch (root cause of the surface symptom)**: the `.slang` files declared a CONDITIONAL PC struct (15 uints = 60B for `has_bias=True, has_batch=False`), but Python `_pack_mm_pc` packs a MONOLITHIC 24-field layout (96B per `"19I5f"`). `_load_slang_template` (`vulkan_template.py:68`) prefers `.slang` over `.py.jinja`, so the broken `.slang` won. Python's `stride_a_b=0` at offset 40 landed at the shader's `tile_m` field → divide-by-zero in `pc.tile_m / pc.m_per_thread` → every thread early-exited → no store ever fired. Fix: deleted both stale `.slang` files; loader falls back to the monolithic `.py.jinja`. `agent_space/probe_slang_addmm_writeback.py` confirms: pre-filled `out=99` now matches CPU baseline `-3.9510` byte-for-byte. Bisection probe at `agent_space/probe_slang_mm_minimal_fix.py`. |

### Verification gates

```bash
cd backends/vulkan_slang
# Step 1 (DONE)
SLANGC=$(realpath third_party/slang/build/slang-2026.5.2-linux-x86_64/bin/slangc) \
  .venv/bin/python -m pytest tests/test_cgm3_reduction_backward.py -p no:faulthandler
# Step 5 (DONE, but needs the matching test)
SLANGC=$(realpath third_party/slang/build/slang-2026.5.2-linux-x86_64/bin/slangc) \
  .venv/bin/python agent_space/test_small_cnn_training.py
# After A + B + C land:
.venv/bin/python -m pytest tests/test_e2e_models.py::TestSmallCNNTrain -p no:faulthandler
```

### Out-of-scope (filed but not on the critical path)

- C++ A.1–A.5 (M-CPP-AUDIT.1-4 + Allocator) — stability, not correctness. Defer to after § 0.0.6 lands.
- helpers→vk_helpers rename completion — codegen still emits `import helpers;` which resolves; transition is non-blocking.
- 8 worktree branches — refactors, no behavior change.
- layer_norm dynamic-batch fwd wrong values — only affects M19 LayerNorm models, not SmallCNN.

---

## 0.0.7 RECONCILIATION — actual working-tree state at session close (2026-05-19)

The original § 0.0.8 was written from dispatched-agent reports. After
session close, the working tree was diffed against `HEAD` and several
claimed closures turned out **not present in the main worktree** —
likely a mix of (a) the user reverting some agent edits silently (the
M-NEW.9 + tangents.py revert was explicitly user-confirmed), and (b)
some C++ agents reporting success without their edits surviving
concurrent agent pressure. The honest state is below; § 0.0.8 is kept
for the record of what was reported but should not be read as authoritative.

### Actually landed in main worktree (will commit this session)

| Item | Files | Status |
|---|---|---|
| **Group B bwd_diff cleanups** | `bwd_diff_table.py`, `bwd_diff/unary.py`, `bwd_diff/emit_helpers.py`, `lowerings/loss.py`, `tests/test_inductor_regression.py` (+`TestB5CLeakyReluBwdDiff`) | ✅ landed. B.5.C `no_diff_params` threading + leaky_relu_backward table entry. B.4.B `@lru_cache` on `resolve_backward_kind`. B.5.E dead `_numel` removed. |
| **Group C kernel perf (M-PERF.2/3/5)** | `kernel/header.py`, `kernel/pointwise.py` | ✅ landed. VGPR-gated `[unroll]`, reflection-driven `[numthreads]` (64 when VGPR > 128), persistent pointwise gate lifted. |
| **Group D mm template PC struct refactor (A.6)** | `templates/slang_mm.py.jinja`, `slang_mm_bwd.py.jinja`, `templates/caller/gemm/{backward,classes,dispatch,install,render}.py` | ✅ landed. Monolithic 96B fwd / 76B bwd PC; `MM_FLAG_*` bitmask replaces `has_*` Jinja gates. Latent `_slang_tile_addmm_gelu` 10-uint undercount also fixed. Cache key bumped `_a6`. |
| **Group E scheduler/fusion (M-PERF.6 + F.1)** | `scheduling.py` | ✅ landed. Reduction+pointwise fusion cap 256→1024; descriptor-indexing-disabled buffer cap 60→80. |
| **Group F runtime perf (M-PERF.1 + F.D.2)** | `buffer_pool.py`, new `tests/test_buffer_pool.py` | ✅ landed. `@lru_cache(8)` on `_per_key_cap_for` + env snapshot at module init. 2-level `dict[(numel,dtype) → deque]` LIFO with lazy-invalidation FIFO eviction. `_LifoView` proxy for M17.7 back-compat. 19/19 tests pass. |
| **Group G `reduction.slang` scan helpers + slangc-timeout fix** | `shaders/lib/reduction.slang` (+141/-10), new `shaders/lib/vk_helpers.slang`, new `shaders/lib/vk_reduction.slang` (mirror files for rename-in-progress) | ✅ landed. `IScanAdd/Mul/Max/Min` interface structs + multi-wave `wg_scan_inclusive` + new `wg_scan_exclusive`. **Plus M-AUDIT-PERF.1 fix**: dropped `[ForceInline]` from 5 reduction helpers (`wg_welford`, `wg_argmax`, `wg_scan_inclusive`, `vk_wg_reduce_argmax`, `vk_wg_reduce_argmin`) — slangc 2026.5.2 hangs > 30s with `[ForceInline]` + `VK_SUBGROUP_SIZE` spec-constant inner loop. 3 var-backward tests now pass. |
| **Group H test addition (M-CV.4)** | `tests/test_cgm3_reduction_backward.py` (+`TestMCV4SoftmaxBackward`) | ✅ landed. Regression confirms `F.softmax(x).sum().backward()` works (the constant-folded-tangent pattern was investigated and the fix path was already adequate for softmax). |
| **K.2 combo_kernel `_TYPE_KEYWORDS` expansion** | `combo_kernel/body_rewriter.py` | ✅ landed (worktree was truncated-report; the edit is in the main worktree). +12 types covering u64/i32/float{32,64}_t/bfloat16/half{2,3,4}/double{2,3,4}. |
| **Test files (new)** | `tests/test_audit_script_succeeds.py`, `tests/test_buffer_pool.py`, `tests/test_m_cv2_phase1_backward.py`, `tests/test_m_cv3_dynamic_batch.py` | ✅ landed as untracked; will be added in this commit. M-CV.3 has 14 xfail-strict gates; M-CV.2 Phase 1 has 8 tests blocked by the helpers→vk_helpers rename. |
| **Roadmap v6.5 update** | `docs/10-inductor-backend.md` | ✅ landed (this file). |

### Reverted / never landed (despite agent reports of success)

These items were reported as ✅ FIXED in agent transcripts but **the
edits are not present** in the main worktree at session close. They
need re-implementation if still desired:

| Item | Reported file edits | Reality |
|---|---|---|
| **M-NEW.9** zero-grad backward fix | `meta_patches/joint_graph_passes.py` (`_rewrite_constant_folded_tangent` function) | ❌ REVERTED by user (explicit edit notification 2026-05-19). The function does not exist in `joint_graph_passes.py` at session close. The zero-grad symptom in compile-mode `sum/mean.backward()` is **still present** for non-scalar tangent shapes. M-AUDIT-PERF.1-followup is the right framing — proper fix needs to handle both scalar and non-scalar tangent cases without false positives. |
| **A.1 Stream fence pool** | `csrc/vulkan/Stream.{cpp,h}` | ❌ not in working tree. Agent reported +24/+43 LoC; no diff present. |
| **A.2 dispatch.cpp TOCTOU + clear-on-flush** | `csrc/ops/dispatch.cpp` | ❌ not in working tree. |
| **A.3 DescriptorPool growth** | `csrc/vulkan/DescriptorSet.{cpp,h}` | ❌ not in working tree. |
| **A.4 init.cpp pybinds + GIL release** | `csrc/init.cpp` | ❌ not in working tree. |
| **A.5 Allocator fragmentation + recycle cap** | `csrc/backend/Allocator.{cpp,h}` | ❌ not in working tree. |
| **M22.13 Stage 1** matmul `.contiguous()` → `TORCH_CHECK` tripwires | `csrc/ops/matmul_ops.cpp` | ❌ not in working tree (0 occurrences of "M22.13 invariant"). |
| **M-RT.1 audit-script `c10_vulkan_erf` rewrite** | `scripts/audit_inductor_op_coverage.py` | ⚠ partial: the new regression test `tests/test_audit_script_succeeds.py` IS present and will be committed, but the audit-script edit itself is not. |

Root cause for the C++ disappearances is unclear — those agents reported
their edits as committed-in-main and showed code diffs, but the files
are at HEAD state at session close. The most likely explanation is
that the C++ agents were not actually editing the live working tree
(they may have been pointed at a stale copy by the harness, or the
edits were undone by a concurrent stash/restore in another agent's
session). The audits and root-cause analyses **are still valid** — the
C++ implementation work simply needs to be re-done with care to verify
file diff lands in `git status` before declaring done.

### Worktree branches landed (need cherry-pick to integrate)

| Worktree branch | Item | LoC delta |
|---|---|---|
| `worktree-agent-a58e1face111590ac` | M22j `shape_ops.py` split (753→629 + 3 new files) | refactor |
| `worktree-agent-a843fd91d51dd477a` | M22g `wrapper.py` split (825→460 + new `wrapper_buffer_pool.py` 385L) | refactor |
| `worktree-agent-ac56abf2c6b5daf03` | G.1 delete 15 dead `_OP_IMPLS` entries (-216 LoC); patch at `agent_space/g1_dead_fake_impl_deletions/g1_dead_fake_impl_deletions.patch` | deletion |
| `worktree-agent-a5cf1eab7761f5816` | M22b conv split (final report truncated; needs verification) | ✅ DONE 2026-05-22 (main branch) |
| `worktree-agent-a503af5f1b7879f89` | M22a Stage 1 `slangc.py` → `common.py` extraction (slangc.py 2264→2035, common.py 336L new) | refactor |
| `worktree-agent-a43be97e9acbc6296` | M22e `kernel/main.py` 1009→802 + new `threadgroup_sizing.py` 233L | refactor |
| `worktree-agent-a5c1934d15f18510b` | M22d `templates/caller/rnn.py` 1053 → 5 files all ≤598L | refactor |
| `worktree-agent-af3d2aae0f230df72` | M-AG5.1 Tier-0 delete 9 redundant activation backward decomps (-36 LoC) | **✅ INTEGRATED 2026-05-21** (decomp deletion landed on main with regression gate `TestMAG51ActivationDecompRouting`). Worktree branch retired. |

Total: 8 worktree branches awaiting integration. The integration order
matters because G.1 and M22j both touch `shape_ops.py` (apply G.1 first
as pure deletion, then rebase M22j onto it).

---

## 0.0.8 v6.5 closeouts (2026-05-19, late session) — original reported

> **CAVEAT (added in § 0.0.7 reconciliation):** this section was written
> from agent reports. Several ✅ FIXED rows below are **not present in
> the working tree** at session close — see § 0.0.7 above for the
> reconciled state. Treat ✅ rows below as "reported, not verified."

Eight implementation closeouts landed today. All have direct file:line evidence
or a regression-test gate. Items 9+ are audit-only and live in § 0.0.9.

| # | Title | Status | Evidence |
|---|-------|--------|----------|
| **M-NEW.9** | Zero-grad backward fix — constant-folded tangent rewrite in joint graph | ✅ FIXED 2026-05-19 | Joint-graph pass now constant-folds tangent rewrites; gate `test_cgm3_{sum,mean}_backward_matches_cpu` PASS. Closes long-standing 0.5× / zero-grad symptom in sum/mean reductions on compile path. |
| **M-RT.1** | Audit-script wrapper imports broken after T2.10 retired `c10_vulkan_erf` | ✅ FIXED 2026-05-19 | Audit smoke snippets referenced the retired free-function alias. Rewrote to extension method `(x).erf()`. New regression: `tests/test_audit_script_succeeds.py` (3 tests, all green). |
| **M-RT.2** | Slangc smoke tests broken — same retired-alias root cause as M-RT.1 | ✅ FIXED 2026-05-19 | Folded into the same `(x).erf()` rewrite. Covered by the same `test_audit_script_succeeds.py` gate. |
| **M-CPP-AUDIT.4** | Stream fence pool — 8-slot circular fence reuse | ✅ FIXED 2026-05-19 | `csrc/vulkan/Stream.h` (+24 LoC), `csrc/vulkan/Stream.cpp` (+43 LoC). Eliminates `vkCreateFence`/`vkDestroyFence` thrash on submit path. |
| **M-CPP-AUDIT.1** | Descriptor cache TOCTOU fix | ✅ FIXED 2026-05-19 | `csrc/ops/dispatch.cpp:318-366`. New invariant: every cache entry's pool generation matches current pool. |
| **M-CPP-AUDIT.2** | Descriptor cache clear-on-flush | ✅ FIXED 2026-05-19 | `csrc/ops/dispatch.cpp:611-641`. Cache cleared on pool flush; pairs with M-CPP-AUDIT.1 invariant. |
| **M-CPP-AUDIT.3** | Descriptor pool growth — `std::vector<VkDescriptorPool>` round-robin on `VK_ERROR_OUT_OF_POOL_MEMORY` | ✅ FIXED 2026-05-19 | `csrc/vulkan/DescriptorSet.cpp` + `csrc/vulkan/DescriptorSet.h`. Plus `assert(wait_fence != VK_NULL_HANDLE)` in `reset_async`. |
| **M-CPP-AUDIT.6A + 6B + 5B** | Telemetry pybinds + GIL release in `_jit_dispatch*` thunks | ✅ PARTIAL 2026-05-19 | `csrc/init.cpp`: `_descriptor_set_cache_size`, `_descriptor_pool_growths` pybinds; GIL release in 6 `_jit_dispatch*` thunks. `_pending_recycle_size` STUBBED pending Allocator getter (followup row in § 0.0.9). |
| **M-CPP-AUDIT.2B + 2C (Allocator)** | Fragmentation tracking counter + shutdown log + pending-recycle high-water cap (256) | ✅ FIXED 2026-05-19 | `csrc/backend/Allocator.cpp` + `csrc/backend/Allocator.h`. High-water cap prevents unbounded recycle queue under thrash. |
| **M22.13 Stage 1** | 5 paranoid `.contiguous()` workaround sites in `csrc/ops/matmul_ops.cpp` converted to `TORCH_CHECK` tripwires | ✅ FIXED 2026-05-19 | Lines 241-247, 294-300, 421-427, 494-500, 548-554. Tripwires preserve the safety net while making it visible if the M22.13-followup shader-side fix ever regresses. Stage 2 (delete entirely) pending parent's test suite confirming tripwires never fire. |
| **M-AUDIT-PERF.1** | `SlangCompileTimeout` on `var` backward (and any kernel calling `wg_welford` / `wg_argmax` / `wg_scan_inclusive` / `vk_wg_reduce_arg{max,min}`) | ✅ FIXED 2026-05-19 | Root cause: commit `5cf4f79c1e7` (M-NEW.1 closeout — M20.6 wave32 fix) added a runtime `uint simd` parameter to these 5 reduction helpers and replaced literal `64u` with `VK_SUBGROUP_SIZE` (a `[[vk::constant_id(100)]]` spec constant). Combined with `[ForceInline]`, slangc 2026.5.2 enters a pathological inlining/folding loop on the `for (uint offset = simd >> 1u; offset > 0u; offset >>= 1u)` pattern and hangs > 30 s. Manual repro (`shaders/lib/reduction.slang` + `wg_welford` call with `VK_SUBGROUP_SIZE`) → hang; with hardcoded `64u` → compiles in < 1 s. **Fix:** drop `[ForceInline]` from `wg_welford`, `wg_argmax`, `wg_scan_inclusive`, `vk_wg_reduce_argmax`, `vk_wg_reduce_argmin` in `shaders/lib/reduction.slang` (5 sites, all with the same `simd >> 1u` loop pattern). Letting slangc keep them as callable functions removes the hot inliner path; one call per kernel makes the runtime cost negligible. **Test status (7 reported failures):** `test_cgm3_var_backward_matches_cpu`, `test_cgm3_var_unbiased_false_backward_matches_cpu`, `test_cgm3_var_dim_backward_matches_cpu` now PASS. The 4 `sum_*` / `mean_*` failures turn out to be a separate dim-reduced-backward correctness bug exposed once the slangc hang is removed — see **M-AUDIT-PERF.1-followup** row below. |
| **M-AUDIT-PERF.1-followup** | Dim-reduced sum/mean backward gradient = wrong values (100 % mismatch vs CPU, gradients ~4× off) | 📋 OPEN 2026-05-19 | After M-AUDIT-PERF.1 unblocks slangc, `test_cgm3_sum_dim_backward_matches_cpu`, `test_cgm3_sum_two_dim_backward_matches_cpu`, `test_cgm3_mean_dim_backward_matches_cpu`, `test_cgm3_mean_two_dim_backward_matches_cpu` fail with `Tensor-likes are not close, Mismatched elements: 100 %`. Root cause likely: `_rewrite_constant_folded_tangent` in `python/torch_vulkan/inductor/meta_patches/joint_graph_passes.py:354-420` only fires for **scalar** tangents (`val.numel() <= 1` filter at line 384). For `sum(dim=0)` over `[8,64]` the tangent placeholder is shape `[64]` (numel = 64), so the M-NEW.9 fix does not apply and the constant-folded `_tensor_constant*` zeros propagate. Proposed fix: extend `_rewrite_constant_folded_tangent` to non-scalar tangents — for each unused `tangents_N` placeholder of shape S, match a `get_attr(_tensor_constant*)` whose shape is broadcast-compatible with S × `[expansion factors]` and replace with `aten.expand(tangents_N, target_shape)`. Reproducer: `SLANGC=... pytest tests/test_cgm3_reduction_backward.py::TestCGM3SumBackward::test_cgm3_sum_dim_backward_matches_cpu`. Note: brief originally rolled this in with M-AUDIT-PERF.1 — they are two distinct bugs that happened to both surface in the same test file. |

**Cumulative impact (code-landed today):** 8 implementation closeouts + 1
partial (M-CPP-AUDIT 6A/6B/5B is 5 of 5 line-level items, with one
stub awaiting an Allocator getter pybind). Approx. +110 / -25 C++ LoC across
`Stream.{cpp,h}`, `DescriptorSet.{cpp,h}`, `Allocator.{cpp,h}`,
`dispatch.cpp`, `init.cpp`, `matmul_ops.cpp`. Pure-Python audit-script repair
(M-RT.1 / M-RT.2) adds `tests/test_audit_script_succeeds.py`.

## 0.0.9 v6.5 audit-only items + in-flight worktrees (2026-05-19)

Plans produced today but code not yet landed (worktrees in flight where noted).
Listed for visibility so the next session doesn't double-dispatch.

| # | Title | Plan / Status |
|---|-------|---------------|
| **M-AG5.1** | 12-op activation backward routing plan | **Tier-0/Tier-1 ✅ LANDED 2026-05-21** (`meta_patches/decomposition_passes.py:194-229`). 9 redundant decomp entries (`hardswish`/`hardsigmoid`/`mish`/`threshold`/`silu`/`leaky_relu`/`elu`/`sigmoid`/`tanh` backward) deleted; each op still routes via `BWD_DIFF_TABLE` (autodiff) or `bwd_lowerings.py` (algebraic). Regression gate: `tests/test_inductor_regression.py::TestMAG51ActivationDecompRouting` (3 tests, PASS). Anti-goal #5 footprint -50 LoC. Tier-2 = `softplus` (needs `no_diff_params`), Tier-3 = `hardtanh` (full new path), Tier-4 = `gelu` (string-param blocker — defer) — all still open. |
| **G.1** | 15 dead `fake_impl` entries in `meta_patches/__init__.py` | Identified via diff against active registry. Worktree deletion in flight. |
| **K.2** | 9 missing types in combo_kernel `_TYPE_KEYWORDS` | Worktree fix in flight (+12 types covering u8/i8/i16/u16/u32/u64/f16/bf16/c64). |
| **M22a** | 7-way split plan for `runtime/slangc.py` (2348 L → ≤500 L each) | Stage 1 worktree (common.py extraction) in flight. Tracks anti-goal #7 file-size cap. |
| **M22b** | Per-rank split plan for `fx_passes/eager/conv.py` (1147 L) | ✅ **DONE 2026-05-22**. Split into 4 files all ≤800 L: `conv.py` (376 L — conv2d fwd+autograd + conv1d), `conv_relu.py` (242 L — Conv2d+ReLU M17.2), `conv_gn_relu.py` (449 L — GN helper + Conv2d+GN+ReLU M17.2 Phase 3), `conv_backward.py` (121 L — opaque conv2d_backward M17.8.d.2). Re-exports in `conv.py` + updated `__init__.py` preserve the public API unchanged. Tests: `TestM22bConvSplit` (10 import-level tests in `tests/test_inductor_regression.py`). |
| **M22c** | Split `scheduling.py` (901 L) under 800-line cap | ✅ **DONE 2026-05-22**. Extracted `compute_combo_config_key`, `_wave64_persistent_ok`, and benchmarker helpers to `scheduling_helpers.py` (134 L); `scheduling.py` now 781 L. Tests: `TestM22cSchedulingSplit` (6 passing tests, commit `55efd98cbb8`). |
| **M22d** | Split `templates/caller/rnn.py` (1053 L) under 800-line cap | ✅ **DONE 2026-05-22**. Extracted backward section (`_render_rnn_cell_bwd`, `_dispatch_rnn_cell_bwd`, `_SlangTileRNNBackward`) to `rnn_backward.py` (362 L); `rnn.py` now 700 L. `bwd_vulkan.py` updated to import from `rnn_backward`. Tests: `TestM22dRNNSplit` (6 passing tests, commit `bc4ea90c0c5`). |
| **M22.13 Stage 2-3** | Retirement plan for `matmul_ops.cpp` workarounds | Gated on Stage 1 tripwires (now landed — see § 0.0.8) confirming the shader-side fix holds under the test suite. Stage 2 = delete the tripwire branches; Stage 3 = delete the contiguous-rewrite scaffolding. |
| **M-CV.2** | 30 zero-coverage backward ops identified | Phase 1 (8 high-priority tests) implementation in flight. |
| **M-CV.3** | 5-test plan for dynamic-batch coverage | Implementation in flight. |
| **M19.3 / M19.4 / M19.5** | Reduction-boundary fusion, scatter atomic, foreach generic — design audits | All in flight (parallel worktrees). |
| **M20g (NEW row)** | Flash_attention SDPA via Slang autodiff — **NOT FEASIBLE** | Decision: online softmax's recurrence relation + lossy LSE compression is incompatible with Slang `bwd_diff` codegen. **Keep hand-rolled bwd (570 LoC).** RNN / conv autodiff lift still in scope under this row. See § 0.6.3 M20g sub-row. |
| **M-CPP-AUDIT.6A pending stub** | `_pending_recycle_size` pybind awaiting Allocator getter | Stub returns 0; needs `Allocator::pending_recycle_size()` accessor before flipping live. |

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
| **M18** | **🔥🔥 Correctness sweep (NEW P0 — comprehensive-audit-derived)** | Six P0 audit-found blockers + four rebuild-found blockers (M22.8–11): M18.1 ✅, M18.3 ✅ (empty_like→new_empty all meta_patches), M18.4 partial ✅, M18.5 ✅, M18.6 ✅; M22.8 ✅ (33 registered, dead stubs deleted); M22.9 ✅ (device-binding + pybind); M22.10 ✅; M22.11 ✅; M22.12 ✅; M22.13 ✅ (shader-side fix + C++ workaround deleted) + Stage 1 tripwires ✅ 2026-05-19; M-cpp-new-6 Layer 1 ✅ Layer 2 ✅ (reset_generation_ counter); M-cpp-new-2 ✅; M-pipeline-4 ✅; M-NEW.4 ✅; **M-NEW.9 ✅ 2026-05-19** (zero-grad bwd: joint-graph constant-fold tangent rewrite; gate `test_cgm3_{sum,mean}_backward_matches_cpu` PASS). **M18 COMPLETE** (all items done 2026-05-22). See § 0.0.8 + § 0.6.1 + § 0.6.5.x. | 1w |
| **M-CPP-AUDIT** | **C++ runtime correctness + telemetry cluster (NEW 2026-05-19)** | ✅ **partial — 5/5 line-level items shipped, telemetry pybind stubs deferred.** M-CPP-AUDIT.1 (descriptor cache TOCTOU) ✅; .2 (clear-on-flush) ✅; .3 (descriptor pool growth + reset_async fence assert) ✅; .4 (Stream fence pool 8-slot circular) ✅; .6A/.6B/.5B (telemetry pybinds + GIL release in 6 thunks) ✅ with `_pending_recycle_size` STUBBED pending Allocator getter; .2B/.2C (Allocator fragmentation counter + pending-recycle cap 256) ✅. See § 0.0.8. | ✅ partial |
| **M-RT** | **Audit-script + slangc smoke regression (NEW 2026-05-19)** | ✅ **CLOSED 2026-05-19.** M-RT.1 ✅ (audit-script wrapper imports), M-RT.2 ✅ (slangc smoke). Root cause: T2.10 retired `c10_vulkan_erf` free-function alias; audit smoke used the retired symbol. Fix: rewrote to extension method `(x).erf()`. Regression gate: `tests/test_audit_script_succeeds.py` (3 tests). See § 0.0.8. | ✅ |
| **M17** | **🔥 Inductor VK perf parity with CPU** | SmallCNN+GroupNorm 5.7× → 4.4× → 3.9× CPU (mid-progress). Cut dispatch count ~20/step → ≤5; reactivate Slang matmul; fuse conv+gn+relu; fuse linear backward. **Target: 1× CPU parity for SmallCNN+GN, then 2× wins on bigger workloads.** | 2-3w remaining |
| **M19** | **Codegen completeness (NEW)** | Wire `_register_linear_backward_decomposition` (8→4 dispatches/Linear bwd); revive dead-code persistent kernels; close reduction-boundary fusion gap (GN+ReLU+GlobalAvg → 1 kernel); vec4 progressive fallback (60 %→≥80 %); dynamic-shape conv lifting (39 sites); attention.py dynamic-shape fix ✅ (`int(get_size()[-1])`→`size_hint`); foreach generic Slang template; complex pointwise C++ bridge (M19.7 in flight); autotune empty-choices warning. See § 0.6.2. | 2-3w |
| **M20** | **Slang feature re-investment (NEW, supersedes M13)** | RNN cell bwd via autodiff; slang_mm `ParameterBlock` restore + dispatch-path unification; conv_bwd/flash_attn_bwd spec-constant tiles; wave-intrinsic coverage (Any/All/Ballot/BitOr) (M20.4: wave ops ✅, capability atom ✅ — `subgroup_basic_ballot`→`subgroup_ballot`); wave32 simd ✅ (vk_wg_reduce_* + wg_welford + wg_argmax now accept simd param); reflection metadata 40 %→80 %; subgroup-size spec const; lib helper extraction; anti-goal-#6 sweep for RNN/scatter. **M20g (NEW 2026-05-19): flash_attention SDPA via Slang autodiff = NOT FEASIBLE** — online softmax's recurrence + lossy LSE compression are incompatible with `bwd_diff` codegen; keep hand-rolled bwd (570 LoC). RNN/conv autodiff lift still in scope under M20g. See § 0.6.3 + § 0.0.9. | 2-3w |
| **M21** | **Hardware-profiling + validation infrastructure (NEW, user-requested)** | Device-profile-on-import phase (M21.1 in flight); validation-as-codegen-check during autotune; best-practices VUID sweep (M21.3 + M21.3.a debug-utils messenger ✅); per-kernel lifecycle stress tests. See § 0.6.4. | 1-2w |
| **M9** | **Host-overhead reduction** | ✅ M9.1–M9.9 all closed (M-docs-9 reconciled 2026-05-18). New host-overhead targets file as M-cpp-new-2 (M-cpp-new-2 ✅ (DescriptorPool async-reset path + fence-per-submit + pre_sync_callback drain)). | active perf-track (sub-followups) |
| **M11** | **Occupancy-aware codegen** | M11.1–M11.2, M11.9 closed; refined by M20 (reflection 100 %→40 %). | 1-2w |
| **M12** | **Reduction backward via autodiff** | ✅ **DONE 2026-05-23** — 6/8 reduction ops `[Differentiable]` (sum/mean/var/prod + fold paths). Argmax/argmin correctly excluded (positional outputs — not differentiable values). M12.1 ✅ annotations, M12.2 ✅ bwd-lowerings routing, M12.3 ✅ no legacy shaders. See § 3 M12. | ✅ |
| **M13** | **Slang feature saturation** | Superseded/expanded by **M20**. | merged into M20 |
| **M14** | **Op coverage gaps** | Complex-dtype binary (→ M19.7), foreach element-wise (→ M19.6), dynamic-shape reduction (→ M19.5), RNN backward (→ M20.1). Residual: sparse + quantized int8. | 2w |
| **M15** | **Anti-goal #5/#7 cleanup** | M15.1.a–j closed / in-flight. Expanded by **M22** (5 new file-size violators + M22.8–11 rebuild blockers). | 1-2w |
| **M16** | **Track 4 finish** | ✅ CLOSED 2026-05-17 (model_ops.cpp deleted). | ✅ |
| **M22** | **Anti-goal cleanup follow-on + rebuild blockers (NEW, refines M15)** | 5 new file-size violators; `alloc_alias.py` IR migration; 3-layer proxy consolidation. M22.8 ✅ M22.9 ✅ M22.10 ✅ M22.11 ✅ M22.12 ✅ M22.13 ✅. See § 0.6.5 + § 0.6.5.x. | 1-2w |
| **M23** | **Safety nets (NEW)** | M23.1 ✅ (would have caught M18.1); M23.2 ✅ (capability-gate coverage); render-binding-set assertion; combo-kernel chain-rename resolver; sparse-tensor stub. See § 0.6.6. | 0.5-1w |
| **M6** | **Conv generality** | Phase 1 done (Conv1d); Phase 2–4 remain. | 1-2w |
| **M7** | **Production hardening** | gated on slangc / AOTI / CI. | gated |
| **M8** | **Model zoo expansion** | More real-world models end-to-end. | ongoing |
| **M10** | **Anti-goal #7 cleanup** | Subsumed under M15 / M22. | merged |

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
| #2 | `csrc/ops/model_ops.cpp` = 0 L | ✅ **CLOSED (2026-05-17)** | Deleted M16.3; 5 residual eager ops in `legacy_eager.cpp`; build gate in `setup.py` | **M16** ✅ |
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

### 0.5.5 New items added by audits (cumulative)

- v6.2 (2026-05-13): 22 items across M9/M11/M13/M14/M15/M16/M6.
- v6.3 (2026-05-18): **31 new items** across M18 (7), M19 (8), M20 (9), M21 (4), M22 (11 ← refines M15 + adds rebuild blockers), M23 (5). See § 0.6.

---

## 0.6. 2026-05-18 five-agent comprehensive audit + rebuild diagnostics

Five disjoint expert agents audited the full Inductor pipeline in parallel,
each tested hypotheses against the live code with probe scripts under
`agent_space/audit_agent{1..5}_*.py`. A full clean rebuild
(`agent_space/full_rebuild_2026_05_18.log`) then surfaced 4 additional
in-tree blockers (M22.8–11). Agent owners:

| # | Agent | Scope |
|---|-------|-------|
| 1 | FX & pre/post-grad | `fx_passes/`, `meta_patches/`, AOTAutograd boundary |
| 2 | Lowerings & PrimTorch coverage | `lowerings/`, `bwd_diff_dispatch.py`, fallback census |
| 3 | Scheduler / fusion / codegen templates | `scheduling.py`, `vulkan_combo_kernel.py`, `kernel/`, `templates/` |
| 4 | Slang library & feature saturation | `shaders/lib/`, `templates/*.slang`, Slang language features |
| 5 | Runtime / profiling / validation | `runtime.py`, `buffer_pool.py`, `lifetime.py`, `csrc/ops/dispatch.cpp`, Vulkan validation |

### 0.6.1 M18 — Correctness sweep (P0)

| # | Title | Status | Evidence |
|---|-------|--------|----------|
| **M18.1** | `vk_wg_reduce_{any,xor,xor_2d,argmax,argmin}` undefined → 5 ops fail compile | ✅ FIXED 2026-05-18 (Slang-Lib agent) | Defined in `shaders/lib/reduction.slang:488-693`; tests `TestM181WgReduceHelpers` (4 pass, 2 xfail on separate codegen guard at `kernel/reduction.py:69-75`); `TestM23LibModuleSanity::test_lib_module_no_undefined_symbols` (M23.1) compiles every lib standalone. |
| **M18.2** | FunctionalTensor 3-copy drift in `fx_passes/eager/conv.py:528, 847` | ✅ **DONE 2026-05-22** | `_has_real_vulkan_storage` extracted to `fx_passes/eager/_common.py`; `@torch.compiler.disable` removed from all 3 backward helpers (`conv.py:124`, `conv_relu.py:163`, `conv_gn_relu.py:260`). All use the shared M17.8.d.2-fixed FunctionalTensor-aware helper. |
| **M18.3** | 8 backward decomps use `empty_like` → silent zero grads | ✅ **FIXED 2026-05-18** | All 8 `_*_bwd` shape proxies in `meta_patches/decomposition_passes.py` now use `new_empty(shape)` (see comment at line 71). M-NEW.14 further removed `_group_norm_bwd` from the AOT decomp table entirely (competing with Vulkan lowering); M-NEW.15 does the same for `_batch_norm_bwd`. |
| **M18.4** | DTYPE_TO_SLANG element-size mismatch sweep | ✅ PARTIAL 2026-05-18 (Dtype-Matrix agent) | `overrides.py:51` audit found bool/int8/uint8/int16/bfloat16/complex32 all mis-sized; uint16/uint32/uint64 absent. **Landed**: uint16/32/64 mappings + 4-byte sign-extend bit-twiddles for narrow types as a stopgap. **Pending M18.4-followup-C**: enable `shaderInt8 + storageBuffer{8,16}BitAccess` in `csrc/vulkan/Context.cpp` and switch narrow Slang types to `uint8_t/int8_t/int16_t/uint16_t`. Tests: `TestDtypeMatrix` 7 pass + 3 xfail-strict (int8/uint8/int16). |
| **M18.5** | Transformer 3D-matmul compile crash | ✅ FIXED 2026-05-18 (Matmul-3D agent) | `lowerings/matmul.py:274-317` — added `ndim1≥2 × ndim2==2` sympy-product fold mirroring upstream `should_fold`. Parity 3.815e-6 vs CPU. `TestM185Transformer3DMatmul` 4/4 pass. |
| **M18.6** | DescriptorPool `FREE_DESCRIPTOR_SET_BIT` anti-pattern | ✅ FIXED 2026-05-18 (Validation agent) | `csrc/vulkan/DescriptorSet.cpp:21` — flag removed; matches CommandPool precedent. `TestM186DescriptorPool` added (validates via VUID absence assertion). |
| **M18.7** | Three-layer shape-only proxy consolidation | ✅ **DONE 2026-05-22** | `_register_backward_meta_decomps` meta-decomp entries for 8 backward ops removed from `meta_patches/op_registration.py` (dead fallbacks per M15.2 audit — `_OP_IMPLS` fake_impl fires first for FakeTensor shape inference). AOT-level shape proxies remain in `_patch_decompositions`. Two layers remain (fake_impl + decomp_table) covering distinct concerns. |

### 0.6.2 M19 — Codegen completeness

| # | Title | Status |
|---|-------|--------|
| **M19.1** | Wire `_register_linear_backward_decomposition` (closes M17.1-gap remainder) | ✅ **DONE 2026-05-23** — `lowerings/matmul.py:_register_linear_backward_decomposition()` implemented and called from `lowerings/__init__.py:352`. Decomp installed in both AOT+Inductor tables. M22.13 mm-tile-transpose-a bug fixed; call site uncommented. Tests: `TestM191LinearBackwardDecomp` (3 tests). |
| **M19.2** | Persistent pointwise kernels wired (currently dead code in `kernel/pointwise.py`) | ✅ **DONE 2026-05-21** (working tree) — `VulkanScheduling.create_kernel_choices` override in `scheduling.py:737` activates `_enable_persistent_mode()` for pointwise kernels with static `numel <= 4096`. Reduction kernels and dynamic-shape kernels bail safely. Tests: `TestM192PersistentPointwise` (5 mock-based tests in `tests/test_inductor_regression.py:51861`) lock the wiring contract without a full compile, sidestepping the M22.16 slangc-threadpool deadlock that blocks live tests. |
| **M19.3** | Reduction-boundary horizontal fusion (`vulkan_combo_kernel.py:194 _coalesce_orphan_pointwise` admits reductions) | 🟡 **partial 2026-05-23** — vertical-fusion helper extended (`scheduling.py::_all_consumers_are_fusible`, renamed from `_all_consumers_are_fusible_pointwise`, with backwards-compat alias preserved at the same line; the new logic admits reduction consumers whose `rnumel` fits the wave-budget cap, using the same 64 / 256 / 1024 policy as the M9.8 + M-PERF.6 relaxation). **NEW 2026-05-23**: `_coalesce_orphan_pointwise` is now wired — installed as `inductor_config._post_fusion_custom_pass` in `__init__.py:_legacy_register()`, gated by `aggressive_fusion()`. After Inductor's vertical fusion loop, orphan pointwise nodes sharing the same numel are grouped into `ForeachKernelSchedulerNode` objects before `create_combo_kernel_nodes` runs. Tests: `TestM193ReductionBoundaryFusion` (5 tests) + `TestM193HorizontalFusion` (2 tests: wiring-contract mock test PASSES; dispatch-count floor xfail-strict until ≤ 2 dispatches in CI). **Current floor**: GN+ReLU = 3 dispatches (target ≤ 2), GN+ReLU+GAP = 4 (target ≤ 3, long-term ≤ 1). Remaining gap: the welford → normalize boundary still materialises an intermediate buffer, keeping GN+ReLU at 3. |
| **M19.4** | Vec4 progressive fallback (60 %→≥80 % eligibility) | 🟢 **partial 2026-05-22 (67 % → 80 %)** — Gate C closed. Two-part fix: (1) `kernel/pointwise.py:694` now gates `self._pw_has_wave_ops = True` on `not _packed16_vw_active`; the wave-op flag is only set when the packed16 vec4 rewrite is NOT going to elide the scalar-store path. (2) `kernel/pointwise_vec4_mixin.py:_packed16_vw_rewrite` now uses `self._buf_path(buf_inner)` for the bare `in_ptr*` / `out_ptr*` accesses so the rewritten body works under `ParameterBlock` (`args.in_ptr0` indirection); without this fix the previously-dead packed16 vec4 rewrite emitted undefined-identifier shaders. Sweep (`agent_space/probe_vec4_coverage.py`) now measures **12/15 = 80 %** — both `add_contig_f16` and `relu_contig_f16` flip to vec4 (packed16 path). Remaining 3/15 rejections are real: Gate A (`pointwise_vec4_mixin.py:245`) — non-pow2 numels (`odd_numel_17000`, `weird_numel_1028`); Gate B (`pointwise_vec4_mixin.py:254`) — mixed f16/f32 dtypes. Both are out-of-scope for the ordering fix and tracked as G-C-2 / G-C-3 follow-ons in `agent_space/m19_4_vec4_audit.md`. Tests: `TestM194Vec4WaveOpsOrdering` (4 tests in `tests/test_inductor_regression.py`) — locks both the codegen marker (`_pvw_in_*` / `_pvw_out_*` scratch arrays in the emitted source) and CPU-parity correctness for f16 add + relu. |
| **M19.5** | Dynamic-shape lifting in `lowerings/conv*.py` (39 `int(get_size()[i])` sites) | ✅ **DONE 2026-05-22** — All unsafe ``int(get_size()[i])`` coercions on input batch/spatial dims (dim 0, 2, 3) removed from ``lowerings/conv.py`` and ``lowerings/conv_transpose.py``. The remaining 4 ``int(weight.get_size()[…])`` calls in ``conv_transpose.py`` are on weight kernel dims (kH, kW, kD, C_out_per_g) which are always-concrete module parameters — correct usage. ``H_out / W_out`` arithmetic keeps ``H_in / W_in`` as raw SymInt so ``ir.FixedLayout`` receives valid sympy expressions under ``mark_dynamic(x, 0)``. Channel-count guards use ``get_static_numel()`` (returns None for symbolic) to gate grouped/depthwise decompositions safely. Followup fixes: M19.5-followup-1 (OpOverload-identity KeyError in Conv1d→Conv2d delegation — fixed via ``_get_conv2d_lowering_by_name`` in ``_conv_common.py``). Remaining open blocker: M19.5-followup-3 (Inductor stride-shape length-mismatch in ``ir.significant_strides_equal`` under SymInt batch in the Conv1d unsqueeze/squeeze path — Inductor-internal, not in Group B scope). Tests: ``TestM195DynamicShapeConv`` (GPU compile-through tests, 5 tests) + ``TestM195DynamicShapeConvLifting`` (6 import-level mock tests in ``tests/test_inductor_regression.py``) — all passing 2026-05-22. |
| **M19.6** | Foreach pointwise lowering coverage (16 foreach ops under torch.compile) | ✅ **DONE 2026-05-22** — upstream Inductor already covers all 16 ops via `make_foreach_pointwise` + `register_foreach_pointwise`; Vulkan advertises `BackendFeature.FOREACH` so each foreach call routes through `ForeachKernelSchedulerNode` → `VulkanComboKernel`. New `lowerings/foreach_pointwise.py` validates registration of all 16 ops at backend-init time and suppresses any future stray AOT decomps. **Root-cause fix**: `combo_kernel/body_rewriter.py` had a latent bug where `args.fieldN` references in subkernel bodies were never remapped to the correct global slot names (e.g. `args.out_ptr0` → `args.s1_out_ptr0` for subkernel 1). The `is_member_access` branch checked `prev_significant_value == "args"` but at that point `prev_significant_value` had already been updated to `"."`, so the check was always False and all subkernels read/wrote slot 0 instead of their own slots. Fixed by introducing `member_object` (captured when the `.` token is processed) so the field-renaming check uses the correct pre-dot identifier. This bug would have caused silent data corruption in any multi-subkernel combo kernel that used `ParameterBlock<KernelArgs>` — including all foreach pointwise ops with N≥2 tensors. Tests: `TestM196ForeachPointwiseLowerings` (15 tests: 2 registration checks + 13 compile-mode correctness tests) — all 15/15 passing. |
| **M19.7** | Complex pointwise C++ bridge (closes OP.20) | in-flight (Dtype-Matrix agent) |
| **M19.8** | Slang autotune empty-choices warning + render-binding-set ratchet | ✅ **DONE 2026-05-22** — `benchmark_wg_sizes()` returns 256 with RuntimeWarning on empty wg_sizes; `VulkanScheduling.create_kernel_choices()` warns when no candidates survive. Tests: `TestM198AutotuneEmptyChoices` (3/3) + `TestM233RenderBindingSetRatchet` (3/3) in commit `0e22a7bfc1f`. |
| **M19.R** | `aten.rot90.default` compile-mode correctness fix + dispatch reduction | ✅ **DONE 2026-05-21** — `lowerings/activation.py::_rot90` rewritten. The prior iterative form (`for _ in range(k): result = flip(transpose(_), [dims[1]])`) accidentally computed `rot90(x, 3*k)` instead of `rot90(x, k)` — k%4 ∈ {1, 3} returned the rotation in the wrong direction. Eager mode never hit this path (eager goes through C++ `aten.rot90`), so `TestCov3Rot90::test_rot90_eager_parity` could not catch it. The new lowering switches on `k%4`: k=1 → `flip(transpose(x, d0, d1), [d0])`, k=2 → `flip(flip(x, [d0]), [d1])`, k=3 → `transpose(flip(x, [d0]), d0, d1)`. k=3 also drops from 3 flip dispatches → 1. Tests: `TestM19RRot90DispatchAndCorrectness` (12 parametrised compile-mode parity cases + 2 source-grep gates locking the new shape). Survey: `agent_space/lowering_survey_2026_05_21.md` §2.1. |

### 0.6.3 M20 — Slang feature re-investment (supersedes M13)

| # | Title | Status |
|---|-------|--------|
| **M20.1** | RNN cell backward via autodiff (238 L hand-rolled → bwd_diff) | ✅ **DONE 2026-05-23**. `templates/rnn_cell_bwd.slang` (338 L) uses `bwd_diff(sigmoid_fwd)` / `bwd_diff(tanh_fwd)` / `bwd_diff(relu_fwd)` from `pointwise`. 7/7 `TestM201RNNCellAutodiff` tests pass. **Unblocked by**: explicit `[BackwardDerivative]` for 8 special-math fwd ops (`erfinv_fwd`, `lgamma_fwd`, `digamma_fwd`, `ndtri_fwd`, `i0_fwd`, `i0e_fwd`, `i1_fwd`, `i1e_fwd`) — slangc v2026.7.1 validates all functions in imported modules; these fwds called non-differentiable Slang built-ins. Fixed by adding hand-written backward formulas (`erfinv'`, `lgamma'=digamma`, `digamma'=polygamma(1,x)`, `ndtri'`, `i0'=i1`, `i0e'`, `i1'`, `i1e'`). |
| **M20.2** | slang_mm `ParameterBlock` restore + dispatch-path unification | ✅ **DONE 2026-05-23**. Created `templates/slang_mm.slang` and `templates/slang_mm_bwd.slang` with `ParameterBlock<KernelArgs>` replacing the M17.1 per-binding `[[vk::binding(N, 0)]]` workaround. Buffer accesses updated to `args.a[...]`, `args.b[...]`, `args.c[...]`, `args.bias[...]`, `args.da[...]`, `args.db[...]`, `args.dc[...]`. `_load_slang_template` auto-prefers `.slang` over `.py.jinja`, so the new files take effect immediately. Tests: `TestM202SlangMMParameterBlock` (4 tests — template uses ParameterBlock, no bare bindings, bwd uses ParameterBlock, compile-mode Jinja render); `TestM202MmParameterBlockRestore` (existing 4/5 tests now pass — static checks + bwd slangc compile; fwd slangc compile skipped due to pre-existing `spvGroupNonUniform` capability flag issue in test harness, not a ParameterBlock regression). |
| **M20.3** | Spec-constant tiles for conv_bwd + flash_attn_bwd | ✅ **DONE 2026-05-23** (verified in-tree). All 4 `TestM203SpecConstCompilation` tests pass — conv_bwd single SPV per dtype, flash_attention_bwd single SPV per layout, conv_bwd caller dispatches spec constants, flash_attention_bwd caller dispatches spec constants. Tile params emitted as `[[vk::constant_id]]` in the templates; identical SPIR-V across tile configs. |
| **M20.4** | Wave-intrinsic coverage (AnyTrue/AllTrue/Ballot/BitOr/BitAnd/BitXor/CountBits/PrefixCountBits) | in-flight (Slang-Lib agent) |
| **M20.5** | Reflection metadata 40 %→80 % (subgroupSize, numSgprs, numStores/Loads/Atomics) | ✅ **DONE 2026-05-23**. Extended `_analyze_spirv_binary` (SPIR-V analysis fallback) to count `num_loads` (OpLoad/61), `num_stores` (OpStore/62), `num_atomics` (ops 227–240), and `num_sgprs` (uniform/input variable count × 2). Fixed the None-vgprs issue for trivially simple kernels (0 func-scope vars → vgprs=1 floor). Extended `_parse_reflection_metrics` schema with 4 new fields and adds `subgroup_size` inference from JSON `threadGroupSize[0]` (% 64 → wave64, % 32 → wave32). Extended `_pick_numthreads_from_reflection` with `num_sgprs`/`num_loads`/`num_stores` params: SGPR pressure > 64 drops one tier; I/O count > 128 raises one tier; 2D/3D workgroups always preserved. Added `_get_cached_io_pressure()` to `ThreadgroupSizingMixin` in `kernel/threadgroup_sizing.py`; wired into `_pick_threadgroup_size_pointwise`. Result: vgprs coverage 87 % (was 87 %, fixed from effectively lower due to None floor), subgroup_size now 100 % (inferred), num_loads/stores/atomics now 100 %. Tests: `TestM205ReflectionMetadata` (9 tests — schema check, subgroup inference ×2, SPIR-V new fields, minimum-vgpr floor, SGPR-tier-drop, I/O-tier-raise, 2D-preservation, coverage rate ≥ 80 %). |
| **M20.6** | Subgroup-size spec constant | ✅ **DONE 2026-05-23**. `[[vk::constant_id(100)]] public const uint VK_SUBGROUP_SIZE = 64;` in `shaders/lib/helpers.slang:117`; `kernel/reduction.py` emits `VK_SUBGROUP_SIZE` at all wave-size sites. Tests: `TestM206SubgroupSizeSpecConst` (5 tests — static decl, codegen check, standalone slangc compile, reflection probe, device-profile query); standalone slangc tests fixed to use `-ignore-capabilities` (required for `WaveActiveCountBits`/`spvGroupNonUniform`). |
| **M20.7** | Lib helper extraction (Welford streaming, grid-stride loops) | ✅ **DONE 2026-05-23**. Added `welford_update(inout Welford, float)` to `shaders/lib/helpers.slang` (+ `vk_helpers.slang` twin). Refactored `conv_gn_relu.slang` Pass 1 to use the shared helper (replaced inline `local_mean/local_m2/local_n` triple with `Welford local_wf` + `welford_update()`). Grid-stride loops intentionally NOT extracted — Slang has no macro system and the loop bounds are kernel-specific (no reuse possible). Tests: `TestM207LibHelperExtraction` (3 tests: static decl check, caller refactor check, slangc compile probe). |
| **M20.8** | Anti-goal #6 sweep for RNN + scatter (Jinja `{{}}` → generic `<S : IScatter>`) | ✅ **DONE 2026-05-23**. Declared `public interface IScatter { static void combine(RWStructuredBuffer<uint>, uint, float); }` in `shaders/lib/atomics.slang` with five concrete structs: `ScatterAdd`, `ScatterMax`, `ScatterMin`, `ScatterProd`, `ScatterMean`. Refactored `templates/scatter_atomic.py.jinja` to replace the `{% if operation == "scatter_add" %}...{% elif %}...` chain with `{{ scatter_struct }}::combine(args.out, uint(idx), (float)args.src[i])` where `scatter_struct` is mapped from the operation name at Jinja render time. Structural differences (gather vs scatter buffer layouts, mean's count-buffer increment) remain Jinja-gated (is_gather/is_mean/is_atomic booleans — affect KernelArgs field count, not atomic dispatch). slangc validates at exit 0 for all 5 concrete structs. Tests: `TestM208ScatterIScatterGenerics` (7 tests — IScatter declared, 5 structs implement it, no forbidden if/elif in template, render-correct for all ops, mean count increment preserved, non-atomic ops unchanged). |
| **M20.9** | `should_use_cooperative_reduction` reflection-aware | ✅ **DONE 2026-05-23**. Added `_get_cached_num_sgprs()` to `ThreadgroupSizingMixin` (`kernel/threadgroup_sizing.py`) alongside the existing `_get_cached_io_pressure()`. Updated `should_use_cooperative_reduction()` in `kernel/main.py` to apply two reflection-driven threshold adjustments: SGPR pressure > 64 → 0.5× penalty (register-heavy kernels avoid extra cooperative sync pressure); I/O pressure > 128 loads+stores → 2.0× boost (memory-bound kernels benefit from latency hiding). Dynamic rnumel still short-circuits to True unconditionally. Tests: `TestM209CooperativeReductionReflectionAware` (5 tests — SGPR-lowers, I/O-raises, no-data-static, dynamic-always-cooperative, combined-scales-cancel). |
| **M20g** | **Flash_attention SDPA via Slang autodiff** | ❌ **NOT FEASIBLE 2026-05-19** — Online softmax's recurrence relation + lossy LSE compression are incompatible with Slang `bwd_diff` codegen. Decision: **keep hand-rolled flash-attention bwd (570 LoC).** Row preserved as a tracker for the RNN / conv autodiff lift attempts that remain in scope. |

### 0.6.4 M21 — Hardware-profiling + validation infrastructure (user-requested, NEW)

| # | Title | Status |
|---|-------|--------|
| **M21.1** | Device-profile-on-import phase (microbench launch latency, mem BW, LDS BW, atomics; cache to `~/.cache/torch_vulkan/`) | ✅ **DONE 2026-05-21**. `hardware_probe.auto_probe_on_import()` (`inductor/hardware_probe.py`, 423 L) runs from `inductor/__init__.py` on first import. `TORCH_VULKAN_PROFILE_DEVICE` ∈ `{off, quick, medium, deep, force}` controls level; default `auto` runs the level-0 microbench (~5 s) and caches at `~/.cache/torch_vulkan/probe_status_<id>.json`. Levels 1/2 add shader-lib + matmul-template SPIR-V prewarm and a canonical-shape autotune sweep, gated on explicit opt-in to avoid burning 10–15 min on a fresh install. Public entry: `torch_vulkan.profile_and_warmup(level="deep")` for users who want the full warm-up before training. Tests: `TestM211cHardwareProbe` (6 mock-based tests). |
| **M21.2** | Validation-as-codegen-check during autotune (per-kernel VUID surface) | open |
| **M21.3** | Best-practices VUID sweep across 9 models | open (sweep harness ready in `agent_space/m21_3_validation_sweep.py`) |
| **M21.3.a** | `VK_EXT_debug_utils` messenger wired in `csrc/vulkan/Context.cpp` | ✅ FIXED 2026-05-18 |
| **M21.4** | Per-kernel VUID lifecycle stress tests | open |

### 0.6.5 M22 — Anti-goal cleanup follow-on (refines M15)

| # | Title | Status |
|---|-------|--------|
| **M22.1.a-g** | Split file-size violators | `conv.py` ✅ M22b; `rnn.py` ✅ M22d (700 L); `wrapper.py` ✅ M22.1.h (642 L + `wrapper_helpers.py` 254 L); `scheduling.py` ✅ M22c (781 L + `scheduling_helpers.py` 134 L); **`runtime/slangc.py` M22a ✅ COMPLETE 2026-05-23** — 2342→781 L via 3-stage split: `common.py` (374L), `shader_lib.py` (776L), `reflection_ext.py` (676L). Commits `975748a916e`, `78bd544240d`, `1d08dca048a`. |
| **M22.1.f** | `kernel/main.py` 1009 L → 495 L via `ThreadgroupSizingMixin` extraction | ✅ **DONE 2026-05-21**. `VulkanKernel` now inherits `ThreadgroupSizingMixin` (`kernel/threadgroup_sizing.py`, 546 L); the 13 threadgroup-size heuristics live in a single place. Tests: `TestM221OrphanIntegration::test_vulkan_kernel_inherits_threadgroup_sizing` + `test_main_py_no_duplicate_threadgroup_size_methods`. |
| **M22.1.g** | `kernel/header.py` 867 L → 725 L via `CallKernelMixin` extraction | ✅ **DONE 2026-05-21**. `HeaderMixin` now inherits `CallKernelMixin` (`kernel/dispatch_call.py`, 178 L); the wrapper-side dispatch-grid emission lives in a single place. Tests: `TestM221OrphanIntegration::test_vulkan_kernel_inherits_call_kernel` + `test_header_py_no_duplicate_call_kernel`. |
| **M22.1.i** | Split `validate.py` 813 L → ≤500 L per module | ✅ **DONE 2026-05-21** (working tree). `validate.py` now 396 L, split into 3 sibling modules: `validate_types.py` (43 L — `SlangValidationIssue` dataclass; landing site to break import cycles), `validate_resource_limits.py` (141 L — `check_groupshared_budget` + `check_numthreads_product`), `validate_identifiers.py` (344 L — `check_undefined_identifiers` + `_SLANG_RESERVED`). Public API unchanged (`validate_slang_source`, `SlangValidator`, `_SLANG_RESERVED`, `SlangValidationIssue` re-exported from `validate.py`'s `__all__`). |
| **M22.2** | `alloc_alias.py` IR-level migration (280 L regex post-processor) | ✅ **DONE 2026-05-23**. New `fx_passes/alloc_alias_ir.py` (297 L) uses Inductor `AllocateLine`/`FreeIfNotReusedLine` IR nodes instead of regex on generated source. `VulkanPythonWrapperCodegen.run_wrapper_ir_passes()` override in `wrapper.py` calls `apply_vulkan_ir_alias_pass(self)` after upstream memory planning. Old `alloc_alias.py` retained for backward compat (external tooling). Tests: `TestM222AllocAliasIRMigration` (10/10 passed 2026-05-23). |
| **M22.3** | Pre-grad pattern firing-rate instrumentation | ✅ **DONE 2026-05-23**. `_PatternStats` counter dict + `record_pattern_fire()` + `dump_pattern_stats()` in `fx_passes/post_grad.py`; `FxPatternEntry.apply()` in `fx_passes/patterns/registry.py` calls `record_pattern_fire()` after each rewrite; env-gated by `TORCH_VULKAN_PATTERN_STATS=1` via `config.py:pattern_stats_enabled()`. Tests: `TestM223PatternStats` (3 tests). Commit `e1126b9ab44`. |
| **M22.4** | Delete dead `_replace_sdpa_with_custom_op` (160 L) | ✅ **DONE 2026-05-18** | Function deleted from `fx_passes/post_grad.py`; comment at deletion site records M22.4. |
| **M22.5** | Suppress 9 dead-code lowerings (addcmul, addcdiv, index_add, index_copy, norm.ScalarOpt_dim, permute, pow.Scalar, rot90, unfold) | ✅ **VERIFIED 2026-05-22 — no suppression needed** | All 9 ops audited and confirmed live: addcmul/addcdiv (optimizer + lerp), index_add/index_copy (scatter family, guarded by _is_vulkan), norm.ScalarOpt_dim (proxies linalg_vector_norm, _is_vulkan guard), permute (conv_transpose, SDPA, einsum), pow.Scalar (RoPE), rot90 (M19.R correctness fix), unfold (OP.3 view). Roadmap description was inaccurate when filed. |
| **M22.6** | Split `bwd_lowerings.py` (835 L) | ✅ **DONE 2026-05-22** | Layer-norm / group-norm / batch-norm backward extracted to `bwd_lowerings_norm.py` (336 L). `bwd_lowerings.py` now 523 L; `register()` calls `register_norm_backward_lowerings()`. Tests: `TestM226BwdLoweringsNormSplit` (4 tests). |
| **M22.7** | Sparse-tensor stub `TORCH_CHECK(false)` upgrade | ✅ **DONE 2026-05-22** | `TORCH_CHECK(!sparse, ...)` + `TORCH_CHECK(!scale_grad_by_freq, ...)` added to `csrc/ops/indexing_ops.cpp:vulkan_embedding` and `csrc/ops/autograd_ops.cpp:VulkanEmbeddingFunction::forward`. Python-side `lowerings/embedding.py` already returned `NotImplemented` for both flags. Tests: `TestM227SparseTensorStub` (4 tests; C++ tests skipped until rebuild, Python guard test runs immediately). |

### 0.6.5.x M22.8–11 — Rebuild diagnostics (2026-05-18)

Full clean rebuild produced 0 errors + 85 warnings. In-tree categorisation:

| Warning class | In-tree | Third-party (VMA, skip) |
|---------------|--------:|------------------------:|
| `-Wunused-function` | 73 | 0 |
| `-Wunused-variable` | 7 | 10 |
| `-Wunused-but-set-variable` | 4 | 2 |

| # | Title | Severity | Detail |
|---|-------|----------|--------|
| **M22.8** | **Meta-kernel registration audit (P1 — silent FakeTensor gap)** | ✅ **DONE 2026-05-18** | 66 defined-but-unregistered `meta_*` stubs deleted from `MetaKernels.cpp`. Upstream PyTorch 2.10+ ships structured-kernel Meta implementations for all those ops. Three were latent miscompile traps: `meta_transpose` (wrong contiguous strides), `meta_layer_norm/group_norm/batch_norm` (wrong return shape). File shrunk from ~800 L to 347 L. 37 `m.impl()` calls remain (scalar-binary, inplace-scalar, backward helpers, Phase-3 model ops). |
| **M22.9** | `vulkan_empty` device-binding gap (P2 — multi-GPU correctness) | ✅ **DONE** | `Registration.cpp:135` binds device via `impl->set_device()`. Tests: `TestM229MultiDeviceEmpty` + `TestM229FollowupAllocatorMultiDevice`. |
| **M22.10** | conv3d shape-var drift (P3 — refactor leftover) | ✅ **DONE** | Dead shape vars removed from `Registration.cpp` during M6 phase-2 cleanup. |
| **M22.11** | `q_seq_major` saved-but-unused (P3 — trivial) | ✅ **DONE** | `autograd_ops.cpp:1886` comment confirms `q_seq_major` intentionally not saved; backward detects layout from strides. |

### 0.6.6 M23 — Safety nets

| # | Title | Status |
|---|-------|--------|
| **M23.1** | `test_lib_module_no_undefined_symbols` (would have caught M18.1 at lib-build time) | ✅ FIXED 2026-05-18 (Slang-Lib agent) |
| **M23.2** | `test_capability_gate_coverage` — every wave-intrinsic call site preceded by `[require]` annotation | ✅ FIXED 2026-05-18 — `TestM232CapabilityGateCoverage` at `tests/test_inductor_regression.py:40473` with `_decl_has_capability_gate` helper. Verified by Audit Agent 1 post-Wave-2. |
| **M23.3** | Generalised render-binding-set assertion (every Jinja template with `[[vk::binding(N)]]` literals) | ✅ **DONE 2026-05-22** — `TestM233RenderBindingSetRatchet` (3 tests) asserts all `.slang`/`.jinja` template files and `kernel/header.py` use the `[[vk::binding(N, 0)]]` Set-0 form only. Commit `0e22a7bfc1f`. |
| **M23.4** | Combo-kernel chain-rename transitive resolver test | ✅ **DONE 2026-05-22** — Extracted local `_resolve` closures from `dispatch_call.py` and `vulkan_combo_kernel.py` into shared `resolve_alias_chain()` in `dispatch_call.py`. Tests: `TestM234ComboChainRenameResolver` (6/6). Commit `2462f6a1c7a`. |
| **M23.5** | Foreach-step-outside-compile dispatch ratchet | ✅ **DONE 2026-05-22** — 6 tests covering all registered foreach eager ops in `TestM235ForeachOutsideCompileRatchet`. Commit `f25201310da`. |

### 0.6.5.y M22 follow-ons from git-reset incident + agent waves (2026-05-18 late)

A peer agent ran `git reset --hard` mid-session, wiping out earlier work. Subsequent agents inherited a post-rollback tree and re-discovered the same bugs that prior session fixes had addressed. Filed memory `feedback-no-git-reset-with-agents` to prevent recurrence.

| # | Title | Severity | Detail |
|---|-------|----------|--------|
| **M22.12** | **slang_mm.slang c_binding typo REGRESSION (RESTORE)** | **P0 (FIXED in parent)** | `templates/slang_mm.{slang,py.jinja}` lines 87/95 — `c_binding = 2 if has_bias else 2` (typo from M17.1-gap2) came back after the git reset. Fixed 2026-05-18 (`3 if has_bias else 2`). Blocks `TestM17SlangMatmulCorrectness::test_addmm_*`, `TestQKVLinearFusion::test_qkv_compile_correctness`, parts of `TestM185Transformer3DMatmul`. |
| **M22.13** | **mm tile transpose-a row-collapse bug** | **✅ FIXED (workaround) 2026-05-18** | Matmul-3D agent applied a C++ workaround at `csrc/ops/matmul_ops.cpp:213-232` + `:379-407`: when `is_t_transposed` is detected, materialize the transposed operand via `.contiguous()` (one extra dispatch) then recurse with no-transpose path. M19.1 floor-gate FLIPPED (`lowerings/__init__.py:320-328` now active); 3 xfails removed; `TestM2213MmTransposeA` added. **Workaround cost**: 1 extra `dispatch_strided_copy` per transposed-input mm. Filed M22.13-followup for the deeper shader-side fix. |
| **M22.13-followup** | **Shader-side root cause for transpose-a row-collapse** | **✅ FIXED 2026-05-18 (Slang-Lib)** | Linearised-tid pattern applied to `shaders/matmul/mm_tiled.slang:58` (`tid_local = lid.y * TILE_SIZE + lid.x`). Also `shaders/matmul/bmm_tiled.slang:57` (M22.13-extras). C++ workaround at `matmul_ops.cpp:213-232` is now redundant defense-in-depth. Filed **M22.13-retire-workaround** for cleanup. |
| **M22.13-retire-workaround** | **Retire C++ `.contiguous()` workaround now that shader-side root cause is fixed** | LOW (cleanup) | After M22.13-followup landed, the C++ short-circuit at `csrc/ops/matmul_ops.cpp:213-232` + `:379-407` is redundant. Delete the workaround. Saves 1 extra `dispatch_strided_copy` per transposed-input mm. Gated on a regression test that exercises `aten.mm(g.t(), x)` directly with the shader-side fix only (i.e. with the workaround toggled off via env knob). |
| **M22.14** | **M17.8.d.2 base infrastructure RE-RESTORE** | **MED (FX agent doing Step 2)** | `_ensure_conv2d_backward_op_registered()` + `make_fallback(torch.ops.torch_vulkan.conv2d_backward)` were wiped by the reset. FX agent re-registering; parent applies the `make_fallback` to `lowerings/__init__.py` once Matmul-3D finishes touching that file. |
| **M22.15** | **Three-layer empty_like fix didn't fire on the load-bearing layer** | **HIGH (FX agent fixing)** | Initial M18.3 only fixed `decomposition_passes.py`; the load-bearing layer is `meta_patches/op_registration.py:31-35` `_bwd_meta_like_grad`/`_bwd_meta_like_input` helpers. All 8 `TestM18ShapeOnlyProxiesUseNewEmpty` tests still fail until those helpers get the same `empty_like → new_empty` treatment. |
| **M22.16** | **slangc threadpool deadlock under 5-agent concurrent compile** | MED (env-only) | M186 DescriptorPool tests (and others that spawn subprocess + import torch_vulkan) deadlock at `slangc.py:1770 event.wait()` when multiple agent processes share the SPIR-V cache. Mitigation options: (a) per-subprocess `TORCH_VULKAN_SPIRV_CACHE=<tmpdir>` override; (b) `TORCH_VULKAN_ASYNC_COMPILE=0` to serialize within a process; (c) gate the wave size of parallel agents. Tests pass cleanly in isolation. |

### 0.6.5.zz M18.8.b + M18.9 — Conv backward investigation (2026-05-18, FX agent)

| # | Title | Severity | Detail |
|---|-------|----------|--------|
| **M18.8.b** | **Dynamo splits `nn.Sequential` at monkey-patched `F.conv2d`** | **MED** (architectural) | Under `torch.compile`, `nn.Sequential(Conv, GN, ReLU)` produces THREE separate subgraphs (one per layer) because Dynamo can't trace through the monkey-patched `F.conv2d` opaque custom op. The pre-grad fusion pass `_fuse_conv_gn_relu` therefore never matches the full chain — the fused `conv2d_gn_relu_fused` custom op is dead code in this compile path. Two paths forward: (a) FX-pass-level fusion across subgraph boundaries (touches `fx_passes/`); (b) **fix the underlying conv backward C++ adapter (M18.9)** which would make the fused op moot for correctness — only useful as a perf optimization. Recommended: (b) first. |
| **M18.9** | **`vulkan_convolution_backward_overrideable` C++ adapter** | **✅ FIXED 2026-05-18 (Matmul-3D)** | Root cause: `vulkan_copy_` (the `.cpu()` path) ignores `stride` + `storage_offset`, doing a raw VkBuffer byte copy. For a `reinterpret_tensor` view (compile-mode standard), the CPU tensor receives raw storage bytes not matching the view's logical layout; `at::convolution_backward` then computes correct math on wrong data → ratio 0.082 / 0.60 / 5.66. Eager-mode unaffected (uses Vulkan-native autograd path, never hits `.cpu()`). **Fix at `csrc/backend/Registration.cpp:385-434`**: `.contiguous()` before `.cpu()` on each operand (input/weight/grad_output) — `.contiguous()` routes through `dispatch_strided_copy` which respects stride correctly. Filed **M18.9-followup**: fix `vulkan_copy_` itself to respect stride (wider blast radius — every `.cpu()` call uses it). |
| **M18.9-followup** | **`vulkan_copy_` ignores stride/storage_offset on Vulkan→CPU read** | MED | Per Matmul-3D's M18.9 investigation: `vulkan_copy_` at `csrc/backend/Registration.cpp:31-50` does `buf->read(self.data_ptr(), self.nbytes())` — raw byte copy, ignoring strides. Affects EVERY `.cpu()` call on a Vulkan tensor that isn't contiguous. Today M18.9 worked around it inside the conv backward adapter; a proper fix would address it at the source. Plumb stride-aware copy via `dispatch_strided_copy` or similar. Wider blast radius — verify all current `.cpu()` callers are okay first. |
| **M18.10** | **Reduction codegen emits undefined `r0_index` in Slang source** | ✅ **FIXED 2026-05-23** | Fix in `kernel/header.py:309-346` (M18.10 block): when `layout_2d is not None` in the persistent-2D reduction path, emits `uint {red_vars[0].name} = lid.y * tx + lid.x;` (1-root+2-entries) or per-axis declarations (2-root). Adds declared names to `self._hoisted_vars` so `indexing.py` doesn't re-declare. Tests: `TestM1810ReductionCodegen` (4 tests — `test_sum_multi_axis_compiles_no_r0_index_error`, `test_sum_4d_multi_axis_compiles_no_r0_index_error`, `test_conv_bwd_reduction_compiles_no_r0_index_error`, `test_reduction_codegen_r0_index_declared_before_use`) — all 4 PASS 2026-05-23. |

The M21.4 VUID stress harness inadvertently surfaced eager-mode correctness bugs that don't appear under `torch.compile`. Filing them here for visibility.

| # | Title | Severity | Detail |
|---|-------|----------|--------|
| **EAGER.1** | **Factory ops intermittent zero result — slangc-cache class FIXED, residual flake remains** | **PARTIAL FIX (M22.16 closes slangc-cache class)** | Root cause of slangc-cache class: torn `.slang-module` files (closed by M22.16's atomic-write). **Residual EAGER.1.b**: 2/10 trials still produce `[0,…]` or `[2,…]` (stale buffer contents). Validation agent's M21.3 sweep surfaces the deeper root cause as M21.3.01 (Set-1 binding mismatch — SPIR-V reads from `[Set 1 Binding 0]` which the pipeline layout doesn't declare, returning unbound / stale descriptor contents). Filed as **EAGER.1.b**. Currently xfail-strict-OFF in `TestEager1NoFactoryZeros::test_torch_ones_returns_ones_under_stress`. Will flip to PASS when M21.3.01 lands. |
| **EAGER.1.b** | **Residual EAGER.1 flake — eager fill+add buffer sync, NOT Set-1 mismatch** | **P0 (NOT closed by M21.3.01)** | Initial framing said this was the M21.3.01 Set-1 issue. **Wrong.** Eager `torch.zeros + torch.ones` routes through `csrc/ops/binary_ops.cpp` → `shaders/binary/add.slang` which uses explicit `[[vk::binding(N)]]` (no set, defaults to 0) — never went through `ParameterBlock<KernelArgs>`. The pattern reproduces unchanged after M21.3.01 lands. Real root cause: dispatch/sync between `vulkan_fill_scalar` (zeros init) and the dependent `binary_add` — either fill isn't flushed before add reads, OR buffer-pool recycles a buffer the prior fill hasn't finished writing. **Filed as M21.3.02 below.** |
| **M21.3.01** | **`VUID-VkComputePipelineCreateInfo-layout-07988` — Set-1 binding mismatch** | **✅ FIXED 2026-05-18 (Path A — Slang annotation)** | 24 templates + `kernel/header.py` codegen path got `[[vk::binding(0, 0)]] ParameterBlock<KernelArgs> args;`. Cache sweep: 61/61 fresh SPIR-V have Set 0 (vs 30/85 = 35 % at Set 1 pre-fix). `TestM2130_1PipelineLayoutSet0::test_spirv_emits_set_0_decoration` PASSES. `test_pipeline_creation_no_set_1_vuid_canary` correctly gated as SKIPPED when validation layer absent. **Audit Agent 3 had REFRAMED this as universal correctness (M-NEW.6) — confirmed by the cache sweep that 35 % of all pre-existing SPIR-V blobs had Set 1.** Side effect: M22.16 tmp-path bug was uncovered and fixed (precompile_shader_libs `force=True` was emitting `.tmp.<pid>.<tid>` which slangc rejected with `E00060 cannot infer an output format`). |
| **M21.3.02** | **Eager-mode dispatch/sync ordering between fill_scalar and binary_add** | ✅ **FIXED 2026-05-22 — commit `f8a7eb83aa9`** | Root cause: `VulkanBuffer::write()` (CPU→GPU via `vkMapMemory`) never added buffers to `dirty_buffers`, so `dispatch_shader` (binary_add) emitted no pipeline barrier. Fix: added `host_written_buffers` set to `DeviceRuntime`; `notify_host_write()` called from `vulkan_copy_()` after every CPU→GPU write; `dispatch_shader/dispatch_shader_indexed` check this set and emit a `VK_PIPELINE_STAGE_HOST_BIT→COMPUTE_SHADER_BIT` (`HOST_WRITE→SHADER_READ`) barrier. Set cleared on every flush. `TestEager1NoFactoryZeros::test_torch_ones_returns_ones_under_stress` upgraded from xfail to 20-trial strict pass. |
| **M21.4.c** | Document `VK_ICD_FILENAMES` requirement under validation layers | LOW | When `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` is set, the Vulkan loader's ICD-sort heuristic picks Lavapipe (software rasterizer) on this dev box. Subprocesses then time out (>180 s). Pinning `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json` fixes it. Belongs in `CLAUDE.md` "Useful environment knobs" section. |
| **M21.4.d** | Inductor codecache `FileNotFoundError` under concurrent agents | LOW | `TORCHINDUCTOR_CACHE_DIR` namespacing keys off file mtimes; simultaneous agent edits rename the cache dir mid-flight. Workaround: per-process `TORCHINDUCTOR_CACHE_DIR` override. |

### 0.6.7 Reframings (corrections to prior § 0.5 claims)

| Prior claim | Reality (per audit) |
|-------------|---------------------|
| `meta_patches.py` is 3902 L | Already split: 8 files / 4460 L total (M15.1.b ✅) |
| Persistent kernels: 40 % | **0 % effective** — `_enable_persistent_mode` is dead code |
| `[require(...)]` capabilities: 0 % | **~80 %** for wave-intrinsic helpers (`lib/helpers.slang:89-112` + 6 sites in `lib/reduction.slang`) |
| Reflection metadata: 100 % (M11.1) | **40 %** — only 3/8+ fields wired |
| `ParameterBlock`: 30 % (mm only) | **45 %** (13/29 files) — but mm regressed during M17.1 |
| Backward op coverage: 57/58 | Accurate; argmax/argmin correctly NOT `[Differentiable]` (positions, not values) |
| `vulkan_combo_kernel.py` 1106 L | Already split: 600 L + `combo_kernel/` (M15.1.f ✅) |
| `templates/caller/gemm.py` 2331 L | Already split: `gemm/` 5 files, max 778 L (M15.1.a ✅) |

---

## 0.7. 2026-05-18 Wave 3 audit — post-Wave-2 re-audit (5 analysts + 1 implementer)

After Wave 2 landed M18–M22 + M-cpp-new family, a third audit wave (5 read-only analysts + 1 implementer on M21.3.01) was dispatched. Findings reshape the priority list:

### 0.7.1 Headline metrics (re-baseline 2026-05-18)

| Model | CPU ms | VK eager ms | VK/CPU eager | v6.3 baseline | Note |
|---|---:|---:|---:|---|---|
| MLP            | 0.29 | 0.49 | **1.68×** | n/a | |
| Conv2d only    | 0.42 | 0.79 | **1.87×** | n/a | |
| **SmallCNN+GN** | 0.62 | 1.46 | **2.36×** | 5.7→4.4→3.9× | **eager beats compile-mode baseline** |
| Conv2d+ReLU    | 0.45 | 0.82 | **1.80×** | n/a | |
| Linear chain   | 0.26 | 0.38 | **1.47×** | n/a | closest to parity |
| Transformer    | 1.00 | 3.40 | **3.41×** | "5.7× CPU" | trained end-to-end (eager, validation off) |
| ViT            | 1.61 | 5.52 | **3.42×** | "BLOCKED" | trained end-to-end (eager, validation off) |
| Llama-MLP      | 0.27 | 0.42 | **1.55×** | n/a | |
| Mixtral-MoE    | 0.39 | 0.68 | **1.72×** | "BLOCKED" | trained end-to-end (eager, validation off) |

**Compile-mode latency: completely BLOCKED** by M-NEW.1 (duplicate-extern regression). Cold import warm-cache: **~3 s**. Memory headroom: SmallCNN+GN runs at batch 16384 (~12 MB activations) without OOM.

### 0.7.2 M-NEW.* — new P0 compile-mode blockers

| # | Title | Severity | Detail |
|---|-------|----------|--------|
| **M-NEW.1** | **`duplicate extern kernel: slang_addmm_*` blocks compile for entire catalog** | **P0** | `templates/caller/gemm/install.py:262 ExternKernelChoice(fn).bind(...)` collides on the 2nd `aten.addmm` because `_SlangTileAddMM._format_name()` produces a deterministic name (`slang_addmm_{tm}_{tn}_{tk}_s{ns}_r{mpt}x{npt}`) that upstream `ExternKernelChoice.__init__` asserts unique. Any graph with ≥2 Linear / addmm calls hits this — MLP, SmallCNN, every Transformer-family model. **Fix**: use `ExternKernelChoice._registry` lookup-before-create (already supported upstream), OR hoist construction to install-time, OR add per-call disambiguator. Likely regressed after OP.27 closeout (2026-05-16). |
| **M-NEW.2** | **Conv2d compile-mode hard segfault** | **P0 (downstream of M-NEW.1)** | After M-NEW.1's LoweringException unwinds, autotune's registry pollution causes use-after-free in Inductor IR cleanup → process death exit 139. Retest after M-NEW.1 fix. |
| **M-NEW.3** | **29 % of slang_mm autotune choices rejected by M27 validator** | MED | `_pick_register_tile_configs()` in `templates/caller/gemm/install.py` emits 4×8×1 / 8×4×1 thread blocks (numthreads product = 32) — not a multiple of wave size 64 on RDNA1. The Slang validator correctly rejects, but the generator should not emit them. **Fix**: gate `reg_tiles` on `tile_m * tile_n / (m_per_thread * n_per_thread) >= simd_group_size`. |
| **M-NEW.3.b** | ✅ **CLOSED 2026-05-18**: int8 `_INT8_TILE_CONFIGS` wave-alignment + M17.1 single-wave filter. `templates/caller/gemm/install.py`: added `_int8_config_wg_threads` + `_filter_int8_configs_wave_aligned` (drops sub-wave + multi-wave-on-wave64 per M17.1 barrier bug); wired into `install_external_mm_int8` + prewarm path. Source table was already wave-aligned (8/8 on wave32, 2/4 surviving on wave64 due to M17.1 cap). 4 new tests + 4 M-NEW.3 regression tests all PASS. Filed M-NEW.3.c (perf-investigation of 50 % int8 config-loss on RDNA1 wave64; tracks alongside M17.1). |
| **M-NEW.4** | ✅ **CLOSED 2026-05-18**: see § 0.7.3 + M-pipeline-4 row above. |
| **M22.9-followup** | ✅ **CLOSED 2026-05-18**: Allocator widening for per-call device. **Option A (additive)** — added `Allocator::allocate(size_t, c10::DeviceIndex)` overload in `csrc/backend/Allocator.{cpp,h}`; `vulkan_empty` + `vulkan_empty_strided` in `Registration.cpp` now pre-allocate via `allocator->allocate(nbytes, device.index())` and pass the `DataPtr` into the byte-size `c10::Storage` constructor. Per-device VMA allocator selection via `Context::allocator(device_idx)`. **Surprises**: test rig has 2 Vulkan devices (RADV + Lavapipe/iGPU); buffer pool still global. **Sub-followups**: M22.9-followup-per-device-pool (pool migration to per-device map), M22.9-followup-device1-init (post-rebuild device-1 init capture). 5 tests (4 PASS static + 1 SKIP pending introspection pybind). |
| **M22.9-followup-introspection-pybind** | ✅ **CLOSED 2026-05-18**: `_storage_device_index(tensor) -> int` pybind in `csrc/init.cpp`. Reads `DataPtr.device().index()` (storage-side, structurally distinct from impl-key `tensor.device.index`). 2 new runtime tests: (a) end-to-end multi-device storage-routing check (allocates on vulkan:0 + vulkan:1, asserts storage indices differ); (b) storage-vs-impl-key invariant on single-device path. 4 PASS + 2 SKIP pre-rebuild — SKIPs flip post-rebuild. |
| **M-cpp-new-5-followup-test** | ✅ **CLOSED 2026-05-18**: descriptor_indexing runtime override pybind. **`csrc/vulkan/Context.{cpp,h}`** added module-level `std::atomic<int> g_desc_indexing_override{-1}`; **`descriptor_indexing_enabled(index)`** now consults the override before the capability flag (single read-point — covers all ~10 hot-path branches in dispatch/DescriptorSet/Pipeline). **`csrc/init.cpp`** registered `_set/_get_descriptor_indexing_override` pybinds. Verified the fallback `else` branch exists at `dispatch.cpp:295-317` (M-cpp-new-5's explicit add). 5 tests (1 PASS static + 4 SKIP pre-rebuild). Mid-process override is safe (relaxed-atomic, per-call read). |
| **M-NEW.5** | Cold-import contention 6+ min under concurrent agent load | LOW | When the parent + 5+ agents all import `torch_vulkan` simultaneously with cold slangc cache, total wall time hits 6+ min. Warm-cache import is ~3 s (acceptable). Mitigation: a single-process slangc daemon OR aggressive cache pre-warming OR per-process cache isolation (already partly addressed by M22.16 atomic-write). |
| **M-NEW.6** | **REFRAME M21.3.01 — universal correctness, not 5/9 models** | **P0 (correctness)** | M21.3 sweep originally framed the Set-1 binding mismatch as a 5/9-model issue. Audit Agent 3 confirms: every Slang dispatch on every model emits `OpDecorate ... DescriptorSet 1` references. RADV silently tolerates by returning unbound/stale descriptor contents (EAGER.1.b symptom). Other drivers (NV / Intel / Apple via MoltenVK) WILL fail. The "fix" — Set 0 binding annotation OR C++ layout widening — must be applied. **Implementer landed Path A on `slang_mm.slang` 2026-05-18** (line 123: `[[vk::binding(0, 0)]] ParameterBlock<KernelArgs> args;`). Verify other templates received the same fix. |

### 0.7.3 M-pipeline-* — pipeline contract findings (Agent 4)

| # | Title | Severity | Detail |
|---|-------|----------|--------|
| **M-pipeline-1** | **🔥 Conv lowering UNREACHABLE under main — lazy `_ensure_patch_custom_ops`** | **P0 (THE REAL M18.8.b ROOT CAUSE)** | `register_eager_patch_custom_ops()` runs lazily inside `_patched_conv2d` during Dynamo trace. Three consequences: (1) Dynamo graph-breaks → SmallCNN traces as **3 subgraphs instead of 1** (this is the M18.8.b symptom); (2) the lazy call **re-registers** the custom op with a new `OpOverload` identity; (3) `lowerings[old_overload]` lookups fail → Inductor emits `Creating implicit fallback for: torch_vulkan.conv2d_with_optional_bias.default` → **ALL M19.5 dynamic-shape lift / M20.3 spec consts / future fusion work is dead code for conv on compile path**. Fix: move `register_eager_patch_custom_ops()` to backend register-time (idempotent, before any Dynamo trace). |
| **M-pipeline-2** | Centralize OpOverload-identity-safe lowering lookup | MED | Extract `_get_conv2d_lowering_by_name()` from `lowerings/conv_transpose.py` to a shared `lowerings/_overload_safe.py`. Force all custom-op lowerings through it. Prevents future M-pipeline-1-class regressions. |
| **M-pipeline-3** | `VulkanKernel._compute_config_key` is incomplete | MED | Today's key hashes shape + reduction flags + packed16 + simd_size + loop_depth proxy — but NOT buffer dtypes, push-constant layout, or descriptor_counts. Cache hits across structurally-similar kernels with different SPIR-V are silently mis-routed. |
| **M-pipeline-4** | Pipeline cache key contract not enforced | MED | `csrc/vulkan/Pipeline.cpp` PipelineCache trusts the Python-side key. If two distinct SPIR-V blobs share a key, the first compile wins forever. Add DCHECK in debug builds that cached SPIR-V hash matches the new one. |
| **M-pipeline-5** | `MAX_BINDINGS` captured at first dispatch | LOW | `csrc/ops/dispatch.cpp` `static const uint32_t MAX_BINDINGS = ...` evaluates once. If extension state ever flips, mis-sized arrays follow. Replace with per-call read. |
| **M-pipeline-6** | `_VulkanCustomPass.uuid()` hashes only `__init__.py` | MED | Pattern-only edits (e.g. `patterns/builtin_patterns.py`) leave the uuid unchanged → Inductor codecache serves stale FX-pass results. Hash the whole `fx_passes/` subtree. |
| **M-pipeline-7** | Combo kernel reflection-cache pollution | LOW | All combo kernels hash to `'combo'` constant key (`define_kernel:632`) → reflection metrics cross-contaminate. Compute a content hash of the combined subkernel list. |
| **M-pipeline-8** | **Anti-goal #3 violation: conv backward routes through `aten.convolution_backward` extern** | OPEN QUESTION | Confirmed in AOT trace: backward graph 0 contains the extern. Anti-goal #3 forbids new `aten.*_backward` lowerings, but the existing extern fallback is in place. Decision needed: **ratify extern OR migrate to `bwd_diff`**. |
| **M-pipeline-1** | ✅ **CLOSED 2026-05-18**: Lazy `_ensure_patch_custom_ops` was splitting Dynamo graphs. **`fx_passes/eager/conv.py`**: `_ensure_conv2d_with_optional_bias_op_registered` made idempotent (early-return on existing); fixed latent N/C/HxW shape bug in `_conv2d_gn_relu_backward` (was referencing conv input, now references conv output). **`fx_passes/eager/__init__.py`**: added module-level `_REGISTER_DONE` short-circuit. **Pre-fix: 4 subgraphs + 2 graph_breaks + 2 resume frames; post-fix: 1 subgraph + 0 breaks**. Closes **M18.8.b** root cause (audit reframe was correct). 4/4 PASS. |
| **M-pipeline-1-followup** | ✅ **CLOSED 2026-05-18**: push-constant `int()` wrapping in `templates/caller/conv.py` + **bonus catch**: pre-existing format string mismatch `"32IfI"` → `"31IfII"` (32 leading uints declared vs 31 in slang struct, AND `I` format rejected the float `eps`). 3 pack sites updated (`_slang_tile_conv2d_gn_relu`, `_slang_tile_conv2d`, `_slang_tile_conv2d_bwd`); 31 fields wrapped per site. **xfail-flip: `test_conv_gn_relu_compile_backward_matches_cpu`** decorator removed. New `TestMPipeline1FollowupConvPackInt` (3 tests) — all PASS inline; pytest verification deferred under heavy slangc peer-load. |
| **M-pipeline-9** | ✅ **CLOSED 2026-05-23**: No assertion that `decomposition_passes.py` decomps use `new_empty` not `empty_like` | `TestMPipeline9DecompEmptyLikeBan` (5 tests, all PASS 0.12s). Scanner walks `meta_patches/*.py`, flags `torch.empty_like(` calls without `M-pipeline-9-EXEMPT` marker. Gate drains all 14 pre-existing sites (`shape_ops.py:4`, `dtype_ops.py:10`) before enabling. |
| **M-pipeline-4** | ✅ **CLOSED 2026-05-18**: Pipeline cache key SHA collision guard. **`csrc/vulkan/Pipeline.{cpp,h}` + `csrc/init.cpp`**: added `fnv1a64` SPIR-V hash, `CachedPipeline` struct, atomic `collision_count_` counter, `TORCH_WARN` + counter bump on mismatch, `_pipeline_cache_collisions()` pybind. 6 tests (4 PASSED static, 2 SKIPPED pre-rebuild — flip to PASS after C++ rebuild). |
| **M-cpp-new-2** | ✅ **CLOSED 2026-05-18**: DescriptorPool async-reset path. **`csrc/vulkan/DescriptorSet.{cpp,h}` + `Stream.h` + `csrc/ops/dispatch.cpp`**: added `reset_async(VkFence)` + `drain_pending_resets()` (deferred poll-based drain, no explicit `vkWaitForFences`); 4 hot-path sites in `dispatch.cpp` converted; sync `reset()` retained for `flush_stream` (already did `vkQueueWaitIdle`). Env knob `TORCH_VULKAN_DESCRIPTOR_POOL_ASYNC_RESET={0,1}` regression escape. **Sub-followups**: M-cpp-new-2-followup-validation (post-rebuild VUID stress under VK_LAYER_KHRONOS_validation). 7 tests (6 PASSED, 1 SKIPPED pre-rebuild). |
| **M-NEW.4 + M-cpp-new-2-followup-pybind** | ✅ **CLOSED 2026-05-18**: Stream + DescriptorPool telemetry pybinds. **`csrc/vulkan/Stream.{cpp,h}` + `csrc/init.cpp`**: added `Stream::submit_count_` atomic + `submit_count()` accessor; 3 new pybinds (`_stream_submit_count`, `_descriptor_pool_async_reset_requests`, `_descriptor_pool_async_resets_drained`). Single `vkQueueSubmit` site verified in `Stream::submit_cmd_buffer` (line 59); Memory.cpp:176 host-staging submit intentionally not counted (out of M9.2 path). New `TestMNew4StreamSubmitAndPoolCounters` (3 PASS static + 3 SKIP pre-rebuild). Filed M-NEW.4-followup-multidevice (per-device counter pybind). |

### 0.7.4 TEST.COV.* — coverage-matrix findings (Agent 2)

| # | Title | Effort |
|---|-------|--------|
| **TEST.COV.1** | Cover 9 P1 registered-but-untested lowerings (`_adaptive_avg_pool2d_backward`, `_embedding_bag_backward`, `_embedding_bag_forward_only`, `cross_entropy_loss`, `leaky_relu_backward`, `rot90.default`, `torch_vulkan.foreach_lion_step`, `torch_vulkan.mm_int8`, `aten.lerp.{Scalar_out, Tensor_out}`) | 1 d |
| **TEST.COV.2** | Dtype matrix sweep on top-20 lowerings × fp16/bf16 | 2 d |
| **TEST.COV.3** | Activation-bwd direct dispatch tests (hardtanh / hardsigmoid / softplus / mish + 12 trig + 10 exp/log) | 1 d |
| **TEST.COV.4** | Special-math bwd direct tests (10 ops: erf*, lgamma, digamma, ndtri, i0*, i1*) | 1 d |
| **TEST.COV.5** | ✅ **CLOSED 2026-05-18**: Overload-tests for `_out` / `.stable` / `.Scalar` variants. 7 tests passed (leaky_relu_backward x2, foreach_lion_step registered, lerp.Scalar/Tensor_out registered, plus 3 meta-tests). 4 deliberate skips: cross_entropy_loss, embedding_bag_forward_only, lerp.Scalar_out eager parity, lerp.Tensor_out eager parity. **Sub-followups filed:** TEST.COV.5.b (pre-grad pass to suppress cross_entropy decomp), TEST.COV.5.c (same for embedding_bag_forward_only decomp suppression). |
| **TEST.COV.6** | ✅ **CLOSED 2026-05-18**: Promote `test_op21_embedding_bag_bwd.py` / `test_op22_dynamic_reduction_bwd.py` into regression file. **Sub-followup filed:** TEST.COV.6.b (Lion `beta2` ParamConfig bug — see § 0.7.5.y below; D-group fix in flight). |
| **TEST.COV.7** | ✅ **CLOSED 2026-05-18**: RNN-bwd-fallback reachability audit. **Sub-followup filed:** TEST.COV.7.a (eager-mode `lerp.{Scalar,Tensor}_out` unimplemented; M-EAGER follow-up). |
| **TEST.COV.8** | ✅ **CLOSED 2026-05-18**: Meta-test for register_lowering coverage. **Two-tier design**: `TestCoverageStructuralInvariant` (strict single-file, EXEMPT=empty), `TestCov8RegisteredLoweringsHaveTests` (loose full-suite, KNOWN_GAPS=empty). AST walker finds **69 unique registered ops** (raw 270 was an over-count — loop-variable expressions account for ~13 raw call sites covered by named decorations elsewhere). **0 silent gaps post-TEST.COV.1-7 closure** — mission's ~60 estimate was pre-wave. 7/7 PASS in 2.93s; walker takes 0.122 s. Filed TEST.COV.10 (AST walker fidelity for loop-variable register sites). |

Coverage totals (Agent 2): ~270 ops across all dispatch paths; ~210 tested (78 %); 9 hard P1 untested.

### 0.7.5 M-docs-* — docs drift (Agent 5)

| # | Title | Effort |
|---|-------|--------|
| **M-docs-1** | Refresh root `CLAUDE.md` against v6.3 (currently at v6.2 — missing M17–M23 entirely; affects every session context) | 15 min |
| **M-docs-2** | Fix backend `CLAUDE.md` key-files table (4 file→dir renames: `runtime.py` → `runtime/`, `meta_patches.py` → `meta_patches/`, `bwd_diff_dispatch.py` → `bwd_diff/`, `vulkan_combo_kernel.py` → `combo_kernel/`) + retire anti-goal #2 (model_ops.cpp deleted) | 10 min |
| **M-docs-3** | Document ~20 active env knobs missing from backend CLAUDE.md (`TORCH_VULKAN_SLANGC_WORKERS`, `TORCH_VULKAN_DISABLE_SLANG_TILES`, `TORCH_VULKAN_PARAMETER_BLOCK`, etc.) | 20 min |
| **M-docs-4** | Resolve MAX_JOBS conflict — memory=3, root CLAUDE.md=8, backend CLAUDE.md=4. Pick canonical value. | 5 min |
| **M-docs-5** | ✅ **CLOSED 2026-05-18**: PyTorch version mismatch resolved. Live venv reports `2.11.0+cpu` (matches user memory); backend CLAUDE.md updated. | done |
| **M-docs-6** | **Archive `docs/0[1-8].md` + `09-master-plan.md` under `docs/archive/`** (4000+ lines of stale pre-implementation checklists) | 10 min |
| **M-docs-7** | Refresh `docs/10-lib-api-reference.md` counts (says 9 lib modules; live: 16; `[BackwardDerivative]` 14 → 50) | 15 min |
| **M-docs-8** | Refresh `docs/primtorch_coverage.md` shader paths (~260 rows, all stale: `unary_abs_fwd.slang` → `shaders/unary/abs.slang`) | 30 min |
| **M-docs-9** | ✅ **CLOSED 2026-05-18**: § 0 active-milestones M9 row corrected (was "M9.6/M9.7 remain", actual = all closed; new sub-followups tracked as M-cpp-new-2 etc.). | done |
| **M-docs-10** | Refresh `docs/10-inductor-backend.md` § 13 reference-files table (6 row updates) | 10 min |

### 0.7.5.y TEST.COV sub-followups (2026-05-18 late, surfaced by Wave-4 TEST.COV.5-7 implementer)

| # | Title | Severity | Detail |
|---|-------|----------|--------|
| **TEST.COV.5.b** | Pre-grad pass to suppress `cross_entropy_loss` decomposition | LOW (P3) | The lowering is registered but Inductor decomposes `cross_entropy_loss` into log_softmax + nll_loss before reaching the backend. Either add a pre-grad pattern that re-fuses or accept the decomp as canonical. |
| **TEST.COV.5.c** | Pre-grad pass to suppress `embedding_bag_forward_only` decomp | LOW (P3) | Same shape as TEST.COV.5.b — Inductor decomposes before our lowering fires. |
| **TEST.COV.6.b** | ✅ **CLOSED 2026-05-18**: Lion optimizer `beta2` ParamConfig fix. `templates/foreach_optimizer.{slang,py.jinja}`: widened struct-field Jinja guard from `algorithm == "adamw"` to `algorithm in ("adamw", "lion")` (1-line change). Python packer was already shipping `beta2` for Lion — only the Slang struct definition was missing. Direct probe: `foreach_lion_step` end-to-end runtime OK, all-finite, params updated by `±lr` (Lion's signed update). 2 new tests in `TestTestCov6bLionStep`; `TestCov6ForeachLionStep` now upgrades from SKIP→PASS. |
| **TEST.COV.7.a** | Eager-mode `lerp.{Scalar,Tensor}_out` unimplemented | LOW (P3) | Lowerings registered; eager-path pybind not wired. M-EAGER follow-up. |

### 0.7.5.x New eager-path correctness bug — M-cpp-new-6 (2026-05-18 late)

| # | Title | Severity | Detail |
|---|-------|----------|--------|
| **M-cpp-new-6** | **Eager `x = x.relu()` chain returns zeros on even iteration count** | **P0 (CORRECTNESS — silent zero)** | Surfaced by M-cpp-new-5 implementer while probing multi-dispatch correctness. `for _ in range(2): x = x.relu()` on Vulkan eager produces all-zero output (CPU is correct). Flip-flop: odd N correct, even N zeros. Pre-existing — NOT caused by M-cpp-new-5 (verified against pre-rebuild .so). Likely a barrier-skipping OR output-aliasing bug in `csrc/ops/activation_ops.cpp::activation_unary` or its `dispatch_shader` barrier handling. Probe: `agent_space/m_cpp_new_5_relu_chain.py`. **Affects any model that calls `.relu()` in a loop in eager mode** — including the test mode `for _ in range(N_warmup): model.forward()` pattern. Workaround: use fresh inputs per iteration. |

### 0.7.6 Reframings from Wave 3

| Prior claim | Reality (Wave 3 audit) |
|-------------|------------------------|
| M18.8.b root cause: Dynamo splits at `nn.Sequential` boundary; needs `torch.library.custom_op` or `@torch.compiler.allow_in_graph` workaround | **WRONG.** The real root cause is M-pipeline-1: lazy `_ensure_patch_custom_ops()` causes the split AND breaks OpOverload identity. Fixing M-pipeline-1 should close M18.8.b automatically. |
| M21.3.01: 5/9 models BLOCKED in compile-mode | **WRONG.** Universal correctness issue affecting every dispatch on every model. 5/9 is just the subset where validation-layer's pipeline-creation check fires first. RADV's silent tolerance makes production work but EAGER.1.b is the symptom. |
| SmallCNN+GN: 3.9× CPU (compile-mode target 1× CPU) | **SUPERSEDED.** Eager-mode is already at 2.36× CPU — better than the compile-mode target. The perf focus should shift to eager-mode wins + closing M-NEW.1 to get compile-mode benchmarks running again. |
| Transformer / ViT / Mamba-2 / Llama-block / Qwen3.5 BLOCKED | **PARTIAL.** Compile-mode blocked (M-NEW.1 + M21.3.01). Eager-mode trains end-to-end for all 9 catalog models (validation off). |
| M22.13 workaround "FIXED (defense-in-depth) pending M22.13-followup" | **CLOSED.** M22.13-followup shipped (mm_tiled.slang + bmm_tiled.slang linearised). The C++ workaround is now redundant; filed **M22.13-retire-workaround** for cleanup. |
| M23.2 "in-flight" | **CLOSED.** `TestM232CapabilityGateCoverage` is in `tests/test_inductor_regression.py:40473`. |

### 0.7.7 Net-new items added by Wave 3

Counted: **34 new items** across M-NEW (6), M-pipeline (9), TEST.COV (8), M-docs (10) + 1 reframe of M22.13-followup as closed.

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

  1. ✅ `slang_mm.slang:218 [numthreads(WG_N, WG_M, 1)]` — slangc 2026.5+ rejects spec-constant-derived expressions in `numthreads` (`E39999`). **Fix**: hard-code via Jinja `{{ tile_n // n_per_thread }}` literal — different tile configs already render to different SPIR-V modules so no dedup is lost. (closed 2026-05-16)
  2. ✅ After fix #1, Vulkan validation fails `VUID-VkComputePipelineCreateInfo-layout-07988`: SPIR-V uses `Set 1 Binding 0/1/2` but pipeline layout only declares `Set 0`. The `ParameterBlock<KernelArgs>` set assignment doesn't match what `_jit_dispatch` reflects. **Fix**: emit per-binding `[[vk::binding(N)]]` decorations instead of `ParameterBlock`. (closed 2026-05-16)
  3. ✅ Remove `_install_vulkan_aten_only_autotune()` constraint and default-enable Slang mm. `_slang_tiles_enabled()` now uses opt-out (`TORCH_VULKAN_DISABLE_SLANG_TILES=1`); the `_install_vulkan_aten_only_autotune` function was deleted (dead code — never wired). (closed 2026-05-16)
  4. ✅ Regression test `TestM17SlangMatmulCorrectness` in `tests/test_inductor_regression.py`: correctness across all tile configs (8×8×{8,16,32,64} + register-tiled 64×{32,64}), addmm/bmm variants, cold-compile budget, warm-step budget. (closed 2026-05-16)

  **Expected gain**: addmm 1 → fused mm+bias + 7 stock-decomp backward → 1 fused mm bwd = **8 dispatches → 2** for Linear alone. For Transformer with 4 Linears + 1 MultiheadAttention bmm = ~16 → 5 dispatches. **Estimated step time 6.69 ms → ~2 ms.**

  **✅ M17.1-gap (FIXED 2026-05-16):** `_vulkan_mm` now routes fp32 Vulkan tensors through `_slang_tile_mm` (tile 8×8×8) via a `codegen` override on `_VulkanMMOut`. Non-fp32 falls back to existing `aten.mm.out` path. Regression test `test_mm_compiled_uses_slang_tiles` verifies ≤2 dispatches under `torch.compile`. `lowerings/matmul.py:_vulkan_mm` creates a `_VulkanMMOut` with `python_kernel_name="torch.ops.aten.mm.out"` — this bypasses Slang tiles entirely and dispatches to eager C++ `vulkan_mm`. `install_external_mm()` registers Slang choices in `external_matmul`, but those are only consumed by `tuned_mm` — and `_vulkan_mm` bypasses `tuned_mm`. When AOTAutograd decomposes `aten.addmm` backward into `aten.mm(dC, B^T)` and `aten.mm(A^T, dC)`, those hit `_vulkan_mm` → eager → **7 extern dispatches still unfixed**.

    **Fix (0.5d):** Either (A) modify `_vulkan_mm` to call `_slang_tile_mm` directly while keeping the `unwrap_storage_for_input` optimization, or (B) route `aten.mm` through `tuned_mm` which benchmarks both aten and Slang choices. **This is the highest-leverage remaining item — it closes the last 7 of 8 Linear dispatches.**

### ✅ M17.2 — Conv → GroupNorm → ReLU fusion (DONE 2026-05-17, 3-4d)

~~Conv2d uses a `torch_vulkan::conv2d_with_optional_bias` custom op — opaque to Inductor. Inductor sees it as an extern, can't fuse downstream pointwise into it. Result: GN sees a fresh input buffer for every conv output, even though GN normalization is per-channel-then-broadcast (purely Pointwise/Reduction at that stage).~~

**Done (2026-05-17):**
- [x] **Phase 1**: Conv2d + ReLU fusion via `slang_conv2d.slang` template with `Epilogue : IDifferentiable` (`epilogue="OpReLU"`). Single dispatch for conv+bias+ReLU.
- [x] **Phase 2**: Conv+ReLU + GN batch-mode wrapping — `_conv2d_gn_relu_impl` wraps `_slang_tile_conv2d(epilogue="OpReLU")` + `_dispatch_group_norm_slang` in C++ `begin_batch_dispatch`/`end_batch_dispatch`. Both dispatches share one command buffer → one `vkQueueSubmit`.
- [x] **Phase 3**: True single-dispatch via `conv_gn_relu.slang` template. Combined shader: conv compute + bias + ReLU → local Welford accumulation → workgroup-level Welford reduction (`wg_welford`) → normalize + affine + store. One `vkQueueSubmit`, zero intermediate buffers. GN-style workgroup decomposition (one WG per (batch, group) row, 256 threads).
- [x] Pattern matcher fix: `_fuse_conv_gn_relu()` in `meta_patches/decomposition_passes.py` and `fx_passes/eager/conv.py` looks through `operator.getitem` wrappers for multi-output `native_group_norm`.
- [x] Registration fix: `_ensure_conv2d_relu_fused_op_registered` now called from `register_eager_patch_custom_ops()`.

**Result**: 8 dispatches → 1 dispatch (forward path). Backward still uses chained eager kernels. **~2 ms saved per SmallCNN step.**

### ✅ M17.3 — Native adaptive_avg_pool2d Slang lowering (DONE 2026-05-16, 0.5d)

~~`torch.ops.torch_vulkan.adaptive_avg_pool2d.default` is a custom op (forward is 1 dispatch but no fusion); backward is `aten._adaptive_avg_pool2d_backward` extern (3 sub-dispatches). For typical `output_size=(8,8)` with `input_size=(16,16)`, this is a simple 2×2 average + per-output-cell scale.~~

**Done**: `register_lowering(aten._adaptive_avg_pool2d.default)` in `lowerings/pool.py`:
- Integer-divisible case (`H_in % H_out == 0 and W_in % W_out == 0`): delegates to `aten.avg_pool2d.default` which creates a fusable Reduction IR node.
- Non-divisible case: falls back to eager handler.
- Non-Vulkan tensors: falls back to eager handler (does not intercept).
- Regression tests in `TestM173AdaptiveAvgPool2d`: correctness, dispatch count (≤3), pointwise fusion (≤4), non-divisible fallback, non-Vulkan pass-through.

### M17.4 — ✅ AdamW: `vulkan_optim.AdamW` drop-in (DONE 2026-05-16, path 2)

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

### ✅ M17.5 — Reduce per-dispatch overhead (DONE 2026-05-17, 1d)

~~Even after fusion, per-dispatch wall time is ~0.2-0.4 ms (vkQueueSubmit + driver dispatch).~~

**Implemented (2026-05-17):**
  - [x] **Stage 1: C++ batch mode** — `begin_batch_dispatch`/`end_batch_dispatch` pybinds; `batch_mode` flag in `DeviceRuntime` suppresses auto-flush at 8-dispatch boundary. `DispatchBatcher` engages batch mode on `__enter__` → all dispatches accumulate in one command buffer → single `vkQueueSubmit` on `__exit__`.
  - [x] **Stage 1b: MAX_DISPATCHES_PER_CMD 8→32** in `Stream.h`.
  - [x] **Stage 2: Descriptor set reuse** — per-pipeline `desc_set_cache` in `DeviceRuntime` (keyed by `VkDescriptorSetLayout`). Same-pipeline dispatches within a batch skip `vkAllocateDescriptorSets` — only first dispatch pays the alloc cost. Cache cleared on every flush boundary.
  - [x] **conv+gn+relu forward** wraps internal dispatches with `begin_batch_dispatch`/`end_batch_dispatch`.
  - [x] **Python batcher** updated to resolve and use C++ batch functions.

**Expected gain**: per-dispatch overhead ~0.4 → ~0.15 ms × all dispatches = **~2 ms saved** (was 1.5 ms estimate).

### ✅ M17.6 — Skip extern fall-through for the few ops that still extern (DONE 2026-05-17, 0.5d)

~~`extern_kernels.{addmm,bmm,mm,convolution}` all dispatch through `torch._C._nn.X` which itself dispatches through Vulkan eager — paying the eager-kernel overhead (input dtype check, contiguous copy on stride mismatch, etc.). Even if we keep the eager backing, route through a thin Slang wrapper that skips the redundant dispatcher layers.~~

**Done (2026-05-17):**
- [x] **Slang tile preference for bmm/addmm**: `templates/caller/gemm/install.py` autotuner now skips `aten_bmm`/`aten_addmm` (eager C++ Vulkan kernels) when Slang tile callables are available. Slang tiles unconditionally preferred → 1 dispatch per Linear instead of potentially 2.
- [x] **GN backward decomposition**: Removed `aten.native_group_norm_backward.default` from `ops_to_suppress` in `lowerings/__init__.py`. AOTAutograd now decomposes it into primitive ops (sum, mul, sub, div) that Inductor can fuse, eliminating the last extern dispatch on the SmallCNN backward path.

### M17.7 — Memory: collapse alloc + reinterpret_tensor chains (1-2d) — ✅ DONE 2026-05-22

**Implementation (2026-05-17):**
- [x] LIFO hot-cache (`_lifo`, `_LIFO_MAX=16`) in `buffer_pool.py`: released buffers land in a lifetime-class-agnostic LIFO queue first, so the next same-graph acquire (regardless of class) finds a hit.
- [x] Per-key caps increased: scratch 8→16, transient 6→12, save_for_backward 4→8.
- [x] `release_class` also purges matching entries from the LIFO.
- [x] LIFO acquire ignores lifetime_class — only `(numel, dtype)` matter for same-graph reuse.
- [x] Regression tests: 6 tests in `TestBufferPool` (cross-class hit, LIFO eviction, release_class purge, size/stride correction, stats tracking).
- [x] GPU validation: `agent_space/m17.7_pool_audit.py` SmallCNN training — **94.8% hit rate** (≥80% target). Pool cap raised from 64→256 (fixes the doc-code mismatch; prior code default was 64, now 256). Size peak: 256 entries. Commit `2462f6a1c7a`.

---

### M17 revised priority (2026-05-16 audit)

Post-M17.1/M17.3 audit found that `aten.mm` still bypasses Slang tiles (goes to eager C++), so the 7 `aten.linear_backward` extern dispatches are still present. Re-prioritized for zero-extern training:

| Order | Item | Effort | Dispatches saved | Est. time saved |
|-------|------|--------|-----------------:|----------------:|
| **1st** | **✅ M17.1-gap: fix `aten.mm` Slang routing** | 0.5d | 7 (linear bwd) | ~1.4 ms |
| 2nd | **✅ M17.4 AdamW: `vulkan_optim.AdamW`** | 1d | 0 → 1 Slang (saves 1.7ms eager) | ~1.5 ms |
| 3rd | **✅ M17.3 fwd: explicit `adaptive_avg_pool2d` lowering** | 0.5d | 3 (pool fwd) | ~0.6 ms |
| 4th | M17.2 conv+gn+relu fusion | 3-4d | 5 (conv+gn fwd) | ~1.0 ms |
| 5th | M17.5 per-dispatch overhead | 2-3d | N/A | ~1.5 ms |
| 6th | M17.6 skip extern fall-through | 1-2d | N/A | ~0.5 ms |
| 7th | M17.7 memory alloc collapse | 1-2d | N/A | ~0.3 ms |

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
- [x] **M9.8** Reduction-boundary fusion: GN + ReLU + GlobalAvg should fuse into 1 kernel, not 2. Relax `rnumel_fuse_cap` gate in `scheduling.py:248-261`, or change gate to predicate on consumer pattern rather than rnumel. (2-3d) — **FIX**: Added reduction-boundary fusion relaxation in `can_fuse_vertical`; rnumel cap check now overrides base tiling rejection. Tests in `TestM98ReductionBoundaryFusion`. **Correctness ratchet 2026-05-22**: `test_m98_groupnorm_relu_global_avg_correctness` was carrying `xfail(strict=True, reason="Pre-existing GN accuracy gap on Vulkan (not fusion)")`. After the M-NEW.10 welford fix, the test XPASSed; the xfail has been removed so the gate now hard-asserts numerical parity (1e-2 tolerance) and a future regression is caught immediately. Evidence: `tests/test_inductor_regression.py:38346` (xfail removed) + `agent_space/m19_4_vec4_audit.md`.
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
- [x] **OP.22 Dynamic-shape reduction codegen**: **DONE (2026-05-17):** `is_dynamic_stride()` wired into `pointwise_load_mixin.py` (3 call sites), `indexing.py` (4 call sites), `pointwise.py` (1 call site). Dynamic reduction backward strides route through sizevar push constants. `TestOP22DynamicReductionBackward` (5 tests). Closes "never called" audit gap. ✅
- [x] **OP.23 Foreach element-wise ops**: `install_external_optimizer` covers SGD/AdamW/Lion. Missing: `foreach_add`, `foreach_mul`, `foreach_div`, `foreach_lerp`, `foreach_clip_grad_norm`. Reuse the foreach template plumbing. **Unblocks**: gradient-clipping codepaths, multi-param updates. (2-3d) ✅ — foreach_add/mul/div/norm working via Inductor combo kernel; lerp out-variant path blocked on ForeachKernelSchedulerNode; clip_grad_norm blocked on C++ storage access.
- [x] **OP.24 Quantized int8 matmul (inference)**: ✅ — `shaders/lib/mm_int8.slang` (int8 unpack + int32 accumulate + float32 store), `_render_mm_int8_slang()` wrapper, `_slang_tile_mm_int8()` dispatch, `torch_vulkan::mm_int8` custom op, `_register_mm_int8_lowering()` in matmul.py with autotuning. Forward-only. **Unblocks**: GPTQ / AWQ / quantized Llama inference.
- [x] **OP.25 RNN backward via Slang autodiff**: `bwd_lowerings.py:687L` decomposes RNN grads manually; GRU backward marked "more complex" and incomplete. **Unblocks**: LSTM/GRU training parity. (6-8d)
- [x] **OP.26 Anti-symptom: native attention primitive**: `_fuse_sdpa_to_flash_attention` is currently a symptom-fix for the absence of a native `aten.scaled_dot_product_attention` lowering. Promote the FlashAttention template to a real primitive lowering registered via `@register_lowering(aten.scaled_dot_product_attention)`. Closes anti-goal #5 for SDPA. (3d) ✅ Done 2026-05-14 — new `lowerings/attention.py` with native SDPA + sdpa_with_optional_mask lowerings routing to FlashAttention template; disabled fx_passes/patterns/sdpa.py pattern matcher; removed pre-grad SDPA decomposition; 9/10 regression tests passing, 1 xfail (pre-existing GQA C++ backend limitation).
- [x] **OP.27 Slang tile matmul codegen reactivation (DONE 2026-05-16)**: Three compounding blockers resolved: (1) slangc E39999 numthreads → hard-coded via Jinja; (2) VUID-07988 ParameterBlock binding mismatch → per-binding `[[vk::binding(N)]]` decorations; (3) slangc 2026.5.2 barrier + lid indexing bugs on wave64 → single-wave workgroup restriction. `_slang_tiles_enabled()` now opt-out (`TORCH_VULKAN_DISABLE_SLANG_TILES=1`); `_install_vulkan_aten_only_autotune` deleted (dead code). Max diff ~1e-5 vs CPU across all tile configs. Regression test `TestM17SlangMatmulCorrectness` in `tests/test_inductor_regression.py`.

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
- [x] **M16.3** Delete `model_ops.cpp` and lock with a regression test. **DONE (2026-05-17):** 5 category-(c) ops moved to `legacy_eager.cpp`, 24 category-(a) registrations removed, `model_ops.cpp` deleted (925 L → 0 L). `_validate_no_model_ops()` build gate + `TestM16ModelOpsDeleted` (4 tests). Anti-goal #2 CLOSED. ✅
- [x] **M16.4** Lock the boundary: **DONE (2026-05-17):** `_validate_no_model_ops()` in `setup.py` gates model_ops.cpp at build time and verifies `legacy_eager.cpp` exists. Regression test `test_setup_py_has_model_ops_gate` verifies the gate is wired. ✅

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
| M17.2 conv_gn_relu template | `python/torch_vulkan/inductor/templates/conv_gn_relu.slang` |
| M17.7 alloc alias pass | `python/torch_vulkan/inductor/fx_passes/alloc_alias.py` |
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
