# CLAUDE.md — PyTorch Vulkan / Slang Inductor Backend

Out-of-tree PyTorch backend: Vulkan compute + Slang shaders compiled to SPIR-V.
Registered via `PrivateUse1` as `torch.device("vulkan")`. All ops run on GPU.
No CPU fallbacks. `TORCH_CHECK(false, …)` for unimplemented ops.

**Mission.** Ship a fully optimizing, training-grade `torch.compile(backend="inductor")`
backend on Vulkan/Slang that supports **any PyTorch model**. Every kernel is
auto-generated. No per-model `.slang` files. No per-model `csrc/ops/*.cpp`
entries. No CPU fallbacks. No Python at deployment.

---

# Where to work

**The roadmap is [`docs/16-inductor-backend.md`](docs/16-inductor-backend.md).**
Read **§ v16** at session start — that's the active plan (5 pillars, 18 milestones,
dependency graph, file:line ownership, Slang smart-feature audit). v7-v15 are
closed and are reference-only;
pre-v7 history lives in
[`docs/archive/v6.x-snapshot-2026-05-27.md`](docs/archive/v6.x-snapshot-2026-05-27.md);
search it for prior decisions, don't extend it.

Pick the highest-priority unblocked v16 milestone, ship it, lock a
regression test, mark it ✅ in the v16 table, move to the next.
**Never stop to ask if you should continue** — work autonomously until
manually stopped. If blocked, skip and note why in the roadmap. Don't
symptom-patch; if a fix needs a new primitive, file it as its own
roadmap item.

## v7 pillars (snapshot)

1. **M-CG** — codegen-only Inductor backend. No `extern_kernels.X` to
   aten / PrivateUse1 eager Vulkan inside compiled wrappers. No
   "if device != vulkan: aten fallback" branches inside custom-op impls
   that the compile path can hit.
2. **M-SF** — smart Slang feature usage. ParameterBlock + generics +
   interfaces + spec constants + `[BackwardDerivative]` + reflection.
   String-substituted Jinja is the exception.
3. **M-VAL** — validation-driven codegen. Vulkan validation layer
   mandatory in tests (`TORCH_VULKAN_VUID_AS_ERROR=1`); VUID during
   autotune → rejected candidate; VUID on landed kernel → test failure.
4. **M-PROBE** — `torch_vulkan.prepare_device(level, timeout_s)` is the
   canonical entry point. Run it once at process start; `torch.compile`
   after that is fast.

## What's already closed (v7)

* **M-CG.1, M-CG.2** ✅ — Explore-agent audit (2026-05-27): zero
  genuine eager-fallback leaks on the compile path. See § v7 audit
  evidence in the roadmap.
* **M-VAL.1** ✅ — VUID counter pybind + autouse fixture; **default-on**
  after M-VAL.3 closed the residual best-practices backlog (zero VUIDs
  across 9 catalog models). Opt-out: `TORCH_VULKAN_VUID_AS_ERROR=0`.
* **M-VAL.3** ✅ — Best-practices VUID sweep across 9 catalog models
  (2026-05-27). Sweep harness at `agent_space/m21_3_validation_sweep.py`.
  Result: zero VUIDs, all prior P0/P1 VUIDs (M21.3.01, M21.3.02,
  EAGER.1.b, M-cpp-new-6) fully closed.
* **M-SF.1** ✅ — ParameterBlock<KernelArgs> at 100% coverage on all
  actively-used templates (2026-05-27).  The 2 stale `.py.jinja` files
  with manual bindings were deleted; `.slang` is the canonical format.
* **M-CG.4** ✅ — M19.1 linear-backward decomposition (2026-05-27).
  Replaces the 7-8 dispatch C++ eager extern with mm+mm+sum primitives
  routing through Slang kernels.  Unblocked by M22.13 mm transpose-a fix.
* **M-VAL.2** ✅ — Per-kernel autotune VUID gate (2026-05-27).
  `get_codegen_validation_mode()` defaults to `error` when
  `TORCH_VULKAN_VUID_AS_ERROR` is not "0" — autotune candidates
  emitting VUIDs are rejected via `RuntimeError`.
* **M-SF.2** ✅ — [BackwardDerivative] on combine_sum_nan +
  combine_prod_nan in vk_reduction.slang (2026-05-27).  Reduction
  coverage: 2→4 ops; total 45 manual derivatives across hot elementals.
* **M-SF.3** ✅ — Spec constants + ParameterBlock now compatible
  (2026-05-27).  Removed CG.M14 constraint in kernel/header.py; all
  pointwise/reduction kernels can now use `[[vk::constant_id(N)]]`.
* **M-VAL.4** ✅ — Pre-slangc static AST validator (2026-05-27).
  Already integrated at slangc.py:161 (M22.1.i); raises RuntimeError
  on any codegen mistake before subprocess invocation.
* **M-SF.4** ✅ — Jinja has_bias eliminated from conv templates;
  mm render defaults to link-time module path (2026-05-27).
* **M-SF.5** ✅ — num_atomics reflection field wired into WG sizing
  heuristic; 8/8 fields now used for smart Slang features (2026-05-27).
* **M-PROBE.1, M-PROBE.2, M-PROBE.3** ✅ — prepare_device() API,
  auto_probe_on_import off, timeout enforcement (2026-05-27).

🎉 **All 16 v7 milestones closed.** The v7 roadmap is complete.

## Companion docs

| Doc | Purpose |
|-----|---------|
| [`docs/10-inductor-backend.md`](docs/10-inductor-backend.md) | **The roadmap.** § v7 is the active plan. |
| [`docs/archive/v6.x-snapshot-2026-05-27.md`](docs/archive/v6.x-snapshot-2026-05-27.md) | Frozen pre-v7 audit logs, milestone closeouts, reconciliations. |
| [`docs/codegen-optimization-roadmap.md`](docs/codegen-optimization-roadmap.md) | Op coverage + Slang feature exploitation tracker. |
| [`docs/how-to-compile-and-codegen.md`](docs/how-to-compile-and-codegen.md) | Pipeline reference (compile APIs, dispatch tables). |
| [`docs/inductor-pipeline-analysis.md`](docs/inductor-pipeline-analysis.md) | Architecture overview, Slang/Vulkan integration. |
| [`docs/agent-prompt-implement-plan.md`](docs/agent-prompt-implement-plan.md) | Copy-paste prompt for spawning sub-agents. |

---

# Anti-Goals (durable)

1. No new model-specific `.slang` files — templates only.
2. ✅ **CLOSED 2026-05-17**: `csrc/ops/model_ops.cpp` deleted (M16.3); a
   build gate in `setup.py:_validate_no_model_ops()` prevents
   re-introduction. The 5 residual eager ops live in
   `csrc/ops/legacy_eager.cpp`.
3. No new `aten.<op>_backward` lowerings — backward routes through
   `bwd_diff_table.py` → Slang `bwd_diff()` / `[BackwardDerivative]`.
4. No hand-tuned shader that isn't auto-generated.
5. No symptom-fixes in `meta_patches/` (since the M15.1 split this is a
   directory; the offenders are in `op_registration.py`,
   `decomposition_passes.py`, etc.) that paper over a missing primitive.
6. No string-based template parameters — use Slang `interface` generics
   (e.g. `<Op : IPointwise>`), not Jinja `{{ epilogue }}::apply()`.
7. No file in `python/torch_vulkan/inductor/` exceeds 800 lines.
   Current state (2026-05-27): one active violator,
   `kernel/pointwise.py` (820 L). The rest of the historical
   M15.1 / M22.1 violator list has been split — most of the larger
   files now sit in the 700-800 L band (`scheduling.py`, `runtime/
   slangc.py`, `runtime/shader_lib.py`, `lowerings/conv.py`,
   `kernel/header.py`, `buffer_pool.py`, `__init__.py`).
8. No CPU fallbacks.

# Discipline (durable)

1. Every roadmap item names a regression test in `tests/test_inductor_regression.py`.
2. Correctness before performance. Gradient parity with CPU is the exit criterion.
3. Floor-gate-then-ratchet: land `xfail(strict=True)` first, then flip.
4. Items that turn out wrong get removed, not annotated.
5. One commit per milestone. Title format
   `vulkan: M-X.Y — short why`. No laundry-list multi-purpose commits.

---

# Hardware & Environment

- **GPU**: AMD Radeon RX 5600 XT (NAVI10/RDNA1) at `/dev/dri/renderD128`.
  16 CUs, 1024 max WG, **wave64**, 64 KB LDS, 6 GB VRAM. Always test on real GPU.
- **Python venv**: `backends/vulkan_slang/.venv/` (PyTorch 2.11.0+cpu — no CUDA libs; the Vulkan backend wraps PrivateUse1 and routes ops to Vulkan).
  Activate: `source backends/vulkan_slang/.venv/bin/activate`.
- **Slang**: submodule `third_party/slang` (v2026.7.1). `slangc` is auto-resolved
  by `conftest.py` from `third_party/slang/build/slang-*/bin/slangc`.
- **Software-Vulkan diagnostics** only (not for correctness oracle — see
  GPU.1-3 in the roadmap for RDNA1-only bugs):
  - Lavapipe: `export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json`
  - SwiftShader: `export VK_ICD_FILENAMES=/home/amit/swiftshader-build/build/Linux/vk_swiftshader_icd.json`

---

# Build

```bash
cd backends/vulkan_slang

# Incremental rebuild (~3 s, 1 file + relink)
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MAX_JOBS=3 python setup.py build_ext --inplace

# Full clean build (~5 min)
rm -rf build/temp.linux-x86_64-cpython-312
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MAX_JOBS=3 python setup.py build_ext --inplace

# Recompile Slang shader libs (after editing shaders/lib/*.slang)
python -c "from torch_vulkan.inductor.runtime import precompile_shader_libs; precompile_shader_libs(force=True)"
```

`MAX_JOBS=3` is the project default per user memory `feedback_build_config`.
Drop to `MAX_JOBS=2` when multiple agents share the GPU box.

`TORCH_DEVICE_BACKEND_AUTOLOAD=0` keeps a broken in-flight `.so` from blocking
the import path during rebuild.

If you get **missing symbols**, **multiple definition** errors, or
**incremental build skips changed files**, delete
`build/temp.linux-x86_64-cpython-312/` and rebuild clean.

---

# Testing

Always run tests from `backends/vulkan_slang/`. Vulkan cleanup segfaults on
exit, so `-p no:faulthandler` is required.

```bash
# Full suite (~90 s with xdist)
python -m pytest tests/ -n 4 --timeout=120 -p no:faulthandler

# Inductor regression suite
python -m pytest tests/test_inductor_regression.py -n 4 --timeout=300 -p no:faulthandler

# Single test
python -m pytest tests/test_file.py::TestClass::test_name --timeout=30 -p no:faulthandler

# Force-regenerate Inductor cache
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python -m pytest …
rm -rf /tmp/torchinductor_$(whoami)   # clear between runs

# M-VAL.1 (v7): treat any Vulkan VUID emitted during a test as a failure.
# Opt-in until M-VAL.3 closes the residual best-practices-VUID backlog.
TORCH_VULKAN_VUID_AS_ERROR=1 python -m pytest tests/test_file.py::test_name \
    --timeout=30 -p no:faulthandler
```

---

# Recommended user pattern (v7 — M-PROBE.1)

`torch_vulkan.prepare_device(level, timeout_s)` is the v7 canonical
entry point. Run it once at process start before any `torch.compile` to
pay the cold cost up front:

```python
import torch
import torch_vulkan

# Profile the GPU (launch latency, mem BW, LDS BW, atomic throughput) +
# compile the shader-lib + matmul/conv autotune sweep. Cached at
# ~/.cache/torch_vulkan/probe_status_<id>.json; second call short-circuits.
torch_vulkan.prepare_device(level="deep", timeout_s=900)

# … model + optimizer setup …
compiled = torch.compile(model, backend="inductor")
for batch in train_loader:
    train_step(compiled, batch)
```

Levels: `"quick"` (~5 s, microbench only), `"medium"` (~30 s warm /
minutes cold, + shader-lib + matmul SPIR-V), `"deep"` (~3 min warm / up
to 15 min cold, + canonical-shape autotune sweep). Returns a dict with
per-stage timings and the device profile; sets `timed_out=True` if
`timeout_s` is exceeded (background warmup keeps running).

Auto-probe-on-import is gated by `TORCH_VULKAN_PROFILE_DEVICE`
(default `"auto"` → level 0 microbench only on cold import). Set to
`"off"` in CI / scripted environments where you'd rather call
`prepare_device` explicitly.

---

# Parallel agent dispatch

Read `docs/agent-prompt-implement-plan.md` for the full prompt template.

**File group ownership** (disjoint scopes for safe parallel edits):

| Group | Scope |
|-------|-------|
| A | C++ autograd: `csrc/backend/Registration.cpp`, `csrc/autocast/` |
| B | Lowerings + bwd_diff: `lowerings/*.py`, `bwd_diff/` (split via M15.1.h), `bwd_diff_table.py` |
| C | Kernel codegen: `kernel/*.py`, `codegen.py`, `expr_printer.py` |
| D | Templates: `vulkan_template{,_caller}.py`, `templates/` |
| E | Scheduler / fusion: `scheduling.py`, `combo_kernel/` (split via M15.1.f) |
| F | Runtime: `runtime/` (split via M15.1.c: `slangc.py`, `dispatch.py`, `batcher.py`, `profile.py`, `reflection.py`, `validation_codegen.py`), `buffer_pool.py`, `wrapper.py`, `lifetime.py` |
| G | Slang lib: `shaders/lib/*.slang` |
| H | FX passes: `fx_passes/` (includes `fx_passes/eager/` and `fx_passes/post_grad.py`) |

Rules: never dispatch two agents into the same group; C++ rebuild is serial
(only the parent does it, only after all C++ edits land); the parent
integrates, rebuilds C++ once, runs the test suite.

---

# Scratch space

`agent_space/` (git-ignored, at `backends/vulkan_slang/agent_space/`) is for
throwaway scripts and probes. Do not commit anything from this directory.

---

# Useful environment knobs

Source of truth: `python/torch_vulkan/inductor/config.py` +
`runtime/slangc.py` (parallelism / cache) + `csrc/init.cpp` + a few
ad-hoc readers in `csrc/`. If you add a new knob, document it here.

## Profiling / observability

| Var | Effect | Default |
|-----|--------|---------|
| `TORCH_VULKAN_INDUCTOR_STATS=1` | Per-kernel call_count / total_us | off |
| `TORCH_VULKAN_PROFILE_DISPATCHES=1` | Per-dispatch timing | off |
| `TORCH_VULKAN_PROFILE_DISPATCH=1` | C++ side timing breakdown (pipeline cache / desc alloc / buffer-info / barrier-check / cmd-record) | off |
| `TORCH_VULKAN_PROFILE_DEVICE=1` | M21.1 device-profile-on-import phase | off |
| `TORCH_VULKAN_TRACE=1` | Print every JIT dispatch (Python side) | off |
| `TORCH_VULKAN_TRACE_DISPATCH=1` | C++ side `DISPATCH[n] key=... barrier=...` trace | off |
| `TORCH_VULKAN_TRACE_MAXPOOL2D=1` | Trace max_pool2d kernel selection | off |
| `TORCH_VULKAN_POOL_STATS=1` | Buffer-pool hit/miss counts | off |
| `TORCH_VULKAN_DEBUG_UTILS=1` | M21.3.a — opt the debug-utils messenger to INFO-level so BestPractices VUIDs reach stderr | off (WARN+) |
| `TORCH_LOGS=output_code` | Print generated Slang source | off |

## Compilation / caching

| Var | Effect | Default |
|-----|--------|---------|
| `TORCH_VULKAN_ASYNC_COMPILE={0,1}` | Run slangc in thread pool | 1 |
| `TORCH_VULKAN_PARALLEL_COMPILE={0,1}` | Parallel slangc batch compile (`runtime/slangc.py:1966`) | 1 |
| `TORCH_VULKAN_SLANGC_WORKERS=<n>` | slangc thread pool size (`runtime/common.py:_default_max_workers`). Default capped at `min(2, cpu_count)` to bound concurrent-agent box pressure | 2 |
| `TORCH_VULKAN_SLANGC_TIMEOUT_S=<n>` | Per-shader slangc timeout | 60 |
| `TORCH_VULKAN_SPIRV_CACHE=<dir>` | Override SPIR-V cache location | `/tmp/torch_vulkan_spirv_<user>` |
| `TORCH_VULKAN_SLANG_MODULE_CACHE=<dir>` | Override Slang `.slang-module` precompile cache | adjacent to SPIR-V cache |
| `TORCH_VULKAN_NO_PREWARM=1` | Skip prewarm-on-import shader compilation | off (prewarm enabled) |
| `TORCH_VULKAN_MAX_AUTOTUNE={0,1,2}` | WG autotune level | 1 |
| `TORCH_VULKAN_MM_TILES="64x64x16,128x64x32"` | Override mm tile choices | autotune-driven |
| `TORCH_VULKAN_VALIDATE_CODEGEN={off,warn,error}` | M21.2 codegen-correctness gate (validation-as-codegen-check) | warn |
| `TORCH_VULKAN_VALIDATE_SLANG={0,1}` | Lint generated Slang source pre-compile | 0 |
| `TORCH_VULKAN_DUMP_FX=<dir>` | Dump FX graphs from pre-grad / post-grad passes | unset |
| `TORCH_VULKAN_FORCE_CUSTOM_OP_RELOAD=1` | Re-register custom ops on every import (dev only) | off |

## Codegen toggles (off-by-default `_NO_*` and on-by-default features)

| Var | Effect | Default |
|-----|--------|---------|
| `TORCH_VULKAN_DESCRIPTOR_INDEXING={0,1}` | Override `VK_EXT_descriptor_indexing` detection (M-cpp-new-5 / M-cpp-new-6 hinge on this) | auto-detect |
| `TORCH_VULKAN_REFLECTION={0,1}` | Use slangc reflection metadata for WG sizing | 1 |
| `TORCH_VULKAN_REFLECTION_ROUTING={0,1}` | DR.7 Pass-2: VGPR/LDS → numthreads pick | 1 |
| `TORCH_VULKAN_REGISTER_AWARE_WG={0,1}` | Register-pressure aware WG sizing | 1 |
| `TORCH_VULKAN_DYNAMIC_SHAPES={0,1}` | Lift dynamic-shape conv / pointwise | 1 |
| `TORCH_VULKAN_SPEC_CONSTANTS={0,1}` | Use spec constants vs push constants where applicable | 1 |
| `TORCH_VULKAN_STATIC_SPECIALIZATION={0,1}` | Static-specialise tile params at compile time | 1 |
| `TORCH_VULKAN_LINK_TIME_SPEC={0,1}` | Link-time tile specialisation (blocked by slangc E30600) | 0 |
| `TORCH_VULKAN_PARAMETER_BLOCK={0,1}` | Slang `ParameterBlock<T>` codegen (mm only) | 1 |
| `TORCH_VULKAN_PARAMETER_ARRAY={0,1}` | Parameter-array binding form | 1 |
| `TORCH_VULKAN_DCE={0,1}` | Dead-code elimination on emitted Slang | 1 |
| `TORCH_VULKAN_DISABLE_SLANG_TILES=1` | M17.1.3 — opt out of Slang tiles, use plain numthreads | off |
| `TORCH_VULKAN_REGISTER_TILE={2,3,4}` | Register-tile size for mm | autotune |
| `TORCH_VULKAN_NO_REGISTER_TILE=1` | Disable register tiling entirely | off |
| `TORCH_VULKAN_BANK_CONFLICT_PAD={0,1}` | Pad LDS to avoid bank conflicts | 1 |
| `TORCH_VULKAN_BATCH_DISPATCH={0,1}` | M9.2 deferred cmd-buf batching | 1 |
| `TORCH_VULKAN_WRAPPER_FASTPATH={0,1}` | Skip wrapper validation for hot ops | 1 |
| `TORCH_VULKAN_GRID_AWARE_WG={0,1}` | M11.9 grid-aware WG sizing | 1 |
| `TORCH_VULKAN_AGGRESSIVE_FUSION={0,1}` | Pull marginal fuse candidates | 1 |
| `TORCH_VULKAN_PERSISTENT_POINTWISE={0,1}` | Persistent pointwise kernel (small numel) | 1 |
| `TORCH_VULKAN_OCCUPANCY_GATE={0,1}` | Gate WG choices on occupancy estimate | 1 |
| `TORCH_VULKAN_STRICT_OCCUPANCY={0,1}` | Reject WG choices that miss occupancy target | 0 |
| `TORCH_VULKAN_ROUND_WG_TO_WAVE={0,1}` | Round numthreads to wave multiple | 1 |
| `TORCH_VULKAN_VEC4_AUDIT={0,1}` | Audit-only vec4 pass (no codegen change) | 0 |
| `TORCH_VULKAN_NO_VEC4_POINTWISE=1` | Disable vec4 pointwise lowering | off |
| `TORCH_VULKAN_NO_PACKED16=1` | Disable packed-fp16 codegen | off |
| `TORCH_VULKAN_NO_LOAD_HOIST=1` | Disable load hoisting | off |
| `TORCH_VULKAN_NO_WG_TUNE=1` | Disable WG autotuning loop | off |
| `TORCH_VULKAN_NO_EXTERN_EPILOGUE=1` | Disable extern-epilogue inlining | off |
| `TORCH_VULKAN_NO_CACHE_NS=1` | Disable per-namespace SPIR-V cache | off |
| `TORCH_VULKAN_BUFFER_POOL={0,1}` | Per-key buffer pool (M9.1) | 1 |
| `TORCH_VULKAN_BUFFER_POOL_PER_KEY={0,1}` | Per-key keying for the pool | 1 |
| `TORCH_VULKAN_BUFFER_POOL_SIZE=<n>` | Max pool size (entries) | 256 |
| `TORCH_VULKAN_MAX_STORAGE_BUFS=<n>` | Override per-stage max storage buffers cap | platform max |
| `TORCH_VULKAN_MAX_GROUPSHARED_BYTES=<n>` | Override LDS budget per WG | platform max |
| `TORCH_VULKAN_MAX_NUMTHREADS_PRODUCT=<n>` | Override numthreads(x*y*z) cap | platform max |
| `TORCH_VULKAN_MAX_PUSH_CONSTANT_BYTES=<n>` | Override push-constant byte cap | platform max |
| `TORCH_VULKAN_WAVE_SIZE={32,64}` | Force a wave size (debug-only) | auto-detect |
| `TORCH_VULKAN_DENORMALS={flush,preserve}` | f32 denormal mode | preserve |
| `TORCH_VULKAN_INLINE_BWD_DIFF={0,1}` | Inline `bwd_diff()` call sites | 1 |
| `TORCH_VULKAN_RNN_CPU_FALLBACK={0,1}` | RNN backward CPU fallback (anti-goal #8 violation; clear before training) | 0 |
| `TORCH_VULKAN_TRUST_INDUCTOR={0,1}` | Skip extra validation when Inductor reports OK | 1 |
| `TORCH_VULKAN_DEBUG_FOREACH=1` | Foreach lowering debug print | off |
| `TORCH_VULKAN_COMBO_DEBUG_ASSERT=1` | Assert on combo-kernel renaming hazards | off |

## Validation layer

| Var | Effect |
|-----|--------|
| `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` | Vulkan validation layer |
| `VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT` | + best-practices hints (the M186 / M21.3 test scaffold uses this) |
| `VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_BIT` | GPU-assisted validation (catches `VUID-VkWriteDescriptorSet-dstSet-04611`; ~3× slower) |
| `TORCH_VULKAN_VALIDATION=<layers>` | Alias / convenience wrapper around `VK_INSTANCE_LAYERS` |
| `VK_ICD_FILENAMES=…` | Override the ICD (force Lavapipe / SwiftShader instead of RADV; pin under validation per M21.4.c) |

---

# Key files (load-bearing — for full inventory see roadmap docs)

| Concern | File / Directory | Group |
|---------|------------------|-------|
| Backend registration | `python/torch_vulkan/inductor/__init__.py` | B |
| Scheduler / fusion | `python/torch_vulkan/inductor/scheduling.py` + `combo_kernel/` | E |
| Kernel codegen | `python/torch_vulkan/inductor/kernel/` (`main.py`, `header.py`, `pointwise.py`, `reduction.py`, `threadgroup_sizing.py`, `dispatch_call.py`, …) | C |
| Lowerings | `python/torch_vulkan/inductor/lowerings/` | B |
| bwd_diff dispatch / table | `python/torch_vulkan/inductor/bwd_diff/` (`unary.py`, `binary.py`, `emit_helpers.py`) + `bwd_diff_table.py` | B |
| Meta-patches (anti-goal #5) | `python/torch_vulkan/inductor/meta_patches/` (`op_registration.py`, `decomposition_passes.py`, `shape_ops.py`, `joint_graph_passes.py`, `dtype_ops.py`, `autograd_registrations.py`, `faketensor_hooks.py`) | H |
| FX passes | `python/torch_vulkan/inductor/fx_passes/` (`post_grad.py`, `eager/`) | H |
| Runtime / slangc | `python/torch_vulkan/inductor/runtime/` (`slangc.py`, `common.py`, `shader_lib.py`, `reflection_ext.py`, `dispatch.py`, `batcher.py`, `profile.py`, `reflection.py`, `validation_codegen.py`) | F |
| Hardware probe | `python/torch_vulkan/inductor/hardware_probe.py` (M-PROBE.1: `prepare_device(level, timeout_s)` entry point) | F |
| Template kernel / caller | `python/torch_vulkan/inductor/vulkan_template{,_caller}.py` + `templates/caller/gemm/` | D |
| Slang lib modules | `shaders/lib/*.slang` (18 modules) | G |
| Slang templates | `python/torch_vulkan/inductor/templates/*.{jinja,slang}` | D |
| C++ autograd registration | `csrc/backend/Registration.cpp` | A |
| C++ runtime dispatch | `csrc/ops/dispatch.{cpp,h}` (deferred cmd-buf, descriptor cache, barrier tracking) | A |
| C++ Vulkan context | `csrc/vulkan/Context.{cpp,h}` (instance + debug messenger + M-VAL.1 VUID counter) | A |
| C++ pybinds | `csrc/init.cpp` (`_validation_errors_count`, `_storage_device_index`, etc.) | A |
| C++ legacy eager ops | `csrc/ops/legacy_eager.cpp` (5 residual ops post-M16; `model_ops.cpp` is deleted) | A |
| pytest conftest (incl. M-VAL.1 fixture) | `conftest.py` | — |
| Regression tests | `tests/test_inductor_regression.py` | — |
