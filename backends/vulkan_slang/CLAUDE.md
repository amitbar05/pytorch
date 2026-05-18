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

**The roadmap is `docs/10-inductor-backend.md`.** Read § 0 (active milestones)
and § 0.5 (latest audit) at session start. Pick the highest-priority unblocked
item and ship it. After each item: update its checkbox, write a regression
test, move to the next. **Never stop to ask if you should continue** — work
autonomously until manually stopped.

If blocked, skip and note why in the roadmap. Don't symptom-patch; if a fix
needs a new primitive, add a roadmap item for it.

Companion docs:

| Doc | Purpose |
|-----|---------|
| `docs/codegen-optimization-roadmap.md` | Op coverage + Slang feature exploitation tracker |
| `docs/how-to-compile-and-codegen.md` | Pipeline reference (compile APIs, dispatch tables) |
| `docs/inductor-pipeline-analysis.md` | Architecture overview, Slang/Vulkan integration |
| `docs/agent-prompt-implement-plan.md` | Copy-paste prompt for spawning sub-agents |

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
   Current top violators after the M15.1 splits: `runtime/slangc.py`
   (~2100 L — M-cpp-new-2 candidate), `templates/caller/gemm/install.py`,
   `fx_passes/eager/conv.py`, `templates/caller/rnn.py`, `kernel/main.py`,
   `fx_passes/post_grad.py` (700 L post-M22.4, no longer in violation),
   `kernel/header.py`, `wrapper.py`, `validate.py`.
8. No CPU fallbacks.

# Discipline (durable)

1. Every roadmap item names a regression test in `tests/test_inductor_regression.py`.
2. Correctness before performance. Gradient parity with CPU is the exit criterion.
3. Floor-gate-then-ratchet: land `xfail(strict=True)` first, then flip.
4. Items that turn out wrong get removed, not annotated.
5. ✅ Track 4 is **locked**: `csrc/ops/model_ops.cpp` is deleted and
   `setup.py` enforces it. Anti-goal #2 is closed.

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
Drop further to `MAX_JOBS=2` when multiple agents share the GPU box
(M22.16 caution).

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
```

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
| `TORCH_VULKAN_SLANGC_WORKERS=<n>` | slangc thread pool size (`runtime/slangc.py:221`). M22.16 default = 2 to limit concurrent-agent box pressure | 2 |
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
| Kernel codegen | `python/torch_vulkan/inductor/kernel/` | C |
| Lowerings | `python/torch_vulkan/inductor/lowerings/` | B |
| bwd_diff dispatch / table | `python/torch_vulkan/inductor/bwd_diff/` (dir; `unary.py`, `binary.py`, `emit_helpers.py`) + `bwd_diff_table.py` | B |
| Meta-patches (anti-goal #5) | `python/torch_vulkan/inductor/meta_patches/` (dir; `op_registration.py`, `decomposition_passes.py`, `shape_ops.py`, `joint_graph_passes.py`, `dtype_ops.py`, `autograd_registrations.py`, `faketensor_hooks.py`) | H |
| FX passes | `python/torch_vulkan/inductor/fx_passes/` (`post_grad.py`, `eager/`) | H |
| Runtime / slangc | `python/torch_vulkan/inductor/runtime/` (dir; `slangc.py` is the 2100 L M-cpp-new-2 candidate, plus `dispatch.py`, `batcher.py`, `profile.py`, `reflection.py`, `validation_codegen.py`) | F |
| Template kernel / caller | `python/torch_vulkan/inductor/vulkan_template{,_caller}.py` + `templates/caller/gemm/` | D |
| Slang lib modules | `shaders/lib/*.slang` (16 modules: `helpers`, `dtype_pack`, `philox`, `special_math`, `bucket`, `mm`, `mm_tile`, `mm_int8`, `atomics`, `conv`, `norm`, `pointwise`, `pointwise_generic`, `reduction`, `losses`, `tensor_layout`) | G |
| Slang templates | `python/torch_vulkan/inductor/templates/*.{jinja,slang}` | D |
| C++ autograd registration | `csrc/backend/Registration.cpp` | A |
| C++ runtime dispatch | `csrc/ops/dispatch.{cpp,h}` (deferred cmd-buf, descriptor cache, barrier tracking) | A |
| C++ legacy eager ops | `csrc/ops/legacy_eager.cpp` (5 residual ops post-M16; not `model_ops.cpp` — deleted) | A |
| Regression tests | `tests/test_inductor_regression.py` | — |
