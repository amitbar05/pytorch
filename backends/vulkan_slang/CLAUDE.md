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
2. `csrc/ops/model_ops.cpp` is slated for deletion at end of Track 4.
3. No new `aten.<op>_backward` lowerings — backward routes through
   `bwd_diff_table.py` → Slang `bwd_diff()` / `[BackwardDerivative]`.
4. No hand-tuned shader that isn't auto-generated.
5. No symptom-fixes in `meta_patches.py` that paper over a missing primitive.
6. No string-based template parameters — use Slang `interface` generics
   (e.g. `<Op : IPointwise>`), not Jinja `{{ epilogue }}::apply()`.
7. No file in `inductor/` exceeds 800 lines.
8. No CPU fallbacks.

# Discipline (durable)

1. Every roadmap item names a regression test in `tests/test_inductor_regression.py`.
2. Correctness before performance. Gradient parity with CPU is the exit criterion.
3. Floor-gate-then-ratchet: land `xfail(strict=True)` first, then flip.
4. Items that turn out wrong get removed, not annotated.
5. Track 4 is irreversible — once `csrc/ops/model_ops.cpp` is deleted, don't
   bring it back.

---

# Hardware & Environment

- **GPU**: AMD Radeon RX 5600 XT (NAVI10/RDNA1) at `/dev/dri/renderD128`.
  16 CUs, 1024 max WG, **wave64**, 64 KB LDS, 6 GB VRAM. Always test on real GPU.
- **Python venv**: `backends/vulkan_slang/.venv/` (PyTorch 2.11.0+cu130).
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
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MAX_JOBS=8 python setup.py build_ext --inplace

# Full clean build (~5 min)
rm -rf build/temp.linux-x86_64-cpython-312
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MAX_JOBS=8 python setup.py build_ext --inplace

# Recompile Slang shader libs (after editing shaders/lib/*.slang)
python -c "from torch_vulkan.inductor.runtime import precompile_shader_libs; precompile_shader_libs(force=True)"
```

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
| B | Lowerings + bwd_diff: `lowerings/*.py`, `bwd_diff_dispatch.py`, `bwd_diff_table.py` |
| C | Kernel codegen: `kernel/*.py`, `codegen.py`, `expr_printer.py` |
| D | Templates: `vulkan_template{,_caller}.py`, `templates/` |
| E | Scheduler / fusion: `scheduling.py`, `vulkan_combo_kernel.py` |
| F | Runtime: `runtime.py`, `buffer_pool.py`, `wrapper.py`, `lifetime.py` |
| G | Slang lib: `shaders/lib/*.slang` |
| H | FX passes: `fx_passes/` |

Rules: never dispatch two agents into the same group; C++ rebuild is serial
(only the parent does it, only after all C++ edits land); the parent
integrates, rebuilds C++ once, runs the test suite.

---

# Scratch space

`agent_space/` (git-ignored, at `backends/vulkan_slang/agent_space/`) is for
throwaway scripts and probes. Do not commit anything from this directory.

---

# Useful environment knobs

| Var | Effect |
|-----|--------|
| `TORCH_VULKAN_INDUCTOR_STATS=1` | Per-kernel call_count / total_us |
| `TORCH_VULKAN_PROFILE_DISPATCHES=1` | Per-dispatch timing |
| `TORCH_VULKAN_TRACE=1` | Print every JIT dispatch |
| `TORCH_VULKAN_ASYNC_COMPILE={0,1}` | Run slangc in thread pool |
| `TORCH_VULKAN_SPIRV_CACHE=<dir>` | Override SPIR-V cache location |
| `TORCH_VULKAN_MAX_AUTOTUNE={0,1,2}` | WG autotune level |
| `TORCH_VULKAN_MM_TILES="64x64x16,128x64x32"` | Override mm tile choices |
| `TORCH_LOGS=output_code` | Print generated Slang source |
| `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` | Vulkan validation layer |
| `VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT` | + best-practices hints |

---

# Key files (load-bearing — for full inventory see roadmap docs)

| Concern | File | Group |
|---------|------|-------|
| Backend registration | `python/torch_vulkan/inductor/__init__.py` | B |
| Scheduler / fusion | `python/torch_vulkan/inductor/scheduling.py` | E |
| Kernel codegen | `python/torch_vulkan/inductor/kernel/` | C |
| Lowerings | `python/torch_vulkan/inductor/lowerings/` | B |
| bwd_diff dispatch / table | `python/torch_vulkan/inductor/bwd_diff_{dispatch,table}.py` | B |
| FX passes | `python/torch_vulkan/inductor/fx_passes/` | H |
| Runtime / slangc | `python/torch_vulkan/inductor/runtime.py` | F |
| Template kernel / caller | `python/torch_vulkan/inductor/vulkan_template{,_caller}.py` | D |
| Slang lib modules | `shaders/lib/{helpers,dtype_pack,philox,special_math,bucket,mm,mm_tile,atomics,conv,norm,pointwise,reduction,losses,tensor_layout}.slang` | G |
| Slang templates | `python/torch_vulkan/inductor/templates/*.{jinja,slang}` | D |
| C++ autograd registration | `csrc/backend/Registration.cpp` | A |
| Regression tests | `tests/test_inductor_regression.py` | — |
