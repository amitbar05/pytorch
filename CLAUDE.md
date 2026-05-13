# CLAUDE.md — PyTorch fork (root)

This fork's primary work surface is the out-of-tree
**`backends/vulkan_slang/`** Inductor backend (Vulkan compute + Slang shaders
→ SPIR-V, registered as `torch.device("vulkan")` via PrivateUse1). Upstream
`torch/_inductor/` work happens occasionally — when it does, the rules in
this file apply. For backend work, the canonical source is
**`backends/vulkan_slang/CLAUDE.md`**, which has its own build / test /
style / anti-goal rules. Don't conflate the two scopes.

---

# Where to look first

| Doing what? | Read this |
|-------------|-----------|
| Any backend (vulkan_slang) work — almost always | **`backends/vulkan_slang/CLAUDE.md`** and **`backends/vulkan_slang/docs/10-inductor-backend.md`** |
| Triage of a closed milestone | `backends/vulkan_slang/docs/10-inductor-backend-history.md` |
| Upstream `torch/_inductor/` change | This file, § *Upstream Inductor reference* below |
| Open question on which roadmap item to pick | Roadmap § 0 (active milestones) + § 0.5 (audit numbers) |

---

# Roadmap-driven workflow

**`backends/vulkan_slang/docs/10-inductor-backend.md` is the canonical source
of truth for what to work on.** Start every session by reading § 0
(active milestones) and § 0.5 (latest audit). Pick the highest-priority
unblocked item.

Loop:
1. Pick the highest unchecked item in the earliest unblocked milestone.
2. Implement → write a regression test → mark `[x]`. Move to the next item.
3. Work autonomously; don't pause to ask "should I continue".
4. Blocked? Skip, note the blocker in the roadmap, take the next item.
5. Found a gap too big for the current change? Add it as a new roadmap
   item (clear title, what's missing, rough effort) in the right milestone.
6. Don't symptom-patch in `meta_patches.py`. If a fix needs a new
   primitive, add a roadmap item for the primitive instead.

## v6.2 active milestones (snapshot 2026-05-13)

| # | Milestone | What it closes |
|---|-----------|----------------|
| **M9** | Host-overhead reduction | 96 % CPU-overhead gap; M9.1 / M9.3 closed, **M9.2 cmd-buf batching** is next |
| **M11** | Occupancy-aware codegen | Wire reflection (VGPR/LDS) → WG sizing (currently 0 % used) |
| **M12** | Reduction backward via autodiff | Closes anti-goal #3 for reductions |
| **M13** | Slang feature saturation | Lift conv / SDPA / reduction / pointwise to the mm gold standard |
| **M14** | Op coverage gaps | Complex binary, sparse, dynamic-shape reduction, foreach, int8, RNN bwd |
| **M15** | Anti-goal #5 / #7 cleanup | 10 file-size violators; `meta_patches.py` symptom-fix audit |
| **M16** | Track 4 finish | Delete `csrc/ops/model_ops.cpp` (925 L). Irreversible. |
| **M6** | Conv generality | Phase 1 closed; Phase 2-4 (depthwise / 3D / transposed) remain |
| **M7** | Production hardening | Gated on slangc / build infra |
| **M8** | Model zoo expansion | Ongoing — 9 architectures already train end-to-end |

Refer to the roadmap doc for the per-item checklist; don't re-derive
priority here.

---

# Scratch space

- `./agent_space/` (root) — throwaway scripts for upstream-Inductor work.
- `backends/vulkan_slang/agent_space/` — throwaway scripts for backend work.

Both are git-ignored. Don't commit from either.

---

# Build

Two distinct builds — pick by what you're touching:

**Upstream PyTorch** (rare — you're editing `torch/_inductor/`):
```bash
MAX_JOBS=6 pip install -e . -v --no-build-isolation
```
Ask the user for env vars (`USE_CUDA=…`, `BUILD_TEST=…`, etc.) before running.

**Vulkan/Slang backend** (most sessions — you're editing `backends/vulkan_slang/`):
```bash
cd backends/vulkan_slang
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MAX_JOBS=8 python setup.py build_ext --inplace
```
See `backends/vulkan_slang/CLAUDE.md` for incremental / clean-build details.

**Never** mix the two. The upstream build rebuilds libtorch; the backend
build rebuilds `_C_ext` against the already-installed libtorch in
`backends/vulkan_slang/.venv/`.

---

# Linting

`spin lint` to lint, `spin fixlint` for autofix. `spin help` lists more.
Only use lint commands provided by `spin`.

---

# Commit messages

Don't commit unless explicitly asked. Keep messages concise: describe
**why**, not a bullet list of what. Disclose Claude authorship via
`Co-Authored-By: Claude <model> <noreply@anthropic.com>` trailer.
Preserve `ghstack-source-id` and `Pull-Request` trailers when amending.

---

# Upstream Inductor reference

**Use only when editing `torch/_inductor/*` directly.** For backend work,
`backends/vulkan_slang/docs/inductor-pipeline-analysis.md` is the
project-specific equivalent.

## Compilation pipeline (upstream)
`torch.compile` → Dynamo FX capture → pre-grad passes
(`torch/_inductor/fx_passes/pre_grad.py`) → AOTAutograd (joint fwd+bwd) →
post-grad passes (`fx_passes/post_grad.py`) → `compile_fx.py:compile_fx_inner`
→ `graph.py:GraphLowering` → `scheduler.py:Scheduler` → device codegen
(`codegen/triton.py`, `codegen/cpp.py`, …) → wrapper codegen
(`codegen/wrapper.py` Python; `codegen/cpp_wrapper_*.py` C++ AOTI) →
compilation cache → returns `CompiledFxGraph` or `CompiledAOTI`.

## Key files (upstream)
| Change | Files |
|---|---|
| Lowering for a new op | `lowering.py`, `ir.py` |
| Kernel template | `codegen/triton.py`, `kernel/{mm,conv}.py` |
| Fusion pass | `fx_passes/post_grad.py`, `pattern_matcher.py` |
| Fusion / tiling heuristics | `scheduler.py`, `choices.py` |
| Wrapper output | `codegen/wrapper.py`, `codegen/cpp_wrapper_*.py` |
| Config option | `config.py` (top-level entries; nested via `Config.find(...)`) |
| Memory planning | `memory.py`, `codegen/memory_planning.py` |
| Autotuning | `select_algorithm.py`, `autotune_process.py`, `runtime/triton_heuristics.py` |
| Dynamic shapes | `compile_fx.py`, `scheduler.py`, `ir.py`, `codegen/triton.py` |
| AOTInductor | `compile_fx.py` (AOTI path), `codegen/cpp_wrapper_*.py`, `codegen/aoti_runtime/interface.cpp` |
| Performance regression | `scheduler.py`, `choices.py`, `config.py`; use `TORCHINDUCTOR_PROFILE=1` |
| Correctness regression | `compile_fx.py` repro → `test/inductor/test_torchinductor.py` |

---

# Coding conventions

## Style — `torch/_inductor/` (upstream files)
- `from __future__ import annotations` first in every file.
- Imports: stdlib → third-party → `torch.*` → relative (`.sibling`).
  `TYPE_CHECKING` block at bottom of import section.
- PascalCase classes, snake_case functions, `_` prefix for private.
- No trivial 1-2 line helpers used once — inline them.
- No dynamic `setattr` / `getattr` — explicit class members.
- Match existing patterns: `V.ops.foo()` define-by-run,
  `@ir_dataclass(frozen=True)` IR nodes, `@cache_on_self` method caching,
  virtualised globals (`V.graph`, `V.ops`, `V.kernel`) over explicit args.
- **File sizes are intentionally large in `torch/_inductor/`. Do not split them.**
  (Conflicts with the backend rule below — scope matters.)

## Style — `backends/vulkan_slang/` (backend files)
- All of the above, **plus** anti-goal #7: **no file in
  `python/torch_vulkan/inductor/` exceeds 800 lines.** When a file approaches
  the cap, split — that's the explicit M15 work.
- See `backends/vulkan_slang/CLAUDE.md` for the full anti-goal / discipline list.

## Debug logging (upstream)
```python
# tlparse-visible structured artifact
from torch._logging import trace_structured
trace_structured("artifact", metadata_fn=lambda: {"name": "x", "encoding": "string"},
                 payload_fn=lambda: my_string)

# Topic-specific logger
log = torch._logging.getArtifactLogger(__name__, "fusion")
log.debug("...")
```
Common artifacts: `fusion`, `schedule`, `output_code`, `autotuning`,
`kernel_code`, `ir_pre_fusion`, `ir_post_fusion`, `loop_ordering`,
`cudagraphs`, `benchmarking`. Enable with `TORCH_COMPILE_DEBUG=1`.

---

# Testing

## Upstream `test/inductor/`
```bash
pytest -n 6 test/inductor/test_torchinductor.py -k test_feature
pytest -n 6 test/inductor/test_torchinductor.py GPUTests.test_feature
```
Test base: `from torch._inductor.test_case import run_tests, TestCase`.
Use `@torch._inductor.config.patch({...})` for temporary config overrides
— never manual save/restore.

Inspecting generated code:
```python
from torch._inductor import utils as inductor_utils
result, code = inductor_utils.run_and_get_code(fn, *args)
result, triton_code = inductor_utils.run_and_get_triton_code(fn, *args)
```

## Backend (vulkan_slang) tests
See `backends/vulkan_slang/CLAUDE.md` — different cwd, requires
`-p no:faulthandler` (Vulkan cleanup segfaults on exit), uses
`-n 4 --timeout=120`. Don't run the upstream pytest command against
the backend tree — caches and conftest don't match.

---

# Common workflows (upstream `torch/_inductor/` only)

| Task | Steps |
|---|---|
| Add an op lowering | `lowering.py` (follow existing pattern) → convert inputs to `TensorBox` via `ops.foo()` / `ir.Pointwise.create()` → fall through to `FallbackKernel` for unsupported → add OpInfo or explicit test in `test/inductor/test_torchinductor.py` |
| Add / modify a fusion pass | Add at the right position in `fx_passes/post_grad.py:post_grad_passes()`; pattern-match-and-replace via `PatternMatcherPass`; structural fusion in `scheduler.py:fuse()` |
| Fix a correctness regression | `run_and_get_code()` minimal repro → bisect with `config.patch(pattern_matcher=False)` etc. → regression test |
| Fix a perf regression | `scheduler.py` + `choices.py`; profile with `TORCHINDUCTOR_PROFILE=1` |
| Add a config option | `config.py` `Config` class with `alias="ns.name"` for nested keys |

For backend work, the equivalent workflows live in
`backends/vulkan_slang/CLAUDE.md` § Build / Testing / Parallel agent
dispatch — they use different files and different conventions.
