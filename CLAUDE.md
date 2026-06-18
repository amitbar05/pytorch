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
| Any backend (vulkan_slang) work — almost always | **`backends/vulkan_slang/CLAUDE.md`** and **`backends/vulkan_slang/docs/ROADMAP.md`** |
| Open question on which roadmap item to pick | `ROADMAP.md § 2` (pillars A–F, file:line + regression test per item) |
| Triage of a closed milestone or past audit | `backends/vulkan_slang/docs/10-inductor-backend-history.md`, then `docs/archive/v6.x-snapshot-2026-05-27.md` |
| Upstream `torch/_inductor/` change | This file, § *Upstream Inductor reference* below |

---

# Roadmap-driven workflow

**`backends/vulkan_slang/docs/ROADMAP.md` is the single canonical source
of truth for what to work on.** Start every session by reading it: § 1 is the
current-state scorecard, § 2 the prioritized forward plan (pillars A–F).
It replaced the old `docs/10/14/15/16-inductor-backend.md` series +
`codegen-optimization-roadmap.md` (deleted 2026-06-15). Closed-milestone history
is `docs/10-inductor-backend-history.md`; pre-v7 is `docs/archive/`. Search those
for prior decisions, don't extend them.

Loop:
1. Pick the highest-priority unblocked ROADMAP.md item.
2. Implement → write a regression test → mark ✅ in the ROADMAP.md
   scorecard → move to the next.
3. Work autonomously; don't pause to ask "should I continue".
4. Blocked? Skip, note the blocker in ROADMAP.md, take the next item.
5. Found a gap too big for the current change? Add it as a new sub-item
   under the right ROADMAP.md pillar (clear title, what's missing, rough effort).
6. Don't symptom-patch in `meta_patches/` — that directory exists as
   anti-goal #4; if a fix needs a new primitive, file a roadmap sub-item for
   the primitive instead.

## Durable pillar goals

| # | Pillar | Goal |
|---|--------|------|
| **Codegen-only** | No `extern_kernels.X` to aten / PrivateUse1 eager Vulkan inside compiled wrappers. No "if device != vulkan: aten fallback" branches inside custom-op impls that the compile path can hit. |
| **Slang-smart** | ParameterBlock + generics + interfaces + `[BackwardDerivative]` + spec consts + reflection metadata. String-substituted Jinja is the exception. |
| **Validation-driven** | Vulkan validation layer mandatory in tests (`TORCH_VULKAN_VUID_AS_ERROR=1`); VUID during autotune → rejected candidate; VUID on landed kernel → test failure. |
| **Profile-and-warmup** | `torch_vulkan.prepare_device(level, timeout_s)` once at process start — pay the cold cost up front so `torch.compile` after that is fast. |

The prioritized open-work breakdown (pillars A–F, each item with file:line +
named regression test) lives in `backends/vulkan_slang/docs/ROADMAP.md § 2`.

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
MAX_JOBS=3 pip install -e . -v --no-build-isolation
```
Ask the user for env vars (`USE_CUDA=…`, `BUILD_TEST=…`, etc.) before running.

**Vulkan/Slang backend** (most sessions — you're editing `backends/vulkan_slang/`):
```bash
cd backends/vulkan_slang
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MAX_JOBS=3 python setup.py build_ext --inplace
```
See `backends/vulkan_slang/CLAUDE.md` for incremental / clean-build details.

`MAX_JOBS=3` is the project default (per user memory `feedback_build_config`).
When multiple agents share the box, drop to `MAX_JOBS=2`.

**Never** mix the two. The upstream build rebuilds libtorch; the backend
build rebuilds `_C_ext` against the already-installed libtorch in
`backends/vulkan_slang/.venv/`.

---

# Linting

`spin lint` to lint, `spin fixlint` for autofix. `spin help` lists more.
Only use lint commands provided by `spin`.

---

# Commit policy

**Test and validate before committing.** Once the relevant tests are green
(unit + targeted regression — and a smoke run when the change touches
runtime / GPU code), commit the work as a single logical unit. You do
not need to wait for a separate "please commit" instruction; the green
test run is the authorisation.

Skip commits only when: (a) tests are still failing, (b) the change is
purely exploratory / scratch (`agent_space/`), or (c) the user said "don't
commit yet" for the specific item.

When you do commit:

* Keep messages concise: describe **why**, not a bullet list of what.
* Disclose Claude authorship via
  `Co-Authored-By: Claude <model> <noreply@anthropic.com>` trailer.
* Preserve `ghstack-source-id` and `Pull-Request` trailers when amending.

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
  `python/torch_vulkan/inductor/` exceeds 800 lines.** When a file
  approaches the cap, split.
- See `backends/vulkan_slang/CLAUDE.md` for the full anti-goal /
  discipline list.

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
