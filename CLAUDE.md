# Scratch Space

Use `agent_space/` (git-ignored, at repo root) for temporary scripts and throwaway experiments. Do not commit files from this directory.

# Inductor Roadmap

**`backends/vulkan_slang/docs/10-inductor-backend.md` is the canonical source of truth for what to work on.** All Inductor work is organized and prioritized through the roadmap. The coding agent's primary directive is to drive roadmap items to completion.

## Agent Workflow

1. **Start every session by reading the roadmap.** `backends/vulkan_slang/docs/10-inductor-backend.md`. Pick the highest unchecked item in the earliest unblocked track. If the user gave a specific task, map it to the roadmap or add it.
2. **Drive items to completion.** Focus on one item at a time. Implement, test, and land it before moving to the next. When you finish writing code, write tests, then move to the next item.
3. **Keep moving.** After completing one task, immediately move to the next unchecked item. Do not pause to ask if you should continue — work autonomously until manually stopped.
4. **If blocked, skip and note why.** Skip to the next unblocked item and document the blocker in the roadmap doc. Come back when the dependency resolves.
5. **When you encounter a component that needs improvement but is too large to fix in the current change, add it to the roadmap.** This includes: missing op coverages, performance bottlenecks in codegen, heuristic improvements needed, missing tests, refactoring opportunities, or patterns that need a new fusion pass. Give each item a clear title, description, approach, and place it in the right priority section.
6. **Keep the roadmap updated as you work.** When an item is complete, mark it `[x]`. When status changes, update it. When priorities shift, reorder. Delete stale items. Keep checkboxes and status tables current.
7. **Each roadmap item should be a meaningful unit of work** — not a micro-task and not an entire quarter's project. Aim for items that take roughly a day to a week of focused work.

## Roadmap Structure

The roadmap is organized into numbered tracks — thematic work streams with entry conditions and verification gates. Each track contains checkboxes. The roadmap also includes a critical path showing dependencies between tracks, a status table at the top, and anti-goals / discipline rules at the bottom.

Add items anywhere in the right track or section. Move items between tracks as priorities change. When blocked, note the dependency. Use `[x]` for completed items.

# Build

Always ask for build environment variables before running build.
`MAX_JOBS=6 pip install -e . -v --no-build-isolation` is the only build command. Never run any other build approach.

# Linting

Use `spin lint` to lint and `spin fixlint` for automatic fixes. `spin help` lists other commands. Only use lint commands provided by `spin`.

# Commit messages

Don't commit unless explicitly asked. Keep messages concise: describe why, not a bullet list of what. Disclose Claude authorship. Preserve `ghstack-source-id` and `Pull-Request` trailers.

# Inductor Compilation Pipeline

The full path from `torch.compile` to generated code:

1. **FX graph capture** (Dynamo) → produces an `fx.GraphModule`
2. **Pre-grad passes** — `torch/_inductor/fx_passes/pre_grad.py`
3. **AOTAutograd** — traces joint forward+backward, functionalizes
4. **Post-grad passes** — `torch/_inductor/fx_passes/post_grad.py`: pattern matcher, fusions (group batch, split-cat, pad_mm), re-inplacing
5. **`compile_fx.py:compile_fx_inner`** — checks `FxGraphCache`, delegates to `GraphLowering`
6. **`graph.py:GraphLowering`** — converts FX graph → Inductor IR (`ir.Operation`, `ir.Buffer`); dead code elimination
7. **`scheduler.py:Scheduler`** — builds a `BaseSchedulerNode` DAG; fuses producer-consumer nodes; partitions into subgraphs; assigns streams
8. **Device-specific codegen** — `codegen/triton.py` (GPU), `codegen/cpp.py` (CPU), `codegen/simd.py` (CPU SIMD), `codegen/halide.py`, `codegen/mps.py`, etc.
9. **Wrapper codegen** — `codegen/wrapper.py` (Python), `codegen/cpp_wrapper_cpu.py` / `codegen/cpp_wrapper_gpu.py` (C++ wrapper), `codegen/wrapper_fxir.py` (FX IR wrapper)
10. **Compilation** — Python wrapper → `PyCodeCache`; C++ wrapper → `CppWrapperCodeCache` / `cpp_builder.py`
11. **Return** `OutputCode` subclass (`CompiledFxGraph` or `CompiledAOTI`)

## Key files for common changes

| Change | Primary files |
|---|---|
| Add lowering for a new op | `lowering.py`, `ir.py` |
| Add a new kernel template | `codegen/triton.py`, `kernel/mm.py` or `kernel/conv.py` |
| Add a fusion pass | `fx_passes/post_grad.py`, `fx_passes/pattern_matcher.py` |
| Change fusion heuristics | `scheduler.py` (fusion), `choices.py` (tiling) |
| Change codegen output | `codegen/wrapper.py`, `codegen/triton.py`, `codegen/cpp.py` |
| Add a config option | `config.py` |
| Fix a correctness bug | `compile_fx.py` → reproduction; `test/inductor/test_torchinductor.py` → regression test |
| Fix a performance regression | `scheduler.py`, `choices.py`, `config.py`; use `TORCHINDUCTOR_PROFILE=1` |
| AOTInductor | `compile_fx.py` (AOTI path), `codegen/cpp_wrapper_*.py`, `codegen/aoti_runtime/interface.cpp` |
| Autotuning | `select_algorithm.py`, `autotune_process.py`, `runtime/triton_heuristics.py` |
| Pattern matching | `pattern_matcher.py`, `fx_passes/post_grad.py` |
| Memory planning | `memory.py`, `codegen/memory_planning.py` |
| CUDA graphs | `cudagraph_trees.py`, `config.triton.cudagraph_trees` |
| Dynamic shapes | `compile_fx.py`, `scheduler.py`, `ir.py`, `codegen/triton.py` |
| Distributed/comms | `comms.py`, `fx_passes/bucketing.py`, `fx_passes/ddp_fusion.py` |

# Coding Conventions

## Style

- `from __future__ import annotations` first in every inductor file
- Imports: stdlib → third-party → `torch.*` → relative imports (`.sibling`)
- `TYPE_CHECKING` block for type-only imports at bottom of import section
- Classes: PascalCase. Functions/methods: snake_case. Private: `_` prefix
- No trivial 1-2 line helpers used once. Prefer inline.
- No dynamic `setattr`/`getattr` on objects. Explicit class members only.
- Match existing patterns: `V.ops.foo()` for define-by-run, `@ir_dataclass(frozen=True)` for IR nodes, `@cache_on_self` for method caching. Dynamic dispatch via virtualized globals (`V.graph`, `V.ops`, `V.kernel`, etc) rather than explicit arguments.
- File sizes are large intentionally. Do not try to split large files.

## Debug logging

For diagnostics visible in production (tlparse), use:
```python
from torch._logging import trace_structured
trace_structured("artifact", metadata_fn=lambda: {"name": "my_artifact", "encoding": "string"}, payload_fn=lambda: my_string)
```

For topic-specific debug logging:
```python
log = torch._logging.getArtifactLogger(__name__, "fusion")
log.debug("...")
```

Common artifact loggers: `"fusion"`, `"schedule"`, `"output_code"`, `"autotuning"`, `"kernel_code"`, `"ir_pre_fusion"`, `"ir_post_fusion"`, `"loop_ordering"`, `"cudagraphs"`, `"benchmarking"`.

Enable debug artifacts with `TORCH_COMPILE_DEBUG=1` or `config.trace.enabled = True`.

# Testing

## Test infrastructure

Use the Inductor test case base:
```python
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_GPU, HAS_TRITON, GPU_TYPE
from torch.testing._internal.common_utils import parametrize, instantiate_parametrized_tests

class MyTests(TestCase):
    @config.patch(max_fusion_size=128)
    def test_feature(self):
        ...

if __name__ == "__main__":
    run_tests()
```

Tests live in `test/inductor/`. Main test files:
- `test_torchinductor.py` — largest test suite; uses `CommonTemplate` + `copy_tests()` to generate CPU/GPU variants
- `test_torchinductor_opinfo.py` — OpInfo-based tests
- `test_torchinductor_dynamic_shapes.py` — dynamic shape tests
- `test_aot_inductor.py` — AOTInductor tests
- `test_pattern_matcher.py` — pattern matcher tests
- `test_inductor_scheduler.py` — scheduler-specific tests
- `test_cpu_repro.py` / `test_cuda_repro.py` — repro extraction

## Inspecting generated code

```python
from torch._inductor import utils as inductor_utils

result, code = inductor_utils.run_and_get_code(fn, *args)        # get Python wrapper + kernel source
result, triton_code = inductor_utils.run_and_get_triton_code(fn, *args)  # Triton kernel source only
```

Use `FileCheck` from `torch.testing._internal.common_utils` to assert patterns in generated code.

## Common decorators

- `@config.patch(max_fusion_size=128)` — config overrides (preferred; never manual save/restore)
- `@unittest.skipIf(not HAS_GPU, "...")` / `@unittest.skipIf(not HAS_TRITON, "...")`
- `@requires_gpu()` / `@requires_gpu_and_triton()` from `torch.testing._internal.triton_utils`
- `@skipIfRocm` / `@skipIfXpu` / `@skipCUDAIf(not SM80OrLater, "...")`

## Config patching

Use `torch._inductor.config.patch` as a decorator or context manager for temporarily overriding config. Never manually save/restore:
```python
@torch._inductor.config.patch(max_fusion_size=128)
def test_foo(self):
    ...

with torch._inductor.config.patch({"triton.cudagraphs": False}):
    ...
```

## Running tests

Always use `-n 6` to run tests in parallel:
```bash
pytest -n 6 test/inductor/test_torchinductor.py -k test_feature
pytest -n 6 test/inductor/test_torchinductor.py GPUTests.test_feature  # specific GPU test class
```

# Common Task Workflows

## Profile and analyze performance

```bash
TORCHINDUCTOR_PROFILE=1 python script.py          # prints kernel execution times
TORCHINDUCTOR_BENCHMARK_KERNEL=1 python script.py  # benchmarks individual kernels
TORCH_COMPILE_DEBUG=1 python script.py             # dumps IR, graphs, generated code to `torch_compile_debug/`
```

## Add a lowering for a new ATen op

1. Add the lowering function in `lowering.py`; follow existing patterns for similar ops
2. Convert FX node inputs → Inductor `TensorBox` outputs; use `ops.foo()` for sub-operations or `ir.Pointwise.create()` / `ir.Reduction.create()`
3. For ops that can't be natively lowered, use `FallbackKernel` (wraps ATen call)
4. Add OpInfo tests or explicit tests in `test/inductor/test_torchinductor.py`

## Add or modify a fusion pass

1. Passes run in `fx_passes/post_grad.py:post_grad_passes()` — add your pass at the right position
2. For pattern-match-and-replace: define patterns extending `PatternMatcherPass` in `pattern_matcher.py` or `fx_passes/post_grad.py`
3. For structural fusion: modify `scheduler.py` fusion logic (e.g., `fuse()` methods)
4. Most fusion reordering happens in `scheduler.py` — the scheduler builds a DAG, then fuses producers into consumers based on heuristics

## Fix a correctness regression

1. Extract a minimal repro using `torch._inductor.utils.run_and_get_code()` to inspect what's different
2. Bisect config options to isolate the cause (e.g., `config.patch(pattern_matcher=False)`)
3. Pin the regression with `TORCHINDUCTOR_ABI_COMPATIBLE=1` to check ABI issues
4. Add a regression test with the exact failing case and the expected output

## Add a config option

Add to `config.py` using the `Config` class. Config keys use dots for nesting:
```python
# in config.py
triton_cudagraphs = Config(True, alias="triton.cudagraphs")
```
Nested config classes (e.g., `triton = Config.find(...)` namespace) are defined after top-level entries.
Use `config.patch({"max_fusion_size": 128})` in tests — nested names work: `config.patch({"triton.cudagraphs": False})`.
