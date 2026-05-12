# How to Compile & Codegen Any PyTorch Model

> **Last updated: 2026-05-07**

A comprehensive guide to PyTorch model compilation through the Inductor
pipeline — covering all entry points, the complete lowering→codegen flow,
backend dispatch, and how each device backend generates kernel source code.

---

## Table of Contents

1. [Compilation Entry Points](#1-compilation-entry-points)
2. [The Complete Lowering Pipeline](#2-the-complete-lowering-pipeline)
3. [Backend Dispatch: Device → Scheduling → Codegen](#3-backend-dispatch-device--scheduling--codegen)
4. [Kernel Codegen Deep Dive](#4-kernel-codegen-deep-dive)
5. [Full Codegen Walkthrough: aten.add → GPU](#5-full-codegen-walkthrough-atenadd--gpu)
6. [ExternKernel / Fallback Path](#6-externkernel--fallback-path)
7. [AOT Compilation to Shared Library](#7-aot-compilation-to-shared-library)
8. [Reference Tables](#8-reference-tables)

---

## 1. Compilation Entry Points

There are **four ways** to initiate Inductor compilation. All converge on the same
internal pipeline.

### 1.1. `torch.compile()` — the standard API

```python
import torch

# Simplest: just wrap your model
@torch.compile
def my_model(x):
    return x + 1

# Or with options
model = torch.compile(
    original_model,
    backend="inductor",       # default: Inductor
    fullgraph=True,           # no graph breaks
    dynamic=False,            # static shapes
    mode="max-autotune",      # "default" | "reduce-overhead" | "max-autotune"
    options={
        "triton.cudagraphs": True,
        "max_autotune": True,
    },
)
output = model(input_data)
```

**Flow:**
```
torch.compile(model, backend="inductor")
  └─> torch._dynamo.optimize(backend)(model)
        └─> _TorchCompileInductorWrapper.__call__()
              └─> torch._inductor.compile_fx(fx_graph, example_inputs)
```

### 1.2. `torch._inductor.aot_compile()` — ahead-of-time compile to .so

```python
import torch
from torch.export import export

# Export the model first
ep = export(model, (example_input,))

# Then AOT-compile
so_path = torch._inductor.aot_compile(
    ep.module(),
    args=(example_input,),      # example inputs
    options={
        "aot_inductor.output_path": "/tmp/my_model.so",
        "max_autotune": True,
    },
)

# At runtime: load and call
callable_fn = torch._inductor.aoti_load_package(so_path)
output = callable_fn(input_data)
```

**Flow:**
```
torch._inductor.aot_compile(gm, args)
  └─> compile_fx_aot()  [sets V.aot_compilation=True, cpp_wrapper=True]
        └─> compile_fx() → AotCodeCompiler.compile() → .so
```

### 1.3. `torch._inductor.standalone_compile()` — cached, serializable compilation

```python
import torch

gm = torch.export.export(model, (x,)).module()
artifact = torch._inductor.standalone_compile(
    gm,
    example_inputs=[x],
    dynamic_shapes="from_graph",
    aot=False,                    # True for AOT serialized callable
    options={"max_autotune": True},
)

# Save to disk
artifact.save(path="model_cache.bin", format="binary")

# Later, in a separate process:
reloaded = torch._inductor.CompiledArtifact.load(
    path="model_cache.bin", format="binary"
)
output = reloaded(*inputs)
```

**Flow:**
```
standalone_compile(gm, inputs)
  └─> compile_fx(gm, inputs)
        └─> CompiledFxGraph (wrapped in CacheCompiledArtifact or AOTCompiledArtifact)
```

### 1.4. Direct `compile_fx()` call — for custom compilers

```python
from torch._inductor.compile_fx import compile_fx, compile_fx_inner

# Option A: full orchestrator (handles AOTAutograd, partitioning, etc.)
result = compile_fx(
    fx_graph_module,
    example_inputs,
    config_patches={"max_autotune": True},
)

# Option B: inner callback (assumes already-partitioned fwd/bwd graphs)
result = compile_fx_inner(
    fx_graph_module,
    example_inputs,
    is_backward=False,
    graph_id=0,
)
```

---

## 2. The Complete Lowering Pipeline

Every compilation entry point converges on `_InProcessFxCompile.codegen_and_compile()`.
Here is the full pipeline, stage by stage:

```
                           Input: FX GraphModule
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                        STAGE 1                                │
    │                    pre_grad_passes()                          │
    │                                                               │
    │  • view_to_reshape() — canonicalize views for layout safety   │
    │  • FakeTensorProp — infer shapes/dtypes on all nodes          │
    │  • Pattern matcher — substitute subgraphs with fused ops      │
    │  • Decomposition selection                                    │
    └───────────────────────────────┬───────────────────────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                        STAGE 2                                │
    │                   post_grad_passes()                          │
    │                                                               │
    │  • Joint graph cleanup (reinplacing)                          │
    │  • Layout optimization                                        │
    │  • Constant folding                                           │
    │  • Dead node elimination                                      │
    └───────────────────────────────┬───────────────────────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                        STAGE 3                                │
    │              GraphLowering.run(*example_inputs)               │
    │                                                               │
    │  For each FX node (e.g., aten.add.Tensor):                    │
    │                                                               │
    │  ┌─────────────────────────────────────────────────────────┐  │
    │  │ STEP 3a: call_function() — Lowering Lookup              │  │
    │  │                                                         │  │
    │  │  target = aten.add.Tensor                               │  │
    │  │                                                         │  │
    │  │  if target in lowerings:                                │  │
    │  │    return lowerings[target](*args, **kwargs)   ← NATIVE │  │
    │  │  elif target in FALLBACK_ALLOW_LIST:                    │  │
    │  │    return fallback_handler(target)(*args)      ← FALLBK │  │
    │  │  elif decomposition exists:                             │  │
    │  │    raise MissingOperatorWithDecomp                      │  │
    │  │  else:                                                  │  │
    │  │    raise MissingOperatorWithoutDecomp                   │  │
    │  └───────────────────────┬─────────────────────────────────┘  │
    │                          │                                     │
    │  ┌───────────────────────▼─────────────────────────────────┐  │
    │  │ STEP 3b: IR Node Creation                              │  │
    │  │                                                         │  │
    │  │  Native lowering:                                       │  │
    │  │    Pointwise(device, dtype, inner_fn, ranges)            │  │
    │  │    → TensorBox(StorageBox(Pointwise(...)))              │  │
    │  │                                                         │  │
    │  │  Fallback:                                              │  │
    │  │    FallbackKernel(layout, aten_op, inputs)              │  │
    │  │    → TensorBox(StorageBox(FallbackKernel(...)))         │  │
    │  │                                                         │  │
    │  │  StorageBox.realize():                                  │  │
    │  │    → ComputedBuffer(FlexibleLayout, data=Pointwise)     │  │
    │  │    → registered in GraphLowering.operations[]            │  │
    │  └─────────────────────────────────────────────────────────┘  │
    └───────────────────────────────┬───────────────────────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                        STAGE 4                                │
    │                 GraphLowering.codegen()                       │
    │                                                               │
    │  ┌─────────────────────────────────────────────────────────┐  │
    │  │ STEP 4a: Scheduler                                     │  │
    │  │                                                         │  │
    │  │  Scheduler(GraphLowering.operations)                     │  │
    │  │                                                         │  │
    │  │  For each IR operation:                                 │  │
    │  │    ComputedBuffer → SchedulerNode                        │  │
    │  │    ExternKernel   → ExternKernelSchedulerNode            │  │
    │  │                                                         │  │
    │  │  Scheduler._init():                                     │  │
    │  │    • fuse_nodes()  — horizontal + vertical fusion       │  │
    │  │    • merge_loops() — loop coalescing                    │  │
    │  │    • dead_node_elimination()                            │  │
    │  │    • topological_sort()                                 │  │
    │  └───────────────────────┬─────────────────────────────────┘  │
    │                          │                                     │
    │  ┌───────────────────────▼─────────────────────────────────┐  │
    │  │ STEP 4b: Scheduler._codegen()                          │  │
    │  │                                                         │  │
    │  │  For each SchedulerNode:                                │  │
    │  │                                                         │  │
    │  │  if node.is_extern():                                   │  │
    │  │    codegen_extern_call(node)     → wrapper op call      │  │
    │  │  elif node.is_template():                               │  │
    │  │    get_backend(device).codegen_template(node)           │  │
    │  │  else:                                                  │  │
    │  │    get_backend(device).codegen_node(node)               │  │
    │  │      → SIMDScheduling.codegen_node()                    │  │
    │  │        → generate_node_schedule()  (tiling plan)        │  │
    │  │        → create_kernel_choices()   (kernel instances)   │  │
    │  │        → kernel.codegen_kernel()   (source code string) │  │
    │  │        → define_kernel(src_code)   (register in wrapper)│  │
    │  │        → kernel.call_kernel(name)  (emit launch code)   │  │
    │  └───────────────────────┬─────────────────────────────────┘  │
    │                          │                                     │
    │  ┌───────────────────────▼─────────────────────────────────┐  │
    │  │ STEP 4c: Wrapper Codegen                               │  │
    │  │                                                         │  │
    │  │  wrapper_code.generate(is_inference)                     │  │
    │  │    → Python source code as a string                     │  │
    │  │    → Compiled via PyCodeCache/Python's compile()        │  │
    │  │    → Returns a callable module                          │  │
    │  └─────────────────────────────────────────────────────────┘  │
    └───────────────────────────────┬───────────────────────────────┘
                                    │
                            ┌───────▼──────┐
                            │  OutputCode  │  callable
                            └──────────────┘
```

---

## 3. Backend Dispatch: Device → Scheduling → Codegen

Inductor uses a **three-level dispatch** to route compilation to the right backend:

### 3.1. Registration Table

The global registry is in `torch/_inductor/codegen/common.py`:

```python
# common.py:305-311
@dataclasses.dataclass
class DeviceCodegen:
    scheduling: SchedulingConstructor       # e.g., TritonScheduling
    wrapper_codegen: WrapperConstructor     # e.g., PythonWrapperCodegen
    cpp_wrapper_codegen: WrapperConstructor | None = None
    fx_wrapper_codegen: WrapperConstructor | None = None

device_codegens: dict[str, DeviceCodegen] = {}
```

Populated via `register_backend_for_device()`:

```python
register_backend_for_device(
    device="cuda",
    device_scheduling=CUDACombinedScheduling,
    device_wrapper_codegen=PythonWrapperCodegen,
    device_cpp_wrapper_codegen=CppWrapperGpu,
)
```

### 3.2. Complete Device Mapping

| Device | Scheduling | Kernel Type | Wrapper (Python) | Wrapper (C++) | Wrapper (FX) |
|--------|-----------|-------------|-------------------|---------------|--------------|
| `cpu` | `CppScheduling` | `CppKernel` / `CppVecKernel` | `PythonWrapperCodegen` | `CppWrapperCpu` | `WrapperFxCodegen` |
| `cuda` | `CUDACombinedScheduling` | `TritonKernel` / `CUTLASSKernel` | `PythonWrapperCodegen` | `CppWrapperGpu` | `WrapperFxCodegen` |
| `xpu` | `TritonScheduling` | `TritonKernel` | `PythonWrapperCodegen` | `CppWrapperGpu` | `WrapperFxCodegen` |
| `mps` | `MetalScheduling` | `MetalKernel` | `PythonWrapperCodegen` | `CppWrapperMps` | `WrapperFxCodegen` |
| `tpu` | `PallasScheduling` | `PallasKernel` | `PythonWrapperCodegen` | — | — |
| `mtia` | `TritonScheduling` | `TritonKernel` | `PythonWrapperMtia` | `CppWrapperGpu` | `WrapperFxCodegen` |
| `vulkan` | `VulkanScheduling` | `VulkanKernel` | `VulkanPythonWrapperCodegen` | — | — |

### 3.3. Dispatch Chain

```
GraphLowering.codegen()
  │
  ├── init_wrapper_code()
  │     └── get_wrapper_codegen_for_device(device_type)
  │            → device_codegens["cuda"].wrapper_codegen
  │            → PythonWrapperCodegen
  │
  └── Scheduler._codegen()
        └── self.get_backend(device)
              └── get_scheduling_for_device(device.type)
                    → device_codegens["cuda"].scheduling
                    → CUDACombinedScheduling(scheduler)

CUDACombinedScheduling delegates to TritonScheduling / CUTLASSScheduling
  └── TritonScheduling.kernel_type = TritonKernel
  └── TritonScheduling.codegen_node()
        └── create_kernel_choices() → TritonKernel(*args)
        └── kernel.codegen_kernel() → string of Triton source
        └── define_kernel(triton_source)
        └── kernel.call_kernel(name) → wrapper.define_kernel()
```

---

## 4. Kernel Codegen Deep Dive

### 4.1. Kernel Class Hierarchy

```
CodeGen (common.py)                   ← context manager, enter/exit hooks
  │
  └── Kernel[CSEVariableType] (common.py)
        │   • loads/compute/stores: IndentedBuffer (3 separate code buffers)
        │   • cse: CSE (common subexpression elimination)
        │   • load(name, index) → CSEVariable
        │   • store(name, index, value)
        │   • reduction(dtype, src_dtype, reduction_type, value)
        │
        └── SIMDKernel[CSEVariableType] (simd.py)
              │   • range_trees: list[IterationRangesRoot]
              │   • body: IndentedBuffer
              │   • indexing_code: IndentedBuffer
              │   • split_and_set_ranges(sizes) → index_vars
              │   • finalize_indexing()
              │   • codegen_body() — stitches loads+compute+stores
              │   • codegen_kernel() → str
              │
              ├── TritonKernel (triton.py)
              │     • generates: tl.load, tl.store, tl.reduce, @triton.jit
              │
              ├── CppKernel (cpp.py)
              │     • generates: C++ with loop nests, vector intrinsics
              │
              ├── HalideKernel (halide.py)
              │     • generates: Halide scheduling language
              │
              ├── MetalKernel (mps.py)
              │     • generates: Metal Shading Language
              │
              ├── PallasKernel (pallas.py)
              │     • generates: JAX Pallas (TPU)
              │
              └── VulkanKernel (backends/vulkan_slang/.../kernel/)
                    • generates: Slang compute shader → SPIR-V
```

### 4.2. The Two-Pass Codegen Pattern

Every `SIMDKernel` subclass follows this two-pass approach:

**Pass 1 — Indexing & Layout:**
```
For each node in the fused schedule:
  1. kernel.split_and_set_ranges(node.get_ranges())
       → Maps the node's output sizes onto the kernel's range_trees
       → For Triton: xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
       → For Vulkan: uint gid_x = gl_GlobalInvocationID.x;
  2. node.decide_inplace_update()
       → Can outputs reuse input buffers? (saves memory)
  3. Collect all indexing expressions (for indirect indexing)
```

**Pass 2 — Body Codegen:**
```
For each node in the fused schedule:
  1. node.codegen(index_vars)
       → Calls self._body(*index_vars)
       → _body() evaluates the LoopBody's FX graph
       → Each op is intercepted by CSEProxy
         → Kernel.load(name, index)  → loads buffer: "float tmp0 = buf1[idx];"
         → CSE.generate(compute, expr) → to compute buffer: "float tmp1 = tmp0 + 1.0f;"
         → Kernel.store(name, index, val) → to stores buffer: "buf0[idx] = tmp1;"
```

### 4.3. CSE (Common Subexpression Elimination)

```python
class CSE:
    _cache: dict[tuple, CSEVariable]

    def generate(self, buffer, expr, *, bounds, write=True, ...):
        cache_key = (expr, bounds, dtype)
        if cache_key in self._cache:
            return self._cache[cache_key]         # REUSE cached variable

        var = self.newvar(bounds, dtype)
        self._cache[cache_key] = var
        if write:
            buffer.writeline(f"{var} = {expr};")  # EMIT new assignment
        return var
```

This means `a * b` computed twice produces only one line of generated code.

### 4.4. ExprPrinter — SymPy → Source Language

Index expressions (stored as SymPy expressions in the IR) are converted to
source code by device-specific printers:

| Backend | Printer Class | Example: FloorDiv(x, 4) |
|---------|--------------|--------------------------|
| Triton/CUDA | `PythonPrinter` | `(x // 4)` |
| C++ | `cexpr` | `c10::div_floor_integer(x, 4L)` |
| Halide | Halide expressions | `(x/4)` |
| Vulkan/Slang | `VulkanExprPrinter` | `((x) / (4) - ...)` (branchless floor) |

### 4.5. Code Assembly: codegen_kernel()

The final source is assembled by stitching buffers:

```python
def codegen_kernel(self, name=None) -> str:
    self.codegen_body()             # stitches loads + compute + stores
    code = IndentedBuffer()

    # 1. Emit header (imports, function signature, grid/block config)
    #    e.g.: @triton.jit  or  [numthreads(256,1,1)]
    code.writeline("[numthreads(256, 1, 1)]")
    code.writeline("void kernel_main(")

    # 2. Emit buffer declarations
    #    e.g.: RWStructuredBuffer<float> buf0 : register(u0);
    for buf in output_buffers:
        code.writeline(f"RWStructuredBuffer<{dtype}> {name} : register(u{slot});")

    # 3. Emit helper functions
    self._emit_helpers(code)

    # 4. Emit the body
    code.splice(self.body)

    # 5. Emit indexing code
    code.splice(self.indexing_code)

    return code.getvalue()
```

---

## 5. Full Codegen Walkthrough: aten.add → GPU

Here is a concrete trace of how `aten.add.Tensor(x, y)` becomes a GPU kernel:

### Step 1: FX Node

```
FX Graph:
  %add : Tensor = aten.add.Tensor(%x, %y)
```

### Step 2: Lowering → IR

```python
# lowering.py (registered lowering for aten.add.Tensor)
@register_lowering(aten.add.Tensor, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
def add(x, y):
    return TensorBox(Pointwise.create(
        device=x.get_device(),
        dtype=result_dtype,
        inner_fn=lambda x_idx, y_idx: ops.add(
            load_input(x, x_idx),
            load_input(y, y_idx),
        ),
        ranges=[size],
    ))
```

### Step 3: Scheduler → Fused Node

```
SchedulerNode("add_0")
  .group = (numel=1024, rnumel=1)
  .node = ComputedBuffer(
    layout = FixedLayout(device="cuda", dtype=float32, size=[1024]),
    data = Pointwise(inner_fn=<lambda>, ranges=[1024])
  )
```

### Step 4: Kernel Creation (Triton)

```python
# TritonScheduling.codegen_node()
features = SIMDKernelFeatures(node_schedule, numel=1024, rnumel=1)
kernel = TritonKernel(
    tiling={"XBLOCK": 256, "R0BLOCK": 1},
    numel=1024,
    rnumel=1,
)
```

### Step 5: Two-Pass Codegen

**Pass 1 — set up ranges:**
```
kernel.split_and_set_ranges(ranges=[1024])
  → range_trees = [
      IterationRangesRoot("xindex", 1024, "XBLOCK"),
    ]
  → index_vars = [x0]  where x0 = xindex (with xindex = pid*256 + arange(256))
```

**Pass 2 — evaluate inner_fn:**
```
CSEProxy intercepted ops:
  load("arg0_1", x0)  → loads: "tmp0 = tl.load(in_ptr0 + x0, ...)"
  load("arg1_1", x0)  → loads: "tmp1 = tl.load(in_ptr1 + x0, ...)"
  ops.add(tmp0, tmp1)  → compute: "tmp2 = tmp0 + tmp1"
  store("buf0", x0, tmp2) → stores: "tl.store(out_ptr0 + x0, tmp2, ...)"
```

### Step 6: Final Triton Source

```python
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < 1024
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)
```

### Step 7: Wrapper

```python
# Generated Python wrapper
def call(args):
    arg0_1, arg1_1 = args
    buf0 = empty_strided_cuda((1024,), (1,), torch.float32, "cuda")
    triton_[256](arg0_1, arg1_1, buf0, 1024)
    del arg0_1, arg1_1
    return (buf0,)
```

---

## 6. ExternKernel / Fallback Path

When no Inductor-native lowering exists for an op, it takes the **fallback path**:

```
No lowering in lowerings dict
  ↓
Check FALLBACK_ALLOW_LIST (or implicit_fallbacks config)
  ↓
fallback_handler(aten.complex_op)()
  ↓
FallbackKernel(ExternKernelAlloc)
  ↓
Registered as ExternKernelSchedulerNode in scheduler
  ↓
codegen_extern_call(node):
  emits: buf0 = torch.ops.aten.complex_op.default(arg0, arg1)
```

This bypasses all loop codegen — the op is emitted as a direct function call in
the wrapper code. No kernel source is generated; the runtime dispatches to the
framework's native op implementation.

**Fusion implications:** ExternKernels are fission boundaries. Pointwise/reduction
ops cannot be fused across an ExternKernel, so each side becomes a separate kernel.

---

## 7. AOT Compilation to Shared Library

For deployment (no Python runtime), Inductor can produce a standalone `.so`:

### Pipeline

```
FX GraphModule
  ↓
aot_compile(gm, args)
  ↓
compile_fx_aot()
  ├── Sets: V.aot_compilation = True
  ├── Sets: cpp_wrapper = True
  └── aot_export_module(model, args, trace_joint=False)
        ↓
      Unlift graph (parameters → inputs)
        ↓
      compile_fx_inner()  [same as normal path]
        ↓
      _InProcessFxCompile.codegen_and_compile()
        ├── post_grad_passes()
        ├── GraphLowering.run()
        └── graph.codegen_with_cpp_wrapper()
              ├── wrapper_code: C++ source (not Python!)
              ├── kernel_code: Triton/C++ kernels
              └── extern_kernel_nodes: serialized
        ↓
      AotCodeCompiler.compile(
          graph,
          wrapper_code,      # C++ host code
          kernel_code,       # Triton/C++ kernel code
          extern_kernel_nodes,
      )
        ↓
      Produces: model.so + weights
```

### At Runtime

```python
# Load the compiled .so
runner = torch._inductor.runtime.aoti_runtime_wrapper.AOTIModelContainerRunnerCuda(
    so_path="/tmp/model.so",
    num_models=1,
)

# Call it directly (no Python overhead)
output = runner.run([input_tensor])
```

---

## 8. Reference Tables

### 8.1. Key Files

| File | Role |
|------|------|
| `torch/__init__.py` | `torch.compile()` API definition |
| `torch/_inductor/compile_fx.py` | `compile_fx()`, `compile_fx_inner()`, `compile_fx_aot()` |
| `torch/_inductor/_compile_fx.py` | `_InProcessFxCompile.codegen_and_compile()` (the core) |
| `torch/_inductor/graph.py` | `GraphLowering` — FX interpreter, lowering dispatch, `codegen()` |
| `torch/_inductor/lowering.py` | `register_lowering()`, `lowerings` dict, `fallback_handler()` |
| `torch/_inductor/ir.py` | IR nodes: `TensorBox`, `Pointwise`, `Reduction`, `ExternKernel`, `Buffer` |
| `torch/_inductor/scheduler.py` | `Scheduler`, `BaseScheduling`, fusion, `_codegen()` |
| `torch/_inductor/codegen/common.py` | `Kernel`, `SIMDKernel`, `CSE`, `OpOverrides`, backend registry |
| `torch/_inductor/codegen/simd.py` | `SIMDScheduling`, `SIMDKernel` base, loop generation |
| `torch/_inductor/codegen/wrapper.py` | `PythonWrapperCodegen` — host orchestration |
| `torch/_inductor/codegen/triton.py` | `TritonKernel`, `TritonScheduling` — Triton backend |
| `torch/_inductor/codegen/cpp.py` | `CppKernel`, `CppScheduling`, `CppVecKernel` — CPU backend |
| `torch/_inductor/codecache.py` | `PyCodeCache`, `FxGraphCache`, `AotCodeCompiler` |
| `torch/_inductor/output_code.py` | `OutputCode`, `CompiledFxGraph`, `CompiledAOTI` |
| `torch/_inductor/standalone_compile.py` | `standalone_compile()`, `CompiledArtifact` |

### 8.2. Key Class Hierarchy

```
CodeGen
  └── Kernel
        ├── SIMDKernel → TritonKernel, HalideKernel, MetalKernel, PallasKernel, VulkanKernel
        ├── CppKernel → CppVecKernel → CppTile2DKernel
        ├── CppTemplateKernel
        └── Template kernels: CUTLASSKernel, ROCmKernel, NVUniversalGemmKernel

BaseScheduling
  ├── CppScheduling (kernel_type = CppKernel)
  ├── SIMDScheduling → TritonScheduling, HalideScheduling, MetalScheduling, VulkanScheduling
  └── CUDACombinedScheduling (delegates to Triton/CUTLASS/CuteDSL)

PythonWrapperCodegen
  ├── SubgraphPythonWrapperCodegen
  ├── CppWrapperCpu → CppWrapperCpuArrayRef
  ├── CppWrapperGpu
  ├── CppWrapperMps
  ├── VulkanPythonWrapperCodegen
  ├── PythonWrapperMtia
  └── WrapperFxCodegen
```

### 8.3. OutputCode Types

| Type | Contains | Callable? | Serializable? |
|------|---------|-----------|---------------|
| `OutputCode` (base) | Cache key, debug info | No | — |
| `CompiledFxGraph` | `current_callable`, graph, output strides | Yes (`__call__`) | Yes (FxGraphCache) |
| `CompiledAOTI` | `.so` path, model runner | Yes (`__call__`) | Already a file |
| `CacheCompiledArtifact` | `CompiledFxGraph` + cache artifacts | Yes | Yes (binary/unpacked) |
| `AOTCompiledArtifact` | `BundledAOTAutogradSerializableCallable` | Yes | Yes (binary) |

### 8.4. Common config_patches / options

| Key | Effect |
|-----|--------|
| `"max_autotune": True` | Enable autotuning for matmul/conv/templates |
| `"triton.cudagraphs": True` | Enable CUDA graphs for reduced launch overhead |
| `"cpp_wrapper": True` | Generate C++ wrapper (for AOT export) |
| `"fx_wrapper": True` | Generate FX IR wrapper |
| `"aot_inductor.output_path"` | Output path for AOT .so |
| `"aot_inductor.package": True` | Package compiled artifacts |
| `"dynamic_shapes": True` | Don't specialize on input shapes |
| `"coordinate_descent_tuning": True` | Use coordinate descent for autotuning |
| `"epilogue_fusion": True` | Fuse epilogues into matmul/convs |
