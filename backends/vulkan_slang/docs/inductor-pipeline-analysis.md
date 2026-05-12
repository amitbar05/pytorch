# PyTorch Inductor Pipeline: Component Breakdown & Slang/Codegen Integration

> **Last updated: 2026-05-07**

This document provides a comprehensive breakdown of the PyTorch Inductor
compiler pipeline, with deep-dive analysis of how **Slang** and the **Vulkan
codegen** are integrated as a custom backend.

---

## Table of Contents

1. [High-Level Pipeline Overview](#1-high-level-pipeline-overview)
2. [Pipeline Stages (Diagram)](#2-pipeline-stages-diagram)
3. [Stage-by-Stage Breakdown](#3-stage-by-stage-breakdown)
4. [The Slang/Vulkan Backend: Where & How It Plugs In](#4-the-slangvulkan-backend-where--how-it-plugs-in)
5. [Slang/Codegen Subsystem Architecture](#5-slangcodegen-subsystem-architecture)
6. [Codegen Flow: From Aten Op to SPIR-V Dispatch](#6-codegen-flow-from-aten-op-to-spir-v-dispatch)
7. [Key File Map](#7-key-file-map)

---

## 1. High-Level Pipeline Overview

PyTorch Inductor is the default `torch.compile` backend. It takes an
AOTAutograd-produced FX graph and lowers it through a multi-stage pipeline:

```
User Model
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  torch.compile (Dynamo)                                       │
│  - Python bytecode → FX trace                                 │
│  - Guards, graph breaks, recompilation                        │
└──────────────────────────┬───────────────────────────────────┘
                           │  FX GraphModule
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  AOTAutograd                                                   │
│  - Forward + backward joint graph                             │
│  - Functionalization, partitioner                              │
│  - Min-cut rematerialization                                  │
└──────────────────────────┬───────────────────────────────────┘
                           │  Partitioned fwd/bwd FX graphs
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Inductor (compile_fx_inner)  ◄── THIS DOCUMENT               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 1. Pre-grad passes                    (fx_passes/)       │ │
│  │ 2. Post-grad passes                   (fx_passes/)       │ │
│  │ 3. GraphLowering                      (graph.py)         │ │
│  │    ├── Lowering (aten→IR)             (lowering.py)      │ │
│  │    ├── Scheduler (fusion)             (scheduler.py)     │ │
│  │    ├── Codegen (IR→source)            (codegen/)         │ │
│  │    └── Wrapper (source→Python/C++)    (codegen/wrapper)   │ │
│  │ 4. Compile to module /.so                                 │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Pipeline Stages (Diagram)

```
                        ┌──────────────┐
                        │   FX Graph   │  from AOTAutograd
                        └──────┬───────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    ▼                     │
          │  ┌──────────────────────────────────┐   │
          │  │  STAGE 1: pre_grad_passes()      │   │
          │  │  - Pattern matching (fusions)    │   │
          │  │  - Decomposition selection       │   │
          │  │  - View→reshape canonicalization │   │
          │  └──────────────┬───────────────────┘   │
          │                 │                        │
          │  ┌──────────────▼───────────────────┐   │
          │  │  STAGE 2: post_grad_passes()    │   │
          │  │  - Joint graph cleanup          │   │
          │  │  - Layout optimization          │   │
          │  │  - Constant folding             │   │
          │  └──────────────┬───────────────────┘   │
          │                 │                        │
          │  ┌──────────────▼───────────────────┐   │
          │  │  STAGE 3: GraphLowering()        │   │
          │  │                                  │   │
          │  │  ┌────────────────────────────┐  │   │
          │  │  │ 3a. LOWERING               │  │   │
          │  │  │   aten op → IR node        │  │   │
          │  │  │   (TensorBox, Pointwise,   │  │   │
          │  │  │    Reduction, etc.)        │  │   │
          │  │  │                            │  │   │
          │  │  │  • register_lowering()     │  │   │
          │  │  │  • fallback_handler()      │  │   │
          │  │  │  • decomposition           │  │   │
          │  │  └────────────┬───────────────┘  │   │
          │  │               │                   │   │
          │  │  ┌────────────▼───────────────┐  │   │
          │  │  │ 3b. SCHEDULER              │  │   │
          │  │  │   IR nodes → fused groups  │  │   │
          │  │  │                            │  │   │
          │  │  │  • can_fuse() heuristics   │  │   │
          │  │  │  • Horizontal fusion       │  │   │
          │  │  │  • Vertical fusion         │  │   │
          │  │  │  • Template partitioning   │  │   │
          │  │  │  • Memory planning         │  │   │
          │  │  └────────────┬───────────────┘  │   │
          │  │               │                   │   │
          │  │  ┌────────────▼───────────────┐  │   │
          │  │  │ 3c. CODEGEN               │  │   │
          │  │  │   Fused groups → Kernel    │  │   │
          │  │  │   source code             │  │   │
          │  │  │                            │  │   │
          │  │  │  • SIMDKernel subclasses   │  │   │
          │  │  │    - TritonKernel (CUDA)   │  │   │
          │  │  │    - CppKernel (CPU)       │  │   │
          │  │  │    - VulkanKernel (GPU)    │  │   │
          │  │  │  • TemplateKernel          │  │   │
          │  │  │  • CSE, index simplification│  │   │
          │  │  │  • ExprPrinter (SymPy→src) │  │   │
          │  │  └────────────┬───────────────┘  │   │
          │  │               │                   │   │
          │  │  ┌────────────▼───────────────┐  │   │
          │  │  │ 3d. WRAPPER               │  │   │
          │  │  │   Orchestrates calls      │  │   │
          │  │  │                            │  │   │
          │  │  │  • Emit Python/C++ host   │  │   │
          │  │  │    code calling kernels    │  │   │
          │  │  │  • Buffer allocation       │  │   │
          │  │  │  • Device synchronization  │  │   │
          │  │  │  • Input/output asserts    │  │   │
          │  │  └────────────┬───────────────┘  │   │
          │  │               │                   │   │
          │  └───────────────┼───────────────────┘   │
          │                  │                        │
          │  ┌───────────────▼───────────────────┐   │
          │  │  STAGE 4: compile_to_module()    │   │
          │  │  - PyCodeCache → .py → compiled  │   │
          │  │  - AotCodeCompiler → .so         │   │
          │  └──────────────┬───────────────────┘   │
          │                 │                        │
          └─────────────────┼────────────────────────┘
                            │
                    ┌───────▼──────┐
                    │  OutputCode  │   callable / .so
                    └──────────────┘
```

---

## 3. Stage-by-Stage Breakdown

### Stage 1: Pre-Grad Passes (`torch/_inductor/fx_passes/pre_grad.py`)

Applied to the forward graph **before** joint graph construction:

| Pass | Purpose |
|------|---------|
| `view_to_reshape()` | Canonicalize views → reshapes (layout opt safety) |
| `pre_grad_passes()` | Pattern-matching fusions (e.g. `nn.functional.linear`) |
| FakeTensor propagation | `FakeTensorProp` — shape/dtype inference on all nodes |

### Stage 2: Post-Grad Passes (`torch/_inductor/fx_passes/post_grad.py`)

Applied **after** AOTAutograd produces the joint (fwd+bwd) graph:

| Pass | Purpose |
|------|---------|
| `joint_graph_passes()` | Joint-graph cleanup, reinplacing |
| `post_grad_passes()` | Dead-node elimination, layout optimization, constant folding |
| Decomposition | `select_decomp_table()` — pick operator decompositions |
| Pattern matching | `pattern_matcher.py` — match subgraphs → fused ops |

### Stage 3a: Lowering (`torch/_inductor/lowering.py`)

The lowering stage converts each **FX node** (`aten.op`) into one or more
**Inductor IR nodes** (`TensorBox`, `Pointwise`, `Reduction`, `ExternKernel`, etc.).

```python
# Example: aten.add.Tensor → Pointwise IR node
register_lowering(aten.add.Tensor, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)(
    lambda x, y: TensorBox(Pointwise.create(...))
)
```

**Dispatch order:**
1. Check `lowerings` dict for a registered lowering
2. Check `FALLBACK_ALLOW_LIST` → `fallback_handler` (ExternKernel)
3. Check decomposition tables
4. Error: `MissingOperatorWithoutDecomp`

Key classes produced:
- `TensorBox` — wrapper around any IR node
- `Pointwise` — elementwise ops (load→compute→store)
- `Reduction` — sum/mean/max/min/etc.
- `ExternKernel` — opaque external calls (fallback, cuBLAS, etc.)
- `MultiOutput` — multiple returns (e.g. `native_layer_norm`)

### Stage 3b: Scheduler (`torch/_inductor/scheduler.py`)

The scheduler groups IR nodes into **fusion groups** (kernels):

```
Pointwise{B}  Pointwise{C}
       \          /
    Pointwise{A}   Reduction{R}
           \         /
          [Fused kernel 1]
```

Key decisions:
- `can_fuse(node1, node2)` — per-backend fusion heuristic
- **Horizontal fusion**: sibling pointwise ops → one kernel
- **Vertical fusion**: producer→consumer chain → one kernel
- **Template partitioning**: matmul/conv get their own template kernel
- **Memory planning**: buffer lifetimes, in-place reuse

### Stage 3c: Codegen (`torch/_inductor/codegen/`)

Each fused scheduler node generates **kernel source code**:

| Backend | Kernel Class | Output Language |
|---------|-------------|-----------------|
| CUDA    | `TritonKernel` (`triton.py`) | Triton (Python DSL) |
| CPU     | `CppKernel` (`cpp.py`) | C++ (with vector intrinsics) |
| Vulkan  | `VulkanKernel` (`kernel/`) | **Slang** (→ SPIR-V) |
| MPS     | `MPSKernel` (`mps.py`) | Metal Shading Language |
| Halide  | `HalideKernel` (`halide.py`) | Halide |

All kernel classes inherit from `SIMDKernel` (`codegen/simd.py`) which provides:
- CSE (Common Subexpression Elimination)
- Index expression simplification (`simplify_indexing`)
- Loop body generation
- Load/compute/store buffer management

The **ExprPrinter** converts SymPy index expressions to the target language:
- `PythonPrinter` → Python for Triton
- `VulkanExprPrinter` → Slang (with HLSL-like arithmetic semantics)

### Stage 3d: Wrapper (`torch/_inductor/codegen/wrapper.py`)

The wrapper generates the **host-side orchestration code** that:
1. Allocates intermediate buffers
2. Calls each kernel with the correct dispatch parameters
3. Handles device synchronization
4. Returns outputs

Each device has its own wrapper:
| Backend | Wrapper Class |
|---------|---------------|
| CUDA/Triton | `PythonWrapperCodegen` |
| CPU | `CppWrapperCpu` |
| Vulkan | `VulkanPythonWrapperCodegen` |
| MPS | `CppWrapperMps` |

### Stage 4: Compile to Module (`GraphLowering.compile_to_module()`)

For non-AOT mode:
1. Wrapper source → `PyCodeCache` → compiled Python module
2. Loaded as `CompiledModule` with `.call()` entry point

For AOT mode:
1. Wrapper + kernel source → `AotCodeCompiler` → `.so` / `.pt2`
2. Returns `CompiledAOTI`

---

## 4. The Slang/Vulkan Backend: Where & How It Plugs In

The Vulkan/Slang backend is an **out-of-tree backend** (at `backends/vulkan_slang/`)
that registers with Inductor via `PrivateUse1` as `torch.device("vulkan")`.

### 4.1. Registration Entry Point

```
backends/vulkan_slang/python/torch_vulkan/__init__.py
    └── import torch_vulkan.inductor
        └── backends/vulkan_slang/python/torch_vulkan/inductor/__init__.py
            └── SlangVulkanBackend.register()
                └── _legacy_register()
```

`_legacy_register()` installs **every hook** Inductor needs to route Vulkan
tensors through the Slang codegen:

```python
# Simplified registration logic
def _legacy_register():
    # 1. Register device-op overrides
    register_device_op_overrides("vulkan", VulkanDeviceOpOverrides())

    # 2. Register wrapper codegen
    register_wrapper_codegen("vulkan", VulkanPythonWrapperCodegen)

    # 3. Register scheduling backend
    register_scheduling_for_device("vulkan", VulkanScheduling)

    # 4. Register kernel type (used by scheduler)
    #    VulkanScheduling.kernel_type = VulkanKernel

    # 5. Register lowerings for vulkan-specific ops
    from .lowerings import register as register_lowerings
    register_lowerings()

    # 6. Install custom op integrations (mm, rng, flash attention, optimizer)
    install_external_mm()
    install_external_rng()
    install_external_flash_attention()
    install_external_optimizer()

    # 7. Pre-warm slangc, compile shader libraries
    precompile_shader_libs()
```

### 4.2. The Four Integration Points

The Vulkan/Slang backend hooks into Inductor at exactly **four seams**:

```
┌────────────────────────────────────────────────────────────┐
│                    Inductor Core Pipeline                   │
│                                                            │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐ │
│  │ Lowering │──▶│ Scheduler│──▶│ Codegen  │──▶│ Wrapper │ │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬────┘ │
│       │              │              │              │        │
└───────┼──────────────┼──────────────┼──────────────┼────────┘
        │              │              │              │
        │     ┌────────┼──────────────┼──────────────┼─────┐
        │     │        │    Vulkan / Slang Backend    │     │
        │     │        │                              │     │
        ├─────┼────────┤  ① Lowerings                 │     │
        │     │        │     lowerings/*.py           │     │
        │     │        │     device_op_overrides.py   │     │
        │     │        │                              │     │
        │     ├────────┤  ② Scheduler                 │     │
        │     │        │     scheduling.py            │     │
        │     │        │     (VulkanScheduling)       │     │
        │     │        │                              │     │
        │     │  ┌─────┤  ③ Codegen                   │     │
        │     │  │     │     kernel/main.py           │     │
        │     │  │     │       (VulkanKernel)         │     │
        │     │  │     │     expr_printer.py          │     │
        │     │  │     │     overrides.py             │     │
        │     │  │     │     slang_helpers.py         │     │
        │     │  │     │     vulkan_template.py       │     │
        │     │  │     │                              │     │
        │     │  │  ┌──┤  ④ Wrapper + Runtime          │     │
        │     │  │  │  │     wrapper.py               │     │
        │     │  │  │  │     runtime.py (slangc)      │     │
        │     │  │  │  │     buffer_pool.py           │     │
        │     │  │  │  └──────────────────────────────┘     │
        └─────┴──┴──┴───────────────────────────────────────┘
```

---

## 5. Slang/Codegen Subsystem Architecture

```
backends/vulkan_slang/python/torch_vulkan/inductor/
│
├── __init__.py            ← Entry: _legacy_register()
├── backend.py             ← SlangVulkanBackend façade
├── config.py              ← Kill-switches (TORCH_VULKAN_*)
│
├── lowerings/             ← ① LOWERING INTEGRATION
│   ├── __init__.py        ←   Register vulkan-specific lowerings
│   └── bwd_diff.py        ←   Backward-diff lowering dispatch
│
├── scheduling.py          ← ② SCHEDULER INTEGRATION
│   └── VulkanScheduling(SIMDScheduling)
│       - kernel_type = VulkanKernel
│       - can_fuse() heuristics (Vulkan-specific limits)
│       - get_backend_features() (FOREACH, SCAN, SORT, etc.)
│
├── codegen.py             ← ③ CODEGEN re-export shim
│   └── OpClass, CodegenStrategy, CODEGEN_STRATEGIES
│
├── kernel/                ← ③ CODEGEN (VulkanKernel)
│   ├── __init__.py        ←   VulkanKernel (SIMDKernel subclass)
│   ├── main.py            ←   Core class + VulkanCSE
│   ├── header.py          ←   Shader header, buffer decls, dispatch
│   ├── pointwise.py       ←   Pointwise load/compute/store
│   ├── reduction.py       ←   1D/2D reduction, welford, scan, sort
│   ├── indexing.py        ←   Indirect indexing
│   └── symbolic.py        ←   Dynamic shape support
│
├── expr_printer.py        ← ③ Slang SymPy→Slang printer
│   └── VulkanExprPrinter(ExprPrinter)
│       - FloorDiv, ModularIndexing, bool casting
│       - Subscript-context int casting
│
├── overrides.py            ← ③ Op→Slang snippet mapping
│   ├── DTYPE_TO_SLANG      ←   torch dtype → Slang type
│   ├── VulkanOverrides     ←   abs→"abs(x)", exp→"exp(x)", etc.
│   └── value_to_slang()    ←   Python scalar → Slang literal
│
├── slang_helpers.py        ← ③ Shared helper emission
│   ├── emit_helpers()      ←   f16/bf16 pack/unpack
│   └── emit_packed16_helpers()
│
├── vulkan_template.py      ← ③ Template (Jinja) kernel system
│   ├── VulkanTemplateKernel ←   Jinja→Slang template expansion
│   ├── SlangTemplate        ←   Analogous to TritonTemplate
│   └── _unwrap_slang_template() ←   .slang-embedded Jinja unwrapper
│
├── vulkan_template_caller.py ← ③ Template call-site installers
│   └── install_external_{mm,rng,flash_attention,optimizer}
│
├── templates/              ← ③ Jinja/.slang templates
│   ├── slang_mm.slang      ←   Matmul with IPointwise epilogue
│   ├── slang_conv2d.slang  ←   Conv2D im2col
│   ├── philox_rng.py.jinja ←   Philox RNG
│   ├── foreach_optimizer.py.jinja ← SGD/AdamW/Lion foreach
│   └── flash_attention.py.jinja   ← Flash attention
│
├── vulkan_combo_kernel.py  ← ③ Multi-kernel combiner (FOREACH)
│
├── wrapper.py              ← ④ WRAPPER INTEGRATION
│   └── VulkanPythonWrapperCodegen(PythonWrapperCodegen)
│       - Buffer pool allocation
│       - Kernel call emission
│       - Assert elision (trust_inductor)
│
├── runtime.py              ← ④ RUNTIME: Slang→SPIR-V compile + dispatch
│   ├── _resolve_slangc()   ←   Locate slangc binary
│   ├── SlangKernel         ←   Compile: slangc → SPIR-V
│   └── make_vulkan_kernel()←   Create dispatchable kernel
│
├── buffer_pool.py          ← ④ Buffer lifecycle management
│   └── vulkan_pool_{acquire,release}()
│
├── device_op_overrides.py  ← ④ Device-level helpers
│   └── VulkanDeviceOpOverrides (single-device, no-op guards)
│
├── device_interface.py     ← ④ Vulkan device interface
│   └── (Worker, device properties, stream mgmt)
│
├── fx_passes/              ← FX graph pre/post processing
├── heuristics/             ← Autotune heuristics
├── compile_graph.py        ← P4.7 Vulkan-graph (cmd buffer) replay
├── lifetime.py             ← T6.2 gradient release hooks
├── autotune.py             ← Max-autotune integration
└── ...
```

### 5.1. The Shader Library (`shaders/lib/`)

Precompiled Slang modules that provide generic, reusable GPU primitives:

```
shaders/lib/
├── helpers.slang       ←   Buffer ops, indexing, warp primitives
├── pointwise.slang     ←   IPointwise interface, activation fwd/bwd
├── reduction.slang     ←   IReduction, IWaveReduction generics
├── mm.slang            ←   Matrix multiply primitives
├── mm_tiled.slang      ←   Tiled matmul with shared memory
├── conv.slang          ←   Convolution primitives
├── norm.slang          ←   LayerNorm, BatchNorm
├── attention.slang     ←   Flash attention primitive
├── tensor_layout.slang ←   Layout helpers (NCHW→NHWC, etc.)
├── training.slang      ←   Optimizer step primitives
├── atomics.slang       ←   Atomic operations
└── losses.slang        ←   Loss function primitives
```

These libraries are precompiled to `.slang-module` at backend import time
(`runtime.py:precompile_shader_libs()`), so generated kernels `import` them
with zero cold-compile overhead.

---

## 6. Codegen Flow: From Aten Op to SPIR-V Dispatch

### 6.1. IR-Codegen Path (Pointwise / Reduction)

This is the primary path for elementwise ops and reductions:

```
aten.add.Tensor(x, y)    ← FX node in the graph
        │
        ▼
┌─────────────────────────────────────────────┐
│ 1. LOWERING (lowerings/__init__.py)          │
│    register_lowering(aten.add.Tensor)(...)    │
│    → TensorBox(Pointwise.create(...))         │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ 2. SCHEDULER (scheduling.py)                │
│    VulkanScheduling.can_fuse(add, next_op)   │
│    → Group with adjacent pointwise ops       │
│    → Create SchedulerNode (fusion group)     │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ 3. CODEGEN: VulkanKernel.codegen_kernel()   │
│    a. header.py: emit buffer declarations    │
│       RWStructuredBuffer<float> buf0;        │
│       StructuredBuffer<float> buf1;          │
│                                              │
│    b. pointwise.py: compute loop body        │
│       loads:  float tmp0 = buf0[gid.x];     │
│       compute: float tmp1 = tmp0 + alpha;    │
│       stores:  buf1[gid.x] = tmp1;          │
│                                              │
│    c. expr_printer.py: SymPy→Slang          │
│       floor_div(x, 4) → (x/4)               │
│       modular_indexing(x,4,2) → (x/4)%2     │
│                                              │
│    d. slang_helpers.py: emit shared helpers  │
│       _vk_f16_to_f32(), _vk_f32_to_f16()    │
│                                              │
│    OUTPUT: Slang compute shader source       │
└──────────────────┬──────────────────────────┘
                   │ Slang source string
                   ▼
┌─────────────────────────────────────────────┐
│ 4. WRAPPER (wrapper.py)                     │
│    def call(args):                           │
│      buf0 = empty_strided_vulkan(...)       │
│      _vk_make_kernel(src0, name0)(          │
│        buf0, buf1, alpha, ...               │
│      )                                       │
└──────────────────┬──────────────────────────┘
                   │ Python wrapper source
                   ▼
┌─────────────────────────────────────────────┐
│ 5. RUNTIME (runtime.py)                     │
│    make_vulkan_kernel(source, name):         │
│      hash = sha256(source)                   │
│      if hash not in cache:                   │
│        slangc source.slang -target spirv     │
│        → source.spv                          │
│      return SlangKernel(hash, spirv)         │
│                                              │
│    SlangKernel.__call__(*buffers, *args):    │
│      _C._jit_dispatch(spirv, bindings)       │
└──────────────────┬──────────────────────────┘
                   │ Vulkan compute dispatch
                   ▼
              ┌──────────┐
              │   GPU    │
              │ (SPIR-V) │
              └──────────┘
```

### 6.2. Template Path (Matmul / Conv / Attention)

Heavy ops use **Jinja-templated Slang shaders** with autotuned tile sizes:

```
aten.mm(A, B)          ← FX node
        │
        ▼
┌───────────────────────────────────────────┐
│ 1. LOWERING                               │
│    matched by pattern → template call      │
│    → ExternKernel (SlangTemplateCaller)    │
└──────────────────┬────────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────────┐
│ 2. TEMPLATE EXPANSION (vulkan_template.py)│
│    _load_slang_template("slang_mm")        │
│    → unwrap .slang-embedded Jinja markers  │
│    → Jinja2.render(                        │
│        TILE_M=64, TILE_N=64, TILE_K=32,   │
│        DTYPE=float, EPILOGUE=relu, ...     │
│      )                                     │
│    → Slang compute shader source           │
└──────────────────┬────────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────────┐
│ 3. AUTOTUNE (if max-autotune)             │
│    Try tile configurations on GPU          │
│    → Pick fastest (TILE_M, TILE_N, TILE_K)│
└──────────────────┬────────────────────────┘
                   │
                   ▼
          [Same runtime path as IR-codegen]
```

### 6.3. Backward Differentiation Path

Two mechanisms generate backward kernels:

**A. Slang `[BackwardDerivative]` (preferred):**
```slang
// shaders/lib/pointwise.slang
[Differentiable]
[BackwardDerivative(relu_fast_bwd)]
public float relu_fwd(float x) { return max(x, 0.0f); }

void relu_fast_bwd(inout DifferentialPair<float> x, float dout) {
    float xv = x.p;
    x = diffPair(xv, xv > 0.0f ? dout : 0.0f);
}
```
The forward kernel `import`s the library function; `slangc` auto-generates
the backward through Slang's built-in autodiff system.

**B. `bwd_diff_dispatch` (fallback for ops without `[BackwardDerivative]`):**
```python
# lowerings/bwd_diff.py
_UNARY_BWD_DIFF_OPS = {
    aten.silu_backward: "silu_bwd",
    ...
}
# Routes backward ops to decomposed pointwise primitives
```

---

## 7. Key File Map

### Core Inductor (upstream)

| File | Role |
|------|------|
| `torch/_inductor/compile_fx.py` | Entry: `compile_fx()` → `_compile_fx_inner()` → pipeline orchestration |
| `torch/_inductor/graph.py` | `GraphLowering` — FX interpreter, lowering dispatch, `codegen()`, `compile_to_module()` |
| `torch/_inductor/lowering.py` | `register_lowering()`, `lowerings` dict, `fallback_handler` |
| `torch/_inductor/scheduler.py` | `Scheduler`, `BaseScheduling`, fusion groups |
| `torch/_inductor/ir.py` | IR nodes: `TensorBox`, `Pointwise`, `Reduction`, `ExternKernel`, `Buffer` |
| `torch/_inductor/codegen/common.py` | `Kernel`, `SIMDKernel`, `OpOverrides`, `DeviceOpOverrides`, `WrapperCodegen`, backend registry |
| `torch/_inductor/codegen/simd.py` | `SIMDKernel` base class (loop generation, index simplification) |
| `torch/_inductor/codegen/wrapper.py` | `PythonWrapperCodegen` — host-code orchestration |
| `torch/_inductor/codegen/triton.py` | `TritonKernel` — Triton codegen (CUDA) |
| `torch/_inductor/codegen/cpp.py` | `CppKernel` — C++/vectorized codegen (CPU) |
| `torch/_inductor/fx_passes/pre_grad.py` | Pre-grad pattern matching, fusion |
| `torch/_inductor/fx_passes/post_grad.py` | Post-grad cleanup, layout opts |
| `torch/_inductor/pattern_matcher.py` | Subgraph pattern matching engine |

### Vulkan/Slang Backend (out-of-tree)

| File | Role |
|------|------|
| `backends/vulkan_slang/python/torch_vulkan/__init__.py` | Package init, `SLANGC` discovery, backend import trigger |
| `backends/vulkan_slang/python/torch_vulkan/inductor/__init__.py` | **`_legacy_register()`** — installs all 4 integration hooks |
| `backends/vulkan_slang/python/torch_vulkan/inductor/backend.py` | `SlangVulkanBackend` façade class |
| `backends/vulkan_slang/python/torch_vulkan/inductor/lowerings/__init__.py` | Vulkan-specific op lowerings (layer_norm, softmax, bwd activations) |
| `backends/vulkan_slang/python/torch_vulkan/inductor/scheduling.py` | `VulkanScheduling` — fusion heuristics, backend features |
| `backends/vulkan_slang/python/torch_vulkan/inductor/kernel/main.py` | `VulkanKernel(SIMDKernel)` + `VulkanCSE` |
| `backends/vulkan_slang/python/torch_vulkan/inductor/kernel/header.py` | `HeaderMixin` — shader emission, buffer decls, dispatch grid |
| `backends/vulkan_slang/python/torch_vulkan/inductor/kernel/pointwise.py` | `PointwiseMixin` — load/compute/store codegen |
| `backends/vulkan_slang/python/torch_vulkan/inductor/kernel/reduction.py` | `ReductionMixin` — 1D/2D reduction, welford, scan, sort |
| `backends/vulkan_slang/python/torch_vulkan/inductor/expr_printer.py` | `VulkanExprPrinter` — SymPy → Slang |
| `backends/vulkan_slang/python/torch_vulkan/inductor/overrides.py` | `VulkanOverrides(OpOverrides)` — op→Slang mapping, dtypes |
| `backends/vulkan_slang/python/torch_vulkan/inductor/slang_helpers.py` | `emit_helpers()` — f16/bf16 pack/unpack, barrier helpers |
| `backends/vulkan_slang/python/torch_vulkan/inductor/vulkan_template.py` | `VulkanTemplateKernel`, `SlangTemplate`, Jinja unwrapper |
| `backends/vulkan_slang/python/torch_vulkan/inductor/vulkan_template_caller.py` | Template installers (mm, rng, flash_attention, optimizer) |
| `backends/vulkan_slang/python/torch_vulkan/inductor/vulkan_combo_kernel.py` | `VulkanComboKernel` — FOREACH multi-kernel merger |
| `backends/vulkan_slang/python/torch_vulkan/inductor/wrapper.py` | `VulkanPythonWrapperCodegen` — host code + buffer pool |
| `backends/vulkan_slang/python/torch_vulkan/inductor/runtime.py` | `make_vulkan_kernel()` — `slangc` → SPIR-V → dispatch |
| `backends/vulkan_slang/python/torch_vulkan/inductor/buffer_pool.py` | Pool-aware buffer lifecycle |
| `backends/vulkan_slang/python/torch_vulkan/inductor/device_op_overrides.py` | `VulkanDeviceOpOverrides` — single-device stubs |
| `backends/vulkan_slang/shaders/lib/*.slang` | Precompiled Slang library modules |
| `backends/vulkan_slang/python/torch_vulkan/inductor/templates/*` | Jinja/.slang templates for heavy ops |

### C++ Runtime

| File | Role |
|------|------|
| `backends/vulkan_slang/csrc/init.cpp` | Pybind module init |
| `backends/vulkan_slang/csrc/ops/` | Hand-written C++ op dispatchers (eager fallback) |
| `backends/vulkan_slang/csrc/vulkan/` | Vulkan runtime: device, allocator, command submission |
| `backends/vulkan_slang/csrc/backend/` | Inductor-backend specific C++ helpers |
| `backends/vulkan_slang/csrc/autocast/` | AMP autocast support |

---

## Summary: The Complete Data Flow

```
   User: torch.compile(model, backend="inductor")
                │
    ┌───────────▼───────────┐
    │  Dynamo: trace → FX   │
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │  AOTAutograd: joint   │
    │  graph + partition    │
    └───────────┬───────────┘
                │
    ┌───────────▼───────────────────────────────────────────┐
    │  Inductor compile_fx_inner()                          │
    │                                                       │
    │  pre_grad_passes → post_grad_passes                   │
    │                                                       │
    │  ┌─────────────────────────────────────────────────┐  │
    │  │ GraphLowering.run()                             │  │
    │  │                                                 │  │
    │  │  FX node: aten.add(x, y)                        │  │
    │  │    │                                            │  │
    │  │    ├─ Lowering ──▶ Pointwise IR                 │  │
    │  │    ├─ Scheduler ─▶ Fused group {"kernel_0"}     │  │
    │  │    │                                            │  │
    │  │    ├─ Codegen ──▶ VulkanKernel.codegen_kernel() │  │
    │  │    │   ┌──────────────────────────────────────┐ │  │
    │  │    │   │  [numthreads(256,1,1)]                │ │  │
    │  │    │   │  RWStructuredBuffer<float> buf0;       │ │  │
    │  │    │   │  StructuredBuffer<float> buf1;         │ │  │
    │  │    │   │  float tmp0 = buf1[gid.x];             │ │  │
    │  │    │   │  buf0[gid.x] = tmp0 + 1.0f;           │ │  │
    │  │    │   └──────────────────────────────────────┘ │  │
    │  │    │     ↑ Slang compute shader source          │  │
    │  │    │                                            │  │
    │  │    └─ Wrapper ─▶ VulkanPythonWrapperCodegen     │  │
    │  │         ┌──────────────────────────────────────┐ │  │
    │  │         │  buf0 = empty_strided_vulkan(...)     │ │  │
    │  │         │  _vk_make_kernel(src0, name0)(args)  │ │  │
    │  │         └──────────────────────────────────────┘ │  │
    │  └─────────────────────────────────────────────────┘  │
    │                                                       │
    │  compile_to_module() → PyCodeCache → callable          │
    └───────────────────────┬───────────────────────────────┘
                            │
    ┌───────────────────────▼───────────────────────────┐
    │  At Runtime (first call):                         │
    │    make_vulkan_kernel(slang_source, "kernel_0")    │
    │      → slangc -target spirv kernel_0.slang        │
    │      → SPIR-V binary cached on disk               │
    │      → _C._jit_dispatch(spirv, buffers)           │
    │        → vkQueueSubmit → GPU executes SPIR-V      │
    └───────────────────────────────────────────────────┘
```
