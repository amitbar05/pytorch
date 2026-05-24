# Slang Shaders & Compilation Pipeline

## Why Slang as the Sole Shader Language

All GPU shaders in this project are written in **Slang** (https://shader-slang.org/). No GLSL. Reasons:

1. **Autodiff eliminates hand-written backward shaders.** `[Differentiable]` + `bwd_diff()` auto-generates backward-mode derivative propagation at compile time. This cuts backward-pass work by 50-70%.
2. **Generics everywhere.** `<T : IFloat>` covers f32 and f16 for every kernel — including GEMM and attention — without preprocessor hacks or specialization-constant gymnastics.
3. **Full compute shader capabilities.** `groupshared` memory, `GroupMemoryBarrierWithGroupSync()`, workgroup dispatch, subgroup ops — everything needed for tiled GEMM and flash attention. Compiles to the same SPIR-V as GLSL.
4. **Modular compilation.** `import` instead of `#include` pasting. Separate compilation, linking, incremental builds.
5. **Multi-target.** Same source → SPIR-V (Vulkan), CUDA, MSL (Metal), **and C++ (CPU)**. The CPU target is critical for the dev workflow (see `05-cpu-testing.md`).
6. **Production-proven.** Valve ships Counter-Strike 2 and Dota 2 through Slang's SPIR-V codegen. Khronos hosts the project. It's in the Vulkan SDK.
7. **Best-in-class tooling.** VS Code LSP with IntelliSense, RenderDoc debugging, meaningful compile errors at definition site.

For porting GLSL reference code (e.g., llama.cpp's `mul_mm.comp`): Slang provides a GLSL compatibility module and the syntax is close enough that porting is mechanical.

---

## Shader Directory Layout

```
shaders/
├── lib/                            # Library modules (SSOT for Inductor codegen)
│   ├── helpers.slang               # Math extensions, Welford, Philox, type-packing
│   ├── pointwise.slang             # Activations, IPointwise interface, operator structs
│   ├── reduction.slang             # Workgroup reductions, IWaveReduction interface
│   ├── norm.slang                  # Layer/RMS norm element functions, INormAffine
│   ├── losses.slang                # MSE/l1/smooth_l1/huber/bce/kl_div elementals
│   ├── mm.slang                    # Tiled GEMM with epilogue interface
│   ├── conv.slang                  # Direct convolution with bias interface
│   ├── atomics.slang               # GPU atomics (f32, i32, u32, packed f16)
│   └── tensor_layout.slang         # N-dim offset/unravel/ravel/broadcast helpers
├── unary/                          # Unary element-wise (all [Differentiable])
│   ├── neg.slang, exp.slang, log.slang, sqrt.slang, rsqrt.slang
│   ├── abs.slang, sign.slang, ceil_floor.slang
│   └── cast.slang                 # dtype conversion (non-differentiable)
├── binary/                        # Binary element-wise with broadcasting
│   ├── add.slang, mul.slang, sub.slang, div.slang, pow.slang
├── activation/                    # Activation functions (all [Differentiable])
│   ├── relu.slang, gelu.slang, silu.slang, sigmoid.slang, tanh.slang
│   └── softmax.slang              # Fused numerically-stable softmax
├── matmul/                        # Matrix multiplication
│   ├── mm_naive.slang             # Baseline (correctness reference)
│   ├── mm_tiled.slang             # Shared-memory tiled GEMM (BM=64,BN=64,BK=16)
│   ├── mm_coopmat.slang           # KHR_cooperative_matrix GEMM
│   └── mm_splitk.slang            # Split-K for tall-skinny matrices
├── conv/                          # Convolution shaders
│   ├── im2col.slang, col2im.slang
│   └── conv2d_direct.slang        # Direct conv for small kernels (3×3, 1×1)
├── attention/                     # Flash-attention style shaders
│   ├── flash_attn_fwd.slang
│   └── flash_attn_bwd.slang       # Recomputation-based backward
├── norm/                          # Normalization (all [Differentiable])
│   ├── layer_norm.slang, batch_norm.slang, group_norm.slang
├── loss/                          # Loss functions ([Differentiable])
│   ├── mse_loss.slang, nll_loss.slang, cross_entropy.slang
├── reduce/                        # Reductions
│   ├── sum.slang, mean.slang, max_min.slang, argmax.slang
├── index/                         # Indexing & scatter/gather
│   ├── gather.slang, scatter.slang, index_select.slang, embedding.slang
├── compare/                       # Comparison ops (non-differentiable)
│   ├── cmp_ops.slang, where.slang
├── pooling/                       # Pooling ops
│   ├── max_pool2d.slang, avg_pool2d.slang, adaptive_avg_pool2d.slang
├── random/                        # RNG shaders (non-differentiable)
│   ├── philox.slang, dropout.slang
└── copy/                          # Memory operations
    ├── copy.slang, fill.slang, contiguous.slang
```

---

## Slang Patterns

### Pattern 1: Element-wise with Autodiff (majority of ops)

```slang
// shaders/activation/gelu.slang
import common.types;

[Differentiable]
float gelu_exact(float x) {
    let k = 0.7978845608f;
    let c = 0.044715f;
    let inner = k * (x + c * x * x * x);
    return 0.5f * x * (1.0f + tanh(inner));
}

[shader("compute")] [numthreads(256, 1, 1)]
void computeMain(
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform uint numel,
    uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= numel) return;
    output[tid.x] = gelu_exact(input[tid.x]);
}

// Backward entry — compile_shaders.py auto-generates this wrapper
[shader("compute")] [numthreads(256, 1, 1)]
void bwd_computeMain(
    uniform StructuredBuffer<float> input,
    uniform StructuredBuffer<float> grad_output,
    uniform RWStructuredBuffer<float> grad_input,
    uniform uint numel,
    uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= numel) return;
    DifferentialPair<float> dp = diffPair(input[tid.x], 0.0f);
    bwd_diff(gelu_exact)(dp, grad_output[tid.x]);
    grad_input[tid.x] = dp.getDifferential();
}
```

### Pattern 2: Tiled GEMM with Groupshared Memory (performance-critical)

```slang
// shaders/matmul/mm_tiled.slang
static const uint BM = 64;
static const uint BN = 64;
static const uint BK = 16;
static const uint TM = 8;
static const uint TN = 8;

groupshared float smem_a[BM * BK];  // Slang groupshared = GLSL shared
groupshared float smem_b[BK * BN];

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(
    uniform StructuredBuffer<float> A,   // [M, K]
    uniform StructuredBuffer<float> B,   // [K, N]
    uniform RWStructuredBuffer<float> C, // [M, N]
    uniform uint M, uniform uint N, uniform uint K,
    uint3 gid : SV_GroupID,
    uint3 lid : SV_GroupThreadID,
    uint tidx : SV_GroupIndex)
{
    uint block_row = gid.x;
    uint block_col = gid.y;

    float reg_c[TM * TN];
    for (uint i = 0; i < TM * TN; i++) reg_c[i] = 0.0f;

    for (uint bk = 0; bk < K; bk += BK)
    {
        // Cooperative load A tile into smem_a
        for (uint i = tidx; i < BM * BK; i += 256) {
            uint row = block_row * BM + i / BK;
            uint col = bk + i % BK;
            smem_a[i] = (row < M && col < K) ? A[row * K + col] : 0.0f;
        }
        // Cooperative load B tile into smem_b
        for (uint i = tidx; i < BK * BN; i += 256) {
            uint row = bk + i / BN;
            uint col = block_col * BN + i % BN;
            smem_b[i] = (row < K && col < N) ? B[row * N + col] : 0.0f;
        }

        GroupMemoryBarrierWithGroupSync();

        uint thread_row = (tidx / (BN / TN)) * TM;
        uint thread_col = (tidx % (BN / TN)) * TN;

        for (uint k = 0; k < BK; k++) {
            for (uint tm = 0; tm < TM; tm++) {
                float a_val = smem_a[(thread_row + tm) * BK + k];
                for (uint tn = 0; tn < TN; tn++) {
                    reg_c[tm * TN + tn] += a_val * smem_b[k * BN + thread_col + tn];
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // Write results
    for (uint tm = 0; tm < TM; tm++) {
        for (uint tn = 0; tn < TN; tn++) {
            uint row = block_row * BM + (tidx / (BN / TN)) * TM + tm;
            uint col = block_col * BN + (tidx % (BN / TN)) * TN + tn;
            if (row < M && col < N)
                C[row * N + col] = reg_c[tm * TN + tn];
        }
    }
}
```

### Pattern 3: Generic Dtype via Slang Generics

```slang
// shaders/binary/add.slang
[Differentiable]
T typed_add<T : IFloat>(T a, T b) { return a + b; }

// f32 variant
[shader("compute")] [numthreads(256, 1, 1)]
void add_f32(uniform StructuredBuffer<float> a, uniform StructuredBuffer<float> b,
             uniform RWStructuredBuffer<float> out, uniform uint numel,
             uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= numel) return;
    out[tid.x] = typed_add<float>(a[tid.x], b[tid.x]);
}

// f16 variant — same logic, different type
[shader("compute")] [numthreads(256, 1, 1)]
void add_f16(uniform StructuredBuffer<half> a, uniform StructuredBuffer<half> b,
             uniform RWStructuredBuffer<half> out, uniform uint numel,
             uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= numel) return;
    out[tid.x] = typed_add<half>(a[tid.x], b[tid.x]);
}
```

---

## Shader Compilation Pipeline

```
compile_shaders.py                     compile_cpu_tests.py
        │                                       │
        ▼                                       ▼
  For each .slang:                        For each .slang:
  ├─ slangc → fwd.spv (forward)          └─ slangc -target cpp → .cpp/.h
  ├─ slangc → bwd.spv (backward)            (CPU-runnable shader math)
  └─ embed as C++ byte arrays
     in csrc/generated/shaders.h         → cpu_tests/generated/
```

### Backward Entry Point Convention

`compile_shaders.py` auto-generates backward wrappers for every `.slang` with `[Differentiable]`. Shader authors only write forward logic. The script produces `{op}_fwd.spv` and `{op}_bwd.spv`.

### Implementation Tasks

- [x] `shaders/lib/*.slang` (9 modules): f32/f16/fp8 dtypes, generic interfaces, `[Differentiable]` annotations, `[BackwardDerivative]` on elementals
- [x] `compile_shaders.py`: batch compile, generate forward + backward SPIR-V + C++ test targets
- [x] `compile_cpu_tests.py`: generates C++ from Slang for CPU-side unit tests
- [x] `tools/lib_graph.py`: library-module dependency graph + dead-shader detector

**Testing gate:**
- [x] `slangc` compiles a trivial compute shader to valid SPIR-V
- [x] `slangc -target cpp` produces compilable C++ from the same source
- [x] Backward entry generation works for `[Differentiable]` functions
- [x] Slang SPIR-V loads and executes correctly on SwiftShader
- [x] CPU-compiled Slang functions produce same outputs as SPIR-V path
- [x] Pin Slang version, documented in `tools/slang_version.txt`

---

## Slang Autodiff Integration with PyTorch Autograd

PyTorch autograd and Slang autodiff are complementary layers:

- **PyTorch autograd** = op-level graph. Chains gradients across operations.
- **Slang autodiff** = shader-level. Generates the math *inside* each backward kernel.

```
PyTorch autograd graph:
    x → [relu] → [mm] → [softmax] → [cross_entropy] → loss
                                                          │
    loss.backward() walks the graph:                      ▼
    ┌──────────────────────────────────────────────────────┐
    │ cross_entropy_bwd  ← Slang bwd_diff (auto)          │
    │ softmax_bwd        ← Slang bwd_diff (auto)          │
    │ mm_bwd             ← PyTorch autograd (decomposes    │
    │                       to mm + transpose, reuses fwd) │
    │ relu_bwd           ← Slang bwd_diff (auto)          │
    └──────────────────────────────────────────────────────┘
```

### Three Tiers of Backward Support

**Tier 1 — Slang autodiff (majority):** Forward is `[Differentiable]` → backward SPIR-V auto-generated at build time. Covers: all unary, binary, activations, norms, losses, softmax, avg_pool2d, reductions.

**Tier 2 — PyTorch autograd decomposition:** PyTorch decomposes backward into forward ops. E.g., `mm` backward = two more `mm` calls with transposed inputs. No backward shader needed.

**Tier 3 — Hand-written Slang backward:** For complex ops where Slang autodiff can't handle global memory patterns. Use `[BackwardDerivative(custom_fn)]`. Covers: flash attention (recomputation-based), batch_norm training mode, embedding_backward (scatter_add).

---

## Reference Materials

### Library Module API Reference (Canonical Signatures)

These 9 modules in `shaders/lib/` form the Slang vocabulary that all
Inductor-codegen'd kernels and templates consume. **Every public symbol below
is a contract.** Breaking a signature requires a backward-compatible fallback
or a synchronized codegen update.

#### `helpers.slang` — Foundational Primitives

| Symbol | Signature | Role |
|--------|-----------|------|
| `Welford` (struct) | `{float mean, m2, n}` | Running mean/variance accumulator |
| `welford_combine` | `(Welford a, Welford b) → Welford` | Merge two Welford states |
| `welford_finalize` | `(Welford w) → float2(mean, var)` | Finalize mean + population variance |
| `f16_to_f32` / `f32_to_f16` | `(uint) → float` / `(float) → uint` | IEEE half ↔ float |
| `bf16_to_f32` / `f32_to_bf16` | `(uint) → float` / `(float) → uint` | BFloat16 ↔ float |
| `unpack_f16x2` / `pack_f16x2` | `(uint) → float2` / `(float,float) → uint` | Bulk pack/unpack |
| `philox_rand` / `philox_randn` | `(uint counter, uint seed) → float` | Philox RNG (uniform / gaussian) |
| `wave_sum<T>` / `wave_max<T>` etc. | `(T v) → T` where `T : __BuiltinFloatingPointType` | Wave-intrinsic wrappers |
| `wg_sum_smem` | `(float v, uint tid, uint size) → float` | Tree reduction for legacy paths |
| Extension `float.erf()` | `float → float`, `[ForceInline]` | Error function |
| Extension `float.log1p()` | `float → float`, `[ForceInline]` | log(1+x) |
| Extension `float.expm1()` | `float → float`, `[ForceInline]` | exp(x)-1 |
| Extension `float.digamma()` | `float → float`, `[ForceInline]` | Digamma function |
| Extension `float.lgamma()` | `float → float`, `[ForceInline]` | Log-gamma |
| `_vk_unpack_{u8,i8,i16}` | `(StructuredBuffer<uint>, uint idx) → float` | Sub-32-bit dtype unpack |

#### `pointwise.slang` — Elementwise Primitives

| Symbol | Signature | Attributes |
|--------|-----------|------------|
| `interface IPointwise` | `static float apply(float x)` | — |
| `interface IPointwiseBinary` | `static float apply(float a, float b)` | — |
| `relu_fwd` / `elu_fwd` / `hardswish_fwd` / `hardsigmoid_fwd` / `gelu_fwd` / `silu_fwd` / `softplus_fwd` / `mish_fwd` | `(float x) → float` | `[Differentiable]`, `[BackwardDerivative(...)]` |
| `sigmoid_fwd` / `tanh_fwd` | `(float x) → float` | `[Differentiable]`, `[BackwardDerivative(...)]` |
| `OpReLU` / `OpGELU` … (18 structs) | `: IPointwise` | Operator structs for template dispatch |
| `OpAdd` / `OpMul` … (7 structs) | `: IPointwiseBinary` | Operator structs for binary template dispatch |
| `pointwise_unary_apply<Op>` | `(StructuredBuffer input, RWStructuredBuffer output, uint idx, uint n)` | `[ForceInline]` |
| `pointwise_binary_apply<Op>` | `(StructuredBuffer in_a, in_b, RWStructuredBuffer output, uint idx, uint n)` | `[ForceInline]` |

#### `reduction.slang` — Workgroup Reductions

| Symbol | Signature | Attributes |
|--------|-----------|------------|
| `interface IReduction` | `static float identity()`, `static float combine(float,float)` | — |
| `interface IWaveReduction : IReduction` | `static float wave_reduce(float v)` | Extends IReduction |
| `OpSum` / `OpProd` / `OpMaxReduce` / `OpMinReduce` | `: IReduction, IWaveReduction` | — |
| `OpAny` / `OpAll` / `OpXorSum` | `: IReduction` | — |
| `wg_reduce<R : IReduction>` | `(float v, uint tid, uint size) → float` | Tree reduction (always correct) |
| `wg_reduce_wave<W : IWaveReduction>` | `(float v, uint tid, uint n_waves, uint simd) → float` | Wave-intrinsic two-stage fast path |
| `wg_argmax` / `wg_argmin` | `(ArgPair v, uint tid, uint size) → ArgPair` | Arg reduction |

#### `norm.slang` — Normalization Primitives

| Symbol | Signature | Attributes |
|--------|-----------|------------|
| `interface INormAffine` | `static float apply(float normalized, uint i, StructuredBuffer weight, StructuredBuffer bias)` | — |
| `ln_affine_elem` / `ln_no_affine_elem` | `(float x, no_diff float mean, no_diff float rstd, [float w, float b]) → float` | `[Differentiable]`, `[BackwardDerivative]` |
| `rms_affine_elem` / `rms_no_affine_elem` | `(float x, no_diff float rstd, [float w]) → float` | `[Differentiable]`, `[BackwardDerivative]` |
| `layer_norm_row<Affine>` | `(StructuredBuffer X, weight, bias, RWStructuredBuffer Y, uint row, D, float eps, uint tid, tg)` | — |
| `rms_norm_row<Affine>` | `(StructuredBuffer X, weight, bias, RWStructuredBuffer Y, uint row, D, float eps, uint tid, tg)` | — |
| `wg_welford` | `(float3 v, uint tid, uint size) → float3` | Workgroup Welford combine |

#### `losses.slang` — Loss Element Functions

| Symbol | Signature | Attributes |
|--------|-----------|------------|
| `mse_elem` | `(float pred, float target) → float` | `[Differentiable]` |
| `l1_elem` | `(float pred, float target) → float` | `[Differentiable]` |
| `smooth_l1_elem` | `(float pred, float target, no_diff float beta) → float` | `[Differentiable]` |
| `huber_elem` | `(float pred, float target, no_diff float delta) → float` | `[Differentiable]` |
| `bce_elem` | `(float pred, float target) → float` | `[Differentiable]` |
| `bce_with_logits_elem` | `(float logit, float target) → float` | `[Differentiable]` |
| `kl_div_elem` | `(float log_pred, float target) → float` | `[Differentiable]` |

#### `mm.slang` — Matrix Multiplication

| Symbol | Signature | Attributes |
|--------|-----------|------------|
| `TILE_M` / `TILE_N` / `TILE_K` | `public const uint` | `[[vk::constant_id(N)]]` specialization constants |
| `interface IEpilogue` | `static float apply(float acc, uint m, uint n)` | — |
| `mm_tiled<Epi : IEpilogue>` | `(StructuredBuffer A, B, RWStructuredBuffer C, uint M, N, K, uint3 gid, uint3 lid)` | — |

#### `conv.slang` — Convolution

| Symbol | Signature | Attributes |
|--------|-----------|------------|
| `KH` … `DILATION_W` (8 consts) | `public const uint` | `[[vk::constant_id(N)]]` |
| `ConvShape` | `struct {uint N, Cin, H, W, Cout, OH, OW}` | — |
| `interface IConvBias` | `static float load(StructuredBuffer bias, uint cout)` | — |
| `conv2d_direct<Bias : IConvBias>` | `(StructuredBuffer X, W, bias, RWStructuredBuffer Y, ConvShape s, uint n, cout, oh, ow)` | — |

#### `atomics.slang` — GPU Atomics

| Symbol | Signature | Role |
|--------|-----------|------|
| `atomic_add_f32` | `(RWByteAddressBuffer buf, uint byte_offset, float value)` | 32-bit float CAS-locked add |
| `atomic_add_i32` / `atomic_add_u32` | `(RWByteAddressBuffer buf, uint byte_offset, int/uint value)` | 32-bit int/uint atomics |
| `atomic_add_f16_packed` | `(RWByteAddressBuffer buf, uint word_byte_offset, uint lane, float value)` | Packed f16 atomic add |
| `c10_vulkan_atomic_add` | `(RWStructuredBuffer<uint> buf, uint idx, float value)` | Legacy StructuredBuffer atomic add |

#### `tensor_layout.slang` — Index Arithmetic

| Symbol | Signature | Attributes |
|--------|-----------|------------|
| `contiguous_offset<N>` | `(uint linear, uint shape[N]) → uint` | `[ForceInline]` |
| `strided_offset<N>` | `(uint linear, uint shape[N], uint stride[N]) → uint` | `[ForceInline]` |
| `broadcast_offset<N>` | `(uint out_linear, uint out_shape[N], uint src_shape[N], uint src_stride[N]) → uint` | `[ForceInline]` |
| `unravel_index<N>` | `(uint linear, uint shape[N], out uint coords[N])` | — |
| `ravel_index<N>` | `(uint coords[N], uint stride[N]) → uint` | `[ForceInline]` |
| `numel<N>` | `(uint shape[N]) → uint` | `[ForceInline]` |

#### Conventions

- **`[ForceInline]`** — All small math helpers use this. Inline into call sites; no slangc code-gen object overhead.
- **`[Differentiable]`** — Activation/loss/norm element primitives. Enables `bwd_diff()` auto-backward generation.
- **`[BackwardDerivative(fn)]`** — Fast hand-written backward for hot-path ops (sigmoid, tanh, gelu, silu, norm elements). Replaces what auto-diff would generate with a `_fast_bwd` counterpart.
- **Interfaces** — `IEpilogue` (mm), `IConvBias` (conv), `INormAffine` (norm), `IPointwise`/`IPointwiseBinary` (pointwise), `IReduction`/`IWaveReduction` (reduction). Used for template dispatch.
- **`no_diff` parameters** — Mark hyperparameters (beta, delta, mean, rstd) that should not propagate gradients. Required by Slang's autodiff API.
- **Backward functions are private** — `sigmoid_fast_bwd`, `tanh_fast_bwd`, `ln_affine_elem_bwd`, etc., are NOT `public`. They exist only to satisfy `[BackwardDerivative(fn)]` linkage.

### Slang
- Homepage / GitHub: https://shader-slang.org/ / https://github.com/shader-slang/slang
- Autodiff Guide: https://shader-slang.org/slang/user-guide/autodiff
- Autodiff Tutorial: https://docs.shader-slang.org/en/stable/auto-diff-tutorial-1.html
- Generics: https://shader-slang.org/slang/user-guide/interfaces-generics
- Coming from GLSL: https://shader-slang.org/slang/user-guide/coming-from-glsl
- Cooperative Vectors: https://shader-slang.org/blog/2025/01/30/coop-vec-available/

### GEMM Optimization
- CUDA GEMM Worklog (concepts transfer): https://siboehm.com/articles/22/CUDA-MMM
- Cooperative Matrix: https://developer.nvidia.com/blog/machine-learning-acceleration-vulkan-cooperative-matrices/

### Existing Implementations (Study & Port)
- **llama.cpp ggml-vulkan:** https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-vulkan — Study `mul_mm.comp` GEMM tiling, port to Slang
- **Sascha Willems' Vulkan Samples (Slang):** 170 Slang shaders for Vulkan — excellent reference
