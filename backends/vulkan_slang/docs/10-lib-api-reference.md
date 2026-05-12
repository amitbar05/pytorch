# 10 — Lib API Reference (Canonical Slang Signatures)

> **Track 2.1.** One canonical signature per primitive across the 9
> `shaders/lib/*.slang` modules. This document is the contract — every lib
> function documented here must keep its signature stable. Breaking changes
> require a version bump in the CI gate.

**Last verified:** 2026-05-03
**`[BackwardDerivative]` count:** 14 (10 pointwise + 4 norm)

---

## `atomics.slang`

Atomic floating-point operations for scatter/gather patterns.

| Function | Signature |
|----------|-----------|
| `c10_vulkan_atomic_add` | `[ForceInline] void c10_vulkan_atomic_add(RWStructuredBuffer<uint> buf, uint idx, float value)` |
| `atomic_add_f32` | `[ForceInline] void atomic_add_f32(RWByteAddressBuffer buf, uint byte_offset, float value)` |
| `atomic_add_i32` | `[ForceInline] void atomic_add_i32(RWByteAddressBuffer buf, uint byte_offset, int value)` |
| `atomic_add_u32` | `[ForceInline] void atomic_add_u32(RWByteAddressBuffer buf, uint byte_offset, uint value)` |
| `atomic_add_f16_packed` | `[ForceInline] void atomic_add_f16_packed(RWByteAddressBuffer buf, uint word_byte_offset, uint lane, float value)` |

---

## `conv.slang`

Direct convolution with specialization-constant kernel geometry.

### Specialization Constants
| Name | Declaration |
|------|-------------|
| `KH` — `KW` — `STRIDE_H` — `STRIDE_W` — `PAD_H` — `PAD_W` — `DILATION_H` — `DILATION_W` | `[[vk::constant_id(N)]]` with defaults |

### Struct `ConvShape`
| Field | Type |
|-------|------|
| `N`, `Cin`, `H`, `W`, `Cout`, `OH`, `OW` | `uint` |

### Interface `IConvBias`
| Method | Signature |
|--------|-----------|
| `load` | `static float load(StructuredBuffer<float> bias, uint cout)` |

### Structs implementing `IConvBias`
| Struct | `load()` returns |
|--------|------------------|
| `BiasNone` | `0.0f` |
| `BiasFromBuffer` | `bias[cout]` |

### Public Function
| Function | Signature |
|----------|-----------|
| `conv2d_direct<Bias : IConvBias>` | `void conv2d_direct(StructuredBuffer<float> X, StructuredBuffer<float> W, StructuredBuffer<float> bias, RWStructuredBuffer<float> Y, ConvShape s, uint n, uint cout, uint oh, uint ow)` |

---

## `helpers.slang`

Dtype conversion, Welford, Philox RNG, wave intrinsics, extension methods.

### Struct `Welford`
| Field | Type |
|-------|------|
| `mean`, `m2`, `n` | `float` |

### Welford Functions
| Function | Signature |
|----------|-----------|
| `welford_combine` | `Welford welford_combine(Welford a, Welford b)` |
| `welford_finalize` | `float2 welford_finalize(Welford w)` |

### f16/bf16 Conversion
| Function | Signature |
|----------|-----------|
| `f16_to_f32` / `f32_to_f16` | `[ForceInline] float f16_to_f32(uint h)` / `uint f32_to_f16(float f)` |
| `bf16_to_f32` / `f32_to_bf16` | same pattern for bf16 |
| `unpack_f16x2` / `pack_f16x2` | `float2 unpack_f16x2(uint word)` / `uint pack_f16x2(float lo, float hi)` |
| `unpack_bf16x2` / `pack_bf16x2` | same pattern for bf16 |
| `_vk_unpack_f16` / `_vk_pack_f16` | `float _vk_unpack_f16(uint word, uint lane)` / `uint _vk_pack_f16(float lo, float hi)` |
| `_vk_unpack_bf16` / `_vk_pack_bf16` | same pattern for bf16 |
| `_vk_unpack_f16_2d` / `_vk_pack_f16_2d` | 2D row-col variants |
| `_vk_unpack_bf16_2d` / `_vk_pack_bf16_2d` | same pattern |
| `_vk_unpack_u8` / `_vk_unpack_i8` / `_vk_unpack_i16` | Sub-32-bit dtype unpack from `StructuredBuffer<uint>` |

### Philox RNG
| Function | Signature |
|----------|-----------|
| `philox_mulhi32` | `[ForceInline] uint philox_mulhi32(uint a, uint b)` |
| `philox_round` | `[ForceInline] uint2 philox_round(uint2 ctr, uint2 key)` |
| `philox_bumpkey` | `[ForceInline] uint2 philox_bumpkey(uint2 key)` |
| `philox_rand` | `[ForceInline] float philox_rand(uint counter, uint seed)` |
| `philox_randn` | `[ForceInline] float philox_randn(uint counter, uint seed)` |

### Wave Intrinsics (generic)
| Function | Signature |
|----------|-----------|
| `wave_sum<T : __BuiltinFloatingPointType>` | `T wave_sum(T v)` |
| `wave_max<T>` / `wave_min<T>` / `wave_prod<T>` | same pattern |
| `wg_sum_smem` | `float wg_sum_smem(float v, uint tid, uint size)` |

### `extension float` Methods
| Method | Signature |
|--------|-----------|
| `erf()`, `log1p()`, `expm1()`, `digamma()`, `lgamma()` | `[ForceInline] float` |

### Free-Function Aliases
`c10_vulkan_erf`, `c10_vulkan_log1p`, `c10_vulkan_expm1`, `c10_vulkan_hypot`, `c10_vulkan_digamma`, `c10_vulkan_lgamma`

---

## `losses.slang`

All 7 loss elementals are `[Differentiable]` (auto-diff, no custom `[BackwardDerivative]`).

| Function | Signature |
|----------|-----------|
| `mse_elem` | `[Differentiable] float mse_elem(float pred, float target)` |
| `l1_elem` | `[Differentiable] float l1_elem(float pred, float target)` |
| `smooth_l1_elem` | `[Differentiable] float smooth_l1_elem(float pred, float target, no_diff float beta)` |
| `huber_elem` | `[Differentiable] float huber_elem(float pred, float target, no_diff float delta)` |
| `bce_elem` | `[Differentiable] float bce_elem(float pred, float target)` |
| `bce_with_logits_elem` | `[Differentiable] float bce_with_logits_elem(float logit, float target)` |
| `kl_div_elem` | `[Differentiable] float kl_div_elem(float log_pred, float target)` |

**`[BackwardDerivative]` annotations:** 0 (auto-diff sufficient for scalar elementals)

---

## `mm.slang`

Tiled matrix multiply with epilogue interface.

### Specialization Constants
`TILE_M = 16`, `TILE_N = 16`, `TILE_K = 16` (all `[[vk::constant_id(N)]]`)

### Interface `IEpilogue`
| Method | Signature |
|--------|-----------|
| `apply` | `static float apply(float acc, uint m, uint n)` |

### Structs implementing `IEpilogue`
| Struct | `apply()` |
|--------|-----------|
| `EpilogueIdentity` | `acc` |
| `EpilogueReLU` | `max(acc, 0.0f)` |
| `EpilogueClampPositive` | `clamp(acc, 0.0f, 6.0f)` |

### Public Function
| Function | Signature |
|----------|-----------|
| `mm_tiled<Epi : IEpilogue>` | `void mm_tiled(StructuredBuffer<float> A, StructuredBuffer<float> B, RWStructuredBuffer<float> C, uint M, uint N, uint K, uint3 gid, uint3 lid)` |

**`[BackwardDerivative]` annotations:** 0 (void functions with barriers — defer to Track 4 template codegen)

---

## `norm.slang`

Layer norm and RMS norm with generic affine policy. **4 `[BackwardDerivative]` annotations.**

### `[BackwardDerivative]` Annotations
| # | Forward | Backward |
|---|---------|----------|
| 1 | `[Differentiable] [BackwardDerivative(ln_affine_elem_bwd)] float ln_affine_elem(float x, no_diff float mean, no_diff float rstd, float w, float b)` | `void ln_affine_elem_bwd(inout DifferentialPair<float> x, no_diff float mean, no_diff float rstd, inout DifferentialPair<float> w, inout DifferentialPair<float> b, float dout)` |
| 2 | `[Differentiable] [BackwardDerivative(ln_no_affine_elem_bwd)] float ln_no_affine_elem(float x, no_diff float mean, no_diff float rstd)` | same pattern |
| 3 | `[Differentiable] [BackwardDerivative(rms_affine_elem_bwd)] float rms_affine_elem(float x, no_diff float rstd, float w)` | same pattern |
| 4 | `[Differentiable] [BackwardDerivative(rms_no_affine_elem_bwd)] float rms_no_affine_elem(float x, no_diff float rstd)` | same pattern |

### Interface `INormAffine`
| Method | Signature |
|--------|-----------|
| `apply` | `static float apply(float normalized, uint i, StructuredBuffer<float> weight, StructuredBuffer<float> bias)` |

### Structs implementing `INormAffine`
| Struct | `apply()` |
|--------|-----------|
| `AffineNone` | `n` |
| `AffineWeightOnly` | `n * w[i]` |
| `AffineWeightBias` | `n * w[i] + b[i]` |

### Public Functions
| Function | Signature |
|----------|-----------|
| `wg_welford` | `float3 wg_welford(float3 v, uint tid, uint size)` |
| `layer_norm_row<Affine : INormAffine>` | `void layer_norm_row(StructuredBuffer<float> X, StructuredBuffer<float> weight, StructuredBuffer<float> bias, RWStructuredBuffer<float> Y, uint row, uint D, float eps, uint tid, uint tg)` |
| `rms_norm_row<Affine : INormAffine>` | `void rms_norm_row(StructuredBuffer<float> X, StructuredBuffer<float> weight, StructuredBuffer<float> bias, RWStructuredBuffer<float> Y, uint row, uint D, float eps, uint tid, uint tg)` |

---

## `pointwise.slang`

The canonical pointwise dispatcher. **10 `[BackwardDerivative]` annotations.**

### `[BackwardDerivative]` Annotations (all 10)

| # | Forward | Backward Function |
|---|---------|-------------------|
| 1 | `[Differentiable] [BackwardDerivative(relu_fast_bwd)] float relu_fwd(float x)` | `void relu_fast_bwd(inout DifferentialPair<float> x, float dout)` |
| 2 | `[Differentiable] [BackwardDerivative(sigmoid_fast_bwd)] float sigmoid_fwd(float x)` | same pattern |
| 3 | `[Differentiable] [BackwardDerivative(tanh_fast_bwd)] float tanh_fwd(float x)` | same pattern |
| 4 | `[Differentiable] [BackwardDerivative(gelu_fast_bwd)] float gelu_fwd(float x)` | same pattern |
| 5 | `[Differentiable] [BackwardDerivative(silu_fast_bwd)] float silu_fwd(float x)` | same pattern |
| 6 | `[Differentiable] [BackwardDerivative(elu_fast_bwd)] float elu_fwd(float x)` | same pattern |
| 7 | `[Differentiable] [BackwardDerivative(hardswish_fast_bwd)] float hardswish_fwd(float x)` | same pattern |
| 8 | `[Differentiable] [BackwardDerivative(hardsigmoid_fast_bwd)] float hardsigmoid_fwd(float x)` | same pattern |
| 9 | `[Differentiable] [BackwardDerivative(softplus_fast_bwd)] float softplus_fwd(float x)` | same pattern |
| 10 | `[Differentiable] [BackwardDerivative(mish_fast_bwd)] float mish_fwd(float x)` | same pattern |

### Interfaces
| Interface | Method |
|-----------|--------|
| `IPointwise` | `static float apply(float x)` |
| `IPointwiseBinary` | `static float apply(float a, float b)` |

### `IPointwise` Structs (18)
`OpReLU`, `OpSigmoid`, `OpTanh`, `OpGELU`, `OpSiLU`, `OpELU`, `OpHardSwish`, `OpHardSigmoid`, `OpMish`, `OpSoftplus`, `OpRelu6`, `OpAbs`, `OpNeg`, `OpExp`, `OpLog`, `OpSqrt`, `OpRsqrt`, `OpReciprocal`

### `IPointwiseBinary` Structs (7)
`OpAdd`, `OpSub`, `OpMul`, `OpDiv`, `OpMin`, `OpMax`, `OpPow`

### Public Functions
| Function | Signature |
|----------|-----------|
| `pointwise_unary_apply<Op : IPointwise>` | `[ForceInline] void pointwise_unary_apply(StructuredBuffer<float> input, RWStructuredBuffer<float> output, uint idx, uint n)` |
| `pointwise_binary_apply<Op : IPointwiseBinary>` | `[ForceInline] void pointwise_binary_apply(StructuredBuffer<float> in_a, StructuredBuffer<float> in_b, RWStructuredBuffer<float> output, uint idx, uint n)` |

---

## `reduction.slang`

Generic workgroup and wave-level reduction operators.

### Interfaces
| Interface | Extends | Methods |
|-----------|---------|---------|
| `IReduction` | — | `static float identity();` `static float combine(float a, float b);` |
| `IWaveReduction` | `IReduction` | `static float wave_reduce(float v);` |

### Reduction Operator Structs
| Struct | Implements | `identity()` | `combine()` | `wave_reduce()` |
|--------|-----------|-------------|------------|-----------------|
| `OpSum` | `IReduction, IWaveReduction` | `0.0f` | `a + b` | `WaveActiveSum(v)` |
| `OpProd` | same | `1.0f` | `a * b` | `WaveActiveProduct(v)` |
| `OpMaxReduce` | same | `−FLT_MAX` | `max(a, b)` | `WaveActiveMax(v)` |
| `OpMinReduce` | same | `FLT_MAX` | `min(a, b)` | `WaveActiveMin(v)` |
| `OpAny` | `IReduction` | `0.0f` | `(a or b) ? 1.0f : 0.0f` | — |
| `OpAll` | `IReduction` | `1.0f` | `(a and b) ? 1.0f : 0.0f` | — |
| `OpXorSum` | `IReduction` | `0.0f` | bitwise-xor | — |

### Utility Struct
| Struct | Fields |
|--------|--------|
| `ArgPair` | `float val; float idx;` |

### Public Functions
| Function | Signature |
|----------|-----------|
| `wg_reduce<R : IReduction>` | `[ForceInline] float wg_reduce(float v, uint tid, uint size)` |
| `wg_reduce_wave<W : IWaveReduction>` | `[ForceInline] float wg_reduce_wave(float v, uint tid, uint n_waves, uint simd)` |
| `wg_argmax` | `[ForceInline] ArgPair wg_argmax(ArgPair v, uint tid, uint size)` |
| `wg_argmin` | `[ForceInline] ArgPair wg_argmin(ArgPair v, uint tid, uint size)` |

---

## `tensor_layout.slang`

Generic N-D indexing utilities parameterized by compile-time rank.

### Public Functions
| Function | Signature |
|----------|-----------|
| `contiguous_offset<let N : int>` | `[ForceInline] uint contiguous_offset(uint linear, uint shape[N])` |
| `strided_offset<let N : int>` | `[ForceInline] uint strided_offset(uint linear, uint shape[N], uint stride[N])` |
| `broadcast_offset<let N : int>` | `[ForceInline] uint broadcast_offset(uint out_linear, uint out_shape[N], uint src_shape[N], uint src_stride[N])` |
| `unravel_index<let N : int>` | `void unravel_index(uint linear, uint shape[N], out uint coords[N])` |
| `ravel_index<let N : int>` | `[ForceInline] uint ravel_index(uint coords[N], uint stride[N])` |
| `numel<let N : int>` | `[ForceInline] uint numel(uint shape[N])` |

---

## Summary

| Module | Lines | Interfaces | Structs | Functions | `[BackwardDerivative]` |
|--------|-------|-----------|---------|-----------|----------------------|
| `atomics.slang` | 116 | 0 | 0 | 5 | 0 |
| `conv.slang` | 60 | 1 | 3 | 1 | 0 |
| `helpers.slang` | 439 | 0 | 1 | 29 | 0 |
| `losses.slang` | 49 | 0 | 0 | 7 | 0 |
| `mm.slang` | 61 | 1 | 3 | 1 | 0 |
| `norm.slang` | 141 | 1 | 3 | 3 | 4 |
| `pointwise.slang` | 289 | 2 | 25 | 2 | 10 |
| `reduction.slang` | 134 | 2 | 8 | 4 | 0 |
| `tensor_layout.slang` | 59 | 0 | 0 | 6 | 0 |

**Total:** 9 modules, 7 interfaces, 43 structs, 58 functions, 14 `[BackwardDerivative]` annotations.
