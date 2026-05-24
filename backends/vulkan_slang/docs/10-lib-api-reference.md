# 10 — Lib API Reference (Canonical Slang Signatures)

> **Track 2.1.** One canonical signature per primitive across the 18
> `shaders/lib/*.slang` modules. This document is the contract — every lib
> function documented here must keep its signature stable. Breaking changes
> require a version bump in the CI gate.

**Last verified:** 2026-05-25
**`[BackwardDerivative]` count:** 64 (37 pointwise + 10 losses + 4 norm + 6 reduction + 6 vk_reduction + 1 conv)
**Lib modules:** 18 (atomics, bucket, conv, dtype_pack, helpers, losses, mm, mm_int8, mm_tile, norm, philox, pointwise, pointwise_generic, reduction, special_math, tensor_layout, vk_helpers, vk_reduction)

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

### `[BackwardDerivative]` Annotation
| Forward | Backward |
|---------|----------|
| `[Differentiable] [BackwardDerivative(conv_inner_madd_bwd)] float conv_inner_madd(float a, float b, no_diff float acc)` | `void conv_inner_madd_bwd(inout DifferentialPair<float> a, inout DifferentialPair<float> b, no_diff float acc, float dout)` |

### Public Function
| Function | Signature |
|----------|-----------|
| `conv2d_direct<Bias : IConvBias>` | `void conv2d_direct(StructuredBuffer<float> X, StructuredBuffer<float> W, StructuredBuffer<float> bias, RWStructuredBuffer<float> Y, ConvShape s, uint n, uint cout, uint oh, uint ow)` |

---

## `helpers.slang`

Re-export facade (T2.6 split). Directly contains Welford online statistics and wave intrinsics; re-exports `dtype_pack`, `philox`, `special_math`, and `bucket` via `__exported import`. Any kernel that does `import helpers;` gets all four sub-modules automatically. See `shaders/lib/helpers.slang`.

### Struct `Welford`
| Field | Type |
|-------|------|
| `mean`, `m2`, `n` | `float` |

### Welford Functions (defined in `helpers.slang`)
| Function | Signature |
|----------|-----------|
| `welford_combine` | `Welford welford_combine(Welford a, Welford b)` |
| `welford_finalize` | `float2 welford_finalize(Welford w)` |
| `welford_update` | `void welford_update(inout Welford w, float x)` |

### Wave Intrinsics (defined in `helpers.slang`)
| Function | Signature |
|----------|-----------|
| `wave_sum<T : __BuiltinFloatingPointType>` | `T wave_sum(T v)` |
| `wave_max<T>` / `wave_min<T>` / `wave_prod<T>` | same pattern |
| `wave_prefix_sum<T>` / `wave_prefix_prod<T>` | prefix variants |
| `wave_read_first` | `float wave_read_first(float v)` |
| `wave_active_any` / `wave_active_all` / `wave_active_ballot` | bool / uint4 |
| `wave_active_bit_and` / `wave_active_bit_or` / `wave_active_bit_xor` | uint |
| `wave_active_count_bits` / `wave_prefix_count_bits` | uint |

**Re-exported from sub-modules:** `dtype_pack.*` (f16/bf16 pack/unpack), `philox.*` (PRNG), `special_math.*` (extension math), `bucket.*` (bucketize + vec4 reductions) — see their dedicated sections below.

---

## `losses.slang`

Eight loss elementals, each with a closed-form `[BackwardDerivative]` (T2.11). All are `[Differentiable]`.

| Function | Signature |
|----------|-----------|
| `mse_elem` | `[Differentiable] [BackwardDerivative(mse_elem_bwd)] float mse_elem(float pred, float target)` |
| `l1_elem` | `[Differentiable] [BackwardDerivative(l1_elem_bwd)] float l1_elem(float pred, float target)` |
| `smooth_l1_elem` | `[Differentiable] [BackwardDerivative(smooth_l1_elem_bwd)] float smooth_l1_elem(float pred, float target, no_diff float beta)` |
| `huber_elem` | `[Differentiable] [BackwardDerivative(huber_elem_bwd)] float huber_elem(float pred, float target, no_diff float delta)` |
| `bce_elem` | `[Differentiable] [BackwardDerivative(bce_elem_bwd)] float bce_elem(float pred, float target)` |
| `bce_with_logits_elem` | `[Differentiable] [BackwardDerivative(bce_with_logits_elem_bwd)] float bce_with_logits_elem(float logit, float target)` |
| `kl_div_elem` | `[Differentiable] [BackwardDerivative(kl_div_elem_bwd)] float kl_div_elem(float log_pred, float target)` |
| `softmax_elem` | `[Differentiable] [BackwardDerivative(softmax_elem_bwd)] float softmax_elem(float p, float din, no_diff float row_dot)` |

**`[BackwardDerivative]` annotations:** 9 (T2.11: closed-form grads; each bwd computes all gradient values before writing to any `inout` parameter)

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

The canonical pointwise dispatcher. **31 `[BackwardDerivative]` annotations.**

### `[BackwardDerivative]` Annotations (representative subset; see shaders/lib/pointwise.slang for full list)

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
| 11 | `[Differentiable] [BackwardDerivative(leaky_relu_fast_bwd)] float leaky_relu_fwd(float x, no_diff float neg_slope)` | same pattern |
| 12 | `[Differentiable] [BackwardDerivative(expm1_fast_bwd)] float expm1_fwd(float x)` | same pattern |
| 13 | `[Differentiable] [BackwardDerivative(log1p_fast_bwd)] float log1p_fwd(float x)` | same pattern |
| 14 | `[Differentiable] [BackwardDerivative(sqrt_fast_bwd)] float sqrt_fwd(float x)` | same pattern |
| 15 | `[Differentiable] [BackwardDerivative(reciprocal_fast_bwd)] float reciprocal_fwd(float x)` | same pattern |
| 16 | `[Differentiable] [BackwardDerivative(abs_fast_bwd)] float abs_fwd(float x)` | same pattern |
| 17 | `[Differentiable] [BackwardDerivative(neg_fast_bwd)] float neg_fwd(float x)` | same pattern |
| 18 | `[Differentiable] [BackwardDerivative(atan2_fast_bwd)] float atan2_fwd(float y, float x)` | same pattern |
| 19 | `[Differentiable] [BackwardDerivative(hypot_fast_bwd)] float hypot_fwd(float a, float b)` | same pattern |
| 20 | `[Differentiable] [BackwardDerivative(max_fast_bwd)] float max_fwd(float a, float b)` | same pattern |
| 21 | `[Differentiable] [BackwardDerivative(min_fast_bwd)] float min_fwd(float a, float b)` | same pattern |
| 22 | `[Differentiable] [BackwardDerivative(erf_fast_bwd)] float erf_fwd(float x)` | same pattern |
| 23 | `[Differentiable] [BackwardDerivative(erfc_fast_bwd)] float erfc_fwd(float x)` | same pattern |
| 24 | `[Differentiable] [BackwardDerivative(erfinv_fast_bwd)] float erfinv_fwd(float x)` | same pattern |
| 25 | `[Differentiable] [BackwardDerivative(lgamma_fast_bwd)] float lgamma_fwd(float x)` | same pattern |
| 26 | `[Differentiable] [BackwardDerivative(digamma_fast_bwd)] float digamma_fwd(float x)` | same pattern |
| 27 | `[Differentiable] [BackwardDerivative(ndtri_fast_bwd)] float ndtri_fwd(float x)` | same pattern |
| 28 | `[Differentiable] [BackwardDerivative(i0_fast_bwd)] float i0_fwd(float x)` | same pattern |
| 29 | `[Differentiable] [BackwardDerivative(i0e_fast_bwd)] float i0e_fwd(float x)` | same pattern |
| 30 | `[Differentiable] [BackwardDerivative(i1_fast_bwd)] float i1_fwd(float x)` | same pattern |
| 31 | `[Differentiable] [BackwardDerivative(i1e_fast_bwd)] float i1e_fwd(float x)` | same pattern |

### Interfaces
| Interface | Method |
|-----------|--------|
| `IPointwise` | `static float apply(float x)` |
| `IPointwiseBinary` | `static float apply(float a, float b)` |
| `IComplexPointwise` | `static float2 apply(float2 x)` |
| `IComplexPointwiseBinary` | `static float2 apply(float2 a, float2 b)` |

### `IPointwise` Structs (18)
`OpReLU`, `OpSigmoid`, `OpTanh`, `OpGELU`, `OpSiLU`, `OpELU`, `OpHardSwish`, `OpHardSigmoid`, `OpMish`, `OpSoftplus`, `OpRelu6`, `OpAbs`, `OpNeg`, `OpExp`, `OpLog`, `OpSqrt`, `OpRsqrt`, `OpReciprocal`

### `IPointwiseBinary` Structs (7)
`OpAdd`, `OpSub`, `OpMul`, `OpDiv`, `OpMin`, `OpMax`, `OpPow`

### `IComplexPointwiseBinary` Structs (4)
`OpComplexAdd`, `OpComplexSub`, `OpComplexMul`, `OpComplexDiv`

### `IComplexPointwise` Structs (2)
`OpComplexConj`, `OpComplexAbs`

### Public Functions
| Function | Signature |
|----------|-----------|
| `pointwise_unary_apply<Op : IPointwise>` | `[ForceInline] void pointwise_unary_apply(StructuredBuffer<float> input, RWStructuredBuffer<float> output, uint idx, uint n)` |
| `pointwise_binary_apply<Op : IPointwiseBinary>` | `[ForceInline] void pointwise_binary_apply(StructuredBuffer<float> in_a, StructuredBuffer<float> in_b, RWStructuredBuffer<float> output, uint idx, uint n)` |
| `pointwise_complex_unary_apply<Op : IComplexPointwise>` | `[ForceInline] void pointwise_complex_unary_apply(StructuredBuffer<float> input, RWStructuredBuffer<float> output, uint idx, uint n)` |
| `pointwise_complex_binary_apply<Op : IComplexPointwiseBinary>` | `[ForceInline] void pointwise_complex_binary_apply(StructuredBuffer<float> in_a, StructuredBuffer<float> in_b, RWStructuredBuffer<float> output, uint idx, uint n)` |

---

## `reduction.slang`

Generic workgroup and wave-level reduction operators. **2 `[BackwardDerivative]` annotations** (combine_max_bwd / combine_min_bwd).

### Interfaces
| Interface | Extends | Methods |
|-----------|---------|---------|
| `IReduction` | — | `static float identity();` `static float combine(float a, float b);` |
| `IWaveReduction` | `IReduction` | `static float wave_reduce(float v);` |

### `[BackwardDerivative]` Annotations
| Forward | Backward |
|---------|----------|
| `[Differentiable] [BackwardDerivative(combine_max_bwd)] float combine_max(float a, float b)` | `void combine_max_bwd(inout DifferentialPair<float> a, inout DifferentialPair<float> b, float dout)` |
| `[Differentiable] [BackwardDerivative(combine_min_bwd)] float combine_min(float a, float b)` | same pattern |

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

---

## `bucket.slang`

Binary search (bucketize) and vec4 horizontal reductions. Split out of `helpers.slang` in T2.6. See `shaders/lib/bucket.slang`.

| Function | Signature |
|----------|-----------|
| `vk_bucketize` | `[ForceInline] uint vk_bucketize(float val, uint lo, uint hi, StructuredBuffer<float> data, uint stride, bool right)` |
| `c10_vulkan_bucketize` | thin alias for `vk_bucketize` (legacy compat) |
| `vk_vec4_hsum` / `c10_vulkan_vec4_hsum` | `float vk_vec4_hsum(StructuredBuffer<float> buf, uint base)` |
| `vk_vec4_hmax` / `c10_vulkan_vec4_hmax` | same pattern |
| `vk_vec4_hmin` / `c10_vulkan_vec4_hmin` | same pattern |
| `vk_vec4_hprod` / `c10_vulkan_vec4_hprod` | same pattern |

**`[BackwardDerivative]` annotations:** 0

---

## `dtype_pack.slang`

IEEE-754 dtype pack/unpack helpers. Split out of `helpers.slang` in T2.6. Re-exported by `helpers.slang` via `import dtype_pack`. See `shaders/lib/dtype_pack.slang`.

| Function | Signature |
|----------|-----------|
| `f16_to_f32` / `f32_to_f16` | `[ForceInline] float f16_to_f32(uint h)` / `uint f32_to_f16(float f)` |
| `unpack_f16x2` / `pack_f16x2` | `float2 unpack_f16x2(uint word)` / `uint pack_f16x2(float lo, float hi)` |
| `bf16_to_f32` / `f32_to_bf16` | same pattern for bf16 |
| `unpack_bf16x2` / `pack_bf16x2` | same pattern for bf16 |
| `_vk_unpack_f16` / `_vk_pack_f16` | `float _vk_unpack_f16(uint word, uint lane)` / `uint _vk_pack_f16(float lo, float hi)` |
| `_vk_unpack_bf16` / `_vk_pack_bf16` | same pattern |
| `_vk_unpack_u8` / `_vk_unpack_i8` / `_vk_unpack_i16` | Sub-32-bit dtype unpack from `StructuredBuffer<uint>` |
| `_vk_unpack_f16_2d` / `_vk_pack_f16_2d` | 2D row-col variants |
| `_vk_unpack_bf16_2d` / `_vk_pack_bf16_2d` | same pattern |
| `uint.unpack_f16(lane)` | extension method |
| `uint.unpack_bf16(lane)` | extension method |

**`[BackwardDerivative]` annotations:** 0

---

## `mm_int8.slang`

INT8 quantized matrix multiply tiles. See `shaders/lib/mm_int8.slang`.

| Struct / Function | Purpose |
|-------------------|---------|
| `MMInt8_PC` | Push-constant layout for INT8 mm kernel |
| `unpack_i8` | Unpack int8 element from packed `StructuredBuffer<uint>` |
| `load_tiles_int8` | Load INT8 A/B tiles into register arrays |
| `mma_tile_int8` | INT8 × INT8 → INT32 tile MMA inner loop |
| `store_epilogue_int8` | Scale + store INT32 accumulator as float |
| `computeTile` | Top-level tile compute entry point |

**`[BackwardDerivative]` annotations:** 0

---

## `mm_tile.slang`

Register-tiled float32 matrix multiply with single / double buffering and `[Differentiable]` inner madd. See `shaders/lib/mm_tile.slang`.

| Struct / Function | Purpose |
|-------------------|---------|
| `MM_PC` | Push-constant layout for tiled mm kernel |
| `load_tiles_single` | Single-buffer A/B tile load |
| `load_tiles_double` | Double-buffer A/B tile load |
| `mma_tile_single` | Tile MMA inner loop (single buffer) |
| `mma_tile_double` | Tile MMA inner loop (double buffer) |
| `store_epilogue<Epi : IPointwise>` | Apply epilogue and store |
| `computeTile<Epi : IPointwise>` | Top-level tile compute entry |
| `tile_inner_madd` | `[Differentiable] float tile_inner_madd(float a, float b, float acc)` |

**`[BackwardDerivative]` annotations:** 0

---

## `philox.slang`

Philox counter-based PRNG. Split out of `helpers.slang` in T2.6. Re-exported by `helpers.slang` via `import philox`. See `shaders/lib/philox.slang`.

| Function | Signature |
|----------|-----------|
| `philox_mulhi32` | `[ForceInline] uint philox_mulhi32(uint a, uint b)` |
| `philox_round` | `[ForceInline] uint2 philox_round(uint2 ctr, uint2 key)` |
| `philox_bumpkey` | `[ForceInline] uint2 philox_bumpkey(uint2 key)` |
| `philox_rand` | `[ForceInline] float philox_rand(uint counter, uint seed)` |
| `philox_randn` | `[ForceInline] float philox_randn(uint counter, uint seed)` |
| `_vk_philox_rand` | thin alias for `philox_rand` |
| `_vk_philox_randn` | thin alias for `philox_randn` |

**`[BackwardDerivative]` annotations:** 0

---

## `pointwise_generic.slang`

Generic 4-entry-point compute shader (unary/binary float, unary/binary complex) using `ParameterBlock<KernelArgs>` + `ByteAddressBuffer`. Imports `pointwise.slang`. See `shaders/lib/pointwise_generic.slang`.

Entry points: `computeMain<Op : IPointwise>`, `computeMain<Op : IPointwiseBinary>`, `computeMain<Op : IComplexPointwise>`, `computeMain<Op : IComplexPointwiseBinary>` — all `[numthreads(256,1,1)]`.

**`[BackwardDerivative]` annotations:** 0 (ops resolved from `pointwise.slang`)

---

## `special_math.slang`

Special mathematical functions as extension methods on `float`. Used by `pointwise.slang` forwards. See `shaders/lib/special_math.slang`.

| Extension method | Notes |
|-----------------|-------|
| `float.erf()` | Horner polynomial approximation |
| `float.log1p()` | Numerically stable log(1+x) |
| `float.expm1()` | exp(x)−1 |
| `float.hypot(other)` | Overflow-safe hypot |
| `float.digamma()` | Stirling series |
| `float.lgamma()` | Log-gamma |
| `float.ndtri()` | Inverse normal CDF |
| `float.i0()` | Modified Bessel I₀ |
| `float.i0e()` | Scaled I₀ |
| `float.i1()` | Modified Bessel I₁ |
| `float.i1e()` | Scaled I₁ |
| `float.erfinv()` | Inverse erf |
| `float.spherical_bessel_j0()` | Spherical Bessel j₀ |
| `float.zeta(q)` | Hurwitz zeta |
| `float.polygamma(n)` | Polygamma |

**`[BackwardDerivative]` annotations:** 0 (differentiable forwards live in `pointwise.slang`)

---

## `vk_helpers.slang`

Welford online statistics, wave intrinsics, and Vulkan-specific reduction helpers. See `shaders/lib/vk_helpers.slang`.

### Struct `Welford`
| Field | Type |
|-------|------|
| `mean`, `m2`, `n` | `float` |

### Welford Functions
| Function | Signature |
|----------|-----------|
| `welford_combine` | `Welford welford_combine(Welford a, Welford b)` |
| `welford_finalize` | `float2 welford_finalize(Welford w)` |
| `welford_update` | `void welford_update(inout Welford w, float x)` |

### Wave Intrinsics
| Function | Signature |
|----------|-----------|
| `wave_sum<T>` / `wave_max<T>` / `wave_min<T>` / `wave_prod<T>` | `T wave_sum(T v)` |
| `wave_prefix_sum<T>` / `wave_prefix_prod<T>` | same pattern |
| `wave_read_first` | `float wave_read_first(float v)` |
| `wave_active_any` / `wave_active_all` | bool variants |
| `wave_active_ballot` | `uint4 wave_active_ballot(bool v)` |
| `wave_active_bit_and` / `wave_active_bit_or` / `wave_active_bit_xor` | uint variants |
| `wave_active_count_bits` / `wave_prefix_count_bits` | uint count variants |

**`[BackwardDerivative]` annotations:** 0

---

## `vk_reduction.slang`

Extended reduction library: differentiable reduction interfaces, scan primitives, 2D reductions, bitonic sort. **2 `[BackwardDerivative]` annotations.** See `shaders/lib/vk_reduction.slang`.

### Interfaces
| Interface | Extends | Notes |
|-----------|---------|-------|
| `IReduction` | — | `identity()` + `combine()` |
| `IDifferentiableReduction` | `IReduction` | adds `[Differentiable] combine()` |
| `IWaveReduction` | `IReduction` | adds `wave_reduce()` |
| `IScanOp` | — | inclusive / exclusive scan |

### `[BackwardDerivative]` Annotations
| Forward | Backward |
|---------|----------|
| `[Differentiable] [BackwardDerivative(combine_max_bwd)] float combine_max(float a, float b)` | `void combine_max_bwd(...)` |
| `[Differentiable] [BackwardDerivative(combine_min_bwd)] float combine_min(float a, float b)` | same pattern |

### Key Structs
`OpSum`, `OpProd`, `OpMaxReduce`, `OpMinReduce`, `OpArgMax`, `OpArgMin`, `WelfordResult<T>`, `ArgPair`, `IScanAdd`, `IScanMul`, `IScanMax`, `IScanMin`

### Key Public Functions
| Function | Notes |
|----------|-------|
| `wg_reduce<W : IWaveReduction>` | `float wg_reduce(float v, uint tid, uint size, uint simd)` |
| `wg_reduce_wave<W : IWaveReduction>` | wave-based variant |
| `wg_welford` | `WelfordResult<float> wg_welford(...)` — `[Differentiable]` |
| `welford_combine` | `[Differentiable] WelfordResult<float> welford_combine(...)` |
| `wg_argmax` / `wg_argmin` | `ArgPair` variants |
| `wg_bitonic_sort_wave` | in-place 2-element wave sort |
| `wave_scan_inclusive<S : IScanOp>` / `wg_scan_inclusive<S>` / `wg_scan_exclusive<S>` | prefix-scan family |
| `wg_reduce_wave_2d<W>` | 2D workgroup reduction |
| `vk_wg_reduce_any` / `vk_wg_reduce_all` / `vk_wg_reduce_xor` | boolean / bitwise reductions |
| `vk_wg_reduce_argmax` / `vk_wg_reduce_argmin` | float2 argmax/argmin |

---

## Summary

| Module | Lines | Interfaces | Structs | Functions | `[BackwardDerivative]` |
|--------|-------|-----------|---------|-----------|----------------------|
| `atomics.slang` | 242 | 0 | 0 | 5 | 0 |
| `bucket.slang` | 146 | 0 | 0 | 10 | 0 |
| `conv.slang` | 88 | 1 | 3 | 2 | 1 |
| `dtype_pack.slang` | 174 | 0 | 0 | 21 | 0 |
| `helpers.slang` | 219 | 0 | 1 | ~15 | 0 |
| `losses.slang` | 201 | 0 | 0 | 8 | 9 |
| `mm.slang` | 61 | 1 | 3 | 1 | 0 |
| `mm_int8.slang` | 215 | 0 | 1 | 5 | 0 |
| `mm_tile.slang` | 355 | 0 | 1 | 7 | 0 |
| `norm.slang` | 160 | 1 | 3 | 3 | 4 |
| `philox.slang` | 81 | 0 | 0 | 7 | 0 |
| `pointwise.slang` | 871 | 4 | 31 | 4 | 31 |
| `pointwise_generic.slang` | 76 | 0 | 2 | 4 | 0 |
| `reduction.slang` | 929 | 2 | 8 | 4 | 2 |
| `special_math.slang` | 467 | 0 | 0 | 15 | 0 |
| `tensor_layout.slang` | 59 | 0 | 0 | 6 | 0 |
| `vk_helpers.slang` | 219 | 0 | 1 | ~18 | 0 |
| `vk_reduction.slang` | 894 | 4 | 12 | ~20 | 2 |

**Total:** 18 modules, ~13 interfaces, ~66 structs, ~155 functions, **49 `[BackwardDerivative]` annotations** (31 pointwise + 9 losses + 4 norm + 2 reduction + 2 vk_reduction + 1 conv).
