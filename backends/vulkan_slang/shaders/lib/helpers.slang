module helpers;

// T2.6 (2026-05-08): `helpers.slang` was 827L and mixed five unrelated
// concerns (IEEE-754 dtype packing, Philox RNG, special-function math,
// bucketize, wave intrinsics). Split into 4 focused sub-modules — this
// file now re-exports them so existing kernels with `import helpers;`
// keep resolving every name they used to.
//
// The `__exported import` declarations make the public symbols of each
// sub-module visible to any source unit that does `import helpers;`,
// matching the pre-split behaviour. Slang's `__exported import` is the
// idiomatic way to compose modules without forcing every kernel to
// enumerate its dependencies.
//
// ── N+1.14 __target_switch usage guide ─────────────────────────────────
// Slang's `__target_switch` lets one shader source emit different
// SPIR-V / HLSL / etc. per compilation target. Use this pattern when
// the same logical operation needs different implementations per
// backend (e.g. AMD RADV workarounds, SwiftShader quirks, lavapipe
// divergence).
//
// Pattern:
//   __target_switch
//   {
//   case spirv:
//       // SPIR-V path (production — Vulkan on RDNA1+)
//       void my_helper(...) { ... }
//   case hlsl:
//       // HLSL path (debug / validation on D3D12)
//       void my_helper(...) { ... }
//   }
//
// Add new target-specific paths:
//   1. Identify the divergence point (env-var gated workaround,
//      GPU-model quirk, capability gap).
//   2. Wrap the divergent code in `__target_switch { case spirv: ...;
//      case hlsl: ...; }`.
//   3. Precompile BOTH paths with `slangc -target spirv` and
//      `slangc -target hlsl` to verify they compile.
//   4. Remove the Python-side env-var cascade once the switch lands.
//
// See N+1.14 in docs/10-inductor-backend.md for the RADV dynamic-
// push-constant workaround migration plan.

__exported import dtype_pack;
__exported import philox;
__exported import special_math;
__exported import bucket;

// ── Welford accumulator (used by norm reductions) ─────────────────────

public struct Welford {
    public float mean;
    public float m2;
    public float n;
};

public Welford welford_combine(Welford a, Welford b) {
    float n = a.n + b.n;
    float inv = n > 0.0f ? 1.0f / n : 0.0f;
    float delta = b.mean - a.mean;
    Welford r;
    r.mean = a.mean + delta * b.n * inv;
    r.m2 = a.m2 + b.m2 + delta * delta * a.n * b.n * inv;
    r.n = n;
    return r;
}

public float2 welford_finalize(Welford w) {
    float v = w.n > 0.0f ? w.m2 / w.n : 0.0f;
    return float2(w.mean, v);
}

// ── Wave intrinsic wrappers ───────────────────────────────────────────
// Generic `T : __BuiltinFloatingPointType` wrappers around the SPIR-V
// `WaveActive*` / `WavePrefix*` intrinsics. Keeping them in `helpers`
// (rather than `reduction`) avoids a circular dep: `reduction.slang`
// already `import helpers;` to use these.
//
// N+1.13 — `[require(...)]` capability annotations.
// `WaveActive*` / `WavePrefix*` map to SPIR-V `OpGroupNonUniform*Add/Min/Max`
// (capability `GroupNonUniformArithmetic`, Slang capability atom
// `subgroup_arithmetic`). `WaveReadLaneFirst` maps to `OpGroupNonUniformBroadcast`
// (capability `GroupNonUniformBallot`, Slang atom `subgroup_ballot`).
// On RDNA1 / Vulkan 1.2+ both subgroup_arithmetic and subgroup_ballot are
// always available; we annotate so a future port to a tile-class GPU
// without these gets a clean slangc error rather than a SPIR-V link miscompile.

[require(spirv, subgroup_arithmetic)]
public T wave_sum<T : __BuiltinFloatingPointType>(T v) { return WaveActiveSum(v); }
[require(spirv, subgroup_arithmetic)]
public T wave_max<T : __BuiltinFloatingPointType>(T v) { return WaveActiveMax(v); }
[require(spirv, subgroup_arithmetic)]
public T wave_min<T : __BuiltinFloatingPointType>(T v) { return WaveActiveMin(v); }
[require(spirv, subgroup_arithmetic)]
public T wave_prod<T : __BuiltinFloatingPointType>(T v) { return WaveActiveProduct(v); }
[require(spirv, subgroup_arithmetic)]
public T wave_prefix_sum<T : __BuiltinFloatingPointType>(T v) { return WavePrefixSum(v); }
[require(spirv, subgroup_arithmetic)]
public T wave_prefix_prod<T : __BuiltinFloatingPointType>(T v) { return WavePrefixProduct(v); }
[require(spirv, subgroup_ballot)]
public float wave_read_first(float v) { return WaveReadLaneFirst(v); }

// ── N+1.14 __target_switch — device-divergence example ────────────────
// When the same logical operation needs different implementations per
// target, use __target_switch instead of Python-side env-var cascades.
// This example provides a skeleton for future AMD RADV dynamic-push-
// constant workarounds and SwiftShader/lavapipe quirks.

// SPIR-V path (production — Vulkan on RDNA1+).
// WaveActiveSum is natively available via subgroup_arithmetic.
[require(spirv, subgroup_arithmetic)]
void wave_reduce_sum(float val, inout float acc) {
    acc = WaveActiveSum(val);
}

// T2.7: `wg_sum_smem` retired in favor of canonical `wg_reduce<OpSum>`
// from `reduction.slang` (groupshared float[1024 + 32] — +32 bank-conflict
// padding M20). All call sites must `import reduction;` and use
// `wg_reduce<OpSum>(v, tid, size)`.
