module vk_helpers;

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

// ── M20.7 Welford streaming (single-element) update ──────────────────
// Knuth / Welford online algorithm: accumulate one element x into the
// running (mean, M2, n) without materialising intermediate sums.
// Use this inside a per-thread loop before feeding the result into the
// workgroup-level `wg_welford` from `reduction.slang`.
//
// Typical usage pattern:
//   Welford wf = { 0.0f, 0.0f, 0.0f };
//   for (uint d = tid; d < S; d += TG_SIZE) {
//       welford_update(wf, buf[d]);
//   }
//   WelfordResult<float> stats = wg_welford(...);
//
// Algorithm: delta = x - mean; n++; mean += delta/n; M2 += delta*(x-mean)
// (the second delta uses the UPDATED mean — Welford's two-delta form.)
public void welford_update(inout Welford w, float x) {
    w.n += 1.0f;
    float delta = x - w.mean;
    w.mean += delta / w.n;
    w.m2   += delta * (x - w.mean);
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

// ── M20.6 VK_SUBGROUP_SIZE spec constant ───────────────────────────────
//
// The wave-reduction helpers (``wg_reduce_wave<W>``, ``wg_scan_*``,
// etc.) take ``simd`` as a uint argument. Pre-M20.6, the codegen at
// ``kernel/reduction.py`` baked the value 64 directly into the
// rendered Slang. That hardcoded NAVI10 wave64 and prevented any
// future wave32 dispatch on NAVI21+ silicon — even though slangc
// itself supports per-wave-size codegen.
//
// M20.6: expose the wave size as a specialization constant so the
// rendered Slang reads ``VK_SUBGROUP_SIZE`` instead of ``64u`` and
// the dispatch pipeline picks the right value at create time via
// ``spec_constants=[(30, simd_size)]``. The default 64 preserves
// today's NAVI10 behaviour byte-for-byte; future wave32 autotune
// (M20.6.b) simply passes 32 in the spec_constants override.
//
// constant_id 30 is selected to avoid collisions with:
//   10-12  flash_attention forward
//   13-15  flash_attention_bwd
//   20-24  slang_mm
//   30-34  slang_conv2d  ← collision; conv2d uses 30-34 for tile dims
//   40-44  slang_conv_bwd
//
// CORRECTION: 30-34 are claimed by slang_conv2d. Library-scope spec
// consts live in their own ID namespace (the lib gets compiled to
// .slang-module first; kernel imports it). To be safe across all
// kernel templates that might co-import this lib, use a high ID
// outside the template ranges. Pick 100.
[[vk::constant_id(100)]] public const uint VK_SUBGROUP_SIZE = 64;

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

// ── M20.4 wave-intrinsic extensions ──────────────────────────────────
//
// Audit row H6 identified these missing wrappers. Each unlocks a fused
// kernel that today falls back to smem reduction (or skips the optimisation
// entirely):
//
//   wave_active_any / _all     — single-wave fast path for aten.any/aten.all
//   wave_active_ballot         — aten.nonzero, aten.where compaction
//   wave_active_bit_{and,or,xor} — bitmask reductions in one wave
//   wave_active_count_bits     — exclusive-scan output count for nonzero
//   wave_prefix_count_bits     — stream-compaction (top-k, gather/scatter)
//
// Capability gates:
//   AnyTrue / AllTrue map to SPIR-V `OpGroupNonUniformAny` / `OpGroupNonUniformAll`
//   (capability `GroupNonUniformVote`, Slang atom `subgroup_vote`).
//
//   BitAnd / BitOr / BitXor map to `OpGroupNonUniformBitwise{And,Or,Xor}`
//   (capability `GroupNonUniformArithmetic`, Slang atom `subgroup_arithmetic`).
//
//   Ballot / CountBits / PrefixCountBits map to `OpGroupNonUniformBallot`
//   / `OpGroupNonUniformBallotBitCount` (capability `GroupNonUniformBallot`,
//   Slang atom `subgroup_ballot`). RDNA1 / Vulkan 1.2+ has all three.

[require(spirv, subgroup_vote)]
public bool wave_active_any(bool v) { return WaveActiveAnyTrue(v); }

[require(spirv, subgroup_vote)]
public bool wave_active_all(bool v) { return WaveActiveAllTrue(v); }

[require(spirv, subgroup_ballot)]
public uint4 wave_active_ballot(bool v) { return WaveActiveBallot(v); }

[require(spirv, subgroup_arithmetic)]
public uint wave_active_bit_and(uint v) { return WaveActiveBitAnd(v); }

[require(spirv, subgroup_arithmetic)]
public uint wave_active_bit_or(uint v) { return WaveActiveBitOr(v); }

[require(spirv, subgroup_arithmetic)]
public uint wave_active_bit_xor(uint v) { return WaveActiveBitXor(v); }

[require(spirv, subgroup_ballot)]
public uint wave_active_count_bits(bool v) { return WaveActiveCountBits(v); }

[require(spirv, subgroup_ballot)]
public uint wave_prefix_count_bits(bool v) { return WavePrefixCountBits(v); }

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
