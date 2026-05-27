module vk_reduction;
// M20.6: __exported so kernels that `import reduction;` also gain access
// to vk_helpers.slang's VK_SUBGROUP_SIZE specialization constant (declared
// at constant_id=100 with default 64). Without the re-export, the
// codegen-emitted reductions that reference VK_SUBGROUP_SIZE would fail
// with "declaration not accessible" — the spec const is declared
// `public` in vk_helpers, but a plain `import` doesn't transitively expose
// imported symbols to downstream consumers.
// NOTE: "helpers" is a reserved module name in slangc v2026.5.2 (causes segfault).
__exported import vk_helpers;

// ── Reduction interfaces ──────────────────────────────────────────────

public interface IReduction {
    static float identity();
    static float combine(float a, float b);
};

// Differentiable reduction — combine can be auto-diff'd.
public interface IDifferentiableReduction : IReduction {
    [Differentiable]
    static float combine(float a, float b);
};

// Wave-intrinsic reduction primitive. 2-4x faster than tree reduction.
// CG.M13: `finalize` lets the interface handle per-op post-reduction
// processing (e.g. mean divides by count) without string-based dispatch.
public interface IWaveReduction : IReduction {
    static float wave_reduce(float v);
    // Post-reduction finalization: default is identity.
    // count = total number of elements reduced.
    [ForceInline]
    static float finalize(float reduced_val, uint count) { return reduced_val; }
};


// ── NaN-aware combine helpers ──────────────────────────────────────

// CG.M11: Closed-form backward for combine_sum_nan.
// da = (isnan(b) ? 0 : dout), db = (isnan(a) ? 0 : dout).
[Differentiable]
[BackwardDerivative(combine_sum_nan_bwd)]
[ForceInline]
public float combine_sum_nan(float a, float b) {
    if (isnan(a)) return a;
    if (isnan(b)) return b;
    return a + b;
}

public void combine_sum_nan_bwd(
    inout DifferentialPair<float> dpa,
    inout DifferentialPair<float> dpb,
    float dout)
{
    float a = dpa.p;
    float b = dpb.p;
    if (isnan(a)) {
        dpa = diffPair(a, dpa.d + dout);
        dpb = diffPair(b, dpb.d);
    } else if (isnan(b)) {
        dpa = diffPair(a, dpa.d);
        dpb = diffPair(b, dpb.d + dout);
    } else {
        dpa = diffPair(a, dpa.d + dout);
        dpb = diffPair(b, dpb.d + dout);
    }
}

// CG.M11: Closed-form backward for combine_prod_nan.
// da = (isnan(b) ? 0 : b * dout), db = (isnan(a) ? 0 : a * dout).
[Differentiable]
[BackwardDerivative(combine_prod_nan_bwd)]
[ForceInline]
public float combine_prod_nan(float a, float b) {
    if (isnan(a)) return a;
    if (isnan(b)) return b;
    return a * b;
}

public void combine_prod_nan_bwd(
    inout DifferentialPair<float> dpa,
    inout DifferentialPair<float> dpb,
    float dout)
{
    float a = dpa.p;
    float b = dpb.p;
    if (isnan(a)) {
        dpa = diffPair(a, dpa.d + dout);
        dpb = diffPair(b, dpb.d);
    } else if (isnan(b)) {
        dpa = diffPair(a, dpa.d);
        dpb = diffPair(b, dpb.d + dout);
    } else {
        dpa = diffPair(a, dpa.d + b * dout);
        dpb = diffPair(b, dpb.d + a * dout);
    }
}

// CG.M16: Capability-gated wave-intrinsic wrappers.
// These directly invoke SPIR-V WaveActive* intrinsics which require
// subgroup_arithmetic (SPIR-V: GroupNonUniformArithmetic).
// RDNA1 (wave64): always available on Vulkan 1.2+.
// Portable fallback: tree reduction in groupshared memory (no wave ops).
[require(spirv, subgroup_arithmetic)]
[ForceInline]
public float wave_active_sum_nan(float v) {
    return WaveActiveSum(v);
}

[require(spirv, subgroup_arithmetic)]
[ForceInline]
public float wave_active_product_nan(float v) {
    return WaveActiveProduct(v);
}

// ── CG.M11: Differentiable max/min combine with closed-form backward ──
// These are free functions (not struct methods) because Slang requires
// [BackwardDerivative] targets to be free functions. The OpMaxReduce /
// OpMinReduce struct methods delegate to these.
//
// Backward of combine_max(a,b): gradient routes to the max position.
// If a > b: da = dout, db = 0.  If b > a: da = 0, db = dout.
// If equal: split evenly.

[Differentiable]
[BackwardDerivative(combine_max_bwd)]
[ForceInline]
public float combine_max(float a, float b) {
    return a > b ? a : b;
}

public void combine_max_bwd(
    inout DifferentialPair<float> dpa,
    inout DifferentialPair<float> dpb,
    float dout)
{
    float a = dpa.p;
    float b = dpb.p;
    if (a > b) {
        dpa = diffPair(a, dpa.d + dout);
        dpb = diffPair(b, dpb.d);
    } else if (b > a) {
        dpa = diffPair(a, dpa.d);
        dpb = diffPair(b, dpb.d + dout);
    } else {
        float half = dout * 0.5f;
        dpa = diffPair(a, dpa.d + half);
        dpb = diffPair(b, dpb.d + half);
    }
}

[Differentiable]
[BackwardDerivative(combine_min_bwd)]
[ForceInline]
public float combine_min(float a, float b) {
    return a < b ? a : b;
}

public void combine_min_bwd(
    inout DifferentialPair<float> dpa,
    inout DifferentialPair<float> dpb,
    float dout)
{
    float a = dpa.p;
    float b = dpb.p;
    if (a < b) {
        dpa = diffPair(a, dpa.d + dout);
        dpb = diffPair(b, dpb.d);
    } else if (b < a) {
        dpa = diffPair(a, dpa.d);
        dpb = diffPair(b, dpb.d + dout);
    } else {
        float half = dout * 0.5f;
        dpa = diffPair(a, dpa.d + half);
        dpb = diffPair(b, dpb.d + half);
    }
}

// ── Concrete reduction ops (float-monomorphized) ──────────────────────

// CG.M13: Concrete reduction ops implement `IWaveReduction` (with `finalize`).
// The codegen emits `wg_reduce_wave<OpSum>(...)` — per-op variants are
// Slang type instantiations, not string-template parameters.
public struct OpSum : IReduction, IWaveReduction, IDifferentiableReduction {
    static float identity() { return 0.0f; }
    [Differentiable]
    static float combine(float a, float b) { return combine_sum_nan(a, b); }
    static float wave_reduce(float v) { return wave_active_sum_nan(v); }
    // sum finalize is identity (no division needed — mean is handled
    // separately by the codegen as sum + broadcast divide)
};

public struct OpProd : IReduction, IWaveReduction, IDifferentiableReduction {
    static float identity() { return 1.0f; }
    [Differentiable]
    static float combine(float a, float b) { return combine_prod_nan(a, b); }
    static float wave_reduce(float v) { return wave_active_product_nan(v); }
};

// CG.M11: OpMaxReduce / OpMinReduce implement IDifferentiableReduction.
// combine() delegates to the free functions combine_max / combine_min
// which carry [BackwardDerivative] for closed-form gradient routing
// to the argmax/argmin position.
public struct OpMaxReduce : IReduction, IWaveReduction, IDifferentiableReduction {
    static float identity() { return -3.402823e38f; }
    [Differentiable]
    static float combine(float a, float b) { return combine_max(a, b); }
    static float wave_reduce(float v) { return WaveActiveMax(v); }
};

public struct OpMinReduce : IReduction, IWaveReduction, IDifferentiableReduction {
    static float identity() { return 3.402823e38f; }
    [Differentiable]
    static float combine(float a, float b) { return combine_min(a, b); }
    static float wave_reduce(float v) { return WaveActiveMin(v); }
};

// CG.M13: ArgMax / ArgMin as proper IWaveReduction structs.
// These operate on (value, index) pairs packed into ArgPair.
// identity() returns a sentinel ArgPair; combine picks the winning
// element; wave_reduce uses WaveActiveMax/WaveActiveMin on the
// packed pair (float2 compares lexicographically by .x).
public struct OpArgMax : IWaveReduction {
    static float identity() { return -3.402823e38f; }
    static float combine(float a, float b) { return a > b ? a : b; }
    static float wave_reduce(float v) { return WaveActiveMax(v); }
};

public struct OpArgMin : IWaveReduction {
    static float identity() { return 3.402823e38f; }
    static float combine(float a, float b) { return a < b ? a : b; }
    static float wave_reduce(float v) { return WaveActiveMin(v); }
};

// ── Wave reduction main entry ────────────────────────────────────────

// CG.M13: Unified wave-intrinsic two-stage reduction.
// Phase 1: each thread reduces its local value via wave_reduce.
// Phase 2: cross-wave reduction via shared memory.
// n_waves=1 fast-path skips shared memory entirely.
// finalize() is available on IWaveReduction for per-op post-processing
// (e.g. mean = sum / count); callers should apply it after wg_reduce_wave
// returns if the reduction requires count-dependent finalization.
//
// CG.M16: wg_reduce_wave calls W::wave_reduce() which dispatches through
// the concrete op's wave_reduce (e.g. OpSum.wave_reduce → wave_active_sum_nan
// which carries [require(spirv, subgroup_arithmetic)]). Capability
// propagation is automatic via the generic instantiation.
groupshared float _wg_reduce_smem[1024 + 32];

[ForceInline]
public float wg_reduce_wave<W : IWaveReduction>(
    float v, uint tid, uint n_waves, uint simd)
{
    if (n_waves == 1u)
        return W::wave_reduce(v);

    uint lane = tid % simd;
    uint wave_id = tid / simd;

    float wave_result = W::wave_reduce(v);
    if (lane == 0u)
        _wg_reduce_smem[wave_id] = wave_result;

    GroupMemoryBarrierWithGroupSync();

    if (wave_id == 0u) {
        float smem_val = (tid < n_waves) ? _wg_reduce_smem[tid] : W::identity();
        float final_result = W::wave_reduce(smem_val);
        _wg_reduce_smem[tid] = final_result;
    }

    GroupMemoryBarrierWithGroupSync();

    float result = _wg_reduce_smem[0];
    return result;
}

// Convenience wrapper — 4-arg form used by norm.slang and other lib
// modules. simd is VK_SUBGROUP_SIZE (spec constant, default 64).
[ForceInline]
public float wg_reduce<W : IWaveReduction>(float v, uint tid, uint size, uint simd)
{
    uint n_waves = (size + simd - 1u) / simd;
    return wg_reduce_wave<W>(v, tid, n_waves, simd);
}

// ── Welford result and reduction ──────────────────────────────────────

public struct WelfordResult<T : __BuiltinFloatingPointType> : IDifferentiable {
    public T mean;
    public T m2;
    public T n;
};

// Float-monomorphized welford combine.
// Forward: combine two (mean, m2, n) pairs into one.
// CG.M11: [BackwardDerivative] for variance backward is deferred —
// the inputs are no_diff (intermediate reduction-tree nodes), so
// autodiff traces through the per-element operations instead.
[Differentiable]
public WelfordResult<float> welford_combine(
    no_diff WelfordResult<float> a,
    no_diff WelfordResult<float> b)
{
    float n_total = a.n + b.n;
    float delta = b.mean - a.mean;
    float na_nb_n = a.n * b.n / n_total;
    WelfordResult<float> r;
    r.mean = a.mean + delta * b.n / n_total;
    r.m2 = a.m2 + b.m2 + delta * delta * na_nb_n;
    r.n = n_total;
    return r;
}


// ── Group welford reduction ──────────────────────────────────────────
// +32 bank-conflict padding (M20)
//
// CG.M16: wg_welford uses WaveReadLaneAt which requires subgroup_shuffle
// (SPIR-V: GroupNonUniformShuffle). RDNA1 (wave64): always available.
groupshared float _welford_smem_mean[1024 + 32];
groupshared float _welford_smem_m2[1024 + 32];
groupshared float _welford_smem_n[1024 + 32];

[require(spirv, subgroup_shuffle)]
// M-NEW.10: [ForceInline] re-added — see reduction.slang for the bug
// rationale (out-of-line callable produces half-magnitude m2 in the
// multistage code path).  M-AUDIT-PERF.1's slangc-hang justification
// no longer reproduces on slangc 2026.7.1.
[ForceInline]
public WelfordResult<float> wg_welford(
    WelfordResult<float> v, uint tid, uint size, uint simd)
{
    if (size == 1u)
        return v;

    uint lane = tid % simd;
    uint wave_id = tid / simd;
    uint n_waves = (size + simd - 1u) / simd;

    // Wave-level reduction
    // M-NEW.10: see reduction.slang for the full bug write-up.  Summary:
    // (1) clamp combine to `wave_valid` (within-wave partner only); the
    //     prior `lane + offset < size` was always true once size > simd,
    //     so high lanes did OOB WaveReadLaneAt picking up undefined data.
    // (2) snapshot all three neighbor reads before mutating local acc.
    // (3) broadcast lane 0's result for n_waves==1 so unguarded post-
    //     reduction pointwise stores see a consistent value.
    WelfordResult<float> acc = v;
    uint wave_base = wave_id * simd;
    uint wave_valid = min(simd, size > wave_base ? size - wave_base : 0u);
    for (uint offset = simd >> 1u; offset > 0u; offset >>= 1u) {
        uint _src = (lane + offset) & (simd - 1u);
        float other_mean = WaveReadLaneAt(acc.mean, _src);
        float other_m2 = WaveReadLaneAt(acc.m2, _src);
        float other_n = WaveReadLaneAt(acc.n, _src);
        if (lane + offset < wave_valid) {
            float n_total = acc.n + other_n;
            float inv = n_total > 0.0f ? 1.0f / n_total : 0.0f;
            float delta = other_mean - acc.mean;
            float new_mean = acc.mean + delta * other_n * inv;
            float new_m2 = acc.m2 + other_m2 + delta * delta * acc.n * other_n * inv;
            acc.mean = new_mean;
            acc.m2 = new_m2;
            acc.n = n_total;
        }
    }

    if (n_waves == 1u) {
        WelfordResult<float> bcast;
        bcast.mean = WaveReadLaneAt(acc.mean, 0u);
        bcast.m2 = WaveReadLaneAt(acc.m2, 0u);
        bcast.n = WaveReadLaneAt(acc.n, 0u);
        return bcast;
    }

    if (lane == 0u) {
        _welford_smem_mean[wave_id] = acc.mean;
        _welford_smem_m2[wave_id] = acc.m2;
        _welford_smem_n[wave_id] = acc.n;
    }

    GroupMemoryBarrierWithGroupSync();

    // Every thread runs the serial fold redundantly — no `if (tid == 0u)`
    // means no divergence around the final state, no second barrier
    // required, and no chance for slangc to elide per-lane stores around
    // a divergent barrier.  See reduction.slang for the full rationale.
    WelfordResult<float> r = WelfordResult<float>(
        _welford_smem_mean[0], _welford_smem_m2[0], _welford_smem_n[0]);
    for (uint i = 1u; i < n_waves; ++i) {
        float b_mean = _welford_smem_mean[i];
        float b_m2 = _welford_smem_m2[i];
        float b_n = _welford_smem_n[i];
        if (b_n > 0.0f) {
            float n_total = r.n + b_n;
            float inv = 1.0f / n_total;
            float delta = b_mean - r.mean;
            r.mean = r.mean + delta * b_n * inv;
            r.m2 = r.m2 + b_m2 + delta * delta * r.n * b_n * inv;
            r.n = n_total;
        }
    }
    return r;
}

// ── 2D wave reduction for partitioned layouts ────────────────────────

groupshared float _wg_2d_smem[32];

[ForceInline]
public float wg_reduce_wave_2d<W : IWaveReduction>(
    float v, uint tid_x, uint tid_y, uint tx, uint ty, uint simd)
{
    return W::wave_reduce(v);
}

// ── Argmin/argmax scanned reduction ──────────────────────────────────

public struct ArgPair {
    public float val;
    public float idx;
};

// +32 bank-conflict padding (M20)
//
// CG.M16: wg_argmax uses WaveReadLaneAt which requires subgroup_shuffle.
groupshared float _wg_arg_smem_val[1024 + 32];
groupshared float _wg_arg_smem_idx[1024 + 32];

[require(spirv, subgroup_shuffle)]
[ForceInline]
public ArgPair wg_argmax(ArgPair v, uint tid, uint size, uint simd)
{
    if (size == 1u) return v;
    uint lane = tid % simd;
    uint wave_id = tid / simd;

    ArgPair acc = v;
    for (uint offset = simd >> 1u; offset > 0u; offset >>= 1u) {
        uint _src = (lane + offset) & (simd - 1u);
        float other_val = WaveReadLaneAt(acc.val, _src);
        if (lane + offset < size && other_val > acc.val) {
            acc.val = other_val;
            acc.idx = WaveReadLaneAt(acc.idx, _src);
        }
    }

    uint n_waves = (size + simd - 1u) / simd;
    if (lane == 0u) {
        _wg_arg_smem_val[wave_id] = acc.val;
        _wg_arg_smem_idx[wave_id] = acc.idx;
    }
    GroupMemoryBarrierWithGroupSync();

    ArgPair result;
    if (wave_id == 0u) {
        result.val = (tid < n_waves) ? _wg_arg_smem_val[tid] : -3.402823e38f;
        result.idx = (tid < n_waves) ? _wg_arg_smem_idx[tid] : -1.0f;
        GroupMemoryBarrierWithGroupSync();
        for (uint offset = simd >> 1u; offset > 0u; offset >>= 1u) {
            uint _src = (lane + offset) & (simd - 1u);
            float other_val = WaveReadLaneAt(result.val, _src);
            if (lane + offset < n_waves && other_val > result.val) {
                result.val = other_val;
                result.idx = WaveReadLaneAt(result.idx, _src);
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    result.val = _wg_arg_smem_val[0];
    result.idx = _wg_arg_smem_idx[0];
    return result;
}

// ── Bitonic sort helpers (N.2) ──────────────────────────────────────

// CG.M16: wg_bitonic_sort_wave uses WaveReadLaneAt which requires subgroup_shuffle.
[require(spirv, subgroup_shuffle)]
[ForceInline]
public void wg_bitonic_sort_wave(inout float2 v, uint lane) {
    for (uint mask = 1u; mask < 64u; mask <<= 1u) {
        float2 other = WaveReadLaneAt(v, lane ^ mask);
        bool swap = (lane & mask) != 0u;
        float2 a = swap ? other : v;
        float2 b = swap ? v : other;
        if (a.x > b.x || (a.x == b.x && a.y > b.y)) {
            v = swap ? a : b;
        } else {
            v = swap ? b : a;
        }
    }
}

// ── Differentiable fold functions for autodiff ──────────────────────
// These are the scalar fold primitives used by reduction trees.
// bwd_diff(reduce_fold_sum)(dpa, dpb, dout) yields dpa.d = dpb.d = dout
//   → every element's gradient = grad_output (sum reduction).
// bwd_diff(reduce_fold_prod)(dpa, dpb, dout) yields
//   dpa.d = dout * b, dpb.d = dout * a
//   → grad_in[i] = grad_out * prod / x_i (prod reduction).
// The [Differentiable] annotations are verified by slangc during
// compilation; the backward dispatches as broadcast kernels via
// bwd_diff_dispatch.dispatch_reduction_bwd.
//
// CG.M11: OpSum/OpProd folds already differentiate trivially (broadcast
// grad_output for sum; grad_output * other_element for prod). The
// max/min folds now have [BackwardDerivative] on combine_max /
// combine_min above. OpMaxReduce.combine / OpMinReduce.combine delegate
// to these free functions.

[Differentiable]
public float reduce_fold_sum(float a, float b) {
    return combine_sum_nan(a, b);
}

[Differentiable]
public float reduce_fold_prod(float a, float b) {
    return combine_prod_nan(a, b);
}

// ── Inclusive / exclusive scan ────────────────────────────────────────
//
// M20h: full multi-wave workgroup scan.
//
// Phase 1 — per-lane in-wave inclusive scan via Hillis-Steele shuffle
//           (works for any associative IScanOp; the WavePrefixSum/
//           WavePrefixProduct fast path is op-specific so we don't take
//           it here — the shuffle ladder costs log2(simd) shuffles).
//
// Phase 2 — last lane of each wave writes its wave total to smem.
//           A single GroupMemoryBarrierWithGroupSync makes the totals
//           visible to all waves.
//
// Phase 3 — wave 0 loads the wave-total vector, runs the same shuffle
//           scan in-place, and writes the *exclusive* prefix (i.e.
//           total-of-prior-waves) back to smem.
//
// Phase 4 — every lane adds its wave's exclusive prefix onto its
//           Phase-1 inclusive scan. Result is the workgroup-wide
//           inclusive prefix scan.
//
// Exclusive variant: compute inclusive, then `combine(identity, prev_lane)`
//                    via shuffle — i.e. shift right by one lane within the
//                    workgroup. Lane 0 of WG returns identity.

public interface IScanOp {
    static float identity();
    static float combine(float a, float b);
};

// Concrete IScanOp ops (codegen names: kernel/reduction.py:_probe_scan_op).
// Each maps a torch op (add / mul / maximum / minimum) to its identity
// + associative combiner. combine uses the NaN-aware sum/prod helpers
// for parity with the wg_reduce reductions; max/min just use the
// inline comparison (matches OpMaxReduce / OpMinReduce).
public struct IScanAdd : IScanOp {
    static float identity() { return 0.0f; }
    static float combine(float a, float b) { return combine_sum_nan(a, b); }
};

public struct IScanMul : IScanOp {
    static float identity() { return 1.0f; }
    static float combine(float a, float b) { return combine_prod_nan(a, b); }
};

public struct IScanMax : IScanOp {
    static float identity() { return -3.402823e38f; }
    static float combine(float a, float b) { return a > b ? a : b; }
};

public struct IScanMin : IScanOp {
    static float identity() { return 3.402823e38f; }
    static float combine(float a, float b) { return a < b ? a : b; }
};

// Per-wave Hillis-Steele inclusive scan. Generic over IScanOp so we
// don't depend on op-specific wave-prefix intrinsics (WavePrefixSum
// covers + only; max/min have no direct intrinsic).
//
// CG.M16: WaveReadLaneAt requires subgroup_shuffle (RDNA1: always on).
[require(spirv, subgroup_shuffle)]
[ForceInline]
public float wave_scan_inclusive<S : IScanOp>(float v, uint lane, uint simd)
{
    float acc = v;
    for (uint offset = 1u; offset < simd; offset <<= 1u) {
        // Slang's WaveReadLaneAt requires non-negative indices; lane is
        // uint so (lane - offset) underflows benignly for the guarded
        // `lane >= offset` lanes — only the unmasked side reads it.
        float other = WaveReadLaneAt(acc, lane - offset);
        if (lane >= offset)
            acc = S::combine(acc, other);
    }
    return acc;
}

// +32 bank-conflict padding (matches existing smem layouts in this file).
// 33 entries is enough (1024 / 32 = 32 waves max on wave32; only 16 on
// wave64) but keeping it generous + aligned with the surrounding pattern.
groupshared float _wg_scan_smem[1024 + 32];

[require(spirv, subgroup_shuffle)]
[ForceInline]
public float wg_scan_inclusive<S : IScanOp>(float v, uint tid, uint size, uint simd)
{
    if (size == 1u) return v;
    uint lane = tid % simd;
    uint wave_id = tid / simd;
    uint n_waves = (size + simd - 1u) / simd;

    // Phase 1: per-wave inclusive scan.
    float wave_incl = wave_scan_inclusive<S>(v, lane, simd);

    if (n_waves == 1u) return wave_incl;

    // Phase 2: last active lane of each wave writes its wave total.
    // Last lane of wave w covers index min(simd-1, size-1-w*simd).
    uint wave_base = wave_id * simd;
    uint wave_last_lane = (size - wave_base >= simd) ? (simd - 1u)
                                                     : (size - wave_base - 1u);
    if (lane == wave_last_lane && tid < size)
        _wg_scan_smem[wave_id] = wave_incl;

    GroupMemoryBarrierWithGroupSync();

    // Phase 3: wave 0 scans the wave-total vector and writes back the
    // *exclusive* prefix (i.e. exclusive wave-id w = inclusive wave-id (w-1)).
    if (wave_id == 0u) {
        float total = (lane < n_waves) ? _wg_scan_smem[lane] : S::identity();
        float incl = wave_scan_inclusive<S>(total, lane, simd);
        // Exclusive = shift right by one lane (lane 0 → identity).
        float prev = WaveReadLaneAt(incl, lane - 1u);
        float excl = (lane == 0u) ? S::identity() : prev;
        if (lane < n_waves)
            _wg_scan_smem[lane] = excl;
    }

    GroupMemoryBarrierWithGroupSync();

    // Phase 4: every lane combines its wave's exclusive prefix with its
    // intra-wave inclusive scan.
    float wave_prefix = _wg_scan_smem[wave_id];
    return S::combine(wave_prefix, wave_incl);
}

// Exclusive scan: shift the inclusive result right by one lane within
// the workgroup. Lane 0 (tid 0) returns the identity element.
//
// We avoid recomputing the scan by deriving it from wg_scan_inclusive:
//   exclusive(tid) = inclusive(tid - 1)   if tid > 0
//                  = identity             if tid == 0
//
// Across a wave boundary we use WaveReadLaneAt to fetch lane (simd-1)
// of the previous wave from smem (wave_prefix already holds that).
// Within a wave we use WaveReadLaneAt on the per-lane inclusive value.
[require(spirv, subgroup_shuffle)]
[ForceInline]
public float wg_scan_exclusive<S : IScanOp>(float v, uint tid, uint size, uint simd)
{
    if (size == 1u) return S::identity();
    uint lane = tid % simd;
    uint wave_id = tid / simd;
    uint n_waves = (size + simd - 1u) / simd;

    // Phase 1: per-wave inclusive scan (reused for both incl + excl).
    float wave_incl = wave_scan_inclusive<S>(v, lane, simd);

    if (n_waves == 1u) {
        // Single-wave WG: exclusive is just inclusive shifted right.
        float prev = WaveReadLaneAt(wave_incl, lane - 1u);
        return (lane == 0u) ? S::identity() : prev;
    }

    // Phase 2: per-wave total to smem.
    uint wave_base = wave_id * simd;
    uint wave_last_lane = (size - wave_base >= simd) ? (simd - 1u)
                                                     : (size - wave_base - 1u);
    if (lane == wave_last_lane && tid < size)
        _wg_scan_smem[wave_id] = wave_incl;

    GroupMemoryBarrierWithGroupSync();

    // Phase 3: wave 0 scans the wave totals and writes back exclusive
    // wave prefixes.
    if (wave_id == 0u) {
        float total = (lane < n_waves) ? _wg_scan_smem[lane] : S::identity();
        float incl = wave_scan_inclusive<S>(total, lane, simd);
        float prev = WaveReadLaneAt(incl, lane - 1u);
        float excl = (lane == 0u) ? S::identity() : prev;
        if (lane < n_waves)
            _wg_scan_smem[lane] = excl;
    }

    GroupMemoryBarrierWithGroupSync();

    // Phase 4: combine wave's exclusive prefix with intra-wave *exclusive* scan.
    // Intra-wave exclusive = shift wave_incl right by one lane.
    float wave_prefix = _wg_scan_smem[wave_id];
    float prev_incl = WaveReadLaneAt(wave_incl, lane - 1u);
    float intra_excl = (lane == 0u) ? S::identity() : prev_incl;
    return S::combine(wave_prefix, intra_excl);
}

// ── M18.1 vk_wg_reduce_* helpers ─────────────────────────────────────
//
// M20.6: each helper accepts a `simd` parameter (VK_SUBGROUP_SIZE spec
// constant, default 64). On wave32 hardware (NAVI21+) the spec constant
// is overridden at pipeline-create time and flows through to lane masks,
// n_waves computation, and cross-wave smem boundaries.

// vk_wg_reduce_any — boolean-OR via float-count reduction.
// Codegen passes `value ? 1.0f : 0.0f` and post-tests `!= 0.0f`, so any
// nonzero sum signals "at least one thread saw true". OpSum already gives
// us NaN-aware add via combine_sum_nan; that's fine here because the
// inputs are always 0.0f or 1.0f (no NaN), and the return type is the
// reduced count.
//
// M20.4 fast path: when the reduction fits in a single wave
// (red_size <= 64 on RDNA1), `WaveActiveAnyTrue` collapses the entire
// workgroup-vote in one instruction with zero LDS traffic. The smem
// hop in `wg_reduce_wave<OpSum>` is wasted work in this regime.
[require(spirv, subgroup_vote)]
[ForceInline]
public float vk_wg_reduce_any(float value, uint tid, uint red_size, uint simd)
{
    if (red_size <= simd) {
        // Single-wave fast path: no smem, no barrier, no cross-wave hop.
        return WaveActiveAnyTrue(value != 0.0f) ? 1.0f : 0.0f;
    }
    uint n_waves = (red_size + simd - 1u) / simd;
    return wg_reduce_wave<OpSum>(value, tid, n_waves, simd);
}

// vk_wg_reduce_all — boolean-AND dual of vk_wg_reduce_any (M20.4).
// Not currently called by the kernel/reduction.py codegen (which lowers
// aten.all via the argmin pathway), but exposed for completeness and
// future codegen wiring. Same single-wave fast path; the cross-wave
// path multiplies float values via OpProd, where any zero kills the
// product — matching the boolean-AND semantics.
[require(spirv, subgroup_vote)]
[ForceInline]
public float vk_wg_reduce_all(float value, uint tid, uint red_size, uint simd)
{
    if (red_size <= simd) {
        return WaveActiveAllTrue(value != 0.0f) ? 1.0f : 0.0f;
    }
    uint n_waves = (red_size + simd - 1u) / simd;
    return wg_reduce_wave<OpProd>(value, tid, n_waves, simd);
}

// ── Bitwise XOR workgroup reduction (uint) ────────────────────────────
//
// `wg_reduce_wave<W>` is float-only (matches OpSum/OpProd/OpMaxReduce/
// OpMinReduce — all `IWaveReduction` over float). bitwise_xor needs a
// uint reduction, so we implement it directly with `WaveActiveBitXor`
// + `groupshared uint` smem. The same wave-then-cross-wave two-stage
// shape that `wg_reduce_wave` uses.
//
// CG.M16: `WaveActiveBitXor` requires `subgroup_arithmetic`; smem
// shuffle uses `subgroup_shuffle` for the cross-wave stage. RDNA1
// (wave64) always has both.

// +32 bank-conflict padding (matches _wg_reduce_smem layout).
groupshared uint _wg_xor_smem[1024 + 32];

[require(spirv, subgroup_arithmetic)]
[ForceInline]
public uint vk_wg_reduce_xor(uint value, uint tid, uint red_size, uint simd)
{
    if (red_size <= simd) {
        return WaveActiveBitXor(value);
    }

    uint lane = tid % simd;
    uint wave_id = tid / simd;
    uint n_waves = (red_size + simd - 1u) / simd;

    uint wave_result = WaveActiveBitXor(value);
    if (lane == 0u)
        _wg_xor_smem[wave_id] = wave_result;

    GroupMemoryBarrierWithGroupSync();

    if (wave_id == 0u) {
        uint smem_val = (tid < n_waves) ? _wg_xor_smem[tid] : 0u;
        uint final_result = WaveActiveBitXor(smem_val);
        _wg_xor_smem[tid] = final_result;
    }

    GroupMemoryBarrierWithGroupSync();

    return _wg_xor_smem[0];
}

// 2D variant — partitioned (ty × tx) layout, flat id = lid.y * tx + lid.x,
// total = tx * ty. The codegen guarantees total <= max_threadgroup_size
// (1024 on RDNA1), so the same wave-then-cross-wave structure applies.
// `tx` is accepted for signature parity with the codegen call (see
// kernel/reduction.py:306) but the flat id already encodes the 2D layout
// so we only need (flat_idx, total) for the reduction itself.
[require(spirv, subgroup_arithmetic)]
[ForceInline]
public uint vk_wg_reduce_xor_2d(uint value, uint tx, uint flat_idx, uint total, uint simd)
{
    return vk_wg_reduce_xor(value, flat_idx, total, simd);
}

// ── argmax / argmin workgroup reduction (float2) ──────────────────────
//
// Codegen packs (value, index) as `float2(val, float(idx))` and expects
// the same float2 back — the caller extracts `.x` (val) and `.y` (idx).
// We mirror the existing `wg_argmax(ArgPair, …)` pattern but accept /
// return `float2` to match the codegen string.
//
// On tie (acc.val == other.val) we propagate the *smaller* index for both
// argmax and argmin — this matches PyTorch's behaviour (first occurrence
// wins on ties).
//
// CG.M16: WaveReadLaneAt requires subgroup_shuffle; RDNA1 (wave64): yes.

// +32 bank-conflict padding (matches _wg_arg_smem_* layout).
groupshared float _wg_argxx_smem_val[1024 + 32];
groupshared float _wg_argxx_smem_idx[1024 + 32];

[require(spirv, subgroup_shuffle)]
[ForceInline]
public float2 vk_wg_reduce_argmax(float2 v, uint tid, uint red_size, uint simd)
{
    if (red_size == 1u) return v;
    uint lane = tid % simd;
    uint wave_id = tid / simd;

    float2 acc = v;
    for (uint offset = simd >> 1u; offset > 0u; offset >>= 1u) {
        uint _src = (lane + offset) & (simd - 1u);
        float other_val = WaveReadLaneAt(acc.x, _src);
        float other_idx = WaveReadLaneAt(acc.y, _src);
        if (lane + offset < red_size) {
            bool take = (other_val > acc.x) ||
                        (other_val == acc.x && other_idx < acc.y);
            if (take) {
                acc.x = other_val;
                acc.y = other_idx;
            }
        }
    }

    uint n_waves = (red_size + simd - 1u) / simd;
    if (n_waves == 1u) return acc;

    if (lane == 0u) {
        _wg_argxx_smem_val[wave_id] = acc.x;
        _wg_argxx_smem_idx[wave_id] = acc.y;
    }
    GroupMemoryBarrierWithGroupSync();

    if (wave_id == 0u) {
        float2 r;
        r.x = (tid < n_waves) ? _wg_argxx_smem_val[tid] : -3.402823e38f;
        r.y = (tid < n_waves) ? _wg_argxx_smem_idx[tid] : 0.0f;
        for (uint offset = simd >> 1u; offset > 0u; offset >>= 1u) {
            uint _src = (lane + offset) & (simd - 1u);
            float other_val = WaveReadLaneAt(r.x, _src);
            float other_idx = WaveReadLaneAt(r.y, _src);
            if (lane + offset < n_waves) {
                bool take = (other_val > r.x) ||
                            (other_val == r.x && other_idx < r.y);
                if (take) {
                    r.x = other_val;
                    r.y = other_idx;
                }
            }
        }
        _wg_argxx_smem_val[tid] = r.x;
        _wg_argxx_smem_idx[tid] = r.y;
    }
    GroupMemoryBarrierWithGroupSync();

    float2 result;
    result.x = _wg_argxx_smem_val[0];
    result.y = _wg_argxx_smem_idx[0];
    return result;
}

[require(spirv, subgroup_shuffle)]
[ForceInline]
public float2 vk_wg_reduce_argmin(float2 v, uint tid, uint red_size, uint simd)
{
    if (red_size == 1u) return v;
    uint lane = tid % simd;
    uint wave_id = tid / simd;

    float2 acc = v;
    for (uint offset = simd >> 1u; offset > 0u; offset >>= 1u) {
        uint _src = (lane + offset) & (simd - 1u);
        float other_val = WaveReadLaneAt(acc.x, _src);
        float other_idx = WaveReadLaneAt(acc.y, _src);
        if (lane + offset < red_size) {
            bool take = (other_val < acc.x) ||
                        (other_val == acc.x && other_idx < acc.y);
            if (take) {
                acc.x = other_val;
                acc.y = other_idx;
            }
        }
    }

    uint n_waves = (red_size + simd - 1u) / simd;
    if (n_waves == 1u) return acc;

    if (lane == 0u) {
        _wg_argxx_smem_val[wave_id] = acc.x;
        _wg_argxx_smem_idx[wave_id] = acc.y;
    }
    GroupMemoryBarrierWithGroupSync();

    if (wave_id == 0u) {
        float2 r;
        r.x = (tid < n_waves) ? _wg_argxx_smem_val[tid] : 3.402823e38f;
        r.y = (tid < n_waves) ? _wg_argxx_smem_idx[tid] : 0.0f;
        for (uint offset = simd >> 1u; offset > 0u; offset >>= 1u) {
            uint _src = (lane + offset) & (simd - 1u);
            float other_val = WaveReadLaneAt(r.x, _src);
            float other_idx = WaveReadLaneAt(r.y, _src);
            if (lane + offset < n_waves) {
                bool take = (other_val < r.x) ||
                            (other_val == r.x && other_idx < r.y);
                if (take) {
                    r.x = other_val;
                    r.y = other_idx;
                }
            }
        }
        _wg_argxx_smem_val[tid] = r.x;
        _wg_argxx_smem_idx[tid] = r.y;
    }
    GroupMemoryBarrierWithGroupSync();

    float2 result;
    result.x = _wg_argxx_smem_val[0];
    result.y = _wg_argxx_smem_idx[0];
    return result;
}
