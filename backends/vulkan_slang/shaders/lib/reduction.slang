module reduction;
import helpers;

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

[Differentiable]
[ForceInline]
public float combine_sum_nan(float a, float b) {
    if (isnan(a)) return a;
    if (isnan(b)) return b;
    return a + b;
}

[Differentiable]
[ForceInline]
public float combine_prod_nan(float a, float b) {
    if (isnan(a)) return a;
    if (isnan(b)) return b;
    return a * b;
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

// Convenience wrapper — 3-arg form used by norm.slang and other lib
// modules. Computes n_waves and simd from size, assuming wave64.
[ForceInline]
public float wg_reduce<W : IWaveReduction>(float v, uint tid, uint size)
{
    uint simd = 64u;
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
[ForceInline]
public WelfordResult<float> wg_welford(
    WelfordResult<float> v, uint tid, uint size)
{
    if (size == 1u)
        return v;

    uint lane = tid % 64u;
    uint wave_id = tid / 64u;
    uint n_waves = (size + 63u) / 64u;

    // Wave-level reduction
    WelfordResult<float> acc = v;
    for (uint offset = 32u; offset > 0u; offset >>= 1u) {
        float other_n = WaveReadLaneAt(acc.n, lane + offset);
        if (lane + offset < size) {
            float n_total = acc.n + other_n;
            float delta = WaveReadLaneAt(acc.mean, lane + offset) - acc.mean;
            acc.mean = acc.mean + delta * other_n / n_total;
            acc.m2 = acc.m2 + WaveReadLaneAt(acc.m2, lane + offset) + delta * delta * acc.n * other_n / n_total;
            acc.n = n_total;
        }
    }

    if (n_waves == 1u)
        return acc;

    if (lane == 0u) {
        _welford_smem_mean[wave_id] = acc.mean;
        _welford_smem_m2[wave_id] = acc.m2;
        _welford_smem_n[wave_id] = acc.n;
    }

    GroupMemoryBarrierWithGroupSync();

    if (wave_id == 0u) {
        WelfordResult<float> r;
        r.mean = (tid < n_waves) ? _welford_smem_mean[tid] : 0.0f;
        r.m2 = (tid < n_waves) ? _welford_smem_m2[tid] : 0.0f;
        r.n = (tid < n_waves) ? _welford_smem_n[tid] : 0.0f;

        for (uint offset = 32u; offset > 0u; offset >>= 1u) {
            float other_n = WaveReadLaneAt(r.n, lane + offset);
            if (lane + offset < n_waves && other_n > 0.0f) {
                float n_total = r.n + other_n;
                float delta = WaveReadLaneAt(r.mean, lane + offset) - r.mean;
                r.mean = r.mean + delta * other_n / n_total;
                r.m2 = r.m2 + WaveReadLaneAt(r.m2, lane + offset) + delta * delta * r.n * other_n / n_total;
                r.n = n_total;
            }
        }

        _welford_smem_mean[tid] = r.mean;
        _welford_smem_m2[tid] = r.m2;
        _welford_smem_n[tid] = r.n;
    }

    GroupMemoryBarrierWithGroupSync();

    WelfordResult<float> result;
    result.mean = _welford_smem_mean[0];
    result.m2 = _welford_smem_m2[0];
    result.n = _welford_smem_n[0];
    return result;
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
public ArgPair wg_argmax(ArgPair v, uint tid, uint size)
{
    if (size == 1u) return v;
    uint lane = tid % 64u;
    uint wave_id = tid / 64u;

    ArgPair acc = v;
    for (uint offset = 32u; offset > 0u; offset >>= 1u) {
        float other_val = WaveReadLaneAt(acc.val, lane + offset);
        if (lane + offset < size && other_val > acc.val) {
            acc.val = other_val;
            acc.idx = WaveReadLaneAt(acc.idx, lane + offset);
        }
    }

    uint n_waves = (size + 63u) / 64u;
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
        for (uint offset = 32u; offset > 0u; offset >>= 1u) {
            float other_val = WaveReadLaneAt(result.val, lane + offset);
            if (lane + offset < n_waves && other_val > result.val) {
                result.val = other_val;
                result.idx = WaveReadLaneAt(result.idx, lane + offset);
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

// ── Inclusive scan ────────────────────────────────────────────────────

public interface IScanOp {
    static float identity();
    static float combine(float a, float b);
};

// CG.M16: wg_scan_inclusive uses WaveReadLaneAt which requires subgroup_shuffle.
[require(spirv, subgroup_shuffle)]
[ForceInline]
public float wg_scan_inclusive<S : IScanOp>(float v, uint tid, uint size, uint simd)
{
    if (size == 1u) return v;
    uint lane = tid % simd;
    float acc = v;
    for (uint offset = 1u; offset < simd; offset <<= 1u) {
        float other = WaveReadLaneAt(acc, lane - offset);
        if (lane >= offset)
            acc = S::combine(acc, other);
    }
    return acc;
}
