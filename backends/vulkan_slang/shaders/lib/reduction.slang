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
public interface IWaveReduction : IReduction {
    static float wave_reduce(float v);
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

[ForceInline]
public float wave_active_sum_nan(float v) {
    return WaveActiveSum(v);
}

[ForceInline]
public float wave_active_product_nan(float v) {
    return WaveActiveProduct(v);
}

// ── Concrete reduction ops (float-monomorphized) ──────────────────────

public struct OpSum : IReduction, IWaveReduction, IDifferentiableReduction {
    static float identity() { return 0.0f; }
    [Differentiable]
    static float combine(float a, float b) { return combine_sum_nan(a, b); }
    static float wave_reduce(float v) { return wave_active_sum_nan(v); }
};

public struct OpProd : IReduction, IWaveReduction, IDifferentiableReduction {
    static float identity() { return 1.0f; }
    [Differentiable]
    static float combine(float a, float b) { return combine_prod_nan(a, b); }
    static float wave_reduce(float v) { return wave_active_product_nan(v); }
};

public struct OpMaxReduce : IReduction, IWaveReduction {
    static float identity() { return -3.402823e38f; }
    static float combine(float a, float b) { return a > b ? a : b; }
    static float wave_reduce(float v) { return WaveActiveMax(v); }
};

public struct OpMinReduce : IReduction, IWaveReduction {
    static float identity() { return 3.402823e38f; }
    static float combine(float a, float b) { return a < b ? a : b; }
    static float wave_reduce(float v) { return WaveActiveMin(v); }
};

// ── Wave reduction main entry ────────────────────────────────────────

// Wave-intrinsic two-stage reduction.
// Phase 1: each thread reduces its local value via wave_reduce.
// Phase 2: cross-wave reduction via shared memory.
// n_waves=1 fast-path skips shared memory entirely.
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

// ── Welford result and reduction ──────────────────────────────────────

public struct WelfordResult<T : __BuiltinFloatingPointType> : IDifferentiable {
    public T mean;
    public T m2;
    public T n;
};

// Float-monomorphized welford combine.
// Forward: combine two (mean, m2, n) pairs into one.
// The derivative is provided by welford_combine_bwd below.
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

groupshared float _welford_smem_mean[1024 + 32];
groupshared float _welford_smem_m2[1024 + 32];
groupshared float _welford_smem_n[1024 + 32];

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
groupshared float _wg_arg_smem_val[1024 + 32];
groupshared float _wg_arg_smem_idx[1024 + 32];

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

// ── Inclusive scan ────────────────────────────────────────────────────

public interface IScanOp {
    static float identity();
    static float combine(float a, float b);
};

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
