module norm;
import helpers;
import reduction;

[Differentiable]
[BackwardDerivative(ln_affine_elem_bwd)]
public float ln_affine_elem(float x, no_diff float mean, no_diff float rstd, float w, float b) {
    return (x - mean) * rstd * w + b;
}
void ln_affine_elem_bwd(inout DifferentialPair<float> x, no_diff float mean, no_diff float rstd,
                         inout DifferentialPair<float> w, inout DifferentialPair<float> b,
                         float dout) {
    float dx = dout * rstd * w.p;
    x = diffPair(x.p, x.d + dx);
    w = diffPair(w.p, w.d + dout * (x.p - mean) * rstd);
    b = diffPair(b.p, b.d + dout);
}

// ═══ CG.M4: Fully differentiable norm chains (no no_diff on stats) ═══
// These allow bwd_diff to produce gradients for mean and rstd alongside x,
// weight, and bias.  The caller sums the per-element d_mean and d_rstd
// contributions and feeds them through welford_combine_bwd to get the
// full reduction-coupled input gradient.

[Differentiable]
public float layer_norm_chain(
    float x,       // input element
    float mean,    // per-row mean (differentiable)
    float rstd,    // per-row reciprocal std (differentiable)
    float weight,  // affine weight
    float bias     // affine bias
) {
    return (x - mean) * rstd * weight + bias;
}

[Differentiable]
public float layer_norm_chain_no_affine(
    float x,
    float mean,
    float rstd
) {
    return (x - mean) * rstd;
}

[Differentiable]
public float rms_norm_chain(
    float x,
    float rstd,
    float weight
) {
    return x * rstd * weight;
}

[Differentiable]
public float rms_norm_chain_no_affine(
    float x,
    float rstd
) {
    return x * rstd;
}

[Differentiable]
[BackwardDerivative(ln_no_affine_elem_bwd)]
public float ln_no_affine_elem(float x, no_diff float mean, no_diff float rstd) {
    return (x - mean) * rstd;
}
void ln_no_affine_elem_bwd(inout DifferentialPair<float> x, no_diff float mean, no_diff float rstd,
                            float dout) {
    x = diffPair(x.p, x.d + dout * rstd);
}

[Differentiable]
[BackwardDerivative(rms_affine_elem_bwd)]
public float rms_affine_elem(float x, no_diff float rstd, float w) {
    return x * rstd * w;
}
void rms_affine_elem_bwd(inout DifferentialPair<float> x, no_diff float rstd,
                          inout DifferentialPair<float> w, float dout) {
    x = diffPair(x.p, x.d + dout * rstd * w.p);
    w = diffPair(w.p, w.d + dout * x.p * rstd);
}

[Differentiable]
[BackwardDerivative(rms_no_affine_elem_bwd)]
public float rms_no_affine_elem(float x, no_diff float rstd) {
    return x * rstd;
}
void rms_no_affine_elem_bwd(inout DifferentialPair<float> x, no_diff float rstd,
                             float dout) {
    x = diffPair(x.p, x.d + dout * rstd);
}

public interface INormAffine {
    static float apply(float normalized, uint i, StructuredBuffer<float> weight, StructuredBuffer<float> bias);
};

public struct AffineNone : INormAffine {
    static float apply(float n, uint i, StructuredBuffer<float> w, StructuredBuffer<float> b) { return n; }
};

public struct AffineWeightOnly : INormAffine {
    static float apply(float n, uint i, StructuredBuffer<float> w, StructuredBuffer<float> b) {
        return n * w[i];
    }
};

public struct AffineWeightBias : INormAffine {
    static float apply(float n, uint i, StructuredBuffer<float> w, StructuredBuffer<float> b) {
        return n * w[i] + b[i];
    }
};

// T2.7: the previous local `groupshared float3 _norm_smem[1024]` plus a
// re-implementation of the Welford combine duplicated `reduction.slang`'s
// canonical `wg_welford` / `WelfordResult<float>` (with +32 bank-conflict
// padding). Both row helpers below now route through `import reduction;`,
// so the bank-conflict padding M20 ships everywhere consistently.

public void layer_norm_row<Affine : INormAffine>(
    StructuredBuffer<float> X,
    StructuredBuffer<float> weight,
    StructuredBuffer<float> bias,
    RWStructuredBuffer<float> Y,
    uint row, uint D, float eps,
    uint tid, uint tg
) {
    float v_local = (tid < D) ? X[row * D + tid] : 0.0f;
    WelfordResult<float> wf;
    wf.mean = (tid < D) ? v_local : 0.0f;
    wf.m2 = 0.0f;
    wf.n = (tid < D) ? 1.0f : 0.0f;
    WelfordResult<float> stats = wg_welford(wf, tid, tg);
    float mean = stats.mean;
    float var = stats.n > 0.0f ? stats.m2 / stats.n : 0.0f;
    float rstd = 1.0f / sqrt(var + eps);
    if (tid < D) {
        float n = (X[row * D + tid] - mean) * rstd;
        Y[row * D + tid] = Affine::apply(n, tid, weight, bias);
    }
}

public void rms_norm_row<Affine : INormAffine>(
    StructuredBuffer<float> X,
    StructuredBuffer<float> weight,
    StructuredBuffer<float> bias,
    RWStructuredBuffer<float> Y,
    uint row, uint D, float eps,
    uint tid, uint tg
) {
    float v = (tid < D) ? X[row * D + tid] : 0.0f;
    float sq = (tid < D) ? (v * v) : 0.0f;
    float sum_sq = wg_reduce<OpSum>(sq, tid, tg);
    // tg threads contribute 1.0f each up to D; rest contribute 0.0f.
    float count = wg_reduce<OpSum>((tid < D) ? 1.0f : 0.0f, tid, tg);
    float mean_sq = count > 0.0f ? sum_sq / count : 0.0f;
    float rstd = 1.0f / sqrt(mean_sq + eps);
    if (tid < D) {
        Y[row * D + tid] = Affine::apply(v * rstd, tid, weight, bias);
    }
}
