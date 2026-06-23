module pointwise;
import helpers;

public interface IPointwise {
    static float apply(float x);
};

public interface IPointwiseBinary {
    static float apply(float a, float b);
};

// CG.M10 — Differentiable scalar epilogue interface.
// Any struct implementing IDifferentiable has a [Differentiable] apply
// method. When a template is parameterized over <Epilogue : IDifferentiable>,
// Slang's autodiff can trace through Epilogue::apply(x) without needing to
// know the concrete type at compile time.
public interface IDifferentiable {
    [Differentiable]
    static float apply(float x);
};

[BackwardDerivative(relu_fast_bwd)]
public float relu_fwd(float x) { return max(x, 0.0f); }

void relu_fast_bwd(inout DifferentialPair<float> x, float dout) {
    float xv = x.p;
    x = diffPair(xv, xv > 0.0f ? dout : 0.0f);
}

// Hand-written `sigmoid_fast_bwd` reuses the forward's own output instead of
// re-evaluating `exp(-x)` — typical hot-path optimization. Registered as the
// custom backward via `[BackwardDerivative]` per P2.4.
[BackwardDerivative(sigmoid_fast_bwd)]
public float sigmoid_fwd(float x) { return 1.0f / (1.0f + exp(-x)); }

void sigmoid_fast_bwd(inout DifferentialPair<float> x, float dout) {
    float y = 1.0f / (1.0f + exp(-x.p));
    x = diffPair(x.p, dout * y * (1.0f - y));
}

// tanh'(x) = 1 - tanh(x)^2. The eager `tanh_backward.slang` saves the
// forward output and avoids recomputing tanh; `[BackwardDerivative]`
// gives us the input `x` (not the output), so the lib bwd recomputes
// `tanh(x)`. Slang's `tanh` lowers to a hardware-accelerated intrinsic
// on most targets — the recompute cost is negligible vs the dispatch
// fragmentation eliminated by routing through `lib`.
[BackwardDerivative(tanh_fast_bwd)]
public float tanh_fwd(float x) { return tanh(x); }

void tanh_fast_bwd(inout DifferentialPair<float> x, float dout) {
    float t = tanh(x.p);
    x = diffPair(x.p, dout * (1.0f - t * t));
}

// Hand-written `gelu_fast_bwd` applies the analytic tanh-approx GELU
// derivative directly. Registered as the custom backward via
// `[BackwardDerivative]` per PF.11 (eager-shader retirement).
// Forward formula: gelu(x) = 0.5 * x * (1 + tanh(c*(x + 0.044715*x^3)))
//   with c = sqrt(2/pi) = 0.7978845608028654.
// Let u = c*(x + 0.044715*x^3); t = tanh(u); sech^2(u) = 1 - t^2;
//   du/dx = c*(1 + 3*0.044715*x^2). Then
//   d(gelu)/dx = 0.5*(1+t) + 0.5*x*(1-t^2)*c*(1 + 3*0.044715*x^2).
// `inner` is clamped to ±88 to avoid tanh overflow NaN on some
// drivers — same guard the eager `gelu_backward.slang` uses.
[BackwardDerivative(gelu_fast_bwd)]
public float gelu_fwd(float x) {
    // erf is not yet differentiable in helpers, so a tanh-approx GELU is used
    // for the autodiff path; the inline expression matches PyTorch's
    // `gelu(x, approximate="tanh")`.
    float c = 0.7978845608028654f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanh(c * (x + 0.044715f * x3)));
}

void gelu_fast_bwd(inout DifferentialPair<float> x, float dout) {
    float c = 0.7978845608028654f;
    float xv = x.p;
    float inner = c * (xv + 0.044715f * xv * xv * xv);
    inner = clamp(inner, -88.0f, 88.0f);
    float t = tanh(inner);
    float sech2 = 1.0f - t * t;
    float d_inner = c * (1.0f + 3.0f * 0.044715f * xv * xv);
    float grad = 0.5f * (1.0f + t) + 0.5f * xv * sech2 * d_inner;
    x = diffPair(xv, dout * grad);
}

// SiLU(x) = x * sigmoid(x); SiLU'(x) = sigmoid(x) * (1 + x - x*sigmoid(x)).
// Hand-rolled `silu_fast_bwd` matches the eager
// `shaders/activation/silu_backward.slang` formula. Registered as the
// custom backward via `[BackwardDerivative]` per PF.11 (eager-shader
// retirement).
[BackwardDerivative(silu_fast_bwd)]
public float silu_fwd(float x) { return x / (1.0f + exp(-x)); }

void silu_fast_bwd(inout DifferentialPair<float> x, float dout) {
    float xv = x.p;
    float sig = 1.0f / (1.0f + exp(-xv));
    float deriv = sig * (1.0f + xv - xv * sig);
    x = diffPair(xv, dout * deriv);
}

// ELU'(x) = 1 if x > 0 else exp(x). Since the `[BackwardDerivative]`
// works on the primal x (not the output), we compute exp(xv) directly
// for the negative branch — this is cheaper than the auto-diff
// save/restore chain that Slang would otherwise emit.
[BackwardDerivative(elu_fast_bwd)]
public float elu_fwd(float x) { return (x > 0.0f) ? x : (exp(x) - 1.0f); }

void elu_fast_bwd(inout DifferentialPair<float> x, float dout) {
    float xv = x.p;
    x = diffPair(xv, xv > 0.0f ? dout : dout * exp(xv));
}

// HardSwish'(x): piecewise — 0 when x<-3, (2x+3)/6 when -3≤x≤3, 1 when x>3.
[BackwardDerivative(hardswish_fast_bwd)]
public float hardswish_fwd(float x) {
    return x * clamp(x + 3.0f, 0.0f, 6.0f) * (1.0f / 6.0f);
}

void hardswish_fast_bwd(inout DifferentialPair<float> x, float dout) {
    float xv = x.p;
    float grad = 0.0f;
    if (xv > 3.0f) grad = 1.0f;
    else if (xv > -3.0f) grad = (2.0f * xv + 3.0f) * (1.0f / 6.0f);
    x = diffPair(xv, dout * grad);
}

// HardSigmoid'(x): piecewise — 0 when |x|>3, 1/6 when -3≤x≤3.
[BackwardDerivative(hardsigmoid_fast_bwd)]
public float hardsigmoid_fwd(float x) {
    return clamp(x + 3.0f, 0.0f, 6.0f) * (1.0f / 6.0f);
}

void hardsigmoid_fast_bwd(inout DifferentialPair<float> x, float dout) {
    float xv = x.p;
    float grad = (xv > -3.0f && xv < 3.0f) ? (1.0f / 6.0f) : 0.0f;
    x = diffPair(xv, dout * grad);
}

[BackwardDerivative(softplus_fast_bwd)]
public float softplus_fwd(float x, no_diff float beta, no_diff float threshold) {
    float bx = beta * x;
    float stable = max(x, 0.0f) + log(1.0f + exp(-abs(bx))) / beta;
    return min(stable, threshold);
}

void softplus_fast_bwd(inout DifferentialPair<float> x, no_diff float beta, no_diff float threshold, float dout) {
    float xv = x.p;
    float bx = beta * xv;
    float sig = 1.0f / (1.0f + exp(-bx));
    float grad = (bx > threshold) ? 1.0f : sig;
    x = diffPair(xv, dout * grad);
}

// Mish'(x) = tanh(sp) + x * sech^2(sp) * sigmoid(x)
//   where sp = softplus(x) = log(1 + exp(x)).
//   For large |x| this approaches identity (grad→1) or zero (grad→0),
//   which matches the analytic limit without special-casing.
[BackwardDerivative(mish_fast_bwd)]
public float mish_fwd(float x) {
    return x * tanh(log(1.0f + exp(x)));
}

void mish_fast_bwd(inout DifferentialPair<float> x, float dout) {
    float xv = x.p;
    float sp = log(1.0f + exp(xv));
    float th = tanh(sp);
    float sech2 = 1.0f - th * th;
    float sig = 1.0f / (1.0f + exp(-xv));
    float grad = th + xv * sech2 * sig;
    x = diffPair(xv, dout * grad);
}

// LeakyReLU(x, alpha) = x if x >= 0 else alpha * x.
// Derivative = 1 if x >= 0 else alpha.
// `alpha` is marked `no_diff` since it's a hyperparameter, not a trainable weight.
[BackwardDerivative(leaky_relu_fast_bwd)]
public float leaky_relu_fwd(float x, no_diff float alpha) {
    return x >= 0.0f ? x : alpha * x;
}

void leaky_relu_fast_bwd(inout DifferentialPair<float> x, no_diff float alpha, float dout) {
    float xv = x.p;
    float grad = xv >= 0.0f ? 1.0f : alpha;
    x = diffPair(xv, dout * grad);
}

// ── CG.M9 fast backward derivatives ───────────────────────────────────
// Hand-written closed-form backward functions for hot-path elementals.
// Each avoids the extra temporaries and recomputation that Slang's
// autodiff (bwd_diff) would otherwise emit.

// expm1'(x) = exp(x) — simpler than autodiff chain through expm1 impl
void expm1_fast_bwd(inout DifferentialPair<float> x, float dout) {
    x = diffPair(x.p, x.d + dout * exp(x.p));
}

// log1p'(x) = 1/(1+x) — simpler than autodiff chain through log1p impl
void log1p_fast_bwd(inout DifferentialPair<float> x, float dout) {
    x = diffPair(x.p, x.d + dout / (1.0f + x.p));
}

// |x|' = sign(x) — avoids autodiff's x/|x| division
void abs_fast_bwd(inout DifferentialPair<float> x, float dout) {
    float xv = x.p;
    x = diffPair(xv, x.d + dout * sign(xv));
}

// (-x)' = -1 — trivial, skips autodiff framing overhead
void neg_fast_bwd(inout DifferentialPair<float> x, float dout) {
    x = diffPair(x.p, x.d - dout);
}

// sqrt'(x) = 1/(2*sqrt(x)). Recomputes sqrt(x.p) since
// [BackwardDerivative] sees the primal x, not the output.
void sqrt_fast_bwd(inout DifferentialPair<float> x, float dout) {
    float s = sqrt(x.p);
    float grad = s > 0.0f ? dout / (2.0f * s) : 0.0f;
    x = diffPair(x.p, x.d + grad);
}

// (1/x)' = -1/x^2
void reciprocal_fast_bwd(inout DifferentialPair<float> x, float dout) {
    float xv = x.p;
    float grad = xv != 0.0f ? -dout / (xv * xv) : 0.0f;
    x = diffPair(xv, x.d + grad);
}

// erf'(x) = 2/sqrt(pi) * exp(-x^2) — dramatically simpler than autodiff
// tracing through erf's polynomial approximation in special_math
void erf_fast_bwd(inout DifferentialPair<float> x, float dout) {
    float xv = x.p;
    float grad = dout * 2.0f / sqrt(3.141592653589793f) * exp(-xv * xv);
    x = diffPair(xv, x.d + grad);
}

// erfc'(x) = -2/sqrt(pi) * exp(-x^2)
void erfc_fast_bwd(inout DifferentialPair<float> x, float dout) {
    float xv = x.p;
    float grad = -dout * 2.0f / sqrt(3.141592653589793f) * exp(-xv * xv);
    x = diffPair(xv, x.d + grad);
}

// erfinv'(x) = sqrt(pi)/2 * exp(erfinv(x)^2)
void erfinv_fast_bwd(inout DifferentialPair<float> x, float dout) {
    float xv = x.p;
    float y = xv.erfinv();
    x = diffPair(xv, x.d + dout * 0.886226925f * exp(y * y));
}

// lgamma'(x) = digamma(x)
void lgamma_fast_bwd(inout DifferentialPair<float> x, float dout) {
    float xv = x.p;
    x = diffPair(xv, x.d + dout * xv.digamma());
}

// digamma'(x) = polygamma(1, x) = trigamma(x)
void digamma_fast_bwd(inout DifferentialPair<float> x, float dout) {
    float xv = x.p;
    x = diffPair(xv, x.d + dout * xv.polygamma(1.0f));
}

// ndtri'(x) = sqrt(2*pi) * exp(ndtri(x)^2 / 2)
void ndtri_fast_bwd(inout DifferentialPair<float> x, float dout) {
    float xv = x.p;
    float y = xv.ndtri();
    x = diffPair(xv, x.d + dout * 2.506628274f * exp(0.5f * y * y));
}

// i0'(x) = i1(x)
void i0_fast_bwd(inout DifferentialPair<float> x, float dout) {
    float xv = x.p;
    x = diffPair(xv, x.d + dout * xv.i1());
}

// i0e'(x) = i1e(x) - sign(x) * i0e(x)
void i0e_fast_bwd(inout DifferentialPair<float> x, float dout) {
    float xv = x.p;
    float sgn = xv > 0.0f ? 1.0f : (xv < 0.0f ? -1.0f : 0.0f);
    x = diffPair(xv, x.d + dout * (xv.i1e() - sgn * xv.i0e()));
}

// i1'(x) = i0(x) - i1(x)/x; limit at x=0 is 0.5
void i1_fast_bwd(inout DifferentialPair<float> x, float dout) {
    float xv = x.p;
    float grad;
    if (abs(xv) < 1e-6f) {
        grad = 0.5f;
    } else {
        grad = xv.i0() - xv.i1() / xv;
    }
    x = diffPair(xv, x.d + dout * grad);
}

// i1e'(x) = i0e(x) - i1e(x)*(1/x + sign(x)); limit at x=0 is 0.5
void i1e_fast_bwd(inout DifferentialPair<float> x, float dout) {
    float xv = x.p;
    float grad;
    if (abs(xv) < 1e-6f) {
        grad = 0.5f;
    } else {
        float sgn = xv > 0.0f ? 1.0f : -1.0f;
        grad = xv.i0e() - xv.i1e() * (1.0f / xv + sgn);
    }
    x = diffPair(xv, x.d + dout * grad);
}

// atan2'(y,x):  d/dy = x/(x^2+y^2),  d/dx = -y/(x^2+y^2)
// Two-argument hand-written backward avoids autodiff emitting an
// atan2 decomposition with multiple temporaries.
void atan2_fast_bwd(inout DifferentialPair<float> y,
                    inout DifferentialPair<float> x,
                    float dout) {
    float yv = y.p;
    float xv = x.p;
    float denom = xv * xv + yv * yv;
    float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
    float dy = dout * xv * inv;
    float dx = dout * (-yv) * inv;
    y = diffPair(yv, y.d + dy);
    x = diffPair(xv, x.d + dx);
}

// hypot'(a,b):  d/da = a/hypot(a,b),  d/db = b/hypot(a,b)
// Reuses hypot result to avoid recomputation.
void hypot_fast_bwd(inout DifferentialPair<float> a,
                    inout DifferentialPair<float> b,
                    float dout) {
    float av = a.p;
    float bv = b.p;
    float h = hypot(av, bv);
    float inv = h > 0.0f ? 1.0f / h : 0.0f;
    float da = dout * av * inv;
    float db = dout * bv * inv;
    a = diffPair(av, a.d + da);
    b = diffPair(bv, b.d + db);
}

// max(a,b)' = (a >= b) ? (1, 0) : (0, 1). Ties route gradient to `a`,
// matching Slang's autodiff convention and PyTorch's `aten.maximum`
// backward when operands differ. The hand-written form skips the
// `DifferentialPair<max>` round-trip the autodiff path produces.
void max_fast_bwd(inout DifferentialPair<float> a,
                  inout DifferentialPair<float> b,
                  float dout) {
    float av = a.p;
    float bv = b.p;
    bool a_wins = av >= bv;
    float da = a_wins ? dout : 0.0f;
    float db = a_wins ? 0.0f : dout;
    a = diffPair(av, a.d + da);
    b = diffPair(bv, b.d + db);
}

// min(a,b)' = (a <= b) ? (1, 0) : (0, 1). Symmetric to max_fast_bwd;
// ties route gradient to `a`.
void min_fast_bwd(inout DifferentialPair<float> a,
                  inout DifferentialPair<float> b,
                  float dout) {
    float av = a.p;
    float bv = b.p;
    bool a_wins = av <= bv;
    float da = a_wins ? dout : 0.0f;
    float db = a_wins ? 0.0f : dout;
    a = diffPair(av, a.d + da);
    b = diffPair(bv, b.d + db);
}

// Identity epilogue: passes through unchanged. Used by matmul/bmm/addmm

// templates when no activation epilogue is requested, so the entry point
// can always be generic over `Epilogue : IPointwise` without branching.
public struct OpIdentity : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return x; }
};

public struct OpReLU : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return relu_fwd(x); }
};

public struct OpSigmoid : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return sigmoid_fwd(x); }
};

public struct OpTanh : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return tanh_fwd(x); }
};

public struct OpGELU : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) {
        // Inline erf(x*0.707...) to avoid extension method call (slangc 2026.5 compat)
        float t = x * 0.7071067811865475f;
        float absx = abs(t);
        float r = 1.0f / (1.0f + 0.3275911f * absx);
        float y = 1.0f - ((((
            1.061405429f * r - 1.453152027f) * r
            + 1.421413741f) * r - 0.284496736f) * r
            + 0.254829592f) * r * exp(-absx * absx);
        float erf_val = (t < 0.0f) ? -y : y;
        return 0.5f * x * (1.0f + erf_val);
    }
};

public struct OpSiLU : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return silu_fwd(x); }
};

public struct OpELU : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return elu_fwd(x); }
};

public struct OpHardSwish : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return hardswish_fwd(x); }
};

public struct OpHardSigmoid : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return hardsigmoid_fwd(x); }
};

public struct OpMish : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return mish_fwd(x); }
};

public struct OpSoftplus : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return softplus_fwd(x, 1.0f, 20.0f); }
};

public struct OpLeakyReLU : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) {
        // Default alpha = 0.01 (PyTorch default negative_slope).
        // Callers needing a different alpha should use leaky_relu_fwd directly.
        return leaky_relu_fwd(x, 0.01f);
    }
};

public struct OpRelu6 : IPointwise {
    static float apply(float x) { return clamp(x, 0.0f, 6.0f); }
};

public struct OpAbs : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return abs_fwd(x); }
};

public struct OpNeg : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return neg_fwd(x); }
};

public struct OpExp : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return exp_fwd(x); }
};

public struct OpLog : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return log_fwd(x); }
};

public struct OpSqrt : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return sqrt_fwd(x); }
};

public struct OpRsqrt : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return rsqrt_fwd(x); }
};

public struct OpReciprocal : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return reciprocal_fwd(x); }
};

public struct OpAdd : IPointwiseBinary {
    static float apply(float a, float b) { return a + b; }
};

public struct OpSub : IPointwiseBinary {
    static float apply(float a, float b) { return a - b; }
};

public struct OpMul : IPointwiseBinary {
    static float apply(float a, float b) { return a * b; }
};

public struct OpDiv : IPointwiseBinary {
    static float apply(float a, float b) { return a / b; }
};

public struct OpMin : IPointwiseBinary {
    static float apply(float a, float b) { return min_fwd(a, b); }
};

public struct OpMax : IPointwiseBinary {
    static float apply(float a, float b) { return max_fwd(a, b); }
};

public struct OpPow : IPointwiseBinary {
    static float apply(float a, float b) { return pow_fwd(a, b); }
};

// ── [Differentiable] basic trig ───────────────────────────────────────

[Differentiable]
public float sin_fwd(float x) { return sin(x); }

[Differentiable]
public float cos_fwd(float x) { return cos(x); }

[Differentiable]
public float tan_fwd(float x) { return tan(x); }

[Differentiable]
public float asin_fwd(float x) { return asin(x); }

[Differentiable]
public float acos_fwd(float x) { return acos(x); }

[Differentiable]
public float atan_fwd(float x) { return atan(x); }

// ── [Differentiable] hyperbolic ───────────────────────────────────────

[Differentiable]
public float sinh_fwd(float x) { return sinh(x); }

[Differentiable]
public float cosh_fwd(float x) { return cosh(x); }

[Differentiable]
public float asinh_fwd(float x) { return asinh(x); }

[Differentiable]
public float acosh_fwd(float x) { return acosh(x); }

[Differentiable]
public float atanh_fwd(float x) { return atanh(x); }

public struct OpCos : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return cos_fwd(x); }
};

public struct OpSin : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return sin_fwd(x); }
};

public struct OpCeil : IPointwise {
    static float apply(float x) { return ceil(x); }
};

public struct OpFloor : IPointwise {
    static float apply(float x) { return floor(x); }
};

public struct OpRound : IPointwise {
    static float apply(float x) { return round(x); }
};

public struct OpSign : IPointwise {
    static float apply(float x) { return sign(x); }
};

// ── [Differentiable] exp / log / power ────────────────────────────────

[Differentiable]
public float exp_fwd(float x) { return exp(x); }

[BackwardDerivative(expm1_fast_bwd)]
public float expm1_fwd(float x) { return x.expm1(); }

[Differentiable]
public float exp2_fwd(float x) { return exp2(x); }

[Differentiable]
public float log_fwd(float x) { return log(x); }

[Differentiable]
public float log2_fwd(float x) { return log2(x); }

[Differentiable]
public float log10_fwd(float x) { return log10(x); }

[BackwardDerivative(log1p_fast_bwd)]
public float log1p_fwd(float x) { return x.log1p(); }

[BackwardDerivative(sqrt_fast_bwd)]
public float sqrt_fwd(float x) { return sqrt(x); }

[Differentiable]
public float rsqrt_fwd(float x) { return 1.0f / sqrt(x); }

[BackwardDerivative(reciprocal_fast_bwd)]
public float reciprocal_fwd(float x) { return 1.0f / x; }

// ── [Differentiable] abs / neg ────────────────────────────────────────

[BackwardDerivative(abs_fast_bwd)]
public float abs_fwd(float x) { return abs(x); }

[BackwardDerivative(neg_fast_bwd)]
public float neg_fwd(float x) { return -x; }

// ── [Differentiable] binary pointwise (CG.M2) ──────────────────────

[Differentiable]
public float pow_fwd(float a, float b) { return pow(a, b); }

[BackwardDerivative(atan2_fast_bwd)]
public float atan2_fwd(float y, float x) { return atan2(y, x); }

[BackwardDerivative(hypot_fast_bwd)]
public float hypot_fwd(float a, float b) { return hypot(a, b); }

[Differentiable]
public float copysign_fwd(float mag, no_diff float sign) { return copysign(mag, sign); }

[BackwardDerivative(max_fast_bwd)]
public float max_fwd(float a, float b) { return max(a, b); }

[BackwardDerivative(min_fast_bwd)]
public float min_fwd(float a, float b) { return min(a, b); }

// ── [Differentiable] special functions (helpers.slang → special_math) ─

[BackwardDerivative(erf_fast_bwd)]
public float erf_fwd(float x) { return (x).erf(); }

[BackwardDerivative(erfc_fast_bwd)]
public float erfc_fwd(float x) { return 1.0f - (x).erf(); }

[BackwardDerivative(erfinv_fast_bwd)]
public float erfinv_fwd(float x) { return (x).erfinv(); }
[BackwardDerivative(lgamma_fast_bwd)]
public float lgamma_fwd(float x) { return (x).lgamma(); }
[BackwardDerivative(digamma_fast_bwd)]
public float digamma_fwd(float x) { return (x).digamma(); }
[BackwardDerivative(ndtri_fast_bwd)]
public float ndtri_fwd(float x) { return (x).ndtri(); }
[BackwardDerivative(i0_fast_bwd)]
public float i0_fwd(float x) { return (x).i0(); }
[BackwardDerivative(i0e_fast_bwd)]
public float i0e_fwd(float x) { return (x).i0e(); }
[BackwardDerivative(i1_fast_bwd)]
public float i1_fwd(float x) { return (x).i1(); }
[BackwardDerivative(i1e_fast_bwd)]
public float i1e_fwd(float x) { return (x).i1e(); }

// ── CG.M22.6: GroupNorm affine elemental ────────────────────────────────
// Per-element GN affine transform: y = (x - mean) * rstd * gamma + bias.
// Only x is differentiated; mean, rstd, gamma, bias are no_diff because
// their gradients require a reduction over the normalized dimension.
// d_weight / d_bias are computed via the fused group_norm_backward shader.
[Differentiable]
[BackwardDerivative(gn_affine_bwd)]
public float gn_affine_fwd(
    float x,
    no_diff float mean,
    no_diff float rstd,
    no_diff float gamma,
    no_diff float bias
) {
    return (x - mean) * rstd * gamma + bias;
}

void gn_affine_bwd(
    inout DifferentialPair<float> x,
    no_diff float mean,
    no_diff float rstd,
    no_diff float gamma,
    no_diff float bias,
    float dout
) {
    float xv = x.p;
    // dy/dx = rstd * gamma, so dx = dout * rstd * gamma
    x = diffPair(xv, dout * rstd * gamma);
}

public struct OpTan : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return tan_fwd(x); }
};

public struct OpAtan : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return atan_fwd(x); }
};

public struct OpLog2 : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return log2_fwd(x); }
};

public struct OpLog10 : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return log10_fwd(x); }
};

public struct OpLog1p : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return log1p_fwd(x); }
};

public struct OpExp2 : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return exp2_fwd(x); }
};

public struct OpExpm1 : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return expm1_fwd(x); }
};

public struct OpAcos : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return acos_fwd(x); }
};

public struct OpAsin : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return asin_fwd(x); }
};

public struct OpCosh : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return cosh_fwd(x); }
};

public struct OpSinh : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return sinh_fwd(x); }
};

public struct OpAsinh : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return asinh_fwd(x); }
};

public struct OpAcosh : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return acosh_fwd(x); }
};

public struct OpAtanh : IPointwise, IDifferentiable {
    [Differentiable]
    static float apply(float x) { return atanh_fwd(x); }
};

public struct OpTrunc : IPointwise {
    static float apply(float x) { return trunc(x); }
};

public struct OpFrac : IPointwise {
    static float apply(float x) { return frac(x); }
};

public struct OpLogicalNot : IPointwise {
    static float apply(float x) { return x == 0.0f ? 1.0f : 0.0f; }
};

public struct OpBitwiseNot : IPointwise {
    static float apply(float x) { return float(~int(x)); }
};

public struct OpFmod : IPointwiseBinary {
    static float apply(float a, float b) { return fmod(a, b); }
};

public struct OpRemainder : IPointwiseBinary {
    static float apply(float a, float b) { return a - b * round(a / b); }
};

public struct OpAtan2 : IPointwiseBinary {
    static float apply(float a, float b) { return atan2_fwd(a, b); }
};

public struct OpHypot : IPointwiseBinary {
    static float apply(float a, float b) { return hypot_fwd(a, b); }
};

public struct OpNextafter : IPointwiseBinary {
    static float apply(float a, float b) { return nextafter(a, b); }
};

public struct OpCopysign : IPointwiseBinary {
    static float apply(float a, float b) { return copysign_fwd(a, b); }
};

// ── Complex-valued pointwise interfaces ────────────────────────────────────
// Complex numbers are stored as interleaved float2 (real=x, imag=y) in the
// SSBO.  Each complex element occupies one float2 slot.  The complex tensor
// numel is the number of complex elements (not the number of float scalars).

public interface IComplexPointwise {
    static float2 apply(float2 x);
};

public interface IComplexPointwiseBinary {
    static float2 apply(float2 a, float2 b);
};

// Complex addition: component-wise (a.x + b.x, a.y + b.y)
public struct OpComplexAdd : IComplexPointwiseBinary {
    static float2 apply(float2 a, float2 b) {
        return a + b;
    }
};

// Complex subtraction: component-wise (a.x - b.x, a.y - b.y)
public struct OpComplexSub : IComplexPointwiseBinary {
    static float2 apply(float2 a, float2 b) {
        return a - b;
    }
};

// Complex multiplication: (a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x)
public struct OpComplexMul : IComplexPointwiseBinary {
    static float2 apply(float2 a, float2 b) {
        return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }
};

// Complex division: a / b = (a * conj(b)) / |b|^2
public struct OpComplexDiv : IComplexPointwiseBinary {
    static float2 apply(float2 a, float2 b) {
        float denom = b.x * b.x + b.y * b.y;
        float inv = denom != 0.0f ? 1.0f / denom : 0.0f;
        return float2(
            (a.x * b.x + a.y * b.y) * inv,
            (a.y * b.x - a.x * b.y) * inv
        );
    }
};

// Complex conjugate: (x.x, -x.y)
public struct OpComplexConj : IComplexPointwise {
    static float2 apply(float2 x) {
        return float2(x.x, -x.y);
    }
};

// Complex absolute value: sqrt(x.x^2 + x.y^2).  Returns (|x|, 0): the
// caller is responsible for extracting the real component when writing
// back to a real-valued output buffer.
public struct OpComplexAbs : IComplexPointwise {
    static float2 apply(float2 x) {
        return float2(sqrt(x.x * x.x + x.y * x.y), 0.0f);
    }
};

// ── Complex dispatch helpers ───────────────────────────────────────────────

[ForceInline]
public void pointwise_complex_unary_apply<Op : IComplexPointwise>(
    StructuredBuffer<float2> input,
    RWStructuredBuffer<float2> output,
    uint idx, uint n
) {
    if (idx < n) output[idx] = Op::apply(input[idx]);
}

[ForceInline]
public void pointwise_complex_binary_apply<Op : IComplexPointwiseBinary>(
    StructuredBuffer<float2> in_a,
    StructuredBuffer<float2> in_b,
    RWStructuredBuffer<float2> output,
    uint idx, uint n
) {
    if (idx < n) output[idx] = Op::apply(in_a[idx], in_b[idx]);
}

// ── Float (real-valued) dispatch ───────────────────────────────────────────

[ForceInline]
public void pointwise_unary_apply<Op : IPointwise>(
    StructuredBuffer<float> input,
    RWStructuredBuffer<float> output,
    uint idx, uint n
) {
    if (idx < n) output[idx] = Op::apply(input[idx]);
}

[ForceInline]
public void pointwise_binary_apply<Op : IPointwiseBinary>(
    StructuredBuffer<float> in_a,
    StructuredBuffer<float> in_b,
    RWStructuredBuffer<float> output,
    uint idx, uint n
) {
    if (idx < n) output[idx] = Op::apply(in_a[idx], in_b[idx]);
}
