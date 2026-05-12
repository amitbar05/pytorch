module special_math;

// Special-function scalar math (Cephes / Numerical Recipes single-precision
// polynomials). Split out of `helpers.slang` in T2.6 (2026-05-08).
// Backward-compat: `helpers.slang` re-exports this module via
// `import special_math;` so existing kernels with `import helpers;` keep
// resolving these names.
//
// 16 `extension float` methods: erf, log1p, expm1, digamma, lgamma, hypot,
// ndtri, i0, i0e, i1, i1e, erfinv, spherical_bessel_j0, zeta, polygamma,
// igamma. Plus 4 `extension float4` h-reductions: hsum/hmax/hmin/hprod.
//
// T2.10 (2026-05-08): retired the M30 deprecated `c10_vulkan_*` free-function
// aliases. Bodies now live directly inside the extension method; callers
// must use `(x).method()` form. The 16 extension methods are the sole
// public surface.

public extension float {
[ForceInline]
public float erf() {
    float x = this;
    float absx = abs(x);
    float t = 1.0f / (1.0f + 0.3275911f * absx);
    float y = 1.0f - ((((
        1.061405429f * t - 1.453152027f) * t
        + 1.421413741f) * t - 0.284496736f) * t
        + 0.254829592f) * t * exp(-absx * absx);
    return (x < 0.0f) ? -y : y;
}
}

public extension float {
[ForceInline]
public float log1p() {
    float x = this;
    if (abs(x) < 0.5f) {
        float t = 1.0f + x;
        return log(t) - (t - 1.0f - x) / t;
    }
    return log(1.0f + x);
}
}

public extension float {
[ForceInline]
public float expm1() {
    float x = this;
    if (abs(x) < 0.7f) {
        float u = exp(x);
        float v = u - 1.0f;
        if (abs(v) > 0.5f) return v;
        return v * x / log(u);
    }
    return exp(x) - 1.0f;
}
}

// Extension form: `(a).hypot(b)`.
public extension float {
[ForceInline]
public float hypot(float other) {
    float abs_a = abs(this);
    float abs_b = abs(other);
    if (abs_a < abs_b) {
        float t = abs_a;
        abs_a = abs_b;
        abs_b = t;
    }
    if (abs_a == 0.0f) return 0.0f;
    float ratio = abs_b / abs_a;
    return abs_a * sqrt(1.0f + ratio * ratio);
}
}

// N+1.15 — [fastopt] lets slangc skip strict math optimization barriers
// and constant-fold more aggressively on this ~32-iteration Cephes loop.
public extension float {
[ForceInline]
public float digamma() {
    float x = max(this, 1e-30f);
    float result = 0.0f;
    uint _digamma_iter = 0;
    while (x < 7.0f && _digamma_iter < 32) {
        result -= 1.0f / x;
        x += 1.0f;
        _digamma_iter++;
    }
    float r = 1.0f / x;
    float r2 = r * r;
    result += log(x) - 0.5f * r;
    r = r * r2;
    result -= r * (0.083333333333f
        + r * (0.001666666666f
        + r * (2.424643878e-5f
        + r * (1.702279432e-6f
        + r * (1.425177771e-7f)))));
    return result;
}
}

public extension float {
[ForceInline]
public float lgamma() {
    float x = this;
    float sign = 1.0f;
    float z = abs(x);
    bool reflect = (z < 0.5f);
    if (reflect) {
        sign = (floor(x + 0.5f) != floor(x)) ? -1.0f : 1.0f;
        z = 1.0f - z;
    }
    z -= 1.0f;
    float g = 7.0f;
    float c0 = 0.99999999999980993f;
    float c1 = 676.5203681218851f;
    float c2 = -1259.1392167224028f;
    float c3 = 771.32342877765313f;
    float c4 = -176.61502916214059f;
    float c5 = 12.507343278686905f;
    float c6 = -0.13857109526572012f;
    float c7 = 9.9843695780195716e-6f;
    float c8 = 1.5056327351493116e-7f;
    float x_ = c0 + c1/(z+1.0f) + c2/(z+2.0f) + c3/(z+3.0f)
        + c4/(z+4.0f) + c5/(z+5.0f) + c6/(z+6.0f)
        + c7/(z+7.0f) + c8/(z+8.0f);
    float t2 = z + g + 0.5f;
    float result = 0.5f * log(6.283185307179586f) + (z + 0.5f) * log(t2) - t2 + log(x_);
    if (reflect) {
        result = log(3.141592653589793238f / sin(3.141592653589793238f * abs(x))) - result;
    }
    return sign < 0.0f ? -result : result;
}
}

// ── ndtri — inverse normal CDF (P3.8 port from slang_helpers.py) ─────

// N+1.15 — [branch] on three-region value-range dispatch suppresses
// flatten-by-default. The central/high/low-tail paths are divergent
// between lanes; flattening would predicate all three polynomials.
public extension float {
[ForceInline]
public float ndtri() {
    float p = this;
    float a[6] = {
        -3.969683028665376e+01f,  2.209460984245205e+02f,
        -2.759285104469687e+02f,  1.383577518672690e+02f,
        -3.066479806614716e+01f,  2.506628277459239e+00f
    };
    float b[5] = {
        -5.447609879822406e+01f,  1.615858368580409e+02f,
        -1.556989798598866e+02f,  6.680131188771972e+01f,
        -1.328068155288572e+01f
    };
    float c[6] = {
        -7.784894002430293e-03f, -3.223964580411365e-01f,
        -2.400758277161838e+00f, -2.549732539343734e+00f,
         4.374664141464968e+00f,  2.938163982698783e+00f
    };
    float d[4] = {
         7.784695709041462e-03f,  3.224671290700398e-01f,
         2.445134137142996e+00f,  3.754408661907416e+00f
    };
    float p_low = 0.02425f;
    float p_high = 1.0f - p_low;
    float q, r;
    [branch] if (p < p_low) {
        q = sqrt(-2.0f * log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0f);
    } else if (p <= p_high) {
        q = p - 0.5f;
        r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0f);
    } else {
        q = sqrt(-2.0f * log(1.0f - p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0f);
    }
}
}

// ── i0 / i0e / i1 / i1e — modified Bessel functions (P3.8 port) ──────
// Numerical Recipes single-precision coefficients.

// N+1.15 — [branch] on two-region modified Bessel approximation.
public extension float {
[ForceInline]
public float i0() {
    float x = this;
    float ax = abs(x);
    [branch] if (ax < 3.75f) {
        float y = (x / 3.75f) * (x / 3.75f);
        float ans1 = 1.0f + y * (3.5156229f + y * (3.0899424f + y * (1.2067492f
            + y * (0.2659732f + y * (0.360768e-1f + y * 0.45813e-2f)))));
        return ans1;
    }
    float z = 3.75f / ax;
    float y = z * z;
    float ans2 = (0.39894228f + y * (0.1328592e-1f + y * (0.225319e-2f
        + y * (-0.157565e-2f + y * (0.916281e-2f + y * (-0.2057706e-1f
        + y * (0.2635537e-1f + y * (-0.1647633e-1f + y * 0.392377e-2f))))))));
    return ans2 * exp(ax) / sqrt(ax);
}
}

// N+1.15 — [branch] on two-region scaled modified Bessel approximation.
public extension float {
[ForceInline]
public float i0e() {
    float x = this;
    float ax = abs(x);
    [branch] if (ax < 3.75f) {
        float y = (x / 3.75f) * (x / 3.75f);
        float ans1 = 1.0f + y * (3.5156229f + y * (3.0899424f + y * (1.2067492f
            + y * (0.2659732f + y * (0.360768e-1f + y * 0.45813e-2f)))));
        return ans1 * exp(-ax);
    }
    float z = 3.75f / ax;
    float y = z * z;
    float ans2 = (0.39894228f + y * (0.1328592e-1f + y * (0.225319e-2f
        + y * (-0.157565e-2f + y * (0.916281e-2f + y * (-0.2057706e-1f
        + y * (0.2635537e-1f + y * (-0.1647633e-1f + y * 0.392377e-2f))))))));
    return ans2 / sqrt(ax);
}
}

// N+1.15 — [branch] on two-region modified Bessel I1 approximation.
public extension float {
[ForceInline]
public float i1() {
    float x = this;
    float ax = abs(x);
    [branch] if (ax < 3.75f) {
        float y = (x / 3.75f) * (x / 3.75f);
        float ans1 = ax * (0.5f + y * (0.87890594f + y * (0.51498869f
            + y * (0.15084934f + y * (0.2658973e-1f + y * (0.301532e-2f
            + y * 0.32411e-3f))))));
        return (x < 0.0f) ? -ans1 : ans1;
    }
    float z = 3.75f / ax;
    float y = z * z;
    float ans2 = 0.39894228f + y * (-0.3988024e-1f + y * (-0.362018e-2f
        + y * (0.163801e-2f + y * (-0.1031555e-1f + y * (0.2282967e-1f
        + y * (-0.2895312e-1f + y * (0.1787654e-1f + y * (-0.420059e-2f))))))));
    ans2 = ans2 * exp(ax) / sqrt(ax);
    return (x < 0.0f) ? -ans2 : ans2;
}
}

// N+1.15 — [branch] on two-region scaled modified Bessel I1 approximation.
public extension float {
[ForceInline]
public float i1e() {
    float x = this;
    float ax = abs(x);
    [branch] if (ax < 3.75f) {
        float y = (x / 3.75f) * (x / 3.75f);
        float ans1 = ax * (0.5f + y * (0.87890594f + y * (0.51498869f
            + y * (0.15084934f + y * (0.2658973e-1f + y * (0.301532e-2f
            + y * 0.32411e-3f))))));
        ans1 = ans1 * exp(-ax);
        return (x < 0.0f) ? -ans1 : ans1;
    }
    float z = 3.75f / ax;
    float y = z * z;
    float ans2 = 0.39894228f + y * (-0.3988024e-1f + y * (-0.362018e-2f
        + y * (0.163801e-2f + y * (-0.1031555e-1f + y * (0.2282967e-1f
        + y * (-0.2895312e-1f + y * (0.1787654e-1f + y * (-0.420059e-2f))))))));
    ans2 = ans2 / sqrt(ax);
    return (x < 0.0f) ? -ans2 : ans2;
}
}

// ── erfinv — inverse error function (P3.8 port) ───────────────────────
// Mike Giles, "Approximating the erfinv function", GPU Gems 4.
// Single-precision coefficients matching CUDA erfinvf.
// N+1.15 — [fastopt] on 9-step Horner-form polynomial approximation.

public extension float {
[ForceInline]
public float erfinv() {
    float x = this;
    float w = -log((1.0f - x) * (1.0f + x));
    float p;
    if (w < 5.0f) {
        w = w - 2.5f;
        p = 2.81022636e-08f;
        p = 3.43273939e-07f + p * w;
        p = -3.5233877e-06f + p * w;
        p = -4.39150654e-06f + p * w;
        p = 0.00021858087f + p * w;
        p = -0.00125372503f + p * w;
        p = -0.00417768164f + p * w;
        p = 0.246640727f + p * w;
        p = 1.50140941f + p * w;
    } else {
        w = sqrt(w) - 3.0f;
        p = -0.000200214257f;
        p = 0.000100950558f + p * w;
        p = 0.00134934322f + p * w;
        p = -0.00367342844f + p * w;
        p = 0.00573950773f + p * w;
        p = -0.0076224613f + p * w;
        p = 0.00943887047f + p * w;
        p = 1.00167406f + p * w;
        p = 2.83297682f + p * w;
    }
    return p * x;
}
}

// ── spherical_bessel_j0 (P3.8 port) ───────────────────────────────────

// N+1.15 — [branch] on two-region spherical Bessel approximation.
public extension float {
[ForceInline]
public float spherical_bessel_j0() {
    float x = this;
    float ax = abs(x);
    [branch] if (ax < 8.0f) {
        float y = x * x;
        return 1.0f + y * (-0.1666666666666667f + y * (0.0083333333333333f
            + y * (-0.0001984126984127f + y * (2.7557319223986e-6f
            + y * (-2.5052108385442e-8f)))));
    }
    return sin(ax) / ax;
}
}

// ── zeta — Hurwitz zeta ζ(s,q) (P3.8 port) ────────────────────────────
// Cephes-style direct sum + Euler-Maclaurin tail.
// Extension form: `(s).zeta(q)` matches PyTorch torch.special.zeta(s, q).

// N+1.15 — [fastopt] on 9-term sum + 6-term Euler-Maclaurin tail.
public extension float {
[ForceInline]
public float zeta(float q) {
    float s = this;
    if (s == 1.0f) return 1e30f;
    if (q <= 0.0f && q == floor(q)) return 1e30f;
    float a = q;
    float sum = 0.0f;
    for (int i = 0; i < 9; i++) {
        sum += pow(a, -s);
        a += 1.0f;
    }
    float b = pow(a, -s);
    sum += b * a / (s - 1.0f);
    sum += 0.5f * b;
    float coef[6] = {
        8.33333333e-2f,
        -1.38888889e-3f,
        3.30687831e-5f,
        -8.26719577e-7f,
        2.08767569e-8f,
        -5.28419013e-10f
    };
    float t = s;
    float a2 = 1.0f / (a * a);
    float bo_a = b / a;
    sum += coef[0] * t * bo_a;
    float pow_factor = bo_a;
    for (int i = 1; i < 6; i++) {
        t = t * (s + 2.0f * float(i) - 1.0f) * (s + 2.0f * float(i));
        pow_factor = pow_factor * a2;
        sum += coef[i] * t * pow_factor;
    }
    return sum;
}
}

// ── polygamma — polygamma function ψ⁽ⁿ⁾(x) (P3.8 port) ────────────────
// Extension form: `(x).polygamma(n)` matches PyTorch torch.polygamma(n, x).
// Internal call to `digamma` uses extension form (T2.10).

// N+1.15 — [fastopt] on ~32-iteration Cephes digamma-like loop.
public extension float {
[ForceInline]
public float polygamma(float n) {
    float x = this;
    if (n == 0.0f) return x.digamma();
    x = max(x, 1e-30f);
    float result = 0.0f;
    uint _pg_iter = 0;
    while (x < 7.0f && _pg_iter < 32) {
        float xn = n;
        for (uint j = 1; j < uint(n); j++) {
            xn *= (n - float(j));
        }
        result += xn / pow(x, n + 1.0f);
        x += 1.0f;
        _pg_iter++;
    }
    float r = 1.0f / (x * x);
    if (n == 1.0f) {
        result += 0.1666666666666667f + r * (0.0333333333333333f
            + r * (0.0238095238095238f
            + r * (0.0333333333333333f
            + r * (0.0757575757575758f))));
    } else if (n == 2.0f) {
        result += -0.0333333333333333f + r * (-0.0238095238095238f
            + r * (-0.0333333333333333f
            + r * (-0.0757575757575758f
            + r * (-0.2531135531135531f))));
    } else {
        result += 0.1666666666666667f + r * 0.0333333333333333f;
    }
    return result;
}
}

// ── igamma — regularized lower incomplete gamma P(a,x) (P3.8 port) ────
// Extension form: `(x).igamma(a)` matches PyTorch torch.igamma(a, x).
// Internal call to `lgamma` uses extension form (T2.10).

// N+1.15 — [fastopt] on 64-iteration series + continued-fraction loops.
public extension float {
[ForceInline]
public float igamma(float a) {
    float x = this;
    if (x < 0.0f || a <= 0.0f) return 0.0f;
    if (x < a + 1.0f) {
        float sum = 1.0f / a;
        float term = 1.0f / a;
        for (uint n = 1; n < 64; n++) {
            term *= x / (a + float(n));
            sum += term;
            if (abs(term) < abs(sum) * 1e-7f) break;
        }
        return sum * exp(-x + a * log(x) - a.lgamma());
    }
    float lg = a.lgamma();
    float b = x + 1.0f - a;
    float c = 1.0f / 1.0e-30f;
    float d = 1.0f / b;
    float h = d;
    for (uint i = 1; i < 64; i++) {
        float an = -float(i) * (float(i) - a);
        b += 2.0f;
        d = an * d + b;
        if (abs(d) < 1.0e-30f) d = 1.0e-30f;
        c = b + an / c;
        if (abs(c) < 1.0e-30f) c = 1.0e-30f;
        d = 1.0f / d;
        float del = d * c;
        h *= del;
        if (abs(del - 1.0f) < 1e-7f) break;
    }
    return 1.0f - h * exp(-x + a * log(x) - lg);
}
}

// Extension form for register-resident float4 horizontal reductions.
// Use when the four lanes are already loaded:
//     float4 v = float4(a, b, c, d);
//     float s = v.hsum();
public extension float4 {
public [ForceInline] float hsum() { return this.x + this.y + this.z + this.w; }
public [ForceInline] float hmax() { return max(max(this.x, this.y), max(this.z, this.w)); }
public [ForceInline] float hmin() { return min(min(this.x, this.y), min(this.z, this.w)); }
public [ForceInline] float hprod() { return this.x * this.y * this.z * this.w; }
}

// Free-function alias for hypot used by pointwise.slang.
[ForceInline] public float hypot(float a, float b) { return a.hypot(b); }

