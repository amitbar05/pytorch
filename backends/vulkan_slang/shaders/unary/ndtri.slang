// Unary ndtri: out[i] = Phi^{-1}(p) — inverse standard normal CDF (probit function).
// Acklam's rational approximation — max |err| < 1.15e-9 for f64; ~1e-6 for f32.
// Splits domain: lower tail (p < 0.02425), central, upper tail (p > 0.97575).
struct Params { uint numel; };
[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

static const float PLO = 0.02425f;
static const float PHI = 0.97575f;

float ndtri_impl(float p) {
    if (p <= 0.0f) return -1.0f / 0.0f; // -Inf
    if (p >= 1.0f) return  1.0f / 0.0f; // +Inf
    if (isnan(p)) return p;

    // Coefficients — central region
    float a0 = -3.969683028665376e+01f;
    float a1 =  2.209460984245205e+02f;
    float a2 = -2.759285104469687e+02f;
    float a3 =  1.383577518672690e+02f;
    float a4 = -3.066479806614716e+01f;
    float a5 =  2.506628277459239e+00f;

    float b0 = -5.447609879822406e+01f;
    float b1 =  1.615858368580409e+02f;
    float b2 = -1.556989798598866e+02f;
    float b3 =  6.680131188771972e+01f;
    float b4 = -1.328068155288572e+01f;

    // Coefficients — tail regions
    float c0 = -7.784894002430293e-03f;
    float c1 = -3.223964580411365e-01f;
    float c2 = -2.400758277161838e+00f;
    float c3 = -2.549732539343734e+00f;
    float c4 =  4.374664141464968e+00f;
    float c5 =  2.938163982698783e+00f;

    float d0 =  7.784695709041462e-03f;
    float d1 =  3.224671290700398e-01f;
    float d2 =  2.445134137142996e+00f;
    float d3 =  3.754408661907416e+00f;

    if (p < PLO) {
        float q = sqrt(-2.0f * log(p));
        return (((((c0*q + c1)*q + c2)*q + c3)*q + c4)*q + c5)
             / ((((d0*q + d1)*q + d2)*q + d3)*q + 1.0f);
    } else if (p <= PHI) {
        float q = p - 0.5f;
        float r = q * q;
        return (((((a0*r + a1)*r + a2)*r + a3)*r + a4)*r + a5)*q
             / (((((b0*r + b1)*r + b2)*r + b3)*r + b4)*r + 1.0f);
    } else {
        float q = sqrt(-2.0f * log(1.0f - p));
        return -((((( c0*q + c1)*q + c2)*q + c3)*q + c4)*q + c5)
              / ((((d0*q + d1)*q + d2)*q + d3)*q + 1.0f);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    out[tid.x] = ndtri_impl(input[tid.x]);
}
