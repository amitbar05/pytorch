// Radix-2 DIT butterfly pass for a power-of-2 FFT.
// Complex data is stored as interleaved float pairs: [re0, im0, re1, im1, ...]
//
// At pass p (0 = first):
//   half_len = 2^p,  bfly_len = 2^(p+1)
//   group    = i / half_len
//   k        = i % half_len
//   twiddle  = exp(±2πi · k / bfly_len)  (- for forward, + for inverse)
//
// Input:  src[b * 2N .. +2N)  interleaved complex floats
// Output: dst[b * 2N .. +2N)

struct Params {
    uint N;          // FFT size (power of 2)
    uint pass_idx;   // 0 .. log2(N)-1
    uint batch;      // number of independent 1-D signals
    uint forward;    // 1 = forward (exp(-2πi)), 0 = inverse (exp(+2πi))
};

[[vk::binding(0)]] StructuredBuffer<float>   src;
[[vk::binding(1)]] RWStructuredBuffer<float> dst;
[[vk::push_constant]] Params params;

static const float PI = 3.14159265358979323846f;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint tid      = gid.x * 256u + lid.x;
    uint half_N   = params.N >> 1u;
    uint b        = tid / half_N;
    uint i        = tid % half_N;
    if (b >= params.batch) return;

    uint half_len = 1u << params.pass_idx;
    uint bfly_len = half_len << 1u;
    uint group    = i / half_len;
    uint k        = i % half_len;

    uint base = b * (2u * params.N);
    uint ia   = base + 2u * (group * bfly_len + k);
    uint ib   = ia + 2u * half_len;

    float ar = src[ia],      ai = src[ia + 1u];
    float br = src[ib],      bi = src[ib + 1u];

    float sign  = (params.forward != 0u) ? -1.0f : 1.0f;
    float angle = sign * 2.0f * PI * float(k) / float(bfly_len);
    float wr    = cos(angle);
    float wi    = sin(angle);

    // twiddle * b
    float tr = wr * br - wi * bi;
    float ti = wr * bi + wi * br;

    dst[ia]      = ar + tr;  dst[ia + 1u] = ai + ti;
    dst[ib]      = ar - tr;  dst[ib + 1u] = ai - ti;
}
