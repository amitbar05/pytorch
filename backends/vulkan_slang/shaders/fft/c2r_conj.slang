// Extend a one-sided Hermitian spectrum (N/2+1 complex values) to full N complex
// values using the symmetry X[N-k] = conj(X[k]) for k = 1 .. N/2-1.

struct Params {
    uint N;      // full FFT size (even)
    uint batch;  // number of signals
};

[[vk::binding(0)]] StructuredBuffer<float>   src;   // input  [batch * 2*(N/2+1)]
[[vk::binding(1)]] RWStructuredBuffer<float> dst;   // output [batch * 2*N]
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint tid = gid.x * 256u + lid.x;
    uint b   = tid / params.N;
    uint k   = tid % params.N;
    if (b >= params.batch) return;

    uint half1    = params.N / 2u + 1u;
    uint src_base = b * 2u * half1;
    uint dst_base = b * 2u * params.N;

    float re, im;
    if (k < half1) {
        re =  src[src_base + 2u * k];
        im =  src[src_base + 2u * k + 1u];
    } else {
        uint kk = params.N - k;   // kk in [1, N/2-1]
        re =  src[src_base + 2u * kk];
        im = -src[src_base + 2u * kk + 1u];   // complex conjugate
    }

    dst[dst_base + 2u * k]      = re;
    dst[dst_base + 2u * k + 1u] = im;
}
