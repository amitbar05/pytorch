// Copy first N/2+1 complex values from a full-N FFT result to an output buffer.
// This extracts the one-sided spectrum for real-to-complex transforms.

struct Params {
    uint N;      // full FFT size
    uint batch;  // number of signals
};

[[vk::binding(0)]] StructuredBuffer<float>   src;   // full FFT  [batch * 2*N]
[[vk::binding(1)]] RWStructuredBuffer<float> dst;   // output    [batch * 2*(N/2+1)]
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint half1 = params.N / 2u + 1u;
    uint tid   = gid.x * 256u + lid.x;
    uint b     = tid / half1;
    uint k     = tid % half1;
    if (b >= params.batch) return;

    uint si = b * 2u * params.N + 2u * k;
    uint di = b * 2u * half1    + 2u * k;
    dst[di]      = src[si];
    dst[di + 1u] = src[si + 1u];
}
