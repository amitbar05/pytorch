// Load real input into interleaved complex buffer with bit-reversal permutation.
// Output is [re0, 0, re1, 0, ...] with elements placed at bit-reversed positions
// so that subsequent DIT butterfly passes produce naturally-ordered output.

struct Params {
    uint N;          // FFT size (power of 2)
    uint log2_N;     // log2(N)
    uint batch;      // number of signals
    uint in_stride;  // float elements per signal in the source buffer
};

[[vk::binding(0)]] StructuredBuffer<float>   src;   // real input  [batch * in_stride]
[[vk::binding(1)]] RWStructuredBuffer<float> dst;   // complex out [batch * 2*N]
[[vk::push_constant]] Params params;

uint bitrev(uint x, uint bits) {
    uint r = 0u;
    for (uint i = 0u; i < bits; i++) {
        r = (r << 1u) | (x & 1u);
        x >>= 1u;
    }
    return r;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint tid = gid.x * 256u + lid.x;
    uint b   = tid / params.N;
    uint n   = tid % params.N;
    if (b >= params.batch) return;

    uint rev_n = bitrev(n, params.log2_N);

    dst[b * 2u * params.N + 2u * n]      = src[b * params.in_stride + rev_n];
    dst[b * 2u * params.N + 2u * n + 1u] = 0.0f;
}
