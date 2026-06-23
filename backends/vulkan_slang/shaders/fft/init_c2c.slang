// Load interleaved complex input into work buffer with bit-reversal permutation,
// preparing for DIT butterfly passes.

struct Params {
    uint N;          // FFT size (power of 2)
    uint log2_N;     // log2(N)
    uint batch;      // number of signals
    uint in_stride;  // complex elements per signal in source (may equal N)
};

[[vk::binding(0)]] StructuredBuffer<float>   src;   // complex input [batch * 2*in_stride]
[[vk::binding(1)]] RWStructuredBuffer<float> dst;   // complex out   [batch * 2*N]
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

    uint rev_n   = bitrev(n, params.log2_N);
    uint src_off = b * 2u * params.in_stride + 2u * rev_n;
    uint dst_off = b * 2u * params.N         + 2u * n;

    dst[dst_off]      = src[src_off];
    dst[dst_off + 1u] = src[src_off + 1u];
}
