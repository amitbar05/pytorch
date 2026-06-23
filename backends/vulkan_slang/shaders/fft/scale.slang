// Scale all elements by a constant factor (used for FFT normalization).

struct Params {
    uint  numel;
    float scale;
};

[[vk::binding(0)]] RWStructuredBuffer<float> buf;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint tid = gid.x * 256u + lid.x;
    if (tid >= params.numel) return;
    buf[tid] *= params.scale;
}
