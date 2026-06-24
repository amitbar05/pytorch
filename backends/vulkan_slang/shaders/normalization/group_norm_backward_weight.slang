// Group normalization backward for grad_weight and grad_bias.
// One workgroup per channel; 256 threads split the N×HxW reduction.
//
// grad_weight[c] = sum_n sum_hw (grad_output[n,c,hw] * x_hat[n,c,hw])
// grad_bias[c]   = sum_n sum_hw  grad_output[n,c,hw]
// where x_hat = (x - mean[n*G + group_of_c]) * rstd[n*G + group_of_c]
//
// GradPar.1: replaced the serial per-channel scan with a 256-thread
// workgroup reduction: threads stride over HxW per batch, reducing
// spatial parallelism from 1 to min(256, HxW) threads per channel.

struct Params {
    uint N;       // batch size
    uint C;       // number of channels
    uint HxW;     // spatial size per channel
    uint G;       // number of groups
    uint cpg;     // channels per group = C / G
    uint compute_weight;
    uint compute_bias;
};

struct KernelArgs {
    StructuredBuffer<float> grad_output;    // [N, C, HxW]
    StructuredBuffer<float> input;          // [N, C, HxW]
    StructuredBuffer<float> mean;           // [N * G]
    StructuredBuffer<float> rstd;           // [N * G]
    RWStructuredBuffer<float> grad_weight;  // [C]
    RWStructuredBuffer<float> grad_bias;    // [C]
};

// M21.3.01: explicit Set 0 binding (slangc 2026.7.1 defaults to Set 1)
[[vk::binding(0, 0)]] ParameterBlock<KernelArgs> args;
[[vk::push_constant]] Params params;

groupshared float smem[256];
groupshared float smem2[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint c = gid.x;
    if (c >= params.C) return;

    uint tid = lid.x;
    uint group_idx = c / params.cpg;

    float sum_gw = 0.0f;
    float sum_gb = 0.0f;

    // Outer loop over batch; inner loop strides 256 threads across HxW.
    for (uint n = 0; n < params.N; n++) {
        uint mg_idx = n * params.G + group_idx;
        float mean_val = args.mean[mg_idx];
        float rstd_val = args.rstd[mg_idx];

        uint base = (n * params.C + c) * params.HxW;
        for (uint hw = tid; hw < params.HxW; hw += 256) {
            float go = args.grad_output[base + hw];
            if (params.compute_weight != 0) {
                float x_hat = (args.input[base + hw] - mean_val) * rstd_val;
                sum_gw += go * x_hat;
            }
            if (params.compute_bias != 0) {
                sum_gb += go;
            }
        }
    }

    smem[tid] = sum_gw;
    smem2[tid] = sum_gb;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
            smem2[tid] += smem2[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        if (params.compute_weight != 0) args.grad_weight[c] = smem[0];
        if (params.compute_bias != 0) args.grad_bias[c] = smem2[0];
    }
}
