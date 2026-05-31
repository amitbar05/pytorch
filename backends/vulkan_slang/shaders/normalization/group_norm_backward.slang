// Fused group normalization backward for grad_input.
// One workgroup per row (N*G rows), each thread handles multiple elements.
//
// Input reshaped as [N*G, group_size] where group_size = cpg * HxW.
// Weight is [C] = [G * cpg], expanded per-spatial-element within each group.
//
// M22.6: Upgraded to ParameterBlock<KernelArgs> for M-SF.1 compliance.
// For each row (batch_group_idx):
//   group_idx = batch_group_idx % G
//   For element d in [0, group_size):
//     channel_in_group = d / HxW
//     w[d] = weight[group_idx * cpg + channel_in_group]
//
// Formula matching PyTorch native_group_norm_backward:
//   x_hat = (x - mean) * rstd
//   go_w = grad_output * w     (w=1 if no gamma)
//   c1 = (1/D) * sum(go_w)
//   c2 = (1/D) * sum(go_w * x_hat)
//   grad_input = rstd * (go_w - c1 - x_hat * c2)

struct Params {
    uint num_rows;    // N * G
    uint group_size;  // cpg * HxW
    uint num_groups;  // G
    uint cpg;         // channels per group
    uint HxW;         // spatial size
    uint has_weight;
};

struct KernelArgs {
    StructuredBuffer<float> grad_output;   // [num_rows, group_size]
    StructuredBuffer<float> input;         // [num_rows, group_size]
    StructuredBuffer<float> mean;          // [num_rows]
    StructuredBuffer<float> rstd;          // [num_rows]
    StructuredBuffer<float> weight;        // [C] = [G * cpg]
    RWStructuredBuffer<float> grad_input;  // [num_rows, group_size]
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
    uint row = gid.x;
    if (row >= params.num_rows) return;

    uint tid = lid.x;
    uint D = params.group_size;
    uint base = row * D;
    float mean_val = args.mean[row];
    float rstd_val = args.rstd[row];

    // Group index for weight lookup
    uint group_idx = row % params.num_groups;
    uint weight_base = group_idx * params.cpg;

    // Step 1: Compute c1 and c2 simultaneously via strided partial sums
    float partial_c1 = 0.0f;
    float partial_c2 = 0.0f;
    for (uint d = tid; d < D; d += 256) {
        float go = args.grad_output[base + d];
        float w = 1.0f;
        if (params.has_weight != 0) {
            uint channel_in_group = d / params.HxW;
            w = args.weight[weight_base + channel_in_group];
        }
        float go_w = go * w;
        float x_hat = (args.input[base + d] - mean_val) * rstd_val;
        partial_c1 += go_w;
        partial_c2 += go_w * x_hat;
    }

    smem[tid] = partial_c1;
    smem2[tid] = partial_c2;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
            smem2[tid] += smem2[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float c1 = smem[0] / float(D);
    float c2 = smem2[0] / float(D);

    // Step 2: Compute grad_input per-element
    for (uint d = tid; d < D; d += 256) {
        float go = args.grad_output[base + d];
        float w = 1.0f;
        if (params.has_weight != 0) {
            uint channel_in_group = d / params.HxW;
            w = args.weight[weight_base + channel_in_group];
        }
        float go_w = go * w;
        float x_hat = (args.input[base + d] - mean_val) * rstd_val;
        args.grad_input[base + d] = rstd_val * (go_w - c1 - x_hat * c2);
    }
}
