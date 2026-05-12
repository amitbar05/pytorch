// nonzero_count.slang — Pass 1: count nonzero elements per workgroup
//
// OP.1.a-fast: Each workgroup scans its assigned elements, counts
// nonzeros, and writes the per-WG total to workspace[wg_id].
// The CPU then computes the exclusive prefix sum over workspace
// before Pass 2 (nonzero_scatter.slang) writes the indices.

import reduction;

struct Params {
    uint numel;
};

[[vk::binding(0, 0)]] StructuredBuffer<float> input;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> workspace;  // per-WG counts
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(
    uint3 tid : SV_DispatchThreadID,
    uint3 gid : SV_GroupID,
    uint3 lid : SV_GroupThreadID)
{
    uint global_idx = tid.x;
    uint local_tid = lid.x;

    // Determine if this element is nonzero (as float 0.0 or 1.0)
    float flag = 0.0f;
    if (global_idx < params.numel && input[global_idx] != 0.0f) {
        flag = 1.0f;
    }

    // Inclusive scan within the workgroup using wave intrinsics.
    // Threads beyond numel contribute 0 (identity for addition), so
    // the total is correct even for the last partial workgroup.
    float inclusive = wg_inclusive_scan_wave<WaveScanAdd>(flag, local_tid, 256u, 32u);

    // The last thread in each workgroup writes the WG total to workspace.
    // For the last WG, even if numel isn't a multiple of 256, the total
    // is correct because out-of-range threads had flag=0.
    if (local_tid == 255u) {
        workspace[gid.x] = uint(inclusive);
    }
}
