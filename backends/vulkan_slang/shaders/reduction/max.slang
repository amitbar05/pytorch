// Parallel max reduction using shared memory
// M15.2: WaveActiveMax (subgroup_arithmetic) replaces the 7-barrier smem
// tree-reduction loop.  RDNA1 / Vulkan 1.2+ always exposes GroupNonUniformArithmetic.

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

import helpers;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(
    uint3 lid : SV_GroupThreadID,
    uint3 gid : SV_GroupID)
{
    uint local_id = lid.x;
    uint global_id = gid.x * 256u + local_id;

    // Per-lane load; out-of-range threads return identity so WaveActiveMax
    // skips them (no smem, no barrier).
    float val = (global_id < params.numel) ? input[global_id] : -3.402823466e+38f;

    // Wave-level reduce: one instruction, no shared memory.
    float lane_max = wg_reduce_wave<OpMaxReduce>(val, local_id, 256u, VK_SUBGROUP_SIZE);

    // Thread 0 writes the workgroup result.
    if (local_id == 0u) {
        out[gid.x] = lane_max;
    }
}
