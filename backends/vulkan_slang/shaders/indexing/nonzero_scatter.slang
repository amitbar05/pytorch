// nonzero_scatter.slang — Pass 2: scatter nonzero indices to output
//
// OP.1.a-fast: Each workgroup reads its exclusive prefix offset from
// workspace (computed by CPU after Pass 1), re-scans its elements,
// and writes the multi-dimensional index of each nonzero to the
// output int64 tensor at position [wg_offset + local_exclusive_prefix].

import reduction;

// Push constants for the scatter pass.
// shape[0..3] stores the tensor dimensions (up to 4D).
// ndim is the actual number of dimensions (1-4).
// numel is the total number of input elements.
struct Params {
    uint ndim;
    uint shape0;
    uint shape1;
    uint shape2;
    uint shape3;
    uint numel;
};

[[vk::binding(0, 0)]] StructuredBuffer<float> input;
[[vk::binding(1, 0)]] StructuredBuffer<uint> workspace;  // per-WG exclusive prefix offsets
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> output;   // int64 indices as uint32 pairs
[[vk::push_constant]] Params params;

// Convert a flat index to multi-dimensional indices (row-major / C-contiguous).
// Writes ndim int64 values (each as 2 uint32s) starting at output[out_base].
void write_multi_index(uint flat, uint out_base, uint ndim,
                       uint s0, uint s1, uint s2, uint s3,
                       RWStructuredBuffer<uint> out)
{
    // Unroll the flat index into up to 4 dimensions.
    // Row-major (C-contiguous): for shape (s0, s1, s2, s3),
    //   flat = ((d0 * s1 + d1) * s2 + d2) * s3 + d3
    uint d0, d1, d2, d3;
    uint tmp = flat;

    d3 = tmp % s3;
    tmp /= s3;
    d2 = tmp % s2;
    tmp /= s2;
    d1 = tmp % s1;
    tmp /= s1;
    d0 = tmp;

    // Write int64 indices as pairs of uint32 (low, high).
    // Indices are non-negative and fit in int32, so high = 0.
    uint base = out_base * ndim * 2u;  // each int64 takes 2 uint32s

    out[base + 0u] = d0;
    out[base + 1u] = 0u;
    if (ndim >= 2u) {
        out[base + 2u] = d1;
        out[base + 3u] = 0u;
    }
    if (ndim >= 3u) {
        out[base + 4u] = d2;
        out[base + 5u] = 0u;
    }
    if (ndim >= 4u) {
        out[base + 6u] = d3;
        out[base + 7u] = 0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(
    uint3 tid : SV_DispatchThreadID,
    uint3 gid : SV_GroupID,
    uint3 lid : SV_GroupThreadID)
{
    uint global_idx = tid.x;
    uint local_tid = lid.x;

    uint ndim = params.ndim;
    uint s0 = params.shape0;
    uint s1 = params.shape1;
    uint s2 = params.shape2;
    uint s3 = params.shape3;
    uint numel = params.numel;

    // Per-WG exclusive prefix offset for this workgroup (computed by CPU)
    uint wg_offset = workspace[gid.x];

    // Determine if this element is nonzero
    float flag = 0.0f;
    if (global_idx < numel && input[global_idx] != 0.0f) {
        flag = 1.0f;
    }

    // Inclusive scan within the workgroup using wave intrinsics.
    // We compute inclusive then subtract our own flag to get exclusive.
    float inclusive = wg_inclusive_scan_wave<WaveScanAdd>(flag, local_tid, 256u, 32u);
    float exclusive = inclusive - flag;  // exclusive = inclusive - own value

    // If this element is nonzero, write its multi-dimensional index
    if (flag > 0.0f) {
        uint out_pos = wg_offset + uint(exclusive);
        write_multi_index(global_idx, out_pos, ndim, s0, s1, s2, s3, output);
    }
}
