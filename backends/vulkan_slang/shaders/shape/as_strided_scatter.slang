// GPU scatter for as_strided_scatter.
// Writes src into out at positions defined by as_strided(size, stride, storage_offset).
//
// Semantics: out starts as a clone of self (done separately via dispatch_copy_buffer).
//            This shader then writes: out[offset + sum(coords * strides)] = src[flat_idx]
//            for each flat index in src (row-major / C-contiguous traversal of size).
//
// Push constants carry the size[] and stride[] arrays (up to 8D).
// Push constant layout (80 bytes total, within Vulkan's 128-byte minimum):
//   [0]  numel_src  — number of elements in src
//   [1]  offset     — storage_offset in elements
//   [2]  ndim       — number of dimensions
//   [3]  _pad
//   [4..11]  size[8]
//   [12..19] stride[8]

struct Params {
    uint numel_src;
    uint offset;
    uint ndim;
    uint _pad;
    uint size[8];
    uint stride[8];
};

[[vk::binding(0)]] StructuredBuffer<float>   src_buf;
[[vk::binding(1)]] RWStructuredBuffer<float> out_buf;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint src_idx = tid.x;
    if (src_idx >= params.numel_src) return;

    // Unravel src_idx into multi-index (row-major: rightmost dim varies fastest)
    uint remaining = src_idx;
    uint dst_idx = params.offset;
    for (int i = (int)params.ndim - 1; i >= 0; i--) {
        uint coord = remaining % params.size[i];
        remaining /= params.size[i];
        dst_idx += coord * params.stride[i];
    }

    out_buf[dst_idx] = src_buf[src_idx];
}
