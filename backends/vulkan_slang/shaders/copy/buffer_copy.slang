// GPU-side buffer copy: out[i] = input[i]
// Replaces host-roundtrip copy_buffer_inplace pattern.
//
// M-cpp-new-4: this shader operates on 32-bit `uint` words, not `float`
// elements. The C++ dispatcher (`dispatch_copy_buffer` in
// `csrc/ops/dispatch.cpp`) computes the copy length as
// `numel * elementSize(dtype) / 4` so that 8-byte dtypes (int64, float64)
// are copied in full instead of having their high 32 bits silently dropped
// by the prior `StructuredBuffer<float>` path. Reinterpreting the storage
// as `uint` is safe because every `VulkanBuffer` is at least 4-byte
// aligned (allocator guarantee) and bit-exact `memcpy`-style semantics are
// what `copy_` requires.

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<uint> input;
[[vk::binding(1)]] RWStructuredBuffer<uint> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;
    out[tid.x] = input[tid.x];
}
