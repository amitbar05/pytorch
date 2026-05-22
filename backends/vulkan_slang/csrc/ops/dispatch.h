#pragma once

#include "../vulkan/Context.h"
#include "../vulkan/CommandBuffer.h"
#include "../vulkan/DescriptorSet.h"
#include "../vulkan/Memory.h"
#include "../vulkan/Pipeline.h"
#include "../vulkan/Stream.h"
#include "../backend/Allocator.h"

#include <torch/torch.h>
#include <unordered_set>
#include <vector>

namespace torch_vulkan { namespace ops {

// CG.M15: SpecConstant = (spec_id, value) pair for VkSpecializationInfo.
using SpecConstant = std::pair<uint32_t, uint32_t>;

// Per-device runtime state (stream, descriptor pool)
struct DeviceRuntime {
    std::unique_ptr<vulkan::Stream> stream;
    std::unique_ptr<vulkan::DescriptorPool> desc_pool;
    // Tracks VkBuffers written by dispatches in the current deferred command buffer.
    // Used for smart barrier insertion: barrier only emitted when a read depends on a prior write.
    std::unordered_set<VkBuffer> dirty_buffers;

    // Tracks VkBuffers written by the CPU host (via vkMapMemory/memcpy in
    // VulkanBuffer::write) since the last flush. A dispatch that reads any
    // of these buffers must first emit a HOST → COMPUTE pipeline barrier so
    // the GPU sees the latest host data. This is separate from dirty_buffers
    // (which covers GPU-written buffers) because the required barrier stages
    // differ: HOST→COMPUTE vs COMPUTE→COMPUTE.
    std::unordered_set<VkBuffer> host_written_buffers;

    // M17.5: batch mode suppresses auto-flush until end_batch_dispatch().
    // When true, dispatch_shader/dispatch_shader_indexed will NOT auto-flush
    // even when MAX_DISPATCHES_PER_CMD is exceeded.
    bool batch_mode = false;

    // M17.5: Per-batch descriptor set cache.
    // Key = (VkDescriptorSetLayout, hash of the bound VkBuffer list).
    //
    // M-cpp-new-6: the cache was originally keyed on layout alone, which
    // is INCORRECT when the same pipeline is dispatched twice in the same
    // batch with *different* buffers (e.g. a chained `x.relu().relu()`).
    // Both dispatches would record `vkCmdBindDescriptorSets` with the same
    // `VkDescriptorSet` handle, then the second `vkUpdateDescriptorSets`
    // call would overwrite the bindings the first dispatch needs — when
    // the cmd buffer executes, both dispatches read the LATEST buffer
    // contents (the 2nd dispatch's), corrupting the chain. Adding the
    // buffer-list hash to the key preserves M17.5's perf win for repeated
    // dispatches on the same buffers (the autotune / multi-launch case it
    // was designed for) while forcing a fresh descriptor set whenever the
    // buffer list changes.
    //
    // The hash needs to combine `VkBuffer` handles in binding order. It
    // is collision-tolerant in the sense that on a collision we'd reuse a
    // set with different buffers and get the same bug back — but
    // `VkBuffer` is a pointer-like handle, so collisions on a 64-bit FNV
    // are vanishingly rare. If a collision is ever observed in practice,
    // switching to `std::vector<VkBuffer>` as the key is the canonical
    // fix and trivial.
    struct DescSetCacheKey {
        VkDescriptorSetLayout layout;
        uint64_t buffers_hash;
        bool operator==(const DescSetCacheKey& o) const noexcept {
            return layout == o.layout && buffers_hash == o.buffers_hash;
        }
    };
    struct DescSetCacheKeyHash {
        size_t operator()(const DescSetCacheKey& k) const noexcept {
            // Mix layout pointer and buffers hash. The buffers hash is
            // already well-mixed (FNV-1a on the VkBuffer handles); the
            // XOR with the layout pointer just folds in the pipeline.
            return static_cast<size_t>(
                k.buffers_hash ^
                (reinterpret_cast<uintptr_t>(k.layout) * 0x9e3779b97f4a7c15ull));
        }
    };
    std::unordered_map<DescSetCacheKey, VkDescriptorSet, DescSetCacheKeyHash>
        desc_set_cache;
};

DeviceRuntime& get_runtime(uint32_t device_index = UINT32_MAX);

// Destroy all per-device runtimes (streams, descriptor pools).
// Must be called before VkDevice destruction.
void cleanup_runtimes();

// Get VkBuffer + size from a Vulkan tensor
struct BufferInfo {
    VkBuffer buffer;
    VkDeviceSize size;
};

BufferInfo get_buffer_info(const at::Tensor& tensor);

// Dispatch a compute shader with storage buffers and optional push constants.
//
// Usage:
//   dispatch_shader("binary_add_fwd", spirv_data, spirv_size,
//                   {input_a, input_b, output}, numel, push_data, push_size);
//
// workgroup_size: threads per workgroup (default 256 for element-wise)
// num_outputs: number of output (RWStructuredBuffer) tensors, counted from the end of `buffers`.
//   Default 1: last tensor is the output.
//   Use 2+ for shaders with multiple output bindings (e.g. rms_norm, max_pool2d_indices).
//   Used for smart barrier insertion: barrier emitted only when a read depends on a prior write.
//
// CG.M15: spec_constants are (constant_id, value) pairs that override
// ``[[vk::constant_id]]`` defaults at pipeline-creation time.  A single
// SPIR-V module can serve multiple tile configurations this way.
void dispatch_shader(
    const std::string& key,
    const uint32_t* spirv_code,
    size_t spirv_size,
    const std::vector<at::Tensor>& buffers,
    uint32_t num_workgroups_x,
    uint32_t num_workgroups_y = 1,
    uint32_t num_workgroups_z = 1,
    const void* push_constants = nullptr,
    uint32_t push_constants_size = 0,
    uint32_t num_outputs = 1,
    const std::vector<SpecConstant>& spec_constants = {});

// N+1.5: dispatch with descriptor-array bindings.
// `descriptor_counts.size()` = number of bindings; sum = total buffers,
// which must equal `buffers.size()`. Each binding consumes
// `descriptor_counts[i]` consecutive entries from `buffers`.
//
// Falls back to the flat path when all counts are 1.
//
// CG.M15: spec_constants are (constant_id, value) pairs for specialization.
void dispatch_shader_indexed(
    const std::string& key,
    const uint32_t* spirv_code,
    size_t spirv_size,
    const std::vector<at::Tensor>& buffers,
    const std::vector<uint32_t>& descriptor_counts,
    uint32_t num_workgroups_x,
    uint32_t num_workgroups_y = 1,
    uint32_t num_workgroups_z = 1,
    const void* push_constants = nullptr,
    uint32_t push_constants_size = 0,
    uint32_t num_outputs = 1,
    const std::vector<SpecConstant>& spec_constants = {});

// Convenience: dispatch element-wise shader with numel push constant
inline void dispatch_elementwise(
    const std::string& key,
    const uint32_t* spirv_code,
    size_t spirv_size,
    const std::vector<at::Tensor>& buffers,
    uint32_t numel) {
    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader(key, spirv_code, spirv_size, buffers,
                    workgroups, 1, 1,
                    &numel, sizeof(numel));
}

// Copy src buffer into dst buffer via GPU shader.
// Handles fp16/bf16 correctly (shader operates on float-sized elements, so
// the element count is adjusted for smaller dtypes to avoid buffer overruns).
void dispatch_copy_buffer(const at::Tensor& src, const at::Tensor& dst);

// Copy strided src to contiguous dst via GPU shader (avoids CPU roundtrip).
// Supports up to 5 dimensions.
void dispatch_strided_copy(const at::Tensor& src, const at::Tensor& dst);

// Notify the dispatch layer that a buffer has been written by the CPU host
// (via vkMapMemory / memcpy, i.e. VulkanBuffer::write). The next dispatch_shader
// call that reads this buffer will emit a HOST → COMPUTE pipeline barrier
// to make the host write visible to the GPU compute stage.
void notify_host_write(VkBuffer buf);

// Flush all pending GPU dispatches (submit deferred command buffer + wait).
// Must be called before any host readback of GPU data.
void flush_stream();

// Flush only if there are pending dispatches (avoids overhead of unnecessary flush).
void flush_if_pending();

// Check if a VkBuffer is referenced in the current deferred command buffer.
bool is_buffer_in_flight(VkBuffer buf);

// ── Perf counters ──────────────────────────────────────────────
uint64_t get_dispatch_count();
uint64_t get_flush_count();
uint64_t get_war_flush_count();
uint64_t get_preread_flush_count();
uint64_t get_capacity_flush_count();
uint64_t get_descpool_flush_count();
uint64_t get_barrier_count();
uint64_t get_barrier_skip_count();
void reset_perf_counters();
void inc_war_flush_count();

// ── M17.5: Batch dispatch mode ─────────────────────────────────
// Begin batched dispatch mode — suppresses auto-flush until end_batch_dispatch().
void begin_batch_dispatch();

// End batched dispatch mode — flushes remaining dispatches and resets descriptor pool.
void end_batch_dispatch();

// ── Per-dispatch timing breakdown (nanoseconds, cumulative) ────
// Only populated when TORCH_VULKAN_PROFILE_DISPATCH=1 is set.
// Divide by get_dispatch_count() to get per-dispatch averages.
bool dispatch_profiling_enabled();
uint64_t get_profile_pipeline_cache_ns();
uint64_t get_profile_get_runtime_ns();
uint64_t get_profile_desc_alloc_ns();
uint64_t get_profile_buffer_info_ns();
uint64_t get_profile_desc_write_ns();
uint64_t get_profile_barrier_check_ns();
uint64_t get_profile_cmd_record_ns();
uint64_t get_profile_dirty_track_ns();
void reset_profile_timers();

}} // namespace torch_vulkan::ops
