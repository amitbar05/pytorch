#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace vulkan {

class DescriptorPool {
public:
    DescriptorPool(VkDevice device, uint32_t max_sets = 4096);
    ~DescriptorPool();

    DescriptorPool(const DescriptorPool&) = delete;
    DescriptorPool& operator=(const DescriptorPool&) = delete;

    VkDescriptorSet allocate(VkDescriptorSetLayout layout);

    // Synchronous reset. Calls the pre-reset callback (which usually does
    // ``flush_stream()`` → ``vkQueueWaitIdle`` so any in-flight descriptor
    // uses complete) and then ``vkResetDescriptorPool``. Safe but expensive
    // because the callback drains in-flight work synchronously.
    void reset();

    // M-cpp-new-2: async reset path. Records ``wait_fence`` (the fence
    // the cmd buffer using this pool's descriptors was submitted with)
    // in a "pending reset" queue. The actual ``vkResetDescriptorPool``
    // fires on the next ``drain_pending_resets()`` call IFF the fence
    // has signaled. If the fence has not signaled, the pool is left in
    // its current state and the next ``allocate`` will see whatever
    // capacity is left.
    //
    // Use this on the M9.2 batched-flush hot path: ``flush_async()``
    // submits the cmd buffer + returns; calling ``reset()`` immediately
    // after would either spec-violate (descriptors still in use) or
    // force a fence wait that defeats the batching. ``reset_async()``
    // defers the reset until the fence signals naturally.
    //
    // Gated by ``TORCH_VULKAN_DESCRIPTOR_POOL_ASYNC_RESET=0`` (default
    // on, set to 0 to fall back to synchronous reset).
    void reset_async(VkFence wait_fence);

    // Drain the pending-reset queue: for each pending entry whose fence
    // has signaled, call ``vkResetDescriptorPool`` once and clear the
    // queue. Pending entries whose fences are still un-signaled stay
    // in the queue for the next drain pass. Called from
    // ``Stream::synchronize()`` and from the next batch's
    // ``flush_async()`` boundary as an opportunistic drain.
    void drain_pending_resets();

    // M-cpp-new-6 Layer 2: monotonically-increasing counter bumped on
    // every ``vkResetDescriptorPool`` (whether via ``reset()`` or
    // ``drain_pending_resets()``). Consumers (``dispatch_shader``)
    // snapshot this counter and clear their own descriptor-set caches
    // when the counter changes, because after ``vkResetDescriptorPool``
    // all previously-allocated ``VkDescriptorSet`` handles become
    // invalid.
    uint64_t reset_generation() const { return reset_generation_; }

    // Telemetry: number of times ``reset_async`` was called vs the
    // number of actual ``vkResetDescriptorPool`` invocations that
    // resulted from those calls. Equal counts means every async reset
    // succeeded at drain time; large drain backlog means fences are
    // not signaling fast enough.
    uint64_t async_reset_requests() const { return async_reset_requests_; }
    uint64_t async_resets_drained() const { return async_resets_drained_; }

    // Set callback invoked before pool reset (to flush pending GPU work).
    // Used by the SYNCHRONOUS reset path. ``reset_async`` does NOT call
    // the callback (the whole point is to skip the flush).
    using PreResetCallback = void(*)();
    void set_pre_reset_callback(PreResetCallback cb) { pre_reset_cb_ = cb; }

    VkDescriptorPool pool() const { return pool_; }

private:
    VkDevice device_;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    PreResetCallback pre_reset_cb_ = nullptr;

    // M-cpp-new-2: pending-reset queue. Each entry is a fence whose
    // signal indicates the descriptors allocated from this pool prior
    // to that fence's submission have completed execution.
    std::vector<VkFence> pending_resets_;

    uint64_t async_reset_requests_ = 0;
    uint64_t async_resets_drained_ = 0;
    uint64_t reset_generation_ = 0;

    // Env-knob cached on first access. Same capture-at-first-use
    // pattern as g_profile_enabled in dispatch.cpp — fine for a
    // process-lifetime knob.
    static bool async_reset_enabled();
};

// Bind storage buffers to a descriptor set
void bind_buffers(VkDevice device,
                  VkDescriptorSet set,
                  const std::vector<VkBuffer>& buffers,
                  const std::vector<VkDeviceSize>& sizes);

// Stack-allocated version to avoid heap allocation per dispatch
void bind_buffers(VkDevice device,
                  VkDescriptorSet set,
                  const VkBuffer* buffers,
                  const VkDeviceSize* sizes,
                  uint32_t count);

// N+1.5: bind buffers to a layout that may include descriptor arrays
// (per-binding descriptorCount > 1). The buffers/sizes arrays are flat:
// total = sum(descriptor_counts). Each binding consumes
// descriptor_counts[i] consecutive entries from buffers/sizes.
//   num_bindings = descriptor_counts_n
//   total_buffers = sum(descriptor_counts[0..n))
// `buffers` / `sizes` must have `total_buffers` entries.
void bind_buffers_indexed(VkDevice device,
                          VkDescriptorSet set,
                          const VkBuffer* buffers,
                          const VkDeviceSize* sizes,
                          const uint32_t* descriptor_counts,
                          uint32_t num_bindings);

} // namespace vulkan
