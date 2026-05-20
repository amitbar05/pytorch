#include "DescriptorSet.h"
#include "Context.h"

#include <cstdlib>
#include <stdexcept>

namespace vulkan {

DescriptorPool::DescriptorPool(VkDevice device, uint32_t max_sets)
    : device_(device) {

    bool desc_idx = Context::instance().descriptor_indexing_enabled();
    // With descriptor indexing: up to 256 bindings per set for aggressive fusion.
    // Without: 16 bindings (half of typical maxPerStageDescriptorStorageBuffers=32).
    uint32_t max_bindings_per_set = desc_idx ? 256 : 16;

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = max_sets * max_bindings_per_set;

    VkDescriptorPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    // No FREE_DESCRIPTOR_SET_BIT — we only ever reset the whole pool via
    // vkResetDescriptorPool() (see DescriptorPool::reset). The per-set free
    // flag adds tracking overhead that the driver flags as a best-practices
    // warning when unused. See validation hint
    // BestPractices-vkCreateDescriptorPool-pool-cleared-or-free-set-bit-set.
    ci.flags = 0;
    if (desc_idx) {
        ci.flags |= VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    }
    ci.maxSets = max_sets;
    ci.poolSizeCount = 1;
    ci.pPoolSizes = &pool_size;

    VkResult result = vkCreateDescriptorPool(device_, &ci, nullptr, &pool_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool");
    }
}

DescriptorPool::~DescriptorPool() {
    if (pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, pool_, nullptr);
    }
}

VkDescriptorSet DescriptorPool::allocate(VkDescriptorSetLayout layout) {
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = pool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &layout;

    VkDescriptorSet set;
    VkResult result = vkAllocateDescriptorSets(device_, &ai, &set);
    if (result != VK_SUCCESS) {
        // Pool exhausted — flush pending GPU work then reset
        if (pre_reset_cb_) pre_reset_cb_();
        reset();
        result = vkAllocateDescriptorSets(device_, &ai, &set);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate descriptor set");
        }
    }
    return set;
}

void DescriptorPool::reset() {
    if (pre_reset_cb_) pre_reset_cb_();
    // Sync path also drains any pending async resets — they're
    // redundant now that we've forced a flush, but draining keeps
    // the pending list bounded.
    drain_pending_resets();
    vkResetDescriptorPool(device_, pool_, 0);
    reset_generation_++;  // M-cpp-new-6 Layer 2: invalidate caches
}

// M-cpp-new-2: env-knob cache. Reads ``TORCH_VULKAN_DESCRIPTOR_POOL_ASYNC_RESET``
// once at first call and remembers. Treats "0" / "false" as disabled
// (fall back to synchronous reset); everything else (including unset)
// is enabled.
bool DescriptorPool::async_reset_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("TORCH_VULKAN_DESCRIPTOR_POOL_ASYNC_RESET");
        if (!env) return true;
        if (env[0] == '0' && env[1] == '\0') return false;
        if (env[0] == 'f' || env[0] == 'F') return false;  // false/False
        return true;
    }();
    return enabled;
}

void DescriptorPool::reset_async(VkFence wait_fence) {
    if (!async_reset_enabled() || wait_fence == VK_NULL_HANDLE) {
        // Fallback: synchronous reset. Caller's ``pre_reset_cb_``
        // (typically ``flush_stream``) has already waited for the
        // queue idle in the legacy path, but here we go through the
        // standard sync path which does the wait too.
        reset();
        return;
    }

    // Blocker F fix: push the new fence BEFORE attempting any drain.
    // ``reset_async(F_N)`` is called immediately after ``flush_async()``
    // submits cmd buffer N with fence ``F_N``; that cmd buffer is now
    // in-flight and still references descriptors allocated from this
    // pool. If we drain BEFORE pushing, ``drain_pending_resets()`` sees
    // only earlier fences (e.g. ``[F_{N-1}]``); if ``F_{N-1}`` has
    // signaled, drain will call ``vkResetDescriptorPool`` while cmd
    // buffer N is still using descriptors — VUID-vkResetDescriptorPool-
    // descriptorPool-00313. Pushing first guarantees the drain pass
    // only resets when ALL queued fences (including the one we just
    // pushed) have signaled.
    pending_resets_.push_back(wait_fence);
    async_reset_requests_++;

    // Poll opportunistically: if ``wait_fence`` is already signaled
    // (e.g. small fast workload), drain immediately to reset the pool.
    // Fences signal in queue submission order, so a signaled
    // ``wait_fence`` implies every earlier pending fence has signaled
    // too — safe to reset.
    VkResult status = vkGetFenceStatus(device_, wait_fence);
    if (status == VK_SUCCESS) {
        drain_pending_resets();
    }
}

void DescriptorPool::drain_pending_resets() {
    if (pending_resets_.empty()) return;

    // Walk the queue; keep entries whose fence has NOT yet signaled.
    //
    // VUID-vkResetDescriptorPool-00313: we must not reset the pool
    // while ANY command buffer using descriptors allocated from it is
    // still in-flight.  Since every pending fence was submitted after
    // the descriptors it guards were recorded, we can only reset when
    // ALL pending fences have signaled — not just any one of them.
    //
    // Fences on a single queue signal in submission order, so once
    // the last entry has signaled, all earlier ones are guaranteed
    // signaled too.  We still walk the whole list (cheap: at most a
    // handful of entries) and keep the suffix that hasn't signaled.
    std::vector<VkFence> still_pending;
    still_pending.reserve(pending_resets_.size());
    for (VkFence f : pending_resets_) {
        VkResult status = vkGetFenceStatus(device_, f);
        if (status != VK_SUCCESS) {
            still_pending.push_back(f);
        }
    }

    // Only reset when all entries drained — otherwise descriptors
    // belonging to the still-pending submissions are still in flight.
    bool all_drained = still_pending.empty();
    pending_resets_ = std::move(still_pending);

    if (all_drained) {
        vkResetDescriptorPool(device_, pool_, 0);
        async_resets_drained_++;
        reset_generation_++;  // M-cpp-new-6 Layer 2: invalidate caches
    }
}

void bind_buffers(VkDevice device,
                  VkDescriptorSet set,
                  const VkBuffer* buffers,
                  const VkDeviceSize* sizes,
                  uint32_t count) {
    // Stack-allocated arrays — capacity grows with descriptor indexing.
    // Without descriptor indexing: 32 bindings (sgd_batch15 uses 30).
    // With descriptor indexing: 256 bindings for aggressive fusion.
    static const uint32_t MAX_BINDINGS =
        Context::instance().descriptor_indexing_enabled() ? 256 : 32;
    VkDescriptorBufferInfo buf_infos[MAX_BINDINGS];
    VkWriteDescriptorSet writes[MAX_BINDINGS];

    for (uint32_t i = 0; i < count; i++) {
        buf_infos[i] = {};
        buf_infos[i].buffer = buffers[i];
        buf_infos[i].offset = 0;
        buf_infos[i].range = sizes[i];

        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buf_infos[i];
    }

    vkUpdateDescriptorSets(device, count, writes, 0, nullptr);
}

void bind_buffers(VkDevice device,
                  VkDescriptorSet set,
                  const std::vector<VkBuffer>& buffers,
                  const std::vector<VkDeviceSize>& sizes) {
    bind_buffers(device, set, buffers.data(), sizes.data(),
                 static_cast<uint32_t>(buffers.size()));
}

void bind_buffers_indexed(VkDevice device,
                          VkDescriptorSet set,
                          const VkBuffer* buffers,
                          const VkDeviceSize* sizes,
                          const uint32_t* descriptor_counts,
                          uint32_t num_bindings) {
    // Same MAX cap as bind_buffers — applies to total buffers, not bindings.
    static const uint32_t MAX_BINDINGS =
        Context::instance().descriptor_indexing_enabled() ? 256 : 32;
    VkDescriptorBufferInfo buf_infos[MAX_BINDINGS];
    VkWriteDescriptorSet writes[MAX_BINDINGS];

    // First, materialize all VkDescriptorBufferInfo entries (one per buffer).
    uint32_t total = 0;
    for (uint32_t i = 0; i < num_bindings; ++i) {
        total += descriptor_counts[i];
    }
    if (total > MAX_BINDINGS) {
        throw std::runtime_error(
            "bind_buffers_indexed: total buffers exceeds MAX_BINDINGS cap");
    }
    for (uint32_t b = 0; b < total; ++b) {
        buf_infos[b] = {};
        buf_infos[b].buffer = buffers[b];
        buf_infos[b].offset = 0;
        buf_infos[b].range = sizes[b];
    }

    // One VkWriteDescriptorSet per binding. Each consumes descriptor_counts[i]
    // consecutive VkDescriptorBufferInfo entries via pBufferInfo[].
    uint32_t buf_offset = 0;
    for (uint32_t i = 0; i < num_bindings; ++i) {
        const uint32_t cnt = descriptor_counts[i];
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorCount = cnt;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buf_infos[buf_offset];
        buf_offset += cnt;
    }

    vkUpdateDescriptorSets(device, num_bindings, writes, 0, nullptr);
}

} // namespace vulkan
