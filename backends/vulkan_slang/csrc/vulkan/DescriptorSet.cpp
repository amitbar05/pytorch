#include "DescriptorSet.h"
#include "Context.h"
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
    ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
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
    vkResetDescriptorPool(device_, pool_, 0);
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
