#include "CommandBuffer.h"
#include <stdexcept>

namespace vulkan {

// ── CommandPool ──────────────────────────────────────────────────
CommandPool::CommandPool(VkDevice device, uint32_t queue_family)
    : device_(device) {
    VkCommandPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    // No RESET_COMMAND_BUFFER_BIT — we only ever reset the whole pool via
    // vkResetCommandPool() (see CommandPool::reset). The per-buffer reset
    // flag adds tracking overhead that the driver flags as a best-practices
    // warning when unused. See validation hint
    // BestPractices-vkCreateCommandPool-command-buffer-reset.
    ci.flags = 0;
    ci.queueFamilyIndex = queue_family;

    VkResult result = vkCreateCommandPool(device_, &ci, nullptr, &pool_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }
}

CommandPool::~CommandPool() {
    if (pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, pool_, nullptr);
    }
}

VkCommandBuffer CommandPool::allocate() {
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = pool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd;
    VkResult result = vkAllocateCommandBuffers(device_, &ai, &cmd);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffer");
    }
    return cmd;
}

void CommandPool::free(VkCommandBuffer cmd) {
    vkFreeCommandBuffers(device_, pool_, 1, &cmd);
}

void CommandPool::reset() {
    vkResetCommandPool(device_, pool_, 0);
}

// ── CommandBuffer ────────────────────────────────────────────────
CommandBuffer::CommandBuffer(VkDevice device, CommandPool& pool)
    : device_(device), pool_(pool) {
    cmd_ = pool_.allocate();
}

CommandBuffer::~CommandBuffer() {
    if (cmd_ != VK_NULL_HANDLE) {
        pool_.free(cmd_);
    }
}

void CommandBuffer::begin() {
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkResult result = vkBeginCommandBuffer(cmd_, &bi);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin command buffer");
    }
    recording_ = true;
}

void CommandBuffer::end() {
    VkResult result = vkEndCommandBuffer(cmd_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to end command buffer");
    }
    recording_ = false;
}

void CommandBuffer::bind_pipeline(VkPipeline pipeline) {
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
}

void CommandBuffer::bind_descriptor_set(VkPipelineLayout layout, VkDescriptorSet set) {
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                             layout, 0, 1, &set, 0, nullptr);
}

void CommandBuffer::push_constants(VkPipelineLayout layout, uint32_t size, const void* data) {
    vkCmdPushConstants(cmd_, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, size, data);
}

void CommandBuffer::dispatch(uint32_t x, uint32_t y, uint32_t z) {
    vkCmdDispatch(cmd_, x, y, z);
}

void CommandBuffer::buffer_barrier(VkBuffer buffer, VkDeviceSize size,
                                    VkAccessFlags src, VkAccessFlags dst) {
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = src;
    barrier.dstAccessMask = dst;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.offset = 0;
    barrier.size = size;

    vkCmdPipelineBarrier(cmd_,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 0, nullptr, 1, &barrier, 0, nullptr);
}

void CommandBuffer::memory_barrier(VkAccessFlags src, VkAccessFlags dst) {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = src;
    barrier.dstAccessMask = dst;

    vkCmdPipelineBarrier(cmd_,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void CommandBuffer::host_to_compute_barrier() {
    // Makes all prior CPU writes (vkMapMemory/memcpy paths) visible to the
    // subsequent compute dispatch. The Vulkan memory model (§ 7.1.2) requires
    // this barrier when a buffer is written by the host and then read by a GPU
    // shader in the same command buffer — even on HOST_COHERENT memory where
    // vkFlushMappedMemoryRanges is not needed, the HOST → COMPUTE pipeline
    // barrier is still required to guarantee visibility in the GPU domain.
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd_,
                          VK_PIPELINE_STAGE_HOST_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 1, &barrier, 0, nullptr, 0, nullptr);
}

} // namespace vulkan
