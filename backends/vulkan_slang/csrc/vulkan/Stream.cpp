#include "Stream.h"
#include <stdexcept>

namespace vulkan {

Stream::Stream(VkDevice device, VkQueue queue, uint32_t queue_family)
    : device_(device), queue_(queue) {
    cmd_pool_ = std::make_unique<CommandPool>(device, queue_family);

    VkFenceCreateInfo fence_ci{};
    fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkResult result = vkCreateFence(device_, &fence_ci, nullptr, &fence_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fence for stream");
    }
}

Stream::~Stream() {
    // Flush any pending deferred work
    if (deferred_cmd_ && pending_dispatches_ > 0) {
        try { flush_sync(); } catch (...) {}
    }
    // Wait for any in-flight work
    if (in_flight_cmd_count_ > 0) {
        try { synchronize(); } catch (...) {}
    }
    deferred_cmd_.reset();
    if (fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(device_, fence_, nullptr);
    }
}

void Stream::submit_and_wait(VkCommandBuffer cmd) {
    submit_cmd_buffer(cmd);
    vkQueueWaitIdle(queue_);
    in_flight_cmd_count_ = 0;
}

VkFence Stream::submit(VkCommandBuffer cmd) {
    submit_cmd_buffer(cmd);
    return fence_;
}

void Stream::submit_cmd_buffer(VkCommandBuffer cmd) {
    // Only reset the fence if no async work is in flight — otherwise
    // the fence from a previous flush_async() is still pending on the
    // GPU and vkResetFences would violate the spec.
    // When in_flight_cmd_count_ > 0, the fence will be reset in
    // synchronize() after vkQueueWaitIdle.
    if (in_flight_cmd_count_ == 0) {
        vkResetFences(device_, 1, &fence_);
    }

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    VkResult result = vkQueueSubmit(queue_, 1, &submit_info, fence_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit command buffer");
    }
}

void Stream::synchronize() {
    if (in_flight_cmd_count_ > 0) {
        vkQueueWaitIdle(queue_);
        in_flight_cmd_count_ = 0;
        // Release in-flight CommandBuffer wrappers. The underlying
        // VkCommandBuffer handles are recycled via cmd_pool_->reset().
        in_flight_cmds_.clear();
        in_flight_buffers_.clear();
        // Safe to reset fence now — GPU is idle.
        vkResetFences(device_, 1, &fence_);
        cmd_pool_->reset();
    }
}

bool Stream::is_idle() const {
    if (in_flight_cmd_count_ == 0) return true;
    VkResult result = vkGetFenceStatus(device_, fence_);
    return result == VK_SUCCESS;
}

// ── Deferred execution ──────────────────────────────────────────

CommandBuffer& Stream::deferred_cmd() {
    if (!deferred_cmd_) {
        deferred_cmd_ = std::make_unique<CommandBuffer>(device_, *cmd_pool_);
        deferred_cmd_->begin();
    }
    return *deferred_cmd_;
}

void Stream::flush_async() {
    if (!deferred_cmd_ || pending_dispatches_ == 0) return;

    deferred_cmd_->end();
    submit_cmd_buffer(deferred_cmd_->handle());

    // Move the CommandBuffer to in_flight_cmds_ so the underlying
    // VkCommandBuffer is not freed until synchronize() drains the list
    // and resets the command pool.
    in_flight_cmds_.push_back(std::move(deferred_cmd_));
    deferred_cmd_.reset();
    // Move buffers from pending to in-flight so is_buffer_pending()
    // still returns true for async-submitted work that hasn't been
    // synchronized yet.
    in_flight_buffers_.insert(pending_buffers_.begin(), pending_buffers_.end());
    pending_buffers_.clear();
    pending_dispatches_ = 0;
    in_flight_cmd_count_++;
}

void Stream::flush_sync() {
    if (!deferred_cmd_ || pending_dispatches_ == 0) {
        // No new dispatches to submit, but there may be in-flight work.
        if (in_flight_cmd_count_ > 0) synchronize();
        return;
    }
    flush_async();
    synchronize();
}

void Stream::flush() {
    flush_sync();
}

} // namespace vulkan
