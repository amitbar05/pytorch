#include "Stream.h"
#include <stdexcept>

namespace vulkan {

Stream::Stream(VkDevice device, VkQueue queue, uint32_t queue_family)
    : device_(device), queue_(queue) {
    cmd_pool_ = std::make_unique<CommandPool>(device, queue_family);
    // Fences are created per-submission in submit_cmd_buffer() — no
    // pre-allocated fence here, to avoid the reuse hazard described by
    // VUID-vkQueueSubmit-fence-00064.
}

Stream::~Stream() {
    // Drop the callback first — the DescriptorPool that registered it
    // may already be destroyed (C++ destroys struct members in reverse
    // declaration order, and desc_pool is declared after stream in
    // DeviceRuntime). Without this, synchronize() below would invoke
    // a callback into a destroyed DescriptorPool.
    pre_sync_callback_ = nullptr;

    // Flush any pending deferred work
    if (deferred_cmd_ && pending_dispatches_ > 0) {
        try { flush_sync(); } catch (...) {}
    }
    // Wait for any in-flight work
    if (in_flight_cmd_count_ > 0) {
        try { synchronize(); } catch (...) {}
    }
    deferred_cmd_.reset();
    // synchronize() should have cleaned up all fences, but be defensive
    // in case it threw or wasn't called.
    for (VkFence f : retired_fences_) {
        if (f != VK_NULL_HANDLE) vkDestroyFence(device_, f, nullptr);
    }
    retired_fences_.clear();
    if (current_fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(device_, current_fence_, nullptr);
        current_fence_ = VK_NULL_HANDLE;
    }
}

void Stream::submit_and_wait(VkCommandBuffer cmd) {
    submit_cmd_buffer(cmd);
    vkQueueWaitIdle(queue_);
    in_flight_cmd_count_ = 0;

    // Drain any pending descriptor-pool resets before destroying
    // fences (the DescriptorPool may hold references to these fences).
    if (pre_sync_callback_) pre_sync_callback_();

    // All fences have signaled — safe to destroy.
    for (VkFence f : retired_fences_) {
        if (f != VK_NULL_HANDLE) vkDestroyFence(device_, f, nullptr);
    }
    retired_fences_.clear();
    if (current_fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(device_, current_fence_, nullptr);
        current_fence_ = VK_NULL_HANDLE;
    }
}

VkFence Stream::submit(VkCommandBuffer cmd) {
    submit_cmd_buffer(cmd);
    return current_fence_;
}

void Stream::submit_cmd_buffer(VkCommandBuffer cmd) {
    // Create a fresh fence for this submission. Per Vulkan spec
    // VUID-vkQueueSubmit-fence-00064, a fence must not be associated
    // with any queue command that has not yet completed execution.
    // A unique fence per submission is the simplest way to guarantee
    // this — VkFence creation is cheap (driver-internal struct, no
    // GPU round-trip) and they're bulk-destroyed in synchronize()
    // after vkQueueWaitIdle guarantees all have signaled.
    VkFence new_fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fence_ci{};
    fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkResult result = vkCreateFence(device_, &fence_ci, nullptr, &new_fence);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fence for submission");
    }

    // Retire the previous fence (it will be cleaned up in synchronize()
    // after vkQueueWaitIdle guarantees it has signaled).
    if (current_fence_ != VK_NULL_HANDLE) {
        retired_fences_.push_back(current_fence_);
    }
    current_fence_ = new_fence;

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    result = vkQueueSubmit(queue_, 1, &submit_info, current_fence_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit command buffer");
    }
    // M-NEW.4: telemetry — increment AFTER a successful submit. The
    // counter is the M9.2 batching health signal: post-fix the ratio
    // ``g_dispatch_count / submit_count_`` should approach
    // ``MAX_DISPATCHES_PER_CMD`` (32). A ratio close to 1 means the
    // deferred-cmd-buffer batching is defeated.
    submit_count_.fetch_add(1, std::memory_order_relaxed);
}

void Stream::synchronize() {
    if (in_flight_cmd_count_ > 0) {
        vkQueueWaitIdle(queue_);
        in_flight_cmd_count_ = 0;
        // Release in-flight CommandBuffer wrappers. The underlying
        // VkCommandBuffer handles are recycled via cmd_pool_->reset().
        in_flight_cmds_.clear();
        in_flight_buffers_.clear();

        // Drain any pending descriptor-pool resets BEFORE destroying
        // fences. After vkQueueWaitIdle, all submission fences have
        // signaled, so the DescriptorPool's pending_resets_ queue will
        // be fully drained — and the vkGetFenceStatus() calls inside
        // drain_pending_resets() will operate on valid fence handles.
        if (pre_sync_callback_) pre_sync_callback_();

        // All fences have now signaled — safe to destroy retired ones
        // and reset the current fence for reuse on the next submission.
        for (VkFence f : retired_fences_) {
            if (f != VK_NULL_HANDLE) vkDestroyFence(device_, f, nullptr);
        }
        retired_fences_.clear();
        if (current_fence_ != VK_NULL_HANDLE) {
            vkDestroyFence(device_, current_fence_, nullptr);
            current_fence_ = VK_NULL_HANDLE;
        }

        cmd_pool_->reset();
    }
}

bool Stream::is_idle() const {
    if (in_flight_cmd_count_ == 0) return true;
    if (current_fence_ == VK_NULL_HANDLE) return true;
    VkResult result = vkGetFenceStatus(device_, current_fence_);
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
