#pragma once

#include "CommandBuffer.h"

#include <vulkan/vulkan.h>
#include <memory>
#include <unordered_set>

namespace vulkan {

class Stream {
public:
    Stream(VkDevice device, VkQueue queue, uint32_t queue_family);
    ~Stream();

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    // Submit a command buffer and wait
    void submit_and_wait(VkCommandBuffer cmd);

    // Submit a command buffer with fence
    VkFence submit(VkCommandBuffer cmd);

    // ── Deferred execution (command buffer batching) ──────────────
    // Maximum dispatches per deferred command buffer before auto-submit.
    // Balances submission amortization against descriptor pool pressure.
    static constexpr uint32_t MAX_DISPATCHES_PER_CMD = 32;

    // Get the active command buffer for recording dispatches.
    // Automatically begins a new command buffer if none is active.
    CommandBuffer& deferred_cmd();

    // Number of dispatches recorded in the current deferred command buffer.
    uint32_t pending_dispatches() const { return pending_dispatches_; }
    void inc_pending() { pending_dispatches_++; }

    // ── Flush control ────────────────────────────────────────────
    // Flush-async: end the active command buffer, submit to GPU,
    // immediately return without waiting. Starts a new command buffer
    // for subsequent dispatches. No-op if no dispatches are pending.
    // Callers that need CPU-visible results must call synchronize() or
    // flush_sync() afterwards.
    void flush_async();

    // Flush-sync: submit all pending work and block until GPU completes.
    // Equivalent to flush_async() + synchronize(). Use when CPU readback
    // is needed (e.g. before tensor .cpu() or .item()).
    void flush_sync();

    // Deprecated: alias for flush_sync(). Kept for backward compat.
    void flush();

    // Block until all in-flight GPU submissions complete.
    void synchronize();

    // Check if any submissions are in-flight (not yet completed).
    bool has_in_flight_work() const { return in_flight_cmd_count_ > 0; }

    // Check if fence is signaled (all submits complete, non-blocking).
    bool is_idle() const;

    // Track buffers used in current deferred batch for WAR hazard detection.
    void track_buffer(VkBuffer buf) { pending_buffers_.insert(buf); }

    // Check if a buffer is referenced by ANY pending or in-flight command buffer.
    // Must return true for both deferred (not yet submitted) and in-flight
    // (submitted via flush_async but not yet synchronized) work, so that
    // VulkanBuffer::read() can correctly decide whether to flush before reading.
    bool is_buffer_pending(VkBuffer buf) const {
        return pending_buffers_.count(buf) > 0 ||
               in_flight_buffers_.count(buf) > 0;
    }

    VkQueue queue() const { return queue_; }
    CommandPool& command_pool() { return *cmd_pool_; }

private:
    // Submit a single command buffer (internal, no wait).
    void submit_cmd_buffer(VkCommandBuffer cmd);

    VkDevice device_;
    VkQueue queue_;
    std::unique_ptr<CommandPool> cmd_pool_;
    VkFence fence_ = VK_NULL_HANDLE;

    // Deferred command buffer state
    std::unique_ptr<CommandBuffer> deferred_cmd_;
    uint32_t pending_dispatches_ = 0;
    std::unordered_set<VkBuffer> pending_buffers_;

    // In-flight command buffers: submitted but not yet completed.
    // Held here to prevent vkFreeCommandBuffers until synchronize()
    // resets the command pool (after vkQueueWaitIdle).
    std::vector<std::unique_ptr<CommandBuffer>> in_flight_cmds_;

    // Buffers referenced by command buffers submitted via flush_async()
    // but not yet synchronized. Used by is_buffer_pending() so that
    // VulkanBuffer::read() flushes before reading a buffer that has
    // async-submitted GPU work in flight.
    std::unordered_set<VkBuffer> in_flight_buffers_;

    // In-flight tracking: how many submissions are pending.
    uint32_t in_flight_cmd_count_ = 0;
};

} // namespace vulkan
