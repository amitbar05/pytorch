#pragma once

#include "CommandBuffer.h"

#include <vulkan/vulkan.h>
#include <atomic>
#include <cstdint>
#include <functional>
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

    // M-cpp-new-2: expose the submission fence so callers can use the
    // ``DescriptorPool::reset_async(fence)`` path. A fresh fence is
    // created for every ``vkQueueSubmit`` call (per
    // VUID-vkQueueSubmit-fence-00064), so this always returns the most
    // recent submission's fence — exactly what ``reset_async`` needs to
    // defer pool reset until that submission's descriptors are done.
    VkFence fence() const { return current_fence_; }

    // M-NEW.4: cumulative ``vkQueueSubmit`` call count from this
    // stream's batched-flush hot path (``submit_cmd_buffer``). Read
    // via the ``_stream_submit_count()`` pybind in ``init.cpp``.
    //
    // The canonical M9.2 batching telemetry: the ratio
    // ``g_dispatch_count / submit_count`` should approach
    // ``MAX_DISPATCHES_PER_CMD`` (32) post-fix, since M9.2 collapses
    // up to 32 dispatches per ``vkQueueSubmit`` call. A ratio close
    // to 1 means batching is defeated (regression).
    uint64_t submit_count() const noexcept {
        return submit_count_.load(std::memory_order_relaxed);
    }

    // Set a callback invoked inside synchronize() / submit_and_wait()
    // after vkQueueWaitIdle but before fence destruction. Used by
    // DescriptorPool to drain its pending_resets_ queue while the
    // fences it references are still alive.
    void set_pre_sync_callback(std::function<void()> cb) {
        pre_sync_callback_ = std::move(cb);
    }

private:
    // Submit a single command buffer (internal, no wait).
    void submit_cmd_buffer(VkCommandBuffer cmd);

    VkDevice device_;
    VkQueue queue_;
    std::unique_ptr<CommandPool> cmd_pool_;

    // Per-submission fence: created fresh in submit_cmd_buffer() for
    // every vkQueueSubmit call, retired to retired_fences_ on the next
    // submission. retired_fences_ are bulk-destroyed in synchronize()
    // after vkQueueWaitIdle guarantees all have signaled.
    VkFence current_fence_ = VK_NULL_HANDLE;
    std::vector<VkFence> retired_fences_;

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

    // M-NEW.4: cumulative ``vkQueueSubmit`` count, incremented in
    // ``submit_cmd_buffer`` immediately after a successful submit.
    // Atomic so the ``_stream_submit_count()`` pybind can read it
    // from any thread without locking.
    std::atomic<uint64_t> submit_count_{0};

    // Invoked by synchronize() after vkQueueWaitIdle but before
    // fence destruction. DescriptorPool registers drain_pending_resets
    // so that its pending_resets_ queue is drained while fences are
    // still valid.
    std::function<void()> pre_sync_callback_;
};

} // namespace vulkan
