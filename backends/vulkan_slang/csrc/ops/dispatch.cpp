#include "dispatch.h"
#include "../generated/shaders.h"

#include <atomic>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace torch_vulkan { namespace ops {

// Forward declarations for callbacks
static void flush_stream_callback();
static void desc_pool_flush_callback();
static bool is_buffer_in_flight_callback(VkBuffer buf);

// ── Per-device runtime (lazy init) ──────────────────────────────
static std::unordered_map<uint32_t, DeviceRuntime> g_runtimes;
static std::recursive_mutex g_runtime_mutex;
static bool g_callback_registered = false;

// ── Perf counters ──────────────────────────────────────────────
static std::atomic<uint64_t> g_dispatch_count{0};
static std::atomic<uint64_t> g_flush_count{0};
static std::atomic<uint64_t> g_war_flush_count{0};
static std::atomic<uint64_t> g_preread_flush_count{0};
static std::atomic<uint64_t> g_capacity_flush_count{0};
static std::atomic<uint64_t> g_descpool_flush_count{0};
static std::atomic<uint64_t> g_barrier_count{0};      // barriers actually emitted
static std::atomic<uint64_t> g_barrier_skip_count{0}; // barriers skipped (independent dispatches)

// ── Per-dispatch timing breakdown (only active with env var) ────
static bool g_profile_enabled = (getenv("TORCH_VULKAN_PROFILE_DISPATCH") != nullptr);
static std::atomic<uint64_t> g_profile_pipeline_cache_ns{0};
static std::atomic<uint64_t> g_profile_get_runtime_ns{0};
static std::atomic<uint64_t> g_profile_desc_alloc_ns{0};
static std::atomic<uint64_t> g_profile_buffer_info_ns{0};
static std::atomic<uint64_t> g_profile_desc_write_ns{0};
static std::atomic<uint64_t> g_profile_barrier_check_ns{0};
static std::atomic<uint64_t> g_profile_cmd_record_ns{0};
static std::atomic<uint64_t> g_profile_dirty_track_ns{0};

// Helper: wall-clock time in nanoseconds
static inline uint64_t _now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count());
}

uint64_t get_dispatch_count() { return g_dispatch_count.load(); }
uint64_t get_flush_count() { return g_flush_count.load(); }
uint64_t get_war_flush_count() { return g_war_flush_count.load(); }
uint64_t get_preread_flush_count() { return g_preread_flush_count.load(); }
uint64_t get_capacity_flush_count() { return g_capacity_flush_count.load(); }
uint64_t get_descpool_flush_count() { return g_descpool_flush_count.load(); }
uint64_t get_barrier_count() { return g_barrier_count.load(); }
uint64_t get_barrier_skip_count() { return g_barrier_skip_count.load(); }

bool dispatch_profiling_enabled() { return g_profile_enabled; }
uint64_t get_profile_pipeline_cache_ns() { return g_profile_pipeline_cache_ns.load(); }
uint64_t get_profile_get_runtime_ns() { return g_profile_get_runtime_ns.load(); }
uint64_t get_profile_desc_alloc_ns() { return g_profile_desc_alloc_ns.load(); }
uint64_t get_profile_buffer_info_ns() { return g_profile_buffer_info_ns.load(); }
uint64_t get_profile_desc_write_ns() { return g_profile_desc_write_ns.load(); }
uint64_t get_profile_barrier_check_ns() { return g_profile_barrier_check_ns.load(); }
uint64_t get_profile_cmd_record_ns() { return g_profile_cmd_record_ns.load(); }
uint64_t get_profile_dirty_track_ns() { return g_profile_dirty_track_ns.load(); }
void reset_profile_timers() {
    g_profile_pipeline_cache_ns = 0;
    g_profile_get_runtime_ns = 0;
    g_profile_desc_alloc_ns = 0;
    g_profile_buffer_info_ns = 0;
    g_profile_desc_write_ns = 0;
    g_profile_barrier_check_ns = 0;
    g_profile_cmd_record_ns = 0;
    g_profile_dirty_track_ns = 0;
}

void reset_perf_counters() {
    g_dispatch_count = 0;
    g_flush_count = 0;
    g_war_flush_count = 0;
    g_preread_flush_count = 0;
    g_capacity_flush_count = 0;
    g_barrier_count = 0;
    g_barrier_skip_count = 0;
    g_descpool_flush_count = 0;
    reset_profile_timers();
}
void inc_war_flush_count() { g_war_flush_count++; }

// ── M17.5: Batch dispatch mode ─────────────────────────────────

void begin_batch_dispatch() {
    auto& ctx = vulkan::Context::instance();
    auto& rt = get_runtime(ctx.current_device());
    rt.batch_mode = true;
}

void end_batch_dispatch() {
    auto& ctx = vulkan::Context::instance();
    auto& rt = get_runtime(ctx.current_device());
    if (!rt.batch_mode) return;
    rt.batch_mode = false;
    // Flush any remaining dispatches + reset descriptor pool.
    // M-cpp-new-2: use ``reset_async(fence)`` so the M9.2 batching win
    // isn't defeated by a synchronous fence-wait at every batch end.
    // The actual ``vkResetDescriptorPool`` fires on the next batch's
    // drain pass once the fence has signaled.
    if (rt.stream && rt.stream->pending_dispatches() > 0) {
        rt.stream->flush_async();
        rt.desc_pool->reset_async(rt.stream->fence());
        rt.dirty_buffers.clear();
        rt.read_buffers.clear();  // S2.0d
        rt.host_written_buffers.clear();
        {
            std::lock_guard<std::mutex> _lk(rt.desc_set_mutex_);
            rt.desc_set_cache.clear();  // M17.5: clear on batch end
        }
    }
}

// Fast path cache for single-device case (avoids mutex + map lookup per dispatch)
static thread_local DeviceRuntime* g_cached_runtime = nullptr;
static thread_local uint32_t g_cached_device_index = UINT32_MAX;

DeviceRuntime& get_runtime(uint32_t device_index) {
    auto& ctx = vulkan::Context::instance();
    if (device_index == UINT32_MAX) device_index = ctx.current_device();

    // Fast path: return cached runtime for same device
    if (g_cached_runtime && g_cached_device_index == device_index) {
        return *g_cached_runtime;
    }

    std::lock_guard<std::recursive_mutex> lock(g_runtime_mutex);
    auto it = g_runtimes.find(device_index);
    if (it != g_runtimes.end()) {
        g_cached_runtime = &it->second;
        g_cached_device_index = device_index;
        return it->second;
    }

    auto& rt = g_runtimes[device_index];
    rt.stream = std::make_unique<vulkan::Stream>(
        ctx.device(device_index),
        ctx.compute_queue(device_index),
        ctx.compute_queue_family(device_index));
    rt.desc_pool = std::make_unique<vulkan::DescriptorPool>(
        ctx.device(device_index), 4096);
    rt.desc_pool->set_pre_reset_callback(desc_pool_flush_callback);

    // Wire the Stream's pre-sync callback to drain the DescriptorPool's
    // pending_resets_ queue BEFORE fence destruction.  This fixes the
    // M-cpp-new-2 use-after-free: synchronize() destroys fences after
    // vkQueueWaitIdle, but reset_async() had stored those fence handles
    // in pending_resets_.  By draining first, fences are still valid
    // when vkGetFenceStatus is called on them.
    //
    // The lambda captures by value (not reference) so it is safe even
    // if rt moves in the unordered_map.  Both unique_ptrs are stable.
    rt.stream->set_pre_sync_callback([desc_pool = rt.desc_pool.get()]() {
        desc_pool->drain_pending_resets();
    });

    // Register callbacks so VulkanBuffer::read() auto-flushes only when needed
    if (!g_callback_registered) {
        vulkan::set_pre_read_callback(flush_stream_callback);
        vulkan::set_is_buffer_in_flight_callback(is_buffer_in_flight_callback);
        g_callback_registered = true;
    }
    g_cached_runtime = &rt;
    g_cached_device_index = device_index;
    return rt;
}

void cleanup_runtimes() {
    std::lock_guard<std::recursive_mutex> lock(g_runtime_mutex);
    // Flush any pending work before destroying runtimes
    for (auto& [idx, rt] : g_runtimes) {
        if (rt.stream && rt.stream->pending_dispatches() > 0) {
            try { rt.stream->flush_sync(); } catch (...) {}
        }
    }
    g_runtimes.clear();
    g_cached_runtime = nullptr;
    g_cached_device_index = UINT32_MAX;
}

// ── Buffer extraction ───────────────────────────────────────────
BufferInfo get_buffer_info(const at::Tensor& tensor) {
    if (tensor.storage_offset() > 0) {
        // Offset view: data_ptr() = opaque_base + storage_offset * itemsize.
        // A naive get_buffer(data_ptr()) collides when another tensor's opaque
        // ID happens to equal that offset-shifted value. Recover the base by
        // arithmetic subtraction so the lookup is always to the parent buffer.
        VkDeviceSize off_bytes =
            static_cast<VkDeviceSize>(tensor.storage_offset()) *
            static_cast<VkDeviceSize>(tensor.element_size());
        static bool _dbg = (getenv("TORCH_VULKAN_DEBUG_OFFSET") != nullptr);
        if (_dbg) {
            fprintf(stderr, "DBG get_buffer_info: storage_offset=%ld off_bytes=%lu\n",
                    (long)tensor.storage_offset(), (unsigned long)off_bytes);
            fflush(stderr);
        }
        void* base = reinterpret_cast<void*>(
            reinterpret_cast<uintptr_t>(tensor.data_ptr()) -
            static_cast<uintptr_t>(off_bytes));
        auto* buf = VulkanAllocator::instance().get_buffer(base);
        if (buf != nullptr && buf->is_valid()) {
            return {buf->buffer(), buf->size(), off_bytes};
        }
        // Arithmetic path failed; fall back to C10 storage pointer.
        buf = VulkanAllocator::instance().get_buffer(
            tensor.storage().data_ptr().get());
        TORCH_CHECK(buf && buf->is_valid(),
                    "Tensor has no backing Vulkan buffer");
        return {buf->buffer(), buf->size(), off_bytes};
    }
    auto* buf = VulkanAllocator::instance().get_buffer(tensor.data_ptr());
    TORCH_CHECK(buf && buf->is_valid(),
                "Tensor has no backing Vulkan buffer");
    return {buf->buffer(), buf->size(), 0};
}

// ── Shader dispatch (deferred) ──────────────────────────────────
void dispatch_shader(
    const std::string& key,
    const uint32_t* spirv_code,
    size_t spirv_size,
    const std::vector<at::Tensor>& tensors,
    uint32_t num_workgroups_x,
    uint32_t num_workgroups_y,
    uint32_t num_workgroups_z,
    const void* push_constants,
    uint32_t push_constants_size,
    uint32_t num_outputs,
    const std::vector<SpecConstant>& spec_constants,
    const std::vector<VkDeviceSize>* per_tensor_byte_offsets) {

    uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0;
    if (g_profile_enabled) t0 = _now_ns();

    auto& ctx = vulkan::Context::instance();
    auto device_idx = ctx.current_device();
    VkDevice device = ctx.device(device_idx);

    // Get or create cached pipeline
    auto* pipeline = vulkan::PipelineCache::instance().get_or_create(
        device, key, spirv_code, spirv_size,
        static_cast<uint32_t>(tensors.size()),
        push_constants_size, spec_constants);

    if (g_profile_enabled) { t1 = _now_ns(); g_profile_pipeline_cache_ns += (t1 - t0); }

    // Get runtime (stream + descriptor pool)
    auto& rt = get_runtime(device_idx);

    if (g_profile_enabled) { t2 = _now_ns(); g_profile_get_runtime_ns += (t2 - t1); }

    // M9.2 batched submission: submit async every 8 dispatches so the
    // GPU can overlap compute with CPU recording the next batch.
    // M-cpp-new-2: ``reset_async(fence)`` defers the actual
    // ``vkResetDescriptorPool`` until the just-submitted cmd buffer's
    // fence signals. The synchronous ``reset()`` here would either
    // spec-violate (descriptors still in use by the in-flight cmd
    // buffer) or force a fence-wait that defeats the M9.2 batching.
    if (!rt.batch_mode && rt.stream->pending_dispatches() >= vulkan::Stream::MAX_DISPATCHES_PER_CMD) {
        g_capacity_flush_count++;
        rt.stream->flush_async();
        rt.desc_pool->reset_async(rt.stream->fence());
        rt.dirty_buffers.clear();
        rt.read_buffers.clear();  // S2.0d
        rt.host_written_buffers.clear();
        {
            std::lock_guard<std::mutex> _lk(rt.desc_set_mutex_);
            rt.desc_set_cache.clear();  // M17.5: clear on pool reset
        }
    }

    // M-cpp-new-5: gate the M17.5 descriptor-set cache on descriptor
    // indexing. The cache lets us reuse a single `VkDescriptorSet` for
    // the same pipeline across multiple dispatches in a batch — which
    // means we call `vkUpdateDescriptorSets` (inside `bind_buffers`)
    // on a set that may already be bound by a previously-recorded but
    // un-submitted command buffer. Per Vulkan spec § 14.2.1 this is UB
    // unless the pool was created with
    // `VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT`, and that flag
    // is only set when `VK_EXT_descriptor_indexing` is available
    // (see `DescriptorSet.cpp:27-29`). On platforms / drivers where
    // the extension is missing M17.5's cache reuse would silently
    // violate VUID-VkWriteDescriptorSet-dstSet-04611.
    //
    // M-cpp-new-6: even with UPDATE_AFTER_BIND, sharing one descriptor
    // set between two recorded dispatches with DIFFERENT buffers is
    // wrong — both bind calls reference the same `VkDescriptorSet`
    // handle, and the second `vkUpdateDescriptorSets` overwrites the
    // bindings the first dispatch needed. When the cmd buffer executes,
    // both dispatches see the LATEST descriptor contents, corrupting
    // any chain like `x.relu().relu()`. The fix below caches by
    // (layout, buffer-list-hash) instead of (layout) alone, so the
    // M17.5 fast path still hits for repeated dispatches with the
    // SAME buffers (the autotune / multi-launch case M17.5 was designed
    // for) but a chain with different intermediates gets a fresh set
    // per dispatch.
    // M-pipeline-5: per-call read of descriptor-indexing state, NOT a
    // `static const` capture-at-first-use. The latter would freeze the
    // first-call value for the whole process — fine today (the env knob
    // is read once at backend init per M-cpp-new-5) but a latent footgun
    // for future test fixtures or `Context::reset()` paths that toggle
    // the state at runtime.
    const bool kUseDescCache =
        vulkan::Context::instance().descriptor_indexing_enabled();

    // M-pipeline-5: ``MAX_BINDINGS_CAP`` is the COMPILE-TIME stack-array
    // size (the larger of the two possible values: 256 for descriptor
    // indexing, 32 legacy). ``max_bindings`` is the per-call RUNTIME
    // cap used for the `n <= max_bindings` precondition check elsewhere.
    // Allocating the upper bound on stack costs at most ~3 KB of stack
    // (256 × 8 B VkBuffer + 256 × 8 B VkDeviceSize = 4 KB), negligible
    // vs. PyTorch's default stack budget. The previous `static const`
    // expression forced GCC into VLA-extension territory; constexpr
    // makes it a pure C++ stack array.
    constexpr uint32_t MAX_BINDINGS_CAP = 256;
    const uint32_t max_bindings = kUseDescCache ? 256u : 32u;
    VkBuffer vk_buffers_arr[MAX_BINDINGS_CAP];
    VkDeviceSize vk_sizes_arr[MAX_BINDINGS_CAP];
    VkDeviceSize vk_offsets_arr[MAX_BINDINGS_CAP];
    uint32_t n = static_cast<uint32_t>(tensors.size());
    TORCH_CHECK(n <= max_bindings,
        "dispatch_shader: total buffer count ", n,
        " exceeds max_bindings ", max_bindings,
        " (descriptor_indexing=", kUseDescCache, ")");

    for (uint32_t i = 0; i < n; ++i) {
        auto info = get_buffer_info(tensors[i]);
        vk_buffers_arr[i] = info.buffer;
        vk_sizes_arr[i] = info.size;
        if (per_tensor_byte_offsets && i < per_tensor_byte_offsets->size() &&
            (*per_tensor_byte_offsets)[i] != VK_WHOLE_SIZE) {
            vk_offsets_arr[i] = (*per_tensor_byte_offsets)[i];
        } else {
            vk_offsets_arr[i] = info.offset;
        }
    }

    if (g_profile_enabled) { t4 = _now_ns(); g_profile_buffer_info_ns += (t4 - t3); }

    // FNV-1a 64-bit hash of the bound VkBuffer handles + byte offsets, in
    // order. Offsets must be included so that two reinterpret_tensor views
    // into the same VkBuffer at different storage_offsets get distinct
    // descriptor sets (otherwise the cache would return a stale set with the
    // wrong VkDescriptorBufferInfo.offset, corrupting the binding).
    uint64_t buffers_hash = 0xcbf29ce484222325ull;
    for (uint32_t i = 0; i < n; ++i) {
        uint64_t v = reinterpret_cast<uint64_t>(vk_buffers_arr[i]);
        for (int b = 0; b < 8; ++b) {
            buffers_hash ^= (v >> (b * 8)) & 0xff;
            buffers_hash *= 0x100000001b3ull;
        }
        uint64_t o = static_cast<uint64_t>(vk_offsets_arr[i]);
        for (int b = 0; b < 8; ++b) {
            buffers_hash ^= (o >> (b * 8)) & 0xff;
            buffers_hash *= 0x100000001b3ull;
        }
    }

    // M17.5 + M-cpp-new-6: cache lookup keyed on (layout, buffer-list+offsets).
    VkDescriptorSet desc_set = VK_NULL_HANDLE;
    if (kUseDescCache) {
        DeviceRuntime::DescSetCacheKey key{
            pipeline->descriptor_set_layout(), buffers_hash};
        {
            std::lock_guard<std::mutex> _lk(rt.desc_set_mutex_);
            auto cache_it = rt.desc_set_cache.find(key);
            if (cache_it != rt.desc_set_cache.end()) {
                desc_set = cache_it->second;
            } else {
                // M-cpp-new-6 Layer 2: snapshot reset generation before
                // allocate() in case pool exhaustion triggers an internal
                // reset. If the generation changed, the cache holds stale
                // VkDescriptorSet handles from the pre-reset pool — clear.
                uint64_t gen_before = rt.desc_pool->reset_generation();
                desc_set = rt.desc_pool->allocate(
                    pipeline->descriptor_set_layout());
                if (rt.desc_pool->reset_generation() != gen_before) {
                    rt.desc_set_cache.clear();
                }
                rt.desc_set_cache[key] = desc_set;
            }
        }
    } else {
        // Legacy path: allocate fresh; do not cache. Pool reset on
        // batch end frees these in bulk via `vkResetDescriptorPool`.
        desc_set = rt.desc_pool->allocate(pipeline->descriptor_set_layout());
    }

    if (g_profile_enabled) { t3 = _now_ns(); g_profile_desc_alloc_ns += (t3 - t2); }

    vulkan::bind_buffers(device, desc_set, vk_buffers_arr, vk_sizes_arr, vk_offsets_arr, n);

    if (g_profile_enabled) { t5 = _now_ns(); g_profile_desc_write_ns += (t5 - t4); }

    // Record into the deferred command buffer (no submit yet)
    auto& cmd = rt.stream->deferred_cmd();
    cmd.bind_pipeline(pipeline->pipeline());
    cmd.bind_descriptor_set(pipeline->layout(), desc_set);

    if (push_constants && push_constants_size > 0) {
        cmd.push_constants(pipeline->layout(), push_constants_size, push_constants);
    }

    // HOST→COMPUTE barrier: if any input buffer was written by the CPU host
    // (via VulkanBuffer::write / vkMapMemory+memcpy), emit a HOST→COMPUTE
    // pipeline barrier before the dispatch. This makes host writes visible
    // to the GPU compute stage (Vulkan spec §7.1.2 requires this even on
    // HOST_COHERENT memory — coherency only waives vkFlushMappedMemoryRanges,
    // not the pipeline barrier for visibility into the GPU domain).
    bool needs_host_barrier = false;
    for (uint32_t i = 0; i < n && !needs_host_barrier; ++i) {
        if (rt.host_written_buffers.count(vk_buffers_arr[i])) {
            needs_host_barrier = true;
        }
    }
    if (needs_host_barrier) {
        cmd.host_to_compute_barrier();
        rt.host_written_buffers.clear();
        g_barrier_count++;
    }

    // Smart barrier: emit when this dispatch has a hazard against a prior one.
    //   RAW / WAW — any of this dispatch's buffers was *written* before
    //               (``dirty_buffers``); covers reads and writes of dirty bufs.
    //   WAR (S2.0d) — any *output* buffer of this dispatch was *read* before
    //               (``read_buffers``). Inductor exact-reuse aliasing turns a
    //               just-read buffer into a new output, so without this the
    //               write races the prior read (gradient corruption in stacked
    //               conv+GN backward). The barrier uses a combined access mask
    //               so it serialises against prior reads *and* writes.
    uint32_t first_output = (num_outputs < n) ? (n - num_outputs) : 0;
    bool needs_barrier = false;
    for (uint32_t i = 0; i < n && !needs_barrier; ++i) {
        if (rt.dirty_buffers.count(vk_buffers_arr[i])) {
            needs_barrier = true;
        }
    }
    for (uint32_t i = first_output; i < n && !needs_barrier; ++i) {
        if (rt.read_buffers.count(vk_buffers_arr[i])) {
            needs_barrier = true;  // WAR
        }
    }
    if (needs_barrier) {
        cmd.memory_barrier(
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
        rt.dirty_buffers.clear();
        rt.read_buffers.clear();
        g_barrier_count++;
    } else if (!rt.dirty_buffers.empty() || !rt.read_buffers.empty()) {
        g_barrier_skip_count++;
    }

    if (g_profile_enabled) { t6 = _now_ns(); g_profile_barrier_check_ns += (t6 - t5); }

    cmd.dispatch(num_workgroups_x, num_workgroups_y, num_workgroups_z);

    if (g_profile_enabled) { t7 = _now_ns(); g_profile_cmd_record_ns += (t7 - t6); }

    // DEBUG: trace dispatch keys
    static bool trace_enabled = (getenv("TORCH_VULKAN_TRACE_DISPATCH") != nullptr);
    if (trace_enabled) {
        fprintf(stderr, "DISPATCH[%llu] key=%s buffers=%u wg=(%u,%u,%u) outputs=%u barrier=%d\n",
                (unsigned long long)g_dispatch_count.load(), key.c_str(), n,
                num_workgroups_x, num_workgroups_y, num_workgroups_z,
                num_outputs, needs_barrier ? 1 : 0);
        fflush(stderr);
    }

    // Mark output buffers dirty; track input buffers as read (S2.0d WAR).
    for (uint32_t i = 0; i < n; ++i) {
        if (i >= first_output) {
            rt.dirty_buffers.insert(vk_buffers_arr[i]);
        } else {
            rt.read_buffers.insert(vk_buffers_arr[i]);
        }
    }

    // Track buffers for WAR hazard detection
    for (uint32_t i = 0; i < n; ++i) {
        rt.stream->track_buffer(vk_buffers_arr[i]);
    }
    rt.stream->inc_pending();
    g_dispatch_count++;

    if (g_profile_enabled) { t0 = _now_ns(); g_profile_dirty_track_ns += (t0 - t7); }
}

// ── N+1.5: dispatch with descriptor-array bindings ──────────────
void dispatch_shader_indexed(
    const std::string& key,
    const uint32_t* spirv_code,
    size_t spirv_size,
    const std::vector<at::Tensor>& tensors,
    const std::vector<uint32_t>& descriptor_counts,
    uint32_t num_workgroups_x,
    uint32_t num_workgroups_y,
    uint32_t num_workgroups_z,
    const void* push_constants,
    uint32_t push_constants_size,
    uint32_t num_outputs,
    const std::vector<SpecConstant>& spec_constants) {

    // Validate: sum(descriptor_counts) == tensors.size()
    uint32_t total = 0;
    for (uint32_t c : descriptor_counts) total += c;
    TORCH_CHECK(total == tensors.size(),
        "dispatch_shader_indexed: sum(descriptor_counts)=", total,
        " != tensors.size()=", tensors.size());

    // Fast path: all counts are 1 → identical to dispatch_shader.
    bool all_ones = true;
    for (uint32_t c : descriptor_counts) {
        if (c != 1) { all_ones = false; break; }
    }
    if (all_ones) {
        dispatch_shader(key, spirv_code, spirv_size, tensors,
                        num_workgroups_x, num_workgroups_y, num_workgroups_z,
                        push_constants, push_constants_size, num_outputs);
        return;
    }

    auto& ctx = vulkan::Context::instance();
    auto device_idx = ctx.current_device();
    VkDevice device = ctx.device(device_idx);

    TORCH_CHECK(ctx.descriptor_indexing_enabled(),
        "dispatch_shader_indexed: descriptor-array binding requested but "
        "VK_EXT_descriptor_indexing is not enabled. Set "
        "TORCH_VULKAN_DESCRIPTOR_INDEXING=1.");

    // Get or create cached pipeline (descriptor-counts variant).
    auto* pipeline = vulkan::PipelineCache::instance().get_or_create(
        device, key, spirv_code, spirv_size,
        descriptor_counts, push_constants_size, spec_constants);

    auto& rt = get_runtime(device_idx);

    if (!rt.batch_mode && rt.stream->pending_dispatches() >= vulkan::Stream::MAX_DISPATCHES_PER_CMD) {
        g_capacity_flush_count++;
        rt.stream->flush_async();
        // M-cpp-new-2: async reset (see comment in `dispatch_shader`).
        rt.desc_pool->reset_async(rt.stream->fence());
        rt.dirty_buffers.clear();
        rt.read_buffers.clear();  // S2.0d
        rt.host_written_buffers.clear();
        {
            std::lock_guard<std::mutex> _lk(rt.desc_set_mutex_);
            rt.desc_set_cache.clear();  // M17.5: clear on pool reset
        }
    }

    // M-cpp-new-5 / M-cpp-new-6 note: unlike `dispatch_shader` we know
    // descriptor indexing is on (asserted above), so UPDATE_AFTER_BIND
    // is available; the M-cpp-new-6 bug (same descriptor-set handle bound
    // by two recorded dispatches with different buffer contents) still
    // applies here in principle, so we use the same (layout, buffer-list)
    // cache key as `dispatch_shader`.
    // M-pipeline-5: per-call read (not `static const` capture-at-first-
    // use). See `dispatch_shader` above for the rationale. This path is
    // already gated by `TORCH_CHECK(ctx.descriptor_indexing_enabled())`
    // above, so `max_bindings` always evaluates to 256 today — but the
    // `static const` capture would freeze the wrong value if a future
    // fixture flipped the assert into a soft warning + fallback.
    constexpr uint32_t MAX_BINDINGS_CAP = 256;
    const uint32_t max_bindings =
        ctx.descriptor_indexing_enabled() ? 256u : 32u;
    VkBuffer vk_buffers_arr[MAX_BINDINGS_CAP];
    VkDeviceSize vk_sizes_arr[MAX_BINDINGS_CAP];
    VkDeviceSize vk_offsets_arr[MAX_BINDINGS_CAP];
    const uint32_t n = static_cast<uint32_t>(tensors.size());
    TORCH_CHECK(n <= max_bindings,
        "dispatch_shader_indexed: total buffer count ", n,
        " exceeds max_bindings ", max_bindings);

    for (uint32_t i = 0; i < n; ++i) {
        auto info = get_buffer_info(tensors[i]);
        vk_buffers_arr[i] = info.buffer;
        vk_sizes_arr[i] = info.size;
        vk_offsets_arr[i] = info.offset;
    }

    uint64_t buffers_hash = 0xcbf29ce484222325ull;
    for (uint32_t i = 0; i < n; ++i) {
        uint64_t v = reinterpret_cast<uint64_t>(vk_buffers_arr[i]);
        for (int b = 0; b < 8; ++b) {
            buffers_hash ^= (v >> (b * 8)) & 0xff;
            buffers_hash *= 0x100000001b3ull;
        }
        uint64_t o = static_cast<uint64_t>(vk_offsets_arr[i]);
        for (int b = 0; b < 8; ++b) {
            buffers_hash ^= (o >> (b * 8)) & 0xff;
            buffers_hash *= 0x100000001b3ull;
        }
    }

    // M17.5: Reuse cached descriptor set keyed on (layout, buffer-list+offsets).
    VkDescriptorSet desc_set = VK_NULL_HANDLE;
    {
        std::lock_guard<std::mutex> _lk(rt.desc_set_mutex_);
        DeviceRuntime::DescSetCacheKey key{
            pipeline->descriptor_set_layout(), buffers_hash};
        auto cache_it = rt.desc_set_cache.find(key);
        if (cache_it != rt.desc_set_cache.end()) {
            desc_set = cache_it->second;
        } else {
            desc_set = rt.desc_pool->allocate(pipeline->descriptor_set_layout());
            rt.desc_set_cache[key] = desc_set;
        }
    }

    vulkan::bind_buffers_indexed(
        device, desc_set,
        vk_buffers_arr, vk_sizes_arr, vk_offsets_arr,
        descriptor_counts.data(),
        static_cast<uint32_t>(descriptor_counts.size()));

    auto& cmd = rt.stream->deferred_cmd();
    cmd.bind_pipeline(pipeline->pipeline());
    cmd.bind_descriptor_set(pipeline->layout(), desc_set);

    if (push_constants && push_constants_size > 0) {
        cmd.push_constants(pipeline->layout(),
                           push_constants_size, push_constants);
    }

    // HOST→COMPUTE barrier for host-written buffers (same logic as dispatch_shader).
    bool needs_host_barrier_indexed = false;
    for (uint32_t i = 0; i < n && !needs_host_barrier_indexed; ++i) {
        if (rt.host_written_buffers.count(vk_buffers_arr[i])) {
            needs_host_barrier_indexed = true;
        }
    }
    if (needs_host_barrier_indexed) {
        cmd.host_to_compute_barrier();
        rt.host_written_buffers.clear();
        g_barrier_count++;
    }

    // S2.0d: RAW/WAW (dirty_buffers) + WAR (output overlaps a prior read).
    uint32_t first_output = (num_outputs < n) ? (n - num_outputs) : 0;
    bool needs_barrier = false;
    for (uint32_t i = 0; i < n && !needs_barrier; ++i) {
        if (rt.dirty_buffers.count(vk_buffers_arr[i])) {
            needs_barrier = true;
        }
    }
    for (uint32_t i = first_output; i < n && !needs_barrier; ++i) {
        if (rt.read_buffers.count(vk_buffers_arr[i])) {
            needs_barrier = true;  // WAR
        }
    }
    if (needs_barrier) {
        cmd.memory_barrier(
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
        rt.dirty_buffers.clear();
        rt.read_buffers.clear();
        g_barrier_count++;
    } else if (!rt.dirty_buffers.empty() || !rt.read_buffers.empty()) {
        g_barrier_skip_count++;
    }

    cmd.dispatch(num_workgroups_x, num_workgroups_y, num_workgroups_z);

    // Mark output buffers dirty; track inputs as read (S2.0d WAR).
    for (uint32_t i = 0; i < n; ++i) {
        if (i >= first_output) {
            rt.dirty_buffers.insert(vk_buffers_arr[i]);
        } else {
            rt.read_buffers.insert(vk_buffers_arr[i]);
        }
    }
    for (uint32_t i = 0; i < n; ++i) {
        rt.stream->track_buffer(vk_buffers_arr[i]);
    }
    rt.stream->inc_pending();
    g_dispatch_count++;
}

// ── Host-write notification ─────────────────────────────────────
// Called from Registration.cpp / vulkan_copy_() after VulkanBuffer::write().
// Records that the GPU buffer was just filled from the CPU so that the next
// dispatch_shader reads it with a HOST→COMPUTE barrier in place.
void notify_host_write(VkBuffer buf) {
    auto& ctx = vulkan::Context::instance();
    auto& rt = get_runtime(ctx.current_device());
    rt.host_written_buffers.insert(buf);
}

// ── Flush all pending GPU work ──────────────────────────────────
void flush_stream() {
    std::lock_guard<std::recursive_mutex> lock(g_runtime_mutex);
    for (auto& [idx, rt] : g_runtimes) {
        if (rt.stream->pending_dispatches() > 0) {
            g_flush_count++;
            rt.stream->flush_sync();
            rt.desc_pool->reset();
        } else if (rt.stream->has_in_flight_work()) {
            // M9.2: work was already submitted async (flush_async) but not
            // yet completed. Must wait for it before CPU readback.
            g_flush_count++;
            rt.stream->synchronize();
            rt.desc_pool->reset();
        }
        rt.dirty_buffers.clear();
        rt.read_buffers.clear();  // S2.0d
        rt.host_written_buffers.clear();
        {
            std::lock_guard<std::mutex> _lk(rt.desc_set_mutex_);
            rt.desc_set_cache.clear();  // M17.5: clear on pool reset (flush)
        }
    }
    // Drain quarantined buffers back into the reuse pool now that
    // the command buffer they were referenced by has completed.
    VulkanAllocator::instance().drain_pending_recycle();
}

// Flush only if there are actually pending dispatches
void flush_if_pending() {
    // Quick check without lock — avoid lock overhead when no work pending
    bool any_pending = false;
    for (auto& [idx, rt] : g_runtimes) {
        if (rt.stream->pending_dispatches() > 0 ||
            rt.stream->has_in_flight_work()) {
            any_pending = true;
            break;
        }
    }
    if (any_pending) flush_stream();
}

// Check if a VkBuffer is used in the current deferred batch
bool is_buffer_in_flight(VkBuffer buf) {
    std::lock_guard<std::recursive_mutex> lock(g_runtime_mutex);
    for (auto& [idx, rt] : g_runtimes) {
        if (rt.stream->is_buffer_pending(buf)) return true;
    }
    return false;
}

// Static callback for VulkanBuffer::read()
static void flush_stream_callback() {
    g_preread_flush_count++;
    flush_stream();
}

// Static callback to check if a buffer is pending in any stream
static bool is_buffer_in_flight_callback(VkBuffer buf) {
    return is_buffer_in_flight(buf);
}

// Static callback for descriptor pool reset
static void desc_pool_flush_callback() {
    g_descpool_flush_count++;
    flush_stream();
}

// ── Byte-precision buffer→buffer copy (OP.1.c) ──────────────────
// Records `vkCmdCopyBuffer` into the deferred command buffer for sub-32-bit
// dtypes (Bool, Byte, Char) where the float-shader copy path silently zeroes
// trailing bytes within each 4-byte slot. Uses the same dirty-buffer / WAR
// tracking as `dispatch_shader` so it composes with the deferred batch.
static void dispatch_copy_buffer_byte(const at::Tensor& src, const at::Tensor& dst) {
    auto src_info = get_buffer_info(src);
    auto dst_info = get_buffer_info(dst);
    VkDeviceSize copy_bytes = static_cast<VkDeviceSize>(dst.nbytes());
    TORCH_CHECK(copy_bytes <= src_info.size && copy_bytes <= dst_info.size,
                "dispatch_copy_buffer_byte: tensor nbytes exceeds buffer capacity");
    if (copy_bytes == 0) return;

    auto& rt = get_runtime(vulkan::Context::instance().current_device());

    // Match dispatch_shader's flush-on-near-full-pool behavior so byte copies
    // recorded after many shader dispatches don't blow the descriptor pool budget.
    if (rt.stream->pending_dispatches() >= vulkan::Stream::MAX_DISPATCHES_PER_CMD) {
        g_capacity_flush_count++;
        rt.stream->flush_async();
        // M-cpp-new-2: async reset (see comment in `dispatch_shader`).
        rt.desc_pool->reset_async(rt.stream->fence());
        rt.dirty_buffers.clear();
        rt.read_buffers.clear();  // S2.0d
        rt.host_written_buffers.clear();
        {
            std::lock_guard<std::mutex> _lk(rt.desc_set_mutex_);
            rt.desc_set_cache.clear();  // M17.5: clear on pool reset
        }
    }

    auto& cmd = rt.stream->deferred_cmd();
    VkCommandBuffer raw_cmd = cmd.handle();

    // HOST→TRANSFER barrier: if src or dst was host-written, make host writes
    // visible to the transfer stage (vkCmdCopyBuffer is TRANSFER, not COMPUTE).
    bool needs_host_barrier_copy =
        rt.host_written_buffers.count(src_info.buffer) ||
        rt.host_written_buffers.count(dst_info.buffer);
    if (needs_host_barrier_copy) {
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(raw_cmd,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, &mb, 0, nullptr, 0, nullptr);
        rt.host_written_buffers.clear();
        g_barrier_count++;
    }

    // Pre-barrier: if src or dst was written by a prior compute dispatch,
    // synchronize the prior shader-write with this copy's transfer-read/write.
    bool needs_barrier =
        rt.dirty_buffers.count(src_info.buffer) ||
        rt.dirty_buffers.count(dst_info.buffer);
    if (needs_barrier) {
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(raw_cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, &mb, 0, nullptr, 0, nullptr);
        rt.dirty_buffers.clear();
        g_barrier_count++;
    } else if (!rt.dirty_buffers.empty()) {
        g_barrier_skip_count++;
    }

    VkBufferCopy region{};
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size = copy_bytes;
    vkCmdCopyBuffer(raw_cmd, src_info.buffer, dst_info.buffer, 1, &region);

    // Post-barrier: transfer-write → subsequent shader-read so a shader
    // dispatch that consumes `dst` next sees the byte copy.
    {
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(raw_cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    // Mark dst dirty + track buffers for WAR detection (same as dispatch_shader).
    rt.dirty_buffers.insert(dst_info.buffer);
    rt.stream->track_buffer(src_info.buffer);
    rt.stream->track_buffer(dst_info.buffer);
    rt.stream->inc_pending();
    g_dispatch_count++;
}

// ── Dtype-aware buffer copy ─────────────────────────────────────
//
// M-cpp-new-4: the `copy_buffer_copy_fwd` shader now operates on 32-bit
// `uint` words (see `shaders/copy/buffer_copy.slang`). The copy length
// passed to the shader is `numel * elementSize(dtype) / 4`, which is the
// number of 4-byte words covering the contiguous tensor storage. This
// preserves all bits for any dtype whose element size is a multiple of
// 4 — most importantly `int64` and `float64`, which the prior
// `StructuredBuffer<float>` path silently truncated by half (the high
// 32 bits of every element were dropped on every `.to('vulkan')` /
// `.contiguous()` round-trip).
//
// Sub-32-bit dtypes (Bool, Byte, Char, Float8_*) cannot use the word-copy
// shader because their nbytes() is not guaranteed to be 4-aligned for
// arbitrary numel; they continue to route through `dispatch_copy_buffer_byte`
// (OP.1.c — `vkCmdCopyBuffer` with byte precision). Half / BFloat16
// (2 B/elem) packed pairs ride the word-copy path when numel is even and
// fall through to the byte-copy path otherwise so the trailing 2 bytes
// don't read past the end of storage.
void dispatch_copy_buffer(const at::Tensor& src, const at::Tensor& dst) {
    uint32_t numel = static_cast<uint32_t>(dst.numel());
    if (numel == 0) return;

    auto dtype = dst.scalar_type();

    // OP.1.c — sub-32-bit dtypes go through byte-precision `vkCmdCopyBuffer`.
    // The float-shader path packs 4 source bytes into one uint and writes one
    // uint per "element" — for `[T,F,T,F]` (4 bytes = `0x00010001` LE) the kernel
    // reads/writes one float (= 65537.0 when reinterpreted), zeroing the trailing
    // 3 elements. byte-precision copy preserves the 1-byte-per-element layout
    // and is also robust to numel that isn't a multiple of 4.
    if (dtype == c10::ScalarType::Bool ||
        dtype == c10::ScalarType::Byte ||
        dtype == c10::ScalarType::Char ||
        dtype == c10::ScalarType::Float8_e4m3fn ||
        dtype == c10::ScalarType::Float8_e5m2) {
        dispatch_copy_buffer_byte(src, dst);
        return;
    }

    // 2 B/elem dtypes (Half, BFloat16) only fit a whole-word copy when
    // numel is even. An odd numel leaves a trailing 2-byte tail; rather
    // than risk reading past storage, defer to the byte-precision copy.
    const auto elem_size =
        static_cast<uint32_t>(c10::elementSize(dtype));
    const uint64_t nbytes =
        static_cast<uint64_t>(numel) * static_cast<uint64_t>(elem_size);
    if (nbytes % 4 != 0) {
        dispatch_copy_buffer_byte(src, dst);
        return;
    }

    // The shader copies 32-bit `uint` words; one word covers 4 bytes of
    // storage regardless of dtype. `numel * elem_size / 4` is the exact
    // word count, so int64 / float64 are copied in full (the M-cpp-new-4
    // bug was passing `numel` here against a 4-B/elem shader).
    TORCH_CHECK(elem_size >= 1, "dispatch_copy_buffer: dtype must have positive element size");
    const uint64_t copy_units_u64 = nbytes / 4ull;
    TORCH_CHECK(
        copy_units_u64 <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
        "dispatch_copy_buffer: tensor too large for 32-bit dispatch (",
        copy_units_u64,
        " uint words)");
    const uint32_t copy_units = static_cast<uint32_t>(copy_units_u64);

    dispatch_elementwise("copy_buffer_copy_fwd",
                         shaders::copy_buffer_copy_fwd,
                         shaders::copy_buffer_copy_fwd_size,
                         {src, dst}, copy_units);
}

// ── Strided buffer copy ──────────────────────────────────────────
void dispatch_strided_copy(const at::Tensor& src, const at::Tensor& dst) {
    uint32_t numel = static_cast<uint32_t>(dst.numel());
    if (numel == 0) return;

    auto ndim = static_cast<uint32_t>(src.dim());
    TORCH_CHECK(ndim <= 5, "strided_copy: max 5 dimensions");

    // Build push constants with sizes, strides, and storage offset.
    // storage_offset_src handles Inductor _reinterpret_tensor views whose
    // data_ptr() is offset from the registered VkBuffer base.
    // Strides are in elements (float-sized), not bytes.
    struct {
        uint32_t numel;
        uint32_t ndim;
        uint32_t sizes0, sizes1, sizes2, sizes3, sizes4;
        uint32_t strides0, strides1, strides2, strides3, strides4;
        uint32_t storage_offset_src;
    } params{};

    params.numel = numel;
    params.ndim = ndim;
    // Always pass storage_offset via push constant. kBaseOffsets below forces the
    // descriptor to bind at offset=0, so there is no double-counting: the shader
    // reads src[storage_offset_src + computed_index] from the buffer base.
    params.storage_offset_src = static_cast<uint32_t>(src.storage_offset());

    auto sizes = src.sizes();
    auto strides = src.strides();

    uint32_t s[5] = {1, 1, 1, 1, 1};
    uint32_t st[5] = {0, 0, 0, 0, 0};
    for (uint32_t d = 0; d < ndim; d++) {
        s[d] = static_cast<uint32_t>(sizes[d]);
        st[d] = static_cast<uint32_t>(strides[d]);
    }
    params.sizes0 = s[0]; params.sizes1 = s[1]; params.sizes2 = s[2];
    params.sizes3 = s[3]; params.sizes4 = s[4];
    params.strides0 = st[0]; params.strides1 = st[1]; params.strides2 = st[2];
    params.strides3 = st[3]; params.strides4 = st[4];

    uint32_t workgroups = (numel + 255) / 256;
    // Force both buffers to bind at descriptor offset=0 so storage_offset_src
    // push constant is not double-counted for aligned src offsets.
    static const std::vector<VkDeviceSize> kBaseOffsets = {0, 0};
    dispatch_shader("copy_strided_copy_fwd",
                    shaders::copy_strided_copy_fwd,
                    shaders::copy_strided_copy_fwd_size,
                    {src, dst},
                    workgroups, 1, 1,
                    &params, sizeof(params),
                    1, {}, &kBaseOffsets);
}

}} // namespace torch_vulkan::ops
