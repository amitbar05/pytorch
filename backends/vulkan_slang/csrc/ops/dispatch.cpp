#include "dispatch.h"
#include "../generated/shaders.h"

#include <atomic>
#include <chrono>
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
    auto* buf = VulkanAllocator::instance().get_buffer(tensor.data_ptr());
    TORCH_CHECK(buf && buf->is_valid(),
                "Tensor has no backing Vulkan buffer");
    return {buf->buffer(), buf->size()};
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
    const std::vector<SpecConstant>& spec_constants) {

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
    if (rt.stream->pending_dispatches() >= vulkan::Stream::MAX_DISPATCHES_PER_CMD) {
        g_capacity_flush_count++;
        rt.stream->flush_async();
        rt.desc_pool->reset();
        rt.dirty_buffers.clear();
    }

    // Allocate descriptor set and bind buffers
    VkDescriptorSet desc_set = rt.desc_pool->allocate(
        pipeline->descriptor_set_layout());

    if (g_profile_enabled) { t3 = _now_ns(); g_profile_desc_alloc_ns += (t3 - t2); }

    // Stack-allocated arrays to avoid heap allocation per dispatch.
    // Capacity grows with descriptor indexing enabled (256 vs 32).
    static const uint32_t MAX_BINDINGS =
        vulkan::Context::instance().descriptor_indexing_enabled() ? 256 : 32;
    VkBuffer vk_buffers_arr[MAX_BINDINGS];
    VkDeviceSize vk_sizes_arr[MAX_BINDINGS];
    uint32_t n = static_cast<uint32_t>(tensors.size());

    for (uint32_t i = 0; i < n; ++i) {
        auto info = get_buffer_info(tensors[i]);
        vk_buffers_arr[i] = info.buffer;
        vk_sizes_arr[i] = info.size;
    }

    if (g_profile_enabled) { t4 = _now_ns(); g_profile_buffer_info_ns += (t4 - t3); }

    vulkan::bind_buffers(device, desc_set, vk_buffers_arr, vk_sizes_arr, n);

    if (g_profile_enabled) { t5 = _now_ns(); g_profile_desc_write_ns += (t5 - t4); }

    // Record into the deferred command buffer (no submit yet)
    auto& cmd = rt.stream->deferred_cmd();
    cmd.bind_pipeline(pipeline->pipeline());
    cmd.bind_descriptor_set(pipeline->layout(), desc_set);

    if (push_constants && push_constants_size > 0) {
        cmd.push_constants(pipeline->layout(), push_constants_size, push_constants);
    }

    // Smart barrier: only emit if this dispatch reads a buffer written by a previous dispatch.
    // Check if any current tensor overlaps with the dirty set from prior dispatches.
    bool needs_barrier = false;
    for (uint32_t i = 0; i < n && !needs_barrier; ++i) {
        if (rt.dirty_buffers.count(vk_buffers_arr[i])) {
            needs_barrier = true;
        }
    }
    if (needs_barrier) {
        cmd.memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        rt.dirty_buffers.clear();
        g_barrier_count++;
    } else if (!rt.dirty_buffers.empty()) {
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

    // Mark output buffers dirty (last num_outputs tensors are outputs).
    uint32_t first_output = (num_outputs < n) ? (n - num_outputs) : 0;
    for (uint32_t i = first_output; i < n; ++i) {
        rt.dirty_buffers.insert(vk_buffers_arr[i]);
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

    if (rt.stream->pending_dispatches() >= vulkan::Stream::MAX_DISPATCHES_PER_CMD) {
        g_capacity_flush_count++;
        rt.stream->flush_async();
        rt.desc_pool->reset();
        rt.dirty_buffers.clear();
    }

    VkDescriptorSet desc_set =
        rt.desc_pool->allocate(pipeline->descriptor_set_layout());

    static const uint32_t MAX_BINDINGS =
        ctx.descriptor_indexing_enabled() ? 256 : 32;
    VkBuffer vk_buffers_arr[MAX_BINDINGS];
    VkDeviceSize vk_sizes_arr[MAX_BINDINGS];
    const uint32_t n = static_cast<uint32_t>(tensors.size());
    TORCH_CHECK(n <= MAX_BINDINGS,
        "dispatch_shader_indexed: total buffer count exceeds MAX_BINDINGS");

    for (uint32_t i = 0; i < n; ++i) {
        auto info = get_buffer_info(tensors[i]);
        vk_buffers_arr[i] = info.buffer;
        vk_sizes_arr[i] = info.size;
    }

    vulkan::bind_buffers_indexed(
        device, desc_set,
        vk_buffers_arr, vk_sizes_arr,
        descriptor_counts.data(),
        static_cast<uint32_t>(descriptor_counts.size()));

    auto& cmd = rt.stream->deferred_cmd();
    cmd.bind_pipeline(pipeline->pipeline());
    cmd.bind_descriptor_set(pipeline->layout(), desc_set);

    if (push_constants && push_constants_size > 0) {
        cmd.push_constants(pipeline->layout(),
                           push_constants_size, push_constants);
    }

    bool needs_barrier = false;
    for (uint32_t i = 0; i < n && !needs_barrier; ++i) {
        if (rt.dirty_buffers.count(vk_buffers_arr[i])) {
            needs_barrier = true;
        }
    }
    if (needs_barrier) {
        cmd.memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        rt.dirty_buffers.clear();
        g_barrier_count++;
    } else if (!rt.dirty_buffers.empty()) {
        g_barrier_skip_count++;
    }

    cmd.dispatch(num_workgroups_x, num_workgroups_y, num_workgroups_z);

    // Mark output buffers dirty (last num_outputs tensors).
    uint32_t first_output = (num_outputs < n) ? (n - num_outputs) : 0;
    for (uint32_t i = first_output; i < n; ++i) {
        rt.dirty_buffers.insert(vk_buffers_arr[i]);
    }
    for (uint32_t i = 0; i < n; ++i) {
        rt.stream->track_buffer(vk_buffers_arr[i]);
    }
    rt.stream->inc_pending();
    g_dispatch_count++;
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
        rt.desc_pool->reset();
        rt.dirty_buffers.clear();
    }

    auto& cmd = rt.stream->deferred_cmd();
    VkCommandBuffer raw_cmd = cmd.handle();

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

    // The copy shader uses StructuredBuffer<float> (4 bytes per element).
    // For smaller dtypes, adjust the copy count to avoid buffer overruns.
    uint32_t copy_units = numel;
    if (dtype == c10::ScalarType::Half || dtype == c10::ScalarType::BFloat16) {
        copy_units = (numel + 1) / 2;  // 2 bytes/element → 2 elements per float
    }

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

    // Build push constants with sizes and strides
    // Strides are in elements (float-sized), not bytes
    struct {
        uint32_t numel;
        uint32_t ndim;
        uint32_t sizes0, sizes1, sizes2, sizes3, sizes4;
        uint32_t strides0, strides1, strides2, strides3, strides4;
    } params{};

    params.numel = numel;
    params.ndim = ndim;

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
    dispatch_shader("copy_strided_copy_fwd",
                    shaders::copy_strided_copy_fwd,
                    shaders::copy_strided_copy_fwd_size,
                    {src, dst},
                    workgroups, 1, 1,
                    &params, sizeof(params));
}

}} // namespace torch_vulkan::ops
