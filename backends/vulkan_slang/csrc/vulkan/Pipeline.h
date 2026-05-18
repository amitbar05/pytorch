#pragma once

#include <vulkan/vulkan.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vulkan {

class Pipeline {
public:
    // CG.M15: SpecConstant = (spec_id, value) pair for VkSpecializationInfo.
    using SpecConstant = std::pair<uint32_t, uint32_t>;

    // Legacy ctor: every binding has descriptorCount=1 (one buffer per slot).
    Pipeline(VkDevice device,
             const uint32_t* spirv_code,
             size_t spirv_size,
             uint32_t num_buffers,
             uint32_t push_constant_size = 0,
             const std::vector<SpecConstant>& spec_constants = {});

    // N+1.5 ctor: per-binding descriptorCount (`descriptor_counts.size()`
    // = number of bindings; each entry = how many buffers in that slot's
    // descriptor array). Sum(descriptor_counts) = total buffers bound.
    // Requires VK_EXT_descriptor_indexing for any count > 1.
    Pipeline(VkDevice device,
             const uint32_t* spirv_code,
             size_t spirv_size,
             const std::vector<uint32_t>& descriptor_counts,
             uint32_t push_constant_size,
             const std::vector<SpecConstant>& spec_constants = {});
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    VkPipeline pipeline() const { return pipeline_; }
    VkPipelineLayout layout() const { return layout_; }
    VkDescriptorSetLayout descriptor_set_layout() const { return desc_set_layout_; }

    // Per-binding descriptorCount (size = num_bindings). For legacy
    // pipelines all entries are 1.
    const std::vector<uint32_t>& descriptor_counts() const {
        return descriptor_counts_;
    }
    // Total buffer count = sum(descriptor_counts_).
    uint32_t total_buffers() const { return total_buffers_; }

private:
    void create_pipeline_objects(const uint32_t* spirv_code,
                                 size_t spirv_size,
                                 uint32_t push_constant_size,
                                 const std::vector<SpecConstant>& spec_constants);

    VkDevice device_;
    VkShaderModule shader_module_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    std::vector<uint32_t> descriptor_counts_;
    uint32_t total_buffers_ = 0;
};

class PipelineCache {
public:
    static PipelineCache& instance();

    // Legacy: all bindings have descriptorCount=1.
    // CG.M15: spec_constants are (spec_id, value) pairs for VkSpecializationInfo.
    Pipeline* get_or_create(
        VkDevice device,
        const std::string& key,
        const uint32_t* spirv_code,
        size_t spirv_size,
        uint32_t num_buffers,
        uint32_t push_constant_size = 0,
        const std::vector<Pipeline::SpecConstant>& spec_constants = {});

    // N+1.5: per-binding descriptorCount (descriptor arrays).
    // The cache key must encode the binding shape, otherwise lookups
    // collide between flat and array-of-buffers pipelines.
    // CG.M15: spec_constants parameter added so different tile configs
    // can share one SPIR-V module.
    Pipeline* get_or_create(
        VkDevice device,
        const std::string& key,
        const uint32_t* spirv_code,
        size_t spirv_size,
        const std::vector<uint32_t>& descriptor_counts,
        uint32_t push_constant_size,
        const std::vector<Pipeline::SpecConstant>& spec_constants = {});

    void clear();

    // M-pipeline-4: cache-key collision telemetry. A non-zero value
    // means the Python-side cache key (e.g. ``config_key``) is not
    // content-aware enough — two distinct Slang sources mapped to the
    // same key. The C++ side detects this by comparing SPIR-V hashes
    // and treats the second request as a cache miss (recompiles),
    // preventing silent miscompiles. Read via ``init.cpp``'s
    // ``_pipeline_cache_collisions()`` pybind.
    uint64_t collision_count() const noexcept {
        return collision_count_.load(std::memory_order_relaxed);
    }

private:
    PipelineCache() = default;

    // M-pipeline-4: cache entry now stores the SPIR-V hash alongside
    // the compiled pipeline. On a cache hit, we compare the new
    // request's hash to the stored one — equal means a true cache
    // hit (the common case); unequal signals a Python-side key
    // collision and we treat the request as a miss + recompile.
    struct CachedPipeline {
        std::unique_ptr<Pipeline> pipeline;
        uint64_t spirv_hash;
    };

    std::mutex mutex_;
    std::unordered_map<std::string, CachedPipeline> cache_;

    // Number of (key, SPIR-V hash) mismatches observed since process
    // start. Atomically incremented under the cache lock; readable
    // from any thread.
    std::atomic<uint64_t> collision_count_{0};
};

} // namespace vulkan
