#include "Pipeline.h"
#include "Context.h"
#include <stdexcept>

namespace vulkan {

Pipeline::Pipeline(VkDevice device,
                   const uint32_t* spirv_code,
                   size_t spirv_size,
                   uint32_t num_buffers,
                   uint32_t push_constant_size,
                   const std::vector<SpecConstant>& spec_constants)
    : device_(device),
      descriptor_counts_(num_buffers, 1u),
      total_buffers_(num_buffers) {
    create_pipeline_objects(spirv_code, spirv_size, push_constant_size,
                            spec_constants);
}

Pipeline::Pipeline(VkDevice device,
                   const uint32_t* spirv_code,
                   size_t spirv_size,
                   const std::vector<uint32_t>& descriptor_counts,
                   uint32_t push_constant_size,
                   const std::vector<SpecConstant>& spec_constants)
    : device_(device), descriptor_counts_(descriptor_counts) {
    total_buffers_ = 0;
    for (uint32_t c : descriptor_counts_) {
        total_buffers_ += c;
    }
    create_pipeline_objects(spirv_code, spirv_size, push_constant_size,
                            spec_constants);
}

void Pipeline::create_pipeline_objects(const uint32_t* spirv_code,
                                       size_t spirv_size,
                                       uint32_t push_constant_size,
                                       const std::vector<SpecConstant>& spec_constants) {
    // Create shader module
    VkShaderModuleCreateInfo sm_ci{};
    sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm_ci.codeSize = spirv_size;
    sm_ci.pCode = spirv_code;

    VkResult result = vkCreateShaderModule(device_, &sm_ci, nullptr, &shader_module_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }

    // Create descriptor set layout
    bool desc_idx = Context::instance().descriptor_indexing_enabled();
    const uint32_t num_bindings =
        static_cast<uint32_t>(descriptor_counts_.size());

    // Sanity: any descriptorCount > 1 requires descriptor indexing on this
    // backend (we set UPDATE_AFTER_BIND on every binding when desc_idx is on).
    // Without it, the runtime path that consumes array bindings is unsafe.
    bool has_array_binding = false;
    for (uint32_t c : descriptor_counts_) {
        if (c > 1) { has_array_binding = true; break; }
    }
    if (has_array_binding && !desc_idx) {
        vkDestroyShaderModule(device_, shader_module_, nullptr);
        shader_module_ = VK_NULL_HANDLE;
        throw std::runtime_error(
            "Pipeline: descriptorCount>1 binding requested but "
            "VK_EXT_descriptor_indexing is disabled "
            "(set TORCH_VULKAN_DESCRIPTOR_INDEXING=1 and ensure device support)");
    }

    std::vector<VkDescriptorSetLayoutBinding> bindings(num_bindings);
    std::vector<VkDescriptorBindingFlags> binding_flags(num_bindings, 0);

    for (uint32_t i = 0; i < num_bindings; i++) {
        bindings[i] = {};
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = descriptor_counts_[i];
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        if (desc_idx) {
            binding_flags[i] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
        }
    }

    VkDescriptorSetLayoutBindingFlagsCreateInfo flags_ci{};
    VkDescriptorSetLayoutCreateInfo dsl_ci{};
    dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    if (desc_idx) {
        flags_ci.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        flags_ci.bindingCount = num_bindings;
        flags_ci.pBindingFlags = binding_flags.data();
        dsl_ci.pNext = &flags_ci;
        // Spec (VUID-VkDescriptorSetLayoutCreateInfo-flags-03000): if any
        // binding flag has UPDATE_AFTER_BIND_BIT set, the layout flags must
        // include UPDATE_AFTER_BIND_POOL_BIT. We set the binding flag on
        // every binding when desc_idx is enabled, so the layout flag must
        // match unconditionally.
        dsl_ci.flags =
            VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    }
    dsl_ci.bindingCount = num_bindings;
    dsl_ci.pBindings = bindings.data();

    VkResult result_dsl =
        vkCreateDescriptorSetLayout(device_, &dsl_ci, nullptr, &desc_set_layout_);
    if (result_dsl != VK_SUCCESS) {
        vkDestroyShaderModule(device_, shader_module_, nullptr);
        shader_module_ = VK_NULL_HANDLE;
        throw std::runtime_error("Failed to create descriptor set layout");
    }

    // Create pipeline layout
    VkPipelineLayoutCreateInfo pl_ci{};
    pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount = 1;
    pl_ci.pSetLayouts = &desc_set_layout_;

    VkPushConstantRange pc_range{};
    if (push_constant_size > 0) {
        pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc_range.offset = 0;
        pc_range.size = push_constant_size;
        pl_ci.pushConstantRangeCount = 1;
        pl_ci.pPushConstantRanges = &pc_range;
    }

    VkResult result_pl = vkCreatePipelineLayout(device_, &pl_ci, nullptr, &layout_);
    if (result_pl != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device_, desc_set_layout_, nullptr);
        vkDestroyShaderModule(device_, shader_module_, nullptr);
        desc_set_layout_ = VK_NULL_HANDLE;
        shader_module_ = VK_NULL_HANDLE;
        throw std::runtime_error("Failed to create pipeline layout");
    }

    // Build specialization info (CG.M15: [[vk::constant_id]] overrides).
    // Spec constants let a single SPIR-V module serve multiple tile
    // configurations — the specialization happens at pipeline-creation
    // time (fast, no slangc recompilation).
    std::vector<VkSpecializationMapEntry> spec_entries;
    std::vector<uint32_t> spec_data;
    VkSpecializationInfo spec_info{};
    if (!spec_constants.empty()) {
        spec_entries.reserve(spec_constants.size());
        spec_data.reserve(spec_constants.size());
        for (const auto& sc : spec_constants) {
            VkSpecializationMapEntry entry{};
            entry.constantID = sc.first;
            entry.offset = static_cast<uint32_t>(spec_data.size() * sizeof(uint32_t));
            entry.size = sizeof(uint32_t);
            spec_entries.push_back(entry);
            spec_data.push_back(sc.second);
        }
        spec_info.mapEntryCount = static_cast<uint32_t>(spec_entries.size());
        spec_info.pMapEntries = spec_entries.data();
        spec_info.dataSize = spec_data.size() * sizeof(uint32_t);
        spec_info.pData = spec_data.data();
    }

    // Create compute pipeline
    VkComputePipelineCreateInfo cp_ci{};
    cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp_ci.stage.module = shader_module_;
    cp_ci.stage.pName = "main";
    cp_ci.stage.pSpecializationInfo = spec_constants.empty() ? nullptr : &spec_info;
    cp_ci.layout = layout_;

    result = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &pipeline_);
    if (result != VK_SUCCESS) {
        vkDestroyPipelineLayout(device_, layout_, nullptr);
        vkDestroyDescriptorSetLayout(device_, desc_set_layout_, nullptr);
        vkDestroyShaderModule(device_, shader_module_, nullptr);
        layout_ = VK_NULL_HANDLE;
        desc_set_layout_ = VK_NULL_HANDLE;
        shader_module_ = VK_NULL_HANDLE;
        throw std::runtime_error("Failed to create compute pipeline");
    }
}

Pipeline::~Pipeline() {
    if (pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, pipeline_, nullptr);
    if (layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, layout_, nullptr);
    if (desc_set_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, desc_set_layout_, nullptr);
    if (shader_module_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, shader_module_, nullptr);
}

// ── PipelineCache ────────────────────────────────────────────────
PipelineCache& PipelineCache::instance() {
    static PipelineCache cache;
    return cache;
}

Pipeline* PipelineCache::get_or_create(
    VkDevice device,
    const std::string& key,
    const uint32_t* spirv_code,
    size_t spirv_size,
    uint32_t num_buffers,
    uint32_t push_constant_size,
    const std::vector<Pipeline::SpecConstant>& spec_constants) {

    // Fast path: check without lock (safe because cache_ is never modified
    // after initial population, and pointer reads are atomic on x86/ARM)
    {
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second.get();
        }
    }

    // Slow path: acquire lock and create pipeline
    std::lock_guard<std::mutex> lock(mutex_);

    // Double-check after acquiring lock
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return it->second.get();
    }

    auto pipeline = std::make_unique<Pipeline>(
        device, spirv_code, spirv_size, num_buffers, push_constant_size,
        spec_constants);
    auto* ptr = pipeline.get();
    cache_[key] = std::move(pipeline);
    return ptr;
}

Pipeline* PipelineCache::get_or_create(
    VkDevice device,
    const std::string& key,
    const uint32_t* spirv_code,
    size_t spirv_size,
    const std::vector<uint32_t>& descriptor_counts,
    uint32_t push_constant_size,
    const std::vector<Pipeline::SpecConstant>& spec_constants) {

    {
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second.get();
        }
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return it->second.get();
    }

    auto pipeline = std::make_unique<Pipeline>(
        device, spirv_code, spirv_size, descriptor_counts, push_constant_size,
        spec_constants);
    auto* ptr = pipeline.get();
    cache_[key] = std::move(pipeline);
    return ptr;
}

void PipelineCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
}

} // namespace vulkan
