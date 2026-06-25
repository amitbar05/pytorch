#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

#include "Context.h"
#include "Pipeline.h"
#include "../ops/dispatch.h"
#include "../backend/Allocator.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace vulkan {

// Blocker F regression test counter — incremented inside ``debug_callback``
// whenever a validation message identifies VUID-vkResetDescriptorPool-
// descriptorPool-00313. Exposed via the static accessor and the
// ``_descriptor_pool_reset_validation_errors`` pybind so tests can
// assert that no descriptor-pool race fires under a tight dispatch loop.
static std::atomic<uint64_t> g_descriptor_pool_reset_validation_errors{0};

uint64_t Context::descriptor_pool_reset_validation_errors() {
    return g_descriptor_pool_reset_validation_errors.load(
        std::memory_order_relaxed);
}

// M-VAL.1 (v7) — generic VUID counter for the validation-driven codegen
// pillar. Incremented for every WARNING+ validation message the layer
// emits. ``reset_validation_errors_count()`` lets the pytest autouse
// fixture (gated by ``TORCH_VULKAN_VUID_AS_ERROR=1``) snapshot a
// per-test baseline so a VUID surfaced during the test fails it.
static std::atomic<uint64_t> g_validation_errors_count{0};

uint64_t Context::validation_errors_count() {
    return g_validation_errors_count.load(std::memory_order_relaxed);
}

void Context::reset_validation_errors_count() {
    g_validation_errors_count.store(0, std::memory_order_relaxed);
}

// ── Debug callback ───────────────────────────────────────────────
VkBool32 VKAPI_CALL Context::debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* /*user_data*/) {
    // Map severity bit to a short tag so the M21.3 sweep parser can
    // categorize hints. BestPractices hints arrive at INFO severity, so we
    // accept everything from INFO and above when TORCH_VULKAN_DEBUG_UTILS=1
    // is set; otherwise only WARNING+ to keep production logs quiet.
    const char* sev_tag = "UNKNOWN";
    switch (severity) {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT: sev_tag = "VERBOSE"; break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:    sev_tag = "INFO";    break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT: sev_tag = "WARNING"; break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:   sev_tag = "ERROR";   break;
        default: break;
    }
    // Type bits are flags — concatenate the active ones.
    std::string type_tag;
    if (type & VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT)     type_tag += "GENERAL|";
    if (type & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT)  type_tag += "VALIDATION|";
    if (type & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT) type_tag += "PERFORMANCE|";
    if (!type_tag.empty()) type_tag.pop_back();  // drop trailing '|'

    // Blocker F regression hook: count VUID-vkResetDescriptorPool-
    // descriptorPool-00313 hits independent of stderr verbosity. The
    // message id stays stable across validation-layer versions, so
    // tests can assert this counter == 0 even when WARNING+ logging
    // is suppressed.
    if (data && data->pMessageIdName &&
        std::strstr(data->pMessageIdName,
                    "VUID-vkResetDescriptorPool-descriptorPool-00313") !=
            nullptr) {
        g_descriptor_pool_reset_validation_errors.fetch_add(
            1, std::memory_order_relaxed);
    }

    // When TORCH_VULKAN_DEBUG_UTILS is set, surface every message
    // (including INFO-level best-practices hints) so the M21.3 sweep
    // can collect them. Otherwise keep the WARNING+ floor for production.
    static const bool debug_utils_full =
        []() {
            const char* e = getenv("TORCH_VULKAN_DEBUG_UTILS");
            return e && strcmp(e, "0") != 0;
        }();
    if (debug_utils_full ||
        severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "[Vulkan VUID] " << sev_tag << " " << type_tag << " "
                  << (data && data->pMessage ? data->pMessage : "")
                  << std::endl;
    }
    // M-VAL.1: count every WARNING+ validation/performance message so the
    // VUID-as-error pytest fixture can assert zero VUIDs per test. We
    // count VALIDATION + PERFORMANCE types only (GENERAL is loader noise
    // like "device created" — not a spec violation).
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT &&
        (type & (VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                 VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT))) {
        g_validation_errors_count.fetch_add(1, std::memory_order_relaxed);
    }
    return VK_FALSE;
}

// ── Singleton ────────────────────────────────────────────────────
Context& Context::instance() {
    static Context ctx;
    return ctx;
}

Context::Context() {
    init_instance();
    init_devices();
}

void Context::shutdown() {
    if (shutdown_done_) return;
    shutdown_done_ = true;

    // Wait for all GPU work to finish before destroying anything
    for (auto& d : devices_) {
        if (d.logical) {
            vkDeviceWaitIdle(d.logical);
        }
    }

    // Destroy dependent singletons in reverse-dependency order:
    // 1. Runtimes (Streams with fences, CommandPools, DescriptorPools)
    torch_vulkan::ops::cleanup_runtimes();
    // 2. Pipelines (VkPipeline, VkShaderModule, VkPipelineLayout, VkDescriptorSetLayout)
    PipelineCache::instance().clear();
    // 3. Allocator buffers (VkBuffer via VMA — must happen while VmaAllocator alive)
    torch_vulkan::VulkanAllocator::instance().release_all();
}

Context::~Context() {
    shutdown();

    for (auto& d : devices_) {
        if (d.allocator) {
            vmaDestroyAllocator(d.allocator);
            d.allocator = VK_NULL_HANDLE;
        }
        if (d.logical) {
            vkDestroyDevice(d.logical, nullptr);
            d.logical = VK_NULL_HANDLE;
        }
    }
    if (debug_messenger_) {
        auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (func) func(instance_, debug_messenger_, nullptr);
        debug_messenger_ = VK_NULL_HANDLE;
    }
    if (instance_) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

// ── Instance creation ────────────────────────────────────────────
void Context::init_instance() {
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "torch_vulkan";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "torch_vulkan";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    // Check for validation layer
    uint32_t layer_count = 0;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    std::vector<VkLayerProperties> layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, layers.data());

    // TORCH_VULKAN_VALIDATION: default ON. Set to 0 to disable for production.
    const char* val_env = getenv("TORCH_VULKAN_VALIDATION");
    bool validation_wanted = !val_env || strcmp(val_env, "0") != 0;

    bool has_validation = false;
    const char* validation_layer = "VK_LAYER_KHRONOS_validation";
    for (const auto& layer : layers) {
        if (strcmp(layer.layerName, validation_layer) == 0) {
            has_validation = true;
            break;
        }
    }
    has_validation = has_validation && validation_wanted;

    // Check for debug utils extension
    uint32_t ext_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> exts(ext_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, exts.data());

    bool has_debug_utils = false;
    for (const auto& ext : exts) {
        if (strcmp(ext.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
            has_debug_utils = true;
            break;
        }
    }

    std::vector<const char*> enabled_layers;
    std::vector<const char*> enabled_extensions;

    if (has_validation) {
        enabled_layers.push_back(validation_layer);
    }
    if (has_debug_utils) {
        enabled_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledLayerCount = static_cast<uint32_t>(enabled_layers.size());
    create_info.ppEnabledLayerNames = enabled_layers.data();
    create_info.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions.size());
    create_info.ppEnabledExtensionNames = enabled_extensions.data();

    VkResult result = vkCreateInstance(&create_info, nullptr, &instance_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance");
    }

    // Set up debug messenger
    // TORCH_VULKAN_DEBUG_UTILS=1 opts into the full M21.3 sweep mode:
    // INFO-level severity is enabled so BestPractices hints (which the
    // Khronos layer emits at INFO) reach userspace stderr. Production
    // and ordinary tests get the WARNING+ floor.
    const char* du_env = getenv("TORCH_VULKAN_DEBUG_UTILS");
    bool debug_utils_full = du_env && strcmp(du_env, "0") != 0;
    // S2.3: when the validation layer is active, also capture INFO-level
    // BestPractices hints so validate=True warm-up surfaces them in-process.
    if (validation_wanted) {
        debug_utils_full = true;
    }
    if (has_validation && has_debug_utils) {
        VkDebugUtilsMessengerCreateInfoEXT dbg_info{};
        dbg_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        dbg_info.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        if (debug_utils_full) {
            dbg_info.messageSeverity |=
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
        }
        dbg_info.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        dbg_info.pfnUserCallback = debug_callback;

        auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
        if (func) {
            func(instance_, &dbg_info, nullptr, &debug_messenger_);
        }
    }
}

// ── Device enumeration & creation ────────────────────────────────
void Context::init_devices() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0) return;

    std::vector<VkPhysicalDevice> physical_devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, physical_devices.data());

    devices_.resize(count);
    for (uint32_t i = 0; i < count; i++) {
        devices_[i].physical = physical_devices[i];
        init_device(i);
    }
}

void Context::init_device(uint32_t index) {
    auto& dev = devices_[index];

    // Query properties
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(dev.physical, &props);
    dev.caps.device_name = props.deviceName;
    dev.caps.device_type = props.deviceType;
    dev.caps.max_workgroup_size = props.limits.maxComputeWorkGroupInvocations;
    dev.caps.max_compute_shared_memory = props.limits.maxComputeSharedMemorySize;

    // Query features
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(dev.physical, &features);
    dev.caps.float64 = features.shaderFloat64;
    dev.caps.int64 = features.shaderInt64;
    // M18.4-followup-C: shaderInt16 lives on the base VkPhysicalDeviceFeatures
    // (not the Vulkan 1.2 aggregate). Required for declaring 16-bit
    // arithmetic / 16-bit element-typed structured buffers.
    dev.caps.int16 = features.shaderInt16;

    // Check for float16 / int8 / 8-bit / 16-bit storage support. The Vulkan
    // 1.2 aggregated features struct already covers 8-bit storage and the
    // 1.1 aggregate covers 16-bit storage. Chain both through pNext so a
    // single ``vkGetPhysicalDeviceFeatures2`` populates all of them.
    VkPhysicalDeviceDescriptorIndexingFeatures desc_idx_features{};
    desc_idx_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;

    // 2026-05-20: query VK_KHR_maintenance4 feature. Enables
    // SPIR-V LocalSizeId execution mode (slangc 2026.5.2 emits this
    // for kernels with spec-constant numthreads).
    VkPhysicalDeviceMaintenance4Features maint4_features{};
    maint4_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
    maint4_features.pNext = &desc_idx_features;

    // M18.4-followup-C: Vulkan 1.1 features expose
    // ``storageBuffer16BitAccess`` + ``uniformAndStorageBuffer16BitAccess``.
    VkPhysicalDeviceVulkan11Features vk11_features{};
    vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    vk11_features.pNext = &maint4_features;

    VkPhysicalDeviceVulkan12Features vk12_features{};
    vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk12_features.pNext = &vk11_features;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &vk12_features;
    vkGetPhysicalDeviceFeatures2(dev.physical, &features2);
    dev.caps.float16 = vk12_features.shaderFloat16;
    dev.caps.int8 = vk12_features.shaderInt8;
    // M18.4-followup-C: stash 8/16-bit storage caps for the device-create
    // step below. We enable them only when the device reports support
    // (defensive — Lavapipe / older drivers may lack one or both).
    dev.caps.storage_buffer_8bit = vk12_features.storageBuffer8BitAccess;
    dev.caps.uniform_and_storage_buffer_8bit =
        vk12_features.uniformAndStorageBuffer8BitAccess;
    dev.caps.storage_buffer_16bit = vk11_features.storageBuffer16BitAccess;
    dev.caps.uniform_and_storage_buffer_16bit =
        vk11_features.uniformAndStorageBuffer16BitAccess;
    // 2026-05-20: stash maintenance4 cap for device-create step.
    dev.caps.maintenance4 = maint4_features.maintenance4;

    // ── Descriptor indexing support ─────────────────────────
    // Gate: env var TORCH_VULKAN_DESCRIPTOR_INDEXING (default 1 on
    // real GPUs, 0 on Lavapipe / llvmpipe where update-after-bind
    // is unreliable).
    bool desc_idx_supported =
        vk12_features.descriptorIndexing &&
        desc_idx_features.descriptorBindingStorageBufferUpdateAfterBind;
    const char* env_val = getenv("TORCH_VULKAN_DESCRIPTOR_INDEXING");
    bool env_gate =
        (env_val == nullptr) ? true : (strcmp(env_val, "1") == 0);
    // Lavapipe / llvmpipe auto-detect: descriptor indexing is flaky
    // on software rasterizers; default to off unless explicitly enabled.
    bool is_lavapipe =
        (dev.caps.device_name.find("llvmpipe") != std::string::npos ||
         dev.caps.device_name.find("Lavapipe") != std::string::npos);
    if (is_lavapipe && env_val == nullptr) {
        env_gate = false; // default off on Lavapipe
    }
    dev.caps.descriptor_indexing = desc_idx_supported && env_gate;

    // Query the actual maxPerStageDescriptorStorageBuffers limit
    dev.caps.max_per_stage_storage_buffers =
        props.limits.maxPerStageDescriptorStorageBuffers;

    // Subgroup properties
    VkPhysicalDeviceSubgroupProperties subgroup_props{};
    subgroup_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroup_props;
    vkGetPhysicalDeviceProperties2(dev.physical, &props2);
    dev.caps.subgroup_size = subgroup_props.subgroupSize;

    // Find compute queue family
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev.physical, &qf_count, nullptr);
    std::vector<VkQueueFamilyProperties> qf_props(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev.physical, &qf_count, qf_props.data());

    // Prefer dedicated compute queue, fall back to any compute-capable
    uint32_t compute_family = UINT32_MAX;
    for (uint32_t i = 0; i < qf_count; i++) {
        if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_family = i;
            // Prefer queue without graphics
            if (!(qf_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                break;
            }
        }
    }
    if (compute_family == UINT32_MAX) {
        throw std::runtime_error("No compute queue family found on device " +
                                 dev.caps.device_name);
    }
    dev.compute_queue_family = compute_family;

    // Create logical device
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_ci{};
    queue_ci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_ci.queueFamilyIndex = compute_family;
    queue_ci.queueCount = 1;
    queue_ci.pQueuePriorities = &queue_priority;

    // Enable Vulkan 1.2 + 1.1 aggregated features + descriptor indexing.
    // Spec (VUID-VkDeviceCreateInfo-pNext-02830): when VkPhysicalDeviceVulkan12Features
    // is in the pNext chain, the legacy per-feature structs (including
    // VkPhysicalDeviceDescriptorIndexingFeatures and the standalone
    // 8-bit / 16-bit storage feature structs) must NOT also be present —
    // the 1.x aggregated structs already cover those bits.
    //
    // M18.4-followup-C: also enable the 8/16-bit storage bits from the
    // 1.1+1.2 aggregates, plus ``shaderInt16`` on the base
    // VkPhysicalDeviceFeatures struct. With these flipped on, the
    // generated SPIR-V is allowed to declare ``RWStructuredBuffer<T>``
    // with ``T ∈ {int8_t, uint8_t, int16_t, uint16_t}`` which matches
    // PyTorch's native 1B/2B allocation for narrow-int dtypes and
    // closes the M17.8.d.3 tail-corruption bug class for the full set.
    VkPhysicalDeviceVulkan11Features enabled_vk11{};
    enabled_vk11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    enabled_vk11.storageBuffer16BitAccess =
        dev.caps.storage_buffer_16bit ? VK_TRUE : VK_FALSE;
    enabled_vk11.uniformAndStorageBuffer16BitAccess =
        dev.caps.uniform_and_storage_buffer_16bit ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceVulkan12Features enabled_vk12{};
    enabled_vk12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    enabled_vk12.pNext = &enabled_vk11;
    enabled_vk12.shaderFloat16 = dev.caps.float16;
    enabled_vk12.shaderInt8 = dev.caps.int8;
    enabled_vk12.storageBuffer8BitAccess =
        dev.caps.storage_buffer_8bit ? VK_TRUE : VK_FALSE;
    enabled_vk12.uniformAndStorageBuffer8BitAccess =
        dev.caps.uniform_and_storage_buffer_8bit ? VK_TRUE : VK_FALSE;
    enabled_vk12.timelineSemaphore = VK_TRUE;
    enabled_vk12.bufferDeviceAddress = VK_FALSE;
    enabled_vk12.descriptorIndexing =
        dev.caps.descriptor_indexing ? VK_TRUE : VK_FALSE;
    enabled_vk12.descriptorBindingStorageBufferUpdateAfterBind =
        dev.caps.descriptor_indexing ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceFeatures enabled_features{};
    enabled_features.shaderFloat64 = dev.caps.float64;
    // shaderInt64 is required by SPIR-V kernels emitted by Inductor that use
    // 64-bit integer arithmetic for indexing (cat/view, gather, scatter).
    enabled_features.shaderInt64 = dev.caps.int64;
    // M18.4-followup-C: shaderInt16 unlocks 16-bit arithmetic in shaders.
    enabled_features.shaderInt16 = dev.caps.int16 ? VK_TRUE : VK_FALSE;

    // 2026-05-20: enable VK_KHR_maintenance4 if supported. slangc 2026.5.2
    // emits SPIR-V using OpExecutionMode LocalSizeId, which requires
    // maintenance4 OR Vulkan 1.3. Probe device-level extension support
    // and only enable when the device actually advertises it.
    VkPhysicalDeviceMaintenance4Features enabled_maint4{};
    enabled_maint4.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
    enabled_maint4.maintenance4 = dev.caps.maintenance4 ? VK_TRUE : VK_FALSE;
    if (dev.caps.maintenance4) {
        enabled_maint4.pNext = enabled_vk12.pNext;
        enabled_vk12.pNext = &enabled_maint4;
    }

    std::vector<const char*> device_extensions;
    if (dev.caps.maintenance4) {
        uint32_t dev_ext_count = 0;
        vkEnumerateDeviceExtensionProperties(
            dev.physical, nullptr, &dev_ext_count, nullptr);
        std::vector<VkExtensionProperties> dev_exts(dev_ext_count);
        vkEnumerateDeviceExtensionProperties(
            dev.physical, nullptr, &dev_ext_count, dev_exts.data());
        for (const auto& ext : dev_exts) {
            if (strcmp(ext.extensionName, "VK_KHR_maintenance4") == 0) {
                device_extensions.push_back("VK_KHR_maintenance4");
                break;
            }
        }
    }

    VkDeviceCreateInfo device_ci{};
    device_ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_ci.pNext = &enabled_vk12;
    device_ci.queueCreateInfoCount = 1;
    device_ci.pQueueCreateInfos = &queue_ci;
    device_ci.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
    device_ci.ppEnabledExtensionNames =
        device_extensions.empty() ? nullptr : device_extensions.data();
    device_ci.pEnabledFeatures = &enabled_features;

    VkResult result = vkCreateDevice(dev.physical, &device_ci, nullptr, &dev.logical);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device for " +
                                 dev.caps.device_name);
    }

    vkGetDeviceQueue(dev.logical, compute_family, 0, &dev.compute_queue);

    // Create VMA allocator
    VmaVulkanFunctions vma_funcs{};
    vma_funcs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vma_funcs.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo alloc_ci{};
    alloc_ci.vulkanApiVersion = VK_API_VERSION_1_2;
    alloc_ci.physicalDevice = dev.physical;
    alloc_ci.device = dev.logical;
    alloc_ci.instance = instance_;
    alloc_ci.pVulkanFunctions = &vma_funcs;

    result = vmaCreateAllocator(&alloc_ci, &dev.allocator);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA allocator for " +
                                 dev.caps.device_name);
    }
}

// ── Public API ───────────────────────────────────────────────────
uint32_t Context::device_count() const {
    return static_cast<uint32_t>(devices_.size());
}

void Context::set_device(uint32_t index) {
    if (index >= devices_.size()) {
        throw std::runtime_error("Device index " + std::to_string(index) +
                                 " out of range (have " +
                                 std::to_string(devices_.size()) + ")");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    current_device_ = index;
}

uint32_t Context::current_device() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_device_;
}

VkPhysicalDevice Context::physical_device(uint32_t index) const {
    if (index == UINT32_MAX) index = current_device();
    return devices_.at(index).physical;
}

VkDevice Context::device(uint32_t index) const {
    if (index == UINT32_MAX) index = current_device();
    return devices_.at(index).logical;
}

VkQueue Context::compute_queue(uint32_t index) const {
    if (index == UINT32_MAX) index = current_device();
    return devices_.at(index).compute_queue;
}

uint32_t Context::compute_queue_family(uint32_t index) const {
    if (index == UINT32_MAX) index = current_device();
    return devices_.at(index).compute_queue_family;
}

VmaAllocator Context::allocator(uint32_t index) const {
    if (index == UINT32_MAX) index = current_device();
    return devices_.at(index).allocator;
}

const DeviceCapabilities& Context::capabilities(uint32_t index) const {
    if (index == UINT32_MAX) index = current_device();
    return devices_.at(index).caps;
}

std::string Context::device_name(uint32_t index) const {
    return capabilities(index).device_name;
}

// M-cpp-new-5-followup-test: process-global runtime override.
// -1 = use the capability (default); 0 = force off; 1 = force on.
// Atomic so tests can flip it concurrently with dispatch threads
// reading it.
static std::atomic<int> g_desc_indexing_override{-1};

bool Context::descriptor_indexing_enabled(uint32_t index) const {
    // M-cpp-new-5-followup-test: consult the override BEFORE the
    // capability flag. The atomic load is uncontended in the common
    // case (override stays at -1 in production) and adds ~1 ns to
    // the hot path.
    const int override_val = g_desc_indexing_override.load(
        std::memory_order_relaxed);
    if (override_val == 0) return false;
    if (override_val == 1) return true;
    // override_val == -1 (or any other sentinel): fall through to
    // the underlying capability flag.
    return capabilities(index).descriptor_indexing;
}

void Context::set_descriptor_indexing_override(int value) {
    g_desc_indexing_override.store(value, std::memory_order_relaxed);
}

int Context::get_descriptor_indexing_override() {
    return g_desc_indexing_override.load(std::memory_order_relaxed);
}

} // namespace vulkan
