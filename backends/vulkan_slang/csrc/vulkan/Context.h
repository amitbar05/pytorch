#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace vulkan {

struct DeviceCapabilities {
    bool float16 = false;
    bool int8 = false;
    // M18.4-followup-C: shaderInt16 (from VkPhysicalDeviceFeatures.shaderInt16)
    // gates 16-bit arithmetic in shaders. Required to declare
    // ``RWStructuredBuffer<int16_t>`` / ``<uint16_t>`` element types so the
    // Slang side's element size matches PyTorch's 2 B/elem allocation
    // for int16/uint16 buffers.
    bool int16 = false;
    bool int64 = false;
    bool float64 = false;
    uint32_t subgroup_size = 0;
    bool cooperative_matrix = false;
    uint32_t max_workgroup_size = 0;
    uint32_t max_compute_shared_memory = 0;
    std::string device_name;
    uint32_t device_type = 0; // VkPhysicalDeviceType
    // VK_EXT_descriptor_indexing / Vulkan 1.2 descriptor indexing
    bool descriptor_indexing = false;
    uint32_t max_per_stage_storage_buffers = 16; // default pre-descriptor-indexing limit
    // M18.4-followup-C: 8-bit + 16-bit storage buffer access. Required to
    // declare narrow-int element types on SSBOs without device-validation
    // failure. ``storageBuffer{8,16}BitAccess`` is the SSBO-side bit;
    // ``uniformAndStorageBuffer{8,16}BitAccess`` covers UBO+SSBO. The
    // pointwise codegen needs SSBO access; we track both for future UBO
    // use (e.g. push-constant arrays of narrow ints).
    bool storage_buffer_8bit = false;
    bool uniform_and_storage_buffer_8bit = false;
    bool storage_buffer_16bit = false;
    bool uniform_and_storage_buffer_16bit = false;
    // 2026-05-20: VK_KHR_maintenance4 enables SPIR-V LocalSizeId execution
    // mode (slangc 2026.5.2 emits it for spec-constant numthreads kernels).
    // Without it, kernels using SPIR-V 1.5+ LocalSizeId fail validation:
    // "LocalSizeId is used but maintenance4 feature was not enabled".
    bool maintenance4 = false;
};

class Context {
public:
    static Context& instance();

    // No copy/move
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    // Device management
    uint32_t device_count() const;
    void set_device(uint32_t index);
    uint32_t current_device() const;

    // Vulkan handles for current device
    VkInstance vk_instance() const { return instance_; }
    VkPhysicalDevice physical_device(uint32_t index = UINT32_MAX) const;
    VkDevice device(uint32_t index = UINT32_MAX) const;
    VkQueue compute_queue(uint32_t index = UINT32_MAX) const;
    uint32_t compute_queue_family(uint32_t index = UINT32_MAX) const;
    VmaAllocator allocator(uint32_t index = UINT32_MAX) const;

    const DeviceCapabilities& capabilities(uint32_t index = UINT32_MAX) const;
    std::string device_name(uint32_t index = UINT32_MAX) const;

    bool is_available() const { return !devices_.empty(); }

    // Descriptor indexing (UPDATE_AFTER_BIND) support for current device.
    // Gated by VK_EXT_descriptor_indexing + env var TORCH_VULKAN_DESCRIPTOR_INDEXING.
    //
    // M-cpp-new-5-followup-test: this method also consults the
    // process-global runtime override
    // (``g_desc_indexing_override``) set via
    // ``set_descriptor_indexing_override(int)``. Override values:
    //   -1 = use the capability flag (default; same as pre-fix)
    //    0 = force "off" (exercises the non-cached fallback path
    //        for stress testing)
    //    1 = force "on"  (asserts the cached path even on drivers
    //        that the capability flag claims don't support it)
    //
    // The override is the only way to flip descriptor indexing
    // mid-process — the env var ``TORCH_VULKAN_DESCRIPTOR_INDEXING``
    // is read once at Context init and then captured in the
    // device-capability struct. Use the override exclusively for
    // tests; never in production code.
    bool descriptor_indexing_enabled(uint32_t index = UINT32_MAX) const;

    // M-cpp-new-5-followup-test: runtime override for the
    // descriptor-indexing capability check. See
    // ``descriptor_indexing_enabled`` for the contract. Reading
    // returns the current override state; writing flips it
    // atomically (safe to call from any thread).
    static void set_descriptor_indexing_override(int value);
    static int get_descriptor_indexing_override();

    // Release all Vulkan resources held by other singletons before
    // destroying VkDevice. Called automatically from the destructor.
    void shutdown();

private:
    Context();
    ~Context();

    void init_instance();
    void init_devices();
    void init_device(uint32_t index);

    static VkBool32 VKAPI_CALL debug_callback(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT type,
        const VkDebugUtilsMessengerCallbackDataEXT* data,
        void* user_data);

    bool shutdown_done_ = false;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;

    struct DeviceState {
        VkPhysicalDevice physical = VK_NULL_HANDLE;
        VkDevice logical = VK_NULL_HANDLE;
        VkQueue compute_queue = VK_NULL_HANDLE;
        uint32_t compute_queue_family = 0;
        VmaAllocator allocator = VK_NULL_HANDLE;
        DeviceCapabilities caps;
    };

    std::vector<DeviceState> devices_;
    uint32_t current_device_ = 0;
    mutable std::mutex mutex_;
};

} // namespace vulkan
