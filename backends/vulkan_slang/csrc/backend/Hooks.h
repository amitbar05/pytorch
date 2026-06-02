#pragma once

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/core/Generator.h>

namespace torch_vulkan {

struct VulkanHooksInterface : public at::PrivateUse1HooksInterface {
    bool hasPrimaryContext(c10::DeviceIndex device_index) const override;
    bool isBuilt() const override { return true; }
    bool isAvailable() const override;
    at::Generator getNewGenerator(
        c10::DeviceIndex device_index = -1) const override;
    const at::Generator& getDefaultGenerator(
        c10::DeviceIndex device_index) const override;
    // AOTI.4: getDeviceFromPtr is called when creating tensors from CPU
    // blobs (e.g. aoti_torch_create_tensor_from_blob).  Vulkan is
    // single-device, so always return device 0.
    c10::Device getDeviceFromPtr(void* /*data*/) const override {
        return c10::Device(c10::DeviceType::PrivateUse1, 0);
    }
};

} // namespace torch_vulkan
