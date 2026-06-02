// Vulkan AOTI device header — minimal shim for Inductor's C++ wrapper codegen.
// The real Vulkan allocation/dispatch functions are declared in the backend's
// AotiRuntime.h, which is included in the generated C++ wrapper by
// VulkanCppWrapperGpu.write_header().
//
// This file provides the declarations expected by the upstream Inductor's
// C++ wrapper codegen (e.g. aoti_torch_empty_strided_vulkan).
#pragma once

#include <torch/csrc/inductor/aoti_include/common.h>

// Forward-declare Vulkan allocation entry point.
// Defined in the backend's AOTI runtime (AotiRuntime.cpp), linked into
// the generated .so.
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_empty_strided_vulkan(
    int64_t ndim,
    const int64_t* sizes,
    const int64_t* strides,
    int32_t dtype,
    int32_t device_idx,
    AtenTensorHandle* out);
