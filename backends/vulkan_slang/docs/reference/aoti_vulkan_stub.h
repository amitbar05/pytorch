// Reference copy of the torch AOTI vulkan.h that must be deployed to:
//   <venv>/lib/python3.12/site-packages/torch/include/torch/csrc/inductor/aoti_include/vulkan.h
//
// This file provides inline stub implementations for all aoti_torch_vulkan_*
// functions that Inductor's C++ wrapper emits.  These stubs allow the generated
// .so to compile and link without undefined symbols.
//
// The allocation stub (aoti_torch_empty_strided_vulkan) delegates to the
// standard CPU allocator, which routes through our Vulkan allocator via
// PrivateUse1 hooks.
//
// All other stubs throw std::runtime_error at runtime until proper Vulkan
// dispatching implementations are added to AotiRuntime.cpp and linked via
// config.aot_inductor.custom_op_libs.
//
// To deploy: copy this file to the torch include path.
// Last updated: 2026-06-02 — AOTI.4 milestone.

#pragma once

#include <torch/csrc/inductor/aoti_include/common.h>
#include <stdexcept>
#include <string>

#define AOTI_VULKAN_STUB(name)                                                \
  static inline AOTITorchError name {                                         \
    throw std::runtime_error(                                                 \
        std::string("Vulkan AOTI stub not implemented: ") + std::string(#name) \
        + "().  Rebuild _C_ext with AOTI runtime support, or link against "   \
        "libtorch_vulkan_aoti.so.");                                          \
    return AOTI_RUNTIME_FAILURE;                                              \
  }

// Allocation — must work (called for every output buffer).
static inline AOTITorchError aoti_torch_empty_strided_vulkan(
    int64_t ndim, const int64_t* sizes, const int64_t* strides,
    int32_t dtype, int32_t device_idx, AtenTensorHandle* out)
{
    return aoti_torch_empty_strided(ndim, sizes, strides, dtype, device_idx,
                                    device_idx, out);
}

// Linear algebra
AOTI_VULKAN_STUB(aoti_torch_vulkan_mm_out(
    AtenTensorHandle out, AtenTensorHandle self, AtenTensorHandle mat2))

// ... (remaining stubs omitted for brevity — see deployed copy for full list)
// Full file: ~200 functions covering linear algebra, convolution,
// normalization, activations, pooling, element-wise, reduction, copy,
// view, loss, optimizer, random, and utility ops.

#undef AOTI_VULKAN_STUB
