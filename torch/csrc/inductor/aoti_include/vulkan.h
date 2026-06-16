#pragma once

// Vulkan backend AOTI declarations.
// This stub is replaced by the full header during install
// (pip install -e . copies from build artifacts).

#ifdef AOTI_VULKAN_FULL_HEADER
// Full header already included
#else

// ── Forward declarations ───────────────────────────────────────────
// aoti_torch_empty_strided_vulkan is defined in _C.so (aoti_shims.cpp).
// Declare with extern "C" to prevent static inline shadows from
// overriding the runtime-linked implementation.
extern "C" {
int aoti_torch_empty_strided_vulkan(
    int64_t ndim,
    const int64_t* sizes,
    const int64_t* strides,
    int32_t dtype,
    int32_t device_idx,
    void** out_handle);
}

#endif // AOTI_VULKAN_FULL_HEADER
