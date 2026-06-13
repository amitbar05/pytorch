// AOTI shim layer for the Vulkan device backend.
//
// Generated kernels emit calls like `aoti_torch_empty_strided_vulkan(...)`
// with unmangled C-linkage.  These definitions must live at global scope
// with `extern "C"` so the dynamic linker can resolve them.
//
// V13 (2026-06-04) fixes applied here:
//   1. `aoti_torch_empty_strided_vulkan` signature corrected from 7-arg
//      (with size_len/stride_len) to 6-arg (ndim, sizes, strides, dtype,
//      device_idx, &out_handle) to match the wrapper emitted by
//      inductor/aoti_wrapper.cpp.
//   2. `aoti_torch_vulkan_mm_out` overrides the inline stub from
//      vulkan.h that throws "Vulkan AOTI stub not implemented" — it now
//      calls the CPU matmul fallback so the wrapper can execute.
//
// Allocation path: we go through `at::empty_strided(...)` with an explicit
// kPrivateUse1 + device_idx so the tensor lands on the correct Vulkan device
// without relying on Context::current_device() (which races under
// multi-GPU / multi-threaded compilation).

#include <ATen/core/Tensor.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/full.h>
#include <ATen/ops/mm.h>

#include <cstdint>
#include <cstdlib>

// Forward declaration for AOTI C shim types.
struct AtenTensorOpaque;
using AtenTensorHandle = AtenTensorOpaque*;

// AOTI CPU matmul fallback stub — used by generated wrappers when
// Vulkan-specific mm_out is not implemented yet.
// NOTE: must NOT use extern "C" — the wrapper is C++ and looks up the
// mangled symbol name (e.g. _Z21aoti_torch_cpu_mm_outP16AtenTensorOpaqueS0_S0_).
int aoti_torch_cpu_mm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2) {
  try {
    auto out_t = reinterpret_cast<at::Tensor*>(out);
    auto self_cpu = reinterpret_cast<at::Tensor*>(self)->cpu();
    auto mat2_cpu = reinterpret_cast<at::Tensor*>(mat2)->cpu();
    auto result = at::mm(self_cpu, mat2_cpu);
    *out_t = result.to(out_t->device());
    return 0;
  } catch (...) {
    return 1;
  }
}

// Override the inline stub from vulkan.h — we implement it by calling
// the CPU fallback above instead of throwing.  This is called from
// the AOTI wrapper's generated C++ code for the matmul node.
int aoti_torch_vulkan_mm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2) {
  return aoti_torch_cpu_mm_out(out, self, mat2);
}

extern "C" {

// The AOTI C++ wrapper calls this with the upstream-style 6-arg signature:
//   (ndim, sizes, strides, dtype, device_idx, &out_handle)
// Do NOT add size_len/stride_len — the wrapper already passes ndim as
// the first argument and the sizes/strides pointers as the second/third.
int aoti_torch_empty_strided_vulkan(
    int64_t ndim,
    const int64_t* sizes,
    const int64_t* strides,
    int32_t dtype,
    int32_t device_idx,
    void** out_handle) {
  try {
    auto size = at::IntArrayRef(sizes, static_cast<int64_t>(ndim));
    auto stride = at::IntArrayRef(strides, static_cast<int64_t>(ndim));
    auto options = at::TensorOptions()
        .device(at::kPrivateUse1, device_idx)
        .dtype(static_cast<at::ScalarType>(dtype));
    auto tensor = at::empty_strided(size, stride, options);

    if (out_handle) *out_handle = tensor.unsafeGetTensorImpl();
    return 0;
  } catch (...) {
    if (out_handle) *out_handle = nullptr;
    return 1;
  }
}

int aoti_torch_as_strided_vulkan(
    void* self_handle,
    int64_t* size_ptr, int64_t size_len,
    int64_t* stride_ptr, int64_t stride_len,
    int64_t storage_offset,
    void** out_handle) {
  try {
    auto* self = reinterpret_cast<at::Tensor*>(self_handle);
    auto size = at::IntArrayRef(size_ptr, static_cast<int64_t>(size_len));
    auto stride = at::IntArrayRef(stride_ptr, static_cast<int64_t>(stride_len));

    auto result = self->as_strided(size, stride, storage_offset);

    if (out_handle) *out_handle = result.unsafeGetTensorImpl();
    return 0;
  } catch (...) {
    if (out_handle) *out_handle = nullptr;
    return 1;
  }
}

int aoti_torch_delete(void* handle) {
  (void)handle;
  return 0;
}

int aoti_torch_zeros_vulkan(
    int64_t* size_ptr, int64_t size_len,
    int64_t aoti_dtype,
    void** out_handle) {
  try {
    auto size = at::IntArrayRef(size_ptr, static_cast<int64_t>(size_len));
    auto dtype = static_cast<at::ScalarType>(aoti_dtype);

    auto options = at::TensorOptions()
        .device(at::kPrivateUse1, 0)
        .dtype(dtype);
    auto tensor = at::zeros(size, options);

    if (out_handle) *out_handle = tensor.unsafeGetTensorImpl();
    return 0;
  } catch (...) {
    if (out_handle) *out_handle = nullptr;
    return 1;
  }
}

int aoti_torch_ones_vulkan(
    int64_t* size_ptr, int64_t size_len,
    int64_t aoti_dtype,
    void** out_handle) {
  try {
    auto size = at::IntArrayRef(size_ptr, static_cast<int64_t>(size_len));
    auto dtype = static_cast<at::ScalarType>(aoti_dtype);

    auto options = at::TensorOptions()
        .device(at::kPrivateUse1, 0)
        .dtype(dtype);
    auto tensor = at::ones(size, options);

    if (out_handle) *out_handle = tensor.unsafeGetTensorImpl();
    return 0;
  } catch (...) {
    if (out_handle) *out_handle = nullptr;
    return 1;
  }
}

int aoti_torch_full_vulkan(
    int64_t* size_ptr, int64_t size_len,
    double fill_value,
    int64_t aoti_dtype,
    void** out_handle) {
  try {
    auto size = at::IntArrayRef(size_ptr, static_cast<int64_t>(size_len));
    auto dtype = static_cast<at::ScalarType>(aoti_dtype);

    auto options = at::TensorOptions()
        .device(at::kPrivateUse1, 0)
        .dtype(dtype);
    auto tensor = at::full(size, fill_value, options);

    if (out_handle) *out_handle = tensor.unsafeGetTensorImpl();
    return 0;
  } catch (...) {
    if (out_handle) *out_handle = nullptr;
    return 1;
  }
}

}  // extern "C"
