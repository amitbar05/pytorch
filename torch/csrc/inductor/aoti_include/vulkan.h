// Vulkan AOTI device header — shim stubs for Inductor's C++ wrapper.
//
// When Inductor generates C++ wrapper code for Vulkan, it emits calls like
//   aoti_torch_vulkan_mm_out(out, self, mat2)
// following the same pattern as aoti_torch_cuda_mm_out.  These stubs are
// declared here as inline functions with weak linkage — they throw clear
// errors at runtime until proper implementations are added to AotiRuntime.cpp
// and linked via config.aot_inductor.custom_op_libs.
#pragma once

#include <torch/csrc/inductor/aoti_include/common.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

// Forward-declare CPU mm (used as fallback in our Vulkan shim).
// Full declaration lives in aoti_torch/generated/c_shim_cpu.h.
// Must be extern "C" to match the C-linkage definition in c_shim_cpu.h.
extern "C" AOTITorchError aoti_torch_cpu_mm_out(
    AtenTensorHandle out, AtenTensorHandle self, AtenTensorHandle mat2);

// Forward-declare Vulkan AOTI runtime opaque handles.
struct AotiVulkanKernelHandle;
struct AotiVulkanModelHandle;

// ── Stub helper ──────────────────────────────────────────────────────
#define AOTI_VULKAN_STUB(name)                                                \
  static inline AOTITorchError name {                                         \
    throw std::runtime_error(                                                 \
        std::string("Vulkan AOTI stub not implemented: ") + std::string(#name) \
        + "().  Rebuild _C_ext with AOTI runtime support, or link against "   \
        "libtorch_vulkan_aoti.so.");                                          \
  }

// ── Allocation ──────────────────────────────────────────────────────
// aoti_torch_empty_strided_vulkan is defined in _C.so (aoti_shims.cpp).
// Declare it here so the generated wrapper sees the correct 6-arg
// signature.  Do NOT inline — the real implementation is linked at
// runtime from the Vulkan backend.
extern "C" AOTITorchError aoti_torch_empty_strided_vulkan(
    int64_t ndim,
    const int64_t* sizes,
    const int64_t* strides,
    int32_t dtype,
    int32_t device_idx,
    AtenTensorHandle* out);

// mm_out / addmm_out: bridge to CPU mm via aoti_torch_cpu_mm_out.
// aoti_torch_cpu_mm_out is provided by aoti_preload.so (LD_PRELOAD).
// The wrapper calls these as normal functions (not inline), so the
// preloaded implementation wins at link time.
extern "C" AOTITorchError aoti_torch_vulkan_mm_out(
    AtenTensorHandle out, AtenTensorHandle self, AtenTensorHandle mat2);
extern "C" AOTITorchError aoti_torch_vulkan_addmm_out(
    AtenTensorHandle out, AtenTensorHandle self,
    AtenTensorHandle mat1, AtenTensorHandle mat2,
    double beta, double alpha);


AOTI_VULKAN_STUB(aoti_torch_vulkan_bmm_out(
    AtenTensorHandle out, AtenTensorHandle self, AtenTensorHandle mat2))

AOTI_VULKAN_STUB(aoti_torch_vulkan_addbmm(
    AtenTensorHandle self, AtenTensorHandle batch1, AtenTensorHandle batch2,
    double beta, double alpha, AtenTensorHandle* ret0))

AOTI_VULKAN_STUB(aoti_torch_vulkan_mm(
    AtenTensorHandle self, AtenTensorHandle mat2, AtenTensorHandle* ret0))

// ── Convolution ─────────────────────────────────────────────────────
AOTI_VULKAN_STUB(aoti_torch_vulkan_convolution(
    AtenTensorHandle input, AtenTensorHandle weight, AtenTensorHandle* bias,
    const int64_t* stride, int64_t stride_len_,
    const int64_t* padding, int64_t padding_len_,
    const int64_t* dilation, int64_t dilation_len_,
    int32_t transposed,
    const int64_t* output_padding, int64_t output_padding_len_,
    int64_t groups,
    AtenTensorHandle* ret0))

AOTI_VULKAN_STUB(aoti_torch_vulkan_convolution_backward(
    AtenTensorHandle grad_output, AtenTensorHandle input, AtenTensorHandle weight,
    const int64_t** bias_sizes, int64_t bias_sizes_len_,
    const int64_t* stride, int64_t stride_len_,
    const int64_t* padding, int64_t padding_len_,
    const int64_t* dilation, int64_t dilation_len_,
    int32_t transposed,
    const int64_t* output_padding, int64_t output_padding_len_,
    int64_t groups,
    const int32_t* output_mask, int64_t output_mask_len_,
    AtenTensorHandle* ret0, AtenTensorHandle* ret1, AtenTensorHandle* ret2))

// ── Normalization ───────────────────────────────────────────────────
AOTI_VULKAN_STUB(aoti_torch_vulkan_native_batch_norm(
    AtenTensorHandle input, AtenTensorHandle* weight, AtenTensorHandle* bias,
    AtenTensorHandle* running_mean, AtenTensorHandle* running_var,
    int32_t training, double momentum, double eps,
    AtenTensorHandle* ret0, AtenTensorHandle* ret1, AtenTensorHandle* ret2))

AOTI_VULKAN_STUB(aoti_torch_vulkan_native_batch_norm_backward(
    AtenTensorHandle grad_out, AtenTensorHandle input, AtenTensorHandle* weight,
    AtenTensorHandle* running_mean, AtenTensorHandle* running_var,
    AtenTensorHandle* save_mean, AtenTensorHandle* save_invstd,
    int32_t train, double eps,
    const int32_t* output_mask, int64_t output_mask_len_,
    AtenTensorHandle* ret0, AtenTensorHandle* ret1, AtenTensorHandle* ret2))

AOTI_VULKAN_STUB(aoti_torch_vulkan_native_group_norm(
    AtenTensorHandle input, AtenTensorHandle* weight, AtenTensorHandle* bias,
    int64_t N, int64_t C, int64_t HxW,
    int64_t group, double eps,
    AtenTensorHandle* ret0, AtenTensorHandle* ret1, AtenTensorHandle* ret2))

AOTI_VULKAN_STUB(aoti_torch_vulkan_native_group_norm_backward(
    AtenTensorHandle grad_out, AtenTensorHandle input, AtenTensorHandle* weight,
    AtenTensorHandle* save_mean, AtenTensorHandle* save_invstd,
    const int32_t* output_mask, int64_t output_mask_len_,
    int64_t N, int64_t C, int64_t HxW, int64_t group,
    AtenTensorHandle* ret0, AtenTensorHandle* ret1, AtenTensorHandle* ret2))

AOTI_VULKAN_STUB(aoti_torch_vulkan_native_layer_norm(
    AtenTensorHandle input, const int64_t* normalized_shape, int64_t normalized_shape_len_,
    AtenTensorHandle* weight, AtenTensorHandle* bias,
    double eps,
    AtenTensorHandle* ret0, AtenTensorHandle* ret1, AtenTensorHandle* ret2))

AOTI_VULKAN_STUB(aoti_torch_vulkan_native_layer_norm_backward(
    AtenTensorHandle grad_out, AtenTensorHandle input,
    const int64_t* normalized_shape, int64_t normalized_shape_len_,
    AtenTensorHandle* save_mean, AtenTensorHandle* save_invstd,
    AtenTensorHandle* weight, AtenTensorHandle* bias,
    const int32_t* output_mask, int64_t output_mask_len_,
    AtenTensorHandle* ret0, AtenTensorHandle* ret1, AtenTensorHandle* ret2))

// ── Activations ─────────────────────────────────────────────────────
AOTI_VULKAN_STUB(aoti_torch_vulkan_relu(
    AtenTensorHandle self, AtenTensorHandle* ret0))

AOTI_VULKAN_STUB(aoti_torch_vulkan_gelu(
    AtenTensorHandle self, const char* approximate, AtenTensorHandle* ret0))

AOTI_VULKAN_STUB(aoti_torch_vulkan_gelu_backward(
    AtenTensorHandle grad_output, AtenTensorHandle self,
    const char* approximate,
    AtenTensorHandle* ret0))

AOTI_VULKAN_STUB(aoti_torch_vulkan_silu(
    AtenTensorHandle self, AtenTensorHandle* ret0))

// ── Pooling ─────────────────────────────────────────────────────────
AOTI_VULKAN_STUB(aoti_torch_vulkan_max_pool2d(
    AtenTensorHandle self, const int64_t* kernel_size, int64_t kernel_size_len_,
    const int64_t* stride, int64_t stride_len_,
    const int64_t* padding, int64_t padding_len_,
    const int64_t* dilation, int64_t dilation_len_,
    int32_t ceil_mode,
    AtenTensorHandle* ret0, AtenTensorHandle* ret1))

AOTI_VULKAN_STUB(aoti_torch_vulkan_avg_pool2d(
    AtenTensorHandle self, const int64_t* kernel_size, int64_t kernel_size_len_,
    const int64_t* stride, int64_t stride_len_,
    const int64_t* padding, int64_t padding_len_,
    int32_t ceil_mode, int32_t count_include_pad,
    int64_t* divisor_override,
    AtenTensorHandle* ret0))

// ── Element-wise & reductions ───────────────────────────────────────
AOTI_VULKAN_STUB(aoti_torch_vulkan_add_Tensor(
    AtenTensorHandle self, AtenTensorHandle other, double alpha,
    AtenTensorHandle* ret0))

AOTI_VULKAN_STUB(aoti_torch_vulkan_mul_Tensor(
    AtenTensorHandle self, AtenTensorHandle other, AtenTensorHandle* ret0))

AOTI_VULKAN_STUB(aoti_torch_vulkan_sum_dim_IntList(
    AtenTensorHandle self, const int64_t* dim, int64_t dim_len_,
    int32_t keepdim, int32_t* dtype,
    AtenTensorHandle* ret0))

AOTI_VULKAN_STUB(aoti_torch_vulkan_mean_dim(
    AtenTensorHandle self, const int64_t* dim, int64_t dim_len_,
    int32_t keepdim, int32_t* dtype,
    AtenTensorHandle* ret0))

// ── Copy / transpose / view ─────────────────────────────────────────
AOTI_VULKAN_STUB(aoti_torch_vulkan_copy_(
    AtenTensorHandle self, AtenTensorHandle src, int32_t non_blocking))

AOTI_VULKAN_STUB(aoti_torch_vulkan_t_copy_(
    AtenTensorHandle self, AtenTensorHandle* ret0))

AOTI_VULKAN_STUB(aoti_torch_vulkan_view(
    AtenTensorHandle self, const int64_t* size, int64_t size_len_,
    AtenTensorHandle* ret0))

AOTI_VULKAN_STUB(aoti_torch_vulkan_permute(
    AtenTensorHandle self, const int64_t* dims, int64_t dims_len_,
    AtenTensorHandle* ret0))

// ── Loss functions ──────────────────────────────────────────────────
AOTI_VULKAN_STUB(aoti_torch_vulkan_mse_loss_out(
    AtenTensorHandle out, AtenTensorHandle self, AtenTensorHandle target,
    int64_t reduction))

AOTI_VULKAN_STUB(aoti_torch_vulkan_cross_entropy_loss(
    AtenTensorHandle self, AtenTensorHandle target, AtenTensorHandle* weight,
    int64_t reduction, int64_t ignore_index, double label_smoothing,
    AtenTensorHandle* ret0))

// ── Optimizer ops ───────────────────────────────────────────────────
AOTI_VULKAN_STUB(aoti_torch_vulkan_addcmul_out(
    AtenTensorHandle out, AtenTensorHandle self, AtenTensorHandle tensor1,
    AtenTensorHandle tensor2, double value))

AOTI_VULKAN_STUB(aoti_torch_vulkan_addcdiv_out(
    AtenTensorHandle out, AtenTensorHandle self, AtenTensorHandle tensor1,
    AtenTensorHandle tensor2, double value))

// ── Random / dropout ────────────────────────────────────────────────
AOTI_VULKAN_STUB(aoti_torch_vulkan_bernoulli__float(
    AtenTensorHandle self, double p, AtenGeneratorHandle* generator))

AOTI_VULKAN_STUB(aoti_torch_vulkan_uniform_(
    AtenTensorHandle self, double from_val, double to_val,
    AtenGeneratorHandle* generator))

// ── Utility ─────────────────────────────────────────────────────────
AOTI_VULKAN_STUB(aoti_torch_vulkan_set__source_Storage_storage_offset(
    AtenTensorHandle self, AtenTensorHandle source, int64_t storage_offset,
    const int64_t* size, int64_t size_len_,
    const int64_t* stride, int64_t stride_len_,
    AtenTensorHandle* ret0))

#undef AOTI_VULKAN_STUB

// ── Vulkan AOTI runtime ABI (from AotiRuntime.h) ────────────────────
// Implemented in csrc/backend/AotiRuntime.cpp, linked into the .so.
// Declared here so the AOTI wrapper can include aoti_include/vulkan.h
// and get the full dispatch surface without importing Python.
#ifdef __cplusplus
extern "C" {
#endif

int torch_vulkan_aoti_make_kernel(
    const uint32_t* spirv_words,
    size_t spirv_words_n,
    const char* key,
    uint32_t n_buffers,
    uint32_t pc_size_bytes,
    AotiVulkanKernelHandle** out_handle);

int torch_vulkan_aoti_make_kernel_from_reflection(
    const uint32_t* spirv_words,
    size_t spirv_words_n,
    const char* reflection_json,
    size_t reflection_json_len,
    const char* key,
    AotiVulkanKernelHandle** out_handle);

int torch_vulkan_aoti_dispatch(
    AotiVulkanKernelHandle* handle,
    void** tensor_handles,
    size_t n_tensors,
    const void* push_constants,
    size_t push_constants_size,
    uint32_t wg_x,
    uint32_t wg_y,
    uint32_t wg_z,
    uint32_t num_outputs);

void torch_vulkan_aoti_destroy_kernel(AotiVulkanKernelHandle* handle);

const char* torch_vulkan_aoti_last_error(void);

// ── Model-level AOTI ──────────────────────────────────────────────
int torch_vulkan_aoti_model_load(
    const char* path,
    AotiVulkanModelHandle** out_handle);

int torch_vulkan_aoti_model_run(
    AotiVulkanModelHandle* handle,
    void** inputs,
    size_t n_inputs,
    void** outputs,
    size_t n_outputs);

void torch_vulkan_aoti_model_free(AotiVulkanModelHandle* handle);

// ── T7.4 specializations ──────────────────────────────────────────
int torch_vulkan_aoti_philox_advance(
    uint64_t* seed_state,
    size_t n_elements);

int torch_vulkan_aoti_scatter_atomic(
    AotiVulkanKernelHandle* kernel_handle,
    void** tensor_handles,
    size_t n_tensors,
    uint32_t numel,
    uint32_t src_numel,
    uint32_t out_numel,
    uint32_t num_outputs);

int torch_vulkan_aoti_foreach_optimizer(
    AotiVulkanKernelHandle* kernel_handle,
    void** tensor_handles,
    size_t n_tensors,
    const void* push_constants,
    size_t push_constants_size,
    uint32_t numel_per_param,
    uint32_t n_params,
    uint32_t num_outputs);

int torch_vulkan_aoti_flash_attention(
    AotiVulkanKernelHandle* kernel_handle,
    void** tensor_handles,
    size_t n_tensors,
    const void* push_constants,
    size_t push_constants_size,
    uint32_t wg_x,
    uint32_t wg_y,
    uint32_t wg_z,
    uint32_t num_outputs);

#ifdef __cplusplus
}  // extern "C"
#endif
