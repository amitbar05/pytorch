// ── Legacy eager-only ops ───────────────────────────────────────
// These four ops are the residual surface that cannot be replaced by an
// Inductor lowering because they handle low-level memory management:
//   - contiguous:       non-contiguous → contiguous GPU copy
//   - _to_copy:         device / dtype conversion
//   - as_strided:       view with arbitrary strides (GPU shader)
//   - resize_:          storage resize (in-place, for the opaque allocator)
//
// They live here — not in model_ops.cpp — so that the "no model_ops.cpp"
// anti-goal can be enforced at build time (M16.3 / M16.4).

#include "ops.h"
#include "dispatch.h"
#include "dtype_utils.h"
#include "../generated/shaders.h"
#include "../backend/Allocator.h"

namespace torch_vulkan { namespace ops {

// ── triu ───────────────────────────────────────────────────────
// Used by attention_ops.cpp for the SDPA causal-mask cache.
at::Tensor vulkan_triu(const at::Tensor& self, int64_t diagonal) {
    auto self_c = self.contiguous();
    TORCH_CHECK(self_c.dim() >= 2, "triu: input must be at least 2D");
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);

    int64_t rows = self_c.size(-2);
    int64_t cols = self_c.size(-1);

    auto flat_in = self_c.reshape({-1});
    auto output = at::empty_like(flat_in);
    uint32_t numel = static_cast<uint32_t>(flat_in.numel());
    if (numel == 0) return cast_from_float32(output.reshape(self_c.sizes()), orig_dtype);

    struct { uint32_t rows; uint32_t cols; int32_t diagonal; uint32_t numel_val; } params{
        static_cast<uint32_t>(rows),
        static_cast<uint32_t>(cols),
        static_cast<int32_t>(diagonal),
        numel};
    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("copy_triu_fwd",
                    shaders::copy_triu_fwd, shaders::copy_triu_fwd_size,
                    {flat_in, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output.reshape(self_c.sizes()), orig_dtype);
}

// ── contiguous ─────────────────────────────────────────────────
at::Tensor vulkan_contiguous(const at::Tensor& self,
                              at::MemoryFormat memory_format) {
    if (self.is_contiguous(memory_format)) return self;
    auto output = at::empty(self.sizes(), self.options());
    if (self.scalar_type() == at::kFloat) {
        dispatch_strided_copy(self, output);
    } else {
        TORCH_CHECK(false, "vulkan_contiguous: non-contiguous non-float32 tensor not supported "
                    "(dtype=", self.scalar_type(), ")");
    }
    return output;
}

// ── _to_copy ───────────────────────────────────────────────────
at::Tensor vulkan_to_copy(const at::Tensor& self,
                           std::optional<at::ScalarType> dtype,
                           std::optional<at::Layout> layout,
                           std::optional<at::Device> device,
                           std::optional<bool> pin_memory,
                           bool non_blocking,
                           std::optional<at::MemoryFormat> memory_format) {
    auto target_dtype = dtype.value_or(self.scalar_type());
    auto target_device = device.value_or(self.device());
    auto out = at::empty(self.sizes(), self.options()
        .dtype(target_dtype)
        .device(target_device));
    out.copy_(self, non_blocking);
    return out;
}

// ── as_strided ─────────────────────────────────────────────────
at::Tensor vulkan_as_strided(const at::Tensor& self, at::IntArrayRef size,
                              at::IntArrayRef stride,
                              std::optional<int64_t> storage_offset) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "as_strided");
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);
    TORCH_CHECK(size.size() <= 8, "Vulkan as_strided: max 8 dimensions supported");

    int64_t numel = 1;
    for (auto s : size) numel *= s;
    auto output = at::empty(size, self_c.options());
    if (numel == 0) return output;

    struct {
        uint32_t numel;
        uint32_t ndim;
        uint32_t sizes[8];
        uint32_t strides[8];
        uint32_t storage_offset;
    } params{};
    params.numel = static_cast<uint32_t>(numel);
    params.ndim = static_cast<uint32_t>(size.size());
    params.storage_offset = static_cast<uint32_t>(storage_offset.value_or(0));
    for (size_t i = 0; i < size.size(); i++) {
        params.sizes[i] = static_cast<uint32_t>(size[i]);
        params.strides[i] = static_cast<uint32_t>(stride[i]);
    }

    uint32_t workgroups = (params.numel + 255) / 256;
    dispatch_shader("copy_as_strided_fwd",
                    shaders::copy_as_strided_fwd, shaders::copy_as_strided_fwd_size,
                    {self_c, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

// ── resize_ ────────────────────────────────────────────────────
const at::Tensor& vulkan_resize_(const at::Tensor& self, at::IntArrayRef size,
                                  std::optional<at::MemoryFormat> memory_format) {
    if (self.sizes() == size) return self;

    auto dtype = self.scalar_type();
    size_t nbytes = c10::elementSize(dtype);
    for (auto s : size) nbytes *= s;
    if (nbytes == 0) nbytes = 1;

    auto& alloc = VulkanAllocator::instance();
    auto data_ptr = alloc.allocate(nbytes);

    auto* impl = self.unsafeGetTensorImpl();
    auto new_storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        static_cast<int64_t>(nbytes),
        std::move(data_ptr),
        &alloc,
        /*resizable=*/false);
    impl->set_storage_and_dtype(std::move(new_storage), caffe2::TypeMeta::fromScalarType(dtype));
    impl->set_sizes_contiguous(size);

    return self;
}

}} // namespace torch_vulkan::ops
