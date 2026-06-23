#include "ops.h"
#include "dispatch.h"
#include "dtype_utils.h"
#include "../generated/shaders.h"

#include <torch/library.h>

namespace torch_vulkan { namespace ops {

// ── Structural / view prims ──────────────────────────────────────
// All ops below operate only on real (non-complex) tensors.
// Complex tensors are not supported — TORCH_CHECK(false) for those.

at::Tensor vulkan_conj(const at::Tensor& self) {
    TORCH_CHECK(!self.is_complex(),
        "conj: complex tensors not supported on Vulkan backend");
    // Real tensors: conjugate is identity
    return self;
}

at::Tensor vulkan_conj_physical(const at::Tensor& self) {
    TORCH_CHECK(!self.is_complex(),
        "conj_physical: complex tensors not supported on Vulkan backend");
    // Real tensors: physical conjugate = contiguous clone
    return self.clone(c10::MemoryFormat::Contiguous);
}

at::Tensor vulkan_real(const at::Tensor& self) {
    TORCH_CHECK(!self.is_complex(),
        "real: complex tensors not supported on Vulkan backend");
    // Real tensors: real part = self (zero-copy view)
    return self;
}

at::Tensor vulkan_imag(const at::Tensor& self) {
    TORCH_CHECK(!self.is_complex(),
        "imag: complex tensors not supported on Vulkan backend");
    // Real tensors: imaginary part = zeros with same shape/dtype
    return at::zeros_like(self);
}

at::Tensor vulkan_view_dtype(const at::Tensor& self, at::ScalarType dtype) {
    auto src_size = c10::elementSize(self.scalar_type());
    auto dst_size = c10::elementSize(dtype);
    TORCH_CHECK(src_size == dst_size,
        "view.dtype: source dtype (", self.scalar_type(), ", ", src_size,
        " bytes) and destination dtype (", dtype, ", ", dst_size,
        " bytes) must have the same element size");
    TORCH_CHECK(self.is_contiguous(),
        "view.dtype: tensor must be contiguous");

    // Share the same underlying storage with a new dtype/shape interpretation.
    c10::Storage storage = self.storage();
    auto new_dtype = caffe2::TypeMeta::fromScalarType(dtype);
    auto out = at::detail::make_tensor<c10::TensorImpl>(
        std::move(storage),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        new_dtype);
    auto* impl = out.unsafeGetTensorImpl();
    impl->set_sizes_and_strides(self.sizes(), self.strides());
    impl->set_storage_offset(self.storage_offset());
    return out;
}

at::Tensor vulkan_as_strided_scatter(
    const at::Tensor& self,
    const at::Tensor& src,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    std::optional<int64_t> storage_offset) {

    // GPU implementation: clone self, then scatter src at strided positions.
    TORCH_CHECK(size.size() <= 8,
        "as_strided_scatter: only up to 8-D supported on Vulkan (got ", size.size(), "D)");

    // Promote to f32 for the GPU scatter shader; restore dtype at end.
    auto self_f32  = ensure_float32(self.contiguous());
    auto src_f32   = ensure_float32(src.contiguous());
    auto out       = self_f32.clone(c10::MemoryFormat::Contiguous);

    uint32_t numel_src = static_cast<uint32_t>(src_f32.numel());
    if (numel_src > 0) {
        uint32_t offset = storage_offset.has_value()
                          ? static_cast<uint32_t>(*storage_offset) : 0u;
        uint32_t ndim   = static_cast<uint32_t>(size.size());

        struct Params {
            uint32_t numel_src, offset, ndim, _pad;
            uint32_t sz[8], st[8];
        } params{};
        params.numel_src = numel_src;
        params.offset    = offset;
        params.ndim      = ndim;
        for (uint32_t i = 0; i < ndim; i++) {
            params.sz[i] = static_cast<uint32_t>(size[i]);
            params.st[i] = static_cast<uint32_t>(stride[i]);
        }

        uint32_t wg = (numel_src + 255) / 256;
        dispatch_shader("shape_as_strided_scatter_fwd",
                        shaders::shape_as_strided_scatter_fwd,
                        shaders::shape_as_strided_scatter_fwd_size,
                        {src_f32, out}, wg, 1, 1,
                        &params, sizeof(params));
    }
    return cast_from_float32(out, self.scalar_type());
}

// ── Control-flow effect tokens ───────────────────────────────────
// Tokens are scalar float tensors with no semantic content at runtime.

at::Tensor vulkan_make_token() {
    return at::zeros({}, at::TensorOptions()
        .dtype(at::kFloat)
        .device(c10::Device(c10::DeviceType::PrivateUse1, 0)));
}

void vulkan_sink_tokens(at::TensorList /*tokens*/) {
    // No-op — effect tokens have no runtime semantics
}

// ── Out-of-place RNG prims ───────────────────────────────────────

at::Tensor vulkan_normal_prim(double mean, double std, c10::SymIntArrayRef shape,
    at::ScalarType dtype, at::Device device,
    std::optional<at::Generator> generator) {
    std::vector<int64_t> sizes;
    sizes.reserve(shape.size());
    for (const auto& s : shape) sizes.push_back(s.expect_int());
    auto out = at::empty(sizes, at::TensorOptions().dtype(dtype).device(device));
    return vulkan_normal_(out, mean, std, generator);
}

at::Tensor vulkan_uniform_helper(c10::SymIntArrayRef shape, double low, double high,
    at::ScalarType dtype, at::Device device,
    std::optional<at::Generator> generator) {
    std::vector<int64_t> sizes;
    sizes.reserve(shape.size());
    for (const auto& s : shape) sizes.push_back(s.expect_int());
    auto out = at::empty(sizes, at::TensorOptions().dtype(dtype).device(device));
    return vulkan_uniform_(out, low, high, generator);
}

}} // namespace torch_vulkan::ops
