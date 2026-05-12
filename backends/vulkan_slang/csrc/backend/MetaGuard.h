// Null-storage guards for PrivateUse1 ops called with FakeTensor inputs.
//
// Inductor's `fake_tensor_prop` and AOT autograd's joint-graph trace dispatch
// our PrivateUse1 ops with FakeTensor inputs (null storage). Without a guard,
// any op that touches `data_ptr()` or a shader buffer crashes with
// `Cannot access data pointer of Tensor (e.g. FakeTensor)`.
//
// We return a *meta-device* tensor with the right shape/dtype. FakeTensorMode
// re-wraps the result and tracks the original (vulkan) device via input
// propagation, so downstream consumers see the correct device.
//
// (For ops where AOT autograd's C++ backward formulas dispatch directly with
// FakeTensors — `MulBackward0` calling `aten::mul`, etc. — we additionally
// register Python `fake_impl`s in `meta_patches.py` that intercept BEFORE
// our PrivateUse1 op runs at all, side-stepping the device-propagation
// mismatch that FakeTensorMode would otherwise hit when wrapping a meta
// output of a PrivateUse1 op.)
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/Functions.h>
#include <ATen/ExpandUtils.h>

namespace torch_vulkan {

inline bool is_null_storage(const at::Tensor& t) {
    return t.defined() && t.has_storage() && t.storage().data() == nullptr;
}

inline bool any_null_storage(const at::Tensor& a) {
    return is_null_storage(a);
}

inline bool any_null_storage(const at::Tensor& a, const at::Tensor& b) {
    return is_null_storage(a) || is_null_storage(b);
}

inline bool any_null_storage(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c) {
    return is_null_storage(a) || is_null_storage(b) || is_null_storage(c);
}

// DEPRECATED in PF.50. Returning meta-device tensors from null-storage
// guards causes downstream autograd-codegen formulas (see PF.13) to
// propagate meta through backward graphs, producing 0.5×/NaN gradient
// bugs. New sites should use ``make_vulkan_null{,_strided,_broadcast}``
// instead. Kept here so any straggler call site still compiles.
inline at::Tensor make_meta(at::IntArrayRef sizes, at::ScalarType dtype) {
    return at::empty(sizes, at::TensorOptions().dtype(dtype).device(c10::kMeta));
}

inline at::Tensor make_meta_like(const at::Tensor& self) {
    return at::empty(self.sizes().vec(),
                     at::TensorOptions().dtype(self.scalar_type()).device(c10::kMeta));
}

inline at::Tensor make_meta_broadcast(const at::Tensor& a, const at::Tensor& b) {
    auto out_size = at::infer_size_dimvector(a.sizes(), b.sizes());
    auto dtype = at::promote_types(a.scalar_type(), b.scalar_type());
    return make_meta(out_size, dtype);
}

// PF.13: View-op fake-storage helpers. PyTorch's view-op dispatch on
// PrivateUse1 short-circuits Python dispatch, so view ops on FakeTensors
// (whose underlying storage is meta-device) must construct an output
// tensor that carries the vulkan dispatch key *and* a vulkan-device null
// storage. Returning a meta tensor — or copying the meta storage from the
// FakeTensor's underlying tensor — propagates a meta-device tensor through
// downstream autograd-codegen formulas (e.g. SumBackward0::apply ->
// grad.expand_symint -> meta tensor flowing into mul/add), polluting the
// partitioner-saved bw outputs with uninitialized garbage.
//
// Critical invariant: ``Tensor::device()`` is derived from the *storage's*
// device, not the dispatch key set. So we must construct a fresh storage
// with PrivateUse1 device + null data ptr, not reuse the FakeTensor's
// meta storage.
inline at::Tensor make_vulkan_null_strided(
        const at::Tensor& src,
        at::IntArrayRef sizes,
        at::IntArrayRef strides,
        at::ScalarType dtype) {
    auto type_meta = caffe2::TypeMeta::fromScalarType(dtype);
    int64_t nbytes = static_cast<int64_t>(type_meta.itemsize());
    for (auto s : sizes) nbytes *= s;

    c10::Device device(c10::DeviceType::PrivateUse1,
                       src.device().is_privateuseone() ? src.device().index() : 0);
    c10::DataPtr data_ptr(nullptr, device);
    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        nbytes,
        std::move(data_ptr),
        /*allocator=*/nullptr,
        /*resizable=*/false);

    auto tensor = at::detail::make_tensor<c10::TensorImpl>(
        std::move(storage),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        type_meta);
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides);
    return tensor;
}

inline at::Tensor make_vulkan_null(
        const at::Tensor& src,
        at::IntArrayRef sizes,
        at::ScalarType dtype) {
    // Build contiguous strides (last dim = 1, then prefix-product).
    int64_t ndim = static_cast<int64_t>(sizes.size());
    std::vector<int64_t> strides(ndim);
    if (ndim > 0) {
        strides[ndim - 1] = 1;
        for (int64_t i = ndim - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * std::max<int64_t>(sizes[i + 1], 1);
        }
    }
    return make_vulkan_null_strided(src, sizes, strides, dtype);
}

inline at::Tensor make_vulkan_null(
        const at::Tensor& src,
        at::IntArrayRef sizes) {
    return make_vulkan_null(src, sizes, src.scalar_type());
}

// PF.50: null-storage helper for broadcasting binary ops. Picks the
// operand that already lives on a privateuse1 device as the device
// source (falls back to ``a``); infers the output shape via
// ``infer_size_dimvector`` and the output dtype via ``promote_types``,
// matching what the corresponding eager op would compute on a real
// vulkan tensor.
inline at::Tensor make_vulkan_null_broadcast(
        const at::Tensor& a,
        const at::Tensor& b) {
    auto out_size = at::infer_size_dimvector(a.sizes(), b.sizes());
    auto dtype = at::promote_types(a.scalar_type(), b.scalar_type());
    const at::Tensor& src = a.device().is_privateuseone() ? a : b;
    return make_vulkan_null(src, out_size, dtype);
}

}  // namespace torch_vulkan
