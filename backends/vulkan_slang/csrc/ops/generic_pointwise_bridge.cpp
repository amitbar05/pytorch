#include "generic_pointwise_bridge.h"
#include "dtype_utils.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <mutex>

namespace torch_vulkan { namespace ops {

namespace py = pybind11;

static py::object _get_module() {
    static py::object mod;
    static std::once_flag flag;
    std::call_once(flag, []() {
        py::gil_scoped_acquire gil;
        mod = py::module_::import(
            "torch_vulkan.inductor.generic_pointwise_dispatch");
    });
    return mod;
}

at::Tensor generic_unary_pointwise(const at::Tensor& self, const char* aten_op) {
    check_supported_float(self, aten_op);
    auto orig_dtype = self.scalar_type();
    auto self_f32 = ensure_float32(self.contiguous());

    if (self_f32.numel() == 0) {
        return cast_from_float32(at::empty(self_f32.sizes(), self_f32.options()), orig_dtype);
    }

    py::gil_scoped_acquire gil;
    auto mod = _get_module();
    at::Tensor result_f32 = mod.attr("dispatch_unary_pointwise")(
        aten_op, self_f32).cast<at::Tensor>();
    return cast_from_float32(result_f32, orig_dtype);
}

at::Tensor generic_binary_pointwise(const at::Tensor& self, const at::Tensor& other, const char* aten_op) {
    check_supported_float(self, aten_op);
    auto orig_dtype = self.scalar_type();

    auto other_dev = other;
    if (other_dev.device() != self.device())
        other_dev = other_dev.to(self.device());

    auto self_f32 = ensure_float32(self.contiguous());
    auto other_f32 = ensure_float32(other_dev.contiguous());

    // Broadcast to common shape
    auto bcast_shape = at::infer_size(self_f32.sizes(), other_f32.sizes());
    auto self_c = self_f32.expand(bcast_shape).contiguous();
    auto other_c = other_f32.expand(bcast_shape).contiguous();

    if (self_c.numel() == 0) {
        return cast_from_float32(at::empty(self_c.sizes(), self_c.options()), orig_dtype);
    }

    py::gil_scoped_acquire gil;
    auto mod = _get_module();
    at::Tensor result_f32 = mod.attr("dispatch_binary_pointwise")(
        aten_op, self_c, other_c).cast<at::Tensor>();
    return cast_from_float32(result_f32, orig_dtype);
}

}} // namespace torch_vulkan::ops
