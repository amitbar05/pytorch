#pragma once

#include <torch/torch.h>

namespace torch_vulkan { namespace ops {

at::Tensor generic_unary_pointwise(const at::Tensor& self, const char* aten_op);
at::Tensor generic_binary_pointwise(const at::Tensor& self, const at::Tensor& other, const char* aten_op);

}} // namespace torch_vulkan::ops
