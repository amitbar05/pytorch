"""FakeTensor fake_impls for dtype casting, type promotion, mixed-precision,
activation backward, loss backward, view backward, and forward pointwise/
reduction ops.

Each function returns a ``FakeTensor`` with the correct output dtype,
shape, and device. Registered via ``_OP_IMPLS`` in ``__init__.py``.
"""

from __future__ import annotations

import torch

# ── Activation backward ops ──────────────────────────────────────────────────


def _threshold_backward_fake(grad_output, input, threshold):
    return torch.empty(
        list(grad_output.shape), dtype=grad_output.dtype, device=grad_output.device
    )


def _leaky_relu_backward_fake(
    grad_output, input_or_result, negative_slope, self_is_result
):
    return torch.empty(
        list(grad_output.shape), dtype=grad_output.dtype, device=grad_output.device
    )


def _elu_backward_fake(
    grad_output, alpha, scale, input_scale, is_result, self_or_result
):
    return torch.empty(
        list(grad_output.shape), dtype=grad_output.dtype, device=grad_output.device
    )


def _selu_backward_fake(
    grad_output, input_or_result, alpha, scale, input_scale, is_result
):
    return torch.empty(
        list(grad_output.shape), dtype=grad_output.dtype, device=grad_output.device
    )


def _hardtanh_backward_fake(grad_output, input, min_val, max_val):
    return torch.empty(
        list(grad_output.shape), dtype=grad_output.dtype, device=grad_output.device
    )


def _silu_backward_fake(grad_output, input_or_result):
    return torch.empty(
        list(grad_output.shape), dtype=grad_output.dtype, device=grad_output.device
    )


def _gelu_backward_fake(grad_output, input, approximate="none"):
    return torch.empty(
        list(grad_output.shape), dtype=grad_output.dtype, device=grad_output.device
    )


def _mish_backward_fake(grad_output, input):
    return torch.empty(
        list(grad_output.shape), dtype=grad_output.dtype, device=grad_output.device
    )


def _hardswish_backward_fake(grad_output, input):
    return torch.empty(
        list(grad_output.shape), dtype=grad_output.dtype, device=grad_output.device
    )


def _hardsigmoid_backward_fake(grad_output, input):
    return torch.empty(
        list(grad_output.shape), dtype=grad_output.dtype, device=grad_output.device
    )


def _softplus_backward_fake(grad_output, input, beta, threshold):
    return torch.empty(
        list(grad_output.shape), dtype=grad_output.dtype, device=grad_output.device
    )


def _sigmoid_backward_fake(grad_output, output):
    return torch.empty(
        list(grad_output.shape), dtype=grad_output.dtype, device=grad_output.device
    )


def _tanh_backward_fake(grad_output, output):
    return torch.empty(
        list(grad_output.shape), dtype=grad_output.dtype, device=grad_output.device
    )


def _softmax_backward_data_fake(grad_output, output, dim, input_dtype):
    return torch.empty(
        list(grad_output.shape), dtype=grad_output.dtype, device=grad_output.device
    )


def _log_softmax_backward_data_fake(grad_output, output, dim, input_dtype):
    return torch.empty(
        list(grad_output.shape), dtype=grad_output.dtype, device=grad_output.device
    )


def _avg_pool2d_backward_fake(
    grad_output,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    return torch.empty(list(input.shape), dtype=input.dtype, device=input.device)


def _max_pool2d_with_indices_backward_fake(
    grad_output, input, kernel_size, stride, padding, dilation, ceil_mode, indices
):
    return torch.empty(list(input.shape), dtype=input.dtype, device=input.device)


def _linear_backward_fake(input, grad_output, weight, output_mask):
    gi = (
        torch.empty(list(input.shape), dtype=input.dtype, device=input.device)
        if output_mask[0]
        else torch.empty(0, dtype=input.dtype, device=input.device)
    )
    gw = (
        torch.empty(list(weight.shape), dtype=weight.dtype, device=weight.device)
        if output_mask[1]
        else torch.empty(0, dtype=weight.dtype, device=weight.device)
    )
    gb = (
        torch.empty(
            [weight.size(0)], dtype=grad_output.dtype, device=grad_output.device
        )
        if output_mask[2]
        else torch.empty(0, dtype=grad_output.dtype, device=grad_output.device)
    )
    return gi, gw, gb


# ── Loss backward ops ────────────────────────────────────────────────────────


def _nll_loss_backward_fake(
    grad_output, input, target, weight, reduction, ignore_index, total_weight
):
    return torch.empty(list(input.shape), dtype=input.dtype, device=input.device)


def _cross_entropy_loss_backward_fake(
    grad_output,
    input,
    target,
    weight,
    reduction,
    ignore_index,
    label_smoothing,
    avg_factor,
):
    return torch.empty(list(input.shape), dtype=input.dtype, device=input.device)


# ── View ops needed by backward functionalization ────────────────────────────


def _reshape_alias_fake(self, size, stride):
    return torch.empty_strided(size, stride, dtype=self.dtype, device=self.device)


def _select_backward_fake(grad_output, input_sizes, dim, index):
    sizes = list(input_sizes)
    return torch.empty(sizes, dtype=grad_output.dtype, device=grad_output.device)


def _slice_backward_fake(grad_output, input_sizes, dim, start, end, step):
    sizes = list(input_sizes)
    return torch.empty(sizes, dtype=grad_output.dtype, device=grad_output.device)


def _unsqueeze_backward_fake(grad_output, input_sizes, dim):
    ndim = len(input_sizes)
    d = dim if dim >= 0 else ndim + dim + 1
    sizes = list(grad_output.size())
    sizes.pop(d)
    return torch.empty(sizes, dtype=grad_output.dtype, device=grad_output.device)


def _squeeze_backward_fake(grad_output, input_sizes, dim):
    ndim = len(input_sizes)
    d = dim if dim >= 0 else ndim + dim
    sizes = list(input_sizes)
    sizes[d] = input_sizes[d]
    return torch.empty(sizes, dtype=grad_output.dtype, device=grad_output.device)


def _t_backward_fake(grad_output, input_sizes):
    return torch.empty(
        list(input_sizes), dtype=grad_output.dtype, device=grad_output.device
    )


def _transpose_backward_fake(grad_output, input_sizes, dim0, dim1):
    return torch.empty(
        list(input_sizes), dtype=grad_output.dtype, device=grad_output.device
    )


def _view_backward_fake(grad_output, input_sizes):
    return torch.empty(
        list(input_sizes), dtype=grad_output.dtype, device=grad_output.device
    )


def _permute_backward_fake(grad_output, input_sizes, dims):
    return torch.empty(
        list(input_sizes), dtype=grad_output.dtype, device=grad_output.device
    )


def _expand_backward_fake(grad_output, input_sizes):
    return torch.empty(
        list(input_sizes), dtype=grad_output.dtype, device=grad_output.device
    )


def _repeat_backward_fake(grad_output, input_sizes):
    return torch.empty(
        list(input_sizes), dtype=grad_output.dtype, device=grad_output.device
    )


# ── Forward pointwise / reduction fake_impls ────────────────────────────────
#
# Required because PrivateUse1 outranks Meta, so PyTorch's built-in meta
# kernels never fire on Vulkan FakeTensors. Without these, AOT autograd's
# joint-graph trace dispatches into our C++ ops with FakeTensor inputs and
# we end up returning kMeta tensors that break downstream device propagation
# in backward formulas (`MulBackward0` doing `grad * saved`).


def _unary_fake(self):
    # M-pipeline-9: `t.new_empty(t.shape)` not `torch.empty_like(t)` —
    # see M18.3 closure (joint-graph partitioner collapses `empty_like`
    # to `aten.full(shape, 0)` if it classifies the fake as shape-only).
    return self.new_empty(self.shape)


def _binary_fake(self, other, alpha=1):
    self_is_t = isinstance(self, torch.Tensor)
    other_is_t = isinstance(other, torch.Tensor)
    if self_is_t and other_is_t:
        out_shape = torch.broadcast_shapes(self.shape, other.shape)
        out_dtype = torch.promote_types(self.dtype, other.dtype)
        device = self.device
    elif self_is_t:
        out_shape = self.shape
        out_dtype = self.dtype
        device = self.device
    else:
        # other must be a tensor for us to be on this Vulkan dispatch path.
        out_shape = other.shape
        out_dtype = other.dtype
        device = other.device
    return torch.empty(out_shape, dtype=out_dtype, device=device)


def _binary_no_alpha_fake(self, other):
    return _binary_fake(self, other)


def _binary_scalar_fake(self, other, alpha=1):
    return self.new_empty(self.shape)  # M-pipeline-9: not `torch.empty_like` — see M18.3.


def _comparison_fake(self, other):
    if isinstance(other, torch.Tensor):
        out_shape = torch.broadcast_shapes(self.shape, other.shape)
    else:
        out_shape = self.shape
    return torch.empty(out_shape, dtype=torch.bool, device=self.device)


def _gelu_fake(self, *, approximate="none"):
    return self.new_empty(self.shape)  # M-pipeline-9: not `torch.empty_like` — see M18.3.


def _leaky_relu_fake(self, negative_slope=0.01):
    return self.new_empty(self.shape)  # M-pipeline-9: not `torch.empty_like` — see M18.3.


def _elu_fake(self, alpha=1.0, scale=1.0, input_scale=1.0):
    return self.new_empty(self.shape)  # M-pipeline-9: not `torch.empty_like` — see M18.3.


def _hardtanh_fake(self, min_val=-1.0, max_val=1.0):
    return self.new_empty(self.shape)  # M-pipeline-9: not `torch.empty_like` — see M18.3.


def _softplus_fake(self, beta=1.0, threshold=20.0):
    return self.new_empty(self.shape)  # M-pipeline-9: not `torch.empty_like` — see M18.3.


def _clamp_fake(self, min=None, max=None):
    return self.new_empty(self.shape)  # M-pipeline-9: not `torch.empty_like` — see M18.3.


def _clamp_min_fake(self, min):
    return self.new_empty(self.shape)  # M-pipeline-9: not `torch.empty_like` — see M18.3.


def _clamp_max_fake(self, max):
    return self.new_empty(self.shape)  # M-pipeline-9: not `torch.empty_like` — see M18.3.


def _where_fake(condition, self, other):
    out_shape = torch.broadcast_shapes(condition.shape, self.shape, other.shape)
    out_dtype = torch.promote_types(self.dtype, other.dtype)
    return torch.empty(out_shape, dtype=out_dtype, device=self.device)


def _softmax_fake(self, dim, half_to_float=False):
    out_dtype = torch.float32 if half_to_float else self.dtype
    return torch.empty(self.shape, dtype=out_dtype, device=self.device)


def _log_softmax_fake(self, dim, half_to_float=False):
    return _softmax_fake(self, dim, half_to_float)


def _reduction_shape(self, dim, keepdim):
    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        return [1] * self.dim() if keepdim else []
    if isinstance(dim, int):
        dim = [dim]
    reduced = {(d % self.dim()) for d in dim}
    out = []
    for i in range(self.dim()):
        if i in reduced:
            if keepdim:
                out.append(1)
        else:
            out.append(self.size(i))
    return out


def _sum_dim_fake(self, dim=None, keepdim=False, *, dtype=None):
    out_shape = _reduction_shape(self, dim, keepdim)
    out_dtype = dtype or self.dtype
    return torch.empty(out_shape, dtype=out_dtype, device=self.device)


def _mean_dim_fake(self, dim=None, keepdim=False, *, dtype=None):
    return _sum_dim_fake(self, dim, keepdim, dtype=dtype)


def _amax_amin_fake(self, dim=(), keepdim=False):
    out_shape = _reduction_shape(self, list(dim) if dim else None, keepdim)
    return torch.empty(out_shape, dtype=self.dtype, device=self.device)


def _argmax_argmin_fake(self, dim=None, keepdim=False):
    if dim is None:
        out_shape = [1] * self.dim() if keepdim else []
    else:
        out_shape = _reduction_shape(self, [dim], keepdim)
    return torch.empty(out_shape, dtype=torch.int64, device=self.device)


def _max_dim_fake(self, dim, keepdim=False):
    out_shape = _reduction_shape(self, [dim], keepdim)
    vals = torch.empty(out_shape, dtype=self.dtype, device=self.device)
    idxs = torch.empty(out_shape, dtype=torch.int64, device=self.device)
    return vals, idxs


def _min_dim_fake(self, dim, keepdim=False):
    return _max_dim_fake(self, dim, keepdim)


def _linear_fake(input, weight, bias=None):
    out_shape = list(input.shape)
    out_shape[-1] = weight.size(0)
    return torch.empty(out_shape, dtype=input.dtype, device=input.device)
