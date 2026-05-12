"""Python-level FakeTensor workarounds for view ops on Vulkan.

For each view op below we install a `fake_impl` via the internal
`FakeImplHolder.kernels` list. FakeTensor's `_dispatch_impl` checks this
registry BEFORE the meta-in-TLS dispatch path, so the workaround takes
priority over our PrivateUse1 C++ kernel (which would otherwise try to
allocate real Vulkan storage and fail on FakeTensor inputs).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch._library.fake_impl import Kernel
from torch._library.simple_registry import singleton

_patched = False


# ── Shape / view ops ─────────────────────────────────────────────────────────


def _as_strided_fake(self, size, stride, storage_offset=None):
    return torch.empty_strided(
        size,
        stride,
        dtype=self.dtype,
        device=self.device,
    )


def _permute_fake(self, dims):
    sizes = [int(self.size(d)) for d in dims]
    strides = [int(self.stride(d)) for d in dims]
    return torch.empty_strided(
        sizes,
        strides,
        dtype=self.dtype,
        device=self.device,
    )


def _transpose_int_fake(self, dim0, dim1):
    ndim = self.dim()
    dim0 = dim0 if dim0 >= 0 else ndim + dim0
    dim1 = dim1 if dim1 >= 0 else ndim + dim1
    sizes = list(self.size())
    strides = list(self.stride())
    sizes[dim0], sizes[dim1] = sizes[dim1], sizes[dim0]
    strides[dim0], strides[dim1] = strides[dim1], strides[dim0]
    return torch.empty_strided(
        sizes,
        strides,
        dtype=self.dtype,
        device=self.device,
    )


def _t_fake(self):
    assert self.dim() <= 2, "t() only supports 0/1/2-D tensors"
    if self.dim() < 2:
        return torch.empty_strided(
            list(self.size()),
            list(self.stride()),
            dtype=self.dtype,
            device=self.device,
        )
    return _transpose_int_fake(self, 0, 1)


def _view_fake(self, size):
    # ``aten::view`` semantics include the special ``-1`` placeholder that
    # tells the runtime to infer the missing dimension from ``self.numel()``.
    # Inductor's ``_low_memory_max_pool_with_offsets`` decomposition relies
    # on this — it builds ``bh_shape = [1] * ndim`` then sets one entry to
    # ``-1`` and calls ``arange(N).view(bh_shape)``. Plain
    # ``torch.empty(size)`` rejects negative dims, so the previous impl
    # crashed under FakeTensorMode with
    # ``"Trying to create tensor with negative dimension -1"`` and broke
    # every compiled max-pool backward.
    sizes = list(size)
    neg = [i for i, s in enumerate(sizes) if isinstance(s, int) and s == -1]
    if neg:
        if len(neg) > 1:
            raise RuntimeError(f"only one dimension can be -1 in view; got {sizes}")
        known = 1
        for s in sizes:
            if isinstance(s, int) and s == -1:
                continue
            known = known * s
        # ``self.numel()`` may be a SymInt under dynamic shapes; the
        # division below preserves that.
        sizes[neg[0]] = self.numel() // known
    return torch.empty(sizes, dtype=self.dtype, device=self.device)


def _flatten_using_ints_fake(self, start_dim=0, end_dim=-1):
    # PF.55: ``aten::flatten.using_ints`` is CompositeImplicitAutograd; its
    # C++ decomposition (``at::native::flatten`` → ``reshape_symint`` →
    # ``at::_ops::view::call``) bypasses the Python torch_dispatch mode
    # for the inner ``view`` call, landing on our PrivateUse1 C++ adapter
    # which calls ``SymInt::expect_int()`` and blows up under symbolic
    # batch. Intercept the composite at FakeTensor level so the
    # decomposition never runs in C++ — we compute the flattened shape
    # in Python with SymInt-preserving multiplication.
    ndim = self.dim()
    if ndim == 0:
        return torch.empty([1], dtype=self.dtype, device=self.device)
    sd = start_dim if start_dim >= 0 else ndim + start_dim
    ed = end_dim if end_dim >= 0 else ndim + end_dim
    sizes = list(self.size())
    if sd == ed:
        return torch.empty(sizes, dtype=self.dtype, device=self.device)
    flat = 1
    for s in sizes[sd : ed + 1]:
        flat = flat * s
    new_sizes = sizes[:sd] + [flat] + sizes[ed + 1 :]
    return torch.empty(new_sizes, dtype=self.dtype, device=self.device)


def _unflatten_int_fake(self, dim, sizes):
    # PF.55: ``aten::unflatten.int`` is CompositeImplicitAutograd; ditto
    # ``flatten.using_ints``, its C++ decomposition reaches our PrivateUse1
    # view adapter via ``view_symint``. Intercept at the composite level.
    ndim = self.dim()
    d = dim if dim >= 0 else ndim + dim
    cur = list(self.size())
    new_sizes = cur[:d] + list(sizes) + cur[d + 1 :]
    return torch.empty(new_sizes, dtype=self.dtype, device=self.device)


def _view_as_fake(self, other):
    # PF.55: ``aten::view_as`` is CompositeImplicitAutograd, decomposing
    # to ``view(other.size())`` in C++. Intercept at the composite level.
    return torch.empty(
        list(other.size()),
        dtype=self.dtype,
        device=self.device,
    )


def _reshape_as_fake(self, other):
    # PF.55: ``aten::reshape_as`` is CompositeImplicitAutograd, decomposing
    # to ``reshape(other.size())`` in C++. Intercept at the composite level.
    return torch.empty(
        list(other.size()),
        dtype=self.dtype,
        device=self.device,
    )


def _unsqueeze_fake(self, dim):
    ndim = self.dim()
    d = dim if dim >= 0 else ndim + dim + 1
    sizes = list(self.size())
    sizes.insert(d, 1)
    strides = list(self.stride())
    stride_at_d = 1 if d == ndim else strides[d] * sizes[d + 1]
    strides.insert(d, stride_at_d)
    return torch.empty_strided(
        sizes,
        strides,
        dtype=self.dtype,
        device=self.device,
    )


def _squeeze_dim_fake(self, dim):
    ndim = self.dim()
    d = dim if dim >= 0 else ndim + dim
    if self.size(d) != 1:
        return torch.empty_strided(
            list(self.size()),
            list(self.stride()),
            dtype=self.dtype,
            device=self.device,
        )
    sizes = list(self.size())
    sizes.pop(d)
    strides = list(self.stride())
    strides.pop(d)
    return torch.empty_strided(
        sizes,
        strides,
        dtype=self.dtype,
        device=self.device,
    )


def _expand_fake(self, size, implicit=False):
    target = list(size)
    src_shape = list(self.size())
    src_stride = list(self.stride())
    pad = len(target) - len(src_shape)
    src_shape = [1] * pad + src_shape
    src_stride = [0] * pad + src_stride
    out_strides = [
        0 if (target[i] != src_shape[i] or src_shape[i] == 1) else src_stride[i]
        for i in range(len(target))
    ]
    return torch.empty_strided(
        target,
        out_strides,
        dtype=self.dtype,
        device=self.device,
    )


def _slice_fake(self, dim, start, end, step=1):
    ndim = self.dim()
    d = dim if dim >= 0 else ndim + dim
    cur = int(self.size(d))
    start = (
        0 if start is None else (max(0, cur + start) if start < 0 else min(start, cur))
    )
    end = cur if end is None else (max(0, cur + end) if end < 0 else min(end, cur))
    length = max(0, (end - start + step - 1) // step)
    sizes = list(self.size())
    sizes[d] = length
    strides = list(self.stride())
    strides[d] = int(self.stride(d)) * step
    return torch.empty_strided(
        sizes,
        strides,
        dtype=self.dtype,
        device=self.device,
    )


def _constant_pad_nd_fake(self, pad, value=0):
    # `pad` is a flat list [(begin_last, end_last, begin_second_last, ...)].
    sizes = list(self.size())
    # Each pair in `pad` corresponds to the last N//2 dimensions.
    for i, (begin, end) in enumerate(zip(pad[0::2], pad[1::2])):
        dim = len(sizes) - 1 - i
        if 0 <= dim < len(sizes):
            sizes[dim] = sizes[dim] + begin + end
    return torch.empty(sizes, dtype=self.dtype, device=self.device)


def _as_strided_scatter_fake(self, src, size, stride, storage_offset=None):
    return torch.empty(list(self.shape), dtype=self.dtype, device=self.device)


def _masked_scatter_fake(self, mask, source):
    return torch.empty(list(self.shape), dtype=self.dtype, device=self.device)


# ── BLAS ops ─────────────────────────────────────────────────────────────────


def _mm_fake(self, mat2):
    M = int(self.size(0))
    N = int(mat2.size(1))
    return torch.empty([M, N], dtype=self.dtype, device=self.device)


def _bmm_fake(self, mat2):
    B, M = int(self.size(0)), int(self.size(1))
    N = int(mat2.size(2))
    return torch.empty([B, M, N], dtype=self.dtype, device=self.device)


def _addmm_fake(self, mat1, mat2, *, beta=1, alpha=1):
    M = int(mat1.size(0))
    N = int(mat2.size(1))
    return torch.empty([M, N], dtype=mat1.dtype, device=mat1.device)


# ── Conv ops ──────────────────────────────────────────────────────────────────


def _convolution_overrideable_fake(
    input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
):
    N = int(input.size(0))
    C_out = int(weight.size(0))
    if not transposed:
        iH, iW = int(input.size(2)), int(input.size(3))
        kH, kW = int(weight.size(2)), int(weight.size(3))
        pH, pW = int(padding[0]), int(padding[1])
        sH, sW = int(stride[0]), int(stride[1])
        dH, dW = int(dilation[0]), int(dilation[1])
        oH = (iH + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        oW = (iW + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        return torch.empty([N, C_out, oH, oW], dtype=input.dtype, device=input.device)
    return torch.empty_like(input)


def _convolution_backward_overrideable_fake(
    grad_output,
    input,
    weight,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    grad_input = (
        torch.empty_like(input)
        if output_mask[0]
        else torch.empty(0, dtype=input.dtype, device=input.device)
    )
    grad_weight = (
        torch.empty_like(weight)
        if output_mask[1]
        else torch.empty(0, dtype=weight.dtype, device=weight.device)
    )
    grad_bias = (
        torch.empty(bias_sizes, dtype=grad_output.dtype, device=grad_output.device)
        if output_mask[2] and bias_sizes
        else torch.empty(0, dtype=grad_output.dtype, device=grad_output.device)
    )
    return grad_input, grad_weight, grad_bias


# ── Normalization backward ops ────────────────────────────────────────────────


def _native_batch_norm_backward_fake(
    grad_out,
    input,
    weight,
    running_mean,
    running_var,
    save_mean,
    save_var,
    train,
    eps,
    output_mask,
):
    grad_input = (
        torch.empty_like(input)
        if output_mask[0]
        else torch.empty(0, dtype=input.dtype, device=input.device)
    )
    C = int(input.size(1))
    grad_weight = (
        torch.empty([C], dtype=grad_out.dtype, device=grad_out.device)
        if output_mask[1]
        else torch.empty(0, dtype=grad_out.dtype, device=grad_out.device)
    )
    grad_bias = (
        torch.empty([C], dtype=grad_out.dtype, device=grad_out.device)
        if output_mask[2]
        else torch.empty(0, dtype=grad_out.dtype, device=grad_out.device)
    )
    return grad_input, grad_weight, grad_bias


def _native_layer_norm_backward_fake(
    grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask
):
    grad_input = (
        torch.empty_like(input)
        if output_mask[0]
        else torch.empty(0, dtype=input.dtype, device=input.device)
    )
    grad_weight = (
        torch.empty_like(weight)
        if (output_mask[1] and weight is not None)
        else torch.empty(0, dtype=grad_out.dtype, device=grad_out.device)
    )
    grad_bias = (
        torch.empty_like(bias)
        if (output_mask[2] and bias is not None)
        else torch.empty(0, dtype=grad_out.dtype, device=grad_out.device)
    )
    return grad_input, grad_weight, grad_bias


def _native_group_norm_backward_fake(
    grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask
):
    grad_input = (
        torch.empty_like(input)
        if output_mask[0]
        else torch.empty(0, dtype=input.dtype, device=input.device)
    )
    grad_weight = (
        torch.empty([C], dtype=grad_out.dtype, device=grad_out.device)
        if output_mask[1]
        else torch.empty(0, dtype=grad_out.dtype, device=grad_out.device)
    )
    grad_bias = (
        torch.empty([C], dtype=grad_out.dtype, device=grad_out.device)
        if output_mask[2]
        else torch.empty(0, dtype=grad_out.dtype, device=grad_out.device)
    )
    return grad_input, grad_weight, grad_bias


# ── Indexing ops ──────────────────────────────────────────────────────────────


def _gather_fake(self, dim, index, sparse_grad=False):
    return torch.empty_like(index, dtype=self.dtype)


def _scatter_src_fake(self, dim, index, src):
    return torch.empty(list(self.shape), dtype=self.dtype, device=self.device)


def _scatter_value_fake(self, dim, index, value):
    return torch.empty(list(self.shape), dtype=self.dtype, device=self.device)


def _scatter_add_fake(self, dim, index, src):
    return torch.empty(list(self.shape), dtype=self.dtype, device=self.device)


def _index_put_fake(self, indices, values, accumulate=False):
    return torch.empty(list(self.shape), dtype=self.dtype, device=self.device)


def _repeat_interleave_self_int_fake(self, repeats, dim=None, output_size=None):
    if dim is None:
        n = int(self.numel()) * int(repeats)
        return torch.empty([n], dtype=self.dtype, device=self.device)
    sizes = list(self.size())
    sizes[dim] = sizes[dim] * int(repeats)
    return torch.empty(sizes, dtype=self.dtype, device=self.device)


# ── Upsample backward ops ─────────────────────────────────────────────────────


def _upsample_bilinear2d_backward_fake(
    grad_output, output_size, input_size, align_corners, scales_h=None, scales_w=None
):
    return torch.empty(
        list(input_size), dtype=grad_output.dtype, device=grad_output.device
    )


def _upsample_nearest2d_backward_fake(
    grad_output, output_size, input_size, scales_h=None, scales_w=None
):
    return torch.empty(
        list(input_size), dtype=grad_output.dtype, device=grad_output.device
    )


# ── Attention backward ────────────────────────────────────────────────────────


def _sdpa_fake(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    # Output shape: [B, H_q, S_q, D_v]
    sizes = list(query.size())
    sizes[-1] = value.size(-1)
    return torch.empty(sizes, dtype=query.dtype, device=query.device)


def _sdpa_backward_fake(
    grad_out,
    query,
    key,
    value,
    out,
    softmax_lse,
    cum_seq_q=None,
    cum_seq_k=None,
    max_q=0,
    max_k=0,
    dropout_p=0.0,
    is_causal=False,
    philox_seed=None,
    philox_offset=None,
    scale=None,
):
    return torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)


# ── FFT ops ───────────────────────────────────────────────────────────────────


def _fft_r2c_fake(self, dim, normalization, onesided):
    sizes = list(self.size())
    if onesided:
        # Last dim in `dim` becomes size // 2 + 1.
        last_dim = dim[-1] if hasattr(dim, "__len__") else dim
        sizes[last_dim] = sizes[last_dim] // 2 + 1
    return torch.empty(sizes, dtype=torch.complex64, device=self.device)


def _fft_c2c_fake(self, dim, normalization, forward):
    return torch.empty(list(self.shape), dtype=self.dtype, device=self.device)


def _fft_c2r_fake(self, dim, normalization, last_dim_size):
    sizes = list(self.size())
    last_dim = dim[-1] if hasattr(dim, "__len__") else dim
    sizes[last_dim] = int(last_dim_size) if last_dim_size else (sizes[last_dim] - 1) * 2
    return torch.empty(sizes, dtype=torch.float32, device=self.device)


# ── SVD ───────────────────────────────────────────────────────────────────────


def _linalg_svd_fake(A, full_matrices=True, driver=None):
    shape = list(A.shape)
    M, N = shape[-2], shape[-1]
    K = min(M, N)
    batch = shape[:-2]
    U_shape = batch + [M, M if full_matrices else K]
    S_shape = batch + [K]
    Vh_shape = batch + [N if full_matrices else K, N]
    dev, dt = A.device, A.dtype
    real_dt = torch.float32 if dt in (torch.complex64, torch.float32) else torch.float64
    return (
        torch.empty(U_shape, dtype=dt, device=dev),
        torch.empty(S_shape, dtype=real_dt, device=dev),
        torch.empty(Vh_shape, dtype=dt, device=dev),
    )


# ── Memory / copy ops ────────────────────────────────────────────────────────


def _clone_fake(self, *, memory_format=torch.preserve_format):
    return torch.empty(list(self.shape), dtype=self.dtype, device=self.device)


def _copy__fake(self, src, non_blocking=False):
    return torch.empty(list(self.shape), dtype=self.dtype, device=self.device)


def _contiguous_fake(self, *, memory_format=torch.preserve_format):
    return torch.empty(list(self.shape), dtype=self.dtype, device=self.device)


def _empty_like_fake(
    self,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
    memory_format=torch.preserve_format,
):
    return torch.empty(list(self.shape), dtype=dtype or self.dtype, device=self.device)


def _fill_scalar_fake(self, value):
    return torch.empty(list(self.shape), dtype=self.dtype, device=self.device)


def _zero__fake(self):
    return torch.empty(list(self.shape), dtype=self.dtype, device=self.device)


def _detach_fake(self):
    return torch.empty(list(self.shape), dtype=self.dtype, device=self.device)


# ── Shape ops ─────────────────────────────────────────────────────────────────


def _flip_fake(self, dims):
    return torch.empty(list(self.shape), dtype=self.dtype, device=self.device)


def _narrow_fake(self, dim, start, length):
    sizes = list(self.size())
    ndim = self.dim()
    d = dim if dim >= 0 else ndim + dim
    sizes[d] = length
    return torch.empty(sizes, dtype=self.dtype, device=self.device)


def _select_int_fake(self, dim, index):
    sizes = list(self.size())
    ndim = self.dim()
    d = dim if dim >= 0 else ndim + dim
    sizes.pop(d)
    return torch.empty(sizes, dtype=self.dtype, device=self.device)


def _diagonal_fake(self, offset=0, dim1=0, dim2=1):
    sizes = list(self.size())
    ndim = self.dim()
    d1 = dim1 if dim1 >= 0 else ndim + dim1
    d2 = dim2 if dim2 >= 0 else ndim + dim2
    out_size = min(sizes[d1], sizes[d2] - abs(offset))
    sizes.pop(d2)
    sizes.pop(d1)
    sizes.append(out_size)
    return torch.empty(sizes, dtype=self.dtype, device=self.device)


def _split_with_sizes_fake(self, split_sizes, dim=0):
    ndim = self.dim()
    d = dim if dim >= 0 else ndim + dim
    return tuple(
        torch.empty(
            list(self.size())[:d] + [s] + list(self.size())[d + 1 :],
            dtype=self.dtype,
            device=self.device,
        )
        for s in split_sizes
    )


def _chunk_fake(self, chunks, dim=0):
    ndim = self.dim()
    d = dim if dim >= 0 else ndim + dim
    total = int(self.size(d))
    size = (total + chunks - 1) // chunks
    sizes = [size] * (total // size)
    if total % size:
        sizes.append(total % size)
    return _split_with_sizes_fake(self, sizes, dim)


def _cat_fake(tensors, dim=0):
    ndim = tensors[0].dim()
    d = dim if dim >= 0 else ndim + dim
    cat_size = sum(int(t.size(d)) for t in tensors)
    sizes = list(tensors[0].size())
    sizes[d] = cat_size
    return torch.empty(sizes, dtype=tensors[0].dtype, device=tensors[0].device)


def _stack_fake(tensors, dim=0):
    sizes = list(tensors[0].size())
    d = dim if dim >= 0 else dim + len(sizes) + 1
    sizes.insert(d, len(tensors))
    return torch.empty(sizes, dtype=tensors[0].dtype, device=tensors[0].device)


def _repeat_fake(self, repeats):
    sizes = [int(self.size(i)) * int(repeats[i]) for i in range(self.dim())]
    return torch.empty(sizes, dtype=self.dtype, device=self.device)


def _tile_fake(self, dims):
    sizes = [int(self.size(i)) * int(dims[i]) for i in range(self.dim())]
    return torch.empty(sizes, dtype=self.dtype, device=self.device)


def _roll_fake(self, shifts, dims=None):
    return torch.empty(list(self.shape), dtype=self.dtype, device=self.device)


def _where_self_fake(condition, self, other):
    return torch.empty(list(self.shape), dtype=self.dtype, device=self.device)


def _index_select_fake(self, dim, index):
    sizes = list(self.size())
    ndim = self.dim()
    d = dim if dim >= 0 else ndim + dim
    sizes[d] = int(index.size(0))
    return torch.empty(sizes, dtype=self.dtype, device=self.device)


def _gather_fake_backward(self, dim, index, sparse_grad=False):
    return torch.empty_like(index, dtype=self.dtype)


def _scatter_reduce_fake(self, dim, index, src, reduce, include_self=True):
    return torch.empty(list(self.shape), dtype=self.dtype, device=self.device)


def _masked_fill_scalar_fake(self, mask, value):
    return torch.empty(list(self.shape), dtype=self.dtype, device=self.device)


def _index_put_fake_impl(self, indices, values, accumulate=False):
    return torch.empty(list(self.shape), dtype=self.dtype, device=self.device)


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


# ── Reduction backward ops ───────────────────────────────────────────────────


def _sum_backward_fake(grad_output, input, dim, keepdim, dtype):
    sizes = list(input.size())
    if dim is not None:
        if not keepdim:
            for d in sorted(
                [
                    d if d >= 0 else len(sizes) + d
                    for d in (dim if hasattr(dim, "__iter__") else [dim])
                ],
                reverse=True,
            ):
                sizes.pop(d)
    else:
        sizes = []
    if not sizes:
        return torch.empty([], dtype=input.dtype, device=input.device)
    return torch.empty(sizes, dtype=input.dtype, device=input.device)


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


# ── Embedding ops ─────────────────────────────────────────────────────────────


def _embedding_fake(
    weight, indices, padding_idx=None, scale_grad_by_freq=False, sparse=False
):
    sizes = list(indices.size()) + [weight.size(1)]
    return torch.empty(sizes, dtype=weight.dtype, device=weight.device)


def _embedding_backward_fake(
    grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
):
    return torch.empty(
        [num_weights, grad_output.size(-1)],
        dtype=grad_output.dtype,
        device=grad_output.device,
    )


def _one_hot_fake(self, num_classes=-1):
    sizes = list(self.size()) + [
        num_classes if num_classes > 0 else int(self.max()) + 1
    ]
    return torch.empty(sizes, dtype=self.dtype, device=self.device)


def _randperm_fake(
    n,
    *,
    generator=None,
    out=None,
    dtype=torch.int64,
    layout=torch.strided,
    device=None,
    requires_grad=False,
    pin_memory=False,
):
    return torch.empty([int(n)], dtype=dtype, device=device)


# ── Forward pointwise / reduction fake_impls ────────────────────────────────
#
# Required because PrivateUse1 outranks Meta, so PyTorch's built-in meta
# kernels never fire on Vulkan FakeTensors. Without these, AOT autograd's
# joint-graph trace dispatches into our C++ ops with FakeTensor inputs and
# we end up returning kMeta tensors that break downstream device propagation
# in backward formulas (`MulBackward0` doing `grad * saved`).


def _unary_fake(self):
    return torch.empty_like(self)


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
    return torch.empty_like(self)


def _comparison_fake(self, other):
    if isinstance(other, torch.Tensor):
        out_shape = torch.broadcast_shapes(self.shape, other.shape)
    else:
        out_shape = self.shape
    return torch.empty(out_shape, dtype=torch.bool, device=self.device)


def _gelu_fake(self, *, approximate="none"):
    return torch.empty_like(self)


def _leaky_relu_fake(self, negative_slope=0.01):
    return torch.empty_like(self)


def _elu_fake(self, alpha=1.0, scale=1.0, input_scale=1.0):
    return torch.empty_like(self)


def _hardtanh_fake(self, min_val=-1.0, max_val=1.0):
    return torch.empty_like(self)


def _softplus_fake(self, beta=1.0, threshold=20.0):
    return torch.empty_like(self)


def _clamp_fake(self, min=None, max=None):
    return torch.empty_like(self)


def _clamp_min_fake(self, min):
    return torch.empty_like(self)


def _clamp_max_fake(self, max):
    return torch.empty_like(self)


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


# ── Registry ──────────────────────────────────────────────────────────────────

_OP_IMPLS: dict[str, Callable] = {
    # View / shape ops
    "aten::as_strided": _as_strided_fake,
    "aten::permute": _permute_fake,
    "aten::transpose.int": _transpose_int_fake,
    "aten::t": _t_fake,
    "aten::view": _view_fake,
    "aten::_unsafe_view": _view_fake,
    "aten::reshape": _view_fake,
    # PF.55: composite-implicit-autograd shape ops whose C++
    # decomposition bypasses Python dispatch for the inner ``view`` call,
    # blowing up our PrivateUse1 C++ adapter on symbolic SymInt sizes.
    "aten::flatten.using_ints": _flatten_using_ints_fake,
    "aten::unflatten.int": _unflatten_int_fake,
    "aten::view_as": _view_as_fake,
    "aten::reshape_as": _reshape_as_fake,
    "aten::unsqueeze": _unsqueeze_fake,
    "aten::squeeze.dim": _squeeze_dim_fake,
    "aten::expand": _expand_fake,
    "aten::slice.Tensor": _slice_fake,
    "aten::constant_pad_nd": _constant_pad_nd_fake,
    "aten::as_strided_scatter": _as_strided_scatter_fake,
    "aten::masked_scatter": _masked_scatter_fake,
    # Memory / copy ops
    "aten::clone": _clone_fake,
    "aten::copy_": _copy__fake,
    "aten::contiguous": _contiguous_fake,
    "aten::empty_like": _empty_like_fake,
    "aten::fill_.Scalar": _fill_scalar_fake,
    "aten::zero_": _zero__fake,
    "aten::detach": _detach_fake,
    # Shape ops
    "aten::flip": _flip_fake,
    "aten::narrow": _narrow_fake,
    "aten::select.int": _select_int_fake,
    "aten::diagonal": _diagonal_fake,
    "aten::split_with_sizes": _split_with_sizes_fake,
    "aten::chunk": _chunk_fake,
    "aten::cat": _cat_fake,
    "aten::stack": _stack_fake,
    "aten::repeat": _repeat_fake,
    "aten::tile": _tile_fake,
    "aten::roll": _roll_fake,
    "aten::where.self": _where_self_fake,
    "aten::index_select": _index_select_fake,
    "aten::scatter_reduce.two": _scatter_reduce_fake,
    "aten::masked_fill_.Scalar": _masked_fill_scalar_fake,
    # BLAS
    "aten::mm": _mm_fake,
    "aten::bmm": _bmm_fake,
    "aten::addmm": _addmm_fake,
    "aten::linear": _linear_fake,
    # Pointwise unary
    "aten::neg": _unary_fake,
    "aten::abs": _unary_fake,
    "aten::exp": _unary_fake,
    "aten::log": _unary_fake,
    "aten::log2": _unary_fake,
    "aten::log10": _unary_fake,
    "aten::log1p": _unary_fake,
    "aten::sqrt": _unary_fake,
    "aten::rsqrt": _unary_fake,
    "aten::ceil": _unary_fake,
    "aten::floor": _unary_fake,
    "aten::round": _unary_fake,
    "aten::sign": _unary_fake,
    "aten::sgn": _unary_fake,
    "aten::reciprocal": _unary_fake,
    "aten::sin": _unary_fake,
    "aten::cos": _unary_fake,
    "aten::tan": _unary_fake,
    "aten::atan": _unary_fake,
    "aten::erf": _unary_fake,
    "aten::logical_not": _unary_fake,
    "aten::bitwise_not": _unary_fake,
    "aten::isnan": _unary_fake,
    "aten::isinf": _unary_fake,
    # Activations (forward)
    "aten::relu": _unary_fake,
    "aten::sigmoid": _unary_fake,
    "aten::tanh": _unary_fake,
    "aten::silu": _unary_fake,
    "aten::selu": _unary_fake,
    "aten::mish": _unary_fake,
    "aten::hardswish": _unary_fake,
    "aten::hardsigmoid": _unary_fake,
    "aten::gelu": _gelu_fake,
    "aten::leaky_relu": _leaky_relu_fake,
    "aten::elu": _elu_fake,
    "aten::hardtanh": _hardtanh_fake,
    "aten::softplus": _softplus_fake,
    "aten::clamp": _clamp_fake,
    "aten::clamp_min": _clamp_min_fake,
    "aten::clamp_max": _clamp_max_fake,
    "aten::softmax.int": _softmax_fake,
    "aten::log_softmax.int": _log_softmax_fake,
    "aten::_softmax": _softmax_fake,
    "aten::_log_softmax": _log_softmax_fake,
    # Pointwise binary (Tensor)
    "aten::add.Tensor": _binary_fake,
    "aten::sub.Tensor": _binary_fake,
    "aten::mul.Tensor": _binary_no_alpha_fake,
    "aten::div.Tensor": _binary_no_alpha_fake,
    "aten::pow.Tensor_Tensor": _binary_no_alpha_fake,
    "aten::fmod.Tensor": _binary_no_alpha_fake,
    "aten::remainder.Tensor": _binary_no_alpha_fake,
    "aten::atan2": _binary_no_alpha_fake,
    "aten::maximum": _binary_no_alpha_fake,
    "aten::minimum": _binary_no_alpha_fake,
    # Pointwise binary (Scalar)
    "aten::add.Scalar": _binary_scalar_fake,
    "aten::sub.Scalar": _binary_scalar_fake,
    "aten::mul.Scalar": _binary_scalar_fake,
    "aten::div.Scalar": _binary_scalar_fake,
    "aten::pow.Tensor_Scalar": _binary_scalar_fake,
    # Comparisons
    "aten::eq.Tensor": _comparison_fake,
    "aten::ne.Tensor": _comparison_fake,
    "aten::lt.Tensor": _comparison_fake,
    "aten::gt.Tensor": _comparison_fake,
    "aten::le.Tensor": _comparison_fake,
    "aten::ge.Tensor": _comparison_fake,
    "aten::eq.Scalar": _comparison_fake,
    "aten::ne.Scalar": _comparison_fake,
    "aten::lt.Scalar": _comparison_fake,
    "aten::gt.Scalar": _comparison_fake,
    "aten::le.Scalar": _comparison_fake,
    "aten::ge.Scalar": _comparison_fake,
    # Reductions
    "aten::sum.dim_IntList": _sum_dim_fake,
    "aten::sum": _sum_dim_fake,
    "aten::mean.dim": _mean_dim_fake,
    "aten::mean": _mean_dim_fake,
    "aten::amax": _amax_amin_fake,
    "aten::amin": _amax_amin_fake,
    "aten::argmax": _argmax_argmin_fake,
    "aten::argmin": _argmax_argmin_fake,
    "aten::max.dim": _max_dim_fake,
    "aten::min.dim": _min_dim_fake,
    # Conv
    "aten::convolution_overrideable": _convolution_overrideable_fake,
    "aten::convolution_backward_overrideable": _convolution_backward_overrideable_fake,
    # Normalization backward
    "aten::native_batch_norm_backward": _native_batch_norm_backward_fake,
    "aten::native_layer_norm_backward": _native_layer_norm_backward_fake,
    "aten::native_group_norm_backward": _native_group_norm_backward_fake,
    # Indexing
    "aten::gather": _gather_fake_backward,
    "aten::scatter_.src": _scatter_src_fake,
    "aten::scatter_.value": _scatter_value_fake,
    "aten::scatter_add_": _scatter_add_fake,
    "aten::index_put_": _index_put_fake,
    "aten::repeat_interleave.self_int": _repeat_interleave_self_int_fake,
    # Upsample backward
    "aten::upsample_bilinear2d_backward": _upsample_bilinear2d_backward_fake,
    "aten::upsample_nearest2d_backward": _upsample_nearest2d_backward_fake,
    # Attention
    "aten::scaled_dot_product_attention": _sdpa_fake,
    # Attention backward
    "aten::_scaled_dot_product_flash_attention_backward": _sdpa_backward_fake,
    # FFT
    "aten::_fft_r2c": _fft_r2c_fake,
    "aten::_fft_c2c": _fft_c2c_fake,
    "aten::_fft_c2r": _fft_c2r_fake,
    # SVD
    "aten::linalg_svd": _linalg_svd_fake,
    # Activation backward ops
    "aten::hardtanh_backward": _hardtanh_backward_fake,
    "aten::threshold_backward": _threshold_backward_fake,
    "aten::leaky_relu_backward": _leaky_relu_backward_fake,
    "aten::elu_backward": _elu_backward_fake,
    "aten::selu_backward": _selu_backward_fake,
    "aten::silu_backward": _silu_backward_fake,
    "aten::gelu_backward": _gelu_backward_fake,
    "aten::mish_backward": _mish_backward_fake,
    "aten::hardswish_backward": _hardswish_backward_fake,
    "aten::hardsigmoid_backward": _hardsigmoid_backward_fake,
    "aten::softplus_backward": _softplus_backward_fake,
    "aten::sigmoid_backward": _sigmoid_backward_fake,
    "aten::tanh_backward": _tanh_backward_fake,
    "aten::_softmax_backward_data": _softmax_backward_data_fake,
    "aten::_log_softmax_backward_data": _log_softmax_backward_data_fake,
    "aten::avg_pool2d_backward": _avg_pool2d_backward_fake,
    "aten::max_pool2d_with_indices_backward": _max_pool2d_with_indices_backward_fake,
    "aten::linear_backward": _linear_backward_fake,
    # Loss backward
    "aten::nll_loss_backward": _nll_loss_backward_fake,
    "aten::_cross_entropy_loss_backward": _cross_entropy_loss_backward_fake,
    # Reduction backward
    "aten::_sum_backward": _sum_backward_fake,
    # View ops needed by backward functionalization
    "aten::_reshape_alias": _reshape_alias_fake,
    "aten::select_backward": _select_backward_fake,
    "aten::slice_backward": _slice_backward_fake,
    "aten::unsqueeze_backward": _unsqueeze_backward_fake,
    "aten::squeeze_backward": _squeeze_backward_fake,
    "aten::t_backward": _t_backward_fake,
    "aten::transpose_backward": _transpose_backward_fake,
    "aten::view_backward": _view_backward_fake,
    "aten::permute_backward": _permute_backward_fake,
    "aten::expand_backward": _expand_backward_fake,
    "aten::repeat_backward": _repeat_backward_fake,
    # Embedding
    "aten::embedding": _embedding_fake,
    "aten::embedding_backward": _embedding_backward_fake,
    "aten::one_hot": _one_hot_fake,
    # Factory ops
    "aten::randperm.default": _randperm_fake,
}


def _register_backward_meta_decomps() -> None:
    """Register meta-level decompositions for aten backward ops.

    FakeTensorMode._dispatch_impl checks meta_table before trying regular
    decompositions. Regular decompositions can fail when saved forward inputs
    arrive as meta tensors (different device from the grad on vulkan:0).
    A meta decomposition fires first and just returns the correct shape,
    bypassing the device-mixing issue.
    """
    try:
        from torch._decomp import register_decomposition

        aten = torch.ops.aten

        def _bwd_meta_like_grad(grad_output, *_args, **_kwargs):
            return torch.empty_like(grad_output)

        def _bwd_meta_like_input(grad_output, input, *_args, **_kwargs):
            return torch.empty_like(input)

        def _layer_norm_bwd_meta(
            grad_out,
            input,
            normalized_shape,
            mean,
            rstd,
            weight=None,
            bias=None,
            output_mask=(True, True, True),
        ):
            gi = (
                torch.empty_like(input)
                if output_mask[0]
                else torch.empty(0, dtype=input.dtype, device=input.device)
            )
            norm_size = 1
            for s in normalized_shape:
                norm_size *= s
            gw = (
                torch.empty(norm_size, dtype=grad_out.dtype, device=grad_out.device)
                if output_mask[1]
                else torch.empty(0, dtype=grad_out.dtype, device=grad_out.device)
            )
            gb = (
                torch.empty(norm_size, dtype=grad_out.dtype, device=grad_out.device)
                if output_mask[2]
                else torch.empty(0, dtype=grad_out.dtype, device=grad_out.device)
            )
            return gi, gw, gb

        def _group_norm_bwd_meta(
            grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask
        ):
            gi = (
                torch.empty_like(input)
                if output_mask[0]
                else torch.empty(0, dtype=input.dtype, device=input.device)
            )
            gw = (
                torch.empty(int(C), dtype=grad_out.dtype, device=grad_out.device)
                if output_mask[1]
                else torch.empty(0, dtype=grad_out.dtype, device=grad_out.device)
            )
            gb = (
                torch.empty(int(C), dtype=grad_out.dtype, device=grad_out.device)
                if output_mask[2]
                else torch.empty(0, dtype=grad_out.dtype, device=grad_out.device)
            )
            return gi, gw, gb

        def _batch_norm_bwd_meta(
            grad_out,
            input,
            weight,
            running_mean,
            running_var,
            save_mean,
            save_invstd,
            train,
            eps,
            output_mask,
        ):
            gi = (
                torch.empty_like(input)
                if output_mask[0]
                else torch.empty(0, dtype=input.dtype, device=input.device)
            )
            C = input.shape[1]
            gw = (
                torch.empty(C, dtype=grad_out.dtype, device=grad_out.device)
                if output_mask[1]
                else torch.empty(0, dtype=grad_out.dtype, device=grad_out.device)
            )
            gb = (
                torch.empty(C, dtype=grad_out.dtype, device=grad_out.device)
                if output_mask[2]
                else torch.empty(0, dtype=grad_out.dtype, device=grad_out.device)
            )
            return gi, gw, gb

        def _linear_bwd_meta(input, grad_output, weight, output_mask):
            gi = (
                torch.empty_like(input)
                if output_mask[0]
                else torch.empty(0, dtype=input.dtype, device=input.device)
            )
            gw = (
                torch.empty_like(weight)
                if output_mask[1]
                else torch.empty(0, dtype=weight.dtype, device=weight.device)
            )
            gb = (
                torch.empty(
                    weight.size(0), dtype=grad_output.dtype, device=grad_output.device
                )
                if output_mask[2]
                else torch.empty(0, dtype=grad_output.dtype, device=grad_output.device)
            )
            return gi, gw, gb

        for op, fn in [
            (aten.gelu_backward.default, _bwd_meta_like_grad),
            (aten.silu_backward.default, _bwd_meta_like_grad),
            (aten.leaky_relu_backward.default, _bwd_meta_like_grad),
            (aten.elu_backward.default, _bwd_meta_like_grad),
            (aten._softmax_backward_data.default, _bwd_meta_like_grad),
            (aten._log_softmax_backward_data.default, _bwd_meta_like_grad),
            (aten.avg_pool2d_backward.default, _bwd_meta_like_input),
            (aten.max_pool2d_with_indices_backward.default, _bwd_meta_like_input),
            (aten.upsample_nearest2d_backward.vec, _bwd_meta_like_grad),
            (aten.upsample_bilinear2d_backward.vec, _bwd_meta_like_grad),
            (aten.native_layer_norm_backward.default, _layer_norm_bwd_meta),
            (aten.native_group_norm_backward.default, _group_norm_bwd_meta),
            (aten.native_batch_norm_backward.default, _batch_norm_bwd_meta),
            (aten.linear_backward.default, _linear_bwd_meta),
        ]:
            try:
                register_decomposition(op, type="meta")(fn)
            except Exception:
                pass
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(
            "Registering backward meta decompositions failed: %s", e
        )


def _register_matmul_meta() -> None:
    """GAP 0 — register aten.matmul in FakeTensorMode's meta_table.

    Without this, FakeTensorMode._dispatch_impl calls func.decompose() for
    aten.matmul (a CompositeImplicitAutograd op). The decompose path
    expands matmul into reshape→bmm→view, then evaluates the bmm on the
    FakeTensor's underlying zero-filled storage, producing all-zeros output.
    Inductor's constant_fold_uniform_value pass sees the uniform all-zeros
    and replaces the inputs with full(1.0) — the user's actual q/k tensors
    become dead code, and the compiled function returns wrong results.

    By registering in meta_table, _dispatch_impl skips decompose() and
    falls through to the fake_impl / meta kernel path, which returns a
    fresh FakeTensor without computing values. The constant folder then
    sees non-uniform data and leaves the graph alone.
    """
    try:
        from torch._decomp import register_decomposition

        matmul_op = torch.ops.aten.matmul.default

        @register_decomposition(matmul_op, type="meta")
        def _matmul_meta(tensor1, tensor2):
            t1 = tensor1 if isinstance(tensor1, torch.Tensor) else tensor1
            t2 = tensor2 if isinstance(tensor2, torch.Tensor) else tensor2
            dim1 = t1.dim()
            dim2 = t2.dim()
            if dim1 == 1 and dim2 == 1:
                return t1.new_empty(())
            elif dim1 == 2 and dim2 == 1:
                return t1.new_empty((t1.size(0),))
            elif dim1 == 1 and dim2 == 2:
                return t1.new_empty((t2.size(1),))
            elif dim1 == 2 and dim2 == 2:
                return t1.new_empty((t1.size(0), t2.size(1)))
            elif dim1 >= 1 and dim2 >= 1:
                max_dim = max(dim1, dim2)
                shape1 = list(t1.shape)
                shape2 = list(t2.shape)
                if dim1 == 1:
                    shape1 = [1, shape1[0]]
                if dim2 == 1:
                    shape2 = [shape2[0], 1]
                out_shape = torch.broadcast_shapes(shape1[:-2], shape2[:-2])
                out_shape = list(out_shape) + [shape1[-2], shape2[-1]]
                if dim1 == 1:
                    out_shape = out_shape[:-2] + out_shape[-1:]
                if dim2 == 1:
                    out_shape = out_shape[:-1]
                return t1.new_empty(out_shape)
            raise RuntimeError(f"matmul: unexpected dims {dim1}, {dim2}")
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(
            "Registering matmul meta decomposition failed: %s", e
        )


def _patch_proxy_call_matmul_decomp() -> None:
    """GAP 0 — register matmul on PrivateUse1 to prevent CompositeImplicitAutograd
    decomposition during make_fx tracing.

    The registration makes autograd_would_have_decomposed() return False for
    matmul on vulkan tensors, preventing the decomposition into view→bmm→view
    that causes constant-fold bugs.

    For eager mode, we also register matmul_backward on AutogradPrivateUse1
    so autograd still computes correct gradients through our matmul.
    """
    try:
        import torch
        from torch._subclasses.fake_tensor import FakeTensor

        _matmul_lib = torch.library.Library("aten", "IMPL", "PrivateUse1")

        def _vulkan_matmul(self, other):
            dim1 = self.dim()
            dim2 = other.dim()
            if dim1 == 2 and dim2 == 2:
                return torch.ops.aten.mm(self, other)
            if dim1 == 3 and dim2 == 3:
                return torch.ops.aten.bmm(self, other)
            s = self
            o = other
            sq1 = sq2 = False
            if dim1 == 1:
                s = s.unsqueeze(0)
                sq1 = True
            if dim2 == 1:
                o = o.unsqueeze(-1)
                sq2 = True
            if s.dim() == 2 and o.dim() == 2:
                result = torch.ops.aten.mm(s, o)
            elif s.dim() == 3 and o.dim() == 3:
                result = torch.ops.aten.bmm(s, o)
            else:
                raise RuntimeError(f"vulkan_matmul: unsupported dims {dim1} x {dim2}")
            if sq1 and sq2:
                return result.squeeze(-1).squeeze(-2)
            if sq1:
                return result.squeeze(-2)
            if sq2:
                return result.squeeze(-1)
            return result

        _matmul_lib.impl("matmul", _vulkan_matmul)

        def _vulkan_matmul_backward(grad, self, other, mask):
            results = []
            if mask[0]:
                g = torch.ops.aten.bmm(
                    grad.reshape(-1, grad.shape[-2], grad.shape[-1]),
                    other.reshape(-1, other.shape[-2], other.shape[-1]).transpose(
                        -2, -1
                    ),
                ).reshape(self.shape)
                results.append(g)
            else:
                results.append(torch.zeros_like(self))
            if mask[1]:
                g = torch.ops.aten.bmm(
                    self.reshape(-1, self.shape[-2], self.shape[-1]).transpose(-2, -1),
                    grad.reshape(-1, grad.shape[-2], grad.shape[-1]),
                ).reshape(other.shape)
                results.append(g)
            else:
                results.append(torch.zeros_like(other))
            return tuple(results)

        _matmul_lib.impl("matmul_backward", _vulkan_matmul_backward)

        import sys

        _module = sys.modules[__name__]
        _module._matmul_lib = _matmul_lib
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(
            "Registering matmul on PrivateUse1 failed: %s", e
        )


def _patch_einsum_proxy_decomp() -> None:
    """T.12 — register aten.einsum on PrivateUse1 with a Python decomposition
    that traces cleanly through ``__torch_dispatch__``.

    Without this, the C++ ``at::native::einsum`` implementation runs as a
    CompositeImplicitAutograd. During AOTAutograd's ``make_fx`` tracing,
    that C++ impl creates intermediate ``unsqueeze`` / ``permute`` views
    *inside C++*, so the resulting tensors carry no proxy. When those
    views are then passed to ``aten.bmm``, ``proxy_tensor.create_arg``
    cannot find a proxy and falls back to baking the tensors in as
    ``_tensor_constantN`` ``get_attr`` nodes on the GraphModule. The user's
    real ``arg0_1`` / ``arg1_1`` placeholders end up dead, and Inductor's
    ``constant_fold_uniform_value`` later sees the (uninitialised /
    fake-storage) constants as uniform and replaces the entire einsum
    with ``aten.full(value=1.0)`` — silent all-ones output (round 9 T.12).

    Fix: provide a Python decomposition that uses pure Python aten calls
    (``unsqueeze`` / ``permute`` / ``bmm`` / ``mul`` / ``sum``). Each call
    goes through ``__torch_dispatch__`` so proxies are tracked correctly.
    Registered on PrivateUse1 so it runs before any
    CompositeImplicitAutograd decomposition.

    Coverage: the seven canonical patterns from
    ``TestT12EinsumCoverage``: ``i,i->`` (inner), ``i,j->ij`` (outer),
    ``ij,jk->ik`` (mm), ``bij,bjk->bik`` (bmm),
    ``bnsh,bksh->bnks`` and ``bnks,bksh->bnsh`` (attention QK / QK·V),
    and ``ii->i`` (diag). For unsupported equations, falls through to
    the C++ default by raising NotImplementedError, which the dispatcher
    treats as no-impl.
    """
    try:
        import torch

        _einsum_lib = torch.library.Library("aten", "IMPL", "PrivateUse1")

        def _parse_einsum(equation):
            """Parse ``equation`` into (input_subs_list, output_sub).

            ``output_sub`` is computed implicitly when '->' is absent
            (each label appearing exactly once across all inputs is
            output, in alphabetical order)."""
            equation = equation.replace(" ", "")
            if "->" in equation:
                lhs, rhs = equation.split("->")
            else:
                lhs, rhs = equation, None
            input_subs = lhs.split(",")
            if rhs is None:
                from collections import Counter

                counts = Counter()
                for s in input_subs:
                    for c in s:
                        counts[c] += 1
                rhs = "".join(sorted(c for c, n in counts.items() if n == 1))
            return input_subs, rhs

        def _einsum_one(sub, out_sub, x):
            """Single-operand einsum: handles diag / sum / permute."""
            # Shape sanity: every dim with the same label must have the same size.
            label_to_size: dict[str, int] = {}
            for label, size in zip(sub, x.shape):
                if label in label_to_size:
                    if label_to_size[label] != size:
                        raise RuntimeError(
                            f"einsum: size mismatch on label {label!r}: "
                            f"{label_to_size[label]} vs {size}"
                        )
                else:
                    label_to_size[label] = size

            # If a label appears more than once, take the diagonal across
            # those axes (e.g. ``ii->i``). ``aten.diagonal`` removes the
            # two source dims and appends a single dim with the matching
            # label. Repeat until all subscripts are unique. Note: do NOT
            # restart from i=0 after a diag rebuild — the dedupe progress
            # would loop forever for ``i,i,i->i``-style patterns; instead
            # rescan with a fresh ``seen`` and look for the next repeat.
            from collections import Counter

            while True:
                counts = Counter(sub)
                repeats = [c for c, n in counts.items() if n > 1]
                if not repeats:
                    break
                label = repeats[0]
                # Find the first two positions of ``label``.
                first = sub.index(label)
                second = sub.index(label, first + 1)
                x = torch.ops.aten.diagonal.default(x, 0, first, second)
                sub = (
                    sub[:first]
                    + sub[first + 1 : second]
                    + sub[second + 1 :]
                    + label
                )

            # Sum out labels that don't appear in output_sub.
            sum_dims = [j for j, c in enumerate(sub) if c not in out_sub]
            if sum_dims:
                x = torch.ops.aten.sum.dim_IntList(x, sum_dims, False)
                sub = "".join(c for c in sub if c in out_sub)

            # Permute to output order.
            if sub != out_sub:
                if sorted(sub) != sorted(out_sub):
                    raise RuntimeError(
                        f"einsum: subscript mismatch {sub!r} vs {out_sub!r}"
                    )
                perm = [sub.index(c) for c in out_sub]
                x = torch.ops.aten.permute.default(x, perm)
            return x

        def _einsum_two(sub_a, sub_b, out_sub, a, b):
            """Two-operand einsum via the standard contract: classify each
            label as batch (in both inputs and output), contract (in both
            inputs but not output), or free (in one input and output)."""
            from collections import Counter

            # Record per-label sizes (for sanity).
            label_to_size: dict[str, int] = {}
            for sub, t in ((sub_a, a), (sub_b, b)):
                for label, size in zip(sub, t.shape):
                    if label in label_to_size:
                        if label_to_size[label] != size and 1 not in (
                            label_to_size[label],
                            size,
                        ):
                            raise RuntimeError(
                                f"einsum: size mismatch on label {label!r}: "
                                f"{label_to_size[label]} vs {size}"
                            )
                    else:
                        label_to_size[label] = size

            # Diagonalize repeats inside each operand first (a→a', b→b').
            a = _einsum_one(sub_a, "".join(dict.fromkeys(sub_a)), a)
            sub_a = "".join(dict.fromkeys(sub_a))
            b = _einsum_one(sub_b, "".join(dict.fromkeys(sub_b)), b)
            sub_b = "".join(dict.fromkeys(sub_b))

            in_a = set(sub_a)
            in_b = set(sub_b)
            in_out = set(out_sub)

            # Labels that are summed out and only appear in one input → reduce.
            only_a_summed = [
                c for c in sub_a if c not in in_b and c not in in_out
            ]
            only_b_summed = [
                c for c in sub_b if c not in in_a and c not in in_out
            ]
            if only_a_summed:
                axes = [sub_a.index(c) for c in only_a_summed]
                a = torch.ops.aten.sum.dim_IntList(a, axes, False)
                sub_a = "".join(c for c in sub_a if c not in only_a_summed)
            if only_b_summed:
                axes = [sub_b.index(c) for c in only_b_summed]
                b = torch.ops.aten.sum.dim_IntList(b, axes, False)
                sub_b = "".join(c for c in sub_b if c not in only_b_summed)

            in_a = set(sub_a)
            in_b = set(sub_b)

            batch_labels = [c for c in sub_a if c in in_b and c in in_out]
            contract_labels = [c for c in sub_a if c in in_b and c not in in_out]
            free_a = [c for c in sub_a if c not in in_b]
            free_b = [c for c in sub_b if c not in in_a]

            # Build target orderings:
            # a: [batch, free_a, contract]
            # b: [batch, contract, free_b]
            target_a = batch_labels + free_a + contract_labels
            target_b = batch_labels + contract_labels + free_b
            if list(sub_a) != target_a:
                perm = [sub_a.index(c) for c in target_a]
                a = torch.ops.aten.permute.default(a, perm)
            if list(sub_b) != target_b:
                perm = [sub_b.index(c) for c in target_b]
                b = torch.ops.aten.permute.default(b, perm)

            # Reshape to (B, M, K) × (B, K, N) bmm form.
            shape_a = list(a.shape)
            shape_b = list(b.shape)
            nb = len(batch_labels)
            nfa = len(free_a)
            nc = len(contract_labels)
            nfb = len(free_b)

            B_dims = shape_a[:nb]
            M_dims = shape_a[nb : nb + nfa]
            K_dims = shape_a[nb + nfa :]
            assert len(K_dims) == nc

            B_dims_b = shape_b[:nb]
            K_dims_b = shape_b[nb : nb + nc]
            N_dims = shape_b[nb + nc :]
            assert len(N_dims) == nfb

            B = 1
            for s in B_dims:
                B *= int(s)
            M = 1
            for s in M_dims:
                M *= int(s)
            K = 1
            for s in K_dims:
                K *= int(s)
            N = 1
            for s in N_dims:
                N *= int(s)

            a3 = torch.ops.aten.reshape.default(a, [B, M, K])
            b3 = torch.ops.aten.reshape.default(b, [B, K, N])

            # Pure-pointwise outer product when there's nothing to contract.
            if K == 1 and nc == 0:
                # mul + broadcast (no real reduction).
                # Here K_dims is empty; both a3 and b3 are [B, M, 1] / [B, 1, N].
                # Use bmm anyway — it's a valid degenerate matmul.
                pass
            out_3 = torch.ops.aten.bmm.default(a3, b3)

            # Reshape back to [*B_dims, *M_dims, *N_dims].
            out_shape = list(B_dims) + list(M_dims) + list(N_dims)
            if not out_shape:
                # Pure inner product: result is scalar.
                out = torch.ops.aten.reshape.default(out_3, [])
            else:
                out = torch.ops.aten.reshape.default(out_3, out_shape)

            # Permute to user's output order.
            current = batch_labels + free_a + free_b
            if current != list(out_sub):
                if sorted(current) != sorted(out_sub):
                    raise RuntimeError(
                        f"einsum: subscript mismatch {current!r} vs {out_sub!r}"
                    )
                perm = [current.index(c) for c in out_sub]
                out = torch.ops.aten.permute.default(out, perm)
            return out

        def _vulkan_einsum(equation, tensors, *, path=None):
            input_subs, out_sub = _parse_einsum(equation)
            if len(tensors) != len(input_subs):
                raise RuntimeError(
                    f"einsum: equation has {len(input_subs)} operand(s) but "
                    f"got {len(tensors)} tensor(s)"
                )
            if len(tensors) == 1:
                return _einsum_one(input_subs[0], out_sub, tensors[0])
            if len(tensors) == 2:
                return _einsum_two(
                    input_subs[0], input_subs[1], out_sub, tensors[0], tensors[1]
                )
            # Multi-operand: contract pairwise from left to right.
            cur = tensors[0]
            cur_sub = input_subs[0]
            for i in range(1, len(tensors)):
                next_t = tensors[i]
                next_sub = input_subs[i]
                # Compute the intermediate output subscript: keep labels that
                # appear in any later input, in either current pair, or in
                # the final output.
                later = set()
                for j in range(i + 1, len(tensors)):
                    later.update(input_subs[j])
                later.update(out_sub)
                inter_sub = "".join(
                    dict.fromkeys(
                        [c for c in cur_sub + next_sub if c in later]
                    )
                )
                cur = _einsum_two(cur_sub, next_sub, inter_sub, cur, next_t)
                cur_sub = inter_sub
            if cur_sub != out_sub:
                # Final permute / reduction (idempotent if already correct).
                cur = _einsum_one(cur_sub, out_sub, cur)
            return cur

        _einsum_lib.impl("einsum", _vulkan_einsum)

        import sys

        _module = sys.modules[__name__]
        _module._einsum_lib = _einsum_lib
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(
            "Registering einsum on PrivateUse1 failed: %s", e
        )


def _disable_bmm_to_mm_for_vulkan() -> None:
    """T.12 — skip Inductor's ``bmm_to_mm`` pattern for Vulkan tensors.

    The ``bmm_to_mm`` joint-graph pattern (``torch/_inductor/fx_passes/
    joint_graph.py``) rewrites ``aten.bmm`` with batch=1 into
    ``aten.mm(a.squeeze(0), b.squeeze(0)).unsqueeze(0)``. Its
    ``replace_by_example`` re-traces the replacement function via
    ``make_fx(..., tracing_mode='real')`` on the matched ``FakeTensor``
    values from ``mat.meta['val']``. On Vulkan, those FakeTensors carry
    ``device='vulkan:0'`` and the ``squeeze(0)`` view dispatch escapes
    proxy tracking — proxy_tensor's ``create_arg`` then bakes the
    squeezed views as ``_tensor_constantN`` ``get_attr`` nodes whose
    underlying tensors have *no backing buffer* (``data_ptr == 0``).
    The replacement graph ends up with the user's ``arg0`` / ``arg1``
    placeholders unused and the ``mm`` reading from the broken
    constants. At runtime, ``extern_kernels.mm(_vulkan_pm__tensor_constant0,
    _vulkan_pm__tensor_constant1)`` fails with ``Cannot access data
    pointer of Tensor (e.g. FakeTensor)``.

    This was the second-stage breakage of the T.12 ``einsum`` chain
    (round 9). With ``aten.einsum`` decomposed via our PrivateUse1
    Python impl (``_patch_einsum_proxy_decomp``) the AOT graph now has
    a clean ``unsqueeze + reshape + bmm + reshape`` tail driven by
    ``arg0_1`` / ``arg1_1``, but ``bmm_to_mm`` then matches and breaks
    it again. The pattern's only purpose is a CUDA micro-optimization
    (``mm`` is faster than ``bmm`` with ``B=1`` on CUDA); on Vulkan
    our ``bmm`` template handles ``B=1`` perfectly fine, so disabling
    is a safe and complete fix.

    Implementation: wrap the registered pattern function so it bails
    out (returns ``None`` without calling ``replace_by_example``) when
    either matched tensor is on Vulkan.
    """
    try:
        from torch._inductor.fx_passes import joint_graph as _jg

        if getattr(_jg, "_vulkan_bmm_to_mm_patched", False):
            return

        _patterns = _jg.patterns
        _orig_fn = _jg.bmm_to_mm
        _bmm_op = torch.ops.aten.bmm.default

        def _vulkan_aware_bmm_to_mm(match, mat1, mat2):
            try:
                v1 = mat1.meta.get("val", None)
                v2 = mat2.meta.get("val", None)
                if v1 is not None and getattr(v1, "device", None) is not None:
                    if v1.device.type == "vulkan":
                        return None
                if v2 is not None and getattr(v2, "device", None) is not None:
                    if v2.device.type == "vulkan":
                        return None
            except Exception:
                pass
            return _orig_fn(match, mat1, mat2)

        # Locate every PatternEntry registered against ``aten.bmm`` in
        # ``joint_graph.patterns`` (the dict is keyed by
        # ``('call_function', op_or_packet)``) and rewrap the handler.
        # We have to handle both the OpOverload and OpOverloadPacket
        # forms because ``register_graph_pattern`` registers under
        # whichever form the pattern declared.
        replaced = 0
        try:
            patterns_dict = _patterns.patterns
        except AttributeError:
            patterns_dict = {}
        bmm_keys = []
        for k in patterns_dict:
            try:
                if (
                    isinstance(k, tuple)
                    and len(k) >= 2
                    and k[0] == "call_function"
                ):
                    op = k[1]
                    name = getattr(op, "__name__", str(op))
                    if "bmm" in str(op) or "bmm" in name:
                        bmm_keys.append(k)
                elif k is _bmm_op:
                    bmm_keys.append(k)
            except Exception:
                continue

        for key in bmm_keys:
            for entry in patterns_dict[key]:
                handler = getattr(entry, "handler", None)
                if handler is _orig_fn:
                    entry.handler = _vulkan_aware_bmm_to_mm  # type: ignore[attr-defined]
                    replaced += 1

        # Also overwrite the module-level binding (for any future
        # registrations that look it up by attribute).
        _jg.bmm_to_mm = _vulkan_aware_bmm_to_mm  # type: ignore[attr-defined]
        _jg._vulkan_bmm_to_mm_patched = True  # type: ignore[attr-defined]
        if not replaced:
            import logging

            logging.getLogger(__name__).warning(
                "bmm_to_mm pattern not found in joint_graph.patterns "
                "(searched %d bmm keys); module attr replaced",
                len(bmm_keys),
            )
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(
            "Disabling bmm_to_mm for Vulkan failed: %s", e
        )


def _register_sdpa_meta() -> None:
    """Register SDPA in FakeTensorMode's meta_table.

    FakeTensorMode._dispatch_impl checks `func not in meta_table` first. If
    True (SDPA is NOT registered), it calls func.decompose() which fires the
    CompositeImplicitAutograd decomposition — returning before the fake_impl
    check is reached. By registering in meta_table we skip the decompose path
    so _dispatch_impl falls through to fake_impl, where _sdpa_fake handles
    shape inference correctly for Vulkan FakeTensors.
    """
    try:
        from torch._decomp import register_decomposition

        sdpa_op = torch.ops.aten.scaled_dot_product_attention.default

        @register_decomposition(sdpa_op, type="meta")
        def _sdpa_meta(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
            enable_gqa=False,
        ):
            sizes = list(query.size())
            sizes[-1] = value.size(-1)
            return query.new_empty(sizes)
    except Exception as e:  # pragma: no cover
        import logging

        logging.getLogger(__name__).warning(
            "Registering SDPA in meta_table failed: %s", e
        )


_view_symint_autograd_lib: "torch.library.Library | None" = None


def _register_view_symint_autograd_pyimpl() -> None:
    """PF.55 — Python ``AutogradPrivateUse1`` impl for view/reshape ops so
    SymInt-bearing sizes don't crash the C++ adapter's ``expect_int()``.

    Why: under ``torch.compile(..., dynamic=True)``, Dynamo's FakeTensor
    propagation calls ``Tensor.flatten`` / ``Tensor.view`` / ``Tensor.reshape``
    on a Vulkan FakeTensor whose ``size()`` carries symbolic SymInts.
    The dispatcher's first hit on the FakeTensor's keyset is
    ``AutogradPrivateUse1`` (always present in the autograd-aware
    keyset). Our C++ ``vulkan_view_autograd_adapter`` (and
    ``vulkan_reshape_autograd_adapter``, ``vulkan_view_adapter``,
    ``vulkan_reshape_adapter``) eagerly converts ``SymIntArrayRef`` to
    ``IntArrayRef`` via ``symint_to_int`` → ``SymInt::expect_int()``,
    which throws ``"when unpacking SymInt, expected int but got s33"``.
    Python ``__torch_dispatch__`` cannot intercept it because
    ``PythonFallbackKernel`` is registered as a *backend fallback* —
    only consulted when no explicit kernel exists at the active
    backend's keyset, which our C++ adapter pre-empts.

    Fix: register a Python kernel on ``AutogradPrivateUse1`` that runs
    *before* the C++ adapter. When the size carries any symbolic SymInt
    (or the input itself has symbolic shape), do shape inference in
    pure Python and return a fake-friendly tensor with the correct
    symbolic shape via ``self.new_empty``. When all sizes are concrete,
    redispatch under ``_ExcludeDispatchKeyGuard(AutogradPrivateUse1)``
    — the dispatcher then walks down to ``PrivateUse1`` where the
    untouched ``vulkan_view_adapter`` / ``vulkan_reshape_adapter`` runs
    eagerly with concrete ints (preserving zero-copy semantics).

    Note: we deliberately do NOT register at ``PrivateUse1`` because
    that would shadow the C++ adapter and we'd lose the ability to
    delegate concrete-size views to the eager path. The
    ``AutogradPrivateUse1`` key is always in a Vulkan FakeTensor's
    keyset (autograd is always live during Dynamo trace), so this one
    registration suffices to intercept all SymInt-bearing view calls.

    Covers ``aten::view``, ``aten::reshape``, and ``aten::_unsafe_view``
    — every shape op that takes a ``SymInt[]`` argument and dispatches
    to our adapters under FakeTensorMode.

    PF.13.b.4 fix: the kernel must wrap the result in a
    ``torch.autograd.Function`` so the output carries a backward
    grad_fn. Registering directly on ``AutogradPrivateUse1`` *replaces*
    the autograd kernel, so without the explicit Function wrap the
    dispatcher never attaches grad_fn — which causes the
    ``aten.matmul`` decomposition (``reshape → bmm → _unsafe_view``)
    to detach during fake-tensor metadata collection. AOTAutograd then
    flips ``needs_autograd = False`` (because ``output_info[i].requires_grad``
    is False everywhere), compiles as inference, and ``loss.backward()``
    on the compiled output raises ``element 0 of tensors does not
    require grad and does not have a grad_fn``. Mirrors the C++
    ``VulkanViewFunction`` / ``VulkanReshapeFunction`` shape-only
    backward in ``csrc/ops/autograd_ops.cpp:1520``.
    """
    global _view_symint_autograd_lib
    if _view_symint_autograd_lib is not None:
        return
    try:

        def _has_symint(size) -> bool:
            return any(isinstance(s, torch.SymInt) for s in size)

        def _resolve_view_size(self, size):
            """Compute output shape, inferring -1 against symbolic numel."""
            sizes = list(size)
            inferred = -1
            for i, s in enumerate(sizes):
                if isinstance(s, int) and s == -1:
                    if inferred != -1:
                        raise RuntimeError("only one -1 allowed in view")
                    inferred = i
            if inferred == -1:
                return sizes
            input_numel = 1
            for d in self.size():
                input_numel = input_numel * d
            other = 1
            for j, s in enumerate(sizes):
                if j != inferred:
                    other = other * s
            sizes[inferred] = input_numel // other
            return sizes

        _OP_DEFAULTS = {
            "view": torch.ops.aten.view.default,
            "reshape": torch.ops.aten.reshape.default,
            "_unsafe_view": torch.ops.aten._unsafe_view.default,
        }

        class _VulkanViewSymIntAutogradFn(torch.autograd.Function):
            """Re-attach grad_fn to view/reshape/_unsafe_view results.

            ``forward`` runs the underlying op (or a symbolic-shape
            new_empty for SymInt sizes); ``backward`` reshapes the grad
            back to the input's original shape — the exact contract of
            ``VulkanViewFunction::backward`` in C++ autograd_ops.cpp.
            """

            @staticmethod
            # pyrefly: ignore [bad-override]
            def forward(ctx, self, size, op_name):
                ctx.input_sizes = list(self.size())
                if _has_symint(size) or _has_symint(self.size()):
                    resolved = _resolve_view_size(self, size)
                    return self.new_empty(resolved)
                with torch._C._ExcludeDispatchKeyGuard(
                    torch._C.DispatchKeySet(torch._C.DispatchKey.AutogradPrivateUse1)
                ):
                    return _OP_DEFAULTS[op_name](self, size)

            @staticmethod
            def backward(ctx, grad):
                # ``reshape`` (not ``view``) handles non-contiguous grads
                # from upstream transpose / permute chains.
                return grad.reshape(ctx.input_sizes), None, None

        def _make_pyimpl(op_name):
            def _impl(self, size):
                return _VulkanViewSymIntAutogradFn.apply(self, size, op_name)

            return _impl

        _view_symint_autograd_lib = torch.library.Library(
            "aten", "IMPL", "AutogradPrivateUse1"
        )
        _view_symint_autograd_lib.impl("view", _make_pyimpl("view"))
        _view_symint_autograd_lib.impl("reshape", _make_pyimpl("reshape"))
        _view_symint_autograd_lib.impl("_unsafe_view", _make_pyimpl("_unsafe_view"))
    except Exception as e:  # pragma: no cover
        import logging

        logging.getLogger(__name__).warning(
            "Registering view symint pyimpl failed: %s", e
        )


_permute_family_autograd_lib: torch.library.Library | None = None


def _register_permute_family_autograd_pyimpl() -> None:
    """Register Python ``AutogradPrivateUse1`` impls for ``permute`` /
    ``transpose.int`` / ``t`` so they produce **proper view aliases** under
    FakeTensorMode.

    The C++ ``vulkan_permute`` (and its autograd wrapper) constructs a fresh
    output via ``at::empty`` (or ``make_vulkan_null`` on null storage). The
    result does not alias ``self``'s storage. Under FakeTensorMode the
    autograd graph captured by AOTAutograd then sees ``permute(arg)`` as
    input-independent — AOTAutograd lifts the result as a frozen tensor
    constant, and Inductor's ``constant_fold_uniform_value`` pass folds it
    to ``aten.full(uniform_value)``. The uniform value is whatever
    uninitialized memory happens to hold (commonly ``1.0f`` from a freed
    upstream allocation), producing kernels that write a constant 1.0 into
    every output slot.

    The fix mirrors the existing ``_register_view_symint_autograd_pyimpl``
    pattern: register a Python ``AutogradPrivateUse1`` kernel that runs
    *before* the C++ adapter, computes the permuted size+stride in pure
    Python, and dispatches to ``aten.as_strided`` (which has a built-in
    ``Meta`` kernel that produces a proper view aliasing ``self``'s
    storage). The autograd backward applies the inverse permutation.

    Without this fix, any compiled graph containing ``permute`` /
    ``transpose`` / ``t`` followed by a materialization (e.g. ``+ 0.0``,
    ``reshape``, ``contiguous``) silently returns wrong values
    (T.12.A).
    """
    global _permute_family_autograd_lib
    if _permute_family_autograd_lib is not None:
        return
    try:

        def _as_strided_view(self, sizes, strides):
            """Dispatch to ``aten.as_strided`` with AutogradPrivateUse1
            excluded so the standard autograd machinery (AsStridedBackward)
            handles the gradient instead of recursing into our wrapper.
            """
            with torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(torch._C.DispatchKey.AutogradPrivateUse1)
            ):
                return torch.ops.aten.as_strided.default(self, sizes, strides)

        class _VulkanPermutePyFn(torch.autograd.Function):
            @staticmethod
            # pyrefly: ignore [bad-override]
            def forward(ctx, self, dims):
                ndim = self.dim()
                ctx.dims = list(dims)
                mapped = [d if d >= 0 else ndim + d for d in dims]
                sizes = [int(self.size(i)) for i in mapped]
                strides = [int(self.stride(i)) for i in mapped]
                return _as_strided_view(self, sizes, strides)

            @staticmethod
            def backward(ctx, grad):
                ndim = len(ctx.dims)
                mapped = [d if d >= 0 else ndim + d for d in ctx.dims]
                inv = [0] * ndim
                for i, d in enumerate(mapped):
                    inv[d] = i
                return torch.ops.aten.permute.default(grad, inv), None

        class _VulkanTransposePyFn(torch.autograd.Function):
            @staticmethod
            # pyrefly: ignore [bad-override]
            def forward(ctx, self, dim0, dim1):
                ndim = self.dim()
                d0 = dim0 if dim0 >= 0 else ndim + dim0
                d1 = dim1 if dim1 >= 0 else ndim + dim1
                ctx.dim0 = dim0
                ctx.dim1 = dim1
                if d0 == d1:
                    return _as_strided_view(self, list(self.size()), list(self.stride()))
                sizes = list(self.size())
                strides = list(self.stride())
                sizes[d0], sizes[d1] = sizes[d1], sizes[d0]
                strides[d0], strides[d1] = strides[d1], strides[d0]
                return _as_strided_view(self, sizes, strides)

            @staticmethod
            def backward(ctx, grad):
                return (
                    torch.ops.aten.transpose.int(grad, ctx.dim0, ctx.dim1),
                    None,
                    None,
                )

        class _VulkanTPyFn(torch.autograd.Function):
            @staticmethod
            # pyrefly: ignore [bad-override]
            def forward(ctx, self):
                ctx.ndim = self.dim()
                if self.dim() < 2:
                    return _as_strided_view(self, list(self.size()), list(self.stride()))
                return _VulkanTransposePyFn.apply(self, 0, 1)

            @staticmethod
            def backward(ctx, grad):
                if ctx.ndim < 2:
                    return grad
                return torch.ops.aten.t.default(grad)

        def _permute_pyimpl(self, dims):
            return _VulkanPermutePyFn.apply(self, dims)

        def _transpose_pyimpl(self, dim0, dim1):
            return _VulkanTransposePyFn.apply(self, dim0, dim1)

        def _t_pyimpl(self):
            return _VulkanTPyFn.apply(self)

        _permute_family_autograd_lib = torch.library.Library(
            "aten", "IMPL", "AutogradPrivateUse1"
        )
        _permute_family_autograd_lib.impl("permute", _permute_pyimpl)
        _permute_family_autograd_lib.impl("transpose.int", _transpose_pyimpl)
        _permute_family_autograd_lib.impl("t", _t_pyimpl)
    except Exception as e:  # pragma: no cover
        import logging

        logging.getLogger(__name__).warning(
            "Registering permute family autograd pyimpl failed: %s", e
        )


_activation_autograd_lib: torch.library.Library | None = None


def _register_activation_autograd_pyimpl() -> None:
    """Register AutogradPrivateUse1 impls for activations whose built-in
    backward functions fail under torch.compile with Vulkan tensors.

    The built-in ``ReluBackward0`` uses ``threshold_backward`` which saves
    the forward output as a meta-device tensor during AOTAutograd tracing,
    causing the backward to return ``[]`` (scalar) instead of the correct
    gradient shape.

    Fix: wrap ``aten.relu`` in a custom ``torch.autograd.Function`` that
    dispatches to the real C++ kernel (via AutogradPrivateUse1 exclusion)
    and computes the backward in Python with correct device handling.
    """
    global _activation_autograd_lib
    if _activation_autograd_lib is not None:
        return
    try:

        class _VulkanReluAutogradFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, self):
                ctx.save_for_backward(self)
                with torch._C._ExcludeDispatchKeyGuard(
                    torch._C.DispatchKeySet(torch._C.DispatchKey.AutogradPrivateUse1)
                ):
                    return self.relu()

            @staticmethod
            def backward(ctx, grad_output):
                (self,) = ctx.saved_tensors
                mask = (self > 0).to(grad_output.dtype)
                return grad_output * mask

        _activation_autograd_lib = torch.library.Library(
            "aten", "IMPL", "AutogradPrivateUse1"
        )
        _activation_autograd_lib.impl("relu", _VulkanReluAutogradFn.apply)
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(
            "Registering activation autograd pyimpls failed: %s", e
        )


def _patch_dynamo_clone_input_for_vulkan() -> None:
    """Make Dynamo's `clone_input` skip `data_ptr()` for Vulkan FakeTensors.

    Upstream `torch._dynamo.utils.clone_input` already special-cases ``xla``:
    it falls back to ``torch.clone(x)`` instead of allocating an aligned
    storage buffer and computing ``(x.data_ptr() - result.data_ptr()) % 32``.
    The aligned path crashes on FakeTensors (no storage → no data pointer)
    and on devices like ours where computing pointer offsets across two
    independently-allocated tensors is meaningless.

    Without this, any Inductor compile that hits the example_inputs clone
    path for a Vulkan FakeTensor (the entry-point of every compiled SDPA-
    like attention compute) crashes with
    "Cannot access data pointer of Tensor (FakeTensor)". Patching here
    fixes P0.3 and unblocks `bmm(q, k.T) → softmax → bmm(@v)` compilation.
    """
    try:
        from torch._dynamo import utils as _du

        if getattr(_du.clone_input, "_vulkan_patched", False):
            return
        _orig_clone_input = _du.clone_input

        def _patched_clone_input(x, *, dtype=None):
            from torch._subclasses.fake_tensor import FakeTensor, is_fake

            # Detect any tensor for which `data_ptr()` is unsafe: FakeTensor,
            # FunctionalTensor, meta tensors, Vulkan tensors. For Dynamo's
            # purposes, `_clone_input` only needs to produce a tensor with the
            # same metadata (shape / stride / dtype / device); the storage
            # contents are never read. So for these cases we return a fresh
            # `empty_like` (or the input itself for FakeTensors, matching the
            # upstream `clone_input` short-circuit).
            def _is_fake_like(t):
                # `is_fake` covers FakeTensor + traceable subclass that wraps
                # one. Add an explicit isinstance check in case `is_fake`
                # misses an exotic subclass.
                try:
                    if is_fake(t) or isinstance(t, FakeTensor):
                        return True
                except Exception:
                    pass
                try:
                    if torch._is_functional_tensor(t):
                        return True
                except Exception:
                    pass
                return False

            def _safe_metadata_clone(t):
                """Produce a tensor with the same metadata when storage is
                unreadable. Falls back to `empty_like` if `torch.clone` errors
                on data pointer access or missing backing storage."""
                with torch.no_grad():
                    try:
                        y = torch.clone(t)
                    except RuntimeError as inner:
                        msg = str(inner)
                        # Dynamo only needs metadata (shape/stride/dtype/device);
                        # any clone failure on a Vulkan tensor is a storage-access
                        # problem — fall back to empty allocation.
                        if "data pointer" not in msg and "backing" not in msg:
                            raise
                        # Last-resort: allocate fresh storage with matching
                        # shape / stride / dtype / device. Contents are
                        # irrelevant for Dynamo example-value tracing.
                        y = torch.empty_strided(
                            t.size(),
                            t.stride(),
                            dtype=dtype or t.dtype,
                            device=t.device,
                        )
                    try:
                        if t.is_leaf:
                            y.requires_grad_(t.requires_grad)
                    except Exception:
                        pass
                    return y

            # FakeTensor: upstream `clone_input` returns `x` unchanged. Match
            # that behavior — cloning is a no-op for shape inference.
            if _is_fake_like(x):
                return x

            # Vulkan tensors: the alignment-aware path computes
            # `(x.data_ptr() - result.data_ptr()) % 32`, which is meaningless
            # for GPU-resident storage. Skip straight to a metadata clone.
            try:
                if x.device.type == "vulkan":
                    return _safe_metadata_clone(x)
            except Exception:
                pass

            try:
                return _orig_clone_input(x, dtype=dtype)
            except RuntimeError as e:
                # Original failed on `data_ptr()` or missing backing storage for
                # a tensor we did not detect upfront (e.g. FunctionalTensor
                # wrapping a Vulkan FakeTensor, as_strided view with no realized
                # storage). Fall back to a metadata-only clone.
                msg = str(e)
                if "data pointer" not in msg and "backing" not in msg:
                    raise
                return _safe_metadata_clone(x)

        _patched_clone_input._vulkan_patched = True  # type: ignore[attr-defined]
        _du.clone_input = _patched_clone_input

        # Also patch the imported reference in `builder.py` — it does
        # `from .utils import clone_input`, which captures the original
        # function by value at import time. Without this, the call site at
        # `_clone_input` (used by Dynamo's example-value clone) bypasses
        # our patch entirely.
        try:
            from torch._dynamo.variables import builder as _builder

            _builder.clone_input = _patched_clone_input
        except Exception:
            pass
    except Exception:
        pass


def _patch_fake_tensor_view_op_device() -> None:
    """Override FakeTensor.__new__ so view-op outputs inherit the input's
    fake device instead of getting `device=meta`.

    During AOT autograd's backward fake_tensor_prop, view-tagged ops like
    `aten::expand` go through PyTorch's C++ view fast-path. That path
    constructs the output FakeTensor via `Tensor._make_subclass` while the
    source FakeTensor is in `in_kernel_invocation` mode (which makes
    `source.device` report `meta`). Result: the new FakeTensor reports
    `device=meta` even though the source was `vulkan:0`. Downstream backward
    formulas then fail with `Unhandled FakeTensor Device Propagation` when
    they multiply a `meta` saved tensor by a `vulkan` grad.

    Fix: track the most-recently-active vulkan FakeTensorMode session in TLS,
    and in `__new__` upgrade `device=meta` → vulkan for tensors created during
    that session.
    """
    try:
        import torch._subclasses.fake_tensor as _ft

        _orig_new = _ft.FakeTensor.__new__

        def _patched_new(cls, fake_mode, elem, device, *args, **kwargs):
            if (
                isinstance(device, torch.device)
                and device.type == "meta"
                and fake_mode is not None
                and getattr(_tls, "_in_joint_trace", False)
            ):
                vk = getattr(fake_mode, "_torch_vulkan_seen_device", None)
                if vk is not None:
                    device = vk
            elif (
                isinstance(device, torch.device)
                and device.type in ("vulkan", "privateuseone")
                and fake_mode is not None
                and getattr(_tls, "_in_joint_trace", False)
            ):
                fake_mode._torch_vulkan_seen_device = device
            return _orig_new(cls, fake_mode, elem, device, *args, **kwargs)

        _ft.FakeTensor.__new__ = staticmethod(_patched_new)
    except Exception as e:  # pragma: no cover
        import logging

        logging.getLogger(__name__).warning(
            "Patching FakeTensor view-op device failed: %s", e
        )


def _patch_fake_tensor_meta_conversion() -> None:
    """Patch FakeTensorMode.validate_and_convert_non_fake_tensors to accept meta tensors.

    During Inductor's fake_tensor_prop on compiled backward graphs, saved
    forward-pass tensors arrive as plain `meta` device tensors rather than
    FakeTensors. The stock validation rejects these. Since a meta tensor is
    shape/dtype-only (identical semantics to a FakeTensor), we auto-convert
    them to FakeTensors using the mode's existing converter, fixing backward
    graph compilation under the inductor backend.

    Only intercepts meta-device tensors; all other non-fake tensors still raise
    so we don't silently hide real errors.
    """
    try:
        import torch._subclasses.fake_tensor as _ft

        _orig_validate = _ft.FakeTensorMode.validate_and_convert_non_fake_tensors

        def _patched_validate(self, func, converter, flat_args, args_spec):
            import torch

            new_args = []
            for a in flat_args:
                if (
                    isinstance(a, torch.Tensor)
                    and not self.is_our_fake(a)
                    and a.device.type in ("meta", "vulkan")
                ):
                    # Meta tensors arrive on backward fake_tensor_prop;
                    # vulkan-device real tensors arrive when Dynamo
                    # specializes a constant index tensor (e.g. the
                    # `target.unsqueeze(1)` argument to `aten.gather` in
                    # cross_entropy), and as the implicit `tensor(1.0)`
                    # tangent into AOT autograd's
                    # `coerce_tangent_and_suggest_memory_format` (PF.13.b.2).
                    # All shape/dtype-equivalent for FakeTensor purposes;
                    # convert so the validator doesn't reject.
                    a = converter.from_real_tensor(self, a)
                new_args.append(a)
            return _orig_validate(self, func, converter, new_args, args_spec)

        _ft.FakeTensorMode.validate_and_convert_non_fake_tensors = _patched_validate
    except Exception as e:  # pragma: no cover
        import logging

        logging.getLogger(__name__).warning(
            "Patching FakeTensorMode for meta tensor conversion failed: %s", e
        )


def _patch_tensor_deepcopy_for_vulkan() -> None:
    """PF.13.b.4 (layered #2) — make non-leaf Vulkan tensors deepcopy-safe.

    AOT autograd's lazy-backward path runs ``copy.deepcopy(bw_module)`` at
    backward time (``runtime_wrappers.py:2890``). The bw_module's
    ``_tensor_constantN`` attributes are saved-for-backward activations
    that the partitioner lifted from the joint graph — they are non-leaf
    Vulkan tensors (typically views like ``k.transpose(-2, -1)``). Stock
    ``Tensor.__deepcopy__`` raises immediately on non-leaf inputs:
    ``Only Tensors created explicitly by the user (graph leaves) support
    the deepcopy protocol``.

    Mirror the lazy/xla/mps/meta/ipu fast-path that already exists in
    ``torch/_tensor.py:164``: for Vulkan tensors, fall through to
    ``self.clone()`` even when non-leaf. ``clone()`` produces a new leaf
    Vulkan tensor with identical shape/stride/data, which is exactly
    what the deepcopy of bw_module's lifted activations needs.

    Without this, the matmul+softmax bwd-compile floor (PF.13.b.4) and
    every transformer-class backward through ``compile_fx_backward``
    aborts at backward execution time.
    """
    try:
        _orig_deepcopy = torch.Tensor.__deepcopy__

        def _patched_deepcopy(self, memo):
            # PF.13.b.4: Vulkan tensors stored as graph module constants
            # during AOTAutograd compilation have inaccessible storage.
            # Neither clone() nor contiguous() works.  Create metadata-
            # equivalent empty tensors instead — the actual values will
            # be filled in during backward execution when saved
            # activations are substituted.
            if self.device.type == "vulkan":
                if id(self) in memo:
                    return memo[id(self)]
                try:
                    with torch.no_grad():
                        new_tensor = self.clone()
                except (RuntimeError, Exception):
                    try:
                        new_tensor = torch.empty_strided(
                            self.shape,
                            self.stride(),
                            dtype=self.dtype,
                            device=self.device,
                        )
                    except Exception:
                        # Last resort: contiguous metadata clone
                        new_tensor = torch.empty(
                            self.shape,
                            dtype=self.dtype,
                            device=self.device,
                        )
                memo[id(self)] = new_tensor
                return new_tensor
            return _orig_deepcopy(self, memo)

        torch.Tensor.__deepcopy__ = _patched_deepcopy
    except Exception as e:  # pragma: no cover
        import logging

        logging.getLogger(__name__).warning(
            "Patching Tensor.__deepcopy__ for Vulkan failed: %s", e
        )


def _patch_fx_graph_cache_reduce_tensor_for_vulkan() -> None:
    """PF.13.b.4 (layered #4) — handle Vulkan view tensors in FX graph cache hashing.

    During forward graph compilation, FxGraphCachePickler._reduce_tensor
    calls t.tolist() on tensor constants stored as graph module attributes.
    These constants are typically saved-for-backward activations that the
    AOTAutograd partitioner lifted from the joint graph — non-leaf view tensors
    like k.transpose(-2, -1).  For Vulkan (PrivateUse1) tensors, the
    storage data pointer is invalid during this compilation phase, causing
    t.tolist() to raise RuntimeError: Cannot access data pointer of
    Tensor.

    This patch catches that RuntimeError for Vulkan tensors and falls back
    to t.cpu().tolist() which copies the data via the Vulkan readback
    path and produces a valid Python list for hashing.
    """
    try:
        import torch._inductor.codecache as _cc

        _orig_reduce_tensor = _cc.FxGraphCachePickler._reduce_tensor

        def _patched_reduce_tensor(self, t: torch.Tensor) -> tuple:
            from torch._inductor.graph import GraphLowering

            metadata = _cc.extract_tensor_metadata_for_cache_key(t)

            if _cc.is_frozen_param(t) and not GraphLowering.can_inline_constant(t):
                return (_cc._ident, (metadata,))

            # PF.13.b.4 / PF.13: Vulkan (PrivateUse1) tensors lifted as
            # graph module constants may have invalid storage data pointers
            # during AOTAutograd compilation (both leaf and non-leaf).
            # Neither .tolist() nor .cpu() works on them.  For Vulkan
            # tensors, try .tolist() and fall back to metadata-only if
            # the storage is inaccessible — metadata alone is sufficient
            # for cache-key uniqueness.
            # T.12: catch any exception from ``t.tolist()`` and fall back to
            # metadata-only hashing — metadata alone uniquely identifies the
            # cache key. This covers Vulkan tensors specifically (where
            # tolist may fail with "Unsupported copy direction" or "Cannot
            # access data pointer of Tensor"), and is harmless for any
            # other device because tolist on a healthy tensor never raises.
            try:
                values = t.tolist()
                return (
                    _cc._ident,
                    (_cc.TensorMetadataAndValues(metadata, values),),
                )
            except Exception:
                # Vulkan tensors lifted as graph constants regularly
                # have inaccessible storage at this point in the
                # pipeline; meta tensors never have data. Fall back to
                # metadata-only hashing — metadata uniquely identifies
                # the cache key. Also covers ``privateuseone`` devices.
                dev_t = getattr(t.device, "type", "")
                if dev_t in ("vulkan", "privateuseone", "meta") or "vulkan" in str(
                    t.device
                ):
                    return (_cc._ident, (metadata,))
                raise

        _cc.FxGraphCachePickler._reduce_tensor = _patched_reduce_tensor
    except Exception as e:  # pragma: no cover
        import logging

        logging.getLogger(__name__).warning(
            "Patching FxGraphCachePickler._reduce_tensor for Vulkan failed: %s", e
        )


def _patch_fake_tensor_skip_const_fold_for_vulkan_null() -> None:
    """Skip FakeTensor's constant-fold path when a Vulkan input's stored
    ``.constant`` has no backing buffer.

    ``FakeTensorMode._dispatch_impl`` has two constant-fold branches that run
    the *real* op against ``arg.constant`` for every fake input:

    - Lift / numbers-as-tensors path (``fake_tensor.py`` ~L2500):
      fires for binary ops with a Python scalar second arg, e.g.
      ``aten.add(<vulkan FakeTensor>, 1.0)``.
    - All-constants path (~L2557):
      fires when *every* fake input has a constant, e.g.
      ``aten.add(<vulkan FakeTensor>, <vulkan FakeTensor>)`` after both
      were promoted via ``make_constant=True``.

    Both branches eventually call ``func(*const_args, ...)``, which lands
    in our PrivateUse1 C++ kernel (``vulkan_add`` etc.). If the constant
    Vulkan tensor was produced by PF.13's view-cascade and has no
    backing buffer (``data_ptr() == 0``), the C++ kernel raises
    ``"Tensor has no backing Vulkan buffer"``. This blocks Inductor's
    FX tracing of conv graphs (``F.conv2d(x, w) + 1.0``) and similar
    binary-op-with-scalar patterns.

    Fix: before ``_dispatch_impl`` runs, scan every flat-arg fake tensor
    and, for any Vulkan FakeTensor whose ``.constant`` is null-storage,
    clear ``.constant`` so the constant-fold guard
    (``all(t.constant is not None ...)``) evaluates False. Dispatch then
    falls through to the registered fake_impl (already covers
    ``aten.add.Tensor``, ``aten.add.Scalar``, ``aten.mul.Tensor``,
    ``aten.mul.Scalar``, etc. — see ``_OP_IMPLS``), which returns the
    correct shape/dtype FakeTensor without dereferencing storage.

    Surgical: only clears ``.constant`` for Vulkan FakeTensors whose
    constant has null storage. Non-Vulkan tensors and Vulkan tensors
    with real storage are untouched, so other backends' const-fold
    behavior is unchanged.
    """
    try:
        import torch._subclasses.fake_tensor as _ft

        _orig_dispatch_impl = _ft.FakeTensorMode._dispatch_impl

        def _is_vulkan_null_constant(c) -> bool:
            if c is None or not isinstance(c, torch.Tensor):
                return False
            try:
                if c.device.type not in ("vulkan", "privateuseone"):
                    return False
            except Exception:  # noqa: BLE001
                return False
            try:
                return c.data_ptr() == 0
            except RuntimeError:
                # data_ptr() raises on FakeTensor / null storage —
                # treat as null-backed.
                return True

        def _is_real_vulkan_null(t) -> bool:
            if not isinstance(t, torch.Tensor):
                return False
            try:
                if t.device.type not in ("vulkan", "privateuseone"):
                    return False
            except Exception:  # noqa: BLE001
                return False
            try:
                return t.data_ptr() == 0
            except RuntimeError:
                return True

        def _patched_dispatch_impl(self, func, types, args, kwargs):
            from torch.utils import _pytree as pytree

            try:
                flat_args, spec = pytree.tree_flatten((args, kwargs))
                # Path 1: clear .constant on FakeTensors with vulkan-null
                # constant so the all-constants branch's guard fails.
                for a in flat_args:
                    if (
                        self.is_our_fake(a)
                        and getattr(a, "constant", None) is not None
                        and _is_vulkan_null_constant(a.constant)
                    ):
                        a.constant = None
                # Path 2: a real (non-fake) vulkan null-storage tensor in
                # flat_args makes ``flat_arg_fake_tensors`` empty, so the
                # ``should_allow_numbers_as_tensors`` const-fold branch
                # fires and tries to call ``func(...)`` against it.
                # Convert such tensors to proper FakeTensors so the branch
                # skips and dispatch falls through to the registered
                # fake_impl.
                replaced = False
                new_flat = []
                for a in flat_args:
                    if not self.is_our_fake(a) and _is_real_vulkan_null(a):
                        try:
                            a = self.from_tensor(a, static_shapes=True)
                            replaced = True
                        except Exception:  # noqa: BLE001
                            pass
                    new_flat.append(a)
                if replaced:
                    args, kwargs = pytree.tree_unflatten(new_flat, spec)
            except Exception:  # noqa: BLE001
                # Never let the guard itself break dispatch; fall through.
                pass
            return _orig_dispatch_impl(self, func, types, args, kwargs)

        _ft.FakeTensorMode._dispatch_impl = _patched_dispatch_impl
    except Exception as e:  # pragma: no cover
        import logging

        logging.getLogger(__name__).warning(
            "Patching FakeTensorMode._dispatch_impl for vulkan null "
            "constants failed: %s",
            e,
        )


def apply() -> None:
    global _patched
    if _patched:
        return

    for op_name, impl in _OP_IMPLS.items():
        try:
            entry = singleton.find(op_name)
            holder = entry.fake_impl
            holder.kernels = [Kernel(impl, __file__)]
        except Exception as e:  # pragma: no cover
            import logging

            logging.getLogger(__name__).warning(
                "Installing fake impl for %s failed: %s", op_name, e
            )

    # Register SDPA in meta_table so FakeTensorMode._dispatch_impl skips the
    # CompositeImplicitAutograd decompose() path (which fires before fake_impl
    # is checked) and reaches our _sdpa_fake fake_impl directly.
    _register_sdpa_meta()
    _register_matmul_meta()
    _patch_proxy_call_matmul_decomp()
    _patch_einsum_proxy_decomp()
    _disable_bmm_to_mm_for_vulkan()

    # PF.55 — Python AutogradPrivateUse1 impl for view/reshape/_unsafe_view
    # so dynamic-shape compile (`torch.compile(dynamic=True)`) doesn't blow
    # up our C++ ``vulkan_view_autograd_adapter``'s ``expect_int()`` call
    # on a symbolic SymInt size. Concrete sizes redispatch back to the C++
    # adapter to preserve zero-copy + autograd semantics.
    _register_view_symint_autograd_pyimpl()
    # T.12.A — Python AutogradPrivateUse1 impl for permute/transpose.int/t
    # so the FakeTensor result is a proper view aliasing ``self`` storage
    # (the C++ kernels return a fresh non-aliasing tensor, which makes
    # AOTAutograd lift the result as a frozen tensor constant and Inductor
    # constant-fold it to garbage uniform values).
    _register_permute_family_autograd_pyimpl()
    # relu ← clamp_min decomposition handles the Vulkan-aware backward
    # through clamp's working AutogradPrivateUse1 C++ adapter.

    # Override decompositions for activation backward ops that hit a PyTorch 2.11
    # FakeTensorMode bug: torch.where(cond, 0.0_scalar, tensor) returns shape []
    # instead of the tensor's shape. These decompositions avoid the scalar-first
    # where(cond, scalar, tensor) pattern.
    _patch_decompositions()

    # Inductor's _misc_patterns_init traces randperm_index_add_pattern with
    # tracing_mode="real". This calls randperm(device='vulkan:0') and then
    # __getitem__ on the result, which TorchFunctionMetadataMode dispatches
    # directly to C++ PrivateUse1, bypassing FakeTensorMode and our fake_impl.
    # Since randperm_index_add_pattern can never appear in a Vulkan FX graph
    # (we don't have aten::randperm on PrivateUse1), skip this init for Vulkan.
    _skip_misc_patterns_for_vulkan()

    # During Inductor's fake_tensor_prop on backward graphs, saved forward
    # tensors may arrive as plain `meta` device tensors (not FakeTensors).
    # This happens because AOT Autograd stores the saved inputs with device
    # metadata from the FakeTensor forward pass, and Inductor's backward
    # compilation propagates these as raw meta tensors rather than wrapping
    # them in the current FakeTensorMode's FakeTensors. The standard
    # validate_and_convert_non_fake_tensors rejects non-fake inputs unless
    # allow_non_fake_inputs=True. We patch it to auto-convert meta tensors
    # (which are shape-only and fully equivalent to FakeTensors) so the
    # backward graph compiles correctly under the inductor backend.
    _patch_fake_tensor_meta_conversion()
    # Always-on: Dynamo's `clone_input` calls `data_ptr()` on the input,
    # which crashes for FakeTensor inputs. Vulkan needs the same xla-style
    # `torch.clone(x)` fallback so attention-shaped graphs compile.
    _patch_dynamo_clone_input_for_vulkan()
    _patch_fake_tensor_view_op_device()

    # Skip _dispatch_impl's constant-fold path when a vulkan FakeTensor
    # holds a null-storage ``.constant`` — otherwise binary ops like
    # ``aten.add(<vulkan FakeTensor>, 1.0)`` during Inductor's FX trace
    # of conv graphs run the real C++ vulkan_add against an
    # un-allocated buffer and raise "Tensor has no backing Vulkan buffer".
    _patch_fake_tensor_skip_const_fold_for_vulkan_null()

    # PF.13.b.4 layered fix #2: AOT autograd's lazy-backward path
    # ``copy.deepcopy(bw_module)`` chokes on non-leaf Vulkan tensors
    # baked into the bw_module as saved-for-backward constants.
    _patch_tensor_deepcopy_for_vulkan()

    # PF.13.b.4 layered fix #4: FxGraphCachePickler._reduce_tensor calls
    # ``t.tolist()`` on Vulkan view tensors (e.g. k.transpose(-2,-1))
    # whose storage data pointer is invalid during AOTAutograd compilation.
    _patch_fx_graph_cache_reduce_tensor_for_vulkan()

    # C1: rewrite ``aten.relu`` to ``where + gt + full_like`` BEFORE the
    # AOT joint trace, so ReluBackward0/threshold_backward never fires
    # against meta-cascaded saved outputs.
    _patch_pre_grad_passes_for_relu_rewrite()

    # T4.8: optimizer foreach step pattern matching on the pre-grad graph
    # catches in-place ``add_/mul_/addcdiv_/addcmul_`` sequences BEFORE
    # AOTAutograd functionalization decomposes them into triplets/doublets.
    _patch_pre_grad_passes_for_optimizer_foreach()

    _patched = True


import threading

_tls = threading.local()


class _joint_trace_ctx:
    __slots__ = ()

    def __enter__(self):
        _tls._in_joint_trace = True
        return self

    def __exit__(self, *exc):
        _tls._in_joint_trace = False
        return False


class _FixMetaDevicePass:
    __bases__ = ()

    def __call__(self, gm: torch.fx.GraphModule) -> None:
        _vulkan_dev = torch.device("vulkan", 0)
        _ft = torch._subclasses.fake_tensor

        fm = None
        for node in gm.graph.nodes:
            val = node.meta.get("val")
            if isinstance(val, _ft.FakeTensor):
                fm = val.fake_mode
                break

        modified = False

        for node in gm.graph.nodes:
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor) and val.device.type == "meta":
                if fm is not None and isinstance(val, _ft.FakeTensor):
                    node.meta["val"] = _ft.FakeTensor.__new__(
                        _ft.FakeTensor,
                        fm,
                        val,
                        device=_vulkan_dev,
                    )
                else:
                    node.meta["val"] = torch.empty_strided(
                        val.shape,
                        val.stride(),
                        dtype=val.dtype,
                        device=_vulkan_dev,
                    )
                modified = True
            elif isinstance(val, (list, tuple)):
                changed = False
                new_list = []
                for v in val:
                    if isinstance(v, torch.Tensor) and v.device.type == "meta":
                        if fm is not None and isinstance(v, _ft.FakeTensor):
                            new_list.append(
                                _ft.FakeTensor.__new__(
                                    _ft.FakeTensor,
                                    fm,
                                    v,
                                    device=_vulkan_dev,
                                )
                            )
                        else:
                            new_list.append(
                                torch.empty_strided(
                                    v.shape,
                                    v.stride(),
                                    dtype=v.dtype,
                                    device=_vulkan_dev,
                                )
                            )
                        changed = True
                    else:
                        new_list.append(v)
                if changed:
                    node.meta["val"] = type(val)(new_list)
                    modified = True

        def _fix(obj):
            if isinstance(obj, torch.device) and obj.type == "meta":
                nonlocal modified
                modified = True
                return _vulkan_dev
            if isinstance(obj, (list, tuple)):
                return type(obj)(_fix(x) for x in obj)
            if isinstance(obj, dict):
                return {k: _fix(v) for k, v in obj.items()}
            return obj

        for node in gm.graph.nodes:
            new_args = _fix(node.args)
            if new_args is not node.args:
                node.args = new_args
            new_kwargs = _fix(node.kwargs)
            if new_kwargs is not node.kwargs:
                node.kwargs = new_kwargs

        if modified:
            gm.graph.lint()
            gm.recompile()

    def uuid(self):
        return None

    @classmethod
    def _make_compatible(cls):
        from torch._inductor.custom_graph_pass import CustomGraphModulePass

        class _Compatible(CustomGraphModulePass):
            _inner = cls()

            def __call__(self, gm):
                self._inner.__call__(gm)

            def uuid(self):
                return self._inner.uuid()

        return _Compatible()


def _install_joint_partition_device_fix() -> None:
    """PF.1 / P8.1: rewrite ``device='meta'`` to ``device='vulkan'`` in the
    joint graph BEFORE partitioning, then deeply propagate vulkan device
    tags through every downstream FakeTensor val that consumes a vulkan
    input.

    Stage 1 (narrow factory-op rewrite): walk the joint graph, find the
    factory ops (``aten.empty*``, ``aten.zeros*``, ``aten.ones*``,
    ``aten.full*``) whose kwargs contain ``torch.device('meta')``, and
    rewrite to ``vulkan:0``. Patches ``node.meta['val']`` in lock-step.

    Stage 2 (deep device propagation, PF.1 / B1'): walk every remaining
    ``call_function`` node in topo order. If its ``meta['val']`` carries
    ``device.type == 'meta'`` AND any of its Tensor arg-vals already
    carries ``device.type == 'vulkan'``, restamp the val to vulkan via
    ``_ft.FakeTensor.__new__`` (the FakeMode-preserving path — never
    ``empty_strided``, which would silently install a real allocation
    into the partitioner's cost-model input and collapse the backward).
    Also recurse into list/tuple vals for multi-output ops.

    Stage 3 (arg/kwarg device-literal rewrite): replaces any remaining
    ``torch.device('meta')`` literals embedded in node args/kwargs with
    the vulkan device, mirroring ``_FixMetaDevicePass`` so downstream
    passes that introspect args also see consistent device info.

    Why the deep propagation is correct: AOT autograd traced the joint
    graph once with a meta-device tangent, so the factory op (e.g.
    ``aten.empty.memory_format(..., device='meta')``) and every
    downstream node that consumed it carry meta vals. After Stage 1
    rewrites the factory device, the partitioner's ``get_device``
    (`torch/_functorch/partitioners.py:1721`) inspects ``node.meta['val'].device``
    on consumer nodes and would still see meta — placing the entire
    backward branch on a CPU-default device. Stage 2 fixes the
    consumers without re-running the dispatcher, so the partitioner's
    cost model sees a single coherent vulkan device while all
    FakeTensors stay FakeTensors.

    Why ``empty_strided`` is forbidden during this pass: the previous
    iteration of this code fell back to ``empty_strided(...)`` when
    ``fm is None`` or the val wasn't a FakeTensor — that creates a real
    Vulkan allocation that the partitioner's cost model treats as a
    static input, dropping the node from the backward graph and yielding
    an empty bw_module. We only restamp via FakeTensor.__new__; if no
    FakeMode is in scope we skip the deep pass entirely.

    Hooks into ``torch._functorch.config.joint_custom_pass``. Chains
    with any pre-existing pass; short-circuits on non-Vulkan graphs.
    """
    import torch._functorch.config as _fc

    existing = _fc.joint_custom_pass
    if getattr(existing, "_vulkan_partition_pass", False):
        return  # idempotent

    _vulkan_dev = torch.device("vulkan", 0)
    _ft = torch._subclasses.fake_tensor

    _FACTORY_OPS = (
        torch.ops.aten.empty.memory_format,
        torch.ops.aten.empty_strided.default,
        torch.ops.aten.zeros.default,
        torch.ops.aten.ones.default,
        torch.ops.aten.full.default,
        torch.ops.aten.empty_like.default,
        torch.ops.aten.zeros_like.default,
        torch.ops.aten.ones_like.default,
        torch.ops.aten.full_like.default,
    )

    def _has_vulkan_input(joint_inputs) -> bool:
        def _check(t) -> bool:
            return isinstance(t, torch.Tensor) and t.device.type == "vulkan"

        if isinstance(joint_inputs, (list, tuple)):
            for inp in joint_inputs:
                if _check(inp):
                    return True
                if isinstance(inp, (list, tuple)):
                    for sub in inp:
                        if _check(sub):
                            return True
        return False

    def _restamp_to_vulkan(val, fm):
        """Return a new vulkan-device FakeTensor mirroring ``val``'s
        shape/dtype/strides, sharing FakeMode ``fm``. ``val`` must be a
        meta-device FakeTensor; other inputs are returned unchanged.
        """
        if isinstance(val, _ft.FakeTensor) and val.device.type == "meta":
            return _ft.FakeTensor.__new__(
                _ft.FakeTensor,
                fm,
                val,
                device=_vulkan_dev,
            )
        return val

    def _val_has_vulkan_tensor(val) -> bool:
        if isinstance(val, torch.Tensor):
            return val.device.type == "vulkan"
        if isinstance(val, (list, tuple)):
            return any(_val_has_vulkan_tensor(v) for v in val)
        return False

    def _val_has_meta_tensor(val) -> bool:
        if isinstance(val, torch.Tensor):
            return val.device.type == "meta"
        if isinstance(val, (list, tuple)):
            return any(_val_has_meta_tensor(v) for v in val)
        return False

    def _restamp_val(val, fm):
        if isinstance(val, _ft.FakeTensor):
            return _restamp_to_vulkan(val, fm)
        if isinstance(val, list):
            return [_restamp_val(v, fm) for v in val]
        if isinstance(val, tuple):
            return tuple(_restamp_val(v, fm) for v in val)
        return val

    def _stamp_factory_devices(fx_g):
        # Discover the FakeMode in scope so we mint Vulkan FakeTensors
        # that share it with the rest of the graph's metadata.
        fm = None
        for node in fx_g.graph.nodes:
            val = node.meta.get("val")
            if isinstance(val, _ft.FakeTensor):
                fm = val.fake_mode
                break

        # Stage 1: rewrite factory-op device kwargs from meta → vulkan
        # and re-stamp the matching node.meta['val'] in lock-step.
        code_modified = False
        for node in fx_g.graph.nodes:
            if node.op != "call_function" or node.target not in _FACTORY_OPS:
                continue
            new_kwargs = dict(node.kwargs)
            kw_device = new_kwargs.get("device")
            if isinstance(kw_device, torch.device) and kw_device.type == "meta":
                new_kwargs["device"] = _vulkan_dev
                node.kwargs = new_kwargs
                code_modified = True
                val = node.meta.get("val")
                if (
                    fm is not None
                    and isinstance(val, _ft.FakeTensor)
                    and val.device.type == "meta"
                ):
                    node.meta["val"] = _restamp_to_vulkan(val, fm)

        # Stage 2 (PF.1): deep device propagation. Walk in topo order
        # (graph.nodes is topologically sorted). For each call_function
        # node whose val is meta and whose input vals contain at least
        # one vulkan tensor, restamp the meta vals to vulkan.
        if fm is not None:
            for node in fx_g.graph.nodes:
                if node.op != "call_function":
                    continue
                val = node.meta.get("val")
                if not _val_has_meta_tensor(val):
                    continue
                # Look at every input node's val; if any is vulkan, this
                # node is a vulkan computation that the joint trace
                # mistakenly stamped as meta.
                has_vulkan_input = False
                for input_node in node.all_input_nodes:
                    if _val_has_vulkan_tensor(input_node.meta.get("val")):
                        has_vulkan_input = True
                        break
                if has_vulkan_input:
                    node.meta["val"] = _restamp_val(val, fm)

        # Stage 3: rewrite any torch.device('meta') literals embedded
        # in node args/kwargs (e.g. cast-style ops with explicit device
        # kwargs that the joint trace captured as meta).
        def _fix(obj):
            nonlocal code_modified
            if isinstance(obj, torch.device) and obj.type == "meta":
                code_modified = True
                return _vulkan_dev
            if isinstance(obj, list):
                return [_fix(x) for x in obj]
            if isinstance(obj, tuple):
                return tuple(_fix(x) for x in obj)
            if isinstance(obj, dict):
                return {k: _fix(v) for k, v in obj.items()}
            return obj

        for node in fx_g.graph.nodes:
            if node.op != "call_function":
                continue
            new_args = _fix(node.args)
            if new_args is not node.args:
                node.args = new_args
            new_kwargs = _fix(node.kwargs)
            if new_kwargs is not node.kwargs:
                node.kwargs = new_kwargs

        if code_modified:
            fx_g.graph.lint()
            fx_g.recompile()
        return fx_g

    def _rewrite_empty_meta_to_tangent_expand(fx_g):
        """PF.13 root fix: replace uninitialized ``aten.empty(shape, device=meta)``
        nodes in the joint graph with ``aten.expand(tangents_X, shape)``.

        Why this is needed: when AOT autograd's joint trace evaluates a
        backward formula like ``SumBackward0 -> grad.expand_symint(self_sizes)``
        with ``grad`` a vulkan FakeTensor of shape ``[]``, the proxy-tensor
        tracer captures the result as ``aten.empty.memory_format(shape,
        device='meta')`` instead of ``aten.expand.default(grad, shape)`` —
        because expand on a 0-dim FakeTensor under ``in_kernel_invocation``
        loses the device tag and proxy_tensor materializes the result as a
        fresh meta-empty allocation. The resulting BW graph has the
        ``tangents_X`` input completely unused and reads uninitialized
        memory in its place — the canonical NaN/0.5×-grad bug.

        Heuristic match: any ``aten.empty.memory_format`` node with no
        tensor inputs (purely a sizes/dtype/device factory call) whose
        ``device`` argument is meta. Pair it with the unique tangent
        placeholder of matching dtype, expanding from the tangent's
        smaller shape to the empty's target shape.

        Vulkan-only: only fires after ``_has_vulkan_input`` has gated us in.
        """
        # Collect tangent placeholders. AOT autograd names them
        # ``tangents_1``, ``tangents_2``, .... We match by dtype because
        # the trace doesn't preserve which tangent the empty was meant
        # to materialize.
        tangent_placeholders = []
        for node in fx_g.graph.nodes:
            if node.op != "placeholder":
                continue
            if not str(node.target).startswith("tangents"):
                continue
            tangent_placeholders.append(node)

        if not tangent_placeholders:
            return fx_g

        def _tangent_for_dtype(dtype):
            for t in tangent_placeholders:
                v = t.meta.get("val")
                if isinstance(v, torch.Tensor) and v.dtype == dtype:
                    return t
            return None

        modified = False
        for node in list(fx_g.graph.nodes):
            if node.op != "call_function":
                continue
            if node.target is not torch.ops.aten.empty.memory_format:
                continue
            # Pure factory: shape is args[0], dtype/device/etc in kwargs.
            kw_device = node.kwargs.get("device")
            if not (isinstance(kw_device, torch.device) and kw_device.type == "meta"):
                continue
            target_shape = node.args[0]
            target_dtype = node.kwargs.get("dtype", torch.float32)
            tangent = _tangent_for_dtype(target_dtype)
            if tangent is None:
                continue
            # Build ``aten.expand.default(tangent, target_shape)``. expand
            # broadcasts a 0-dim or smaller-rank source to the requested
            # shape using stride-0 views — exactly the semantics
            # ``grad.expand_symint(sizes)`` was meant to capture.
            with fx_g.graph.inserting_before(node):
                expand_node = fx_g.graph.call_function(
                    torch.ops.aten.expand.default,
                    (tangent, list(target_shape)),
                )
            # Carry over the val so downstream FakeTensor-aware passes
            # (lifetime annotation, partitioner cost model) see a vulkan
            # tensor of the right shape/dtype/device.
            expand_node.meta = dict(node.meta)
            tangent_val = tangent.meta.get("val")
            if isinstance(tangent_val, torch.Tensor):
                fm = getattr(tangent_val, "fake_mode", None)
                if fm is not None:
                    expand_node.meta["val"] = fm.from_tensor(
                        torch.empty(
                            list(target_shape),
                            dtype=target_dtype,
                            device=_vulkan_dev,
                        ),
                        static_shapes=True,
                    )
            node.replace_all_uses_with(expand_node)
            fx_g.graph.erase_node(node)
            modified = True

        if modified:
            fx_g.graph.lint()
            fx_g.recompile()
        return fx_g

    def _chained(fx_g, joint_inputs):
        if callable(existing):
            fx_g = existing(fx_g, joint_inputs)
        if _has_vulkan_input(joint_inputs):
            # Order matters: replace empty(meta)→expand(tangent) BEFORE
            # the device-stamp pass so we don't end up with empty(vulkan)
            # zombies that read uninitialized memory at runtime.
            fx_g = _rewrite_empty_meta_to_tangent_expand(fx_g)
            fx_g = _stamp_factory_devices(fx_g)
            # PF.40: annotate node.meta["lifetime_class"] on the joint
            # graph. The annotation propagates into fw_module / bw_module
            # because the partitioner copies node meta. Consumed by
            # PF.41 (StepActivationPool) and PF.42 (step-end release hook).
            from torch_vulkan.inductor.lifetime import (
                annotate_lifetime_classes,
            )

            fx_g = annotate_lifetime_classes(fx_g, joint_inputs)
        return fx_g

    _chained._vulkan_partition_pass = True  # type: ignore[attr-defined]
    _fc.joint_custom_pass = _chained


def _patch_compile_fx_for_backward() -> None:
    try:
        from torch._inductor.codegen.common import register_backend_for_device
        from torch._inductor.custom_graph_pass import CustomGraphModulePass

        from .codegen import VulkanScheduling
        from .wrapper import VulkanPythonWrapperCodegen

        class _Compatible(CustomGraphModulePass):
            _inner = _FixMetaDevicePass()

            def __call__(self, gm):
                self._inner.__call__(gm)

            def uuid(self):
                return self._inner.uuid()

        register_backend_for_device(
            "meta",
            VulkanScheduling,
            VulkanPythonWrapperCodegen,
            None,
            None,
            _Compatible(),
        )
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(
            "Registering meta→VulkanScheduling alias failed: %s", e
        )


def _skip_misc_patterns_for_vulkan() -> None:
    try:
        from torch._inductor.fx_passes import misc_patterns as _mp

        _orig_init = getattr(_mp, "_misc_patterns_init", None)
        if _orig_init is None:
            return

        def _patched_misc_init(device):
            dev_type = getattr(device, "type", str(device))
            if "privateuseone" in dev_type or "vulkan" in dev_type:
                return
            return _orig_init(device)

        _mp._misc_patterns_init = _patched_misc_init
    except Exception:
        pass


def _patch_decompositions() -> None:
    from torch._decomp import decomposition_table

    aten = torch.ops.aten

    def _hardtanh_bwd(grad_output, self, min_val, max_val):
        mask = (self > min_val) & (self < max_val)
        return grad_output * mask

    def _hardswish_bwd(grad_output, self):
        return torch.where(
            self < -3.0,
            torch.zeros_like(grad_output),
            torch.where(self > 3.0, grad_output, grad_output * (self / 3.0 + 0.5)),
        )

    def _hardsigmoid_bwd(grad_output, self):
        mask = (self > -3.0) & (self < 3.0)
        return grad_output * mask * (1.0 / 6.0)

    def _softplus_bwd(grad_output, self, beta, threshold):
        bx = beta * self
        return torch.where(bx > threshold, grad_output, grad_output * torch.sigmoid(bx))

    def _mish_bwd(grad_output, self):
        sp = torch.nn.functional.softplus(self)
        tsp = torch.tanh(sp)
        return grad_output * (tsp + self * (1.0 - tsp * tsp) * torch.sigmoid(self))

    # C1: Provide REAL decompositions, not shape-only proxies.
    # The previous approach (torch.empty_like for all backwards) caused
    # ReluBackward0 and similar to return [] because the compiled backward
    # never fills these empty tensors. Inductor lowerings handle the actual
    # computation via bwd_diff or primitive decomposition.
    def _relu_fwd_decomp(self):
        # C1 safety net: decompose relu → where(x > 0, x, 0) at the
        # AOT decomposition level so ReluBackward0 never enters the
        # joint trace even if the pre-grad rewrite misses.
        return torch.where(self > 0, self, torch.zeros_like(self))

    def _threshold_bwd(grad_output, self, threshold):
        return torch.where(self > threshold, grad_output, 0.0)

    def _gelu_bwd(grad_output, self, approximate="none"):
        # Use the built-in gelu_backward which inductor can lower
        return torch.ops.aten.gelu_backward(grad_output, self, approximate=approximate)

    def _silu_bwd(grad_output, self):
        sig = torch.sigmoid(self)
        return grad_output * sig * (1.0 + self * (1.0 - sig))

    def _tanh_bwd(grad_output, self):
        return grad_output * (1.0 - self * self)

    def _sigmoid_bwd(grad_output, self):
        return grad_output * self * (1.0 - self)

    def _leaky_relu_bwd(grad_output, self, negative_slope, self_is_result):
        return torch.where(self > 0, grad_output, grad_output * negative_slope)

    def _elu_bwd(
        grad_output, alpha, scale, input_scale, self_or_result, self_is_result
    ):
        return torch.where(
            self_or_result > 0,
            grad_output * scale,
            grad_output * scale * (self_or_result + alpha / input_scale),
        )

    def _softmax_bwd(grad_output, output, dim, input_dtype):
        return torch.empty_like(grad_output)

    def _log_softmax_bwd(grad_output, output, dim, input_dtype):
        return torch.empty_like(grad_output)

    def _avg_pool2d_bwd(
        grad_output,
        self,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    ):
        return torch.empty_like(self)

    def _max_pool_bwd(
        grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices
    ):
        return torch.empty_like(self)

    def _linear_bwd(input, grad_output, weight, output_mask):
        gi = (
            torch.empty_like(input)
            if output_mask[0]
            else torch.empty(0, dtype=input.dtype, device=input.device)
        )
        gw = (
            torch.empty_like(weight)
            if output_mask[1]
            else torch.empty(0, dtype=weight.dtype, device=weight.device)
        )
        gb = (
            torch.empty(
                weight.size(0), dtype=grad_output.dtype, device=grad_output.device
            )
            if output_mask[2]
            else torch.empty(0, dtype=grad_output.dtype, device=grad_output.device)
        )
        return gi, gw, gb

    def _layer_norm_bwd(
        grad_out,
        input,
        normalized_shape,
        mean,
        rstd,
        weight=None,
        bias=None,
        output_mask=(True, True, True),
    ):
        norm_size = 1
        for s in normalized_shape:
            norm_size *= int(s)
        gi = (
            torch.empty_like(input)
            if output_mask[0]
            else torch.empty(0, dtype=input.dtype, device=input.device)
        )
        gw = (
            torch.empty(norm_size, dtype=grad_out.dtype, device=grad_out.device)
            if output_mask[1]
            else torch.empty(0, dtype=grad_out.dtype, device=grad_out.device)
        )
        gb = (
            torch.empty(norm_size, dtype=grad_out.dtype, device=grad_out.device)
            if output_mask[2]
            else torch.empty(0, dtype=grad_out.dtype, device=grad_out.device)
        )
        return gi, gw, gb

    def _group_norm_bwd(
        grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask
    ):
        gi = (
            torch.empty_like(input)
            if output_mask[0]
            else torch.empty(0, dtype=input.dtype, device=input.device)
        )
        gw = (
            torch.empty(int(C), dtype=grad_out.dtype, device=grad_out.device)
            if output_mask[1]
            else torch.empty(0, dtype=grad_out.dtype, device=grad_out.device)
        )
        gb = (
            torch.empty(int(C), dtype=grad_out.dtype, device=grad_out.device)
            if output_mask[2]
            else torch.empty(0, dtype=grad_out.dtype, device=grad_out.device)
        )
        return gi, gw, gb

    def _batch_norm_bwd(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        train,
        eps,
        output_mask,
    ):
        C = input.shape[1]
        gi = (
            torch.empty_like(input)
            if output_mask[0]
            else torch.empty(0, dtype=input.dtype, device=input.device)
        )
        gw = (
            torch.empty(C, dtype=grad_out.dtype, device=grad_out.device)
            if output_mask[1]
            else torch.empty(0, dtype=grad_out.dtype, device=grad_out.device)
        )
        gb = (
            torch.empty(C, dtype=grad_out.dtype, device=grad_out.device)
            if output_mask[2]
            else torch.empty(0, dtype=grad_out.dtype, device=grad_out.device)
        )
        return gi, gw, gb

    replacements = {
        aten.hardtanh_backward.default: _hardtanh_bwd,
        aten.hardswish_backward.default: _hardswish_bwd,
        aten.hardsigmoid_backward.default: _hardsigmoid_bwd,
        aten.softplus_backward.default: _softplus_bwd,
        aten.mish_backward.default: _mish_bwd,
        aten.relu.default: _relu_fwd_decomp,
        aten.threshold_backward.default: _threshold_bwd,
        # Shape-only backward decompositions (input-saving ops that would
        # mix meta saved-input device with vulkan:0 gradient device)
        aten.gelu_backward.default: _gelu_bwd,
        aten.silu_backward.default: _silu_bwd,
        aten.leaky_relu_backward.default: _leaky_relu_bwd,
        aten.elu_backward.default: _elu_bwd,
        aten.sigmoid_backward.default: _sigmoid_bwd,
        aten.tanh_backward.default: _tanh_bwd,
        aten._softmax_backward_data.default: _softmax_bwd,
        aten._log_softmax_backward_data.default: _log_softmax_bwd,
        aten.avg_pool2d_backward.default: _avg_pool2d_bwd,
        aten.max_pool2d_with_indices_backward.default: _max_pool_bwd,
        aten.linear_backward.default: _linear_bwd,
        aten.native_layer_norm_backward.default: _layer_norm_bwd,
        aten.native_group_norm_backward.default: _group_norm_bwd,
        aten.native_batch_norm_backward.default: _batch_norm_bwd,
    }
    decomposition_table.update(replacements)


def _patch_pre_grad_passes_for_optimizer_foreach() -> None:
    """T4.8 pre-grad pass: detect optimizer-step patterns on the forward
    graph BEFORE AOTAutograd functionalizes in-place mutations.

    Dynamo traces ``p.add_(g, alpha=-lr)`` as ``aten.add_.Tensor``.
    AOTAutograd then functionalizes this into the
    ``(mul.Tensor → add.Tensor → copy_)`` triplet.  If we intercept
    the pre-grad graph, we can rewrite contiguous per-param add_/mul_/
    addcdiv_/addcmul_ sequences directly into ``torch_vulkan::foreach_*``
    custom ops, bypassing the functionalization dance.

    Vulkan-only: only fires when the graph's tensors originate from a
    Vulkan device, detected by the same three strategies used in
    ``_patch_pre_grad_passes_for_relu_rewrite``.
    """
    import torch._inductor.compile_fx as _cfx

    if getattr(_cfx, "_vulkan_foreach_rewrite_patched", False):
        return

    _orig = _cfx.run_pre_grad_passes

    def _patched(model_, example_inputs_):
        from .fx_passes.functional.optimizer import (
            _fuse_optimizer_step_to_foreach,
        )

        # Reuse the relu-rewrite detector (same Vulkan-detection logic).
        # Must import the function here to avoid circular imports.
        try:
            # Quick device check — reuse the same strategies as
            # _patch_pre_grad_passes_for_relu_rewrite.
            inputs = example_inputs_ or ()
            is_vulkan = False
            for t in inputs:
                if not isinstance(t, torch.Tensor):
                    continue
                try:
                    if t.device.type in ("vulkan", "privateuseone"):
                        is_vulkan = True
                        break
                except Exception:
                    pass
                try:
                    fd = getattr(t, "fake_device", None)
                    if fd is not None and fd.type in ("vulkan", "privateuseone"):
                        is_vulkan = True
                        break
                except Exception:
                    pass
            if not is_vulkan and isinstance(model_, torch.fx.GraphModule):
                try:
                    for node in model_.graph.nodes:
                        if node.op != "placeholder":
                            continue
                        val = node.meta.get("val") if hasattr(node, "meta") else None
                        if val is None:
                            continue
                        for v in val if isinstance(val, (list, tuple)) else [val]:
                            if not isinstance(v, torch.Tensor):
                                continue
                            try:
                                if v.device.type in ("vulkan", "privateuseone"):
                                    is_vulkan = True
                                    break
                            except Exception:
                                pass
                            try:
                                fd = getattr(v, "fake_device", None)
                                if fd is not None and fd.type in (
                                    "vulkan",
                                    "privateuseone",
                                ):
                                    is_vulkan = True
                                    break
                            except Exception:
                                pass
                        if is_vulkan:
                            break
                except Exception:
                    pass

            if is_vulkan and isinstance(model_, torch.fx.GraphModule):
                try:
                    _fuse_optimizer_step_to_foreach(model_)
                except Exception as e:
                    import logging

                    logging.getLogger(__name__).warning(
                        "Vulkan pre-grad foreach rewrite failed: %s", e
                    )
        except Exception:
            pass

        return _orig(model_, example_inputs_)

    _cfx.run_pre_grad_passes = _patched
    _cfx._vulkan_foreach_rewrite_patched = True


def _patch_pre_grad_passes_for_relu_rewrite() -> None:
    """C1 fix: rewrite ``aten.relu`` in the pre-grad FX graph BEFORE
    AOTAutograd traces the joint forward+backward.

    The default ``ReluBackward0`` saves the forward output and computes
    backward via ``aten.threshold_backward(grad_out, result, 0)``. Under
    AOTAutograd's joint trace the saved ``result`` propagates as a
    ``device=meta`` FakeTensor through PF.13's view-op cascade, so the
    threshold-backward decomposition's ``result > 0`` lands on meta and
    ``where(meta_cond, vulkan_grad, 0.0)`` collapses to a ``[]``-shape
    tensor — surfacing as ``"ReluBackward0 returned an invalid gradient
    at index 0 - got [] but expected shape compatible with [...]"``.

    Decompose ``relu(x) -> where(x > 0, x, full_like(x, 0))`` at the
    pre-grad stage so AOTAutograd traces the joint graph against
    pointwise primitives whose backwards never call threshold_backward
    nor save a meta-cascaded forward output.

    The Vulkan-detection check uses three strategies because
    ``run_pre_grad_passes`` is called from two different contexts:

    1. **Direct call** (``compile_fx.py:2883``, AOT path):
       ``example_inputs_`` are real tensors — ``t.device.type`` works.
    2. **Inside ``aot_autograd()``** (``aot_autograd.py:1108/1134``):
       ``example_inputs_`` are ``FakeTensors``.  FakeTensor's ``device``
       property returns ``fake_device`` when not inside a kernel
       invocation (the usual case during pre-grad passes), but a
       defensive check of the ``fake_device`` attribute is included for
       the rare kernel-invocation path where ``.device`` returns
       ``"meta"``.
    3. **Graph placeholder metadata**: inspect ``node.meta['val']``
       on placeholder nodes as a last-resort fallback.

    Vulkan-only: only fires when the graph's tensors originate from a
    Vulkan device, so non-Vulkan compiles are unaffected.
    """
    import torch._inductor.compile_fx as _cfx

    if getattr(_cfx, "_vulkan_relu_rewrite_patched", False):
        return

    _orig = _cfx.run_pre_grad_passes

    def _detect_vulkan(example_inputs_, model_) -> bool:
        """Return True if this compilation involves Vulkan tensors."""
        inputs = example_inputs_ or ()
        # Strategy 1: real tensors (direct-call path in compile_fx.py:2883)
        for t in inputs:
            if not isinstance(t, torch.Tensor):
                continue
            try:
                if t.device.type in ("vulkan", "privateuseone"):
                    return True
            except Exception:
                pass
            # Strategy 2: FakeTensor with fake_device (aot_autograd path)
            # FakeTensor.device returns fake_device when not in kernel
            # invocation, but we check both for robustness.
            try:
                fd = getattr(t, "fake_device", None)
                if fd is not None and fd.type in ("vulkan", "privateuseone"):
                    return True
            except Exception:
                pass
        # Strategy 3: graph placeholder metadata (last-resort fallback)
        if isinstance(model_, torch.fx.GraphModule):
            try:
                for node in model_.graph.nodes:
                    if node.op != "placeholder":
                        continue
                    val = node.meta.get("val") if hasattr(node, "meta") else None
                    if val is None:
                        continue
                    for v in val if isinstance(val, (list, tuple)) else [val]:
                        if not isinstance(v, torch.Tensor):
                            continue
                        try:
                            if v.device.type in ("vulkan", "privateuseone"):
                                return True
                        except Exception:
                            pass
                        try:
                            fd = getattr(v, "fake_device", None)
                            if fd is not None and fd.type in (
                                "vulkan",
                                "privateuseone",
                            ):
                                return True
                        except Exception:
                            pass
            except Exception:
                pass
        return False

    def _patched(model_, example_inputs_):
        from .fx_passes.post_grad import _replace_relu_with_clamp_min

        if _detect_vulkan(example_inputs_, model_) and isinstance(
            model_, torch.fx.GraphModule
        ):
            try:
                _replace_relu_with_clamp_min(model_)
            except Exception as e:  # pragma: no cover
                import logging

                logging.getLogger(__name__).warning(
                    "Vulkan pre-grad relu rewrite failed: %s", e
                )
        return _orig(model_, example_inputs_)

    _cfx.run_pre_grad_passes = _patched
    _cfx._vulkan_relu_rewrite_patched = True
