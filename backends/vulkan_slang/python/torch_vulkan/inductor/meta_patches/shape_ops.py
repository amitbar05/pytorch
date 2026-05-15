"""Genuine shape-inference fake_impls for Vulkan view/shape/BLAS/conv/norm/
indexing/attention/FFT/SVD/memory/embedding ops.

Each function below is a ``fake_impl`` registered via ``_OP_IMPLS``
(in ``__init__.py``) that returns a ``FakeTensor`` with the correct
output shape and dtype — no storage allocation, no C++ dispatch.
"""

from __future__ import annotations

import torch

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


# ── Attention ────────────────────────────────────────────────────────


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


# ── Reduction backward ───────────────────────────────────────────────────


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
