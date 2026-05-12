#include "dispatch.h"
#include "dtype_utils.h"
#include "../generated/shaders.h"

#include <cstdint>
#include <vector>

#include <torch/library.h>

namespace torch_vulkan { namespace ops {

// ── Embedding ───────────────────────────────────────────────────
at::Tensor vulkan_embedding(const at::Tensor& weight, const at::Tensor& indices,
                            int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
    auto weight_c = weight.contiguous();
    check_supported_float(weight_c, "embedding");
    auto orig_dtype = weight_c.scalar_type();

    int64_t num_indices = indices.numel();
    int64_t embedding_dim = weight.size(1);

    // Create output in the original dtype (no upcast needed for lookup)
    std::vector<int64_t> out_shape(indices.sizes().begin(), indices.sizes().end());
    out_shape.push_back(embedding_dim);

    // For f16/bf16 weights: use raw uint32 copy shader to avoid 2x memory upcast.
    // f16/bf16 each pack 2 values per uint32. f32 packs 1 per uint32.
    const bool is_packed16 = (orig_dtype == at::kHalf || orig_dtype == at::kBFloat16);
    if (is_packed16) {
        // Use embedding_raw shader: treat weight buffer as uint32 (2 f16/bf16 per uint32).
        // embedding_dim must be even (always true for transformer hidden sizes).
        TORCH_CHECK(embedding_dim % 2 == 0,
                    "embedding: f16/bf16 weight requires even embedding_dim, got ", embedding_dim);
        uint32_t dim_u32 = static_cast<uint32_t>(embedding_dim / 2);

        auto output = at::empty(out_shape, weight_c.options());  // bf16/f16 output
        if (num_indices == 0) return output;

        // Get indices as int32 on Vulkan (reinterpret as float buffer for dispatch).
        // The embedding_raw shader uses asint() on StructuredBuffer<float>, so int32
        // data in a float-typed buffer works correctly with no copy overhead.
        at::Tensor indices_vulkan;

        if (indices.device().type() == c10::DeviceType::PrivateUse1 &&
            indices.scalar_type() == at::kLong) {
            auto indices_c = indices.contiguous();
            indices_vulkan = at::empty({num_indices}, weight_c.options());
            struct { uint32_t numel; } i2i_params{static_cast<uint32_t>(num_indices)};
            dispatch_shader("indexing_i64_to_i32_fwd",
                            shaders::indexing_i64_to_i32_fwd,
                            shaders::indexing_i64_to_i32_fwd_size,
                            {indices_c, indices_vulkan},
                            (static_cast<uint32_t>(num_indices) + 255) / 256, 1, 1,
                            &i2i_params, sizeof(i2i_params));
        } else if (indices.device().type() == c10::DeviceType::PrivateUse1 &&
                   indices.scalar_type() == at::kInt) {
            // Reinterpret int32 buffer as float — same 4-byte layout, no copy needed.
            auto indices_c = indices.contiguous();
            auto impl = c10::make_intrusive<at::TensorImpl>(
                c10::Storage(indices_c.storage()),
                indices_c.key_set(),
                at::scalarTypeToTypeMeta(at::kFloat));
            std::vector<int64_t> sz = {num_indices}, st = {1};
            impl->set_sizes_and_strides(sz, st);
            impl->set_storage_offset(indices_c.storage_offset());
            indices_vulkan = at::Tensor(std::move(impl));
        } else {
            indices_vulkan = at::empty({num_indices}, weight_c.options());
            auto indices_cpu = indices.cpu().to(at::kInt).contiguous();
            auto& alloc = VulkanAllocator::instance();
            auto* buf = alloc.get_buffer(indices_vulkan.data_ptr());
            TORCH_CHECK(buf, "Failed to get Vulkan buffer for indices");
            buf->write(indices_cpu.data_ptr(), static_cast<VkDeviceSize>(num_indices * sizeof(int32_t)));
        }

        struct { uint32_t num_indices; uint32_t embedding_dim_u32; } params{
            static_cast<uint32_t>(num_indices), dim_u32
        };
        uint32_t total_u32 = static_cast<uint32_t>(num_indices) * dim_u32;
        uint32_t workgroups = (total_u32 + 255) / 256;
        dispatch_shader("indexing_embedding_raw_fwd",
                        shaders::indexing_embedding_raw_fwd,
                        shaders::indexing_embedding_raw_fwd_size,
                        {weight_c, indices_vulkan, output},
                        workgroups, 1, 1,
                        &params, sizeof(params));
        return output;
    }

    // f32 path: standard upcast-lookup-downcast (original behavior)
    weight_c = ensure_float32(weight_c);

    auto output = at::empty(out_shape, weight_c.options());
    if (num_indices == 0) return output;

    // Get indices as int32 on Vulkan. Avoid CPU roundtrip when possible.
    // The embedding shader uses StructuredBuffer<float> for indices and reads them
    // via asint() — so we can pass int32 data in a float-typed buffer with no copy.
    at::Tensor indices_vulkan;

    if (indices.device().type() == c10::DeviceType::PrivateUse1 &&
        indices.scalar_type() == at::kLong) {
        // Int64 on Vulkan: convert to int32 on GPU via shader
        // Int64 is stored as pairs of uint32 (little-endian), we extract low bits
        auto indices_c = indices.contiguous();
        indices_vulkan = at::empty({num_indices}, weight_c.options().dtype(at::kFloat));
        struct { uint32_t numel; } i2i_params{static_cast<uint32_t>(num_indices)};
        dispatch_shader("indexing_i64_to_i32_fwd",
                        shaders::indexing_i64_to_i32_fwd,
                        shaders::indexing_i64_to_i32_fwd_size,
                        {indices_c, indices_vulkan},
                        (static_cast<uint32_t>(num_indices) + 255) / 256, 1, 1,
                        &i2i_params, sizeof(i2i_params));
    } else if (indices.device().type() == c10::DeviceType::PrivateUse1 &&
               indices.scalar_type() == at::kInt) {
        // Int32 on Vulkan: reinterpret the buffer as float (same 4-byte layout, no copy).
        // The embedding shader uses asint(buffer[i]) to read indices, so passing int32
        // data in a float-typed binding works correctly.
        auto indices_c = indices.contiguous();
        auto impl = c10::make_intrusive<at::TensorImpl>(
            c10::Storage(indices_c.storage()),
            indices_c.key_set(),
            at::scalarTypeToTypeMeta(at::kFloat));
        std::vector<int64_t> sz2 = {num_indices}, st2 = {1};
        impl->set_sizes_and_strides(sz2, st2);
        impl->set_storage_offset(indices_c.storage_offset());
        indices_vulkan = at::Tensor(std::move(impl));
    } else {
        // CPU path: convert and upload
        indices_vulkan = at::empty({num_indices}, weight_c.options().dtype(at::kFloat));
        auto indices_cpu = indices.cpu().to(at::kInt).contiguous();
        auto& alloc = VulkanAllocator::instance();
        auto* buf = alloc.get_buffer(indices_vulkan.data_ptr());
        TORCH_CHECK(buf, "Failed to get Vulkan buffer for indices");
        buf->write(indices_cpu.data_ptr(), static_cast<VkDeviceSize>(num_indices * sizeof(int32_t)));
    }

    struct { uint32_t num_indices; uint32_t embedding_dim; } params{
        static_cast<uint32_t>(num_indices),
        static_cast<uint32_t>(embedding_dim)
    };

    uint32_t total = static_cast<uint32_t>(num_indices * embedding_dim);
    uint32_t workgroups = (total + 255) / 256;

    dispatch_shader("indexing_embedding_fwd",
                    shaders::indexing_embedding_fwd, shaders::indexing_embedding_fwd_size,
                    {weight_c, indices_vulkan, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return output;  // already f32 — no cast needed
}

// ── Index Select ────────────────────────────────────────────────
at::Tensor vulkan_index_select(const at::Tensor& self, int64_t dim, const at::Tensor& index) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "index_select");
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);

    dim = at::maybe_wrap_dim(dim, self_c.dim());

    // For non-zero dims, transpose to make dim=0, then use GPU shader
    if (dim != 0) {
        auto transposed = self_c.movedim(dim, 0).contiguous();
        auto result = vulkan_index_select(transposed, 0, index);
        return cast_from_float32(result.movedim(0, dim).contiguous(), orig_dtype);
    }

    auto indices_cpu = index.cpu().to(at::kInt).contiguous();
    int64_t num_indices = indices_cpu.numel();
    int64_t slice_size = self_c.numel() / self_c.size(0);

    std::vector<int64_t> out_shape = {num_indices};
    for (int64_t i = 1; i < self_c.dim(); i++) out_shape.push_back(self_c.size(i));
    auto output = at::empty(out_shape, self_c.options());

    if (num_indices == 0) return output;

    // Transfer indices
    auto indices_vulkan = at::empty({num_indices}, self_c.options().dtype(at::kFloat));
    {
        auto& alloc = VulkanAllocator::instance();
        auto* buf = alloc.get_buffer(indices_vulkan.data_ptr());
        buf->write(indices_cpu.data_ptr(), static_cast<VkDeviceSize>(num_indices * sizeof(int32_t)));
    }

    struct { uint32_t num_indices; uint32_t slice_size; } params{
        static_cast<uint32_t>(num_indices),
        static_cast<uint32_t>(slice_size)
    };

    uint32_t total = static_cast<uint32_t>(num_indices * slice_size);
    uint32_t workgroups = (total + 255) / 256;

    dispatch_shader("indexing_index_select_fwd",
                    shaders::indexing_index_select_fwd, shaders::indexing_index_select_fwd_size,
                    {self_c, indices_vulkan, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

// ── Masked Fill ─────────────────────────────────────────────────
at::Tensor& vulkan_masked_fill(at::Tensor& self, const at::Tensor& mask, const at::Scalar& value) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "masked_fill_");

    auto mask_float = mask.to(at::kFloat).contiguous().to(self.device());
    uint32_t numel = static_cast<uint32_t>(self_c.numel());

    if (numel == 0) return self;

    auto output = at::empty_like(self_c);

    struct { float fill_value; uint32_t numel; } params{
        value.toFloat(), numel
    };

    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("indexing_masked_fill_fwd",
                    shaders::indexing_masked_fill_fwd, shaders::indexing_masked_fill_fwd_size,
                    {self_c, mask_float, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));

    // Copy result back to self via GPU
    dispatch_copy_buffer(output, self);

    return self;
}

// ── Masked Scatter ──────────────────────────────────────────────
at::Tensor vulkan_masked_scatter(const at::Tensor& self, const at::Tensor& mask, const at::Tensor& source) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "masked_scatter");
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);

    // Flatten everything for element-wise operation
    auto self_flat = self_c.reshape({-1});
    uint32_t numel = static_cast<uint32_t>(self_flat.numel());
    if (numel == 0) return self_c.reshape(self.sizes());

    // Get mask as bool on CPU to build index mapping
    auto mask_flat = mask.reshape({-1}).to(at::kBool).cpu();
    auto mask_accessor = mask_flat.accessor<bool, 1>();

    // Build index buffer on CPU: for each position, store the source index
    // if mask is true, or 0xFFFFFFFF if mask is false
    auto indices_cpu = at::empty({numel}, at::kInt);
    auto* indices_ptr = indices_cpu.data_ptr<int32_t>();
    int32_t src_idx = 0;
    for (uint32_t i = 0; i < numel; i++) {
        if (mask_accessor[i]) {
            indices_ptr[i] = src_idx++;
            TORCH_CHECK(src_idx <= static_cast<int32_t>(source.numel()),
                        "masked_scatter: source doesn't have enough values");
        } else {
            indices_ptr[i] = static_cast<int32_t>(0xFFFFFFFF);
        }
    }

    // Upload index buffer to Vulkan (pack as float — shader reads as uint)
    auto indices_vulkan = at::empty({static_cast<int64_t>(numel)}, self_c.options());
    {
        auto& alloc = VulkanAllocator::instance();
        auto* buf = alloc.get_buffer(indices_vulkan.data_ptr());
        TORCH_CHECK(buf, "Failed to get Vulkan buffer for indices");
        buf->write(indices_cpu.data_ptr(), static_cast<VkDeviceSize>(numel * sizeof(int32_t)));
    }

    // Ensure source is contiguous float32 on Vulkan
    auto src_c = source.contiguous();
    src_c = ensure_float32(src_c);
    if (src_c.device() != self_c.device()) {
        src_c = src_c.to(self_c.device());
    }

    auto output = at::empty_like(self_flat);

    struct { uint32_t numel; } params{ numel };

    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("indexing_masked_scatter_fwd",
                    shaders::indexing_masked_scatter_fwd, shaders::indexing_masked_scatter_fwd_size,
                    {self_flat, indices_vulkan, src_c.reshape({-1}), output},
                    workgroups, 1, 1,
                    &params, sizeof(params));

    return cast_from_float32(output.reshape(self.sizes()), orig_dtype);
}

// In-place version
at::Tensor& vulkan_masked_scatter_(at::Tensor& self, const at::Tensor& mask, const at::Tensor& source) {
    auto result = vulkan_masked_scatter(self, mask, source);
    dispatch_copy_buffer(result, self);
    return self;
}

// ── OP.1.a-fast: GPU-native nonzero via two-pass scan ───────────
// Returns int64 tensor of shape (N, ndim) listing the indices of nonzero
// elements in row-major (C-contiguous) order, where N = count of nonzero
// elements in `self`.
//
// Two-pass GPU scan:
//   Pass 1 (nonzero_count):  scans input, computes per-workgroup nonzero
//                            counts using wg_inclusive_scan_wave<WaveScanAdd>
//                            and writes them to workspace.
//   CPU prefix sum:          computes exclusive prefix sum over per-WG
//                            counts (small — O(num_workgroups)).
//   Pass 2 (nonzero_scatter): re-scans input using the prefix-sum offsets
//                            to write each nonzero's multi-dimensional
//                            index into the output int64 tensor.
//
// Falls back to CPU roundtrip when the input is not on Vulkan or cannot
// be converted to float32.
at::Tensor vulkan_nonzero(const at::Tensor& self) {
    // Fallback for non-Vulkan tensors
    if (self.device().type() != c10::DeviceType::PrivateUse1) {
        auto cpu_input = self.cpu();
        auto cpu_result = at::nonzero(cpu_input);
        return cpu_result.to(self.device());
    }

    // GPU path: two-pass scan
    auto self_c = self.contiguous();
    int64_t ndim = self_c.dim();
    // Fall back to CPU for 0D or >4D tensors (shader supports 1D-4D)
    if (ndim == 0 || ndim > 4) {
        auto cpu_input = self.cpu();
        auto cpu_result = at::nonzero(cpu_input);
        return cpu_result.to(self.device());
    }

    // Ensure float32 for the shader (handles f16/bf16 via GPU cast)
    auto orig_dtype = self_c.scalar_type();
    if (!is_supported_float(orig_dtype) && orig_dtype != at::kBool) {
        // For integer types not easily cast to float, fall back to CPU
        auto cpu_input = self.cpu();
        auto cpu_result = at::nonzero(cpu_input);
        return cpu_result.to(self.device());
    }
    self_c = ensure_float32(self_c);

    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) {
        return at::empty({0, ndim}, self_c.options().dtype(at::kLong));
    }

    uint32_t num_workgroups = (numel + 255) / 256;

    // ── Allocate workspace buffer (uint32, one per workgroup) ──
    auto workspace = at::empty({static_cast<int64_t>(num_workgroups)},
                               self_c.options().dtype(at::kInt));

    // ── Pass 1: count nonzero elements per workgroup ──
    {
        struct { uint32_t numel; } count_params{numel};
        dispatch_shader("nonzero_count",
                        shaders::indexing_nonzero_count_fwd,
                        shaders::indexing_nonzero_count_fwd_size,
                        {self_c, workspace},
                        num_workgroups, 1, 1,
                        &count_params, sizeof(count_params));
    }

    // Flush GPU work and read workspace back to CPU for prefix sum
    flush_stream();
    auto workspace_cpu = workspace.cpu().to(at::kInt);
    auto* ws_ptr = workspace_cpu.data_ptr<int32_t>();

    // ── CPU: compute exclusive prefix sum over per-WG counts ──
    uint32_t total_nonzero = 0;
    std::vector<uint32_t> prefix(num_workgroups);
    for (uint32_t i = 0; i < num_workgroups; i++) {
        prefix[i] = total_nonzero;
        total_nonzero += static_cast<uint32_t>(ws_ptr[i]);
    }

    // Handle empty result
    if (total_nonzero == 0) {
        return at::empty({0, ndim}, self_c.options().dtype(at::kLong));
    }

    // Write exclusive prefix sum back to workspace on GPU
    {
        auto& alloc = VulkanAllocator::instance();
        auto* buf = alloc.get_buffer(workspace.data_ptr());
        TORCH_CHECK(buf, "Failed to get Vulkan buffer for workspace");
        buf->write(prefix.data(),
                   static_cast<VkDeviceSize>(num_workgroups * sizeof(uint32_t)));
    }

    // ── Allocate output tensor: (total_nonzero, ndim) int64 ──
    // The shader writes int64 as pairs of uint32, so we allocate
    // int64 buffer and treat it as uint32 in the shader.
    auto output = at::empty({static_cast<int64_t>(total_nonzero), ndim},
                            self_c.options().dtype(at::kLong));

    // ── Pass 2: scatter nonzero indices to output ──
    {
        // Build shape array for push constants (up to 4D)
        uint32_t s0 = static_cast<uint32_t>(self_c.size(0));
        uint32_t s1 = ndim >= 2 ? static_cast<uint32_t>(self_c.size(1)) : 1u;
        uint32_t s2 = ndim >= 3 ? static_cast<uint32_t>(self_c.size(2)) : 1u;
        uint32_t s3 = ndim >= 4 ? static_cast<uint32_t>(self_c.size(3)) : 1u;

        struct {
            uint32_t ndim;
            uint32_t shape0;
            uint32_t shape1;
            uint32_t shape2;
            uint32_t shape3;
            uint32_t numel;
        } scatter_params{static_cast<uint32_t>(ndim), s0, s1, s2, s3, numel};

        dispatch_shader("nonzero_scatter",
                        shaders::indexing_nonzero_scatter_fwd,
                        shaders::indexing_nonzero_scatter_fwd_size,
                        {self_c, workspace, output},
                        num_workgroups, 1, 1,
                        &scatter_params, sizeof(scatter_params));
    }

    return output;
}

}} // namespace torch_vulkan::ops
