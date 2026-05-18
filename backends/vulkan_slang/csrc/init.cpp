#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

#include <cstdlib>

#include "vulkan/Context.h"
#include "vulkan/Pipeline.h"  // M-pipeline-4: PipelineCache::collision_count()
#include "vulkan/DescriptorSet.h"  // M-cpp-new-2-followup-pybind: async reset counters
#include "vulkan/Stream.h"  // M-NEW.4: submit_count()
#include "backend/Allocator.h"
#include "backend/Hooks.h"
#include "ops/ops.h"
#include "ops/dispatch.h"
#include "backend/AotiRuntime.h"

namespace torch_vulkan {

// Defined in Profiler.cpp
void register_profiler_stubs();

bool is_available() {
    return vulkan::Context::instance().is_available();
}

int64_t device_count() {
    return static_cast<int64_t>(vulkan::Context::instance().device_count());
}

int64_t current_device() {
    // M22.9: currently single-device (device 0). Multi-GPU support
    // requires per-thread device-context tracking.
    return 0;
}

std::string get_device_name(int64_t device_index) {
    return vulkan::Context::instance().device_name(
        static_cast<uint32_t>(device_index));
}

void synchronize(int64_t device_index) {
    auto& ctx = vulkan::Context::instance();
    auto device = ctx.device(static_cast<uint32_t>(device_index));
    vkDeviceWaitIdle(device);
}

PYBIND11_MODULE(_C, m) {
    // Register PrivateUse1 backend as "vulkan"
    c10::register_privateuse1_backend("vulkan");

    // Register allocator
    auto* alloc = &VulkanAllocator::instance();
    c10::SetAllocator(c10::DeviceType::PrivateUse1, alloc);

    // Expose shutdown for Python atexit to call
    m.def("_shutdown", []() { vulkan::Context::instance().shutdown(); });

    // Register hooks
    static VulkanHooksInterface hooks;
    at::RegisterPrivateUse1HooksInterface(&hooks);

    // Register profiler stubs
    register_profiler_stubs();

    m.def("_is_available", &is_available);
    m.def("_device_count", &device_count);
    m.def("_current_device", &current_device);
    m.def("_get_device_name", &get_device_name);
    m.def("_synchronize", &synchronize);
    m.def("_manual_seed", [](uint64_t seed) { ops::vulkan_manual_seed(seed); });
    m.def("_empty_cache", []() { VulkanAllocator::instance().empty_cache(); });
    m.def("_memory_cached", []() -> int64_t {
        return static_cast<int64_t>(VulkanAllocator::instance().cached_bytes());
    });
    m.def("_empty_strided_fast", [](const std::vector<int64_t>& size,
                                     const std::vector<int64_t>& stride,
                                     c10::ScalarType dtype) -> at::Tensor {
        auto options = at::TensorOptions()
            .dtype(dtype)
            .device(c10::Device(c10::DeviceType::PrivateUse1, 0));
        return at::empty_strided(size, stride, options);
    });

    // Custom ops
    m.def("rope", [](const at::Tensor& input, double theta) {
        return ops::vulkan_rope_autograd(input, theta);
    }, py::arg("input"), py::arg("theta") = 10000.0);

    // Direct SDPA bypass — F.scaled_dot_product_attention dispatches through
    // CompositeImplicitAutograd which fails with None mask on PrivateUse1
    m.def("_sdpa", [](const at::Tensor& query, const at::Tensor& key,
                       const at::Tensor& value,
                       const std::optional<at::Tensor>& attn_mask,
                       double dropout_p, bool is_causal,
                       std::optional<double> scale) {
        return ops::vulkan_sdpa_autograd(query, key, value, attn_mask,
                                          dropout_p, is_causal, scale);
    }, py::arg("query"), py::arg("key"), py::arg("value"),
       py::arg("attn_mask") = py::none(), py::arg("dropout_p") = 0.0,
       py::arg("is_causal") = false, py::arg("scale") = py::none());

    m.def("rms_norm", [](const at::Tensor& input, const at::Tensor& weight, double eps) {
        return ops::vulkan_rms_norm_autograd(input, weight, eps);
    }, py::arg("input"), py::arg("weight"), py::arg("eps") = 1e-6);

    m.def("swiglu", [](const at::Tensor& gate, const at::Tensor& up) {
        return ops::vulkan_swiglu_autograd(gate, up);
    }, py::arg("gate"), py::arg("up"));

    // Fused attention score: scale * (q @ k.T) in 1 dispatch instead of 2
    // q: [B, M, K] contiguous, k: [B, N, K] contiguous (NOT pre-transposed)
    // Returns: scale * (q @ k.T) = [B, M, N]
    // Saves 1 dispatch vs (q @ k.T) then * scale
    m.def("scaled_bmm", [](const at::Tensor& q, const at::Tensor& k, double scale) {
        return ops::vulkan_scaled_bmm_autograd(q, k, scale);
    }, py::arg("q"), py::arg("k"), py::arg("scale"));


    // Fused Add + RMSNorm: h_new = residual + shortcut; normed = weight * (h_new / rms(h_new))
    // Returns (normed, h_new). Saves 1 dispatch vs separate add + rms_norm.
    m.def("add_rms_norm", [](const at::Tensor& residual, const at::Tensor& shortcut,
                              const at::Tensor& weight, double eps) {
        return ops::vulkan_add_rms_norm_apply(residual, shortcut, weight, eps);
    }, py::arg("residual"), py::arg("shortcut"), py::arg("weight"), py::arg("eps") = 1e-6);

    // Flash Attention: fused QK^T + softmax + @V (7 dispatches → 1)
    // Q: [B,H,N,D] or [B,S,H,D] (seq-major), K/V: matching layout
    // Returns output [B,H,N,D]. Fully differentiable via flash attention backward.
    // q_seq_major=True: Q/K/V in [B,S,H,D] layout — skips 3 contiguous() copies per call.
    m.def("flash_attention", [](const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V,
                                 double scale, bool is_causal, bool q_seq_major) {
        return ops::vulkan_flash_attention_autograd(Q, K, V, scale, is_causal, q_seq_major);
    }, py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("scale"),
       py::arg("is_causal") = true, py::arg("q_seq_major") = false);

    // RMSNormGated: weight * rms_norm(input) * silu(gate) — Qwen3.5 GatedDeltaNet
    m.def("rms_norm_gated", [](const at::Tensor& input, const at::Tensor& gate,
                                const at::Tensor& weight, double eps) {
        return ops::vulkan_rms_norm_gated_autograd(input, gate, weight, eps);
    }, py::arg("input"), py::arg("gate"), py::arg("weight"), py::arg("eps") = 1e-6);

    // Fused SGD step
    m.def("_sgd_step", [](at::Tensor& param, const at::Tensor& grad,
                           at::Tensor& momentum_buf,
                           double lr, double momentum, double dampening,
                           double weight_decay, bool nesterov,
                           bool has_momentum_buf) {
        ops::vulkan_sgd_step(param, grad, momentum_buf,
                              static_cast<float>(lr),
                              static_cast<float>(momentum),
                              static_cast<float>(dampening),
                              static_cast<float>(weight_decay),
                              nesterov, has_momentum_buf);
    }, py::arg("param"), py::arg("grad"), py::arg("momentum_buf"),
       py::arg("lr"), py::arg("momentum"), py::arg("dampening"),
       py::arg("weight_decay"), py::arg("nesterov"),
       py::arg("has_momentum_buf"));

    // Batched SGD step (no momentum): up to 15 params per dispatch
    m.def("_sgd_batch_step", [](std::vector<at::Tensor> params,
                                 std::vector<at::Tensor> grads,
                                 double lr, double weight_decay) {
        std::vector<at::Tensor*> param_ptrs;
        std::vector<const at::Tensor*> grad_ptrs;
        param_ptrs.reserve(params.size());
        grad_ptrs.reserve(grads.size());
        for (auto& p : params) param_ptrs.push_back(&p);
        for (auto& g : grads)  grad_ptrs.push_back(&g);
        ops::vulkan_sgd_batch_step(param_ptrs, grad_ptrs,
                                    static_cast<float>(lr),
                                    static_cast<float>(weight_decay));
    }, py::arg("params"), py::arg("grads"), py::arg("lr"), py::arg("weight_decay"));

    // Batched AdamW step: up to 7 params per dispatch
    m.def("_adamw_batch_step", [](std::vector<at::Tensor> params,
                                   std::vector<at::Tensor> grads,
                                   std::vector<at::Tensor> m_bufs,
                                   std::vector<at::Tensor> v_bufs,
                                   double lr, double beta1, double beta2,
                                   double eps, double weight_decay,
                                   double bc1, double bc2) {
        std::vector<at::Tensor*> pp, mb, vb;
        std::vector<const at::Tensor*> gp;
        for (auto& p : params) pp.push_back(&p);
        for (auto& g : grads)  gp.push_back(&g);
        for (auto& m : m_bufs) mb.push_back(&m);
        for (auto& v : v_bufs) vb.push_back(&v);
        ops::vulkan_adamw_batch_step(pp, gp, mb, vb,
                                      static_cast<float>(lr),
                                      static_cast<float>(beta1),
                                      static_cast<float>(beta2),
                                      static_cast<float>(eps),
                                      static_cast<float>(weight_decay),
                                      static_cast<float>(bc1),
                                      static_cast<float>(bc2));
    }, py::arg("params"), py::arg("grads"), py::arg("m_bufs"), py::arg("v_bufs"),
       py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"),
       py::arg("weight_decay"), py::arg("bc1"), py::arg("bc2"));

    // Fused AdamW step
    m.def("_adamw_step", [](at::Tensor& param, const at::Tensor& grad,
                             at::Tensor& m_buf, at::Tensor& v_buf,
                             double lr, double beta1, double beta2, double eps,
                             double weight_decay, int64_t step) {
        ops::vulkan_adamw_step(param, grad, m_buf, v_buf,
                               static_cast<float>(lr),
                               static_cast<float>(beta1),
                               static_cast<float>(beta2),
                               static_cast<float>(eps),
                               static_cast<float>(weight_decay),
                               step);
    }, py::arg("param"), py::arg("grad"), py::arg("m"), py::arg("v"),
       py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"),
       py::arg("weight_decay"), py::arg("step"));

    // ── JIT dispatch for Inductor codegen ──────────────────────────
    // Cached variants: pipeline arg is unused (pipeline cache handles dedup).
    // We pass nullptr SPIR-V — PipelineCache uses the key for lookups.

    m.def("_jit_dispatch_cached_nopc", [](int64_t pipeline,
                                            const std::vector<at::Tensor>& tensors,
                                            int64_t wg_x, int64_t wg_y, int64_t wg_z,
                                            int64_t num_outputs) {
        (void)pipeline;
        std::string key = "vk_" + std::to_string(pipeline);
        ops::dispatch_shader(key, nullptr, 0, tensors,
                              static_cast<uint32_t>(wg_x),
                              static_cast<uint32_t>(wg_y),
                              static_cast<uint32_t>(wg_z),
                              nullptr, 0,
                              static_cast<uint32_t>(num_outputs));
    }, py::arg("pipeline"), py::arg("tensors"),
       py::arg("wg_x"), py::arg("wg_y"), py::arg("wg_z"),
       py::arg("num_outputs"));

    m.def("_jit_dispatch_cached", [](int64_t pipeline,
                                       const std::vector<at::Tensor>& tensors,
                                       int64_t wg_x, int64_t wg_y, int64_t wg_z,
                                       const py::bytes& pc_bytes,
                                       int64_t num_outputs) {
        (void)pipeline;
        std::string key = "vk_" + std::to_string(pipeline);
        std::string pc = pc_bytes;
        const void* pc_data = pc.empty() ? nullptr : pc.data();
        uint32_t pc_size = static_cast<uint32_t>(pc.size());
        ops::dispatch_shader(key, nullptr, 0, tensors,
                              static_cast<uint32_t>(wg_x),
                              static_cast<uint32_t>(wg_y),
                              static_cast<uint32_t>(wg_z),
                              pc_data, pc_size,
                              static_cast<uint32_t>(num_outputs));
    }, py::arg("pipeline"), py::arg("tensors"),
       py::arg("wg_x"), py::arg("wg_y"), py::arg("wg_z"),
       py::arg("push_constants"), py::arg("num_outputs"));

    // M9.5: cached indexed dispatch — same as _jit_dispatch_cached but
    // routes through the descriptor-array path for kernels with
    // descriptorCount > 1 bindings.
    m.def("_jit_dispatch_indexed_cached_nopc", [](int64_t pipeline,
                                            const std::vector<at::Tensor>& tensors,
                                            const std::vector<uint32_t>& descriptor_counts,
                                            int64_t wg_x, int64_t wg_y, int64_t wg_z,
                                            int64_t num_outputs) {
        (void)pipeline;
        std::string key = "vk_" + std::to_string(pipeline);
        ops::dispatch_shader_indexed(key, nullptr, 0, tensors,
                                      descriptor_counts,
                                      static_cast<uint32_t>(wg_x),
                                      static_cast<uint32_t>(wg_y),
                                      static_cast<uint32_t>(wg_z),
                                      nullptr, 0,
                                      static_cast<uint32_t>(num_outputs));
    }, py::arg("pipeline"), py::arg("tensors"),
       py::arg("descriptor_counts"),
       py::arg("wg_x"), py::arg("wg_y"), py::arg("wg_z"),
       py::arg("num_outputs"));

    m.def("_jit_dispatch_indexed_cached", [](int64_t pipeline,
                                       const std::vector<at::Tensor>& tensors,
                                       const std::vector<uint32_t>& descriptor_counts,
                                       int64_t wg_x, int64_t wg_y, int64_t wg_z,
                                       const py::bytes& pc_bytes,
                                       int64_t num_outputs) {
        (void)pipeline;
        std::string key = "vk_" + std::to_string(pipeline);
        std::string pc = pc_bytes;
        const void* pc_data = pc.empty() ? nullptr : pc.data();
        uint32_t pc_size = static_cast<uint32_t>(pc.size());
        ops::dispatch_shader_indexed(key, nullptr, 0, tensors,
                                      descriptor_counts,
                                      static_cast<uint32_t>(wg_x),
                                      static_cast<uint32_t>(wg_y),
                                      static_cast<uint32_t>(wg_z),
                                      pc_data, pc_size,
                                      static_cast<uint32_t>(num_outputs));
    }, py::arg("pipeline"), py::arg("tensors"),
       py::arg("descriptor_counts"),
       py::arg("wg_x"), py::arg("wg_y"), py::arg("wg_z"),
       py::arg("push_constants"), py::arg("num_outputs"));

    // Raw dispatch with key+spirv (used by dispatch() and compile_and_dispatch()).
    // CG.M15: spec_constants accepts [(constant_id, value), ...] for
    // VkSpecializationInfo at pipeline-creation time.
    m.def("_jit_dispatch", [](const std::string& key,
                               const py::bytes& spirv_bytes,
                               const std::vector<at::Tensor>& tensors,
                               int64_t wg_x, int64_t wg_y, int64_t wg_z,
                               const py::bytes& pc_bytes,
                               int64_t num_outputs,
                               const std::vector<std::pair<uint32_t, uint32_t>>& spec_constants) {
        std::string spv = spirv_bytes;
        const auto* code = spv.empty() ? nullptr
            : reinterpret_cast<const uint32_t*>(spv.data());
        size_t code_size = spv.size();
        std::string pc = pc_bytes;
        const void* pc_data = pc.empty() ? nullptr : pc.data();
        uint32_t pc_size = static_cast<uint32_t>(pc.size());
        ops::dispatch_shader(key, code, code_size, tensors,
                              static_cast<uint32_t>(wg_x),
                              static_cast<uint32_t>(wg_y),
                              static_cast<uint32_t>(wg_z),
                              pc_data, pc_size,
                              static_cast<uint32_t>(num_outputs),
                              spec_constants);
    }, py::arg("key"), py::arg("spirv"), py::arg("tensors"),
       py::arg("wg_x"), py::arg("wg_y") = 1, py::arg("wg_z") = 1,
       py::arg("push_constants") = py::bytes(),
       py::arg("num_outputs") = 1,
       py::arg("spec_constants") = std::vector<std::pair<uint32_t, uint32_t>>{});

    // Pipeline factory — returns a callable that creates cached pipelines.
    // Called as: pipeline = get_pipeline(key, spirv, n_buffers, pc_size_bytes)
    m.def("_jit_pipeline", [](const std::string& key,
                               const py::bytes& spirv_bytes,
                               int64_t n_buffers,
                               int64_t pc_size_bytes) -> int64_t {
        // Use a simple counter-based pipeline handle.
        // Real pipeline handles would use VkPipelineCache.
        static std::atomic<int64_t> next_id{1};
        (void)key; (void)spirv_bytes; (void)n_buffers; (void)pc_size_bytes;
        return next_id.fetch_add(1, std::memory_order_relaxed);
    }, py::arg("key"), py::arg("spirv"),
       py::arg("n_buffers"), py::arg("pc_size_bytes"));

    // N+1.5: descriptor-array variant of `_jit_dispatch`.
    // `descriptor_counts` is a per-binding count array. For flat layouts
    // (every binding has count=1) callers should use `_jit_dispatch`.
    //
    // Usage:
    //   _vk._jit_dispatch_indexed(
    //       key, spirv, tensors,           # tensors flattened in binding order
    //       descriptor_counts,             # e.g. [1, 4, 4, 4, 4] for adamw_b4
    //       wg_x, wg_y, wg_z, push_constants, num_outputs)
    // CG.M15: spec_constants for VkSpecializationInfo.
    m.def("_jit_dispatch_indexed", [](const std::string& key,
                                        const py::bytes& spirv_bytes,
                                        const std::vector<at::Tensor>& tensors,
                                        const std::vector<uint32_t>& descriptor_counts,
                                        int64_t wg_x, int64_t wg_y, int64_t wg_z,
                                        const py::bytes& pc_bytes,
                                        int64_t num_outputs,
                                        const std::vector<std::pair<uint32_t, uint32_t>>& spec_constants) {
        std::string spv = spirv_bytes;
        const auto* code = spv.empty() ? nullptr
            : reinterpret_cast<const uint32_t*>(spv.data());
        size_t code_size = spv.size();
        std::string pc = pc_bytes;
        const void* pc_data = pc.empty() ? nullptr : pc.data();
        uint32_t pc_size = static_cast<uint32_t>(pc.size());
        ops::dispatch_shader_indexed(key, code, code_size, tensors,
                                      descriptor_counts,
                                      static_cast<uint32_t>(wg_x),
                                      static_cast<uint32_t>(wg_y),
                                      static_cast<uint32_t>(wg_z),
                                      pc_data, pc_size,
                                      static_cast<uint32_t>(num_outputs),
                                      spec_constants);
    }, py::arg("key"), py::arg("spirv"), py::arg("tensors"),
       py::arg("descriptor_counts"),
       py::arg("wg_x"), py::arg("wg_y") = 1, py::arg("wg_z") = 1,
       py::arg("push_constants") = py::bytes(),
       py::arg("num_outputs") = 1,
       py::arg("spec_constants") = std::vector<std::pair<uint32_t, uint32_t>>{});

    // Probe for whether `descriptorCount > 1` bindings will work on this
    // device (descriptor indexing enabled + extension supported).
    m.def("_descriptor_indexing_enabled", []() -> bool {
        return vulkan::Context::instance().descriptor_indexing_enabled();
    });

    // M-pipeline-4: PipelineCache collision telemetry. Non-zero means
    // the Python-side cache key (e.g. ``kernel.config_key`` or
    // ``compute_combo_config_key``) is not content-aware enough — two
    // distinct Slang sources mapped to the same key. The C++ side
    // detected the mismatch via SPIR-V hash and recompiled (preventing
    // silent miscompile), but the counter records the event so tests
    // can assert == 0 under a normal training workload.
    m.def("_pipeline_cache_collisions", []() -> uint64_t {
        return vulkan::PipelineCache::instance().collision_count();
    });

    // M-cpp-new-2-followup-pybind: expose the DescriptorPool async-
    // reset counters added in M-cpp-new-2. Together they let Python
    // tests assert the M9.2 batching win is preserved (the async
    // path fires + drains at a healthy ratio).
    //
    // ``_descriptor_pool_async_reset_requests()`` — total
    // ``reset_async`` calls into the pool since process start.
    //
    // ``_descriptor_pool_async_resets_drained()`` — total
    // ``vkResetDescriptorPool`` calls actually executed by the
    // drainer (each one resets all then-outstanding pending
    // entries in a single shot). Always ≤ async_reset_requests.
    m.def("_descriptor_pool_async_reset_requests", []() -> uint64_t {
        auto& rt = ops::get_runtime();
        return rt.desc_pool ? rt.desc_pool->async_reset_requests() : 0;
    });
    m.def("_descriptor_pool_async_resets_drained", []() -> uint64_t {
        auto& rt = ops::get_runtime();
        return rt.desc_pool ? rt.desc_pool->async_resets_drained() : 0;
    });

    // M-NEW.4: cumulative ``vkQueueSubmit`` calls from the M9.2
    // batched-flush hot path on the current device's Stream. The
    // canonical M9.2 win telemetry — post-fix the ratio
    // ``dispatch_count / submit_count`` should approach
    // ``MAX_DISPATCHES_PER_CMD`` (32). A ratio near 1 indicates the
    // deferred-cmd-buffer batching is defeated (regression).
    m.def("_stream_submit_count", []() -> uint64_t {
        auto& rt = ops::get_runtime();
        return rt.stream ? rt.stream->submit_count() : 0;
    });

    // M22.9-followup-introspection-pybind: return the device index
    // stored on a tensor's storage ``DataPtr`` — NOT the tensor's
    // impl-key device.
    //
    // The two can differ on multi-GPU rigs pre-M22.9-followup:
    //   - ``tensor.device.index`` reads the impl's dispatch-key
    //     device (set by ``_change_backend_component_keys`` per
    //     M22.9).
    //   - ``_storage_device_index(tensor)`` reads the underlying
    //     ``DataPtr.device().index()`` which was wired by the
    //     allocator (the M22.9-followup fix routes the device
    //     index through ``VulkanAllocator::allocate(size_t,
    //     DeviceIndex)``).
    //
    // The two MUST agree post-M22.9-followup for any tensor
    // constructed via ``vulkan_empty`` / ``vulkan_empty_strided``.
    // Disagreement → silent multi-GPU correctness bug.
    m.def("_storage_device_index", [](const at::Tensor& t) -> int64_t {
        const auto& storage = t.storage();
        return static_cast<int64_t>(storage.data_ptr().device().index());
    });

    // M-cpp-new-5-followup-test: runtime override for the
    // descriptor-indexing capability check. Lets tests force the
    // non-cached fallback path on rigs where the capability flag
    // would otherwise return true.
    //
    // Override values:
    //   -1 → use the capability flag (default; production behaviour)
    //    0 → force off (stresses the non-cached fallback path —
    //        the only safe path on drivers without
    //        UPDATE_AFTER_BIND_BIT)
    //    1 → force on (asserts the cached path)
    //
    // The override is the only way to flip descriptor indexing
    // mid-process; the env var ``TORCH_VULKAN_DESCRIPTOR_INDEXING``
    // is captured at Context init. Use exclusively for tests.
    m.def("_set_descriptor_indexing_override", [](int v) {
        vulkan::Context::set_descriptor_indexing_override(v);
    });
    m.def("_get_descriptor_indexing_override", []() -> int {
        return vulkan::Context::get_descriptor_indexing_override();
    });

    // M18.4-followup-C: device-feature dictionary. Returns the enabled
    // bits on the current device so Python-side tests can confirm 8/16-bit
    // storage and shaderInt8/16 are on (without going through vulkaninfo).
    // Used by TestM184FollowUpCDeviceFeatures.
    m.def("_device_caps", []() -> py::dict {
        const auto& caps = vulkan::Context::instance().capabilities();
        py::dict d;
        d["float16"] = caps.float16;
        d["int8"] = caps.int8;
        d["int16"] = caps.int16;
        d["int64"] = caps.int64;
        d["float64"] = caps.float64;
        d["storage_buffer_8bit"] = caps.storage_buffer_8bit;
        d["uniform_and_storage_buffer_8bit"] =
            caps.uniform_and_storage_buffer_8bit;
        d["storage_buffer_16bit"] = caps.storage_buffer_16bit;
        d["uniform_and_storage_buffer_16bit"] =
            caps.uniform_and_storage_buffer_16bit;
        d["descriptor_indexing"] = caps.descriptor_indexing;
        d["subgroup_size"] = caps.subgroup_size;
        d["max_workgroup_size"] = caps.max_workgroup_size;
        d["max_compute_shared_memory"] = caps.max_compute_shared_memory;
        d["device_name"] = caps.device_name;
        return d;
    });

    // ── AOTI runtime bindings (P3.4) ────────────────────────────
    // Kernel-level AOTI: create, dispatch, destroy a single kernel.
    m.def("_aoti_make_kernel", [](const py::bytes& spv_bytes,
                                    const std::string& key,
                                    int64_t n_buffers,
                                    int64_t pc_size_bytes) -> int64_t {
        std::string spv = spv_bytes;
        const auto* code = spv.empty() ? nullptr
            : reinterpret_cast<const uint32_t*>(spv.data());
        size_t code_words = spv.size() / sizeof(uint32_t);
        AotiVulkanKernelHandle* handle = nullptr;
        int ret = torch_vulkan_aoti_make_kernel(
            code, code_words, key.c_str(),
            static_cast<uint32_t>(n_buffers),
            static_cast<uint32_t>(pc_size_bytes),
            &handle);
        if (ret != 0) {
            throw std::runtime_error(torch_vulkan_aoti_last_error());
        }
        return reinterpret_cast<int64_t>(handle);
    }, py::arg("spirv"), py::arg("key"),
       py::arg("n_buffers"), py::arg("pc_size_bytes"));

    m.def("_aoti_dispatch", [](int64_t handle_int,
                                  const std::vector<at::Tensor>& tensors,
                                  int64_t wg_x, int64_t wg_y, int64_t wg_z,
                                  const py::bytes& pc_bytes,
                                  int64_t num_outputs) {
        auto* handle = reinterpret_cast<AotiVulkanKernelHandle*>(handle_int);
        std::vector<void*> tensor_ptrs;
        tensor_ptrs.reserve(tensors.size());
        for (const auto& t : tensors) {
            tensor_ptrs.push_back(
                const_cast<void*>(reinterpret_cast<const void*>(&t)));
        }
        std::string pc = pc_bytes;
        int ret = torch_vulkan_aoti_dispatch(
            handle, tensor_ptrs.data(), tensor_ptrs.size(),
            pc.empty() ? nullptr : pc.data(), pc.size(),
            static_cast<uint32_t>(wg_x),
            static_cast<uint32_t>(wg_y),
            static_cast<uint32_t>(wg_z),
            static_cast<uint32_t>(num_outputs));
        if (ret != 0) {
            throw std::runtime_error(torch_vulkan_aoti_last_error());
        }
    }, py::arg("handle"), py::arg("tensors"),
       py::arg("wg_x"), py::arg("wg_y"), py::arg("wg_z"),
       py::arg("push_constants") = py::bytes(),
       py::arg("num_outputs") = 1);

    m.def("_aoti_destroy_kernel", [](int64_t handle_int) {
        auto* handle = reinterpret_cast<AotiVulkanKernelHandle*>(handle_int);
        torch_vulkan_aoti_destroy_kernel(handle);
    }, py::arg("handle"));

    // Model-level AOTI: load, run, free an entire model.
    m.def("_aoti_model_load", [](const std::string& path) -> int64_t {
        AotiVulkanModelHandle* handle = nullptr;
        int ret = torch_vulkan_aoti_model_load(path.c_str(), &handle);
        if (ret != 0) {
            throw std::runtime_error(torch_vulkan_aoti_last_error());
        }
        return reinterpret_cast<int64_t>(handle);
    }, py::arg("path"));

    m.def("_aoti_model_run", [](int64_t handle_int,
                                   const std::vector<at::Tensor>& inputs,
                                   const std::vector<at::Tensor>& outputs) {
        auto* handle = reinterpret_cast<AotiVulkanModelHandle*>(handle_int);
        std::vector<void*> input_ptrs;
        std::vector<void*> output_ptrs;
        for (const auto& t : inputs) {
            input_ptrs.push_back(
                const_cast<void*>(reinterpret_cast<const void*>(&t)));
        }
        for (const auto& t : outputs) {
            output_ptrs.push_back(
                const_cast<void*>(reinterpret_cast<const void*>(&t)));
        }
        int ret = torch_vulkan_aoti_model_run(
            handle, input_ptrs.data(), input_ptrs.size(),
            output_ptrs.data(), output_ptrs.size());
        if (ret != 0) {
            throw std::runtime_error(torch_vulkan_aoti_last_error());
        }
    }, py::arg("handle"), py::arg("inputs"), py::arg("outputs"));

    m.def("_aoti_model_free", [](int64_t handle_int) {
        auto* handle = reinterpret_cast<AotiVulkanModelHandle*>(handle_int);
        torch_vulkan_aoti_model_free(handle);
    }, py::arg("handle"));

    // M17.5: Batch dispatch mode — suppresses auto-flush until end_batch_dispatch()
    m.def("begin_batch_dispatch", []() {
        ops::begin_batch_dispatch();
    }, "Begin batched dispatch mode — suppresses auto-flush until end_batch_dispatch()");

    m.def("end_batch_dispatch", []() {
        ops::end_batch_dispatch();
    }, "End batched dispatch mode — flushes remaining dispatches");

    // Flush pending GPU work (for benchmarking / synchronization)
    m.def("_flush", []() { ops::flush_stream(); });

    // Perf counters
    m.def("_get_dispatch_count", []() -> int64_t { return ops::get_dispatch_count(); });
    m.def("_get_flush_count", []() -> int64_t { return ops::get_flush_count(); });
    m.def("_get_war_flush_count", []() -> int64_t { return ops::get_war_flush_count(); });
    m.def("_get_preread_flush_count", []() -> int64_t { return ops::get_preread_flush_count(); });
    m.def("_get_capacity_flush_count", []() -> int64_t { return ops::get_capacity_flush_count(); });
    m.def("_get_descpool_flush_count", []() -> int64_t { return ops::get_descpool_flush_count(); });
    m.def("_get_barrier_count", []() -> int64_t { return ops::get_barrier_count(); });
    m.def("_get_barrier_skip_count", []() -> int64_t { return ops::get_barrier_skip_count(); });
    m.def("_reset_perf_counters", []() { ops::reset_perf_counters(); });

    // Per-dispatch timing breakdown (nanoseconds, cumulative)
    // Only populated when TORCH_VULKAN_PROFILE_DISPATCH=1.
    m.def("_profiling_enabled", []() -> bool { return ops::dispatch_profiling_enabled(); });
    m.def("_profile_pipeline_cache_ns", []() -> int64_t { return ops::get_profile_pipeline_cache_ns(); });
    m.def("_profile_get_runtime_ns", []() -> int64_t { return ops::get_profile_get_runtime_ns(); });
    m.def("_profile_desc_alloc_ns", []() -> int64_t { return ops::get_profile_desc_alloc_ns(); });
    m.def("_profile_buffer_info_ns", []() -> int64_t { return ops::get_profile_buffer_info_ns(); });
    m.def("_profile_desc_write_ns", []() -> int64_t { return ops::get_profile_desc_write_ns(); });
    m.def("_profile_barrier_check_ns", []() -> int64_t { return ops::get_profile_barrier_check_ns(); });
    m.def("_profile_cmd_record_ns", []() -> int64_t { return ops::get_profile_cmd_record_ns(); });
    m.def("_profile_dirty_track_ns", []() -> int64_t { return ops::get_profile_dirty_track_ns(); });
    m.def("_reset_profile_timers", []() { ops::reset_profile_timers(); });
}

} // namespace torch_vulkan
