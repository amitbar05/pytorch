// PF.31 — Stable C++ AOTI runtime entry (TA7).
//
// Inductor's Python wrapper calls `_vk_make_kernel`, a Python import, to build
// per-kernel dispatch closures. An AOTI-compiled `.so` cannot dispatch on a
// host without our Python package — half the "AOT" mission. This header
// exposes a frozen extern-C ABI that AOTI-emitted C++ wrappers (and a Python
// loader bridge for tests) call directly. The ABI never imports Python.
//
// Lifetime: `make_kernel` returns an opaque heap handle owning a reference
// into the global `vulkan::PipelineCache` (the cache itself owns the
// underlying Pipeline). `destroy_kernel` frees the handle; the Pipeline stays
// resident in the cache so repeat make_kernel calls with the same key are
// essentially free.
//
// Thread-safety: `PipelineCache::get_or_create` is mutex-locked; the dispatch
// path uses the same machinery as `_jit_dispatch_cached` (per-device stream).
// Returned handles can be used from any thread; concurrent dispatches go
// through the device's deferred-command-buffer ordering invariants.

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle: AOTI wrappers treat this as a `void*`.
struct AotiVulkanKernelHandle;

// Make a kernel from already-compiled SPIR-V plus an explicit
// (n_buffers, pc_size_bytes) layout. This is the path AOTI-emitted code calls.
//
// Parameters:
//   spirv_words   — SPIR-V bytecode as 32-bit words.
//   spirv_words_n — number of 32-bit words.
//   key           — null-terminated cache key (stable across runs; AOTI uses
//                   the same hash the Inductor wrapper computes).
//   n_buffers     — descriptor binding count (storage buffers).
//   pc_size_bytes — push-constant block size in bytes (0 if unused).
//   out_handle    — receives a non-null handle on success.
//
// Returns 0 on success, non-zero on failure (no exceptions cross the ABI).
int torch_vulkan_aoti_make_kernel(
    const uint32_t* spirv_words,
    size_t spirv_words_n,
    const char* key,
    uint32_t n_buffers,
    uint32_t pc_size_bytes,
    AotiVulkanKernelHandle** out_handle);

// Same, but derive (n_buffers, pc_size_bytes) from a slangc reflection JSON
// blob. PF.30 / PF.32 ship reflection JSON next to the SPV in the AOTI
// package; this entry lets the AOTI wrapper pass it through without parsing
// in generated code.
int torch_vulkan_aoti_make_kernel_from_reflection(
    const uint32_t* spirv_words,
    size_t spirv_words_n,
    const char* reflection_json,
    size_t reflection_json_len,
    const char* key,
    AotiVulkanKernelHandle** out_handle);

// Dispatch a previously-built handle. `tensor_handles` is an array of
// `at::Tensor*` (the AOTI codegen passes raw `&tensor` pointers from its
// stack-allocated `RAIIAtenTensorHandle::get()` results; for our test
// bridge the Python side passes the same).
//
// `push_constants` is the raw push-constant bytes (or null when pc_len=0).
// `num_outputs` is the trailing-RW-buffer count (default 1; matches the
// Python `make_vulkan_kernel` `n_outputs` arg).
//
// Returns 0 on success, non-zero on failure.
int torch_vulkan_aoti_dispatch(
    AotiVulkanKernelHandle* handle,
    void** tensor_handles,
    size_t n_tensors,
    const void* push_constants,
    size_t push_constants_size,
    uint32_t wg_x,
    uint32_t wg_y,
    uint32_t wg_z,
    uint32_t num_outputs);

// Free a handle. Idempotent on null. The underlying pipeline stays in the
// global cache (PipelineCache); only the per-call wrapper is freed.
void torch_vulkan_aoti_destroy_kernel(AotiVulkanKernelHandle* handle);

// ── Model-level AOTI (P3.4) ─────────────────────────────────────
// Opaque handle for a loaded AOTI model (collection of kernels + metadata).
struct AotiVulkanModelHandle;

// Load a compiled AOTI model from a directory or .pt2 archive.
// The path should point to a directory containing:
//   model.so       — compiled C++ wrapper (optional, for full AOT)
//   kernels.bin    — serialized kernel metadata + SPIR-V bundle
// Returns 0 on success, non-zero on failure.
int torch_vulkan_aoti_model_load(
    const char* path,
    AotiVulkanModelHandle** out_handle);

// Run all kernels in dispatch order with the given input tensors.
// inputs: array of at::Tensor* for model inputs.
// outputs: array of at::Tensor* for model outputs (pre-allocated).
// Returns 0 on success, non-zero on failure.
int torch_vulkan_aoti_model_run(
    AotiVulkanModelHandle* handle,
    void** inputs,
    size_t n_inputs,
    void** outputs,
    size_t n_outputs);

// Free a model handle. Idempotent on null.
void torch_vulkan_aoti_model_free(AotiVulkanModelHandle* handle);

// Last-error string for the calling thread. Returns a pointer to a
// thread-local C string; the contents are invalidated by the next call to
// any torch_vulkan_aoti_* entry on the same thread. Never returns null.
const char* torch_vulkan_aoti_last_error(void);

// ── T7.4 — Extern-ABI specializations ──────────────────────────────
// All four entries below are thin glue: they format a stable cache-key
// (matching the Python dispatcher's keys, so SPV from the AOTI package
// resolves) and route into the existing `torch_vulkan_aoti_dispatch`
// path (same `dispatch_shader` infrastructure used by JIT). The Python
// shims in `philox_dispatch.py` / `vulkan_template_caller.py` exist to
// render Slang source; under AOTI the SPV is precompiled at package
// time, so all that remains at runtime is buffer-binding + push-
// constant assembly + descriptor-set dispatch — exactly what these
// ABIs do without crossing into Python.

// Advance a Philox 64-bit counter by N elements (host-side; no GPU).
// `seed_state` is a pointer to a uint64_t holding the current counter
// (typically derived from `(seed_lo, seed_hi)` packed little-endian).
// Increments the counter by ((n_elements + 3) / 4) — Philox-4x32-10
// produces 4 outputs per round, so N elements consume ceil(N/4) rounds.
// This matches how the Python `_philox_seed_from_torch` + `offset`
// pairing advances the global generator across compiled-graph calls.
//
// Returns 0 on success, non-zero on failure.
int torch_vulkan_aoti_philox_advance(
    uint64_t* seed_state,
    size_t n_elements);

// Dispatch a scatter/gather/index_put template with the standard 3-uint
// push-constant payload `(numel, src_numel, out_numel)`.
//
// `kernel_handle` must have been built via `torch_vulkan_aoti_make_kernel`
// from the precompiled scatter SPV (cache-key format
// `"slang_scatter_{op}_{dtype}_{index_dtype}"` matching the Python
// dispatcher). Tensor order: `[src, indices, output]` (output last so
// dirty-buffer tracking marks it as written). For `scatter_reduce_mean`,
// pass `count_buffer` as the 4th tensor and `num_outputs=2`.
//
// Returns 0 on success, non-zero on failure.
int torch_vulkan_aoti_scatter_atomic(
    AotiVulkanKernelHandle* kernel_handle,
    void** tensor_handles,    // [src, indices, output, (count_buffer)]
    size_t n_tensors,         // 3 or 4
    uint32_t numel,
    uint32_t src_numel,
    uint32_t out_numel,
    uint32_t num_outputs);    // 1 or 2

// Dispatch a foreach-optimizer template with already-assembled push
// constants. The caller (AOTI codegen) builds the
// `(uint n_params, uint _pad[3], ParamConfig[batch_size])` payload
// matching the Python `_slang_foreach_optimizer` layout, since the
// per-algorithm config struct depends on `algorithm`. This entry only
// owns the buffer-binding + grid + dispatch step.
//
// `tensor_handles` order must match the Python convention:
//   [param0, grad0, param1, grad1, ..., paramN-1, gradN-1,
//    momentum0..N-1?, v0..N-1?]
// where the optional momentum/v slots are present per algorithm.
//
// Grid: `wg_x = ceil(numel_per_param / 256)`, `wg_y = n_params`, `wg_z = 1`.
//
// Returns 0 on success, non-zero on failure.
int torch_vulkan_aoti_foreach_optimizer(
    AotiVulkanKernelHandle* kernel_handle,
    void** tensor_handles,
    size_t n_tensors,
    const void* push_constants,
    size_t push_constants_size,
    uint32_t numel_per_param,
    uint32_t n_params,
    uint32_t num_outputs);

// Dispatch a flash-attention template. Tensor order:
//   [q, k, v, out, (lse?)]
// with `lse` present iff the variant emits log-sum-exp (training fwd).
// Push constants are caller-assembled to match the variant's PC struct.
//
// Returns 0 on success, non-zero on failure.
int torch_vulkan_aoti_flash_attention(
    AotiVulkanKernelHandle* kernel_handle,
    void** tensor_handles,    // [q, k, v, out, (lse)]
    size_t n_tensors,         // 4 or 5
    const void* push_constants,
    size_t push_constants_size,
    uint32_t wg_x,
    uint32_t wg_y,
    uint32_t wg_z,
    uint32_t num_outputs);    // 1 (out only) or 2 (out + lse)

#ifdef __cplusplus
}  // extern "C"
#endif
