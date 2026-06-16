// PF.31 — Stable C++ AOTI runtime entry (TA7) implementation.
#include "AotiRuntime.h"

#include "../vulkan/Context.h"
#include "../vulkan/Pipeline.h"
#include "../ops/dispatch.h"

#include <ATen/core/Tensor.h>

#include <fstream>
#include <cstring>

#include <algorithm>
#include <cstring>
#include <exception>
#include <limits>
#include <string>
#include <vector>

namespace {

// Per-thread last-error buffer. `torch_vulkan_aoti_last_error` returns a
// pointer into this string; AOTI callers can read it immediately after a
// non-zero return without worrying about exceptions crossing the ABI.
thread_local std::string g_last_error;

void set_last_error(const std::string& msg) { g_last_error = msg; }
void clear_last_error() { g_last_error.clear(); }

struct AotiKernelHandleImpl {
    vulkan::Pipeline* pipeline = nullptr;
    uint32_t n_buffers = 0;
    uint32_t pc_size_bytes = 0;
    std::string key;
};

// Minimal slangc reflection-JSON probe. Two facts to extract:
//   - n_buffers: count of `"kind": "descriptorTableSlot"` occurrences
//   - push_constant_size: when at least one `"kind": "pushConstantBuffer"`
//     is present, the largest `offset + size` over any `"kind": "uniform"`
//     entry. This works for both reflection schemas slangc emits — the
//     top-level `cbuffer` form (fields under `type.elementType.fields[]`)
//     and the entry-point uniform form (`uniform` directly in
//     entryPoints[*].parameters[*].binding). For the single-field shaders
//     used by Inductor today the maximum `offset + size` equals the total
//     block size; multi-field blocks come out the same because slangc
//     packs fields in declaration order with monotonic offsets.
// Avoids pulling nlohmann/json / rapidjson for two integers.
struct ReflLayout {
    uint32_t n_buffers = 0;
    uint32_t pc_size = 0;
};

inline bool key_eq(const char* p, const char* end, const char* s, size_t n) {
    return static_cast<size_t>(end - p) >= n && std::memcmp(p, s, n) == 0;
}

inline uint32_t parse_uint_after(const char* p, const char* end) {
    while (p < end && (*p == ' ' || *p == ':' || *p == '\t')) ++p;
    uint32_t v = 0;
    while (p < end && *p >= '0' && *p <= '9') {
        v = v * 10 + static_cast<uint32_t>(*p - '0');
        ++p;
    }
    return v;
}

ReflLayout parse_reflection_layout(const char* json, size_t len) {
    ReflLayout out;
    if (json == nullptr || len < 2) return out;
    const char* end = json + len;
    bool saw_pc = false;
    uint32_t max_extent = 0;

    for (const char* p = json; p < end - 1; ++p) {
        if (*p != '"') continue;
        const char* k = p + 1;
        const char* q = k;
        while (q < end && *q != '"') {
            if (*q == '\\' && q + 1 < end) ++q;
            ++q;
        }
        if (q >= end) break;
        size_t klen = static_cast<size_t>(q - k);

        if (klen == 4 && std::memcmp(k, "kind", 4) == 0) {
            // Skip to the value string after `"kind":`
            const char* r = q + 1;
            while (r < end && (*r == ' ' || *r == ':' || *r == '\t')) ++r;
            if (r < end && *r == '"') {
                const char* v = r + 1;
                if (key_eq(v, end, "descriptorTableSlot", 19)) {
                    out.n_buffers += 1;
                } else if (key_eq(v, end, "pushConstantBuffer", 18)) {
                    saw_pc = true;
                }
            }
        } else if (saw_pc && klen == 6
                && std::memcmp(k, "offset", 6) == 0) {
            // Parse `"offset": <N>` and look ahead for `"size": <M>` —
            // `offset + size` gives the field's end-byte; track the max.
            uint32_t off = parse_uint_after(q + 1, end);
            // Scan forward (within ~200 chars, bounded) for `"size": <N>`.
            const char* s = q + 1;
            const char* limit = std::min(end, s + 256);
            while (s < limit - 6) {
                if (*s == '"' && std::memcmp(s + 1, "size", 4) == 0
                        && s[5] == '"') {
                    uint32_t sz = parse_uint_after(s + 6, end);
                    if (off + sz > max_extent) max_extent = off + sz;
                    break;
                }
                ++s;
            }
        }
        p = q;
    }
    if (saw_pc) out.pc_size = max_extent;
    return out;
}

}  // namespace

extern "C" {

int torch_vulkan_aoti_make_kernel(
    const uint32_t* spirv_words,
    size_t spirv_words_n,
    const char* key,
    uint32_t n_buffers,
    uint32_t pc_size_bytes,
    AotiVulkanKernelHandle** out_handle) {
    clear_last_error();
    if (out_handle == nullptr) {
        set_last_error("torch_vulkan_aoti_make_kernel: out_handle is null");
        return 1;
    }
    *out_handle = nullptr;
    if (spirv_words == nullptr || spirv_words_n == 0) {
        set_last_error("torch_vulkan_aoti_make_kernel: empty SPIR-V");
        return 2;
    }
    if (key == nullptr) {
        set_last_error("torch_vulkan_aoti_make_kernel: null cache key");
        return 3;
    }
    try {
        auto& ctx = vulkan::Context::instance();
        if (!ctx.is_available()) {
            set_last_error("torch_vulkan_aoti_make_kernel: no Vulkan device");
            return 4;
        }
        VkDevice device = ctx.device(ctx.current_device());
        std::string skey(key);
        auto* pipeline = vulkan::PipelineCache::instance().get_or_create(
            device, skey, spirv_words,
            spirv_words_n * sizeof(uint32_t),
            n_buffers, pc_size_bytes);
        if (pipeline == nullptr) {
            set_last_error("torch_vulkan_aoti_make_kernel: pipeline cache returned null");
            return 5;
        }
        auto* h = new AotiKernelHandleImpl();
        h->pipeline = pipeline;
        h->n_buffers = n_buffers;
        h->pc_size_bytes = pc_size_bytes;
        h->key = std::move(skey);
        *out_handle = reinterpret_cast<AotiVulkanKernelHandle*>(h);
        return 0;
    } catch (const std::exception& e) {
        set_last_error(std::string("torch_vulkan_aoti_make_kernel: ") + e.what());
        return 6;
    } catch (...) {
        set_last_error("torch_vulkan_aoti_make_kernel: unknown exception");
        return 7;
    }
}

int torch_vulkan_aoti_make_kernel_from_reflection(
    const uint32_t* spirv_words,
    size_t spirv_words_n,
    const char* reflection_json,
    size_t reflection_json_len,
    const char* key,
    AotiVulkanKernelHandle** out_handle) {
    auto layout = parse_reflection_layout(reflection_json, reflection_json_len);
    if (layout.n_buffers == 0) {
        clear_last_error();
        set_last_error("torch_vulkan_aoti_make_kernel_from_reflection: "
                       "could not parse n_buffers from reflection JSON");
        if (out_handle != nullptr) *out_handle = nullptr;
        return 8;
    }
    return torch_vulkan_aoti_make_kernel(
        spirv_words, spirv_words_n, key,
        layout.n_buffers, layout.pc_size, out_handle);
}

int torch_vulkan_aoti_dispatch(
    AotiVulkanKernelHandle* handle,
    void** tensor_handles,
    size_t n_tensors,
    const void* push_constants,
    size_t push_constants_size,
    uint32_t wg_x,
    uint32_t wg_y,
    uint32_t wg_z,
    uint32_t num_outputs) {
    clear_last_error();
    if (handle == nullptr) {
        set_last_error("torch_vulkan_aoti_dispatch: null handle");
        return 1;
    }
    auto* h = reinterpret_cast<AotiKernelHandleImpl*>(handle);
    if (h->pipeline == nullptr) {
        set_last_error("torch_vulkan_aoti_dispatch: handle has null pipeline");
        return 2;
    }
    if (n_tensors > 0 && tensor_handles == nullptr) {
        set_last_error("torch_vulkan_aoti_dispatch: null tensor array with n>0");
        return 3;
    }
    try {
        std::vector<at::Tensor> buffers;
        buffers.reserve(n_tensors);
        for (size_t i = 0; i < n_tensors; ++i) {
            auto* t = reinterpret_cast<at::Tensor*>(tensor_handles[i]);
            if (t == nullptr) {
                set_last_error("torch_vulkan_aoti_dispatch: null tensor at slot "
                               + std::to_string(i));
                return 4;
            }
            buffers.push_back(*t);
        }
        // Dispatch through the existing shader dispatch infrastructure.
        // The pipeline was already created by torch_vulkan_aoti_make_kernel
        // and cached under h->key.  Passing nullptr SPIR-V makes
        // PipelineCache::get_or_create do a key-only lookup.
        torch_vulkan::ops::dispatch_shader(
            h->key, /*spirv_code=*/nullptr, /*spirv_size=*/0,
            buffers,
            wg_x, wg_y, wg_z,
            push_constants, static_cast<uint32_t>(push_constants_size),
            num_outputs);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "AOTI_DISPATCH ERROR: %s\n", e.what());
        fflush(stderr);
        set_last_error(std::string("torch_vulkan_aoti_dispatch: ") + e.what());
        return 5;
    } catch (...) {
        set_last_error("torch_vulkan_aoti_dispatch: unknown exception");
        return 6;
    }
}

void torch_vulkan_aoti_destroy_kernel(AotiVulkanKernelHandle* handle) {
    if (handle == nullptr) return;
    delete reinterpret_cast<AotiKernelHandleImpl*>(handle);
}

const char* torch_vulkan_aoti_last_error(void) {
    return g_last_error.c_str();
}

// ── Model-level AOTI (P3.4) ─────────────────────────────────────

namespace {

// Serialized per-kernel metadata in the kernels.bin file.
// Header: 8-byte magic + uint32_t kernel_count, then kernel entries.
struct AotiKernelEntry {
    uint32_t spirv_size;       // in uint32_t words
    uint32_t n_buffers;
    uint32_t pc_size_bytes;
    uint32_t key_len;          // length of cache key string
    // followed by: key_len bytes (cache key), then spirv_size*4 bytes (SPIR-V)
};

// Minimal header for kernels.bin
struct AotiBinHeader {
    char magic[8];             // "vk_aoti\n"
    uint32_t kernel_count;
};

struct AotiModelHandleImpl {
    std::vector<AotiKernelHandleImpl*> kernels;
    // For future: dispatch order / buffer-layout metadata
    std::string path;
};

// Simple binary-read helper
inline bool read_u32_le(std::istream& in, uint32_t& out) {
    char buf[4];
    if (!in.read(buf, 4)) return false;
    out = static_cast<uint32_t>(static_cast<unsigned char>(buf[0])) |
          (static_cast<uint32_t>(static_cast<unsigned char>(buf[1])) << 8) |
          (static_cast<uint32_t>(static_cast<unsigned char>(buf[2])) << 16) |
          (static_cast<uint32_t>(static_cast<unsigned char>(buf[3])) << 24);
    return true;
}

}  // namespace

int torch_vulkan_aoti_model_load(
    const char* path,
    AotiVulkanModelHandle** out_handle) {
    clear_last_error();
    if (out_handle == nullptr) {
        set_last_error("torch_vulkan_aoti_model_load: out_handle is null");
        return 1;
    }
    *out_handle = nullptr;
    if (path == nullptr) {
        set_last_error("torch_vulkan_aoti_model_load: null path");
        return 2;
    }

    try {
        auto& ctx = vulkan::Context::instance();
        if (!ctx.is_available()) {
            set_last_error("torch_vulkan_aoti_model_load: no Vulkan device");
            return 3;
        }

        // Build path to kernels.bin
        std::string dir(path);
        while (!dir.empty() && (dir.back() == '/' || dir.back() == '\\'))
            dir.pop_back();
        std::string bin_path = dir + "/kernels.bin";

        std::ifstream in(bin_path, std::ios::binary);
        if (!in) {
            set_last_error("torch_vulkan_aoti_model_load: cannot open " + bin_path);
            return 4;
        }

        AotiBinHeader hdr{};
        if (!in.read(hdr.magic, 8) ||
            std::memcmp(hdr.magic, "vk_aoti\n", 8) != 0) {
            set_last_error("torch_vulkan_aoti_model_load: bad magic in " + bin_path);
            return 5;
        }
        if (!read_u32_le(in, hdr.kernel_count) || hdr.kernel_count == 0) {
            set_last_error("torch_vulkan_aoti_model_load: zero kernel count");
            return 6;
        }

        auto* model = new AotiModelHandleImpl();
        model->path = dir;

        for (uint32_t k = 0; k < hdr.kernel_count; ++k) {
            AotiKernelEntry entry{};
            if (!read_u32_le(in, entry.spirv_size) ||
                !read_u32_le(in, entry.n_buffers) ||
                !read_u32_le(in, entry.pc_size_bytes) ||
                !read_u32_le(in, entry.key_len)) {
                set_last_error("torch_vulkan_aoti_model_load: truncated entry " +
                               std::to_string(k));
                delete model;
                return 7;
            }

            // Read cache key
            std::string key(entry.key_len, '\0');
            if (!in.read(&key[0], entry.key_len)) {
                set_last_error("torch_vulkan_aoti_model_load: truncated key " +
                               std::to_string(k));
                delete model;
                return 8;
            }

            // Read SPIR-V
            std::vector<uint32_t> spv(entry.spirv_size);
            if (entry.spirv_size > 0) {
                if (!in.read(reinterpret_cast<char*>(spv.data()),
                             entry.spirv_size * sizeof(uint32_t))) {
                    set_last_error("torch_vulkan_aoti_model_load: truncated SPIR-V " +
                                   std::to_string(k));
                    delete model;
                    return 9;
                }
            }

            // Create kernel via make_kernel
            AotiVulkanKernelHandle* kh = nullptr;
            int ret = torch_vulkan_aoti_make_kernel(
                spv.data(), entry.spirv_size,
                key.c_str(), entry.n_buffers, entry.pc_size_bytes,
                &kh);
            if (ret != 0) {
                set_last_error("torch_vulkan_aoti_model_load: make_kernel failed " +
                               std::to_string(k) + ": " +
                               std::string(torch_vulkan_aoti_last_error()));
                delete model;
                return 10;
            }
            model->kernels.push_back(
                reinterpret_cast<AotiKernelHandleImpl*>(kh));
        }

        *out_handle = reinterpret_cast<AotiVulkanModelHandle*>(model);
        return 0;
    } catch (const std::exception& e) {
        set_last_error(std::string("torch_vulkan_aoti_model_load: ") + e.what());
        return 11;
    } catch (...) {
        set_last_error("torch_vulkan_aoti_model_load: unknown exception");
        return 12;
    }
}

int torch_vulkan_aoti_model_run(
    AotiVulkanModelHandle* handle,
    void** inputs,
    size_t n_inputs,
    void** outputs,
    size_t n_outputs) {
    clear_last_error();
    if (handle == nullptr) {
        set_last_error("torch_vulkan_aoti_model_run: null handle");
        return 1;
    }
    auto* model = reinterpret_cast<AotiModelHandleImpl*>(handle);

    // ── Simplified single-kernel dispatch ─────────────────────
    // Extended in later milestones for multi-kernel dispatch with
    // proper buffer alias tracking and workgroup derivation from
    // tensor shapes. For now: dispatch all kernels in order with
    // the provided tensors treated as buffers.
    //
    // Each kernel is dispatched with all tensors (inputs interleaved
    // with outputs). Full multi-kernel scheduling with per-kernel
    // buffer layouts and intermediate tensor management is the next
    // increment (P4.x).
    if (model->kernels.empty()) {
        set_last_error("torch_vulkan_aoti_model_run: no kernels in model");
        return 2;
    }

    try {
        // Build combined tensor vector: inputs then outputs
        std::vector<at::Tensor> all_tensors;
        all_tensors.reserve(n_inputs + n_outputs);
        for (size_t i = 0; i < n_inputs; ++i) {
            auto* t = reinterpret_cast<at::Tensor*>(inputs[i]);
            if (t == nullptr) {
                set_last_error("torch_vulkan_aoti_model_run: null input at slot "
                               + std::to_string(i));
                return 3;
            }
            all_tensors.push_back(*t);
        }
        for (size_t i = 0; i < n_outputs; ++i) {
            auto* t = reinterpret_cast<at::Tensor*>(outputs[i]);
            if (t == nullptr) {
                set_last_error("torch_vulkan_aoti_model_run: null output at slot "
                               + std::to_string(i));
                return 4;
            }
            all_tensors.push_back(*t);
        }

        // Dispatch all kernels in order with the same tensor set.
        // Each kernel reads from the inputs and writes to the outputs
        // (simplified: full buffer set per kernel).
        for (size_t ki = 0; ki < model->kernels.size(); ++ki) {
            auto* kh = model->kernels[ki];
            // Default workgroup: 1D dispatch sized from first tensor
            uint32_t wg_x = 1, wg_y = 1, wg_z = 1;
            if (n_inputs > 0) {
                auto* t = reinterpret_cast<at::Tensor*>(inputs[0]);
                uint32_t numel = static_cast<uint32_t>(t->numel());
                wg_x = (numel + 255) / 256;
            } else if (n_outputs > 0) {
                auto* t = reinterpret_cast<at::Tensor*>(outputs[0]);
                uint32_t numel = static_cast<uint32_t>(t->numel());
                wg_x = (numel + 255) / 256;
            }

            torch_vulkan::ops::dispatch_shader(
                kh->key, /*spirv_code=*/nullptr, /*spirv_size=*/0,
                all_tensors, wg_x, wg_y, wg_z,
                /*push_constants=*/nullptr, /*pc_size=*/0,
                static_cast<uint32_t>(n_outputs));
        }

        // Flush all pending dispatches
        torch_vulkan::ops::flush_stream();
        return 0;
    } catch (const std::exception& e) {
        set_last_error(std::string("torch_vulkan_aoti_model_run: ") + e.what());
        return 5;
    } catch (...) {
        set_last_error("torch_vulkan_aoti_model_run: unknown exception");
        return 6;
    }
}

void torch_vulkan_aoti_model_free(AotiVulkanModelHandle* handle) {
    if (handle == nullptr) return;
    auto* model = reinterpret_cast<AotiModelHandleImpl*>(handle);
    for (auto* kh : model->kernels) {
        delete kh;
    }
    delete model;
}

// ── T7.4 — Extern-ABI specializations ──────────────────────────────
//
// Each entry is a thin glue layer over `torch_vulkan_aoti_dispatch`:
// validate args, compute the workgroup grid for that kernel family,
// and forward to the generic dispatch. The kernel handle was already
// built via `torch_vulkan_aoti_make_kernel` from precompiled SPV at
// package-load time — these entries do not invoke slangc, JSON, or
// Python.

int torch_vulkan_aoti_philox_advance(
    uint64_t* seed_state,
    size_t n_elements) {
    clear_last_error();
    if (seed_state == nullptr) {
        set_last_error("torch_vulkan_aoti_philox_advance: null seed_state");
        return 1;
    }
    // Philox-4x32-10 emits 4 random words per round. Advance the
    // counter by ceil(n_elements / 4) so consecutive calls produce
    // disjoint streams (matches the Python `offset` semantics in
    // `_dispatch_philox_rng` where the caller passes `offset=0` and
    // each subsequent dispatch increments by total_elements/4).
    uint64_t rounds = static_cast<uint64_t>((n_elements + 3) / 4);
    // Saturating add — wrapping is well-defined for unsigned, but
    // bound the addend so callers passing absurd values (e.g.,
    // SIZE_MAX) trip on a checked overflow rather than silently
    // wrapping the global counter.
    if (rounds > (std::numeric_limits<uint64_t>::max() - *seed_state)) {
        set_last_error("torch_vulkan_aoti_philox_advance: counter overflow");
        return 2;
    }
    *seed_state += rounds;
    return 0;
}

int torch_vulkan_aoti_scatter_atomic(
    AotiVulkanKernelHandle* kernel_handle,
    void** tensor_handles,
    size_t n_tensors,
    uint32_t numel,
    uint32_t src_numel,
    uint32_t out_numel,
    uint32_t num_outputs) {
    clear_last_error();
    if (n_tensors < 3) {
        set_last_error("torch_vulkan_aoti_scatter_atomic: need >=3 tensors "
                       "[src, indices, output, (count_buffer)]");
        return 1;
    }
    // PC layout matches `_dispatch_scatter_atomic`:
    //   uint numel; uint src_numel; uint out_numel;
    uint32_t pc[3] = {numel, src_numel, out_numel};
    // Grid: ceil(numel / 256), 1, 1 — matches the Python dispatcher.
    uint32_t wg_x = (numel + 255u) / 256u;
    return torch_vulkan_aoti_dispatch(
        kernel_handle, tensor_handles, n_tensors,
        pc, sizeof(pc),
        wg_x, 1, 1,
        num_outputs);
}

int torch_vulkan_aoti_foreach_optimizer(
    AotiVulkanKernelHandle* kernel_handle,
    void** tensor_handles,
    size_t n_tensors,
    const void* push_constants,
    size_t push_constants_size,
    uint32_t numel_per_param,
    uint32_t n_params,
    uint32_t num_outputs) {
    clear_last_error();
    if (n_params == 0) {
        set_last_error("torch_vulkan_aoti_foreach_optimizer: n_params=0");
        return 1;
    }
    if (numel_per_param == 0) {
        set_last_error("torch_vulkan_aoti_foreach_optimizer: numel_per_param=0");
        return 2;
    }
    // Grid: X = ceil(numel / 256), Y = n_params, Z = 1.  Matches
    // `_slang_foreach_optimizer` in vulkan_template_caller.py.
    uint32_t wg_x = (numel_per_param + 255u) / 256u;
    uint32_t wg_y = n_params;
    return torch_vulkan_aoti_dispatch(
        kernel_handle, tensor_handles, n_tensors,
        push_constants, push_constants_size,
        wg_x, wg_y, 1,
        num_outputs);
}

int torch_vulkan_aoti_flash_attention(
    AotiVulkanKernelHandle* kernel_handle,
    void** tensor_handles,
    size_t n_tensors,
    const void* push_constants,
    size_t push_constants_size,
    uint32_t wg_x,
    uint32_t wg_y,
    uint32_t wg_z,
    uint32_t num_outputs) {
    clear_last_error();
    if (n_tensors < 4) {
        set_last_error("torch_vulkan_aoti_flash_attention: need >=4 tensors "
                       "[q, k, v, out, (lse)]");
        return 1;
    }
    // Workgroup is variant-specific (D, B, H, N tile shape) — caller
    // computed it at AOTI compile time. Just forward.
    return torch_vulkan_aoti_dispatch(
        kernel_handle, tensor_handles, n_tensors,
        push_constants, push_constants_size,
        wg_x, wg_y, wg_z,
        num_outputs);
}

}  // extern "C"
