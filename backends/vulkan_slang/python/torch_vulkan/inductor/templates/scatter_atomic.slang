// Scatter / Gather / IndexPut compute shader — T4.5 / T4.11
//
// One template handles nine operations by switching on the Jinja `operation`
// parameter at render time:
//   gather                — out[i] = src[indices[i]]           (pointwise, no atomics)
//   scatter               — out[indices[i]] = src[i]           (non-atomic store)
//   scatter_add           — out[indices[i]] += src[i]          (atomic float add)
//   index_put             — out[indices[i]] = values[i]        (non-atomic store)
//   index_put_accumulate  — out[indices[i]] += values[i]       (atomic float add)
//   scatter_reduce_amax   — out[indices[i]] = max(out[..], v)  (T4.11, atomic_max_f32)
//   scatter_reduce_amin   — out[indices[i]] = min(out[..], v)  (T4.11, atomic_min_f32)
//   scatter_reduce_prod   — out[indices[i]] *= src[i]          (T4.11, atomic_mul_f32)
//   scatter_reduce_mean   — out[indices[i]] += src[i] AND count[indices[i]] += 1
//                           (T4.11, atomic_add_f32 + InterlockedAdd; post-pass
//                            divides on the host side / second kernel).
//
// Buffer layout (canonical struct field order, matching dispatch caller):
//   args.src     — source / values (StructuredBuffer<dtype>)
//   args.indices — indices (StructuredBuffer<index_dtype>)
//   args.out     — output (RWStructuredBuffer<uint> for scatter,
//                          RWStructuredBuffer<dtype> for gather)
//
// NOTE: int64 index buffers are declared as uint2 (the backend convention
// for devices without shaderInt64). The index value is read from the .x
// component (low 32 bits), which is sufficient for all realistic tensor
// dimensions (< 2^31 elements).
//
// N+1.10 — Buffer bindings are wrapped in a single
// ``ParameterBlock<KernelArgs>``.  Slang assigns one descriptorTableSlot
// per struct field in declaration order (set 0, binding 0..N-1),
// matching the descriptorCount=1 layout the runtime expects.  The
// dispatch caller (``_dispatch_scatter_atomic`` in
// ``vulkan_template_caller.py``) passes tensors in the same order as
// struct fields here (src, indices, out), so no `[[vk::binding(N)]]`
// literals are emitted.  Mirrors the foreach_optimizer + philox_rng
// round-3/4 pattern.  No array-of-structs of buffers, so the
// descriptor-array caveat documented in foreach_optimizer.py.jinja
// does not bite us here.  The uint2-encoded int64 indices remain a
// plain ``StructuredBuffer<uint2>`` field — this works cleanly inside
// a ParameterBlock because slangc maps it to a single
// descriptorTableSlot like any other StructuredBuffer<T>.

/*{%*/ set is_gather = (operation == "gather") /*%}*/
/*{%*/ set is_mean = (operation == "scatter_reduce_mean") /*%}*/
/*{%*/ set is_atomic = (operation in (
    "scatter_add",
    "index_put_accumulate",
    "scatter_reduce_amax",
    "scatter_reduce_amin",
    "scatter_reduce_prod",
    "scatter_reduce_mean",
)) /*%}*/
/*# The Vulkan backend stores int64 tensors as uint2 (vec2 of uint32) #*/
/*# because many Vulkan devices do not support the shaderInt64 capability. #*/
/*# StructuredBuffer<int64_t> would fail to create a pipeline on those #*/
/*# devices. We follow the same convention as the Inductor codegen: #*/
/*# declare uint2 buffers and read .x for the low 32 bits of the index. #*/
/*{%*/ set idx_buf_type = "uint2" if index_dtype == "int64_t" else index_dtype /*%}*/
/*{%*/ set idx_read = "asint(args.indices[i].x)" if index_dtype == "int64_t" else "args.indices[i]" /*%}*/

struct PC {
    uint numel;         // elements to process (= len(indices) for scatter/gather)
    uint src_numel;     // elements in src / values buffer (bounds check)
    uint out_numel;     // elements in output buffer (bounds check)
};

[[vk::push_constant]] PC pc;

// All buffer bindings live in this ParameterBlock.  Field order is the
// canonical binding order; the dispatch caller passes tensors in the
// same order (src, indices, out).  No `[[vk::binding(N)]]` literals
// appear anywhere in this template.
struct KernelArgs {
    StructuredBuffer</*{{*/ dtype /*}}float*/> src;
    StructuredBuffer</*{{*/ idx_buf_type /*}}uint2*/> indices;
/*{%*/ if is_gather /*%}*/
    RWStructuredBuffer</*{{*/ dtype /*}}float*/> out;
/*{%*/ else /*%}*/
    // Scatter / index_put writes into a uint-view of the output so the CAS-based
    // float atomic-add can use InterlockedCompareExchange on the raw bits.
    RWStructuredBuffer<uint> out;
/*{%*/ endif /*%}*/
/*{%*/ if is_mean /*%}*/
    // Mean-mode also tracks how many elements landed on each output slot
    // so a follow-on kernel (or host-side pass) can divide for the mean.
    // Field is only emitted for `scatter_reduce_mean` — other operations
    // keep the original 3-field layout for cache-key stability.
    RWStructuredBuffer<uint> count;
/*{%*/ endif /*%}*/
};
// M21.3.01: explicit Set 0 binding (slangc 2026.7.1 defaults to Set 1)
[[vk::binding(0, 0)]] ParameterBlock<KernelArgs> args;

/*{%*/ if is_atomic /*%}*/
// ── Import the canonical atomic-add implementation from lib/atomics.slang
// PF.21.b: the precompiled module carries the correct SPIR-V decorations
// (Coherent) for InterlockedCompareExchange to work on Vulkan drivers.
import atomics;
/*{%*/ endif /*%}*/

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= pc.numel) return;

    // Read index from the index buffer.  For int64 index tensors the
    // buffer is declared as uint2 and we take the low 32 bits; this
    // matches the backend convention in header.py:_binding_dtype().
    // Use asint() (bit reinterpretation) not int() (value conversion)
    // so that negative int64 indices survive the uint2 round-trip.
    int idx = /*{{*/ idx_read /*}}asint(args.indices[i].x)*/;

    // Guard against reading past the end of the indices buffer.
    // pc.numel is the dispatch size (≤ indices.numel()); i is the
    // global thread id.  The grid-level check (i >= pc.numel) above
    // already prevents overshoot, but this is belt-and-suspenders.
    // This also guards the src[i] read below for scatter/index_put paths.
    if (i >= pc.src_numel) return;

/*{%*/ if is_gather /*%}*/
    // gather: out[i] = src[indices[i]]
    // The index refers to the source buffer, so we bounds-check against
    // src_numel (not out_numel).  The output index i is already guarded
    // by the pc.numel check above.
    if (idx < 0 || uint(idx) >= pc.src_numel) return;
    args.out[i] = args.src[uint(idx)];
/*{%*/ elif operation == "scatter" /*%}*/
    // scatter: out[indices[i]] = src[i]  (last-write-wins for duplicates)
    // The index refers to the output buffer; guard with out_numel.
    if (idx < 0 || uint(idx) >= pc.out_numel) return;
    args.out[uint(idx)] = asuint(args.src[i]);
/*{%*/ elif operation == "scatter_add" /*%}*/
    // scatter_add: out[indices[i]] += src[i]  (atomic)
    if (idx < 0 || uint(idx) >= pc.out_numel) return;
    c10_vulkan_atomic_add(args.out, uint(idx), (float)args.src[i]);
/*{%*/ elif operation == "index_put" /*%}*/
    // index_put: out[indices[i]] = values[i]  (last-write-wins)
    if (idx < 0 || uint(idx) >= pc.out_numel) return;
    args.out[uint(idx)] = asuint(args.src[i]);
/*{%*/ elif operation == "index_put_accumulate" /*%}*/
    // index_put(accumulate=True): out[indices[i]] += values[i]  (atomic)
    if (idx < 0 || uint(idx) >= pc.out_numel) return;
    c10_vulkan_atomic_add(args.out, uint(idx), (float)args.src[i]);
/*{%*/ elif operation == "scatter_reduce_amax" /*%}*/
    // scatter_reduce(reduce='amax'): out[indices[i]] = max(out[indices[i]], src[i])
    // T4.11 — float atomic max via CAS loop in lib/atomics.slang.
    if (idx < 0 || uint(idx) >= pc.out_numel) return;
    atomic_max_f32(args.out, uint(idx), (float)args.src[i]);
/*{%*/ elif operation == "scatter_reduce_amin" /*%}*/
    // scatter_reduce(reduce='amin'): out[indices[i]] = min(out[indices[i]], src[i])
    // T4.11 — float atomic min via CAS loop in lib/atomics.slang.
    if (idx < 0 || uint(idx) >= pc.out_numel) return;
    atomic_min_f32(args.out, uint(idx), (float)args.src[i]);
/*{%*/ elif operation == "scatter_reduce_prod" /*%}*/
    // scatter_reduce(reduce='prod'): out[indices[i]] *= src[i]
    // T4.11 — float atomic multiply via CAS loop in lib/atomics.slang.
    if (idx < 0 || uint(idx) >= pc.out_numel) return;
    atomic_mul_f32(args.out, uint(idx), (float)args.src[i]);
/*{%*/ elif operation == "scatter_reduce_mean" /*%}*/
    // scatter_reduce(reduce='mean'): same atomic-add as sum-mode plus a
    // companion uint-counter SSBO that increments once per src element.
    // The host (or a follow-on divide kernel) computes
    //   out[k] = (include_self ? out[k] + sum(src[..])  : sum(src[..])) /
    //            (include_self ? count[k] + 1           : count[k])
    // Per-thread we just emit the atomic add + atomic increment; the
    // post-pass is the caller's responsibility (see _dispatch_scatter_atomic).
    if (idx < 0 || uint(idx) >= pc.out_numel) return;
    c10_vulkan_atomic_add(args.out, uint(idx), (float)args.src[i]);
    uint _orig_count;
    InterlockedAdd(args.count[uint(idx)], 1u, _orig_count);
/*{%*/ endif /*%}*/
}
