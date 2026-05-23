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
// M20.8 — Anti-goal #6 sweep:
//   Atomic operations (scatter_add, index_put_accumulate, scatter_reduce_*)
//   are now dispatched via the Slang generic `<S : IScatter>` interface
//   declared in `shaders/lib/atomics.slang`.  The Jinja `operation` string
//   selects the concrete struct name (scatter_struct) at render time, but
//   there is no per-operation if/elif chain inside the shader body for
//   atomic ops — only `ScatterXxx::combine(args.out, idx, val)`.
//   Structural differences (gather vs scatter buffer layout, mean's count
//   buffer) remain Jinja-gated because they affect the struct layout.
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
/*# M20.8: Map operation name to the concrete IScatter struct.            #*/
/*# Non-atomic operations (gather, scatter, index_put) leave this unused. #*/
/*{%*/ set _scatter_struct_map = {
    "scatter_add":           "ScatterAdd",
    "index_put_accumulate":  "ScatterAdd",
    "scatter_reduce_amax":   "ScatterMax",
    "scatter_reduce_amin":   "ScatterMin",
    "scatter_reduce_prod":   "ScatterProd",
    "scatter_reduce_mean":   "ScatterMean",
} /*%}*/
/*{%*/ set scatter_struct = _scatter_struct_map.get(operation, "ScatterAdd") /*%}*/
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
// ── Import the canonical atomic helpers from lib/atomics.slang (M20.8).
// PF.21.b: the precompiled module carries the correct SPIR-V decorations
// (Coherent) for InterlockedCompareExchange to work on Vulkan drivers.
// M20.8: also declares IScatter + ScatterAdd/Max/Min/Prod/Mean structs.
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
/*{%*/ elif operation == "index_put" /*%}*/
    // index_put: out[indices[i]] = values[i]  (last-write-wins)
    if (idx < 0 || uint(idx) >= pc.out_numel) return;
    args.out[uint(idx)] = asuint(args.src[i]);
/*{%*/ elif is_atomic /*%}*/
    // M20.8 — Anti-goal #6: dispatch via IScatter generic instead of
    // per-operation if/elif chain.  The concrete struct (/*{{*/ scatter_struct /*}}ScatterAdd*/)
    // is substituted at Jinja render time; the shader body is operation-agnostic.
    if (idx < 0 || uint(idx) >= pc.out_numel) return;
    /*{{*/ scatter_struct /*}}ScatterAdd*/::combine(args.out, uint(idx), (float)args.src[i]);
/*{%*/ if is_mean /*%}*/
    // scatter_reduce(reduce='mean'): companion uint-counter SSBO gets one
    // atomic increment per landed element.  The host post-pass divides
    //   out[k] / count[k] to compute the mean.  This increment is
    // Jinja-gated (is_mean) because it requires the extra `count` buffer
    // declared in KernelArgs — a structural difference, not an op difference.
    uint _orig_count;
    InterlockedAdd(args.count[uint(idx)], 1u, _orig_count);
/*{%*/ endif /*%}*/
/*{%*/ endif /*%}*/
}
