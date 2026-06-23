// ═══════════════════════════════════════════════════════════════════════════
// CG.M12 — Generic pointwise kernel family.
//
// ONE source file, ONE set of bindings, FOUR generic entry points.
// Uses ByteAddressBuffer so the binding layout is identical for unary,
// binary, real, and complex — buf_in1 is simply unused for unary ops.
// The concrete op struct is resolved at SPIR-V compile time via slangc's
// generic specialization (e.g. -entry "computeMain<OpAbs>").
//
// Anti-goal #6: no string-based template parameters — Slang generics only.
// ═══════════════════════════════════════════════════════════════════════════

module pointwise_generic;
import pointwise;

// ── Unified push constants and buffer bindings ────────────────────────────
// CG.M14: ParameterBlock<KernelArgs> groups all buffer bindings into a
// single descriptor table. Slang auto-assigns binding indices in
// struct-field order (buf_in0=0, buf_in1=1, buf_out=2), matching the
// dispatch order in generic_pointwise_dispatch.py.

struct PC { uint numel; };
[[vk::push_constant]] PC pc;

struct KernelArgs {
    ByteAddressBuffer buf_in0;
    ByteAddressBuffer buf_in1;
    RWByteAddressBuffer buf_out;
};
[[vk::binding(0, 0)]] ParameterBlock<KernelArgs> args;

// ── Unary float entry point ───────────────────────────────────────────────

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain<Op : IPointwise>(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.numel) return;
    float val = args.buf_in0.Load<float>(tid.x * 4);
    float result = Op::apply(val);
    args.buf_out.Store<float>(tid.x * 4, result);
}

// ── Binary float entry point ──────────────────────────────────────────────

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain<Op : IPointwiseBinary>(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.numel) return;
    float a = args.buf_in0.Load<float>(tid.x * 4);
    float b = args.buf_in1.Load<float>(tid.x * 4);
    float result = Op::apply(a, b);
    args.buf_out.Store<float>(tid.x * 4, result);
}

// ── Unary complex (float2) entry point ────────────────────────────────────

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain<Op : IComplexPointwise>(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.numel) return;
    float2 val = args.buf_in0.Load<float2>(tid.x * 8);
    float2 result = Op::apply(val);
    args.buf_out.Store<float2>(tid.x * 8, result);
}

// ── Binary complex (float2) entry point ───────────────────────────────────

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain<Op : IComplexPointwiseBinary>(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.numel) return;
    float2 a = args.buf_in0.Load<float2>(tid.x * 8);
    float2 b = args.buf_in1.Load<float2>(tid.x * 8);
    float2 result = Op::apply(a, b);
    args.buf_out.Store<float2>(tid.x * 8, result);
}
