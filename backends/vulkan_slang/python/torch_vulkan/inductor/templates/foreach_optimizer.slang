// Foreach optimizer template — B1: Slang interface IOptimizer replaces
// per-algorithm Jinja branches and runtime pc.algorithm switch.
//
// Compile-time specialization via <Algo : IOptimizer> generic parameter.
// The algorithm_type is selected by the entry-point name passed to slangc
// (e.g. "computeMain<AdamWImpl>") — matching the conv/epilogue pattern.
//
// All four Impl structs are always emitted; the entry point picks one.
//
// T4.8: Replaces optimizer_ops.cpp body + sgd_batch*.slang / adamw_batch*.slang
// with a single template parameterized over:
//   batch_size   — number of parameters per dispatch (1, 7, 15, 21, 32)
//   output_dtype — f32, f16, bf16
//   parameter_array — bool (N+1.5.b). When True, emit
//                     ``ParamSlot params[BATCH_SIZE]`` array-of-structs
//                     inside the ParameterBlock and index directly via
//                     ``args.params[param_idx].p[idx]``. When False
//                     (default), fall back to the round-3 layout with
//                     ``param0..paramN-1`` flat fields and a
//                     ``switch (param_idx)`` cascade to dispatch
//                     dynamically.
//
// B1: Algorithm is a COMPILE-TIME generic parameter <Algo : IOptimizer>
// selected by the slangc entry-point name.  No per-line {% if algorithm %}
// branches in the template; no pc.algorithm push-constant field.
//
// Dispatch: 2D grid.  X = element index (256 threads/wg).
//                       Y = parameter index (0..n_params-1).
//
// Push constants carry per-param config (always full 7-float layout).
// Field order MUST match the Python-side ``struct.pack`` sequence in
// ``templates/caller/optimizer.py``.

// ── Optimizer interface ───────────────────────────────────────────────────
// B1: Anti-goal #6 — Slang interface generics, not string-based template params.
interface IOptimizer {
    // Compute one optimizer step for a single element.
    // Returns the new param value.
    // state1 = momentum, state2 = v (AdamW).  Both are updated in-place.
    // Algorithms that don't use a state leave it unchanged.
    static float step(
        float grad, float param,
        inout float state1, inout float state2,
        float lr, float weight_decay,
        float beta1, float beta2, float eps
    );
}

// ── Concrete algorithm implementations ────────────────────────────────────
// All four are always present.  The entry point selects one via
// ``computeMain<AdamWImpl>`` etc.

struct SGDImpl : IOptimizer {
    static float step(
        float grad, float param,
        inout float state1, inout float state2,
        float lr, float weight_decay,
        float beta1, float beta2, float eps
    ) {
        // SGD: p = p - lr * g  (no momentum, no weight decay)
        return param - lr * grad;
    }
};

struct SGD MomentumImpl : IOptimizer {
    static float step(
        float grad, float param,
        inout float state1, inout float state2,
        float lr, float weight_decay,
        float beta1, float beta2, float eps
    ) {
        // SGD+Momentum: buf = μ * buf + g;  p = p - lr * buf
        float buf = beta1 * state1 + grad;
        state1 = buf;
        return param - lr * buf;
    }
};

struct AdamWImpl : IOptimizer {
    static float step(
        float grad, float param,
        inout float state1, inout float state2,
        float lr, float weight_decay,
        float beta1, float beta2, float eps
    ) {
        // AdamW: m = β₁*m + (1-β₁)*g;  v = β₂*v + (1-β₂)*g²
        //        p = p - lr * (m_hat / (√v_hat + ε) + wd * p)
        state1 = beta1 * state1 + (1.0f - beta1) * grad;
        state2 = beta2 * state2 + (1.0f - beta2) * grad * grad;
        return param - lr * (state1 / (sqrt(state2) + eps) + weight_decay * param);
    }
};

struct LionImpl : IOptimizer {
    static float step(
        float grad, float param,
        inout float state1, inout float state2,
        float lr, float weight_decay,
        float beta1, float beta2, float eps
    ) {
        // Lion: update = β₁ * m + (1-β₁) * g
        //       p = p - lr * sign(β₁ * m + (1-β₁) * g)
        //       m = β₂ * m + (1-β₂) * g
        float update = beta1 * state1 + (1.0f - beta1) * grad;
        state1 = beta2 * state1 + (1.0f - beta2) * grad;
        return param - (update > 0.0f ? lr : -lr);
    }
};

// ── Per-param config (packed for GPU alignment) ───────────────────────────
// B1: Always full 7-float layout — no Jinja conditionals.
// Unused fields are zeroed by the Python packer for algorithms that
// don't need them (e.g., SGD leaves momentum/beta2/eps at 0).
struct ParamConfig {
    uint  numel;
    float lr;
    float weight_decay;
    float momentum;   // SGD+Momentum: μ; AdamW/Lion: β₁
    float beta2;      // AdamW/Lion: β₂; SGD: unused
    float eps;        // AdamW: ε; SGD/Lion: unused
    float _pad;       // align to 32B (7 floats = 28B, pad to 32B)
};

struct PushConstants {
    uint n_params;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    ParamConfig cfg[/*{{*/ batch_size /*}}7*/];
};
[[vk::push_constant]] PushConstants pc;

// ── KernelArgs ────────────────────────────────────────────────────────────
// B1: All buffer bindings are always declared — no conditional Momentum/V.
// The Python caller always passes all 4 buffer types per param index.
// Field order is the canonical binding order; the dispatch caller
// passes tensors in the same order.
struct KernelArgs {
/*{%*/ for i in range(batch_size) /*%}*/
    RWStructuredBuffer</*{{*/ output_dtype /*}}float*/> param/*{{*/ i /*}}0*/;
    StructuredBuffer</*{{*/ output_dtype /*}}float*/>   grad/*{{*/ i /*}}0*/;
    RWStructuredBuffer</*{{*/ output_dtype /*}}float*/> momentum/*{{*/ i /*}}0*/;
    RWStructuredBuffer</*{{*/ output_dtype /*}}float*/> v/*{{*/ i /*}}0*/;
/*{%*/ endfor /*%}*/
};
// M21.3.01: explicit Set 0 binding (slangc 2026.7.1 defaults to Set 1)
[[vk::binding(0, 0)]] ParameterBlock<KernelArgs> args;

// ── Entry point ───────────────────────────────────────────────────────────
// B1: Generic over <Algo : IOptimizer>.  The algorithm_type is selected by
// the entry-point name passed to slangc, e.g. ``entry="computeMain<AdamWImpl>"``.
// This follows the same pattern as conv ``computeMain<OpReLU>``.
// All math + state updates happen through Algo::step() with zero
// runtime dispatch overhead (monomorphized at SPIR-V compile time).

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain<Algo : IOptimizer>(
    uint3 gid : SV_GroupID,
    uint3 lid : SV_GroupThreadID
) {
    uint param_idx = gid.y;
    if (param_idx >= pc.n_params) return;

    ParamConfig cfg = pc.cfg[param_idx];
    uint elem = gid.x * 256u + lid.x;
    if (elem >= cfg.numel) return;

    float p, g;
    switch (param_idx) {
/*{%*/ for i in range(batch_size) /*%}*/
        case /*{{*/ i /*}}0*/: p = (float)args.param/*{{*/ i /*}}0*/[elem]; g = (float)args.grad/*{{*/ i /*}}0*/[elem]; break;
/*{%*/ endfor /*%}*/
        default: return;
    }

    // B1: Load momentum/variance buffers UNCONDITIONALLY — the Python
    // caller always provides real or dummy buffers for all algorithms.
    float m_val = 0.0f;
    float v_val = 0.0f;
    switch (param_idx) {
/*{%*/ for i in range(batch_size) /*%}*/
        case /*{{*/ i /*}}0*/: m_val = (float)args.momentum/*{{*/ i /*}}0*/[elem]; v_val = (float)args.v/*{{*/ i /*}}0*/[elem]; break;
/*{%*/ endfor /*%}*/
        default: break;
    }

    // ── B1: Compile-time algorithm dispatch via Slang interface ────────
    // Algo::step() is monomorphized at SPIR-V compile time — zero
    // runtime overhead vs. the old switch(pc.algorithm) approach.
    float p_new = Algo::step(
        g, p, m_val, v_val,
        cfg.lr, cfg.weight_decay,
        cfg.momentum, cfg.beta2, cfg.eps
    );

    // ── Write-back ─────────────────────────────────────────────────────
    // B1: Always write back param, momentum, and v unconditionally.
    // The Python caller provides dummy buffers for unused state slots.
/*{%*/ if parameter_array /*%}*/
    args.params[param_idx].p[elem] = (/*{{*/ output_dtype /*}}float*/)p_new;
    args.params[param_idx].m[elem] = (/*{{*/ output_dtype /*}}float*/)m_val;
    args.params[param_idx].v[elem] = (/*{{*/ output_dtype /*}}float*/)v_val;
/*{%*/ else /*%}*/
    switch (param_idx) {
/*{%*/ for i in range(batch_size) /*%}*/
        case /*{{*/ i /*}}0*/:
            args.param/*{{*/ i /*}}0*/[elem]    = (/*{{*/ output_dtype /*}}float*/)p_new;
            args.momentum/*{{*/ i /*}}0*/[elem] = (/*{{*/ output_dtype /*}}float*/)m_val;
            args.v/*{{*/ i /*}}0*/[elem]        = (/*{{*/ output_dtype /*}}float*/)v_val;
            break;
/*{%*/ endfor /*%}*/
    }
/*{%*/ endif /*%}*/
}
