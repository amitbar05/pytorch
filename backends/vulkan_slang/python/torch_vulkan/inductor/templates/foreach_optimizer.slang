// Foreach optimizer template — M8: runtime algorithm selector replaces Jinja branches.
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
// M8: Algorithm is now a RUNTIME push-constant selector (uint algorithm)
// instead of a Jinja conditional.  The kernel body uses a switch on
// ``pc.algorithm`` (ALG_* constants below) at lines ~160–240.
//
//   algorithm == 0: SGD (no momentum, no weight decay)
//   algorithm == 1: SGD+Momentum (with optional weight decay)
//   algorithm == 2: AdamW
//   algorithm == 3: Lion
//
// This matches M-SF.4 (slang_conv_bwd.slang) — always declare all buffers,
// gate usage at runtime.
//
// Dispatch: 2D grid.  X = element index (256 threads/wg).
//                       Y = parameter index (0..n_params-1).
//
// Push constants carry per-param config + algorithm selector.
// Field order MUST match the Python-side ``struct.pack`` sequence in
// ``templates/caller/optimizer.py`` (around line 173-193).

// Per-param config (packed for GPU alignment).
// M8: Always full 7-float layout — no Jinja conditionals.
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
    uint algorithm;   // M8: runtime algorithm selector (see ALG_* below)
    uint _pad0;
    uint _pad1;
    ParamConfig cfg[/*{{*/ batch_size /*}}7*/];
};
[[vk::push_constant]] PushConstants pc;

// M8: All buffer bindings live in this ParameterBlock.
// Unlike the old Jinja-conditional layout, ALL buffers are always
// declared — the shader gates reads/writes at runtime.
// Field order is the canonical binding order; the dispatch caller
// passes tensors in the same order.  No `[[vk::binding(N)]]` literals
// appear anywhere in this template.
// N+1.10 flat layout — one binding per buffer
// (descriptorCount=1), per-param dynamic dispatch via switch cascade.
// NOTE: N+1.5.b array-of-structs was removed because Vulkan spec
// requires array indices to be dynamically uniform; `param_idx` from
// SV_GroupID.y is NOT uniform across the grid and the driver would
// need to broadcast. The round-3 switch cascade is the portable path.
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

// M8: Algorithm selector constants (must match Python-side enum).
#define ALG_SGD            0u
#define ALG_SGD_MOMENTUM   1u
#define ALG_ADAMW          2u
#define ALG_LION           3u

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
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

    // M8: Load momentum/variance buffers unconditionally — the algorithm
    // selector below gates whether they are used.  This matches M-SF.4.
    float m_val = 0.0f;
    float v_val = 0.0f;
    switch (param_idx) {
/*{%*/ for i in range(batch_size) /*%}*/
        case /*{{*/ i /*}}0*/: m_val = (float)args.momentum/*{{*/ i /*}}0*/[elem]; v_val = (float)args.v/*{{*/ i /*}}0*/[elem]; break;
/*{%*/ endfor /*%}*/
        default: break;
    }

    // M8: Runtime algorithm dispatch via push-constant selector.
    // This replaces the old per-algorithm Jinja branches.
    float p_new;
    switch (pc.algorithm) {
        case ALG_SGD: {
            // SGD: p = p - lr * g  (no weight decay in pure SGD)
            p_new = p - cfg.lr * g;
            break;
        }
        case ALG_SGD_MOMENTUM: {
            // SGD+Momentum: buf = μ * buf + g;  p = p - lr * buf
            float buf = cfg.momentum * m_val + g;
            p_new = p - cfg.lr * buf;
            m_val = buf;  // write back momentum
            break;
        }
        case ALG_ADAMW: {
            // AdamW: m = β₁*m + (1-β₁)*g;  v = β₂*v + (1-β₂)*g²
            //        p = p - lr * (m_hat / (√v_hat + ε) + wd * p)
            m_val = cfg.momentum * m_val + (1.0f - cfg.momentum) * g;
            v_val = cfg.beta2 * v_val + (1.0f - cfg.beta2) * g * g;
            p_new = p - cfg.lr * (m_val / (sqrt(v_val) + cfg.eps) + cfg.weight_decay * p);
            break;
        }
        case ALG_LION: {
            // Lion: update = β₁ * m + (1-β₁) * g
            //       p = p - lr * sign(β₁ * m + (1-β₁) * g)
            //       m = β₂ * m + (1-β₂) * g
            float update = cfg.momentum * m_val + (1.0f - cfg.momentum) * g;
            m_val = cfg.beta2 * m_val + (1.0f - cfg.beta2) * g;
            p_new = p - (update > 0.0f ? cfg.lr : -cfg.lr);
            break;
        }
        default: {
            // Fallback: plain SGD if algorithm selector is invalid
            p_new = p - cfg.lr * g;
            break;
        }
    }

/*{%*/ if parameter_array /*%}*/
    args.params[param_idx].p[elem] = (/*{{*/ output_dtype /*}}float*/)p_new;
    // M8: Write back momentum/variance only when algorithm uses them.
    // The runtime check avoids unnecessary memory writes for SGD.
    if (pc.algorithm == ALG_SGD_MOMENTUM || pc.algorithm == ALG_ADAMW || pc.algorithm == ALG_LION) {
        args.params[param_idx].m[elem] = (/*{{*/ output_dtype /*}}float*/)m_val;
    }
    if (pc.algorithm == ALG_ADAMW) {
        args.params[param_idx].v[elem] = (/*{{*/ output_dtype /*}}float*/)v_val;
    }
/*{%*/ else /*%}*/
    switch (param_idx) {
/*{%*/ for i in range(batch_size) /*%}*/
        case /*{{*/ i /*}}0*/: args.param/*{{*/ i /*}}0*/[elem] = (/*{{*/ output_dtype /*}}float*/)p_new; break;
/*{%*/ endfor /*%}*/
    }
    // M8: Write back momentum/variance only when algorithm uses them.
    if (pc.algorithm == ALG_SGD_MOMENTUM || pc.algorithm == ALG_ADAMW || pc.algorithm == ALG_LION) {
        switch (param_idx) {
/*{%*/ for i in range(batch_size) /*%}*/
            case /*{{*/ i /*}}0*/: args.momentum/*{{*/ i /*}}0*/[elem] = (/*{{*/ output_dtype /*}}float*/)m_val; break;
/*{%*/ endfor /*%}*/
        }
    }
    if (pc.algorithm == ALG_ADAMW) {
        switch (param_idx) {
/*{%*/ for i in range(batch_size) /*%}*/
            case /*{{*/ i /*}}0*/: args.v/*{{*/ i /*}}0*/[elem] = (/*{{*/ output_dtype /*}}float*/)v_val; break;
/*{%*/ endfor /*%}*/
        }
    }
/*{%*/ endif /*%}*/
}
