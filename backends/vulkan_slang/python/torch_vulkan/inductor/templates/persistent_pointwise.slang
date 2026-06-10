// Persistent pointwise micro-batching template (.slang migration from .py.jinja)
//
// Grid-stride loop kernel that processes multiple small pointwise
// operations in a single dispatch. One kernel stays resident and loops
// over all N operations.
//
// M-SF.3: buffer pools use fixed compile-time upper bounds (MAX_*)
// instead of Jinja {% for %} loops, which Slang cannot express in
// struct definitions. The runtime loop iterates only over pc.num_ops.
// Unused buffer slots are simply not referenced.

// M-SF.3: fixed upper bounds for buffer pools
[[vk::constant_id(42)]] const uint MAX_INPUT_BUFFERS = 8u;
[[vk::constant_id(43)]] const uint MAX_OUTPUT_BUFFERS = 8u;
[[vk::constant_id(44)]] const uint THREADS = 64u;

struct OpRange {
    uint start_idx;   // offset into the flat element space
    uint numel;       // number of elements for this op
    uint op_type;     // operation selector (0=copy, 1=relu, 2=sigmoid, ...)
    uint in_buf_id;   // input buffer binding index (0..MAX_INPUT_BUFFERS-1)
    uint out_buf_id;  // output buffer binding index (0..MAX_OUTPUT_BUFFERS-1)
    float scalar;     // scalar argument (e.g. alpha for mul, beta for add)
};

struct PushConstants {
    uint total_elements;   // sum of all numels
    uint num_ops;          // number of OpRange entries
    uint num_input_bufs;   // actual count (<= MAX_INPUT_BUFFERS)
    uint num_output_bufs;  // actual count (<= MAX_OUTPUT_BUFFERS)
    OpRange ranges[32];    // upper bound: 32 op ranges per dispatch
};
[[vk::push_constant]] PushConstants pc;

struct KernelArgs {
    StructuredBuffer<float> in_bufs[MAX_INPUT_BUFFERS];
    RWStructuredBuffer<float> out_bufs[MAX_OUTPUT_BUFFERS];
};
// M21.3.01: explicit Set 0 binding (slangc 2026.7.1 defaults to Set 1)
[[vk::binding(0, 0)]] ParameterBlock<KernelArgs> args;

// Operation implementations

[ForceInline]
float op_relu(float x, OpRange cfg) {
    return max(x, 0.0f);
}

[ForceInline]
float op_sigmoid(float x, OpRange cfg) {
    return 1.0f / (1.0f + exp(-x));
}

[ForceInline]
float op_tanh(float x, OpRange cfg) {
    return tanh(x);
}

[ForceInline]
float op_gelu_tanh(float x, OpRange cfg) {
    float sqrt_2_over_pi = 0.7978845608f;
    float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + tanh(inner));
}

[ForceInline]
float op_add_scalar(float x, OpRange cfg) {
    return x + cfg.scalar;
}

[ForceInline]
float op_mul_scalar(float x, OpRange cfg) {
    return x * cfg.scalar;
}

[ForceInline]
float op_identity(float x, OpRange cfg) {
    return x;
}

// Dispatch helper: apply a single op to one element
[ForceInline]
float apply_op(float x, OpRange cfg) {
    switch (cfg.op_type) {
        case 0u: return op_identity(x, cfg);
        case 1u: return op_relu(x, cfg);
        case 2u: return op_sigmoid(x, cfg);
        case 3u: return op_tanh(x, cfg);
        case 4u: return op_gelu_tanh(x, cfg);
        case 5u: return op_add_scalar(x, cfg);
        case 6u: return op_mul_scalar(x, cfg);
        default: return x;
    }
}

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint tid = gid.x * THREADS + lid.x;
    uint total_elements = pc.total_elements;

    // Grid-stride loop: each thread processes multiple elements,
    // potentially spanning multiple operations.
    for (uint i = tid; i < total_elements; i += THREADS * get_group_count().x) {
        // Find which operation this element belongs to (linear scan
        // over OpRange entries — small N, overhead negligible).
        uint offset = 0u;
        uint op_idx = pc.num_ops;  // sentinel
        for (uint o = 0u; o < pc.num_ops; o++) {
            if (i >= pc.ranges[o].start_idx && i < pc.ranges[o].start_idx + pc.ranges[o].numel) {
                op_idx = o;
                offset = pc.ranges[o].start_idx;
                break;
            }
        }
        if (op_idx >= pc.num_ops) continue;  // shouldn't happen

        OpRange cfg = pc.ranges[op_idx];
        uint local_i = i - offset;

        // Read input (runtime dispatch on in_buf_id)
        float x = 0.0f;
        if (cfg.in_buf_id < pc.num_input_bufs) {
            x = args.in_bufs[cfg.in_buf_id][local_i];
        }

        // Apply operation
        float y = apply_op(x, cfg);

        // Write output (runtime dispatch on out_buf_id)
        if (cfg.out_buf_id < pc.num_output_bufs) {
            args.out_bufs[cfg.out_buf_id][local_i] = y;
        }
    }
}
