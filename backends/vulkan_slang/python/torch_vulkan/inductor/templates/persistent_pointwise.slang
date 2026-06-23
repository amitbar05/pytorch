// Persistent pointwise micro-batching template
//
// Grid-stride loop kernel that processes multiple small pointwise
// operations in a single dispatch. One kernel stays resident and loops
// over all N operations.
//
// C6.3 (2026-06-18): OpRange data moved from push constants to a
// StructuredBuffer to avoid the 128B push-constant limit on RDNA1.
// 32 OpRange entries × 24B = 768B, which exceeds maxPushConstantsSize.
// Now only 16B of push constants (total_elements, num_ops,
// num_input_bufs, num_output_bufs).

// Fixed upper bounds for buffer pools
[[vk::constant_id(42)]] const uint MAX_INPUT_BUFFERS = 8u;
[[vk::constant_id(43)]] const uint MAX_OUTPUT_BUFFERS = 8u;
[[vk::constant_id(44)]] const uint THREADS = 64u;
[[vk::constant_id(45)]] const uint MAX_OPS = 32u;

struct OpRange {
    uint start_idx;   // offset into the flat element space
    uint numel;       // number of elements for this op
    uint op_type;     // operation selector (0=identity, 1=relu, 2=sigmoid, ...)
    uint in_buf_id;   // input buffer binding index (0..MAX_INPUT_BUFFERS-1)
    uint out_buf_id;  // output buffer binding index (0..MAX_OUTPUT_BUFFERS-1)
    float scalar;     // scalar argument (e.g. alpha for mul, beta for add)
};

// C6.3: Push constants reduced to 16B (fits in RDNA1 128B limit).
struct PushConstants {
    uint total_elements;   // sum of all numels
    uint num_ops;          // number of OpRange entries
    uint num_input_bufs;   // actual count (<= MAX_INPUT_BUFFERS)
    uint num_output_bufs;  // actual count (<= MAX_OUTPUT_BUFFERS)
};
[[vk::push_constant]] PushConstants pc;

struct KernelArgs {
    StructuredBuffer<float> in_bufs[MAX_INPUT_BUFFERS];
    RWStructuredBuffer<float> out_bufs[MAX_OUTPUT_BUFFERS];
    // C6.3: OpRange data lives in a storage buffer, not push constants.
    StructuredBuffer<OpRange> op_ranges;
};
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

// C6.3: New ops for loss backward support
[ForceInline]
float op_sub(float x, OpRange cfg) {
    // sub: x - scalar.  Uses scalar field for the subtrahend.
    return x - cfg.scalar;
}

[ForceInline]
float op_pow(float x, OpRange cfg) {
    // pow: x ^ scalar.  Uses scalar field for the exponent.
    return pow(x, cfg.scalar);
}

[ForceInline]
float op_fill(float x, OpRange cfg) {
    // fill: output = scalar (ignore input x)
    return cfg.scalar;
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
        case 7u: return op_sub(x, cfg);
        case 8u: return op_pow(x, cfg);
        case 9u: return op_fill(x, cfg);
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
            OpRange cfg = args.op_ranges[o];
            if (i >= cfg.start_idx && i < cfg.start_idx + cfg.numel) {
                op_idx = o;
                offset = cfg.start_idx;
                break;
            }
        }
        if (op_idx >= pc.num_ops) continue;  // shouldn't happen

        OpRange cfg = args.op_ranges[op_idx];
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
