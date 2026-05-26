module bucket;

// Bucketize + buffer-form vec4 horizontal reductions. Split out of
// `helpers.slang` in T2.6 (2026-05-08). All four take a
// `StructuredBuffer<float>` self argument, so they intentionally retain
// free-function shape — Slang extensions on `StructuredBuffer<T>` would
// invert the API in ways that don't match the Inductor codegen call shape.
// Backward-compat: `helpers.slang` re-exports this module via
// `import bucket;` so existing kernels with `import helpers;` keep
// resolving these names.

// ── bucketize — binary search over sorted boundary buffer (P3.8 port) ─
// M30 — INTENTIONALLY NOT MIGRATED to extension form. The signature takes
// a `StructuredBuffer<float> data` plus four scalar args (lo, hi, stride,
// right). Slang extensions on `StructuredBuffer<T>` would force the
// boundary buffer to be `this` while the value being bucketized becomes a
// parameter — the resulting `(boundaries).bucketize(val, lo, hi, stride, right)`
// reads worse than the free-function form. Kept as a free function.
// T2.10b — renamed to `vk_bucketize`; `c10_vulkan_bucketize` retained as a
// thin alias for any out-of-scope caller still using the legacy name.

[ForceInline]
public uint vk_bucketize(float val, uint lo, uint hi,
                         StructuredBuffer<float> data, uint stride,
                         bool right) {
    while (lo < hi) {
        uint mid = (lo + hi) >> 1u;
        float bval = data[mid * stride];
        if (right) {
            if (val < bval) hi = mid; else lo = mid + 1u;
        } else {
            if (val <= bval) hi = mid; else lo = mid + 1u;
        }
    }
    return lo;
}

[ForceInline]
public uint c10_vulkan_bucketize(float val, uint lo, uint hi,
                                StructuredBuffer<float> data, uint stride,
                                bool right) {
    return vk_bucketize(val, lo, hi, data, stride, right);
}

// ── vec4_reduce_* — horizontal ops on 4-element buffer windows (P3.8) ─
// M30 — buffer-form (`StructuredBuffer<float> buf, uint base`) variants are
// INTENTIONALLY NOT MIGRATED to extension methods. Slang extensions on
// `StructuredBuffer<float>` would invert the API to
// `(buf).vec4_hsum(base)` while the existing call site in
// `kernel/pointwise.py` already passes the buffer + base as positional
// args. Keeping free functions matches the codegen call shape one-to-one.
// The float4-self extension methods (`(v).hsum()` etc.) live in
// `special_math.slang` for callers that already have the four lanes loaded
// into a register; new code paths should prefer them.
// T2.10b — renamed to `vk_vec4_h{sum,max,min,prod}`; the legacy
// `c10_vulkan_vec4_h*` names are kept as transitional [ForceInline] aliases.

[ForceInline]
public float vk_vec4_hsum(StructuredBuffer<float> buf, uint base) {
    return buf[base] + buf[base + 1u] + buf[base + 2u] + buf[base + 3u];
}

[ForceInline]
public float vk_vec4_hmax(StructuredBuffer<float> buf, uint base) {
    return max(max(buf[base], buf[base + 1u]), max(buf[base + 2u], buf[base + 3u]));
}

[ForceInline]
public float vk_vec4_hmin(StructuredBuffer<float> buf, uint base) {
    return min(min(buf[base], buf[base + 1u]), min(buf[base + 2u], buf[base + 3u]));
}

[ForceInline]
public float vk_vec4_hprod(StructuredBuffer<float> buf, uint base) {
    return buf[base] * buf[base + 1u] * buf[base + 2u] * buf[base + 3u];
}

[ForceInline]
public float c10_vulkan_vec4_hsum(StructuredBuffer<float> buf, uint base) {
    return vk_vec4_hsum(buf, base);
}

[ForceInline]
public float c10_vulkan_vec4_hmax(StructuredBuffer<float> buf, uint base) {
    return vk_vec4_hmax(buf, base);
}

[ForceInline]
public float c10_vulkan_vec4_hmin(StructuredBuffer<float> buf, uint base) {
    return vk_vec4_hmin(buf, base);
}

[ForceInline]
public float c10_vulkan_vec4_hprod(StructuredBuffer<float> buf, uint base) {
    return vk_vec4_hprod(buf, base);
}

// ── N.1.b-fast — GPU-native searchsorted ───────────────────────────────
// Parallel binary search: each thread computes one output element.
// No shared memory, no barriers — pure compute, log2(seq_len) iterations.
// This replaces the CPU-roundtrip in lowerings/searchsorted.py.
//
// Usage: dispatch with grid = ceil(num_values / 256), push constants
// carrying (seq_len, num_values, right).

struct SearchsortedPC {
    uint seq_len;
    uint num_values;
    uint right;  // boolean: 0 = left-most, 1 = right-most
};

[[vk::push_constant]] SearchsortedPC searchsorted_pc;

struct SearchsortedArgs {
    StructuredBuffer<float> sorted_seq;
    StructuredBuffer<float> values;
    RWStructuredBuffer<uint> output;
};
[[vk::binding(0, 0)]] ParameterBlock<SearchsortedArgs> searchsorted_args;

[shader("compute")]
[numthreads(256, 1, 1)]
void searchsorted(uint3 gtid : SV_DispatchThreadID) {
    uint tid = gtid.x;
    if (tid >= searchsorted_pc.num_values) {
        return;
    }

    float val = searchsorted_args.values[tid];
    uint lo = 0u;
    uint hi = searchsorted_pc.seq_len;

    bool right = searchsorted_pc.right != 0u;

    // Binary search: log2(seq_len) iterations
    while (lo < hi) {
        uint mid = (lo + hi) >> 1u;
        float bval = searchsorted_args.sorted_seq[mid];
        if (right ? bval <= val : bval < val) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    searchsorted_args.output[tid] = lo;
}
