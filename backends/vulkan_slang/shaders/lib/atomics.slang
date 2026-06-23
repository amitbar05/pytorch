module atomics;

// Float-atomic-add via CAS loop. Vulkan/SPIR-V on RDNA1 has no native
// float atomics, so float-add reinterprets the buffer as uint and uses
// InterlockedCompareExchange. The eager-backend shaders had this inlined
// in three places (index_add, scatter_add, embedding_backward_atomic);
// PF.21 hoists it here so future atomic-scatter lowerings (PF.22 /
// PF.25) compose without re-deriving the pattern.

// ─── M20.8: IScatter — Anti-goal #6 generic interface ─────────────────────
//
// Replaces the Jinja `{% if operation == "scatter_add" %}...{% elif %}...`
// chain in `templates/scatter_atomic.py.jinja` with a Slang generic
// `<S : IScatter>` parameterisation.  The Jinja `operation` string is now
// only used to select the concrete struct name (scatter_struct), not to
// branch inside the shader body.
//
// Interface contract: `combine` writes one value into a uint-view SSBO.
// All scatter atomic operations share the same buffer type
// (RWStructuredBuffer<uint> — the CAS trick that reinterprets float bits
// as uint for InterlockedCompareExchange) so the interface is uniform.
// The `scatter_reduce_mean` count-increment remains Jinja-gated (is_mean)
// because it binds a *second* buffer — a structural difference, not an
// operational one.
public interface IScatter {
    static void combine(RWStructuredBuffer<uint> out, uint idx, float val);
};

// scatter_add / index_put_accumulate — atomic float-add
public struct ScatterAdd : IScatter {
    public static void combine(RWStructuredBuffer<uint> out, uint idx, float val) {
        vk_atomic_add(out, idx, val);
    }
};

// scatter_reduce(reduce='amax') — atomic float max
public struct ScatterMax : IScatter {
    public static void combine(RWStructuredBuffer<uint> out, uint idx, float val) {
        atomic_max_f32(out, idx, val);
    }
};

// scatter_reduce(reduce='amin') — atomic float min
public struct ScatterMin : IScatter {
    public static void combine(RWStructuredBuffer<uint> out, uint idx, float val) {
        atomic_min_f32(out, idx, val);
    }
};

// scatter_reduce(reduce='prod') — atomic float multiply
public struct ScatterProd : IScatter {
    public static void combine(RWStructuredBuffer<uint> out, uint idx, float val) {
        atomic_mul_f32(out, idx, val);
    }
};

// scatter_reduce(reduce='mean') — same as ScatterAdd for the value slot;
// the count-buffer increment is handled separately (is_mean Jinja gate).
public struct ScatterMean : IScatter {
    public static void combine(RWStructuredBuffer<uint> out, uint idx, float val) {
        vk_atomic_add(out, idx, val);
    }
};

// Inductor codegen emits atomic-scatter stores into a
// `RWStructuredBuffer<uint>` (the dest tensor reinterpreted as raw u32).
// PF.21.b retires `slang_helpers._emit_helpers_impl`'s inline copy of
// this helper — kernels now `import atomics;` and call this version.
//
// T2.10b — `vk_atomic_add` is the new canonical name; `c10_vulkan_atomic_add`
// is preserved as a [ForceInline] alias because Group D's
// `templates/scatter_atomic.py.jinja` (out of T2.10b's edit scope) still
// emits the old name, and a regression test asserts the symbol exists in
// `lib/atomics.slang`. New code should call `vk_atomic_add` directly.
[ForceInline]
public void vk_atomic_add(RWStructuredBuffer<uint> buf, uint idx, float value) {
    uint expected, desired, original;
    [loop] do {
        expected = buf[idx];
        desired = asuint(asfloat(expected) + value);
        InterlockedCompareExchange(buf[idx], expected, desired, original);
    } while (original != expected);
    // RDNA1 memory model: ensure the successful store is visible to
    // other wavefronts before they read. Without this, concurrent
    // CAS loops on the same address can observe stale values and
    // silently drop contributions, corrupting dW/dX accumulators.
    DeviceMemoryBarrier();
}

[ForceInline]
public void c10_vulkan_atomic_add(RWStructuredBuffer<uint> buf, uint idx, float value) {
    vk_atomic_add(buf, idx, value);
}

[ForceInline]
public void atomic_add_f32(RWByteAddressBuffer buf, uint byte_offset, float value) {
    uint expected = buf.Load(byte_offset);
    [loop]
    for (;;) {
        uint desired = asuint(asfloat(expected) + value);
        uint original;
        buf.InterlockedCompareExchange(byte_offset, expected, desired, original);
        if (original == expected) break;
        expected = original;
    }
}

// Integer atomics map to native HLSL/SPIR-V intrinsics — no CAS needed.
[ForceInline]
public void atomic_add_i32(RWByteAddressBuffer buf, uint byte_offset, int value) {
    uint original;
    buf.InterlockedAdd(byte_offset, asuint(value), original);
}

[ForceInline]
public void atomic_add_u32(RWByteAddressBuffer buf, uint byte_offset, uint value) {
    uint original;
    buf.InterlockedAdd(byte_offset, value, original);
}

// ─── T4.11: float atomic max / min / mul (CAS loops) ──────────────────────
//
// Vulkan/SPIR-V on RDNA1 has no native float atomics for max/min/mul, so
// these helpers reinterpret the buffer as raw uint and use the same
// InterlockedCompareExchange CAS loop pattern as `vk_atomic_add`.
// They're consumed by `templates/scatter_atomic.py.jinja` for
// `aten.scatter_reduce_(reduce='amax'|'amin'|'prod'|'mean')` (mean is sum +
// post-pass divide and so reuses `vk_atomic_add` plus a count
// buffer; the atomic count increment uses `InterlockedAdd` on a separate
// `RWStructuredBuffer<uint>`).
//
// NaN handling: HLSL/Slang `max(NaN, x) = NaN` and `min(NaN, x) = NaN`
// on most GPUs (true on AMD/RDNA1 — `v_max_f32` returns NaN if either
// operand is NaN). PyTorch's CPU `scatter_reduce(amax)` propagates NaN
// the same way, so we don't need an explicit NaN guard.
//
// Early-out: the max/min variants check `val <= prev` / `val >= prev`
// before the CAS. This is purely a performance optimization (skips the
// CAS round-trip when the candidate is uninteresting); correctness only
// requires the inner CAS loop to converge. We do NOT early-out on prod
// because the multiplied value can be smaller in magnitude (signed
// floats) and there's no monotone guard.
[ForceInline]
public void atomic_max_f32(RWStructuredBuffer<uint> buf, uint idx, float val) {
    uint cur = buf[idx];
    [loop] for (;;) {
        float prev_f = asfloat(cur);
        if (val <= prev_f) return;  // already greater; no update needed
        uint new_bits = asuint(max(prev_f, val));
        uint expected = cur;
        InterlockedCompareExchange(buf[idx], expected, new_bits, cur);
        if (cur == expected) break;
    }
}

[ForceInline]
public void atomic_min_f32(RWStructuredBuffer<uint> buf, uint idx, float val) {
    uint cur = buf[idx];
    [loop] for (;;) {
        float prev_f = asfloat(cur);
        if (val >= prev_f) return;  // already smaller; no update needed
        uint new_bits = asuint(min(prev_f, val));
        uint expected = cur;
        InterlockedCompareExchange(buf[idx], expected, new_bits, cur);
        if (cur == expected) break;
    }
}

[ForceInline]
public void atomic_mul_f32(RWStructuredBuffer<uint> buf, uint idx, float val) {
    uint cur = buf[idx];
    [loop] for (;;) {
        float prev_f = asfloat(cur);
        uint new_bits = asuint(prev_f * val);
        uint expected = cur;
        InterlockedCompareExchange(buf[idx], expected, new_bits, cur);
        if (cur == expected) break;
    }
}

// Packed-half atomic add: each 32-bit word holds two f16 lanes. Update one
// lane via CAS while preserving the other. `lane` selects 0 (low 16 bits)
// or 1 (high 16 bits). The float input is rounded to f16 once, then folded
// in repeatedly — the helper round-trip lives in helpers.f32_to_f16.
[ForceInline]
public void atomic_add_f16_packed(
    RWByteAddressBuffer buf, uint word_byte_offset, uint lane, float value
) {
    uint shift = lane * 16u;
    uint mask = 0xFFFFu << shift;
    uint expected = buf.Load(word_byte_offset);
    [loop]
    for (;;) {
        uint half_bits = (expected >> shift) & 0xFFFFu;
        // Bit-extend f16 → f32 via the well-known formula: (sign|exp|mant) →
        // (sign| (exp+112) | mant<<13). Inlined here so atomics is
        // self-contained (no `import helpers;` chain forced on consumers).
        float f;
        {
            uint h = half_bits;
            uint sign = h >> 15u;
            uint exp = (h >> 10u) & 0x1Fu;
            uint mant = h & 0x3FFu;
            if (exp == 31u) {
                f = (mant != 0u) ? asfloat(0x7FC00000u)
                                 : (sign != 0u ? asfloat(0xFF800000u) : asfloat(0x7F800000u));
            } else if (exp == 0u && mant == 0u) {
                f = sign != 0u ? -0.0f : 0.0f;
            } else {
                uint bits = (sign << 31u) | ((exp + 112u) << 23u) | (mant << 13u);
                f = asfloat(bits);
            }
        }
        f += value;
        // f32 → f16 round-to-nearest-even (IEEE 754 binary16).
        uint new_h;
        {
            uint bits = asuint(f);
            uint sign = bits >> 31u;
            int exp = int((bits >> 23u) & 0xFFu) - 127;
            uint mant = bits & 0x7FFFFFu;
            if (exp == 128 && mant != 0u) {
                new_h = (sign << 15u) | 0x7E00u;
            } else if (exp >= 16) {
                new_h = (sign << 15u) | 0x7C00u;
            } else if (exp < -24) {
                new_h = (sign << 15u);
            } else if (exp < -14) {
                uint sh = uint(-14 - exp);
                new_h = (sign << 15u) | ((mant | 0x800000u) >> (sh + 13u));
            } else {
                uint h_exp = uint(exp + 15);
                uint h_mant = mant >> 13u;
                uint round = (mant >> 12u) & 1u;
                uint sticky = (mant & 0xFFFu) != 0u ? 1u : 0u;
                if (round != 0u && (sticky != 0u || (h_mant & 1u) != 0u)) h_mant++;
                if (h_mant == 0x400u) { h_mant = 0u; h_exp++; }
                new_h = (sign << 15u) | (h_exp << 10u) | h_mant;
            }
        }
        uint desired = (expected & ~mask) | ((new_h & 0xFFFFu) << shift);
        uint original;
        buf.InterlockedCompareExchange(word_byte_offset, expected, desired, original);
        if (original == expected) break;
        expected = original;
    }
}
