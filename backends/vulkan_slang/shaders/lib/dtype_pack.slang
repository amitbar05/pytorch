module dtype_pack;

// IEEE-754 dtype pack/unpack helpers. Split out of `helpers.slang` in T2.6
// (2026-05-08). PyTorch packs bool/uint8 at 1 B/elem, int8/int16 at 1/2 B,
// f16/bf16 at 2 B/elem, but the Vulkan binding layer exposes the buffer as
// `StructuredBuffer<uint>` / `StructuredBuffer<int>` for ABI uniformity.
// These helpers extract / insert the logical element from / to the correct
// word lane.
//
// `helpers.slang` re-exports this module via `import dtype_pack;` so existing
// kernels with `import helpers;` keep resolving these names.

// N+1.15 — [branch] on dtype-conversion if/else chains suppresses
// flatten-by-default penalty. The NaN/Inf/zero/subnormal paths are
// truly divergent between lanes; flattening them into predicated
// selects costs VGPRs for no throughput benefit.
[ForceInline]
public float f16_to_f32(uint h) {
    uint sign = h >> 15u;
    uint exp = (h >> 10u) & 0x1Fu;
    uint mant = h & 0x3FFu;
    [branch] if (exp == 31u) {
        if (mant != 0u) return asfloat(0x7FC00000u);
        return sign != 0u ? asfloat(0xFF800000u) : asfloat(0x7F800000u);
    }
    [branch] if (exp == 0u) {
        if (mant == 0u) return sign != 0u ? -0.0f : 0.0f;
        float f = float(mant) / 1024.0f;
        return sign != 0u ? -(f * (1.0f / 16384.0f)) : (f * (1.0f / 16384.0f));
    }
    return asfloat((sign << 31u) | ((exp + 112u) << 23u) | (mant << 13u));
}

// N+1.15 — [branch] on dtype-conversion if/else chains.
[ForceInline]
public uint f32_to_f16(float f) {
    uint bits = asuint(f);
    uint sign = bits >> 31u;
    int exp = int((bits >> 23u) & 0xFFu) - 127;
    uint mant = bits & 0x7FFFFFu;
    [branch] if (exp == 128 && mant != 0u) return (sign << 15u) | 0x7E00u;
    [branch] if (exp >= 16) return (sign << 15u) | 0x7C00u;
    [branch] if (exp < -24) return (sign << 15u);
    [branch] if (exp < -14) {
        uint sh = uint(-14 - exp);
        return (sign << 15u) | ((mant | 0x800000u) >> (sh + 13u));
    }
    uint h_exp = uint(exp + 15);
    uint h_mant = mant >> 13u;
    uint round = (mant >> 12u) & 1u;
    uint sticky = (mant & 0xFFFu) != 0u ? 1u : 0u;
    if (round != 0u && (sticky != 0u || (h_mant & 1u) != 0u)) h_mant++;
    if (h_mant == 0x400u) { h_mant = 0u; h_exp++; }
    return (sign << 15u) | (h_exp << 10u) | h_mant;
}

[ForceInline]
public float2 unpack_f16x2(uint word) {
    return float2(f16_to_f32(word & 0xFFFFu), f16_to_f32(word >> 16u));
}

[ForceInline]
public uint pack_f16x2(float lo, float hi) {
    return f32_to_f16(lo) | (f32_to_f16(hi) << 16u);
}

[ForceInline]
public float bf16_to_f32(uint h) {
    return asfloat(h << 16u);
}

[ForceInline]
public uint f32_to_bf16(float f) {
    uint bits = asuint(f);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) return (bits >> 16u) | 0x0040u;
    uint lsb = (bits >> 16u) & 1u;
    uint round = (bits >> 15u) & 1u;
    uint sticky = (bits & 0x7FFFu) != 0u ? 1u : 0u;
    bits += (round & (lsb | sticky)) << 15u;
    return bits >> 16u;
}

[ForceInline]
public float2 unpack_bf16x2(uint word) {
    return float2(bf16_to_f32(word & 0xFFFFu), bf16_to_f32(word >> 16u));
}

[ForceInline]
public uint pack_bf16x2(float lo, float hi) {
    return f32_to_bf16(lo) | (f32_to_bf16(hi) << 16u);
}

[ForceInline]
public float _vk_unpack_f16(uint word, uint lane) {
    return f16_to_f32((word >> (lane * 16u)) & 0xFFFFu);
}

[ForceInline]
public uint _vk_pack_f16(float lo, float hi) {
    return f32_to_f16(lo) | (f32_to_f16(hi) << 16u);
}

[ForceInline]
public float _vk_unpack_bf16(uint word, uint lane) {
    return bf16_to_f32((word >> (lane * 16u)) & 0xFFFFu);
}

[ForceInline]
public uint _vk_pack_bf16(float lo, float hi) {
    return f32_to_bf16(lo) | (f32_to_bf16(hi) << 16u);
}

// M30 — extension-method form of the scalar-uint unpackers. Inductor
// codegen now emits `(word).unpack_f16(lane)` instead of
// `_vk_unpack_f16(word, lane)`. The free-function forms above remain as
// transitional aliases. The buffer-form unpackers below take
// `StructuredBuffer<...>` arguments and are intentionally NOT migrated to
// extension methods — Slang extensions on buffer types are awkward and
// rarely match the call-site shape.
public extension uint {
public [ForceInline] float unpack_f16(uint lane) {
    return f16_to_f32((this >> (lane * 16u)) & 0xFFFFu);
}
public [ForceInline] float unpack_bf16(uint lane) {
    return bf16_to_f32((this >> (lane * 16u)) & 0xFFFFu);
}
}

// ── sub-32-bit dtype unpack (P3.8 port from slang_helpers.py) ──────────

[ForceInline]
public float _vk_unpack_u8(StructuredBuffer<uint> buf, uint idx) {
    return float((buf[idx >> 2u] >> ((idx & 3u) << 3u)) & 0xFFu);
}

[ForceInline]
public float _vk_unpack_i8(StructuredBuffer<uint> buf, uint idx) {
    uint b = (buf[idx >> 2u] >> ((idx & 3u) << 3u)) & 0xFFu;
    int s = (b & 0x80u) != 0u ? int(b) - 256 : int(b);
    return float(s);
}

[ForceInline]
public float _vk_unpack_i16(StructuredBuffer<uint> buf, uint idx) {
    uint h = (buf[idx >> 1u] >> ((idx & 1u) << 4u)) & 0xFFFFu;
    int s = (h & 0x8000u) != 0u ? int(h) - 65536 : int(h);
    return float(s);
}

// ── packed16 2D helpers (P3.8 port from slang_helpers.py) ─────────────

[ForceInline]
public float _vk_unpack_f16_2d(RWStructuredBuffer<uint> buf, uint row, uint col, uint stride) {
    uint idx = row * stride + col;
    return f16_to_f32((buf[idx >> 1u] >> ((idx & 1u) * 16u)) & 0xFFFFu);
}

[ForceInline]
public void _vk_pack_f16_2d(RWStructuredBuffer<uint> buf, uint row, uint col, uint stride, float lo, float hi) {
    uint idx = row * stride + col;
    buf[idx >> 1u] = f32_to_f16(lo) | (f32_to_f16(hi) << 16u);
}

[ForceInline]
public float _vk_unpack_bf16_2d(RWStructuredBuffer<uint> buf, uint row, uint col, uint stride) {
    uint idx = row * stride + col;
    return bf16_to_f32((buf[idx >> 1u] >> ((idx & 1u) * 16u)) & 0xFFFFu);
}

[ForceInline]
public void _vk_pack_bf16_2d(RWStructuredBuffer<uint> buf, uint row, uint col, uint stride, float lo, float hi) {
    uint idx = row * stride + col;
    buf[idx >> 1u] = f32_to_bf16(lo) | (f32_to_bf16(hi) << 16u);
}
