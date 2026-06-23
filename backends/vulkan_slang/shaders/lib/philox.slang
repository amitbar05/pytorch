module philox;

// Philox 4x32-10 counter-based RNG. Split out of `helpers.slang` in T2.6
// (2026-05-08). Backward-compat: `helpers.slang` re-exports this module via
// `import philox;` so existing kernels with `import helpers;` keep
// resolving these names.
//
// PF.27 — `_vk_*` aliases match the inline-emitted names in
// `slang_helpers._emit_helpers_impl` so Inductor codegen
// (`overrides.rand` / `overrides.randn` / `overrides.rand_eager` /
// `overrides.randint64`) compiles cleanly under both helper paths:
// the inline body path AND the `import helpers;` fast path.

[ForceInline]
public uint philox_mulhi32(uint a, uint b) {
    uint a_lo = a & 0xFFFFu;
    uint a_hi = a >> 16;
    uint b_lo = b & 0xFFFFu;
    uint b_hi = b >> 16;
    uint p0 = a_lo * b_lo;
    uint p1 = a_lo * b_hi;
    uint p2 = a_hi * b_lo;
    uint p3 = a_hi * b_hi;
    uint carry = ((p0 >> 16) + (p1 & 0xFFFFu) + (p2 & 0xFFFFu)) >> 16;
    return p3 + (p1 >> 16) + (p2 >> 16) + carry;
}

[ForceInline]
public uint2 philox_round(uint2 ctr, uint2 key) {
    uint lo = ctr.x;
    uint hi = ctr.y;
    uint product_lo = lo * 0xD2511F53u;
    uint product_hi = philox_mulhi32(lo, 0xD2511F53u);
    uint2 r;
    r.x = hi ^ key.x ^ product_hi;
    r.y = product_lo;
    return r;
}

[ForceInline]
public uint2 philox_bumpkey(uint2 key) {
    key.x += 0x9E3779B9u;
    key.y += 0xBB67AE85u;
    return key;
}

[ForceInline]
public float philox_rand(uint counter, uint seed) {
    uint2 ctr = uint2(counter, 0);
    uint2 key = uint2(seed, 0);
    [unroll] for (int i = 0; i < 9; i++) {
        ctr = philox_round(ctr, key);
        key = philox_bumpkey(key);
    }
    ctr = philox_round(ctr, key);
    return float(ctr.x) / 4294967296.0f;
}

[ForceInline]
public float philox_randn(uint counter, uint seed) {
    uint2 ctr = uint2(counter, 0);
    uint2 key = uint2(seed, 0);
    [unroll] for (int i = 0; i < 9; i++) {
        ctr = philox_round(ctr, key);
        key = philox_bumpkey(key);
    }
    ctr = philox_round(ctr, key);
    float u1 = max(float(ctr.x) / 4294967296.0f, 1.401298464e-45f);
    float u2 = max(float(ctr.y) / 4294967296.0f, 1.401298464e-45f);
    return sqrt(-2.0f * log(u1)) * cos(6.283185307179586f * u2);
}

[ForceInline]
public float _vk_philox_rand(uint counter, uint seed) {
    return philox_rand(counter, seed);
}

[ForceInline]
public float _vk_philox_randn(uint counter, uint seed) {
    return philox_randn(counter, seed);
}
