// mm_int8.slang — Tiled int8 matrix multiplication for inference.
// OP.24 / M14: Int8 weights × int8 activations → int32 accumulation → float32 output.
//
// Each uint32 in the StructuredBuffer contains 4 packed int8 values (little-endian).
// The shader unpacks at load time, accumulates in int32 to avoid overflow,
// and converts to float32 on store.
//
// Tile sizes: static const int (link-time specializable). Sensible defaults
// allow standalone compilation for syntax-check; importing wrappers override
// by defining the same constants BEFORE `import mm_int8;`.
//
// Usage (precompile once):
//   slangc mm_int8.slang -emit-ir -o mm_int8.slang-module
//
// Usage (specialize at link time):
//   slangc wrapper.slang -I <module_dir> -o kernel.spv
//   where wrapper.slang defines `static const int TILE_M = 64;` etc.
//   BEFORE `import mm_int8;`.

module mm_int8;

// ── Link-time specialization constants ──────────────────────────────────
// Sensible defaults allow standalone compilation for syntax-check and
// `-emit-ir`.  Importing wrappers override these at link time by defining
// the same constants BEFORE the `import mm_int8;` statement.
static const int TILE_M = 64;
static const int TILE_N = 64;
static const int TILE_K = 16;
static const int M_PER_THREAD = 4;
static const int N_PER_THREAD = 4;

// ── Derived constants ───────────────────────────────────────────────────
static const int WG_M = TILE_M / M_PER_THREAD;
static const int WG_N = TILE_N / N_PER_THREAD;
static const int TILE_A_ELEMS = TILE_M * TILE_K;
static const int TILE_B_ELEMS = TILE_K * TILE_N;
static const int WG_SIZE = WG_M * WG_N;

// ── Push-constant struct ────────────────────────────────────────────────
public struct MMInt8_PC {
    uint M;
    uint N;
    uint K;
    uint stride_a_m;
    uint stride_a_k;
    uint stride_b_k;
    uint stride_b_n;
    uint stride_c_m;
    uint stride_c_n;
};

// +32 bank-conflict padding (M20) — shifts bank mapping for
// column-major access patterns when TILE_K or TILE_N is a multiple of 32.
groupshared int _tile_a[TILE_A_ELEMS + 32];
groupshared int _tile_b[TILE_B_ELEMS + 32];

// ── Int8 unpack helper ──────────────────────────────────────────────────
// Extracts one int8 value from a packed uint32 buffer. `idx` is the
// element index (in int8 space, 4 elements per uint32 word).
[ForceInline]
public int unpack_i8(StructuredBuffer<uint> buf, uint idx) {
    uint word = buf[idx >> 2u];
    uint lane = idx & 3u;
    uint b = (word >> (lane << 3u)) & 0xFFu;
    // Sign-extend: if MSB is set, value is negative
    return (b & 0x80u) != 0u ? int(b) - 256 : int(b);
}

// ── Load helpers ────────────────────────────────────────────────────────

[ForceInline]
public void load_tiles_int8(
    uint k_base,
    uint row_base,
    uint col_base,
    uint M, uint N, uint K,
    uint stride_a_m, uint stride_a_k,
    uint stride_b_k, uint stride_b_n,
    StructuredBuffer<uint> A,
    StructuredBuffer<uint> B,
    uint3 lid)
{
    uint tid = lid.y * WG_N + lid.x;

    // Load A tile: TILE_M * TILE_K int8 values → groupshared int.
    [unroll]
    for (uint i = tid; i < (uint)TILE_A_ELEMS; i += (uint)WG_SIZE) {
        uint local_m = i / (uint)TILE_K;
        uint local_k = i % (uint)TILE_K;
        uint global_m = row_base + local_m;
        uint global_k = k_base + local_k;
        int v = 0;
        if (global_m < M && global_k < K) {
            uint elem_idx = global_m * stride_a_m + global_k * stride_a_k;
            v = unpack_i8(A, elem_idx);
        }
        _tile_a[i] = v;
    }

    // Load B tile: TILE_K * TILE_N int8 values → groupshared int.
    [unroll]
    for (uint j = tid; j < (uint)TILE_B_ELEMS; j += (uint)WG_SIZE) {
        uint local_k = j / (uint)TILE_N;
        uint local_n = j % (uint)TILE_N;
        uint global_k = k_base + local_k;
        uint global_n = col_base + local_n;
        int v = 0;
        if (global_k < K && global_n < N) {
            uint elem_idx = global_k * stride_b_k + global_n * stride_b_n;
            v = unpack_i8(B, elem_idx);
        }
        _tile_b[j] = v;
    }
}

// ── MMA (matrix-multiply-accumulate) helper ─────────────────────────────

[ForceInline]
public void mma_tile_int8(
    uint k_base,
    uint K,
    uint3 lid,
    inout int acc[M_PER_THREAD][N_PER_THREAD])
{
    uint k_end = min((uint)TILE_K, K - k_base);
    [unroll(TILE_K <= 16 ? TILE_K : 16)]
    for (uint kk = 0; kk < k_end; kk++) {
        int a_reg[M_PER_THREAD];
        [unroll]
        for (uint mi = 0; mi < (uint)M_PER_THREAD; mi++) {
            uint local_m = lid.y * (uint)M_PER_THREAD + mi;
            a_reg[mi] = _tile_a[local_m * (uint)TILE_K + kk];
        }
        int b_reg[N_PER_THREAD];
        [unroll]
        for (uint ni = 0; ni < (uint)N_PER_THREAD; ni++) {
            uint local_n = lid.x * (uint)N_PER_THREAD + ni;
            b_reg[ni] = _tile_b[kk * (uint)TILE_N + local_n];
        }
        [unroll]
        for (uint mi = 0; mi < (uint)M_PER_THREAD; mi++) {
            [unroll]
            for (uint ni = 0; ni < (uint)N_PER_THREAD; ni++) {
                acc[mi][ni] += a_reg[mi] * b_reg[ni];
            }
        }
    }
}

// ── Store epilogue: int32 → float32 ─────────────────────────────────────

[ForceInline]
public void store_epilogue_int8(
    uint row_base, uint col_base,
    uint M, uint N,
    uint stride_c_m, uint stride_c_n,
    RWStructuredBuffer<float> C,
    int acc[M_PER_THREAD][N_PER_THREAD],
    uint3 lid)
{
    [unroll]
    for (uint mi = 0; mi < (uint)M_PER_THREAD; mi++) {
        uint row = row_base + lid.y * (uint)M_PER_THREAD + mi;
        if (row >= M) continue;
        [unroll]
        for (uint ni = 0; ni < (uint)N_PER_THREAD; ni++) {
            uint col = col_base + lid.x * (uint)N_PER_THREAD + ni;
            if (col >= N) continue;
            C[row * stride_c_m + col * stride_c_n] = float(acc[mi][ni]);
        }
    }
}

// ── Full computeTile entry point ────────────────────────────────────────
// Orchestrates the full tiled int8 matmul: load → mma → store, for a single
// workgroup. The caller (wrapper entry point) provides the dispatch
// geometry, push constants, and buffer bindings.

[ForceInline]
public void computeTile(
    uint row_base,
    uint col_base,
    MMInt8_PC pc,
    StructuredBuffer<uint> A,
    StructuredBuffer<uint> B,
    RWStructuredBuffer<float> C,
    uint3 gid,
    uint3 lid)
{
    int acc[M_PER_THREAD][N_PER_THREAD];
    [unroll]
    for (uint _mi = 0; _mi < (uint)M_PER_THREAD; _mi++) {
        [unroll]
        for (uint _ni = 0; _ni < (uint)N_PER_THREAD; _ni++) {
            acc[_mi][_ni] = 0;
        }
    }

    uint k_blocks = (pc.K + (uint)TILE_K - 1u) / (uint)TILE_K;

    for (uint kb = 0; kb < k_blocks; kb++) {
        uint k_base = kb * (uint)TILE_K;
        load_tiles_int8(k_base,
            row_base, col_base, pc.M, pc.N, pc.K,
            pc.stride_a_m, pc.stride_a_k,
            pc.stride_b_k, pc.stride_b_n, A, B, lid);
        GroupMemoryBarrierWithGroupSync();
        mma_tile_int8(k_base, pc.K, lid, acc);
        GroupMemoryBarrierWithGroupSync();
    }

    store_epilogue_int8(
        row_base, col_base, pc.M, pc.N,
        pc.stride_c_m, pc.stride_c_n, C, acc, lid);
}
