// mm_tile.slang — Link-time-specializable tiled matmul module.  P3.2 / M14.
//
// Tile sizes are declared as `static const int` with sensible defaults so the
// module is compilable standalone (for `slangc -emit-ir`). At link time,
// importing wrappers override these by defining the same `static const int`
// BEFORE the `import mm_tile;` statement, or via `-specialize-const` flags.
// Slang resolves the importing module's definitions over the defaults.
//
// This is the modern replacement for `mm_tiled.slang` (which uses
// `#ifndef` / `#define` preprocessor guards). The module body is compiled
// ONCE to `.slang-module`; per-tile-config wrappers import it without
// re-parsing.
//
// Usage (precompile once per dtype):
//   slangc mm_tile.slang -emit-ir -o mm_tile.slang-module
//
// Usage (specialize at link time):
//   slangc wrapper.slang -I <module_dir> -o kernel.spv
//   where wrapper.slang defines `static const int TILE_M = 64;` etc.
//   BEFORE `import mm_tile;`.

module mm_tile;

// `__exported import` re-exports `pointwise`'s public symbols (the
// IPointwise interface + `OpGELU` / `OpReLU` / `OpIdentity` / etc. struct
// hierarchy) so that wrappers which `import mm_tile;` can name those
// types as the generic argument to `mm_tile::computeTile<Epi>` without
// also having to `import pointwise;` themselves.  Required by the N+1.9
// link-time specialization wrapper rendered in
// `vulkan_template_caller._render_mm_linktime_wrapper_slang`.
__exported import pointwise;

// ── Link-time specialization constants ──────────────────────────────────
// Sensible defaults allow standalone compilation for syntax-check and
// `-emit-ir`.  Importing wrappers override these at link time by defining
// the same constants BEFORE the `import mm_tile;` statement.
// Slang resolves the wrapper's definitions over these module defaults.
static const int TILE_M = 64;
static const int TILE_N = 64;
static const int TILE_K = 16;
static const int M_PER_THREAD = 4;
static const int N_PER_THREAD = 4;
// NUM_STAGES is a preprocessor macro (not static const int) because
// it is used in #if guards for groupshared array sizing.  Importing
// wrappers override via #define NUM_STAGES N before import.
#ifndef NUM_STAGES
#define NUM_STAGES 1
#endif

// ── Derived constants ───────────────────────────────────────────────────
static const int WG_M = TILE_M / M_PER_THREAD;
static const int WG_N = TILE_N / N_PER_THREAD;
static const int TILE_A_ELEMS = TILE_M * TILE_K;
static const int TILE_B_ELEMS = TILE_K * TILE_N;
static const int TILE_A_2X = 2 * TILE_A_ELEMS;
static const int TILE_B_2X = 2 * TILE_B_ELEMS;
static const int WG_SIZE = WG_M * WG_N;

// ── Push-constant struct (public: used by wrappers) ─────────────────────
public struct MM_PC {
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
#if NUM_STAGES > 1
groupshared float _tile_a[TILE_A_2X + 32];
groupshared float _tile_b[TILE_B_2X + 32];
#else
groupshared float _tile_a[TILE_A_ELEMS + 32];
groupshared float _tile_b[TILE_B_ELEMS + 32];
#endif

// ── Load helpers ────────────────────────────────────────────────────────

[ForceInline]
public void load_tiles_single(
    uint k_base,
    uint row_base,
    uint col_base,
    uint M, uint N, uint K,
    uint stride_a_m, uint stride_a_k,
    uint stride_b_k, uint stride_b_n,
    StructuredBuffer<float> A,
    StructuredBuffer<float> B,
    uint3 lid)
{
    uint tid = lid.y * WG_N + lid.x;

    // Load A tile: TILE_M * TILE_K floats.
    [unroll]
    for (uint i = tid; i < (uint)TILE_A_ELEMS; i += (uint)WG_SIZE) {
        uint local_m = i / (uint)TILE_K;
        uint local_k = i % (uint)TILE_K;
        uint global_m = row_base + local_m;
        uint global_k = k_base + local_k;
        float v = 0.0f;
        if (global_m < M && global_k < K) {
            v = A[global_m * stride_a_m + global_k * stride_a_k];
        }
        _tile_a[i] = v;
    }

    // Load B tile: TILE_K * TILE_N floats.
    [unroll]
    for (uint j = tid; j < (uint)TILE_B_ELEMS; j += (uint)WG_SIZE) {
        uint local_k = j / (uint)TILE_N;
        uint local_n = j % (uint)TILE_N;
        uint global_k = k_base + local_k;
        uint global_n = col_base + local_n;
        float v = 0.0f;
        if (global_k < K && global_n < N) {
            v = B[global_k * stride_b_k + global_n * stride_b_n];
        }
        _tile_b[j] = v;
    }
}

#if NUM_STAGES > 1
[ForceInline]
public void load_tiles_double(
    uint stage,
    uint k_base,
    uint row_base,
    uint col_base,
    uint M, uint N, uint K,
    uint stride_a_m, uint stride_a_k,
    uint stride_b_k, uint stride_b_n,
    StructuredBuffer<float> A,
    StructuredBuffer<float> B,
    uint3 lid)
{
    uint off_a = stage * (uint)TILE_A_ELEMS;
    uint off_b = stage * (uint)TILE_B_ELEMS;
    uint tid = lid.y * WG_N + lid.x;

    [unroll]
    for (uint i = tid; i < (uint)TILE_A_ELEMS; i += (uint)WG_SIZE) {
        uint local_m = i / (uint)TILE_K;
        uint local_k = i % (uint)TILE_K;
        uint global_m = row_base + local_m;
        uint global_k = k_base + local_k;
        float v = 0.0f;
        if (global_m < M && global_k < K) {
            v = A[global_m * stride_a_m + global_k * stride_a_k];
        }
        _tile_a[off_a + i] = v;
    }

    [unroll]
    for (uint j = tid; j < (uint)TILE_B_ELEMS; j += (uint)WG_SIZE) {
        uint local_k = j / (uint)TILE_N;
        uint local_n = j % (uint)TILE_N;
        uint global_k = k_base + local_k;
        uint global_n = col_base + local_n;
        float v = 0.0f;
        if (global_k < K && global_n < N) {
            v = B[global_k * stride_b_k + global_n * stride_b_n];
        }
        _tile_b[off_b + j] = v;
    }
}
#endif

// ── MMA (matrix-multiply-accumulate) helpers ────────────────────────────

[ForceInline]
public void mma_tile_single(
    uint k_base,
    uint K,
    uint3 lid,
    inout float acc[M_PER_THREAD][N_PER_THREAD])
{
    uint k_end = min((uint)TILE_K, K - k_base);
    [unroll(TILE_K <= 16 ? TILE_K : 16)]
    for (uint kk = 0; kk < k_end; kk++) {
        float a_reg[M_PER_THREAD];
        [unroll]
        for (uint mi = 0; mi < (uint)M_PER_THREAD; mi++) {
            uint local_m = lid.y * (uint)M_PER_THREAD + mi;
            a_reg[mi] = _tile_a[local_m * (uint)TILE_K + kk];
        }
        float b_reg[N_PER_THREAD];
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

#if NUM_STAGES > 1
[ForceInline]
public void mma_tile_double(
    uint stage,
    uint k_base,
    uint K,
    uint3 lid,
    inout float acc[M_PER_THREAD][N_PER_THREAD])
{
    uint off_a = stage * (uint)TILE_A_ELEMS;
    uint off_b = stage * (uint)TILE_B_ELEMS;
    uint k_end = min((uint)TILE_K, K - k_base);
    [unroll(TILE_K <= 16 ? TILE_K : 16)]
    for (uint kk = 0; kk < k_end; kk++) {
        float a_reg[M_PER_THREAD];
        [unroll]
        for (uint mi = 0; mi < (uint)M_PER_THREAD; mi++) {
            uint local_m = lid.y * (uint)M_PER_THREAD + mi;
            a_reg[mi] = _tile_a[off_a + local_m * (uint)TILE_K + kk];
        }
        float b_reg[N_PER_THREAD];
        [unroll]
        for (uint ni = 0; ni < (uint)N_PER_THREAD; ni++) {
            uint local_n = lid.x * (uint)N_PER_THREAD + ni;
            b_reg[ni] = _tile_b[off_b + kk * (uint)TILE_N + local_n];
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
#endif

// ── Epilogue store ──────────────────────────────────────────────────────
// Generic over the epilogue type (IPointwise), resolved at SPIR-V link time.
[ForceInline]
public void store_epilogue<Epi : IPointwise>(
    uint row_base, uint col_base,
    uint M, uint N,
    uint stride_c_m, uint stride_c_n,
    RWStructuredBuffer<float> C,
    float acc[M_PER_THREAD][N_PER_THREAD],
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
            float v = acc[mi][ni];
            v = Epi::apply(v);
            C[row * stride_c_m + col * stride_c_n] = v;
        }
    }
}

// ── Full computeTile entry point ────────────────────────────────────────
// Orchestrates the full tiled matmul: load → mma → store, for a single
// workgroup. The caller (wrapper entry point) provides the dispatch
// geometry, push constants, buffer bindings, and epilogue type.
//
// Supports both single-stage and double-buffered (NUM_STAGES>1) execution
// selected at link time via the NUM_STAGES extern constant.  The
// `#if NUM_STAGES > 1` guards are constant-folded by Slang's linker after
// specialization, so there's zero runtime branch overhead.
[ForceInline]
public void computeTile<Epi : IPointwise>(
    uint row_base,
    uint col_base,
    MM_PC pc,
    StructuredBuffer<float> A,
    StructuredBuffer<float> B,
    RWStructuredBuffer<float> C,
    uint3 gid,
    uint3 lid)
{
    float acc[M_PER_THREAD][N_PER_THREAD];
    [unroll]
    for (uint _mi = 0; _mi < (uint)M_PER_THREAD; _mi++) {
        [unroll]
        for (uint _ni = 0; _ni < (uint)N_PER_THREAD; _ni++) {
            acc[_mi][_ni] = 0.0f;
        }
    }

    uint k_blocks = (pc.K + (uint)TILE_K - 1u) / (uint)TILE_K;

#if NUM_STAGES > 1
    // Double-buffered pipeline: prefetch stage 0, then loop over K blocks
    // computing on current stage while loading the next.
    if (k_blocks >= 1) {
        load_tiles_double(0, 0,
            row_base, col_base, pc.M, pc.N, pc.K,
            pc.stride_a_m, pc.stride_a_k,
            pc.stride_b_k, pc.stride_b_n, A, B, lid);
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint kb = 0; kb < k_blocks; kb++) {
        mma_tile_double(kb & 1u, kb * (uint)TILE_K, pc.K, lid, acc);
        if (kb + 1 < k_blocks) {
            load_tiles_double((kb + 1u) & 1u, (kb + 1u) * (uint)TILE_K,
                row_base, col_base, pc.M, pc.N, pc.K,
                pc.stride_a_m, pc.stride_a_k,
                pc.stride_b_k, pc.stride_b_n, A, B, lid);
            GroupMemoryBarrierWithGroupSync();
        }
    }
#else
    // Single-stage pipeline: load → mma → barrier — repeat for each K block.
    for (uint kb = 0; kb < k_blocks; kb++) {
        uint k_base = kb * (uint)TILE_K;
        load_tiles_single(k_base,
            row_base, col_base, pc.M, pc.N, pc.K,
            pc.stride_a_m, pc.stride_a_k,
            pc.stride_b_k, pc.stride_b_n, A, B, lid);
        GroupMemoryBarrierWithGroupSync();
        mma_tile_single(k_base, pc.K, lid, acc);
        GroupMemoryBarrierWithGroupSync();
    }
#endif

    store_epilogue<Epi>(
        row_base, col_base, pc.M, pc.N,
        pc.stride_c_m, pc.stride_c_n, C, acc, lid);
}

// ── CG.M5: Differentiable scalar inner for autodiff ───────────────────
// Extracted from the per-element mma loop so Slang's autodiff can
// produce dA, dB gradients.  The tile-loop wrapper (in
// slang_mm_bwd.py.jinja) calls bwd_diff(tile_inner_madd) for each
// (m,n) element and aggregates dA/dB contributions across the K
// dimension.
//
// Autodiff derivatives:
//   dL/da = dL/dout * b    (gradient w.r.t. A element)
//   dL/db = dL/dout * a    (gradient w.r.t. B element)
[Differentiable]
public float tile_inner_madd(float a, float b, float acc)
{
    return acc + a * b;
}
