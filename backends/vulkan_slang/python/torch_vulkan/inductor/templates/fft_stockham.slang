// Stockham (out-of-place, self-sorting) radix-2 FFT for power-of-2 sizes.
//
// Each workgroup computes one batch element's 1-D FFT using groupshared memory.
// Twiddle factors are precomputed once and cached in groupshared for the
// lifetime of the workgroup.
//
// Complex data is stored as interleaved float pairs: [re0, im0, re1, im1, ...]
//
// Template parameters (Jinja2 — computed in Python, passed as scalars):
//   N          — FFT size (power of 2, compile-time constant)
//   half_N     — N // 2
//   log2_N     — integer log2 of N
//   n_threads  — number of threads (min(half_N, 256))
//   dir_sign   — -1.0 for forward, +1.0 for inverse
//   norm_scale — 1/sqrt(N) if normalized, else None (omitted)
//   dtype      — Slang type string for storage ("float")
//
// Dispatch grid:  (batch, 1, 1)          — one workgroup per batch element
// Numthreads:     (n_threads, 1, 1)      — each thread handles multiple
//                  butterfly pairs per stage and loads/stores multiple elements

static const uint N_VAL = /*{{*/ N /*}}256*/u;
static const uint HALF_N = /*{{*/ half_N /*}}128*/u;
static const uint LOG2_N = /*{{*/ log2_N /*}}8*/u;
static const float DIR_SIGN = /*{{*/ dir_sign /*}}-1.0*/f;

/*{%*/ if norm_scale is not none /*%}*/
static const float NORM_SCALE = /*{{*/ norm_scale /*}}0.0625*/f;
/*{%*/ endif /*%}*/

struct PC {
    uint batch_in;     // number of independent 1-D signals
    uint in_stride;    // stride in floats between signals (= 2*N for contiguous complex)
    uint out_stride;   // stride in floats between output signals
    uint batch_out;    // number of output signals (may differ for r2c/c2r)
};

[[vk::push_constant]] PC pc;

struct KernelArgs {
    StructuredBuffer</*{{*/ dtype /*}}float*/> src;      // input
    RWStructuredBuffer</*{{*/ dtype /*}}float*/> dst;    // output
};
ParameterBlock<KernelArgs> args;

// Groupshared: work buffer (2*N floats for interleaved complex) + twiddle cache
groupshared /*{{*/ dtype /*}}float*/ gs_data[/*{{*/ 2 * N /*}}512*/];
groupshared /*{{*/ dtype /*}}float*/ gs_twiddle[/*{{*/ N /*}}256*/];  // N floats: interleaved cos/sin for N/2 values

// ── Twiddle precomputation ──────────────────────────────────────────────
// All threads cooperate to fill twiddle[0..N-1] with cos(θ_k), sin(θ_k).
void precompute_twiddles(uint tid, uint total_threads) {
    [unroll]
    for (uint i = tid; i < HALF_N; i += total_threads) {
        float angle = DIR_SIGN * 2.0f * 3.14159265358979323846f * float(i) / float(N_VAL);
        gs_twiddle[2u * i]      = cos(angle);
        gs_twiddle[2u * i + 1u] = sin(angle);
    }
}

// ── Stockham butterfly stage ────────────────────────────────────────────
// Performs one stage with step = 2^s, using precomputed twiddle factors
// stored in groupshared memory.
//
// Stockham (self-sorting) addressing for stage s:
//   half  = 2^s
//   bfly_len = 2 * half
//   For each butterfly j (0 .. N/2-1):
//     group   = j / half
//     k       = j % half
//     idx_a   = 2 * bfly_len * group + 2 * k       → top element base
//     idx_b   = idx_a + 2 * half                    → bottom element base
//     twiddle_idx = k * (N / bfly_len)
void stockham_stage(uint s, uint tid, uint total_threads) {
    uint half = 1u << s;
    uint bfly_len = half << 1u;

    [unroll]
    for (uint j = tid; j < HALF_N; j += total_threads) {
        uint group = j / half;
        uint k     = j % half;

        uint idx_a = 2u * (bfly_len * group + k);
        uint idx_b = idx_a + 2u * half;

        float ar = gs_data[idx_a];
        float ai = gs_data[idx_a + 1u];
        float br = gs_data[idx_b];
        float bi = gs_data[idx_b + 1u];

        uint tw_k = k * (N_VAL / bfly_len);
        float wr = gs_twiddle[2u * tw_k];
        float wi = gs_twiddle[2u * tw_k + 1u];

        // Butterfly: A = a + W*b, B = a - W*b
        float tr = wr * br - wi * bi;
        float ti = wr * bi + wi * br;

        gs_data[idx_a]      = ar + tr;
        gs_data[idx_a + 1u] = ai + ti;
        gs_data[idx_b]      = ar - tr;
        gs_data[idx_b + 1u] = ai - ti;
    }
}

// ── Main entry point ────────────────────────────────────────────────────
[shader("compute")]
[numthreads(/*{{*/ n_threads /*}}256*/, 1, 1)]
void computeMain(
    uint3 gid  : SV_GroupID,
    uint3 lid  : SV_GroupThreadID,
    uint3 gtid : SV_DispatchThreadID)
{
    uint batch_idx = gid.x;
    if (batch_idx >= pc.batch_in && batch_idx >= pc.batch_out) return;

    uint tid = lid.x;
    uint total_threads = /*{{*/ n_threads /*}}256*/u;

    // ── Phase 0: Precompute twiddle factors ─────────────────────────
    precompute_twiddles(tid, total_threads);
    GroupMemoryBarrierWithGroupSync();

    // ── Phase 1: Load input into groupshared ────────────────────────
    if (batch_idx < pc.batch_in) {
        uint base_in = batch_idx * pc.in_stride;
        [unroll]
        for (uint i = tid; i < 2u * N_VAL; i += total_threads) {
            gs_data[i] = args.src[base_in + i];
        }
    } else {
        // Out-of-range batch: zero-fill
        [unroll]
        for (uint i = tid; i < 2u * N_VAL; i += total_threads) {
            gs_data[i] = 0.0f;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // ── Phase 2: Stockham butterfly stages ──────────────────────────
    [unroll]
    for (uint s = 0u; s < LOG2_N; s++) {
        stockham_stage(s, tid, total_threads);
        GroupMemoryBarrierWithGroupSync();
    }

    // ── Phase 3: Store output ───────────────────────────────────────
    if (batch_idx < pc.batch_out) {
        uint base_out = batch_idx * pc.out_stride;
        [unroll]
        for (uint i = tid; i < 2u * N_VAL; i += total_threads) {
            float val = gs_data[i];
/*{%*/ if norm_scale is not none /*%}*/
            val *= NORM_SCALE;
/*{%*/ endif /*%}*/
            args.dst[base_out + i] = (/*{{*/ dtype /*}}float*/)val;
        }
    }
}
