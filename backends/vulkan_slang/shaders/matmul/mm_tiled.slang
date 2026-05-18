// Tiled matrix multiplication using shared memory.
//   C[M,N] = op(A) * op(B)
// When transpose_a=0: A is [M,K]  (contiguous), op(A) = A
// When transpose_a=1: A is [K,M]  (contiguous), op(A) = A^T
// When transpose_b=0: B is [K,N]  (contiguous), op(B) = B
// When transpose_b=1: B is [N,K]  (contiguous), op(B) = B^T
//
// Each thread block computes a TILE_SIZE x TILE_SIZE tile of C.
//
// M22.13-followup: derive ``lrow`` / ``lcol`` from a linearised
// ``tid_local`` instead of reading ``lid.x`` / ``lid.y`` directly. The
// raw ``lid.x``/``lid.y`` form triggered a slangc 2026.5+ wave64
// scalarisation bug on RDNA1 (3+ structured-buffer bindings, both
// ``lid.x`` and ``lid.y`` used to address different buffers): the
// generated SPIR-V folded ``lid.x`` to a constant for lanes with
// ``lid.x >= 1``, collapsing rows 1..TILE_SIZE-1 of every output tile
// onto row 0 (the visually-buggy "all rows identical" symptom from
// M22.13). The pattern is the analog of M17.1's ``slang_mm.slang``
// fix (see ``templates/slang_mm.slang`` lines 239-246).
//
// Linearisation rule used here: this shader's workgroup is
// ``numthreads(TILE_SIZE, TILE_SIZE, 1)``, so the SPIR-V lane ordering
// is ``lane = lid.y * TILE_SIZE + lid.x``. Reconstructing ``lid.x``
// and ``lid.y`` from that lane via integer math IS algebraically
// identical to the direct read — but slangc's wave64 scalarisation
// pass does not recognise the recovered values as wave-uniform, so
// the buggy fold doesn't fire.

static const uint TILE_SIZE = 16;

groupshared float sA[TILE_SIZE][TILE_SIZE];
groupshared float sB[TILE_SIZE][TILE_SIZE];

struct Params {
    uint M;
    uint N;
    uint K;
    uint transpose_a;
    uint transpose_b;
};

[[vk::binding(0)]] StructuredBuffer<float> A;
[[vk::binding(1)]] StructuredBuffer<float> B;
[[vk::binding(2)]] RWStructuredBuffer<float> C;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void computeMain(
    uint3 lid : SV_GroupThreadID,
    uint3 gid : SV_GroupID)
{
    // M22.13-followup: derive per-thread coordinates from a linearised
    // ``tid_local``. Algebraically identical to reading lid.x/lid.y
    // directly, but the SPIR-V wave64-scalarisation pass treats the
    // recovered values as per-lane (correct) rather than wave-uniform
    // (the buggy fold).
    uint tid_local = lid.y * TILE_SIZE + lid.x;
    uint lrow = tid_local % TILE_SIZE;  // == lid.x — but recovered
    uint lcol = tid_local / TILE_SIZE;  // == lid.y — but recovered
    uint row = gid.x * TILE_SIZE + lrow;
    uint col = gid.y * TILE_SIZE + lcol;

    float sum = 0.0f;

    uint num_tiles = (params.K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        // Load tile of A into shared memory
        uint ak = t * TILE_SIZE + lcol;
        if (params.transpose_a != 0) {
            // A is [K, M], we want A^T[row, k] = A[k, row]
            if (row < params.M && ak < params.K)
                sA[lrow][lcol] = A[ak * params.M + row];
            else
                sA[lrow][lcol] = 0.0f;
        } else {
            // A is [M, K]
            if (row < params.M && ak < params.K)
                sA[lrow][lcol] = A[row * params.K + ak];
            else
                sA[lrow][lcol] = 0.0f;
        }

        // Load tile of B into shared memory
        uint bk = t * TILE_SIZE + lrow;
        if (params.transpose_b != 0) {
            // B is [N, K], we want B^T[k, col] = B[col, k]
            if (bk < params.K && col < params.N)
                sB[lrow][lcol] = B[col * params.K + bk];
            else
                sB[lrow][lcol] = 0.0f;
        } else {
            // B is [K, N]
            if (bk < params.K && col < params.N)
                sB[lrow][lcol] = B[bk * params.N + col];
            else
                sB[lrow][lcol] = 0.0f;
        }

        GroupMemoryBarrierWithGroupSync();

        // Multiply tiles
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += sA[lrow][k] * sB[k][lcol];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < params.M && col < params.N) {
        C[row * params.N + col] = sum;
    }
}
