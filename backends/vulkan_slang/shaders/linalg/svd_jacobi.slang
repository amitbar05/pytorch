// One-sided Jacobi SVD.
// Computes A = U * diag(S) * V^T for a single M×N matrix (M >= N).
//
// Algorithm:
//   Repeat until convergence:
//     For each column pair (p, q) with p < q:
//       Compute the 2×2 symmetric sub-matrix of A^T*A for columns p,q.
//       Compute Givens rotation angle to diagonalise that 2×2.
//       Apply rotation to columns p and q of A (and accumulate in V).
//   Singular values  S[j]    = norm of column j of A.
//   Left vectors     U[:,j]  = column j of A / S[j].
//   Right vectors    V                        (accumulated rotations).
//
// Dispatch: one workgroup per matrix in the batch.
// Shared memory layout:
//   smem_A : float[M * N]  — working copy of A (column-major)
//   smem_V : float[N * N]  — right singular vectors
//
// Limits: M <= MAX_M (256), N <= MAX_N (32).

#ifndef MAX_M
#define MAX_M 256
#endif
#ifndef MAX_N
#define MAX_N 32
#endif

groupshared float smem_A[MAX_M * MAX_N];
groupshared float smem_V[MAX_N * MAX_N];
groupshared float smem_S[MAX_N];
groupshared uint  smem_order[MAX_N];
groupshared float g_alpha, g_beta, g_gamma;

struct Params {
    uint M;           // rows
    uint N;           // cols  (M >= N)
    uint batch;       // number of matrices
    uint max_sweeps;  // Jacobi sweep limit (default 30)
    uint full_matrices; // 1 = output U[M,M], 0 = U[M,N] (thin)
};

[[vk::binding(0)]] StructuredBuffer<float>   A_in;   // [batch * M * N] row-major
[[vk::binding(1)]] RWStructuredBuffer<float> U_out;  // [batch * M * Ku] row-major (Ku = full ? M : N)
[[vk::binding(2)]] RWStructuredBuffer<float> S_out;  // [batch * N]
[[vk::binding(3)]] RWStructuredBuffer<float> V_out;  // [batch * N * N] row-major
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint b   = gid.x;
    uint tid = lid.x;
    if (b >= params.batch) return;

    uint M = params.M;
    uint N = params.N;

    // --- Load A (column-major into smem_A) ---
    // A_in is row-major: A_in[b*M*N + row*N + col]
    // smem_A column-major: smem_A[col*M + row]
    for (uint i = tid; i < M * N; i += 256) {
        uint row = i / N;
        uint col = i % N;
        smem_A[col * M + row] = A_in[b * M * N + row * N + col];
    }

    // --- Initialise V = I ---
    for (uint i = tid; i < N * N; i += 256) {
        uint r = i / N, c = i % N;
        smem_V[i] = (r == c) ? 1.0f : 0.0f;
    }

    GroupMemoryBarrierWithGroupSync();

    // --- Jacobi sweeps ---
    for (uint sweep = 0u; sweep < params.max_sweeps; sweep++) {
        bool converged = true;

        for (uint p = 0u; p < N - 1u; p++) {
            for (uint q = p + 1u; q < N; q++) {
                // Compute 2x2 submatrix of A^T A for columns p,q.
                // Only thread 0 does the angle computation (simple serial per-pair).
                float alpha = 0.0f, beta = 0.0f, gamma_val = 0.0f;
                if (tid == 0u) {
                    for (uint i = 0u; i < M; i++) {
                        float ap = smem_A[p * M + i];
                        float aq = smem_A[q * M + i];
                        alpha += ap * ap;
                        beta  += aq * aq;
                        gamma_val += ap * aq;
                    }
                }
                GroupMemoryBarrierWithGroupSync();

                // Broadcast alpha/beta/gamma_val via smem (thread 0 writes, all read)
                if (tid == 0u) { g_alpha = alpha; g_beta = beta; g_gamma = gamma_val; }
                GroupMemoryBarrierWithGroupSync();
                alpha     = g_alpha;
                beta      = g_beta;
                gamma_val = g_gamma;

                // Skip pair if already orthogonal
                if (abs(gamma_val) < 1e-10f * sqrt(alpha * beta + 1e-30f)) continue;
                converged = false;

                // Symmetric Schur decomposition of [[alpha, gamma], [gamma, beta]]
                float zeta = (alpha - beta) / (2.0f * gamma_val);
                float t    = sign(zeta) / (abs(zeta) + sqrt(1.0f + zeta * zeta));
                float c    = 1.0f / sqrt(1.0f + t * t);
                float s    = c * t;

                // Apply Givens rotation to columns p and q of A and V
                for (uint i = tid; i < M; i += 256) {
                    float ap = smem_A[p * M + i];
                    float aq = smem_A[q * M + i];
                    smem_A[p * M + i] =  c * ap + s * aq;
                    smem_A[q * M + i] = -s * ap + c * aq;
                }
                for (uint i = tid; i < N; i += 256) {
                    float vp = smem_V[p * N + i];
                    float vq = smem_V[q * N + i];
                    smem_V[p * N + i] =  c * vp + s * vq;
                    smem_V[q * N + i] = -s * vp + c * vq;
                }
                GroupMemoryBarrierWithGroupSync();
            }
        }
        if (converged) break;
    }

    GroupMemoryBarrierWithGroupSync();

    // --- Extract singular values and sort (descending) ---
    // S[j] = norm of column j
    if (tid < N) {
        float sq = 0.0f;
        for (uint i = 0u; i < M; i++) {
            float v = smem_A[tid * M + i];
            sq += v * v;
        }
        smem_S[tid] = sqrt(sq);
    }
    GroupMemoryBarrierWithGroupSync();

    // Insertion sort on thread 0 (N <= 32, cheap)
    if (tid < N) smem_order[tid] = tid;
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0u) {
        for (uint i = 1u; i < N; i++) {
            uint key_idx = smem_order[i];
            float key_s  = smem_S[key_idx];
            int j = (int)i - 1;
            while (j >= 0 && smem_S[smem_order[j]] < key_s) {
                smem_order[j + 1] = smem_order[j];
                j--;
            }
            smem_order[j + 1] = key_idx;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // --- Write outputs (permuted by sort order) ---
    uint Ku = params.full_matrices ? M : N;

    // S_out[b*N + k] = S[smem_order[k]]
    if (tid < N) {
        S_out[b * N + tid] = smem_S[smem_order[tid]];
    }

    // U_out: thin U[M, N] — normalised columns of A in sorted order
    for (uint k = 0u; k < N; k++) {
        uint col = smem_order[k];
        float sv = smem_S[col];
        float inv_sv = (sv > 1e-30f) ? (1.0f / sv) : 0.0f;
        for (uint i = tid; i < M; i += 256) {
            float u_val = smem_A[col * M + i] * inv_sv;
            U_out[b * M * Ku + i * Ku + k] = u_val;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    // If full_matrices, remaining columns of U are zero (orthonormal complement not computed)
    if (params.full_matrices) {
        for (uint k = N; k < M; k++) {
            for (uint i = tid; i < M; i += 256)
                U_out[b * M * M + i * M + k] = 0.0f;
            GroupMemoryBarrierWithGroupSync();
        }
    }

    // V_out[b*N*N + row*N + col]: V columns are smem_V rows in sorted order
    // smem_V is column-major: smem_V[col * N + row]
    for (uint k = 0u; k < N; k++) {
        uint col = smem_order[k];
        for (uint i = tid; i < N; i += 256) {
            V_out[b * N * N + i * N + k] = smem_V[col * N + i];
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
