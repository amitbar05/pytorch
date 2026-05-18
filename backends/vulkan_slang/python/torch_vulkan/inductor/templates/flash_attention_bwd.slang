// CG.M7 — SDPA backward via [Differentiable] flash forward
// Single-kernel backward computes dQ, dK, dV in one fused dispatch.
//
// Algorithm: For each (b, h, q_tile) workgroup:
//   1. Load Q_tile [BQ, D] and dO_tile [BQ, D] into shared memory.
//   2. For each K/V tile along the S axis:
//      a. Load K_tile [BK, D] and V_tile [BK, D] into shared memory.
//      b. Recompute S = Q @ K^T * scale for all qi × ki in the tile.
//      c. Apply causal mask if needed.
//      d. Recompute online softmax to get P [BQ, BK].
//      e. Compute dV contribution:  dV += P^T @ dO    (atomic add)
//      f. Compute dP = dO @ V^T                       (scalar inner)
//      g. Compute softmax backward: dS = P * (dP - row_dot(P, dP))
//      h. Compute dQ contribution:  dQ += dS @ K       (register accumulate)
//      i. Compute dK contribution:  dK += dS^T @ Q     (atomic add)
//   3. Write dQ tile to output.
//
// Grid:  (B, H * N_q_tiles, 1) — same as forward.
// numthreads is 1-D (D threads per workgroup).
//
// dQ is accumulated per-workgroup in registers (same pattern as O in forward).
// dK and dV use InterlockedAdd because multiple Q tiles contribute to the
// same K/V positions.
//
// Parameters:
//   head_dim      — D ∈ {32, 64, 128, 256}
//   BK            — K/V tile size along S axis (default 64)
//   BQ            — Q tile size along N axis (default 32)
//
// M20.3: BQ, BK, HEAD_DIM are specialization constants
// (``[[vk::constant_id]]``). They determine groupshared array sizes and
// numthreads at SPIR-V compile time. One SPIR-V module per (dtype,
// is_causal, head_layout) serves every (BQ, BK, HEAD_DIM) tuple — the
// caller passes the tuple via Vulkan spec-constant overrides.
//
// Constant ID assignment (file-local 13-15; sequential after the
// forward template's 10-12 so dispatching forward+backward in the same
// pipeline-cache run doesn't collide):
//   13 — BQ        (default 32)
//   14 — BK        (default 64)
//   15 — HEAD_DIM  (default 64)
//
// We don't use the forward's 10-12 because backward maintains its own
// pipeline cache key; reusing the same IDs would force a forced
// override at every backward dispatch and lose the cleanliness of the
// per-template ID space.

// dK/dV use CAS-based vk_atomic_add (Mesa ACO on RDNA1 cannot lower
// `buffer_atomic_add_f32`; the lib's CAS path is portable).
import atomics;

/*{%*/ set wg_size = head_dim if head_dim <= 256 else 256 /*%}*/

// M20.3: Specialization constants for tile shape + workgroup dimensions.
// HEAD_DIM doubles as the workgroup x-dim — ``numthreads`` references it
// directly because Slang's compile-time-constant analysis only recognises
// spec consts in attribute arguments when the spec-const symbol is used
// without an intermediate ``static const`` alias (verified standalone:
// ``numthreads(X, 1, 1)`` works for ``[[vk::constant_id(N)]] const uint X``
// but ``static const uint Y = X; numthreads(Y, 1, 1)`` does not).
[[vk::constant_id(13)]] const uint BQ = /*{{*/ BQ /*}}32*/;
[[vk::constant_id(14)]] const uint BK = /*{{*/ BK /*}}64*/;
[[vk::constant_id(15)]] const uint HEAD_DIM = /*{{*/ head_dim /*}}64*/;

struct BwdPC {
    uint B;           // batch size
    uint H;           // number of query heads
    uint KV_H;        // number of KV heads (for GQA)
    uint N;           // query sequence length
    uint S;           // key/value sequence length
    uint D;           // head dimension
    float scale;       // 1/sqrt(D) or user-specified
    uint is_causal;   // 1 = apply causal mask
    uint q_layout;    // 0 = [B,H,N,D], 1 = [B,N,H,D]
};

[[vk::push_constant]] BwdPC pc;

// ── Buffer bindings (ParameterBlock) ────────────────────────────────────
struct KernelArgs {
    StructuredBuffer<float> q;         // saved forward input
    StructuredBuffer<float> k;         // saved forward input
    StructuredBuffer<float> v;         // saved forward input
    StructuredBuffer<float> lse;       // saved log-sum-exp [B, H, N]
    StructuredBuffer<float> dO;        // gradient of loss w.r.t. output O
    RWStructuredBuffer<float> dQ;      // output: gradient w.r.t. Q (per-workgroup accumulation, no atomic)
    // dK/dV are accumulated atomically across Q-tiles; bound as uint so we
    // can CAS on the raw bit pattern (RDNA1 ACO has no native float atomic).
    RWStructuredBuffer<uint> dK;
    RWStructuredBuffer<uint> dV;
};
// M21.3.01: explicit Set 0 binding (slangc 2026.7.1 defaults to Set 1)
[[vk::binding(0, 0)]] ParameterBlock<KernelArgs> args;

// Shared memory for Q, K, V, dO tiles.
// M20.3: groupshared array dimensions use spec-constant BQ, BK, HEAD_DIM.
groupshared float Q_smem[BQ * HEAD_DIM];
groupshared float K_smem[BK * HEAD_DIM];
groupshared float V_smem[BK * HEAD_DIM];
groupshared float dO_smem[BQ * HEAD_DIM];

// ── CG.M7: Differentiable scalar inner functions (inlined) ──────────────
// These are copies of loss.slang::sdpa_score / sdpa_output / softmax_elem
// inlined for standalone template compilation.  The [Differentiable]
// annotation lets bwd_diff produce derivative contributions.

[Differentiable]
float sdpa_score(float a, float b, no_diff float acc)
{
    return acc + a * b;
}

[Differentiable]
float sdpa_output(float a, float b, no_diff float acc)
{
    return acc + a * b;
}

// ── Layout helpers ───────────────────────────────────────────────────────

[ForceInline]
uint _q_offset(uint b, uint h, uint q_pos, uint d) {
    if (pc.q_layout != 0u) {
        return b * pc.N * pc.H * pc.D + q_pos * pc.H * pc.D + h * pc.D + d;
    } else {
        return b * pc.H * pc.N * pc.D + h * pc.N * pc.D + q_pos * pc.D + d;
    }
}

[ForceInline]
uint _kv_offset(uint b, uint kv_h, uint pos, uint d) {
    if (pc.q_layout != 0u) {
        return b * pc.S * pc.KV_H * pc.D + pos * pc.KV_H * pc.D + kv_h * pc.D + d;
    } else {
        return b * pc.KV_H * pc.S * pc.D + kv_h * pc.S * pc.D + pos * pc.D + d;
    }
}

// ── Entry point — one workgroup per (b, h, q_tile) ───────────────────────

[shader("compute")]
[numthreads(HEAD_DIM, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint tid = lid.x;

    // Decode global id: matches forward grid encoding.
    uint b = gid.x;
    uint N_q_tiles = (pc.N + BQ - 1u) / BQ;
    uint h = gid.y / N_q_tiles;
    uint q_start = (gid.y % N_q_tiles) * BQ;
    uint kv_h = (h * pc.KV_H) / pc.H;  // GQA head mapping

    if (b >= pc.B || h >= pc.H) return;

    // ── Load Q and dO into shared memory ──────────────────────────────
    [unroll]
    for (uint qi = 0; qi < BQ; qi++) {
        uint q_pos = q_start + qi;
        float qv = 0.0f;
        float dov = 0.0f;
        if (q_pos < pc.N && tid < pc.D) {
            qv = args.q[_q_offset(b, h, q_pos, tid)];
            dov = args.dO[_q_offset(b, h, q_pos, tid)];
        }
        Q_smem[qi * HEAD_DIM + tid] = qv;
        dO_smem[qi * HEAD_DIM + tid] = dov;
    }
    GroupMemoryBarrierWithGroupSync();

    // ── Per-thread dQ accumulator (registers, one per qi × d) ─────────
    // dQ has same shape as Q: [BQ, D].  Each thread accumulates one d-slot
    // and writes it at the end — same pattern as O in the forward.
    float dQ_acc[BQ];
    [unroll]
    for (uint qi = 0; qi < BQ; qi++) {
        dQ_acc[qi] = 0.0f;
    }

    // ── Iterate over K/V tiles ────────────────────────────────────────
    uint kv_tiles = (pc.S + BK - 1u) / BK;
    for (uint kvt = 0; kvt < kv_tiles; kvt++) {
        uint kv_start = kvt * BK;

        // ── Load K tile into shared memory ──────────────────────────
        [unroll]
        for (uint ki_load = 0; ki_load < BK; ki_load++) {
            uint k_pos = kv_start + ki_load;
            float kv_val = 0.0f;
            if (k_pos < pc.S && tid < pc.D) {
                kv_val = args.k[_kv_offset(b, kv_h, k_pos, tid)];
            }
            K_smem[ki_load * HEAD_DIM + tid] = kv_val;
        }

        // ── Load V tile into shared memory ──────────────────────────
        [unroll]
        for (uint ki_load = 0; ki_load < BK; ki_load++) {
            uint k_pos = kv_start + ki_load;
            float vv = 0.0f;
            if (k_pos < pc.S && tid < pc.D) {
                vv = args.v[_kv_offset(b, kv_h, k_pos, tid)];
            }
            V_smem[ki_load * HEAD_DIM + tid] = vv;
        }
        GroupMemoryBarrierWithGroupSync();

        // ── Phase 1: Recompute softmax P and accumulate dV ─────────
        // Online softmax state for each qi in this Q tile.
        float m[BQ];
        float l[BQ];
        // Accumulated dV per ki for this thread's d-slot.
        float dV_local[BK];
        [unroll]
        for (uint qi = 0; qi < BQ; qi++) {
            m[qi] = -1e38f;
            l[qi] = 0.0f;
        }
        [unroll]
        for (uint ki = 0; ki < BK; ki++) {
            dV_local[ki] = 0.0f;
        }

        // Recompute scores and softmax (same as forward).
        // Also accumulate dV contributions using the recomputed P.
        [unroll]
        for (uint qi = 0; qi < BQ; qi++) {
            uint q_pos = q_start + qi;
            if (q_pos >= pc.N) break;

            [unroll]
            for (uint ki = 0; ki < BK; ki++) {
                uint k_pos = kv_start + ki;
                if (k_pos >= pc.S) break;

                // Causality check.
                if (pc.is_causal != 0u && k_pos > q_pos) continue;

                // Full Q·K dot product (redundant across threads for
                // consistent softmax state on every lane).
                float score = 0.0f;
                for (uint d = 0; d < pc.D; d++) {
                    score += Q_smem[qi * HEAD_DIM + d]
                           * K_smem[ki * HEAD_DIM + d];
                }
                score *= pc.scale;

                // Online softmax step.
                float m_new = max(m[qi], score);
                float alpha = exp(m[qi] - m_new);
                float p_val = exp(score - m_new);
                l[qi] = l[qi] * alpha + p_val;
                m[qi] = m_new;

                // Normalize p for dV accumulation (must use final l after
                // all tiles — we'll rescale dV contributions at the end).
                // For now, accumulate raw p * dO contributions.
                // dV_contrib[ki, d] += p_val * dO_smem[qi, d]
                // (per d-slot; only this thread's tid dimension)
                if (tid < pc.D) {
                    float do_val = dO_smem[qi * HEAD_DIM + tid];
                    dV_local[ki] += p_val * do_val;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // ── Phase 1b: Rescale dV contributions and atomic-add ──────
        // dV_local was accumulated with unnormalized p values.
        // Normalize by the final softmax denominator l[qi] for each qi.
        // Since p_val = exp(score - m_new) and l is the running sum of
        // these, each p_val contributed to l[qi] with the right factor.
        // We need dV = sum_qi P_final[qi,ki] * dO[qi,d]
        // where P_final[qi,ki] = exp(score - m_global) / l_global
        // Our accumulated dV_local[ki] = sum_qi p_val_qi * dO[qi,d]
        // = sum_qi exp(score_qi - m_qi_final) * dO[qi,d]
        // But l_global = sum_ki exp(score_ki - m_global), which is the
        // sum across ALL tiles.  Since we don't have m_global / l_global
        // yet (those require a second pass), we defer the final rescale
        // and atomic-add to after all KV tiles are processed.
        //
        // SIMPLIFICATION: Since each KV tile's online softmax state
        // (m, l) is independent across tiles, the correct p values must
        // be recomputed with the GLOBAL m_global / l_global.  We store
        // the per-tile scores and do a second pass.
        //
        // To keep the template simpler: we recompute P properly using
        // the saved LSE from the forward pass, avoiding the need for
        // a second pass.  LSE[m,n] = m_global + log(l_global).
        //
        // P_i = exp(score_i - m_global) / l_global
        //     = exp(score_i - LSE_qi)
        //
        // We can use the LSE buffer to get m_global/l_global without
        // a second pass!

        // ── Phase 2: Recompute P using saved LSE, compute gradients ─
        // For each qi × ki in this tile:
        //   1. Recompute score (S)
        //   2. Compute P = exp(S * scale - LSE_qi)
        //   3. Compute dV contribution (atomic add)
        //   4. Compute dP = sum over d of (dO[qi,d] * V[ki,d])
        //   5. Accumulate row_dot = sum_ki(P[qi,ki] * dP[qi,ki])
        //   6. Compute dS = P * (dP - row_dot) (softmax backward)
        //   7. Accumulate dQ += dS * K  (register)
        //   8. Atomic-add dK += dS^T * Q  (atomic)

        float P_cache[BQ][BK];
        float dP_cache[BQ][BK];
        float dS_cache[BQ][BK];
        [unroll]
        for (uint qi = 0; qi < BQ; qi++) {
            [unroll]
            for (uint ki = 0; ki < BK; ki++) {
                P_cache[qi][ki] = 0.0f;
                dP_cache[qi][ki] = 0.0f;
                dS_cache[qi][ki] = 0.0f;
            }
        }

        [unroll]
        for (uint qi = 0; qi < BQ; qi++) {
            uint q_pos = q_start + qi;
            if (q_pos >= pc.N) break;

            // Load LSE for this query position.
            float lse_qi = args.lse[b * pc.H * pc.N + h * pc.N + q_pos];

            // Compute row_dot for softmax backward: sum_ki(P[qi,ki] * dP[qi,ki])
            float row_dot = 0.0f;

            [unroll]
            for (uint ki = 0; ki < BK; ki++) {
                uint k_pos = kv_start + ki;
                if (k_pos >= pc.S) break;
                if (pc.is_causal != 0u && k_pos > q_pos) continue;

                // Recompute score S = Q @ K^T * scale.
                float score = 0.0f;
                for (uint d = 0; d < pc.D; d++) {
                    score += Q_smem[qi * HEAD_DIM + d]
                           * K_smem[ki * HEAD_DIM + d];
                }
                score *= pc.scale;

                // P = exp(score - lse_qi)  (online-softmax: lse = m + log(l))
                float p_val = exp(score - lse_qi);

                // dP = dO @ V^T: dP[qi,ki] = sum_d(dO[qi,d] * V[ki,d])
                float dp_val = 0.0f;
                for (uint d = 0; d < pc.D; d++) {
                    dp_val += dO_smem[qi * HEAD_DIM + d]
                            * V_smem[ki * HEAD_DIM + d];
                }

                P_cache[qi][ki] = p_val;
                dP_cache[qi][ki] = dp_val;
                row_dot += p_val * dp_val;
            }

            // Softmax backward: dS = P * (dP - row_dot)
            [unroll]
            for (uint ki = 0; ki < BK; ki++) {
                uint k_pos = kv_start + ki;
                if (k_pos >= pc.S) break;
                if (pc.is_causal != 0u && k_pos > q_pos) continue;

                float ds = P_cache[qi][ki]
                         * (dP_cache[qi][ki] - row_dot);
                dS_cache[qi][ki] = ds;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // ── Phase 3: Accumulate dQ, dK, dV from dS and P ───────────
        // dQ[qi,d] += sum_ki(dS[qi,ki] * K[ki,d])
        // dK[ki,d] += sum_qi(dS[qi,ki] * Q[qi,d])   (atomic)
        // dV[ki,d] += sum_qi(P[qi,ki] * dO[qi,d])   (atomic)

        [unroll]
        for (uint qi = 0; qi < BQ; qi++) {
            uint q_pos = q_start + qi;
            if (q_pos >= pc.N) break;

            // dQ accumulation: dQ[qi,d] += sum_ki(dS[qi,ki] * K[ki,d])
            if (tid < pc.D) {
                float dq_val = 0.0f;
                [unroll]
                for (uint ki = 0; ki < BK; ki++) {
                    uint k_pos = kv_start + ki;
                    if (k_pos >= pc.S) break;
                    if (pc.is_causal != 0u && k_pos > q_pos) continue;
                    dq_val += dS_cache[qi][ki]
                            * K_smem[ki * HEAD_DIM + tid];
                }
                dQ_acc[qi] += dq_val * pc.scale;  // chain-rule scale factor
            }
        }

        // dK and dV: atomic-add (multiple Q tiles contribute).
        [unroll]
        for (uint ki = 0; ki < BK; ki++) {
            uint k_pos = kv_start + ki;
            if (k_pos >= pc.S) break;
            if (tid >= pc.D) continue;

            // dV[ki,d] += sum_qi(P[qi,ki] * dO[qi,d])
            float dv_val = 0.0f;
            [unroll]
            for (uint qi = 0; qi < BQ; qi++) {
                uint q_pos = q_start + qi;
                if (q_pos >= pc.N) break;
                if (pc.is_causal != 0u && k_pos > q_pos) continue;
                dv_val += P_cache[qi][ki]
                        * dO_smem[qi * HEAD_DIM + tid];
            }
            if (dv_val != 0.0f) {
                uint dv_idx = _kv_offset(b, kv_h, k_pos, tid);
                vk_atomic_add(args.dV, dv_idx, dv_val);
            }

            // dK[ki,d] += sum_qi(dS[qi,ki] * Q[qi,d]) * scale
            float dk_val = 0.0f;
            [unroll]
            for (uint qi = 0; qi < BQ; qi++) {
                uint q_pos = q_start + qi;
                if (q_pos >= pc.N) break;
                if (pc.is_causal != 0u && k_pos > q_pos) continue;
                dk_val += dS_cache[qi][ki]
                        * Q_smem[qi * HEAD_DIM + tid];
            }
            if (dk_val != 0.0f) {
                uint dk_idx = _kv_offset(b, kv_h, k_pos, tid);
                vk_atomic_add(args.dK, dk_idx, dk_val * pc.scale);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // ── Write dQ output ────────────────────────────────────────────────
    [unroll]
    for (uint qi = 0; qi < BQ; qi++) {
        uint q_pos = q_start + qi;
        if (q_pos >= pc.N) break;
        if (tid < pc.D) {
            args.dQ[_q_offset(b, h, q_pos, tid)] = dQ_acc[qi];
        }
    }
}
