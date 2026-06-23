// Combined Conv2D + GroupNorm + ReLU — single Vulkan dispatch.
// M17.2 Phase 3: true 1-dispatch fusion, eliminating the intermediate
// conv-output buffer and the second dispatch of the Phase 2 approach.
//
// Workgroup decomposition: one WG per (batch, group) row.
// Grid:  (N * G, 1, 1)   where G = num_groups for GroupNorm
// numthreads: (64, 1, 1) — single wave64 (avoids slangc 2026.7.1
//   multi-wave write-coverage miscompile, Group-D 2026-05-21).
//
// Each WG handles group_size = (C_out / G) * oH * oW output elements.
// 64 threads stride across the group_size elements.
//
// PyTorch semantics being matched (SmallCNN-style):
//   ``F.relu(F.group_norm(F.conv2d(x, w, bias)))``
//   i.e. compose order is  ReLU ∘ GN ∘ (Conv+bias).
//
// Algorithm (2-pass, no LDS output buffer — works for any group_size):
//   1. Each thread computes its share of conv outputs (strided),
//      applies bias, and accumulates Welford stats over the
//      pre-normalisation conv+bias outputs.  (NO ReLU here — GN
//      must see the raw conv+bias values, otherwise the mean/var
//      collapse onto the clamped distribution and the post-GN
//      result diverges from CPU by O(1).)
//   2. Workgroup-level Welford reduction → mean, var, rstd.
//   3. Each thread recomputes its share, normalises with the
//      Welford stats, applies the GN affine (weight, bias),
//      finally applies ReLU, and writes to output.
//
// Conv groups=1 only (not depthwise).  GN groups can be any divisor of C_out.
// f32 only.
//
// Bug-history note (Group-D fix, 2026-05-21):
// Earlier versions of this shader applied ReLU **before** the Welford
// accumulation, computing ``gn(relu(conv(x)))`` instead of the requested
// ``relu(gn(conv(x)))``.  The two are not equivalent — the Welford stats
// differ, and the unrelued post-GN tensor still contains negative values
// (which never appear in the CPU oracle), so SmallCNN
// ``test_forward_vs_cpu`` failed with ``max abs err ≈ 0.87``.
//
// Jinja2 parameters:
//   group_size      — (C_out / G) * oH * oW  (literal for documentation)
//   has_bias        — whether conv bias tensor is present

import reduction;
import pointwise;  // for IDifferentiable (OpReLU, OpIdentity, etc.)

struct PC {
    // ── Conv dimensions ──
    uint N;
    uint C_in;
    uint C_out;
    uint iH;
    uint iW;
    uint oH;
    uint oW;
    uint kH;
    uint kW;
    uint sH;
    uint sW;
    uint pH;
    uint pW;
    uint dH;
    uint dW;
    // ── Strides for input [N, C_in, iH, iW] ──
    uint stride_in_n;
    uint stride_in_c;
    uint stride_in_h;
    uint stride_in_w;
    // ── Strides for weight [C_out, C_in, kH, kW] ──
    uint stride_w_out;
    uint stride_w_in;
    uint stride_w_kh;
    uint stride_w_kw;
    // ── Strides for output [N, C_out, oH, oW] ──
    uint stride_out_n;
    uint stride_out_c;
    uint stride_out_h;
    uint stride_out_w;
    // ── GroupNorm params ──
    // (channels_per_group = C_out / num_groups, derived in shader)
    // (spatial_size = oH * oW, derived in shader)
    uint num_groups;
    uint group_size;
    float eps;
    // ── Bias stride (0 if no bias) ──
    uint stride_bias;
};  // PC struct = 30 uints + 1 float + 1 uint = 128 bytes (RDNA1 push-constant cap)

[[vk::push_constant]] PC pc;

struct KernelArgs {
    StructuredBuffer<float> input;      // [N, C_in, iH, iW]
    StructuredBuffer<float> weight;     // [C_out, C_in, kH, kW]
    StructuredBuffer<float> bias;       // [C_out]  (or dummy if no bias)
    StructuredBuffer<float> gn_weight;  // [C_out]  gamma
    StructuredBuffer<float> gn_bias;    // [C_out]  beta
    RWStructuredBuffer<float> out;      // [N, C_out, oH, oW]
};
// M21.3.01: explicit Set 0 binding (slangc 2026.7.1 defaults to Set 1)
[[vk::binding(0, 0)]] ParameterBlock<KernelArgs> args;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(
    uint3 gid : SV_GroupID,
    uint3 lid : SV_GroupThreadID)
{
    uint row = gid.x;  // n * G + group_idx
    uint tid = lid.x;

    if (row >= pc.N * pc.num_groups) return;

    uint n         = row / pc.num_groups;
    uint group_idx = row % pc.num_groups;
    // Derived (kept out of PC to fit 128B push-constant cap):
    uint channels_per_group = pc.C_out / pc.num_groups;
    uint spatial_size       = pc.oH * pc.oW;
    uint co_start  = group_idx * channels_per_group;
    uint S         = pc.group_size;

    // Precompute base offsets
    uint in_batch_base  = n * pc.stride_in_n;
    uint out_batch_base = n * pc.stride_out_n;

    // ═══════════════════════════════════════════════════════════════
    // Pass 1: compute conv + bias, accumulate Welford stats over the
    // pre-norm distribution.  ReLU is intentionally NOT applied here
    // because PyTorch composes as ``relu(gn(conv(x)))`` — GN must see
    // the raw conv+bias values.
    // ═══════════════════════════════════════════════════════════════
    // M20.7: replaced inline (local_mean, local_m2, local_n) triple
    // with `Welford` struct + `welford_update()` from helpers.slang
    // (re-exported by reduction.slang).  The accumulator is
    // zero-initialised and welford_update() maintains the invariant
    // that (mean, M2, n) satisfy Welford's online algorithm after each
    // element.
    Welford local_wf = { 0.0f, 0.0f, 0.0f };

    for (uint d = tid; d < S; d += 64) {
        // Decode (co, oh, ow) from flat index within the group row
        uint co_local    = d / spatial_size;
        uint spatial_idx = d % spatial_size;
        uint oh          = spatial_idx / pc.oW;
        uint ow          = spatial_idx % pc.oW;
        uint co          = co_start + co_local;

        // ── Conv2d inner loop ──
        float sum = 0.0f;
        uint w_co_base = co * pc.stride_w_out;

        for (uint ci = 0u; ci < pc.C_in; ci++) {
            uint in_c_base   = in_batch_base + ci * pc.stride_in_c;
            uint w_ci_base   = w_co_base + ci * pc.stride_w_in;

            for (uint kh = 0u; kh < pc.kH; kh++) {
                int ih = int(oh * pc.sH + kh * pc.dH) - int(pc.pH);
                if (ih < 0 || uint(ih) >= pc.iH) continue;

                uint in_h_base  = in_c_base + uint(ih) * pc.stride_in_h;
                uint w_kh_base  = w_ci_base + kh * pc.stride_w_kh;

                for (uint kw = 0u; kw < pc.kW; kw++) {
                    int iw = int(ow * pc.sW + kw * pc.dW) - int(pc.pW);
                    if (iw < 0 || uint(iw) >= pc.iW) continue;

                    float in_val = args.input[in_h_base + uint(iw) * pc.stride_in_w];
                    float w_val  = args.weight[w_kh_base + kw * pc.stride_w_kw];
                    sum += in_val * w_val;
                }
            }
        }

        // ── Bias (always emitted; dummy zero buffer when no bias) ──
        sum += args.bias[co];

        // ── Online Welford via shared helper ──
        // Pre-norm value; the fix vs the old shader is that we do NOT
        // pre-clamp at ReLU before accumulating.
        welford_update(local_wf, sum);
    }

    // ═══════════════════════════════════════════════════════════════
    // Pass 2: workgroup-level Welford reduction → mean, var, rstd
    // ═══════════════════════════════════════════════════════════════
    WelfordResult<float> wf;
    wf.mean = local_wf.mean;
    wf.m2   = local_wf.m2;
    wf.n    = local_wf.n;
    WelfordResult<float> stats = wg_welford(wf, tid, 64u, VK_SUBGROUP_SIZE);

    float mean = stats.mean;
    float var  = (stats.n > 0.0f) ? stats.m2 / stats.n : 0.0f;
    float rstd = 1.0f / sqrt(var + pc.eps);

    // ═══════════════════════════════════════════════════════════════
    // Pass 3: recompute conv + bias, normalize with the Welford
    // statistics, apply GN affine, then ReLU, then store.
    // PyTorch semantics: ``relu(gn(conv(x) + bias))`` — ReLU is the
    // OUTERMOST op, so it runs after the affine transform.
    //
    // NOTE (Group-D 2026-05-21): slangc 2026.7.1 has a write-coverage
    // miscompile in this shader — Wave 1 (lanes 64..127 on RDNA1
    // wave64) silently drops its ``OpStore`` when Pass 3's stored
    // value depends on both the conv-load chain AND the welford
    // results.  Verified via probe_cgr_write_pattern.py.  Multiple
    // loop-shape rewrites in this file did not lift the miscompile,
    // so the Python caller (``_slang_tile_conv2d_gn_relu``) now
    // bypasses this kernel and falls back to a 3-dispatch decomp
    // (conv + group_norm + relu) until either the slangc bug is
    // narrowed or a different shader structure is found.  This block
    // is preserved as a regression baseline + documentation.
    // M-CG.3 (2026-05-27): reduced WG from 256→64 to avoid the slangc
    // 2026.7.1 multi-wave write-coverage miscompile.  With single-wave,
    // the store is never dropped.
    // ═══════════════════════════════════════════════════════════════
    for (uint d = tid; d < S; d += 64) {
        uint co_local    = d / spatial_size;
        uint spatial_idx = d % spatial_size;
        uint oh          = spatial_idx / pc.oW;
        uint ow          = spatial_idx % pc.oW;
        uint co          = co_start + co_local;

        // ── Conv2d inner loop (recomputed) ──
        float sum = 0.0f;
        uint w_co_base = co * pc.stride_w_out;

        for (uint ci = 0u; ci < pc.C_in; ci++) {
            uint in_c_base   = in_batch_base + ci * pc.stride_in_c;
            uint w_ci_base   = w_co_base + ci * pc.stride_w_in;

            for (uint kh = 0u; kh < pc.kH; kh++) {
                int ih = int(oh * pc.sH + kh * pc.dH) - int(pc.pH);
                if (ih < 0 || uint(ih) >= pc.iH) continue;

                uint in_h_base  = in_c_base + uint(ih) * pc.stride_in_h;
                uint w_kh_base  = w_ci_base + kh * pc.stride_w_kh;

                for (uint kw = 0u; kw < pc.kW; kw++) {
                    int iw = int(ow * pc.sW + kw * pc.dW) - int(pc.pW);
                    if (iw < 0 || uint(iw) >= pc.iW) continue;

                    float in_val = args.input[in_h_base + uint(iw) * pc.stride_in_w];
                    float w_val  = args.weight[w_kh_base + kw * pc.stride_w_kw];
                    sum += in_val * w_val;
                }
            }
        }

        // ── Bias (always emitted; dummy zero buffer when no bias) ──
        sum += args.bias[co];

        // ── Normalize the raw conv+bias value, then apply GN affine ──
        // Mirrors ``F.group_norm`` exactly: ``(x - mean) * rstd * gamma + beta``.
        float normed = (sum - mean) * rstd;
        float affine = normed * args.gn_weight[co] + args.gn_bias[co];

        // ── ReLU (outermost op) ──
        float result = max(affine, 0.0f);

        uint out_idx = out_batch_base
                     + co * pc.stride_out_c
                     + oh * pc.stride_out_h
                     + ow * pc.stride_out_w;
        args.out[out_idx] = result;
    }
}
