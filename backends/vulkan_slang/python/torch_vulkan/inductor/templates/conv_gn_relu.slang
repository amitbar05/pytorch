// Combined Conv2D + GroupNorm + ReLU — single Vulkan dispatch.
// M17.2 Phase 3: true 1-dispatch fusion, eliminating the intermediate
// conv-output buffer and the second dispatch of the Phase 2 approach.
//
// Workgroup decomposition: one WG per (batch, group) row.
// Grid:  (N * G, 1, 1)   where G = num_groups for GroupNorm
// numthreads: (256, 1, 1)
//
// Each WG handles group_size = (C_out / G) * oH * oW output elements.
// 256 threads stride across the group_size elements.
//
// Algorithm (2-pass, no LDS output buffer — works for any group_size):
//   1. Each thread computes its share of conv outputs (strided),
//      applies bias + ReLU, and accumulates Welford stats locally.
//   2. Workgroup-level Welford reduction → mean, var, rstd.
//   3. Each thread recomputes its share, normalizes, applies
//      GN affine (weight, bias), and writes to output.
//
// Conv groups=1 only (not depthwise).  GN groups can be any divisor of C_out.
// f32 only.
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
    uint num_groups;
    uint group_size;
    uint channels_per_group;
    uint spatial_size;
    float eps;
    // ── Bias stride (0 if no bias) + pad ──
    uint stride_bias;
    uint _pad;
};

[[vk::push_constant]] PC pc;

struct KernelArgs {
    StructuredBuffer<float> input;      // [N, C_in, iH, iW]
    StructuredBuffer<float> weight;     // [C_out, C_in, kH, kW]
    StructuredBuffer<float> bias;       // [C_out]  (or dummy if no bias)
    StructuredBuffer<float> gn_weight;  // [C_out]  gamma
    StructuredBuffer<float> gn_bias;    // [C_out]  beta
    RWStructuredBuffer<float> out;      // [N, C_out, oH, oW]
};
ParameterBlock<KernelArgs> args;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(
    uint3 gid : SV_GroupID,
    uint3 lid : SV_GroupThreadID)
{
    uint row = gid.x;  // n * G + group_idx
    uint tid = lid.x;

    if (row >= pc.N * pc.num_groups) return;

    uint n         = row / pc.num_groups;
    uint group_idx = row % pc.num_groups;
    uint co_start  = group_idx * pc.channels_per_group;
    uint S         = pc.group_size;

    // Precompute base offsets
    uint in_batch_base  = n * pc.stride_in_n;
    uint out_batch_base = n * pc.stride_out_n;

    // ═══════════════════════════════════════════════════════════════
    // Pass 1: compute conv + bias + ReLU, accumulate Welford stats
    // ═══════════════════════════════════════════════════════════════
    float local_mean = 0.0f;
    float local_m2   = 0.0f;
    float local_n    = 0.0f;

    for (uint d = tid; d < S; d += 256) {
        // Decode (co, oh, ow) from flat index within the group row
        uint co_local    = d / pc.spatial_size;
        uint spatial_idx = d % pc.spatial_size;
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

        // ── Bias ──
        /*{%*/ if has_bias /*%}*/
        sum += args.bias[co];
        /*{%*/ endif /*%}*/

        // ── ReLU ──
        float relu_val = sum > 0.0f ? sum : 0.0f;

        // ── Online Welford (single-pass mean + M2) ──
        float delta = relu_val - local_mean;
        local_n += 1.0f;
        local_mean += delta / local_n;
        local_m2   += delta * (relu_val - local_mean);
    }

    // ═══════════════════════════════════════════════════════════════
    // Pass 2: workgroup-level Welford reduction → mean, var, rstd
    // ═══════════════════════════════════════════════════════════════
    WelfordResult<float> wf;
    wf.mean = (local_n > 0.0f) ? local_mean : 0.0f;
    wf.m2   = (local_n > 0.0f) ? local_m2   : 0.0f;
    wf.n    = local_n;
    WelfordResult<float> stats = wg_welford(wf, tid, 256u);

    float mean = stats.mean;
    float var  = (stats.n > 0.0f) ? stats.m2 / stats.n : 0.0f;
    float rstd = 1.0f / sqrt(var + pc.eps);

    // ═══════════════════════════════════════════════════════════════
    // Pass 3: recompute conv + bias + ReLU, normalize, affine, store
    // ═══════════════════════════════════════════════════════════════
    for (uint d = tid; d < S; d += 256) {
        uint co_local    = d / pc.spatial_size;
        uint spatial_idx = d % pc.spatial_size;
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

        // ── Bias ──
        /*{%*/ if has_bias /*%}*/
        sum += args.bias[co];
        /*{%*/ endif /*%}*/

        // ── ReLU ──
        float relu_val = sum > 0.0f ? sum : 0.0f;

        // ── Normalize + affine + store ──
        float normed = (relu_val - mean) * rstd;
        float result = normed * args.gn_weight[co] + args.gn_bias[co];

        uint out_idx = out_batch_base
                     + co * pc.stride_out_c
                     + oh * pc.stride_out_h
                     + ow * pc.stride_out_w;
        args.out[out_idx] = result;
    }
}
