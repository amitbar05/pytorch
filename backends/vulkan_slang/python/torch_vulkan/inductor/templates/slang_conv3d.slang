// Tiled direct Conv3D: out[N, C_out, oD, oH, oW] = conv3d(input[N, C_in, iD, iH, iW], weight[C_out, C_in, kD, kH, kW])
// Optional bias: out += bias[C_out] broadcast to [1, C_out, 1, 1, 1]
//
// Grid:  (ceil(oW / TILE_W), ceil(oH / TILE_H), N * ceil(oD) * ceil(C_out / TILE_C))
// Each workgroup handles one depth slice (od) — no depth tiling (simplification).
// numthreads: (THREADS_W, THREADS_H, 1) where THREADS_W*THREADS_H <= 256
//
// Supported: groups==1, f32 only.
//
// All stride computation is derived from sizes in the shader — inputs
// must be contiguous (enforced by the dispatch caller via .contiguous()).
//
// Push-constant limit: 128B on RDNA1. With 5D shapes we compute strides
// from sizes to stay within the budget.
//
// Epilogue via ``<Epilogue : IDifferentiable>`` (same as Conv2d).
//
// Spec constants (CG.SF.01):
//   50 — TILE_W      (default 4)
//   51 — TILE_H      (default 4)
//   52 — TILE_C      (default 4)
//   53 — THREADS_W   (default 8)
//   54 — THREADS_H   (default 8)

/*{%*/ set tile_w = tile_w | default(4) /*%}*/
/*{%*/ set tile_h = tile_h | default(4) /*%}*/
/*{%*/ set tile_c = tile_c | default(4) /*%}*/
/*{%*/ set threads_w = threads_w | default(8) /*%}*/
/*{%*/ set threads_h = threads_h | default(8) /*%}*/

[[vk::constant_id(50)]] const uint TILE_W = /*{{*/ tile_w /*}}4*/;
[[vk::constant_id(51)]] const uint TILE_H = /*{{*/ tile_h /*}}4*/;
[[vk::constant_id(52)]] const uint TILE_C = /*{{*/ tile_c /*}}4*/;
[[vk::constant_id(53)]] const uint THREADS_W = /*{{*/ threads_w /*}}8*/;
[[vk::constant_id(54)]] const uint THREADS_H = /*{{*/ threads_h /*}}8*/;

struct PC {
    uint N;
    uint C_in;
    uint C_out;
    uint iD;
    uint iH;
    uint iW;
    uint oD;
    uint oH;
    uint oW;
    uint kD;
    uint kH;
    uint kW;
    uint sD;
    uint sH;
    uint sW;
    uint pD;
    uint pH;
    uint pW;
    uint dD;
    uint dH;
    uint dW;
    // CG.SF.01: Runtime tile sizes — loop bounds use these.
    uint tile_w;
    uint tile_h;
    uint tile_c;
    // M-SF.4: always present — gated at runtime by stride_bias != 0
    uint stride_bias;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    // Total: 28 uints = 112B <= 128B Vulkan push-constant limit
};

[[vk::push_constant]] PC pc;

import pointwise;

struct KernelArgs {
    StructuredBuffer<float> input;
    StructuredBuffer<float> weight;
    // M-SF.4: always present — gated at runtime by stride_bias != 0
    StructuredBuffer<float> bias;
    RWStructuredBuffer<float> out;
};
// M21.3.01: explicit Set 0 binding
[[vk::binding(0, 0)]] ParameterBlock<KernelArgs> args;

[shader("compute")]
[numthreads(THREADS_W, THREADS_H, 1)]
void computeMain<Epilogue : IDifferentiable>(
    uint3 gtid : SV_DispatchThreadID,
    uint3 gid : SV_GroupID,
    uint3 lid : SV_GroupThreadID)
{
    // -- Compute strides from sizes (all inputs are contiguous NCDHW) --
    // input [N, C_in, iD, iH, iW]: strides [C_in*iD*iH*iW, iD*iH*iW, iH*iW, iW, 1]
    uint stride_in_n = pc.C_in * pc.iD * pc.iH * pc.iW;
    uint stride_in_c = pc.iD * pc.iH * pc.iW;
    uint stride_in_d = pc.iH * pc.iW;
    uint stride_in_h = pc.iW;
    // uint stride_in_w = 1;

    // weight [C_out, C_in, kD, kH, kW]: strides [C_in*kD*kH*kW, kD*kH*kW, kH*kW, kW, 1]
    uint stride_w_out = pc.C_in * pc.kD * pc.kH * pc.kW;
    uint stride_w_in = pc.kD * pc.kH * pc.kW;
    uint stride_w_kd = pc.kH * pc.kW;
    uint stride_w_kh = pc.kW;
    // uint stride_w_kw = 1;

    // output [N, C_out, oD, oH, oW]: strides [C_out*oD*oH*oW, oD*oH*oW, oH*oW, oW, 1]
    uint stride_out_n = pc.C_out * pc.oD * pc.oH * pc.oW;
    uint stride_out_c = pc.oD * pc.oH * pc.oW;
    uint stride_out_d = pc.oH * pc.oW;
    uint stride_out_h = pc.oW;
    // uint stride_out_w = 1;

    // -- Decode workgroup position --
    uint tile_w_idx = gid.x;
    uint tile_h_idx = gid.y;
    // gid.z encodes: od + oD * (n * tile_c_count + co_tile)
    uint tile_c_count = (pc.C_out + pc.tile_c - 1u) / pc.tile_c;
    uint nz_co = gid.z;
    uint od = nz_co % pc.oD;
    uint batch_co_tile = nz_co / pc.oD;
    uint n = batch_co_tile / tile_c_count;
    uint co_tile = batch_co_tile % tile_c_count;

    if (n >= pc.N || co_tile >= tile_c_count) return;

    uint co_start = co_tile * pc.tile_c;
    uint co_end = min(co_start + pc.tile_c, pc.C_out);

    uint ow_start = tile_w_idx * pc.tile_w;
    uint oh_start = tile_h_idx * pc.tile_h;
    uint ow_end = min(ow_start + pc.tile_w, pc.oW);
    uint oh_end = min(oh_start + pc.tile_h, pc.oH);

    // -- Per-thread output position (linearised to avoid slangc wave64 bug) --
    uint tid_local = lid.y * THREADS_W + lid.x;
    uint local_ow = tid_local % THREADS_W;
    uint local_oh = tid_local / THREADS_W;

    // CG.SF.01: Early exit for threads outside the runtime workgroup.
    if (local_ow >= pc.tile_w || local_oh >= pc.tile_h) return;

    uint ow = ow_start + local_ow;
    uint oh = oh_start + local_oh;

    // -- Precompute input base for this batch --
    uint in_batch_base = n * stride_in_n;

    // -- Compute input depth base for this output depth --
    int id_base_signed = int(od * pc.sD) - int(pc.pD);

    // -- Iterate over output channels assigned to this workgroup --
    for (uint co = co_start; co < co_end; co++) {
        if (ow >= ow_end || oh >= oh_end) {
            // Outside tile boundary — skip store.
        }

        float sum = 0.0f;

        // Weight base for this output channel
        uint w_co_base = co * stride_w_out;

        // Iterate over input channels and 3D kernel
        for (uint ci = 0; ci < pc.C_in; ci++) {
            uint in_c_base = in_batch_base + ci * stride_in_c;
            uint w_ci_base = w_co_base + ci * stride_w_in;

            for (uint kd = 0; kd < pc.kD; kd++) {
                int id_val = id_base_signed + int(kd * pc.dD);
                if (id_val < 0 || uint(id_val) >= pc.iD) continue;

                uint in_d_base = in_c_base + uint(id_val) * stride_in_d;
                uint w_kd_base = w_ci_base + kd * stride_w_kd;

                for (uint kh = 0; kh < pc.kH; kh++) {
                    int ih = int(oh * pc.sH + kh * pc.dH) - int(pc.pH);
                    if (ih < 0 || uint(ih) >= pc.iH) continue;

                    uint in_h_base = in_d_base + uint(ih) * stride_in_h;
                    uint w_kh_base = w_kd_base + kh * stride_w_kh;

                    for (uint kw = 0; kw < pc.kW; kw++) {
                        int iw = int(ow * pc.sW + kw * pc.dW) - int(pc.pW);
                        if (iw < 0 || uint(iw) >= pc.iW) continue;

                        float in_val = args.input[in_h_base + uint(iw)];
                        float w_val = args.weight[w_kh_base + kw];
                        sum += in_val * w_val;
                    }
                }
            }
        }

        // Write output
        if (ow < ow_end && oh < oh_end) {
            uint out_idx = n * stride_out_n
                         + co * stride_out_c
                         + od * stride_out_d
                         + oh * stride_out_h
                         + ow;
            // M-SF.4: bias always present, gated at runtime
            if (pc.stride_bias != 0u) sum += args.bias[co * pc.stride_bias];
/*{%*/ if epilogue /*%}*/
            sum = Epilogue::apply(sum);
/*{%*/ endif /*%}*/
            args.out[out_idx] = sum;
        }
    }
}
