// CG.M6 Conv3D backward via [Differentiable] conv_inner_madd
// Single-kernel backward computes dX, dW, and dB in one dispatch.
//
// Autodiff over conv_inner_madd(a, b, acc) gives:
//   dL/da = dL/dout * b    -> dX element
//   dL/db = dL/dout * a    -> dW element
//
// Grid:  (ceil(oW / TILE_W), ceil(oH / TILE_H), N * oD * ceil(C_out / TILE_C))
// Each workgroup handles one output-depth slice (od).
// numthreads: (THREADS_W, THREADS_H, 1)
//
// Supported: groups==1, f32 only.
//
// All strides computed from sizes in the shader (inputs must be contiguous).
//
// Atomic accumulators on dX/dW/dB via vk_atomic_add (CAS loop, RDNA1-safe).
//
// Constant ID assignment (file-local 50-54; avoids 10-12 used by
// flash_attention, 20-24 by slang_mm, 30-34 by slang_conv2d fwd,
// 40-44 by slang_conv_bwd, 13-15 by flash_attention_bwd):
//   50 -- TILE_W      (default 4)
//   51 -- TILE_H      (default 4)
//   52 -- TILE_C      (default 4)
//   53 -- THREADS_W   (default 8)
//   54 -- THREADS_H   (default 8)
import atomics;

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

// -- Push-constant struct (fits in 128B Vulkan limit) --
// All strides derived from sizes. Inputs must be contiguous.
struct BwdPC {
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
    // Runtime tile sizes (CG.SF.01)
    uint tile_w;
    uint tile_h;
    uint tile_c;
    // has_bias: 1 if bias gradient is needed
    uint has_bias;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    // Total: 28 uints = 112B <= 128B Vulkan push constant limit
};

[[vk::push_constant]] BwdPC pc;

// -- Buffer bindings (ParameterBlock) --
struct KernelArgs {
    StructuredBuffer<float> input;
    StructuredBuffer<float> weight;
    StructuredBuffer<float> grad_out;
    // dX/dW/dB are accumulated atomically; bound as uint so we can CAS.
    RWStructuredBuffer<uint> grad_input;
    RWStructuredBuffer<uint> grad_weight;
    RWStructuredBuffer<uint> grad_bias;
};
// M21.3.01: explicit Set 0 binding
[[vk::binding(0, 0)]] ParameterBlock<KernelArgs> args;

// -- CG.M6: Differentiable scalar inner --
[Differentiable]
[BackwardDerivative(conv_inner_madd_bwd)]
float conv_inner_madd(float a, float b, no_diff float acc)
{
    return acc + a * b;
}

void conv_inner_madd_bwd(
    inout DifferentialPair<float> dpa,
    inout DifferentialPair<float> dpb,
    no_diff float acc,
    float dout)
{
    dpa = diffPair(dpa.p, dpa.d + dout * dpb.p);
    dpb = diffPair(dpb.p, dpb.d + dout * dpa.p);
}

// -- Entry point --
[shader("compute")]
[numthreads(THREADS_W, THREADS_H, 1)]
void computeMain(
    uint3 gtid : SV_DispatchThreadID,
    uint3 gid : SV_GroupID,
    uint3 lid : SV_GroupThreadID)
{
    // -- Compute strides from sizes (all contiguous NCDHW) --
    uint stride_in_n = pc.C_in * pc.iD * pc.iH * pc.iW;
    uint stride_in_c = pc.iD * pc.iH * pc.iW;
    uint stride_in_d = pc.iH * pc.iW;
    uint stride_in_h = pc.iW;
    // uint stride_in_w = 1;

    uint stride_w_out = pc.C_in * pc.kD * pc.kH * pc.kW;
    uint stride_w_in = pc.kD * pc.kH * pc.kW;
    uint stride_w_kd = pc.kH * pc.kW;
    uint stride_w_kh = pc.kW;
    // uint stride_w_kw = 1;

    uint stride_go_n = pc.C_out * pc.oD * pc.oH * pc.oW;
    uint stride_go_c = pc.oD * pc.oH * pc.oW;
    uint stride_go_d = pc.oH * pc.oW;
    uint stride_go_h = pc.oW;
    // uint stride_go_w = 1;

    // -- Decode workgroup position --
    uint tile_w_idx = gid.x;
    uint tile_h_idx = gid.y;
    uint nz_co = gid.z;
    uint od = nz_co % pc.oD;
    uint batch_co_tile = nz_co / pc.oD;

    uint tile_c_count = (pc.C_out + TILE_C - 1u) / TILE_C;
    uint n = batch_co_tile / tile_c_count;
    uint co_tile = batch_co_tile % tile_c_count;

    if (n >= pc.N || co_tile >= tile_c_count) return;

    uint co_start = co_tile * TILE_C;
    uint co_end = min(co_start + TILE_C, pc.C_out);

    uint ow_start = tile_w_idx * TILE_W;
    uint oh_start = tile_h_idx * TILE_H;
    uint ow_end = min(ow_start + TILE_W, pc.oW);
    uint oh_end = min(oh_start + TILE_H, pc.oH);

    // -- Per-thread output position (linearised for slangc wave64 bug) --
    uint tid_local = lid.y * THREADS_W + lid.x;
    uint local_ow = tid_local % THREADS_W;
    uint local_oh = tid_local / THREADS_W;

    uint ow = ow_start + local_ow;
    uint oh = oh_start + local_oh;

    // -- Precompute bases for this batch --
    uint in_batch_base = n * stride_in_n;
    uint go_batch_base = n * stride_go_n;
    uint dx_batch_base = n * stride_in_n  /*dx same as input*/;

    // -- Per-thread dB accumulator (local to this workgroup tile) --
    float db_local[TILE_C];
    [unroll]
    for (uint ci_init = 0; ci_init < TILE_C; ci_init++) {
        db_local[ci_init] = 0.0f;
    }

    // -- Iterate over output channels assigned to this workgroup --
    uint co_idx = 0;
    for (uint co = co_start; co < co_end; co++, co_idx++) {
        if (ow >= ow_end || oh >= oh_end) {
            continue;
        }

        // -- Load grad_output for this (n, co, od, oh, ow) --
        uint go_idx = go_batch_base
                    + co * stride_go_c
                    + od * stride_go_d
                    + oh * stride_go_h
                    + ow;
        float dout = args.grad_out[go_idx];

        // Bias gradient: dB[co] accumulates over all spatial+batch dims
        db_local[co_idx] += dout;

        // Weight base for this output channel
        uint w_co_base = co * stride_w_out;
        // grad_weight base for this output channel
        uint dw_co_base = co * stride_w_out  /*dw same as weight*/;

        // -- Compute input depth base for this output depth --
        int id_base_signed = int(od * pc.sD) - int(pc.pD);

        // -- Iterate over input channels and 3D kernel --
        for (uint ci = 0; ci < pc.C_in; ci++) {
            uint in_c_base = in_batch_base + ci * stride_in_c;
            uint dx_c_base = dx_batch_base + ci * stride_in_c  /*dx same*/;
            uint w_ci_base = w_co_base + ci * stride_w_in;
            uint dw_ci_base = dw_co_base + ci * stride_w_in  /*dw same*/;

            for (uint kd = 0; kd < pc.kD; kd++) {
                int id_val = id_base_signed + int(kd * pc.dD);
                if (id_val < 0 || uint(id_val) >= pc.iD) continue;

                uint in_d_base = in_c_base + uint(id_val) * stride_in_d;
                uint dx_d_base = dx_c_base + uint(id_val) * stride_in_d  /*dx same*/;
                uint w_kd_base = w_ci_base + kd * stride_w_kd;
                uint dw_kd_base = dw_ci_base + kd * stride_w_kd  /*dw same*/;

                for (uint kh = 0; kh < pc.kH; kh++) {
                    int ih = int(oh * pc.sH + kh * pc.dH) - int(pc.pH);
                    if (ih < 0 || uint(ih) >= pc.iH) continue;

                    uint in_h_base = in_d_base + uint(ih) * stride_in_h;
                    uint dx_h_base = dx_d_base + uint(ih) * stride_in_h  /*dx same*/;
                    uint w_kh_base = w_kd_base + kh * stride_w_kh;
                    uint dw_kh_base = dw_kd_base + kh * stride_w_kh  /*dw same*/;

                    for (uint kw = 0; kw < pc.kW; kw++) {
                        int iw = int(ow * pc.sW + kw * pc.dW) - int(pc.pW);
                        if (iw < 0 || uint(iw) >= pc.iW) continue;

                        float in_val = args.input[in_h_base + uint(iw)];
                        float w_val = args.weight[w_kh_base + kw];

                        // -- bwd_diff(conv_inner_madd) --
                        // d_input = dout * w_val, d_weight = dout * in_val
                        DifferentialPair<float> dp_in = diffPair(in_val, 0.0f);
                        DifferentialPair<float> dp_w = diffPair(w_val, 0.0f);
                        bwd_diff(conv_inner_madd)(dp_in, dp_w, 0.0f, dout);
                        float dx_contrib = dp_in.getDifferential();
                        float dw_contrib = dp_w.getDifferential();

                        // -- Atomic accumulate into dX --
                        uint dx_idx = dx_h_base + uint(iw)  /*dx same as input*/;
                        vk_atomic_add(args.grad_input, dx_idx, dx_contrib);

                        // -- Atomic accumulate into dW --
                        uint dw_idx = dw_kh_base + kw  /*dw same as weight*/;
                        vk_atomic_add(args.grad_weight, dw_idx, dw_contrib);
                    }
                }
            }
        }
    }

    // -- Atomic accumulate dB contributions --
    if (pc.has_bias != 0u) {
        [unroll]
        for (uint ci_db = 0; ci_db < TILE_C; ci_db++) {
            uint co_db = co_start + ci_db;
            if (co_db >= co_end) break;
            if (db_local[ci_db] != 0.0f) {
                vk_atomic_add(args.grad_bias, co_db * 1u, db_local[ci_db]);
            }
        }
    }
}
