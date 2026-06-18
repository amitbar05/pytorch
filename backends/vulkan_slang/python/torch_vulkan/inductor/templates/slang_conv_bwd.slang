// CG.M6 — Conv2D backward via [Differentiable] conv_inner_madd
// Single-kernel backward computes dX, dW, and dB in one dispatch.
//
// Autodiff over conv_inner_madd(a, b, acc) gives:
//   dL/da = dL/dout * b    → dX element
//   dL/db = dL/dout * a    → dW element
//
// Grid:  (ceil(oW / TILE_W), ceil(oH / TILE_H), N * ceil(C_out / TILE_C))
// numthreads: (THREADS_W, THREADS_H, 1)
//
// Supported: groups==1, f32 only.
//
// Atomic accumulators on dX/dW/dB go through `vk_atomic_add` (CAS loop on
// uint-reinterpreted buffer) instead of native `InterlockedAdd<float>`.
// Mesa ACO on RDNA1 cannot lower `buffer_atomic_add_f32` and crashes at
// shader assembly; the CAS path is portable and the canonical fallback
// established by `lib/atomics.slang`.
//
// M20.3: Tile sizes are specialization constants (``[[vk::constant_id]]``).
// One SPIR-V module per (dtype, has_bias) configuration is reused across
// every (TILE_W, TILE_H, TILE_C, THREADS_W, THREADS_H) tuple — the caller
// passes the tuple via Vulkan spec-constant overrides at pipeline-create
// time. The Jinja default after ``/*}}*/`` keeps the raw .slang file
// passing ``slangc --syntax-check`` standalone.
//
// Constant ID assignment (file-local 40-44; avoids 10-12 used by
// flash_attention, 20-24 by slang_mm, 30-34 by slang_conv2d fwd, and
// 13-15 by flash_attention_bwd):
//   40 — TILE_W      (default 8)
//   41 — TILE_H      (default 8)
//   42 — TILE_C      (default 8)
//   43 — THREADS_W   (default 16)
//   44 — THREADS_H   (default 16)
import atomics;

/*{%*/ set tile_w = tile_w | default(8) /*%}*/
/*{%*/ set tile_h = tile_h | default(8) /*%}*/
/*{%*/ set tile_c = tile_c | default(8) /*%}*/
/*{%*/ set threads_w = threads_w | default(16) /*%}*/
/*{%*/ set threads_h = threads_h | default(16) /*%}*/

// M20.3: Specialization constants — overridden at dispatch time so the
// same SPIR-V module serves all autotuned tile configs.
[[vk::constant_id(40)]] const uint TILE_W = /*{{*/ tile_w /*}}8*/;
[[vk::constant_id(41)]] const uint TILE_H = /*{{*/ tile_h /*}}8*/;
[[vk::constant_id(42)]] const uint TILE_C = /*{{*/ tile_c /*}}8*/;
[[vk::constant_id(43)]] const uint THREADS_W = /*{{*/ threads_w /*}}16*/;
[[vk::constant_id(44)]] const uint THREADS_H = /*{{*/ threads_h /*}}16*/;

// ── Push-constant struct (fits in 128B Vulkan limit) ──────────────────
// grad_input strides = input strides (same NCHW shape).
// grad_weight strides = weight strides (same [C_out, C_in, kH, kW] shape).
// grad_bias is always 1-D contiguous, stride = 1.
struct BwdPC {
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
    // Strides for input / grad_input (NCHW) — same layout
    uint stride_in_n;
    uint stride_in_c;
    uint stride_in_h;
    uint stride_in_w;
    // Strides for weight / grad_weight [C_out, C_in, kH, kW] — same layout
    uint stride_w_out;
    uint stride_w_in;
    uint stride_w_kh;
    uint stride_w_kw;
    // Strides for grad_output (NCHW)
    uint stride_go_n;
    uint stride_go_c;
    uint stride_go_h;
    uint stride_go_w;
    // M-SF.4: stride_grad_bias is always present — gated at runtime by != 0
    // (replaces the legacy has_bias Jinja conditional that produced
    // 27-uint vs 28-uint struct variants; same pattern as fwd stride_bias).
    uint stride_grad_bias;  // = grad_bias.view(-1).stride(0) or 0
};

[[vk::push_constant]] BwdPC pc;

// ── Buffer bindings (ParameterBlock) ────────────────────────────────────
struct KernelArgs {
    StructuredBuffer<float> input;
    StructuredBuffer<float> weight;
    StructuredBuffer<float> grad_out;
    // dX/dW/dB are accumulated atomically; bound as uint so we can CAS on
    // the raw bit pattern. The host pre-zero-fills these tensors so
    // `asfloat(0)==0.0f` is the correct accumulator identity.
    // M-SF.4: grad_bias is always declared — the runtime gate below
    // (stride_grad_bias != 0) skips accumulation when has_bias=False.
    RWStructuredBuffer<uint> grad_input;
    RWStructuredBuffer<uint> grad_weight;
    RWStructuredBuffer<uint> grad_bias;
};
// M21.3.01: explicit Set 0 binding (slangc 2026.7.1 defaults to Set 1)
[[vk::binding(0, 0)]] ParameterBlock<KernelArgs> args;

// ── CG.M6: Differentiable scalar inner — inlined for standalone compilation ──
// Uses [BackwardDerivative] to avoid autodiff coverage warnings.
[Differentiable]
[BackwardDerivative(conv_inner_madd_bwd)]
float conv_inner_madd(float a, float b, no_diff float acc)
{
    return acc + a * b;
}

// Hand-written fast backward for conv_inner_madd.
void conv_inner_madd_bwd(
    inout DifferentialPair<float> dpa,
    inout DifferentialPair<float> dpb,
    no_diff float acc,
    float dout)
{
    dpa = diffPair(dpa.p, dpa.d + dout * dpb.p);
    dpb = diffPair(dpb.p, dpb.d + dout * dpa.p);
}

// ── Entry point ─────────────────────────────────────────────────────────
// M20.3: numthreads now references the spec-constant THREADS_W/THREADS_H so
// the same SPIR-V module handles any (THREADS_W, THREADS_H) combo.
[shader("compute")]
[numthreads(THREADS_W, THREADS_H, 1)]
void computeMain(
    uint3 gtid : SV_DispatchThreadID,
    uint3 gid : SV_GroupID,
    uint3 lid : SV_GroupThreadID)
{
    // ── Decode workgroup position ────────────────────────────────────
    uint tile_w_idx = gid.x;
    uint tile_h_idx = gid.y;
    uint batch_co_tile = gid.z;

    // M20.3: TILE_C / TILE_W / TILE_H are spec constants (vk::constant_id
    // 40-42). Same SPIR-V module serves all autotune tile choices.
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

    // ── Per-thread output position ───────────────────────────────────
    // M-slang-new-4: derive local_ow / local_oh from a linearised
    // ``tid_local`` instead of reading ``lid.x`` / ``lid.y`` directly.
    // Same slangc 2026.5+ wave64 scalarisation bug as M22.13 / M22.13-
    // extras: 5+ StructuredBuffer bindings (args.input, args.weight,
    // args.grad_out, args.grad_input, args.grad_weight, optional
    // args.grad_bias) with both ``lid.x`` and ``lid.y`` reads addressing
    // different buffers triggers the row-collapse fold on RDNA1 wave64.
    //
    // Linearisation rule: workgroup is ``numthreads(THREADS_W,
    // THREADS_H, 1)``, so SPIR-V lane ordering is
    // ``lane = lid.y * THREADS_W + lid.x``. Reconstructing via modular
    // arithmetic is algebraically identical to direct ``lid`` reads but
    // sidesteps the slangc fold. See ``mm_tiled.slang`` /
    // ``slang_conv2d.slang`` for the precedent.
    uint tid_local = lid.y * THREADS_W + lid.x;
    uint local_ow = tid_local % THREADS_W;  // == lid.x — but recovered
    uint local_oh = tid_local / THREADS_W;  // == lid.y — but recovered

    uint ow = ow_start + local_ow;
    uint oh = oh_start + local_oh;

    // ── Precompute bases for this batch ──────────────────────────────
    uint in_batch_base = n * pc.stride_in_n;
    uint go_batch_base = n * pc.stride_go_n;
    uint dx_batch_base = n * pc.stride_in_n  /*dx same as input*/;

    // ── Per-thread dB accumulator (local to this workgroup tile) ─────
    // M20.3: array dimension uses spec-constant TILE_C. SPIR-V supports
    // spec-constant-sized arrays for local (non-groupshared) declarations.
    float db_local[TILE_C];
    [unroll]
    for (uint ci = 0; ci < TILE_C; ci++) {
        db_local[ci] = 0.0f;
    }

    // ── Iterate over output channels assigned to this workgroup ──────
    uint co_idx = 0;
    for (uint co = co_start; co < co_end; co++, co_idx++) {
        if (ow >= ow_end || oh >= oh_end) {
            continue;
        }

        // ── Load grad_output for this (n, co, oh, ow) ───────────────
        uint go_idx = go_batch_base
                    + co * pc.stride_go_c
                    + oh * pc.stride_go_h
                    + ow * pc.stride_go_w;
        float dout = args.grad_out[go_idx];

        // Bias gradient: dB[co] accumulates over all spatial+batch dims
        db_local[co_idx] += dout;

        // Weight base for this output channel
        uint w_co_base = co * pc.stride_w_out;
        // grad_weight base for this output channel
        uint dw_co_base = co * pc.stride_w_out  /*dw same as weight*/;

        // ── Iterate over input channels and kernel ──────────────────
        for (uint ci = 0; ci < pc.C_in; ci++) {
            uint in_c_base = in_batch_base + ci * pc.stride_in_c;
            uint dx_c_base = dx_batch_base + ci * pc.stride_in_c  /*dx same as input*/;
            uint w_ci_base = w_co_base + ci * pc.stride_w_in;
            uint dw_ci_base = dw_co_base + ci * pc.stride_w_in  /*dw same as weight*/;

            for (uint kh = 0; kh < pc.kH; kh++) {
                int ih = int(oh * pc.sH + kh * pc.dH) - int(pc.pH);
                if (ih < 0 || uint(ih) >= pc.iH) continue;

                uint in_h_base = in_c_base + uint(ih) * pc.stride_in_h;
                uint dx_h_base = dx_c_base + uint(ih) * pc.stride_in_h  /*dx same as input*/;
                uint w_kh_base = w_ci_base + kh * pc.stride_w_kh;
                uint dw_kh_base = dw_ci_base + kh * pc.stride_w_kh  /*dw same as weight*/;

                for (uint kw = 0; kw < pc.kW; kw++) {
                    int iw = int(ow * pc.sW + kw * pc.dW) - int(pc.pW);
                    if (iw < 0 || uint(iw) >= pc.iW) continue;

                    float in_val = args.input[in_h_base + uint(iw) * pc.stride_in_w];
                    float w_val = args.weight[w_kh_base + kw * pc.stride_w_kw];

                    // ── bwd_diff(conv_inner_madd) ──────────────────
                    // d_input = dout * w_val, d_weight = dout * in_val
                    DifferentialPair<float> dp_in = diffPair(in_val, 0.0f);
                    DifferentialPair<float> dp_w = diffPair(w_val, 0.0f);
                    bwd_diff(conv_inner_madd)(dp_in, dp_w, 0.0f, dout);
                    float dx_contrib = dp_in.getDifferential();
                    float dw_contrib = dp_w.getDifferential();

                    // ── Atomic accumulate into dX (CAS-based, RDNA1-safe) ──
                    uint dx_idx = dx_h_base + uint(iw) * pc.stride_in_w  /*dx same as input*/;
                    vk_atomic_add(args.grad_input, dx_idx, dx_contrib);

                    // ── Atomic accumulate into dW (CAS-based, RDNA1-safe) ──
                    uint dw_idx = dw_kh_base + kw * pc.stride_w_kw  /*dw same as weight*/;
                    vk_atomic_add(args.grad_weight, dw_idx, dw_contrib);
                }
            }
        }
    }

    // ── Atomic accumulate dB contributions ────────────────────────────
    // M-SF.4: always compiled; runtime-gated by stride_grad_bias (0 = no bias).
    if (pc.stride_grad_bias != 0u) {
        [unroll]
        for (uint ci = 0; ci < TILE_C; ci++) {
            uint co = co_start + ci;
            if (co >= co_end) break;
            if (db_local[ci] != 0.0f) {
                vk_atomic_add(args.grad_bias, co * pc.stride_grad_bias, db_local[ci]);
            }
        }
    }
}
