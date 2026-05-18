// Tiled direct Conv2D: out[N, C_out, oH, oW] = conv2d(input[N, C_in, iH, iW], weight[C_out, C_in, kH, kW])
// Optional bias: out += bias[C_out] broadcast to [1, C_out, 1, 1]
//
// Workgroup decomposition: each workgroup computes a tile of the output spatial
// dimensions. Within a workgroup, threads cooperatively handle the channel and
// kernel loops. The workgroup grid covers (batch, output channel, output tile).
//
// Grid:  (ceil(oW / TILE_W), ceil(oH / TILE_H), N * ceil(C_out / TILE_C))
// numthreads: (THREADS_W, THREADS_H, 1) where THREADS_W*THREADS_H <= 256
//
// Supported: groups==1, f32 only.
//
// M17.2: Optional epilogue via ``<Epilogue : IDifferentiable>``.
// When ``has_epilogue`` is set, ``Epilogue::apply(sum)`` is called before
// writing the output.  Common epilogues: OpReLU, OpGELU, OpSiLU, OpIdentity.
// The concrete epilogue type is resolved at SPIR-V compile time via the
// ``entry`` parameter (e.g. ``entry="computeMain<OpReLU>"``).
//
// CG.M15: Tile sizes are specialization constants (``[[vk::constant_id]]``).
// They determine numthreads at SPIR-V compile time.  Actual loop bounds use
// push-constant values (``pc.tile_w``, ``pc.tile_h``, ``pc.tile_c``) so the
// same SPIR-V module serves multiple tile configurations.

/*{%*/ set tile_w = tile_w | default(8) /*%}*/
/*{%*/ set tile_h = tile_h | default(8) /*%}*/
/*{%*/ set tile_c = tile_c | default(8) /*%}*/
/*{%*/ set threads_w = threads_w | default(16) /*%}*/
/*{%*/ set threads_h = threads_h | default(16) /*%}*/

// CG.M15: Specialization constants for workgroup shape.
[[vk::constant_id(30)]] const uint TILE_W = /*{{*/ tile_w /*}}8*/;
[[vk::constant_id(31)]] const uint TILE_H = /*{{*/ tile_h /*}}8*/;
[[vk::constant_id(32)]] const uint TILE_C = /*{{*/ tile_c /*}}8*/;
[[vk::constant_id(33)]] const uint THREADS_W = /*{{*/ threads_w /*}}16*/;
[[vk::constant_id(34)]] const uint THREADS_H = /*{{*/ threads_h /*}}16*/;

struct PC {
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
    // Strides for input (NCHW): [C_in*iH*iW, iH*iW, iW, 1]
    uint stride_in_n;
    uint stride_in_c;
    uint stride_in_h;
    uint stride_in_w;
    // Strides for weight: [C_out, C_in, kH, kW] -> [C_in*kH*kW, kH*kW, kW, 1]
    uint stride_w_out;
    uint stride_w_in;
    uint stride_w_kh;
    uint stride_w_kw;
    // Strides for output (NCHW): [C_out*oH*oW, oH*oW, oW, 1]
    uint stride_out_n;
    uint stride_out_c;
    uint stride_out_h;
    uint stride_out_w;
    // CG.M15: Runtime tile sizes — loop bounds use these so the same
    // SPIR-V module (with fixed numthreads from specialization constants)
    // handles multiple tile configurations.
    uint tile_w;
    uint tile_h;
    uint tile_c;
/*{%*/ if has_bias /*%}*/
    uint stride_bias;
    uint _pad; // keep struct size aligned
/*{%*/ endif /*%}*/
};

[[vk::push_constant]] PC pc;

// N+1.10 — Buffer bindings are wrapped in a single
// ``ParameterBlock<KernelArgs>``.  Slang assigns one descriptorTableSlot
// per struct field in declaration order (set 0, binding 0..N-1),
// matching the descriptorCount=1 layout the runtime expects.  The
// dispatch caller (``_slang_tile_conv2d`` in ``vulkan_template_caller.py``)
// passes tensors in struct-field order (input, weight, [bias,] out).
// No `[[vk::binding(N)]]` literals are emitted.

import pointwise;

struct KernelArgs {
    StructuredBuffer<float> input;
    StructuredBuffer<float> weight;
/*{%*/ if has_bias /*%}*/
    StructuredBuffer<float> bias;
/*{%*/ endif /*%}*/
    RWStructuredBuffer<float> out;
};
// M21.3.01: explicit Set 0 binding (slangc 2026.7.1 defaults to Set 1)
[[vk::binding(0, 0)]] ParameterBlock<KernelArgs> args;

[shader("compute")]
[numthreads(THREADS_W, THREADS_H, 1)]
void computeMain<Epilogue : IDifferentiable>(
    uint3 gtid : SV_DispatchThreadID,
    uint3 gid : SV_GroupID,
    uint3 lid : SV_GroupThreadID)
{
    // ── Decode workgroup position ──────────────────────────────────────
    uint tile_w_idx = gid.x;
    uint tile_h_idx = gid.y;
    uint batch_co_tile = gid.z;

    // CG.M15: tile_c_count uses pc.tile_c so the grid decode is correct
    // for the chosen specialization.
    uint tile_c_count = (pc.C_out + pc.tile_c - 1u) / pc.tile_c;
    uint n = batch_co_tile / tile_c_count;
    uint co_tile = batch_co_tile % tile_c_count;

    if (n >= pc.N || co_tile >= tile_c_count) return;

    uint co_start = co_tile * pc.tile_c;
    uint co_end = min(co_start + pc.tile_c, pc.C_out);

    uint ow_start = tile_w_idx * pc.tile_w;
    uint oh_start = tile_h_idx * pc.tile_h;
    uint ow_end = min(ow_start + pc.tile_w, pc.oW);
    uint oh_end = min(oh_start + pc.tile_h, pc.oH);

    // ── Per-thread output position ─────────────────────────────────────
    // M-slang-new-4: derive local_ow / local_oh from a linearised
    // ``tid_local`` instead of reading ``lid.x`` / ``lid.y`` directly.
    // The raw form triggers the same slangc 2026.5+ wave64 scalarisation
    // bug on RDNA1 that affected ``mm_tiled.slang`` pre-M22.13-followup
    // and ``bmm_tiled.slang`` pre-M22.13-extras: with 3+ StructuredBuffer
    // bindings (args.input, args.weight, args.out, optional args.bias)
    // and both ``lid.x`` and ``lid.y`` reads used to address different
    // buffers, slangc folds ``lid.x`` to a constant for lanes with
    // ``lid.x >= 1``, collapsing rows of the C-tile.
    //
    // Linearisation rule: workgroup is ``numthreads(THREADS_W,
    // THREADS_H, 1)``, so SPIR-V lane ordering is
    // ``lane = lid.y * THREADS_W + lid.x``. Reconstructing the per-thread
    // coords via modular arithmetic is algebraically identical to direct
    // ``lid`` reads, but slangc's wave64 scalarisation pass treats the
    // recovered values as per-lane (correct) rather than wave-uniform
    // (the buggy fold). Same pattern as ``mm_tiled.slang`` and
    // ``templates/slang_mm.slang``.
    uint tid_local = lid.y * THREADS_W + lid.x;
    uint local_ow = tid_local % THREADS_W;  // == lid.x — but recovered
    uint local_oh = tid_local / THREADS_W;  // == lid.y — but recovered

    // CG.M15: Early exit for threads outside the runtime workgroup
    // (spec-constant THREADS_W/THREADS_H may be larger than needed).
    if (local_ow >= pc.tile_w || local_oh >= pc.tile_h) return;

    uint ow = ow_start + local_ow;
    uint oh = oh_start + local_oh;

    // ── Precompute input base for this batch ───────────────────────────
    uint in_batch_base = n * pc.stride_in_n;

    // ── Iterate over output channels assigned to this workgroup ────────
    for (uint co = co_start; co < co_end; co++) {
        if (ow >= ow_end || oh >= oh_end) {
            // This thread is outside the tile boundary but still needs to
            // participate in the co loop for uniform control flow with
            // groupshared loads (if added later). For now just skip store.
        }

        float sum = 0.0f;

        // Weight base for this output channel
        uint w_co_base = co * pc.stride_w_out;

        // Iterate over input channels and kernel
        for (uint ci = 0; ci < pc.C_in; ci++) {
            uint in_c_base = in_batch_base + ci * pc.stride_in_c;
            uint w_ci_base = w_co_base + ci * pc.stride_w_in;

            for (uint kh = 0; kh < pc.kH; kh++) {
                int ih = int(oh * pc.sH + kh * pc.dH) - int(pc.pH);
                if (ih < 0 || uint(ih) >= pc.iH) continue;

                uint in_h_base = in_c_base + uint(ih) * pc.stride_in_h;
                uint w_kh_base = w_ci_base + kh * pc.stride_w_kh;

                for (uint kw = 0; kw < pc.kW; kw++) {
                    int iw = int(ow * pc.sW + kw * pc.dW) - int(pc.pW);
                    if (iw < 0 || uint(iw) >= pc.iW) continue;

                    float in_val = args.input[in_h_base + uint(iw) * pc.stride_in_w];
                    float w_val = args.weight[w_kh_base + kw * pc.stride_w_kw];
                    sum += in_val * w_val;
                }
            }
        }

        // Write output
        if (ow < ow_end && oh < oh_end) {
            uint out_idx = n * pc.stride_out_n
                         + co * pc.stride_out_c
                         + oh * pc.stride_out_h
                         + ow * pc.stride_out_w;
/*{%*/ if has_bias /*%}*/
            sum += args.bias[co * pc.stride_bias];
/*{%*/ endif /*%}*/
/*{%*/ if epilogue /*%}*/
            sum = Epilogue::apply(sum);
/*{%*/ endif /*%}*/
            args.out[out_idx] = sum;
        }
    }
}
