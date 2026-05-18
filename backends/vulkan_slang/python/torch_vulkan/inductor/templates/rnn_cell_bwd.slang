// M20.1 — RNN cell backward template (LSTM / GRU / RNN-tanh / RNN-relu).
//
// Pre-M20.1, this template hand-rolled the chain rule including the
// inverse derivatives of sigmoid and tanh as inline helpers (d-sigmoid
// and d-tanh of saved primal). Three known chain-rule-class bugs
// (M17.8.d.3, M18.2, M18.3) traced to copy-paste errors in this region.
//
// Post-M20.1, the activation backwards delegate to lib/pointwise.slang
// primitives via bwd_diff(sigmoid_fwd) / bwd_diff(tanh_fwd) /
// bwd_diff(relu_fwd). Those primitives carry validator-clean
// [BackwardDerivative] partners (sigmoid_fast_bwd, tanh_fast_bwd,
// relu_fast_bwd) that are unit-tested in the lib. The cell-state chain
// rule (LSTM's c' = f*c + i*g; h' = o*tanh(c')) is still composed
// manually here, but it's small (~10 LoC for LSTM, fewer for others)
// and the activation-derivative bug class is eliminated.
//
// Why not full bwd_diff() over a single Slang-Differentiable
// cell-update? The slang_validator at slang_validate/bwd_diff_scan.py
// requires every Slang-Differentiable-tagged function to have a
// paired BackwardDerivative-or-ForwardDerivative annotation;
// bwd_diff(name)(...) invocations don't satisfy that contract from
// the validator's regex point of view. Patching the validator is
// Group F (out of templates/ lane), so M20.1 ratchets to the
// per-primitive bwd_diff() form and leaves the
// autodiff-over-the-whole-cell variant for a later milestone after
// the validator is tightened to recognise bwd_diff() use sites.

import pointwise;
// M20.1.c: weight + bias grad accumulators use vk_atomic_add to dedupe
// cross-batch / cross-time-step accumulation. Mirrors slang_conv_bwd.slang's
// dX/dW/dB accumulators which have used vk_atomic_add since OP.25.
import atomics;

//# ── Cell-type specific constants ── #//
/*{%*/ if cell_type == "lstm" /*%}*/
/*{%*/ set gate_size = 4 /*%}*/
/*{%*/ set has_cell_state = true /*%}*/
/*{%*/ elif cell_type == "gru" /*%}*/
/*{%*/ set gate_size = 3 /*%}*/
/*{%*/ set has_cell_state = false /*%}*/
/*{%*/ elif cell_type in ("rnn_tanh", "rnn_relu") /*%}*/
/*{%*/ set gate_size = 1 /*%}*/
/*{%*/ set has_cell_state = false /*%}*/
/*{%*/ else /*%}*/
/*{%*/ set gate_size = 1 /*%}*/
/*{%*/ set has_cell_state = false /*%}*/
/*{%*/ endif /*%}*/

struct BwdPC {
    uint hidden_size;
    uint input_size;
    uint stride_w_ih;
    uint stride_w_hh;
    uint stride_x;
    uint stride_h;
/*{%*/ if has_cell_state /*%}*/
    uint stride_c;
/*{%*/ endif /*%}*/
    uint has_bias;
};

[[vk::push_constant]] BwdPC pc;

struct KernelArgs {
    StructuredBuffer</*{{*/ dtype /*}}float*/> x_t;
    StructuredBuffer</*{{*/ dtype /*}}float*/> h_prev;
/*{%*/ if has_cell_state /*%}*/
    StructuredBuffer</*{{*/ dtype /*}}float*/> c_prev;
/*{%*/ endif /*%}*/
    StructuredBuffer</*{{*/ dtype /*}}float*/> w_ih;
    StructuredBuffer</*{{*/ dtype /*}}float*/> w_hh;
    StructuredBuffer</*{{*/ dtype /*}}float*/> b_ih;
    StructuredBuffer</*{{*/ dtype /*}}float*/> b_hh;
    StructuredBuffer</*{{*/ dtype /*}}float*/> grad_h;
/*{%*/ if has_cell_state /*%}*/
    StructuredBuffer</*{{*/ dtype /*}}float*/> grad_c;
/*{%*/ endif /*%}*/
    RWStructuredBuffer</*{{*/ dtype /*}}float*/> grad_x;
    RWStructuredBuffer</*{{*/ dtype /*}}float*/> grad_h_prev;
/*{%*/ if has_cell_state /*%}*/
    RWStructuredBuffer</*{{*/ dtype /*}}float*/> grad_c_prev;
/*{%*/ endif /*%}*/
    // M20.1.c: bind weight + bias grad accumulators as RWStructuredBuffer<uint>
    // so the CAS-based vk_atomic_add (in lib/atomics.slang) can reinterpret
    // the float values via asuint/asfloat. The host-side caller zeros each
    // buffer before the BPTT loop (lowerings/rnn/bwd_vulkan.py uses
    // torch.zeros_like(w_ih) etc.) — atomic-add into a zero-init buffer is
    // the correct semantics for cross-batch + cross-time-step accumulation.
    RWStructuredBuffer<uint> grad_w_ih;
    RWStructuredBuffer<uint> grad_w_hh;
    RWStructuredBuffer<uint> grad_b_ih;
    RWStructuredBuffer<uint> grad_b_hh;
};
// M21.3.01: explicit Set 0 binding (slangc 2026.7.1 defaults to Set 1)
[[vk::binding(0, 0)]] ParameterBlock<KernelArgs> args;

groupshared float scratch_in[/*{{*/ hidden_size /*}}256*/];

// ── M20.1: thin wrappers around lib primitive bwd_diffs ────────────────
//
// Each helper returns the gradient w.r.t. the input given the gradient
// w.r.t. the output and the saved forward primal. They route through
// bwd_diff(<lib primitive>) so the activation-derivative formulae live
// in one place (lib/pointwise.slang) and bug fixes there propagate to
// every consumer.

[ForceInline]
float _sigmoid_grad(float x_primal, float grad_out) {
    DifferentialPair<float> dp = diffPair(x_primal, 0.0f);
    bwd_diff(sigmoid_fwd)(dp, grad_out);
    return dp.d;
}

[ForceInline]
float _tanh_grad(float x_primal, float grad_out) {
    DifferentialPair<float> dp = diffPair(x_primal, 0.0f);
    bwd_diff(tanh_fwd)(dp, grad_out);
    return dp.d;
}

[ForceInline]
float _relu_grad(float x_primal, float grad_out) {
    DifferentialPair<float> dp = diffPair(x_primal, 0.0f);
    bwd_diff(relu_fwd)(dp, grad_out);
    return dp.d;
}

[shader("compute")]
[numthreads(/*{{*/ hidden_size /*}}256*/, 1, 1)]
void computeMain(uint3 gtid : SV_DispatchThreadID,
                 uint3 lid : SV_GroupThreadID,
                 uint3 gid : SV_GroupID) {
    uint batch = gid.x;
    uint hid = lid.x;

    if (hid >= pc.hidden_size) {
        return;
    }

    uint x_off = batch * pc.stride_x;
    uint h_off = batch * pc.stride_h;
/*{%*/ if has_cell_state /*%}*/
    uint c_off = batch * pc.stride_c;
/*{%*/ endif /*%}*/

    // ── Recompute pre-activation raw gate sums (forward mirror). ─
    float ih_gate[/*{{*/ gate_size /*}}4*/];
    float hh_gate[/*{{*/ gate_size /*}}4*/];
    for (uint g = 0u; g < /*{{*/ gate_size /*}}4*/u; g++) {
        uint gate_row = g * pc.hidden_size + hid;
        float acc_ih = 0.0f;
        for (uint k = 0u; k < pc.input_size; k++) {
            acc_ih += (float)args.x_t[x_off + k]
                    * (float)args.w_ih[gate_row * pc.stride_w_ih + k];
        }
        float acc_hh = 0.0f;
        for (uint k = 0u; k < pc.hidden_size; k++) {
            acc_hh += (float)args.h_prev[h_off + k]
                    * (float)args.w_hh[gate_row * pc.stride_w_hh + k];
        }
        if (pc.has_bias != 0u) {
            acc_ih += (float)args.b_ih[gate_row];
            acc_hh += (float)args.b_hh[gate_row];
        }
        ih_gate[g] = acc_ih;
        hh_gate[g] = acc_hh;
    }

    // ── Cell-type-specific gate-gradient computation ─────────────
    // The activation derivatives use bwd_diff(<lib primitive>) — the
    // cell-state chain rule (where present) is composed manually.
/*{%*/ if cell_type == "lstm" /*%}*/
    // LSTM: c' = f*c + i*g; h' = o * tanh(c')
    //   ∂L/∂o_raw = ∂σ(o_raw) * (gh * tanh(c'))
    //   ∂L/∂c'    = gh * o * ∂tanh(c') + gc
    //   ∂L/∂i_raw = ∂σ(i_raw) * (dc' * g)
    //   ∂L/∂f_raw = ∂σ(f_raw) * (dc' * c_prev)
    //   ∂L/∂g_raw = ∂tanh(g_raw) * (dc' * i)
    //   ∂L/∂c_prev = dc' * f
    float i_raw = ih_gate[0] + hh_gate[0];
    float f_raw = ih_gate[1] + hh_gate[1];
    float g_raw = ih_gate[2] + hh_gate[2];
    float o_raw = ih_gate[3] + hh_gate[3];
    float i_val = sigmoid_fwd(i_raw);
    float f_val = sigmoid_fwd(f_raw);
    float g_val = tanh_fwd(g_raw);
    float o_val = sigmoid_fwd(o_raw);
    float c_prev_val = (float)args.c_prev[c_off + hid];
    float c_val = f_val * c_prev_val + i_val * g_val;
    float tanh_c = tanh_fwd(c_val);

    float gh = (float)args.grad_h[h_off + hid];
    float gc = (float)args.grad_c[h_off + hid];
    // dc' propagation: gh contributes through h' = o * tanh(c'), so
    // dc' includes (gh * o) routed through bwd_diff(tanh_fwd).
    float dc = _tanh_grad(c_val, gh * o_val) + gc;
    // di / df / dg / do via per-primitive bwd_diff:
    float di = _sigmoid_grad(i_raw, dc * g_val);
    float df = _sigmoid_grad(f_raw, dc * c_prev_val);
    float dg = _tanh_grad(g_raw, dc * i_val);
    float dout_do = _sigmoid_grad(o_raw, gh * tanh_c);

    float gate_grad[4] = { di, df, dg, dout_do };
    float grad_c_prev_val = dc * f_val;
    args.grad_c_prev[h_off + hid] = (/*{{*/ dtype /*}}float*/)grad_c_prev_val;
    float dh_prev_direct = 0.0f;
/*{%*/ elif cell_type == "gru" /*%}*/
    // GRU: r = σ(...), z = σ(...), n = tanh(ih_n + r * hh_n);
    //      h' = (1-z)*n + z*h_prev
    //   dn         = gh * (1 - z)
    //   dz         = gh * (h_prev - n)
    //   dh_prev (z-route) = gh * z
    //   dn_raw = bwd_diff(tanh_fwd)(ih_n + r*hh_n, dn) gives gradient
    //           w.r.t. the tanh input. Split between ih_n and r*hh_n.
    //   dr = dn_raw * hh_n; then ∂σ(r_raw); dr_sig = dz; etc.
    float r_raw = ih_gate[0] + hh_gate[0];
    float z_raw = ih_gate[1] + hh_gate[1];
    float r_val = sigmoid_fwd(r_raw);
    float z_val = sigmoid_fwd(z_raw);
    float n_raw_input = ih_gate[2] + r_val * hh_gate[2];
    float n_val = tanh_fwd(n_raw_input);
    float h_prev_val = (float)args.h_prev[h_off + hid];

    float gh = (float)args.grad_h[h_off + hid];
    float dn = gh * (1.0f - z_val);
    float dz = gh * (h_prev_val - n_val);
    float dh_prev_direct = gh * z_val;

    // bwd through tanh on the n-gate input.
    float dn_raw = _tanh_grad(n_raw_input, dn);
    float dr = dn_raw * hh_gate[2];
    dr = _sigmoid_grad(r_raw, dr);
    dz = _sigmoid_grad(z_raw, dz);

    // Gate gradients applied to ih vs hh differ for the n-gate:
    //   ih path of n: dn_raw (direct contribution)
    //   hh path of n: dn_raw * r_val (through the r-scaling)
    float gate_grad[3] = { dr, dz, dn_raw };
    float gate_grad_hh[3] = { dr, dz, dn_raw * r_val };
/*{%*/ elif cell_type == "rnn_tanh" /*%}*/
    float raw = ih_gate[0] + hh_gate[0];
    float gh = (float)args.grad_h[h_off + hid];
    float d_raw = _tanh_grad(raw, gh);
    float gate_grad[1] = { d_raw };
    float gate_grad_hh[1] = { d_raw };
    float dh_prev_direct = 0.0f;
/*{%*/ elif cell_type == "rnn_relu" /*%}*/
    float raw = ih_gate[0] + hh_gate[0];
    float gh = (float)args.grad_h[h_off + hid];
    float d_raw = _relu_grad(raw, gh);
    float gate_grad[1] = { d_raw };
    float gate_grad_hh[1] = { d_raw };
    float dh_prev_direct = 0.0f;
/*{%*/ endif /*%}*/

    // ── Accumulate W_ih, W_hh, b_ih, b_hh gradients ─────────────
    // M20.1.c: vk_atomic_add (CAS-based, RDNA1-portable — see
    // lib/atomics.slang) replaces the prior non-atomic read-modify-write.
    // The race surfaces under batch_size > 1: each batch element's
    // workgroup writes to the same (gate_row, k) cell, dropping updates.
    // Mirrors the canonical pattern from templates/slang_conv_bwd.slang.
    for (uint g = 0u; g < /*{{*/ gate_size /*}}4*/u; g++) {
        uint gate_row = g * pc.hidden_size + hid;
        float gg_ih = gate_grad[g];
/*{%*/ if cell_type in ("gru", "rnn_tanh", "rnn_relu") /*%}*/
        float gg_hh = gate_grad_hh[g];
/*{%*/ else /*%}*/
        float gg_hh = gg_ih;
/*{%*/ endif /*%}*/
        for (uint k = 0u; k < pc.input_size; k++) {
            uint idx = gate_row * pc.stride_w_ih + k;
            vk_atomic_add(args.grad_w_ih, idx, gg_ih * (float)args.x_t[x_off + k]);
        }
        for (uint k = 0u; k < pc.hidden_size; k++) {
            uint idx = gate_row * pc.stride_w_hh + k;
            vk_atomic_add(args.grad_w_hh, idx, gg_hh * (float)args.h_prev[h_off + k]);
        }
        if (pc.has_bias != 0u) {
            vk_atomic_add(args.grad_b_ih, gate_row, gg_ih);
            vk_atomic_add(args.grad_b_hh, gate_row, gg_hh);
        }
    }

    // ── Reduce to grad_x[k] ────────────────────────────────────
    for (uint k = 0u; k < pc.input_size; k++) {
        float contrib = 0.0f;
        for (uint g = 0u; g < /*{{*/ gate_size /*}}4*/u; g++) {
            uint gate_row = g * pc.hidden_size + hid;
            contrib += gate_grad[g] * (float)args.w_ih[gate_row * pc.stride_w_ih + k];
        }
        scratch_in[hid] = contrib;
        GroupMemoryBarrierWithGroupSync();

        for (uint s = pc.hidden_size / 2u; s > 0u; s >>= 1u) {
            if (hid < s) {
                scratch_in[hid] += scratch_in[hid + s];
            }
            GroupMemoryBarrierWithGroupSync();
        }

        if (hid == 0u) {
            float val = (float)args.grad_x[x_off + k];
            args.grad_x[x_off + k] = (/*{{*/ dtype /*}}float*/)(val + scratch_in[0]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // ── Reduce to grad_h_prev[k] ───────────────────────────────
    for (uint k = 0u; k < pc.hidden_size; k++) {
        float contrib = 0.0f;
        for (uint g = 0u; g < /*{{*/ gate_size /*}}4*/u; g++) {
            uint gate_row = g * pc.hidden_size + hid;
/*{%*/ if cell_type in ("gru", "rnn_tanh", "rnn_relu") /*%}*/
            contrib += gate_grad_hh[g] * (float)args.w_hh[gate_row * pc.stride_w_hh + k];
/*{%*/ else /*%}*/
            contrib += gate_grad[g] * (float)args.w_hh[gate_row * pc.stride_w_hh + k];
/*{%*/ endif /*%}*/
        }
        if (hid == k) {
            contrib += dh_prev_direct;
        }
        scratch_in[hid] = contrib;
        GroupMemoryBarrierWithGroupSync();

        for (uint s = pc.hidden_size / 2u; s > 0u; s >>= 1u) {
            if (hid < s) {
                scratch_in[hid] += scratch_in[hid + s];
            }
            GroupMemoryBarrierWithGroupSync();
        }

        if (hid == 0u) {
            float val = (float)args.grad_h_prev[h_off + k];
            args.grad_h_prev[h_off + k] = (/*{{*/ dtype /*}}float*/)(val + scratch_in[0]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
