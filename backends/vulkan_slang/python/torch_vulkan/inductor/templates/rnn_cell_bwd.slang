// M20.1 / B2 — RNN cell backward template (LSTM / GRU / RNN-tanh / RNN-relu).
//
// B2: cell_type Jinja branches replaced with Slang `interface IRnnCellBwd`.
// Each cell struct provides its own `bwd_update()` implementing the
// gate-gradient chain rule.  `GATE_SIZE` / `HAS_CELL_STATE` are
// compile-time interface constants (the same `IRnnCell` interface used
// by the forward template).  The cell type is selected by the slangc
// entry-point name, e.g. ``entry="computeMain<LstmCellImpl>"``.
//
// All four cell structs are always emitted; the entry point picks one
// via `<Cell : IRnnCell>` generics — matching the forward
// `rnn_cell.slang` pattern and the `foreach_optimizer.slang` pattern.
//
// Activation backward routes through lib/pointwise.slang primitives
// via bwd_diff(sigmoid_fwd) / bwd_diff(tanh_fwd) / bwd_diff(relu_fwd).

import pointwise;
import atomics;

// ── Cell-type specific constants (via interface, B2) ──────────────────
// Re-use the same IRnnCell interface + concrete cell structs as the
// forward template.  No Jinja `[% if cell_type %]` branches anywhere.

interface IRnnCell {
    static const int GATE_SIZE;
    static const bool HAS_CELL_STATE;

    // Forward update — see rnn_cell.slang for the contract.
    static void update(
        float ih_gate[4],
        float hh_gate[4],
        float c_prev,
        out float h_new,
        out float c_new
    );

    // B2: Backward gate-gradient computation.
    //   ih_gate/hh_gate: saved raw pre-activation gate sums (recomputed).
    //   c_prev: saved cell state (LSTM) or h_prev[h] (GRU/RNN).
    //   grad_h: incoming gradient w.r.t. hidden output.
    //   grad_c: incoming gradient w.r.t. cell state (0.0 for non-LSTM).
    //   gate_grad_ih[g]: gradient w.r.t. ih_gate[g]
    //   gate_grad_hh[g]: gradient w.r.t. hh_gate[g]
    //     (may differ from gate_grad_ih[g] for GRU's r-scaled n-gate)
    //   grad_c_prev: gradient w.r.t. c_prev.
    //   grad_h_prev_direct: direct gradient w.r.t. h_prev (outside the
    //     W_hh dot-product).  For GRU this is gh * z; for others 0.0.
    static void bwd_update(
        float ih_gate[4],
        float hh_gate[4],
        float c_prev,
        float grad_h,
        float grad_c,
        out float gate_grad_ih[4],
        out float gate_grad_hh[4],
        out float grad_c_prev,
        out float grad_h_prev_direct
    );
}

// ── Concrete cell implementations ─────────────────────────────────────
// All four are always present; entry point selects one via
// ``computeMain<LstmCellImpl>`` etc.  Each struct provides both forward
// update() and backward bwd_update().

struct LstmCellImpl : IRnnCell {
    static const int GATE_SIZE = 4;
    static const bool HAS_CELL_STATE = true;

    static void update(
        float ih_gate[4], float hh_gate[4], float c_prev,
        out float h_new, out float c_new
    ) {
        float i = sigmoid_fwd(ih_gate[0] + hh_gate[0]);
        float f = sigmoid_fwd(ih_gate[1] + hh_gate[1]);
        float g = tanh_fwd(ih_gate[2] + hh_gate[2]);
        float o = sigmoid_fwd(ih_gate[3] + hh_gate[3]);
        c_new = f * c_prev + i * g;
        h_new = o * tanh_fwd(c_new);
    }

    static void bwd_update(
        float ih_gate[4], float hh_gate[4],
        float c_prev, float grad_h, float grad_c,
        out float gate_grad_ih[4], out float gate_grad_hh[4],
        out float grad_c_prev, out float grad_h_prev_direct
    ) {
        float i_raw = ih_gate[0] + hh_gate[0];
        float f_raw = ih_gate[1] + hh_gate[1];
        float g_raw = ih_gate[2] + hh_gate[2];
        float o_raw = ih_gate[3] + hh_gate[3];
        float i_val = sigmoid_fwd(i_raw);
        float f_val = sigmoid_fwd(f_raw);
        float g_val = tanh_fwd(g_raw);
        float o_val = sigmoid_fwd(o_raw);
        float c_val = f_val * c_prev + i_val * g_val;
        float tanh_c = tanh_fwd(c_val);

        float gh = grad_h;
        float gc = grad_c;
        // dc' = (gh * o) routed through bwd_diff(tanh_fwd) + gc
        float dc = _tanh_grad(c_val, gh * o_val) + gc;
        float di = _sigmoid_grad(i_raw, dc * g_val);
        float df = _sigmoid_grad(f_raw, dc * c_prev);
        float dg = _tanh_grad(g_raw, dc * i_val);
        float dout_do = _sigmoid_grad(o_raw, gh * tanh_c);

        gate_grad_ih[0] = di;  gate_grad_ih[1] = df;
        gate_grad_ih[2] = dg;  gate_grad_ih[3] = dout_do;
        gate_grad_hh = gate_grad_ih;  // LSTM: same gradient for ih and hh
        grad_c_prev = dc * f_val;
        grad_h_prev_direct = 0.0f;
    }
};

struct GruCellImpl : IRnnCell {
    static const int GATE_SIZE = 3;
    static const bool HAS_CELL_STATE = false;

    static void update(
        float ih_gate[4], float hh_gate[4], float c_prev,
        out float h_new, out float c_new
    ) {
        float r = sigmoid_fwd(ih_gate[0] + hh_gate[0]);
        float z = sigmoid_fwd(ih_gate[1] + hh_gate[1]);
        float n = tanh_fwd(ih_gate[2] + r * hh_gate[2]);
        h_new = (1.0f - z) * n + z * c_prev;
        c_new = 0.0f;
    }

    static void bwd_update(
        float ih_gate[4], float hh_gate[4],
        float c_prev, float grad_h, float grad_c,
        out float gate_grad_ih[4], out float gate_grad_hh[4],
        out float grad_c_prev, out float grad_h_prev_direct
    ) {
        float r_raw = ih_gate[0] + hh_gate[0];
        float z_raw = ih_gate[1] + hh_gate[1];
        float r_val = sigmoid_fwd(r_raw);
        float z_val = sigmoid_fwd(z_raw);
        float n_raw_input = ih_gate[2] + r_val * hh_gate[2];
        float n_val = tanh_fwd(n_raw_input);
        // c_prev carries h_prev[h] for non-cell-state cells
        float h_prev_val = c_prev;

        float gh = grad_h;
        float dn = gh * (1.0f - z_val);
        float dz = gh * (h_prev_val - n_val);
        grad_h_prev_direct = gh * z_val;

        float dn_raw = _tanh_grad(n_raw_input, dn);
        float dr = _sigmoid_grad(r_raw, dn_raw * hh_gate[2]);
        dz = _sigmoid_grad(z_raw, dz);

        gate_grad_ih[0] = dr;  gate_grad_ih[1] = dz;  gate_grad_ih[2] = dn_raw;
        gate_grad_hh[0] = dr;  gate_grad_hh[1] = dz;
        // n-gate hh path: dn_raw * r_val (r-scaled)
        gate_grad_hh[2] = dn_raw * r_val;
        grad_c_prev = 0.0f;
    }
};

struct RnnTanhCellImpl : IRnnCell {
    static const int GATE_SIZE = 1;
    static const bool HAS_CELL_STATE = false;

    static void update(
        float ih_gate[4], float hh_gate[4], float c_prev,
        out float h_new, out float c_new
    ) {
        h_new = tanh_fwd(ih_gate[0] + hh_gate[0]);
        c_new = 0.0f;
    }

    static void bwd_update(
        float ih_gate[4], float hh_gate[4],
        float c_prev, float grad_h, float grad_c,
        out float gate_grad_ih[4], out float gate_grad_hh[4],
        out float grad_c_prev, out float grad_h_prev_direct
    ) {
        float raw = ih_gate[0] + hh_gate[0];
        float d_raw = _tanh_grad(raw, grad_h);
        gate_grad_ih[0] = d_raw;
        gate_grad_hh[0] = d_raw;
        grad_c_prev = 0.0f;
        grad_h_prev_direct = 0.0f;
    }
};

struct RnnReluCellImpl : IRnnCell {
    static const int GATE_SIZE = 1;
    static const bool HAS_CELL_STATE = false;

    static void update(
        float ih_gate[4], float hh_gate[4], float c_prev,
        out float h_new, out float c_new
    ) {
        h_new = relu_fwd(ih_gate[0] + hh_gate[0]);
        c_new = 0.0f;
    }

    static void bwd_update(
        float ih_gate[4], float hh_gate[4],
        float c_prev, float grad_h, float grad_c,
        out float gate_grad_ih[4], out float gate_grad_hh[4],
        out float grad_c_prev, out float grad_h_prev_direct
    ) {
        float raw = ih_gate[0] + hh_gate[0];
        float d_raw = _relu_grad(raw, grad_h);
        gate_grad_ih[0] = d_raw;
        gate_grad_hh[0] = d_raw;
        grad_c_prev = 0.0f;
        grad_h_prev_direct = 0.0f;
    }
};

// ── Push constants / KernelArgs ────────────────────────────────────────
// B2: Always the full LSTM-shaped layout — no Jinja struct-field
// conditionals.  c_prev / grad_c / grad_c_prev / stride_c are always
// present; non-LSTM cells simply don't read/write those slots.

struct BwdPC {
    uint hidden_size;
    uint input_size;
    uint stride_w_ih;
    uint stride_w_hh;
    uint stride_x;
    uint stride_h;
    uint stride_c;       // always present (unused if !HAS_CELL_STATE)
    uint has_bias;
};

[[vk::push_constant]] BwdPC pc;

struct KernelArgs {
    StructuredBuffer</*{{*/ dtype /*}}float*/> x_t;
    StructuredBuffer</*{{*/ dtype /*}}float*/> h_prev;
    StructuredBuffer</*{{*/ dtype /*}}float*/> c_prev;       // always present
    StructuredBuffer</*{{*/ dtype /*}}float*/> w_ih;
    StructuredBuffer</*{{*/ dtype /*}}float*/> w_hh;
    StructuredBuffer</*{{*/ dtype /*}}float*/> b_ih;
    StructuredBuffer</*{{*/ dtype /*}}float*/> b_hh;
    StructuredBuffer</*{{*/ dtype /*}}float*/> grad_h;
    StructuredBuffer</*{{*/ dtype /*}}float*/> grad_c;        // always present
    RWStructuredBuffer</*{{*/ dtype /*}}float*/> grad_x;
    RWStructuredBuffer</*{{*/ dtype /*}}float*/> grad_h_prev;
    RWStructuredBuffer</*{{*/ dtype /*}}float*/> grad_c_prev;  // always present
    RWStructuredBuffer<uint> grad_w_ih;
    RWStructuredBuffer<uint> grad_w_hh;
    RWStructuredBuffer<uint> grad_b_ih;
    RWStructuredBuffer<uint> grad_b_hh;
};
// M21.3.01: explicit Set 0 binding (slangc 2026.7.1 defaults to Set 1)
[[vk::binding(0, 0)]] ParameterBlock<KernelArgs> args;

groupshared float scratch_in[/*{{*/ hidden_size /*}}256*/];

// ── M20.1: thin wrappers around lib primitive bwd_diffs ────────────────

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

// ── Entry point ───────────────────────────────────────────────────────
// B2: Generic over <Cell : IRnnCell>.  The cell type is selected by the
// slangc entry-point name, e.g. ``entry="computeMain<LstmCellImpl>"``.
// Cell.GATE_SIZE and Cell.HAS_CELL_STATE are compile-time constants.

[shader("compute")]
[numthreads(/*{{*/ hidden_size /*}}256*/, 1, 1)]
void computeMain<Cell : IRnnCell>(uint3 gtid : SV_DispatchThreadID,
                 uint3 lid : SV_GroupThreadID,
                 uint3 gid : SV_GroupID) {
    uint batch = gid.x;
    uint hid = lid.x;

    if (hid >= pc.hidden_size) {
        return;
    }

    uint x_off = batch * pc.stride_x;
    uint h_off = batch * pc.stride_h;
    uint c_off = batch * pc.stride_c;

    // ── Recompute pre-activation raw gate sums (forward mirror) ───
    float ih_gate[4];
    float hh_gate[4];
    for (uint g = 0u; g < (uint)Cell.GATE_SIZE; g++) {
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

    // ── Per-cell-type backward via interface bwd_update() ────────
    float c_prev_val;
    if (Cell.HAS_CELL_STATE) {
        c_prev_val = (float)args.c_prev[c_off + hid];
    } else {
        c_prev_val = (float)args.h_prev[h_off + hid];
    }

    float gh = (float)args.grad_h[h_off + hid];
    float gc = Cell.HAS_CELL_STATE ? (float)args.grad_c[h_off + hid] : 0.0f;

    float gate_grad_ih[4];
    float gate_grad_hh[4];
    float grad_c_prev_val;
    float dh_prev_direct;

    Cell.bwd_update(
        ih_gate, hh_gate, c_prev_val, gh, gc,
        gate_grad_ih, gate_grad_hh,
        grad_c_prev_val, dh_prev_direct
    );

    if (Cell.HAS_CELL_STATE) {
        args.grad_c_prev[h_off + hid] = (/*{{*/ dtype /*}}float*/)grad_c_prev_val;
    }

    // ── Accumulate W_ih, W_hh, b_ih, b_hh gradients ─────────────
    for (uint g = 0u; g < (uint)Cell.GATE_SIZE; g++) {
        uint gate_row = g * pc.hidden_size + hid;
        float gg_ih = gate_grad_ih[g];
        float gg_hh = gate_grad_hh[g];
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
        for (uint g = 0u; g < (uint)Cell.GATE_SIZE; g++) {
            uint gate_row = g * pc.hidden_size + hid;
            contrib += gate_grad_ih[g] * (float)args.w_ih[gate_row * pc.stride_w_ih + k];
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
        for (uint g = 0u; g < (uint)Cell.GATE_SIZE; g++) {
            uint gate_row = g * pc.hidden_size + hid;
            contrib += gate_grad_hh[g] * (float)args.w_hh[gate_row * pc.stride_w_hh + k];
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
