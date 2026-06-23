// T.10-fast — Fused multi-time-step RNN cell template (.slang migration from .py.jinja)
//
// Unlike rnn_cell.slang, which dispatches one kernel per time step, this
// template processes ALL time steps for a single (layer, direction,
// batch_element) in ONE kernel dispatch.
//
// Design:
//   - One workgroup per batch element (grid = (batch_size, 1, 1))
//   - hidden_size threads per workgroup, each thread owns one hidden unit
//   - Groupshared memory holds h and c state between time steps
//   - GroupMemoryBarrierWithGroupSync between time steps
//
// Constraints:
//   - hidden_size <= 1024 (groupshared memory limit)
//   - Same weight layout as rnn_cell.slang
//
// M-SF.3: THREADS is a specialization constant (default 64).
// HIDDEN_SIZE is a compile-time constant for groupshared array bounds.
//
// B2: cell_type (lstm/gru/rnn_tanh/rnn_relu) was a Jinja structural
// branch.  Replaced with the same `interface IRnnCell` used by
// rnn_cell.slang (LstmCellImpl / GruCellImpl / RnnTanhCellImpl /
// RnnReluCellImpl) — the cell type is now selected via the slangc
// entry-point name (``computeMain<LstmCellImpl>`` etc.), not Jinja.
// `direction` was already a runtime push-constant gate (M10
// IRnnDirection) — anti-goal #5 compliant as-is, unchanged here.

import pointwise;

// ── Cell interface (shared shape with rnn_cell.slang) ──────────────────
interface IRnnCell {
    static const int GATE_SIZE;
    static const bool HAS_CELL_STATE;

    static void update(
        float ih_gate[/*{{*/ gate_size_max /*}}4*/],
        float hh_gate[/*{{*/ gate_size_max /*}}4*/],
        float c_prev,
        out float h_new,
        out float c_new
    );
}

struct LstmCellImpl : IRnnCell {
    static const int GATE_SIZE = 4;
    static const bool HAS_CELL_STATE = true;

    static void update(
        float ih_gate[/*{{*/ gate_size_max /*}}4*/],
        float hh_gate[/*{{*/ gate_size_max /*}}4*/],
        float c_prev,
        out float h_new,
        out float c_new
    ) {
        float i = sigmoid_fwd(ih_gate[0] + hh_gate[0]);
        float f = sigmoid_fwd(ih_gate[1] + hh_gate[1]);
        float g = tanh_fwd(ih_gate[2] + hh_gate[2]);
        float o = sigmoid_fwd(ih_gate[3] + hh_gate[3]);
        c_new = f * c_prev + i * g;
        h_new = o * tanh_fwd(c_new);
    }
};

struct GruCellImpl : IRnnCell {
    static const int GATE_SIZE = 3;
    static const bool HAS_CELL_STATE = false;

    static void update(
        float ih_gate[/*{{*/ gate_size_max /*}}4*/],
        float hh_gate[/*{{*/ gate_size_max /*}}4*/],
        float c_prev,
        out float h_new,
        out float c_new
    ) {
        // c_prev carries h_prev[hid] for non-cell-state cells (see the
        // entry-point comment) — GRU needs h_prev directly for the
        // (1-z)*n + z*h_prev mix.
        float r = sigmoid_fwd(ih_gate[0] + hh_gate[0]);
        float z = sigmoid_fwd(ih_gate[1] + hh_gate[1]);
        float n = tanh_fwd(ih_gate[2] + r * hh_gate[2]);
        h_new = (1.0f - z) * n + z * c_prev;
        c_new = 0.0f;
    }
};

struct RnnTanhCellImpl : IRnnCell {
    static const int GATE_SIZE = 1;
    static const bool HAS_CELL_STATE = false;

    static void update(
        float ih_gate[/*{{*/ gate_size_max /*}}4*/],
        float hh_gate[/*{{*/ gate_size_max /*}}4*/],
        float c_prev,
        out float h_new,
        out float c_new
    ) {
        h_new = tanh_fwd(ih_gate[0] + hh_gate[0]);
        c_new = 0.0f;
    }
};

struct RnnReluCellImpl : IRnnCell {
    static const int GATE_SIZE = 1;
    static const bool HAS_CELL_STATE = false;

    static void update(
        float ih_gate[/*{{*/ gate_size_max /*}}4*/],
        float hh_gate[/*{{*/ gate_size_max /*}}4*/],
        float c_prev,
        out float h_new,
        out float c_new
    ) {
        h_new = relu_fwd(ih_gate[0] + hh_gate[0]);
        c_new = 0.0f;
    }
};

struct PC {
    uint hidden_size;
    uint input_size;
    uint seq_len;
    uint stride_w_ih;       // = input_size
    uint stride_w_hh;       // = hidden_size
    uint stride_x_tbatch;   // stride between time steps = batch * input_size
    uint stride_x_batch;    // stride within a time step for batch = input_size
    uint stride_h_batch;    // stride for batch in h/c = hidden_size
    uint stride_out_tbatch; // stride between time steps in output = hidden_size
    uint stride_out_batch;  // stride between batch elements in output = seq_len * hidden_size
    uint has_bias;          // boolean: 0=no bias, 1=has bias
    uint direction;         // M10: 0=forward, 1=reverse
};

[[vk::push_constant]] PC pc;

// M-SF.3: THREADS (numthreads X) is a specialization constant (default 64).
// Actual value is provided at dispatch time by the caller.
[[vk::constant_id(40)]] const uint THREADS = 64u;
// Groupshared memory size is a compile-time constant (Vulkan requirement).
// We use a single fixed upper bound; c_shared is allocated unconditionally
// — for non-cell-state cells it is simply unused (small, fixed LDS cost).
[[vk::constant_id(41)]] const uint HIDDEN_SIZE = 1024u;

struct KernelArgs {
    StructuredBuffer</*{{*/ dtype /*}}float*/> x_seq;      // [seq_len, batch, input_size]
    StructuredBuffer</*{{*/ dtype /*}}float*/> h0;          // [batch, hidden_size]
    StructuredBuffer</*{{*/ dtype /*}}float*/> c0;          // [batch, hidden_size] (LSTM) or dummy
    StructuredBuffer</*{{*/ dtype /*}}float*/> w_ih;        // [gate_size*hidden_size, input_size]
    StructuredBuffer</*{{*/ dtype /*}}float*/> w_hh;        // [gate_size*hidden_size, hidden_size]
    StructuredBuffer</*{{*/ dtype /*}}float*/> b_ih;        // [gate_size*hidden_size]
    StructuredBuffer</*{{*/ dtype /*}}float*/> b_hh;        // [gate_size*hidden_size]
    RWStructuredBuffer</*{{*/ dtype /*}}float*/> out_seq;   // [batch, seq_len, hidden_size]
    RWStructuredBuffer</*{{*/ dtype /*}}float*/> h_last;    // [batch, hidden_size]
    RWStructuredBuffer</*{{*/ dtype /*}}float*/> c_last;    // [batch, hidden_size] (LSTM) or dummy
};
// M21.3.01: explicit Set 0 binding (slangc 2026.7.1 defaults to Set 1)
[[vk::binding(0, 0)]] ParameterBlock<KernelArgs> args;

// Groupshared memory for communicating hidden/cell state between time steps.
// Each thread writes its own element; all threads read all elements in the
// hh dot-product loop. Barrier after each time step ensures consistency.
// M-SF.3: size is the specialization constant HIDDEN_SIZE (Vulkan requires
// compile-time constant array bound).  pc.hidden_size must be <= HIDDEN_SIZE.
groupshared float h_shared[HIDDEN_SIZE];
groupshared float c_shared[HIDDEN_SIZE];

// ── Entry point ───────────────────────────────────────────────────────
// B2: Generic over <Cell : IRnnCell>; cell type selected via the
// slangc entry-point name (``entry="computeMain<LstmCellImpl>"``).
// For non-LSTM cells, h_shared[hid] is passed as the "c_prev" argument
// to Cell::update() (see rnn_cell.slang's GRU comment) so GruCellImpl
// can read the previous hidden state without a second interface method.
[shader("compute")]
[numthreads(THREADS, 1, 1)]
void computeMain<Cell : IRnnCell>(uint3 gtid : SV_DispatchThreadID,
                 uint3 lid : SV_GroupThreadID,
                 uint3 gid : SV_GroupID) {
    uint batch = gid.x;
    uint hid = lid.x;  // which hidden unit this thread owns

    if (hid >= pc.hidden_size) {
        return;
    }

    // Load initial state into groupshared memory
    uint h_batch_off = batch * pc.stride_h_batch;
    h_shared[hid] = (float)args.h0[h_batch_off + hid];
    if (Cell.HAS_CELL_STATE) {
        c_shared[hid] = (float)args.c0[h_batch_off + hid];
    }
    GroupMemoryBarrierWithGroupSync();

    uint out_batch_off = batch * pc.stride_out_batch;

    // Time-step loop
    for (uint t = 0u; t < pc.seq_len; t++) {
        float ih_gate[/*{{*/ gate_size_max /*}}4*/];
        float hh_gate[/*{{*/ gate_size_max /*}}4*/];

        // M10: IRnnDirection — input index computed from direction push constant.
        // For reverse (pc.direction == 1u), read x_seq in reverse order without
        // requiring the caller to pre-flip the input tensor.
        uint x_t_off;
        if (pc.direction == 0u) {
            x_t_off = t * pc.stride_x_tbatch + batch * pc.stride_x_batch;
        } else {
            x_t_off = (pc.seq_len - 1u - t) * pc.stride_x_tbatch + batch * pc.stride_x_batch;
        }

        for (uint g = 0u; g < (uint)Cell.GATE_SIZE; g++) {
            uint gate_row = g * pc.hidden_size + hid;

            float acc_ih = 0.0f;
            for (uint k = 0u; k < pc.input_size; k++) {
                acc_ih += (float)args.x_seq[x_t_off + k]
                        * (float)args.w_ih[gate_row * pc.stride_w_ih + k];
            }

            float acc_hh = 0.0f;
            for (uint k = 0u; k < pc.hidden_size; k++) {
                acc_hh += h_shared[k]
                        * (float)args.w_hh[gate_row * pc.stride_w_hh + k];
            }

            ih_gate[g] = acc_ih;
            hh_gate[g] = acc_hh;

            if (pc.has_bias != 0u) {
                ih_gate[g] += (float)args.b_ih[gate_row];
                hh_gate[g] += (float)args.b_hh[gate_row];
            }
        }

        float c_prev_val = Cell.HAS_CELL_STATE ? c_shared[hid] : h_shared[hid];
        float h_new, c_new;
        Cell.update(ih_gate, hh_gate, c_prev_val, h_new, c_new);

        args.out_seq[out_batch_off + t * pc.stride_out_tbatch + hid] = (/*{{*/ dtype /*}}float*/)h_new;

        h_shared[hid] = h_new;
        if (Cell.HAS_CELL_STATE) {
            c_shared[hid] = c_new;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    args.h_last[h_batch_off + hid] = (/*{{*/ dtype /*}}float*/)h_shared[hid];
    if (Cell.HAS_CELL_STATE) {
        args.c_last[h_batch_off + hid] = (/*{{*/ dtype /*}}float*/)c_shared[hid];
    }
}
