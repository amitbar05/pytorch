// RNN cell template: fused LSTM/GRU/RNN-tanh/RNN-relu forward kernel.
//
// CP.3 / OP.6: Replaces the CPU-roundtrip in lowerings/rnn.py with a
// Vulkan-native compute shader.  One dispatch per (time_step, layer,
// direction) tuple; the workgroup cooperatively computes all hidden
// units for one batch element.  Inner reduction uses WaveActiveSum
// for the dot-product accumulate step.
//
// B2: cell_type (lstm/gru/rnn_tanh/rnn_relu) was a Jinja structural
// branch (an "if cell_type == lstm" chain).  Replaced with a Slang
// `interface IRnnCell` + generic `<Cell : IRnnCell>` entry point,
// matching the `IOptimizer` / `computeMain<AdamWImpl>` pattern in
// foreach_optimizer.slang.  `GATE_SIZE` / `HAS_CELL_STATE` are
// `static const` interface requirements (see
// static-const-in-interface.slang in the Slang test suite) — each
// concrete cell struct supplies its own value, and the entry point
// reads them via `Cell.GATE_SIZE`.  The concrete cell struct is
// selected by the slangc entry-point name, e.g.
// ``entry="computeMain<LstmCellImpl>"``.
//
// Cell types:
//   LSTM (gate_size=4):  i, f, g, o gates  → c_t, h_t
//   GRU  (gate_size=3):  r, z, n gates     → h_t
//   RNN  (gate_size=1):  single gate        → h_t
//
// Weight layout (PyTorch convention, contiguous gate blocks):
//   w_ih: [gate_size * hidden_size, input_size]
//     LSTM: rows 0..hs-1=i, hs..2hs-1=f, 2hs..3hs-1=g, 3hs..4hs-1=o
//     GRU:  rows 0..hs-1=r, hs..2hs-1=z, 2hs..3hs-1=n
//     RNN:  rows 0..hs-1 = single gate
//   w_hh: [gate_size * hidden_size, hidden_size] — same gate ordering
//   b_ih: [gate_size * hidden_size]
//   b_hh: [gate_size * hidden_size]
//
// Dispatch grid:  (batch_size, 1, 1)
// Workgroup:      (hidden_size, 1, 1)
//
// Each thread (lid.x = hidden_unit) computes the cell outputs for its
// hidden unit: it runs a serial dot product over input_size for the
// input→hidden transform and over hidden_size for the hidden→hidden
// transform, then applies the cell update equations.
//
// For batched execution: the grid's gid.x dimension enumerates batch
// elements.  Each workgroup processes one batch element independently.

import pointwise;

// ── Cell interface ──────────────────────────────────────────────────────
// B2: Anti-goal #5 — Slang interface generics, not string-based Jinja
// branches, select the cell type.  GATE_SIZE / HAS_CELL_STATE are
// compile-time interface requirements satisfied per concrete struct.
interface IRnnCell {
    static const int GATE_SIZE;
    static const bool HAS_CELL_STATE;

    // Computes the new hidden state (and cell state, for LSTM) from the
    // raw ih_gate/hh_gate sums.  c_prev is unused for non-cell-state
    // types (pass 0.0f); c_new is only meaningful when HAS_CELL_STATE.
    static void update(
        float ih_gate[/*{{*/ gate_size_max /*}}4*/],
        float hh_gate[/*{{*/ gate_size_max /*}}4*/],
        float c_prev,
        out float h_new,
        out float c_new
    );
}

// ── Concrete cell implementations ─────────────────────────────────────
// All four are always present; the entry point selects one via
// ``computeMain<LstmCellImpl>`` etc.  The interface array size is fixed
// at the host's GATE_SIZE_MAX (4, for LSTM); cells with smaller
// gate_size simply ignore the unused trailing slots.

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
        float r = sigmoid_fwd(ih_gate[0] + hh_gate[0]);
        float z = sigmoid_fwd(ih_gate[1] + hh_gate[1]);
        float n = tanh_fwd(ih_gate[2] + r * hh_gate[2]);
        // h_prev arrives via c_prev's slot is NOT used here — GRU's
        // "previous hidden" term is folded into hh_gate by the caller
        // through the ordinary dot-product (hh_gate already contains
        // h_prev · W_hh); the (1-z)*n + z*h_prev mix needs h_prev
        // directly, so the entry point passes h_prev via c_prev for
        // non-cell-state cells (see the entry point comment below).
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

// ── Push constants / KernelArgs ────────────────────────────────────────
// B2: Always the full LSTM-shaped layout (stride_c always present) —
// no Jinja struct-field conditionals.  Non-cell-state cells leave
// stride_c / c_prev / c_t unused; the Python caller always provides a
// real or dummy buffer.
struct PC {
    uint hidden_size;
    uint input_size;
    uint stride_w_ih;   // = input_size (row-major)
    uint stride_w_hh;   // = hidden_size (row-major)
    uint stride_x;      // batch stride for x_t = input_size
    uint stride_h;      // batch stride for h = hidden_size
    uint stride_c;      // batch stride for c = hidden_size (unused if !HAS_CELL_STATE)
    uint has_bias;      // boolean: 0=no bias, 1=has bias
};

[[vk::push_constant]] PC pc;

struct KernelArgs {
    StructuredBuffer</*{{*/ dtype /*}}float*/> x_t;       // [batch, input_size]
    StructuredBuffer</*{{*/ dtype /*}}float*/> h_prev;    // [batch, hidden_size]
    StructuredBuffer</*{{*/ dtype /*}}float*/> c_prev;    // [batch, hidden_size] (LSTM) or dummy
    StructuredBuffer</*{{*/ dtype /*}}float*/> w_ih;      // [gate_size*hidden_size, input_size]
    StructuredBuffer</*{{*/ dtype /*}}float*/> w_hh;      // [gate_size*hidden_size, hidden_size]
    StructuredBuffer</*{{*/ dtype /*}}float*/> b_ih;      // [gate_size*hidden_size]
    StructuredBuffer</*{{*/ dtype /*}}float*/> b_hh;      // [gate_size*hidden_size]
    RWStructuredBuffer</*{{*/ dtype /*}}float*/> h_t;     // [batch, hidden_size]
    RWStructuredBuffer</*{{*/ dtype /*}}float*/> c_t;     // [batch, hidden_size] (LSTM) or dummy
};
// M21.3.01: explicit Set 0 binding (slangc 2026.7.1 defaults to Set 1)
[[vk::binding(0, 0)]] ParameterBlock<KernelArgs> args;

// ── Dot-product helpers (file-scope, called from computeMain) ──────────

[ForceInline]
float _dot_ih(
    StructuredBuffer</*{{*/ dtype /*}}float*/> x_t, uint x_off,
    StructuredBuffer</*{{*/ dtype /*}}float*/> w_ih, uint stride_w_ih,
    uint gate_row, uint input_size)
{
    float acc = 0.0f;
    for (uint k = 0u; k < input_size; k++) {
        acc += (float)x_t[x_off + k]
             * (float)w_ih[gate_row * stride_w_ih + k];
    }
    return acc;
}

[ForceInline]
float _dot_hh(
    StructuredBuffer</*{{*/ dtype /*}}float*/> h_prev, uint h_off,
    StructuredBuffer</*{{*/ dtype /*}}float*/> w_hh, uint stride_w_hh,
    uint gate_row, uint hidden_size)
{
    float acc = 0.0f;
    for (uint k = 0u; k < hidden_size; k++) {
        acc += (float)h_prev[h_off + k]
             * (float)w_hh[gate_row * stride_w_hh + k];
    }
    return acc;
}

// ── Entry point ───────────────────────────────────────────────────────
// B2: Generic over <Cell : IRnnCell>.  The cell type is selected by the
// slangc entry-point name, e.g. ``entry="computeMain<LstmCellImpl>"``
// — matching the conv ``computeMain<OpReLU>`` / optimizer
// ``computeMain<AdamWImpl>`` pattern.  ``Cell.GATE_SIZE`` and
// ``Cell.HAS_CELL_STATE`` are compile-time constants, so the gate loop
// and cell-state load/store are monomorphized per cell type with zero
// runtime branching overhead.
//
// Note: for GRU (no real cell state), the entry point passes h_prev
// through the ``c_prev`` argument of ``Cell.update`` so GruCellImpl can
// compute ``(1-z)*n + z*h_prev`` without a separate interface method —
// the "cell state" slot is simply repurposed as "extra previous-state
// scalar" for non-LSTM cells.

[shader("compute")]
[numthreads(/*{{*/ hidden_size /*}}256*/, 1, 1)]
void computeMain<Cell : IRnnCell>(uint3 gtid : SV_DispatchThreadID,
                 uint3 lid : SV_GroupThreadID,
                 uint3 gid : SV_GroupID) {
    uint batch = gid.x;
    uint hid = lid.x;  // which hidden unit this thread owns

    if (hid >= pc.hidden_size) {
        return;
    }

    // Batch offsets into the input/hidden tensors.
    uint x_off = batch * pc.stride_x;
    uint h_off = batch * pc.stride_h;
    uint c_off = batch * pc.stride_c;

    // ── Compute input-part + hidden-part + bias for each gate ──────
    // Each gate = sum_k(x[k] * W_ih[gate_row, k]) + bias_ih[gate_row]
    //           + sum_k(h[k] * W_hh[gate_row, k]) + bias_hh[gate_row]
    //
    // We accumulate in float (even for half inputs) for numerical
    // stability across the dot-product reduction.  GATE_SIZE_MAX (4)
    // is the fixed array bound; Cell.GATE_SIZE (≤ 4) is the number of
    // slots actually used.
    float ih_gate[/*{{*/ gate_size_max /*}}4*/];
    float hh_gate[/*{{*/ gate_size_max /*}}4*/];
    for (uint g = 0u; g < (uint)Cell.GATE_SIZE; g++) {
        uint gate_row = g * pc.hidden_size + hid;
        ih_gate[g] = _dot_ih(args.x_t, x_off, args.w_ih, pc.stride_w_ih,
                             gate_row, pc.input_size);
        hh_gate[g] = _dot_hh(args.h_prev, h_off, args.w_hh, pc.stride_w_hh,
                              gate_row, pc.hidden_size);
        if (pc.has_bias != 0u) {
            ih_gate[g] += (float)args.b_ih[gate_row];
            hh_gate[g] += (float)args.b_hh[gate_row];
        }
    }

    // ── Cell update via the compile-time-selected Cell::update() ───
    // For HAS_CELL_STATE cells (LSTM), c_prev is the real saved cell
    // state.  For others, c_prev carries h_prev[hid] (see the GRU
    // comment above the entry point) — RNN-tanh/relu ignore it.
    float c_prev_val;
    if (Cell.HAS_CELL_STATE) {
        c_prev_val = (float)args.c_prev[c_off + hid];
    } else {
        c_prev_val = (float)args.h_prev[h_off + hid];
    }

    float h_new, c_new;
    Cell.update(ih_gate, hh_gate, c_prev_val, h_new, c_new);

    args.h_t[h_off + hid] = (/*{{*/ dtype /*}}float*/)h_new;
    if (Cell.HAS_CELL_STATE) {
        args.c_t[c_off + hid] = (/*{{*/ dtype /*}}float*/)c_new;
    }
}
