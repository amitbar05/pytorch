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

{% if cell_type == "lstm" %}
{# LSTM constants #}
{% set gate_size = 4 %}
{% set has_cell_state = true %}
{% elif cell_type == "gru" %}
{# GRU constants #}
{% set gate_size = 3 %}
{% set has_cell_state = false %}
{% elif cell_type == "rnn_tanh" %}
{# RNN-tanh constants #}
{% set gate_size = 1 %}
{% set has_cell_state = false %}
{% elif cell_type == "rnn_relu" %}
{# RNN-relu constants #}
{% set gate_size = 1 %}
{% set has_cell_state = false %}
{% endif %}

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
};

[[vk::push_constant]] PC pc;

// M-SF.3: THREADS (numthreads X) is a specialization constant (default 64).
// Actual value is provided at dispatch time by the caller.
[[vk::constant_id(40)]] const uint THREADS = 64u;
// Groupshared memory size is a compile-time constant (Vulkan requirement).
// We use a single fixed upper bound; c_shared is unused for non-LSTM but
// the memory cost is small (hidden_size * 4 bytes, typically <= 4KB).
[[vk::constant_id(41)]] const uint HIDDEN_SIZE = 1024u;

struct KernelArgs {
    StructuredBuffer<{{ dtype }}> x_seq;      // [seq_len, batch, input_size]
    StructuredBuffer<{{ dtype }}> h0;          // [batch, hidden_size]
{% if has_cell_state %}
    StructuredBuffer<{{ dtype }}> c0;          // [batch, hidden_size]
{% else %}
    StructuredBuffer<{{ dtype }}> _c0_dummy;   // unused placeholder (groupshared budget)
{% endif %}
    StructuredBuffer<{{ dtype }}> w_ih;        // [gate_size*hidden_size, input_size]
    StructuredBuffer<{{ dtype }}> w_hh;        // [gate_size*hidden_size, hidden_size]
    StructuredBuffer<{{ dtype }}> b_ih;        // [gate_size*hidden_size]
    StructuredBuffer<{{ dtype }}> b_hh;        // [gate_size*hidden_size]
    RWStructuredBuffer<{{ dtype }}> out_seq;   // [batch, seq_len, hidden_size]
    RWStructuredBuffer<{{ dtype }}> h_last;    // [batch, hidden_size]
{% if has_cell_state %}
    RWStructuredBuffer<{{ dtype }}> c_last;    // [batch, hidden_size]
{% else %}
    RWStructuredBuffer<{{ dtype }}> _c_last_dummy; // unused placeholder
{% endif %}
};
// M21.3.01: explicit Set 0 binding (slangc 2026.7.1 defaults to Set 1)
[[vk::binding(0, 0)]] ParameterBlock<KernelArgs> args;

// Groupshared memory for communicating hidden/cell state between time steps.
// Each thread writes its own element; all threads read all elements in the
// hh dot-product loop. Barrier after each time step ensures consistency.
// M-SF.3: size is the specialization constant HIDDEN_SIZE (Vulkan requires
// compile-time constant array bound).  pc.hidden_size must be <= HIDDEN_SIZE.
groupshared float h_shared[HIDDEN_SIZE];
{% if has_cell_state %}
groupshared float c_shared[HIDDEN_SIZE];
{% endif %}

// Activation helpers

[ForceInline]
float _sigmoid(float v) {
    return 0.5f + 0.5f * tanh(v * 0.5f);
}

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void computeMain(uint3 gtid : SV_DispatchThreadID,
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
{% if has_cell_state %}
    c_shared[hid] = (float)args.c0[h_batch_off + hid];
{% endif %}
    GroupMemoryBarrierWithGroupSync();

    uint out_batch_off = batch * pc.stride_out_batch;

    // Time-step loop
    for (uint t = 0u; t < pc.seq_len; t++) {
        float ih_gate[{{ gate_size }}];
        float hh_gate[{{ gate_size }}];

        uint x_t_off = t * pc.stride_x_tbatch + batch * pc.stride_x_batch;

        for (uint g = 0u; g < {{ gate_size }}u; g++) {
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

{% if cell_type == "lstm" %}
        // LSTM cell update
        float i = _sigmoid(ih_gate[0] + hh_gate[0]);
        float f = _sigmoid(ih_gate[1] + hh_gate[1]);
        float gate = tanh(ih_gate[2] + hh_gate[2]);
        float o = _sigmoid(ih_gate[3] + hh_gate[3]);
        float c_new = f * c_shared[hid] + i * gate;
        float h_new = o * tanh(c_new);
{% elif cell_type == "gru" %}
        // GRU cell update
        float r = _sigmoid(ih_gate[0] + hh_gate[0]);
        float z = _sigmoid(ih_gate[1] + hh_gate[1]);
        float n = tanh(ih_gate[2] + r * hh_gate[2]);
        float h_new = (1.0f - z) * n + z * h_shared[hid];
{% elif cell_type == "rnn_tanh" %}
        // RNN-tanh cell update
        float h_new = tanh(ih_gate[0] + hh_gate[0]);
{% elif cell_type == "rnn_relu" %}
        // RNN-relu cell update
        float val = ih_gate[0] + hh_gate[0];
        float h_new = (val > 0.0f) ? val : 0.0f;
{% endif %}

        args.out_seq[out_batch_off + t * pc.stride_out_tbatch + hid] = ({{ dtype }})h_new;

        h_shared[hid] = h_new;
{% if has_cell_state %}
        c_shared[hid] = c_new;
{% endif %}

        GroupMemoryBarrierWithGroupSync();
    }

    args.h_last[h_batch_off + hid] = (float)h_shared[hid];
{% if has_cell_state %}
    args.c_last[h_batch_off + hid] = (float)c_shared[hid];
{% endif %}
}
