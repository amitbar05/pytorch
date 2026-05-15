// OP.25 — RNN cell backward template (LSTM / GRU / RNN-tanh / RNN-relu).
// Valid Slang with Jinja2 markers in comments — used by slangc validation.

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
    RWStructuredBuffer</*{{*/ dtype /*}}float*/> grad_w_ih;
    RWStructuredBuffer</*{{*/ dtype /*}}float*/> grad_w_hh;
    RWStructuredBuffer</*{{*/ dtype /*}}float*/> grad_b_ih;
    RWStructuredBuffer</*{{*/ dtype /*}}float*/> grad_b_hh;
};
ParameterBlock<KernelArgs> args;

groupshared float scratch_in[/*{{*/ hidden_size /*}}256*/];

[ForceInline]
float _sigmoid(float v) {
    return 0.5f + 0.5f * tanh(v * 0.5f);
}

[ForceInline]
float _dsigmoid_from_val(float sigmoid_val) {
    return sigmoid_val * (1.0f - sigmoid_val);
}

[ForceInline]
float _dtanh_from_val(float tanh_val) {
    return 1.0f - tanh_val * tanh_val;
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

/*{%*/ if cell_type == "lstm" /*%}*/
    float i_val = _sigmoid(ih_gate[0] + hh_gate[0]);
    float f_val = _sigmoid(ih_gate[1] + hh_gate[1]);
    float g_val = tanh(ih_gate[2] + hh_gate[2]);
    float o_val = _sigmoid(ih_gate[3] + hh_gate[3]);
    float c_prev_val = (float)args.c_prev[c_off + hid];
    float c_val = f_val * c_prev_val + i_val * g_val;
    float tanh_c = tanh(c_val);

    float gh = (float)args.grad_h[h_off + hid];
    float gc = 0.0f;
    gc += (float)args.grad_c[h_off + hid];
    float dc = gh * o_val * _dtanh_from_val(tanh_c) + gc;
    float di = dc * g_val * _dsigmoid_from_val(i_val);
    float df = dc * c_prev_val * _dsigmoid_from_val(f_val);
    float dg = dc * i_val * _dtanh_from_val(g_val);
    float dout_do = gh * tanh_c * _dsigmoid_from_val(o_val);

    float gate_grad[4] = { di, df, dg, dout_do };
    float grad_c_prev_val = dc * f_val;
    args.grad_c_prev[h_off + hid] = (/*{{*/ dtype /*}}float*/)grad_c_prev_val;
/*{%*/ elif cell_type == "gru" /*%}*/
    float r_val = _sigmoid(ih_gate[0] + hh_gate[0]);
    float z_val = _sigmoid(ih_gate[1] + hh_gate[1]);
    float n_raw = ih_gate[2] + r_val * hh_gate[2];
    float n_val = tanh(n_raw);
    float h_prev_val = (float)args.h_prev[h_off + hid];

    float gh = (float)args.grad_h[h_off + hid];
    float dn = gh * (1.0f - z_val);
    float dz = gh * (h_prev_val - n_val);
    float dh_prev_direct = gh * z_val;

    float dn_raw = dn * _dtanh_from_val(n_val);
    float dr = dn_raw * hh_gate[2];
    dr = dr * _dsigmoid_from_val(r_val);
    dz = dz * _dsigmoid_from_val(z_val);

    float gate_grad[3] = { dr, dz, dn_raw };
/*{%*/ elif cell_type == "rnn_tanh" /*%}*/
    float raw = ih_gate[0] + hh_gate[0];
    float h_val = tanh(raw);
    float gh = (float)args.grad_h[h_off + hid];
    float d_raw = gh * _dtanh_from_val(h_val);
    float gate_grad[1] = { d_raw };
/*{%*/ elif cell_type == "rnn_relu" /*%}*/
    float raw = ih_gate[0] + hh_gate[0];
    float gh = (float)args.grad_h[h_off + hid];
    float d_raw = (raw > 0.0f) ? gh : 0.0f;
    float gate_grad[1] = { d_raw };
/*{%*/ endif /*%}*/

    for (uint g = 0u; g < /*{{*/ gate_size /*}}4*/u; g++) {
        uint gate_row = g * pc.hidden_size + hid;
        float gg = gate_grad[g];
        for (uint k = 0u; k < pc.input_size; k++) {
            uint idx = gate_row * pc.stride_w_ih + k;
            float val = (float)args.grad_w_ih[idx];
            args.grad_w_ih[idx] = (/*{{*/ dtype /*}}float*/)(val + gg * (float)args.x_t[x_off + k]);
        }
        for (uint k = 0u; k < pc.hidden_size; k++) {
            uint idx = gate_row * pc.stride_w_hh + k;
            float val = (float)args.grad_w_hh[idx];
            args.grad_w_hh[idx] = (/*{{*/ dtype /*}}float*/)(val + gg * (float)args.h_prev[h_off + k]);
        }
        if (pc.has_bias != 0u) {
            args.grad_b_ih[gate_row] = (/*{{*/ dtype /*}}float*/)((float)args.grad_b_ih[gate_row] + gg);
            args.grad_b_hh[gate_row] = (/*{{*/ dtype /*}}float*/)((float)args.grad_b_hh[gate_row] + gg);
        }
    }

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

    for (uint k = 0u; k < pc.hidden_size; k++) {
        float contrib = 0.0f;
        for (uint g = 0u; g < /*{{*/ gate_size /*}}4*/u; g++) {
            uint gate_row = g * pc.hidden_size + hid;
            contrib += gate_grad[g] * (float)args.w_hh[gate_row * pc.stride_w_hh + k];
        }
/*{%*/ if cell_type == "gru" /*%}*/
        if (hid == k) {
            contrib += dh_prev_direct;
        }
/*{%*/ endif /*%}*/
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
