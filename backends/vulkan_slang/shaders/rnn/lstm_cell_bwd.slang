// Fused LSTM cell backward. Inputs: grad_hy, grad_cy, cx, cy, workspace (post-activation i,f,g,o).
// Outputs: grad_gates[B, 4H] (same as grad_input_gates AND grad_hidden_gates; caller clones),
//          grad_cx[B, H]
//
// Activation-derivative formulas:
//   total_grad_cy = grad_cy + grad_hy * o * (1 - tanh_cy²)
//   grad_o_pre = grad_hy * tanh_cy * o * (1 - o)
//   grad_f_pre = total_grad_cy * cx * f * (1 - f)
//   grad_i_pre = total_grad_cy * g * i * (1 - i)
//   grad_g_pre = total_grad_cy * i * (1 - g²)
//   grad_cx    = total_grad_cy * f
//
// When grad_hy or grad_cy is missing (undefined), binding 3 / 4 is replaced by
// a small "zero scalar buffer"; `has_ghy`/`has_gcy` flags select the path.

struct Params {
    uint B;
    uint H;
    uint has_ghy;
    uint has_gcy;
};

[[vk::binding(0)]] StructuredBuffer<float> cx;         // [B, H]
[[vk::binding(1)]] StructuredBuffer<float> cy;         // [B, H]
[[vk::binding(2)]] StructuredBuffer<float> workspace;  // [B, 4H]
[[vk::binding(3)]] StructuredBuffer<float> grad_hy;    // [B, H]
[[vk::binding(4)]] StructuredBuffer<float> grad_cy;    // [B, H]
[[vk::binding(5)]] RWStructuredBuffer<float> grad_gates;  // [B, 4H]
[[vk::binding(6)]] RWStructuredBuffer<float> grad_cx;     // [B, H]
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = params.B * params.H;
    if (idx >= total) return;
    uint b = idx / params.H;
    uint h = idx % params.H;

    uint base_g = b * 4 * params.H + h;
    float i = workspace[base_g + 0 * params.H];
    float f = workspace[base_g + 1 * params.H];
    float g = workspace[base_g + 2 * params.H];
    float o = workspace[base_g + 3 * params.H];
    float cxv = cx[idx];
    float cyv = cy[idx];
    float tanh_cy = tanh(cyv);

    float ghy = (params.has_ghy != 0) ? grad_hy[idx] : 0.0f;
    float gcy = (params.has_gcy != 0) ? grad_cy[idx] : 0.0f;

    float total_gcy = gcy + ghy * o * (1.0f - tanh_cy * tanh_cy);

    float grad_o_pre = ghy * tanh_cy * o * (1.0f - o);
    float grad_f_pre = total_gcy * cxv * f * (1.0f - f);
    float grad_i_pre = total_gcy * g   * i * (1.0f - i);
    float grad_g_pre = total_gcy * i   * (1.0f - g * g);

    grad_gates[base_g + 0 * params.H] = grad_i_pre;
    grad_gates[base_g + 1 * params.H] = grad_f_pre;
    grad_gates[base_g + 2 * params.H] = grad_g_pre;
    grad_gates[base_g + 3 * params.H] = grad_o_pre;
    grad_cx[idx] = total_gcy * f;
}
