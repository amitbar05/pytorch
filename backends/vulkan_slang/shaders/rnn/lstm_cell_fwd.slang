// Fused LSTM cell forward. Computes for each (b, h):
//   i = sigmoid(IG[b, 0*H+h] + HG[b, 0*H+h] + bi[0*H+h] + bh[0*H+h])
//   f = sigmoid(IG[b, 1*H+h] + HG[b, 1*H+h] + bi[1*H+h] + bh[1*H+h])
//   g = tanh   (IG[b, 2*H+h] + HG[b, 2*H+h] + bi[2*H+h] + bh[2*H+h])
//   o = sigmoid(IG[b, 3*H+h] + HG[b, 3*H+h] + bi[3*H+h] + bh[3*H+h])
//   cy = f * cx + i * g
//   hy = o * tanh(cy)
//   workspace[b, 0*H+h] = i, [1*H+h] = f, [2*H+h] = g, [3*H+h] = o

struct Params { uint B; uint H; uint has_bias; };

[[vk::binding(0)]] StructuredBuffer<float> input_gates;   // [B, 4H]
[[vk::binding(1)]] StructuredBuffer<float> hidden_gates;  // [B, 4H]
[[vk::binding(2)]] StructuredBuffer<float> cx;            // [B, H]
[[vk::binding(3)]] StructuredBuffer<float> bias_ih;       // [4H] (ignored if has_bias=0)
[[vk::binding(4)]] StructuredBuffer<float> bias_hh;       // [4H]
[[vk::binding(5)]] RWStructuredBuffer<float> hy;          // [B, H]
[[vk::binding(6)]] RWStructuredBuffer<float> cy;          // [B, H]
[[vk::binding(7)]] RWStructuredBuffer<float> workspace;   // [B, 4H]
[[vk::push_constant]] Params params;

float sigmoidf(float x) { return 1.0f / (1.0f + exp(-x)); }

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = params.B * params.H;
    if (idx >= total) return;
    uint b = idx / params.H;
    uint h = idx % params.H;

    uint base_g = b * 4 * params.H + h;
    float i_pre = input_gates[base_g + 0 * params.H] + hidden_gates[base_g + 0 * params.H];
    float f_pre = input_gates[base_g + 1 * params.H] + hidden_gates[base_g + 1 * params.H];
    float g_pre = input_gates[base_g + 2 * params.H] + hidden_gates[base_g + 2 * params.H];
    float o_pre = input_gates[base_g + 3 * params.H] + hidden_gates[base_g + 3 * params.H];
    if (params.has_bias != 0) {
        i_pre += bias_ih[0 * params.H + h] + bias_hh[0 * params.H + h];
        f_pre += bias_ih[1 * params.H + h] + bias_hh[1 * params.H + h];
        g_pre += bias_ih[2 * params.H + h] + bias_hh[2 * params.H + h];
        o_pre += bias_ih[3 * params.H + h] + bias_hh[3 * params.H + h];
    }
    float i = sigmoidf(i_pre);
    float f = sigmoidf(f_pre);
    float g = tanh(g_pre);
    float o = sigmoidf(o_pre);

    float cxv = cx[idx];
    float cy_v = f * cxv + i * g;
    float hy_v = o * tanh(cy_v);

    hy[idx] = hy_v;
    cy[idx] = cy_v;
    workspace[base_g + 0 * params.H] = i;
    workspace[base_g + 1 * params.H] = f;
    workspace[base_g + 2 * params.H] = g;
    workspace[base_g + 3 * params.H] = o;
}
