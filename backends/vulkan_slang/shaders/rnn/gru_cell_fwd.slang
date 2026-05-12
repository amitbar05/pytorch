// Fused GRU cell forward. Computes for each (b, h):
//   gi = IG + (bias_ih if has_bias); split as gi_r, gi_z, gi_n
//   gh = HG + (bias_hh if has_bias); split as gh_r, gh_z, gh_n (== hn)
//   r = sigmoid(gi_r + gh_r)
//   z = sigmoid(gi_z + gh_z)
//   n = tanh(gi_n + r * gh_n)
//   hy = n + z * (hx - n)
//   workspace[b, 0*H+h] = r, 1*H+h = z, 2*H+h = n, 3*H+h = gh_n (hn), 4*H+h = hx

struct Params { uint B; uint H; uint has_bias; };

[[vk::binding(0)]] StructuredBuffer<float> input_gates;   // [B, 3H]
[[vk::binding(1)]] StructuredBuffer<float> hidden_gates;  // [B, 3H]
[[vk::binding(2)]] StructuredBuffer<float> hx;            // [B, H]
[[vk::binding(3)]] StructuredBuffer<float> bias_ih;       // [3H]
[[vk::binding(4)]] StructuredBuffer<float> bias_hh;       // [3H]
[[vk::binding(5)]] RWStructuredBuffer<float> hy;          // [B, H]
[[vk::binding(6)]] RWStructuredBuffer<float> workspace;   // [B, 5H]
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

    uint base_g = b * 3 * params.H + h;
    float gi_r = input_gates [base_g + 0 * params.H];
    float gi_z = input_gates [base_g + 1 * params.H];
    float gi_n = input_gates [base_g + 2 * params.H];
    float gh_r = hidden_gates[base_g + 0 * params.H];
    float gh_z = hidden_gates[base_g + 1 * params.H];
    float gh_n = hidden_gates[base_g + 2 * params.H];
    if (params.has_bias != 0) {
        gi_r += bias_ih[0 * params.H + h]; gh_r += bias_hh[0 * params.H + h];
        gi_z += bias_ih[1 * params.H + h]; gh_z += bias_hh[1 * params.H + h];
        gi_n += bias_ih[2 * params.H + h]; gh_n += bias_hh[2 * params.H + h];
    }
    float r = sigmoidf(gi_r + gh_r);
    float z = sigmoidf(gi_z + gh_z);
    float n = tanh(gi_n + r * gh_n);
    float hxv = hx[idx];
    float hy_v = n + z * (hxv - n);

    hy[idx] = hy_v;
    uint base_w = b * 5 * params.H + h;
    workspace[base_w + 0 * params.H] = r;
    workspace[base_w + 1 * params.H] = z;
    workspace[base_w + 2 * params.H] = n;
    workspace[base_w + 3 * params.H] = gh_n;
    workspace[base_w + 4 * params.H] = hxv;
}
