// Fused GRU cell backward. Inputs: grad_hy, workspace (r, z, n, gh_n, hx).
// Outputs: grad_input_gates[B, 3H], grad_hidden_gates[B, 3H], grad_hx[B, H].
//
// grad_n_post = grad_hy * (1 - z)
// grad_z_post = grad_hy * (hx - n)
// grad_hx     = grad_hy * z
// grad_z_pre  = grad_z_post * z * (1 - z)
// grad_n_pre  = grad_n_post * (1 - n²)
// grad_gi_n   = grad_n_pre
// grad_gh_n   = grad_n_pre * r
// grad_r_post = grad_n_pre * gh_n
// grad_r_pre  = grad_r_post * r * (1 - r)
// grad_input_gates  = [grad_r_pre, grad_z_pre, grad_gi_n]
// grad_hidden_gates = [grad_r_pre, grad_z_pre, grad_gh_n]

struct Params { uint B; uint H; };

[[vk::binding(0)]] StructuredBuffer<float> grad_hy;      // [B, H]
[[vk::binding(1)]] StructuredBuffer<float> workspace;    // [B, 5H]
[[vk::binding(2)]] RWStructuredBuffer<float> grad_ig;    // [B, 3H]
[[vk::binding(3)]] RWStructuredBuffer<float> grad_hg;    // [B, 3H]
[[vk::binding(4)]] RWStructuredBuffer<float> grad_hx;    // [B, H]
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = params.B * params.H;
    if (idx >= total) return;
    uint b = idx / params.H;
    uint h = idx % params.H;

    uint base_w = b * 5 * params.H + h;
    float r    = workspace[base_w + 0 * params.H];
    float z    = workspace[base_w + 1 * params.H];
    float n    = workspace[base_w + 2 * params.H];
    float gh_n = workspace[base_w + 3 * params.H];
    float hxv  = workspace[base_w + 4 * params.H];

    float ghy = grad_hy[idx];
    float grad_n_post = ghy * (1.0f - z);
    float grad_z_post = ghy * (hxv - n);
    float grad_z_pre  = grad_z_post * z * (1.0f - z);
    float grad_n_pre  = grad_n_post * (1.0f - n * n);
    float grad_gi_n   = grad_n_pre;
    float grad_gh_n_v = grad_n_pre * r;
    float grad_r_post = grad_n_pre * gh_n;
    float grad_r_pre  = grad_r_post * r * (1.0f - r);

    uint base_g = b * 3 * params.H + h;
    grad_ig[base_g + 0 * params.H] = grad_r_pre;
    grad_ig[base_g + 1 * params.H] = grad_z_pre;
    grad_ig[base_g + 2 * params.H] = grad_gi_n;
    grad_hg[base_g + 0 * params.H] = grad_r_pre;
    grad_hg[base_g + 1 * params.H] = grad_z_pre;
    grad_hg[base_g + 2 * params.H] = grad_gh_n_v;
    grad_hx[idx] = ghy * z;
}
