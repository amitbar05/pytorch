module conv;

// ── CG.M6: Differentiable scalar inner for conv backward ─────────────
// Autodiff derivatives of conv_inner_madd:
//   dL/da = dL/dout * b    (gradient w.r.t. input element)
//   dL/db = dL/dout * a    (gradient w.r.t. weight element)
// `acc` is no_diff — it chains the output gradient into autodiff.
[Differentiable]
[BackwardDerivative(conv_inner_madd_bwd)]
public float conv_inner_madd(float a, float b, no_diff float acc)
{
    return acc + a * b;
}

// Hand-written fast backward for conv_inner_madd:
//   dL/da = dL/dout * b  (gradient w.r.t. input element)
//   dL/db = dL/dout * a  (gradient w.r.t. weight element)
// `acc` is no_diff — its gradient always flows through.
public void conv_inner_madd_bwd(
    inout DifferentialPair<float> dpa,
    inout DifferentialPair<float> dpb,
    no_diff float acc,
    float dout)
{
    // dL/da = dL/dout * b
    dpa = diffPair(dpa.p, dpa.d + dout * dpb.p);
    // dL/db = dL/dout * a
    dpb = diffPair(dpb.p, dpb.d + dout * dpa.p);
}

[[vk::constant_id(0)]] public const uint KH = 3;
[[vk::constant_id(1)]] public const uint KW = 3;
[[vk::constant_id(2)]] public const uint STRIDE_H = 1;
[[vk::constant_id(3)]] public const uint STRIDE_W = 1;
[[vk::constant_id(4)]] public const uint PAD_H = 0;
[[vk::constant_id(5)]] public const uint PAD_W = 0;
[[vk::constant_id(6)]] public const uint DILATION_H = 1;
[[vk::constant_id(7)]] public const uint DILATION_W = 1;

public struct ConvShape {
    public uint N;
    public uint Cin;
    public uint H;
    public uint W;
    public uint Cout;
    public uint OH;
    public uint OW;
};

public interface IConvBias {
    static float load(StructuredBuffer<float> bias, uint cout);
};

public struct BiasNone : IConvBias {
    static float load(StructuredBuffer<float> bias, uint cout) { return 0.0f; }
};

public struct BiasFromBuffer : IConvBias {
    static float load(StructuredBuffer<float> bias, uint cout) { return bias[cout]; }
};

public void conv2d_direct<Bias : IConvBias>(
    StructuredBuffer<float> X,
    StructuredBuffer<float> W,
    StructuredBuffer<float> bias,
    RWStructuredBuffer<float> Y,
    ConvShape s,
    uint n, uint cout, uint oh, uint ow
) {
    if (n >= s.N || cout >= s.Cout || oh >= s.OH || ow >= s.OW) return;

    float acc = Bias::load(bias, cout);
    for (uint cin = 0u; cin < s.Cin; cin++) {
        for (uint kh = 0u; kh < KH; kh++) {
            for (uint kw = 0u; kw < KW; kw++) {
                int ih = int(oh * STRIDE_H + kh * DILATION_H) - int(PAD_H);
                int iw = int(ow * STRIDE_W + kw * DILATION_W) - int(PAD_W);
                if (ih >= 0 && ih < int(s.H) && iw >= 0 && iw < int(s.W)) {
                    uint x_idx = ((n * s.Cin + cin) * s.H + uint(ih)) * s.W + uint(iw);
                    uint w_idx = ((cout * s.Cin + cin) * KH + kh) * KW + kw;
                    acc += X[x_idx] * W[w_idx];
                }
            }
        }
    }
    uint y_idx = ((n * s.Cout + cout) * s.OH + oh) * s.OW + ow;
    Y[y_idx] = acc;
}
