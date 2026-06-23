module losses;

// ── T2.11: closed-form [BackwardDerivative] for all 7 loss elementals.
// Auto-diff would re-derive these per backward call (and allocate an
// `exp`-tape for `bce_with_logits`); each closed-form below is a single
// mul/sub. Style mirrors `lib/pointwise.slang`: forward carries
// `[BackwardDerivative(name)]`, the named bwd is a separate `void` that
// mutates `inout DifferentialPair<float>` per differentiable input.
//
// Aliasing contract (cf. T3.11 on norm.slang): each bwd computes ALL
// gradient values into local `dx_*` floats BEFORE writing to any
// `inout` parameter. This keeps the order-of-writes irrelevant and
// avoids the read-then-write fragility flagged on `ln_affine_elem_bwd`.

[Differentiable]
[BackwardDerivative(mse_elem_bwd)]
public float mse_elem(float pred, float target) {
    float d = pred - target;
    return d * d;
}
void mse_elem_bwd(inout DifferentialPair<float> pred,
                  inout DifferentialPair<float> target,
                  float dout) {
    float d = pred.p - target.p;
    float dx_pred = dout * 2.0f * d;
    float dx_target = -dx_pred;
    pred = diffPair(pred.p, pred.d + dx_pred);
    target = diffPair(target.p, target.d + dx_target);
}

[Differentiable]
[BackwardDerivative(l1_elem_bwd)]
public float l1_elem(float pred, float target) {
    return abs(pred - target);
}
void l1_elem_bwd(inout DifferentialPair<float> pred,
                 inout DifferentialPair<float> target,
                 float dout) {
    float s = sign(pred.p - target.p);
    float dx_pred = dout * s;
    float dx_target = -dx_pred;
    pred = diffPair(pred.p, pred.d + dx_pred);
    target = diffPair(target.p, target.d + dx_target);
}

[Differentiable]
[BackwardDerivative(smooth_l1_elem_bwd)]
public float smooth_l1_elem(float pred, float target, no_diff float beta) {
    float d = pred - target;
    float ad = abs(d);
    if (ad < beta) {
        return 0.5f * d * d / beta;
    }
    return ad - 0.5f * beta;
}
void smooth_l1_elem_bwd(inout DifferentialPair<float> pred,
                        inout DifferentialPair<float> target,
                        no_diff float beta,
                        float dout) {
    float d = pred.p - target.p;
    float ad = abs(d);
    float grad = (ad < beta) ? (d / beta) : sign(d);
    float dx_pred = dout * grad;
    float dx_target = -dx_pred;
    pred = diffPair(pred.p, pred.d + dx_pred);
    target = diffPair(target.p, target.d + dx_target);
}

[Differentiable]
[BackwardDerivative(huber_elem_bwd)]
public float huber_elem(float pred, float target, no_diff float delta) {
    float d = pred - target;
    float ad = abs(d);
    if (ad < delta) {
        return 0.5f * d * d;
    }
    return delta * (ad - 0.5f * delta);
}
void huber_elem_bwd(inout DifferentialPair<float> pred,
                    inout DifferentialPair<float> target,
                    no_diff float delta,
                    float dout) {
    float d = pred.p - target.p;
    float ad = abs(d);
    float grad = (ad < delta) ? d : (delta * sign(d));
    float dx_pred = dout * grad;
    float dx_target = -dx_pred;
    pred = diffPair(pred.p, pred.d + dx_pred);
    target = diffPair(target.p, target.d + dx_target);
}

[Differentiable]
[BackwardDerivative(bce_elem_bwd)]
public float bce_elem(float pred, float target) {
    float p = clamp(pred, 1e-7f, 1.0f - 1e-7f);
    return -(target * log(p) + (1.0f - target) * log(1.0f - p));
}
void bce_elem_bwd(inout DifferentialPair<float> pred,
                  inout DifferentialPair<float> target,
                  float dout) {
    // Match the forward's clamp so the backward gradient saturates at the
    // same boundary the forward log() saw — no NaN at p=0/1.
    float p = clamp(pred.p, 1e-7f, 1.0f - 1e-7f);
    float t = target.p;
    float dx_pred = dout * (p - t) / (p * (1.0f - p));
    float dx_target = dout * (log(1.0f - p) - log(p));
    pred = diffPair(pred.p, pred.d + dx_pred);
    target = diffPair(target.p, target.d + dx_target);
}

[Differentiable]
[BackwardDerivative(bce_with_logits_elem_bwd)]
public float bce_with_logits_elem(float logit, float target) {
    float max_l = max(logit, 0.0f);
    return max_l - logit * target + log(1.0f + exp(-abs(logit)));
}
void bce_with_logits_elem_bwd(inout DifferentialPair<float> logit,
                              inout DifferentialPair<float> target,
                              float dout) {
    // sigmoid(logit) - target — numerically stable, no exp-tape.
    float sig = 1.0f / (1.0f + exp(-logit.p));
    float dx_logit = dout * (sig - target.p);
    float dx_target = dout * (-logit.p);
    logit = diffPair(logit.p, logit.d + dx_logit);
    target = diffPair(target.p, target.d + dx_target);
}

[Differentiable]
[BackwardDerivative(kl_div_elem_bwd)]
public float kl_div_elem(float log_pred, float target) {
    return target * (log(target) - log_pred);
}
void kl_div_elem_bwd(inout DifferentialPair<float> log_pred,
                     inout DifferentialPair<float> target,
                     float dout) {
    float dx_log_pred = dout * (-target.p);
    // d/d(target) of target*(log(target) - log_pred)
    //   = (log(target) - log_pred) + target * (1/target)
    //   = log(target) - log_pred + 1
    float dx_target = dout * (log(target.p) - log_pred.p + 1.0f);
    log_pred = diffPair(log_pred.p, log_pred.d + dx_log_pred);
    target = diffPair(target.p, target.d + dx_target);
}

// ── CG.M7: SDPA softmax scalar chain ──────────────────────────────────
// The softmax forward operates on a pair (score, running_max, running_sum)
// implementing online softmax.  The backward is the well-known formula:
//   dS = P * (dP - sum(P * dP))
// where P = exp(score - max) / sum(exp(score - max)).
//
// softmax_exp_sub: computes exp(score - max_sub) — the core softmax numerator.
// Marked [Differentiable] so bwd_diff can chain through the exp.
[Differentiable]
public float softmax_exp_sub(float score, no_diff float max_sub) {
    return exp(score - max_sub);
}

// ── CG.M7: SDPA scalar inner functions ────────────────────────────────
// These mirror tile_inner_madd from mm_tile.slang; inlined here so the
// flash attention backward template is self-contained without importing
// the mm module.  Autodiff derivatives:
//   dL/da = dL/dout * b    (used for Q·K score and P·V output accum)
//   dL/db = dL/dout * a    (used for grad w.r.t. second operand)
[Differentiable]
public float sdpa_score(float a, float b, no_diff float acc) {
    return acc + a * b;
}

[Differentiable]
public float sdpa_output(float a, float b, no_diff float acc) {
    return acc + a * b;
}

// ── CG.M7: Per-element softmax backward (closed-form fast bwd) ────────
// softmax_elem implements one element of the online-softmax forward:
//   p_i = exp(score_i - max_global) / sum_global
// The backward formula for a single element:
//   dp_i = dout * (p_i * dout_i - p_i * sum_j(p_j * dout_j))
// which simplifies to:
//   dscore_i = p_i * (din_i - acc)   where acc = sum_j(p_j * din_j)
//
// This function takes the per-element softmax probability p, the per-element
// incoming gradient din, and the precomputed row-wise dot(probs, din).
// It returns the gradient w.r.t. the original score (pre-softmax).
[Differentiable]
[BackwardDerivative(softmax_elem_bwd)]
public float softmax_elem(float p, float din, no_diff float row_dot) {
    return p * (din - row_dot);
}
void softmax_elem_bwd(inout DifferentialPair<float> p,
                      inout DifferentialPair<float> din,
                      no_diff float row_dot,
                      float dout) {
    // Forward: y = p * (din - row_dot)
    // dL/dp = dout * (din - row_dot)
    // dL/d(din) = dout * p
    float dx_p = dout * (din.p - row_dot);
    float dx_din = dout * p.p;
    p = diffPair(p.p, p.d + dx_p);
    din = diffPair(din.p, din.d + dx_din);
}
