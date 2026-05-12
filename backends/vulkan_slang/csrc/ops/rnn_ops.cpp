#include "ops.h"
#include "dispatch.h"
#include "../generated/shaders.h"

#include <torch/library.h>
#include <optional>

namespace torch_vulkan { namespace ops {

// Returns (hy, cy, workspace). If fast path doesn't apply, returns tensors with
// zero-initialized state — caller checks std::get<0>().defined().
std::tuple<at::Tensor, at::Tensor, at::Tensor>
vulkan_lstm_cell_fused_fwd(
    const at::Tensor& input_gates, const at::Tensor& hidden_gates,
    const at::Tensor& cx,
    const std::optional<at::Tensor>& input_bias,
    const std::optional<at::Tensor>& hidden_bias) {
    bool have_ib = input_bias.has_value()  && input_bias->defined();
    bool have_hb = hidden_bias.has_value() && hidden_bias->defined();
    if (input_gates.scalar_type() != at::kFloat
        || hidden_gates.scalar_type() != at::kFloat
        || cx.scalar_type() != at::kFloat
        || (have_ib && input_bias->scalar_type() != at::kFloat)
        || (have_hb && hidden_bias->scalar_type() != at::kFloat)
        || input_gates.dim() != 2 || hidden_gates.dim() != 2 || cx.dim() != 2
        || input_gates.sizes() != hidden_gates.sizes()
        || cx.size(0) != input_gates.size(0)
        || (input_gates.size(1) % 4) != 0
        || cx.size(1) != input_gates.size(1) / 4) {
        return {at::Tensor(), at::Tensor(), at::Tensor()};
    }
    auto ig = input_gates.contiguous();
    auto hg = hidden_gates.contiguous();
    auto cxc = cx.contiguous();
    int64_t B = ig.size(0);
    int64_t H = ig.size(1) / 4;
    auto opts = ig.options();
    auto hy = at::empty({B, H}, opts);
    auto cy = at::empty({B, H}, opts);
    auto ws = at::empty({B, 4 * H}, opts);
    at::Tensor bi = have_ib ? input_bias->contiguous()
                            : at::zeros({4 * H}, opts);
    at::Tensor bh = have_hb ? hidden_bias->contiguous()
                            : at::zeros({4 * H}, opts);
    bool has_bias = have_ib || have_hb;
    struct { uint32_t B, H, has_bias; } p{
        (uint32_t)B, (uint32_t)H, has_bias ? 1u : 0u};
    uint32_t total = (uint32_t)(B * H);
    uint32_t wg = (total + 255) / 256;
    dispatch_shader(
        "rnn_lstm_cell_fwd",
        shaders::rnn_lstm_cell_fwd_fwd,
        shaders::rnn_lstm_cell_fwd_fwd_size,
        {ig, hg, cxc, bi, bh, hy, cy, ws},
        wg, 1, 1, &p, sizeof(p), 3);
    return std::make_tuple(hy, cy, ws);
}

// Returns (grad_gates [B,4H], grad_cx [B,H]). Empty pair if fast path doesn't
// apply. Caller is responsible for cloning grad_gates for grad_hidden_gates.
std::tuple<at::Tensor, at::Tensor>
vulkan_lstm_cell_fused_bwd(
    const std::optional<at::Tensor>& grad_hy,
    const std::optional<at::Tensor>& grad_cy,
    const at::Tensor& cx, const at::Tensor& cy,
    const at::Tensor& workspace) {
    bool have_ghy = grad_hy.has_value() && grad_hy->defined();
    bool have_gcy = grad_cy.has_value() && grad_cy->defined();
    if (cx.scalar_type() != at::kFloat || cy.scalar_type() != at::kFloat
        || workspace.scalar_type() != at::kFloat
        || (have_ghy && grad_hy->scalar_type() != at::kFloat)
        || (have_gcy && grad_cy->scalar_type() != at::kFloat)
        || cx.dim() != 2 || cy.dim() != 2 || workspace.dim() != 2
        || cx.sizes() != cy.sizes()
        || workspace.size(0) != cx.size(0)
        || (workspace.size(1) % 4) != 0
        || workspace.size(1) / 4 != cx.size(1)) {
        return {at::Tensor(), at::Tensor()};
    }
    auto cxc = cx.contiguous();
    auto cyc = cy.contiguous();
    auto wsc = workspace.contiguous();
    int64_t B = cxc.size(0);
    int64_t H = cxc.size(1);
    auto opts = cxc.options();
    at::Tensor ghy_t = have_ghy ? grad_hy->contiguous() : at::zeros({B, H}, opts);
    at::Tensor gcy_t = have_gcy ? grad_cy->contiguous() : at::zeros({B, H}, opts);
    auto grad_gates = at::empty({B, 4 * H}, opts);
    auto grad_cx = at::empty({B, H}, opts);
    struct { uint32_t B, H, has_ghy, has_gcy; } p{
        (uint32_t)B, (uint32_t)H,
        have_ghy ? 1u : 0u, have_gcy ? 1u : 0u};
    uint32_t total = (uint32_t)(B * H);
    uint32_t wg = (total + 255) / 256;
    dispatch_shader(
        "rnn_lstm_cell_bwd",
        shaders::rnn_lstm_cell_bwd_fwd,
        shaders::rnn_lstm_cell_bwd_fwd_size,
        {cxc, cyc, wsc, ghy_t, gcy_t, grad_gates, grad_cx},
        wg, 1, 1, &p, sizeof(p), 2);
    return std::make_tuple(grad_gates, grad_cx);
}

std::tuple<at::Tensor, at::Tensor>
vulkan_gru_cell_fused_fwd(
    const at::Tensor& input_gates, const at::Tensor& hidden_gates,
    const at::Tensor& hx,
    const std::optional<at::Tensor>& input_bias,
    const std::optional<at::Tensor>& hidden_bias) {
    bool have_ib = input_bias.has_value()  && input_bias->defined();
    bool have_hb = hidden_bias.has_value() && hidden_bias->defined();
    if (input_gates.scalar_type() != at::kFloat
        || hidden_gates.scalar_type() != at::kFloat
        || hx.scalar_type() != at::kFloat
        || (have_ib && input_bias->scalar_type() != at::kFloat)
        || (have_hb && hidden_bias->scalar_type() != at::kFloat)
        || input_gates.dim() != 2 || hidden_gates.dim() != 2 || hx.dim() != 2
        || input_gates.sizes() != hidden_gates.sizes()
        || hx.size(0) != input_gates.size(0)
        || (input_gates.size(1) % 3) != 0
        || hx.size(1) != input_gates.size(1) / 3) {
        return {at::Tensor(), at::Tensor()};
    }
    auto ig = input_gates.contiguous();
    auto hg = hidden_gates.contiguous();
    auto hxc = hx.contiguous();
    int64_t B = ig.size(0);
    int64_t H = ig.size(1) / 3;
    auto opts = ig.options();
    auto hy = at::empty({B, H}, opts);
    auto ws = at::empty({B, 5 * H}, opts);
    at::Tensor bi = have_ib ? input_bias->contiguous()
                            : at::zeros({3 * H}, opts);
    at::Tensor bh = have_hb ? hidden_bias->contiguous()
                            : at::zeros({3 * H}, opts);
    bool has_bias = have_ib || have_hb;
    struct { uint32_t B, H, has_bias; } p{
        (uint32_t)B, (uint32_t)H, has_bias ? 1u : 0u};
    uint32_t total = (uint32_t)(B * H);
    uint32_t wg = (total + 255) / 256;
    dispatch_shader(
        "rnn_gru_cell_fwd",
        shaders::rnn_gru_cell_fwd_fwd,
        shaders::rnn_gru_cell_fwd_fwd_size,
        {ig, hg, hxc, bi, bh, hy, ws},
        wg, 1, 1, &p, sizeof(p), 2);
    return std::make_tuple(hy, ws);
}

// Returns (grad_input_gates, grad_hidden_gates, grad_hx). grad_input_bias /
// grad_hidden_bias computed by caller via sum if has_bias is true.
std::tuple<at::Tensor, at::Tensor, at::Tensor>
vulkan_gru_cell_fused_bwd(const at::Tensor& grad_hy, const at::Tensor& workspace) {
    if (grad_hy.scalar_type() != at::kFloat
        || workspace.scalar_type() != at::kFloat
        || grad_hy.dim() != 2 || workspace.dim() != 2
        || workspace.size(0) != grad_hy.size(0)
        || (workspace.size(1) % 5) != 0
        || workspace.size(1) / 5 != grad_hy.size(1)) {
        return {at::Tensor(), at::Tensor(), at::Tensor()};
    }
    auto ghy = grad_hy.contiguous();
    auto wsc = workspace.contiguous();
    int64_t B = ghy.size(0);
    int64_t H = ghy.size(1);
    auto opts = ghy.options();
    auto grad_ig = at::empty({B, 3 * H}, opts);
    auto grad_hg = at::empty({B, 3 * H}, opts);
    auto grad_hx = at::empty({B, H}, opts);
    struct { uint32_t B, H; } p{(uint32_t)B, (uint32_t)H};
    uint32_t total = (uint32_t)(B * H);
    uint32_t wg = (total + 255) / 256;
    dispatch_shader(
        "rnn_gru_cell_bwd",
        shaders::rnn_gru_cell_bwd_fwd,
        shaders::rnn_gru_cell_bwd_fwd_size,
        {ghy, wsc, grad_ig, grad_hg, grad_hx},
        wg, 1, 1, &p, sizeof(p), 3);
    return std::make_tuple(grad_ig, grad_hg, grad_hx);
}

}} // namespace torch_vulkan::ops
