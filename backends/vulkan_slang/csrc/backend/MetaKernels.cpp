// Meta (FakeTensor) kernel registrations for torch.compile support.
// These describe output shape/dtype/device without running actual computation.
// Required for AOT Autograd and Inductor tracing.
//
// M22.8 (2026-05-18): cleaned out 66 defined-but-unregistered meta_* stubs.
// Anti-goal #5 (no dead symptom-patches) — upstream PyTorch 2.10+ ships
// structured-kernel Meta implementations for every standard ATen op those
// stubs covered, so leaving them around was silent dead code that risked
// being wired up later with incorrect semantics. Three of the deletions
// were latent miscompile traps:
//
//   * `meta_transpose` returned `at::empty(swapped_sizes)` which produces
//     CONTIGUOUS strides — not the swapped-stride view that real
//     `aten::transpose` produces. Upstream's transpose-Meta correctly
//     synthesises the swapped-stride view. If we had ever registered
//     this stub, FakeTensor strides would diverge from real strides and
//     downstream Inductor / AOT-Autograd would silently miscompile.
//   * `meta_layer_norm`, `meta_group_norm`, `meta_batch_norm` were
//     defined to return a single tensor (or a 3-tuple with wrong
//     stat shapes for layer_norm) — but the real ops return
//     `(out, mean, rstd)` 3-tuples with broadcast-shaped stats.
//     Registering would have crashed at import or produced wrong shapes.
//
// Keep set (37 explicit `m.impl(...)` calls below) covers only ops where
// PyTorch's built-in dispatch on `PrivateUse1` does not already produce a
// correct Meta shape — namely scalar binary / pow / inplace-scalar
// variants, backward helpers (which routinely lack upstream Meta
// coverage), and a handful of Phase-3 model-coverage ops.

#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <ATen/Functions.h>
#include <ATen/ExpandUtils.h>

namespace torch_vulkan {

// ── Helpers ─────────────────────────────────────────────────────

// Wrap at::empty to avoid ambiguity with std::empty in C++20.
static at::Tensor meta_empty(std::vector<int64_t> sizes, at::TensorOptions opts) {
    return at::empty(c10::IntArrayRef(sizes), opts);
}

// Make meta tensor with same dtype/device as input but given shape.
static at::Tensor meta_like(const at::Tensor& input, at::IntArrayRef sizes) {
    return at::empty(sizes, input.options().device(at::kMeta));
}

static at::Tensor meta_like_self(const at::Tensor& self) {
    return at::empty(self.sizes().vec(), self.options().device(at::kMeta));
}

// ── Scalar binary ops: same shape as self ───────────────────────

static at::Tensor meta_binary_scalar(const at::Tensor& self, const at::Scalar&, const at::Scalar&) {
    return meta_like_self(self);
}

static at::Tensor meta_binary_scalar_no_alpha(const at::Tensor& self, const at::Scalar&) {
    return meta_like_self(self);
}

static at::Tensor meta_comparison_scalar(const at::Tensor& self, const at::Scalar&) {
    return at::empty(self.sizes().vec(), self.options().dtype(at::kBool).device(at::kMeta));
}

static at::Tensor meta_pow_tensor_scalar(const at::Tensor& self, const at::Scalar&) {
    return meta_like_self(self);
}

// ── In-place scalar ops: return self ────────────────────────────

static at::Tensor& meta_inplace_scalar(at::Tensor& self, const at::Scalar&, const at::Scalar&) {
    return self;
}

static at::Tensor& meta_inplace_scalar_no_alpha(at::Tensor& self, const at::Scalar&) {
    return self;
}

// ── Backward helper meta kernels ────────────────────────────────

static at::Tensor meta_threshold_backward(const at::Tensor& grad_output, const at::Tensor&, const at::Scalar&) {
    return meta_like_self(grad_output);
}

static at::Tensor meta_sigmoid_backward(const at::Tensor& grad_output, const at::Tensor&) {
    return meta_like_self(grad_output);
}

static at::Tensor meta_tanh_backward(const at::Tensor& grad_output, const at::Tensor&) {
    return meta_like_self(grad_output);
}

static at::Tensor meta_gelu_backward(const at::Tensor& grad_output, const at::Tensor&, c10::string_view) {
    return meta_like_self(grad_output);
}

static at::Tensor meta_silu_backward(const at::Tensor& grad_output, const at::Tensor&) {
    return meta_like_self(grad_output);
}

static at::Tensor meta_leaky_relu_backward(const at::Tensor& grad_output, const at::Tensor&, const at::Scalar&, bool) {
    return meta_like_self(grad_output);
}

static at::Tensor meta_elu_backward(const at::Tensor& grad_output, const at::Scalar&, const at::Scalar&, const at::Scalar&, bool, const at::Tensor&) {
    return meta_like_self(grad_output);
}

static at::Tensor meta_softmax_backward_data(const at::Tensor& grad_output, const at::Tensor&, int64_t, at::ScalarType) {
    return meta_like_self(grad_output);
}

static at::Tensor meta_log_softmax_backward_data(const at::Tensor& grad_output, const at::Tensor&, int64_t, at::ScalarType) {
    return meta_like_self(grad_output);
}

static at::Tensor meta_avg_pool2d_backward(const at::Tensor&, const at::Tensor& self,
    at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool, bool, std::optional<int64_t>) {
    return meta_like_self(self);
}

static std::tuple<at::Tensor, at::Tensor> meta_max_pool2d_with_indices(
    const at::Tensor& self, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef, bool) {
    int64_t kH = kernel_size[0], kW = kernel_size.size() > 1 ? kernel_size[1] : kH;
    int64_t sH = stride.empty() ? kH : stride[0];
    int64_t sW = stride.empty() ? kW : (stride.size() > 1 ? stride[1] : sH);
    int64_t pH = padding.empty() ? 0 : padding[0];
    int64_t pW = padding.empty() ? 0 : (padding.size() > 1 ? padding[1] : pH);
    int64_t oH = (self.size(2) + 2*pH - kH) / sH + 1;
    int64_t oW = (self.size(3) + 2*pW - kW) / sW + 1;
    auto opts = self.options().device(at::kMeta);
    return std::make_tuple(
        meta_empty({self.size(0), self.size(1), oH, oW}, opts),
        meta_empty({self.size(0), self.size(1), oH, oW}, opts.dtype(at::kLong)));
}

static at::Tensor meta_max_pool2d_with_indices_backward(
    const at::Tensor&, const at::Tensor& self,
    at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool, const at::Tensor&) {
    return meta_like_self(self);
}

static at::Tensor meta_embedding_dense_backward(
    const at::Tensor& grad_output, const at::Tensor&,
    c10::SymInt num_weights, c10::SymInt, bool) {
    return at::empty({num_weights.expect_int(), grad_output.size(-1)},
                     grad_output.options().device(at::kMeta));
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> meta_native_layer_norm_backward(
    const at::Tensor&, const at::Tensor& input,
    c10::SymIntArrayRef, const at::Tensor&, const at::Tensor&,
    const std::optional<at::Tensor>& weight, const std::optional<at::Tensor>&,
    std::array<bool, 3> output_mask) {
    auto opts = input.options().device(at::kMeta);
    return std::make_tuple(
        output_mask[0] ? at::empty(input.sizes().vec(), opts) : at::Tensor(),
        output_mask[1] && weight.has_value() ? at::empty(weight->sizes().vec(), opts) : at::Tensor(),
        output_mask[2] && weight.has_value() ? at::empty(weight->sizes().vec(), opts) : at::Tensor());
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> meta_native_group_norm_backward(
    const at::Tensor&, const at::Tensor& input,
    const at::Tensor&, const at::Tensor&,
    const std::optional<at::Tensor>& weight,
    c10::SymInt, c10::SymInt C, c10::SymInt, int64_t,
    std::array<bool, 3> output_mask) {
    auto opts = input.options().device(at::kMeta);
    return std::make_tuple(
        output_mask[0] ? at::empty(input.sizes().vec(), opts) : at::Tensor(),
        output_mask[1] ? at::empty({C.expect_int()}, opts) : at::Tensor(),
        output_mask[2] ? at::empty({C.expect_int()}, opts) : at::Tensor());
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> meta_native_batch_norm_backward(
    const at::Tensor&, const at::Tensor& input,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>&, const std::optional<at::Tensor>&,
    const std::optional<at::Tensor>&, const std::optional<at::Tensor>&,
    bool, double, std::array<bool, 3> output_mask) {
    auto opts = input.options().device(at::kMeta);
    int64_t C = input.size(1);
    return std::make_tuple(
        output_mask[0] ? at::empty(input.sizes().vec(), opts) : at::Tensor(),
        output_mask[1] ? at::empty({C}, opts) : at::Tensor(),
        output_mask[2] ? at::empty({C}, opts) : at::Tensor());
}

// ── linear_backward meta kernel ──────────────────────────────────

static std::tuple<at::Tensor, at::Tensor, at::Tensor> meta_linear_backward(
    const at::Tensor& self, const at::Tensor& grad_output,
    const at::Tensor& weight, std::array<bool, 3> output_mask) {
    auto opts = self.options().device(at::kMeta);
    return std::make_tuple(
        output_mask[0] ? at::empty(self.sizes().vec(), opts) : at::Tensor(),
        output_mask[1] ? at::empty(weight.sizes().vec(), opts) : at::Tensor(),
        output_mask[2] ? at::empty({weight.size(0)}, opts) : at::Tensor());
}

// ── Upsample backward ───────────────────────────────────────────

static at::Tensor meta_upsample_nearest2d_backward(const at::Tensor& grad_output,
    at::IntArrayRef output_size, at::IntArrayRef input_size,
    std::optional<double>, std::optional<double>) {
    return meta_empty({input_size[0], input_size[1], input_size[2], input_size[3]},
                     grad_output.options().device(at::kMeta));
}

static at::Tensor meta_upsample_bilinear2d_backward(const at::Tensor& grad_output,
    at::IntArrayRef output_size, at::IntArrayRef input_size,
    bool, std::optional<double>, std::optional<double>) {
    return meta_empty({input_size[0], input_size[1], input_size[2], input_size[3]},
                     grad_output.options().device(at::kMeta));
}

// ── Phase 3: Model coverage Meta kernels ────────────────────────

// triu/tril: same shape
static at::Tensor meta_triu(const at::Tensor& self, int64_t /*diagonal*/) {
    return meta_like_self(self);
}
static at::Tensor meta_tril(const at::Tensor& self, int64_t /*diagonal*/) {
    return meta_like_self(self);
}

// constant_pad_nd: compute padded shape
static at::Tensor meta_constant_pad_nd(const at::Tensor& self,
                                        c10::SymIntArrayRef pad,
                                        const at::Scalar& /*value*/) {
    auto sizes = self.sizes().vec();
    int64_t ndim = sizes.size();
    int64_t npairs = pad.size() / 2;
    for (int64_t i = 0; i < npairs; i++) {
        int64_t dim = ndim - 1 - i;
        sizes[dim] += pad[2 * i].expect_int() + pad[2 * i + 1].expect_int();
    }
    return meta_like(self, sizes);
}

// erf: same shape
static at::Tensor meta_erf(const at::Tensor& self) {
    return meta_like_self(self);
}

// flip: same shape
static at::Tensor meta_flip(const at::Tensor& self, at::IntArrayRef /*dims*/) {
    return meta_like_self(self);
}

// roll: same shape
static at::Tensor meta_roll(const at::Tensor& self, c10::SymIntArrayRef /*shifts*/,
                             at::IntArrayRef /*dims*/) {
    return meta_like_self(self);
}

// fmod/remainder: broadcast binary, same shape
static at::Tensor meta_fmod(const at::Tensor& self, const at::Tensor& other) {
    auto out_size = at::infer_size_dimvector(self.sizes(), other.sizes());
    return meta_like(self, out_size);
}

// cumprod: same shape as input
static at::Tensor meta_cumprod(const at::Tensor& self, int64_t, std::optional<at::ScalarType>) {
    return meta_like_self(self);
}

// ── Registration ────────────────────────────────────────────────
//
// NOTE: PyTorch 2.10+ has built-in Meta kernels for most standard ATen ops.
// We only register Meta implementations for ops where PyTorch's built-in
// decompositions or meta kernels don't cover our custom dispatch.
// Registering meta kernels with wrong signatures causes fatal errors at
// import. If torch.compile needs a meta kernel for a specific op, add it
// here with the EXACT signature matching the op schema — and PROVE the
// upstream Meta is missing or incorrect for our dispatch before adding it.
//
// Audit Agent 2 (2026-05-18) verified that the 66 previously-defined
// stubs (deleted in this commit) all had correct upstream Meta coverage
// for every relevant model in our nine-model end-to-end suite. If a new
// op needs Meta here, document the gap and ship a regression test
// alongside the registration.

TORCH_LIBRARY_IMPL(aten, Meta, m) {
    // Scalar-promoted binary ops — these override PyTorch's default dispatch
    // on PrivateUse1 so torch.compile's FakeTensor tracing needs Meta kernels.
    m.impl("add.Scalar", meta_binary_scalar);
    m.impl("sub.Scalar", meta_binary_scalar);
    m.impl("mul.Scalar", meta_binary_scalar_no_alpha);
    m.impl("div.Scalar", meta_binary_scalar_no_alpha);
    m.impl("pow.Tensor_Scalar", meta_pow_tensor_scalar);

    // In-place scalar variants
    m.impl("add_.Scalar", meta_inplace_scalar);
    m.impl("sub_.Scalar", meta_inplace_scalar);
    m.impl("mul_.Scalar", meta_inplace_scalar_no_alpha);
    m.impl("div_.Scalar", meta_inplace_scalar_no_alpha);

    // Scalar comparison ops
    m.impl("eq.Scalar", meta_comparison_scalar);
    m.impl("ne.Scalar", meta_comparison_scalar);
    m.impl("lt.Scalar", meta_comparison_scalar);
    m.impl("gt.Scalar", meta_comparison_scalar);
    m.impl("le.Scalar", meta_comparison_scalar);
    m.impl("ge.Scalar", meta_comparison_scalar);

    // Backward helper ops (for torch.compile tracing)
    m.impl("threshold_backward", meta_threshold_backward);
    m.impl("sigmoid_backward", meta_sigmoid_backward);
    m.impl("tanh_backward", meta_tanh_backward);
    m.impl("gelu_backward", meta_gelu_backward);
    m.impl("silu_backward", meta_silu_backward);
    m.impl("leaky_relu_backward", meta_leaky_relu_backward);
    m.impl("elu_backward", meta_elu_backward);
    m.impl("_softmax_backward_data", meta_softmax_backward_data);
    m.impl("_log_softmax_backward_data", meta_log_softmax_backward_data);
    m.impl("avg_pool2d_backward", meta_avg_pool2d_backward);
    m.impl("max_pool2d_with_indices", meta_max_pool2d_with_indices);
    m.impl("max_pool2d_with_indices_backward", meta_max_pool2d_with_indices_backward);
    m.impl("embedding_dense_backward", meta_embedding_dense_backward);
    m.impl("native_layer_norm_backward", meta_native_layer_norm_backward);
    m.impl("native_group_norm_backward", meta_native_group_norm_backward);
    m.impl("native_batch_norm_backward", meta_native_batch_norm_backward);
    m.impl("linear_backward", meta_linear_backward);

    // Phase 3: Model coverage ops
    m.impl("triu", meta_triu);
    m.impl("tril", meta_tril);
    m.impl("constant_pad_nd", meta_constant_pad_nd);
    m.impl("erf", meta_erf);
    m.impl("flip", meta_flip);
    m.impl("roll", meta_roll);

    // Upsample backward
    m.impl("upsample_nearest2d_backward", meta_upsample_nearest2d_backward);
    m.impl("upsample_bilinear2d_backward", meta_upsample_bilinear2d_backward);

    // New ops
    m.impl("fmod.Tensor", meta_fmod);
    m.impl("remainder.Tensor", meta_fmod);  // same shape logic
    m.impl("cumprod", meta_cumprod);
}

} // namespace torch_vulkan
