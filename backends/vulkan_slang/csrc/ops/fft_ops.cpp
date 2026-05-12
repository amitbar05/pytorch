#include "ops.h"
#include "dispatch.h"
#include "dtype_utils.h"
#include "../generated/shaders.h"

#include <torch/library.h>
#include <cmath>

namespace torch_vulkan { namespace ops {

// ─── helpers ────────────────────────────────────────────────────────────────

static bool is_pow2(int64_t n) { return n > 0 && (n & (n - 1)) == 0; }

static int ilog2(int64_t n) {
    int r = 0;
    while (n > 1) { n >>= 1; ++r; }
    return r;
}

// normalization codes (fft_norm_mode enum, already direction-adjusted by PyTorch):
//   0 = none:      no scale (used for forward in "backward" convention, or
//                            inverse in "forward" convention)
//   1 = by_root_n: scale by 1/sqrt(N) (ortho)
//   2 = by_n:      scale by 1/N (used for inverse in "backward" convention, or
//                                forward in "forward" convention)
// PyTorch calls norm_from_string(norm, is_forward) before dispatching, so the
// normalization code already accounts for direction — forward param is unused here.
static float fft_scale(int normalization, int64_t N, bool /*forward*/) {
    if (normalization == 1) return 1.0f / std::sqrt(float(N));
    if (normalization == 2) return 1.0f / float(N);
    return 1.0f;  // 0 = none: no scale
}

// ─── core 1-D FFT engine ────────────────────────────────────────────────────
// Executes log2(N) DIT butterfly passes, ping-ponging between work and tmp.
// On entry: work holds bit-reversed complex input (interleaved floats, [batch, 2N]).
// Returns the tensor that holds the final result.
static at::Tensor run_fft_passes(at::Tensor work, at::Tensor tmp,
                                  int64_t N, int64_t batch, bool forward) {
    int m = ilog2(N);
    struct BflyParams { uint32_t N, pass_idx, batch, forward; };

    for (int pass = 0; pass < m; ++pass) {
        BflyParams p{ (uint32_t)N, (uint32_t)pass, (uint32_t)batch, forward ? 1u : 0u };
        uint32_t wg = ((uint32_t)(batch * N / 2) + 255u) / 256u;
        at::Tensor src = work.reshape({-1});
        at::Tensor dst = tmp.reshape({-1});
        dispatch_shader("fft_butterfly_fwd",
                        shaders::fft_butterfly_fwd, shaders::fft_butterfly_fwd_size,
                        {src, dst}, wg, 1u, 1u, &p, sizeof(p), 1u);
        std::swap(work, tmp);
    }
    return work;
}

// ─── _fft_r2c ───────────────────────────────────────────────────────────────

at::Tensor vulkan_fft_r2c(const at::Tensor& self,
                           at::IntArrayRef   dim,
                           int64_t           normalization,
                           bool              onesided) {
    TORCH_CHECK(self.is_floating_point(),
                "_fft_r2c: input must be a real floating-point tensor");
    TORCH_CHECK(dim.size() == 1,
                "_fft_r2c: only 1-D FFT supported, got dim.size()=", dim.size());

    auto self_c = self.contiguous();
    check_supported_float(self_c, "_fft_r2c");
    auto self_f32 = ensure_float32(self_c);

    int64_t fft_dim = at::maybe_wrap_dim(dim[0], self_f32.dim());
    int64_t N = self_f32.size(fft_dim);

    TORCH_CHECK(is_pow2(N), "_fft_r2c: FFT size must be a power of 2, got ", N);
    TORCH_CHECK(N <= (1 << 24), "_fft_r2c: FFT size too large, got ", N);

    int64_t batch = self_f32.numel() / N;

    // Move fft_dim to last so memory layout is [batch, N]
    auto x = self_f32;
    if (fft_dim != x.dim() - 1)
        x = x.movedim(fft_dim, -1).contiguous();

    at::TensorOptions opts = at::TensorOptions().dtype(at::kFloat).device(self.device());
    auto work = at::empty({batch, 2 * N}, opts);
    auto tmp  = at::empty({batch, 2 * N}, opts);

    // Load real input with bit-reversal into complex work buffer (imag = 0)
    {
        struct { uint32_t N, log2_N, batch, in_stride; } p{
            (uint32_t)N, (uint32_t)ilog2(N), (uint32_t)batch, (uint32_t)N
        };
        uint32_t wg = ((uint32_t)(batch * N) + 255u) / 256u;
        at::Tensor x_flat   = x.reshape({-1});
        at::Tensor work_flat = work.reshape({-1});
        dispatch_shader("fft_init_r2c_fwd",
                        shaders::fft_init_r2c_fwd, shaders::fft_init_r2c_fwd_size,
                        {x_flat, work_flat}, wg, 1u, 1u, &p, sizeof(p), 1u);
    }

    at::Tensor result = run_fft_passes(work, tmp, N, batch, true);

    // Normalization
    float sc = fft_scale((int)normalization, N, true);
    if (sc != 1.0f) {
        struct { uint32_t numel; float scale; } sp{ (uint32_t)(batch * 2 * N), sc };
        uint32_t wg = ((uint32_t)(batch * 2 * N) + 255u) / 256u;
        at::Tensor res_flat = result.reshape({-1});
        dispatch_shader("fft_scale_fwd",
                        shaders::fft_scale_fwd, shaders::fft_scale_fwd_size,
                        {res_flat}, wg, 1u, 1u, &sp, sizeof(sp), 1u);
    }

    // Extract one-sided spectrum (first N/2+1 complex values)
    int64_t out_N = onesided ? N / 2 + 1 : N;
    at::Tensor out_float;
    if (onesided) {
        out_float = at::empty({batch, 2 * out_N}, opts);
        struct { uint32_t N, batch; } p{ (uint32_t)N, (uint32_t)batch };
        uint32_t wg = ((uint32_t)(batch * out_N) + 255u) / 256u;
        at::Tensor res_flat = result.reshape({-1});
        at::Tensor out_flat = out_float.reshape({-1});
        dispatch_shader("fft_r2c_out_fwd",
                        shaders::fft_r2c_out_fwd, shaders::fft_r2c_out_fwd_size,
                        {res_flat, out_flat}, wg, 1u, 1u, &p, sizeof(p), 1u);
    } else {
        out_float = result;
    }

    // Reinterpret [batch, 2*out_N] → [batch, out_N, 2] → complex64 → restore batch shape
    auto out_2 = out_float.view({batch, out_N, 2});
    at::Tensor out_complex = at::view_as_complex(out_2.contiguous());

    auto out_shape = self.sizes().vec();
    out_shape[fft_dim] = out_N;
    return out_complex.view(out_shape);
}

// ─── _fft_c2c ───────────────────────────────────────────────────────────────

at::Tensor vulkan_fft_c2c(const at::Tensor& self,
                           at::IntArrayRef   dim,
                           int64_t           normalization,
                           bool              forward) {
    TORCH_CHECK(self.is_complex(),
                "_fft_c2c: input must be a complex tensor");
    TORCH_CHECK(dim.size() == 1,
                "_fft_c2c: only 1-D FFT supported, got dim.size()=", dim.size());

    int64_t fft_dim = at::maybe_wrap_dim(dim[0], self.dim());
    int64_t N = self.size(fft_dim);
    TORCH_CHECK(is_pow2(N), "_fft_c2c: FFT size must be a power of 2, got ", N);
    TORCH_CHECK(N <= (1 << 24), "_fft_c2c: FFT size too large, got ", N);

    int64_t batch = self.numel() / N;

    // Reinterpret complex as float pairs: [..., N] → [..., N, 2]
    auto x_real = at::view_as_real(self.contiguous());
    if (fft_dim != self.dim() - 1)
        x_real = x_real.movedim(fft_dim, -2).contiguous();

    at::TensorOptions opts = at::TensorOptions().dtype(at::kFloat).device(self.device());
    auto work = at::empty({batch, 2 * N}, opts);
    auto tmp  = at::empty({batch, 2 * N}, opts);

    // Load complex input with bit-reversal
    {
        struct { uint32_t N, log2_N, batch, in_stride; } p{
            (uint32_t)N, (uint32_t)ilog2(N), (uint32_t)batch, (uint32_t)N
        };
        uint32_t wg = ((uint32_t)(batch * N) + 255u) / 256u;
        at::Tensor x_flat    = x_real.reshape({-1});
        at::Tensor work_flat = work.reshape({-1});
        dispatch_shader("fft_init_c2c_fwd",
                        shaders::fft_init_c2c_fwd, shaders::fft_init_c2c_fwd_size,
                        {x_flat, work_flat}, wg, 1u, 1u, &p, sizeof(p), 1u);
    }

    at::Tensor result = run_fft_passes(work, tmp, N, batch, forward);

    float sc = fft_scale((int)normalization, N, forward);
    if (sc != 1.0f) {
        struct { uint32_t numel; float scale; } sp{ (uint32_t)(batch * 2 * N), sc };
        uint32_t wg = ((uint32_t)(batch * 2 * N) + 255u) / 256u;
        at::Tensor res_flat = result.reshape({-1});
        dispatch_shader("fft_scale_fwd",
                        shaders::fft_scale_fwd, shaders::fft_scale_fwd_size,
                        {res_flat}, wg, 1u, 1u, &sp, sizeof(sp), 1u);
    }

    auto out_2 = result.view({batch, N, 2});
    at::Tensor out = at::view_as_complex(out_2.contiguous());
    return out.view(self.sizes());
}

// ─── _fft_c2r ───────────────────────────────────────────────────────────────

at::Tensor vulkan_fft_c2r(const at::Tensor& self,
                           at::IntArrayRef   dim,
                           int64_t           normalization,
                           int64_t           last_dim_size) {
    TORCH_CHECK(self.is_complex(),
                "_fft_c2r: input must be a complex tensor");
    TORCH_CHECK(dim.size() == 1,
                "_fft_c2r: only 1-D FFT supported, got dim.size()=", dim.size());

    int64_t N = last_dim_size;
    TORCH_CHECK(is_pow2(N), "_fft_c2r: output size must be a power of 2, got ", N);
    TORCH_CHECK(N <= (1 << 24), "_fft_c2r: FFT size too large, got ", N);

    int64_t fft_dim = at::maybe_wrap_dim(dim[0], self.dim());
    TORCH_CHECK(self.size(fft_dim) == N / 2 + 1,
                "_fft_c2r: expected input size N/2+1=", N/2+1,
                " along dim ", fft_dim, " got ", self.size(fft_dim));

    int64_t batch = self.numel() / (N / 2 + 1);
    at::TensorOptions opts = at::TensorOptions().dtype(at::kFloat).device(self.device());

    auto x_real = at::view_as_real(self.contiguous());
    if (fft_dim != self.dim() - 1)
        x_real = x_real.movedim(fft_dim, -2).contiguous();
    at::Tensor x_flat = x_real.reshape({-1});

    // Extend Hermitian spectrum to full N
    auto full = at::empty({batch, 2 * N}, opts);
    {
        struct { uint32_t N, batch; } p{ (uint32_t)N, (uint32_t)batch };
        uint32_t wg = ((uint32_t)(batch * N) + 255u) / 256u;
        at::Tensor full_flat = full.reshape({-1});
        dispatch_shader("fft_c2r_conj_fwd",
                        shaders::fft_c2r_conj_fwd, shaders::fft_c2r_conj_fwd_size,
                        {x_flat, full_flat}, wg, 1u, 1u, &p, sizeof(p), 1u);
    }

    // Bit-reversal + inverse FFT
    auto work = at::empty({batch, 2 * N}, opts);
    auto tmp  = at::empty({batch, 2 * N}, opts);
    {
        struct { uint32_t N, log2_N, batch, in_stride; } p{
            (uint32_t)N, (uint32_t)ilog2(N), (uint32_t)batch, (uint32_t)N
        };
        uint32_t wg = ((uint32_t)(batch * N) + 255u) / 256u;
        at::Tensor full_flat = full.reshape({-1});
        at::Tensor work_flat = work.reshape({-1});
        dispatch_shader("fft_init_c2c_fwd",
                        shaders::fft_init_c2c_fwd, shaders::fft_init_c2c_fwd_size,
                        {full_flat, work_flat}, wg, 1u, 1u, &p, sizeof(p), 1u);
    }

    at::Tensor result = run_fft_passes(work, tmp, N, batch, false);

    float sc = fft_scale((int)normalization, N, false);
    if (sc != 1.0f) {
        struct { uint32_t numel; float scale; } sp{ (uint32_t)(batch * 2 * N), sc };
        uint32_t wg = ((uint32_t)(batch * 2 * N) + 255u) / 256u;
        at::Tensor res_flat = result.reshape({-1});
        dispatch_shader("fft_scale_fwd",
                        shaders::fft_scale_fwd, shaders::fft_scale_fwd_size,
                        {res_flat}, wg, 1u, 1u, &sp, sizeof(sp), 1u);
    }

    // Extract real parts: [batch, 2N] → [batch, N, 2] → select index 0 on last dim
    auto real_out = result.view({batch, N, 2}).select(-1, 0).contiguous();

    auto out_shape = self.sizes().vec();
    out_shape[fft_dim] = N;
    return real_out.view(out_shape);
}

// ─── linalg_svd (one-sided Jacobi, GPU, M >= N, N <= 32, M <= 256) ──────────

// Internal helper: runs Jacobi SVD on GPU, always computes U and V.
static std::tuple<at::Tensor, at::Tensor, at::Tensor>
svd_jacobi(const at::Tensor& A, bool full_matrices) {
    auto A_c = A.contiguous();
    check_supported_float(A_c, "linalg_svd");
    auto A_f = ensure_float32(A_c);

    int64_t M = A_f.size(-2);
    int64_t N = A_f.size(-1);
    int64_t batch = A_f.numel() / (M * N);

    bool transposed = (M < N);
    if (transposed) {
        A_f = A_f.mT().contiguous();
        std::swap(M, N);
    }

    TORCH_CHECK(N <= 32,
                "linalg_svd: GPU Jacobi supports N <= 32 columns, got N=", N);
    TORCH_CHECK(M <= 256,
                "linalg_svd: GPU Jacobi supports M <= 256 rows, got M=", M);

    at::TensorOptions opts = at::TensorOptions().dtype(at::kFloat).device(A.device());
    int64_t Ku = full_matrices ? M : N;
    auto U_out = at::empty({batch, M, Ku}, opts);
    auto S_out = at::empty({batch, N}, opts);
    auto V_out = at::empty({batch, N, N}, opts);

    struct { uint32_t M, N, batch, max_sweeps, full_matrices; } p{
        (uint32_t)M, (uint32_t)N, (uint32_t)batch, 30u, full_matrices ? 1u : 0u
    };

    at::Tensor A_flat = A_f.reshape({-1});
    at::Tensor U_flat = U_out.reshape({-1});
    at::Tensor S_flat = S_out.reshape({-1});
    at::Tensor V_flat = V_out.reshape({-1});
    dispatch_shader("linalg_svd_jacobi_fwd",
                    shaders::linalg_svd_jacobi_fwd, shaders::linalg_svd_jacobi_fwd_size,
                    {A_flat, U_flat, S_flat, V_flat},
                    (uint32_t)batch, 1u, 1u, &p, sizeof(p), 3u);

    auto batch_shape = A.sizes().slice(0, A.dim() - 2).vec();
    auto S_shape = batch_shape; S_shape.push_back(N);
    auto U_rows = transposed ? N : M;
    auto V_rows = N;
    auto U_shape = batch_shape; U_shape.push_back(U_rows); U_shape.push_back(Ku);
    auto V_shape = batch_shape; V_shape.push_back(V_rows); V_shape.push_back(N);

    at::Tensor U_r = U_out.view(U_shape);
    at::Tensor S_r = S_out.view(S_shape);
    at::Tensor Vh_r = V_out.view(V_shape).mT().contiguous();

    if (transposed)
        return {Vh_r.mT().contiguous(), S_r, U_r.mT().contiguous()};
    return {U_r, S_r, Vh_r};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
vulkan_linalg_svd(const at::Tensor& A, bool full_matrices, bool compute_uv,
                  std::optional<c10::string_view> /*driver*/) {
    TORCH_CHECK(A.dim() >= 2,
                "linalg_svd: expected at least 2-D input, got ", A.dim(), "-D");
    if (compute_uv) {
        return svd_jacobi(A, full_matrices);
    }
    // compute_uv=false: only S is meaningful. Return empty U and Vh of correct
    // dtype/device so downstream shape checks pass without computing them.
    auto [U, S, Vh] = svd_jacobi(A, full_matrices);
    at::TensorOptions opts = at::TensorOptions().dtype(U.scalar_type()).device(A.device());
    auto U_empty = at::empty({0}, opts);
    auto Vh_empty = at::empty({0}, opts);
    return {U_empty, S, Vh_empty};
}

at::Tensor
vulkan_linalg_svdvals(const at::Tensor& A,
                      std::optional<c10::string_view> /*driver*/) {
    TORCH_CHECK(A.dim() >= 2,
                "linalg_svdvals: expected at least 2-D input, got ", A.dim(), "-D");
    auto [U, S, Vh] = svd_jacobi(A, false);
    return S;
}

}} // namespace torch_vulkan::ops
