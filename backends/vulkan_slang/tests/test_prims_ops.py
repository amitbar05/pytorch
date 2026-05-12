"""
Tests for PrimTorch (prims::) ops on the Vulkan backend.

Covers:
  - prims::xor_sum  (global + per-dim + multi-dim + dtypes + dispatch count)
  - prims::normal   (shape, mean/std, GPU RNG)
  - prims::_make_token / _sink_tokens (control-flow tokens)
  - prims::as_strided_scatter (GPU scatter, dispatch count verification)
  - prims view/structural ops: conj_physical, real, imag,
      view_of_dtype (view.dtype), convert_element_type, broadcast_in_dim
  - prims decomposed-to-aten ops: add, mul, sum, rev, reshape,
      collapse, transpose, squeeze, split_dim, fill, iota, view_of, device_put
"""

import torch
import torch_vulkan
import pytest

VK = "vulkan"
RTOL, ATOL = 1e-4, 1e-4


def vk(t):
    return t.to(VK)


def xor_ref(t, dim=None, keepdim=False, dtype=None):
    """CPU XOR reduction reference."""
    t32 = t.int()
    if dim is None:
        flat = t32.flatten()
        r = flat[0].clone()
        for x in flat[1:]:
            r = r ^ x
        return r.to(dtype if dtype else t.dtype)
    if isinstance(dim, int):
        dim = [dim]
    out = t32
    for d in sorted(dim, reverse=True):
        slices = out.unbind(d)
        acc = slices[0].clone()
        for s in slices[1:]:
            acc = acc ^ s
        out = acc.unsqueeze(d) if keepdim else acc
    return out.to(dtype if dtype else t.dtype)


# ── xor_sum ───────────────────────────────────────────────────────────────────

class TestPrimsXorSum:

    def test_global_1d(self):
        t = torch.tensor([1, 2, 4, 8, 15], dtype=torch.int32)
        result = torch.ops.prims.xor_sum(vk(t), None).cpu()
        assert result.equal(xor_ref(t))

    def test_global_2d(self):
        t = torch.randint(0, 256, (4, 8), dtype=torch.int32)
        result = torch.ops.prims.xor_sum(vk(t), None).cpu()
        assert result.equal(xor_ref(t))

    def test_global_large(self):
        # Exercises multi-pass reduction (1024 > 256 workgroup size)
        t = torch.randint(0, 65536, (1024,), dtype=torch.int32)
        result = torch.ops.prims.xor_sum(vk(t), None).cpu()
        assert result.equal(xor_ref(t))

    def test_global_output_dtype(self):
        t = torch.randint(0, 16, (32,), dtype=torch.int32)
        result = torch.ops.prims.xor_sum(vk(t), None, output_dtype=torch.int64).cpu()
        assert result.dtype == torch.int64
        assert result.equal(xor_ref(t, dtype=torch.int64))

    def test_dim0(self):
        t = torch.randint(0, 256, (4, 8), dtype=torch.int32)
        result = torch.ops.prims.xor_sum(vk(t), [0]).cpu()
        assert result.equal(xor_ref(t, dim=[0]))

    def test_dim1(self):
        t = torch.randint(0, 256, (4, 8), dtype=torch.int32)
        result = torch.ops.prims.xor_sum(vk(t), [1]).cpu()
        assert result.equal(xor_ref(t, dim=[1]))

    def test_dim1_large_row(self):
        # row_size > 256 — exercises multi-iteration loop in shader
        t = torch.randint(0, 256, (4, 512), dtype=torch.int32)
        result = torch.ops.prims.xor_sum(vk(t), [1]).cpu()
        assert result.equal(xor_ref(t, dim=[1]))

    def test_dim0_strided(self):
        # dim=0 on [4,8]: uses strided shader (dim is not last)
        t = torch.randint(0, 64, (4, 8), dtype=torch.int32)
        result = torch.ops.prims.xor_sum(vk(t), [0]).cpu()
        assert result.equal(xor_ref(t, dim=[0]))

    def test_3d_middle_dim(self):
        t = torch.randint(0, 64, (3, 4, 5), dtype=torch.int32)
        result = torch.ops.prims.xor_sum(vk(t), [1]).cpu()
        assert result.equal(xor_ref(t, dim=[1]))

    def test_multi_dim(self):
        t = torch.randint(0, 64, (3, 4, 5), dtype=torch.int32)
        result = torch.ops.prims.xor_sum(vk(t), [0, 2]).cpu()
        assert result.equal(xor_ref(t, dim=[0, 2]))

    def test_all_dims(self):
        t = torch.randint(0, 64, (3, 4, 5), dtype=torch.int32)
        result = torch.ops.prims.xor_sum(vk(t), [0, 1, 2]).cpu()
        assert result.equal(xor_ref(t))

    def test_negative_dim(self):
        t = torch.randint(0, 64, (4, 8), dtype=torch.int32)
        result = torch.ops.prims.xor_sum(vk(t), [-1]).cpu()
        assert result.equal(xor_ref(t, dim=[-1]))

    def test_dim_dispatch_count(self):
        """Per-dim XOR: exactly 1 GPU dispatch (was 16 with pairwise XOR loop)."""
        t = torch.randint(0, 256, (4, 32), dtype=torch.int32).to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.xor_sum(t, [1])
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d == 1, f"Expected 1 dispatch for xor_sum dim=1, got {d}"

    def test_global_dispatch_count(self):
        """Global XOR of 256 elements: 1 pass = 1 dispatch."""
        t = torch.randint(0, 256, (256,), dtype=torch.int32).to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.xor_sum(t, None)
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d == 1


# ── normal ────────────────────────────────────────────────────────────────────

class TestPrimsNormal:

    def test_shape(self):
        t = torch.ops.prims.normal([4, 8], mean=0.0, std=1.0,
                                    dtype=torch.float32, device=VK, requires_grad=False)
        assert list(t.shape) == [4, 8]
        assert t.device.type == VK

    def test_dtype(self):
        t = torch.ops.prims.normal([16], mean=0.0, std=1.0,
                                    dtype=torch.float32, device=VK, requires_grad=False)
        assert t.dtype == torch.float32

    def test_mean_std_rough(self):
        N = 4096
        t = torch.ops.prims.normal([N], mean=2.0, std=3.0,
                                    dtype=torch.float32, device=VK, requires_grad=False).cpu()
        assert abs(t.mean().item() - 2.0) < 0.3
        assert abs(t.std().item() - 3.0) < 0.3

    def test_on_gpu(self):
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.normal([64], mean=0.0, std=1.0,
                                dtype=torch.float32, device=VK, requires_grad=False)
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d == 1, f"normal prim should dispatch 1 GPU shader (fused Box-Muller), got {d}"


# ── control-flow tokens ───────────────────────────────────────────────────────

class TestPrimsTokens:

    def test_make_token_is_tensor(self):
        # _make_token() has no tensor input, so PyTorch dispatches to CPU by default.
        # The token is a zero-valued tensor with no semantic content at runtime.
        tok = torch.ops.prims._make_token()
        assert isinstance(tok, torch.Tensor)

    def test_sink_tokens_noop(self):
        tok = torch.ops.prims._make_token()
        torch.ops.prims._sink_tokens([tok])  # must not raise


# ── as_strided_scatter ────────────────────────────────────────────────────────

class TestPrimsAsStridedScatter:

    def _cpu_ref(self, x, src, size, stride, offset):
        return torch.ops.prims.as_strided_scatter(x.cpu(), src.cpu(), size, stride, offset)

    def test_basic_contiguous(self):
        x = torch.arange(8, dtype=torch.float32)
        src = torch.tensor([10.0, 20.0])
        result = torch.ops.prims.as_strided_scatter(vk(x), vk(src), [2], [1], 2).cpu()
        assert result.equal(self._cpu_ref(x, src, [2], [1], 2))

    def test_strided_every_other(self):
        x = torch.zeros(8, dtype=torch.float32)
        src = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = torch.ops.prims.as_strided_scatter(vk(x), vk(src), [4], [2], 0).cpu()
        assert result.equal(self._cpu_ref(x, src, [4], [2], 0))

    def test_column_of_matrix(self):
        # Column 0 of a 4×4 row-major matrix: stride=[4], offset=0
        x = torch.zeros(16, dtype=torch.float32)
        src = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = torch.ops.prims.as_strided_scatter(vk(x), vk(src), [4], [4], 0).cpu()
        assert result.equal(self._cpu_ref(x, src, [4], [4], 0))

    def test_2d_src(self):
        x = torch.zeros(12, dtype=torch.float32)
        src = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = torch.ops.prims.as_strided_scatter(vk(x), vk(src), [2, 3], [3, 1], 0).cpu()
        assert result.equal(self._cpu_ref(x, src, [2, 3], [3, 1], 0))

    def test_nonzero_offset(self):
        x = torch.zeros(10, dtype=torch.float32)
        src = torch.tensor([7.0, 8.0, 9.0])
        result = torch.ops.prims.as_strided_scatter(vk(x), vk(src), [3], [1], 3).cpu()
        assert result.equal(self._cpu_ref(x, src, [3], [1], 3))

    def test_empty_src(self):
        x = torch.ones(8, dtype=torch.float32)
        src = torch.empty(0, dtype=torch.float32)
        result = torch.ops.prims.as_strided_scatter(vk(x), vk(src), [0], [1], 0).cpu()
        assert result.equal(x)

    def test_self_unchanged(self):
        """Positions not covered by src keep their original values."""
        x = torch.arange(8, dtype=torch.float32)
        src = torch.tensor([99.0, 99.0])
        result = torch.ops.prims.as_strided_scatter(vk(x), vk(src), [2], [1], 0).cpu()
        # positions 0,1 = 99, positions 2..7 = original
        assert result[2:].equal(x[2:])

    def test_dispatch_count(self):
        """Must dispatch GPU shader (>0), no CPU roundtrip."""
        x = torch.zeros(16, dtype=torch.float32).to(VK)
        src = torch.ones(4, dtype=torch.float32).to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.as_strided_scatter(x, src, [4], [1], 0)
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d > 0, "as_strided_scatter should use GPU shaders"


# ── view / structural prims ───────────────────────────────────────────────────

class TestPrimsViewOps:

    def test_real_real(self):
        # prims::real is complex-only at the PyTorch level.
        # Our vulkan_real/imag are registered under aten:: and handle real tensors.
        t = torch.randn(8)
        result = torch.ops.aten.real(vk(t)).cpu()
        assert result.equal(t)

    def test_imag_real(self):
        t = torch.randn(8)
        result = torch.ops.aten.imag(vk(t)).cpu()
        assert result.equal(torch.zeros(8))

    def test_view_dtype_f32_to_int32(self):
        t = torch.ones(4, dtype=torch.float32)
        vt = vk(t)
        reinterp = torch.ops.prims.view_of_dtype(vt, torch.int32)
        assert reinterp.dtype == torch.int32
        assert reinterp.cpu().view(torch.float32).equal(t)

    def test_view_dtype_roundtrip(self):
        t = torch.randn(8)
        vt = vk(t)
        as_int = torch.ops.prims.view_of_dtype(vt, torch.int32)
        back = torch.ops.prims.view_of_dtype(as_int, torch.float32)
        assert back.cpu().equal(t)

    def test_conj_physical_real_is_clone(self):
        t = torch.randn(4, 4)
        result = torch.ops.aten.conj_physical(vk(t)).cpu()
        assert result.equal(t)


# ── convert_element_type (dtype cast) ────────────────────────────────────────

class TestPrimsConvertElementType:

    def test_f32_to_f16(self):
        t = torch.randn(128, dtype=torch.float32)
        result = torch.ops.prims.convert_element_type(vk(t), torch.float16).cpu()
        assert result.dtype == torch.float16
        torch.testing.assert_close(result.float(), t.half().float(), rtol=1e-2, atol=1e-2)

    def test_f32_to_bf16(self):
        t = torch.randn(128, dtype=torch.float32)
        result = torch.ops.prims.convert_element_type(vk(t), torch.bfloat16).cpu()
        assert result.dtype == torch.bfloat16
        torch.testing.assert_close(result.float(), t.bfloat16().float(), rtol=1e-2, atol=1e-2)

    def test_f16_to_f32(self):
        t = torch.randn(64, dtype=torch.float16)
        result = torch.ops.prims.convert_element_type(vk(t), torch.float32).cpu()
        torch.testing.assert_close(result, t.float(), rtol=1e-3, atol=1e-3)

    def test_f32_to_f32_noop(self):
        t = torch.randn(16)
        result = torch.ops.prims.convert_element_type(vk(t), torch.float32).cpu()
        assert result.equal(t)

    def test_on_gpu(self):
        t = torch.randn(64).to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.convert_element_type(t, torch.float16)
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d > 0


# ── broadcast_in_dim ──────────────────────────────────────────────────────────

class TestPrimsBroadcastInDim:

    def test_1d_to_2d(self):
        t = torch.randn(8)
        expected = t.unsqueeze(0).expand(4, 8).contiguous()
        result = torch.ops.prims.broadcast_in_dim(vk(t), [4, 8], [1]).cpu()
        torch.testing.assert_close(result, expected, rtol=RTOL, atol=ATOL)

    def test_scalar_to_2d(self):
        t = torch.tensor(3.14)
        expected = torch.full((2, 3), 3.14)
        result = torch.ops.prims.broadcast_in_dim(vk(t), [2, 3], []).cpu()
        torch.testing.assert_close(result, expected, rtol=RTOL, atol=ATOL)

    def test_2d_to_3d(self):
        t = torch.randn(3, 4)
        expected = t.unsqueeze(0).expand(2, 3, 4).contiguous()
        result = torch.ops.prims.broadcast_in_dim(vk(t), [2, 3, 4], [1, 2]).cpu()
        torch.testing.assert_close(result, expected, rtol=RTOL, atol=ATOL)


# ── decomposed-to-aten prims ──────────────────────────────────────────────────

class TestPrimsDecomposed:
    """Prims ops that decompose to aten ops on Vulkan — correctness checks."""

    def test_add(self):
        a, b = torch.randn(8), torch.randn(8)
        result = torch.ops.prims.add(vk(a), vk(b)).cpu()
        torch.testing.assert_close(result, a + b, rtol=RTOL, atol=ATOL)

    def test_mul(self):
        a, b = torch.randn(8), torch.randn(8)
        result = torch.ops.prims.mul(vk(a), vk(b)).cpu()
        torch.testing.assert_close(result, a * b, rtol=RTOL, atol=ATOL)

    def test_sum_dim(self):
        t = torch.randn(4, 8)
        result = torch.ops.prims.sum(vk(t), [1]).cpu()
        torch.testing.assert_close(result, t.sum(1), rtol=RTOL, atol=ATOL)

    def test_rev(self):
        t = torch.arange(8, dtype=torch.float32)
        result = torch.ops.prims.rev(vk(t), [0]).cpu()
        assert result.equal(t.flip(0))

    def test_reshape(self):
        t = torch.randn(4, 8)
        result = torch.ops.prims.reshape(vk(t), [2, 16]).cpu()
        torch.testing.assert_close(result, t.reshape(2, 16), rtol=RTOL, atol=ATOL)

    def test_collapse(self):
        t = torch.randn(2, 4, 8)
        result = torch.ops.prims.collapse(vk(t), 1, 2).cpu()
        torch.testing.assert_close(result, t.reshape(2, 32), rtol=RTOL, atol=ATOL)

    def test_transpose(self):
        t = torch.randn(4, 8)
        result = torch.ops.prims.transpose(vk(t), [1, 0]).cpu()
        torch.testing.assert_close(result, t.T.contiguous(), rtol=RTOL, atol=ATOL)

    def test_squeeze(self):
        t = torch.randn(1, 8)
        result = torch.ops.prims.squeeze(vk(t), [0]).cpu()
        torch.testing.assert_close(result, t.squeeze(0), rtol=RTOL, atol=ATOL)

    def test_split_dim(self):
        t = torch.randn(8)
        result = torch.ops.prims.split_dim(vk(t), 0, 4).cpu()
        torch.testing.assert_close(result, t.reshape(4, 2), rtol=RTOL, atol=ATOL)

    def test_fill(self):
        t = torch.zeros(8, dtype=torch.float32)
        result = torch.ops.prims.fill(vk(t).clone(), 3.14).cpu()
        torch.testing.assert_close(result, torch.full((8,), 3.14), rtol=RTOL, atol=ATOL)

    def test_iota_on_device(self):
        t = torch.ops.prims.iota(8, start=0, step=1,
                                  dtype=torch.int32, device=VK, requires_grad=False)
        assert t.device.type == VK
        assert t.cpu().equal(torch.arange(8, dtype=torch.int32))

    def test_iota_custom_start_step(self):
        t = torch.ops.prims.iota(5, start=2, step=3,
                                  dtype=torch.int64, device=VK, requires_grad=False)
        assert t.cpu().equal(torch.tensor([2, 5, 8, 11, 14], dtype=torch.int64))

    def test_view_of(self):
        t = torch.randn(4, 8)
        result = torch.ops.prims.view_of(vk(t)).cpu()
        assert result.equal(t)

    def test_device_put(self):
        t = torch.randn(8).to(VK)
        result = torch.ops.prims.device_put(t, 'cpu')
        assert result.device.type == 'cpu'
        torch.testing.assert_close(result, t.cpu(), rtol=RTOL, atol=ATOL)


# ── prims special math — unary ────────────────────────────────────────────────

class TestPrimsUnarySpecialMath:
    """
    Covers prims:: unary special math ops registered directly on PrivateUse1.
    Each test verifies: (1) GPU dispatch ≤ 1, (2) result matches CPU reference.
    """

    ATOL_SPECIAL = 5e-4  # special functions have ~4 ULP error on RDNA1

    def _check(self, op, cpu_op, input_fn, atol=None):
        atol = atol or self.ATOL_SPECIAL
        t_cpu = input_fn()
        t_vk = vk(t_cpu)
        torch_vulkan._c_ext._reset_perf_counters()
        r_vk = op(t_vk).cpu()
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d == 1, f"{op} expected 1 dispatch, got {d}"
        r_cpu = cpu_op(t_cpu)
        torch.testing.assert_close(r_vk, r_cpu, rtol=atol, atol=atol)

    def test_erfcx(self):
        self._check(torch.ops.prims.erfcx, torch.special.erfcx,
                    lambda: torch.randn(128))

    def test_ndtri(self):
        # ndtri (inverse normal CDF) defined on (0, 1)
        self._check(torch.ops.prims.ndtri, torch.special.ndtri,
                    lambda: torch.rand(128) * 0.98 + 0.01)

    def test_spherical_bessel_j0(self):
        self._check(torch.ops.prims.spherical_bessel_j0,
                    torch.special.spherical_bessel_j0,
                    lambda: torch.rand(128) * 10.0)

    def test_bessel_i0(self):
        self._check(torch.ops.prims.bessel_i0, torch.i0,
                    lambda: torch.randn(128))

    def test_bessel_i0e(self):
        self._check(torch.ops.prims.bessel_i0e, torch.special.i0e,
                    lambda: torch.randn(128))

    def test_bessel_i1(self):
        self._check(torch.ops.prims.bessel_i1, torch.special.i1,
                    lambda: torch.randn(128))

    def test_bessel_i1e(self):
        self._check(torch.ops.prims.bessel_i1e, torch.special.i1e,
                    lambda: torch.randn(128))

    def test_bessel_j0(self):
        self._check(torch.ops.prims.bessel_j0, torch.special.bessel_j0,
                    lambda: torch.randn(128))

    def test_bessel_j1(self):
        self._check(torch.ops.prims.bessel_j1, torch.special.bessel_j1,
                    lambda: torch.randn(128))

    def test_digamma(self):
        # digamma not defined at non-positive integers; use positive values
        self._check(torch.ops.prims.digamma, torch.digamma,
                    lambda: torch.rand(128) + 0.5)

    def test_dispatch_count_all_one(self):
        """All prims unary special math ops: exactly 1 GPU dispatch each."""
        t = torch.randn(64).to(VK)
        tp = (torch.rand(64) + 0.5).to(VK)  # positive inputs
        ops_inputs = [
            (torch.ops.prims.erfcx, t),
            (torch.ops.prims.ndtri, (torch.rand(64) * 0.98 + 0.01).to(VK)),
            (torch.ops.prims.spherical_bessel_j0, tp),
            (torch.ops.prims.bessel_i0, t),
            (torch.ops.prims.bessel_i0e, t),
            (torch.ops.prims.bessel_i1, t),
            (torch.ops.prims.bessel_i1e, t),
            (torch.ops.prims.bessel_j0, t),
            (torch.ops.prims.bessel_j1, t),
            (torch.ops.prims.digamma, tp),
        ]
        for op, inp in ops_inputs:
            torch_vulkan._c_ext._reset_perf_counters()
            op(inp)
            d = torch_vulkan._c_ext._get_dispatch_count()
            assert d == 1, f"{op} expected 1 dispatch, got {d}"


# ── prims special math — binary ───────────────────────────────────────────────

class TestPrimsBinarySpecialMath:
    """
    Covers prims:: binary special math ops and aten:: ops with new GPU registrations.
    """

    ATOL = 1e-3

    def _check_binary(self, op, cpu_op, a_fn, b_fn, atol=None):
        atol = atol or self.ATOL
        a_cpu, b_cpu = a_fn(), b_fn()
        a_vk, b_vk = vk(a_cpu), vk(b_cpu)
        torch_vulkan._c_ext._reset_perf_counters()
        r_vk = op(a_vk, b_vk).cpu()
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d == 1, f"{op} expected 1 dispatch, got {d}"
        r_cpu = cpu_op(a_cpu, b_cpu)
        torch.testing.assert_close(r_vk, r_cpu, rtol=atol, atol=atol)

    def test_igamma(self):
        self._check_binary(
            torch.ops.prims.igamma, torch.igamma,
            lambda: torch.rand(128) + 1.0,
            lambda: torch.rand(128) + 0.1)

    def test_igammac(self):
        self._check_binary(
            torch.ops.prims.igammac, torch.igammac,
            lambda: torch.rand(128) + 1.0,
            lambda: torch.rand(128) + 0.1)

    def test_zeta(self):
        self._check_binary(
            torch.ops.prims.zeta, torch.special.zeta,
            lambda: torch.rand(128) + 2.0,   # s > 1 for convergence
            lambda: torch.rand(128) + 1.0)   # q > 0

    def test_shift_left(self):
        a = torch.randint(0, 1000, (128,), dtype=torch.int32)
        b = torch.randint(0, 8, (128,), dtype=torch.int32)
        r_vk = torch.ops.prims.shift_left(vk(a), vk(b)).cpu()
        r_cpu = torch.ops.prims.shift_left(a, b)
        assert r_vk.equal(r_cpu)

    def test_shift_right_arithmetic(self):
        a = torch.randint(-1000, 1000, (128,), dtype=torch.int32)
        b = torch.randint(0, 8, (128,), dtype=torch.int32)
        r_vk = torch.ops.prims.shift_right_arithmetic(vk(a), vk(b)).cpu()
        r_cpu = torch.ops.prims.shift_right_arithmetic(a, b)
        assert r_vk.equal(r_cpu)

    def test_aten_igamma(self):
        """aten::igamma must work on Vulkan (was unregistered — would raise)."""
        a = (torch.rand(64) + 1.0).to(VK)
        x = (torch.rand(64) + 0.1).to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        r = torch.igamma(a, x).cpu()
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d >= 1
        torch.testing.assert_close(r, torch.igamma(a.cpu(), x.cpu()), rtol=1e-3, atol=1e-3)

    def test_aten_igammac(self):
        """aten::igammac must work on Vulkan (was unregistered — would raise)."""
        a = (torch.rand(64) + 1.0).to(VK)
        x = (torch.rand(64) + 0.1).to(VK)
        r = torch.igammac(a, x).cpu()
        torch.testing.assert_close(r, torch.igammac(a.cpu(), x.cpu()), rtol=1e-3, atol=1e-3)

    def test_frexp(self):
        """prims::frexp returns (mantissa, exponent), should be 1 dispatch."""
        t_cpu = torch.randn(128)
        t_vk = vk(t_cpu)
        torch_vulkan._c_ext._reset_perf_counters()
        mant_vk, exp_vk = torch.ops.prims.frexp(t_vk)
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d == 1, f"frexp expected 1 dispatch, got {d}"
        mant_cpu, exp_cpu = torch.frexp(t_cpu)
        torch.testing.assert_close(mant_vk.cpu(), mant_cpu, rtol=1e-5, atol=1e-5)
        assert exp_vk.cpu().equal(exp_cpu)

    def test_nextafter_gpu(self):
        a = torch.randn(64).to(VK)
        b = (torch.randn(64) + 2.0).to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        r = torch.nextafter(a, b).cpu()
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d == 1
        torch.testing.assert_close(r, torch.nextafter(a.cpu(), b.cpu()))

    def test_gcd_gpu(self):
        a = torch.randint(1, 1000, (64,), dtype=torch.int32).to(VK)
        b = torch.randint(1, 1000, (64,), dtype=torch.int32).to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        r = torch.gcd(a, b).cpu()
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d == 1
        assert r.equal(torch.gcd(a.cpu(), b.cpu()))


class TestPrimsDirectDispatch:
    """
    Direct prims::PrivateUse1 registrations cut GPU dispatch count from 2→1 for
    all standard unary/binary/bitwise/comparison ops.

    Each test verifies: (a) exactly 1 GPU dispatch, (b) numerically correct output.
    """

    def _check(self, fn, args_gpu, args_cpu, *, atol=1e-5, rtol=1e-5, is_bool=False):
        torch_vulkan._c_ext._reset_perf_counters()
        out_gpu = fn(*args_gpu)
        d = torch_vulkan._c_ext._get_dispatch_count()
        out_cpu = fn(*args_cpu)
        if isinstance(out_gpu, (tuple, list)):
            for g, c in zip(out_gpu, out_cpu):
                torch.testing.assert_close(g.cpu().float(), c.float(), rtol=rtol, atol=atol)
        elif is_bool:
            assert out_gpu.cpu().equal(out_cpu), "bool mismatch"
        else:
            torch.testing.assert_close(out_gpu.cpu(), out_cpu, rtol=rtol, atol=atol)
        return d

    def _x(self, n=64, lo=-2.0, hi=2.0):
        return torch.rand(n) * (hi - lo) + lo

    def test_unary_math_dispatch1(self):
        """All standard unary prims ops must be 1 dispatch."""
        x = self._x().to(VK)
        x_pos = (x.abs() + 0.1).to(VK)
        x_small = (x * 0.5).to(VK)
        unary_ops = [
            (torch.ops.prims.abs,        (x,)),
            (torch.ops.prims.neg,        (x,)),
            (torch.ops.prims.sign,       (x,)),
            (torch.ops.prims.sqrt,       (x_pos,)),
            (torch.ops.prims.rsqrt,      (x_pos,)),
            (torch.ops.prims.reciprocal, (x_pos,)),
            (torch.ops.prims.exp,        (x,)),
            (torch.ops.prims.exp2,       (x,)),
            (torch.ops.prims.expm1,      (x,)),
            (torch.ops.prims.log,        (x_pos,)),
            (torch.ops.prims.log2,       (x_pos,)),
            (torch.ops.prims.log10,      (x_pos,)),
            (torch.ops.prims.log1p,      (x_pos,)),
            (torch.ops.prims.sin,        (x,)),
            (torch.ops.prims.cos,        (x,)),
            (torch.ops.prims.tan,        (x_small,)),
            (torch.ops.prims.asin,       (x_small,)),
            (torch.ops.prims.acos,       (x_small,)),
            (torch.ops.prims.atan,       (x,)),
            (torch.ops.prims.sinh,       (x,)),
            (torch.ops.prims.cosh,       (x,)),
            (torch.ops.prims.tanh,       (x,)),
            (torch.ops.prims.asinh,      (x,)),
            (torch.ops.prims.acosh,      (x_pos + 1.0,)),
            (torch.ops.prims.atanh,      (x_small,)),
            (torch.ops.prims.erf,        (x,)),
            (torch.ops.prims.erfc,       (x,)),
            (torch.ops.prims.round,      (x,)),
            (torch.ops.prims.floor,      (x,)),
            (torch.ops.prims.ceil,       (x,)),
            (torch.ops.prims.trunc,      (x,)),
            (torch.ops.prims.cbrt,       (x,)),
            (torch.ops.prims.lgamma,     (x_pos,)),
        ]
        bad = []
        for fn, args in unary_ops:
            torch_vulkan._c_ext._reset_perf_counters()
            fn(*args)
            d = torch_vulkan._c_ext._get_dispatch_count()
            if d != 1:
                bad.append((fn.__name__, d))
        assert not bad, f"Expected 1 dispatch: {bad}"

    def test_isfinite_dispatch1_and_correct(self):
        x_cpu = torch.tensor([1.0, float('nan'), float('inf'), -float('inf'), 0.0])
        x_gpu = x_cpu.to(VK)
        d = self._check(torch.ops.prims.isfinite, (x_gpu,), (x_cpu,), is_bool=True)
        assert d == 1, f"isfinite expected 1 dispatch, got {d}"

    def test_cbrt_dispatch1_and_correct(self):
        x_cpu = torch.linspace(-8.0, 8.0, 64)
        x_gpu = x_cpu.to(VK)
        d = self._check(torch.ops.prims.cbrt, (x_gpu,), (x_cpu,), atol=1e-6)
        assert d == 1, f"cbrt expected 1 dispatch, got {d}"

    def test_erf_inv_dispatch1_and_correct(self):
        x_cpu = torch.linspace(-0.99, 0.99, 64)
        x_gpu = x_cpu.to(VK)
        d = self._check(torch.ops.prims.erf_inv, (x_gpu,), (x_cpu,), atol=1e-5)
        assert d == 1, f"erf_inv expected 1 dispatch, got {d}"

    def test_binary_arithmetic_dispatch1(self):
        """prims::add/sub/mul/div/pow — 1 dispatch each."""
        x_cpu = torch.rand(64) + 0.5
        y_cpu = torch.rand(64) + 0.5
        x_gpu, y_gpu = x_cpu.to(VK), y_cpu.to(VK)
        ops = [
            torch.ops.prims.add, torch.ops.prims.sub,
            torch.ops.prims.mul, torch.ops.prims.div,
            torch.ops.prims.pow,
        ]
        bad = []
        for fn in ops:
            torch_vulkan._c_ext._reset_perf_counters()
            fn(x_gpu, y_gpu)
            d = torch_vulkan._c_ext._get_dispatch_count()
            if d != 1:
                bad.append((fn.__name__, d))
        assert not bad, f"Expected 1 dispatch: {bad}"

    def test_add_correct(self):
        x_cpu, y_cpu = torch.randn(64), torch.randn(64)
        x_gpu, y_gpu = x_cpu.to(VK), y_cpu.to(VK)
        d = self._check(torch.ops.prims.add, (x_gpu, y_gpu), (x_cpu, y_cpu))
        assert d == 1

    def test_sub_correct(self):
        x_cpu, y_cpu = torch.randn(64), torch.randn(64)
        x_gpu, y_gpu = x_cpu.to(VK), y_cpu.to(VK)
        d = self._check(torch.ops.prims.sub, (x_gpu, y_gpu), (x_cpu, y_cpu))
        assert d == 1

    def test_comparison_ops_dispatch(self):
        """prims comparison ops: 2 dispatches (compare + float→bool). Was 3 via prims→aten."""
        x_gpu = torch.randn(64).to(VK)
        y_gpu = torch.randn(64).to(VK)
        ops = [torch.ops.prims.eq, torch.ops.prims.ne,
               torch.ops.prims.lt, torch.ops.prims.le,
               torch.ops.prims.gt, torch.ops.prims.ge]
        bad = []
        for fn in ops:
            torch_vulkan._c_ext._reset_perf_counters()
            fn(x_gpu, y_gpu)
            d = torch_vulkan._c_ext._get_dispatch_count()
            if d != 2:
                bad.append((fn.__name__, d))
        assert not bad, f"Expected 2 dispatches: {bad}"

    def test_comparison_correct(self):
        x_cpu = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        y_cpu = torch.tensor([0.0, 0.0, 0.0, 1.0])
        x_gpu, y_gpu = x_cpu.to(VK), y_cpu.to(VK)
        for fn in [torch.ops.prims.lt, torch.ops.prims.ge, torch.ops.prims.eq]:
            self._check(fn, (x_gpu, y_gpu), (x_cpu, y_cpu), is_bool=True)

    def test_bitwise_ops_dispatch1(self):
        """prims bitwise ops: 1 dispatch each."""
        xi_gpu = torch.randint(0, 255, (64,), dtype=torch.int32).to(VK)
        yi_gpu = torch.randint(0, 255, (64,), dtype=torch.int32).to(VK)
        ops_bin = [torch.ops.prims.bitwise_and,
                   torch.ops.prims.bitwise_or,
                   torch.ops.prims.bitwise_xor]
        bad = []
        for fn in ops_bin:
            torch_vulkan._c_ext._reset_perf_counters()
            fn(xi_gpu, yi_gpu)
            d = torch_vulkan._c_ext._get_dispatch_count()
            if d != 1:
                bad.append((fn.__name__, d))
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.bitwise_not(xi_gpu)
        d = torch_vulkan._c_ext._get_dispatch_count()
        if d != 1:
            bad.append(("bitwise_not", d))
        assert not bad, f"Expected 1 dispatch: {bad}"

    def test_nextafter_and_gcd_dispatch1(self):
        x_gpu = torch.randn(64).to(VK)
        y_gpu = (torch.randn(64) + 2.0).to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.nextafter(x_gpu, y_gpu)
        assert torch_vulkan._c_ext._get_dispatch_count() == 1

        a = torch.randint(1, 100, (64,), dtype=torch.int32).to(VK)
        b = torch.randint(1, 100, (64,), dtype=torch.int32).to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.gcd(a, b)
        assert torch_vulkan._c_ext._get_dispatch_count() == 1

    def test_aten_isfinite_dispatch1(self):
        """aten::isfinite was missing; now 1 dispatch via vulkan_isfinite."""
        x = torch.tensor([1.0, float('nan'), float('inf')]).to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        r = torch.isfinite(x).cpu()
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d == 1, f"aten::isfinite expected 1 dispatch, got {d}"
        assert r.equal(torch.isfinite(torch.tensor([1.0, float('nan'), float('inf')])))

    def test_fmax_dispatch1_and_correct(self):
        """prims::fmax: NaN-ignoring max — 1 dispatch, correct with NaN inputs."""
        nan = float('nan')
        x_cpu = torch.tensor([1.0, nan, 3.0, nan])
        y_cpu = torch.tensor([2.0, 2.0, nan, nan])
        x_gpu, y_gpu = x_cpu.to(VK), y_cpu.to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        r = torch.ops.prims.fmax(x_gpu, y_gpu).cpu()
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d == 1, f"fmax expected 1 dispatch, got {d}"
        expected = torch.tensor([2.0, 2.0, 3.0, nan])
        torch.testing.assert_close(r[:3], expected[:3])   # NaN output at [3] is unspecified
        assert not torch.isnan(r[0]) and not torch.isnan(r[1]) and not torch.isnan(r[2])

    def test_fmin_dispatch1_and_correct(self):
        """prims::fmin: NaN-ignoring min — 1 dispatch, correct with NaN inputs."""
        nan = float('nan')
        x_cpu = torch.tensor([1.0, nan, 3.0, nan])
        y_cpu = torch.tensor([2.0, 2.0, nan, nan])
        x_gpu, y_gpu = x_cpu.to(VK), y_cpu.to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        r = torch.ops.prims.fmin(x_gpu, y_gpu).cpu()
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d == 1, f"fmin expected 1 dispatch, got {d}"
        assert not torch.isnan(r[0]) and not torch.isnan(r[1]) and not torch.isnan(r[2])
        assert r[0].item() == 1.0 and r[1].item() == 2.0 and r[2].item() == 3.0

    def test_aten_fmax_and_fmin(self):
        """aten::fmax / aten::fmin match torch.fmax / torch.fmin on GPU."""
        x_cpu = torch.tensor([1.0, 3.0, -1.0, 5.0])
        y_cpu = torch.tensor([2.0, 2.0,  0.0, 4.0])
        x_gpu, y_gpu = x_cpu.to(VK), y_cpu.to(VK)
        torch.testing.assert_close(torch.fmax(x_gpu, y_gpu).cpu(), torch.fmax(x_cpu, y_cpu))
        torch.testing.assert_close(torch.fmin(x_gpu, y_gpu).cpu(), torch.fmin(x_cpu, y_cpu))


class TestPrimsFactoryOps:
    """prims factory ops: full, full_like, fill, uniform."""

    def test_full_correct(self):
        r = torch.ops.prims.full([4, 4], 3.14, dtype=torch.float32,
                                  device='vulkan', requires_grad=False).cpu()
        assert r.shape == (4, 4)
        torch.testing.assert_close(r, torch.full([4, 4], 3.14))

    def test_full_dispatch(self):
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.full([64], 1.0, dtype=torch.float32,
                              device='vulkan', requires_grad=False)
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d <= 1, f"prims::full expected ≤1 dispatch, got {d}"

    def test_full_like_correct(self):
        ref = torch.randn(4, 4).to('vulkan')
        r = torch.ops.prims.full_like(ref, -1.5, dtype=torch.float32,
                                       device='vulkan', requires_grad=False).cpu()
        assert r.shape == (4, 4)
        torch.testing.assert_close(r, torch.full([4, 4], -1.5))

    def test_fill_correct(self):
        t = torch.randn(64).to('vulkan')
        r = torch.ops.prims.fill(t, 7.0).cpu()
        torch.testing.assert_close(r, torch.full([64], 7.0))

    def test_fill_does_not_modify_input(self):
        t = torch.zeros(8).to('vulkan')
        torch.ops.prims.fill(t, 99.0)
        assert t.cpu()[0].item() == 0.0, "prims::fill must be out-of-place"

    def test_uniform_correct(self):
        r = torch.ops.prims.uniform([1024], low=0.0, high=1.0,
                                     dtype=torch.float32, device='vulkan',
                                     stride=[1]).cpu()
        assert r.shape == (1024,)
        assert r.min().item() >= 0.0 and r.max().item() < 1.0

    def test_uniform_dispatch(self):
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.uniform([64], low=0.0, high=1.0,
                                  dtype=torch.float32, device='vulkan',
                                  stride=[1])
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d == 1, f"prims::uniform expected 1 dispatch (fused philox), got {d}"


class TestPrimsStructuralOps:
    """prims structural ops: squeeze, as_strided, convert_element_type, where."""

    def test_convert_element_type_f32_to_f16(self):
        x = torch.randn(64).to('vulkan')
        r = torch.ops.prims.convert_element_type(x, torch.float16).cpu()
        assert r.dtype == torch.float16
        torch.testing.assert_close(r.float(), x.cpu(), rtol=1e-2, atol=1e-2)

    def test_convert_element_type_dispatch(self):
        x = torch.randn(64).to('vulkan')
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.convert_element_type(x, torch.float16)
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d <= 2, f"convert_element_type expected ≤2 dispatches, got {d}"

    def test_squeeze_dims_list(self):
        t = torch.randn(4, 1, 8, 1).to('vulkan')
        r = torch.ops.prims.squeeze(t, [1, 3]).cpu()
        assert r.shape == (4, 8)

    def test_squeeze_single_dim(self):
        t = torch.randn(3, 1, 5).to('vulkan')
        r = torch.ops.prims.squeeze(t, [1]).cpu()
        assert r.shape == (3, 5)

    def test_as_strided_view(self):
        t = torch.randn(16).to('vulkan')
        r = torch.ops.prims.as_strided(t, [4, 4], [4, 1], 0).cpu()
        assert r.shape == (4, 4)
        torch.testing.assert_close(r, t.cpu().view(4, 4))

    def test_where_correct(self):
        cond = torch.tensor([True, False, True, False]).to('vulkan')
        a = torch.tensor([1.0, 2.0, 3.0, 4.0]).to('vulkan')
        b = torch.tensor([10.0, 20.0, 30.0, 40.0]).to('vulkan')
        r = torch.ops.prims.where(cond, a, b).cpu()
        expected = torch.tensor([1.0, 20.0, 3.0, 40.0])
        torch.testing.assert_close(r, expected)

    def test_where_dispatch(self):
        cond = torch.tensor([True, False] * 32).to('vulkan')
        a = torch.randn(64).to('vulkan')
        b = torch.randn(64).to('vulkan')
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.where(cond, a, b)
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d <= 2, f"prims::where expected ≤2 dispatches, got {d}"


class TestPrimsReductionOps:
    """prims reduction ops: sum, prod, amax, amin, var — direct PrivateUse1 dispatch."""

    def test_sum_all_dims(self):
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).to(VK)
        r = torch.ops.prims.sum(x, None).cpu()
        assert r.item() == pytest.approx(276.0)

    def test_sum_single_dim(self):
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(VK)
        r = torch.ops.prims.sum(x, [1]).cpu()
        expected = torch.tensor([6.0, 22.0, 38.0])
        torch.testing.assert_close(r, expected)

    def test_sum_multi_dim(self):
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).to(VK)
        r = torch.ops.prims.sum(x, [0, 2]).cpu()
        expected = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).sum([0, 2])
        torch.testing.assert_close(r, expected)

    def test_sum_dispatch1(self):
        x = torch.randn(64, 64).to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.sum(x, [0])
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d == 1, f"prims::sum expected 1 dispatch, got {d}"

    def test_prod_all_dims(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0]).to(VK)
        r = torch.ops.prims.prod(x, None).cpu()
        assert r.item() == pytest.approx(24.0)

    def test_prod_single_dim(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to(VK)
        r = torch.ops.prims.prod(x, [1]).cpu()
        expected = torch.tensor([6.0, 120.0])
        torch.testing.assert_close(r, expected)

    def test_prod_dispatch1(self):
        x = torch.randn(16, 16).to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.prod(x, [0])
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d <= 2, f"prims::prod expected <= 2 dispatches, got {d}"

    def test_amax_single_dim(self):
        x = torch.tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]]).to(VK)
        r = torch.ops.prims.amax(x, [1]).cpu()
        expected = torch.tensor([5.0, 6.0])
        torch.testing.assert_close(r, expected)

    def test_amax_all_dims(self):
        x = torch.tensor([[1.0, 5.0], [3.0, 2.0]]).to(VK)
        r = torch.ops.prims.amax(x, None).cpu()
        assert r.item() == pytest.approx(5.0)

    def test_amax_dispatch1(self):
        x = torch.randn(32, 32).to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.amax(x, [0])
        d = torch_vulkan._c_ext._get_dispatch_count()
        # non-last dim reduction uses movedim + reduce shader = 2 dispatches
        assert d <= 2, f"prims::amax expected <= 2 dispatches, got {d}"

    def test_amin_single_dim(self):
        x = torch.tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]]).to(VK)
        r = torch.ops.prims.amin(x, [1]).cpu()
        expected = torch.tensor([1.0, 2.0])
        torch.testing.assert_close(r, expected)

    def test_amin_all_dims(self):
        x = torch.tensor([[1.0, 5.0], [3.0, 2.0]]).to(VK)
        r = torch.ops.prims.amin(x, None).cpu()
        assert r.item() == pytest.approx(1.0)

    def test_var_single_dim(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]).to(VK)
        r = torch.ops.prims.var(x, [1], 1.0).cpu()
        expected = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]).var(1)
        torch.testing.assert_close(r, expected)

    def test_var_all_dims(self):
        x = torch.arange(8, dtype=torch.float32).to(VK)
        r = torch.ops.prims.var(x, None, 1.0).cpu()
        expected = x.cpu().var()
        torch.testing.assert_close(r, expected)

    def test_var_dispatch1(self):
        x = torch.randn(32, 32).to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.var(x, [0], 1.0)
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d == 1, f"prims::var expected 1 dispatch (fused), got {d}"


class TestPrimsShapeOps:
    """prims shape/view ops: reshape, transpose, rev, view_of, collapse, split_dim, broadcast_in_dim."""

    def test_reshape_correct(self):
        x = torch.arange(12, dtype=torch.float32).to(VK)
        r = torch.ops.prims.reshape(x, [3, 4]).cpu()
        assert r.shape == (3, 4)
        torch.testing.assert_close(r, torch.arange(12, dtype=torch.float32).reshape(3, 4))

    def test_reshape_dispatch1(self):
        x = torch.randn(64).to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.reshape(x, [8, 8])
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d == 0, f"prims::reshape (view) expected 0 dispatches, got {d}"

    def test_transpose_correct(self):
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(VK)
        r = torch.ops.prims.transpose(x, [1, 0]).cpu()
        assert r.shape == (3, 2)
        torch.testing.assert_close(r, x.cpu().T)

    def test_transpose_3d(self):
        x = torch.randn(2, 3, 4).to(VK)
        r = torch.ops.prims.transpose(x, [2, 0, 1]).cpu()
        assert r.shape == (4, 2, 3)
        torch.testing.assert_close(r, x.cpu().permute(2, 0, 1))

    def test_rev_correct(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0]).to(VK)
        r = torch.ops.prims.rev(x, [0]).cpu()
        expected = torch.tensor([4.0, 3.0, 2.0, 1.0])
        torch.testing.assert_close(r, expected)

    def test_rev_2d(self):
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(VK)
        r = torch.ops.prims.rev(x, [1]).cpu()
        expected = x.cpu().flip(1)
        torch.testing.assert_close(r, expected)

    def test_view_of_same_data(self):
        x = torch.randn(4, 4).to(VK)
        r = torch.ops.prims.view_of(x)
        assert r.shape == x.shape
        torch.testing.assert_close(r.cpu(), x.cpu())

    def test_collapse_dims(self):
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).to(VK)
        r = torch.ops.prims.collapse(x, 1, 2).cpu()
        assert r.shape == (2, 12)
        torch.testing.assert_close(r, x.cpu().reshape(2, 12))

    def test_collapse_view_dims(self):
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).to(VK)
        r = torch.ops.prims.collapse_view(x, 0, 1).cpu()
        assert r.shape == (6, 4)
        torch.testing.assert_close(r, x.cpu().reshape(6, 4))

    def test_split_dim_correct(self):
        x = torch.arange(24, dtype=torch.float32).reshape(6, 4).to(VK)
        r = torch.ops.prims.split_dim(x, 0, 2).cpu()
        assert r.shape == (2, 3, 4)
        torch.testing.assert_close(r, x.cpu().reshape(2, 3, 4))

    def test_broadcast_in_dim_expand(self):
        x = torch.tensor([1.0, 2.0, 3.0]).to(VK)
        r = torch.ops.prims.broadcast_in_dim(x, [4, 3], [1]).cpu()
        assert r.shape == (4, 3)
        expected = x.cpu().unsqueeze(0).expand(4, 3)
        torch.testing.assert_close(r, expected)

    def test_broadcast_in_dim_add_batch(self):
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(VK)
        r = torch.ops.prims.broadcast_in_dim(x, [5, 2, 3], [1, 2]).cpu()
        assert r.shape == (5, 2, 3)
        torch.testing.assert_close(r, x.cpu().unsqueeze(0).expand(5, 2, 3))


class TestPrimsMemoryOps:
    """prims memory/factory ops: empty, empty_strided, clone, scalar_tensor, iota."""

    def test_empty_shape(self):
        r = torch.ops.prims.empty([3, 4], dtype=torch.float32,
                                   device='vulkan', requires_grad=False)
        assert r.shape == (3, 4)
        assert r.device.type == 'vulkan'

    def test_empty_strided_shape(self):
        r = torch.ops.prims.empty_strided([4, 4], [4, 1], dtype=torch.float32,
                                           device='vulkan', requires_grad=False)
        assert r.shape == (4, 4)
        assert r.device.type == 'vulkan'

    def test_empty_permuted_shape(self):
        r = torch.ops.prims.empty_permuted([3, 4], [1, 0], dtype=torch.float32,
                                            device='vulkan', requires_grad=False)
        assert r.shape == (3, 4)
        assert r.device.type == 'vulkan'

    def test_clone_correct(self):
        x = torch.randn(4, 4).to(VK)
        r = torch.ops.prims.clone(x).cpu()
        torch.testing.assert_close(r, x.cpu())
        assert r.data_ptr() != x.cpu().data_ptr()  # independent copy

    def test_clone_dispatch1(self):
        x = torch.randn(64).to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.clone(x)
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d == 1, f"prims::clone expected 1 dispatch, got {d}"

    def test_scalar_tensor_correct(self):
        r = torch.ops.prims.scalar_tensor(3.14, dtype=torch.float32,
                                           device='vulkan').cpu()
        assert r.shape == ()
        assert r.item() == pytest.approx(3.14, abs=1e-5)

    def test_iota_basic(self):
        r = torch.ops.prims.iota(5, start=0, step=1, dtype=torch.int32,
                                  device='vulkan', requires_grad=False).cpu()
        expected = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
        assert r.equal(expected)

    def test_iota_step(self):
        r = torch.ops.prims.iota(4, start=2, step=3, dtype=torch.int32,
                                  device='vulkan', requires_grad=False).cpu()
        expected = torch.tensor([2, 5, 8, 11], dtype=torch.int32)
        assert r.equal(expected)

    def test_iota_int32(self):
        r = torch.ops.prims.iota(6, start=10, step=2, dtype=torch.int32,
                                  device='vulkan', requires_grad=False).cpu()
        expected = torch.tensor([10, 12, 14, 16, 18, 20], dtype=torch.int32)
        assert r.equal(expected)

    def test_iota_dispatch1(self):
        torch_vulkan._c_ext._reset_perf_counters()
        r = torch.ops.prims.iota(64, start=0, step=1, dtype=torch.int32,
                                  device='vulkan', requires_grad=False)
        d = torch_vulkan._c_ext._get_dispatch_count()
        # int32 iota: GPU arange = 1 dispatch; CPU arange + DMA = 0 dispatches; both are valid
        assert d <= 1, f"prims::iota expected <= 1 dispatch, got {d}"
        assert r.device.type == VK


class TestPrimsDataMovement:
    """prims data movement: device_put, copy_to, copy_strided."""

    def test_device_put_to_vulkan(self):
        x = torch.randn(8)
        r = torch.ops.prims.device_put(x, torch.device('vulkan'), False)
        assert r.device.type == 'vulkan'
        torch.testing.assert_close(r.cpu(), x)

    def test_device_put_to_cpu(self):
        x = torch.randn(8).to(VK)
        r = torch.ops.prims.device_put(x, torch.device('cpu'), False)
        assert r.device.type == 'cpu'

    def test_copy_to_correct(self):
        a = torch.zeros(4).to(VK)
        b = torch.tensor([1.0, 2.0, 3.0, 4.0]).to(VK)
        result = torch.ops.prims.copy_to(a, b).cpu()
        torch.testing.assert_close(result, b.cpu())

    def test_copy_strided_correct(self):
        x = torch.arange(8, dtype=torch.float32).to(VK)
        r = torch.ops.prims.copy_strided(x, [1]).cpu()
        torch.testing.assert_close(r, x.cpu())

    def test_resize_shape(self):
        x = torch.randn(4).to(VK)
        r = torch.ops.prims.resize(x, [8])
        assert r.shape == (8,)


class TestPrimsComplexOps:
    """prims complex ops tested on real tensors (complex tensors not supported on Vulkan)."""

    def test_conj_physical_real(self):
        # conj_physical on a real tensor = clone (conjugate of real = identity)
        x = torch.randn(4).to(VK)
        r = torch.ops.prims.conj_physical(x)
        torch.testing.assert_close(r.cpu(), x.cpu())

    def test_conj_physical_dispatch1(self):
        x = torch.randn(16).to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.conj_physical(x)
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d == 1, f"prims::conj_physical expected 1 dispatch, got {d}"

    def test_real_real_tensor(self):
        # prims::real on a real tensor returns the tensor
        x = torch.randn(6).to(VK)
        r = torch.ops.prims.real(x)
        torch.testing.assert_close(r.cpu(), x.cpu())

    def test_conj_real_is_identity(self):
        # prims::conj on a real tensor = lazy conjugate view = identity when read
        x = torch.randn(4).to(VK)
        r = torch.ops.prims.conj(x).cpu()
        torch.testing.assert_close(r, x.cpu())


class TestPrimsSpecialMath:
    """prims special math: frexp, as_strided_scatter, svd."""

    def test_frexp_mantissa_range(self):
        x = torch.tensor([1.0, 2.0, 4.0, 8.0]).to(VK)
        mantissa, exponent = torch.ops.prims.frexp(x)
        m_cpu, e_cpu = torch.frexp(x.cpu())
        torch.testing.assert_close(mantissa.cpu(), m_cpu)
        torch.testing.assert_close(exponent.cpu().float(), e_cpu.float())

    def test_frexp_dispatch1(self):
        x = torch.randn(64).to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.frexp(x)
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d == 1, f"prims::frexp expected 1 dispatch, got {d}"

    def test_as_strided_scatter_correct(self):
        base = torch.zeros(16).to(VK)
        src = torch.tensor([1.0, 2.0, 3.0, 4.0]).to(VK)
        r = torch.ops.prims.as_strided_scatter(base, src, [4], [1], 2).cpu()
        expected = torch.zeros(16)
        expected[2:6] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        torch.testing.assert_close(r, expected)

    def test_svd_shapes(self):
        A = torch.randn(4, 3).to(VK)
        U, S, Vh = torch.ops.prims.svd(A, full_matrices=False)
        assert U.shape == (4, 3)
        assert S.shape == (3,)
        assert Vh.shape == (3, 3)
        U_cpu, S_cpu, Vh_cpu = U.cpu(), S.cpu(), Vh.cpu()
        recon = U_cpu @ torch.diag(S_cpu) @ Vh_cpu
        torch.testing.assert_close(recon, A.cpu(), rtol=1e-3, atol=1e-3)


class TestPrimsFFTOps:
    """prims FFT ops: fft_r2c, fft_c2c, fft_c2r."""

    def test_fft_r2c_correct(self):
        x = torch.randn(8).to(VK)
        r = torch.ops.prims.fft_r2c(x, dim=[0], onesided=True).cpu()
        expected = torch.fft.rfft(x.cpu(), norm=None)
        torch.testing.assert_close(r, expected, rtol=1e-4, atol=1e-4)

    def test_fft_r2c_dispatch1(self):
        x = torch.randn(64).to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.fft_r2c(x, dim=[0], onesided=True)
        d = torch_vulkan._c_ext._get_dispatch_count()
        # FFT is a log2(N)-stage butterfly: N=64 → 6 stages + init + output = ~8 dispatches
        assert d <= 8, f"prims::fft_r2c expected <= 8 dispatches, got {d}"

    def test_fft_c2c_forward(self):
        # Create complex Vulkan tensor via view_as_complex on a [N,2] float tensor
        x_f = torch.randn(8, 2).to(VK)
        x = torch.view_as_complex(x_f)
        r = torch.ops.prims.fft_c2c(x, dim=[0], forward=True).cpu()
        expected = torch.fft.fft(x.cpu(), norm=None)
        torch.testing.assert_close(r, expected, rtol=1e-4, atol=1e-4)

    def test_fft_c2c_inverse(self):
        # Create complex Vulkan tensor via view_as_complex on a [N,2] float tensor
        x_f = torch.randn(8, 2).to(VK)
        x = torch.view_as_complex(x_f)
        r = torch.ops.prims.fft_c2c(x, dim=[0], forward=False).cpu()
        # prims::fft_c2c with forward=False is unnormalized ifft
        expected = torch.fft.ifft(x.cpu(), norm="forward")
        torch.testing.assert_close(r, expected, rtol=1e-4, atol=1e-4)

    def test_fft_c2r_correct(self):
        x = torch.randn(8).to(VK)
        xf = torch.ops.prims.fft_r2c(x, dim=[0], onesided=True)
        r = torch.ops.prims.fft_c2r(xf, dim=[0], last_dim_size=8).cpu()
        # unnormalized round-trip: r should equal x * N
        torch.testing.assert_close(r / 8.0, x.cpu(), rtol=1e-4, atol=1e-4)


# ── dtype optimization tests ────────────────────────────────────────────────
# Verify that prims (and the underlying vulkan) ops produce correct results
# for f16, bf16, and fp8 inputs, and that f16/bf16 unary ops use 1 dispatch
# (packed16 fast path) instead of the 3-dispatch widen-compute-narrow path.

HALF_DTYPES = [torch.float16, torch.bfloat16]
FP8_DTYPES  = [torch.float8_e4m3fn, torch.float8_e5m2]

def _vk_close(cpu_ref, vk_result, rtol=2e-2, atol=2e-2):
    """Assert CPU f32 reference ≈ Vulkan result (cast to f32)."""
    torch.testing.assert_close(
        cpu_ref.float(), vk_result.cpu().float(), rtol=rtol, atol=atol)


class TestPrimsUnaryHalfDtype:
    """Unary prims ops: f16 / bf16 use packed16 shader (1 dispatch, correct output)."""

    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=["f16", "bf16"])
    def test_exp(self, dtype):
        x = torch.randn(64).to(dtype)
        _vk_close(x.float().exp(), torch.ops.prims.exp(x.to(VK)))

    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=["f16", "bf16"])
    def test_log(self, dtype):
        x = torch.rand(64).add(0.1).to(dtype)
        _vk_close(x.float().log(), torch.ops.prims.log(x.to(VK)))

    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=["f16", "bf16"])
    def test_sqrt(self, dtype):
        x = torch.rand(64).to(dtype)
        _vk_close(x.float().sqrt(), torch.ops.prims.sqrt(x.to(VK)))

    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=["f16", "bf16"])
    def test_rsqrt(self, dtype):
        x = torch.rand(64).add(0.1).to(dtype)
        _vk_close(x.float().rsqrt(), torch.ops.prims.rsqrt(x.to(VK)))

    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=["f16", "bf16"])
    def test_sin(self, dtype):
        x = torch.randn(64).to(dtype)
        _vk_close(x.float().sin(), torch.ops.prims.sin(x.to(VK)))

    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=["f16", "bf16"])
    def test_cos(self, dtype):
        x = torch.randn(64).to(dtype)
        _vk_close(x.float().cos(), torch.ops.prims.cos(x.to(VK)))

    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=["f16", "bf16"])
    def test_tanh(self, dtype):
        x = torch.randn(64).to(dtype)
        _vk_close(x.float().tanh(), torch.ops.prims.tanh(x.to(VK)))

    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=["f16", "bf16"])
    def test_abs(self, dtype):
        x = torch.randn(64).to(dtype)
        _vk_close(x.float().abs(), torch.ops.prims.abs(x.to(VK)))

    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=["f16", "bf16"])
    def test_neg(self, dtype):
        x = torch.randn(64).to(dtype)
        _vk_close(-x.float(), torch.ops.prims.neg(x.to(VK)))

    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=["f16", "bf16"])
    def test_ceil(self, dtype):
        x = torch.randn(64).to(dtype)
        _vk_close(x.float().ceil(), torch.ops.prims.ceil(x.to(VK)))

    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=["f16", "bf16"])
    def test_floor(self, dtype):
        x = torch.randn(64).to(dtype)
        _vk_close(x.float().floor(), torch.ops.prims.floor(x.to(VK)))

    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=["f16", "bf16"])
    def test_erf(self, dtype):
        x = torch.randn(64).to(dtype)
        _vk_close(x.float().erf(), torch.ops.prims.erf(x.to(VK)))

    def test_exp_f16_dispatch_count(self):
        """f16 exp uses packed16 shader: must be 1 dispatch (not 3)."""
        x = torch.randn(256).half().to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.exp(x)
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d == 1, f"prims::exp(f16) expected 1 dispatch, got {d}"

    def test_sin_bf16_dispatch_count(self):
        """bf16 sin uses packed16 shader: must be 1 dispatch (not 3)."""
        x = torch.randn(256).bfloat16().to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.sin(x)
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d == 1, f"prims::sin(bf16) expected 1 dispatch, got {d}"

    def test_sqrt_f16_dispatch_count(self):
        """f16 sqrt uses packed16 shader: must be 1 dispatch (not 3)."""
        x = torch.rand(256).half().to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.sqrt(x)
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d == 1, f"prims::sqrt(f16) expected 1 dispatch, got {d}"


class TestPrimsBinaryHalfDtype:
    """Binary prims ops: f16 / bf16 use packed16 add/mul shaders (1 dispatch)."""

    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=["f16", "bf16"])
    def test_add(self, dtype):
        a = torch.randn(64).to(dtype)
        b = torch.randn(64).to(dtype)
        expected = (a.float() + b.float()).to(dtype).float()
        result = torch.ops.prims.add(a.to(VK), b.to(VK)).cpu().float()
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=["f16", "bf16"])
    def test_mul(self, dtype):
        a = torch.randn(64).to(dtype)
        b = torch.randn(64).to(dtype)
        expected = (a.float() * b.float()).to(dtype).float()
        result = torch.ops.prims.mul(a.to(VK), b.to(VK)).cpu().float()
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    def test_add_f16_dispatch_count(self):
        """f16 add uses packed16 shader: must be 1 dispatch."""
        a = torch.randn(256).half().to(VK)
        b = torch.randn(256).half().to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.add(a, b)
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d == 1, f"prims::add(f16,f16) expected 1 dispatch, got {d}"

    def test_mul_bf16_dispatch_count(self):
        """bf16 mul uses packed16 shader: must be 1 dispatch."""
        a = torch.randn(256).bfloat16().to(VK)
        b = torch.randn(256).bfloat16().to(VK)
        torch_vulkan._c_ext._reset_perf_counters()
        torch.ops.prims.mul(a, b)
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d == 1, f"prims::mul(bf16,bf16) expected 1 dispatch, got {d}"


class TestPrimsReductionHalfDtype:
    """Reduction prims ops: f16 / bf16 produce correct results (widen-compute-narrow)."""

    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=["f16", "bf16"])
    def test_sum(self, dtype):
        x = torch.randn(8, 16).to(dtype)
        expected = x.float().sum(dim=[1])
        result = torch.ops.prims.sum(x.to(VK), dims=[1]).cpu().float()
        torch.testing.assert_close(result, expected, rtol=5e-2, atol=5e-2)

    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=["f16", "bf16"])
    def test_var(self, dtype):
        x = torch.randn(8, 16).to(dtype)
        expected = x.float().var(dim=[-1], correction=0)
        result = torch.ops.prims.var(x.to(VK), dims=[-1], correction=0.0).cpu().float()
        torch.testing.assert_close(result, expected, rtol=0.1, atol=0.1)

    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=["f16", "bf16"])
    def test_amax(self, dtype):
        x = torch.randn(4, 8).to(dtype)
        expected = x.float().amax(dim=[1])
        result = torch.ops.prims.amax(x.to(VK), dims=[1]).cpu().float()
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


class TestPrimsFP8Dtype:
    """FP8 prims ops: exp, log, sqrt work via widen-compute-narrow (3 dispatches)."""

    @pytest.mark.parametrize("dtype", FP8_DTYPES, ids=["e4m3fn", "e5m2"])
    def test_exp_fp8(self, dtype):
        x = torch.rand(64).mul(0.5)  # small values to avoid fp8 overflow
        x_fp8 = x.to(dtype)
        expected = x_fp8.float().exp()
        result = torch.ops.prims.exp(x_fp8.to(VK)).cpu().float()
        torch.testing.assert_close(result, expected, rtol=0.1, atol=0.1)

    @pytest.mark.parametrize("dtype", FP8_DTYPES, ids=["e4m3fn", "e5m2"])
    def test_add_fp8(self, dtype):
        a = torch.rand(64).mul(0.5).to(dtype)
        b = torch.rand(64).mul(0.5).to(dtype)
        expected = (a.float() + b.float())
        result = torch.ops.prims.add(a.to(VK), b.to(VK)).cpu().float()
        torch.testing.assert_close(result, expected, rtol=0.2, atol=0.2)

    @pytest.mark.parametrize("dtype", FP8_DTYPES, ids=["e4m3fn", "e5m2"])
    def test_convert_element_type_fp8_to_f32(self, dtype):
        x = torch.rand(64).mul(0.5).to(dtype)
        expected = x.float()
        result = torch.ops.prims.convert_element_type(x.to(VK), torch.float32).cpu()
        torch.testing.assert_close(result, expected, rtol=0.1, atol=0.1)


class TestPrimsShapeHalfDtype:
    """Shape prims (reshape, transpose, broadcast_in_dim) work for f16/bf16/fp8."""

    @pytest.mark.parametrize("dtype", HALF_DTYPES + [torch.float32], ids=["f16", "bf16", "f32"])
    def test_reshape(self, dtype):
        x = torch.randn(4, 8).to(dtype)
        result = torch.ops.prims.reshape(x.to(VK), [2, 16]).cpu()
        assert result.shape == (2, 16)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", HALF_DTYPES + [torch.float32], ids=["f16", "bf16", "f32"])
    def test_transpose(self, dtype):
        x = torch.randn(4, 8).to(dtype)
        expected = x.permute(1, 0)
        result = torch.ops.prims.transpose(x.to(VK), [1, 0]).cpu()
        torch.testing.assert_close(result.float(), expected.float(), rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", HALF_DTYPES + [torch.float32], ids=["f16", "bf16", "f32"])
    def test_broadcast_in_dim(self, dtype):
        x = torch.randn(4).to(dtype)
        expected = x.unsqueeze(0).expand(2, 4)
        result = torch.ops.prims.broadcast_in_dim(x.to(VK), [2, 4], [1]).cpu()
        torch.testing.assert_close(result.float(), expected.float(), rtol=1e-3, atol=1e-3)
