"""Tests for FFT (r2c, c2c, c2r) and SVD correctness on the Vulkan backend."""

import torch
import pytest
import math
import struct


# ─── helpers ────────────────────────────────────────────────────────────────

def allclose(a, b, atol=1e-4, rtol=1e-4, label=""):
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    if not ok:
        max_err = (a - b).abs().max().item()
        assert ok, f"{label}: max_err={max_err:.3e}"


# ─── FFT r2c ─────────────────────────────────────────────────────────────────

class TestFFTR2C:
    """Real-to-complex 1-D FFT correctness."""

    def _check(self, N, batch_shape=(), norm=None):
        shape = batch_shape + (N,)
        x_cpu = torch.randn(*shape)
        x_vk  = x_cpu.to("vulkan")
        kw = {} if norm is None else {"norm": norm}
        ref = torch.fft.rfft(x_cpu, **kw)
        got = torch.fft.rfft(x_vk,  **kw).cpu()
        allclose(ref.real, got.real, label=f"r2c real N={N}")
        allclose(ref.imag, got.imag, label=f"r2c imag N={N}")

    def test_n4(self):       self._check(4)
    def test_n8(self):       self._check(8)
    def test_n16(self):      self._check(16)
    def test_n32(self):      self._check(32)
    def test_n64(self):      self._check(64)
    def test_n128(self):     self._check(128)
    def test_n256(self):     self._check(256)
    def test_n512(self):     self._check(512)
    def test_n1024(self):    self._check(1024)
    def test_batched(self):  self._check(64, batch_shape=(4,))
    def test_2d_batch(self): self._check(32, batch_shape=(3, 2))
    def test_norm_forward(self): self._check(64, norm="forward")
    def test_norm_ortho(self):   self._check(64, norm="ortho")
    def test_norm_backward(self):self._check(64, norm="backward")


# ─── FFT c2r ─────────────────────────────────────────────────────────────────

class TestFFTC2R:
    """Complex-to-real 1-D IFFT correctness."""

    def _check(self, N, batch_shape=()):
        shape = batch_shape + (N,)
        x_cpu = torch.randn(*shape)
        # CPU reference: rfft → irfft
        ref = torch.fft.irfft(torch.fft.rfft(x_cpu), n=N)
        # Vulkan path
        x_vk = x_cpu.to("vulkan")
        got = torch.fft.irfft(torch.fft.rfft(x_vk), n=N).cpu()
        allclose(ref, got, atol=1e-3, rtol=1e-3, label=f"c2r N={N}")

    def test_n4(self):      self._check(4)
    def test_n8(self):      self._check(8)
    def test_n16(self):     self._check(16)
    def test_n32(self):     self._check(32)
    def test_n64(self):     self._check(64)
    def test_n128(self):    self._check(128)
    def test_n256(self):    self._check(256)
    def test_n512(self):    self._check(512)
    def test_batched(self): self._check(64, batch_shape=(4,))

    def test_roundtrip(self):
        N, x = 128, torch.randn(128)
        recovered = torch.fft.irfft(torch.fft.rfft(x.to("vulkan")), n=N).cpu()
        allclose(x, recovered, atol=1e-4, rtol=1e-4, label="roundtrip")

    def test_roundtrip_batched(self):
        N, x = 64, torch.randn(3, 64)
        recovered = torch.fft.irfft(torch.fft.rfft(x.to("vulkan")), n=N).cpu()
        allclose(x, recovered, atol=1e-4, rtol=1e-4, label="roundtrip_batched")

    def test_norm_ortho(self):
        N, x = 64, torch.randn(64)
        ref = torch.fft.irfft(torch.fft.rfft(x, norm="ortho"), n=N, norm="ortho")
        got = torch.fft.irfft(torch.fft.rfft(x.to("vulkan"), norm="ortho"), n=N, norm="ortho").cpu()
        allclose(ref, got, atol=1e-3, label="c2r ortho")


# ─── FFT c2c ─────────────────────────────────────────────────────────────────

class TestFFTC2C:
    """Complex-to-complex 1-D FFT / IFFT correctness."""

    def _check(self, N, batch_shape=()):
        shape = batch_shape + (N,)
        x_cpu = torch.randn(*shape, dtype=torch.complex64)
        x_vk  = x_cpu.to("vulkan")
        ref = torch.fft.fft(x_cpu)
        got = torch.fft.fft(x_vk).cpu()
        allclose(ref.real, got.real, atol=1e-3, rtol=1e-3, label=f"c2c real N={N}")
        allclose(ref.imag, got.imag, atol=1e-3, rtol=1e-3, label=f"c2c imag N={N}")

    def test_n4(self):     self._check(4)
    def test_n16(self):    self._check(16)
    def test_n64(self):    self._check(64)
    def test_n256(self):   self._check(256)
    def test_batched(self): self._check(32, batch_shape=(4,))

    def test_roundtrip(self):
        N, x = 64, torch.randn(64, dtype=torch.complex64)
        rec = torch.fft.ifft(torch.fft.fft(x.to("vulkan"))).cpu()
        allclose(x.real, rec.real, atol=1e-3, label="c2c roundtrip real")
        allclose(x.imag, rec.imag, atol=1e-3, label="c2c roundtrip imag")

    def test_parseval(self):
        N, x = 64, torch.randn(64, dtype=torch.complex64)
        X = torch.fft.fft(x.to("vulkan")).cpu()
        lhs = (x.real**2 + x.imag**2).sum().item()
        rhs = ((X.real**2 + X.imag**2).sum() / N).item()
        assert abs(lhs - rhs) < 1e-2, f"Parseval mismatch: {lhs:.4f} vs {rhs:.4f}"


# ─── SVD ─────────────────────────────────────────────────────────────────────

class TestSVD:
    """One-sided Jacobi SVD correctness."""

    def _check(self, shape):
        M, N = shape[-2], shape[-1]
        if M < N or N > 32 or M > 256:
            pytest.skip(f"shape {shape} outside GPU Jacobi limits")

        A_cpu = torch.randn(*shape)
        A_vk  = A_cpu.to("vulkan")
        U, S, Vh   = torch.linalg.svd(A_vk, full_matrices=False)
        U, S, Vh   = U.cpu(), S.cpu(), Vh.cpu()
        S_ref      = torch.linalg.svdvals(A_cpu)

        allclose(S_ref, S, atol=1e-4, rtol=1e-4, label=f"svd values {shape}")

        A_rec = (U * S.unsqueeze(-2)) @ Vh
        allclose(A_cpu, A_rec, atol=1e-3, rtol=1e-3, label=f"svd reconstruction {shape}")

        I_u = U.mT @ U
        allclose(I_u, torch.eye(N), atol=1e-3, label=f"U orthonormal {shape}")

        I_v = Vh @ Vh.mT
        allclose(I_v, torch.eye(N), atol=1e-3, label=f"Vh orthonormal {shape}")

    def test_4x4(self):    self._check((4, 4))
    def test_8x4(self):    self._check((8, 4))
    def test_16x4(self):   self._check((16, 4))
    def test_32x8(self):   self._check((32, 8))
    def test_64x16(self):  self._check((64, 16))
    def test_128x32(self): self._check((128, 32))

    def test_rank1(self):
        u, v = torch.randn(10), torch.randn(4)
        A = u.unsqueeze(1) * v.unsqueeze(0)
        _, S, _ = torch.linalg.svd(A.to("vulkan"), full_matrices=False)
        S = S.cpu()
        expected = u.norm() * v.norm()
        assert abs(S[0].item() - expected.item()) < 1e-3, f"rank-1 S[0]={S[0]:.4f} expected={expected:.4f}"
        for k in range(1, 4):
            assert abs(S[k].item()) < 1e-3, f"rank-1 S[{k}]={S[k]:.4f} should be ~0"

    def test_batched(self):
        for b in [2, 4]:
            A_cpu = torch.randn(b, 16, 4)
            U, S, Vh = torch.linalg.svd(A_cpu.to("vulkan"), full_matrices=False)
            U, S, Vh = U.cpu(), S.cpu(), Vh.cpu()
            A_rec = (U * S.unsqueeze(-2)) @ Vh
            allclose(A_cpu, A_rec, atol=1e-3, rtol=1e-3, label=f"batched SVD b={b}")

    def test_values_only(self):
        A_cpu = torch.randn(32, 8)
        S_cpu = torch.linalg.svdvals(A_cpu)
        S_vk  = torch.linalg.svdvals(A_cpu.to("vulkan")).cpu()
        allclose(S_cpu, S_vk, atol=1e-4, rtol=1e-4, label="svdvals")


# ─── Structural view prims ────────────────────────────────────────────────────

class TestPrimsViewOps:
    """Structural view prims (conj, real, imag, view_dtype, as_strided_scatter)."""

    def test_conj_real(self):
        x = torch.randn(10).to("vulkan")
        assert torch.equal(x.conj().cpu(), x.cpu())

    def test_real_real(self):
        x = torch.randn(10).to("vulkan")
        assert torch.equal(torch.real(x).cpu(), x.cpu())

    def test_imag_real(self):
        x = torch.randn(10).to("vulkan")
        out = torch.imag(x).cpu()
        assert (out == 0).all()

    def test_view_dtype_roundtrip(self):
        x = torch.randn(8, dtype=torch.float32).to("vulkan")
        y = x.view(torch.int32).view(torch.float32)
        assert torch.equal(y.cpu(), x.cpu())

    def test_as_strided_scatter(self):
        """as_strided_scatter: set a sub-view back."""
        x_cpu = torch.randn(4, 4)
        src_cpu = torch.randn(2, 2)
        x_vk  = x_cpu.to("vulkan")
        src_vk = src_cpu.to("vulkan")
        # Write src into top-left 2×2 block
        result_vk  = torch.ops.aten.as_strided_scatter(x_vk, src_vk, [2, 2], [4, 1], 0)
        result_cpu = torch.ops.aten.as_strided_scatter(x_cpu, src_cpu, [2, 2], [4, 1], 0)
        allclose(result_cpu, result_vk.cpu(), atol=1e-6, label="as_strided_scatter")
