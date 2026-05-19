"""M-CV.3 — Dynamic-shape variable-batch coverage regression tests.

Five tests covering forward and backward parity vs. CPU oracle when
the batch dimension is symbolic (``mark_dynamic(x, 0)`` /
``dynamic=True``).  Together they form the M-CV.3 audit gate for the
five regions where dynamic-shape coverage was previously absent:

  1. softmax forward                — pointwise / reduce-broadcast fwd
  2. layer_norm forward             — composite normalization fwd
  3. matmul (3D batched) backward   — bmm gradient under symbolic B
  4. conv2d backward                — strided conv gradient under symbolic B
  5. pointwise+reduce chain fwd/bwd — broadcasted add + sum reduction

Each test compiles once with ``dynamic=True`` and then exercises the
compiled callable across several batch sizes.  Numerical parity is
asserted against an eager CPU oracle (atol/rtol ~ 1e-5).  Tests that
fail today are marked ``xfail(strict=True)`` so they will flip to PASS
the moment the underlying primitive ships, acting as a future
regression gate.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch._dynamo
import torch.nn.functional as F

import torch_vulkan  # noqa: F401  # registers the vulkan PrivateUse1 device


def _ensure_env() -> None:
    """Skip the suite when slangc is unavailable; enable dynamic shapes."""
    if not os.environ.get("SLANGC"):
        pytest.skip("SLANGC env var not set — slangc unavailable")
    os.environ.setdefault("TORCH_VULKAN_DYNAMIC_SHAPES", "1")


# --------------------------------------------------------------------------
# 1) softmax forward — pointwise + reduce-broadcast over symbolic B
# --------------------------------------------------------------------------
class TestMCV3SoftmaxDynamicBatch:
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "M-CV.3 gap: dynamic-shape softmax fwd hits slangc "
            "error[E30015] 'undefined identifier r0_indexnumel' in the "
            "generated reduction kernel — codegen drops the symbolic "
            "reduce-extent variable from the kernel body but still "
            "references it inside the store guard. Verified 2026-05-19."
        ),
    )
    @pytest.mark.parametrize("B", [1, 4, 16])
    def test_m_cv3_softmax_dynamic_batch_fwd(self, B: int) -> None:
        _ensure_env()
        torch.manual_seed(42)
        x_cpu = torch.randn(B, 128)
        x_vk = x_cpu.to("vulkan:0")
        torch._dynamo.mark_dynamic(x_vk, 0)

        def fn(x):
            return F.softmax(x, dim=-1)

        compiled = torch.compile(fn, backend="inductor", dynamic=True)
        out_vk = compiled(x_vk).cpu()
        out_cpu = fn(x_cpu)
        torch.testing.assert_close(out_vk, out_cpu, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------
# 2) LayerNorm forward — composite normalization under symbolic B
# --------------------------------------------------------------------------
class TestMCV3LayerNormDynamicBatch:
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "M-CV.3 gap: dynamic-shape LayerNorm fwd numerical mismatch "
            "vs. CPU oracle (100% elements diverge, max abs diff > 0.8 "
            "for the smallest batch). Suggests the per-row mean/variance "
            "reduction in the dynamic-shape norm kernel uses a stale or "
            "wrong row-extent (likely the symbolic feature-dim length). "
            "Verified 2026-05-19."
        ),
    )
    @pytest.mark.parametrize("B", [2, 8, 32])
    def test_m_cv3_layer_norm_dynamic_batch_fwd(self, B: int) -> None:
        _ensure_env()
        torch.manual_seed(42)
        ln_cpu = torch.nn.LayerNorm([128])
        ln_vk = torch.nn.LayerNorm([128]).to("vulkan:0")
        # Match weights so the oracle comparison is valid
        with torch.no_grad():
            ln_vk.weight.copy_(ln_cpu.weight.to("vulkan:0"))
            ln_vk.bias.copy_(ln_cpu.bias.to("vulkan:0"))

        x_cpu = torch.randn(B, 128)
        x_vk = x_cpu.to("vulkan:0")
        torch._dynamo.mark_dynamic(x_vk, 0)

        def fn_vk(x):
            return ln_vk(x)

        def fn_cpu(x):
            return ln_cpu(x)

        compiled = torch.compile(fn_vk, backend="inductor", dynamic=True)
        out_vk = compiled(x_vk).cpu()
        out_cpu = fn_cpu(x_cpu)
        torch.testing.assert_close(out_vk, out_cpu, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------
# 3) 3D matmul backward — bmm-like with symbolic batch
# --------------------------------------------------------------------------
class TestMCV3Matmul3DDynamicBatch:
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "M-CV.3 gap: dynamic-shape 3D matmul backward fails inductor "
            "compile with RuntimeError 'when unpacking SymInt, expected "
            "int but got s17' from c10::SymInt::expect_int() during "
            "at::_ops::expand::call() inside SumBackward0. The backward "
            "lowering for expand under a symbolic batch dim does not "
            "thread the SymInt through — it tries to coerce to int. "
            "Verified 2026-05-19."
        ),
    )
    @pytest.mark.parametrize("B", [1, 4, 16])
    def test_m_cv3_matmul_3d_dynamic_batch_bwd(self, B: int) -> None:
        _ensure_env()
        torch.manual_seed(42)
        M, K, N = 8, 16, 12

        a_cpu = torch.randn(B, M, K, requires_grad=True)
        b_cpu = torch.randn(K, N, requires_grad=True)
        a_vk = a_cpu.detach().to("vulkan:0").requires_grad_(True)
        b_vk = b_cpu.detach().to("vulkan:0").requires_grad_(True)
        torch._dynamo.mark_dynamic(a_vk, 0)

        def fn(a, b):
            return (a @ b).sum()

        compiled = torch.compile(fn, backend="inductor", dynamic=True)
        out_vk = compiled(a_vk, b_vk)
        out_cpu = fn(a_cpu, b_cpu)
        torch.testing.assert_close(
            out_vk.cpu(), out_cpu, atol=1e-4, rtol=1e-4
        )

        out_vk.backward()
        out_cpu.backward()

        torch.testing.assert_close(
            a_vk.grad.cpu(), a_cpu.grad, atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            b_vk.grad.cpu(), b_cpu.grad, atol=1e-4, rtol=1e-4
        )


# --------------------------------------------------------------------------
# 4) Conv2d backward — strided conv with symbolic batch
# --------------------------------------------------------------------------
class TestMCV3Conv2dDynamicBatch:
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "M-CV.3 gap: dynamic-shape Conv2d backward fails inductor "
            "compile with RuntimeError 'when unpacking SymInt, expected "
            "int but got s77' from c10::SymInt::expect_int() during "
            "at::_ops::expand::call() inside SumBackward0. Same "
            "underlying SymInt-in-expand bug that breaks the 3D matmul "
            "backward case; conv2d simply re-exercises it through the "
            "convolution backward chain. Verified 2026-05-19."
        ),
    )
    @pytest.mark.parametrize("B", [2, 8])
    def test_m_cv3_conv2d_dynamic_batch_bwd(self, B: int) -> None:
        _ensure_env()
        torch.manual_seed(42)
        conv_cpu = torch.nn.Conv2d(3, 8, 3, padding=1, stride=2)
        conv_vk = torch.nn.Conv2d(3, 8, 3, padding=1, stride=2).to("vulkan:0")
        with torch.no_grad():
            conv_vk.weight.copy_(conv_cpu.weight.to("vulkan:0"))
            conv_vk.bias.copy_(conv_cpu.bias.to("vulkan:0"))

        x_cpu = torch.randn(B, 3, 32, 32, requires_grad=True)
        x_vk = x_cpu.detach().to("vulkan:0").requires_grad_(True)
        torch._dynamo.mark_dynamic(x_vk, 0)

        def fn_vk(x):
            return conv_vk(x).sum()

        def fn_cpu(x):
            return conv_cpu(x).sum()

        compiled = torch.compile(fn_vk, backend="inductor", dynamic=True)
        out_vk = compiled(x_vk)
        out_cpu = fn_cpu(x_cpu)
        torch.testing.assert_close(
            out_vk.cpu(), out_cpu, atol=1e-3, rtol=1e-3
        )

        out_vk.backward()
        out_cpu.backward()

        torch.testing.assert_close(
            x_vk.grad.cpu(), x_cpu.grad, atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            conv_vk.weight.grad.cpu(),
            conv_cpu.weight.grad,
            atol=1e-3,
            rtol=1e-3,
        )
        torch.testing.assert_close(
            conv_vk.bias.grad.cpu(),
            conv_cpu.bias.grad,
            atol=1e-3,
            rtol=1e-3,
        )


# --------------------------------------------------------------------------
# 5) Pointwise + reduce chain with broadcast under symbolic batch
# --------------------------------------------------------------------------
class TestMCV3PointwiseReduceChainDynamicBatch:
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "M-CV.3 gap: dynamic-shape pointwise+reduce backward fails "
            "with RuntimeError 'when unpacking SymInt, expected int but "
            "got s17' from expect_int() during at::_ops::expand::call() "
            "inside SumBackward0. Forward pass itself completes and the "
            "pre-backward parity assertion succeeds — the breakage is "
            "strictly in the backward expand-of-symbolic-extent path, "
            "the same root cause as the 3D matmul and Conv2d backward "
            "xfails above. Verified 2026-05-19."
        ),
    )
    @pytest.mark.parametrize("B", [1, 4, 8])
    def test_m_cv3_pointwise_reduce_chain_dynamic_batch(self, B: int) -> None:
        _ensure_env()
        torch.manual_seed(42)
        H, W = 16, 16

        x_cpu = torch.randn(B, 1, 1, requires_grad=True)
        y_cpu = torch.randn(B, H, W, requires_grad=True)
        x_vk = x_cpu.detach().to("vulkan:0").requires_grad_(True)
        y_vk = y_cpu.detach().to("vulkan:0").requires_grad_(True)
        torch._dynamo.mark_dynamic(y_vk, 0)

        def fn(x, y):
            return (x + y).sum()

        compiled = torch.compile(fn, backend="inductor", dynamic=True)

        out_vk = compiled(x_vk, y_vk)
        out_cpu = fn(x_cpu, y_cpu)
        torch.testing.assert_close(
            out_vk.cpu(), out_cpu, atol=1e-4, rtol=1e-4
        )

        out_vk.backward()
        out_cpu.backward()

        torch.testing.assert_close(
            x_vk.grad.cpu(), x_cpu.grad, atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            y_vk.grad.cpu(), y_cpu.grad, atol=1e-4, rtol=1e-4
        )
