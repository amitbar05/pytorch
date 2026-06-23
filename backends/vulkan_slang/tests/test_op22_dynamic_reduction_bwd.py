"""OP.22 — Dynamic-shape reduction backward regression tests.

Verifies that reduction backward (sum_backward, mean_backward,
var_backward, prod_backward) produces correct gradients when the
batch dimension is symbolic (``dynamic=True``).  Covers the
symbolic-stride path where stride expressions contain
sympy.Symbol(B) and must flow through push constants.
"""

from __future__ import annotations

import os

import torch
import torch_vulkan  # noqa: F401


def _ensure_env():
    if not os.environ.get("SLANGC"):
        import pytest

        pytest.skip("SLANGC env var not set")
    os.environ["TORCH_VULKAN_DYNAMIC_SHAPES"] = "1"


class TestOP22SumBackward:
    """Dynamic-shape sum backward across batch sizes."""

    def test_sum_backward_cross_batch(self):
        """``sum(dim=-1).sum().backward()`` — grad matches CPU for
        batch sizes {1, 4, 16, 64} with a single compile."""
        _ensure_env()
        from torch_vulkan.inductor.runtime import (
            _COMPILE_STATS,
            reset_compile_stats,
        )

        @torch.compile(backend="inductor", dynamic=True)
        def fn(x):
            return x.sum(dim=-1).sum()

        # Warmup
        x_warm = torch.randn(4, 128, device="vulkan:0")
        fn(x_warm).backward()
        reset_compile_stats()

        for B in [1, 4, 16, 64]:
            x_vk = torch.randn(B, 128, device="vulkan:0")
            x_cpu = x_vk.cpu().requires_grad_(True)

            out_vk = fn(x_vk)
            out_cpu = x_cpu.sum(dim=-1).sum()

            torch.testing.assert_close(
                out_vk.cpu(),
                out_cpu,
                rtol=1e-3,
                atol=1e-3,
                msg=f"OP.22 sum fwd: batch={B}",
            )

            out_vk.backward()
            out_cpu.backward()

            torch.testing.assert_close(
                x_vk.grad.cpu(),
                x_cpu.grad,
                rtol=1e-2,
                atol=1e-2,
                msg=f"OP.22 sum bwd: batch={B}",
            )

        cold = _COMPILE_STATS.get("cold_compiles", 0)
        assert cold == 0, (
            f"OP.22 sum: {cold} recompiles after warmup — expected 0. "
            "The symbolic-shape reduction kernel should be reused."
        )


class TestOP22MeanBackward:
    """Dynamic-shape mean backward across batch sizes."""

    def test_mean_backward_cross_batch(self):
        """``mean(dim=-1).sum().backward()`` — grad matches CPU."""
        _ensure_env()

        @torch.compile(backend="inductor", dynamic=True)
        def fn(x):
            return x.mean(dim=-1).sum()

        x_warm = torch.randn(4, 128, device="vulkan:0")
        fn(x_warm).backward()

        for B in [1, 4, 16, 64]:
            x_vk = torch.randn(B, 128, device="vulkan:0")
            x_cpu = x_vk.cpu().requires_grad_(True)

            out_vk = fn(x_vk)
            out_cpu = x_cpu.mean(dim=-1).sum()

            out_vk.backward()
            out_cpu.backward()

            torch.testing.assert_close(
                x_vk.grad.cpu(),
                x_cpu.grad,
                rtol=1e-2,
                atol=1e-2,
                msg=f"OP.22 mean bwd: batch={B}",
            )


class TestOP22VarBackward:
    """Dynamic-shape var backward across batch sizes."""

    def test_var_backward_cross_batch(self):
        """``var(dim=-1).sum().backward()`` — grad matches CPU."""
        _ensure_env()

        @torch.compile(backend="inductor", dynamic=True)
        def fn(x):
            return x.var(dim=-1).sum()

        x_warm = torch.randn(4, 128, device="vulkan:0")
        fn(x_warm).backward()

        for B in [1, 4, 16, 64]:
            x_vk = torch.randn(B, 128, device="vulkan:0")
            x_cpu = x_vk.cpu().requires_grad_(True)

            out_vk = fn(x_vk)
            out_cpu = x_cpu.var(dim=-1).sum()

            out_vk.backward()
            out_cpu.backward()

            torch.testing.assert_close(
                x_vk.grad.cpu(),
                x_cpu.grad,
                rtol=1e-2,
                atol=1e-2,
                msg=f"OP.22 var bwd: batch={B}",
            )


class TestOP22ProdBackward:
    """Dynamic-shape prod backward across batch sizes."""

    def test_prod_backward_cross_batch(self):
        """``prod(dim=-1).sum().backward()`` — grad matches CPU."""
        _ensure_env()

        @torch.compile(backend="inductor", dynamic=True)
        def fn(x):
            return x.abs().prod(dim=-1).sum()

        x_warm = torch.rand(4, 64, device="vulkan:0") + 0.5
        fn(x_warm).backward()

        for B in [1, 4, 16]:
            x_vk = torch.rand(B, 64, device="vulkan:0") + 0.5
            x_cpu = x_vk.cpu().requires_grad_(True)

            out_vk = fn(x_vk)
            out_cpu = x_cpu.abs().prod(dim=-1).sum()

            out_vk.backward()
            out_cpu.backward()

            torch.testing.assert_close(
                x_vk.grad.cpu(),
                x_cpu.grad,
                rtol=1e-2,
                atol=1e-2,
                msg=f"OP.22 prod bwd: batch={B}",
            )
