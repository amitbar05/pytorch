"""M17.2 — Conv2d + activation fusion via Slang epilogue.

Regression tests for the conv→relu fusion FX pass and the conv2d_relu_fused
custom op.  Verifies correctness vs CPU, dispatch count reduction, grouped conv,
conv+gn+relu chains, and training backward.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class TestM172ConvEpilogue:
    """M17.2 — Conv2d + activation fusion via Slang epilogue.

    Verifies that ``conv2d → relu`` is fused into a single dispatch
    (conv template with ``<Epilogue : IDifferentiable>``) instead of
    separate conv + pointwise dispatches.
    """

    _BUG_ROOT_COMPONENT = "codegen"

    @staticmethod
    def _make_small_cnn_conv_layer():
        """Create a SmallCNN-style conv layer: Conv2d(3, 16, 3, padding=1)."""
        return nn.Conv2d(3, 16, 3, padding=1, bias=True)

    def test_conv_relu_correctness(self):
        """Conv2d→ReLU fused should match CPU reference."""
        conv = self._make_small_cnn_conv_layer().to("vulkan:0")

        @torch.compile(backend="inductor")
        def fn(x, w, b):
            y = torch.nn.functional.conv2d(x, w, b, stride=1, padding=1)
            return torch.relu(y)

        x = torch.randn(2, 3, 16, 16, device="vulkan:0")
        result = fn(x, conv.weight, conv.bias)

        expected = torch.relu(
            torch.nn.functional.conv2d(
                x.cpu(),
                conv.weight.cpu(),
                conv.bias.cpu(),
                stride=1,
                padding=1,
            )
        )
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_conv_relu_dispatch_count(self):
        """Conv2d→ReLU must be ≤2 dispatches (1 fused conv+relu)."""
        import torch_vulkan

        conv = self._make_small_cnn_conv_layer().to("vulkan:0")

        @torch.compile(backend="inductor")
        def fn(x, w, b):
            y = torch.nn.functional.conv2d(x, w, b, stride=1, padding=1)
            return torch.relu(y)

        x = torch.randn(2, 3, 16, 16, device="vulkan:0")
        # Warm-up
        fn(x, conv.weight, conv.bias)

        torch_vulkan._c_ext._reset_perf_counters()
        fn(x, conv.weight, conv.bias)
        d = torch_vulkan._c_ext._get_dispatch_count()
        assert d <= 2, f"conv2d+relu: expected ≤2 dispatches (fused), got {d}"

    def test_conv_no_bias_relu_correctness(self):
        """Conv2d without bias→ReLU fused should match CPU reference."""
        conv = nn.Conv2d(3, 16, 3, padding=1, bias=False).to("vulkan:0")

        @torch.compile(backend="inductor")
        def fn(x, w):
            y = torch.nn.functional.conv2d(x, w, bias=None, stride=1, padding=1)
            return torch.relu(y)

        x = torch.randn(2, 3, 16, 16, device="vulkan:0")
        result = fn(x, conv.weight)

        expected = torch.relu(
            torch.nn.functional.conv2d(
                x.cpu(),
                conv.weight.cpu(),
                bias=None,
                stride=1,
                padding=1,
            )
        )
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_conv_relu_grouped_conv(self):
        """Grouped conv (groups=2)→ReLU: correctness only."""
        conv = nn.Conv2d(4, 4, 3, padding=1, groups=2, bias=True).to("vulkan:0")

        @torch.compile(backend="inductor")
        def fn(x, w, b):
            y = torch.nn.functional.conv2d(
                x,
                w,
                b,
                stride=1,
                padding=1,
                groups=2,
            )
            return torch.relu(y)

        x = torch.randn(2, 4, 16, 16, device="vulkan:0")
        result = fn(x, conv.weight, conv.bias)

        expected = torch.relu(
            torch.nn.functional.conv2d(
                x.cpu(),
                conv.weight.cpu(),
                conv.bias.cpu(),
                stride=1,
                padding=1,
                groups=2,
            )
        )
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_conv_gn_relu_correctness(self):
        """Conv2d→GroupNorm→ReLU: correctness.

        Verifies that the conv→relu epilogue fusion doesn't break when
        a norm sits between conv and relu.  GN prevents the direct
        conv→relu epilogue match, so the epilogue pass should leave
        this chain intact.
        """

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1, bias=True)
                self.gn = nn.GroupNorm(4, 16)

            def forward(self, x):
                y = self.conv(x)
                y = self.gn(y)
                return torch.relu(y)

        model = Block().to("vulkan:0")

        @torch.compile(backend="inductor")
        def fn(m, x):
            return m(x)

        x = torch.randn(2, 3, 16, 16, device="vulkan:0")
        result = fn(model, x)

        expected = torch.relu(
            model.gn(
                torch.nn.functional.conv2d(
                    x.cpu(),
                    model.conv.weight.cpu(),
                    model.conv.bias.cpu(),
                    stride=1,
                    padding=1,
                )
            )
        )
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_conv_relu_training_backward(self):
        """Training: Conv2d→ReLU forward + backward should produce
        correct gradients vs CPU."""
        conv_vk = nn.Conv2d(3, 16, 3, padding=1, bias=True).to("vulkan:0")
        conv_cpu = nn.Conv2d(3, 16, 3, padding=1, bias=True)
        conv_cpu.load_state_dict(conv_vk.state_dict())

        def make_compiled(conv):
            @torch.compile(backend="inductor")
            def fn(x, w, b):
                y = torch.nn.functional.conv2d(x, w, b, stride=1, padding=1)
                return torch.relu(y)

            return fn

        x_vk = torch.randn(2, 3, 16, 16, device="vulkan:0", requires_grad=True)
        x_cpu = x_vk.detach().cpu().requires_grad_(True)

        fn_vk = make_compiled(conv_vk)
        y_vk = fn_vk(x_vk, conv_vk.weight, conv_vk.bias)
        loss_vk = y_vk.sum()
        loss_vk.backward()

        y_cpu = torch.relu(
            torch.nn.functional.conv2d(
                x_cpu,
                conv_cpu.weight,
                conv_cpu.bias,
                stride=1,
                padding=1,
            )
        )
        loss_cpu = y_cpu.sum()
        loss_cpu.backward()

        torch.testing.assert_close(
            x_vk.grad.cpu(),
            x_cpu.grad,
            rtol=1e-3,
            atol=1e-3,
        )
        torch.testing.assert_close(
            conv_vk.weight.grad.cpu(),
            conv_cpu.weight.grad,
            rtol=1e-3,
            atol=1e-3,
        )
        torch.testing.assert_close(
            conv_vk.bias.grad.cpu(),
            conv_cpu.bias.grad,
            rtol=1e-3,
            atol=1e-3,
        )

    def test_slang_conv2d_epilogue_direct(self):
        """Test _slang_tile_conv2d with epilogue=\"OpReLU\" directly
        (bypasses compile path, validates the Slang template + dispatch)."""
        from torch_vulkan.inductor.templates.caller import _slang_tile_conv2d

        N, C_in, C_out, H, W = 1, 3, 8, 16, 16
        kH, kW = 3, 3
        sH = sW = 1
        pH = pW = 1
        oH = (H + 2 * pH - kH) // sH + 1
        oW = (W + 2 * pW - kW) // sW + 1

        x = torch.randn(N, C_in, H, W, device="vulkan:0")
        w = torch.randn(C_out, C_in, kH, kW, device="vulkan:0")

        # With epilogue
        out_relu = torch.empty(N, C_out, oH, oW, device="vulkan:0")
        _slang_tile_conv2d(
            x,
            w,
            out_relu,
            stride=(sH, sW),
            padding=(pH, pW),
            dilation=(1, 1),
            epilogue="OpReLU",
        )

        # CPU reference
        expected = torch.nn.functional.conv2d(
            x.cpu(),
            w.cpu(),
            bias=None,
            stride=1,
            padding=1,
        )
        expected_relu = torch.relu(expected)

        torch.testing.assert_close(
            out_relu.cpu(),
            expected_relu,
            rtol=1e-4,
            atol=1e-4,
        )

    def test_slang_conv2d_epilogue_identity(self):
        """Test _slang_tile_conv2d with epilogue=None (identity path)
        produces same result as the default computeMain path."""
        from torch_vulkan.inductor.templates.caller import _slang_tile_conv2d

        N, C_in, C_out, H, W = 1, 3, 8, 16, 16
        kH, kW = 3, 3
        oH = (H + 2 * 1 - kH) // 1 + 1
        oW = (W + 2 * 1 - kW) // 1 + 1

        x = torch.randn(N, C_in, H, W, device="vulkan:0")
        w = torch.randn(C_out, C_in, kH, kW, device="vulkan:0")

        # No epilogue (uses OpIdentity internally)
        out = torch.empty(N, C_out, oH, oW, device="vulkan:0")
        _slang_tile_conv2d(
            x,
            w,
            out,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            epilogue=None,
        )

        expected = torch.nn.functional.conv2d(
            x.cpu(),
            w.cpu(),
            bias=None,
            stride=1,
            padding=1,
        )
        torch.testing.assert_close(
            out.cpu(),
            expected,
            rtol=1e-4,
            atol=1e-4,
        )
