"""Standalone reproduction test for P0.0 backward graph compilation.

Run with:
  python backends/vulkan_slang/tests/test_backward_repro.py
"""
import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import torch

def test_sum_of_square_plus_one_backward():
    """The simplest possible compiled backward - (x*x+1).sum().backward()"""
    print("Testing (x*x+1).sum().backward() under torch.compile...")

    @torch.compile(backend="inductor")
    def fn(x):
        return (x * x + 1.0).sum()

    x = torch.randn(8, 16, device="vulkan:0", requires_grad=True)
    x_cpu = x.detach().cpu().requires_grad_()

    try:
        result = fn(x)
        print(f"  Forward OK: shape={result.shape}, device={result.device}")

        result.backward()
        print(f"  Backward OK: grad device={x.grad.device}, grad shape={x.grad.shape}")

        ((x_cpu * x_cpu + 1.0).sum()).backward()

        torch.testing.assert_close(x.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)
        print("  PASSED: backward compilation works!")
        return True
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def test_linear_relu_sum_backward():
    """Linear + relu + sum backward under compile."""
    print("Testing linear+relu+sum backward under torch.compile...")
    import torch.nn.functional as F

    @torch.compile(backend="inductor")
    def fn(x, w, b):
        return F.relu(F.linear(x, w, b)).sum()

    x = torch.randn(4, 32, device="vulkan:0", requires_grad=True)
    w = torch.randn(16, 32, device="vulkan:0", requires_grad=True)
    b = torch.randn(16, device="vulkan:0", requires_grad=True)
    x_cpu = x.detach().cpu().requires_grad_()
    w_cpu = w.detach().cpu().requires_grad_()
    b_cpu = b.detach().cpu().requires_grad_()

    try:
        fn(x, w, b).backward()
        F.relu(F.linear(x_cpu, w_cpu, b_cpu)).sum().backward()

        torch.testing.assert_close(x.grad.cpu(), x_cpu.grad, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(w.grad.cpu(), w_cpu.grad, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(b.grad.cpu(), b_cpu.grad, rtol=1e-3, atol=1e-3)
        print("  PASSED!")
        return True
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":

    cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR", f"/tmp/torchinductor_{os.environ.get('USER', 'root')}")
    import shutil
    shutil.rmtree(cache, ignore_errors=True)
    print(f"Cleared cache: {cache}")

    results = []
    results.append(("sum_of_square+1", test_sum_of_square_plus_one_backward()))
    print()
    results.append(("linear_relu_sum", test_linear_relu_sum_backward()))

    print()
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")

    sys.exit(0 if all(p for _, p in results) else 1)
