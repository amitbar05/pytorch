"""PF.13 — Isolate the exact op that causes the empty gradient."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
os.environ["TORCH_VULKAN_TRUST_INDUCTOR"] = "0"
import torch
torch._dynamo.reset()

torch.manual_seed(42)

# Test: just relu
print("=== Test 1: x.relu().sum() ===")
x = torch.randn(4, 16, device="vulkan:0", requires_grad=True)
x_cpu = x.detach().cpu().requires_grad_()
try:
    fn = torch.compile(lambda a: a.relu().sum(), backend="inductor")
    out = fn(x)
    print(f"  forward: {out.cpu().item():.4f}")
    if x.grad is not None:
        diff = (x_cpu.grad - x.grad.cpu()).abs().max().item()
        print(f"  backward: grad diff={diff:.6f}")
    else:
        print("  backward: x.grad is None")
except Exception as e:
    print(f"  FAILED: {e}")

# Test: just sigmoid
print("\n=== Test 2: x.sigmoid().sum() ===")
torch._dynamo.reset()
x = torch.randn(4, 16, device="vulkan:0", requires_grad=True)
x_cpu = x.detach().cpu().requires_grad_()
try:
    fn = torch.compile(lambda a: a.sigmoid().sum(), backend="inductor")
    out = fn(x)
    print(f"  forward: {out.cpu().item():.4f}")
    if x.grad is not None:
        diff = (x_cpu.grad - x.grad.cpu()).abs().max().item()
        print(f"  backward: grad diff={diff:.6f}")
    else:
        print("  backward: x.grad is None")
except Exception as e:
    print(f"  FAILED: {e}")

# Test: x + y (pointwise binary)
print("\n=== Test 3: (x + y).sum() ===")
torch._dynamo.reset()
x = torch.randn(4, 16, device="vulkan:0", requires_grad=True)
y = torch.randn(4, 16, device="vulkan:0", requires_grad=True)
x_cpu = x.detach().cpu().requires_grad_()
y_cpu = y.detach().cpu().requires_grad_()
try:
    fn = torch.compile(lambda a, b: (a + b).sum(), backend="inductor")
    out = fn(x, y)
    print(f"  forward: {out.cpu().item():.4f}")
    for name, t in [('x', x), ('y', y)]:
        if t.grad is not None:
            ref = {'x': x_cpu, 'y': y_cpu}[name]
            diff = (ref.grad - t.grad.cpu()).abs().max().item()
            print(f"  {name}.grad diff: {diff:.6f}")
        else:
            print(f"  {name}.grad: None")
except Exception as e:
    print(f"  FAILED: {e}")

# Test: matmul backward
print("\n=== Test 4: (x @ w).sum() ===")
torch._dynamo.reset()
x = torch.randn(4, 8, device="vulkan:0", requires_grad=True)
w = torch.randn(8, 16, device="vulkan:0", requires_grad=True)
x_cpu = x.detach().cpu().requires_grad_()
w_cpu = w.detach().cpu().requires_grad_()
try:
    fn = torch.compile(lambda a, b: (a @ b).sum(), backend="inductor")
    out = fn(x, w)
    print(f"  forward: {out.cpu().item():.4f}")
    for name, t in [('x', x), ('w', w)]:
        if t.grad is not None:
            ref = {'x': x_cpu, 'w': w_cpu}[name]
            diff = (ref.grad - t.grad.cpu()).abs().max().item()
            print(f"  {name}.grad diff: {diff:.6f}")
        else:
            print(f"  {name}.grad: None")
except Exception as e:
    print(f"  FAILED: {e}")

# Test: clamp (alternative to relu)
print("\n=== Test 5: x.clamp(min=0).sum() ===")
torch._dynamo.reset()
x = torch.randn(4, 16, device="vulkan:0", requires_grad=True)
x_cpu = x.detach().cpu().requires_grad_()
try:
    fn = torch.compile(lambda a: a.clamp(min=0).sum(), backend="inductor")
    out = fn(x)
    print(f"  forward: {out.cpu().item():.4f}")
    if x.grad is not None:
        diff = (x_cpu.grad - x.grad.cpu()).abs().max().item()
        print(f"  backward: grad diff={diff:.6f}")
    else:
        print("  backward: x.grad is None")
except Exception as e:
    print(f"  FAILED: {e}")
