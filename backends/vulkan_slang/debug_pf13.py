"""PF.13 detailed diagnostics - compare backends."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
os.environ["TORCH_VULKAN_TRUST_INDUCTOR"] = "0"
import torch
torch._dynamo.reset()

import torch.nn.functional as F
torch.manual_seed(42)

x = torch.randn(4, 32, device="vulkan:0", requires_grad=True)
w = torch.randn(16, 32, device="vulkan:0", requires_grad=True)
b = torch.randn(16, device="vulkan:0", requires_grad=True)

# CPU baseline
x_cpu = x.detach().cpu().requires_grad_()
w_cpu = w.detach().cpu().requires_grad_()
b_cpu = b.detach().cpu().requires_grad_()
out_cpu = F.linear(x_cpu, w_cpu, b_cpu).relu().sum()
out_cpu.backward()

# Try aot_eager (uses same AOTAutograd but runs eager ops)
print("=== aot_eager backend ===")
torch._dynamo.reset()
x2 = torch.randn(4, 32, device="vulkan:0", requires_grad=True)
w2 = torch.randn(16, 32, device="vulkan:0", requires_grad=True)
b2 = torch.randn(16, device="vulkan:0", requires_grad=True)
try:
    fn_aot = torch.compile(lambda x, w, b: F.linear(x, w, b).relu().sum(), backend="aot_eager")
    result = fn_aot(x2, w2, b2)
    result.cpu()
    print(f"  aot_eager result: {result.cpu().item():.4f}")
    print(f"  x.grad: {'OK' if x2.grad is not None else 'None'}")
    if x2.grad is not None:
        diff = (x_cpu.grad - x2.grad.cpu()).abs().max().item()
        print(f"  x.grad diff: {diff:.6f}")
except Exception as e:
    print(f"  aot_eager FAILED: {e}")

# Try inductor with just relu (no linear first)
print("\n=== pure relu+sum compile ===")
torch._dynamo.reset()
a = torch.randn(4, 16, device="vulkan:0", requires_grad=True)
a_cpu = a.detach().cpu().requires_grad_()
try:
    fn_relu = torch.compile(lambda x: x.relu().sum(), backend="inductor")
    result = fn_relu(a)
    result.cpu()
    print(f"  relu result: {result.cpu().item():.4f}")
    if a.grad is not None:
        print(f"  a.grad shape: {a.grad.shape}")
        a_cpu.relu().sum().backward()
        diff = (a_cpu.grad - a.grad.cpu()).abs().max().item()
        print(f"  a.grad diff: {diff:.6f}")
    else:
        print("  a.grad: None")
except Exception as e:
    print(f"  pure relu FAILED: {e}")

# Try linear only (no relu)
print("\n=== linear+sum compile (no relu) ===")
torch._dynamo.reset()
x3 = torch.randn(4, 32, device="vulkan:0", requires_grad=True)
w3 = torch.randn(16, 32, device="vulkan:0", requires_grad=True)
b3 = torch.randn(16, device="vulkan:0", requires_grad=True)
try:
    fn_lin = torch.compile(lambda x, w, b: F.linear(x, w, b).sum(), backend="inductor")
    result = fn_lin(x3, w3, b3)
    result.cpu()
    print(f"  linear+sum result: {result.cpu().item():.4f}")
    for name, t in [('x', x3), ('w', w3), ('b', b3)]:
        if t.grad is not None:
            diff = ({'x': x_cpu, 'w': w_cpu, 'b': b_cpu}[name].grad - t.grad.cpu()).abs().max().item()
            print(f"  {name}.grad diff: {diff:.6f}")
        else:
            print(f"  {name}.grad: None")
except Exception as e:
    print(f"  linear+sum FAILED: {e}")
