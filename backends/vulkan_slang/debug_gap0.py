"""Targeted repro for known inductor bugs."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
os.environ["TORCH_VULKAN_TRUST_INDUCTOR"] = "0"
import torch

torch._dynamo.reset()

# Test PF.13: bwd compile via view ops
print("=" * 60)
print("TEST PF.13: linear+relu+sum backward (test_linear_relu_sum_backward)")
torch.manual_seed(42)
import torch.nn.functional as F
x = torch.randn(4, 32, device="vulkan:0", requires_grad=True)
w = torch.randn(16, 32, device="vulkan:0", requires_grad=True)
b = torch.randn(16, device="vulkan:0", requires_grad=True)

x_cpu = x.detach().cpu().requires_grad_()
w_cpu = w.detach().cpu().requires_grad_()
b_cpu = b.detach().cpu().requires_grad_()

# Eager vulkan reference
out_ref = F.linear(x, w, b).relu().sum()
out_ref.backward()
ref_grads = {k: v.grad.cpu() for k, v in [('x', x), ('w', w), ('b', b)]}

# CPU reference  
out_cpu = F.linear(x_cpu, w_cpu, b_cpu).relu().sum()
out_cpu.backward()
cpu_grads = {k: v.grad for k, v in [('x', x_cpu), ('w', w_cpu), ('b', b_cpu)]}

torch._dynamo.reset()
try:
    compiled_fn = torch.compile(lambda x, w, b: F.linear(x, w, b).relu().sum(), backend="inductor")
    result = compiled_fn(x, w, b)
    result.cpu()
    for name in ('x', 'w', 'b'):
        g = {'x': x, 'w': w, 'b': b}[name].grad
        if g is not None:
            diff_cpu = (cpu_grads[name] - g.cpu()).abs().max().item()
            diff_vk = (ref_grads[name] - g.cpu()).abs().max().item()
            print(f"  {name}: cpu_diff={diff_cpu:.6f}, vk_diff={diff_vk:.6f} {'PASS' if diff_cpu < 1e-3 else 'FAIL'}")
        else:
            print(f"  {name}: grad is None - FAIL")
except Exception as e:
    print(f"FAIL: {e}")
    # Short traceback
    import traceback; traceback.print_exc(limit=5)

# Test: simple 2-layer MLP backward  
print("\n" + "=" * 60)
print("TEST: 2-layer MLP backward")
import torch.nn as nn
torch.manual_seed(42)
m = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 16)).to("vulkan:0")
x = torch.randn(4, 32, device="vulkan:0", requires_grad=True)
m_cpu = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 16))
m_cpu.load_state_dict({k: v.cpu() for k, v in m.state_dict().items()})
x_cpu = x.detach().cpu().requires_grad_()

out_ref = m(x).sum()
out_ref.backward()
x_ref_grad = x.grad.cpu()

out_cpu = m_cpu(x_cpu).sum()
out_cpu.backward()
x_cpu_grad = x_cpu.grad

torch._dynamo.reset()
try:
    compiled_fn = torch.compile(lambda x: m(x).sum(), backend="inductor")
    result = compiled_fn(x)
    result.cpu()
    if x.grad is not None:
        diff_x = (x_cpu_grad - x.grad.cpu()).abs().max().item()
        print(f"  x grad: cpu_diff={diff_x:.6f} {'PASS' if diff_x < 1e-3 else 'FAIL'}")
    else:
        print("  x.grad is None")
    for name, p in m.named_parameters():
        if p.grad is not None:
            p_cpu = dict(m_cpu.named_parameters())[name]
            diff_p = (p_cpu.grad.cpu() - p.grad.cpu()).abs().max().item()
            print(f"  {name}: cpu_diff={diff_p:.6f} {'PASS' if diff_p < 1e-3 else 'FAIL'}")
        else:
            print(f"  {name}: grad is None")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback; traceback.print_exc(limit=5)
