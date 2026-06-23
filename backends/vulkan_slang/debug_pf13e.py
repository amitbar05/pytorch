"""PF.13 — Check backward under compile (not just forward)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
os.environ["TORCH_VULKAN_TRUST_INDUCTOR"] = "0"
import torch
torch._dynamo.reset()

torch.manual_seed(42)

# Test ALL ops with actual backward call
tests = [
    ("sigmoid", lambda x: x.sigmoid().sum()),
    ("tanh", lambda x: x.tanh().sum()),
    ("relu", lambda x: x.relu().sum()),
    ("silu", lambda x: x.silu().sum()),
    ("gelu", lambda x: x.gelu().sum()),
    ("x + y", lambda a, b: (a + b).sum()),
    ("x * y", lambda a, b: (a * b).sum()),
    ("x @ w", lambda a, b: (a @ b).sum()),
]

for name, fn in tests:
    print(f"\n=== {name} ===")
    torch._dynamo.reset()
    try:
        if "x + y" in name or "x * y" in name:
            a = torch.randn(4, 16, device="vulkan:0", requires_grad=True)
            b = torch.randn(4, 16, device="vulkan:0", requires_grad=True)
            a_cpu = a.detach().cpu().requires_grad_()
            b_cpu = b.detach().cpu().requires_grad_()
            compiled = torch.compile(lambda a, b: fn(a, b), backend="inductor", fullgraph=True)
            out = compiled(a, b)
            out.backward()
            ref = fn(a_cpu, b_cpu)
            ref.backward()
            for name2, t, ref_t in [('a', a, a_cpu), ('b', b, b_cpu)]:
                if t.grad is not None and ref_t.grad is not None:
                    diff = (ref_t.grad - t.grad.cpu()).abs().max().item()
                    print(f"  {name2}.grad diff: {diff:.6f} {'PASS' if diff < 1e-3 else 'FAIL'}")
                else:
                    print(f"  {name2}.grad: {'None' if t.grad is None else 'set'} (ref: {'None' if ref_t.grad is None else 'set'})")
        elif "x @ w" in name:
            a = torch.randn(4, 8, device="vulkan:0", requires_grad=True)
            b = torch.randn(8, 16, device="vulkan:0", requires_grad=True)
            a_cpu = a.detach().cpu().requires_grad_()
            b_cpu = b.detach().cpu().requires_grad_()
            compiled = torch.compile(lambda a, b: fn(a, b), backend="inductor", fullgraph=True)
            out = compiled(a, b)
            out.backward()
            ref = fn(a_cpu, b_cpu)
            ref.backward()
            for name2, t, ref_t in [('a', a, a_cpu), ('b', b, b_cpu)]:
                if t.grad is not None and ref_t.grad is not None:
                    diff = (ref_t.grad - t.grad.cpu()).abs().max().item()
                    print(f"  {name2}.grad diff: {diff:.6f} {'PASS' if diff < 1e-3 else 'FAIL'}")
                else:
                    print(f"  {name2}.grad: {'None' if t.grad is None else 'set'} (ref: {'None' if ref_t.grad is None else 'set'})")
        else:
            a = torch.randn(4, 16, device="vulkan:0", requires_grad=True)
            a_cpu = a.detach().cpu().requires_grad_()
            compiled = torch.compile(lambda a: fn(a), backend="inductor", fullgraph=True)
            out = compiled(a)
            out.backward()
            ref = fn(a_cpu)
            ref.backward()
            if a.grad is not None and a_cpu.grad is not None:
                diff = (a_cpu.grad - a.grad.cpu()).abs().max().item()
                print(f"  grad diff: {diff:.6f} {'PASS' if diff < 1e-3 else 'FAIL'}")
            else:
                print(f"  grad: {'None' if a.grad is None else 'set'} (ref: {'None' if a_cpu.grad is None else 'set'})")
    except Exception as e:
        print(f"  FAILED: {e}")
