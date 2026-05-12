"""Reproduce GAP 1.1 — matmul+softmax backward silently produces zero gradients.

To run:
    cd backends/vulkan_slang
    source .venv/bin/activate
    TORCH_LOGS=output_code python debug_gap1_1.py
"""

import torch
import torch_vulkan.inductor  # registers the backend

torch.manual_seed(0)

# Simple matmul + softmax + sum
@torch.compile(backend="inductor")
def fn(q, k):
    scores = torch.matmul(q, k.transpose(-2, -1))
    return torch.softmax(scores, dim=-1).sum()

# Vulkan tensors with grad
q = torch.randn(2, 4, 64, 32, device="vulkan:0", requires_grad=True)
k = torch.randn(2, 4, 64, 32, device="vulkan:0", requires_grad=True)

# CPU reference
q_cpu = q.detach().cpu().requires_grad_()
k_cpu = k.detach().cpu().requires_grad_()

print("=== Forward pass ===")
loss = fn(q, k)
print(f"Compiled loss: {loss.item():.6f}")

ref = torch.softmax(
    torch.matmul(q_cpu, k_cpu.transpose(-2, -1)), dim=-1
).sum()
print(f"CPU ref loss:  {ref.item():.6f}")
print(f"Forward diff:  {abs(loss.cpu().item() - ref.item()):.6e}")

print("\n=== Backward pass ===")
loss.backward()
ref.backward()

if q.grad is None:
    print("BUG: q.grad is None — backward compile failed (deepcopy/set_source_Storage)")
elif q.grad.mean().cpu().item() == 0.0:
    print(f"BUG: q.grad.mean() = 0.0 — silently zero gradients (GAP 1.1)")
else:
    print(f"q.grad.mean() = {q.grad.mean().cpu().item():.6f}")

if k.grad is None:
    print("BUG: k.grad is None — backward compile failed")
elif k.grad.mean().cpu().item() == 0.0:
    print(f"BUG: k.grad.mean() = 0.0 — silently zero gradients (GAP 1.1)")
else:
    print(f"k.grad.mean() = {k.grad.mean().cpu().item():.6f}")

if q.grad is not None and q.grad.mean().cpu().item() != 0.0:
    match = torch.allclose(q.grad.cpu(), q_cpu.grad, rtol=1e-3, atol=1e-3)
    print(f"\nq.grad matches CPU: {match}")
    if not match:
        print(f"  q.grad max diff: {(q.grad.cpu() - q_cpu.grad).abs().max().item():.6e}")

if k.grad is not None and k.grad.mean().cpu().item() != 0.0:
    match = torch.allclose(k.grad.cpu(), k_cpu.grad, rtol=1e-3, atol=1e-3)
    print(f"k.grad matches CPU: {match}")
    if not match:
        print(f"  k.grad max diff: {(k.grad.cpu() - k_cpu.grad).abs().max().item():.6e}")
