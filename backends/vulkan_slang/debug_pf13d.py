"""PF.13 — Check if autograd device type is registered for Vulkan."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
import torch

# Check which devices are registered for autograd
print("=== Autograd device capabilities ===")
print(f"torch._C._autograd_init: {hasattr(torch._C, '_autograd_init')}")

# Check: does Vulkan appear in any autograd registries?
import torch._C as _C
print("\n=== Registered autograd device types ===")
# Look at what device keys are registered in autograd
try:
    for dev_type in ['cpu', 'cuda', 'mps', 'xla', 'vulkan', 'meta', 'privateuse1']:
        dev = torch.device(dev_type, 0)
        print(f"  {dev_type}:")
        print(f"    is_autograd_capable_device: {torch.is_autograd_capable_device(dev)}")
except Exception as e:
    print(f"  Error: {e}")

# Check: is autograd being properly set up for Vulkan tensors?
print("\n=== Tensor autograd metadata ===")
x = torch.randn(4, 16, device="vulkan:0", requires_grad=True)
y = x.sigmoid()
print(f"x._grad_fn: {x.grad_fn}")
print(f"y._grad_fn: {y.grad_fn}")
print(f"y._backward_hooks: {y._backward_hooks}")
print(f"y.name: {y.name() if hasattr(y, 'name') else 'N/A'}")

# Check PyTorch's internal autograd functions list
print("\n=== Autograd function internals ===")
print(f"torch._C._are_functorch_transforms_active: {torch._C._are_functorch_transforms_active()}")

# Check: does the PrivateUse1 device get special autograd treatment?
print("\n=== PrivateUse1 autograd check ===")
try:
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        x_fake = torch.empty(4, 16, device="vulkan:0")
        y_fake = torch.sigmoid(x_fake)
        print(f"FakeTensor requires_grad: {y_fake.requires_grad}")
        print(f"FakeTensor grad_fn: {y_fake.grad_fn}")
except Exception as e:
    print(f"FakeTensor test error: {e}")

# Check: does .sum() backward work in eager for Vulkan?
print("\n=== Eager backward on Vulkan ===")
x = torch.randn(4, 16, device="vulkan:0", requires_grad=True)
y = x.sigmoid().sum()
y.backward()
print(f"x.grad after backward: {x.grad is not None}")
if x.grad is not None:
    print(f"x.grad shape: {x.grad.shape}")

# Check a very simple test: just backward through sigmoid.sigmoid (no sum)
print("\n=== Eager backward through sigmoid only ===")
x = torch.randn(4, 16, device="vulkan:0", requires_grad=True)
y = x.sigmoid()
# Create a manual grad_output (not sum)
grad_output = torch.randn(4, 16, device="vulkan:0")
torch.autograd.backward([y], [grad_output])
print(f"x.grad after manual backward: {x.grad is not None}")

# Check the output requires_grad chain
print("\n=== requires_grad propagation ===")
x = torch.randn(4, 16, device="vulkan:0", requires_grad=True)
y = torch.sigmoid(x)
print(f"input.requires_grad: {x.requires_grad}")
print(f"sigmoid.requires_grad: {y.requires_grad}")
print(f"sigmoid.grad_fn: {y.grad_fn}")
if y.grad_fn:
    print(f"sigmoid.grad_fn name: {y.grad_fn.name()}")
