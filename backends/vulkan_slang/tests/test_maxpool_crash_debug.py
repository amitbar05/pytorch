"""Minimal debug test to find heap corruption source."""
import torch
import sys

def test_maxpool_bwd_step_by_step(vulkan_device):
    print("\nStep 1: basic tensors", file=sys.stderr, flush=True)
    torch.manual_seed(0)
    x_cpu = torch.randn(1, 2, 4, 4)
    x_vk = x_cpu.to(vulkan_device)
    print("Step 2: cpu max_pool2d fwd", file=sys.stderr, flush=True)
    y_cpu = torch.nn.functional.max_pool2d(x_cpu.clone().requires_grad_(True), 2)
    y_cpu.sum().backward()
    print("Step 3: vk max_pool2d fwd", file=sys.stderr, flush=True)
    x_vk2 = x_vk.clone().requires_grad_(True)
    y_vk = torch.nn.functional.max_pool2d(x_vk2, 2)
    print(f"  y_vk shape: {y_vk.shape}", file=sys.stderr, flush=True)
    print("Step 4: sum().backward()", file=sys.stderr, flush=True)
    y_vk.sum().backward()
    print("Step 5: grad check", file=sys.stderr, flush=True)
    vk_grad = x_vk2.grad.cpu()
    print(f"  done L_inf={vk_grad.abs().max().item():.4f}", file=sys.stderr, flush=True)
