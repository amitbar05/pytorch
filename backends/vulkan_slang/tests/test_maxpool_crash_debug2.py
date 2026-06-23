"""Crash debug with larger tensor."""
import torch, sys

def test_maxpool_bwd_4ch_8x8(vulkan_device):
    torch.manual_seed(0)
    x_cpu = torch.randn(1, 4, 8, 8)
    x_vk = x_cpu.to(vulkan_device)
    print("\nStarting 1,4,8,8 test...", file=sys.stderr, flush=True)
    x_vk2 = x_vk.clone().requires_grad_(True)
    y_vk = torch.nn.functional.max_pool2d(x_vk2, 2)
    print(f"forward done shape={y_vk.shape}", file=sys.stderr, flush=True)
    y_vk.sum().backward()
    print(f"backward done grad_norm={x_vk2.grad.cpu().norm():.4f}", file=sys.stderr, flush=True)
