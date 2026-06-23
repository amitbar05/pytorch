"""Diagnostic pytest test for max_pool backward correctness."""
import torch
import pytest


@pytest.mark.timeout(60)
def test_maxpool_bwd_eager(vulkan_device):
    """Test eager max_pool2d backward with Vulkan indices."""
    import sys
    print("\n[T] starting test", file=sys.stderr, flush=True)
    torch.manual_seed(0)
    x_cpu = torch.randn(1, 4, 8, 8)
    print("[T] made x_cpu", file=sys.stderr, flush=True)
    x_vk = x_cpu.to(vulkan_device)
    print("[T] made x_vk", file=sys.stderr, flush=True)

    # Test 1: autograd-based backward
    x_cpu2 = x_cpu.clone().requires_grad_(True)
    y_cpu = torch.nn.functional.max_pool2d(x_cpu2, 2)
    y_cpu.sum().backward()
    cpu_grad = x_cpu2.grad
    print("[T] cpu backward done", file=sys.stderr, flush=True)

    x_vk2 = x_vk.clone().requires_grad_(True)
    y_vk = torch.nn.functional.max_pool2d(x_vk2, 2)
    print("[T] vk fwd done", file=sys.stderr, flush=True)
    y_vk.sum().backward()
    print("[T] vk bwd done", file=sys.stderr, flush=True)
    vk_grad = x_vk2.grad.cpu()

    l_inf1 = (cpu_grad - vk_grad).abs().max().item()
    print(f"\nTest1 autograd L_inf={l_inf1:.8f} cpu_norm={cpu_grad.norm():.4f} vk_norm={vk_grad.norm():.4f}")
    assert l_inf1 < 1e-4, f"autograd backward L_inf={l_inf1}"

    # Test 2: direct aten call with VK indices
    torch.manual_seed(0)
    x_cpu3 = torch.randn(1, 4, 8, 8)
    x_vk3 = x_cpu3.to(vulkan_device)

    out_cpu, idx_cpu = torch.nn.functional.max_pool2d_with_indices(x_cpu3, 2)
    out_vk, idx_vk = torch.nn.functional.max_pool2d_with_indices(x_vk3, 2)

    print(f"\nidx_cpu[0,0,:2,:2]:\n{idx_cpu[0,0,:2,:2]}")
    print(f"idx_vk[0,0,:2,:2].cpu():\n{idx_vk[0,0,:2,:2].cpu()}")
    idx_match = (idx_cpu == idx_vk.cpu()).all().item()
    print(f"Indices match: {idx_match}")

    grad = torch.ones_like(out_cpu)
    grad_vk_tensor = grad.to(vulkan_device)

    grad_in_cpu = torch.ops.aten.max_pool2d_with_indices_backward.default(
        grad, x_cpu3, [2, 2], [2, 2], [0, 0], [1, 1], False, idx_cpu
    )
    grad_in_vk = torch.ops.aten.max_pool2d_with_indices_backward.default(
        grad_vk_tensor, x_vk3, [2, 2], [2, 2], [0, 0], [1, 1], False, idx_vk
    ).cpu()

    l_inf2 = (grad_in_cpu - grad_in_vk).abs().max().item()
    print(f"\nTest2 direct VK backward L_inf={l_inf2:.8f}")
    assert l_inf2 < 1e-4, f"direct VK backward L_inf={l_inf2}"

    # Test 3: VK indices on CPU backward (tests index correctness only)
    grad_in_vk_idx_cpu = torch.ops.aten.max_pool2d_with_indices_backward.default(
        grad, x_cpu3, [2, 2], [2, 2], [0, 0], [1, 1], False, idx_vk.cpu()
    )
    l_inf3 = (grad_in_cpu - grad_in_vk_idx_cpu).abs().max().item()
    print(f"\nTest3 VK-indices on CPU backward L_inf={l_inf3:.8f}")
    assert l_inf3 < 1e-4, f"VK indices wrong, CPU backward L_inf={l_inf3}"
