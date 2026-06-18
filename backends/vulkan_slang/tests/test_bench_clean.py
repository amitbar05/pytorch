"""Clean benchmark: fwd+bwd timing (no optimizer step to avoid NaN)."""

import time
import pytest
import torch
import torch.nn as nn
import torch_vulkan


@pytest.mark.gpu
def test_bench_clean():
    B, H, W = 4, 32, 32
    N_STEPS = 50
    WARM = 5

    # CPU baseline
    torch.manual_seed(42)
    cpu_mod = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.Conv2d(16, 32, 3, padding=1),
    )
    x_cpu = torch.randn(B, 3, H, W)
    cpu_mod(x_cpu)
    for _ in range(WARM):
        out = cpu_mod(x_cpu)
        out.sum().backward()
        cpu_mod.zero_grad()
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        out = cpu_mod(x_cpu)
        out.sum().backward()
        cpu_mod.zero_grad()
    cpu_ms = (time.perf_counter() - t0) / N_STEPS * 1000

    # Vulkan
    torch.manual_seed(42)
    vk_mod = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.Conv2d(16, 32, 3, padding=1),
    ).to("vulkan:0")
    x_vk = x_cpu.detach().clone().to("vulkan:0")

    # Device warm-up
    t_dw = time.perf_counter()
    torch_vulkan.prepare_device(level="quick", timeout_s=30, verbose=False)
    dw_s = time.perf_counter() - t_dw

    # Model warm-up (compile fwd+bwd)
    print("[VK] Model warm-up...")
    t_mw = time.perf_counter()
    compiled = torch_vulkan.prepare_model(vk_mod, x_vk, verbose=False)
    mw_s = time.perf_counter() - t_mw

    # Benchmark
    for _ in range(WARM):
        out = compiled(x_vk)
        out.sum().backward()
        vk_mod.zero_grad()

    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        out = compiled(x_vk)
        out.sum().backward()
        vk_mod.zero_grad()
    vk_ms = (time.perf_counter() - t0) / N_STEPS * 1000

    speedup = cpu_ms / vk_ms

    print()
    print("=" * 55)
    print(f"  MODEL: 2×Conv2d(3→16→32, k=3)")
    print(f"  INPUT: {B}×3×{H}×{W}")
    print(f"  Device warm-up:  {dw_s:>8.1f}s")
    print(f"  Model warm-up:   {mw_s:>8.1f}s")
    print(f"  Total warm-up:   {dw_s + mw_s:>8.1f}s")
    print(f"  CPU fwd+bwd:     {cpu_ms:>8.1f} ms/step")
    print(f"  VK  fwd+bwd:     {vk_ms:>8.1f} ms/step")
    print(f"  Speedup vs CPU:  {speedup:>8.1f}x")
    print("=" * 55)
