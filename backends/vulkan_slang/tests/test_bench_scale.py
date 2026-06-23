"""Benchmark: 2 conv layers at scale — separate warm-up from training."""

import time
import pytest
import torch
import torch.nn as nn
import torch_vulkan


@pytest.mark.gpu
def test_bench_scale():
    B, H, W = 4, 128, 128
    C1, C2 = 64, 128
    N_STEPS = 20
    WARM = 3

    # ══════ CPU ══════
    torch.manual_seed(42)
    cpu_mod = nn.Sequential(
        nn.Conv2d(3, C1, 3, padding=1),
        nn.Conv2d(C1, C2, 3, padding=1),
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

    # ══════ Vulkan ══════
    torch.manual_seed(42)
    vk_mod = nn.Sequential(
        nn.Conv2d(3, C1, 3, padding=1),
        nn.Conv2d(C1, C2, 3, padding=1),
    ).to("vulkan:0")
    x_vk = x_cpu.detach().clone().to("vulkan:0")

    print("[VK] Device warm-up...")
    t_dw = time.perf_counter()
    torch_vulkan.prepare_device(level="quick", timeout_s=60, verbose=False)
    dw_s = time.perf_counter() - t_dw

    print("[VK] Model warm-up (compile fwd+bwd)...")
    t_mw = time.perf_counter()
    compiled = torch.compile(vk_mod, backend="inductor")
    out = compiled(x_vk)
    out.sum().backward()
    vk_mod.zero_grad()
    mw_s = time.perf_counter() - t_mw

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
    print("=" * 60)
    print(f"  MODEL: 2×Conv2d(3→{C1}→{C2}, k=3)")
    print(f"  INPUT: {B}×3×{H}×{W}")
    print(f"  Device warm-up:  {dw_s:>8.1f}s")
    print(f"  Model warm-up:   {mw_s:>8.1f}s  (compile + 1st fwd+bwd)")
    print(f"  CPU fwd+bwd:     {cpu_ms:>8.1f} ms/step")
    print(f"  VK  fwd+bwd:     {vk_ms:>8.1f} ms/step")
    print(f"  Speedup vs CPU:  {speedup:>8.1f}x")
    print("=" * 60)
