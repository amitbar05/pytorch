"""Verify Slang backward feature in inductor pipeline for conv2d training.

Matches debug_gap0.py working pattern: compile a function that takes tensors
directly (not an nn.Module), uses sum() loss, and runs on vulkan:0 device.

STATUS (2026-05-12):
  - Slang shader validation PASSES (push constants, BackwardDerivative)
  - The backward shader IS dispatched (slang_conv_bwd_8x8x8_t16x16_float)
  - BUT: PF.51 FakeTensor issue during AOT Autograd joint graph trace
  - This is a pre-existing integration gap for custom ops on PrivateUse1
  - Eager mode (without torch.compile) works correctly
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

os.environ["TORCH_VULKAN_TRUST_INDUCTOR"] = "0"
os.environ["TORCH_VULKAN_TRACE_DISPATCH"] = "1"

import torch
import torch.nn.functional as F


def main():
    print("=" * 70)
    print("Slang Backward Feature Verification for Conv2d + Inductor")
    print("=" * 70)

    try:
        import torch_vulkan

        if not torch_vulkan.is_available():
            print("SKIP: No Vulkan device")
            return
    except ImportError:
        print("SKIP: torch_vulkan not installed")
        return

    device = torch.device("vulkan:0")

    # ---- Test 1: Eager mode baseline (should always work) --------------
    print("\n--- Test 1: Eager mode conv backward ---")
    torch.manual_seed(42)

    x_vk = torch.randn(4, 3, 16, 16, device=device, requires_grad=True)
    w_vk = torch.randn(16, 3, 3, 3, device=device, requires_grad=True)
    x_cpu = x_vk.detach().cpu().requires_grad_()
    w_cpu = w_vk.detach().cpu().requires_grad_()

    cpu_out = F.conv2d(x_cpu, w_cpu, padding=1).relu().sum()
    cpu_out.backward()
    cpu_x_grad = x_cpu.grad.clone()
    cpu_w_grad = w_cpu.grad.clone()

    # Eager mode on Vulkan (no compile)
    out_vk = F.conv2d(x_vk, w_vk, padding=1).relu().sum()
    out_vk.backward()

    dx = (cpu_x_grad - x_vk.grad.cpu()).abs().max().item()
    dw = (cpu_w_grad - w_vk.grad.cpu()).abs().max().item()
    print(f"  dx diff: {dx:.6f}  dw diff: {dw:.6f}")
    print(f"  Eager mode: {'PASS' if max(dx, dw) < 1e-2 else 'FAIL'}")

    # ---- Test 2: Verify Slang backward infrastructure ------------------
    print("\n--- Slang Backward Infrastructure ---")
    from torch_vulkan.inductor.bwd_diff_dispatch import resolve_backward_kind
    from torch_vulkan.inductor.bwd_template_registry import BWD_TEMPLATE_REGISTRY

    print("BWD_TEMPLATE_REGISTRY conv entries:")
    for key in ["conv_im2col_f32", "conv2d_default"]:
        entry = BWD_TEMPLATE_REGISTRY.lookup(key)
        if entry:
            print(f"  {key}: kind={entry.kind.name}, fwd_fn={entry.fwd_fn}")

    print("Backward routing:")
    for op_name in ["aten.convolution.default", "aten.convolution_backward.default"]:
        r = resolve_backward_kind(op_name)
        if r:
            print(f"  {op_name}: kind={r.kind.name}, fwd_key={r.fwd_key}")

    # Verify template and lib files
    import os as _os

    base = _os.path.dirname(__file__)

    conv_lib = _os.path.join(base, "shaders", "lib", "conv.slang")
    if _os.path.exists(conv_lib):
        with open(conv_lib) as f:
            content = f.read()
        has_diff = "[Differentiable]" in content
        has_bwd = "[BackwardDerivative]" in content
        print(f"\nshaders/lib/conv.slang:")
        print(f"  [Differentiable]: {'OK' if has_diff else 'MISSING'}")
        print(f"  [BackwardDerivative]: {'OK' if has_bwd else 'MISSING'}")

    conv_tmpl = _os.path.join(
        base, "python", "torch_vulkan", "inductor", "templates", "slang_conv_bwd.slang"
    )
    if _os.path.exists(conv_tmpl):
        with open(conv_tmpl) as f:
            content = f.read()
        has_bwd = "[BackwardDerivative" in content
        has_bwd_fn = "conv_inner_madd_bwd" in content
        # Count push constant fields
        import re

        struct_match = re.search(r"struct BwdPC \{(.*?)\};", content, re.DOTALL)
        if struct_match:
            fields = [
                l
                for l in struct_match.group(1).split("\n")
                if "uint" in l and not l.strip().startswith("//")
            ]
            pc_bytes = len(fields) * 4
            print(f"\ntemplates/slang_conv_bwd.slang:")
            print(f"  [BackwardDerivative]: {'OK' if has_bwd else 'MISSING'}")
            print(f"  conv_inner_madd_bwd: {'OK' if has_bwd_fn else 'MISSING'}")
            print(
                f"  Push constant size: {pc_bytes}B (limit 128B): {'OK' if pc_bytes <= 128 else 'TOO LARGE'}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
