"""Dispatch-builder CPU overhead audit (#7).

Profiles the per-dispatch CPU cost in dispatch_shader() by running a
high-dispatch-count benchmark and printing the timing breakdown for:

  (a) descriptor set writes per dispatch (vkUpdateDescriptorSets)
  (b) buffer-dirty set lookup in smart-barrier insertion
  (c) pipeline cache lookup (lock-free fast path)

Usage:
    TORCH_VULKAN_PROFILE_DISPATCH=1 python benchmarks/profile_dispatch.py \\
        [--model mlp|transformer_block|mnist_cnn] [--steps 50] [--train]

Without TORCH_VULKAN_PROFILE_DISPATCH=1, runs a lightweight dispatch-count
summary (no timing breakdown).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Callable, Optional

import torch
import torch.nn as nn


def _reset_all():
    """Reset perf counters and profile timers."""
    import torch_vulkan

    torch_vulkan._c_ext._reset_perf_counters()
    if torch_vulkan._c_ext._profiling_enabled():
        torch_vulkan._c_ext._reset_profile_timers()


def _get_profile_breakdown(disp_count: int) -> dict:
    """Return per-dispatch timing breakdown (ns)."""
    import torch_vulkan

    ext = torch_vulkan._c_ext
    if not ext._profiling_enabled():
        return {}

    d = float(disp_count) if disp_count > 0 else 1.0
    return {
        "pipeline_cache_ns": round(ext._profile_pipeline_cache_ns() / d, 1),
        "get_runtime_ns": round(ext._profile_get_runtime_ns() / d, 1),
        "desc_alloc_ns": round(ext._profile_desc_alloc_ns() / d, 1),
        "buffer_info_ns": round(ext._profile_buffer_info_ns() / d, 1),
        "desc_write_ns": round(ext._profile_desc_write_ns() / d, 1),
        "barrier_check_ns": round(ext._profile_barrier_check_ns() / d, 1),
        "cmd_record_ns": round(ext._profile_cmd_record_ns() / d, 1),
        "dirty_track_ns": round(ext._profile_dirty_track_ns() / d, 1),
        "total_ns": round(
            sum(
                [
                    ext._profile_pipeline_cache_ns(),
                    ext._profile_get_runtime_ns(),
                    ext._profile_desc_alloc_ns(),
                    ext._profile_buffer_info_ns(),
                    ext._profile_desc_write_ns(),
                    ext._profile_barrier_check_ns(),
                    ext._profile_cmd_record_ns(),
                    ext._profile_dirty_track_ns(),
                ]
            )
            / d,
            1,
        ),
    }


def _build_models() -> dict[str, Callable[[], nn.Module]]:
    """Return a registry of models for profiling."""

    def _mlp():
        return nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 64),
        )

    def _mnist_cnn():
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def _transformer_block():
        class Block(nn.Module):
            def __init__(self, dim=128, heads=4):
                super().__init__()
                self.norm1 = nn.RMSNorm(dim)
                self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
                self.norm2 = nn.RMSNorm(dim)
                self.mlp = nn.Sequential(
                    nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
                )

            def forward(self, x):
                h = self.norm1(x)
                h, _ = self.attn(h, h, h, need_weights=False)
                x = x + h
                x = x + self.mlp(self.norm2(x))
                return x

        return Block()

    def _matmul_heavy():
        """Synthetic: many chained matmuls to stress dispatch path."""
        return nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
        )

    return {
        "mlp": _mlp,
        "mnist_cnn": _mnist_cnn,
        "transformer_block": _transformer_block,
        "matmul_heavy": _matmul_heavy,
    }


def _make_inputs(model: nn.Module, train: bool) -> tuple:
    """Create sensible inputs based on first layer type."""
    # Try to infer from first parameterized layer
    first_linear = None
    first_conv = None
    for m in model.modules():
        if first_linear is None and isinstance(m, nn.Linear):
            first_linear = m
        if first_conv is None and isinstance(m, nn.Conv2d):
            first_conv = m
        if first_linear and first_conv:
            break

    if first_conv is not None:
        # CNN input
        return (torch.randn(8, 1, 28, 28),)
    elif first_linear is not None:
        in_features = first_linear.in_features
        # transformer_block has MultiheadAttention with embed_dim
        batch = 2
        return (torch.randn(batch, 32, in_features),)
    else:
        return (torch.randn(4, 256),)


def run_profile(
    model_name: str,
    steps: int,
    warmup: int,
    train: bool,
) -> dict:
    models = _build_models()
    if model_name not in models:
        raise SystemExit(f"Unknown model {model_name!r}; choices: {list(models)}")

    import torch_vulkan
    import torch_vulkan.inductor  # noqa: F401

    profiling = torch_vulkan._c_ext._profiling_enabled()
    if not profiling:
        print("NOTE: Set TORCH_VULKAN_PROFILE_DISPATCH=1 for timing breakdown.\n")

    # Build model
    model = models[model_name]()
    if train:
        model.train()
    else:
        model.eval()
    model = model.to("vulkan:0")
    if not train:
        for p in model.parameters():
            p.requires_grad_(False)

    inputs = tuple(t.to("vulkan:0") for t in _make_inputs(model, train))
    compiled_fn = torch.compile(model, backend="inductor")

    # Warmup
    for _ in range(warmup):
        _reset_all()
        if train:
            out = compiled_fn(*inputs)
            if isinstance(out, torch.Tensor):
                loss = out.sum() if out.dim() > 0 else out
            else:
                loss = out[0].sum() if out[0].dim() > 0 else out[0]
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
        else:
            compiled_fn(*inputs)
        torch.empty(1, device="vulkan:0").cpu()

    # Timed steps
    disp_counts = []
    wall_times_ms = []
    profile_breakdowns = []

    for step in range(steps):
        _reset_all()
        t0 = time.perf_counter()

        if train:
            out = compiled_fn(*inputs)
            if isinstance(out, torch.Tensor):
                loss = out.sum() if out.dim() > 0 else out
            else:
                loss = out[0].sum() if out[0].dim() > 0 else out[0]
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
        else:
            compiled_fn(*inputs)

        torch.empty(1, device="vulkan:0").cpu()
        dt = (time.perf_counter() - t0) * 1000

        d = int(torch_vulkan._c_ext._get_dispatch_count())
        disp_counts.append(d)
        wall_times_ms.append(dt)

        if profiling:
            profile_breakdowns.append(_get_profile_breakdown(d))

    # Aggregate
    med_disp = statistics.median(disp_counts)
    med_ms = statistics.median(wall_times_ms)

    result = {
        "model": model_name,
        "train": train,
        "steps": steps,
        "warmup": warmup,
        "profiling_enabled": profiling,
        "dispatch_count_median": med_disp,
        "wall_ms_median": round(med_ms, 3),
        "per_dispatch_us": round((med_ms * 1000) / med_disp, 2) if med_disp else None,
    }

    if profiling and profile_breakdowns:
        # Average the per-step breakdowns
        keys = profile_breakdowns[0].keys()
        avg_breakdown = {}
        for k in keys:
            vals = [b[k] for b in profile_breakdowns]
            avg_breakdown[k] = round(statistics.mean(vals), 1)
        result["per_dispatch_ns"] = avg_breakdown
        # Also print per-step data for variability
        result["step_breakdowns"] = profile_breakdowns

    if not profiling:
        result["note"] = (
            "Set TORCH_VULKAN_PROFILE_DISPATCH=1 for per-component timing breakdown."
        )

    return result


def print_report(result: dict) -> None:
    print(
        f"Dispatch-builder overhead audit — {result['model']} "
        f"({'train' if result['train'] else 'inference'})"
    )
    print(f"  steps: {result['steps']} (warmup: {result['warmup']})")
    print(f"  profiling: {'ON' if result['profiling_enabled'] else 'OFF'}")
    print(f"  dispatches/step (median): {result['dispatch_count_median']}")
    print(f"  wall time/step (median): {result['wall_ms_median']} ms")
    if result.get("per_dispatch_us"):
        print(f"  per-dispatch wall time:   {result['per_dispatch_us']} µs")
    print()

    breakdown = result.get("per_dispatch_ns")
    if breakdown:
        print("  Per-dispatch CPU timing breakdown (ns):")
        print(f"    pipeline_cache:  {breakdown['pipeline_cache_ns']:>8.1f}")
        print(f"    get_runtime:     {breakdown['get_runtime_ns']:>8.1f}")
        print(f"    desc_alloc:      {breakdown['desc_alloc_ns']:>8.1f}")
        print(f"    buffer_info:     {breakdown['buffer_info_ns']:>8.1f}")
        print(f"    desc_write:      {breakdown['desc_write_ns']:>8.1f}")
        print(f"    barrier_check:   {breakdown['barrier_check_ns']:>8.1f}")
        print(f"    cmd_record:      {breakdown['cmd_record_ns']:>8.1f}")
        print(f"    dirty_track:     {breakdown['dirty_track_ns']:>8.1f}")
        print(f"    ──────────────────────────")
        print(
            f"    TOTAL:           {breakdown['total_ns']:>8.1f} ns = {breakdown['total_ns'] / 1000:.2f} µs"
        )
        print()

        # Identify bottleneck
        components = [
            ("pipeline_cache_ns", "pipeline cache lookup"),
            ("get_runtime_ns", "runtime (stream+pool) get"),
            ("desc_alloc_ns", "descriptor set alloc (vkAllocateDescriptorSets)"),
            ("buffer_info_ns", "buffer info extraction (get_buffer_info)"),
            ("desc_write_ns", "descriptor writes (vkUpdateDescriptorSets)"),
            ("barrier_check_ns", "smart barrier check (dirty set lookup)"),
            ("cmd_record_ns", "cmd recording (bind+dispatch)"),
            ("dirty_track_ns", "dirty buffer tracking (insert+track)"),
        ]
        max_comp = max(components, key=lambda c: breakdown[c[0]])
        print(
            f"  ⚠ Bottleneck: {max_comp[1]} at {breakdown[max_comp[0]]:.1f} ns/dispatch "
            f"({breakdown[max_comp[0]] / breakdown['total_ns'] * 100:.0f}%)"
        )
        print()
    elif result.get("note"):
        print(f"  {result['note']}")


def main():
    p = argparse.ArgumentParser(description="Dispatch-builder CPU overhead audit (#7)")
    p.add_argument(
        "--model",
        default="transformer_block",
        choices=list(_build_models()),
        help="Model to benchmark",
    )
    p.add_argument("--steps", type=int, default=30, help="Timed steps")
    p.add_argument("--warmup", type=int, default=5, help="Warmup steps")
    p.add_argument(
        "--train", action="store_true", help="Forward+backward (default: fwd only)"
    )
    p.add_argument("--json", action="store_true", help="JSON output only")
    args = p.parse_args()

    result = run_profile(args.model, args.steps, args.warmup, args.train)

    if args.json:
        # Strip step_breakdowns for cleaner JSON
        result.pop("step_breakdowns", None)
        print(json.dumps(result, indent=2, default=str))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
