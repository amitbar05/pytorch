"""Inductor backend training-step benchmark runner.

Runs N steps of a workload first eagerly then under torch.compile and prints:
- dispatches / step
- ms / step
- speedup vs eager

Usage:
    SLANGC=... python benchmarks/inductor_train.py \
        --model mlp [--steps 50] [--warmup 5]

The model registry below is intentionally minimal — extend with new model
constructors as roadmap items target them.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import time
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp() -> tuple[nn.Module, Callable[[], tuple[torch.Tensor, ...]]]:
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.GELU(),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Linear(256, 64),
    )

    def make_inputs() -> tuple[torch.Tensor, ...]:
        return (torch.randn(32, 256),)

    return model, make_inputs


def _resnet18() -> tuple[nn.Module, Callable[[], tuple[torch.Tensor, ...]]]:
    try:
        from torchvision.models import resnet18
    except ImportError:
        raise SystemExit("torchvision not installed; pip install torchvision")
    model = resnet18(num_classes=10)
    return model, lambda: (torch.randn(4, 3, 64, 64),)


def _mnist_cnn() -> tuple[nn.Module, Callable[[], tuple[torch.Tensor, ...]]]:
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 10),
    )
    return model, lambda: (torch.randn(8, 1, 28, 28),)


def _transformer_block() -> tuple[nn.Module, Callable[[], tuple[torch.Tensor, ...]]]:
    class Block(nn.Module):
        def __init__(self, dim: int = 128, heads: int = 4):
            super().__init__()
            self.norm1 = nn.RMSNorm(dim)
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.norm2 = nn.RMSNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.norm1(x)
            h, _ = self.attn(h, h, h, need_weights=False)
            x = x + h
            x = x + self.mlp(self.norm2(x))
            return x

    return Block(), lambda: (torch.randn(2, 32, 128),)


_MODELS: dict[str, Callable[[], tuple[nn.Module, Callable[[], tuple[torch.Tensor, ...]]]]] = {
    "mlp": _mlp,
    "mnist_cnn": _mnist_cnn,
    "resnet18": _resnet18,
    "transformer_block": _transformer_block,
}


def _to_device(model: nn.Module, inputs: tuple[torch.Tensor, ...], device: str):
    model = model.to(device)
    inputs = tuple(t.to(device) for t in inputs)
    return model, inputs


@contextlib.contextmanager
def _no_grad_or_train(train: bool):
    if train:
        yield
    else:
        with torch.no_grad():
            yield


def _bench_step(
    fn: Callable, inputs, train: bool, model: nn.Module
) -> tuple[float, int]:
    """One step: forward (+ optional backward). Returns (ms, dispatches)."""
    import torch_vulkan
    torch_vulkan._c_ext._reset_perf_counters()
    t0 = time.perf_counter()
    if train:
        out = fn(*inputs)
        if not isinstance(out, torch.Tensor):
            out = out[0] if isinstance(out, tuple) else out
        loss = out.sum() if out.dim() > 0 else out
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
    else:
        fn(*inputs)
    # Force GPU sync via a CPU read on a small tensor.
    torch.empty(1, device="vulkan:0").cpu()
    dt = (time.perf_counter() - t0) * 1000
    d = torch_vulkan._c_ext._get_dispatch_count()
    return dt, d


def run(model_name: str, steps: int, warmup: int, train: bool) -> dict:
    if model_name not in _MODELS:
        raise SystemExit(
            f"unknown model {model_name!r}; choices: {list(_MODELS)}"
        )
    builder = _MODELS[model_name]

    # Eager pass
    model_e, make_inputs = builder()
    if train:
        model_e.train()
    else:
        model_e.eval()
    model_e, eager_inputs = _to_device(model_e, make_inputs(), "vulkan:0")
    if not train:
        for p in model_e.parameters():
            p.requires_grad_(False)

    eager_fn = model_e.__call__
    for _ in range(warmup):
        with _no_grad_or_train(train):
            _bench_step(eager_fn, eager_inputs, train, model_e)
    eager_times = []
    eager_disp = []
    for _ in range(steps):
        with _no_grad_or_train(train):
            ms, d = _bench_step(eager_fn, eager_inputs, train, model_e)
        eager_times.append(ms)
        eager_disp.append(d)

    # Compiled pass
    model_c, _ = builder()
    if train:
        model_c.train()
    else:
        model_c.eval()
    model_c, compiled_inputs = _to_device(model_c, make_inputs(), "vulkan:0")
    if not train:
        for p in model_c.parameters():
            p.requires_grad_(False)
    compiled_fn = torch.compile(model_c, backend="inductor")

    # Compile + warmup
    for _ in range(warmup):
        with _no_grad_or_train(train):
            _bench_step(compiled_fn, compiled_inputs, train, model_c)
    compiled_times = []
    compiled_disp = []
    for _ in range(steps):
        with _no_grad_or_train(train):
            ms, d = _bench_step(compiled_fn, compiled_inputs, train, model_c)
        compiled_times.append(ms)
        compiled_disp.append(d)

    eager_med = statistics.median(eager_times)
    compiled_med = statistics.median(compiled_times)
    return {
        "model": model_name,
        "train": train,
        "steps": steps,
        "warmup": warmup,
        "eager_ms_median": round(eager_med, 3),
        "compiled_ms_median": round(compiled_med, 3),
        "speedup": round(eager_med / compiled_med, 3) if compiled_med else None,
        "eager_dispatches": eager_disp[0] if eager_disp else None,
        "compiled_dispatches": compiled_disp[0] if compiled_disp else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlp", choices=list(_MODELS))
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--train", action="store_true",
                   help="run forward+backward (default: forward only)")
    p.add_argument("--json", action="store_true",
                   help="emit JSON only (for scripting)")
    args = p.parse_args()

    import torch_vulkan
    import torch_vulkan.inductor  # registers the backend  # noqa: F401

    result = run(args.model, args.steps, args.warmup, args.train)
    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"model:               {result['model']}")
    print(f"mode:                {'train' if result['train'] else 'inference'}")
    print(f"steps:               {result['steps']} (after {result['warmup']} warmup)")
    print(f"eager dispatches:    {result['eager_dispatches']}")
    print(f"compiled dispatches: {result['compiled_dispatches']}")
    print(f"eager ms (median):   {result['eager_ms_median']}")
    print(f"compiled ms (med.):  {result['compiled_ms_median']}")
    print(f"speedup:             {result['speedup']}x")


if __name__ == "__main__":
    main()
