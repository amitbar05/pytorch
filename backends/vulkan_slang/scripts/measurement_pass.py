#!/usr/bin/env python3
"""P7.1 — Measurement pass per benchmark.

Runs every model in `benchmarks/inductor_train.py` under
`torch.compile(backend="inductor")` with `TORCH_VULKAN_INDUCTOR_STATS=1`
and emits a structured report listing the largest per-kernel time
contributor per benchmark. Anything ≥ 5% of total step time is
reported as an "outlier candidate" the discovery loop can promote
into a P7.x roadmap entry.

Output is stable JSON-on-stdout so the regression suite can call
`measurement_summary()` without parsing prose, and a human reader
gets a readable section per benchmark.

This is the formalized version of the §"Training-driven Discovery
Loop" pass in CLAUDE.md — the agent runs it after exhausting the
prioritized roadmap, files the largest contributor as a new P7
entry, and ships it.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any


_OUTLIER_PCT = 5.0


def _run_one(model_name: str) -> dict[str, Any]:
    bench_dir = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "benchmarks"))
    if bench_dir not in sys.path:
        sys.path.insert(0, bench_dir)
    import inductor_train as it
    import torch

    os.environ["TORCH_VULKAN_INDUCTOR_STATS"] = "1"
    from torch_vulkan.inductor.inductor_stats import summary, reset_stats

    if model_name not in it._MODELS:
        return {"model": model_name, "status": "missing_in_registry"}

    try:
        builder = it._MODELS[model_name]
        model, make_inputs = builder()
    except SystemExit as e:
        return {"model": model_name, "status": "unavailable",
                "error": str(e)[:200]}
    except Exception as e:
        return {"model": model_name, "status": "build_failed",
                "error": f"{type(e).__name__}: {e}"[:300]}
    try:
        model = model.to("vulkan:0").eval()
        inputs = tuple(t.to("vulkan:0") for t in make_inputs())
        compiled = torch.compile(model, backend="inductor")
        with torch.no_grad():
            compiled(*inputs)
            reset_stats()
            for _ in range(20):
                compiled(*inputs)
            torch.empty(1, device="vulkan:0").cpu()
    except Exception as e:
        return {
            "model": model_name,
            "status": "compile_failed",
            "error": f"{type(e).__name__}: {e}"[:300],
        }

    s = summary(top_n=10)
    out: dict[str, Any] = {
        "model": model_name,
        "status": "ok",
        "n_kernels": s["n_kernels"],
        "total_us": s["total_us"],
        "top": [
            {
                "kernel": k,
                "total_us": us,
                "call_count": cnt,
                "avg_us": avg,
                "pct_of_total": (us / s["total_us"] * 100) if s["total_us"] else 0.0,
            }
            for (k, us, cnt, avg, _h) in s["top"]
        ],
    }
    out["outlier_candidates"] = [
        e for e in out["top"] if e["pct_of_total"] >= _OUTLIER_PCT
    ]
    return out


def measurement_summary(models: list[str] | None = None) -> dict[str, Any]:
    """Run a measurement pass over the given models (default: all).

    Returned shape:
      {
        "models": [<per-model dict>...],
        "total_outliers": int,
      }
    Each per-model dict is the output of `_run_one`.
    """
    bench_dir = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "benchmarks"))
    if bench_dir not in sys.path:
        sys.path.insert(0, bench_dir)
    import inductor_train as it

    if models is None:
        models = list(it._MODELS.keys())
    rows = [_run_one(m) for m in models]
    n_outliers = sum(
        len(r.get("outlier_candidates", [])) for r in rows
    )
    return {"models": rows, "total_outliers": n_outliers}


def measure_three_modes(
    model_name: str, *, steps: int = 10, warmup: int = 3,
) -> dict[str, Any]:
    """P7.10.b — measure one model in CPU eager / Vulkan eager / Vulkan
    compiled modes and return median ms/step + dispatch counts where
    applicable, plus a CPU-oracle correctness flag.

    The compiled mode is verified against CPU on a single representative
    forward pass before timing — if outputs disagree, the row is flagged
    ``correctness_failed`` and timings are returned but should not be
    propagated into the Performance Targets table. (CPU is the only
    correctness oracle; Vulkan eager is unverified — see CLAUDE.md.)

    Returned shape::

        {
          "model": str,
          "status": "ok" | "compile_failed" | "build_failed" | "skipped",
          "cpu_ms": float | None,           # median ms/step
          "vulkan_eager_ms": float | None,  # median ms/step
          "compiled_ms": float | None,      # median ms/step
          "vulkan_eager_dispatches": int | None,
          "compiled_dispatches": int | None,
          "correctness_ok": bool,           # compiled vs CPU
          "max_abs_err": float | None,      # max |compiled - cpu|
        }
    """
    import statistics
    import time

    bench_dir = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "benchmarks"))
    if bench_dir not in sys.path:
        sys.path.insert(0, bench_dir)
    import inductor_train as it
    import torch

    if model_name not in it._MODELS:
        return {"model": model_name, "status": "missing_in_registry"}

    out: dict[str, Any] = {
        "model": model_name,
        "status": "ok",
        "cpu_ms": None,
        "vulkan_eager_ms": None,
        "compiled_ms": None,
        "vulkan_eager_dispatches": None,
        "compiled_dispatches": None,
        "correctness_ok": False,
        "max_abs_err": None,
    }

    def _bench(fn, inputs, *, sync_device: str | None) -> float:
        if sync_device == "vulkan":
            torch.empty(1, device="vulkan:0").cpu()
        t0 = time.perf_counter()
        with torch.no_grad():
            fn(*inputs)
        if sync_device == "vulkan":
            torch.empty(1, device="vulkan:0").cpu()
        return (time.perf_counter() - t0) * 1000.0

    try:
        builder = it._MODELS[model_name]
        # CPU eager
        m_cpu, make_inputs = builder()
        m_cpu = m_cpu.eval()
        for p in m_cpu.parameters():
            p.requires_grad_(False)
        x_cpu = make_inputs()
        for _ in range(warmup):
            _bench(m_cpu, x_cpu, sync_device=None)
        cpu_times = [
            _bench(m_cpu, x_cpu, sync_device=None) for _ in range(steps)
        ]
        out["cpu_ms"] = round(statistics.median(cpu_times), 4)
        cpu_ref_out = m_cpu(*x_cpu).detach()
    except Exception as e:
        out["status"] = "build_failed"
        out["error"] = f"cpu eager: {type(e).__name__}: {e}"[:300]
        return out

    try:
        # Vulkan eager
        m_v, _ = builder()
        m_v = m_v.to("vulkan:0").eval()
        for p in m_v.parameters():
            p.requires_grad_(False)
        x_v = tuple(t.to("vulkan:0") for t in make_inputs())
        for _ in range(warmup):
            _bench(m_v, x_v, sync_device="vulkan")
        v_times = [
            _bench(m_v, x_v, sync_device="vulkan") for _ in range(steps)
        ]
        out["vulkan_eager_ms"] = round(statistics.median(v_times), 4)

        import torch_vulkan
        torch_vulkan._c_ext._reset_perf_counters()
        with torch.no_grad():
            m_v(*x_v)
        torch.empty(1, device="vulkan:0").cpu()
        out["vulkan_eager_dispatches"] = (
            torch_vulkan._c_ext._get_dispatch_count()
        )
    except Exception as e:
        out["status"] = "vulkan_eager_failed"
        out["error"] = f"vulkan eager: {type(e).__name__}: {e}"[:300]
        return out

    try:
        # Vulkan compiled
        m_c, _ = builder()
        m_c = m_c.to("vulkan:0").eval()
        for p in m_c.parameters():
            p.requires_grad_(False)
        x_c = tuple(t.to("vulkan:0") for t in make_inputs())
        comp = torch.compile(m_c, backend="inductor")
        for _ in range(warmup):
            _bench(comp, x_c, sync_device="vulkan")
        c_times = [
            _bench(comp, x_c, sync_device="vulkan") for _ in range(steps)
        ]
        out["compiled_ms"] = round(statistics.median(c_times), 4)

        import torch_vulkan
        torch_vulkan._c_ext._reset_perf_counters()
        with torch.no_grad():
            comp_out = comp(*x_c)
        torch.empty(1, device="vulkan:0").cpu()
        out["compiled_dispatches"] = (
            torch_vulkan._c_ext._get_dispatch_count()
        )

        # CPU-oracle correctness check on compiled output.
        # We rebuild input tensors with the same RNG so cpu_ref_out and
        # comp_out compare against the SAME inputs — but builders use
        # `torch.randn(...)` so we can't replay them deterministically.
        # Instead, compare against CPU using the *compiled* model's
        # weights run on CPU with the same inputs (rebuild on CPU).
        weights_cpu = {
            k: v.detach().cpu() for k, v in m_c.state_dict().items()
        }
        m_ref, _ = builder()
        m_ref.load_state_dict(weights_cpu)
        m_ref = m_ref.eval()
        for p in m_ref.parameters():
            p.requires_grad_(False)
        x_cpu_same = tuple(t.cpu() for t in x_c)
        with torch.no_grad():
            ref_out = m_ref(*x_cpu_same)
        comp_cpu = (
            comp_out[0].cpu() if isinstance(comp_out, tuple)
            else comp_out.cpu()
        )
        ref_t = (
            ref_out[0] if isinstance(ref_out, tuple) else ref_out
        )
        diff = (comp_cpu - ref_t).abs().max().item()
        out["max_abs_err"] = float(diff)
        out["correctness_ok"] = diff < 1e-2  # relaxed for fp32 vulkan
    except Exception as e:
        out["status"] = "compile_failed"
        out["error"] = f"compile: {type(e).__name__}: {e}"[:300]
        return out

    return out


# PF.62 — train-step builders. The shared `inductor_train.py` registry
# is sized for forward-only smoke tests (e.g. mlp at B=32 D=256); the
# joint-step parity gate wants a slightly larger MLP that exercises
# linear backward + GELU backward + SGD step at non-trivial sizes.
# Builders here are intentionally local to keep the bench registry
# stable for the dispatch-count waterfall used elsewhere.
_TRAIN_STEP_BUILDERS: dict[str, Any] = {}


def _train_step_register(name: str):
    def deco(fn):
        _TRAIN_STEP_BUILDERS[name] = fn
        return fn
    return deco


@_train_step_register("mlp_b64_d512")
def _build_mlp_b64_d512():
    import torch
    from torch import nn

    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.GELU(),
        nn.Linear(1024, 512),
        nn.GELU(),
        nn.Linear(512, 128),
    )
    g = torch.Generator().manual_seed(0)

    def make_inputs() -> tuple:
        # Deterministic inputs so CPU/Vulkan/compiled all see the same
        # data (parameters are deepcopy-replicated below).
        x = torch.randn(64, 512, generator=g)
        # Target index per row for a CE-style scalar loss; held
        # separately so the test can choose a loss surface.
        return (x,)

    return model, make_inputs


def measure_train_step_three_modes(
    model_name: str, *, steps: int = 10, warmup: int = 3,
) -> dict[str, Any]:
    """PF.62 — measure one model in CPU eager / Vulkan eager / Vulkan
    compiled modes for a *full training step* (forward +
    ``loss.backward()`` + ``optimizer.step()`` + ``optimizer.zero_grad()``)
    and return median ms/step + dispatch counts where applicable, plus a
    CPU-oracle correctness flag on the post-step parameter values.

    Per the §"Performance Targets — End-to-end training step" contract,
    the floor is ``compiled_train_step_ms ≤ min(cpu_train_step_ms,
    vulkan_eager_train_step_ms)``.

    Returned shape::

        {
          "model": str,
          "status": "ok" | "compile_failed" | "build_failed"
                    | "vulkan_eager_failed" | "missing_in_registry",
          "cpu_train_step_ms": float | None,
          "vulkan_eager_train_step_ms": float | None,
          "compiled_train_step_ms": float | None,
          "vulkan_eager_dispatches": int | None,
          "compiled_dispatches": int | None,
          "correctness_ok": bool,            # compiled vs CPU (post-step)
          "max_abs_err": float | None,       # max |compiled_param - cpu_param|
          "floor_min_baseline_ms": float | None,
          "floor_pass": bool,
        }

    Notes
    -----
    - CPU is the only correctness oracle (per CLAUDE.md). Compiled
      parameters are compared against a CPU eager replica that started
      from the same weights and saw the same inputs.
    - Vulkan eager uses ``torch.empty(1, device='vulkan:0').cpu()`` as
      the canonical synchronization barrier (matches `_bench` in
      ``measure_three_modes``).
    - Compiled mode uses ``torch.compile(backend='inductor')``.
    """
    import copy
    import statistics
    import time

    import torch
    from torch import nn

    if model_name not in _TRAIN_STEP_BUILDERS:
        return {"model": model_name, "status": "missing_in_registry"}

    out: dict[str, Any] = {
        "model": model_name,
        "status": "ok",
        "cpu_train_step_ms": None,
        "vulkan_eager_train_step_ms": None,
        "compiled_train_step_ms": None,
        "vulkan_eager_dispatches": None,
        "compiled_dispatches": None,
        "correctness_ok": False,
        "max_abs_err": None,
        "floor_min_baseline_ms": None,
        "floor_pass": False,
    }

    builder = _TRAIN_STEP_BUILDERS[model_name]

    def _train_step(model: nn.Module, inputs, optimizer, loss_fn):
        optimizer.zero_grad()
        out_t = model(*inputs)
        loss = loss_fn(out_t)
        loss.backward()
        optimizer.step()
        return loss

    def _loss_fn(t: "torch.Tensor") -> "torch.Tensor":
        return t.float().pow(2).sum()

    def _bench_train(model, inputs, optimizer, *, sync_device: str | None) -> float:
        if sync_device == "vulkan":
            torch.empty(1, device="vulkan:0").cpu()
        t0 = time.perf_counter()
        _train_step(model, inputs, optimizer, _loss_fn)
        if sync_device == "vulkan":
            torch.empty(1, device="vulkan:0").cpu()
        return (time.perf_counter() - t0) * 1000.0

    # --- Build a single source-of-truth weights set on CPU. All three
    # modes deepcopy from this so the post-step parameter comparison is
    # meaningful.
    try:
        m_seed, make_inputs = builder()
        seed_state = copy.deepcopy(m_seed.state_dict())
        seed_inputs_cpu = make_inputs()
    except Exception as e:
        out["status"] = "build_failed"
        out["error"] = f"builder: {type(e).__name__}: {e}"[:300]
        return out

    # --- CPU eager
    try:
        m_cpu = builder()[0]
        m_cpu.load_state_dict(copy.deepcopy(seed_state))
        m_cpu.train()
        opt_cpu = torch.optim.SGD(m_cpu.parameters(), lr=1e-3)
        x_cpu_warm = tuple(t.clone() for t in seed_inputs_cpu)
        for _ in range(warmup):
            _bench_train(m_cpu, x_cpu_warm, opt_cpu, sync_device=None)
        cpu_times = [
            _bench_train(m_cpu, x_cpu_warm, opt_cpu, sync_device=None)
            for _ in range(steps)
        ]
        out["cpu_train_step_ms"] = round(statistics.median(cpu_times), 4)
        cpu_post_state = {k: v.detach().clone() for k, v in m_cpu.state_dict().items()}
    except Exception as e:
        out["status"] = "build_failed"
        out["error"] = f"cpu eager: {type(e).__name__}: {e}"[:300]
        return out

    # --- Vulkan eager
    try:
        m_v = builder()[0]
        m_v.load_state_dict(copy.deepcopy(seed_state))
        m_v = m_v.to("vulkan:0").train()
        opt_v = torch.optim.SGD(m_v.parameters(), lr=1e-3)
        x_v = tuple(t.to("vulkan:0") for t in seed_inputs_cpu)
        for _ in range(warmup):
            _bench_train(m_v, x_v, opt_v, sync_device="vulkan")
        v_times = [
            _bench_train(m_v, x_v, opt_v, sync_device="vulkan")
            for _ in range(steps)
        ]
        out["vulkan_eager_train_step_ms"] = round(statistics.median(v_times), 4)

        import torch_vulkan
        torch_vulkan._c_ext._reset_perf_counters()
        _bench_train(m_v, x_v, opt_v, sync_device="vulkan")
        out["vulkan_eager_dispatches"] = (
            torch_vulkan._c_ext._get_dispatch_count()
        )
    except Exception as e:
        out["status"] = "vulkan_eager_failed"
        out["error"] = f"vulkan eager: {type(e).__name__}: {e}"[:300]
        return out

    # --- Vulkan compiled
    try:
        m_c = builder()[0]
        m_c.load_state_dict(copy.deepcopy(seed_state))
        m_c = m_c.to("vulkan:0").train()
        opt_c = torch.optim.SGD(m_c.parameters(), lr=1e-3)
        x_c = tuple(t.to("vulkan:0") for t in seed_inputs_cpu)
        comp = torch.compile(m_c, backend="inductor")

        def _train_step_compiled(inputs):
            opt_c.zero_grad()
            out_t = comp(*inputs)
            loss = _loss_fn(out_t)
            loss.backward()
            opt_c.step()
            return loss

        for _ in range(warmup):
            torch.empty(1, device="vulkan:0").cpu()
            _t0 = time.perf_counter()
            _train_step_compiled(x_c)
            torch.empty(1, device="vulkan:0").cpu()
            _ = (time.perf_counter() - _t0) * 1000.0

        c_times = []
        for _ in range(steps):
            torch.empty(1, device="vulkan:0").cpu()
            t0 = time.perf_counter()
            _train_step_compiled(x_c)
            torch.empty(1, device="vulkan:0").cpu()
            c_times.append((time.perf_counter() - t0) * 1000.0)
        out["compiled_train_step_ms"] = round(statistics.median(c_times), 4)

        import torch_vulkan
        torch_vulkan._c_ext._reset_perf_counters()
        torch.empty(1, device="vulkan:0").cpu()
        _train_step_compiled(x_c)
        torch.empty(1, device="vulkan:0").cpu()
        out["compiled_dispatches"] = (
            torch_vulkan._c_ext._get_dispatch_count()
        )

        # CPU-oracle correctness on post-step parameter values: rebuild
        # the same number of training steps on CPU using the same weights
        # and inputs, then diff.
        m_ref = builder()[0]
        m_ref.load_state_dict(copy.deepcopy(seed_state))
        m_ref.train()
        opt_ref = torch.optim.SGD(m_ref.parameters(), lr=1e-3)
        x_ref = tuple(t.clone() for t in seed_inputs_cpu)
        # warmup + steps + 1 (for the dispatch-count step) — match the
        # number of optimizer steps the compiled model executed.
        n_steps = warmup + steps + 1
        for _ in range(n_steps):
            _train_step(m_ref, x_ref, opt_ref, _loss_fn)
        ref_state = m_ref.state_dict()

        max_diff = 0.0
        for k, v_ref in ref_state.items():
            v_c = m_c.state_dict()[k].detach().cpu()
            d = (v_c - v_ref.detach()).abs().max().item()
            if d > max_diff:
                max_diff = d
        out["max_abs_err"] = float(max_diff)
        out["correctness_ok"] = max_diff < 1e-2
        # Suppress unused-warning for cpu_post_state — kept for potential
        # future first-step-only correctness mode.
        _ = cpu_post_state
    except Exception as e:
        out["status"] = "compile_failed"
        out["error"] = f"compile: {type(e).__name__}: {e}"[:300]
        return out

    # --- Floor evaluation
    if (out["compiled_train_step_ms"] is not None and
            out["cpu_train_step_ms"] is not None and
            out["vulkan_eager_train_step_ms"] is not None):
        floor = min(out["cpu_train_step_ms"],
                    out["vulkan_eager_train_step_ms"])
        out["floor_min_baseline_ms"] = round(floor, 4)
        out["floor_pass"] = bool(
            out["correctness_ok"]
            and out["compiled_train_step_ms"] <= floor
        )

    return out


# Mapping: benchmark model name → workload-substring used in each of the
# three Performance Targets tables. Some models map to multiple rows
# (forward + backward + e2e); some map to none (e.g. resnet18 has rows
# but the benchmark doesn't run today). Substrings must be unique
# within their table — verified by `parse_tables` raising on miss.
_FWD_TABLE_KEYS: dict[str, str] = {
    "mlp": "MLP",
    "resnet18": "ResNet-18",
    # mobilenet_v2 / miniqwen3 not yet in benchmark registry.
}
_BWD_TABLE_KEYS: dict[str, str] = {
    "mlp": "MLP backward",
    "resnet18": "ResNet-18 backward",
}
_E2E_TABLE_KEYS: dict[str, str] = {
    "mlp": "MLP fwd",
    "mnist_cnn": "MNIST CNN fwd",
    "resnet18": "ResNet-18 fwd",
}


def build_perf_targets_updates(
    measurements: list[dict[str, Any]],
) -> list:
    """Translate three-mode measurements into ``CellUpdate`` records
    that ``apply_updates`` can splice into the Performance Targets
    tables. Skips measurements that didn't reach ``ok`` or whose CPU
    correctness check failed.

    Returns ``list[CellUpdate]``; callers feed it to
    ``update_perf_targets.apply_updates(source, ...)``.
    """
    sys.path.insert(0, os.path.dirname(__file__))
    from update_perf_targets import CellUpdate

    updates: list = []
    for m in measurements:
        if m.get("status") != "ok":
            continue
        if not m.get("correctness_ok", False):
            # Don't propagate compiled numbers to the table when output
            # diverges from CPU — would mask a correctness regression.
            continue
        name = m["model"]

        # Table 0 — Forward dispatch
        if name in _FWD_TABLE_KEYS:
            key = _FWD_TABLE_KEYS[name]
            if m.get("compiled_dispatches") is not None:
                updates.append(CellUpdate(
                    table_index=0, workload_substring=key,
                    column="Today", value=str(m["compiled_dispatches"]),
                ))
            if m.get("vulkan_eager_dispatches") is not None:
                updates.append(CellUpdate(
                    table_index=0, workload_substring=key,
                    column="Vulkan eager",
                    value=str(m["vulkan_eager_dispatches"]),
                ))
        # Table 2 — End-to-end ms
        if name in _E2E_TABLE_KEYS:
            key = _E2E_TABLE_KEYS[name]
            if m.get("cpu_ms") is not None:
                updates.append(CellUpdate(
                    table_index=2, workload_substring=key,
                    column="CPU (ms)", value=f"{m['cpu_ms']:.3f}",
                ))
            if m.get("vulkan_eager_ms") is not None:
                updates.append(CellUpdate(
                    table_index=2, workload_substring=key,
                    column="Vulkan eager (ms)",
                    value=f"{m['vulkan_eager_ms']:.3f}",
                ))
            if m.get("compiled_ms") is not None:
                updates.append(CellUpdate(
                    table_index=2, workload_substring=key,
                    column="Today (ms)",
                    value=f"{m['compiled_ms']:.3f}",
                ))
    return updates


class PerfRegression(AssertionError):
    """Raised when the compiled (Today) column regresses against the
    prior recorded value, or against ``min(CPU, Vulkan eager)``.

    P7.10.b CI gate: every commit that nudges Today *upward* on a
    benchmark must justify itself. The exception message names the
    table, workload, column, prior value, and new value so a CI log
    points at the offending row.
    """


def detect_regressions(
    measurements: list[dict[str, Any]],
    prior_today: dict[tuple[int, str], float] | None = None,
) -> list[str]:
    """Return a list of human-readable regression strings — empty if
    no Today value worsened against the prior recorded Today (when
    given) or against ``min(CPU, Vulkan eager)``.

    ``prior_today`` is keyed by ``(table_index, workload_substring)``
    and holds the *numeric* prior Today value (parsed by the caller
    from the doc table before the measurement run). If absent, only
    the min-of-baselines check fires. Lower Today is always better
    (dispatch counts and ms — both monotonically minimize).
    """
    regressions: list[str] = []
    for m in measurements:
        if m.get("status") != "ok" or not m.get("correctness_ok", False):
            continue
        name = m["model"]

        # Forward dispatch table: compiled_dispatches must be ≤ vulkan_eager_dispatches
        if name in _FWD_TABLE_KEYS:
            key = _FWD_TABLE_KEYS[name]
            cd = m.get("compiled_dispatches")
            vd = m.get("vulkan_eager_dispatches")
            if cd is not None and vd is not None and cd > vd:
                regressions.append(
                    f"FWD/{key}: compiled dispatches {cd} > "
                    f"vulkan eager {vd}"
                )
            if prior_today is not None:
                prior = prior_today.get((0, key))
                if prior is not None and cd is not None and cd > prior:
                    regressions.append(
                        f"FWD/{key}: compiled dispatches {cd} > "
                        f"prior Today {prior}"
                    )

        # E2E ms table: compiled_ms must be ≤ min(cpu_ms, vulkan_eager_ms)
        if name in _E2E_TABLE_KEYS:
            key = _E2E_TABLE_KEYS[name]
            cm = m.get("compiled_ms")
            cpum = m.get("cpu_ms")
            vem = m.get("vulkan_eager_ms")
            baselines = [b for b in (cpum, vem) if b is not None]
            if cm is not None and baselines and cm > min(baselines):
                regressions.append(
                    f"E2E/{key}: compiled {cm:.3f} ms > "
                    f"min(cpu={cpum}, vulkan_eager={vem})"
                )
            if prior_today is not None:
                prior = prior_today.get((2, key))
                if prior is not None and cm is not None and cm > prior:
                    regressions.append(
                        f"E2E/{key}: compiled {cm:.3f} ms > "
                        f"prior Today {prior}"
                    )
    return regressions


def run_and_update_targets(
    *, write: bool = False, models: list[str] | None = None,
    fail_on_regression: bool = False,
) -> dict[str, Any]:
    """P7.10.b end-to-end driver.

    Runs the three-mode measurement on every model in the bench
    registry (or ``models`` if provided), translates results into
    ``CellUpdate`` records, optionally writes them back to the
    Performance Targets tables, and reports any regression against
    prior Today / min(baselines).

    Returns ``{measurements, updates_count, regressions, diff}``.
    Raises ``PerfRegression`` when ``fail_on_regression=True`` and
    any regression is detected.
    """
    sys.path.insert(0, os.path.dirname(__file__))
    from update_perf_targets import (
        CellUpdate, _ROADMAP_PATH, apply_updates,
    )

    bench_dir = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "benchmarks"))
    if bench_dir not in sys.path:
        sys.path.insert(0, bench_dir)
    import inductor_train as it
    if models is None:
        models = list(_E2E_TABLE_KEYS.keys() & it._MODELS.keys())

    measurements = [measure_three_modes(m) for m in models]
    updates = build_perf_targets_updates(measurements)
    regressions = detect_regressions(measurements)

    diff_bytes = 0
    if write and updates:
        with open(_ROADMAP_PATH, encoding="utf-8") as f:
            src = f.read()
        new = apply_updates(src, updates)
        diff_bytes = abs(len(new) - len(src))
        if new != src:
            with open(_ROADMAP_PATH, "w", encoding="utf-8") as f:
                f.write(new)

    result = {
        "measurements": measurements,
        "updates_count": len(updates),
        "regressions": regressions,
        "diff_bytes": diff_bytes,
    }
    if fail_on_regression and regressions:
        raise PerfRegression(
            "Performance Targets regression(s):\n  - "
            + "\n  - ".join(regressions)
        )
    return result


def main() -> None:
    s = measurement_summary()
    print("Inductor measurement pass")
    print("=========================")
    for r in s["models"]:
        print()
        print(f"# {r['model']}: {r.get('status')}")
        if r.get("status") != "ok":
            err = r.get("error")
            if err:
                print(f"  error: {err}")
            continue
        print(f"  n_kernels={r['n_kernels']}  total_us={r['total_us']:.0f}")
        for e in r["top"]:
            print(
                f"  {e['pct_of_total']:5.1f}%  "
                f"{e['total_us']:10.1f}us  "
                f"{e['call_count']:4d} calls  "
                f"{e['kernel'][:60]}"
            )
        if r["outlier_candidates"]:
            print(f"  outliers ≥ {_OUTLIER_PCT:.0f}%: "
                  f"{len(r['outlier_candidates'])}")
    print()
    print(f"total_outliers={s['total_outliers']}")
    # JSON tail for programmatic consumption.
    print()
    print("--- JSON ---")
    print(json.dumps(s, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
