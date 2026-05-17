"""Vulkan-optimized optimizers.

Provides drop-in replacements for torch.optim classes that route
through fused Slang foreach kernels instead of per-parameter eager dispatch.
"""

from __future__ import annotations

import math
from typing import Any

import torch

_ops_ensured: bool = False


def _ensure_optimizer_ops() -> None:
    """Make sure the foreach optimizer custom ops are registered.

    install_external_optimizer() registers
    torch.ops.torch_vulkan.foreach_{sgd,sgd_momentum,adamw,lion}_step
    and is safe to call multiple times (no-op after first call).
    """
    global _ops_ensured
    if _ops_ensured:
        return
    try:
        from torch_vulkan.inductor.templates.caller.optimizer import (
            install_external_optimizer,
        )

        install_external_optimizer()
    except ImportError:
        pass
    _ops_ensured = True


class AdamW(torch.optim.AdamW):
    """AdamW optimizer that uses the fused Slang foreach_adamw_step kernel.

    When all parameters are on ``device="vulkan"`` and dtype is float32,
    ``step()`` dispatches to ``torch.ops.torch_vulkan.foreach_adamw_step``
    in a single fused dispatch instead of N per-parameter eager dispatches.
    Falls back to ``super().step()`` for non-Vulkan, mixed-device, or
    unsupported configurations (amsgrad, maximize, differentiable).

    Bias correction and decoupled weight decay are handled in Python
    before the kernel call, so the kernel receives bias-corrected
    effective lr/eps and zero weight_decay.  The result numerically
    matches ``torch.optim.AdamW`` on float32 Vulkan tensors.
    """

    def step(self, closure: Any = None) -> Any:
        """Perform a single optimization step.

        Uses the fused Slang foreach_adamw_step kernel when all
        parameters are Vulkan float32 tensors with default settings
        (no amsgrad, no maximize, no differentiable).  Falls back to
        the standard PyTorch AdamW implementation otherwise.
        """
        _ensure_optimizer_ops()

        # ── Pre-check: can we use the fused path? ──────────────────
        # We must check before calling closure, so we don't double-call
        # it if we fall back to super().step(closure).
        use_fused = True
        for group in self.param_groups:
            # Unsupported features — fall back.
            if group.get("amsgrad", False):
                use_fused = False
                break
            if group.get("maximize", False):
                use_fused = False
                break
            if group.get("differentiable", False):
                use_fused = False
                break
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.device.type != "vulkan" or p.dtype != torch.float32:
                    use_fused = False
                    break
            if not use_fused:
                break

        if not use_fused:
            return super().step(closure)

        # ── Fused Slang path ───────────────────────────────────────
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # torch.no_grad(): the parent Adam.step() uses
        # @_use_grad_for_differentiable which calls
        # torch.set_grad_enabled(self.defaults["differentiable"])
        # (i.e. False when differentiable=False).  We mirror that
        # contract so in-place mutations on leaf tensors are legal.
        with torch.no_grad():
            for group in self.param_groups:
                params_with_grad: list[torch.Tensor] = []
                grads: list[torch.Tensor] = []
                m_bufs: list[torch.Tensor] = []
                v_bufs: list[torch.Tensor] = []

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    # Lazy state initialization — compatible with parent
                    # state dict keys: 'step', 'exp_avg', 'exp_avg_sq'
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = torch.tensor(0.0)
                        state["exp_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                    params_with_grad.append(p)
                    grads.append(p.grad)
                    m_bufs.append(state["exp_avg"])
                    v_bufs.append(state["exp_avg_sq"])
                    state["step"] += 1

                if not params_with_grad:
                    continue

                beta1, beta2 = group["betas"]
                lr = group["lr"]
                weight_decay = group.get("weight_decay", 0.0)
                eps = group["eps"]

                # ── bias correction ──────────────────────────────────
                # The Slang kernel computes:
                #   m = β₁·m + (1-β₁)·g
                #   v = β₂·v + (1-β₂)·g²
                #   p = p - lr_k · (m / (√v + ε_k) + wd_k · p)
                #
                # torch.optim.AdamW uses bias-corrected estimates:
                #   denom = √(v / (1-β₂ᵗ)) + ε
                #   p = p - (lr / (1-β₁ᵗ)) · m / denom - lr·wd·p
                #
                # These are equivalent iff:
                #   lr_k = lr · √(1-β₂ᵗ) / (1-β₁ᵗ)
                #   ε_k  = ε · √(1-β₂ᵗ)
                #   wd_k = 0  (applied in Python before the kernel)
                # ─────────────────────────────────────────────────────
                step_t = self.state[params_with_grad[0]]["step"].item()
                bias_correction1 = 1.0 - beta1**step_t
                bias_correction2 = 1.0 - beta2**step_t
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                lr_eff = lr * bias_correction2_sqrt / bias_correction1
                eps_eff = eps * bias_correction2_sqrt

                # ── decoupled weight decay ───────────────────────────
                # torch.optim.AdamW: param.mul_(1 - lr * weight_decay)
                # before the Adam step.  Apply the same pre-scaling here
                # and pass weight_decay=0 to the kernel.
                if weight_decay != 0.0:
                    for p in params_with_grad:
                        p.mul_(1.0 - lr * weight_decay)

                # ── fused Slang dispatch ─────────────────────────────
                # Single Vulkan dispatch for all params in this group.
                torch.ops.torch_vulkan.foreach_adamw_step(
                    params_with_grad,
                    grads,
                    m_bufs,
                    v_bufs,
                    [lr_eff],  # length-1 broadcasts to all params
                    [0.0],  # weight_decay already pre-applied
                    [beta1],
                    [beta2],
                    [eps_eff],
                )

        return loss
