"""Foreach optimizer custom-op registrations (T4.8).

M24 fix: torch.library.infer_schema() rejects PEP 604 unions like
``float | list[float]``. Per-param scalar lists must be ``list[float]``
(or ``Sequence[float]`` — both are accepted). Single-scalar callers
normalize to ``[scalar] * n_params`` at the boundary before calling.

All foreach ops mutate ``params`` (and optionally momentum/m/v buffers)
in-place. We declare them via ``mutates_args=("params", ...)`` so that
Inductor's functionalization understands the alias contract.
"""

from __future__ import annotations


def _ensure_foreach_sgd_step_op_registered() -> "object":
    """Register `torch_vulkan::foreach_sgd_step` (T4.8).

    Vanilla SGD with optional weight decay over a list of parameters:
        p_i := p_i - lr_i * (g_i + wd_i * p_i)   for i in range(N)

    All scalar args are `list[float]` length N. Single-scalar callers
    normalize at the boundary.
    """
    import torch

    op_name = "torch_vulkan::foreach_sgd_step"
    existing = getattr(torch.ops.torch_vulkan, "foreach_sgd_step", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    Tensor = torch.Tensor

    def _foreach_sgd_step_impl(
        params: list[Tensor],
        grads: list[Tensor],
        lr: list[float],
        weight_decay: list[float],
    ) -> None:
        from ...templates.caller import _pick_foreach_optimizer_caller

        n = len(params)
        if n == 0:
            return
        # Boundary normalization — accept length-1 broadcasts.
        if len(lr) == 1 and n > 1:
            lr = list(lr) * n
        if len(weight_decay) == 1 and n > 1:
            weight_decay = list(weight_decay) * n
        # Vulkan path: dispatch the template (one shader for all params).
        if params[0].device.type == "vulkan":
            caller = _pick_foreach_optimizer_caller("sgd", n, "float")
            caller(
                list(params),
                list(grads),
                lr=list(lr),
                weight_decay=list(weight_decay),
            )
            return
        # CPU/other path: standard SGD update.
        for p, g, l, wd in zip(params, grads, lr, weight_decay):
            if wd != 0.0:
                p.add_(p, alpha=wd)  # in-place: g_eff = g + wd*p
            p.add_(g, alpha=-l)

    _foreach_sgd_step_impl.__annotations__ = {
        "params": list[Tensor],
        "grads": list[Tensor],
        "lr": list[float],
        "weight_decay": list[float],
        "return": None,
    }
    sgd_op = torch.library.custom_op(op_name, mutates_args=("params",))(
        _foreach_sgd_step_impl
    )

    def _foreach_sgd_step_fake(params, grads, lr, weight_decay):
        return None

    sgd_op.register_fake(_foreach_sgd_step_fake)
    return torch.ops.torch_vulkan.foreach_sgd_step.default


def _ensure_foreach_sgd_momentum_step_op_registered() -> "object":
    """Register `torch_vulkan::foreach_sgd_momentum_step` (T4.8).

    SGD + momentum:
        buf_i := momentum_i * buf_i + g_i
        p_i   := p_i - lr_i * buf_i
    """
    import torch

    op_name = "torch_vulkan::foreach_sgd_momentum_step"
    existing = getattr(torch.ops.torch_vulkan, "foreach_sgd_momentum_step", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    Tensor = torch.Tensor

    def _impl(
        params: list[Tensor],
        grads: list[Tensor],
        momentum_bufs: list[Tensor],
        lr: list[float],
        weight_decay: list[float],
        momentum: list[float],
    ) -> None:
        from ...templates.caller import _pick_foreach_optimizer_caller

        n = len(params)
        if n == 0:
            return
        if len(lr) == 1 and n > 1:
            lr = list(lr) * n
        if len(weight_decay) == 1 and n > 1:
            weight_decay = list(weight_decay) * n
        if len(momentum) == 1 and n > 1:
            momentum = list(momentum) * n
        if params[0].device.type == "vulkan":
            caller = _pick_foreach_optimizer_caller("sgd_momentum", n, "float")
            caller(
                list(params),
                list(grads),
                lr=list(lr),
                weight_decay=list(weight_decay),
                momentum=list(momentum),
                momentum_bufs=list(momentum_bufs),
            )
            return
        for p, g, buf, l, wd, m in zip(
            params, grads, momentum_bufs, lr, weight_decay, momentum
        ):
            g_eff = g.clone()
            if wd != 0.0:
                g_eff = g_eff.add(p, alpha=wd)
            buf.mul_(m).add_(g_eff)
            p.add_(buf, alpha=-l)

    _impl.__annotations__ = {
        "params": list[Tensor],
        "grads": list[Tensor],
        "momentum_bufs": list[Tensor],
        "lr": list[float],
        "weight_decay": list[float],
        "momentum": list[float],
        "return": None,
    }
    op = torch.library.custom_op(op_name, mutates_args=("params", "momentum_bufs"))(
        _impl
    )

    def _fake(params, grads, momentum_bufs, lr, weight_decay, momentum):
        return None

    op.register_fake(_fake)
    return torch.ops.torch_vulkan.foreach_sgd_momentum_step.default


def _ensure_foreach_adamw_step_op_registered() -> "object":
    """Register `torch_vulkan::foreach_adamw_step` (T4.8).

    AdamW (decoupled weight decay):
        m_i := beta1_i * m_i + (1 - beta1_i) * g_i
        v_i := beta2_i * v_i + (1 - beta2_i) * g_i^2
        p_i := p_i - lr_i * (m_i / (sqrt(v_i) + eps_i) + wd_i * p_i)
    """
    import torch

    op_name = "torch_vulkan::foreach_adamw_step"
    existing = getattr(torch.ops.torch_vulkan, "foreach_adamw_step", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    Tensor = torch.Tensor

    def _impl(
        params: list[Tensor],
        grads: list[Tensor],
        m_bufs: list[Tensor],
        v_bufs: list[Tensor],
        lr: list[float],
        weight_decay: list[float],
        beta1: list[float],
        beta2: list[float],
        eps: list[float],
    ) -> None:
        from ...templates.caller import _pick_foreach_optimizer_caller

        n = len(params)
        if n == 0:
            return
        # Length-1 broadcast normalization at the boundary.
        if len(lr) == 1 and n > 1:
            lr = list(lr) * n
        if len(weight_decay) == 1 and n > 1:
            weight_decay = list(weight_decay) * n
        if len(beta1) == 1 and n > 1:
            beta1 = list(beta1) * n
        if len(beta2) == 1 and n > 1:
            beta2 = list(beta2) * n
        if len(eps) == 1 and n > 1:
            eps = list(eps) * n
        if params[0].device.type == "vulkan":
            caller = _pick_foreach_optimizer_caller("adamw", n, "float")
            # Template uses `momentum` slot for beta1.
            caller(
                list(params),
                list(grads),
                lr=list(lr),
                weight_decay=list(weight_decay),
                momentum=list(beta1),
                beta2=list(beta2),
                eps=list(eps),
                momentum_bufs=list(m_bufs),
                v_bufs=list(v_bufs),
            )
            return
        # CPU reference.
        for p, g, m, v, l, wd, b1, b2, e in zip(
            params, grads, m_bufs, v_bufs, lr, weight_decay, beta1, beta2, eps
        ):
            m.mul_(b1).add_(g, alpha=1.0 - b1)
            v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
            denom = v.sqrt().add_(e)
            update = m / denom
            if wd != 0.0:
                update = update.add(p, alpha=wd)
            p.add_(update, alpha=-l)

    _impl.__annotations__ = {
        "params": list[Tensor],
        "grads": list[Tensor],
        "m_bufs": list[Tensor],
        "v_bufs": list[Tensor],
        "lr": list[float],
        "weight_decay": list[float],
        "beta1": list[float],
        "beta2": list[float],
        "eps": list[float],
        "return": None,
    }
    op = torch.library.custom_op(op_name, mutates_args=("params", "m_bufs", "v_bufs"))(
        _impl
    )

    def _fake(params, grads, m_bufs, v_bufs, lr, weight_decay, beta1, beta2, eps):
        return None

    op.register_fake(_fake)
    return torch.ops.torch_vulkan.foreach_adamw_step.default


def _ensure_foreach_lion_step_op_registered() -> "object":
    """Register `torch_vulkan::foreach_lion_step` (T4.8).

    Lion (EvoLved Sign Momentum):
        update = beta1_i * momentum_i + (1 - beta1_i) * g_i
        p_i   := p_i - lr_i * sign(update)
        momentum_i := beta2_i * momentum_i + (1 - beta2_i) * g_i
    """
    import torch

    op_name = "torch_vulkan::foreach_lion_step"
    existing = getattr(torch.ops.torch_vulkan, "foreach_lion_step", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    Tensor = torch.Tensor

    def _impl(
        params: list[Tensor],
        grads: list[Tensor],
        momentum_bufs: list[Tensor],
        lr: list[float],
        weight_decay: list[float],
        beta1: list[float],
        beta2: list[float],
    ) -> None:
        from ...templates.caller import _pick_foreach_optimizer_caller

        n = len(params)
        if n == 0:
            return
        if len(lr) == 1 and n > 1:
            lr = list(lr) * n
        if len(weight_decay) == 1 and n > 1:
            weight_decay = list(weight_decay) * n
        if len(beta1) == 1 and n > 1:
            beta1 = list(beta1) * n
        if len(beta2) == 1 and n > 1:
            beta2 = list(beta2) * n
        if params[0].device.type == "vulkan":
            caller = _pick_foreach_optimizer_caller("lion", n, "float")
            caller(
                list(params),
                list(grads),
                lr=list(lr),
                weight_decay=list(weight_decay),
                momentum=list(beta1),
                beta2=list(beta2),
                momentum_bufs=list(momentum_bufs),
            )
            return
        for p, g, mom, l, wd, b1, b2 in zip(
            params, grads, momentum_bufs, lr, weight_decay, beta1, beta2
        ):
            update = mom.mul(b1).add_(g, alpha=1.0 - b1)
            p.add_(update.sign(), alpha=-l)
            if wd != 0.0:
                p.add_(p, alpha=-l * wd)
            mom.mul_(b2).add_(g, alpha=1.0 - b2)

    _impl.__annotations__ = {
        "params": list[Tensor],
        "grads": list[Tensor],
        "momentum_bufs": list[Tensor],
        "lr": list[float],
        "weight_decay": list[float],
        "beta1": list[float],
        "beta2": list[float],
        "return": None,
    }
    op = torch.library.custom_op(op_name, mutates_args=("params", "momentum_bufs"))(
        _impl
    )

    def _fake(params, grads, momentum_bufs, lr, weight_decay, beta1, beta2):
        return None

    op.register_fake(_fake)
    return torch.ops.torch_vulkan.foreach_lion_step.default
