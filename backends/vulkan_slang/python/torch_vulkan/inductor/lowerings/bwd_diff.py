"""Bwd-diff custom-op registration factories (PF.6.b / PF.11).

Creates ``torch_vulkan::*_bwd_diff`` eager custom_ops that route backward
graphs through ``bwd_diff_dispatch.py`` at runtime. These are registered
eagerly at backend startup so they're available when cached compiled graphs
are loaded from disk (the lowering phase is skipped on cache hit).
"""

from __future__ import annotations

_UNARY_BWD_DIFF_OPS = frozenset(
    {
        "aten.relu_backward",
        "aten.sigmoid_backward",
        "aten.tanh_backward",
        "aten.gelu_backward",
        "aten.silu_backward",
        "aten.elu_backward",
        "aten.hardswish_backward",
        "aten.hardsigmoid_backward",
        # M-AG5.1 Tier-2 (2026-05-22): aten.softplus_backward removed —
        # the unary bwd_diff custom-op signature ``(grad_output, self)``
        # cannot carry the ``beta``/``threshold`` Scalar params the aten
        # op requires. softplus_backward lowers algebraically via
        # ``bwd_lowerings.py``.
        "aten.mish_backward",
    }
)

_BINARY_LOSS_BWD_DIFF_OPS = frozenset(
    {
        "aten.mse_loss_backward",
        "aten.binary_cross_entropy_backward",
        "aten.binary_cross_entropy_with_logits_backward",
        "aten.smooth_l1_loss_backward",
        "aten.huber_loss_backward",
    }
)

_unary_bwd_diff_cache: dict[str, object] = {}
_binary_loss_bwd_diff_cache: dict[str, object] = {}


def _ensure_unary_bwd_diff_op(aten_op: str):
    """GAP 4.1 / PF.6.b.iii — generic bwd_diff custom_op factory for unary backwards.

    Mirrors ``_ensure_silu_backward_bwd_diff_op_registered`` but works for any
    unary backward in ``BWD_DIFF_TABLE``. Returns the ``OpOverload`` for use
    with ``L.fallback_handler``. Idempotent — returns cached op on subsequent
    calls.
    """
    if aten_op in _unary_bwd_diff_cache:
        return _unary_bwd_diff_cache[aten_op]

    import torch

    short = aten_op.split(".")[-1]
    ns_name = f"{short}_bwd_diff"
    op_name = f"torch_vulkan::{ns_name}"
    existing = getattr(torch.ops.torch_vulkan, ns_name, None)
    if existing is not None:
        _unary_bwd_diff_cache[aten_op] = existing.default
        return existing.default

    Tensor = torch.Tensor

    def _impl(grad_output: Tensor, self_: Tensor) -> Tensor:
        from torch_vulkan.inductor.bwd_diff_dispatch import dispatch_unary_bwd

        return dispatch_unary_bwd(aten_op, self_, grad_output)

    _impl.__annotations__ = {
        "grad_output": Tensor,
        "self_": Tensor,
        "return": Tensor,
    }
    op = torch.library.custom_op(op_name, mutates_args=())(_impl)

    def _fake(grad_output, self_):
        return torch.empty_like(self_)

    op.register_fake(_fake)
    overload = getattr(torch.ops.torch_vulkan, ns_name).default
    _unary_bwd_diff_cache[aten_op] = overload
    return overload


def _ensure_binary_loss_bwd_diff_op(aten_op: str):
    """GAP 4.1 — generic bwd_diff custom_op factory for binary loss backwards.

    Creates a custom_op that calls ``dispatch_binary_bwd`` and returns only
    ``grad_a`` (the grad w.r.t. the first input, typically ``self``). The
    second input's grad (w.r.t. ``target``) is discarded since loss backward
    ops only return grad_input.

    For ops with ``no_diff_params`` (e.g. smooth_l1 has ``beta``, huber has
    ``delta``), the generated custom_op accepts trailing float args that are
    forwarded as ``no_diff_kwargs`` to the dispatcher (T3.2).
    """
    if aten_op in _binary_loss_bwd_diff_cache:
        return _binary_loss_bwd_diff_cache[aten_op]

    import torch

    from torch_vulkan.inductor.bwd_diff_table import BWD_DIFF_TABLE

    entry = BWD_DIFF_TABLE[aten_op]
    short = aten_op.split(".")[-1]
    ns_name = f"{short}_bwd_diff"
    op_name = f"torch_vulkan::{ns_name}"
    existing = getattr(torch.ops.torch_vulkan, ns_name, None)
    if existing is not None:
        _binary_loss_bwd_diff_cache[aten_op] = existing.default
        return existing.default

    no_diff_params = entry.no_diff_params
    param_names = ", ".join(no_diff_params)

    if no_diff_params:
        ns: dict = {"torch": torch}
        impl_src = (
            f"def _impl(grad_output, self_, target, {param_names}):\n"
            f"    from torch_vulkan.inductor.bwd_diff_dispatch import "
            f"dispatch_binary_bwd\n"
            f"    _kwargs = dict(zip(\n"
            f"        ({no_diff_params!r}), ({(', '.join(no_diff_params))})))\n"
            f"    grad_a, _ = dispatch_binary_bwd(\n"
            f"        {aten_op!r}, self_, target, grad_output,\n"
            f"        no_diff_kwargs=_kwargs)\n"
            f"    return grad_a\n"
        )
        fake_src = (
            f"def _fake(grad_output, self_, target, {param_names}):\n"
            f"    return torch.empty_like(self_)\n"
        )
        exec(impl_src + fake_src, ns)
        _impl = ns["_impl"]
        _fake = ns["_fake"]
        ann = {
            "grad_output": torch.Tensor,
            "self_": torch.Tensor,
            "target": torch.Tensor,
            "return": torch.Tensor,
        }
        for p in no_diff_params:
            ann[p] = float
        _impl.__annotations__ = ann
    else:

        def _impl(grad_output, self_, target):
            from torch_vulkan.inductor.bwd_diff_dispatch import dispatch_binary_bwd

            grad_a, _ = dispatch_binary_bwd(aten_op, self_, target, grad_output)
            return grad_a

        _impl.__annotations__ = {
            "grad_output": torch.Tensor,
            "self_": torch.Tensor,
            "target": torch.Tensor,
            "return": torch.Tensor,
        }

        def _fake(grad_output, self_, target):
            return torch.empty_like(self_)

    op = torch.library.custom_op(op_name, mutates_args=())(_impl)
    op.register_fake(_fake)
    overload = getattr(torch.ops.torch_vulkan, ns_name).default
    _binary_loss_bwd_diff_cache[aten_op] = overload
    return overload
