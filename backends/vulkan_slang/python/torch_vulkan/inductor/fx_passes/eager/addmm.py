"""Addmm / scaled-bmm custom-op registrations for fused FX-pattern targets.

Each ``_ensure_*`` function idempotently registers a ``torch_vulkan::*``
custom_op so Inductor has a valid OpOverload target to replace matched
subgraphs with.
"""

from __future__ import annotations


def _ensure_addmm_gelu_op_registered() -> "object":
    """Register `torch_vulkan::addmm_gelu_fused` (PF.5).

    Pattern target for `_fuse_addmm_gelu`. Eager backing dispatches the
    fused tiled-addmm+gelu Slang shader (single GPU dispatch for
    `gelu(a @ b + bias)`). Falls back to two ops on shape/dtype mismatch.
    """
    import torch

    op_name = "torch_vulkan::addmm_gelu_fused"
    existing = getattr(torch.ops.torch_vulkan, "addmm_gelu_fused", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    Tensor = torch.Tensor

    def _addmm_gelu_impl(bias: Tensor, mat1: Tensor, mat2: Tensor) -> Tensor:
        from ...templates.caller import _pick_addmm_gelu_tile

        if (
            mat1.device.type != "vulkan"
            or mat1.dtype not in (torch.float32, torch.float16)
            or mat1.dim() != 2
            or mat2.dim() != 2
            or bias.dim() != 1
        ):
            return torch.nn.functional.gelu(torch.addmm(bias, mat1, mat2))

        caller = _pick_addmm_gelu_tile(mat1.shape[0], mat2.shape[1], mat1.shape[1])
        return caller(bias, mat1, mat2)

    _addmm_gelu_impl.__annotations__ = {
        "bias": Tensor,
        "mat1": Tensor,
        "mat2": Tensor,
        "return": Tensor,
    }
    fused_op = torch.library.custom_op(op_name, mutates_args=())(_addmm_gelu_impl)

    def _addmm_gelu_fake(bias, mat1, mat2):
        return mat1.new_empty((mat1.shape[0], mat2.shape[1]))

    fused_op.register_fake(_addmm_gelu_fake)
    return torch.ops.torch_vulkan.addmm_gelu_fused.default


def _ensure_scaled_bmm_op_registered() -> "object":
    """Register `torch_vulkan::scaled_bmm` as a torch custom_op exactly once.

    The FX pass needs an OpOverload as the rewrite target — Inductor's
    lowering machinery rejects plain Python functions. Wrapping the eager
    dispatch in a custom_op (with a fake_impl for shape inference) makes
    the post-rewrite graph compileable end-to-end. Returns the OpOverload.
    """
    import torch

    op_name = "torch_vulkan::scaled_bmm"
    existing = getattr(torch.ops.torch_vulkan, "scaled_bmm", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    Tensor = torch.Tensor

    def _scaled_bmm_impl(q: Tensor, k: Tensor, scale: float) -> Tensor:
        import torch_vulkan

        return torch_vulkan.scaled_bmm(q, k, scale)

    _scaled_bmm_impl.__annotations__ = {
        "q": Tensor,
        "k": Tensor,
        "scale": float,
        "return": Tensor,
    }
    scaled_bmm = torch.library.custom_op(op_name, mutates_args=())(_scaled_bmm_impl)

    def _scaled_bmm_fake(q, k, scale):
        # bmm(q, k.T): q is (B, M, K); k is (B, N, K) → output (B, M, N).
        b, m, _ = q.shape
        _, n, _ = k.shape
        return q.new_empty((b, m, n))

    scaled_bmm.register_fake(_scaled_bmm_fake)

    return torch.ops.torch_vulkan.scaled_bmm.default
