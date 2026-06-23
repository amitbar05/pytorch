"""QKV projection custom-op registration for fused FX-pattern targets."""

from __future__ import annotations


def _ensure_qkv_cat_op_registered() -> "object":
    """Register ``torch_vulkan::qkv_cat3`` — concatenate 3 tensors along a dim
    via the eager Vulkan dispatch. We need this as an extern OpOverload so the
    QKV FX pass can inject a weight-pack node without going through Inductor's
    cat IR lowering, which crashes on Vulkan (masked-op `dtype=str` bug).
    """
    import torch

    op_name = "torch_vulkan::qkv_cat3"
    existing = getattr(torch.ops.torch_vulkan, "qkv_cat3", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    Tensor = torch.Tensor

    def _cat3_impl(a: Tensor, b: Tensor, c: Tensor, dim: int) -> Tensor:
        return torch.cat([a, b, c], dim=dim)

    _cat3_impl.__annotations__ = {
        "a": Tensor,
        "b": Tensor,
        "c": Tensor,
        "dim": int,
        "return": Tensor,
    }
    cat3_op = torch.library.custom_op(op_name, mutates_args=())(_cat3_impl)

    def _cat3_fake(a, b, c, dim):
        out_shape = list(a.shape)
        out_shape[dim] = a.shape[dim] + b.shape[dim] + c.shape[dim]
        return a.new_empty(out_shape)

    cat3_op.register_fake(_cat3_fake)
    return torch.ops.torch_vulkan.qkv_cat3.default
