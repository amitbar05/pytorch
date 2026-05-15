"""SwiGLU custom-op registration for fused FX-pattern targets."""

from __future__ import annotations


def _ensure_swiglu_op_registered() -> "object":
    """Register `torch_vulkan::swiglu_fused` as a torch custom_op exactly once.

    Mirrors the scaled_bmm registration: the FX rewrite target must be an
    OpOverload, and we wrap the eager `torch_vulkan.swiglu` extern with a
    fake_impl so Inductor's shape inference accepts the rewritten graph.
    """
    import torch

    op_name = "torch_vulkan::swiglu_fused"
    existing = getattr(torch.ops.torch_vulkan, "swiglu_fused", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    Tensor = torch.Tensor

    def _swiglu_impl(gate: Tensor, up: Tensor) -> Tensor:
        import torch_vulkan

        return torch_vulkan.swiglu(gate, up)

    _swiglu_impl.__annotations__ = {"gate": Tensor, "up": Tensor, "return": Tensor}
    swiglu_op = torch.library.custom_op(op_name, mutates_args=())(_swiglu_impl)

    def _swiglu_fake(gate, up):
        return gate.new_empty(gate.shape)

    swiglu_op.register_fake(_swiglu_fake)
    return torch.ops.torch_vulkan.swiglu_fused.default
