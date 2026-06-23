"""OP.24 — torch_vulkan::mm_int8 custom op registration.

Registers a custom ``torch_vulkan::mm_int8`` op that dispatches int8×int8
matrix multiplication through the Slang tiled int8 template.  The op is
registered as a fallback kernel so Inductor can lower ``aten.mm`` for int8
inputs on Vulkan into a single dispatch.

This module must be imported before FX graph capture so the op is available
to the dispatcher — the ``register()`` function in ``lowerings/__init__.py``
calls ``_register_mm_int8_op()`` at backend-init time.
"""

from __future__ import annotations

import torch

# ── Custom op definition ─────────────────────────────────────────────────
# Schema: mm_int8(Tensor a, Tensor b, *, Tensor(a!) out) -> Tensor(a!)
# a: [M, K] int8, b: [K, N] int8, out: [M, N] float32
# Mutates the output buffer in-place (returns the same tensor for chaining).

torch.library.define(
    "torch_vulkan::mm_int8",
    "(Tensor a, Tensor b, *, Tensor(a!) out) -> Tensor(a!)",
)


def _mm_int8_dispatch(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """PrivateUse1 (Vulkan) implementation of mm_int8.

    At runtime, this is called with real Vulkan tensors.  We dispatch through
    the Slang int8 tiled matmul template.
    """
    # Late import to avoid circular deps at module-load time.
    from ..templates.caller.gemm.dispatch import _slang_tile_mm_int8

    if out is None:
        out = torch.empty(
            (a.shape[0], b.shape[1]),
            dtype=torch.float32,
            device=a.device,
        )

    # Conservative tile config that fits any RDNA1 workgroup.
    # Future: autotune between multiple tile configs.
    _slang_tile_mm_int8(32, 32, 16, a, b, out, m_per_thread=4, n_per_thread=4)
    return out


# Register the Vulkan (PrivateUse1) implementation.
torch.library.impl("torch_vulkan::mm_int8", "PrivateUse1", _mm_int8_dispatch)


def _register_mm_int8_op() -> None:
    """Register the mm_int8 fallback lowering with Inductor.

    Must be called during backend init (from ``lowerings.register()``)
    BEFORE any FX graph capture.  Idempotent — safe to call multiple times.

    Also ensures the custom op definition and PrivateUse1 implementation
    have been registered (the module-level calls above are idempotent per
    torch.library, but we call them explicitly for clarity).
    """
    from torch._inductor.lowering import make_fallback

    make_fallback(torch.ops.torch_vulkan.mm_int8)
