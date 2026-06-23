"""Shared utilities for eager custom-op backwards.

Currently exposes ``_has_real_vulkan_storage`` — the canonical FakeTensor /
FunctionalTensor / compile-context detector used by the conv2d family of
backwards (and any future op family that needs to choose between the Slang
dispatch path and the aten decomposition path).

See M17.8.d.2 / M18.2 (roadmap 10-inductor-backend.md § 0.6.1).
"""

from __future__ import annotations

import torch


def _has_real_vulkan_storage(t) -> bool:
    """Return True iff ``t`` has real Vulkan storage we can dispatch a kernel
    against, False if ``t`` is a tracing wrapper (FakeTensor / FunctionalTensor /
    any tensor seen under ``torch.compiler.is_compiling()``).

    Originally each conv backward defined its own copy of this helper that
    only checked ``t.untyped_storage().device.type``.  That check returned
    True for FunctionalTensor wrappers whose underlying storage looked
    Vulkan-like — those flow into the Slang dispatch path, which invokes
    the C++ kernel via direct FFI (invisible to the proxy tracer).  The
    visible ops in the branch are just ``torch.zeros_like(...)``, so the
    traced backward graph collapses to ``aten.full(shape, 0)`` and
    produces literal-zero gradients at runtime.

    The fix (M17.8.d.2 for ``_conv2d_backward``; M18.2 extends to the two
    siblings) is to short-circuit on FakeTensor / FunctionalTensor / any
    ``torch.compiler.is_compiling()`` context BEFORE looking at storage.
    """
    try:
        # FakeTensor exposes .is_fake; FunctionalTensor doesn't but is the
        # wrapper used during AOTAutograd trace.
        if getattr(t, "is_fake", False):
            return False
        # FunctionalTensor → AOTAutograd wraps all inputs in FunctionalTensor
        # during trace; identify by constructor module name to avoid the
        # import dependency.
        if type(t).__name__ == "FunctionalTensor":
            return False
        # Generic safety: any traced tensor under torch.compile.
        if torch.compiler.is_compiling():
            return False
        # Fall back to the storage device check for plain Tensors.
        return t.untyped_storage().device.type != "meta"
    except Exception:
        return True
