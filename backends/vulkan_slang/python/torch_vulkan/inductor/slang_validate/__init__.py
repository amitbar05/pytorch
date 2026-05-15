"""Slang source validation without compilation (Track T.5).

Catches common codegen bugs in-process before invoking ``slangc`` —
every codegen bug that *can* be caught here saves a subprocess creation,
SPIR-V generation, and error-message parse cycle (~0.2-0.5s per kernel).

Checks performed (all O(n) in source length):
  1. **Brace balance** — ``{ }`` mismatch, catches 90% of codegen bugs.
  2. **Binding contiguity** — ``[[vk::binding(N)]]`` declarations must be
     consecutive starting from 0 with no gaps.
  3. **Undefined identifier leaks** — Dynamo/Inductor size symbols (``s27``,
     ``s143``) that leak into generated Slang outside of subscript contexts.
  4. **Groupshared budget** — sum of all ``groupshared`` array sizes must
     not exceed the device limit (default 65536 bytes for RDNA1 — full LDS).
  5. **Numthreads product** — ``[numthreads(X, Y, Z)]`` product must be
     ≤ device limit (default 1024 for RDNA1).
  6. **Workgroup size advisory** — wave-size alignment check (M27).
  7. **Push-constant budget** — struct size ≤ 128 bytes (Vulkan minimum).
  8. **BwdDiff annotation scan** — ``[Differentiable]``/``[BackwardDerivative]``
     pairing check.

Usage (called before every ``slangc`` invocation)::

    from torch_vulkan.inductor.slang_validate import validate_slang_source

    errors = validate_slang_source(src)
    if errors:
        raise RuntimeError(f"Slang validation failed: {errors}")

Environment knobs:
  ``TORCH_VULKAN_VALIDATE_SLANG=0`` — disable validation (default: on).
  ``TORCH_VULKAN_MAX_GROUPSHARED_BYTES=N`` — override budget (default 65536).
  ``TORCH_VULKAN_MAX_NUMTHREADS_PRODUCT=N`` — override limit (default 1024).
  ``TORCH_VULKAN_WAVE_SIZE=N`` — wave size for advisory (default 64).
  ``TORCH_VULKAN_MAX_PUSH_CONSTANT_BYTES=N`` — push-constant budget (default 128).
"""

from __future__ import annotations

from ._config import _ENABLED, ValidationIssue
from .bindings import _check_binding_contiguity
from .braces import _check_brace_balance
from .bwd_diff_scan import _check_differentiable_pairs
from .memory import _check_groupshared_budget, _check_numthreads_product
from .push_constants import _check_push_constant_size
from .symbols import _check_size_symbol_leaks
from .workgroup import _validate_workgroup_size


def validate_slang_source(src: str) -> list[ValidationIssue]:
    """Run all Slang source validation checks.

    Returns a list of ``ValidationIssue`` objects, each with a
    ``.category`` string and ``.message`` for display.  Empty list
    means pass.
    """
    if not _ENABLED:
        return []

    issues: list[ValidationIssue] = []
    for msg in _check_brace_balance(src):
        issues.append(ValidationIssue("brace", msg))
    for msg in _check_binding_contiguity(src):
        issues.append(ValidationIssue("binding", msg))
    for msg in _check_size_symbol_leaks(src):
        issues.append(ValidationIssue("identifier", msg))
    for msg in _check_groupshared_budget(src):
        issues.append(ValidationIssue("groupshared", msg))
    for msg in _check_numthreads_product(src):
        issues.append(ValidationIssue("numthreads", msg))
    for msg in _validate_workgroup_size(src):
        issues.append(ValidationIssue("workgroup", msg))
    for msg in _check_push_constant_size(src):
        issues.append(ValidationIssue("push_constant", msg))
    for msg in _check_differentiable_pairs(src):
        issues.append(ValidationIssue("bwd_diff", msg))
    return issues
