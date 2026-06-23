"""M-pipeline-2 — OpOverload-identity-safe lowering lookups for conv ops.

Background
==========

PyTorch's ``register_lowering`` decorator stamps the lowering function
into ``torch._inductor.lowering.lowerings`` keyed by the ``OpOverload``
object's Python identity (``dict`` uses ``__hash__`` which for
``OpOverload`` defaults to ``id()``).

For ATen ops (``aten.add.Tensor`` etc.) the OpOverload identity is
process-stable: PyTorch creates the singleton on first attribute access
and caches it. Subsequent ``torch.ops.aten.add.Tensor`` accesses return
the same Python object, so ``lowerings[aten.add.Tensor]`` works.

For CUSTOM ops created via ``torch.library.Library(...).define(...)``,
the identity is NOT stable across re-registration. Backend init calls
``register_eager_patch_custom_ops()`` which re-binds custom ops like
``torch_vulkan::conv2d_with_optional_bias``. After re-binding, a fresh
resolution of ``torch.ops.torch_vulkan.conv2d_with_optional_bias.default``
yields a NEW ``OpOverload`` object whose identity does not match the
key the original ``@register_lowering`` decorator stamped. The dict
lookup silently misses with ``KeyError``.

M-pipeline-1 fixes the symptom (move the custom-op re-registration so
it never re-binds after the lowering decorator runs). M-pipeline-2 is
the preventive layer: any lookup that could hit a re-bound custom op
must use ``get_lowering_by_name`` instead of dict-key identity.

API
===

``get_lowering_by_name(lowerings, target)`` — iterate the dict by
``str(key)`` match and return the lowering callable, or ``None`` if
not registered. Drop-in replacement for ``lowerings.get(target)``
that survives identity drift.

``_get_conv2d_lowering_by_name()`` — backwards-compat alias preserved
for `conv.py`; was the original name introduced by the M19.5-followup-1
inline implementation. Use ``get_lowering_by_name`` in new code.

``register_lowering_identity_safe(op)`` — marker decorator that
documents (in source) that the wrapped lowering's call sites must use
``get_lowering_by_name`` for lookups rather than dict-key identity.
Returns the underlying ``register_lowering`` decorator unchanged; the
documentation IS the contract. Apply this to any custom-op lowering
that is known to be re-registered.
"""

from __future__ import annotations

from typing import Any, Callable, Optional


def get_lowering_by_name(
    lowerings: dict,
    target,
) -> Optional[Callable[..., Any]]:
    """Look up a lowering by ``str(key)`` match, surviving OpOverload
    identity drift across ``register_eager_patch_custom_ops()`` reruns.

    Parameters
    ----------
    lowerings : dict
        The ``torch._inductor.lowering.lowerings`` registry (or any
        compatible dict mapping ``OpOverload`` → callable).
    target : OpOverload | str
        Either an ``OpOverload`` object (in which case its ``str()``
        form is matched) or a string already in ``str(OpOverload)``
        form (e.g. ``"aten.add.Tensor"`` or
        ``"torch_vulkan.conv2d_with_optional_bias.default"``).

    Returns
    -------
    Callable | None
        The registered lowering, or ``None`` when no match is found.
        The caller should typically fall through to ``NotImplemented``
        so Inductor routes through the extern fallback.

    Why this is needed
    ------------------
    ``lowerings`` is keyed by ``OpOverload`` Python identity. For
    ATen ops the identity is stable. For custom ops registered via
    ``torch.library.Library(...)``, the identity changes whenever
    the library is re-defined (which our backend init does on every
    fresh process via ``register_eager_patch_custom_ops()``). The
    string form, however, IS stable — ``str(some_op_overload)`` always
    returns ``"namespace.opname.overload_name"``.

    Performance: O(n) linear scan over ``lowerings``. Acceptable
    because ``lowerings`` is typically <1000 entries and this lookup
    fires at most once per lowering invocation (and lowerings are
    cached at the FX level downstream).
    """
    target_str = str(target)
    for k, v in lowerings.items():
        if str(k) == target_str:
            return v
    # Also accept a suffix match (e.g. searching for
    # "torch_vulkan.conv2d_with_optional_bias.default" should match an
    # OpOverload whose str() form includes a leading module path).
    for k, v in lowerings.items():
        if str(k).endswith(target_str):
            return v
    return None


def register_lowering_identity_safe(op):
    """Marker decorator that documents the wrapped lowering must be
    looked up via :func:`get_lowering_by_name`, not via raw
    ``lowerings[op]``.

    Returns the underlying ``register_lowering`` decorator unchanged —
    the contract is enforced by source-level documentation + the
    M-pipeline-2 static regression test in
    ``tests/test_inductor_regression.py::
    TestMPipeline2OpOverloadIdentitySafe::test_conv1d_dynamic_uses_helper``.

    Apply to any custom-op lowering whose underlying ``OpOverload``
    may be re-bound (e.g.
    ``torch_vulkan::conv2d_with_optional_bias``).
    """
    from torch._inductor.lowering import register_lowering

    return register_lowering(op, type_promotion_kind=None)


# ── Backwards-compat alias ───────────────────────────────────────────
#
# The original inline implementation in ``conv.py`` was named
# ``_get_conv2d_lowering_by_name`` and took zero arguments (it
# implicitly looked up the conv2d_with_optional_bias op). The
# generalised helper above takes a ``target`` parameter; this thin
# alias preserves the old zero-arg surface so call sites in
# ``conv.py`` don't need to change.


def _get_conv2d_lowering_by_name() -> Optional[Callable[..., Any]]:
    """Look up the ``torch_vulkan::conv2d_with_optional_bias`` lowering
    by string form. Survives ``register_eager_patch_custom_ops()``
    re-registration. Returns ``None`` if the lowering isn't registered
    yet (caller should return ``NotImplemented``).

    Thin alias around :func:`get_lowering_by_name` preserved for source
    compatibility with the original inline implementation. Prefer
    ``get_lowering_by_name(lowerings, target)`` in new code.
    """
    from torch._inductor.lowering import lowerings as _lowerings

    return get_lowering_by_name(
        _lowerings,
        "torch_vulkan.conv2d_with_optional_bias.default",
    )
