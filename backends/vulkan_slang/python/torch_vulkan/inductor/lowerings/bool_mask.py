"""M16.2 - PrivateUse1 eager override for ``aten::index.Tensor``.

Handles bool-mask reads (``x[mask]``) and integer-index reads without
depending on ``csrc/ops/model_ops.cpp::vulkan_index_tensor``.

When ``torch.compile`` traces ``x[m]`` with a bool-typed ``m``, Dynamo
emits a graph break at ``BINARY_SUBSCR`` because the resulting tensor
shape is data-dependent.  After the graph break, the expression
evaluates eagerly through this override.

The override handles:

  1. **Single bool mask (``x[mask]``):** converts bool mask to int64
     via ``torch.nonzero(mask).squeeze(1)``, then chains through
     ``self.index_select(dim, int_idx)`` using GPU-native
     ``vulkan_index_select`` from ``csrc/ops/indexing_ops.cpp``.

  2. **Single int-index (``x[idx]``):** routes through
     ``self.index_select(dim=0, idx)`` (no ``model_ops.cpp`` dep).

  3. **Multi-index / mixed:** CPU roundtrip for correctness.
     The compiled path uses Inductor's ``index_impl`` Pointwise kernel
     for integer indices, which is GPU-native and fast.

Under ``torch.compile``, integer-only ``aten.index.Tensor`` is lowered
by the upstream Inductor's ``index_impl`` (Pointwise kernel). The bool-mask
path falls back to eager (this override) because upstream raises
``NotImplementedError`` for bool indices.

Prior art: OP.1.d (2026-05-09) CPU-roundtrip; OP.1.d-fast (2026-05-11)
added GPU-native chain but still depended on ``model_ops.cpp`` for integer
paths.  M16.2 (2026-05-13) removes that last dependency.
"""

from __future__ import annotations

_registered = False


def _register_bool_mask_read_lowering() -> None:
    """Idempotently install the ``aten::index.Tensor`` PrivateUse1 override.

    M16.2: Handles bool-mask and integer-index reads without depending on
    ``model_ops.cpp::vulkan_index_tensor``.  Bool masks decompose via
    ``nonzero`` + ``index_select``; integer indices use ``index_select``
    (single-index) or CPU fallback (multi-index).
    """
    global _registered
    if _registered:
        return

    import torch
    from torch.library import Library

    _lib = Library("aten", "IMPL")

    def _vulkan_index_tensor_with_bool(dispatch_keys, self, indices):
        # ── M16.2: GPU-native bool-mask gather (OP.1.d-fast) ──────────
        # Strategy: convert bool mask(s) to int64 indices via
        # ``torch.nonzero`` on Vulkan (now working via CPU-fallback in
        # csrc/ops/indexing_ops.cpp), then chain through
        # ``index_select`` for the gather (all on GPU).
        #
        # Integer-only indices are decomposed into primitives:
        #   1-index → ``index_select(dim=0, idx)`` (vulkan_index_select)
        #   2-index → CPU roundtrip (rare; compiled path uses Inductor's
        #             ``index_impl`` Pointwise, which is faster).
        #
        # This eliminates the dependency on ``model_ops.cpp::vulkan_index_tensor``
        # (anti-goal #2 / M16).

        # Find bool-index positions
        bool_positions = []
        for i, idx in enumerate(indices):
            if idx is not None and idx.dtype == torch.bool:
                bool_positions.append(i)

        # ── Single bool mask (the common x[mask] / x[:, mask]) ──
        if len(bool_positions) == 1:
            axis = bool_positions[0]
            mask = indices[axis]
            # nonzero: GPU→CPU→GPU roundtrip (small index tensor only)
            int_idx = torch.nonzero(mask).squeeze(1)
            return self.index_select(axis, int_idx)

        # ── Convert bool positions to int64 indices ───────────────────
        converted = list(indices)
        for i in bool_positions:
            converted[i] = torch.nonzero(converted[i]).squeeze(1)

        has_bool = len(bool_positions) > 0

        # ── Integer-only: 1-index case ────────────────────────────────
        # Single int-index tensor → ``index_select(dim=0, idx)``.
        # This is correct for the common ``x[idx]`` pattern where idx
        # is a 1-D int64 tensor selecting rows along dim 0.
        if not has_bool and len([i for i in converted if i is not None]) == 1:
            for axis, idx in enumerate(converted):
                if idx is not None:
                    return self.index_select(axis, idx)

        # ── Integer-only: 2-index case ────────────────────────────────
        # Two int-index tensors on a 2-D input: element-wise gather.
        # Fall back to CPU; the compiled path uses Inductor's ``index_impl``
        # Pointwise kernel which is GPU-native and fast.
        #
        # ── Everything else (bool masks, mixed, >2 indices) ───────────
        # CPU roundtrip: move self + indices to CPU, do the indexing, move
        # result back.  This is correct for all edge cases and the data
        # movement is acceptable for the eager path (torch.compile uses
        # Inductor's ``index_impl`` Pointwise for integer indices).
        cpu_self = self.cpu()
        cpu_idxs = tuple(slice(None) if i is None else i.cpu() for i in converted)
        return cpu_self[cpu_idxs].to(self.device)

    try:
        _lib.impl(
            "index.Tensor",
            _vulkan_index_tensor_with_bool,
            "PrivateUse1",
            with_keyset=True,
            allow_override=True,
        )
        _registered = True
    except RuntimeError as exc:
        # If a Python override is already installed (e.g. test re-import),
        # skip silently.  The dispatcher keeps the existing kernel.
        import logging

        logging.getLogger(__name__).warning(
            "OP.1.d: failed to register bool-mask override (already installed?): %s",
            exc,
        )

    # Hold a reference to the library so it isn't garbage-collected
    # (which would unregister our impl).  Mirrors the matmul pattern in
    # ``meta_patches.py``.
    import sys

    sys.modules[__name__]._bool_mask_lib = _lib  # type: ignore[attr-defined]


def _register_index_tensor_lowering() -> None:
    """M16.2 — Inductor lowering for ``aten.index.Tensor`` with bool mask.

    Registers a Vulkan-specific lowering that handles bool-mask
    ``aten.index.Tensor`` by falling back to eager (which our
    PrivateUse1 override in ``_register_bool_mask_read_lowering``
    decomposes to ``nonzero`` + ``index_select``).

    Integer-only indices are delegated to the upstream ``index_impl``
    (Pointwise kernel), which already works correctly on Vulkan.
    """
    import torch
    from torch._inductor.lowering import (
        fallback_handler,
        index_impl,
        register_lowering,
    )

    aten = torch.ops.aten

    @register_lowering(aten.index.Tensor, type_promotion_kind=None)
    def _vulkan_index_tensor(x, indices):
        # Check for bool indices
        has_bool = any(
            idx is not None and idx.get_dtype() == torch.bool for idx in indices
        )
        if has_bool:
            # Bool mask: fall back to eager. Our PrivateUse1 override
            # (bool_mask.py) decomposes to nonzero + index_select.
            x.realize()
            for idx in indices:
                if idx is not None:
                    idx.realize()
            return fallback_handler(aten.index.Tensor, add_to_fallback_set=False)(
                x, indices
            )

        # Integer indices: delegate to upstream index_impl.
        # This creates a Pointwise kernel with indirect loads.
        return index_impl(x, indices, check=True)
