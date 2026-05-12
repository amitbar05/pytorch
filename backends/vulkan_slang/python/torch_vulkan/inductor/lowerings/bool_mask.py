"""OP.1.d-fast — GPU-native bool-mask read (`x[mask]`) via nonzero + index_select.

Why this lives in ``lowerings/`` even though it's an eager override:

When ``torch.compile`` traces ``x[m]`` with a bool-typed ``m``, Dynamo
emits a graph break at ``BINARY_SUBSCR`` because the resulting tensor
shape is data-dependent (``aten.nonzero``'s output shape depends on
input data, not on static shapes).  After the graph break, the
``x[m]`` expression evaluates eagerly — and the eager
``aten.index.Tensor`` PrivateUse1 kernel
(``csrc/ops/model_ops.cpp::vulkan_index_tensor``) does not handle bool
masks at all: it interprets a bool tensor as an int-index tensor,
selecting rows by ``[1,0,1,...]`` interpreted as integers, returning
shape ``(8,)`` for a length-8 mask instead of ``(N,)`` where N is the
number of True elements.

We install a **Python-level override** at the PrivateUse1 dispatch key
for ``aten::index.Tensor`` that:

  1. **Single bool mask (common case):** converts the bool mask to int64
     indices via ``torch.nonzero(mask).squeeze(1)`` (GPU-native, thanks
     to OP.1.a-fast's two-pass scan kernel), then chains through
     ``self.index_select(dim, int_idx)`` — all on GPU, no CPU roundtrip
     for the data tensor.

  2. **Multiple bool masks:** converts each bool mask to int64 indices
     and redispatches to the original C++ kernel via
     ``torch.library.get_kernel``'s ``call_boxed`` shim.

  3. **Int-only indices:** forwards to the original C++ kernel
     via ``call_boxed`` — preserving the existing fast int-indexing
     path (``test_multi_tensor_indexing_*``) untouched.

The override registers with ``with_keyset=True`` so the impl receives
the dispatcher's keyset, which is required for the boxed call into the
original kernel.

Prior art: OP.1.d shipped a CPU-roundtrip version (2026-05-09) that
moved the entire data tensor to CPU, did the gather there, and moved the
result back to device.  OP.1.d-fast (this file) replaces the CPU
roundtrip with a GPU-native chain enabled by OP.1.a-fast's GPU-native
``aten::nonzero``.
"""

from __future__ import annotations

_registered = False


def _register_bool_mask_read_lowering() -> None:
    """Idempotently install the ``aten::index.Tensor`` PrivateUse1 override.

    The override GPU-routes bool-mask cases via ``nonzero`` + ``index_select``
    (OP.1.d-fast) and forwards int-only cases to the original C++ kernel
    (``vulkan_index_tensor``) so existing multi-tensor int indexing keeps
    its GPU dispatch path.
    """
    global _registered
    if _registered:
        return

    import torch
    from torch.library import Library, get_kernel

    # Snapshot the original PrivateUse1 kernel BEFORE we override it.
    # ``get_kernel`` returns a handle whose ``call_boxed`` re-enters the
    # dispatcher at the requested key with the dispatch keyset we
    # received in our impl; this is the canonical "call parent" path
    # for a Python-side conditional override.
    try:
        _orig_index_kernel = get_kernel("aten::index.Tensor", "PrivateUse1")
    except Exception:
        # If the C++ kernel isn't registered (shouldn't happen in
        # normal operation), fall back to CPU roundtrip for ALL paths.
        _orig_index_kernel = None

    _lib = Library("aten", "IMPL")

    def _vulkan_index_tensor_with_bool(dispatch_keys, self, indices):
        has_bool = any(idx is not None and idx.dtype == torch.bool for idx in indices)
        if not has_bool and _orig_index_kernel is not None:
            return _orig_index_kernel.call_boxed(dispatch_keys, self, indices)

        # ── GPU-native bool-mask gather (OP.1.d-fast) ────────────────
        # Strategy: convert bool mask(s) to int64 indices via
        # ``torch.nonzero`` on Vulkan (OP.1.a-fast provides GPU-native
        # nonzero — no CPU roundtrip), then chain through
        # ``index_select`` for the gather (all on GPU).
        #
        # This replaces the OP.1.d CPU roundtrip (which moved the
        # *entire data tensor* to CPU).  Only the small nonzero
        # result touches host memory, and only when the original
        # C++ kernel is used as the fallback (the ``vulkan_index_tensor``
        # shader itself does a small CPU upload of int32 indices).

        # Find bool-index positions
        bool_positions = []
        for i, idx in enumerate(indices):
            if idx is not None and idx.dtype == torch.bool:
                bool_positions.append(i)

        # ── Single bool mask (the common x[mask] / x[:, mask]) ──
        if len(bool_positions) == 1:
            axis = bool_positions[0]
            mask = indices[axis]
            # GPU-native nonzero — stays on Vulkan thanks to OP.1.a-fast
            int_idx = torch.nonzero(mask).squeeze(1)
            return self.index_select(axis, int_idx)

        # ── Multiple bool masks or mixed bool+int: convert→redispatch ──
        converted = list(indices)
        for i in bool_positions:
            converted[i] = torch.nonzero(converted[i]).squeeze(1)

        if _orig_index_kernel is not None and len(bool_positions) <= 2:
            # Convert bool→int, forward to original C++ kernel (handles
            # up to 2 int index tensors; mode 0 = single-index row select,
            # mode 1 = two-index element selection).
            return _orig_index_kernel.call_boxed(dispatch_keys, self, tuple(converted))

        # ── Complex fallback: CPU roundtrip ───────────────────────────
        # Only reached when >2 bool masks or the original kernel is
        # unavailable (should be rare).
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
