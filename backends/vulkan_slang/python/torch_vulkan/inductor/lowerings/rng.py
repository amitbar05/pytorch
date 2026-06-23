"""OP.11 — ``aten.multinomial`` Philox-based sampling lowering.

Decomposes ``aten.multinomial`` (replacement=True only for now) into:
  1. Normalize probabilities (divide by row-wise sum)
  2. Cumulative distribution (cumsum along last dim)
  3. Philox uniform random samples (via ``aten.rand`` → FallbackKernel →
     PrivateUse1 Philox dispatch)
  4. Inverse-CDF lookup via ``aten.searchsorted``

``replacement=False`` (without-replacement sampling) is not yet supported
and will fall through to the standard Inductor path (graph-break or
FallbackKernel).

Dependencies:
  - P1.3 / CP.9: Philox RNG PrivateUse1 dispatch
  - N.1.b: ``aten.searchsorted`` PrivateUse1 override (CPU roundtrip)
  - P4.8: ``aten.cumsum`` Vulkan scan shader
"""

from __future__ import annotations

_registered = False


def _register_multinomial_lowering() -> None:
    global _registered
    if _registered:
        return
    _registered = True

    import torch
    from torch._inductor import lowering as L
    from torch._inductor.lowering import register_lowering

    from . import _is_vulkan

    aten = torch.ops.aten

    # Save the original lowering (if any) so we can fall through for
    # non-Vulkan tensors or replacement=False.
    _orig_multinomial = L.lowerings.get(aten.multinomial)

    @register_lowering(aten.multinomial, type_promotion_kind=None)
    def _vulkan_multinomial(
        probs,
        num_samples,
        replacement=False,
        *,
        generator=None,
    ):
        """Lower aten.multinomial → cumsum + rand + searchsorted.

        Only supports ``replacement=True`` (inverse-CDF method).
        ``replacement=False`` falls through to the upstream path.
        """
        # ── guard: Vulkan tensors only ──────────────────────────────
        if not _is_vulkan(probs):
            if _orig_multinomial is not None:
                return _orig_multinomial(
                    probs,
                    num_samples,
                    replacement=replacement,
                    generator=generator,
                )
            return NotImplemented

        # ── guard: replacement=True only for now ────────────────────
        if not replacement:
            if _orig_multinomial is not None:
                return _orig_multinomial(
                    probs,
                    num_samples,
                    replacement=replacement,
                    generator=generator,
                )
            return NotImplemented

        # ── 1. Normalize probabilities to sum to 1 along last dim ──
        probs_sum = L.lowerings[aten.sum.dim_IntList](probs, [-1], keepdim=True)
        probs_norm = L.lowerings[aten.div.Tensor](probs, probs_sum)

        # ── 2. Compute CDF via cumsum along last dim ─────────────────
        cdf = L.lowerings[aten.cumsum](probs_norm, -1)

        # ── 3. Generate uniform random samples via Philox ────────────
        probs_shape = probs.get_size()
        probs_dtype = probs.get_dtype()
        probs_device = probs.get_device()

        # Output shape: [*batch_dims, num_samples]
        uniform_shape = list(probs_shape[:-1]) + [num_samples]

        uniform = L.lowerings[aten.rand](
            *uniform_shape,
            dtype=probs_dtype,
            device=probs_device,
        )

        # ── 4. Inverse-CDF lookup via searchsorted ──────────────────
        indices = L.lowerings[aten.searchsorted.Tensor](
            cdf,
            uniform,
            out_int32=False,  # int64 output
            right=False,
        )

        # Clamp to [0, num_categories-1] in case floating-point
        # rounding pushes a uniform sample slightly above 1.0 (so
        # searchsorted returns C instead of C-1).
        num_categories = probs_shape[-1]
        indices = L.lowerings[aten.clamp](indices, min=0, max=num_categories - 1)

        return indices
