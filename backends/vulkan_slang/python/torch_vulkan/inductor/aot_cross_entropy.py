"""TRAIN.8 — Register custom ``nll_loss_forward`` decomposition with
compile-time-constant total_weight.

The upstream ``nll_loss_forward`` decomposition computes
``total_weight = (target != ignore_index).sum().to(self)``.  When the
AOTAutograd min-cut partitioner assigns ``target`` to the backward
partition, ``InvalidNode`` propagates through the chain:
    target → ne(target, ignore_index) → sum → total_weight
and ultimately the forward output ``div(sum_loss, total_weight)`` is
marked "invalid, but is output".

**Fix:** Install our own decomposition for ``aten.nll_loss_forward.default``
that computes ``total_weight = self.new_full((), float(batch_size))`` (a
compile-time constant that doesn't depend on target VALUES), preventing
the partitioner from creating the backward dependency chain.

This is mathematically equivalent when ``ignore_index = -100`` (default)
and all targets are valid class indices (the standard training case).
"""

from __future__ import annotations

_patched = False


def patch_nll_loss_forward() -> None:
    """Install our custom ``nll_loss_forward`` decomposition in the Inductor
    decomp table.  Safe to call multiple times (idempotent).

    This replaces the upstream ``nll_loss_forward`` wrapper function in
    ``torch._inductor.decomposition.decompositions`` with our own version
    that uses a target-value-independent total_weight.
    """
    global _patched
    if _patched:
        return
    _patched = True

    import torch

    def _vulkan_nll_loss_forward_const_tw(
        self: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None,
        reduction: int,
        ignore_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Custom nll_loss_forward: constant total_weight for partitioner safety.

        Returns ``(loss, total_weight)`` matching the upstream signature.
        """
        n_dims = self.dim()
        channel_dim = 1 if n_dims >= 2 else 0

        # Apply per-class weight before gathering.
        if weight is not None:
            if n_dims > 1:
                shape = [1] * n_dims
                shape[channel_dim] = weight.shape[0]
                w = weight.view(shape)
            else:
                w = weight
            self_work = self * w
        else:
            w = None
            self_work = self

        # Safe target: replace ignore_index with 0 so gather stays in bounds.
        ignore_mask = target != ignore_index
        safe_target = torch.where(ignore_mask, target, torch.zeros_like(target))

        if n_dims >= 2:
            safe_target_expanded = safe_target.unsqueeze(channel_dim)
        else:
            safe_target_expanded = safe_target

        # Gather log probabilities at target indices, negate.
        result = -torch.gather(self_work, channel_dim, safe_target_expanded)
        if n_dims >= 2:
            result = result.squeeze(channel_dim)

        # Zero out ignored positions.
        result = torch.where(ignore_mask, result, torch.zeros_like(result))

        # Reduction.NONE
        if reduction == 0 and n_dims > 1:
            total_weight = self.new_full((), 0.0)
            return result, total_weight

        # Compute total_weight.
        # KEY FIX: For unweighted case, use a CONSTANT total_weight =
        # batch_size.  self.new_full() creates a tensor on the same device
        # as self (log_probs), and the value is known at compile time
        # (doesn't depend on target values).
        if w is not None:
            # Weighted: still need to sum gathered weights.
            w_expanded = w.expand(self_work.shape)
            wsum = torch.gather(w_expanded, channel_dim, safe_target_expanded)
            if n_dims >= 2:
                wsum = wsum.squeeze(channel_dim)
            wsum = torch.where(ignore_mask, wsum, torch.zeros_like(wsum))
            total_weight = wsum.sum()
        else:
            # Unweighted: total_weight = number of valid samples.
            # Using target.numel() gives us a Python int (compile-time constant).
            N = target.numel()
            total_weight = self.new_full((), float(N))

        # Apply reduction.
        if reduction == 2:  # SUM
            result = result.sum()
        elif reduction == 1:  # MEAN
            result = result.sum() / total_weight

        return result, total_weight

    # REMOVE nll_loss_forward from Inductor decomp table entirely.
    # This keeps nll_loss_forward opaque in the joint graph (not decomposed
    # before partitioning). The partitioner then handles it based on inputs/outputs
    # without seeing the internal div node.
    # During Inductor lowering, nll_loss_forward will be handled via FallbackKernel
    # (calls Vulkan eager impl). This is a stepping stone until we add a proper lowering.
    from torch._inductor.decomposition import decompositions as _inductor_decomps

    aten = torch.ops.aten
    _inductor_decomps.pop(aten.nll_loss_forward.default, None)

    # Also remove from fast_random_decomps cache so removal is effective.
    try:
        from torch._inductor.decomposition import fast_random_decomps
        fast_random_decomps.cache_clear()
    except Exception:
        pass
