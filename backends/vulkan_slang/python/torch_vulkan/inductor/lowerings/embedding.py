"""Embedding-dense-backward + embedding_bag-forward lowerings (B2 / OP.5)."""

from __future__ import annotations

from . import _is_vulkan


def _register_embedding_dense_backward() -> None:
    """B2 — Inductor lowering for ``aten.embedding_dense_backward.default``.

    Without this, compiled models using ``nn.Embedding`` extern-fall on the
    backward path because the hand-written ``embedding_backward.slang`` is
    reachable only on the eager dispatch path. Decomposes the backward into
    ``aten.index_put.default(accumulate=True)`` which is already lowered
    (PF.22), so the scheduler fuses the scatter-add into the backward chain.

    Schema:
      ``embedding_dense_backward(grad_output, indices, num_weights, padding_idx,
        scale_grad_by_freq) -> grad_weight``

    The decomposition creates a zero ``grad_weight`` of shape
    ``(num_weights, embedding_dim)`` and scatters ``grad_output`` rows into it
    at the positions indicated by ``indices``.
    """
    import torch
    from torch._inductor import lowering as L
    from torch._inductor.lowering import register_lowering

    aten = torch.ops.aten

    @register_lowering(aten.embedding_dense_backward, type_promotion_kind=None)
    def _vulkan_embedding_dense_backward(
        grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
    ):
        if not _is_vulkan(grad_output):
            return NotImplemented

        if bool(scale_grad_by_freq):
            return NotImplemented

        num_weights = int(num_weights)
        padding_idx = int(padding_idx)
        if num_weights <= 0:
            return NotImplemented

        embedding_dim_list = list(grad_output.get_size())
        if len(embedding_dim_list) != 2:
            return NotImplemented
        embedding_dim = embedding_dim_list[1]

        grad_weight = L.lowerings[aten.full.default](
            [num_weights, embedding_dim],
            0.0,
            dtype=grad_output.get_dtype(),
            device=grad_output.get_device(),
            pin_memory=False,
        )

        idx_list = list(indices.get_size())
        idx_numel = 1
        for s in idx_list:
            idx_numel = idx_numel * s
        flat_indices = L.lowerings[aten.view.default](indices, [idx_numel])
        flat_grad = L.lowerings[aten.view.default](
            grad_output, [idx_numel, embedding_dim]
        )

        if padding_idx >= 0:
            mask = L.lowerings[aten.ne.Scalar](flat_indices, padding_idx)
            zero_idx = L.lowerings[aten.mul.Scalar](flat_indices, 0)
            safe_indices = L.lowerings[aten.where.self](mask, flat_indices, zero_idx)
            mask_2d = L.lowerings[aten.unsqueeze.default](mask, -1)
            zero_grad = L.lowerings[aten.mul.Scalar](flat_grad, 0.0)
            safe_grad = L.lowerings[aten.where.self](mask_2d, flat_grad, zero_grad)
        else:
            safe_indices = flat_indices
            safe_grad = flat_grad

        result = L.lowerings[aten.index_put.default](
            grad_weight, [safe_indices], safe_grad, True
        )
        return result


def _register_embedding_bag_forward() -> None:
    """OP.5 — Inductor lowering for ``aten._embedding_bag.default`` (forward).

    Upstream Inductor registers ``aten._embedding_bag`` via ``make_fallback``,
    which routes back to the eager kernel. Vulkan has no eager
    ``_embedding_bag`` kernel, so without this lowering compiled models using
    ``nn.EmbeddingBag`` (recommendation engines, word2vec-style models) fail
    with an unimplemented-op error.

    Strategy (Option C from the OP.5 spec): decompose ``mode=sum``,
    ``mode=mean``, and ``mode=max`` into existing primitives. ``mode=max``
    is wired here via ``aten.scatter_reduce.two(reduce='amax')`` — which
    rides on the T4.11 ``atomic_max_f32`` infrastructure (see
    ``shaders/lib/atomics.slang`` + ``templates/scatter_atomic.py.jinja``)
    routed through the OP.2 PrivateUse1 eager dispatch
    (``lowerings/scatter.py:_install_eager_scatter_reduce``).

    Decomposition:
      1. ``rows = embedding(weight, indices)``                  # [N, D]
      2. Build a 1-D ``bag_id[N]`` Pointwise: ``bag_id[i]
         = #{b ∈ [1, num_bags) : i ≥ offsets[b]}`` — a static
         unroll over interior offset boundaries.
      3a. (sum/mean) ``out = zeros([B, D]);
          out.index_put_([bag_id], rows, accumulate=True)``
          (atomic-scatter, same primitive as ``embedding_dense_backward``).
      3b. (max) ``out = full([B, D], -inf);
          out.scatter_reduce_(dim=0, bag_id_2d, rows, reduce='amax',
            include_self=False)`` then ``out = where(bag_size > 0, out, 0)``
          to match CPU semantics (empty-bag rows return 0, not -inf).
      4. (mean) ``out /= clamp(bag_size, 1)`` where ``bag_size`` is
         computed via the same 1-D scatter trick.

    The 3 secondary outputs (offset2bag / bag_size / max_indices) returned
    by ``_embedding_bag`` are zero-filled placeholders sized to match the
    meta schema. They are only consumed by the backward path; for
    forward-only inference (the OP.5 target) these never reach a kernel.

    Schema:
      ``_embedding_bag(weight, indices, offsets, scale_grad_by_freq=False,
        mode=0, sparse=False, per_sample_weights=None,
        include_last_offset=False, padding_idx=-1)
        -> (output, offset2bag, bag_size, max_indices)``

    The earlier ``index_put`` codegen bug (binding-order swap) was fixed
    in round 3 (Group F) — see
    ``inductor/wrapper.py:_install_vulkan_skip_alignment_clone``. All
    three modes (sum / mean / max) now match CPU eager.
    """
    import torch
    from torch._inductor import lowering as L
    from torch._inductor.lowering import register_lowering

    aten = torch.ops.aten

    def _vulkan_embedding_bag(
        weight,
        indices,
        offsets,
        scale_grad_by_freq=False,
        mode=0,
        sparse=False,
        per_sample_weights=None,
        include_last_offset=False,
        padding_idx=-1,
    ):
        if not _is_vulkan(weight):
            return NotImplemented
        # mode: 0=sum, 1=mean, 2=max. All three are wired here.
        if int(mode) not in (0, 1, 2):
            return NotImplemented
        if bool(sparse) or bool(scale_grad_by_freq):
            return NotImplemented
        # mode='max' requires fp32 (T4.11 atomic_max_f32 is fp32-only).
        if int(mode) == 2 and weight.get_dtype() != torch.float32:
            return NotImplemented
        # per_sample_weights is only valid for mode='sum' per upstream schema;
        # ignore the gate for mode='mean' parity (upstream raises in that case).
        if int(mode) == 2 and per_sample_weights is not None:
            return NotImplemented

        weight_size = list(weight.get_size())
        if len(weight_size) != 2:
            return NotImplemented
        embedding_dim = weight_size[1]

        indices_size = list(indices.get_size())
        if len(indices_size) != 1:
            return NotImplemented
        n_tokens = indices_size[0]

        offsets_size = list(offsets.get_size())
        if len(offsets_size) != 1:
            return NotImplemented
        if bool(include_last_offset):
            num_bags = offsets_size[0] - 1
        else:
            num_bags = offsets_size[0]
        if num_bags <= 0:
            return NotImplemented

        weight_dtype = weight.get_dtype()
        weight_device = weight.get_device()
        idx_dtype = indices.get_dtype()
        offsets_dtype = offsets.get_dtype()

        # ── 1. Embedding lookup: rows = embedding(weight, indices) → [N, D]
        rows = L.lowerings[aten.embedding.default](weight, indices, -1, False, False)

        # Optional padding-idx zeroing of contributing rows.
        if int(padding_idx) >= 0:
            keep = L.lowerings[aten.ne.Scalar](indices, int(padding_idx))
            keep_2d = L.lowerings[aten.unsqueeze.default](keep, -1)
            zero_rows = L.lowerings[aten.mul.Scalar](rows, 0.0)
            rows = L.lowerings[aten.where.self](keep_2d, rows, zero_rows)

        # Optional per-sample weighting (mode=sum only — guarded by op schema).
        if per_sample_weights is not None:
            psw_2d = L.lowerings[aten.unsqueeze.default](per_sample_weights, -1)
            rows = L.lowerings[aten.mul.Tensor](rows, psw_2d)

        # ── 2. Build a 1-D ``bag_id[N]`` Pointwise mapping each token to
        # its bag.  We unroll the count of crossed interior boundaries
        # at lowering time so the inner_fn is a static SSA chain.
        from torch._inductor import ir as _ir
        from torch._inductor.virtualized import ops as _ops

        offsets_loader = offsets.make_loader()

        def _bag_id_inner(idx):
            (i,) = idx
            i_int = _ops.index_expr(i, offsets_dtype)
            bid = _ops.constant(0, offsets_dtype)
            for b in range(1, int(num_bags)):
                thresh = offsets_loader([b])
                crossed = _ops.ge(i_int, thresh)
                crossed_int = _ops.to_dtype(crossed, offsets_dtype)
                bid = _ops.add(bid, crossed_int)
            return bid

        bag_id = _ir.Pointwise.create(
            device=weight_device,
            dtype=offsets_dtype,
            inner_fn=_bag_id_inner,
            ranges=[int(n_tokens)],
        )

        # ── 3. Reduce ``rows`` into a [B, D] output, indexed by ``bag_id``.
        # sum/mean: atomic-add scatter via ``index_put(accumulate=True)``.
        # max:      atomic-max scatter via
        #           ``scatter_reduce.two(reduce='amax', include_self=False)``,
        #           which rides on T4.11 atomic_max_f32 through OP.2's
        #           PrivateUse1 eager dispatch.
        if int(mode) in (0, 1):
            zero_out = L.lowerings[aten.full.default](
                [int(num_bags), int(embedding_dim)],
                0.0,
                dtype=weight_dtype,
                device=weight_device,
                pin_memory=False,
            )
            output = L.lowerings[aten.index_put.default](zero_out, [bag_id], rows, True)
        else:
            # mode == 2 (max).  scatter_reduce(dim=0) needs index of shape
            # [N, D] matching ``rows``.  Two constraints we have to thread:
            #
            # (a) The eager scatter impl calls ``index.contiguous()`` —
            #     so the buffer must be contiguous int64 / int32.  An
            #     ``unsqueeze + expand`` view has stride-0 on axis 1 and
            #     trips ``vulkan_clone`` (fp32-only).
            # (b) A 2-D ``Pointwise`` whose ``inner_fn`` calls
            #     ``index_expr(axis0)`` triggers a Vulkan combo-kernel
            #     codegen bug (axis-0's name leaks unbound — see
            #     ``x1_sub1 undefined`` in subkernel-1).
            #
            # Workaround: realize bag_id as a 1-D buffer (no
            # ``index_expr`` in any post-realize Pointwise), then build
            # bag_id_2d by *reading* from that buffer with a 2-D
            # Pointwise whose inner_fn uses ``make_loader`` (a buffer
            # read, not an index expression).  Force materialization via
            # ``aten.clone`` so ScatterFallback's ``reinterpret_tensor``
            # path is not hit.
            import math

            neg_inf_out = L.lowerings[aten.full.default](
                [int(num_bags), int(embedding_dim)],
                -math.inf,
                dtype=weight_dtype,
                device=weight_device,
                pin_memory=False,
            )

            # Realize the 1-D bag_id so the next Pointwise reads from a
            # real buffer, not from an inlined ``index_expr`` chain.
            bag_id.realize()
            bag_id_loader = bag_id.make_loader()

            def _bag_id_2d_inner(idx):
                i, _j = idx
                return bag_id_loader([i])

            bag_id_2d = _ir.Pointwise.create(
                device=weight_device,
                dtype=offsets_dtype,
                inner_fn=_bag_id_2d_inner,
                ranges=[int(n_tokens), int(embedding_dim)],
            )
            # clone() forces a fresh contiguous buffer so the
            # ScatterFallback codegen doesn't introduce a stride-0
            # ``reinterpret_tensor`` view that the eager impl can't
            # ``.contiguous()`` through ``vulkan_clone``.
            bag_id_2d = L.lowerings[aten.clone.default](bag_id_2d)
            output = L.lowerings[aten.scatter_reduce.two](
                neg_inf_out,
                0,
                bag_id_2d,
                rows,
                "amax",
                include_self=False,
            )

        # ── 4. mode-specific post-pass.
        # mean: divide each row by its bag size.
        # max:  replace -inf rows (empty bags) with 0 to match CPU semantics.
        if int(mode) in (1, 2):
            # Compute bag_size via the same 1-D scatter trick (count of
            # tokens landing in each bag).  Reused for both mean (divisor)
            # and max (empty-bag mask).
            zero_count = L.lowerings[aten.full.default](
                [int(num_bags)],
                0.0,
                dtype=weight_dtype,
                device=weight_device,
                pin_memory=False,
            )
            ones = L.lowerings[aten.full.default](
                [int(n_tokens)],
                1.0,
                dtype=weight_dtype,
                device=weight_device,
                pin_memory=False,
            )
            bag_size_1d = L.lowerings[aten.index_put.default](
                zero_count, [bag_id], ones, True
            )
            if int(mode) == 1:
                bag_size_clamped = L.lowerings[aten.clamp.default](
                    bag_size_1d, 1.0, None
                )
                bag_size_2d_div = L.lowerings[aten.unsqueeze.default](
                    bag_size_clamped, -1
                )
                output = L.lowerings[aten.div.Tensor](output, bag_size_2d_div)
            else:  # mode == 2 (max)
                # Empty bags (bag_size == 0) currently hold -inf; replace
                # with 0 to match CPU embedding_bag(mode='max') semantics.
                nonempty = L.lowerings[aten.gt.Scalar](bag_size_1d, 0.0)
                nonempty_2d = L.lowerings[aten.unsqueeze.default](nonempty, -1)
                zero_out_max = L.lowerings[aten.full.default](
                    [int(num_bags), int(embedding_dim)],
                    0.0,
                    dtype=weight_dtype,
                    device=weight_device,
                    pin_memory=False,
                )
                output = L.lowerings[aten.where.self](nonempty_2d, output, zero_out_max)

        # ── Secondary outputs (zero-filled placeholders matching meta schema).
        offset2bag = L.lowerings[aten.full.default](
            [n_tokens],
            0,
            dtype=idx_dtype,
            device=weight_device,
            pin_memory=False,
        )
        bag_size_int = L.lowerings[aten.full.default](
            [num_bags],
            0,
            dtype=idx_dtype,
            device=weight_device,
            pin_memory=False,
        )
        max_indices = L.lowerings[aten.full.default](
            [num_bags, embedding_dim] if int(mode) == 2 else [0],
            0,
            dtype=idx_dtype,
            device=weight_device,
            pin_memory=False,
        )

        return output, offset2bag, bag_size_int, max_indices

    # Both ``aten._embedding_bag`` (training path) and
    # ``aten._embedding_bag_forward_only`` (inference path) share the same
    # signature and same forward semantics — register both.
    register_lowering(aten._embedding_bag.default, type_promotion_kind=None)(
        _vulkan_embedding_bag
    )
    register_lowering(
        aten._embedding_bag_forward_only.default, type_promotion_kind=None
    )(_vulkan_embedding_bag)


def _register_embedding_bag_backward() -> None:
    """OP.21 — Inductor lowering for ``aten._embedding_bag_backward.default``.

    Decomposes the backward into the same primitives the forward uses:
    - mode=0 (sum):  ``index_put(accumulate=True)`` via offset2bag mapping
    - mode=1 (mean): sum-mode with per-bag division by bag_size
    - mode=2 (max):  ``scatter_reduce(reduce='amax')`` via max_indices

    Schema:
      ``_embedding_bag_backward(grad, indices, offsets, offset2bag, bag_size,
        max_indices, num_weights, scale_grad_by_freq, mode, sparse,
        per_sample_weights, padding_idx) -> grad_weight``
    """
    import torch
    from torch._inductor import lowering as L
    from torch._inductor.lowering import register_lowering

    aten = torch.ops.aten

    @register_lowering(aten._embedding_bag_backward, type_promotion_kind=None)
    def _vulkan_embedding_bag_backward(
        grad,
        indices,
        offsets,
        offset2bag,
        bag_size,
        max_indices,
        num_weights,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights=None,
        padding_idx=-1,
    ):
        if not _is_vulkan(grad):
            return NotImplemented

        if bool(scale_grad_by_freq):
            return NotImplemented
        if bool(sparse):
            return NotImplemented
        if per_sample_weights is not None:
            return NotImplemented

        num_weights = int(num_weights)
        padding_idx = int(padding_idx)
        mode = int(mode)
        if num_weights <= 0:
            return NotImplemented

        grad_size = list(grad.get_size())
        if len(grad_size) != 2:
            return NotImplemented
        embedding_dim = grad_size[1]
        weight_device = grad.get_device()
        weight_dtype = grad.get_dtype()

        # Create zero grad_weight
        grad_weight = L.lowerings[aten.full.default](
            [num_weights, embedding_dim],
            0.0,
            dtype=weight_dtype,
            device=weight_device,
            pin_memory=False,
        )

        if mode == 2:
            # max mode: scatter via max_indices
            # max_indices shape: [num_bags, embedding_dim]
            n_tokens = grad_size[0]
            flat_grad = L.lowerings[aten.view.default](grad, [n_tokens, embedding_dim])
            result = L.lowerings[aten.scatter_reduce.two](
                grad_weight,
                0,
                max_indices,
                flat_grad,
                "amax",
                False,
            )
            return result

        # mode 0 (sum) and mode 1 (mean): index_put with accumulate
        n_tokens = grad_size[0]
        flat_grad = L.lowerings[aten.view.default](grad, [n_tokens, embedding_dim])

        if mode == 1:
            # mean: divide each grad row by corresponding bag_size
            bag_size_f = L.lowerings[aten.index.Tensor](bag_size, [offset2bag])
            bag_size_2d = L.lowerings[aten.unsqueeze.default](bag_size_f, -1)
            flat_grad = L.lowerings[aten.div.Tensor](flat_grad, bag_size_2d)

        # Handle padding_idx by zeroing grads at padding_idx position
        if padding_idx >= 0:
            offset2bag_i64 = L.lowerings[aten._to_copy.default](
                offset2bag, dtype=torch.int64
            )
            mask = L.lowerings[aten.eq.Scalar](offset2bag_i64, padding_idx)
            zero_idx = L.lowerings[aten.mul.Scalar](offset2bag_i64, 0)
            safe_indices = L.lowerings[aten.where.self](mask, zero_idx, offset2bag_i64)
        else:
            safe_indices = L.lowerings[aten._to_copy.default](
                offset2bag, dtype=torch.int64
            )

        result = L.lowerings[aten.index_put.default](
            grad_weight, [safe_indices], flat_grad, True
        )
        return result
