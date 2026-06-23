"""View-shape lowerings — permute, as_strided zero-copy paths (PF.4) and
OP.3 view-style lowerings (narrow_copy, repeat_interleave, unfold).

OP.3 background: under torch.compile, several view-style ops (`narrow_copy`,
`repeat_interleave.self_int`, `unfold`) produce wrong results because the
upstream AOT decomps route them through a chain that ends in
``aten.narrow.default`` (or similar view) called at the wrapper level on the
Vulkan tensor, then a stride-aware copy that the Vulkan eager path
mishandles (storage_offset is not propagated into the SSBO descriptor set).

Fix: register Vulkan-specific Inductor lowerings that emit a Pointwise with
the offset/index transform baked into the inner_fn. This bypasses
``aten.narrow.default`` at the wrapper level — the kernel reads directly
from the input buffer with the correct sympy index expression.

We must also pop the upstream Inductor + AOT decomps so the FX graph
contains the original op (not the broken decomp chain). See
``_suppress_upstream_decomps`` in ``__init__.py`` for the pop list.
"""

from __future__ import annotations

from . import _is_vulkan


def _register_view_lowerings() -> None:
    """PF.4 — `aten.permute.default` zero-copy lowering for Vulkan.

    Upstream's `permute` lowering routes through ``PermuteView.create``,
    which builds a ``ReinterpretView`` only when ``is_storage_and_layout(x)``
    holds. In other cases (e.g. when the FX graph fed Inductor a tensor
    that came back as a meta-stride tensor through PrivateUse1's view
    fast-path — see PF.13 / B1''), the downstream ``assert_size_stride``
    in the wrapper trips because Inductor's IR-level stride algebra
    diverges from the meta tensor's strides.

    The fix: explicitly emit an ``as_strided`` IR with the permuted
    size+stride computed from the input. This guarantees Inductor's
    expected output strides match ``_permute_fake``'s output strides
    (both are simple permutations of the input layout), so the
    wrapper's ``assert_size_stride`` agrees with what the runtime
    produces — without needing ``TORCH_VULKAN_TRUST_INDUCTOR=1``.

    Vulkan-only: non-Vulkan tensors fall through to upstream's
    PermuteView path.
    """
    import torch
    from torch._inductor import lowering as L
    from torch._inductor.lowering import register_lowering

    aten = torch.ops.aten

    @register_lowering(aten.permute.default, type_promotion_kind=None)
    def _vulkan_permute_default(x, dims):
        if not _is_vulkan(x):
            # Re-dispatch into the upstream packet so non-Vulkan
            # devices keep the PermuteView path. We do this by calling
            # PermuteView.create directly — the packet-level lowering
            # we just overrode is unreachable from inside the override.
            from torch._inductor import ir
            from torch._inductor.ir import TensorBox

            assert isinstance(x, TensorBox)
            return TensorBox(ir.PermuteView.create(x.data, tuple(dims)))

        # PF.13.b.4 Layer-3: realize the input before reading stride.
        # The bwd graph emits ``view → permute → bmm`` chains where the
        # input is a ``View(StorageBox(Pointwise))`` IRNode whose
        # ``get_stride()`` raises ``NotImplementedError`` because the
        # Pointwise has no materialized layout. Upstream's
        # ``PermuteView.create`` gates on ``is_storage_and_layout`` and
        # returns a symbolic ``PermuteView`` for the non-layout case;
        # our override skips that gate, so we must realize first.
        # ``ExternKernel.realize_input`` calls ``x.realize()`` then
        # converts the BaseView-on-Pointwise to a ReinterpretView with
        # a layout, after which the existing ``as_strided`` emission
        # proceeds unchanged (preserving PF.4's stride-match guarantee).
        from torch._inductor import ir

        x = ir.ExternKernel.realize_input(x)

        # Compute permuted size + stride. _map_neg_dims is the same
        # mapping ``_permute_fake`` uses, so the resulting IR layout
        # bit-matches the FakeTensor stride.
        ndim = len(x.get_size())
        mapped = tuple(d if d >= 0 else ndim + d for d in dims)
        old_size = list(x.get_size())
        old_stride = list(x.get_stride())
        new_size = [old_size[i] for i in mapped]
        new_stride = [old_stride[i] for i in mapped]
        return L.lowerings[aten.as_strided](x, new_size, new_stride)

    _register_op3_view_lowerings()


def _register_op3_view_lowerings() -> None:
    """OP.3 — Vulkan-specific lowerings for view-style copy ops that the
    upstream decomp routes through the broken ``aten.narrow``-at-wrapper path.

    Each lowering emits a Pointwise that bakes the index transform directly
    into the kernel, sidestepping the descriptor-offset bug.

    Covered ops:
      - ``aten.narrow_copy.default`` — reads ``x[..., dim_idx + start, ...]``.
      - ``aten.repeat_interleave.self_int`` — reads
        ``x[..., dim_idx // repeats, ...]``.
      - ``aten.unfold.default`` — reads
        ``x[..., dim_idx*step + last_idx, ...]``; the new last axis carries
        the window index.

    ``aten.roll.default`` works correctly via the upstream decomp (cat of
    two slices) — no Vulkan-specific lowering needed.
    """
    import torch
    from torch._inductor.ir import Pointwise
    from torch._inductor.lowering import register_lowering
    from torch.utils._sympy.functions import FloorDiv

    aten = torch.ops.aten

    @register_lowering(aten.narrow_copy.default, type_promotion_kind=None)
    def _vulkan_narrow_copy(x, dim, start, length):
        ndim = len(x.get_size())
        if dim < 0:
            dim += ndim
        out_size = list(x.get_size())
        out_size[dim] = length
        loader = x.make_loader()

        def inner(idx):
            idx = list(idx)
            idx[dim] = idx[dim] + start
            return loader(idx)

        return Pointwise.create(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=inner,
            ranges=out_size,
        )

    @register_lowering(aten.repeat_interleave.self_int, type_promotion_kind=None)
    def _vulkan_repeat_interleave_self_int(x, repeats, dim=None, output_size=None):
        # When ``dim`` is None the spec flattens the input first.
        if dim is None:
            from torch._inductor.lowering import lowerings as _lowerings

            flat = _lowerings[aten.reshape](x, [-1])
            return _vulkan_repeat_interleave_self_int(flat, repeats, dim=0)

        ndim = len(x.get_size())
        if dim < 0:
            dim += ndim
        sizes = list(x.get_size())
        out_size = list(sizes)
        out_size[dim] = sizes[dim] * repeats
        loader = x.make_loader()

        def inner(idx):
            idx = list(idx)
            # FloorDiv keeps the sympy expression integer-typed; plain "/"
            # gets printed as a float division and reads the wrong elements.
            idx[dim] = FloorDiv(idx[dim], repeats)
            return loader(idx)

        return Pointwise.create(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=inner,
            ranges=out_size,
        )

    @register_lowering(aten.unfold.default, type_promotion_kind=None)
    def _vulkan_unfold(x, dimension, size, step):
        # Output shape: sizes[:dim] + [new_dim_size] + sizes[dim+1:] + [size].
        # The new last axis indexes into the window.
        ndim = len(x.get_size())
        if dimension < 0:
            dimension += ndim
        if ndim == 0:
            # Spec: unfold of a 0-dim tensor returns a 1-dim tensor of length
            # ``size`` after unsqueezing. Defer to upstream's edge-case path.
            from torch._inductor.lowering import lowerings as _lowerings

            return _lowerings[aten.unfold](x, dimension, size, step)

        sizes = list(x.get_size())
        dim_size = sizes[dimension]
        # ``new_dim_size = (dim_size - size) // step + 1`` — match aten spec.
        new_dim_size = FloorDiv(dim_size - size, step) + 1
        out_size = (
            list(sizes[:dimension])
            + [new_dim_size]
            + list(sizes[dimension + 1 :])
            + [size]
        )
        loader = x.make_loader()

        def inner(idx):
            # idx is the index tuple over out_size; the final entry is the
            # window index ``k``, the entry at ``dimension`` is the window
            # start ``i``. The input dim index is ``i * step + k``.
            idx = list(idx)
            i = idx[dimension]
            k = idx[-1]
            new_idx = (
                idx[:dimension]
                + [i * step + k]
                + idx[dimension + 1 : -1]
            )
            return loader(new_idx)

        return Pointwise.create(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=inner,
            ranges=out_size,
        )
