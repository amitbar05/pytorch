"""Complex-dtype elementwise lowerings (OP.20).

Intercepts elementwise ops with complex inputs and decomposes them
through ``view_as_real`` → real pointwise → ``view_as_complex`` so
they route through Inductor's pointwise codegen instead of falling
through to ``ExternKernel`` (eager dispatch).

The Slang-side structs live in ``shaders/lib/pointwise.slang`` as
``OpComplexAdd``, ``OpComplexMul``, ``OpComplexDiv``, ``OpComplexConj``,
``OpComplexAbs`` implementing ``IComplexPointwise`` / ``IComplexPointwiseBinary``
on ``float2`` values.

**Limitation (OP.20 partial):** ``aten.add``, ``aten.mul``, ``aten.div``,
``aten.sub``, and ``aten.abs`` have upstream ``register_pointwise`` lowerings
that fire before ours (first-come-first-served).  These complex ops still
fall through to eager dispatch; the Slang structs and
``dispatch_complex_pointwise`` in ``generic_pointwise_dispatch.py`` provide
the foundation for a future C++ bridge that routes through the JIT complex
path.  ``aten.conj_physical`` does NOT have an upstream lowering and is
fully intercepted here.
"""

from __future__ import annotations

import torch


def _register_complex_lowerings() -> None:
    """Register Vulkan-specific lowerings for complex elementwise ops.

    Called from ``lowerings/__init__.py:register()``.
    """
    from torch._inductor.lowering import lowerings, register_lowering

    aten = torch.ops.aten

    # ── conj_physical ──────────────────────────────────────────────────
    # aten.conj_physical has NO upstream Inductor lowering, so our
    # registration fires first and handles the full op.
    @register_lowering(aten.conj_physical.default, type_promotion_kind=None)
    def _conj_physical(input):
        if not input.get_dtype().is_complex:
            # Real tensor: conj_physical is identity
            return input
        # Decompose: view_as_real → negate imag column → view_as_complex
        real_view = lowerings[aten.view_as_real.default](input)
        real_sizes = real_view.get_size()

        from torch._inductor.ir import Pointwise, ops_wrapper

        def conj_inner_fn(index):
            loader = real_view.make_loader()
            val = loader(index)
            # index is a tuple; index[-1] is 0 (real) or 1 (imag).
            # Use ops to emit a conditional negate at codegen time.
            is_imag = ops_wrapper("eq")(
                index[-1], ops_wrapper("constant")(1, torch.int64)
            )
            return ops_wrapper("where")(is_imag, -val, val)

        conj_real = Pointwise.create(
            device=real_view.get_device(),
            dtype=real_view.get_dtype(),
            inner_fn=conj_inner_fn,
            ranges=real_sizes,
        )
        return lowerings[aten.view_as_complex.default](conj_real)

    # ── abs (complex) ──────────────────────────────────────────────────
    # aten.abs has an upstream register_pointwise lowering; our
    # registration here would be ignored (first-come-first-served loss).
    # Complex abs still falls through to eager for now.

    # ── add / sub / mul / div (complex) ─────────────────────────────────
    # All have upstream register_pointwise / register_lowering entries.
    # These complex ops fall through to eager; the Slang float2 structs
    # in pointwise.slang and dispatch_complex_pointwise() provide the
    # JIT-shader foundation for wiring up in a future C++ bridge update.
