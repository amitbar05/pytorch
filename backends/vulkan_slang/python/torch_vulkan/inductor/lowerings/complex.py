"""Complex-dtype elementwise lowerings (OP.20 / M19.7).

Decomposes complex-valued pointwise ops (add / sub / mul / div / abs /
conj_physical) into real-valued pointwise ops over a ``.view(real_dtype)``
projection so they survive Inductor's complex-fallback gate
(``unsupported_input_tensor`` in ``torch/_inductor/lowering.py``).

Why decompositions and not lowerings: upstream's ``graph.py`` consults
``fallback_node_due_to_unsupported_type`` BEFORE looking up the
lowerings dict, so any op with a complex input falls through to the
``FallbackKernel`` (eager) path regardless of whether we registered a
``@register_lowering(aten.X)``. Decompositions, on the other hand, run
at AOTAutograd time (before Inductor sees the FX graph), so we can
replace ``aten.mul.Tensor(complex_a, complex_b)`` with the equivalent
real-valued math BEFORE the unsupported-input check ever fires.

The Slang-side ``OpComplex{Add,Sub,Mul,Div,Conj,Abs}`` structs in
``shaders/lib/pointwise.slang`` and ``dispatch_complex_pointwise`` in
``generic_pointwise_dispatch.py`` remain available for the eager C++
bridge; this module covers the Inductor compile path via the
view-as-real decomposition. The two paths converge: both produce
correct complex results, but the compile path now generates fused
Slang kernels instead of falling back to the (slow) eager dispatch.

Pattern (mirrors upstream ``torch._inductor.decomposition.add``):

    x.view(real_dtype).view(*init, last // 2, 2)  →  (real_part, imag_part)
    apply complex math via real ops
    stack → flatten(start_dim=-2) → view(complex_dtype)

The compile path then sees only real-valued add/sub/mul/etc. — which
Inductor lowers via our existing Slang pointwise codegen — and the
final ``view(complex_dtype)`` is a no-op alias.
"""

from __future__ import annotations

import torch


def _reshape_complex_as_pairs(x: torch.Tensor) -> torch.Tensor:
    """Reshape ``x`` from complex ``[..., last]`` to real
    ``[..., last, 2]`` via ``view(real_dtype)`` + reshape.

    Mirrors ``reshape_tensor_complex`` in upstream's ``aten.add``
    decomp. Assumes ``x.stride()[-1] == 1`` (complex tensors are
    contiguous in the last dim by construction).
    """
    *initial_dims, last_dim = x.shape
    # x.view(real) doubles the last dim. Re-grouping (n, 2) gives us
    # (real, imag) pairs at the innermost level.
    real_view = x.view(x.real.dtype)
    return real_view.view(*initial_dims, last_dim, 2)


def _requires_fallback_complex(t: torch.Tensor) -> bool:
    """Return True if ``t`` cannot be safely re-viewed as its real
    dtype (last stride must be 1)."""
    if t.ndim == 0:
        return False
    return t.stride()[-1] != 1


def _register_complex_lowerings() -> None:
    """Register Vulkan-specific decompositions for complex elementwise ops.

    Called from ``lowerings/__init__.py:register()``. Installs entries
    in Inductor's ``decompositions`` dict so AOTAutograd replaces the
    complex form with view-as-real-and-back BEFORE Inductor sees the
    node. The actual real math then lowers through our existing
    pointwise / Slang path.

    Implementation note: we bypass ``register_decomposition`` (which
    raises on duplicate keys via ``_add_op_to_registry``) and directly
    assign to the ``decompositions`` dict. Upstream already registers
    ``aten.conj_physical`` and ``aten.add`` for complex inputs; our
    entries overwrite theirs intentionally (we match the same
    view-as-real pattern; tests verify behavioral equivalence).
    """
    from torch._inductor.decomposition import decompositions

    aten = torch.ops.aten

    # ── aten.add.Tensor (complex) ─────────────────────────────────────
    # Component-wise: (a.real + b.real, a.imag + b.imag).
    # We override upstream's complex-add decomp because it does
    # ``x = x + 0`` to materialize conj views, which on Vulkan
    # routes ``aten.add.Scalar(complex, 0)`` to the C++ eager path
    # which isn't implemented.
    def _vulkan_complex_add(
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        alpha: torch.types.Number | None = None,
    ) -> torch.Tensor:
        x_is_complex = torch.is_tensor(x) and x.is_complex()
        y_is_complex = torch.is_tensor(y) and y.is_complex()
        if not (x_is_complex and y_is_complex):
            return NotImplemented
        output_size_zero = x.ndim == 0 and y.ndim == 0
        if x.ndim == 0:
            x = x.reshape(1)
        if y.ndim == 0:
            y = y.reshape(1)
        z = y if alpha is None else alpha * y
        if _requires_fallback_complex(x) or _requires_fallback_complex(z):
            return NotImplemented
        complex_type = torch.promote_types(x.dtype, y.dtype)
        x_pairs = _reshape_complex_as_pairs(x)
        z_pairs = _reshape_complex_as_pairs(z)
        result = torch.flatten(x_pairs + z_pairs, start_dim=-2).view(complex_type)
        if output_size_zero:
            return result[0]
        return result

    # ── aten.sub.Tensor (complex) ─────────────────────────────────────
    # Component-wise: (a.real - b.real, a.imag - b.imag).
    def _vulkan_complex_sub(
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        alpha: torch.types.Number | None = None,
    ) -> torch.Tensor:
        x_is_complex = torch.is_tensor(x) and x.is_complex()
        y_is_complex = torch.is_tensor(y) and y.is_complex()
        if not (x_is_complex and y_is_complex):
            return NotImplemented
        # Handle 0-dim by reshape-to-1 then result[0].
        output_size_zero = x.ndim == 0 and y.ndim == 0
        if x.ndim == 0:
            x = x.reshape(1)
        if y.ndim == 0:
            y = y.reshape(1)
        z = y if alpha is None else alpha * y
        if _requires_fallback_complex(x) or _requires_fallback_complex(z):
            return NotImplemented
        complex_type = torch.promote_types(x.dtype, y.dtype)
        # Note: upstream's add decomp does ``x = x + 0`` here to force
        # materialization of conj views. We omit it because the ``+ 0``
        # itself is a complex aten.add.Scalar that the Vulkan C++ side
        # doesn't implement, and the resolve_conj path is handled at
        # AOTAutograd's functionalization layer for our test surface.
        x_pairs = _reshape_complex_as_pairs(x)
        z_pairs = _reshape_complex_as_pairs(z)
        result = torch.flatten(x_pairs - z_pairs, start_dim=-2).view(complex_type)
        if output_size_zero:
            return result[0]
        return result

    # ── aten.mul.Tensor (complex) ─────────────────────────────────────
    # Cross terms:
    #   result.real = a.real*b.real - a.imag*b.imag
    #   result.imag = a.real*b.imag + a.imag*b.real
    def _vulkan_complex_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_is_complex = torch.is_tensor(x) and x.is_complex()
        y_is_complex = torch.is_tensor(y) and y.is_complex()
        if not (x_is_complex and y_is_complex):
            return NotImplemented
        output_size_zero = x.ndim == 0 and y.ndim == 0
        if x.ndim == 0:
            x = x.reshape(1)
        if y.ndim == 0:
            y = y.reshape(1)
        if _requires_fallback_complex(x) or _requires_fallback_complex(y):
            return NotImplemented
        complex_type = torch.promote_types(x.dtype, y.dtype)
        # No ``x + 0`` materialize (would re-enter the Vulkan eager
        # path for complex aten.add.Scalar which isn't implemented).
        x_pairs = _reshape_complex_as_pairs(x)
        y_pairs = _reshape_complex_as_pairs(y)
        # Index into the trailing axis to pull out real / imag.
        ar = x_pairs[..., 0]
        ai = x_pairs[..., 1]
        br = y_pairs[..., 0]
        bi = y_pairs[..., 1]
        zr = ar * br - ai * bi
        zi = ar * bi + ai * br
        # Stack to (..., last, 2) then flatten + reinterpret.
        stacked = torch.stack([zr, zi], dim=-1)
        result = torch.flatten(stacked, start_dim=-2).view(complex_type)
        if output_size_zero:
            return result[0]
        return result

    # ── aten.div.Tensor (complex) ─────────────────────────────────────
    # a/b = (a * conj(b)) / |b|^2
    #     = ((ar*br + ai*bi) / d, (ai*br - ar*bi) / d)
    # where d = br^2 + bi^2.
    def _vulkan_complex_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_is_complex = torch.is_tensor(x) and x.is_complex()
        y_is_complex = torch.is_tensor(y) and y.is_complex()
        if not (x_is_complex and y_is_complex):
            return NotImplemented
        output_size_zero = x.ndim == 0 and y.ndim == 0
        if x.ndim == 0:
            x = x.reshape(1)
        if y.ndim == 0:
            y = y.reshape(1)
        if _requires_fallback_complex(x) or _requires_fallback_complex(y):
            return NotImplemented
        complex_type = torch.promote_types(x.dtype, y.dtype)
        x_pairs = _reshape_complex_as_pairs(x)
        y_pairs = _reshape_complex_as_pairs(y)
        ar = x_pairs[..., 0]
        ai = x_pairs[..., 1]
        br = y_pairs[..., 0]
        bi = y_pairs[..., 1]
        denom = br * br + bi * bi
        zr = (ar * br + ai * bi) / denom
        zi = (ai * br - ar * bi) / denom
        stacked = torch.stack([zr, zi], dim=-1)
        result = torch.flatten(stacked, start_dim=-2).view(complex_type)
        if output_size_zero:
            return result[0]
        return result

    # ── aten.abs.default (complex → real) ─────────────────────────────
    # |z| = sqrt(z.real^2 + z.imag^2). Produces a real tensor of the
    # corresponding float dtype.
    def _vulkan_complex_abs(x: torch.Tensor) -> torch.Tensor:
        if not (torch.is_tensor(x) and x.is_complex()):
            return NotImplemented
        output_size_zero = x.ndim == 0
        if x.ndim == 0:
            x = x.reshape(1)
        if _requires_fallback_complex(x):
            return NotImplemented
        x_pairs = _reshape_complex_as_pairs(x)
        ar = x_pairs[..., 0]
        ai = x_pairs[..., 1]
        result = torch.sqrt(ar * ar + ai * ai)
        if output_size_zero:
            return result[0]
        return result

    # ── aten.conj_physical (complex) ──────────────────────────────────
    # Component-wise negate of the imaginary part: (z.real, -z.imag).
    # Upstream registers an identity decomp for conj_physical on real
    # tensors (returns ``NotImplemented`` for complex, then falls into
    # the lowering path). Our previous lowering-based implementation
    # was unreachable because the complex-fallback gate fired first.
    # Re-implement as a decomposition for complex inputs.
    def _vulkan_complex_conj_physical(x: torch.Tensor) -> torch.Tensor:
        if not (torch.is_tensor(x) and x.is_complex()):
            # Real tensor: conj_physical is identity.
            return x
        output_size_zero = x.ndim == 0
        if x.ndim == 0:
            x = x.reshape(1)
        if _requires_fallback_complex(x):
            return NotImplemented
        x_pairs = _reshape_complex_as_pairs(x)
        ar = x_pairs[..., 0]
        ai = x_pairs[..., 1]
        # Negate the imaginary part.
        stacked = torch.stack([ar, -ai], dim=-1)
        result = torch.flatten(stacked, start_dim=-2).view(x.dtype)
        if output_size_zero:
            return result[0]
        return result

    # Install via direct dict assignment (bypasses register_decomposition's
    # duplicate-registration check; upstream owns conj_physical and we
    # intentionally override it). The upstream global ``decomposition_table``
    # also gets updated so AOTAutograd sees the same entries.
    from torch._decomp import (
        decomposition_table as _aot_decomps,
        global_decomposition_table as _global_decomps,
    )

    _entries = {
        aten.add.Tensor: _vulkan_complex_add,
        aten.sub.Tensor: _vulkan_complex_sub,
        aten.mul.Tensor: _vulkan_complex_mul,
        aten.div.Tensor: _vulkan_complex_div,
        aten.abs.default: _vulkan_complex_abs,
        aten.conj_physical.default: _vulkan_complex_conj_physical,
    }
    for op, fn in _entries.items():
        decompositions[op] = fn
        _aot_decomps[op] = fn
        # The "post_autograd" registry is the one AOTAutograd consults
        # for joint-graph decomp. Stuff our entries into it so they
        # fire at autograd-tracing time too. The dict may not exist on
        # older PT versions; guard.
        try:
            _global_decomps["post_autograd"][op] = fn
        except (KeyError, TypeError):
            pass

    # Sanity: confirm our entries landed.
    assert decompositions.get(aten.add.Tensor) is _vulkan_complex_add
    assert decompositions.get(aten.sub.Tensor) is _vulkan_complex_sub
    assert decompositions.get(aten.mul.Tensor) is _vulkan_complex_mul
    assert decompositions.get(aten.div.Tensor) is _vulkan_complex_div
    assert decompositions.get(aten.abs.default) is _vulkan_complex_abs
    assert (
        decompositions.get(aten.conj_physical.default) is _vulkan_complex_conj_physical
    )
