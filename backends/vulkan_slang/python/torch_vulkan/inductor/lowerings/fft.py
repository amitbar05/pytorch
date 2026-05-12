"""OP.4 / OP.10 — FFT op lowerings.

Register ``aten._fft_*`` ops as Inductor lowerings so
``torch.fft.{rfft,irfft,fft,ifft,...}`` does not graph-break under
``torch.compile(backend="inductor")``.

Two tiers of coverage:

**Tier 1 (OP.4 — done 2026-05-08):** ``make_fallback`` routes all three
primitives (``_fft_r2c``, ``_fft_c2c``, ``_fft_c2r``) through the eager
C++ kernels in ``csrc/ops/fft_ops.cpp``, which are registered at
PrivateUse1 in ``csrc/backend/Registration.cpp``.

**Tier 2 (OP.10 / CP.2 — this module):** For small power-of-2 sizes
(N ≤ 1024), the Stockham FFT template (``templates/fft_stockham.py.jinja``)
offers a fused single-dispatch alternative via
``vulkan_template_caller.install_external_fft()``.

The high-level wrappers (``aten.fft_rfft``, ``aten.fft_irfft``,
``aten.fft_fft``, ``aten.fft_ifft``) decompose internally into the
underscore-prefixed primitives at the AOTAutograd / Inductor decomp
layer, so registering lowerings for the underscore variants is
sufficient.
"""

from __future__ import annotations


def _register_fft_lowerings() -> None:
    import torch
    from torch._inductor.lowering import fallbacks, lowerings, make_fallback

    aten = torch.ops.aten

    # Forward 1-D FFT primitives. Higher-level wrappers
    # (``fft_rfft``/``fft_irfft``/``fft_fft``/``fft_ifft`` and the
    # 2-D / N-D variants) decompose into these.
    _fft_ops = [
        aten._fft_r2c.default,  # real → complex (forward)
        aten._fft_c2c.default,  # complex → complex (forward + inverse)
        aten._fft_c2r.default,  # complex → real (inverse)
    ]

    for op in _fft_ops:
        # Idempotent: skip ops that already have a fallback (e.g.
        # ``_fft_r2c`` registered upstream). ``make_fallback`` would
        # raise ``AssertionError`` on a duplicate registration.
        if op in lowerings or op in fallbacks:
            continue
        make_fallback(op)

    # ── OP.10 / CP.2: Install Stockham template choices ──────────────
    # The template lowerings are registered as ``ExternKernelChoice``
    # options alongside the eager fallback. Inductor's autotuner picks
    # the fastest at runtime.
    _register_stockham_template()


def _register_stockham_template() -> None:
    """Install Stockham FFT template as an external FFT choice.

    Delegates to ``vulkan_template_caller.install_external_fft()`` which
    registers ``_fft_c2c`` lowering with both the eager C++ fallback and
    Stockham template choices.
    """
    try:
        from torch_vulkan.inductor.vulkan_template_caller import install_external_fft

        install_external_fft()
    except ImportError:
        # Template infrastructure not available (e.g. during docs build).
        pass
