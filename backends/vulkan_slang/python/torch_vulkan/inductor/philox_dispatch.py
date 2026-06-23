"""Track 4.6 — Philox RNG PrivateUse1 dispatch registration.

Registers ``aten.rand``, ``aten.randn``, ``aten.uniform``, and
``aten.native_dropout`` implementations for the Vulkan (PrivateUse1)
dispatch key using the ``philox_rng.py.jinja`` template.

This intercepts both eager-mode calls AND Inductor's ``FallbackKernel``
path (when ``config.fallback_random=True``).  Without this module,
the C++ backend's generic RNG impl would handle Vulkan RNG ops — we
want the Slang Philox template instead.
"""

from __future__ import annotations

from typing import Optional

import torch

from .philox_state import get_philox_state, reset_philox_state  # noqa: F401  # CP.9

# ── Dispatch helpers ─────────────────────────────────────────────────────


def _dispatch_rand(size, *, dtype, device, seed_lo, seed_hi, offset=0):
    """Standalone Philox uniform dispatch via the template caller."""
    from torch_vulkan.inductor.vulkan_template_caller import (
        _SlangPhiloxRNG,
    )

    rng = _SlangPhiloxRNG(rng_mode="uniform")
    return rng(
        list(size),
        dtype=dtype,
        device=device,
        seed_lo=seed_lo,
        seed_hi=seed_hi,
        offset=offset,
    )


def _dispatch_randn(size, *, dtype, device, seed_lo, seed_hi, offset=0):
    from torch_vulkan.inductor.vulkan_template_caller import (
        _SlangPhiloxRNG,
    )

    rng = _SlangPhiloxRNG(rng_mode="normal")
    return rng(
        list(size),
        dtype=dtype,
        device=device,
        seed_lo=seed_lo,
        seed_hi=seed_hi,
        offset=offset,
    )


def _dispatch_dropout(input_tensor, p, train, seed_lo, seed_hi, offset=0):
    from torch_vulkan.inductor.vulkan_template_caller import (
        _SlangPhiloxRNG,
    )

    if not train:
        mask = torch.ones(
            input_tensor.shape, dtype=torch.bool, device=input_tensor.device
        )
        return input_tensor, mask
    rng = _SlangPhiloxRNG(rng_mode="uniform", fused_dropout=True)
    result = rng(
        list(input_tensor.shape),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
        seed_lo=seed_lo,
        seed_hi=seed_hi,
        offset=offset,
        input_tensor=input_tensor,
        dropout_p=p,
    )
    # Simplification: the result tensor has already had dropout applied.
    # Elements that are 0 in the result were dropped. Elements that are
    # non-zero were kept (with scaling). This works for the common case
    # where input values aren't exactly 0 before dropout.
    mask = result != 0.0
    return result, mask


# ── Registration ─────────────────────────────────────────────────────────

_installed = False


_rng_lib: "torch.library.Library | None" = None


def install() -> None:
    """Register PrivateUse1 impls for RNG ops.  Idempotent."""
    global _installed, _rng_lib
    if _installed:
        return
    _installed = True

    # ``Library`` must outlive ``install()`` — when the object is
    # garbage-collected PyTorch unregisters every kernel it owned.  The
    # historical decorator path (``@torch.library.impl(lib, …)``) hides
    # this requirement because the produced wrapper closes over ``lib``;
    # the direct ``lib.impl(name, fn)`` call below does not, so we have
    # to keep a module-level reference.
    _rng_lib = torch.library.Library("aten", "IMPL", "PrivateUse1")

    @torch.library.impl(_rng_lib, "rand")
    def _vulkan_rand(
        size,
        *,
        dtype=None,
        layout=None,
        device=None,
        pin_memory=None,
        generator=None,
    ):
        if generator is not None:
            return NotImplemented
        if device is not None and device.type != "vulkan":
            return NotImplemented
        out_dtype = dtype if dtype is not None else torch.float32
        if out_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            return NotImplemented
        state = get_philox_state()
        num_elements = 1
        for s in size:
            num_elements *= s
        offset = state.advance(num_elements)
        return _dispatch_rand(
            size,
            dtype=out_dtype,
            device=device or torch.device("vulkan"),
            seed_lo=state.seed_lo,
            seed_hi=state.seed_hi,
            offset=offset,
        )

    @torch.library.impl(_rng_lib, "randn")
    def _vulkan_randn(
        size,
        *,
        dtype=None,
        layout=None,
        device=None,
        pin_memory=None,
        generator=None,
    ):
        if generator is not None:
            return NotImplemented
        if device is not None and device.type != "vulkan":
            return NotImplemented
        out_dtype = dtype if dtype is not None else torch.float32
        if out_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            return NotImplemented
        state = get_philox_state()
        num_elements = 1
        for s in size:
            num_elements *= s
        offset = state.advance(num_elements)
        return _dispatch_randn(
            size,
            dtype=out_dtype,
            device=device or torch.device("vulkan"),
            seed_lo=state.seed_lo,
            seed_hi=state.seed_hi,
            offset=offset,
        )

    @torch.library.impl(_rng_lib, "uniform")
    def _vulkan_uniform(self, from_=0, to=1, *, generator=None):
        if generator is not None:
            return NotImplemented
        if self.device.type != "vulkan":
            return NotImplemented
        if self.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            return NotImplemented
        state = get_philox_state()
        offset = state.advance(self.numel())
        result = _dispatch_rand(
            list(self.shape),
            dtype=self.dtype,
            device=self.device,
            seed_lo=state.seed_lo,
            seed_hi=state.seed_hi,
            offset=offset,
        )
        if from_ != 0 or to != 1:
            result = result * (to - from_) + from_
        self.copy_(result)
        return self

    @torch.library.impl(_rng_lib, "native_dropout")
    def _vulkan_native_dropout(input_tensor, p, train):
        if input_tensor.device.type != "vulkan":
            return NotImplemented
        if input_tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            return NotImplemented
        state = get_philox_state()
        offset = state.advance(input_tensor.numel())
        return _dispatch_dropout(
            input_tensor, p, train, state.seed_lo, state.seed_hi, offset=offset
        )

    # ── random_.from: integer uniform in [from, to) ──────────────────────
    # ``torch.randint(low, high, size, device="vulkan")`` allocates an
    # integer tensor then calls ``aten::random_.from(self, from, to)`` to
    # fill it.  Before this impl the Vulkan backend errored with
    # ``Operation 'aten::random_.from … is not yet implemented`` and
    # blocked the SmallCNN training E2E (CrossEntropyLoss target build).
    #
    # Implementation: generate a uniform float in [from, to) via the
    # existing philox path, floor it (truncating toward zero is fine for
    # the non-negative range), then cast into the caller's integer dtype.
    #
    # Note: the dot-syntax overload name ``random_.from`` is NOT accepted
    # by the BC-API ``torch.library.impl(lib, name)`` decorator path, so
    # register via the underlying ``Library.impl(op_name, fn)`` method
    # directly with the OpOverload retrieved by name (``from`` is a Python
    # keyword and can't be accessed via attribute syntax).
    def _vulkan_random_from(self, from_, to=None, *, generator=None):
        if generator is not None:
            return NotImplemented
        if self.device.type != "vulkan":
            return NotImplemented
        if to is None:
            # to=None means "fill with the full positive range of the
            # dtype" — uncommon for randint and not what training loops
            # use.  Defer to upstream by returning NotImplemented so
            # PyTorch's dispatcher tries the next key.
            return NotImplemented
        if from_ >= to:
            raise ValueError(
                f"random_.from requires from < to, got from={from_} to={to}"
            )

        # Allowed target dtypes: integer types we'd reasonably randint into.
        if self.dtype not in (
            torch.int64,
            torch.int32,
            torch.int16,
            torch.int8,
            torch.uint8,
            torch.bool,
        ):
            return NotImplemented

        state = get_philox_state()
        offset = state.advance(self.numel())
        uniform = _dispatch_rand(
            list(self.shape),
            dtype=torch.float32,
            device=self.device,
            seed_lo=state.seed_lo,
            seed_hi=state.seed_hi,
            offset=offset,
        )
        # uniform ∈ [0, 1) → scale to [0, range) → floor → shift → cast.
        range_size = float(to) - float(from_)
        scaled = uniform * range_size
        if from_ != 0:
            scaled = scaled + float(from_)
        # ``floor`` is a Vulkan-supported pointwise op; cast handles the
        # final dtype.  Bool is a special case (1-bit) — use truthiness.
        floored = scaled.floor()
        if self.dtype == torch.bool:
            self.copy_(floored != 0)
        else:
            self.copy_(floored.to(self.dtype))
        return self

    _rng_lib.impl("random_.from", _vulkan_random_from)
