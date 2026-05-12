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
    mask = (result != 0.0) | (input_tensor == 0.0)
    return result, mask


# ── Registration ─────────────────────────────────────────────────────────

_installed = False


def install() -> None:
    """Register PrivateUse1 impls for RNG ops.  Idempotent."""
    global _installed
    if _installed:
        return
    _installed = True

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
