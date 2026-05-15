"""Philox RNG template callers.

Provides rendering, dispatch, and installation for the Philox RNG Slang
template (uniform, normal, fused_dropout).
"""

from __future__ import annotations

import hashlib
import os
import struct
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

from ...buffer_pool import pool_acquire, pool_acquire_scratch, pool_release_scratch
from ...vulkan_template import _load_slang_template
from ...vulkan_template_caller import _dtype_to_slang

_rng_installed = False

_PHILOX_RNG_CACHE: dict[tuple, str] = {}


def _render_philox_rng(
    output_dtype: str = "float",
    rng_mode: str = "uniform",
    fused_dropout: bool = False,
) -> str:
    """Render the philox_rng Jinja2 template with the given parameters.

    Args:
        output_dtype: Slang type string — ``"float"`` (f32), ``"half"`` (f16),
                      or ``"uint"`` (bf16).
        rng_mode: ``"uniform"`` (1 output per thread) or ``"normal"``
                  (Box-Muller, 2 outputs per thread).
        fused_dropout: If True, the shader applies a dropout mask + scale
                       after RNG and reads from a second input binding.
    """
    from jinja2 import Environment

    num_outputs = 2 if rng_mode == "normal" else 1
    key = (output_dtype, rng_mode, fused_dropout)
    if key in _PHILOX_RNG_CACHE:
        return _PHILOX_RNG_CACHE[key]

    src = _load_slang_template("philox_rng")
    if not src:
        raise RuntimeError("philox_rng.py.jinja template not found")

    env = Environment()
    tmpl = env.from_string(src)
    rendered = tmpl.render(
        output_dtype=output_dtype,
        rng_mode=rng_mode,
        fused_dropout=fused_dropout,
        num_outputs=num_outputs,
    )
    _PHILOX_RNG_CACHE[key] = rendered
    return rendered


def _dispatch_philox_rng(
    output_dtype: str,
    rng_mode: str,
    fused_dropout: bool,
    total_elements: int,
    seed_lo: int,
    seed_hi: int,
    offset: int,
    output: torch.Tensor,
    output2: torch.Tensor | None = None,
    input_tensor: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    cache_key: str = "",
) -> None:
    """Dispatch the Philox RNG template as a standalone compute shader.

    Args:
        output_dtype: Slang type for the output buffer(s).
        rng_mode: ``"uniform"`` or ``"normal"``.
        fused_dropout: Whether to apply dropout mask + scale.
        total_elements: Number of random samples to generate.
        seed_lo, seed_hi: 64-bit Philox seed split into two u32s.
        offset: Starting offset in the Philox counter sequence.
        output: Primary output tensor (must be pre-allocated).
        output2: Secondary output for ``rng_mode="normal"`` (Box-Muller).
        input_tensor: Input tensor for fused dropout mode.
        dropout_p: Dropout probability (only for fused_dropout).
        cache_key: Stable cache key for SPIR-V compilation caching.
    """
    from ...runtime import compile_and_dispatch

    numel = total_elements
    threadgroup_size = 256
    grid_x = (numel + threadgroup_size - 1) // threadgroup_size

    # Build push constants: total_elements, seed_lo, seed_hi, offset, [dropout_p]
    # Layout matches the PC struct in philox_rng.py.jinja:
    #   uint total_elements; uint seed_lo; uint seed_hi; uint offset; [float dropout_p;]
    if fused_dropout:
        pc = struct.pack("4If", numel, seed_lo, seed_hi, offset, dropout_p)
    else:
        pc = struct.pack("4I", numel, seed_lo, seed_hi, offset)

    if not cache_key:
        cache_key = (
            f"slang_philox_{rng_mode}"
            f"{'_dropout' if fused_dropout else ''}"
            f"_{output_dtype}"
        )

    src = _render_philox_rng(
        output_dtype=output_dtype,
        rng_mode=rng_mode,
        fused_dropout=fused_dropout,
    )

    tensors: list[torch.Tensor] = [output]
    if rng_mode == "normal" and output2 is not None:
        tensors.append(output2)
    if fused_dropout and input_tensor is not None:
        tensors.append(input_tensor.contiguous())

    if not output.is_contiguous():
        output = output.contiguous()
        tensors[0] = output

    num_outs = len(tensors) - (1 if fused_dropout else 0)

    compile_and_dispatch(
        src,
        tensors,
        grid_x,
        1,
        1,
        push_constants=pc,
        num_outputs=num_outs,
        cache_key=cache_key,
    )


class _SlangPhiloxRNG:
    """Picklable callable for standalone Philox RNG dispatch.

    Caches the rendered Slang source + cache_key per dtype on the instance
    so the per-dispatch path skips the Jinja-render dict lookup.

    ``rng_mode`` selects the template variant:
    - ``"uniform"`` — one float32 output per element.
    - ``"normal"`` — two float32 outputs per element (Box-Muller).
    - ``"fused_dropout"`` — uniform RNG + dropout mask + scale in one pass.
    """

    __slots__ = ("rng_mode", "fused_dropout", "__name__", "_per_dtype")

    def __init__(self, rng_mode: str = "uniform", fused_dropout: bool = False):
        self.rng_mode = rng_mode
        self.fused_dropout = fused_dropout
        self.__name__ = f"slang_philox_{rng_mode}{'_dropout' if fused_dropout else ''}"
        self._per_dtype: dict[str, tuple[str, str, str]] = {}

    def _src_key_dtype(self, dtype_s: str) -> tuple[str, str, str]:
        """Return (src, cache_key, output_dtype_str) for the given dtype."""
        cached = self._per_dtype.get(dtype_s)
        if cached is not None:
            return cached

        output_dtype = dtype_s
        src = _render_philox_rng(
            output_dtype=output_dtype,
            rng_mode=self.rng_mode,
            fused_dropout=self.fused_dropout,
        )
        cache_key = (
            f"slang_philox_{self.rng_mode}"
            f"{'_dropout' if self.fused_dropout else ''}"
            f"_{dtype_s}"
        )
        result = (src, cache_key, output_dtype)
        self._per_dtype[dtype_s] = result
        return result

    def __call__(
        self,
        size: list[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        seed_lo: int = 0,
        seed_hi: int = 0,
        offset: int = 0,
        out: torch.Tensor | None = None,
        input_tensor: torch.Tensor | None = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        """Generate random values via the Philox template.

        Args:
            size: Output shape.
            dtype: Output dtype (f32, f16, bf16).
            device: Target Vulkan device.
            seed_lo, seed_hi: 64-bit Philox seed split into two u32s.
            offset: Starting counter offset.
            out: Pre-allocated output tensor (or None to allocate).
            input_tensor: Input for fused dropout mode.
            dropout_p: Dropout probability (fused_dropout only).

        Returns:
            Output tensor with random values.
        """
        if out is None:
            out = pool_acquire(tuple(size), dtype, device)
            if out is None:
                out = torch.empty(size, dtype=dtype, device=device)

        dtype_s = _dtype_to_slang(dtype)
        src, cache_key, output_dtype = self._src_key_dtype(dtype_s)

        # T6.7: ``output2`` is intermediate scratch — Philox normal-mode
        # produces 2 values per counter and we discard the second one.
        # Route it through the buffer pool's ``scratch`` bucket so the
        # storage is reused across training steps instead of leaking via
        # ``torch.empty`` per call.
        output2: torch.Tensor | None = None
        if self.rng_mode == "normal":
            output2 = pool_acquire_scratch(tuple(size), dtype, device)
            if output2 is None:
                output2 = torch.empty(size, dtype=dtype, device=device)

        total_elements = out.numel()

        _dispatch_philox_rng(
            output_dtype=output_dtype,
            rng_mode=self.rng_mode,
            fused_dropout=self.fused_dropout,
            total_elements=total_elements,
            seed_lo=seed_lo,
            seed_hi=seed_hi,
            offset=offset,
            output=out,
            output2=output2,
            input_tensor=input_tensor,
            dropout_p=dropout_p,
            cache_key=cache_key,
        )

        if self.rng_mode == "normal":
            # For the lowering path, just return the first output.
            # _philox_normal generates 2 values per counter; we consume both
            # but only return the first. Future: pack both as complex or
            # return a tuple for fused applications.
            # T6.7: release the discarded second-output scratch back to
            # the pool. The caller drops its local reference immediately.
            if output2 is not None:
                pool_release_scratch(output2)
                output2 = None
            return out
        return out

    def __reduce__(self):
        return (_SlangPhiloxRNG, (self.rng_mode, self.fused_dropout))


def _philox_seed_from_torch() -> tuple[int, int]:
    """Derive a deterministic Philox seed from the global PyTorch RNG state.

    Returns (seed_lo, seed_hi) as two unsigned 32-bit integers. Uses the
    CPU generator's current seed so ``torch.manual_seed(...)`` controls
    reproducibility across Vulkan compiled graphs.
    """
    gen = torch.default_generator
    raw_state = gen.get_state()
    state_bytes = (
        raw_state.tobytes()
        if hasattr(raw_state, "tobytes")
        else str(raw_state).encode()
    )
    h = hashlib.sha256(state_bytes).digest()[:8]
    seed64 = int.from_bytes(h, "little")
    seed_lo = seed64 & 0xFFFFFFFF
    seed_hi = (seed64 >> 32) & 0xFFFFFFFF
    return seed_lo, seed_hi


def install_external_rng() -> None:
    """Register Vulkan Philox RNG template lowerings for ``aten.rand``,
    ``aten.randn``, ``aten.uniform``, and ``aten.native_dropout``.

    Analogous to ``install_external_mm()`` / ``install_external_addmm()``:
    intercepts the aten ops at the Inductor lowering level and routes them
    through the ``philox_rng.py.jinja`` template instead of the default
    ExternKernel fallback path.

    Safe to call multiple times — only installs once.
    """
    global _rng_installed
    if _rng_installed:
        return
    _rng_installed = True

    import torch
    from torch._inductor import config as inductor_config
    from torch._inductor import lowering as _L
    from torch._inductor.lowering import (
        fallback_handler,
        register_lowering,
    )

    aten = torch.ops.aten

    # Force fallback_random so aten.rand/randn reach our lowering instead
    # of being caught by replace_random.py's FX pass.
    if not inductor_config.fallback_random:
        inductor_config.fallback_random = True

    # Capture original fallbacks (from upstream lowering.py) so we can
    # delegate to them for non-Vulkan tensors.
    _orig_rand_lowering = _L.lowerings.get(aten.rand.default)
    _orig_randn_lowering = _L.lowerings.get(aten.randn.default)
    _orig_uniform_lowering = _L.lowerings.get(aten.uniform.default)
    _orig_dropout_lowering = _L.lowerings.get(aten.native_dropout.default)

    # Pre-build the three template callables.
    _uniform_rng = _SlangPhiloxRNG(rng_mode="uniform")
    _normal_rng = _SlangPhiloxRNG(rng_mode="normal")
    _dropout_rng = _SlangPhiloxRNG(rng_mode="uniform", fused_dropout=True)

    @register_lowering(aten.rand.default, type_promotion_kind=None)
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
            return fallback_handler(aten.rand.generator)(
                size,
                dtype=dtype,
                layout=layout,
                device=device,
                pin_memory=pin_memory,
                generator=generator,
            )
        try:
            dev = device if device is not None else torch.device("cpu")
            if isinstance(dev, str):
                dev = torch.device(dev)
        except Exception:
            dev = torch.device("cpu")

        if dev.type != "vulkan":
            if _orig_rand_lowering is not None:
                return _orig_rand_lowering(
                    size,
                    dtype=dtype,
                    layout=layout,
                    device=device,
                    pin_memory=pin_memory,
                )
            return NotImplemented

        out_dtype = dtype if dtype is not None else torch.float32
        if out_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            if _orig_rand_lowering is not None:
                return _orig_rand_lowering(
                    size,
                    dtype=dtype,
                    layout=layout,
                    device=device,
                    pin_memory=pin_memory,
                )
            return NotImplemented

        # CP.9 / TRAIN.12: Defer to FallbackKernel so the PhiloxState
        # offset advances at runtime instead of being baked in at
        # trace time.  The PrivateUse1 dispatch in philox_dispatch.py
        # calls get_philox_state().advance(n) and passes the dynamic
        # offset to the Slang Philox template.
        import torch.utils._pytree as pytree
        from torch._inductor import ir

        return pytree.tree_map(
            TensorBox.create,
            ir.FallbackKernel.create(
                aten.rand.default,
                size,
                dtype=out_dtype,
                layout=layout,
                device=dev,
                pin_memory=pin_memory,
            ),
        )

    @register_lowering(aten.randn.default, type_promotion_kind=None)
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
            return fallback_handler(aten.randn.generator)(
                size,
                dtype=dtype,
                layout=layout,
                device=device,
                pin_memory=pin_memory,
                generator=generator,
            )
        try:
            dev = device if device is not None else torch.device("cpu")
            if isinstance(dev, str):
                dev = torch.device(dev)
        except Exception:
            dev = torch.device("cpu")

        if dev.type != "vulkan":
            if _orig_randn_lowering is not None:
                return _orig_randn_lowering(
                    size,
                    dtype=dtype,
                    layout=layout,
                    device=device,
                    pin_memory=pin_memory,
                )
            return NotImplemented

        out_dtype = dtype if dtype is not None else torch.float32
        if out_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            if _orig_randn_lowering is not None:
                return _orig_randn_lowering(
                    size,
                    dtype=dtype,
                    layout=layout,
                    device=device,
                    pin_memory=pin_memory,
                )
            return NotImplemented

        # CP.9 / TRAIN.12: Defer to FallbackKernel (see _vulkan_rand).
        import torch.utils._pytree as pytree
        from torch._inductor import ir

        return pytree.tree_map(
            TensorBox.create,
            ir.FallbackKernel.create(
                aten.randn.default,
                size,
                dtype=out_dtype,
                layout=layout,
                device=dev,
                pin_memory=pin_memory,
            ),
        )

    @register_lowering(aten.uniform.default, type_promotion_kind=None)
    def _vulkan_uniform(self, from_=0, to=1, *, generator=None):
        if generator is not None:
            return fallback_handler(aten.uniform.default)(
                self,
                from_=from_,
                to=to,
                generator=generator,
            )
        try:
            dev = self.get_device()
        except Exception:
            dev = torch.device("cpu")

        if dev.type != "vulkan":
            if _orig_uniform_lowering is not None:
                return _orig_uniform_lowering(self, from_=from_, to=to)
            return NotImplemented

        out_dtype = self.get_dtype()
        if out_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            if _orig_uniform_lowering is not None:
                return _orig_uniform_lowering(self, from_=from_, to=to)
            return NotImplemented

        # CP.9 / TRAIN.12: Defer to FallbackKernel (see _vulkan_rand).
        # aten.uniform is in-place; the FallbackKernel handles the
        # copy-back via the PrivateUse1 dispatch which calls self.copy_.
        import torch.utils._pytree as pytree
        from torch._inductor import ir

        return pytree.tree_map(
            TensorBox.create,
            ir.FallbackKernel.create(
                aten.uniform.default,
                self,
                from_,
                to,
            ),
        )

    @register_lowering(aten.native_dropout.default, type_promotion_kind=None)
    def _vulkan_native_dropout(input_tensor, p, train):
        # During Inductor lowering, ``input_tensor`` is an ``ir.TensorBox`` —
        # not a ``torch.Tensor`` — so we cannot call runtime helpers like
        # ``_dropout_rng(...)`` (which would invoke ``.contiguous()`` and
        # ``torch.empty(...)`` on IR nodes). Instead, defer to the
        # ``FallbackKernel`` path, which emits an ``aten.native_dropout``
        # call resolved at runtime to our PrivateUse1 impl in
        # ``philox_dispatch.py:_vulkan_native_dropout`` (which dispatches the
        # fused Philox+dropout Slang shader). This mirrors upstream's
        # ``fallback_random=True`` lowering behaviour (see
        # ``torch/_inductor/lowering.py::native_dropout``).
        try:
            dev = input_tensor.get_device()
        except Exception:
            dev = torch.device("cpu")

        if dev.type != "vulkan":
            if _orig_dropout_lowering is not None:
                return _orig_dropout_lowering(input_tensor, p, train)
            return NotImplemented

        if not train:
            # Eval mode: return input unchanged, mask = all ones (built via
            # the upstream lowering for ``aten.ones`` so the mask is valid IR).
            from torch._inductor.lowering import lowerings as _L_lowerings

            ones_lowering = _L_lowerings.get(aten.ones.default)
            if ones_lowering is None:
                if _orig_dropout_lowering is not None:
                    return _orig_dropout_lowering(input_tensor, p, train)
                return NotImplemented
            mask = ones_lowering(
                list(input_tensor.get_size()),
                dtype=torch.bool,
                device=dev,
                layout=None,
                pin_memory=None,
            )
            return input_tensor, mask

        out_dtype = input_tensor.get_dtype()
        if out_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            if _orig_dropout_lowering is not None:
                return _orig_dropout_lowering(input_tensor, p, train)
            return NotImplemented

        # Defer to FallbackKernel — emits a runtime call to
        # ``aten.native_dropout.default`` which routes to our PrivateUse1
        # Vulkan impl that runs the fused Philox+dropout shader.
        import torch.utils._pytree as pytree
        from torch._inductor import ir
        from torch._inductor.ir import TensorBox

        return pytree.tree_map(
            TensorBox.create,
            ir.FallbackKernel.create(
                aten.native_dropout.default, input_tensor, p, train
            ),
        )

    # ── Pre-warm the Philox template variants ──────────────────────────
    from ...runtime import _slangc_available, prewarm_compile

    if _slangc_available():
        rng_specs: list[tuple[str, str]] = []
        for dt in ("float", "half", "uint"):
            for mode, dropout in (
                ("uniform", False),
                ("normal", False),
                ("uniform", True),
            ):
                key = f"slang_philox_{mode}{'_dropout' if dropout else ''}_{dt}"
                src = _render_philox_rng(
                    output_dtype=dt,
                    rng_mode=mode,
                    fused_dropout=dropout,
                )
                rng_specs.append((key, src))
        prewarm_compile(rng_specs, sync=False)
