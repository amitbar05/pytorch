"""Helpers for adding new fused Vulkan ops without touching multiple files.

P5.5 — see ``docs/10-inductor-backend.md``. Two decorators:

- ``register_lowering(op)`` — wraps Inductor's ``register_lowering`` and adds
  an automatic vulkan device-type guard so the lowering is a no-op on CPU
  / CUDA tensors. The wrapped function returns ``NotImplemented`` for
  non-vulkan inputs, falling back to the upstream lowering.
- ``register_template(pattern, slang_src, name)`` — declares an FX match
  pattern + Slang shader source. The decorator hashes the source into a
  stable cache key, registers a meta_patch shape-inference impl for the
  pattern's root op (caller supplies it), and wires the template into
  ``ExternKernelChoice`` so the FX matcher can route to it.

The intent is "drop one file in ``python/torch_vulkan/inductor/extensions/``
and the new op is wired end-to-end" — the decorators handle the boilerplate.
"""
from __future__ import annotations

from typing import Any, Callable, Optional


def _is_vulkan(node) -> bool:
    try:
        return node.get_device().type == "vulkan"
    except Exception:
        return False


def register_lowering(
    op,
    *,
    type_promotion_kind: Any = None,
    convert_input_to_inductor_ir: bool = True,
):
    """Register a vulkan-only Inductor lowering for ``op``.

    The wrapped function only fires when the first argument is on the vulkan
    device; otherwise ``NotImplemented`` is returned and Inductor falls back
    to its default lowering (typically the ExternKernel path). This avoids
    accidentally hijacking lowerings on other backends sharing the same
    Inductor session.

    Usage:

        @register_lowering(torch.ops.aten.my_op)
        def _vulkan_my_op(x, ...):
            ...
    """
    from torch._inductor.lowering import register_lowering as _upstream_register

    def decorator(fn: Callable) -> Callable:
        def wrapper(x, *args, **kwargs):
            if not _is_vulkan(x):
                return NotImplemented
            return fn(x, *args, **kwargs)
        wrapper.__name__ = fn.__name__
        wrapper.__qualname__ = fn.__qualname__
        wrapper.__doc__ = fn.__doc__
        return _upstream_register(
            op,
            type_promotion_kind=type_promotion_kind,
            convert_input_to_inductor_ir=convert_input_to_inductor_ir,
        )(wrapper)

    return decorator


def register_template(
    name: str,
    slang_src: str,
    *,
    n_buffers: int,
    n_pc: int = 0,
    pc_size_bytes: int = 0,
    n_outputs: int = 1,
):
    """Register a fused Slang shader as a callable template.

    Returns a callable ``dispatch(*tensors, *push_constants, wg=(x, y, z))``
    that JITs the shader (cached by ``name``) and dispatches it. Lowerings or
    FX passes can call the returned callable to schedule the fused op.

    The Slang source is hashed into a stable cache key so the SPIR-V cache
    survives across Python sessions. Pre-warming the slangc pool with this
    source can be done by calling ``runtime.prewarm_compile([(name, src)])``.
    """
    from . import runtime

    def dispatch(*args, wg=(1, 1, 1)):
        if len(args) != n_buffers + n_pc:
            raise TypeError(
                f"register_template({name!r}) expected "
                f"{n_buffers}+{n_pc} args, got {len(args)}"
            )
        tensors = list(args[:n_buffers])
        if n_pc:
            import struct
            pc_bytes = struct.pack(f"{n_pc}I", *args[n_buffers:])
        else:
            pc_bytes = b""
        spv = runtime.compile_slang_to_spirv(slang_src, cache_key=name)
        runtime.dispatch(name, spv, tensors, wg[0], wg[1], wg[2],
                         pc_bytes, n_outputs)

    dispatch.__name__ = f"vulkan_template_{name}"
    dispatch.cache_key = name
    dispatch.slang_src = slang_src
    dispatch.n_buffers = n_buffers
    dispatch.n_pc = n_pc
    return dispatch


def prewarm_template(dispatch_fn) -> None:
    """Submit a template's Slang source to the slangc pool.

    Use after ``register_template`` to compile the shader at backend
    registration time instead of on first dispatch.
    """
    from . import runtime
    runtime.prewarm_compile([(dispatch_fn.cache_key, dispatch_fn.slang_src)])
