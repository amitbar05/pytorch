"""Generic pointwise dispatch via Slang IPointwise/IPointwiseBinary interfaces.

T2.4 — retires ~170 G-category single-op shaders by routing through
``lib/pointwise.slang`` at runtime. Instead of dispatching a pre-compiled
standalone shader (``shaders/unary/abs.slang`` etc.), this module:

1. Looks up the op in ``POINTWISE_TABLE`` to find its Slang struct name.
2. Generates a thin Slang shader that calls ``pointwise_unary_apply<Op>(...)``.
3. Compiles via slangc, caches the SPIR-V, and dispatches.

The generated shader sources are cached by ``(op_struct, arity, dtype)`` so
repeated calls to the same op use the same compiled binary.

Usage (eager path — replaces ``ops::vulkan_abs`` etc.)::

    from .generic_pointwise_dispatch import dispatch_unary_pointwise
    output = dispatch_unary_pointwise("aten.abs", input_tensor)

This is a Python-level shim; the C++ per-op functions can be reduced to a
single call through this dispatcher once it's stable.
"""
from __future__ import annotations

import hashlib
from typing import Optional

import torch

from .generic_dispatch_table import POINTWISE_TABLE, PointwiseEntry
from .runtime import _get_jit_dispatch_cached, _normalize_slang_source

_SRC_CACHE: dict[str, str] = {}
_SPV_CACHE: dict[str, bytes] = {}
_KERNEL_CACHE: dict[str, object] = {}
_THREADGROUP_SIZE = 256

_UNARY_SRC_TEMPLATE = """\
import pointwise;
{% if imports %}
{% for imp in imports %}
import {{ imp }};
{% endfor %}
{% endif %}

struct PC { uint numel; };
[[vk::push_constant]] PC pc;

[[vk::binding(0)]] StructuredBuffer<{{ input_dtype }}> input;
[[vk::binding(1)]] RWStructuredBuffer<{{ output_dtype }}> output;

[numthreads({{ threads }}, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.numel) return;
    float v = (float)input[tid.x];
    output[tid.x] = ({{ output_dtype }}){{ op_struct }}::apply(v);
}
"""

_BINARY_SRC_TEMPLATE = """\
import pointwise;
{% if imports %}
{% for imp in imports %}
import {{ imp }};
{% endfor %}
{% endif %}

struct PC { uint numel; };
[[vk::push_constant]] PC pc;

[[vk::binding(0)]] StructuredBuffer<{{ input_dtype }}> input_a;
[[vk::binding(1)]] StructuredBuffer<{{ input_dtype }}> input_b;
[[vk::binding(2)]] RWStructuredBuffer<{{ output_dtype }}> output;

[numthreads({{ threads }}, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.numel) return;
    float a = (float)input_a[tid.x];
    float b = (float)input_b[tid.x];
    output[tid.x] = ({{ output_dtype }}){{ op_struct }}::apply(a, b);
}
"""


def _render_src(entry: PointwiseEntry, input_dtype: str = "float",
                output_dtype: str = "float") -> str:
    from jinja2 import Environment

    key = f"unary_{entry.op_struct}_{input_dtype}_{output_dtype}" if entry.arity == 1 \
        else f"binary_{entry.op_struct}_{input_dtype}_{output_dtype}"
    if key in _SRC_CACHE:
        return _SRC_CACHE[key]

    env = Environment()
    template = _UNARY_SRC_TEMPLATE if entry.arity == 1 else _BINARY_SRC_TEMPLATE
    tmpl = env.from_string(template)
    rendered = tmpl.render(
        op_struct=entry.op_struct,
        imports=entry.imports,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        threads=_THREADGROUP_SIZE,
    )
    _SRC_CACHE[key] = rendered
    return rendered


def _compile_if_needed(key: str, src: str) -> bytes:
    cache_key = hashlib.sha256(src.encode()).hexdigest()[:16]
    if cache_key in _SPV_CACHE:
        return _SPV_CACHE[cache_key]

    try:
        from torch_vulkan.inductor.runtime import compile_slang_to_spirv
        spv = compile_slang_to_spirv(
            src, f"pointwise_{key}",
            _normalize_slang_source(src),
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to compile generic pointwise shader for {key}: {e}"
        ) from e

    _SPV_CACHE[cache_key] = spv
    return spv


def dispatch_unary_pointwise(
    aten_op: str,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    """Dispatch a unary pointwise op via the generic Slang template.

    Args:
        aten_op: The ATen op name (e.g. ``"aten.abs"``).
        input_tensor: Vulkan tensor, any supported dtype.

    Returns:
        Output Vulkan tensor with same shape and dtype as input.
    """
    entry = POINTWISE_TABLE.get(aten_op)
    if entry is None:
        raise RuntimeError(
            f"Generic pointwise dispatch: {aten_op} not in POINTWISE_TABLE"
        )
    if entry.arity != 1:
        raise RuntimeError(
            f"dispatch_unary_pointwise called for binary op {aten_op}"
        )
    return _dispatch_pointwise_impl(entry, args=(input_tensor,))


def dispatch_binary_pointwise(
    aten_op: str,
    input_a: torch.Tensor,
    input_b: torch.Tensor,
) -> torch.Tensor:
    """Dispatch a binary pointwise op via the generic Slang template.

    Args:
        aten_op: The ATen op name (e.g. ``"aten.add"``).
        input_a, input_b: Vulkan tensors, same shape and dtype.

    Returns:
        Output Vulkan tensor.
    """
    entry = POINTWISE_TABLE.get(aten_op)
    if entry is None:
        raise RuntimeError(
            f"Generic pointwise dispatch: {aten_op} not in POINTWISE_TABLE"
        )
    if entry.arity != 2:
        raise RuntimeError(
            f"dispatch_binary_pointwise called for unary op {aten_op}"
        )
    return _dispatch_pointwise_impl(entry, args=(input_a, input_b))


def _dispatch_pointwise_impl(
    entry: PointwiseEntry,
    args: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    input_dtype = "float"
    output_dtype = "float"

    tensor = args[0]
    if tensor.dtype == torch.float16:
        # Note: float16_t is IEEE half, not bf16. Templates using this
        # would need packed16 path. For now widen to float32.
        input_dtype = "float"
        output_dtype = "float"
    elif tensor.dtype == torch.bfloat16:
        input_dtype = "float"
        output_dtype = "float"

    src = _render_src(entry, input_dtype, output_dtype)
    device = tensor.device
    numel = tensor.numel()

    cache_key = hashlib.sha256(
        f"{entry.op_struct}_{entry.arity}_{input_dtype}_{numel}".encode()
    ).hexdigest()[:12]

    spv = _compile_if_needed(cache_key, src)

    from torch_vulkan.inductor.runtime import make_vulkan_kernel
    n_buffers = entry.arity + 1
    pc_size = 4

    dispatch_cached, dispatch_nopc, get_pipeline = _get_jit_dispatch_cached()

    if cache_key not in _KERNEL_CACHE:
        kernel = make_vulkan_kernel(src, cache_key, n_buffers, pc_size, 0, 1)
        _KERNEL_CACHE[cache_key] = kernel

    kernel = _KERNEL_CACHE[cache_key]

    import struct
    wg_x = (numel + _THREADGROUP_SIZE - 1) // _THREADGROUP_SIZE
    pc_bytes = struct.pack("<I", numel)

    bufs = list(args)
    if entry.arity == 1:
        output = torch.empty_like(tensor)
        bufs.append(output)
    else:
        assert tensor.shape == args[1].shape
        output = torch.empty_like(tensor)
        bufs.append(output)

    kernel(bufs, wg_x, 1, 1, pc_bytes)
    return output


def can_dispatch(aten_op: str) -> Optional[PointwiseEntry]:
    """Check if an aten op is covered by generic pointwise dispatch."""
    return POINTWISE_TABLE.get(aten_op)
