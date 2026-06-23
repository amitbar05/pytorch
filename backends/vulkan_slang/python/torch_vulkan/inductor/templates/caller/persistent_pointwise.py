"""Persistent pointwise micro-batching template caller.

C6.3 (2026-06-18): Wires the ``persistent_pointwise.slang`` template as a
Python-callable dispatch.  Multiple small pointwise ops (identity, relu,
sigmoid, tanh, gelu, add_scalar, mul_scalar, sub, pow, fill) are batched
into a single GPU dispatch via a grid-stride loop.

The OpRange metadata lives in a storage buffer (not push constants) to
stay within the RDNA1 128B push-constant limit.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Optional

import torch

from ...runtime.dispatch import compile_and_dispatch
from ...vulkan_template import _load_slang_template


# Op type enum — must match the switch cases in persistent_pointwise.slang
OP_IDENTITY = 0
OP_RELU = 1
OP_SIGMOID = 2
OP_TANH = 3
OP_GELU_TANH = 4
OP_ADD_SCALAR = 5
OP_MUL_SCALAR = 6
OP_SUB = 7
OP_POW = 8
OP_FILL = 9

# Struct format for a single OpRange: 5 × uint32 + 1 × float32 = 24 bytes
_OP_RANGE_STRUCT = struct.Struct("IIIII f")

# Struct format for PushConstants: 4 × uint32 = 16 bytes
_PC_STRUCT = struct.Struct("IIII")

# Render cache (template has no Jinja variables → single cached instance)
_render_cache: Optional[str] = None

# Default spec constants
DEFAULT_MAX_INPUT_BUFS = 8
DEFAULT_MAX_OUTPUT_BUFS = 8
DEFAULT_THREADS = 64
DEFAULT_MAX_OPS = 32


def _render() -> str:
    """Load and render the persistent_pointwise.slang template."""
    global _render_cache
    if _render_cache is not None:
        return _render_cache
    src = _load_slang_template("persistent_pointwise")
    if not src:
        raise RuntimeError("persistent_pointwise.slang template not found")
    _render_cache = src
    return src


def dispatch_persistent_pointwise(
    ops: list[tuple[int, torch.Tensor, torch.Tensor, float]],
    *,
    max_input_bufs: int = DEFAULT_MAX_INPUT_BUFS,
    max_output_bufs: int = DEFAULT_MAX_OUTPUT_BUFS,
    threads: int = DEFAULT_THREADS,
    max_ops: int = DEFAULT_MAX_OPS,
) -> None:
    """Batch-dispatch multiple pointwise operations through a single kernel.

    Each operation is a tuple of ``(op_type, input_tensor, output_tensor, scalar)``.
    The input and output tensors must be Vulkan-allocated (device='vulkan:0').

    Binding layout (ParameterBlock<KernelArgs> at set=0, binding=0):
      tensors[0..max_input_bufs-1]  = args.in_bufs[i]
      tensors[max_input_bufs..max_input_bufs+max_output_bufs-1] = args.out_bufs[i]
      tensors[max_input_bufs+max_output_bufs] = args.op_ranges

    Args:
        ops: List of (op_type, input, output, scalar) tuples.
        max_input_bufs: Max input buffer slots (spec constant 42).
        max_output_bufs: Max output buffer slots (spec constant 43).
        threads: Threads per workgroup (spec constant 44).
        max_ops: Max op ranges per dispatch (spec constant 45).
    """
    if not ops:
        return
    if len(ops) > max_ops:
        raise ValueError(
            f"Too many ops ({len(ops)}) for persistent_pointwise "
            f"(max {max_ops}); split into multiple dispatches"
        )

    # Collect unique input/output buffers and assign IDs.
    in_buf_map: dict[int, int] = {}  # data_ptr → buf_id
    out_buf_map: dict[int, int] = {}
    in_bufs: list[torch.Tensor] = []
    out_bufs: list[torch.Tensor] = []
    dummy = torch.empty(1, dtype=torch.float32, device="vulkan:0")

    for _op_type, inp, out, _scalar in ops:
        in_ptr = inp.data_ptr()
        if in_ptr not in in_buf_map:
            if len(in_bufs) >= max_input_bufs:
                raise ValueError(
                    f"Too many unique input buffers ({len(in_bufs) + 1}), "
                    f"max {max_input_bufs}"
                )
            in_buf_map[in_ptr] = len(in_bufs)
            in_bufs.append(inp)
        out_ptr = out.data_ptr()
        if out_ptr not in out_buf_map:
            if len(out_bufs) >= max_output_bufs:
                raise ValueError(
                    f"Too many unique output buffers ({len(out_bufs) + 1}), "
                    f"max {max_output_bufs}"
                )
            out_buf_map[out_ptr] = len(out_bufs)
            out_bufs.append(out)

    # Build OpRange buffer data
    total_elements = 0
    op_range_bytes = bytearray()
    for op_type, inp, out, scalar in ops:
        numel = inp.numel()
        start_idx = total_elements
        total_elements += numel
        in_id = in_buf_map[inp.data_ptr()]
        out_id = out_buf_map[out.data_ptr()]
        op_range_bytes.extend(
            _OP_RANGE_STRUCT.pack(start_idx, numel, op_type, in_id, out_id, scalar)
        )

    # Upload OpRange data to a Vulkan buffer
    op_range_tensor = torch.frombuffer(
        bytearray(op_range_bytes), dtype=torch.uint8
    ).to(device="vulkan:0")

    # Pack push constants: total_elements, num_ops, num_input_bufs, num_output_bufs
    pc_bytes = _PC_STRUCT.pack(
        total_elements, len(ops), len(in_bufs), len(out_bufs)
    )

    # Compute grid size: ceil(total_elements / threads), capped at 1024 WGs
    wg_x = min(1024, (total_elements + threads - 1) // threads)

    # Build flat tensor list matching SPIR-V binding order:
    #   in_bufs[0..max-1], out_bufs[0..max-1], op_ranges
    tensors: list[torch.Tensor] = []
    for i in range(max_input_bufs):
        tensors.append(in_bufs[i] if i < len(in_bufs) else dummy)
    for i in range(max_output_bufs):
        tensors.append(out_bufs[i] if i < len(out_bufs) else dummy)
    tensors.append(op_range_tensor)

    # Build spec constants list
    spec_constants: list[tuple[int, int]] = [
        (42, max_input_bufs),
        (43, max_output_bufs),
        (44, threads),
        (45, max_ops),
    ]

    # Compute cache key from op structure (stable across tensor identity)
    cache_key = hashlib.sha256(
        _render().encode()
        + struct.pack("IIII", max_input_bufs, max_output_bufs, threads, max_ops)
        + bytes(op_range_bytes)
    ).hexdigest()

    compile_and_dispatch(
        _render(),
        tensors,
        wg_x,
        wg_y=1,
        wg_z=1,
        push_constants=pc_bytes,
        num_outputs=len(out_bufs),
        entry="computeMain",
        cache_key=cache_key,
        spec_constants=spec_constants,
    )
