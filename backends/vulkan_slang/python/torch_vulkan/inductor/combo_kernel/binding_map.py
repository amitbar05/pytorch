"""Global binding-map construction for Vulkan combo-kernel fusion.

Assigns each outer buffer name (the wrapper-visible ``buf`` name) a unique
Slang binding slot across all subkernels.  Inner names (``in_ptr0``,
``out_ptr1``, ``inout_ptr0``) that collide across subkernels get prefixed
deduplicated global names.  Inplace buffers are declared ``RW`` so the
body's writes are valid l-values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..kernel import VulkanKernel


def build_global_binding_map(
    subkernels: list[tuple["VulkanKernel", int]],
) -> tuple[
    list[tuple[str, str, str]],  # in_decls: [(dtype_str, global_name, outer)] read-only
    list[
        tuple[str, str, str]
    ],  # rw_decls: [(dtype_str, global_name, outer)] read-write (inplace + output)
    list[dict[str, str]],  # per-subkernel inner->global rename map
]:
    """Build the global binding map across all subkernels.

    Each outer buffer name (the wrapper-visible buf name) gets one binding,
    even if multiple subkernels reference it. Inner names (``in_ptr0``,
    ``out_ptr1``, ``inout_ptr0``) collide across subkernels and get prefixed
    with ``s{idx}_`` if already taken. Inplace buffers are declared ``RW`` so
    the body's ``<inner>[idx] = ...`` writes are valid l-values.
    """
    import torch
    from torch._inductor.codegen.common import InplacedBuffer
    from torch._inductor.virtualized import V

    from ..overrides import DTYPE_TO_SLANG

    outer_to_global: dict[str, str] = {}
    outer_is_rw: dict[str, bool] = {}
    in_decls: list[tuple[str, str, str]] = []
    rw_decls: list[tuple[str, str, str]] = []
    per_sub_maps: list[dict[str, str]] = []
    used_globals: set[str] = set()

    def _dtype_str(outer: str) -> str:
        dtype = V.graph.get_dtype(outer)
        if dtype in (torch.float16, torch.bfloat16):
            return "float"
        base = DTYPE_TO_SLANG.get(dtype, "float")
        # int64_t buffers must be declared as uint2 because the
        # pointwise store emits uint2(...) for int64 values (Slang
        # on Vulkan lacks native 64-bit integer atomics, so we
        # bitcast through uint2).  Matches _binding_dtype in
        # kernel/header.py.
        if base == "int64_t":
            return "uint2"
        return base

    # First pass: discover which outer buffers are used as outputs anywhere
    # (so we know to declare them as RW). An outer that appears in BOTH
    # input_buffers and output_buffers in the same subkernel is a
    # read-modify-write (inplace) — we must declare it RW even though
    # ``inplace_buffers`` is empty.
    outers_written: set[str] = set()
    for kernel, _ in subkernels:
        for outer, inplaced in kernel.args.inplace_buffers.items():
            if isinstance(inplaced, InplacedBuffer):
                outers_written.add(outer)
        for outer in kernel.args.output_buffers:
            if outer in kernel.removed_buffers:
                continue
            outers_written.add(outer)

    for idx, (kernel, _) in enumerate(subkernels):
        name_map: dict[str, str] = {}

        def _declare(outer: str, inner: str) -> None:
            if outer in outer_to_global:
                name_map[inner] = outer_to_global[outer]
                return
            # GAP-1.1-B: loop until we find a name not in used_globals.
            # The naive f"s{idx}_{inner}" can itself collide when
            # the same subkernel has TWO different outer buffers that
            # share the same inner name (e.g. both input and output
            # buffers named "in_out_ptr0" in subkernel 1).
            candidate = inner
            while candidate in used_globals:
                candidate = f"s{idx}_{candidate}"
            global_name = candidate
            used_globals.add(global_name)
            outer_to_global[outer] = global_name
            name_map[inner] = global_name
            if outer in outers_written:
                outer_is_rw[outer] = True
                rw_decls.append((_dtype_str(outer), global_name, outer))
            else:
                outer_is_rw[outer] = False
                in_decls.append((_dtype_str(outer), global_name, outer))

        for outer, inplaced in kernel.args.inplace_buffers.items():
            if not isinstance(inplaced, InplacedBuffer):
                continue
            _declare(outer, inplaced.inner_name)

        for outer, inner in kernel.args.input_buffers.items():
            if outer in kernel.args.inplace_buffers:
                continue
            _declare(outer, inner)

        for outer, inner in kernel.args.output_buffers.items():
            if outer in kernel.removed_buffers or outer in kernel.args.inplace_buffers:
                continue
            _declare(outer, inner)

        per_sub_maps.append(name_map)

    return in_decls, rw_decls, per_sub_maps
