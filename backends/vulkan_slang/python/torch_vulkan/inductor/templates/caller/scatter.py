"""Scatter / Gather / IndexPut template callers.

Provides rendering, dispatch, and installation for the scatter_atomic Slang
template.
"""

from __future__ import annotations

import struct as _struct
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

from ...vulkan_template import _load_slang_template

_SCATTER_CACHE: dict[tuple, str] = {}
_scatter_installed = False


def _render_scatter_atomic(
    operation: str,
    dtype: str = "float",
    index_dtype: str = "int",
) -> str:
    """Render the scatter_atomic Jinja2 template.

    Args:
        operation: One of:
            * ``"gather"`` / ``"scatter"`` / ``"scatter_add"`` /
              ``"index_put"`` / ``"index_put_accumulate"`` (T4.5).
            * ``"scatter_reduce_amax"`` / ``"scatter_reduce_amin"`` /
              ``"scatter_reduce_prod"`` / ``"scatter_reduce_mean"`` (T4.11).
        dtype: Slang element type for data buffers ("float", "half").
        index_dtype: Slang type for index buffer ("int", "int64_t").
    """
    from jinja2 import Environment

    valid_ops = {
        "gather",
        "scatter",
        "scatter_add",
        "index_put",
        "index_put_accumulate",
        # T4.11 — non-sum scatter_reduce modes.
        "scatter_reduce_amax",
        "scatter_reduce_amin",
        "scatter_reduce_prod",
        "scatter_reduce_mean",
    }
    if operation not in valid_ops:
        raise ValueError(
            f"Unknown scatter operation '{operation}'. Must be one of: {sorted(valid_ops)}"
        )

    key = (operation, dtype, index_dtype)
    if key in _SCATTER_CACHE:
        return _SCATTER_CACHE[key]

    src = _load_slang_template("scatter_atomic")
    if not src:
        raise RuntimeError("scatter_atomic.slang template not found")

    env = Environment()
    tmpl = env.from_string(src)
    rendered = tmpl.render(
        operation=operation,
        dtype=dtype,
        index_dtype=index_dtype,
    )
    _SCATTER_CACHE[key] = rendered
    return rendered


def _dispatch_scatter_atomic(
    operation: str,
    numel: int,
    src_numel: int,
    out_numel: int,
    output: torch.Tensor,
    src: torch.Tensor,
    indices: torch.Tensor,
    dtype: str = "float",
    index_dtype: str = "int",
    cache_key: str = "",
    count_buffer: torch.Tensor | None = None,
) -> None:
    """Dispatch the scatter/gather/index_put template as a compute shader.

    Args:
        operation: ``"gather"``, ``"scatter"``, ``"scatter_add"``,
                   ``"index_put"``, ``"index_put_accumulate"`` or one of the
                   T4.11 reduce modes (``"scatter_reduce_amax"``,
                   ``"scatter_reduce_amin"``, ``"scatter_reduce_prod"``,
                   ``"scatter_reduce_mean"``).
        numel: Number of work items (= number of indices to process).
        src_numel: Element count of the source/values buffer.
        out_numel: Element count of the output buffer.
        output: Output tensor (must be pre-allocated).
        src: Source/values tensor.
        indices: Index tensor (int32 or int64).
        dtype: Slang type string for data buffers.
        index_dtype: Slang type string for the index buffer.
        cache_key: Stable cache key for SPIR-V compilation caching.
        count_buffer: Required for ``operation="scatter_reduce_mean"`` —
                      a uint32 tensor of length ``out_numel`` that is
                      atomically incremented per landed element so the
                      caller can divide for the mean.  Ignored for all
                      other operations.
    """
    from ...runtime import compile_and_dispatch

    threadgroup_size = 256
    grid_x = (numel + threadgroup_size - 1) // threadgroup_size

    # Push constants: numel, src_numel, out_numel
    pc = _struct.pack("3I", numel, src_numel, out_numel)

    if not cache_key:
        cache_key = f"slang_scatter_{operation}_{dtype}_{index_dtype}"

    src_rendered = _render_scatter_atomic(
        operation=operation,
        dtype=dtype,
        index_dtype=index_dtype,
    )

    # Ensure all tensors are contiguous before dispatch.
    # The shader uses flat (linear) indexing, so views with non-default
    # strides would cause silently-wrong results.
    out_contig = output.contiguous()
    src_contig = src.contiguous()
    idx_contig = indices.contiguous()

    # The C++ dispatch_shader marks the **last** num_outputs buffers as
    # outputs for dirty-buffer / barrier tracking.  Place the output(s)
    # last so the tracking is correct.  Mean-mode binds a second output
    # buffer (the per-target count) immediately after `out` to match the
    # KernelArgs struct field order in scatter_atomic.slang.
    tensors: list[torch.Tensor] = [src_contig, idx_contig, out_contig]
    num_outputs = 1
    if operation == "scatter_reduce_mean":
        if count_buffer is None:
            raise ValueError(
                "scatter_reduce_mean requires a `count_buffer` (uint32 "
                "tensor of length out_numel) so the post-pass divide can "
                "compute the mean."
            )
        tensors.append(count_buffer.contiguous())
        num_outputs = 2

    # If the caller's output was non-contiguous we must copy the result
    # back into the original tensor after the dispatch completes.
    needs_copy_back = out_contig.data_ptr() != output.data_ptr()

    compile_and_dispatch(
        src_rendered,
        tensors,
        grid_x,
        1,
        1,
        push_constants=pc,
        num_outputs=num_outputs,
        cache_key=cache_key,
    )

    if needs_copy_back:
        output.copy_(out_contig)


def install_external_scatter() -> None:
    """Register Vulkan scatter/gather/index_put template lowerings.

    Intercepts ``aten.gather``, ``aten.scatter_add``, ``aten.scatter.src``,
    and ``aten.index_put`` at the Inductor lowering level and routes them
    through the ``scatter_atomic.slang`` template instead of the default
    ExternKernel fallback path.

    Analogous to ``install_external_rng()`` for RNG ops.
    Safe to call multiple times — only installs once.
    """
    global _scatter_installed
    if _scatter_installed:
        return
    _scatter_installed = True

    # We don't replace the lowering — we rely on the existing Inductor
    # codegen for scatter/gather/index_put, which already works correctly
    # via indirect-indexing + atomic-add (see TestGatherScatterAdd,
    # TestIndexSelectAndScatterCodegen).  This install hook pre-warms the
    # template variants so that when the FxPatternRegistry (Track 4) or
    # template_registry routes a SCATTER-class op through the template
    # pipeline, the SPIR-V is already cached.
    from ...runtime import _slangc_available, prewarm_compile

    if not _slangc_available():
        return

    specs: list[tuple[str, str]] = []
    operations = (
        "gather",
        "scatter",
        "scatter_add",
        "index_put",
        "index_put_accumulate",
        # T4.11 — non-sum scatter_reduce modes.
        "scatter_reduce_amax",
        "scatter_reduce_amin",
        "scatter_reduce_prod",
        "scatter_reduce_mean",
    )
    for op in operations:
        for dt in ("float", "half"):
            for idt in ("int", "int64_t"):
                key = f"slang_scatter_{op}_{dt}_{idt}"
                src = _render_scatter_atomic(
                    operation=op,
                    dtype=dt,
                    index_dtype=idt,
                )
                specs.append((key, src))
    prewarm_compile(specs, sync=False)
