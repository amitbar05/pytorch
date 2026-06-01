"""Module-level helpers and monkey-patches for wrapper.py.

Extracted from ``wrapper.py`` to keep that file under the 800-line
anti-goal #7 cap (M22.1 split). Imported by ``wrapper.py``; the
module-level install calls run on import.
"""

from __future__ import annotations

import os

import torch
import torch._inductor.compile_fx
from torch._inductor.virtualized import V


# ── M22.2 — _get_alloc_alias_fn / _alloc_alias_fn removed ──────────
# The M17.7 regex post-processor (alloc_alias.py) has been superseded by
# the IR-level pass in alloc_alias_ir.py, which runs inside
# VulkanPythonWrapperCodegen.run_wrapper_ir_passes(). The lazy-import
# shim _get_alloc_alias_fn / _alloc_alias_fn that previously wired the
# regex pass into wrapper.py was dead code after the IR migration landed;
# it is removed here. alloc_alias.py itself is retained as a legacy
# reference (see its module docstring).


def _install_vulkan_skip_alignment_clone() -> None:
    """OP.5 fix — exclude Vulkan tensors from Inductor's alignment check.

    ``compile_fx.get_input_idxs_to_check`` flags every GPU input whose
    ``data_ptr() % 16 != 0`` for runtime cloning via
    ``align_inputs_from_check_idxs`` → ``copy_misaligned_inputs`` →
    ``clone_preserve_strides``. The clone path on a Vulkan tensor goes
    through PrivateUse1's ``aten::clone``, which (per the documented
    ``dispatch_copy_buffer`` 4B/elem bug) silently truncates int64 /
    float64 storage — turning ``[0, 1, 2, 3]`` into ``[0, 1, 0, 0]``
    on every int64 index buffer that flows through compiled graphs.

    Symptom: ``aten.index_put`` / ``embedding_dense_backward`` /
    ``embedding_bag_forward`` produce numerically wrong outputs even
    though the generated Slang kernel and binding emission are correct.
    Root cause is upstream Inductor speculatively cloning misaligned
    GPU inputs; Vulkan tensors carry synthetic data pointers (1, 2, 3,
    …) that *always* fail the ``% 16`` test, so every int64 index
    buffer triggers the truncating-clone bug.

    Fix: monkey-patch the alignment check to skip Vulkan tensors.
    Vulkan kernels read the underlying ``VkBuffer`` via descriptor sets
    that are device-aligned by construction; the host-side ``data_ptr``
    is a synthetic key into our buffer table, not a memory address, so
    "alignment" is meaningless for our backend. Idempotent.
    """
    _orig = torch._inductor.compile_fx.get_input_idxs_to_check
    if getattr(_orig, "_vulkan_skip_alignment_patched", False):
        return

    def _vulkan_aware_get_input_idxs_to_check(inputs, static_input_idxs):
        ids = list(_orig(inputs, static_input_idxs))
        return [
            i
            for i in ids
            if not (
                i < len(inputs)
                and isinstance(inputs[i], torch.Tensor)
                and inputs[i].device.type == "vulkan"
            )
        ]

    _vulkan_aware_get_input_idxs_to_check._vulkan_skip_alignment_patched = True  # type: ignore[attr-defined]
    torch._inductor.compile_fx.get_input_idxs_to_check = (
        _vulkan_aware_get_input_idxs_to_check
    )


_install_vulkan_skip_alignment_clone()


def _install_vulkan_pattern_matcher_tensor_attr_fix() -> None:
    """TR.14 fix — gracefully skip pattern_matcher replacements whose
    subgraph contains a Vulkan tensor with no backing buffer.

    Upstream's ``ReplacementPatternEntry.replace_with_graph`` builds a
    ``Replacer`` (an ``fx.Interpreter``) that walks the replacement
    subgraph and copies its nodes into the main graph. The
    ``get_attr`` branch of ``Replacer.run_node`` (in
    ``torch/_inductor/pattern_matcher.py``) only handles
    ``GraphModule`` attributes (HOP subgraphs); for plain tensor
    constants it raises::

        NotImplementedError(
            f"NYI: replacement_graph.{target} is not a graph module. Got {sub_gm}."
        )

    The f-string materializes the tensor via ``Tensor.__repr__`` →
    ``Tensor.tolist()``. On Vulkan, this crashes with
    ``RuntimeError: Vulkan tensor has no backing buffer`` whenever the
    replacement traced an op (e.g. ``torch.arange`` inside
    ``scatter_upon_const_tensor``) that ``make_fx`` lifted as
    ``self._tensor_constantN`` — those tensors carry FakeTensor
    metadata but no allocated ``VkBuffer`` (synthetic ``data_ptr=0``).

    Fix: monkey-patch ``ReplacementPatternEntry.replace_with_graph``
    so that when the replacement subgraph contains a ``get_attr`` to
    a Vulkan tensor with ``data_ptr=0``, we **skip the replacement**
    entirely and leave the original (un-replaced) FX nodes in the
    main graph. This unblocks compile (no crash) and falls back to
    the canonical lowering for the original op — which is the
    correct path on Vulkan since we can't faithfully reconstruct the
    constant's values from a missing buffer.

    The narrow contract: TR.14's ``RuntimeError: Vulkan tensor has
    no backing buffer`` no longer fires inside the pattern matcher.
    Upstream patterns that *can* produce a valid replacement (no
    Vulkan tensor constants, or tensor constants with real backing)
    still apply normally.

    Idempotent.
    """
    from torch._inductor import pattern_matcher as _pm

    cls = _pm.ReplacementPatternEntry
    if getattr(cls, "_vulkan_tensor_attr_patched", False):
        return

    _orig = cls.replace_with_graph

    def _replacement_has_unbacked_vulkan_tensor(replacement_graph) -> bool:
        """Return True iff the replacement subgraph contains a
        ``get_attr`` whose target resolves to a Vulkan tensor with
        no backing buffer (``data_ptr=0``).
        """
        if hasattr(replacement_graph, "graph"):
            rmod = replacement_graph
            rgraph = replacement_graph.graph
        else:
            rmod = getattr(replacement_graph, "owning_module", None)
            rgraph = replacement_graph
        if rmod is None:
            return False
        for n in rgraph.nodes:
            if n.op != "get_attr":
                continue
            val = getattr(rmod, n.target, None)
            if not isinstance(val, torch.Tensor):
                continue
            if isinstance(val, torch.fx.GraphModule):
                continue
            if val.device.type == "vulkan" and val.data_ptr() == 0:
                return True
        return False

    @staticmethod  # type: ignore[misc]
    def _patched_replace_with_graph(match, graph, replacement_graph, args):
        if _replacement_has_unbacked_vulkan_tensor(replacement_graph):
            return None
        return _orig(match, graph, replacement_graph, args)

    cls.replace_with_graph = _patched_replace_with_graph
    cls._vulkan_tensor_attr_patched = True  # type: ignore[attr-defined]


_install_vulkan_pattern_matcher_tensor_attr_fix()


def _normalize_strides_to_row_major(
    shape: list[int] | tuple[int, ...],
    stride: list[int] | tuple[int, ...],
) -> list[int]:
    """Normalize strides to standard row-major (C-contiguous) order.

    vulkan_empty_strided always creates row-major contiguous buffers
    regardless of the requested strides. Inductor's kernel codegen uses
    the strides returned here to compute load/store memory offsets, so
    we must ensure they match the actual row-major buffer layout.

    For a tensor of shape (d0, d1, ..., dn), row-major strides are:
        stride[n-1] = 1
        stride[i] = product(shape[i+1:]) for i from n-2 down to 0
    """
    ndim = len(shape)
    if ndim == 0:
        return list(stride)
    row_major = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        row_major[i] = row_major[i + 1] * shape[i + 1]
    return row_major


def _trust_inductor() -> bool:
    # P5.11.a.1: default ON. The asserts are debug-only and the wrapper
    # codegen is mature enough that the per-buffer `assert_size_stride`
    # overhead is no longer a load-bearing safety net (CPU oracle covers
    # correctness; the elision saves ~150 µs/step on small workloads).
    # Opt-OUT preserved via `TORCH_VULKAN_TRUST_INDUCTOR=0` for engineers
    # debugging an Inductor codegen bug.
    return os.environ.get("TORCH_VULKAN_TRUST_INDUCTOR", "1") != "0"


# GPU.1 / GPU.2 / GPU.3 — Config-reading helpers cached at module load.
# These are called during codegen (not runtime dispatch) so the env var
# is captured once at graph-compile time.


def _batch_dispatch_enabled() -> bool:
    """GPU.1: batch dispatch submission.

    Disabled by default (2026-06-01) due to ordering bug in multi-layer
    backward wrappers. Batched pointwise kernels can execute AFTER
    synchronous extern dispatches → stale gradients → frozen loss.
    Set ``TORCH_VULKAN_BATCH_DISPATCH=1`` to re-enable.
    """
    return os.environ.get("TORCH_VULKAN_BATCH_DISPATCH", "0") != "0"


def _wrapper_fastpath_enabled() -> bool:
    """GPU.2: Python wrapper hot-path optimizations."""
    return os.environ.get("TORCH_VULKAN_WRAPPER_FASTPATH", "1") != "0"


def _profile_dispatches_enabled() -> bool:
    """GPU.3: per-dispatch profiling."""
    return os.environ.get("TORCH_VULKAN_PROFILE_DISPATCHES", "0") == "1"


# PF.41: lifetime_class names valid for the buffer pool.
_VALID_LIFETIME_CLASSES = frozenset(
    {"parameter", "gradient", "save_for_backward", "transient", "output", "scratch"}
)


def _lifetime_class_for_name(name: str) -> str:
    """Resolve the lifetime_class string for a buffer by name.

    PF.40 annotates joint-graph FX nodes with
    ``node.meta["lifetime_class"]``; the partitioner copies node meta
    into the per-side fw/bw modules, and Inductor's IR Buffer carries
    a back-pointer to its origin FX node via :meth:`get_origin_node`.
    Returns ``"transient"`` (the safest default for intra-step reuse)
    when the lookup misses for any reason — buffer not found, no
    origin node, no annotation, or an unrecognized class name.
    """
    try:
        buf = V.graph.try_get_buffer(name)
    except Exception:
        return "transient"
    if buf is None:
        return "transient"
    try:
        node = buf.get_origin_node()
    except Exception:
        return "transient"
    if node is None:
        return "transient"
    cls = node.meta.get("lifetime_class")
    if cls in _VALID_LIFETIME_CLASSES:
        return cls
    return "transient"
