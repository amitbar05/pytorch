"""VulkanScheduling — SIMDScheduling subclass for the Vulkan Inductor backend."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Optional

import sympy
import torch
from torch._inductor import config as inductor_config
from torch._inductor.codegen.common import BackendFeature
from torch._inductor.codegen.simd import SIMDScheduling
from torch._inductor.utils import get_kernel_metadata
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from .kernel import VulkanKernel

if TYPE_CHECKING:
    from torch._inductor.scheduler import Scheduler, SchedulerNode


def compute_combo_config_key(sub_config_keys) -> str:
    """M-pipeline-7: derive a content-aware cache key for a combo kernel
    from the tuple of its sub-kernels' ``config_key`` strings.

    Why this exists: the prior implementation used the literal string
    ``"combo"`` as the cache key for every combo kernel in the process.
    The ``_reflection_metrics_by_key`` cross-index in
    ``runtime/slangc.py`` then returned whichever combo happened to be
    compiled first, corrupting WG-sizing heuristics for every
    subsequent combo. Same bug class as M-pipeline-3 (single-kernel
    collisions) at a different cache layer.

    Order-sensitive: the combo's emitted Slang lays out gtid ranges and
    binding slots in sub-kernel iteration order, so combos that differ
    only in sub-kernel order ARE structurally different and must map to
    distinct cache slots.

    Key format: ``combo2_n{N}_{hash16}``. The ``combo2_`` prefix bumps
    the cache version so any in-memory entries from the old
    ``"combo"`` format cannot collide with new entries. The ``nN``
    component is human-debuggable (you can read the combo size off the
    key without rehashing). ``hash16`` is a 16-char SHA-1 prefix over
    ``repr(tuple(sub_config_keys))``.
    """
    sub_keys_tuple = tuple(sub_config_keys)
    n = len(sub_keys_tuple)
    return (
        f"combo2_n{n}_"
        f"{hashlib.sha1(repr(sub_keys_tuple).encode()).hexdigest()[:16]}"
    )


class VulkanScheduling(SIMDScheduling):
    kernel_type = VulkanKernel  # type: ignore[assignment]

    _cached_max_storage_bufs: Optional[int] = None

    def __init__(self, scheduler: "Optional[Scheduler]") -> None:
        super().__init__(scheduler)

    @classmethod
    def _get_max_storage_bufs(cls) -> int:
        """Return the device's usable storage buffer binding limit.

        Queries ``maxPerStageDescriptorStorageBuffers`` from the Vulkan
        device properties.  Reserves a margin of 4 slots for:
          * Push-constant / specialization-constant meta-bindings
          * Internal workspace / reduction buffers
          * Optional epilogue / bias / scale auxiliary buffers

        When ``VK_EXT_descriptor_indexing`` is available AND enabled via
        ``TORCH_VULKAN_DESCRIPTOR_INDEXING=1`` (the default), the cap is
        raised to >=256 (N+1.5.c).  When disabled or unavailable, the
        pre-indexing effective limit of ~60 is used.

        The ``TORCH_VULKAN_MAX_STORAGE_BUFS`` env var overrides this
        entire probe — useful for debugging fusion heuristics.

        Falls back to 16 when device query is unavailable (offline / CI
        without a GPU). The Vulkan spec minimum for
        ``maxPerStageDescriptorStorageBuffers`` is 4; RDNA1 advertises 64.
        """
        if cls._cached_max_storage_bufs is not None:
            return cls._cached_max_storage_bufs

        from .config import descriptor_indexing_enabled, max_storage_bufs_override

        # Explicit override always wins (useful for debugging).
        override = max_storage_bufs_override()
        if override is not None:
            cls._cached_max_storage_bufs = override
            return cls._cached_max_storage_bufs

        try:
            from torch._dynamo.device_interface import get_interface_for_device

            iface = get_interface_for_device("vulkan")
            props = iface.Worker.get_device_properties()
            raw = props.max_storage_buffers

            # N+1.5.c: When descriptor indexing is enabled, raise the cap.
            if descriptor_indexing_enabled():
                from .runtime import _descriptor_indexing_supported

                if _descriptor_indexing_supported():
                    # Descriptor indexing removes the per-stage binding cap.
                    # Use a generous limit: min(device_limit - 4, 256).
                    usable = raw - 4
                    cap = max(12, min(usable, 256))
                    cls._cached_max_storage_bufs = cap
                    import logging

                    logging.getLogger(__name__).info(
                        "VK_EXT_descriptor_indexing active — storage buffer "
                        "binding cap raised to %d (device raw limit: %d)",
                        cap,
                        raw,
                    )
                    return cap

            # Pre-descriptor-indexing path: reserve 4 slots; clamp to [12, 60].
            usable = raw - 4
            cls._cached_max_storage_bufs = max(12, min(usable, 60))
        except Exception:
            cls._cached_max_storage_bufs = 16

        import logging

        logging.getLogger(__name__).info(
            "Vulkan storage buffer binding cap: %d",
            cls._cached_max_storage_bufs,
        )
        return cls._cached_max_storage_bufs

    @classmethod
    def get_backend_features(cls, device):
        # FOREACH enables the combo-kernel path. `VulkanComboKernel` rewrites
        # subkernel buffers + locals into globally unique names so the merged
        # shader is well-formed (see `vulkan_combo_kernel.py`).
        #
        # F.2 / F.3 reconcile (2026-05-08): the 2026-05-08 backend audit
        # claimed SCAN/SORT/TUPLE_REDUCTION were advertised without
        # codegen, but inspection of `kernel/reduction.py` and
        # `shaders/lib/reduction.slang` shows all three paths are wired
        # end-to-end. Each feature below cites the codegen entry point,
        # the lib helper it routes to, and the regression test that
        # locks the behavior:
        #
        #   * SCAN — `kernel/reduction.py:scan()` (line ~481) emits
        #     `wg_scan_inclusive<IScan{Add,Mul,Min,Max}>(...)` from
        #     `shaders/lib/reduction.slang:wg_scan_inclusive` (uses
        #     `WavePrefixSum`/`WavePrefixProduct`/`WavePrefixMin`/
        #     `WavePrefixMax` + groupshared cross-wave propagation).
        #     Locked by `TestBackendFeatureGates.test_cumsum_pointwise_fusion_correctness`
        #     and `TestTrackT2TemplateWiring.test_cumsum_dim_last_compiled`.
        #
        #   * SORT — `kernel/reduction.py:sort()` (line ~562) emits
        #     `wg_bitonic_sort_float2(...)` from
        #     `shaders/lib/reduction.slang:wg_bitonic_sort_float2`.
        #     Locked by `TestTrackT2TemplateWiring.test_topk_basic_compiled`.
        #
        #   * TUPLE_REDUCTION — `kernel/reduction.py:_welford_reduction`
        #     (line ~252) emits `wg_welford(WelfordResult<float>(...))`
        #     from `shaders/lib/reduction.slang:wg_welford` and returns
        #     a `(mean, m2, cnt)` triple of CSE vars (proper multi-output
        #     plumbing). Locked by
        #     `TestVarMeanWelford.test_var_mean_correctness`,
        #     `TestWelfordCooperativeSelection.test_layer_norm_compiles`,
        #     and `TestEmbeddingLayerNormBaseline`.
        #
        # If any of these regress, drop the corresponding flag here so
        # Inductor falls back to ATen extern (slower, correct).
        features = [
            BackendFeature.INPLACE_BUFFERS,
            BackendFeature.TUPLE_REDUCTION,
            BackendFeature.REDUCE_TO_SINGLE_ELEMENT,
            BackendFeature.FOREACH,
            BackendFeature.SCAN,
            BackendFeature.SORT,
        ]
        return OrderedSet(features)

    def can_fuse(self, node1, node2):
        """Vulkan-specific fusion heuristics.

        The default SIMDScheduling.can_fuse is generally good, but we add
        Vulkan-specific overrides:

        1. Reject fusions that would produce kernels with more unique buffers
           than the device supports (queried from maxPerStageDescriptorStorageBuffers,
           halved to leave room for outputs). Falls back to 16 on query failure.
           DR.1+: aggressive_fusion relaxes this for pointwise-only chains.

        2. Prefer NOT fusing pointwise + reduction when the pointwise output
           is only consumed by the reduction (already covered by the base
           class horizontal fusion). But DO fuse when the reduction has a
           small rnumel that fits in one wave — single-wave persistent
           reductions are faster than separate pointwise + reduce kernels
           because we skip the intermediate buffer write and barrier.

        3. Allow extern-kernel → pointwise vertical fusion (Phase A):
           when node1 produces an extern-kernel output and node2 is a
           pointwise node, allow them to be fused into a single pointwise
           kernel that reads the extern output as an input. This collapses
           N extern-tail pointwise dispatches into 1.

        4. TRAIN.7: autocast boundary nodes
           (aten._autocast_to_reduced_precision / _autocast_to_full_precision)
           lower to aten.to.dtype identity casts.  They must not prevent
           fusion — fp32→fp16 and fp16→fp32 boundaries are just dtype
           conversions that operate elementwise, exactly like any other
           pointwise op.  The base class can_fuse handles dtype transitions
           naturally; this entry documents that we rely on that behavior.

        5. DR.1+: Reduction + pointwise tail fusion — when a reduction
           output has a single consumer that is pointwise, allow fusing
           the reduction into the pointwise consumer kernel.
        """
        from torch._inductor import scheduler

        from . import config

        if isinstance(node1, scheduler.ForeachKernelSchedulerNode) or isinstance(
            node2, scheduler.ForeachKernelSchedulerNode
        ):
            return scheduler.ForeachKernelSchedulerNode.can_fuse(node1, node2)

        _, (numel1, rnumel1) = node1.group
        _, (numel2, rnumel2) = node2.group

        # Check if both nodes are pointwise (rnumel == 1 means no reduction)
        both_pointwise = rnumel1 == 1 and rnumel2 == 1

        combined_nodes = list(node1.get_nodes()) + list(node2.get_nodes())
        unique_bufs = set()
        for n in combined_nodes:
            unique_bufs |= n.get_buffer_names()

        max_bufs = self._get_max_storage_bufs()

        # DR.1+: Relax memory threshold for pointwise-only chains.
        # Pointwise ops use mostly registers, so the storage buffer
        # limit is less constraining.  Allow up to 1.5x the normal
        # limit for pointwise-only chains when aggressive fusion is on.
        if config.aggressive_fusion() and both_pointwise:
            pw_max_bufs = max(max_bufs, int(max_bufs * 1.5))
            if len(unique_bufs) > pw_max_bufs:
                return False
        elif len(unique_bufs) > max_bufs:
            return False

        base_result = super().can_fuse(node1, node2)
        if not base_result:
            return False

        # NOTE: ExternKernelSchedulerNode is rejected by Scheduler.can_fuse
        # (scheduler.py:5829) before our backend override is consulted, unless
        # the IR node is a UserDefinedTritonKernel. So this ExternKernelOut
        # check is effectively dead code for non-Triton extern ops.
        #
        # Phase A (epilogue fusion) is already working correctly: the base
        # SIMDScheduling.can_fuse fuses pointwise nodes that follow an
        # ExternKernelOut into a single VulkanKernel. The extern kernel
        # produces its output via eager dispatch, then the fused pointwise
        # epilogue runs as one JIT Slang kernel — collapsing N separate
        # pointwise dispatches into 1.
        #
        # Phase B (future): fuse matmul+epilogue into a single template
        # shader via VulkanTemplateCaller, eliminating the intermediate
        # buffer write entirely. Requires wiring up SlangTemplate (GAP 2).

        # Reduction + pointwise fusion: allow fusing when the
        # reduction's rnumel fits within a reasonable wave budget.
        # DR.1+: aggressive_fusion raises the threshold from 64 to 256.
        # The pointwise epilogue element-wise consumes the reduction
        # output and benefits from skipping the intermediate write.
        rnumel_fuse_cap = 256 if config.aggressive_fusion() else 64
        if node1.is_reduction() and not node2.is_reduction():
            if (
                rnumel1 != 1
                and isinstance(rnumel1, sympy.Integer)
                and int(rnumel1) <= rnumel_fuse_cap
            ):
                return True
        if not node1.is_reduction() and node2.is_reduction():
            if (
                rnumel2 != 1
                and isinstance(rnumel2, sympy.Integer)
                and int(rnumel2) <= rnumel_fuse_cap
            ):
                return True

        return True

    def _get_fusion_group_from_node(self, snode) -> Optional[str]:
        """Extract the ``vulkan_fusion_group`` annotation from a scheduler
        node's underlying IR nodes (checking all sub-nodes for fused nodes).

        Returns the fusion group name if any IR node has the annotation,
        or None otherwise.
        """
        for ir_node in snode.get_nodes():
            fx_node = getattr(ir_node, "fx_node", None)
            if fx_node is not None:
                if hasattr(fx_node, "meta"):
                    fg = fx_node.meta.get("vulkan_fusion_group")
                    if fg is not None:
                        return fg
        return None

    def can_fuse_vertical(self, node1, node2):
        """DR.1 — Fusion-aware vertical fusion with FX-pass-level pattern
        matching for op-classes.

        When both ``node1`` and ``node2`` carry the same
        ``vulkan_fusion_group`` annotation (set by
        ``fx_passes/patterns/op_class_fusion.py``), allow fusion EVEN
        when base heuristics would reject it.  When they carry DIFFERENT
        fusion groups, REJECT fusion to respect template-supported
        epilogue composition boundaries.

        Also retains Phase B (T4.1) template→pointwise epilogue fusion
        via ``vulkan_epilogue`` metadata.

        DR.1+: Multi-consumer fusion — when a buffer has multiple consumers
        but all consumers can be fused into the same kernel (all pointwise
        or small-reduction chains), the scheduler can skip materializing
        that intermediate buffer. This is gated by
        ``TORCH_VULKAN_AGGRESSIVE_FUSION=1`` (default ON).

        Falls back to base SIMDScheduling for unannotated nodes.
        """
        from torch._inductor import scheduler

        from . import config

        base = super().can_fuse_vertical(node1, node2)

        # --- DR.1: Fusion-group aware scheduling ---
        fg1 = self._get_fusion_group_from_node(node1)
        fg2 = self._get_fusion_group_from_node(node2)

        if fg1 is not None and fg2 is not None:
            if fg1 == fg2:
                # M6.7: conv_epilogue fusing template (conv) + reduction (norm)
                # produces invalid Slang in the combo kernel body rewriter.
                # The combo kernel only handles pointwise subkernels — template
                # and reduction kernels must use native template epilogues, not
                # the generic combo kernel path.  Reject this fusion so the
                # scheduler keeps them as separate dispatches.
                from .fx_passes.patterns.op_class_fusion import FusionGroup

                if fg1 == FusionGroup.conv_epilogue:
                    mix_template = node1.is_template() or node2.is_template()
                    mix_reduction = node1.is_reduction() or node2.is_reduction()
                    if mix_template and mix_reduction:
                        return False
                # Same fusion group — allow fusion even if base
                # heuristics said no (overrides memory-footprint
                # rejection for template-compatible epilogues).
                return True
            elif fg1 != fg2:
                # Different fusion groups cross an op-class boundary
                # without template support — prevent fusion.
                return False

        # DR.1+: Multi-consumer fusion — when base says no because node1's
        # output has multiple consumers, check if all those consumers are
        # pointwise nodes that could be fused into the same kernel. If so,
        # allow the fusion — the scheduler can skip materialization.
        if config.aggressive_fusion() and not base:
            if self._all_consumers_are_fusible_pointwise(node1, node2):
                return True

        # When only one node has a fusion group (the other is None),
        # allow the fusion to proceed if base heuristics agree.
        # The annotated node's epilogue can consume the other node's
        # output or be consumed by it.

        # --- Phase B (T4.1): template→pointwise via vulkan_epilogue ---
        # Check if node1's IR node has an fx_node with epilogue metadata.
        # Chain: SchedulerNode -> .node (ir.Operation) -> .fx_node (FX Node)
        ir_node = getattr(node1, "node", None)
        if ir_node is not None:
            fx_node = getattr(ir_node, "fx_node", None)
            if fx_node is not None:
                epilogue = (
                    fx_node.meta.get("vulkan_epilogue")
                    if hasattr(fx_node, "meta")
                    else None
                )
                if epilogue:
                    # Propagate the epilogue struct name to node2's IR nodes
                    # so the codegen can emit the IPointwise interface dispatch.
                    node2_nodes = list(node2.get_nodes())
                    for n in node2_nodes:
                        if not hasattr(n, "meta"):
                            n.meta = {}
                        n.meta["vulkan_epilogue"] = epilogue
                        n.meta["vulkan_is_epilogue"] = True
                    return True

        # Reverse direction: check if node2's IR node has epilogue metadata
        # (covers the case where the FX pass fused the epilogue onto node2).
        ir_node2 = getattr(node2, "node", None)
        if ir_node2 is not None:
            fx_node2 = getattr(ir_node2, "fx_node", None)
            if fx_node2 is not None:
                epilogue2 = (
                    fx_node2.meta.get("vulkan_epilogue")
                    if hasattr(fx_node2, "meta")
                    else None
                )
                if epilogue2:
                    node1_nodes = list(node1.get_nodes())
                    for n in node1_nodes:
                        if not hasattr(n, "meta"):
                            n.meta = {}
                        n.meta["vulkan_epilogue"] = epilogue2
                    return True

        # M9.8: Reduction-boundary fusion relaxation.
        # When base heuristics reject reduction↔pointwise fusion (e.g.
        # due to tiling incompatibility), override if the reduction's
        # rnumel fits within the wave budget.  This allows patterns like
        # GN + ReLU + GlobalAvg to fuse into a single kernel.
        if not base:
            _, (numel1, rnumel1) = node1.group
            _, (numel2, rnumel2) = node2.group
            rnumel_fuse_cap = 256 if config.aggressive_fusion() else 64
            if node1.is_reduction() and not node2.is_reduction():
                if (
                    rnumel1 != 1
                    and isinstance(rnumel1, sympy.Integer)
                    and int(rnumel1) <= rnumel_fuse_cap
                ):
                    return True
            if not node1.is_reduction() and node2.is_reduction():
                if (
                    rnumel2 != 1
                    and isinstance(rnumel2, sympy.Integer)
                    and int(rnumel2) <= rnumel_fuse_cap
                ):
                    return True

        # When no epilogue metadata was found on either node, fall back
        # to the base class result.
        return base

    def _all_consumers_are_fusible_pointwise(self, node1, node2) -> bool:
        """Check if all consumers of node1's output buffers are pointwise
        nodes that can be fused with node2 into the same kernel.

        This enables multi-consumer fusion: when a buffer has multiple
        consumers that are all pointwise, the scheduler can skip
        materialization and fuse them into the consumer kernel.

        DR.1+: gated by aggressive_fusion().
        """
        # Get node1's output buffer names
        node1_bufs = set()
        for n in node1.get_nodes():
            if hasattr(n, "get_buffer_names"):
                node1_bufs |= n.get_buffer_names()
            elif hasattr(n, "get_name"):
                node1_bufs.add(n.get_name())
        if not node1_bufs:
            return False

        # Find all scheduler nodes that consume node1's buffers
        try:
            sched = (
                V.graph.scheduler
                if hasattr(V, "graph") and V.graph is not None
                else None
            )
            if sched is None:
                return False

            all_consumers = []
            for sn in sched.nodes:
                if sn is node2:
                    continue  # node2 is already in the fusion
                # Check if this node reads any of node1's output buffers
                sn_reads = set()
                for n in sn.get_nodes():
                    sn_reads |= n.get_read_names()
                if node1_bufs & sn_reads:
                    all_consumers.append(sn)

            if not all_consumers:
                # No other consumers — let base heuristics handle single-consumer
                return False

            # Check: are all consumers pointwise?
            for consumer in all_consumers:
                if consumer.is_reduction():
                    return False
                # Also check numel compatibility
                _, (c_numel, c_rnumel) = consumer.group
                _, (n2_numel, n2_rnumel) = node2.group
                if c_numel != n2_numel:
                    return False

            return True
        except Exception:
            return False

    def codegen_mix_order_reduction(self, node):
        # T5.13 (2026-05-09): autotune is now safe for mix-order reduction.
        # Previously this method force-disabled
        # ``mix_order_reduction_autotune_split_size``, ``max_autotune``, and
        # ``coordinate_descent_tuning`` as a Track-0 era correctness escape —
        # autotune was producing wrong results for layernorm/rmsnorm fused
        # mix-order reductions. After T5.14 (round 7) and TR.16.A (wave A)
        # fixed the underlying reduction-codegen bugs, autotune is once again
        # safe and we delegate directly to the base implementation.
        super().codegen_mix_order_reduction(node)

    def benchmark_codegened_module(
        self, mod, n_spills_threshold=8, node_names: OrderedSet[str] | None = None
    ) -> tuple[float, str]:
        # Benchmarker registration is module-load-time (see bottom of file)
        # so we don't pay the @register_benchmarker decorator cost on every
        # codegen — that path was being re-entered per autotune candidate.
        # P2.2: also cache the Benchmarker instance on the class. Inductor
        # autotune calls this method per candidate config; constructing
        # `Benchmarker()` on every call repeated its (small) init work.
        try:
            from torch._dynamo.device_interface import get_interface_for_device

            iface = get_interface_for_device("vulkan")
            with iface.device(None):
                args = mod.get_args()
                call = mod.call
                call(*args)
                ms = _get_benchmarker().benchmark(lambda: call(*args), device="vulkan")
            return ms, ""
        except Exception:
            return float("inf"), ""

    def codegen_combo_kernel(self, combo_kernel_node):
        from torch._inductor.codegen.simd import SIMDKernelFeatures

        from .vulkan_combo_kernel import VulkanComboKernel

        subkernel_nodes = combo_kernel_node.get_subkernel_nodes()
        combo = VulkanComboKernel()

        first_kernel: object | None = None
        for sn in subkernel_nodes:
            nodes = sn.get_nodes()
            _, (numel, rnumel) = max(nodes, key=lambda x: int(x.is_reduction())).group
            node_schedule = self.generate_node_schedule(nodes, numel, rnumel)
            features = SIMDKernelFeatures(node_schedule, numel, rnumel)
            tiling = self.select_tiling(node_schedule, numel, rnumel)
            kernel = self.kernel_type(tiling, features=features)

            # Share only the CSE counter (not cache) across subkernels.
            # Each subkernel keeps its own CSE cache (independent variable
            # declarations), but the shared counter prevents tmpN name
            # collisions.  This avoids cross-subkernel variable references
            # that would require complex scope hoisting.  The cross_decls
            # mechanism in _rewrite_body serves as a safety net for any
            # edge-case cross-references.
            if first_kernel is not None:
                combo.share_cse_from(first_kernel, kernel)
            else:
                first_kernel = kernel

            with V.set_kernel_handler(kernel):
                self.process_kernel(kernel, node_schedule)
            combo.create_sub_kernel(kernel, numel)

        src_code = combo.codegen_kernel()
        kernel_name = self._define_combo_kernel(src_code, combo_kernel_node, combo)
        self.codegen_comment(combo_kernel_node.snodes, kernel_name)
        combo.call_kernel(kernel_name)
        self.free_buffers_in_scheduler()

    def _define_combo_kernel(self, src_code: str, combo_kernel_node, combo) -> str:
        """Like `define_kernel` but uses the combo's merged (n_buffers, n_outputs)
        counts instead of a single subkernel's args. Combo kernels currently have
        no push constants (no symbolic shapes through the dispatch wrapper)."""
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]
        kernel_name = f"vulkan_kernel_{wrapper.next_kernel_suffix()}"
        wrapper.src_to_kernel[src_code] = kernel_name

        src_var = f"{kernel_name}_slang"
        src_hash = hashlib.sha1(src_code.encode()).hexdigest()[:12]
        n_buffers = combo.n_buffers
        n_outputs = combo.n_outputs
        n_pc = 0
        pc_size_bytes = 0
        # M-pipeline-7: derive a content-aware combo config_key from the
        # sub-kernels' own (post-M-pipeline-3) config_keys. See
        # `compute_combo_config_key` above for the full rationale.
        sub_keys = tuple(sk.config_key for sk, _numel in combo.subkernels)
        combo_config_key = compute_combo_config_key(sub_keys)

        wrapper.header.splice(
            f"{src_var} = '''{src_code}'''\n"
            f"{src_var}_key = '{kernel_name}_{src_hash}'\n"
            f"{kernel_name} = _vk_make_kernel({src_var}, {src_var}_key, {n_buffers}, {pc_size_bytes}, {n_pc}, {n_outputs}, config_key='{combo_config_key}')\n"
        )
        return kernel_name

    def define_kernel(
        self,
        src_code: str,
        node_schedule: "list[SchedulerNode]",
        kernel: VulkanKernel,
    ) -> str:
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]
        kernel_name = f"vulkan_kernel_{wrapper.next_kernel_suffix()}"
        wrapper.src_to_kernel[src_code] = kernel_name

        origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
        src_var = f"{kernel_name}_slang"
        src_hash = hashlib.sha1(src_code.encode()).hexdigest()[:12]
        from torch._inductor.codegen.common import InplacedBuffer

        # N+1.7: When the kernel is fully-static (all dims and numels
        # are known integers) AND the feature gate is enabled, sizevars
        # are emitted as ``static const uint`` module-scope declarations
        # instead of push-constant struct members.  Therefore n_pc is 0
        # (no push constants).  The gate is TORCH_VULKAN_STATIC_SPECIALIZATION=1
        # (default ON).
        from . import config

        if config.static_specialization() and kernel._is_fully_static():
            n_pc = 0
        else:
            n_pc = sum(1 for _ in kernel.args.sizevars.values()) + sum(
                1
                for v in kernel.active_range_trees()
                if not isinstance(v.numel, sympy.Integer)
                and (not v.is_reduction or kernel.inside_reduction)
            )
        n_inplace = len(
            {
                inplaced.inner_name
                for inplaced in kernel.args.inplace_buffers.values()
                if isinstance(inplaced, InplacedBuffer)
            }
        )
        n_outputs = n_inplace + sum(
            1
            for k, v in kernel.args.output_buffers.items()
            if k not in kernel.removed_buffers and k not in kernel.args.inplace_buffers
        )
        n_inputs = sum(
            1 for k in kernel.args.input_buffers if k not in kernel.args.inplace_buffers
        )
        n_ws = len(kernel.args.workspace_args)
        n_buffers = n_outputs + n_inputs + n_ws
        pc_size_bytes = n_pc * 4

        wrapper.header.splice(
            f"{src_var} = '''{src_code}'''\n"
            f"{src_var}_key = '{kernel_name}_{src_hash}'\n"
            f"{kernel_name} = _vk_make_kernel({src_var}, {src_var}_key, {n_buffers}, {pc_size_bytes}, {n_pc}, {n_outputs}, config_key='{kernel.config_key}')\n"
        )
        return kernel_name

    @classmethod
    def get_tiling_and_scores(
        cls, node_schedule, numel, reduction_numel=sympy.S.One, coalesce_analysis=None
    ):
        return super().get_tiling_and_scores(
            node_schedule, numel, reduction_numel, coalesce_analysis
        )


_BENCHMARKER = None


def _get_benchmarker():
    """Cached `Benchmarker` instance. P2.2 — `benchmark_codegened_module` is
    called per autotune candidate; instantiating fresh per call repeated the
    benchmarker's init work for no benefit. The benchmarker is stateless
    across calls so a single instance is fine.
    """
    global _BENCHMARKER
    if _BENCHMARKER is None:
        from torch._inductor.runtime.benchmarking import Benchmarker

        _BENCHMARKER = Benchmarker()
    return _BENCHMARKER


def _reset_benchmarker_cache() -> None:
    """Test hook — clears the cached benchmarker."""
    global _BENCHMARKER
    _BENCHMARKER = None


def _register_vulkan_benchmarker_once() -> None:
    """Register Vulkan's wall-clock benchmarker exactly once at module load.

    Inductor's `Benchmarker` looks up a per-device entry in the registry on
    every benchmark call. Re-running `@register_benchmarker(..., override=True)`
    inside `benchmark_codegened_module` (the previous shape) replaced the
    entry on every autotune iteration, which is wasted work.
    """
    try:
        from torch._inductor.runtime.benchmarking import register_benchmarker

        @register_benchmarker("vulkan", override=True)
        def _vulkan_bench(self, f, *, warmup, rep, **kw):
            f()
            import time

            timings = []
            t0 = time.perf_counter()
            while True:
                start = time.perf_counter()
                f()
                end = time.perf_counter()
                timings.append((end - start) * 1000)
                if (end - t0) * 1000 > rep:
                    break
            from statistics import median

            return median(timings)
    except Exception:
        pass


_register_vulkan_benchmarker_once()
