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
from .scheduling_helpers import (
    _fusion_has_new_half_reads,
    _get_benchmarker,
    _reset_benchmarker_cache,
    _wave64_persistent_ok,
    compute_combo_config_key,
)

if TYPE_CHECKING:
    from torch._inductor.scheduler import Scheduler, SchedulerNode


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

            # Pre-descriptor-indexing path: reserve 4 slots; clamp to [12, 80].
            # F.1 (2026-05-19): raised cap 60 → 80. Original 60 targeted older
            # mobile-class binding ceilings (~64/stage on Intel UHD / Mali /
            # Adreno). Modern desktop drivers (RADV / NVIDIA / Intel ARC)
            # advertise >=64; RDNA1 specifically advertises 64. 80 stays
            # below the 96/128 tier older mobile drivers enforce while
            # admitting larger fused combos. On RDNA1 with descriptor
            # indexing enabled (default), this branch is not reached.
            usable = raw - 4
            cls._cached_max_storage_bufs = max(12, min(usable, 80))
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

        # TRAIN.12 (2026-05-28): Don't fuse two reductions with different
        # (numel, rnumel) groups.  Upstream `analyze_reads_and_writes`
        # (tiling_utils.py:552) calls `get_pw_red_splits` WITHOUT
        # ``none_if_not_divisible=True``, so if the fused node contains a
        # sub-node whose body sizes don't decompose into the group's
        # pointwise × reduction split, an assertion fires:
        #   assert prod(sizes[0]) == pw_numel * red_numel
        # This happens when 2+ norm layers (GroupNorm, BatchNorm) produce
        # backward reduction nodes with different reduction dimensions that
        # the scheduler considers compatible.  Rejecting the fusion at the
        # backend level is correct: on Vulkan, two reductions with different
        # rnumel can't share a workgroup reduction strategy anyway.
        if not both_pointwise and rnumel1 != 1 and rnumel2 != 1:
            if (numel1, rnumel1) != (numel2, rnumel2):
                return False

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
        # M-PERF.6 (2026-05-19): on RDNA1/wave64 with persistent_pointwise
        # the wave-cooperative reduction in `shaders/lib/vk_reduction`
        # handles rnumel up to 1024 (16 waves of 64). Unlocks layer-norm
        # fusion in stable diffusion / Llama training (~2-3 us/norm).
        rnumel_fuse_cap = 64
        if config.aggressive_fusion():
            rnumel_fuse_cap = 256
            if config.persistent_pointwise() and _wave64_persistent_ok():
                rnumel_fuse_cap = 1024
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

        # GPU.1: AMD RDNA1 L2 cache race — reject fp16/bf16 extra-read fusion
        # after a wg_welford reduction.  See scheduling_helpers._fusion_has_new_half_reads.
        if base and node1.is_reduction():
            if _fusion_has_new_half_reads(node1, node2):
                return False

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
        # nodes that could be fused into the same kernel. If so, allow the
        # fusion — the scheduler can skip materialization.
        #
        # M19.3 (2026-05-21): the consumer-pattern check was previously
        # pointwise-only, which left the GN + ReLU + GlobalAvg chain at
        # 3 dispatches (welford+normalize+ReLU into 1, GAP reduction as a
        # second, plus an output-store boundary as a third).  Allowing
        # consumers whose ``rnumel`` fits the wave-budget cap closes the
        # last gap — the GAP-style reduction folds into the upstream
        # pointwise kernel, taking ``GN+ReLU+GAP`` from 3 down to 2.
        if config.aggressive_fusion() and not base:
            if self._all_consumers_are_fusible(node1, node2):
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
        # M-PERF.6 (2026-05-19): same cap-raise logic as can_fuse_horizontal —
        # wave64 + persistent_pointwise lifts the ceiling to 1024.
        if not base:
            _, (numel1, rnumel1) = node1.group
            _, (numel2, rnumel2) = node2.group
            rnumel_fuse_cap = 64
            if config.aggressive_fusion():
                rnumel_fuse_cap = 256
                if config.persistent_pointwise() and _wave64_persistent_ok():
                    rnumel_fuse_cap = 1024
            if node1.is_reduction() and not node2.is_reduction():
                if (
                    rnumel1 != 1
                    and isinstance(rnumel1, (int, sympy.Integer))
                    and int(rnumel1) <= rnumel_fuse_cap
                ):
                    return True
            if not node1.is_reduction() and node2.is_reduction():
                if (
                    rnumel2 != 1
                    and isinstance(rnumel2, (int, sympy.Integer))
                    and int(rnumel2) <= rnumel_fuse_cap
                ):
                    return True

        # When no epilogue metadata was found on either node, fall back
        # to the base class result.
        return base

    def _all_consumers_are_fusible(self, node1, node2) -> bool:
        """Check if all consumers of ``node1``'s output buffers are fusible
        with ``node2`` into the same kernel.

        This enables multi-consumer fusion: when a buffer has multiple
        consumers and **all** of them either (a) are pointwise with the
        same numel as ``node2`` or (b) are reductions whose ``rnumel``
        fits the wave-budget cap (the same cap policy as the M9.8 /
        M-PERF.6 reduction-boundary relaxation), the scheduler can skip
        materialising the intermediate buffer and fuse them into a single
        kernel.

        DR.1+: gated by ``aggressive_fusion()``.

        M19.3 (2026-05-21): the historical name
        ``_all_consumers_are_fusible_pointwise`` is preserved as a
        backwards-compat alias below — the test suite (and any in-flight
        agent diff) refers to that name.  Internally we use the broader
        name because reductions are now admissible.
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

        # Same wave-budget cap policy as M9.8 + M-PERF.6: 64 default,
        # 256 with aggressive fusion, 1024 with wave64 + persistent
        # pointwise.  Reduction consumers fold into the upstream kernel
        # iff their ``rnumel`` fits this cap.
        from . import config as _config

        rnumel_fuse_cap = 64
        if _config.aggressive_fusion():
            rnumel_fuse_cap = 256
            if _config.persistent_pointwise() and _wave64_persistent_ok():
                rnumel_fuse_cap = 1024

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

            _, (n2_numel, _n2_rnumel) = node2.group
            for consumer in all_consumers:
                _, (c_numel, c_rnumel) = consumer.group
                if consumer.is_reduction():
                    # M19.3: admit small-rnumel reduction consumers.
                    # Bail on symbolic or oversized rnumels — we can't
                    # prove the merged kernel fits in the wave budget
                    # without an integer bound.
                    if not isinstance(c_rnumel, sympy.Integer):
                        return False
                    if int(c_rnumel) > rnumel_fuse_cap:
                        return False
                    # Also bail if numel is symbolic — the broadcast
                    # check below relies on integer comparison.
                    if not isinstance(c_numel, sympy.Integer):
                        return False
                    # The reduction's input iteration space is
                    # ``c_numel * c_rnumel``; that must match node2's
                    # output numel so the merged kernel covers exactly
                    # the upstream pointwise's output.
                    if isinstance(n2_numel, sympy.Integer):
                        if int(c_numel) * int(c_rnumel) != int(n2_numel):
                            return False
                    # Symbolic n2_numel: we already gave up further up
                    # for symbolic c_numel, so fall through and accept
                    # — the base scheduler will catch any leftover
                    # incompatibility downstream.
                else:
                    # Pointwise consumer: original numel-equality contract.
                    if c_numel != n2_numel:
                        return False

            return True
        except Exception:
            return False

    # Backwards-compat alias.  The historical (pre-M19.3) name is
    # referenced by parallel agent diffs and existing regression-test
    # mocks; keeping the alias avoids breaking those at the import-time.
    _all_consumers_are_fusible_pointwise = _all_consumers_are_fusible

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
        # AOTI-FIX: store kernel source for AOTI C++ codegen to compile to SPIR-V
        _set_kernel_source(wrapper, kernel_name, src_code)
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
        # AOTI-FIX: store kernel source for AOTI C++ codegen to compile to SPIR-V
        _set_kernel_source(wrapper, kernel_name, src_code)
        return kernel_name

    @classmethod
    def get_tiling_and_scores(
        cls, node_schedule, numel, reduction_numel=sympy.S.One, coalesce_analysis=None
    ):
        return super().get_tiling_and_scores(
            node_schedule, numel, reduction_numel, coalesce_analysis
        )

    def create_kernel_choices(
        self, kernel_features, kernel_args, kernel_kwargs
    ):
        """M19.2 — wire ``VulkanKernel._enable_persistent_mode()``.

        The grid-stride-loop wrapper in ``kernel/pointwise.py`` was dead
        code (defined, gated by ``config.persistent_pointwise()``, but
        never invoked anywhere in the backend).  Activate it here, right
        after the kernel is instantiated by the base SIMDScheduling, so
        the body emitter in ``kernel/header.py`` (which checks
        ``self._persistent_mode`` after the loads/compute/stores splice)
        actually sees the flag set when ``codegen_kernel()`` runs.

        Activation conditions:
          * ``config.persistent_pointwise()`` is the user gate (default
            on via ``TORCH_VULKAN_PERSISTENT_POINTWISE=1``).
          * Pointwise only — ``reduction_numel == 1``.  The persistent
            loop substitutes ``gtid.x`` and is incompatible with the
            two-tree reduction layout.
          * Small kernels — ``numel <= 4096``.  Below this point the
            per-dispatch overhead (~30-60 us on RDNA1) dominates the
            kernel runtime and the grid-stride loop amortises launches
            across the chain.  Above 4096 elements the dispatch is
            already large enough that the existing single-launch
            path is fine; the persistent path would just add a
            modulo-stride load per element.
        """
        from . import config

        kernels = super().create_kernel_choices(
            kernel_features, kernel_args, kernel_kwargs
        )

        if not kernels:
            import warnings

            warnings.warn(
                "[M19.8] VulkanScheduling.create_kernel_choices returned no candidates "
                "for this node; Inductor will likely fall back to an extern kernel. "
                "Check TORCH_VULKAN_NO_WG_TUNE / occupancy settings.",
                RuntimeWarning,
                stacklevel=2,
            )

        if not config.persistent_pointwise():
            return kernels
        if kernel_features.is_reduction():
            return kernels

        numel = kernel_features.numel
        if not isinstance(numel, sympy.Integer):
            return kernels
        if int(numel) > 4096:
            return kernels

        for kernel in kernels:
            enable = getattr(kernel, "_enable_persistent_mode", None)
            if callable(enable):
                enable()

        return kernels


# ── AOTI kernel source registry ────────────────────────────────────────

# Mapped on the wrapper object at _kernel_name_to_src so the AOTI C++
# wrapper can compile Slang→SPIR-V during codegen (not at runtime).
_KERNEL_NAME_TO_SRC_ATTR = "_kernel_name_to_src"


def _set_kernel_source(wrapper, kernel_name: str, src_code: str) -> None:
    """Store kernel Slang source keyed by kernel_name on the wrapper."""
    if not hasattr(wrapper, _KERNEL_NAME_TO_SRC_ATTR):
        setattr(wrapper, _KERNEL_NAME_TO_SRC_ATTR, {})
    getattr(wrapper, _KERNEL_NAME_TO_SRC_ATTR)[kernel_name] = src_code


def get_kernel_source(wrapper, kernel_name: str) -> str | None:
    """Return the Slang source for a kernel name, or None if not found."""
    d = getattr(wrapper, _KERNEL_NAME_TO_SRC_ATTR, None)
    if d is None:
        return None
    return d.get(kernel_name)


