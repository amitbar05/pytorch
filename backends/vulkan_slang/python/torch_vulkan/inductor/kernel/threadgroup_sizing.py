"""Threadgroup-sizing heuristics for :class:`VulkanKernel`.

Extracted from ``kernel/main.py`` as ``ThreadgroupSizingMixin`` to keep
``main.py`` under the 800-line anti-goal #7 cap (see
``backends/vulkan_slang/CLAUDE.md``).

Owns the register/shared-memory/loop-depth aware heuristics that pick
``[numthreads(...)]`` for pointwise and reduction kernels, plus the
M11.5 wave-rounding helper, the M11.7 occupancy gate, and the
N+1.12 / DR.3 max-unroll-factor helper.

All ``self.xxx`` attribute accesses resolve via MRO against the host
class (``VulkanKernel``); this mixin contributes no state of its own.
"""

from __future__ import annotations

from typing import Optional

import sympy

from .symbolic import is_dynamic


class ThreadgroupSizingMixin:
    """Mixin providing workgroup-size heuristics for :class:`VulkanKernel`."""

    def _pick_threadgroup_size(self) -> int:
        """Pick workgroup size with register-pressure awareness (P5.1 / M4 / DR.3).

        DR.3: Routes to specialized helpers based on op-class so pointwise,
        reduction, and matmul kernels each get a heuristic tuned to their
        register / shared-memory / loop-depth profile.

        Pointwise: optimize for occupancy (more threads, simpler bodies).
        Reduction: fewer threads when register pressure is high (welford,
        multi-axis) and when loop_depth is deep.
        Matmul: tile-size-aware sizing (handled by template configs).
        """
        if self.inside_reduction:
            return self._pick_threadgroup_size_reduction()
        return self._pick_threadgroup_size_pointwise()

    @staticmethod
    def _round_wg_to_wave(wg_size: int, max_wg: int, sgs: int) -> int:
        """M11.5: Round WG size up to next wave-size multiple.

        RDNA1 (wave64) hardware pads partial waves; a WG of 100 threads
        spans 2 waves (128 lanes) wasting 28 lanes.  Rounding up to the
        next multiple guarantees full-wave occupancy.  Never exceeds max_wg.
        Call sites should only invoke when :func:`~config.round_wg_to_wave`
        returns True and ``wg_size % sgs != 0``.
        """
        rounded = ((wg_size + sgs - 1) // sgs) * sgs
        return min(rounded, max_wg)

    def _check_occupancy_gate(self) -> None:
        """M11.7: Warn/fail if estimated occupancy falls below 50 %.

        Called after ``_pick_threadgroup_size`` when WG size is final.
        Uses :func:`~gpu_utilization.estimate_occupancy` with the best
        available VGPR/shared-memory data (reflection if available,
        fallback heuristic otherwise).

        Behaviour is controlled by ``TORCH_VULKAN_STRICT_OCCUPANCY``:
        - unset or ``0``: log a warning via ``trace_structured``.
        - ``1``: raise ``RuntimeError``, failing compilation.
        """
        from .. import config
        from ..gpu_utilization import estimate_occupancy

        if not config.occupancy_gate():
            return

        wg_size = self.max_threadgroup_size
        sgs = self.simd_group_size or 64

        # Best-effort VGPR estimate
        vgprs = self._get_actual_vgprs()
        if vgprs is None:
            vgprs = self._estimate_vgprs()
        if vgprs is None:
            vgprs = 32  # conservative default for float workloads
        vgprs_per_thread = max(1, vgprs // wg_size if wg_size > 0 else vgprs // sgs)

        # Best-effort shared memory estimate
        shared_mem_bytes = getattr(self, "_cached_shared_mem", None) or 0

        est = estimate_occupancy(
            threadgroup_size=wg_size,
            vgprs_per_thread=vgprs_per_thread,
            shared_mem_bytes=shared_mem_bytes,
            simd_size=sgs,
        )

        occupancy_pct = est["occupancy_pct"]
        if occupancy_pct < 50.0:
            msg = (
                f"[M11.7] Low estimated occupancy: {occupancy_pct:.0f}% "
                f"(WG={wg_size}, VGPR={vgprs}, LDS={shared_mem_bytes}B, "
                f"limit={est['limiting_factor']})"
            )
            if config.strict_occupancy():
                raise RuntimeError(msg)
            else:
                from torch._logging import trace_structured

                trace_structured(
                    "artifact",
                    metadata_fn=lambda: {
                        "name": "occupancy_gate",
                        "encoding": "string",
                    },
                    payload_fn=lambda: msg,
                )

    # ── DR.3: Shared helpers for threadgroup-size picking ───────────

    def _classify_vgpr_pressure(self, dtype_bytes: int) -> tuple[str, Optional[int]]:
        """Classify VGPR pressure as 'light' / 'normal' / 'heavy'.

        DR.3: Also retrieves cached shared_mem and loop_depth from
        reflection when available so the caller can factor them into
        the workgroup decision.

        Returns (vgpr_class, estimated_vgprs).
        """
        from .. import config

        vgpr_class = "normal"
        estimated_vgprs = None
        if config.register_aware_wg():
            # P3.3: Try actual VGPR count from SPIR-V reflection first
            actual = self._get_actual_vgprs()
            if actual is not None:
                estimated_vgprs = actual
            else:
                estimated_vgprs = self._estimate_vgprs()

            if estimated_vgprs <= 16:
                vgpr_class = "light"
            elif estimated_vgprs <= 32:
                vgpr_class = "normal"
            else:
                vgpr_class = "heavy"
        else:
            # Legacy: key on dtype and welford only
            if dtype_bytes >= 8:
                vgpr_class = "heavy"  # f64 uses ~2× registers
            elif dtype_bytes <= 2:
                vgpr_class = "light"  # f16/bf16 uses fewer registers
            if self.has_welford:
                vgpr_class = "heavy"  # welford uses ~3× registers (mean,m2,count)
            if getattr(self, "_partitioned_2d_active", False):
                vgpr_class = "heavy"  # 2D partitioned loops use more registers
        return vgpr_class, estimated_vgprs

    def _apply_vgpr_cap(self, max_wg: int, estimated_vgprs, sgs: int) -> int:
        """Apply VGPR-aware workgroup cap (RDNA1 occupancy model).

        RDNA1: 64 VGPRs/SIMD, 256 VGPRs/CU, max 1024 threads/CU.
        To maintain >=2 waves/CU occupancy:
          max_wg = min(1024, 256 * 2 / vgprs * subgroup_size)
        where subgroup_size = 64 (wave64) for Vulkan on RDNA1.
        """
        if estimated_vgprs is not None and estimated_vgprs > 0:
            _cu_vgprs = 256  # RDNA1 VGPRs per CU
            _min_waves = 2  # target >=2 waves/CU
            _subgroup_size = sgs
            if _subgroup_size <= 0:
                _subgroup_size = 64
            _vgpr_max_wg = (_cu_vgprs * _min_waves // estimated_vgprs) * _subgroup_size
            _vgpr_max_wg = min(_vgpr_max_wg, 1024)
            _vgpr_max_wg = max(_vgpr_max_wg, _subgroup_size)
            max_wg = min(max_wg, _vgpr_max_wg)
        return max_wg

    def _get_cached_shared_mem(self) -> Optional[int]:
        """DR.3: Retrieve shared_mem from cached reflection metrics."""
        from .. import config as _cfg

        if not _cfg.reflection_enabled():
            return None
        config_key = self._compute_config_key()
        from torch_vulkan.inductor.runtime import get_cached_metrics_for_key

        metrics = get_cached_metrics_for_key(config_key)
        if metrics is None:
            return None
        sm = metrics.get("shared_mem")
        if sm is None:
            return None
        try:
            return int(sm)
        except (TypeError, ValueError):
            return None

    def _get_cached_loop_depth(self) -> Optional[int]:
        """DR.3: Retrieve loop_depth from cached reflection metrics."""
        from .. import config as _cfg

        if not _cfg.reflection_enabled():
            return None
        config_key = self._compute_config_key()
        from torch_vulkan.inductor.runtime import get_cached_metrics_for_key

        metrics = get_cached_metrics_for_key(config_key)
        if metrics is None:
            return None
        ld = metrics.get("loop_depth")
        if ld is None:
            return None
        try:
            return int(ld)
        except (TypeError, ValueError):
            return None

    def _estimate_loop_depth(self) -> int:
        """DR.3: Structural loop-depth estimate from kernel config.

        Approximates maximum nested loop depth in the emitted Slang code
        from the number of reduction axes and persistence strategy.
        Used as a fallback when cached reflection data is unavailable.
        """
        depth = 0
        for rd in self.range_trees:
            if rd.is_reduction:
                depth += 1
        # Persistent reduction adds an outer staging loop.
        if self.inside_reduction and not self.should_use_cooperative_reduction():
            depth += 1
        # 2D partitioned adds Y-axis loop.
        if getattr(self, "_partitioned_2d_active", False):
            depth += 1
        return max(depth, 1)

    def _apply_loop_depth_penalty(
        self, max_wg: int, vgpr_class: str, loop_depth: int
    ) -> int:
        """DR.3: Reduce workgroup size when loop depth is high.

        Deeply nested loops increase register pressure beyond what
        the VGPR estimate captures.  For every loop level beyond 2,
        drop one occupancy tier.
        """
        if loop_depth <= 2:
            return max_wg
        # loop_depth 3-4: one tier down; 5+: two tiers down
        tiers = {"light": "light", "normal": "light", "heavy": "normal"}
        if loop_depth >= 5:
            tiers = {"light": "normal", "normal": "heavy", "heavy": "heavy"}
        adjusted = tiers.get(vgpr_class, vgpr_class)
        tier_caps = {"light": 256, "normal": 192, "heavy": 128}
        return min(max_wg, tier_caps.get(adjusted, 256))

    def _apply_shared_mem_cap(self, max_wg: int, dtype_bytes: int, sgs: int) -> int:
        """DR.3: Clamp workgroup size when shared-memory usage is high.

        Uses cached reflection shared_mem when available; otherwise falls
        back to the structural estimate from _groupshared_bytes_used.
        """
        lds_budget = getattr(self, "_groupshared_budget_bytes", 64 * 1024)
        lds_used = getattr(self, "_groupshared_bytes_used", 0)

        # DR.3: prefer cached reflection shared_mem (actual SPIR-V)
        cached_sm = self._get_cached_shared_mem()
        if cached_sm is not None and cached_sm > lds_used:
            lds_used = cached_sm

        if lds_used <= 0:
            return max_wg

        # Cooperative reductions allocate groupshared proportional to
        # workgroup size.  Conservative: assume 8 bytes/thread for f32.
        if self.inside_reduction:
            lds_per_thread = 8 if dtype_bytes >= 4 else 4
            lds_max_threads = max((lds_budget - lds_used) // lds_per_thread, sgs)
            max_wg = min(max_wg, lds_max_threads)
        else:
            # Pointwise: if groupshared is used (e.g. vec4 staging),
            # ensure total allocation fits within budget.
            if lds_used > lds_budget // 2:
                max_wg = min(max_wg, sgs * 2)
        return max_wg

    def _max_unroll_factor(self) -> int:
        """N+1.12 / DR.3: Maximum safe [unroll(N)] factor for this kernel.

        High VGPR pressure or deep loop nesting limits how aggressively
        the compiler can unroll without spilling to scratch memory.
        Caps the unroll factor used in codegen_iteration_ranges_entry
        and template emission to avoid VGPR explosion on RDNA1.

        Tier system:
          - Base: heavy=4, normal=8, light=16 (VGPR-pressure-driven)
          - loop_depth >= 3: drop one tier (16→8, 8→4, 4→4)
          - loop_depth >= 5: drop two tiers (16→4, 8→4, 4→4)

        Deep loop nests explode register pressure because every level
        adds index variables and boundary checks; unrolling them would
        multiply that by the unroll factor.

        Returns: 16 (light), 8 (moderate), 4 (heavy / deep loops).
        """
        dtype_bytes = 2 if self._packed16 else 4
        vgpr_class, _ = self._classify_vgpr_pressure(dtype_bytes)
        loop_depth = self._get_cached_loop_depth()
        if loop_depth is None:
            loop_depth = self._estimate_loop_depth()

        # Base unroll tier from VGPR pressure class
        tier_map = {"heavy": 4, "normal": 8, "light": 16}
        base = tier_map.get(vgpr_class, 8)

        # N+1.12: loop-depth-driven tier drops
        if loop_depth >= 5:
            drops = 2  # two tiers: 16→8→4 or 8→4→4 or 4→4→4
        elif loop_depth >= 3:
            drops = 1  # one tier: 16→8 or 8→4 or 4→4
        else:
            drops = 0

        for _ in range(drops):
            if base == 16:
                base = 8
            elif base == 8:
                base = 4
            # 4 is the floor — never drop below

        return base

    # ── DR.3: Op-class-specific threadgroup-size pickers ────────────

    def _pick_threadgroup_size_pointwise(self) -> int:
        """DR.3: Workgroup size for pointwise kernels.

        Pointwise kernels have simple bodies (no reduction loops) and
        low register pressure.  We optimize for occupancy: use larger
        workgroups to saturate the CU, dropping only when total numel
        is small or when VGPR pressure is unexpectedly high.
        """
        from .. import config

        if config.no_wg_tune():
            return 256

        try:
            from torch._dynamo.device_interface import get_interface_for_device

            iface = get_interface_for_device("vulkan")
            props = iface.Worker.get_device_properties()
            max_wg = props.max_workgroup_size
        except Exception:
            max_wg = 256

        sgs = self.simd_group_size or 64
        dtype_bytes = 2 if self._packed16 else 4

        # DR.3: use shared classification + cap helpers
        vgpr_class, estimated_vgprs = self._classify_vgpr_pressure(dtype_bytes)
        max_wg = self._apply_vgpr_cap(max_wg, estimated_vgprs, sgs)
        max_wg = self._apply_shared_mem_cap(max_wg, dtype_bytes, sgs)

        # DR.3: loop_depth penalty — pointwise kernels rarely have deep
        # loops, but vec4/packed16 can add an inner unroll loop.
        loop_depth = self._get_cached_loop_depth()
        if loop_depth is None:
            loop_depth = self._estimate_loop_depth()
        max_wg = self._apply_loop_depth_penalty(max_wg, vgpr_class, loop_depth)

        # ── Numel-driven sizing ────────────────────────────────────
        # Pointwise WG-size cap by VGPR pressure class.
        caps = {"light": 512, "normal": 384, "heavy": 256}
        if all(not is_dynamic(v) for v in self.numels.values()):
            total = 1
            for v in self.numels.values():
                total *= int(v)
            scale = 4 // dtype_bytes  # 1 for f32, 2 for f16/bf16
            cap = caps.get(vgpr_class, 256)
            if total > scale * 256 * 1024:
                wg_size = min(max_wg, cap)
            elif total > scale * 64 * 1024:
                wg_size = min(max_wg, min(384, cap))
            elif total > 0 and total < 256:
                if total <= sgs:
                    wg_size = min(max_wg, sgs)
                else:
                    n = 1
                    while n < total:
                        n <<= 1
                    wg_size = min(max_wg, n)
            else:
                wg_size = min(max_wg, cap)

            # GPU.4+: Grid-size + wave-slot awareness — if the grid
            # would produce fewer WGs than the device has wave slots,
            # shrink the WG size to fill all wave slots on all CUs.
            # RDNA1 can run 4 waves/CU (wave64) or 8 waves/CU (wave32).
            # Target: fill all wave slots for maximum occupancy.
            # Never go below one wave (sgs).
            if config.grid_aware_wg():
                try:
                    from torch._dynamo.device_interface import get_interface_for_device

                    iface = get_interface_for_device("vulkan")
                    props = iface.Worker.get_device_properties()
                    num_cus = getattr(props, "num_compute_units", 20)
                except Exception:
                    num_cus = 20
                # GPU.4+: Target filling all wave slots, not just one WG/CU.
                # wave64 (sgs=64): 4 waves/CU on RDNA1
                # wave32 (sgs=32): 8 waves/CU on RDNA1
                waves_per_cu = 4 if sgs >= 64 else 8
                target_wgs = num_cus * waves_per_cu
                num_wgs = (total + wg_size - 1) // wg_size
                if num_wgs < target_wgs and wg_size > sgs:
                    # Too few WGs for full wave-slot occupancy — reduce
                    # WG size to increase WG count, but never below one wave.
                    target_wg = max(sgs, total // target_wgs)
                    # Round down to power-of-two for hardware efficiency
                    rounded = sgs
                    while rounded * 2 <= target_wg:
                        rounded *= 2
                    wg_size = min(wg_size, max(sgs, rounded))
            # M11.5: round WG size up to wave-size multiple.
            if config.round_wg_to_wave() and wg_size % sgs != 0:
                wg_size = self._round_wg_to_wave(wg_size, max_wg, sgs)
            return wg_size

        # Dynamic-numel fall-through: use the same VGPR-class caps.
        wg_size = min(max_wg, caps.get(vgpr_class, 256))
        # M11.5: round WG size up to wave-size multiple.
        if config.round_wg_to_wave() and wg_size % sgs != 0:
            wg_size = self._round_wg_to_wave(wg_size, max_wg, sgs)
        return wg_size

    def _pick_threadgroup_size_reduction(self) -> int:
        """DR.3: Workgroup size for reduction kernels.

        Reduction kernels have deeper loops (persistent / cooperative
        staging, welford accumulators, multi-axis reductions) and higher
        register pressure.  We bias toward smaller workgroups to keep
        occupancy ≥2 waves/CU, and apply loop_depth + shared_mem
        penalties that are not needed for pointwise.
        """
        from torch._inductor.codegen.simd import prefix_is_reduction

        from .. import config

        if config.no_wg_tune():
            return 256

        try:
            from torch._dynamo.device_interface import get_interface_for_device

            iface = get_interface_for_device("vulkan")
            props = iface.Worker.get_device_properties()
            max_wg = props.max_workgroup_size
        except Exception:
            max_wg = 256

        sgs = self.simd_group_size or 64
        dtype_bytes = 2 if self._packed16 else 4

        # DR.3: shared helpers for VGPR classification + capping
        vgpr_class, estimated_vgprs = self._classify_vgpr_pressure(dtype_bytes)
        max_wg = self._apply_vgpr_cap(max_wg, estimated_vgprs, sgs)
        max_wg = self._apply_shared_mem_cap(max_wg, dtype_bytes, sgs)

        # DR.3: loop_depth penalty is critical for reductions.
        # Multi-axis welford reductions can easily reach loop_depth 4-5,
        # blowing VGPR budget if the workgroup is too large.
        loop_depth = self._get_cached_loop_depth()
        if loop_depth is None:
            loop_depth = self._estimate_loop_depth()
        max_wg = self._apply_loop_depth_penalty(max_wg, vgpr_class, loop_depth)

        rnumel = sympy.S.One
        for prefix, numel in self.numels.items():
            if prefix_is_reduction(prefix):
                rnumel = rnumel * numel
        if is_dynamic(rnumel):
            dyn_caps = {"light": 256, "normal": 256, "heavy": 128}
            wg_size = min(max_wg, dyn_caps.get(vgpr_class, 256))
            if config.round_wg_to_wave() and wg_size % sgs != 0:
                wg_size = self._round_wg_to_wave(wg_size, max_wg, sgs)
            return wg_size
        rn = int(rnumel)
        effective_rn = rn if dtype_bytes >= 4 else max(rn // 2, 1)

        # Reduction caps are more conservative than pointwise.
        # Heavy kernels (welford, f64, deep loops) get significantly
        # smaller workgroups to stay under the VGPR budget.
        rn_caps = {
            "light": (sgs * 4, sgs * 2, sgs),
            "normal": (sgs * 2, sgs, sgs),
            "heavy": (sgs, sgs // 2, max(sgs // 2, 32)),
        }
        cap_large, cap_medium, cap_small = rn_caps.get(vgpr_class, (sgs * 2, sgs, sgs))

        # M11.9: Grid-aware WG sizing for reductions.
        # Reductions have a fixed dispatch grid (one WG per output element),
        # so we can't increase grid count by shrinking WG like pointwise.
        # Instead, scale the WG cap based on total work per CU:
        #   work_per_cu = rnumel * grid_size / num_cus
        # When per-CU work is small, use fewer threads per WG to avoid
        # wasting wave slots.  When per-CU work is large, allow more
        # threads for faster per-element reduction.
        if config.grid_aware_wg():
            try:
                from torch._dynamo.device_interface import get_interface_for_device

                iface = get_interface_for_device("vulkan")
                props = iface.Worker.get_device_properties()
                num_cus = getattr(props, "num_compute_units", 20)
            except Exception:
                num_cus = 20
            # Dispatch grid = product of non-reduction dimensions
            non_red_numel = sympy.S.One
            for prefix, numel in self.numels.items():
                if not prefix_is_reduction(prefix):
                    non_red_numel = non_red_numel * numel
            if not is_dynamic(non_red_numel):
                grid_size = int(non_red_numel)
                # Total reduction elements per CU
                work_per_cu = (rn * grid_size) // num_cus
                if grid_size >= num_cus:
                    # Grid fills all CUs — allow full throughput
                    pass
                elif work_per_cu <= sgs:
                    # Very small problem: use single-wave WG
                    cap_large = cap_medium = cap_small = min(cap_small, sgs)
                elif work_per_cu <= sgs * 2:
                    cap_large = min(cap_large, sgs * 2)
                    cap_medium = min(cap_medium, sgs)

        if effective_rn > cap_large:
            wg_size = min(max_wg, 256)
        elif effective_rn > cap_medium:
            wg_size = min(max_wg, cap_medium)
        else:
            wg_size = min(max_wg, cap_small)
        # M11.5: round WG size up to wave-size multiple.
        if config.round_wg_to_wave() and wg_size % sgs != 0:
            wg_size = self._round_wg_to_wave(wg_size, max_wg, sgs)
        return wg_size
