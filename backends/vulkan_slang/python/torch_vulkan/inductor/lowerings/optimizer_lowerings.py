"""Optimizer step lowerings — _slang_foreach_optimizer template dispatch (CODEGEN.1).

Replaces the ``make_fallback()`` calls that previously handled the 4 optimizer
custom ops (``torch_vulkan::foreach_{sgd,sgd_momentum,adamw,lion}_step``).
Each lowering creates a ``_VulkanForeachOptimizerExternKernel(ir.ExternKernelOut)``
that emits a direct ``_pick_foreach_optimizer_caller(...)``  call during
codegen, eliminating the FallbackKernel indirection.

Exit gate: ``git grep 'make_fallback.*optimizer' lowerings/`` returns 0 hits.
"""

from __future__ import annotations


def _is_vulkan(x) -> bool:
    """Return True if x is a Vulkan IR node."""
    try:
        return x.get_device().type == "vulkan"
    except Exception:
        return False


def _register_optimizer_lowerings() -> None:
    """Register lowerings for torch_vulkan::foreach_{sgd,sgd_momentum,adamw,lion}_step.

    Each lowering:
    1. Validates Vulkan device.
    2. Creates a ``_VulkanForeachOptimizerExternKernel`` whose codegen emits
       ``_pick_foreach_optimizer_caller(algorithm, n, 'float')(params, grads, ...)``
       directly in the generated wrapper — replacing the FallbackKernel path
       that previously called ``torch.ops.torch_vulkan.foreach_*_step(...)``.
    3. Returns None (void in-place op).

    Anti-goal #7: file kept under 800 lines (currently ~250).
    """
    import torch
    from torch._inductor import ir
    from torch._inductor.lowering import (
        add_needs_realized_inputs,
        register_lowering,
    )

    # ═════════════════════════════════════════════════════════════════════
    # CODEGEN.1: _VulkanForeachOptimizerExternKernel
    # ═════════════════════════════════════════════════════════════════════

    class _VulkanForeachOptimizerExternKernel(ir.ExternKernel):
        """Void mutation ExternKernel that dispatches an optimizer step via the
        Slang foreach_optimizer template.

        Inputs layout (flattened):
            [param0, grad0, param1, grad1, ..., param_{n-1}, grad_{n-1},
             mom_buf0, ..., mom_buf_{n-1},        # if algorithm in {sgd_momentum, adamw, lion}
             v_buf0, ..., v_buf_{n-1}]             # if algorithm == "adamw"

        Uses NoneLayout (no output tensor).  All param tensors at even indices
        are registered as mutated so the scheduler preserves ordering.

        The codegen override emits a call to
        ``_pick_foreach_optimizer_caller(algorithm, n, 'float')``  —
        the picklable ``_SlangForeachOptimizer`` factory — which handles
        batch-size selection and multi-dispatch automatically.
        """

        def __init__(
            self,
            inputs,
            algorithm: str,
            n_params: int,
            scalar_kwargs: dict[str, list],
        ):
            dev = inputs[0].get_device()
            # Collect param IR nodes before unwrap_storage (needed for MutationOutput).
            param_inputs = [inputs[i * 2] for i in range(n_params)]
            super().__init__(
                None,
                ir.NoneLayout(device=dev),
                self.unwrap_storage(inputs),
                (),
            )
            self.algorithm = algorithm
            self.n_params = n_params
            self.scalar_kwargs = scalar_kwargs
            # Inductor constraint: each output buffer can declare at most 1 mutation.
            # Use one MutationOutput per param (each with mutation_names=[param_name]),
            # stored in self.mutation_outputs (returned by ExternKernel.get_outputs()).
            # MutationOutput.__init__ calls mark_buffer_mutated internally.
            for p in param_inputs:
                self.mutation_outputs.append(
                    ir.MutationOutput(ir.NoneLayout(device=dev), p, self)
                )
            self.name = ir.V.graph.register_buffer(self)
            ir.V.graph.register_operation(self)

        def has_side_effects(self) -> bool:
            return True

        def should_allocate(self) -> bool:
            return False

        def codegen(self, wrapper):
            """Emit a call to ``_pick_foreach_optimizer_caller`` in the wrapper.

            In AOTI mode, compiles the foreach optimizer Slang template to
            SPIR-V at codegen time and emits C++ dispatch calls through
            the ``emit_aoti_extern_dispatch`` helper.
            """
            from torch._inductor import graph as _inductor_graph

            input_names = [inp.codegen_reference() for inp in self.inputs]
            n = self.n_params

            was_aoti = getattr(_inductor_graph.V.graph, 'aot_mode', False)

            if was_aoti:
                self._codegen_aoti(wrapper, input_names, n)
                return

            # M-NEW.12: flush batcher before direct Vulkan dispatch
            wrapper._flush_batcher_before_direct_call()

            wrapper.add_import_once(
                "from torch_vulkan.inductor.templates.caller.optimizer "
                "import _pick_foreach_optimizer_caller"
            )

            # ── Split inputs into param/grad/momentum/v buffers ────────
            params_refs = [input_names[i * 2] for i in range(n)]
            grads_refs = [input_names[i * 2 + 1] for i in range(n)]

            # ── Scalar keyword arguments (lr, weight_decay, etc.) ──────
            scalar_parts: list[str] = []
            for key, val in self.scalar_kwargs.items():
                scalar_parts.append(f"{key}={val!r}")

            # ── Optional momentum_bufs / v_bufs ────────────────────────
            offset = 2 * n
            remaining = input_names[offset:]
            mom_bufs_str = ""
            v_bufs_str = ""
            if self.algorithm in ("sgd_momentum", "adamw", "lion"):
                mom_refs = remaining[:n]
                mom_bufs_str = f", momentum_bufs=[{', '.join(mom_refs)}]"
                remaining = remaining[n:]
            if self.algorithm == "adamw":
                v_refs = remaining[:n]
                v_bufs_str = f", v_bufs=[{', '.join(v_refs)}]"

            self.codegen_comment(wrapper)
            wrapper.writeline(
                f"_pick_foreach_optimizer_caller('{self.algorithm}', {n}, 'float')"
                f"([{', '.join(params_refs)}], [{', '.join(grads_refs)}]"
                f"{', ' + ', '.join(scalar_parts) if scalar_parts else ''}"
                f"{mom_bufs_str}{v_bufs_str})"
            )
            # NoneLayout — no output tensor to assert sizes on.

        def _codegen_aoti(self, wrapper, input_names, n_params):
            """Emit AOTI C++ dispatch for the foreach optimizer template."""
            import struct

            from ..templates.caller.optimizer import (
                _foreach_cache_key,
                _OPTIMIZER_BATCH_SIZES,
                _render_foreach_optimizer_slang,
                _resolve_algorithm_type,
            )

            algorithm_type = _resolve_algorithm_type(self.algorithm)

            # 1. Pick batch_size: smallest pre-defined size >= n_params
            batch_size = n_params
            for bs in _OPTIMIZER_BATCH_SIZES:
                if bs >= n_params:
                    batch_size = bs
                    break
            if n_params > _OPTIMIZER_BATCH_SIZES[-1]:
                batch_size = _OPTIMIZER_BATCH_SIZES[-1]

            # 2. Render Slang template and compute cache key
            slang_src = _render_foreach_optimizer_slang(
                algorithm_type, batch_size, "float", parameter_array=False
            )
            cache_key = _foreach_cache_key(
                algorithm_type, batch_size, "float", parameter_array=False
            )

            # 3. Extract concrete scalar values from scalar_kwargs
            def _extract_scalar_list(key, default_val):
                vals = self.scalar_kwargs.get(key)
                if vals is None:
                    return [default_val] * n_params
                # Try to extract concrete values from IR nodes
                result = []
                for v in vals:
                    try:
                        # ir.Constant wrapping a Python scalar
                        if hasattr(v, 'value'):
                            result.append(float(v.value))
                        # plain Python number
                        elif isinstance(v, (int, float)):
                            result.append(float(v))
                        # sympy expression
                        else:
                            from sympy import Integer as SymInt, Float as SymFloat
                            if isinstance(v, (SymInt, SymFloat)):
                                result.append(float(v))
                            else:
                                raise ValueError(f"Cannot extract concrete value from {type(v)}: {v}")
                    except Exception as e:
                        raise NotImplementedError(
                            f"Optimizer AOTI: cannot extract scalar {key} value from "
                            f"{type(v).__name__}={v!r}. Use concrete (non-symbolic) "
                            f"optimizer scalars for AOTI compilation."
                        ) from e
                return result

            lr_vals = _extract_scalar_list("lr", 0.01)
            wd_vals = _extract_scalar_list("weight_decay", 0.0)
            momentum_vals = _extract_scalar_list("momentum", 0.0)
            beta2_vals = _extract_scalar_list("beta2", 0.0)
            eps_vals = _extract_scalar_list("eps", 0.0)

            # 4. Determine numels from the first param's layout
            params_ir = self.inputs[: n_params * 2 : 2]  # every other from index 0
            numel = None
            for pi in params_ir:
                try:
                    sz = pi.get_size()
                    if sz:
                        from sympy import prod
                        numel = int(prod(sz))
                        break
                except Exception:
                    pass
            if numel is None:
                raise NotImplementedError(
                    "Optimizer AOTI: cannot determine parameter numel. "
                    "Ensure parameter shapes are statically known."
                )
            numels = [numel] * n_params

            # 5. Build push constants in the B1 format
            # Header: (n_params, pad, pad, pad) as 4 uint32
            pc_values = [n_params, 0, 0, 0]

            for i in range(n_params):
                # Pack ParamConfig: numel (uint32) + 6 floats (as uint32 via struct)
                pc_values.append(numels[i])
                floats_to_pack = [
                    lr_vals[i],
                    wd_vals[i],
                    momentum_vals[i],
                    beta2_vals[i],
                    eps_vals[i],
                    0.0,  # _pad
                ]
                for f in floats_to_pack:
                    packed = struct.unpack("<I", struct.pack("<f", f))[0]
                    pc_values.append(packed)

            # 6. Pad remaining ParamConfig slots for batch_size - n_params
            for _ in range(n_params, batch_size):
                pc_values.append(0)  # numel = 0
                for _f in range(6):
                    pc_values.append(0)

            # 7. Build buffer names in the order the shader expects
            # Layout: [p0, g0, m0, v0, p1, g1, m1, v1, ...]
            buffer_names: list[str] = []
            offset = 2 * n_params  # past param/grad pairs
            remaining = input_names[offset:]

            for i in range(n_params):
                buffer_names.append(input_names[i * 2])      # param_i
                buffer_names.append(input_names[i * 2 + 1])  # grad_i
                # momentum buffer
                if self.algorithm in ("sgd_momentum", "adamw", "lion"):
                    buffer_names.append(remaining[i])
                else:
                    buffer_names.append("_opt_dummy")  # placeholder
                # v buffer
                if self.algorithm == "adamw":
                    # v bufs come after momentum bufs
                    v_idx = n_params + i
                    if v_idx < len(remaining):
                        buffer_names.append(remaining[v_idx])
                    else:
                        buffer_names.append("_opt_dummy")
                else:
                    buffer_names.append("_opt_dummy")

            # Pad remaining slots for batch_size
            for _ in range(n_params, batch_size):
                buffer_names.extend(["_opt_dummy", "_opt_dummy", "_opt_dummy", "_opt_dummy"])

            # 8. Compute grid
            threadgroup_size = 256
            grid_x = (numel + threadgroup_size - 1) // threadgroup_size
            grid_y = n_params
            grid_z = 1

            # 9. Allocate dummy buffer for algorithms that don't need momentum/v buffers
            output_allocations = []
            if "_opt_dummy" in buffer_names:
                output_allocations.append({
                    "name": "_opt_dummy",
                    "shape": [1],
                    "stride": [1],
                    "dtype": "float32",
                })

            # 10. Emit AOTI dispatch
            wrapper.emit_aoti_extern_dispatch(
                slang_src=slang_src,
                cache_key=cache_key,
                buffer_names=buffer_names,
                pc_values=pc_values,
                grid_x=grid_x,
                grid_y=grid_y,
                grid_z=grid_z,
                num_outputs=n_params * 3,
                output_allocations=output_allocations if output_allocations else None,
            )

    # ═════════════════════════════════════════════════════════════════════
    # Lowering helpers
    # ═════════════════════════════════════════════════════════════════════

    def _create_extern_kernel(
        algorithm: str,
        params: list,
        grads: list,
        scalar_kwargs: dict,
        extra_buffers: list | None = None,
    ):
        """Build inputs list and create a void mutation optimizer kernel.

        The kernel self-registers via __init__ (mark_buffer_mutated +
        register_buffer + register_operation), so no return value is needed.
        """
        n = len(params)
        flat_inputs: list = []
        for i in range(n):
            flat_inputs.append(params[i])
            flat_inputs.append(grads[i])
        if extra_buffers:
            flat_inputs.extend(extra_buffers)

        _VulkanForeachOptimizerExternKernel(
            inputs=flat_inputs,
            algorithm=algorithm,
            n_params=n,
            scalar_kwargs=scalar_kwargs,
        )

    # ═════════════════════════════════════════════════════════════════════
    # Per-op lowerings
    # ═════════════════════════════════════════════════════════════════════

    def _register_sgd_step():
        op = getattr(torch.ops.torch_vulkan, "foreach_sgd_step", None)
        if op is None:
            return
        op_default = getattr(op, "default", op)
        add_needs_realized_inputs(op_default)

        @register_lowering(op, type_promotion_kind=None)
        def _vulkan_foreach_sgd_step(params, grads, lr, weight_decay):
            if not params or not _is_vulkan(params[0]):
                return NotImplemented
            n = len(params)
            # Broadcast length-1 scalar lists to full length.
            if len(lr) == 1 and n > 1:
                lr = list(lr) * n
            if len(weight_decay) == 1 and n > 1:
                weight_decay = list(weight_decay) * n
            _create_extern_kernel(
                "sgd", params, grads, {"lr": list(lr), "weight_decay": list(weight_decay)}
            )
            return None  # void op — params mutated in-place

    def _register_sgd_momentum_step():
        op = getattr(torch.ops.torch_vulkan, "foreach_sgd_momentum_step", None)
        if op is None:
            return
        op_default = getattr(op, "default", op)
        add_needs_realized_inputs(op_default)

        @register_lowering(op, type_promotion_kind=None)
        def _vulkan_foreach_sgd_momentum_step(
            params, grads, momentum_bufs, lr, weight_decay, momentum
        ):
            if not params or not _is_vulkan(params[0]):
                return NotImplemented
            n = len(params)
            if len(lr) == 1 and n > 1:
                lr = list(lr) * n
            if len(weight_decay) == 1 and n > 1:
                weight_decay = list(weight_decay) * n
            if len(momentum) == 1 and n > 1:
                momentum = list(momentum) * n
            _create_extern_kernel(
                "sgd_momentum",
                params,
                grads,
                {
                    "lr": list(lr),
                    "weight_decay": list(weight_decay),
                    "momentum": list(momentum),
                },
                extra_buffers=list(momentum_bufs),
            )
            return None  # void op — params and momentum_bufs mutated in-place

    def _register_adamw_step():
        op = getattr(torch.ops.torch_vulkan, "foreach_adamw_step", None)
        if op is None:
            return
        op_default = getattr(op, "default", op)
        add_needs_realized_inputs(op_default)

        @register_lowering(op, type_promotion_kind=None)
        def _vulkan_foreach_adamw_step(
            params, grads, m_bufs, v_bufs, lr, weight_decay, beta1, beta2, eps
        ):
            if not params or not _is_vulkan(params[0]):
                return NotImplemented
            n = len(params)
            if len(lr) == 1 and n > 1:
                lr = list(lr) * n
            if len(weight_decay) == 1 and n > 1:
                weight_decay = list(weight_decay) * n
            if len(beta1) == 1 and n > 1:
                beta1 = list(beta1) * n
            if len(beta2) == 1 and n > 1:
                beta2 = list(beta2) * n
            if len(eps) == 1 and n > 1:
                eps = list(eps) * n
            _create_extern_kernel(
                "adamw",
                params,
                grads,
                {
                    "lr": list(lr),
                    "weight_decay": list(weight_decay),
                    "momentum": list(beta1),  # template uses 'momentum' for beta1
                    "beta2": list(beta2),
                    "eps": list(eps),
                },
                extra_buffers=list(m_bufs) + list(v_bufs),
            )
            return None  # void op — params, m_bufs, v_bufs mutated in-place

    def _register_lion_step():
        op = getattr(torch.ops.torch_vulkan, "foreach_lion_step", None)
        if op is None:
            return
        op_default = getattr(op, "default", op)
        add_needs_realized_inputs(op_default)

        @register_lowering(op, type_promotion_kind=None)
        def _vulkan_foreach_lion_step(
            params, grads, momentum_bufs, lr, weight_decay, beta1, beta2
        ):
            if not params or not _is_vulkan(params[0]):
                return NotImplemented
            n = len(params)
            if len(lr) == 1 and n > 1:
                lr = list(lr) * n
            if len(weight_decay) == 1 and n > 1:
                weight_decay = list(weight_decay) * n
            if len(beta1) == 1 and n > 1:
                beta1 = list(beta1) * n
            if len(beta2) == 1 and n > 1:
                beta2 = list(beta2) * n
            _create_extern_kernel(
                "lion",
                params,
                grads,
                {
                    "lr": list(lr),
                    "weight_decay": list(weight_decay),
                    "momentum": list(beta1),
                    "beta2": list(beta2),
                },
                extra_buffers=list(momentum_bufs),
            )
            return None  # void op — params and momentum_bufs mutated in-place

    # ── Install all 4 lowerings ────────────────────────────────────────
    _register_sgd_step()
    _register_sgd_momentum_step()
    _register_adamw_step()
    _register_lion_step()
