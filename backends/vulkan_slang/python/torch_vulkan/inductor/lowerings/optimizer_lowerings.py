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

    class _VulkanForeachOptimizerExternKernel(ir.ExternKernelOut):
        """ExternKernelOut that dispatches an optimizer step via the Slang
        foreach_optimizer template.

        Inputs layout (flattened):
            [param0, grad0, param1, grad1, ..., param_{n-1}, grad_{n-1},
             mom_buf0, ..., mom_buf_{n-1},        # if algorithm in {sgd_momentum, adamw, lion}
             v_buf0, ..., v_buf_{n-1}]             # if algorithm == "adamw"

        The ``layout`` is derived from the first param (used only for
        scheduler metadata — the kernel mutates params in-place).

        The codegen override emits a call to
        ``_pick_foreach_optimizer_caller(algorithm, n, 'float')``  —
        the picklable ``_SlangForeachOptimizer`` factory — which handles
        batch-size selection and multi-dispatch automatically.
        """

        def __init__(
            self,
            layout,
            inputs,
            algorithm: str,
            n_params: int,
            scalar_kwargs: dict[str, list],
        ):
            super().__init__(
                layout=layout,
                inputs=inputs,
                python_kernel_name=(
                    "torch_vulkan.inductor.templates.caller.optimizer"
                    "._slang_foreach_optimizer"
                ),
                op_overload=None,
            )
            self.algorithm = algorithm
            self.n_params = n_params
            self.scalar_kwargs = scalar_kwargs

        def codegen(self, wrapper):
            """Emit a call to ``_pick_foreach_optimizer_caller`` in the wrapper."""
            # M-NEW.12: flush batcher before direct Vulkan dispatch
            wrapper._flush_batcher_before_direct_call()

            wrapper.add_import_once(
                "from torch_vulkan.inductor.templates.caller.optimizer "
                "import _pick_foreach_optimizer_caller"
            )

            input_names = [inp.codegen_reference() for inp in self.inputs]
            n = self.n_params

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
            self.codegen_size_asserts(wrapper)

    # ═════════════════════════════════════════════════════════════════════
    # Lowering helpers
    # ═════════════════════════════════════════════════════════════════════

    def _make_optimizer_layout(param_ir):
        """Create a FixedLayout matching *param_ir* for the ExternKernelOut."""
        try:
            dev = param_ir.get_device()
            dtype = param_ir.get_dtype()
            size = list(param_ir.get_size())
            stride = list(param_ir.get_stride())
        except Exception:
            # Fallback: minimal layout for scheduling.
            import sympy

            dev = torch.device("vulkan")
            dtype = torch.float32
            size = [sympy.Integer(1)]  # type: ignore[assignment]
            stride = [sympy.Integer(1)]  # type: ignore[assignment]
        return ir.FixedLayout(device=dev, dtype=dtype, size=size, stride=stride)

    def _create_extern_kernel(
        algorithm: str,
        params: list,
        grads: list,
        scalar_kwargs: dict,
        extra_buffers: list | None = None,
    ):
        """Build inputs list + ExternKernelOut for an optimizer step."""
        n = len(params)
        # Flatten: [p0, g0, p1, g1, ...]
        flat_inputs: list = []
        for i in range(n):
            flat_inputs.append(params[i])
            flat_inputs.append(grads[i])
        if extra_buffers:
            flat_inputs.extend(extra_buffers)

        layout = _make_optimizer_layout(params[0])
        kernel = _VulkanForeachOptimizerExternKernel(
            layout=layout,
            inputs=flat_inputs,
            algorithm=algorithm,
            n_params=n,
            scalar_kwargs=scalar_kwargs,
        )
        # Ensure all input tensors are realized before dispatch.
        return ir.TensorBox.create(kernel)

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
            return None  # void op

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
            return None

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
            return None

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
            return None

    # ── Install all 4 lowerings ────────────────────────────────────────
    _register_sgd_step()
    _register_sgd_momentum_step()
    _register_adamw_step()
    _register_lion_step()
