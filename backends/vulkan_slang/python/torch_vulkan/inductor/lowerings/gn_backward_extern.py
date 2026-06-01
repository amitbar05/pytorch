"""M22.6 — GroupNorm backward via fused Slang shaders (ExternKernelOut).

Replaces the old ~15 dispatch decomposition with 2 fused dispatches:
  1. ``group_norm_backward.slang`` — grad_input in one kernel
  2. ``group_norm_backward_weight.slang`` — grad_weight + grad_bias in one kernel

Both shaders use ParameterBlock<KernelArgs> (M-SF.1 compliant).
"""

from __future__ import annotations

import torch as _torch_module
from torch._inductor import ir as _ir_module

_aten = _torch_module.ops.aten


def _vk_realize_then_unwrap(x):
    """Realize Pointwise/Reduction, then unwrap StorageBox → data."""
    if isinstance(x, _ir_module.TensorBox):
        x = x.data
    if isinstance(x, _ir_module.StorageBox) and isinstance(
        x.data, (_ir_module.Pointwise, _ir_module.Reduction)
    ):
        x.realize()
    if isinstance(x, _ir_module.StorageBox):
        return x.data
    return x


class _VulkanGNBwdInputExternKernel(_ir_module.ExternKernelOut):
    """ExternKernelOut that dispatches ``_dispatch_group_norm_backward_slang``.

    The primary output (``layout``) is grad_input.  Input tensors are
    grad_output, input, mean, rstd, and optionally gamma.

    When gamma is None, a dummy 1-element buffer is used as weight so
    the shader's has_weight=0 path applies (no per-element gamma scale).
    """

    @staticmethod
    def unwrap_storage(inputs):
        """Unwrap all computation inputs. Keep the LAST input (gi_buf) as
        StorageBox to preserve data dependency for MutationOutput."""
        if not inputs:
            return []
        inputs_new = []
        for i, x in enumerate(inputs):
            if i < len(inputs) - 1:
                x = _vk_realize_then_unwrap(x)
            # else: keep last (gi_buf) as StorageBox for MutationOutput
            inputs_new.append(x)
        return inputs_new

    def __init__(
        self,
        layout,
        inputs,
        num_groups: int,
    ):
        # inputs: [grad_output, input_t, mean, rstd, weight_or_none, gi_buf_or_none]
        # Filter out None entries for super().__init__ input tracking.
        clean_inputs = [x for x in inputs if x is not None]
        super().__init__(
            layout=layout,
            inputs=clean_inputs,
            python_kernel_name=(
                "torch_vulkan.inductor.fx_passes.eager.conv_gn_relu"
                "._dispatch_group_norm_backward_slang"
            ),
            op_overload=None,
        )
        self.num_groups = num_groups
        self._has_weight = inputs[4] is not None if len(inputs) > 4 else False
        self._gi_buf = inputs[-1]
        # GN-BWD-FIX: mark gi_buf (last clean input) as MutationOutput so the
        # scheduler knows this kernel WRITES to it. Without this, the scheduler
        # treats gi_buf as read-only and allocates a separate output buffer,
        # which stays zero — gradients never reach param.grad.
        from torch._inductor.ir import MutationOutput, NoneLayout
        gi_node = self.inputs[-1]  # grad_input buffer (last non-None input)
        self.mutation_outputs.append(
            MutationOutput(NoneLayout(device=layout.device), gi_node, self)
        )

    def codegen(self, wrapper):
        # M-NEW.12: flush batcher before this direct Vulkan dispatch.
        # Without this flush, any batched kernel (e.g., ReLU backward
        # pointwise) whose output feeds into this GN backward runs AFTER
        # this synchronous dispatch → GN backward reads stale/zero data
        # → zero gradients flow to upstream conv → model doesn't learn.
        wrapper._flush_batcher_before_direct_call()

        wrapper.add_import_once(
            "from torch_vulkan.inductor.fx_passes.eager.conv_gn_relu "
            "import _dispatch_group_norm_backward_slang"
        )

        # GN-BWD-FIX: explicit allocation for primary output (gi_buf is the
        # mutation output; the wrapper won't allocate this buffer because no
        # downstream consumer references the ExternKernelOut's primary output).
        out_name = self.codegen_reference()
        layout = self.get_layout()
        size = list(layout.size)
        stride = list(layout.stride)
        dtype_str = "torch.float32"
        wrapper.writeline(
            f"{out_name} = empty_strided_vulkan({size}, {stride}, {dtype_str}, "
            f"lifetime_class='transient')"
        )

        names = [inp.codegen_reference() for inp in self.inputs]
        # Inputs after None-filtering: [grad_output, input_t, mean, rstd, weight?, gi_buf]
        grad_out = names[0]
        inp = names[1]
        mean = names[2]
        rstd = names[3]
        weight = names[4] if self._has_weight else "None"
        gi_out = names[-1]  # grad_input buffer (last non-None input)

        G = self.num_groups

        wrapper.writeline(f"{gi_out}.zero_()")
        self.codegen_comment(wrapper)
        wrapper.writeline(
            f"_dispatch_group_norm_backward_slang("
            f"{grad_out}, {inp}, {mean}, {rstd}, "
            f"{weight}, {gi_out}, {G})"
        )
        self.codegen_size_asserts(wrapper)
        wrapper.add_import_once("import torch_vulkan")
        wrapper.writeline("torch_vulkan.synchronize(0)")


class _VulkanGNBwdWeightExternKernel(_ir_module.ExternKernelOut):
    """ExternKernelOut that dispatches ``_dispatch_group_norm_backward_weight_slang``.

    The primary output (``layout``) is grad_weight.  grad_bias is passed
    as a pre-allocated buffer that the kernel writes into.
    """

    @staticmethod
    def unwrap_storage(inputs):
        """Unwrap only computational inputs (indices 0-3). Keep gw_buf/gb_buf
        (indices 4+) as StorageBox to preserve data dependency for MutationOutput."""
        inputs_new = []
        for i, x in enumerate(inputs):
            if i <= 3:
                x = _vk_realize_then_unwrap(x)
            # else: keep as StorageBox to preserve dependency for MutationOutput
            inputs_new.append(x)
        return inputs_new

    def __init__(
        self,
        layout,
        inputs,
        num_groups: int,
        compute_bias: bool,
    ):
        super().__init__(
            layout=layout,
            inputs=inputs,
            python_kernel_name=(
                "torch_vulkan.inductor.fx_passes.eager.conv_gn_relu"
                "._dispatch_group_norm_backward_weight_slang"
            ),
            op_overload=None,
        )
        self.num_groups = num_groups
        self.compute_bias = compute_bias
        # GN-BWD-FIX: mark gw_buf (inputs[4]) and gb_buf (inputs[5] if present)
        # as MutationOutput so the scheduler knows this kernel WRITES to them.
        from torch._inductor.ir import MutationOutput, NoneLayout
        gw_node = self.inputs[4]  # grad_weight buffer
        self.mutation_outputs.append(
            MutationOutput(NoneLayout(device=layout.device), gw_node, self)
        )
        if compute_bias and len(self.inputs) > 5:
            gb_node = self.inputs[5]  # grad_bias buffer
            self.mutation_outputs.append(
                MutationOutput(NoneLayout(device=layout.device), gb_node, self)
            )

    def codegen(self, wrapper):
        # M-NEW.12: flush batcher before this direct Vulkan dispatch
        # (same pattern as _VulkanGNBwdInputExternKernel above).
        wrapper._flush_batcher_before_direct_call()

        wrapper.add_import_once(
            "from torch_vulkan.inductor.fx_passes.eager.conv_gn_relu "
            "import _dispatch_group_norm_backward_weight_slang"
        )

        # GN-BWD-FIX: explicit allocation for primary output (gw_buf is the
        # mutation output; the wrapper won't allocate this buffer automatically).
        out_name = self.codegen_reference()
        layout = self.get_layout()
        size = list(layout.size)
        stride = list(layout.stride)
        dtype_str = "torch.float32"
        wrapper.writeline(
            f"{out_name} = empty_strided_vulkan({size}, {stride}, {dtype_str}, "
            f"lifetime_class='transient')"
        )

        names = [inp.codegen_reference() for inp in self.inputs]
        grad_out = names[0]
        inp = names[1]
        mean = names[2]
        rstd = names[3]
        gw_out = names[4]  # grad_weight buffer
        G = self.num_groups

        # grad_bias is the last input if compute_bias is True
        gb_arg = "None"
        if self.compute_bias and len(names) > 5:
            gb_arg = names[5]

        wrapper.writeline(f"{gw_out}.zero_()")
        if self.compute_bias and gb_arg != "None":
            wrapper.writeline(f"{gb_arg}.zero_()")

        self.codegen_comment(wrapper)

        compute_bias_str = "True" if self.compute_bias else "False"
        if gb_arg == "None":
            wrapper.writeline(
                f"_dispatch_group_norm_backward_weight_slang("
                f"{grad_out}, {inp}, {mean}, {rstd}, "
                f"{gw_out}, {gw_out}, {G}, "
                f"compute_weight=True, compute_bias={compute_bias_str})"
            )
        else:
            wrapper.writeline(
                f"_dispatch_group_norm_backward_weight_slang("
                f"{grad_out}, {inp}, {mean}, {rstd}, "
                f"{gw_out}, {gb_arg}, {G}, "
                f"compute_weight=True, compute_bias={compute_bias_str})"
            )
        self.codegen_size_asserts(wrapper)
        wrapper.add_import_once("import torch_vulkan")
        wrapper.writeline("torch_vulkan.synchronize(0)")
