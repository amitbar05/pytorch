"""GN.1 — GroupNorm forward via fused Slang shader (ExternKernelOut).

Replaces the ~10 dispatch decomposition in norm.py with 1 fused dispatch
using the existing ``group_norm.slang`` shader.  Reduces dispatch count
by ~9 per GN layer and eliminates intermediate buffers for var_mean, dx,
var_eps, rstd, etc.

Multi-output: The shader writes save_mean/rstd into pre-allocated buffers
(MutationOutput pattern), in addition to the primary normalized output.

Pattern mirrors ``gn_backward_extern.py`` (GN backward ExternKernelOut).
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


class _VulkanGNFwdExternKernel(_ir_module.ExternKernelOut):
    """ExternKernelOut for fused GroupNorm forward.

    The primary output (``layout``) is the normalized output [N, C, H, W].
    ``save_mean`` [N*G] and ``save_rstd`` [N*G] are pre-allocated buffers
    passed as additional inputs and marked as MutationOutput — the shader
    writes into them for the backward pass.

    Inputs list structure:
        [input_t, weight, bias, save_mean_buf, save_rstd_buf]

    Where save_mean_buf and save_rstd_buf are pre-allocated via
    aten.empty.memory_format in the lowering function.
    """

    @staticmethod
    def unwrap_storage(inputs):
        """Unwrap computational inputs (0-2). Keep save_mean/rstd (3-4)
        as StorageBox for MutationOutput."""
        inputs_new = []
        for i, x in enumerate(inputs):
            if i <= 2:  # input, weight, bias
                x = _vk_realize_then_unwrap(x)
            # else: keep as StorageBox for MutationOutput
            inputs_new.append(x)
        return inputs_new

    def __init__(
        self,
        layout,
        inputs,
        num_groups: int,
        eps: float,
    ):
        # inputs: [input_t, weight, bias, save_mean_buf, save_rstd_buf]
        super().__init__(
            layout=layout,
            inputs=inputs,
            python_kernel_name=(
                "torch_vulkan.inductor.fx_passes.eager.conv_gn_relu"
                "._dispatch_group_norm_slang"
            ),
            op_overload=None,
        )
        self.num_groups = num_groups
        self.eps = eps

        # Mark save_mean and save_rstd as MutationOutput so the scheduler
        # knows this kernel WRITES to them (for use in backward).
        from torch._inductor.ir import MutationOutput, NoneLayout

        # save_mean = inputs[3], save_rstd = inputs[4]
        if len(self.inputs) > 3:
            self.mutation_outputs.append(
                MutationOutput(
                    NoneLayout(device=layout.device),
                    self.inputs[3],
                    self,
                )
            )
        if len(self.inputs) > 4:
            self.mutation_outputs.append(
                MutationOutput(
                    NoneLayout(device=layout.device),
                    self.inputs[4],
                    self,
                )
            )

    def codegen(self, wrapper):
        wrapper._flush_batcher_before_direct_call()

        wrapper.add_import_once(
            "from torch_vulkan.inductor.fx_passes.eager.conv_gn_relu "
            "import _dispatch_group_norm_slang"
        )

        names = [inp.codegen_reference() for inp in self.inputs]
        inp_name = names[0]
        weight_name = names[1]
        bias_name = names[2]

        # Primary output allocation
        out_name = self.codegen_reference()
        layout = self.get_layout()
        size = list(layout.size)
        stride = list(layout.stride)
        wrapper.writeline(
            f"{out_name} = empty_strided_vulkan({size}, {stride}, "
            f"\"torch.float32\", lifetime_class='transient')"
        )

        G = self.num_groups
        eps = self.eps

        # Pass pre-allocated save_mean / save_rstd buffers
        save_mean_name = names[3] if len(names) > 3 else "None"
        save_rstd_name = names[4] if len(names) > 4 else "None"

        self.codegen_comment(wrapper)
        if save_mean_name != "None":
            wrapper.writeline(
                f"_dispatch_group_norm_slang("
                f"{inp_name}, {weight_name}, {bias_name}, "
                f"{out_name}, {G}, {eps}, "
                f"save_mean={save_mean_name}, save_rstd={save_rstd_name})"
            )
        else:
            wrapper.writeline(
                f"_dispatch_group_norm_slang("
                f"{inp_name}, {weight_name}, {bias_name}, "
                f"{out_name}, {G}, {eps})"
            )
        self.codegen_size_asserts(wrapper)
        wrapper.add_import_once("import torch_vulkan")
        wrapper.writeline("torch_vulkan.synchronize(0)")
