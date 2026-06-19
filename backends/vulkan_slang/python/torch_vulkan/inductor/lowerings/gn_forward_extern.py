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
    """Realize Pointwise/Reduction, then unwrap StorageBox → data.

    Handles nested StorageBox (TensorBox → StorageBox → StorageBox → Pointwise)
    by looping until the innermost data is reached.
    Also unwraps View nodes to their inner data.
    """
    if isinstance(x, _ir_module.TensorBox):
        x = x.data
    # S2.0b: unwrap StorageBox/View layers to a fixpoint (interleaved nesting
    # like StorageBox → View → StorageBox → Buffer would otherwise leave a
    # trailing StorageBox that crashes decide_layout).
    while True:
        if isinstance(x, _ir_module.StorageBox):
            x = x.data
            continue
        if isinstance(x, _ir_module.BaseView) and hasattr(x, 'data'):
            x = x.data
            continue
        break
    # If result is not a real Buffer (Pointwise/Reduction/etc.),
    # wrap in a ComputedBuffer so it gets codegen_reference() and allocation.
    if not isinstance(x, (_ir_module.Buffer, _ir_module.ReinterpretView)):
        from torch._inductor.graph import V
        layout = _ir_module.FlexibleLayout(
            device=x.get_device(),
            dtype=x.get_dtype(),
            size=list(x.get_size()),
        )
        buf = _ir_module.ComputedBuffer(name=None, layout=layout, data=x)
        V.graph.register_buffer(buf, set_name=True)
        V.graph.register_operation(buf)  # Sets operation_name for scheduler
        return buf
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
        # A2.5: AOTI mode — emit C++ dispatch calls instead of Python
        from torch._inductor import graph as _inductor_graph

        if getattr(_inductor_graph.V.graph, 'aot_mode', False):
            self._codegen_aoti(wrapper)
            return

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
            f"torch.float32, lifetime_class='transient')"
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

    def _codegen_aoti(self, wrapper):
        """Emit C++ AOTI dispatch for GN forward via pre-compiled SPIR-V."""
        import os
        import struct

        names = [inp.codegen_reference() for inp in self.inputs]
        out_name = self.codegen_reference()
        inp_name = names[0]
        weight_name = names[1]
        bias_name = names[2]
        save_mean_name = names[3] if len(names) > 3 else None
        save_rstd_name = names[4] if len(names) > 4 else None

        G = self.num_groups
        eps = self.eps
        in_layout = self.inputs[0].get_layout()
        N, C, H, W = [int(s) for s in in_layout.size]
        channels_per_group = C // G
        spatial_size = H * W
        group_size = channels_per_group * spatial_size
        num_rows = N * G

        this_dir = os.path.dirname(os.path.abspath(__file__))
        shader_path = os.path.join(
            this_dir, "..", "..", "..", "..", "..",
            "shaders", "normalization", "group_norm.slang",
        )
        with open(shader_path) as f:
            src = f.read()

        pc_bytes = struct.pack("5If", G, group_size, num_rows,
                               channels_per_group, spatial_size, float(eps))
        pc_values = list(struct.unpack(f"{len(pc_bytes) // 4}I", pc_bytes))
        cache_key = f"group_norm_fused_{G}_{channels_per_group}_{spatial_size}_f32_m17"

        buffer_names = [inp_name, weight_name, bias_name, out_name]
        if save_mean_name:
            buffer_names.append(save_mean_name)
        else:
            buffer_names.append("_gn_fwd_dummy_mean")
        if save_rstd_name:
            buffer_names.append(save_rstd_name)
        else:
            buffer_names.append("_gn_fwd_dummy_rstd")

        out_layout = self.get_layout()
        output_allocations = [{
            "name": out_name,
            "shape": [int(s) for s in out_layout.size],
            "stride": [int(s) for s in out_layout.stride],
            "dtype": "float32",
        }]
        if not save_mean_name:
            output_allocations.append({
                "name": "_gn_fwd_dummy_mean", "shape": [num_rows],
                "stride": [1], "dtype": "float32",
            })
        if not save_rstd_name:
            output_allocations.append({
                "name": "_gn_fwd_dummy_rstd", "shape": [num_rows],
                "stride": [1], "dtype": "float32",
            })

        wrapper.emit_aoti_extern_dispatch(
            slang_src=src,
            cache_key=cache_key,
            buffer_names=buffer_names,
            pc_values=pc_values,
            grid_x=num_rows,
            grid_y=1,
            grid_z=1,
            num_outputs=1,
            output_allocations=output_allocations,
        )
