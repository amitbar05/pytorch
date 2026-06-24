"""DR.8 / T7.5 — Vulkan C++ Wrapper GPU (AOTI analog of upstream cpp_wrapper_gpu.py).

Emits C++ code that statically links Vulkan dispatch, producing a `.so` that
runs without the ``torch_vulkan`` Python package. Subclasses the upstream
``CppWrapperCpu`` and overrides Vulkan-specific parts:

1. **Kernel call emission**: Emits C++ calls to ``torch_vulkan_aoti_dispatch``
   with pre-created handles, push-constants, and buffer bindings.

2. **Buffer management**: Emits C++ buffer allocation/release via the existing
   Vulkan allocator (``aoti_torch_empty_strided`` etc.).

3. **SPIR-V bundle**: Collects all SPIR-V binaries from the compile cache and
   embeds them as ``static const uint32_t`` arrays in the C++ source.

4. **Kernel registration**: At ``.so`` load time (static initializer), registers
   all precompiled kernels with the Vulkan runtime.

Architecture follows the same pattern as upstream ``torch/_inductor/codegen/cpp_wrapper_gpu.py``
(CUDA C++ wrapper), replacing CUDA-specific constructs with Vulkan AOTI ABI calls.
"""

from __future__ import annotations

import hashlib
import os
import struct
from typing import Any, Optional

import torch
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
from torch._inductor.virtualized import V

from . import runtime as _vk_rt

# ── Helpers ────────────────────────────────────────────────────────────


def _vulkan_cpp_device_ptr() -> str:
    """C++ type for a Vulkan device pointer (matches the allocator's interface)."""
    return "void*"


def _vulkan_cpp_stream_type() -> str:
    """C++ type for a Vulkan stream (VkQueue handle)."""
    return "void*"


def _spv_to_cpp_array(spv: bytes, name: str) -> str:
    """Convert SPIR-V binary to a C++ static const uint32_t array."""
    words = len(spv) // 4
    # Group into lines of 8 uint32_t values for readability
    lines: list[str] = []
    lines.append(f"static const uint32_t {name}_data[{words}] = {{")
    for i in range(0, words, 8):
        chunk = spv[i * 4 : min((i + 8) * 4, len(spv))]
        hex_vals = ", ".join(
            f"0x{struct.unpack('<I', chunk[j : j + 4])[0]:08x}"
            for j in range(0, len(chunk), 4)
        )
        comma = "," if i + 8 < words else ""
        lines.append(f"    {hex_vals}{comma}")
    lines.append("};")
    return "\n".join(lines)


def _spv_key_to_c_name(key: str) -> str:
    """Convert a cache key to a valid C identifier."""
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return f"_vk_spv_{h}"


# ── Vulkan C++ Wrapper Class ────────────────────────────────────────────


class VulkanCppWrapperGpu(CppWrapperCpu):
    """C++ wrapper codegen that emits Vulkan AOTI dispatch code.

    When ``V.graph.aot_mode`` is True, Inductor selects this wrapper instead
    of ``VulkanPythonWrapperCodegen``.  The emitted C++ calls the Vulkan AOTI
    C ABI (``torch_vulkan_aoti_*``) directly, producing a ``.so`` that runs
    without the Python ``torch_vulkan`` package.
    """

    def __init__(self) -> None:
        self.device = "vulkan"
        # Track collected SPIR-V blobs for embedding in the generated C++.
        self._spv_blobs: dict[str, bytes] = {}  # key → spirv bytes
        self._spv_metadata: dict[
            str, dict[str, Any]
        ] = {}  # key → {n_buffers, pc_size_bytes, ...}
        self._spv_includes_emitted = False
        super().__init__()

    def _flush_batcher_before_direct_call(self) -> None:
        """No-op in AOTI mode: the C++ wrapper doesn't use the batcher."""
        pass

    # ── Factory ─────────────────────────────────────────────────────

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper,
        partition_signatures=None,
    ):
        return VulkanCppWrapperGpu()

    # ── Header / prefix ──────────────────────────────────────────────

    def finalize_prefix(self):
        """Emit SPIR-V data arrays and kernel initialization before the
        main prefix (so they're available for all kernel calls)."""
        # First, emit SPIR-V blobs as static const arrays
        if self._spv_blobs and not self._spv_includes_emitted:
            spv_lines: list[str] = []
            spv_lines.append("// ── Embedded SPIR-V binaries (DR.8 / T7.5) ──")
            spv_lines.append("")
            for key, spv in sorted(self._spv_blobs.items()):
                c_name = _spv_key_to_c_name(key)
                spv_lines.append(_spv_to_cpp_array(spv, c_name))
                spv_lines.append("")
                meta = self._spv_metadata.get(key, {})
                spv_lines.append(
                    f"// key: {key}\n"
                    f"// n_buffers: {meta.get('n_buffers', 0)}, "
                    f"pc_size_bytes: {meta.get('pc_size_bytes', 0)}"
                )
                spv_lines.append("")
            self._spv_includes_emitted = True
            # Insert SPV blobs at the beginning of the prefix
            old_prefix = self.prefix
            self.prefix = self.prefix.__class__()
            for line in spv_lines:
                self.prefix.writeline(line)
            self.prefix.splice(old_prefix)

        # Then do the usual finalize
        super().finalize_prefix()

    # ── Input codegen ────────────────────────────────────────────────

    def codegen_inputs(self):
        """Generate C++ code for input tensor handling."""
        # Vulkan uses PrivateUse1 so we need to handle device tensors
        super().codegen_inputs()

    # ── Allocation / deallocation ────────────────────────────────────

    def make_allocation(
        self,
        name: str,
        device,
        dtype,
        shape,
        stride,
        allocation_shape=None,
        is_pinned=False,
    ) -> str:
        """Emit C++ Vulkan buffer allocation."""
        if device is not None and device.type in ("vulkan", "meta"):
            if allocation_shape is None:
                allocation_shape = shape

            # Use parent's C++-compatible helpers for dtype and array vars
            dtype_code = self.codegen_dtype(dtype)
            size_array_var = self.codegen_int_array_var(
                self.codegen_shape_tuple(shape),
                self.wrapper_call.writeline,
                known_statically=self.is_statically_known_list_of_ints(shape),
                graph=self.get_codegened_graph(),
            )
            alloc_array_var = self.codegen_int_array_var(
                self.codegen_shape_tuple(allocation_shape),
                self.wrapper_call.writeline,
                known_statically=self.is_statically_known_list_of_ints(allocation_shape),
                graph=self.get_codegened_graph(),
            )
            stride_array_var = self.codegen_int_array_var(
                self.codegen_shape_tuple(stride),
                self.wrapper_call.writeline,
                known_statically=self.is_statically_known_list_of_ints(stride),
                graph=self.get_codegened_graph(),
            )

            handle_name = f"{name}_handle"
            self.wrapper_call.writeline(f"AtenTensorHandle {handle_name};")
            args = [
                str(len(shape)),
                alloc_array_var,
                stride_array_var,
                dtype_code,
                "0",  # device_idx
                f"&{handle_name}",
            ]
            self.wrapper_call.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided_vulkan({', '.join(args)}));"
            )

            if allocation_shape != shape:
                # as_strided to reshape
                new_handle = f"{name}_as_strided_handle"
                self.wrapper_call.writeline(f"AtenTensorHandle {new_handle};")
                as_args = [
                    f"{handle_name}",
                    size_array_var,
                    stride_array_var,
                    "0",  # storage_offset
                ]
                self.wrapper_call.writeline(
                    f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_as_strided({', '.join(as_args)}, &{new_handle}));"
                )
                self.wrapper_call.writeline(
                    f"wrap_with_raii_handle_if_needed({handle_name});"
                )
                # Return RAII wrapper — matching parent contract (caller writelines it).
                return f"RAIIAtenTensorHandle {name}({new_handle});"
            # Return RAII wrapper — matching parent contract (caller writelines it).
            return f"RAIIAtenTensorHandle {name}({handle_name});"

        return super().make_allocation(
            name, device, dtype, shape, stride, allocation_shape, is_pinned
        )

    def codegen_dtype(self, dtype):
        """Use upstream AOTI dtype getters instead of cached variables.

        Upstream ``CppWrapperCpu.codegen_dtype`` returns
        ``cached_torch_dtype_<str>`` which is only emitted by the CUDA
        wrapper path.  Vulkan AOTI wrappers must call the exported
        ``aoti_torch_dtype_<str>()`` functions.
        """
        dtype_str = str(dtype).split(".")[-1]
        return f"aoti_torch_dtype_{dtype_str}()"

    def make_buffer_free(self, buffer) -> str:
        """Emit C++ buffer deallocation."""
        try:
            device = buffer.get_device()
        except (AttributeError, NotImplementedError):
            device = None
        if device is not None and device.type == "vulkan":
            name = buffer.get_name()
            # Inputs/outputs are owned by the caller — don't free them.
            if name in V.graph.graph_inputs or name in V.graph.get_output_names():
                return f"// {name} is an input/output, caller owns it"
            return f"aoti_torch_delete({name}){self.ending}"
        return super().make_buffer_free(buffer)

    def write_header(self):
        """Emit Vulkan-specific includes and forward declarations."""
        super().write_header()
        import torch_vulkan

        _pkg_dir = os.path.dirname(torch_vulkan.__file__)
        _backend_root = os.path.dirname(os.path.dirname(_pkg_dir))
        _csrc_include = os.path.join(_backend_root, "csrc")
        self.header.splice(
            f'#include "{os.path.join(_csrc_include, "backend", "AotiRuntime.h")}"\n'
            "#include <cstdint>\n"
            "#include <cstring>\n"
            "#include <vector>\n"
            "\n"
            "// Forward declarations for Vulkan AOTI shim symbols linked via extra_objects.\n"
            'extern "C" int aoti_torch_delete(void* handle);\n'
            "\n"
            "// Vulkan no-op stream guard for AOTI C++ wrapper compatibility.\n"
            "struct AOTIVulkanStreamGuard {\n"
            "    AOTIVulkanStreamGuard(void*, int32_t) {}\n"
            "};\n"
        )

    def generate_extern_kernel_out(self, node) -> None:
        """S4.0: intercept ExternKernelOut for Vulkan mm/addmm/bmm in AOTI mode.

        The parent CppWrapperCpu.generate_extern_kernel_out emits C++ ATen calls
        that bypass the Vulkan AOTI ABI. For Vulkan ExternKernelOut nodes,
        emit the SPIR-V kernel handle + dispatch via _generate_kernel_call_helper.
        Falls back to parent if no SPIR-V cache entry is known.
        """
        try:
            dev = node.get_device()
        except (AttributeError, NotImplementedError):
            dev = None

        if dev is None or dev.type != "vulkan":
            return super().generate_extern_kernel_out(node)

        kernel_name = getattr(node, "python_kernel_name", None)
        if kernel_name is not None and kernel_name in self._spv_blobs:
            input_names = [inp.codegen_reference() for inp in node.inputs]
            out_name = node.codegen_reference()
            call_args = input_names + [out_name, "1", "1", "1"]
            self._generate_kernel_call_helper(
                kernel_name, call_args, device=dev, triton=False,
            )
            return

        import warnings
        warnings.warn(
            f"S4.0: VulkanCppWrapperGpu.generate_extern_kernel_out: "
            f"no SPIR-V cache entry for '{kernel_name}' — falling back to CppWrapperCpu.",
            stacklevel=2,
        )
        super().generate_extern_kernel_out(node)

    def _generate_kernel_call_helper(
        self,
        kernel_name: str,
        call_args,
        *,
        device=None,
        triton=True,
        arg_types=None,
        raw_keys=None,
        raw_args=None,
        triton_meta=None,
        inductor_meta=None,
        graph_name: str = "",
        original_fxnode_name=None,
        current_stream_idx=None,
        **kwargs,
    ) -> None:
        """Emit a C++ Vulkan kernel dispatch call.

        For Vulkan kernels, this emits:
          1. A static initializer block that calls torch_vulkan_aoti_make_kernel
             with the precompiled SPIR-V
          2. A dispatch call to torch_vulkan_aoti_dispatch with the tensor
             handles and push constants
        """
        device = device or V.graph.get_current_device_or_throw()

        if device.type != "vulkan":
            # Non-Vulkan kernel — delegate to parent
            return super()._generate_kernel_call_helper(
                kernel_name,
                call_args,
                device=device,
                triton=triton,
                arg_types=arg_types,
                raw_keys=raw_keys,
                raw_args=raw_args,
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                graph_name=graph_name,
                original_fxnode_name=original_fxnode_name,
                current_stream_idx=current_stream_idx,
            )

        # ── Collect SPIR-V and kernel metadata from the compile cache ──
        from .runtime import _KERNEL_SPIRV_HASH, _disk_cache_read

        # The kernel_name for our Vulkan backend is a cache key for the
        # JIT dispatch.  Look up the SPIR-V from the disk cache.
        key = kernel_name
        spv = _disk_cache_read(key)

        if spv is None:
            # Try to find by the kernel's source hash
            spv_hash = _KERNEL_SPIRV_HASH.get(key, "")
            if spv_hash:
                # The SPIR-V cache in the runtime uses source hash as lookup.
                # Try reading with the hash-based key.
                spv = _disk_cache_read(spv_hash)
                if spv is None:
                    # Try the full key as-is
                    spv = _disk_cache_read(key)

        # AOTI-FIX: When SPIR-V is not yet cached (normal for AOTI —
        # the Python runtime hasn't executed), compile it now from the
        # kernel's Slang source stored during define_kernel().
        if spv is None:
            from .scheduling import get_kernel_source
            from .runtime.slangc import compile_slang_to_spirv

            src = get_kernel_source(V.graph.wrapper_code, key)
            if src is not None:
                # compile_slang_to_spirv returns the SPIR-V bytes
                # directly; no need to re-read from disk cache.
                spv = compile_slang_to_spirv(src, cache_key=key)
            else:
                pass  # fall through to the error below

        if spv is not None:
            spv_c_name = _spv_key_to_c_name(key)
            self._spv_blobs[key] = spv

            # Determine n_buffers from SPIR-V reflection, with fallback to
            # the scheduling codegen's metadata (which knows the exact counts).
            n_buffers = _vk_rt.get_reflected_binding_count(spv)
            if n_buffers is None or n_buffers == 0:
                from .scheduling import get_kernel_meta
                meta = get_kernel_meta(V.graph.wrapper_code, key)
                if meta is not None:
                    n_buffers = meta.get("n_buffers", 0)
                else:
                    n_buffers = _vk_rt._get_reflected_buffer_count_from_cache_key("") or 0

            # S4.0: use SPIR-V reflection for pc_size_bytes — authoritative for all Vulkan kernels.
            # mm/addmm/bmm get 96 (19I5f PC struct); static-specialised pointwise gets 0.
            pc_size_bytes = _vk_rt.get_reflected_pc_size(spv)
            if pc_size_bytes == 0 and inductor_meta:
                # Fallback for old slangc without -reflection-json.
                n_pc = inductor_meta.get("n_pc", 0)
                if n_pc > 0:
                    pc_size_bytes = n_pc * 4

            self._spv_metadata[key] = {
                "n_buffers": n_buffers,
                "pc_size_bytes": pc_size_bytes,
            }

            # ── Emit the kernel initialization (once per kernel) ──
            # Generate a unique handle variable name
            handle_name = f"_handle_{_spv_key_to_c_name(key)}"
            init_line = (
                f"// Kernel handle for: {key}\n"
                f"static AotiVulkanKernelHandle* {handle_name} = nullptr;\n"
                f"if ({handle_name} == nullptr) {{\n"
                f"    int rc = torch_vulkan_aoti_make_kernel(\n"
                f"        {spv_c_name}_data, {len(spv) // 4},\n"
                f'        "{key}",\n'
                f"        {n_buffers}u, {pc_size_bytes}u,\n"
                f"        &{handle_name});\n"
                f"    if (rc != 0) {{\n"
                f'        throw std::runtime_error("Failed to create Vulkan kernel: "\n'
                f"            + std::string(torch_vulkan_aoti_last_error()));\n"
                f"    }}\n"
                f"}}"
            )
            self.writeline(init_line)
        else:
            # AOTI-FIX: SPIR-V not in JIT cache. This means the kernel
            # compilation didn't produce cached SPIR-V before AOTI codegen.
            # Raise a clear error so we can investigate.
            raise RuntimeError(
                f"AOTI: SPIR-V not found in cache for kernel '{key}'. "
                f"Kernel must be compiled before AOTI codegen. "
                f"hash={_KERNEL_SPIRV_HASH.get(key, 'N/A')}"
            )

        # ── Parse call_args to separate: buffers, push-constants, wg dims ──
        # Convention (matching Python wrapper):
        #   call_args = [buf0, buf1, ..., bufN-1, pc0, pc1, ..., wg_x, wg_y, wg_z]
        # where the last 3 args are always workgroup dimensions
        # and any push-constant ints sit between buffers and wg dims.
        n_outputs = inductor_meta.get("n_outputs", 1) if inductor_meta else 1
        # S4.0: derive n_pc from stored n_buffers + call_args length so
        # mm/addmm/bmm ExternKernelOut nodes (inductor_meta=None) get the right split.
        n_args = len(call_args)
        _meta_n_buffers = self._spv_metadata.get(key, {}).get("n_buffers", 0)
        n_pc = max(0, n_args - 3 - _meta_n_buffers)

        # Separate args: last 3 are wg dims, preceding n_pc are push constants
        assert n_args >= 3, (
            f"Expected at least 3 args (wg dims), got {n_args}: {call_args}"
        )

        wg_args = call_args[-3:]  # wg_x, wg_y, wg_z
        buffer_args = call_args[: n_args - 3 - n_pc] if n_pc > 0 else call_args[:-3]
        pc_args = call_args[n_args - 3 - n_pc : n_args - 3] if n_pc > 0 else []

        # ── Emit tensor handle array ──
        # AOTI-FIX: RAIIAtenTensorHandle wraps AtenTensorHandle (=at::Tensor*).
        # Cast to the underlying handle first, then to void*.  Using &arg
        # gets the RAII wrapper address, not the tensor pointer.
        n_tensors = len(buffer_args)
        buf_list = ", ".join(
            f"reinterpret_cast<void*>(static_cast<AtenTensorHandle>({arg}))"
            for arg in buffer_args
        )
        tensor_array_line = f"void* _tensor_handles_{kernel_name}[] = {{ {buf_list} }};"
        self.writeline(tensor_array_line)

        # ── Emit push constant bytes ──
        if pc_args:
            pc_values = ", ".join(f"static_cast<uint32_t>({arg})" for arg in pc_args)
            pc_line = f"uint32_t _pc_{kernel_name}[] = {{ {pc_values} }};"
            self.writeline(pc_line)
            pc_ptr = f"_pc_{kernel_name}"
            pc_size = f"sizeof(_pc_{kernel_name})"
        else:
            pc_ptr = "nullptr"
            pc_size = "0"

        # ── Emit the dispatch call ──
        # Replace Python // (integer division) with C++ / (also integer
        # division for integers) to avoid // being interpreted as a C++ comment.
        # Also qualify bare min() calls as std::min (Python's builtin min
        # is used in the generated workgroup expressions).
        def _cpp_safe(arg: str) -> str:
            return arg.replace("//", "/").replace("min(", "std::min(")

        dispatch_line = (
            f"int _rc_{kernel_name} = torch_vulkan_aoti_dispatch(\n"
            f"    {handle_name},\n"
            f"    _tensor_handles_{kernel_name},\n"
            f"    {n_tensors}u,\n"
            f"    {pc_ptr},\n"
            f"    {pc_size},\n"
            f"    static_cast<uint32_t>({_cpp_safe(wg_args[0])}),\n"
            f"    static_cast<uint32_t>({_cpp_safe(wg_args[1])}),\n"
            f"    static_cast<uint32_t>({_cpp_safe(wg_args[2])}),\n"
            f"    {n_outputs}u);"
        )
        self.writeline(dispatch_line)
        self.writeline(
            f"if (_rc_{kernel_name} != 0) {{\n"
            f'    throw std::runtime_error("Vulkan dispatch failed: "\n'
            f"        + std::string(torch_vulkan_aoti_last_error()));\n"
            f"}}"
        )

    # ── AOTI extern-kernel dispatch ──────────────────────────────────

    def emit_aoti_extern_dispatch(
        self,
        slang_src: str,
        cache_key: str,
        buffer_names: "list[str]",
        pc_values: "list[int]",
        grid_x: int,
        grid_y: int,
        grid_z: int,
        num_outputs: int,
        output_allocations: "list[dict] | None" = None,
    ) -> None:
        """Emit C++ AOTI dispatch for an extern kernel with embedded SPIR-V.

        Compiles the Slang source to SPIR-V at codegen time, stores it in
        ``self._spv_blobs``, and emits C++ code that calls
        ``torch_vulkan_aoti_make_kernel`` + ``torch_vulkan_aoti_dispatch``
        with the precompiled SPIR-V.

        Called from extern-kernel ``codegen()`` methods when
        ``V.graph.aot_mode`` is True, replacing the Python function-call
        emission that would leak Python syntax into the C++ wrapper.
        """
        from .runtime.slangc import compile_slang_to_spirv

        # 1. Compile Slang → SPIR-V (uses disk + in-memory cache, threadsafe)
        try:
            spv = compile_slang_to_spirv(slang_src, cache_key=cache_key)
        except Exception as e:
            raise RuntimeError(
                f"AOTI extern-kernel SPIR-V compilation failed for key "
                f"'{cache_key}': {e}"
            ) from e

        spv_c_name = _spv_key_to_c_name(cache_key)
        self._spv_blobs[cache_key] = spv

        # 2. Determine n_buffers from SPIR-V reflection
        n_buffers = _vk_rt.get_reflected_binding_count(spv)
        if n_buffers is None or n_buffers == 0:
            n_buffers = len(buffer_names)

        # 3. Emit output buffer allocations if requested
        if output_allocations:
            for alloc in output_allocations:
                name = alloc["name"]
                shape = alloc["shape"]
                stride = alloc["stride"]
                dtype_str = alloc.get("dtype", "float32")
                shape_str = ", ".join(str(s) for s in shape)
                stride_str = ", ".join(str(s) for s in stride)
                self.writeline(
                    f"AtenTensorHandle {name}_handle;"
                )
                self.writeline(
                    f"int64_t _shape_{name}[] = {{ {shape_str} }};"
                )
                self.writeline(
                    f"int64_t _stride_{name}[] = {{ {stride_str} }};"
                )
                n_dims = len(shape)
                self.writeline(
                    f"AOTI_TORCH_ERROR_CODE_CHECK("
                    f"aoti_torch_empty_strided_vulkan("
                    f"{n_dims}, _shape_{name}, _stride_{name}, "
                    f"aoti_torch_dtype_{dtype_str}(), "
                    f"0, &{name}_handle));"
                )
                self.writeline(
                    f"RAIIAtenTensorHandle {name}({name}_handle);"
                )

        # 4. Emit buffer handle array for the dispatch
        n_tensors = len(buffer_names)
        buf_list = ", ".join(
            f"reinterpret_cast<void*>(static_cast<AtenTensorHandle>({arg}))"
            for arg in buffer_names
        )
        self.writeline(
            f"void* _tensor_handles_{spv_c_name}[] = {{ {buf_list} }};"
        )

        # 5. Emit push constant data
        if pc_values:
            pc_str = ", ".join(str(v) for v in pc_values)
            pc_size = len(pc_values)
            self.writeline(
                f"uint32_t _pc_{spv_c_name}[{pc_size}] = {{ {pc_str} }};"
            )
            pc_ptr = f"_pc_{spv_c_name}"
            pc_size_bytes = f"sizeof(_pc_{spv_c_name})"
        else:
            pc_ptr = "nullptr"
            pc_size_bytes = "0"

        # 6. Compute number of push constants for metadata
        n_pc = len(pc_values)
        pc_meta_bytes = n_pc * 4 if n_pc > 0 else 0

        self._spv_metadata[cache_key] = {
            "n_buffers": n_buffers,
            "pc_size_bytes": pc_meta_bytes,
        }

        # 7. Emit kernel handle + make_kernel (once)
        handle_name = f"_handle_{spv_c_name}"
        init_line = (
            f"// Extern kernel: {cache_key}\n"
            f"static AotiVulkanKernelHandle* {handle_name} = nullptr;\n"
            f"if ({handle_name} == nullptr) {{\n"
            f"    int rc = torch_vulkan_aoti_make_kernel(\n"
            f"        {spv_c_name}_data, {len(spv) // 4},\n"
            f'        "{cache_key}",\n'
            f"        {n_buffers}u, {pc_meta_bytes}u,\n"
            f"        &{handle_name});\n"
            f"    if (rc != 0) {{\n"
            f'        throw std::runtime_error("Failed to create extern Vulkan kernel: "\n'
            f"            + std::string(torch_vulkan_aoti_last_error()));\n"
            f"    }}\n"
            f"}}"
        )
        self.writeline(init_line)

        # 8. Emit dispatch call
        dispatch_line = (
            f"int _rc_{spv_c_name} = torch_vulkan_aoti_dispatch(\n"
            f"    {handle_name},\n"
            f"    _tensor_handles_{spv_c_name},\n"
            f"    {n_tensors}u,\n"
            f"    {pc_ptr},\n"
            f"    {pc_size_bytes},\n"
            f"    static_cast<uint32_t>({grid_x}),\n"
            f"    static_cast<uint32_t>({grid_y}),\n"
            f"    static_cast<uint32_t>({grid_z}),\n"
            f"    {num_outputs}u);"
        )
        self.writeline(dispatch_line)
        self.writeline(
            f"if (_rc_{spv_c_name} != 0) {{\n"
            f'    throw std::runtime_error("Extern Vulkan dispatch failed: "\n'
            f"        + std::string(torch_vulkan_aoti_last_error()));\n"
            f"}}"
        )

    # ── Stream / device management ───────────────────────────────────

    def write_get_raw_stream(self, device_idx: int, graph_name: str) -> str:
        """Return a C++ expression for the Vulkan stream (VkQueue)."""
        # Vulkan uses a single queue; we don't need per-device stream management
        # in the same way CUDA does. Return a null pointer placeholder.
        return "nullptr"

    def get_autotuning_input_name(self, idx):
        return f"_REAL_AUTOTUNE_INPUT_{idx}"


# ── SPIR-V bundle for AOTI packaging ──────────────────────────────────


def collect_aoti_spv_bundle() -> dict[str, bytes]:
    """Collect all SPIR-V binaries from the compile cache for AOTI packaging.

    Returns a dict mapping cache key → SPIR-V bytes for every kernel
    compiled during the traced session.

    Called during AOTI export to embed SPIR-V into the generated ``.so``.
    """
    from .runtime import _KERNEL_SPIRV_HASH, _disk_cache_read

    bundle: dict[str, bytes] = {}
    seen: set[str] = set()

    for key, spv_hash in _KERNEL_SPIRV_HASH.items():
        if key in seen:
            continue
        seen.add(key)

        spv = _disk_cache_read(key)
        if spv is None and spv_hash:
            spv = _disk_cache_read(spv_hash)

        if spv is not None:
            bundle[key] = spv

    return bundle


def emit_aoti_spv_header(
    bundle: dict[str, bytes],
    metadata: Optional[dict[str, dict]] = None,
) -> str:
    """Generate a C++ header fragment containing all SPIR-V blobs as
    static const arrays, plus initialization code to register them with
    the Vulkan runtime.

    Returns a C++ string suitable for inclusion in the generated ``.cpp``.

    Parameters
    ----------
    bundle:
        Mapping from cache key to SPIR-V bytes.
    metadata:
        Optional per-key overrides: ``{"n_buffers": int, "pc_size_bytes": int}``.
        When omitted, both values are derived from SPIR-V reflection.
    """
    lines: list[str] = []
    lines.append("// ── Auto-generated SPIR-V bundle for Vulkan AOTI (DR.8 / T7.5) ──")
    lines.append("// clang-format off")
    lines.append("")

    c_names: list[tuple[str, str, bytes]] = []  # (key, c_name, spv)

    for key, spv in sorted(bundle.items()):
        c_name = _spv_key_to_c_name(key)
        c_names.append((key, c_name, spv))
        lines.append(_spv_to_cpp_array(spv, c_name))
        lines.append("")

    # Emit a registration function
    lines.append("// ── Kernel initialization ──")
    lines.append("")
    lines.append("struct AotiKernelInit {")
    lines.append("    AotiVulkanKernelHandle* handle;")
    lines.append("    const char* key;")
    lines.append("    const uint32_t* spirv_data;")
    lines.append("    size_t spirv_words;")
    lines.append("    uint32_t n_buffers;")
    lines.append("    uint32_t pc_size_bytes;")
    lines.append("};")
    lines.append("")

    # Count kernels
    lines.append(f"static const size_t _vk_aoti_kernel_count = {len(c_names)};")
    lines.append("")

    lines.append("static AotiKernelInit _vk_aoti_kernels[] = {")
    for key, c_name, spv in c_names:
        key_meta = (metadata or {}).get(key, {})
        n_buf = key_meta["n_buffers"] if "n_buffers" in key_meta else (_vk_rt.get_reflected_binding_count(spv) or 0)
        pc_size_bytes = key_meta["pc_size_bytes"] if "pc_size_bytes" in key_meta else (_vk_rt.get_reflected_pc_size(spv) or 0)
        lines.append(
            f'    {{nullptr, "{key}", {c_name}_data, {len(spv) // 4}, {n_buf}u, {pc_size_bytes}u}},'
        )
    lines.append("};")
    lines.append("")

    lines.append("// ── Static initializer: register all kernels on .so load ──")
    lines.append("static int _vk_aoti_init_kernels() {")
    lines.append("    for (size_t i = 0; i < _vk_aoti_kernel_count; ++i) {")
    lines.append("        auto& k = _vk_aoti_kernels[i];")
    lines.append("        int rc = torch_vulkan_aoti_make_kernel(")
    lines.append("            k.spirv_data, k.spirv_words,")
    lines.append("            k.key, k.n_buffers, k.pc_size_bytes,")
    lines.append("            &k.handle);")
    lines.append("        if (rc != 0) return rc;")
    lines.append("    }")
    lines.append("    return 0;")
    lines.append("}")
    lines.append("")

    lines.append("static int _vk_aoti_init_result = _vk_aoti_init_kernels();")
    lines.append("")
    lines.append("// clang-format on")
    lines.append("")

    return "\n".join(lines)


# ── Kernels.bin binary format (v2, A2.7) ─────────────────────────────


def write_kernels_bin(
    path: str,
    kernels: "list[dict]",
    version: int = 2,
) -> None:
    """Write a ``kernels.bin`` file in the v2 binary format.

    Compatible with ``torch_vulkan_aoti_model_load`` in
    ``csrc/backend/AotiRuntime.cpp``.

    Parameters
    ----------
    path : str
        Output file path (e.g., ``/tmp/model/kernels.bin``).
    kernels : list[dict]
        Each dict describes one kernel:
        - ``spv`` : bytes — SPIR-V binary (required).
        - ``key`` : str — cache key (required).
        - ``n_input_buffers`` : int — number of input buffers.
        - ``n_output_buffers`` : int — number of output buffers.
        - ``wg_x``, ``wg_y``, ``wg_z`` : int — workgroup dimensions.
          If all zero, the runtime computes wg_x from the first buffer.
    version : int
        Binary format version (default 2).
    """
    import struct
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "wb") as f:
        # Header: magic (8B) + version (4B) + kernel_count (4B)
        f.write(b"vk_aoti\n")
        f.write(struct.pack("<I", version))
        f.write(struct.pack("<I", len(kernels)))

        for k in kernels:
            spv = k["spv"]
            spirv_words = len(spv) // 4

            n_input = k.get("n_input_buffers", 0)
            n_output = k.get("n_output_buffers", 1)
            n_bufs = n_input + n_output
            pc_bytes = k.get("pc_size_bytes", 0)
            wg_x = k.get("wg_x", 0)
            wg_y = k.get("wg_y", 0)
            wg_z = k.get("wg_z", 0)
            key = k["key"].encode("utf-8")

            # Entry header (9 × uint32 = 36 bytes)
            f.write(struct.pack(
                "<9I",
                spirv_words,
                n_input,
                n_output,
                n_bufs,
                pc_bytes,
                wg_x,
                wg_y,
                wg_z,
                len(key),
            ))
            # Cache key
            f.write(key)
            # SPIR-V data (4 bytes per word, little-endian)
            f.write(spv)
