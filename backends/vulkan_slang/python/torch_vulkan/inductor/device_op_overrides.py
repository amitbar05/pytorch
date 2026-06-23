"""Device-level helpers the Inductor wrapper uses to emit host code.

Vulkan is treated as a single-device backend — all helpers below collapse to
no-ops or the equivalent of `torch.vulkan.*`. Parallels MPS's own overrides
(MPS is also effectively single-device).
"""
from torch._inductor.codegen.common import (
    DeviceOpOverrides,
    register_device_op_overrides,
)


class VulkanDeviceOpOverrides(DeviceOpOverrides):
    def set_device(self, device_idx: int) -> str:
        assert device_idx == 0, "vulkan backend is single-device"
        return "pass"

    def synchronize(self) -> str:
        return "torch_vulkan._C._flush()"

    def device_guard(self, device_idx: int) -> str:
        assert device_idx == 0
        # PF.30.h.3 — Vulkan is single-device, no real device-context to enter.
        # Mirror MPS's nullcontext approach (mps_device_op_overrides.py:9).
        return "torch._ops.contextlib.nullcontext()"

    def kernel_header(self) -> str:
        return ""

    def kernel_driver(self) -> str:
        return ""

    def cpp_kernel_type(self) -> str:
        return "void*"

    def cpp_device_ptr(self) -> str:
        return "void*"

    def cpp_stream_type(self) -> str:
        return "void*"

    def cpp_device_guard(self) -> str:
        return ""

    def cpp_aoti_device_guard(self) -> str:
        return ""

    def cpp_stream_guard(self) -> str:
        return "AOTIVulkanStreamGuard"

    def cpp_aoti_stream_guard(self) -> str:
        return "AOTIVulkanStreamGuard"

    def cpp_getStreamFromExternal(self, *args, **kwargs) -> str:
        return ""


def register():
    register_device_op_overrides("vulkan", VulkanDeviceOpOverrides())
