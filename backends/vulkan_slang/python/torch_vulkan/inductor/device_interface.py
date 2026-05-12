"""Dynamo DeviceInterface for the vulkan backend.

Inductor queries a `DeviceInterface` via `get_interface_for_device` during
reduction lowering (`Reduction.num_splits` → `DeviceProperties.create`).
"""
from __future__ import annotations

from collections import namedtuple
from typing import Any

import torch
from torch._dynamo.device_interface import (
    DeviceInterface,
    register_interface_for_device,
)


_CACHED_PROPS = None


def _get_device_properties():
    global _CACHED_PROPS
    if _CACHED_PROPS is not None:
        return _CACHED_PROPS
    try:
        import torch_vulkan._C as _c
        caps = _c._get_device_capabilities(0)
    except Exception:
        caps = {}

    VulkanProps = namedtuple(
        "VulkanProperties",
        [
            "multi_processor_count",
            "max_workgroup_size",
            "subgroup_size",
            "max_shared_memory_size",
            "max_storage_buffers",
            "float16",
            "float64",
            "int8",
            "cooperative_matrix",
        ],
    )
    _CACHED_PROPS = VulkanProps(
        multi_processor_count=caps.get("max_workgroup_size", 256) // 64,
        max_workgroup_size=caps.get("max_workgroup_size", 256),
        subgroup_size=caps.get("subgroup_size", 64),
        max_shared_memory_size=caps.get("max_compute_shared_memory", 32768),
        max_storage_buffers=caps.get("max_storage_buffers", 64),
        float16=caps.get("float16", False),
        float64=caps.get("float64", False),
        int8=caps.get("int8", False),
        cooperative_matrix=caps.get("cooperative_matrix", False),
    )
    return _CACHED_PROPS


class _VulkanDeviceCtx:
    """Trivial context manager for `with VulkanInterface.device(idx):`.

    Single-device Vulkan backend: no real device-switch needed. Upstream's
    autotune harness (`GPUDeviceBenchmarkMixin.do_bench`) wraps benchmark
    calls in this context manager (PF.30.h).
    """

    def __init__(self, idx: int = 0) -> None:
        self.idx = int(idx) if idx is not None else 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class VulkanInterface(DeviceInterface):
    @staticmethod
    def is_bf16_supported(including_emulation: bool = False) -> bool:
        return True

    @classmethod
    def is_dtype_supported(
        cls, dtype: torch.dtype, including_emulation: bool = False
    ) -> bool:
        if dtype in (torch.float64, torch.complex128):
            return False
        return True

    @staticmethod
    def is_available() -> bool:
        try:
            import torch_vulkan
            return torch_vulkan.is_available()
        except Exception:
            return False

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def set_device(device: torch.types.Device) -> None:
        # Single-device backend; nothing to do.
        return None

    @staticmethod
    def exchange_device(device: int) -> int:
        # Returns previous device; we only have one.
        return 0

    @staticmethod
    def maybe_exchange_device(device: int) -> int:
        return 0

    @staticmethod
    def device(device: torch.types.Device = None) -> "_VulkanDeviceCtx":
        idx = 0
        if isinstance(device, int):
            idx = device
        elif isinstance(device, torch.device) and device.index is not None:
            idx = device.index
        return _VulkanDeviceCtx(idx)

    @staticmethod
    def device_count() -> int:
        try:
            import torch_vulkan._C as _c
            return int(_c._device_count())
        except Exception:
            return 1

    @staticmethod
    def get_compute_capability(device: torch.types.Device = None) -> str:
        return ""

    @staticmethod
    def synchronize(device: torch.types.Device = None) -> None:
        try:
            import torch_vulkan._C as _c
            _c._flush()
        except Exception:
            pass

    @staticmethod
    def get_device_properties(device: torch.types.Device = None) -> Any:
        return _get_device_properties()

    class Worker:
        @staticmethod
        def get_device_properties(device: torch.types.Device = None) -> Any:
            return _get_device_properties()

        @staticmethod
        def current_device() -> int:
            return 0


def register() -> None:
    register_interface_for_device("vulkan", VulkanInterface)
    register_interface_for_device("vulkan:0", VulkanInterface)
