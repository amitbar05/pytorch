# AOTI Container Vulkan Support

## Patches Required (in venv torch install)

### 1. model_base.h — parse_device_str
File: `<venv>/lib/python3.12/site-packages/torch/include/torch/csrc/inductor/aoti_runtime/model_base.h`

```diff
-  std::regex re("(cpu|cuda|xpu|mps)(:([0-9]+))?");
+  std::regex re("(cpu|cuda|xpu|mps|vulkan)(:([0-9]+))?");

+  } else if (sm[1].str() == "vulkan") {
+    device_type = aoti_torch_device_type_privateuse1();
+    device_idx = 0;
```

### 2. vulkan.h — AOTI shim declarations
File: `<venv>/.../torch/include/torch/csrc/inductor/aoti_include/vulkan.h`

Full stubs + mm_out bridge to `aoti_torch_cpu_mm_out`.
(Deployed in this session.)

## Remaining Blocker

**PrivateUse1 constant loading**: `aoti_torch_create_tensor_from_blob_v2` calls
`getDeviceFromPtr()` on CPU-allocated memory, expecting a device pointer.
Our PrivateUse1 hooks don't handle CPU→Vulkan transfers.

**Options:**
1. Implement `getDeviceFromPtr` to return Vulkan device for any pointer
2. Pre-allocate constants on Vulkan before container creation
3. Use custom model loader that bypasses AOTI container
