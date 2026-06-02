# AOTI Container Vulkan Support

## Status: Container Creation VERIFIED ✅ (2026-06-02)

### Verified Pipeline
```
export → aot_compile → load .so → create container → delete container
```
All steps pass. Container creation handle: valid (non-null, non-zero).

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
✅ Applied and working.

### 2. vulkan.h — AOTI shim declarations
File: `<venv>/.../torch/include/torch/csrc/inductor/aoti_include/vulkan.h`

Full stubs + mm_out bridge to `aoti_torch_cpu_mm_out`.
✅ Deployed and working.

### 3. getDeviceFromPtr — VulkanHooksInterface override
File: `csrc/backend/Hooks.h` (commit `0ff39e4`)

```cpp
c10::Device getDeviceFromPtr(void* /*data*/) const override {
    return c10::Device(c10::DeviceType::PrivateUse1, 0);
}
```
✅ Compiled into `_C.so` (verified with `nm -D`).
Container creation now passes `load_constants()` which calls `aoti_torch_create_tensor_from_blob` → `getDeviceFromPtr`.

## Remaining Work
- **Inference dispatch**: Container creates but Run/GetInput/SetOutput APIs need wiring
- **Better mm shim**: Currently delegates to `aoti_torch_cpu_mm_out`; needs real Vulkan dispatch

## Build Note
Incremental builds may miss header changes. To force recompile after Hooks.h changes:
```bash
touch csrc/backend/Hooks.cpp csrc/init.cpp
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MAX_JOBS=3 python setup.py build_ext --inplace
```
