# Extension Cookbook — Adding a Fused Vulkan Op

Single-page guide for landing a new fused Vulkan op without editing five
files. Maps to roadmap **P5.5**.

## When to add a fused op

If the eager backend already has a hand-fused shader for a pattern that
Inductor would otherwise emit as multiple dispatches (e.g. `silu(gate) *
up`, `add + rmsnorm`), wrap it as an extension. The decision tree:

| Situation | Use |
|-----------|-----|
| Pattern is `op(arg)` with a tighter Slang | `register_lowering(aten.op)` |
| New Slang shader, no existing aten op | `register_template(name, src, ...)` |
| FX rewrite (matching a multi-node pattern) | hand-write an FX pass + use `register_template` for the fused dispatch target |

## `register_lowering` — overriding an aten op

```python
import torch
from torch_vulkan.inductor.extensions import register_lowering
from torch._inductor import lowering as L

aten = torch.ops.aten

@register_lowering(aten.silu)
def _vulkan_silu(x):
    sig = L.lowerings[aten.sigmoid](x)
    return L.lowerings[aten.mul.Tensor](x, sig)
```

The wrapper auto-returns `NotImplemented` when `x` is not a vulkan IR
node — non-vulkan backends sharing the same Inductor session are
unaffected. Add a regression in `tests/test_inductor_regression.py`
asserting the dispatch-count or correctness target.

## `register_template` — adding a fused Slang shader

```python
from torch_vulkan.inductor.extensions import register_template, prewarm_template

_SRC = """
[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(
    uniform StructuredBuffer<float> in0,
    uniform StructuredBuffer<float> in1,
    uniform RWStructuredBuffer<float> out0,
    uniform PushConstants { uint numel; },
    uint3 tid : SV_DispatchThreadID
) {
    if (tid.x >= numel) return;
    out0[tid.x] = in0[tid.x] * in1[tid.x] * 0.5;
}
"""

dispatch_my_fused = register_template(
    name="my_fused_mul_half",
    slang_src=_SRC,
    n_buffers=3,           # 2 inputs + 1 output
    n_pc=1,                # 1 push constant (numel)
    pc_size_bytes=4,
    n_outputs=1,
)

# Optional: pre-compile at import time so first dispatch isn't slow.
prewarm_template(dispatch_my_fused)
```

To use the dispatcher inside a lowering or FX pass:

```python
def _my_fused_lowering(a, b):
    out = empty_strided_vulkan(a.shape, a.stride(), a.dtype)
    wg_x = (a.numel() + 63) // 64
    dispatch_my_fused(a, b, out, a.numel(), wg=(wg_x, 1, 1))
    return out
```

## Wiring into the backend register flow

Put your extension module in
`python/torch_vulkan/inductor/extensions/<your_op>.py` and call its
`register()` from `inductor/__init__.py:register()` after the existing
`_lowerings.register()` line.

## Mandatory regression test

Every new fused op lands with a regression in
`tests/test_inductor_regression.py`. Two minimum assertions:

1. **Dispatch count**: `_get_dispatch_count()` after a single warm call
   matches your target (typically `==1` for a pattern that previously
   needed N dispatches).
2. **Correctness**: `torch.testing.assert_close(compiled_out.cpu(),
   eager_out, rtol=…, atol=…)`.

See `TestRMSNormForward` in the regression file as a worked example.

## Debugging

- `TORCH_LOGS=output_code python my_workload.py` — prints the
  Inductor-generated Slang.
- `TORCH_VULKAN_TRACE=1` — prints every JIT dispatch (key, tensors, WG, pc).
- `slangc` failure → check the source written via TORCH_VULKAN_TRACE; the
  most common bug is a `groupshared` declared inside `computeMain`
  instead of at module scope.
