# Track 0.5 — RNG Determinism Under Compile

## Root Cause

Inductor's `fuse_seed_creation_pass` (in `torch/_inductor/fx_passes/replace_random.py`)
creates a seed buffer input for the compiled graph. The kernel codegen reads from
this buffer via `OpOverrides.load_seed(name, offset)` (common.py:1049) which calls
`ops.load(name, sympy.Integer(offset))`.

On the Triton backend, the runtime wrapper manages `PhiloxState` — it reads the
generator's seed/offset, creates a seed buffer, passes it as a kernel arg, and
advances the generator state after dispatch.

On the Vulkan backend:
1. `VulkanOverrides` inherits `OpOverrides.load_seed` → reads from seed buffer ✓
2. `VulkanOverrides.rand(seed, offset)` emits `_vk_philox_rand((uint)(offset), (uint)(seed))` ✓
3. Slang helpers (`_vk_philox_round`, `_vk_philox_rand`, `_vk_philox_randn`) exist ✓
4. **MISSING**: wrapper codegen does NOT create/advance PhiloxState
5. **MISSING**: seed buffer content is constant from trace time, not dynamic from generator

Result: same seed on every dispatch → identical dropout masks → "deterministic" but broken.

## Implementation Plan

### Step 1: Implement `load_seed` override in `VulkanOverrides`

```python
# overrides.py
@staticmethod
def load_seed(name, offset):
    # On Vulkan, int64 seed is bound as uint2 buffer.
    # Read low 32 bits of the seed word at the given offset.
    var = V.kernel.args.input(name)
    V.kernel.headers.add("random")
    return f"((uint)({var}[{offset}].x))"
```

This reads the seed from the kernel's seed buffer (int64 stored as uint2 pairs).
The low 32 bits contain the seed value passed to Philox.

### Step 2: Wire `seed_offset` in `VulkanKernel.call_kernel`

The wrapper must pass the seed/offset as kernel arguments:

```python
# The fuse_seed_creation_pass creates seed_offset sizevars.
# kernel.py:call_kernel already iterates range trees and sizevars,
# so seed_offset values are already in the args list.
# But they're lifted from the seed buffer, not computed dynamically.
```

### Step 3: Wrapper-level PhiloxState management

```python
# wrapper.py — in make_allocation or generate_kernel_call:
# 1. Before kernel dispatch, read generator seed/offset
# 2. Create seed buffer with correct content  
# 3. Pass seed buffer as kernel arg
# 4. After dispatch, advance generator offset
```

The standard pattern is:
```python
from torch._inductor.runtime.runtime_utils import PhiloxState
# In the wrapper preamble:
seed_state = PhiloxState(generator)
# In the kernel call:
seed, offset = seed_state.get_seed_offset()
seed_buffer = torch.tensor([seed, offset], dtype=torch.int64, device="vulkan")
# Dispatch kernel with seed_buffer as arg
seed_state.advance(offset_increment)
```

### Step 4: Regression test

```python
def test_dropout_compiled_reproduces_with_same_seed(self):
    """Same seed → identical dropout output under compile."""
    @torch.compile(backend="inductor")
    def fn(x):
        return torch.nn.functional.dropout(x, p=0.5, training=True)
    
    x = torch.ones(16, 16, device="vulkan:0")
    torch.manual_seed(42)
    out1 = fn(x)
    torch.manual_seed(42)
    out2 = fn(x)
    assert torch.equal(out1, out2), "GAP 1.3: same seed produced different masks"
```

## Files to Modify

| File | Change |
|------|--------|
| `overrides.py` | Add `load_seed` static method |
| `kernel/main.py` | Verify seed buffers bound as uint2 |
| `wrapper.py` | Add PhiloxState management in preamble + kernel call |
| `runtime.py` | No change needed (Philox helpers already in slang_helpers.py) |
| `tests/test_inductor_regression.py` | Add `test_dropout_compiled_reproduces_with_same_seed` |

## Dependencies

- C++ `VulkanGeneratorImpl` already exists (`csrc/backend/Generator.cpp`)
- `torch.manual_seed()` propagates via `HooksInterface.getNewGenerator()`
- Philox shader helpers already compiled (`shaders/lib/`)
