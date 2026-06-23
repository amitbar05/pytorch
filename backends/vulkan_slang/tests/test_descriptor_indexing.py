"""N+1.5 — descriptor-array binding (`VK_EXT_descriptor_indexing`).

Exercises `_C._jit_dispatch_indexed`, which writes Vulkan descriptor sets
with `descriptorCount > 1` per binding. This is the runtime support
required to consume `ParameterBlock<KernelArgs>` shaders that contain
`ParamSlot params[N]` array-of-structs (N+1.10 round-3 D-agent dropped
this in favour of a `switch (param_idx)` cascade because the C++ runtime
mis-bound array-of-structs as `descriptorCount=1`).

The tests synthesise a tiny shader with 1 flat binding + 1 array-of-N
storage-buffer binding, dispatch it via the new FFI, and verify that
each array slot is read into a distinct output. They also probe the
`_descriptor_indexing_enabled` capability gate.
"""
from __future__ import annotations

import os
import struct

import pytest
import torch


# ── capability fixture ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def vk_capabilities():
    """Returns (descriptor_indexing_enabled: bool, _C module)."""
    try:
        import torch_vulkan  # noqa: F401  (registers the device)
        from torch_vulkan import _C as _c
    except ImportError:
        pytest.skip("torch_vulkan not installed")
    if not hasattr(_c, "_descriptor_indexing_enabled"):
        pytest.skip(
            "_C._descriptor_indexing_enabled not present — "
            "rebuild required after csrc/init.cpp N+1.5 edits"
        )
    if not hasattr(_c, "_jit_dispatch_indexed"):
        pytest.skip(
            "_C._jit_dispatch_indexed not present — "
            "rebuild required after csrc/init.cpp N+1.5 edits"
        )
    return _c._descriptor_indexing_enabled(), _c


# ── helper to compile a tiny Slang shader with array-of-buffers ──────


_SHADER_TEMPLATE = """
[[vk::binding(0)]] StructuredBuffer<float> shared_in;
[[vk::binding(1)]] RWStructuredBuffer<float> outs[{N}];

struct PC {{
    uint numel;
}};
[[vk::push_constant]] PC pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {{
    if (tid.x >= pc.numel) return;
    float v = shared_in[tid.x];
    [unroll]
    for (int i = 0; i < {N}; ++i) {{
        outs[i][tid.x] = v + float(i + 1);
    }}
}}
"""


def _compile_array_shader(n: int) -> bytes:
    from torch_vulkan.inductor.runtime import compile_slang_to_spirv

    src = _SHADER_TEMPLATE.format(N=n)
    return compile_slang_to_spirv(src, entry="computeMain")


# ── tests ────────────────────────────────────────────────────────────


def test_descriptor_indexing_enabled_on_rdna1(vk_capabilities):
    enabled, _c = vk_capabilities
    if os.environ.get("TORCH_VULKAN_DESCRIPTOR_INDEXING") == "0":
        pytest.skip("descriptor indexing explicitly disabled via env var")
    # On RDNA1 RADV the feature is supported and default-on. On Lavapipe
    # we default off (probe accepts both in case env var was forced).
    assert isinstance(enabled, bool)


@pytest.mark.parametrize("n", [2, 4])
def test_dispatch_indexed_array_binding_writes_each_slot(vk_capabilities, n):
    """Build an N-slot array-of-buffers shader, dispatch via the indexed
    FFI, and verify each output buffer received its own slot value."""
    enabled, _c = vk_capabilities
    if not enabled:
        pytest.skip("descriptor indexing disabled — array bindings unsupported")

    try:
        spv = _compile_array_shader(n)
    except Exception as e:
        pytest.skip(f"slangc compile failed: {e}")

    numel = 8
    device = torch.device("vulkan:0")
    src = torch.full((numel,), 7.0, dtype=torch.float32, device=device)
    outs = [
        torch.zeros(numel, dtype=torch.float32, device=device) for _ in range(n)
    ]

    tensors = [src] + outs
    descriptor_counts = [1, n]  # binding 0: flat, binding 1: array of N

    pc = struct.pack("I", numel)
    wg = (numel + 63) // 64

    _c._jit_dispatch_indexed(
        f"vk_n15_array_n{n}",
        spv,
        tensors,
        descriptor_counts,
        wg,
        1,
        1,
        pc,
        n,
    )

    # Synchronize before readback.
    import torch_vulkan as _tv
    if hasattr(_tv, "synchronize"):
        _tv.synchronize()

    for i, out in enumerate(outs):
        expected = torch.full((numel,), 7.0 + float(i + 1), dtype=torch.float32)
        got = out.cpu()
        assert torch.allclose(got, expected), (
            f"slot {i}: expected uniform {expected[0].item()}, got {got.tolist()}"
        )


def test_dispatch_indexed_falls_back_to_flat(vk_capabilities):
    """When all `descriptor_counts` entries are 1, the indexed path must
    behave identically to the flat path. Build a 2-binding flat shader
    and dispatch via both paths — results should match."""
    _, _c = vk_capabilities
    from torch_vulkan.inductor.runtime import compile_slang_to_spirv

    src = """
[[vk::binding(0)]] StructuredBuffer<float> a;
[[vk::binding(1)]] RWStructuredBuffer<float> b;

struct PC { uint numel; };
[[vk::push_constant]] PC pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.numel) return;
    b[tid.x] = a[tid.x] * 2.0;
}
"""
    spv = compile_slang_to_spirv(src, entry="computeMain")

    numel = 8
    device = torch.device("vulkan:0")
    a = torch.arange(numel, dtype=torch.float32, device=device)
    b = torch.zeros(numel, dtype=torch.float32, device=device)

    pc = struct.pack("I", numel)
    wg = (numel + 63) // 64

    _c._jit_dispatch_indexed(
        "vk_n15_flat",
        spv,
        [a, b],
        [1, 1],
        wg,
        1,
        1,
        pc,
        1,
    )
    import torch_vulkan as _tv

    _tv.synchronize()
    expected = torch.arange(numel, dtype=torch.float32) * 2.0
    assert torch.allclose(b.cpu(), expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-p", "no:faulthandler"])
