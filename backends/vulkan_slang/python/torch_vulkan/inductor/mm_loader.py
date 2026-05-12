"""Generic-mm source loader for the ExternKernelChoice path.

P3.4 foundation: provides callers a one-line way to fetch the Slang
generic matmul template (`shaders/lib/mm.slang`) wrapped in an entry
point that wires up storage buffers + push constants + chosen epilogue.
The full migration of `vulkan_template_caller.py` away from Jinja
toward this loader is filed as `P3.4-followup-jinja-retire`; this
module is the future caller's seam.
"""
from __future__ import annotations

import os


_MM_MODULE_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "shaders", "lib", "mm.slang"
))


def generic_mm_kernel_source(epilogue: str = "EpilogueIdentity") -> str:
    """Return a complete Slang compute kernel that imports `mm.slang`
    and dispatches `mm_tiled<Epi>(...)` over the standard
    `(A, B, C, M, N, K)` storage-buffer + push-constant layout.

    The caller can pick spec-constant tile sizes by passing them at
    pipeline-creation time (P0.6); they are NOT hardcoded in the
    rendered source so the same source produces one SPV regardless of
    chosen tile.
    """
    return f"""
import mm;

[[vk::binding(0, 0)]] StructuredBuffer<float> A;
[[vk::binding(1, 0)]] StructuredBuffer<float> B;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> C;
[[vk::push_constant]] cbuffer Push {{ uint M; uint N; uint K; }};

[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {{
    mm_tiled<{epilogue}>(A, B, C, M, N, K, gid, lid);
}}
"""


def mm_module_source() -> str:
    """Return the raw `mm.slang` module source. Useful for inlining
    when the caller doesn't want to depend on the precompiled
    `.slang-module` cache (e.g. fresh checkout, no slangc -emit-ir
    pass yet)."""
    with open(_MM_MODULE_PATH) as f:
        return f.read()
