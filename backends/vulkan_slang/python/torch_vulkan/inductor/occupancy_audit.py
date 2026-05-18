"""P5.5 — VGPR / occupancy regression gate.

Vendor-neutral proxy for register pressure and occupancy: walk the
SPIR-V binary slangc emits for each canonical generic-family entry
point and lock four counts.

  * `workgroup_size`         — product of the three `OpExecutionMode
                               LocalSize` operands. Larger groups → fewer
                               concurrent waves per CU; smaller groups →
                               under-utilization.
  * `groupshared_vars`       — number of `OpVariable` with `StorageClass
                               Workgroup`. Direct proxy for groupshared
                               LDS pressure.
  * `function_local_vars`    — number of `OpVariable` with `StorageClass
                               Function`. Strong proxy for VGPR pressure
                               on AMD RDNA1 (each function-scope local is
                               a register-allocated SSA value before the
                               driver's lowering pass).
  * `instruction_count`      — total SPIR-V instruction count after the
                               header. Catches codegen bloat that doesn't
                               show up as new locals (e.g. inlined helpers
                               that explode the basic block count).

Each commit can lower the LOCKED_OCCUPANCY ceilings but never raise
them. To intentionally raise (e.g. a temporary increase justified by
a feature add), the human flips the constant in this file with a
comment.

Why proxies and not actual VGPR counts: vendor toolchains (RGA on AMD,
nv-prof on NVIDIA) aren't available in CI on the agent's RDNA1 dev box.
SPIR-V-level proxies give a deterministic, cross-driver upper bound — a
shader whose proxy counts shrink will allocate fewer registers; one
whose counts grow gets re-reviewed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

_SPIRV_MAGIC = 0x07230203

# SPIR-V opcodes we care about.
_OP_EXECUTION_MODE = 16
_OP_VARIABLE = 59

# ExecutionMode operand for LocalSize.
_EXEC_MODE_LOCAL_SIZE = 17

# StorageClass values.
_STORAGE_FUNCTION = 7
_STORAGE_WORKGROUP = 4


@dataclass(frozen=True)
class OccupancyMetrics:
    workgroup_size: int
    groupshared_vars: int
    function_local_vars: int
    instruction_count: int


def parse_spirv_metrics(spv: bytes) -> OccupancyMetrics:
    """Walk a SPIR-V binary and extract the four occupancy proxies.

    The format is documented in the SPIR-V spec §2.3 (Physical Layout).
    Header is 5 words; each instruction is a leading word with
    `WordCount << 16 | Opcode` followed by `WordCount-1` operand words.
    """
    if len(spv) < 20:
        raise ValueError("SPIR-V binary too short to parse")

    magic = struct.unpack_from("<I", spv, 0)[0]
    if magic != _SPIRV_MAGIC:
        raise ValueError(
            f"not a SPIR-V binary: magic={magic:#x} expected {_SPIRV_MAGIC:#x}"
        )

    workgroup_size = 0
    groupshared_vars = 0
    function_local_vars = 0
    instruction_count = 0

    offset = 20  # past the 5-word header
    n_words = len(spv) // 4

    while offset < len(spv):
        head = struct.unpack_from("<I", spv, offset)[0]
        word_count = (head >> 16) & 0xFFFF
        opcode = head & 0xFFFF
        if word_count == 0:
            # Malformed: word count of 0 would loop forever.
            break
        instruction_count += 1

        if opcode == _OP_EXECUTION_MODE and word_count >= 3:
            # Layout: OpExecutionMode | EntryPoint | Mode | (literals…)
            mode = struct.unpack_from("<I", spv, offset + 8)[0]
            if mode == _EXEC_MODE_LOCAL_SIZE and word_count >= 6:
                lx = struct.unpack_from("<I", spv, offset + 12)[0]
                ly = struct.unpack_from("<I", spv, offset + 16)[0]
                lz = struct.unpack_from("<I", spv, offset + 20)[0]
                workgroup_size = lx * ly * lz
        elif opcode == _OP_VARIABLE and word_count >= 4:
            # Layout: OpVariable | ResultType | ResultId | StorageClass | (init?)
            sc = struct.unpack_from("<I", spv, offset + 12)[0]
            if sc == _STORAGE_FUNCTION:
                function_local_vars += 1
            elif sc == _STORAGE_WORKGROUP:
                groupshared_vars += 1

        offset += word_count * 4
        if offset > n_words * 4:
            break

    return OccupancyMetrics(
        workgroup_size=workgroup_size,
        groupshared_vars=groupshared_vars,
        function_local_vars=function_local_vars,
        instruction_count=instruction_count,
    )


# Canonical entry-point shaders, one per generic-family library. These
# are the *minimum* exercise of each library — if their proxy counts
# grow, every kernel that imports the same library will too.

_POINTWISE_ENTRY = """
import pointwise;
[[vk::binding(0, 0)]] StructuredBuffer<float> in_x;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> out_y;
[[vk::push_constant]] cbuffer Push { uint n; };
[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    pointwise_unary_apply<OpReLU>(in_x, out_y, tid.x, n);
}
"""

_REDUCTION_ENTRY = """
import reduction;
[[vk::binding(0, 0)]] StructuredBuffer<float> in_x;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> out_y;
[[vk::push_constant]] cbuffer Push { uint n; };
[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID) {
    float v = (tid.x < n) ? in_x[tid.x] : 0.0;
    float r = wg_reduce<OpSum>(v, lid.x, 64, VK_SUBGROUP_SIZE);
    if (lid.x == 0) out_y[0] = r;
}
"""

_MM_ENTRY = """
import mm;
[[vk::binding(0, 0)]] StructuredBuffer<float> in_a;
[[vk::binding(1, 0)]] StructuredBuffer<float> in_b;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> out_y;
[[vk::push_constant]] cbuffer Push { uint M; uint N; uint K; };
[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    mm_tiled<EpilogueIdentity>(in_a, in_b, out_y, M, N, K, gid, lid);
}
"""

_NORM_ENTRY = """
import norm;
[[vk::binding(0, 0)]] StructuredBuffer<float> in_x;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> out_y;
[[vk::push_constant]] cbuffer Push { uint row_size; float eps; };
[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    rms_norm_row<AffineNone>(
        in_x, in_x, in_x, out_y,
        gid.x, row_size, eps, lid.x, 64u, VK_SUBGROUP_SIZE);
}
"""

_LOSSES_ENTRY = """
import losses;
[[vk::binding(0, 0)]] StructuredBuffer<float> in_pred;
[[vk::binding(1, 0)]] StructuredBuffer<float> in_target;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> out_y;
[[vk::push_constant]] cbuffer Push { uint n; };
[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= n) return;
    out_y[tid.x] = mse_elem(in_pred[tid.x], in_target[tid.x]);
}
"""


_REGISTERED_ENTRIES: dict[str, str] = {
    "pointwise_unary_relu": _POINTWISE_ENTRY,
    "reduction_sum": _REDUCTION_ENTRY,
    "mm_identity": _MM_ENTRY,
    "norm_rms_no_affine": _NORM_ENTRY,
    "losses_mse": _LOSSES_ENTRY,
}


def measure_registered() -> dict[str, OccupancyMetrics]:
    """Compile every registered entry to SPIR-V and return its metrics.

    Skips silently if slangc is unavailable (CI without the compiler) so
    the audit's existence doesn't block environments that can't run it.
    """
    from .runtime import _slangc_available, compile_slang_to_spirv

    if not _slangc_available():
        return {}

    out: dict[str, OccupancyMetrics] = {}
    for name, src in _REGISTERED_ENTRIES.items():
        spv = compile_slang_to_spirv(
            src,
            entry="computeMain",
            cache_key=f"p55_occupancy_audit_{name}_v2",
        )
        out[name] = parse_spirv_metrics(spv)
    return out


# Locked occupancy ceilings — measured at lock time on RDNA1 with
# slang-2026.5.2-linux-x86_64 on 2026-04-28. Bump down (never up) as
# Slang upgrades shrink codegen, or as kernel sources tighten. Bumping
# up requires a one-line comment justifying the regression.
LOCKED_OCCUPANCY: dict[str, dict[str, int]] = {
    "pointwise_unary_relu": {
        "workgroup_size": 64,
        "groupshared_vars": 0,
        "function_local_vars": 0,
        "instruction_count": 75,
    },
    "reduction_sum": {
        "workgroup_size": 64,
        "groupshared_vars": 1,
        "function_local_vars": 3,
        "instruction_count": 188,
    },
    "mm_identity": {
        "workgroup_size": 256,
        "groupshared_vars": 2,
        "function_local_vars": 5,
        "instruction_count": 285,
    },
    "norm_rms_no_affine": {
        "workgroup_size": 64,
        "groupshared_vars": 1,
        "function_local_vars": 4,
        "instruction_count": 224,
    },
    "losses_mse": {
        "workgroup_size": 64,
        "groupshared_vars": 0,
        "function_local_vars": 0,
        "instruction_count": 91,
    },
}


def assert_under_locked_ceilings() -> None:
    """Compare every registered entry's metrics to the locked ceilings.

    Raises if any metric grew. workgroup_size is locked exactly (a
    different launch shape is a deliberate change, not noise). Skips
    silently if slangc is unavailable (CI without the compiler).
    """
    measured = measure_registered()
    if not measured:
        return
    failures: list[str] = []
    for name, m in measured.items():
        ceiling = LOCKED_OCCUPANCY.get(name)
        if ceiling is None:
            failures.append(f"{name}: no locked ceiling registered")
            continue
        for metric, val in (
            ("workgroup_size", m.workgroup_size),
            ("groupshared_vars", m.groupshared_vars),
            ("function_local_vars", m.function_local_vars),
            ("instruction_count", m.instruction_count),
        ):
            if metric == "workgroup_size":
                if val != ceiling[metric]:
                    failures.append(
                        f"{name}.{metric}: {val} != locked {ceiling[metric]}"
                    )
            else:
                if val > ceiling[metric]:
                    failures.append(
                        f"{name}.{metric}: {val} > locked {ceiling[metric]} "
                        f"— Slang upgrade or kernel change bloated SPIR-V "
                        f"codegen. See P5.5 in docs/10-inductor-backend.md."
                    )
    if failures:
        raise AssertionError("\n".join(failures))
