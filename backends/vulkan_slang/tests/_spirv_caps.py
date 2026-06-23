"""Pure-Python SPIR-V capability/type-width parser for regression tests.

Used by PF.27.a's `TestRngKernelCompilesOnRdna1` floor and any future test that
needs to assert a kernel does not pull in a capability the target hardware
lacks (RDNA1/RADV: no `Int64`, no `Int8`, etc.). Avoids a hard dep on the
SPIRV-Tools binaries (`spirv-val`, `spirv-dis`) — they are not installed on
the CI host.

Spec reference: SPIR-V 1.3 binary layout. Header is five 32-bit words:
``magic=0x07230203, version, generator, bound, schema=0``. Each instruction is
``(wordCount<<16)|opcode`` followed by ``wordCount-1`` operand words. Only the
opcodes we need are decoded; the rest are skipped by length.
"""

from __future__ import annotations

import os
import struct
from typing import Iterable, NamedTuple

# Subset of SPIR-V capability ids we care about for RDNA1 gating. See
# https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#Capability
CAP_NAMES: dict[int, str] = {
    1: "Shader",
    9: "Float16",
    10: "Float64",
    11: "Int64",
    12: "Int64Atomics",
    22: "Int16",
    39: "Int8",
    61: "GroupNonUniform",
    62: "GroupNonUniformVote",
    63: "GroupNonUniformArithmetic",
    65: "GroupNonUniformShuffle",
    4434: "StorageBuffer16BitAccess",
    4467: "StorageBuffer8BitAccess",
}

_MAGIC = 0x07230203
_OP_CAPABILITY = 17
_OP_TYPE_INT = 21
_OP_TYPE_FLOAT = 22


class SpvSummary(NamedTuple):
    capabilities: frozenset[int]
    int_widths: frozenset[int]
    float_widths: frozenset[int]


def parse_spv(path: str | os.PathLike[str]) -> SpvSummary:
    """Parse a SPIR-V binary and return its capability set + integer/float widths.

    Raises ``ValueError`` if the file is too short or the magic header is wrong.
    """
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 20:
        raise ValueError(f"{path}: too short ({len(data)} bytes) to be SPIR-V")
    magic = struct.unpack_from("<I", data, 0)[0]
    if magic != _MAGIC:
        raise ValueError(f"{path}: bad SPIR-V magic 0x{magic:08x}")

    caps: set[int] = set()
    int_widths: set[int] = set()
    float_widths: set[int] = set()
    pos = 20
    while pos < len(data):
        word = struct.unpack_from("<I", data, pos)[0]
        wcount = word >> 16
        opcode = word & 0xFFFF
        if wcount == 0:
            break
        if opcode == _OP_CAPABILITY:
            caps.add(struct.unpack_from("<I", data, pos + 4)[0])
        elif opcode == _OP_TYPE_INT:
            # OpTypeInt: result_id, width, signedness
            int_widths.add(struct.unpack_from("<I", data, pos + 8)[0])
        elif opcode == _OP_TYPE_FLOAT:
            float_widths.add(struct.unpack_from("<I", data, pos + 8)[0])
        pos += wcount * 4
    return SpvSummary(
        capabilities=frozenset(caps),
        int_widths=frozenset(int_widths),
        float_widths=frozenset(float_widths),
    )


def assert_no_capability(spv_path: str | os.PathLike[str], cap_id: int) -> None:
    """Assert ``OpCapability cap_id`` does not appear in the blob.

    Used by the PF.27.a RDNA1 floor: ``assert_no_capability(spv, 11)`` for Int64.
    """
    summary = parse_spv(spv_path)
    if cap_id in summary.capabilities:
        name = CAP_NAMES.get(cap_id, f"cap{cap_id}")
        all_caps = sorted(
            (c, CAP_NAMES.get(c, f"cap{c}")) for c in summary.capabilities
        )
        raise AssertionError(
            f"{spv_path}: declares OpCapability {name} ({cap_id}); "
            f"all capabilities = {all_caps}"
        )


def find_recent_spv(
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    require_cap: int | None = None,
    forbid_cap: int | None = None,
) -> list[str]:
    """List `.spv` files in the cache, optionally filtered by capability.

    ``cache_dir`` defaults to ``$TORCH_VULKAN_SPIRV_CACHE`` or
    ``~/.cache/torch_vulkan/spirv``. Use ``forbid_cap=11`` after a fix lands to
    prove no kernel pulls in Int64; ``require_cap`` is for the inverse check
    (e.g. asserting all fp16 kernels declare ``Float16``).
    """
    if cache_dir is None:
        cache_dir = os.environ.get(
            "TORCH_VULKAN_SPIRV_CACHE",
            os.path.join(os.path.expanduser("~"), ".cache", "torch_vulkan", "spirv"),
        )
    out: list[str] = []
    for shard in os.listdir(cache_dir):
        sub = os.path.join(cache_dir, shard)
        if not os.path.isdir(sub):
            continue
        for fname in os.listdir(sub):
            if not fname.endswith(".spv"):
                continue
            path = os.path.join(sub, fname)
            if require_cap is None and forbid_cap is None:
                out.append(path)
                continue
            try:
                summary = parse_spv(path)
            except ValueError:
                continue
            if require_cap is not None and require_cap not in summary.capabilities:
                continue
            if forbid_cap is not None and forbid_cap in summary.capabilities:
                continue
            out.append(path)
    return out


def assert_cache_free_of_capability(
    cap_id: int,
    cache_dir: str | os.PathLike[str] | None = None,
) -> None:
    """Walk the SPIR-V cache; fail if any blob declares ``cap_id``.

    Stronger floor than per-kernel asserts: catches regressions even when the
    test only exercises one of many cached kernels (e.g. a side-effect kernel
    compiled by an unrelated graph in the same process).
    """
    offenders = find_recent_spv(cache_dir, require_cap=cap_id)
    if offenders:
        name = CAP_NAMES.get(cap_id, f"cap{cap_id}")
        raise AssertionError(
            f"{len(offenders)} cached SPIR-V blob(s) declare OpCapability "
            f"{name} ({cap_id}): {offenders[:5]}"
            + (" ..." if len(offenders) > 5 else "")
        )


def kernel_hash_to_spv_path(
    hash_key: str,
    cache_dir: str | os.PathLike[str] | None = None,
) -> str:
    """Translate a runtime cache hash key into the on-disk `.spv` path.

    Layout matches ``runtime._disk_cache_read``: ``<dir>/<hash[:2]>/<hash[2:]>.spv``.
    """
    if cache_dir is None:
        cache_dir = os.environ.get(
            "TORCH_VULKAN_SPIRV_CACHE",
            os.path.join(os.path.expanduser("~"), ".cache", "torch_vulkan", "spirv"),
        )
    return os.path.join(str(cache_dir), hash_key[:2], hash_key[2:] + ".spv")


def iter_capabilities(
    spv_paths: Iterable[str | os.PathLike[str]],
) -> dict[str, SpvSummary]:
    """Bulk-summarize a list of SPIR-V files. Useful for audit dumps."""
    return {str(p): parse_spv(p) for p in spv_paths}
