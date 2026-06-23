"""Module-scope Slang helper functions emitted per-kernel.

Extracted from VulkanKernel._emit_helpers so future workstreams (packed16,
2D wg-reduce) can add helpers here without touching kernel.py.
"""

from __future__ import annotations

from torch._inductor.codegen.common import IndentedBuffer


def emit_packed16_helpers(code: IndentedBuffer, dtype: str) -> None:
    """Emit pack/unpack helpers for uint32-packed f16 or bf16 buffers.

    Two half-precision elements are stored per uint32 word:
      bits[15:0]  = element at even index (lo half)
      bits[31:16] = element at odd index  (hi half)

    ``dtype`` is "f16" or "bf16".  The helper names are suffixed accordingly
    so both can coexist in the same shader (mixed-dtype pointwise is not
    currently generated, but guard against future changes).
    """
    if dtype == "f16":
        code.splice("""
            [ForceInline] float _vk_f16_to_f32(uint h) {
                uint sign = h >> 15u;
                uint exp  = (h >> 10u) & 0x1Fu;
                uint mant = h & 0x3FFu;
                if (exp == 31u) {
                    if (mant != 0u) return asfloat(0x7FC00000u);
                    return sign != 0u ? asfloat(0xFF800000u) : asfloat(0x7F800000u);
                }
                if (exp == 0u) {
                    if (mant == 0u) return sign != 0u ? -0.0f : 0.0f;
                    float f = float(mant) / 1024.0f;
                    return sign != 0u ? -(f * (1.0f / 16384.0f)) : (f * (1.0f / 16384.0f));
                }
                return asfloat((sign << 31u) | ((exp + 112u) << 23u) | (mant << 13u));
            }
            [ForceInline] uint _vk_f32_to_f16(float f) {
                uint bits = asuint(f);
                uint sign = bits >> 31u;
                int  exp  = int((bits >> 23u) & 0xFFu) - 127;
                uint mant = bits & 0x7FFFFFu;
                if (exp == 128 && mant != 0u) return (sign << 15u) | 0x7E00u;
                if (exp >= 16)  return (sign << 15u) | 0x7C00u;
                if (exp < -24)  return (sign << 15u);
                if (exp < -14) {
                    uint sh = uint(-14 - exp);
                    return (sign << 15u) | ((mant | 0x800000u) >> (sh + 13u));
                }
                uint h_exp  = uint(exp + 15);
                uint h_mant = mant >> 13u;
                uint round  = (mant >> 12u) & 1u;
                uint sticky = (mant & 0xFFFu) != 0u ? 1u : 0u;
                if (round != 0u && (sticky != 0u || (h_mant & 1u) != 0u)) h_mant++;
                if (h_mant == 0x400u) { h_mant = 0u; h_exp++; }
                return (sign << 15u) | (h_exp << 10u) | h_mant;
            }
            [ForceInline] float _vk_unpack_f16(uint word, uint lane) {
                return _vk_f16_to_f32((word >> (lane * 16u)) & 0xFFFFu);
            }
            [ForceInline] uint _vk_pack_f16(float lo, float hi) {
                return _vk_f32_to_f16(lo) | (_vk_f32_to_f16(hi) << 16u);
            }
        """)
    else:  # bf16
        code.splice("""
            [ForceInline] float _vk_bf16_to_f32(uint h) { return asfloat(h << 16u); }
            [ForceInline] uint _vk_f32_to_bf16(float f) {
                uint bits = asuint(f);
                if ((bits & 0x7FFFFFFFu) > 0x7F800000u) return (bits >> 16u) | 0x0040u;
                uint lsb    = (bits >> 16u) & 1u;
                uint round  = (bits >> 15u) & 1u;
                uint sticky = (bits & 0x7FFFu) != 0u ? 1u : 0u;
                bits += (round & (lsb | sticky)) << 15u;
                return bits >> 16u;
            }
            [ForceInline] float _vk_unpack_bf16(uint word, uint lane) {
                return _vk_bf16_to_f32((word >> (lane * 16u)) & 0xFFFFu);
            }
            [ForceInline] uint _vk_pack_bf16(float lo, float hi) {
                return _vk_f32_to_bf16(lo) | (_vk_f32_to_bf16(hi) << 16u);
            }
        """)


_emit_helpers_cache: dict[tuple, str] = {}

# Headers whose body now lives in `shaders/lib/helpers.slang` (or one of
# its sub-modules — see T2.6 split below). When the kernel's headers set
# is a subset of this catalog, `emit_helpers` ships `import helpers;`
# instead of inlining ~300 lines of bodies — slashes kernel SPV size,
# slangc parse cost, and SPIR-V cache fragmentation (P0.8). Bodies still
# live in `_emit_helpers_impl` as a fallback for kernels needing helpers
# not yet ported to the module.
#
# T2.6 (2026-05-08) split `helpers.slang` (827L) into 4 focused
# sub-modules: `dtype_pack.slang` (f16/bf16/u8/i8/i16 conversion +
# `extension uint` unpackers), `philox.slang` (Philox RNG state +
# rand/randn), `special_math.slang` (16 `extension float` math methods +
# 4 `extension float4` h-reductions), and `bucket.slang` (bucketize +
# buffer-form vec4 reductions). `helpers.slang` now `__exported import`s
# all four — kernels that `import helpers;` continue to resolve every
# name they used to, so this routing table does NOT need per-submodule
# entries. The set below stays organised by header tag, not by physical
# module location.
#
# Wave intrinsic wrappers (`wave_sum`/`wave_max`/`wave_min`/`wave_prod`)
# are public generic functions in `lib/helpers.slang` (`T : __BuiltinFloatingPointType`)
# at module scope — no inline-emit path exists for them, so simply listing
# them here routes kernels that opt into wave reductions through the
# precompiled module without any `_emit_helpers_impl` change.
# P3.8 Phase 1 — quick wins already ported to helpers.slang in P3.5,
# temporarily removed from HELPERS_MODULE_HEADERS until helpers.slang
# was rebuilt. Re-added now that the rebuild is complete.
# P3.8 Phase 2 — easy standalone math functions ported to helpers.slang
# (ndtri, i0, i0e, i1, i1e, erfinv, spherical_bessel_j0, zeta,
# bucketize, vec4_reduce_sum/max/min/prod).
HELPERS_MODULE_HEADERS = frozenset(
    {
        # Phase 1 — already-ported quick wins (P3.5)
        "erf",
        "log1p",
        "expm1",
        "digamma",
        "lgamma",
        # Phase 1 — already-ported (P0.8, PF.49, T5.1)
        "hypot",
        "packed16_f16",
        "packed16_bf16",
        "random",
        "wave_sum",
        "wave_max",
        "wave_min",
        "wave_prod",
        "wave_prefix_sum",
        "wave_prefix_prod",
        "wave_read_first",
        "subdtype_unpack",
        "packed16_2d_f16",
        "packed16_2d_bf16",
        # Phase 2 — easy standalone scalar math (N+1.2)
        "ndtri",
        "i0",
        "i0e",
        "i1",
        "i1e",
        "erfinv",
        "spherical_bessel_j0",
        "zeta",
        "bucketize",
        "vec4_reduce_sum",
        "vec4_reduce_max",
        "vec4_reduce_min",
        "vec4_reduce_prod",
        # Phase 3 — dependent math
        "polygamma",
        "igamma",
        # M20.4.b — wave-intrinsic fast path (single-wave reductions, no LDS)
        "wave_active_any",
        "wave_active_bit_xor",
    }
)
# The generic IWaveReduction interface + OpSum/OpProd/OpMaxReduce/OpMinReduce
# structs + wg_reduce_wave<W> live in `shaders/lib/reduction.slang`
# (precompiled .slang-module). Headers in this set skip inline emission.
# N+1.2 — All reduction helpers now live in shaders/lib/reduction.slang.
# T2.10b — Codegen now emits the new `vk_wg_reduce_*` / `vk_wg_welford`
# names; legacy `c10_vulkan_wg_reduce_*` aliases are still exported by
# `lib/reduction.slang` for any out-of-scope caller.  No inline fallback
# remains.
REDUCTION_MODULE_HEADERS = frozenset(
    {
        "wgreduce",  # sum/prod/max/min via wg_reduce_wave<W : IWaveReduction>
        "wgreduce2d",  # 2D wave reduction via wg_reduce_wave_2d<W : IWaveReduction> (T5.1)
        "wg_scan",  # inclusive scan via wg_inclusive_scan<O : IScanOp> (N.1)
        "wg_sort",  # bitonic sort via wg_bitonic_sort_float/float2 (N.2)
        "welford",  # WelfordResult<T> generic reduction via wg_welford (N.3)
        # N+1.2 — any/xor/argmax/argmin/2d_xor (were inline-only)
        "wgreduce_any",
        "wgreduce_xor",
        "wgreduce_argmax",
        "wgreduce_argmin",
        "wgreduce2d_xor",
        # P2.1/M1 — wave primitive wrappers + IScan generic scan
        "wave_broadcast",  # wave_broadcast(v) → WaveReadLaneFirst(v)
        "wg_scan_exclusive",  # wg_scan_exclusive<S : IScan>(v, lane, simd)
        "wg_scan_inclusive",  # wg_scan_inclusive<S : IScan>(v, tid, size, simd)
        "wg_bitonic_sort_wave",  # bitonic_sort_wave from reduction.slang (N.2)
    }
)



def _reset_emit_helpers_cache() -> None:
    """Test hook — clears the rendered-helpers cache."""
    _emit_helpers_cache.clear()


def emit_helpers(
    code: IndentedBuffer,
    headers,
    max_threadgroup_size: int,
    simd_group_size: int,
) -> None:
    """Emit module-scope helpers (reductions, math functions).

    Output is fully determined by `(frozenset(headers), max_threadgroup_size,
    simd_group_size)`, so the rendered string is cached on the first call per
    triple and spliced unchanged on subsequent calls. P6.3 — saves the
    per-kernel cost of re-rendering identical helper bodies.

    P3.6 — reduction helpers (wgreduce, wgreduce_any, wgreduce_xor,
    wgreduce_argmax, wgreduce_argmin) route through `import reduction;`
    (precompiled .slang-module) instead of per-kernel inline emission.
    """
    headers = set(headers)
    cache_key = (frozenset(headers), max_threadgroup_size, simd_group_size)
    cached = _emit_helpers_cache.get(cache_key)
    if cached is not None:
        code.splice(cached)
        return
    scratch = IndentedBuffer()

    # PF.21.b — atomic_add lives in `lib/atomics.slang`.
    rest = headers - {"atomic_add"}
    if "atomic_add" in headers:
        scratch.splice("import atomics;\n")

    # P3.6 — reduction helpers route through `import reduction;`
    # (precompiled .slang-module via shaders/lib/reduction.slang).
    reduction_headers = rest & REDUCTION_MODULE_HEADERS
    rest -= reduction_headers
    if reduction_headers:
        scratch.splice("import vk_reduction;\n")

    # P0.8 — helpers module headers route through `import vk_helpers;`
    # (precompiled .slang-module via shaders/lib/vk_helpers.slang).
    # Uses vk_helpers instead of helpers to avoid ambiguity with
    # vk_reduction's imported VK_SUBGROUP_SIZE from vk_helpers.
    helper_headers = rest & HELPERS_MODULE_HEADERS
    rest -= helper_headers
    if helper_headers:
        scratch.splice("import vk_helpers;\n")

    # Any remaining headers need inline emission.
    if rest:
        _emit_helpers_impl(scratch, rest, max_threadgroup_size, simd_group_size)

    rendered = scratch.getvalue()
    _emit_helpers_cache[cache_key] = rendered
    code.splice(rendered)


def _emit_helpers_impl(
    code: IndentedBuffer,
    headers,
    max_threadgroup_size: int,
    simd_group_size: int,
) -> None:
    all_known = HELPERS_MODULE_HEADERS | REDUCTION_MODULE_HEADERS | {"atomic_add"}
    unknown = set(headers) - all_known
    if unknown:
        raise AssertionError(
            f"N+1.3: Unknown helper headers reached _emit_helpers_impl: "
            f"{sorted(unknown)}.  Add them to HELPERS_MODULE_HEADERS or REDUCTION_MODULE_HEADERS."
        )
    if headers:
        raise AssertionError(
            f"N+1.3: Known headers leaked through emit_helpers routing: "
            f"{sorted(headers)}."
        )
