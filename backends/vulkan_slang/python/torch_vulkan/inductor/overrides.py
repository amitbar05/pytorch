"""Inductor op overrides → Slang snippets, plus dtype helpers."""

from __future__ import annotations

import math

import torch
from torch._inductor.codegen.common import OpOverrides
from torch._inductor.utils import sympy_index_symbol
from torch._inductor.virtualized import V


class _SlangExpr(str):
    """A Slang source string that also carries a torch dtype.

    Inductor's `DtypePropagationOpsHandler._default` reads `value.dtype` for
    `masked` ops under the triton/cuda backend code path. Our backend goes
    through that path (`get_current_backend()` returns `cuda_backend` for
    non-cpu/mps/xpu devices), so the string we return must satisfy the
    `.dtype` attribute access. We attach `dtype` (and a no-op `shape`) so the
    handler doesn't crash with `AttributeError: 'str' object has no attribute
    'dtype'`.
    """

    __slots__ = ("dtype", "shape")

    def __new__(cls, value: str, dtype=None, shape=None):
        s = super().__new__(cls, value)
        s.dtype = dtype
        s.shape = shape
        return s


def _infer_dtype(value) -> torch.dtype | None:
    if hasattr(value, "dtype") and value.dtype is not None:
        return value.dtype
    return None


def _masked_dtype(body, other) -> torch.dtype:
    """Best-effort dtype for the value of an `ops.masked(mask, body, other)`.

    The propagation handler asserts the output dtype is non-None for backends
    in the triton/cpp tier (which our backend gets routed into). Inputs in
    our codegen are typically Slang source strings that don't carry the
    dtype, so we fall back to ``float32`` when neither operand exposes one.
    """
    return _infer_dtype(body) or _infer_dtype(other) or torch.float32


# M18.4 — DTYPE_TO_SLANG: Slang element type per PyTorch dtype.
#
# CRITICAL ALLOC CONTRACT: the Slang element type's storage width MUST
# match PyTorch's ``element_size()`` for that dtype, OR the buffer pool
# must round the allocation up to match. Otherwise
# ``RWStructuredBuffer<T>`` writes overflow the buffer for any thread
# index past ``alloc_nbytes / sizeof(T)``. The M17.8.d.3 root cause
# was ``bool`` mapped to ``"uint"`` (4B/slot) while PyTorch allocates
# 1B/bool — tail writes overflowed and downstream reads returned the
# zero-padded heap.
#
# History:
#   * M18.4 (2026-05-17): stopgap — narrow integers (bool, int8, uint8,
#     int16) declared as 32-bit slots; the pointwise load/store mixins
#     sign-extend via bit-twiddles. Allocation was still misaligned and
#     ``TestDtypeMatrix`` xfail-strict'd int8/uint8/int16 tail-corruption.
#     Two follow-up paths filed: (a) Slang-side native widths gated on
#     Vulkan device features; (b) Inductor-side 4-byte alloc round-up.
#   * M18.4-followup-C (2026-05-18): path (a) LANDED for {bool, int8,
#     uint8, int16, uint16}. Vulkan device features ``shaderInt8`` +
#     ``shaderInt16`` + ``storageBuffer8BitAccess`` +
#     ``storageBuffer16BitAccess`` + ``uniformAndStorageBuffer{8,16}BitAccess``
#     are now enabled in ``csrc/vulkan/Context.cpp`` (gated on device
#     report — defensively off on Lavapipe / older drivers). The Slang
#     element types match PyTorch's native 1B/2B alloc, so the
#     M17.8.d.3 tail-corruption bug class is CLOSED for the integer
#     half. The matching sign-extend workarounds have been dropped
#     from ``pointwise_load_mixin.py``.
#
# Still open follow-ups:
#   * ``bfloat16``: bound as 32-bit ``uint`` slot (2B-vs-4B mismatch).
#     Works today via the ``packed16_bf16`` load path on eligible
#     fusions; long-term should bind as native ``bfloat16_t``. slangc
#     2026.7.1 does NOT have a ``bfloat16_t`` type
#     (``agent_space/m18_4_slang_dtype_probe_bfloat16.slang`` produces
#     ``E30015``). Tracked as ``M18.4-followup-bfloat16`` — gated on a
#     slangc upgrade or explicit ``uint16_t`` reinterpret in the
#     load/store paths.
#   * ``complex32``: bound as ``float2`` (8B/slot) but PyTorch allocates
#     4B/elem. Same alloc-mismatch bug class. Rarely exercised; tracked
#     as ``M18.4-followup-complex32``.
DTYPE_TO_SLANG: dict[torch.dtype, str] = {
    # M18.4-followup-C: narrow integers now bind at their native width.
    # ``RWStructuredBuffer<{int8_t,uint8_t,int16_t,uint16_t}>`` is legal
    # SPIR-V once ``shaderInt{8,16}`` + 8/16-bit storage features are
    # enabled (see ``csrc/vulkan/Context.cpp``). Sign extension is
    # implicit in the load-side ``(float)(v[i])`` cast — no bit-twiddle
    # workaround needed.
    torch.bool: "uint8_t",
    torch.int8: "int8_t",
    torch.uint8: "uint8_t",
    torch.int16: "int16_t",
    torch.uint16: "uint16_t",
    # bfloat16 still binds as a 32-bit uint slot — see
    # ``M18.4-followup-bfloat16`` above. The ``packed16_bf16`` load path
    # handles the 2B-vs-4B mismatch on eligible fusion shapes.
    torch.bfloat16: "uint",
    # 32-bit native types — width matches PyTorch's 4B alloc.
    torch.int32: "int",
    torch.uint32: "uint",
    torch.float: "float",
    torch.half: "float16_t",
    # 64-bit native types — width matches PyTorch's 8B alloc.
    torch.int64: "int64_t",
    torch.uint64: "uint64_t",
    torch.float64: "double",
    # Complex (complex32 8B-vs-4B alloc mismatch is a known gap;
    # rarely exercised, deferred to ``M18.4-followup-complex32``).
    torch.complex32: "float2",
    torch.complex64: "float2",
    torch.complex128: "double2",
}


def value_to_slang(val) -> str:
    """Render a Python scalar as a Slang literal."""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, float):
        if math.isinf(val):
            return "(1.0/0.0)" if val > 0 else "(-1.0/0.0)"
        if math.isnan(val):
            return "asfloat(0x7FC00000u)"
        return f"{val!r}f"
    return str(val)


class VulkanOverrides(OpOverrides):
    """Inductor op → Slang snippet."""

    @staticmethod
    def to_dtype(x, dtype: torch.dtype, src_dtype=None, use_compute_types=True) -> str:
        if dtype == torch.bfloat16:
            # DTYPE_TO_SLANG[bfloat16] = "uint" (2-bf16-per-slot packing) so a
            # naive ((uint)(x)) would TRUNCATE the float to an integer — wrong.
            # All Vulkan Inductor compute happens in fp32; a logical "to bf16"
            # inside a kernel is a no-op at the fp32 compute level.  The actual
            # bf16 bit-packing is handled by store() via _vk_pack_bf16.
            return f"((float)({x}))"
        s = DTYPE_TO_SLANG.get(dtype, "float")
        return f"(({s})({x}))"

    @staticmethod
    def constant(val, dtype: torch.dtype) -> str:
        return value_to_slang(val)

    @staticmethod
    def abs(x) -> str:
        return f"abs({x})"

    @staticmethod
    def exp(x) -> str:
        return f"exp({x})"

    @staticmethod
    def exp2(x) -> str:
        return f"exp2({x})"

    @staticmethod
    def expm1(x) -> str:
        V.kernel.headers.add("expm1")
        return f"({x}).expm1()"

    @staticmethod
    def sqrt(x) -> str:
        return f"sqrt({x})"

    @staticmethod
    def rsqrt(x) -> str:
        return f"rsqrt({x})"

    @staticmethod
    def log(x) -> str:
        return f"log({x})"

    @staticmethod
    def log2(x) -> str:
        return f"log2({x})"

    @staticmethod
    def xlog1py(x, y) -> str:
        V.kernel.headers.add("log1p")
        return f"((({x}) == 0.0f) ? 0.0f : (({x}) * ({y}).log1p()))"

    @staticmethod
    def special_xlog1py(x, y) -> str:
        return VulkanOverrides.xlog1py(x, y)

    @staticmethod
    def special_xlogy(x, y) -> str:
        return VulkanOverrides.xlogy(x, y)

    @staticmethod
    def to_dtype_bitcast(x, dtype: torch.dtype, src_dtype=None) -> str:
        # N+1.15 — dtype-switch: emits a simple cast expression.
        # When this grows branching Slang (e.g. runtime dtype dispatch),
        # annotate the emitted if/switch with `[branch]` to suppress
        # flatten-by-default penalty. Today all dispatches resolve at
        # codegen time so the Slang output is branch-free.
        # The `[branch]` attribute lives on the callees — f16_to_f32 /
        # f32_to_f16 in dtype_pack.slang already carry it.
        slang = DTYPE_TO_SLANG.get(dtype, "float")
        if src_dtype is not None:
            src_slang = DTYPE_TO_SLANG.get(src_dtype, "float")
            if src_slang == slang:
                return f"({x})"
            if src_slang == "float" and slang == "int":
                return f"asint({x})"
            if src_slang == "float" and slang == "uint":
                return f"asuint({x})"
            if src_slang == "int" and slang == "float":
                return f"asfloat({x})"
            if src_slang == "uint" and slang == "float":
                return f"asfloat({x})"
        if slang == "float":
            return f"asfloat(asuint({x}))"
        if slang == "int":
            return f"asint(asuint({x}))"
        return f"(({slang})({x}))"

    @staticmethod
    def rand(seed, offset) -> str:
        V.kernel.headers.add("random")
        return f"_vk_philox_rand((uint)({offset}), (uint)({seed}))"

    @staticmethod
    def randn(seed, offset) -> str:
        V.kernel.headers.add("random")
        return f"_vk_philox_randn((uint)({offset}), (uint)({seed}))"

    @staticmethod
    def randint64(seed, offset, low, high) -> str:
        V.kernel.headers.add("random")
        return f"((int64_t)(_vk_philox_rand((uint)({offset}), (uint)({seed})) * ((double)({high}) - (double)({low})) + (double)({low})))"

    @staticmethod
    def rand_eager(seed, base_offset, threads_per_round, tid, vec):
        V.kernel.headers.add("random")
        return f"_vk_philox_rand((uint)({base_offset}), (uint)({seed}))"

    @staticmethod
    def frexp(x):
        cache_keys = f"frexp({x})[0]", f"frexp({x})[1]"
        if all(V.kernel.cse.try_get(cache_key) is not None for cache_key in cache_keys):
            return tuple(V.kernel.cse.try_get(cache_key) for cache_key in cache_keys)
        from torch._inductor.codegen.common import BracesBuffer

        code = BracesBuffer()
        exponent = V.kernel.cse.newvar(dtype=torch.int32, shape=x.shape)
        mantissa = V.kernel.cse.newvar(dtype=x.dtype, shape=x.shape)
        code.writeline(f"int {exponent};")
        code.writeline(f"float {mantissa} = frexp((float)({x}), {exponent});")
        V.kernel.compute.splice(code)
        cse_vars = (mantissa, exponent)
        for cache_key, cse_var in zip(cache_keys, cse_vars):
            V.kernel.cse.put(cache_key, cse_var)
        return mantissa, exponent

    @staticmethod
    def ldexp(x, n) -> str:
        return f"ldexp({x}, (int)({n}))"

    @staticmethod
    def nextafter(x, y) -> str:
        return f"nextafter({x}, {y})"

    @staticmethod
    def minimum(a, b) -> str:
        return f"((isnan({a}) || isnan({b})) ? asfloat(0x7FC00000u) : min({a}, {b}))"

    @staticmethod
    def maximum(a, b) -> str:
        return f"((isnan({a}) || isnan({b})) ? asfloat(0x7FC00000u) : max({a}, {b}))"

    @staticmethod
    def recip(x) -> str:
        return f"(1.0f / ({x}))"

    @staticmethod
    def sign(x) -> str:
        return f"sign({x})"

    @staticmethod
    def floor(x) -> str:
        # Slang has the built-in `floor()` (HLSL-inherited).  Upstream's
        # `OpOverrides.floor` raises NotImplementedError and the mps
        # auto-table doesn't cover it.  `aten.frac` decomposes through
        # `floor`, so this missing override blocked frac compilation
        # (and any other consumer of `aten.floor.default`).
        return f"floor({x})"

    @staticmethod
    def ceil(x) -> str:
        return f"ceil({x})"

    @staticmethod
    def round(x) -> str:
        # Slang `round()` is HLSL banker's-rounding (round-half-to-even),
        # matching PyTorch's `torch.round` semantics.
        return f"round({x})"

    @staticmethod
    def isnan(x) -> str:
        # Slang has the built-in `isnan()` (HLSL-inherited).  Upstream
        # `OpsHandler.isnan` raises NotImplementedError.  Used by
        # `aten.nan_to_num` decomposition (`isnan + isinf` + `where`).
        return f"isnan({x})"

    @staticmethod
    def isinf(x) -> str:
        return f"isinf({x})"

    @staticmethod
    def isfinite(x) -> str:
        # `isfinite` not in HLSL; express as `!isnan && !isinf`.
        return f"(!isnan({x}) && !isinf({x}))"

    @staticmethod
    def asinh(x) -> str:
        # `asinh(x) = log(x + sqrt(x^2 + 1))`.  Slang doesn't have a
        # direct `asinh()` builtin in older targets, so synthesize.
        return f"log(({x}) + sqrt(({x}) * ({x}) + 1.0f))"

    @staticmethod
    def acosh(x) -> str:
        # `acosh(x) = log(x + sqrt(x^2 - 1))` for x >= 1.
        return f"log(({x}) + sqrt(({x}) * ({x}) - 1.0f))"

    @staticmethod
    def atanh(x) -> str:
        # `atanh(x) = 0.5 * log((1+x)/(1-x))` for |x| < 1.
        return f"(0.5f * log((1.0f + ({x})) / (1.0f - ({x}))))"

    @staticmethod
    def sinh(x) -> str:
        # `sinh(x) = (exp(x) - exp(-x)) / 2`.  Slang has `sinh()` as
        # part of HLSL intrinsics.
        return f"sinh({x})"

    @staticmethod
    def cosh(x) -> str:
        return f"cosh({x})"

    # ── Basic trig (sin/cos/tan + inverses) ───────────────────────────────
    # These are HLSL/Slang builtins. The default OpsHandler raises
    # NotImplementedError for all six; VulkanOverrides must override them so
    # that backward decompositions that emit aten.sin / aten.cos (e.g.
    # cos_backward decomposes to neg(sin(x)) * grad_out) can compile without
    # falling through to the unimplemented base-class method.

    @staticmethod
    def sin(x) -> str:
        return f"sin({x})"

    @staticmethod
    def cos(x) -> str:
        return f"cos({x})"

    @staticmethod
    def tan(x) -> str:
        return f"tan({x})"

    @staticmethod
    def asin(x) -> str:
        return f"asin({x})"

    @staticmethod
    def acos(x) -> str:
        return f"acos({x})"

    @staticmethod
    def atan(x) -> str:
        return f"atan({x})"

    @staticmethod
    def heaviside(x, values) -> str:
        # `heaviside(x, values)` = `(x > 0) ? 1 : ((x == 0) ? values : 0)`.
        # Mirrors PyTorch's torch.heaviside semantics.
        return f"((({x}) > 0.0f) ? 1.0f : ((({x}) == 0.0f) ? ({values}) : 0.0f))"

    @staticmethod
    def logaddexp(a, b) -> str:
        # log(exp(a) + exp(b)) — numerically stable form:
        # max(a,b) + log1p(exp(-|a-b|)).
        return f"(max(({a}), ({b})) + log1p(exp(-abs(({a}) - ({b})))))"

    @staticmethod
    def logaddexp2(a, b) -> str:
        # log2(2^a + 2^b) — stable form: max(a,b) + log2(1 + 2^(-|a-b|)).
        # Slang has `log2()`, `exp2()` built-ins.
        return f"(max(({a}), ({b})) + log2(1.0f + exp2(-abs(({a}) - ({b})))))"

    @staticmethod
    def rsub(a, b) -> str:
        return f"(({b}) - ({a}))"

    @staticmethod
    def clamp(a, min_val, max_val) -> str:
        return f"clamp({a}, {value_to_slang(min_val)}, {value_to_slang(max_val)})"

    # ── Comparison ops ────────────────────────────────────────────────
    # Return _SlangExpr with dtype=torch.bool so that CSE declares the
    # intermediate variable as `bool` (not `float`).  Bool output buffers
    # are StructuredBuffer<uint>; the store path casts bool→uint.
    #
    # Blocker D (M18.x): Slang lowers `bool_a < bool_b` to `OpULessThan
    # %bool ...`, which SPIR-V rejects (`Expected operands to be scalar or
    # vector int: ULessThan`). This shows up under `torch.max(dim=…)`
    # codegen, which emits `(a != a) > (b != b)` to push NaN to the end.
    # Cast bool operands to `uint` before the ordering compare so SPIR-V
    # sees `OpULessThan %uint ...`. `eq`/`ne` are fine on bools (they
    # lower to `OpLogicalEqual` / `OpLogicalNotEqual`).

    @staticmethod
    def _cast_if_bool(x) -> str:
        if hasattr(x, "dtype") and x.dtype == torch.bool:
            return f"(uint)({x})"
        return f"{x}"

    @staticmethod
    def eq(a, b):
        return _SlangExpr(f"({a} == {b})", dtype=torch.bool)

    @staticmethod
    def ne(a, b):
        return _SlangExpr(f"({a} != {b})", dtype=torch.bool)

    @staticmethod
    def lt(a, b):
        a_s = VulkanOverrides._cast_if_bool(a)
        b_s = VulkanOverrides._cast_if_bool(b)
        return _SlangExpr(f"({a_s} < {b_s})", dtype=torch.bool)

    @staticmethod
    def gt(a, b):
        a_s = VulkanOverrides._cast_if_bool(a)
        b_s = VulkanOverrides._cast_if_bool(b)
        return _SlangExpr(f"({a_s} > {b_s})", dtype=torch.bool)

    @staticmethod
    def le(a, b):
        a_s = VulkanOverrides._cast_if_bool(a)
        b_s = VulkanOverrides._cast_if_bool(b)
        return _SlangExpr(f"({a_s} <= {b_s})", dtype=torch.bool)

    @staticmethod
    def ge(a, b):
        a_s = VulkanOverrides._cast_if_bool(a)
        b_s = VulkanOverrides._cast_if_bool(b)
        return _SlangExpr(f"({a_s} >= {b_s})", dtype=torch.bool)

    @staticmethod
    def logical_not(a) -> str:
        return f"(!({a}))"

    @staticmethod
    def logical_and(a, b) -> str:
        return f"(({a}) && ({b}))"

    @staticmethod
    def logical_or(a, b) -> str:
        return f"(({a}) || ({b}))"

    @staticmethod
    def where(cond, x, y) -> str:
        return f"(({cond}) ? ({x}) : ({y}))"

    @staticmethod
    def tanh(x) -> str:
        return f"tanh({x})"

    @staticmethod
    def prelu(x, weight) -> str:
        return f"(({x}) >= 0.0f ? ({x}) : ({weight} * ({x})))"

    @staticmethod
    def gelu(x, approximate="none") -> str:
        if approximate == "tanh":
            return (
                f"(0.5f * ({x}) * (1.0f + tanh(0.7978845608028654f "
                f"* (({x}) + 0.044715f * ({x}) * ({x}) * ({x})))))"
            )
        V.kernel.headers.add("erf")
        return f"(0.5f * ({x}) * (1.0f + (({x}) * 0.7071067811865475f).erf()))"

    @staticmethod
    def relu(x) -> str:
        return f"max(({x}), 0.0f)"

    @staticmethod
    def relu6(x) -> str:
        return f"min(max(({x}), 0.0f), 6.0f)"

    @staticmethod
    def sigmoid(x) -> str:
        return f"(1.0f / (1.0f + exp(-({x}))))"

    @staticmethod
    def silu(x) -> str:
        return f"(({x}) / (1.0f + exp(-({x}))))"

    @staticmethod
    def hardtanh(x, min_val=-1.0, max_val=1.0) -> str:
        return f"clamp({x}, {value_to_slang(min_val)}, {value_to_slang(max_val)})"

    @staticmethod
    def erf(x) -> str:
        V.kernel.headers.add("erf")
        return f"({x}).erf()"

    @staticmethod
    def erfc(x) -> str:
        V.kernel.headers.add("erf")
        return f"(1.0f - ({x}).erf())"

    @staticmethod
    def erfcx(x) -> str:
        V.kernel.headers.add("erf")
        return f"(exp(({x}) * ({x})) * (1.0f - ({x}).erf()))"

    @staticmethod
    def lgamma(x) -> str:
        V.kernel.headers.add("lgamma")
        return f"({x}).lgamma()"

    @staticmethod
    def ndtr(x) -> str:
        V.kernel.headers.add("erf")
        return f"(0.5f * (1.0f + (({x}) * 0.7071067811865475f).erf()))"

    @staticmethod
    def digamma(x) -> str:
        V.kernel.headers.add("digamma")
        return f"({x}).digamma()"

    @staticmethod
    def log1p(x) -> str:
        V.kernel.headers.add("log1p")
        return f"({x}).log1p()"

    @staticmethod
    def xlogy(x, y) -> str:
        V.kernel.headers.add("log1p")
        return f"((({x}) == 0.0f) ? 0.0f : (({x}) * log(({y}) + (({y}) == 0.0f ? 1.0f : 0.0f)))))"

    # M30 — extension-method forms for the second batch of helpers.
    # Free-function aliases (`c10_vulkan_*`) remain in `helpers.slang` for
    # backward compatibility; T2.10 will retire them.
    @staticmethod
    def hypot(x, y) -> str:
        V.kernel.headers.add("hypot")
        return f"({x}).hypot({y})"

    @staticmethod
    def ndtri(x) -> str:
        V.kernel.headers.add("ndtri")
        return f"({x}).ndtri()"

    @staticmethod
    def spherical_bessel_j0(x) -> str:
        V.kernel.headers.add("spherical_bessel_j0")
        return f"({x}).spherical_bessel_j0()"

    @staticmethod
    def zeta(x, q) -> str:
        # PyTorch / upstream pointwise dispatch: zeta(x=s, q). Slang
        # extension `(s).zeta(q)` matches torch.special.zeta(s, q).
        V.kernel.headers.add("zeta")
        return f"({x}).zeta({q})"

    @staticmethod
    def polygamma(n, x) -> str:
        # Inductor calls polygamma(n, x); extension form `(x).polygamma(n)`
        # mirrors PyTorch's (n, x) ordering at the call site.
        V.kernel.headers.add("polygamma")
        return f"({x}).polygamma({n})"

    @staticmethod
    def igamma(a, x) -> str:
        # PyTorch torch.igamma(a, x). Extension form `(x).igamma(a)`.
        V.kernel.headers.add("igamma")
        return f"({x}).igamma({a})"

    @staticmethod
    def erfinv(x) -> str:
        # Upstream `OpsHandler.erfinv` raises NotImplementedError; the
        # `mps` upstream-overrides table doesn't include erfinv, so we
        # need an explicit Slang override.  Routes to the `(x).erfinv()`
        # extension method in `shaders/lib/special_math.slang` (Mike
        # Giles approximation, single-precision matching CUDA `erfinvf`).
        V.kernel.headers.add("erfinv")
        return f"({x}).erfinv()"

    @staticmethod
    def i0(x) -> str:
        # `(x).i0()` extension in `shaders/lib/special_math.slang`.
        V.kernel.headers.add("i0")
        return f"({x}).i0()"

    @staticmethod
    def i0e(x) -> str:
        V.kernel.headers.add("i0e")
        return f"({x}).i0e()"

    @staticmethod
    def i1(x) -> str:
        V.kernel.headers.add("i1")
        return f"({x}).i1()"

    @staticmethod
    def i1e(x) -> str:
        V.kernel.headers.add("i1e")
        return f"({x}).i1e()"

    @staticmethod
    def trunc(x) -> str:
        # Slang has the built-in `trunc()` (HLSL-inherited).  Upstream's
        # mps override table doesn't cover trunc, so add an explicit
        # mapping here so `aten.trunc` compiles cleanly under
        # `torch.compile`.
        return f"trunc({x})"

    @staticmethod
    def frac(x) -> str:
        # Slang has the built-in `frac()` (HLSL-inherited): returns
        # `x - floor(x)`.  PyTorch's `aten.frac` is `x - trunc(x)`,
        # which differs for negative non-integers (e.g. -2.5 → -0.5
        # vs +0.5).  Use `x - trunc(x)` to match PyTorch semantics.
        return f"(({x}) - trunc({x}))"

    @staticmethod
    def atan2(y, x) -> str:
        # Slang has the built-in `atan2(y, x)` (HLSL-inherited).
        return f"atan2({y}, {x})"

    @staticmethod
    def xlogy(x, y) -> str:
        # `xlogy(x, y) = x * log(y)` with the convention `0 * log(0) = 0`
        # and NaN propagation.  Mirrors `torch.special.xlogy` and the
        # upstream `OpsHandler.xlogy` (which raises NotImplementedError
        # by default).
        return (
            f"((({x}) == 0.0f) ? 0.0f : "
            f"((({y}) <= 0.0f) ? "
            f"(0.0f / 0.0f) : "
            f"(({x}) * log({y}))))"
        )

    @staticmethod
    def xlog1py(x, y) -> str:
        # `x * log1p(y)` with the same `0 * log1p(-1) = 0` convention.
        return (
            f"((({x}) == 0.0f) ? 0.0f : "
            f"((({y}) <= -1.0f) ? "
            f"(0.0f / 0.0f) : "
            f"(({x}) * ({y}).log1p())))"
        )

    @staticmethod
    def pow(a, b) -> str:
        # Slang has no `**` operator. The default `OpOverrides.pow` returns
        # `a ** b` (Python form), which slangc rejects with `unexpected
        # token ('**')`. Adam's update emits `beta ** step` patterns whose
        # CSE locals are float, so `pow(a, b)` (the HLSL/Slang built-in)
        # is the right replacement; it accepts non-integer exponents and
        # mirrors what `metal::pow` does on the MPS backend.
        return f"pow(({a}), ({b}))"

    @staticmethod
    def fmod(a, b) -> str:
        return f"fmod({a}, {b})"

    @staticmethod
    def remainder(a, b) -> str:
        return f"(({a}) - ({b}) * trunc(({a}) / ({b})))"

    @staticmethod
    def copysign(a, b) -> str:
        return f"((({b}) >= 0.0f || (({b}) == -0.0f && ({b}) < 0.0f)) ? abs({a}) : -abs({a}))"

    @staticmethod
    def signbit(x) -> str:
        return f"(({x}) < 0.0f || (asuint({x}) == 0x80000000u))"

    @staticmethod
    def index_expr(expr, dtype):
        with V.kernel._vk_printer.subscript():
            return f"((int)({V.kernel.kexpr(expr)}))"

    @staticmethod
    def indirect_indexing(index_var, size, check=True, wrap_neg=True):
        return sympy_index_symbol(str(index_var))

    @staticmethod
    def masked(mask, body, other) -> str:
        b = body()
        dtype = _infer_dtype(b) or _infer_dtype(other) or torch.float32
        return _SlangExpr(f"(({mask}) ? ({b}) : ({other}))", dtype=dtype)

    @staticmethod
    def vulkan_bwd_diff_unary(fwd_fn, module_name, no_diff_params_json, x_val, grad_out_val, *no_diff_vals):
        """Emit inline bwd_diff(fwd_fn) for unary ops during LoopBody replay.

        Called by the CSEProxy when replaying the FX graph generated by
        ``bwd_diff_inline_lowering.py``'s ``inner_fn``.  The constants
        ``fwd_fn``/``module_name``/``no_diff_params_json`` are captured
        from closure at trace time; ``x_val``/``grad_out_val`` are the
        runtime CSE variable strings produced during replay.
        """
        import json

        from torch_vulkan.inductor.bwd_diff_table import BwdDiffEntry
        from torch_vulkan.inductor.kernel.bwd_diff_inline import emit_inline_unary_bwd

        kernel = V.kernel
        if hasattr(kernel, "_bwd_diff_imports") and module_name not in kernel._bwd_diff_imports:
            kernel._bwd_diff_imports.add(module_name)
            kernel.module_scope_decls.writeline(f"import {module_name};")

        no_diff_params = tuple(json.loads(no_diff_params_json))
        entry = BwdDiffEntry(fwd_fn=fwd_fn, module=module_name, arity=1, no_diff_params=no_diff_params)
        no_diff_scalar_values = (
            {no_diff_params[i]: str(no_diff_vals[i]) for i in range(len(no_diff_params))}
            if no_diff_params
            else None
        )

        body_lines, result_expr = emit_inline_unary_bwd(
            entry,
            x_var=str(x_val),
            grad_out_var=str(grad_out_val),
            dtype="float",
            no_diff_scalar_values=no_diff_scalar_values,
        )
        kernel.compute.writeline(body_lines)
        return result_expr

    @staticmethod
    def vulkan_bwd_diff_binary(fwd_fn, module_name, no_diff_params_json, pred_val, target_val, grad_out_val, *no_diff_vals):
        """Emit inline bwd_diff(fwd_fn) for binary ops during LoopBody replay.

        Like ``vulkan_bwd_diff_unary`` but for binary-input loss backward ops
        (mse_loss, l1_loss, bce, smooth_l1, huber).  Only grad_a (w.r.t.
        ``pred``) is returned; grad_b (w.r.t. ``target``) is discarded
        because Inductor only needs the input gradient for the loss forward.
        """
        import json

        from torch_vulkan.inductor.bwd_diff_table import BwdDiffEntry
        from torch_vulkan.inductor.kernel.bwd_diff_inline import emit_inline_binary_bwd

        kernel = V.kernel
        if hasattr(kernel, "_bwd_diff_imports") and module_name not in kernel._bwd_diff_imports:
            kernel._bwd_diff_imports.add(module_name)
            kernel.module_scope_decls.writeline(f"import {module_name};")

        no_diff_params = tuple(json.loads(no_diff_params_json))
        entry = BwdDiffEntry(fwd_fn=fwd_fn, module=module_name, arity=2, no_diff_params=no_diff_params)
        no_diff_scalar_values = (
            {no_diff_params[i]: str(no_diff_vals[i]) for i in range(len(no_diff_params))}
            if no_diff_params
            else None
        )

        body_lines, result_a_expr, _result_b_expr = emit_inline_binary_bwd(
            entry,
            a_var=str(pred_val),
            b_var=str(target_val),
            grad_out_var=str(grad_out_val),
            dtype="float",
            no_diff_scalar_values=no_diff_scalar_values,
        )
        kernel.compute.writeline(body_lines)
        return result_a_expr


# Upstream's `_initialize_pointwise_overrides` hard-codes its target names to
# {triton, cpp, cppvec, halide, mps}. We reuse the "mps" entries since Metal
# and Slang are both C-like and share most intrinsic names (sin, cos, sqrt,
# etc.). Ops where the Metal form uses `metal::xxx` get shadowed by the
# explicit @staticmethod definitions above.
VulkanOverrides._initialize_pointwise_overrides("mps")


def _register_vulkan_bwd_diff_dtype_rules() -> None:
    """Register dtype and shape propagation rules for our virtual bwd_diff ops.

    ``CSEProxy._default`` (upstream ``codegen/common.py``) calls both
    ``DtypePropagationOpsHandler.<name>`` and ``ShapePropagationOpsHandler.<name>``
    for any op going through the generic CSE path (when backend=="triton", which
    is config.cuda_backend for Vulkan devices).

    Our ops always produce float32 scalars (shape=None means pointwise element).
    Register both handlers once so the singleton instances pick them up.

    The string args (fwd_fn, module_name, no_diff_params_json) confuse
    ``broadcast_shapes_for_args`` in ShapePropagationOpsHandler, so we
    return None (= pointwise/scalar, no block shape) instead.
    """
    import functools

    from torch._inductor.dtype_propagation import DtypePropagationOpsHandler
    from torch._inductor.shape_propagation import ShapePropagationOpsHandler

    _f32_rule = functools.partial(
        DtypePropagationOpsHandler.return_dtype, dtype=torch.float32
    )
    DtypePropagationOpsHandler.vulkan_bwd_diff_unary = _f32_rule  # type: ignore[attr-defined]
    DtypePropagationOpsHandler.vulkan_bwd_diff_binary = _f32_rule  # type: ignore[attr-defined]

    _none_shape: staticmethod = staticmethod(lambda *args, **kwargs: None)
    ShapePropagationOpsHandler.vulkan_bwd_diff_unary = _none_shape  # type: ignore[attr-defined]
    ShapePropagationOpsHandler.vulkan_bwd_diff_binary = _none_shape  # type: ignore[attr-defined]


_register_vulkan_bwd_diff_dtype_rules()
