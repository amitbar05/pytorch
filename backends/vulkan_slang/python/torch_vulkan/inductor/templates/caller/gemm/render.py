"""GEMM template rendering.

Jinja2 template loading and Slang source generation for all matmul variants
(mm, addmm, bmm, mm backward).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from ....vulkan_template import _load_slang_template
from ....vulkan_template_caller import (
    _VALID_IDIFFERENTIABLE_STRUCTS,
    _dtype_to_slang,
    _validate_epilogue_struct,
)

_tile_cache: dict[tuple, str] = {}


def _render_mm_slang(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    dtype_a: str = "float",
    dtype_b: str = "float",
    dtype_c: str = "float",
    dtype_acc: str = "float",
    dtype_bias: str = "float",
    epilogue_struct: str | None = None,
    num_stages: int = 1,
    has_bias: bool = False,
    has_alpha: bool = False,
    has_beta: bool = False,
    has_scale: bool = False,
    has_clamp: bool = False,
    has_batch: bool = False,
    m_per_thread: int = 1,
    n_per_thread: int = 1,
    use_module: bool = False,
) -> str:
    """Render the slang_mm Jinja2 template.

    When ``use_module=True`` (P3.2 / M14), delegates to
    ``_render_mm_linktime_wrapper_slang`` to produce a thin wrapper that
    imports ``mm_tile.slang-module`` instead of inlining the full tile loop.

    `m_per_thread` / `n_per_thread` control register-tile depth. With both at 1
    the workgroup is `(tile_n, tile_m)` and each thread emits one output (legacy
    1-output-per-thread path). With either > 1 the workgroup shrinks to
    `(tile_n / n_per_thread, tile_m / m_per_thread)` and each thread accumulates
    an `(m_per_thread, n_per_thread)` register block — the standard
    register-tile pattern that lets us pick `tile_m * tile_n` > 1024 without
    blowing the threadgroup limit, and amortizes K-loop loads over many outputs.

    CG.M10: ``epilogue_struct`` is a validated ``IDifferentiable`` struct name
    from ``_VALID_IDIFFERENTIABLE_STRUCTS`` (e.g. ``"OpGELU"``).  When
    non-``None``, the Jinja2 template emits the ``Epilogue::apply(...)`` call
    site.  The concrete type is NOT a Jinja variable — it is a Slang generic
    type parameter on ``computeMain<Epilogue : IDifferentiable>``, resolved at
    SPIR-V compile time via the ``entry`` parameter of ``compile_and_dispatch``.
    (Anti-goal #6: no string-based template parameters.)
    """
    from jinja2 import Environment

    # P3.2 / M14: When use_module is True, delegate to the link-time
    # specialization path.  The heavy tile-loop body lives in
    # mm_tile.slang-module (compiled ONCE per dtype).
    if use_module:
        return _render_mm_linktime_wrapper_slang(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            m_per_thread=m_per_thread,
            n_per_thread=n_per_thread,
            num_stages=num_stages,
            has_bias=has_bias,
            epilogue_struct=epilogue_struct,
            dtype_a=dtype_a,
            dtype_b=dtype_b,
            dtype_c=dtype_c,
            dtype_acc=dtype_acc,
            dtype_bias=dtype_bias,
            has_alpha=has_alpha,
            has_beta=has_beta,
            has_scale=has_scale,
            has_clamp=has_clamp,
        )

    # Validate the struct name early — fail at render time, not at SPIR-V
    # compile time, so the error message points at the Python caller.
    epilogue_struct = _validate_epilogue_struct(epilogue_struct)
    has_epilogue = epilogue_struct is not None

    key = (
        tile_m,
        tile_n,
        tile_k,
        dtype_a,
        dtype_b,
        dtype_c,
        dtype_acc,
        dtype_bias,
        epilogue_struct,
        num_stages,
        has_bias,
        has_alpha,
        has_beta,
        has_scale,
        has_clamp,
        has_batch,
        m_per_thread,
        n_per_thread,
    )
    if key in _tile_cache:
        return _tile_cache[key]

    src = _load_slang_template("slang_mm")
    if not src:
        raise RuntimeError("slang_mm.py.jinja template not found")

    env = Environment()
    tmpl = env.from_string(src)
    rendered = tmpl.render(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        dtype_a=dtype_a,
        dtype_b=dtype_b,
        dtype_c=dtype_c,
        dtype_acc=dtype_acc,
        dtype_bias=dtype_bias,
        epilogue=has_epilogue,
        num_stages=num_stages,
        has_bias=has_bias,
        has_alpha=has_alpha,
        has_beta=has_beta,
        has_scale=has_scale,
        has_clamp=has_clamp,
        has_batch=has_batch,
        m_per_thread=m_per_thread,
        n_per_thread=n_per_thread,
    )
    _tile_cache[key] = rendered
    return rendered


# ── P3.2 / M14: Link-time specialization via precompiled mm_tile module ──
# Cached per unique (tile_m, tile_n, tile_k, ..., dtype) tuple.
_linktime_wrapper_cache: dict[tuple, str] = {}


def _render_mm_linktime_wrapper_slang(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_per_thread: int = 1,
    n_per_thread: int = 1,
    num_stages: int = 1,
    has_bias: bool = False,
    epilogue_struct: str | None = None,
    dtype_a: str = "float",
    dtype_b: str = "float",
    dtype_c: str = "float",
    dtype_acc: str = "float",
    dtype_bias: str = "float",
    has_alpha: bool = False,
    has_beta: bool = False,
    has_scale: bool = False,
    has_clamp: bool = False,
) -> str:
    """Render a thin wrapper importing the precompiled mm_tile module.

    P3.2 / M14: The heavy template body lives in ``mm_tile.slang-module``
    (compiled once per dtype). This wrapper (~50 lines) defines the
    tile-size constants via ``static const int`` (link-time resolution
    of ``extern static const int`` in the module), push constants,
    buffer bindings, and the compute entry point.  Compilation is an
    order of magnitude faster because slangc only parses the wrapper and
    links against precompiled IR with constant specialization.

    Uses proper Slang link-time constant resolution:
    ``static const int TILE_M = <value>;`` before ``import mm_tile;``
    satisfies the module's ``extern static const int TILE_M;``
    declaration at link time.
    """
    key = (
        tile_m,
        tile_n,
        tile_k,
        m_per_thread,
        n_per_thread,
        num_stages,
        epilogue_struct,
        has_bias,
        has_alpha,
        has_beta,
        has_scale,
        has_clamp,
        dtype_a,
        dtype_b,
        dtype_c,
        dtype_acc,
        dtype_bias,
    )
    if key in _linktime_wrapper_cache:
        return _linktime_wrapper_cache[key]

    wg_m = tile_m // m_per_thread
    wg_n = tile_n // n_per_thread
    epilogue_type = epilogue_struct if epilogue_struct else "OpIdentity"
    out_binding_idx = 3 if has_bias else 2
    bias_binding_idx = 2

    lines: list[str] = []
    lines.append("// P3.2/M14 mm_tile link-time wrapper — auto-generated")
    lines.append(f"// TILE_M={tile_m} TILE_N={tile_n} TILE_K={tile_k}")
    lines.append(
        f"// M_PER_THREAD={m_per_thread} N_PER_THREAD={n_per_thread}"
        f" NUM_STAGES={num_stages}"
    )
    lines.append("")

    # Link-time specialization: define constants BEFORE import so Slang's
    # linker resolves the module's `extern static const int` declarations.
    lines.append(f"static const int TILE_M = {tile_m};")
    lines.append(f"static const int TILE_N = {tile_n};")
    lines.append(f"static const int TILE_K = {tile_k};")
    lines.append(f"static const int M_PER_THREAD = {m_per_thread};")
    lines.append(f"static const int N_PER_THREAD = {n_per_thread};")
    lines.append(f"#define NUM_STAGES {num_stages}")
    lines.append("")
    lines.append("// NUM_STAGES is a #define (not static const) because mm_tile.slang")
    lines.append("// uses it in #if guards for groupshared array sizing.")
    lines.append("import mm_tile;")
    lines.append("")

    # Push-constant struct
    lines.append("struct PC {")
    lines.append("    uint M;")
    lines.append("    uint N;")
    lines.append("    uint K;")
    lines.append("    uint stride_a_m;")
    lines.append("    uint stride_a_k;")
    lines.append("    uint stride_b_k;")
    lines.append("    uint stride_b_n;")
    lines.append("    uint stride_c_m;")
    lines.append("    uint stride_c_n;")
    if has_bias:
        lines.append("    uint stride_bias_n;")
    if has_alpha:
        lines.append("    float alpha;")
    if has_beta:
        lines.append("    float beta;")
    if has_scale:
        lines.append("    float scale;")
    if has_clamp:
        lines.append("    float clamp_min;")
        lines.append("    float clamp_max;")
    lines.append("};")
    lines.append("")
    lines.append("[[vk::push_constant]] PC pc;")
    lines.append("")

    # Buffer bindings
    lines.append(f"[[vk::binding(0)]] StructuredBuffer<{dtype_a}> a;")
    lines.append(f"[[vk::binding(1)]] StructuredBuffer<{dtype_b}> b;")
    if has_bias:
        lines.append(
            f"[[vk::binding({bias_binding_idx})]] StructuredBuffer<{dtype_bias}> bias;"
        )
    lines.append(f"[[vk::binding({out_binding_idx})]] RWStructuredBuffer<{dtype_c}> c;")
    lines.append("")

    # Entry point — delegates to mm_tile::computeTile<Epilogue>
    lines.append('[shader("compute")]')
    lines.append(f"[numthreads({wg_n}, {wg_m}, 1)]")
    lines.append("void computeMain(")
    lines.append("    uint3 gtid : SV_DispatchThreadID,")
    lines.append("    uint3 lid : SV_GroupThreadID,")
    lines.append("    uint3 gid : SV_GroupID)")
    lines.append("{")
    lines.append("    uint row_base = gid.y * (uint)TILE_M;")
    lines.append("    uint col_base = gid.x * (uint)TILE_N;")
    lines.append("")

    # Build the MM_PC struct for the module call
    lines.append("    mm_tile::MM_PC mm_pc;")
    lines.append("    mm_pc.M = pc.M;")
    lines.append("    mm_pc.N = pc.N;")
    lines.append("    mm_pc.K = pc.K;")
    lines.append("    mm_pc.stride_a_m = pc.stride_a_m;")
    lines.append("    mm_pc.stride_a_k = pc.stride_a_k;")
    lines.append("    mm_pc.stride_b_k = pc.stride_b_k;")
    lines.append("    mm_pc.stride_b_n = pc.stride_b_n;")
    lines.append("    mm_pc.stride_c_m = pc.stride_c_m;")
    lines.append("    mm_pc.stride_c_n = pc.stride_c_n;")
    lines.append("")

    # Call the module's computeTile function (handles load → mma → store).
    # The epilogue (bias, activation, clamp, etc.) is applied by the wrapper
    # AFTER the module's store to allow dtype-cast and alpha/beta blending.
    lines.append(f"    mm_tile::computeTile<{epilogue_type}>(")
    lines.append("        row_base, col_base, mm_pc, a, b, c, gid, lid);")
    lines.append("")

    # Post-module epilogue: alpha, beta, bias, clamp, scale
    # These are applied per-element AFTER the module's store_epilogue
    # to support operations not expressible as pure IPointwise.
    if has_alpha or has_beta or has_bias or has_scale or has_clamp:
        lines.append("    // Post-module epilogue adjustments")
        lines.append("    [unroll]")
        lines.append("    for (uint mi = 0; mi < (uint)M_PER_THREAD; mi++) {")
        lines.append("        uint row = row_base + lid.y * (uint)M_PER_THREAD + mi;")
        lines.append("        if (row >= pc.M) continue;")
        lines.append("        [unroll]")
        lines.append("        for (uint ni = 0; ni < (uint)N_PER_THREAD; ni++) {")
        lines.append(
            "            uint col = col_base + lid.x * (uint)N_PER_THREAD + ni;"
        )
        lines.append("            if (col >= pc.N) continue;")
        lines.append(
            f"            {dtype_acc} v = ({dtype_acc})"
            f"c[row * pc.stride_c_m + col * pc.stride_c_n];"
        )
        if has_alpha:
            lines.append("            v *= pc.alpha;")
        if has_bias:
            lines.append(f"            v += ({dtype_acc})bias[col * pc.stride_bias_n];")
        if has_beta:
            lines.append(
                f"            v = pc.beta * v +"
                f" (1.0 - pc.beta) * ({dtype_acc})"
                f"c[row * pc.stride_c_m + col * pc.stride_c_n];"
            )
        if has_scale:
            lines.append("            v *= pc.scale;")
        if has_clamp:
            lines.append(
                f"            v = clamp(v, ({dtype_acc})pc.clamp_min,"
                f" ({dtype_acc})pc.clamp_max);"
            )
        lines.append(
            f"            c[row * pc.stride_c_m + col * pc.stride_c_n] = ({dtype_c})v;"
        )
        lines.append("        }")
        lines.append("    }")

    lines.append("}")

    src = "\n".join(lines) + "\n"
    _linktime_wrapper_cache[key] = src
    return src


def _render_mm_backward_slang(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    *,
    transpose_a: bool = False,
    transpose_b: bool = False,
    num_stages: int = 1,
    m_per_thread: int = 1,
    n_per_thread: int = 1,
) -> str:
    """Render the tiled matmul template for backward use.

    T4.2: Matmul backward computes dA = dC @ B^T and dB = A^T @ dC.
    Instead of transposing operands on the CPU (which requires a copy),
    this renders the forward mm template with the strides pre-configured
    for the transposed access pattern:

    - ``transpose_a=True`` → the A operand is read with stride_a_m=1
      (i.e. A is logically transposed: A^T has K rows and M columns)
    - ``transpose_b=True`` → the B operand is read with stride_b_k=1
      (i.e. B is logically transposed: B^T has N rows and K columns)

    The rendered shader is functionally identical to the forward template
    but the push-constant stride layout encodes the transposition, avoiding
    a host-side copy + contiguous call.

    Returns the Slang source string.
    """
    # Use the same _render_mm_slang with explicit dtype params.
    # Transposition is handled at dispatch time via the push-constant
    # strides; the template itself is identical.
    src = _render_mm_slang(
        tile_m,
        tile_n,
        tile_k,
        dtype_a="float",
        dtype_b="float",
        dtype_c="float",
        dtype_acc="float",
        num_stages=num_stages,
        m_per_thread=m_per_thread,
        n_per_thread=n_per_thread,
    )
    return src


# ═══════════════════════════════════════════════════════════════════════════
# CG.M5 — Single-kernel matmul backward via [Differentiable] tile_inner_madd
# ═══════════════════════════════════════════════════════════════════════════


def _render_mm_bwd_slang(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    *,
    dtype_a: str = "float",
    dtype_b: str = "float",
    dtype_c: str = "float",
    dtype_acc: str = "float",
    has_batch: bool = False,
    m_per_thread: int = 1,
    n_per_thread: int = 1,
) -> str:
    """Render the CG.M5 slang_mm_bwd Jinja2 template.

    Produces a single-kernel backward that computes BOTH dA and dB in one
    dispatch by wrapping ``bwd_diff(tile_inner_madd)`` in a tiled K-loop.
    This replaces the 2-dispatch decomposition (dA = dC @ B^T, dB = A^T @ dC)
    with 1 fused dispatch.

    The template supports:
      - mm backward (has_batch=False): dA[M,K], dB[K,N]
      - bmm backward (has_batch=True): dA[B,M,K], dB[B,K,N]
      - Register tiling via m_per_thread / n_per_thread
    """
    from jinja2 import Environment

    key = (
        tile_m,
        tile_n,
        tile_k,
        dtype_a,
        dtype_b,
        dtype_c,
        dtype_acc,
        has_batch,
        m_per_thread,
        n_per_thread,
    )
    if key in _tile_cache:
        return _tile_cache[key]

    src = _load_slang_template("slang_mm_bwd")
    if not src:
        raise RuntimeError("slang_mm_bwd.py.jinja template not found")

    env = Environment()
    tmpl = env.from_string(src)
    rendered = tmpl.render(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        dtype_a=dtype_a,
        dtype_b=dtype_b,
        dtype_c=dtype_c,
        dtype_acc=dtype_acc,
        has_batch=has_batch,
        m_per_thread=m_per_thread,
        n_per_thread=n_per_thread,
    )
    _tile_cache[key] = rendered
    return rendered


# ═══════════════════════════════════════════════════════════════════════════
# OP.24 — Int8 matmul (inference-only)
# ═══════════════════════════════════════════════════════════════════════════

_int8_tile_cache: dict[tuple, str] = {}


def _render_mm_int8_slang(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_per_thread: int = 1,
    n_per_thread: int = 1,
) -> str:
    """Render a thin wrapper that imports ``mm_int8.slang`` and dispatches
    ``mm_int8::computeTile`` for int8×int8→int32→float32 matmul.

    OP.24: The wrapper defines tile-size constants via ``static const int``
    BEFORE ``import mm_int8;`` so Slang's linker resolves the module's
    ``extern static const int`` declarations at link time.

    Returns:
        Complete Slang source string ready for compilation.
    """
    key = (tile_m, tile_n, tile_k, m_per_thread, n_per_thread)
    if key in _int8_tile_cache:
        return _int8_tile_cache[key]

    wg_m = tile_m // m_per_thread
    wg_n = tile_n // n_per_thread

    lines: list[str] = []
    lines.append("// OP.24 mm_int8 wrapper — auto-generated")
    lines.append(f"// TILE_M={tile_m} TILE_N={tile_n} TILE_K={tile_k}")
    lines.append(
        f"// M_PER_THREAD={m_per_thread} N_PER_THREAD={n_per_thread}"
    )
    lines.append("")

    # Link-time specialization: define constants BEFORE import.
    lines.append(f"static const int TILE_M = {tile_m};")
    lines.append(f"static const int TILE_N = {tile_n};")
    lines.append(f"static const int TILE_K = {tile_k};")
    lines.append(f"static const int M_PER_THREAD = {m_per_thread};")
    lines.append(f"static const int N_PER_THREAD = {n_per_thread};")
    lines.append("")
    lines.append("import mm_int8;")
    lines.append("")

    # Push-constant struct — matches MMInt8_PC in mm_int8.slang
    lines.append("struct PC {")
    lines.append("    uint M;")
    lines.append("    uint N;")
    lines.append("    uint K;")
    lines.append("    uint stride_a_m;")
    lines.append("    uint stride_a_k;")
    lines.append("    uint stride_b_k;")
    lines.append("    uint stride_b_n;")
    lines.append("    uint stride_c_m;")
    lines.append("    uint stride_c_n;")
    lines.append("};")
    lines.append("")
    lines.append("[[vk::push_constant]] PC pc;")
    lines.append("")

    # Buffer bindings: A/B are packed uint (4×int8 per word), C is float32.
    lines.append("[[vk::binding(0)]] StructuredBuffer<uint> a;")
    lines.append("[[vk::binding(1)]] StructuredBuffer<uint> b;")
    lines.append("[[vk::binding(2)]] RWStructuredBuffer<float> c;")
    lines.append("")

    # Entry point — delegates to mm_int8::computeTile
    lines.append('[shader("compute")]')
    lines.append(f"[numthreads({wg_n}, {wg_m}, 1)]")
    lines.append("void computeMain(")
    lines.append("    uint3 gtid : SV_DispatchThreadID,")
    lines.append("    uint3 lid : SV_GroupThreadID,")
    lines.append("    uint3 gid : SV_GroupID)")
    lines.append("{")
    lines.append("    uint row_base = gid.y * (uint)TILE_M;")
    lines.append("    uint col_base = gid.x * (uint)TILE_N;")
    lines.append("")
    lines.append("    mm_int8::MMInt8_PC mm_pc;")
    lines.append("    mm_pc.M = pc.M;")
    lines.append("    mm_pc.N = pc.N;")
    lines.append("    mm_pc.K = pc.K;")
    lines.append("    mm_pc.stride_a_m = pc.stride_a_m;")
    lines.append("    mm_pc.stride_a_k = pc.stride_a_k;")
    lines.append("    mm_pc.stride_b_k = pc.stride_b_k;")
    lines.append("    mm_pc.stride_b_n = pc.stride_b_n;")
    lines.append("    mm_pc.stride_c_m = pc.stride_c_m;")
    lines.append("    mm_pc.stride_c_n = pc.stride_c_n;")
    lines.append("")
    lines.append("    mm_int8::computeTile(")
    lines.append("        row_base, col_base, mm_pc, a, b, c, gid, lid);")
    lines.append("}")

    src = "\n".join(lines) + "\n"
    _int8_tile_cache[key] = src
    return src
