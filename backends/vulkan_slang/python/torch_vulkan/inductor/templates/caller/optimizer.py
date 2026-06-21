"""Foreach optimizer template callers.

Provides rendering, dispatch, and installation for the foreach optimizer Slang
template (SGD, SGD+momentum, AdamW, Lion).

B1: Uses Slang interface IOptimizer with compile-time specialization.
A single Jinja variable ``algorithm_type`` selects the concrete type
(e.g. "AdamWImpl") — no per-line ``{% if algorithm %}`` branches.
All push constants and buffers are packed uniformly regardless of algorithm.
"""

from __future__ import annotations

import os
import struct
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

from ...buffer_pool import pool_acquire_scratch, pool_release_scratch
from ...runtime import compile_and_dispatch
from ...vulkan_template import _load_slang_template

# Batch sizes the template can be rendered for.  The template unrolls
# over `batch_size` params per dispatch (each param gets its own binding
# slot).  We pre-select a few common sizes; callers pick the smallest
# size that fits their parameter count.
#
# Push-constant layout: 16-byte header + batch_size × 28-byte ParamConfig.
# AMD RDNA1 maxPushConstantsSize = 256 bytes → max batch_size = (256-16)/28 = 8.
# Sizes 15/21/32 are removed to stay within the 256-byte device limit.
# For N > 8 params, _SlangForeachOptimizer.__call__ splits into multiple batches.
_OPTIMIZER_BATCH_SIZES = (1, 7, 8)

# ── B1: Algorithm name → Slang concrete type mapping ──────────────────────
# External callers use the legacy string names; internally we map to the
# Slang interface concrete type names used by the generic entry point.
_ALGORITHM_TO_TYPE: dict[str, str] = {
    "sgd": "SGDImpl",
    "sgd_momentum": "SGDMomentumImpl",
    "adamw": "AdamWImpl",
    "lion": "LionImpl",
}

# Reverse mapping for introspection / debug.
_TYPE_TO_ALGORITHM: dict[str, str] = {v: k for k, v in _ALGORITHM_TO_TYPE.items()}


def _resolve_algorithm_type(algorithm: str) -> str:
    """Map a legacy algorithm name to the Slang concrete type name.

    Accepts both legacy names ("sgd", "adamw", etc.) and the Slang type
    names ("SGDImpl", "AdamWImpl", etc.) for forward compatibility.
    Returns the Slang type name.
    """
    if algorithm in _ALGORITHM_TO_TYPE:
        return _ALGORITHM_TO_TYPE[algorithm]
    if algorithm in _TYPE_TO_ALGORITHM:
        return algorithm
    raise ValueError(
        f"Unknown optimizer algorithm: '{algorithm}'. "
        f"Expected one of: {list(_ALGORITHM_TO_TYPE.keys())} "
        f"or {list(_ALGORITHM_TO_TYPE.values())}"
    )


def _foreach_use_parameter_array() -> bool:
    """Whether the foreach optimizer template should emit the N+1.5.b
    ``ParamSlot params[BATCH_SIZE]`` array-of-structs path with
    descriptor-array bindings (descriptorCount=BATCH_SIZE per buffer
    family) instead of the round-3 flat-binding + switch-cascade layout.

    The new path requires:
      (a) VK_EXT_descriptor_indexing on the device
          (probed via ``_C._descriptor_indexing_enabled``);
      (b) The N+1.5.a Python wiring that extracts per-binding
          ``descriptorCount`` from reflection JSON and routes the
          dispatch through ``_C._jit_dispatch_indexed`` instead of the
          flat ``_C._jit_dispatch`` FFI.
    """
    from ... import config as _cfg

    if not _cfg.parameter_array():
        return False

    try:
        import torch_vulkan as _tv

        ce = getattr(_tv, "_c_ext", None)
        if ce is None:
            return False
        probe = getattr(ce, "_descriptor_indexing_enabled", None)
        if probe is None or not probe():
            return False
        if not hasattr(ce, "_jit_dispatch_indexed"):
            return False
    except Exception:
        return False
    return True


def _render_foreach_optimizer_slang(
    algorithm_type: str,
    batch_size: int,
    output_dtype: str = "float",
    parameter_array: bool | None = None,
) -> str:
    """Render the foreach_optimizer.slang template for a given
    (algorithm_type, batch_size, output_dtype) combination.

    B1: ``algorithm_type`` is the Slang concrete type name (e.g. "AdamWImpl").
    It is substituted once as the generic argument to ``computeMain<...>``
    and as the qualifier in ``Algo::step(...)`` — no per-line algorithm
    branching in the template.
    """
    from jinja2 import Environment

    source_template = _load_slang_template("foreach_optimizer")
    if not source_template:
        raise RuntimeError(
            "foreach_optimizer.slang template not found — "
            "is the Vulkan Slang backend installed correctly?"
        )
    if parameter_array is None:
        parameter_array = _foreach_use_parameter_array()
    env = Environment()
    tmpl = env.from_string(source_template)
    return tmpl.render(
        algorithm_type=algorithm_type,
        batch_size=batch_size,
        output_dtype=output_dtype,
        parameter_array=parameter_array,
    )


def _foreach_cache_key(
    algorithm_type: str,
    batch_size: int,
    output_dtype: str,
    parameter_array: bool,
) -> str:
    """Build the SPIR-V / pipeline cache key for a foreach variant.

    B1: Uses ``algorithm_type`` (Slang concrete type name) instead of
    legacy algorithm string.
    """
    suffix = "_pa" if parameter_array else ""
    return f"slang_foreach_{algorithm_type}_{batch_size}_{output_dtype}{suffix}"


# ── B1: Always-packed push-constant layout ────────────────────────────────
# The ParamConfig struct in the template is always 7 floats (28 B, padded to
# 32 B).  We pack all 7 fields for every algorithm; unused fields are 0.0f.

_PARAM_CONFIG_FORMAT = "Iffffff"      # numel + 6 floats (the 7th is _pad)
_FULL_HEADER_FORMAT = "4I"             # n_params + 3 uint _pad


def _pack_push_constants(
    n_params: int,
    batch_size: int,
    numels: list[int],
    lr: list[float],
    weight_decay: list[float],
    momentum: list[float] | None = None,
    beta2: list[float] | None = None,
    eps: list[float] | None = None,
) -> bytes:
    """Pack push constants in the B1 uniform layout.

    Always packs the full ParamConfig (7 fields) for each param.
    Unused fields are zero-filled.  Padding slots beyond n_params
    are also zero-filled.
    """
    pc_parts: list[bytes] = []
    pc_parts.append(struct.pack(_FULL_HEADER_FORMAT, n_params, 0, 0, 0))

    mom_list = momentum if momentum is not None else [0.0] * n_params
    b2_list = beta2 if beta2 is not None else [0.0] * n_params
    eps_list = eps if eps is not None else [0.0] * n_params

    for i in range(n_params):
        pc_parts.append(struct.pack(
            _PARAM_CONFIG_FORMAT,
            numels[i],
            lr[i],
            weight_decay[i],
            mom_list[i],
            b2_list[i],
            eps_list[i],
            0.0,  # _pad
        ))

    # Pad remaining ParamConfig slots for unused entries.
    for _ in range(n_params, batch_size):
        pc_parts.append(struct.pack(_PARAM_CONFIG_FORMAT, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    return b"".join(pc_parts)


def _slang_foreach_optimizer(
    algorithm: str,
    batch_size: int,
    output_dtype: str,
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    src: str | None = None,
    cache_key: str | None = None,
    momentum_bufs: list[torch.Tensor] | None = None,
    v_bufs: list[torch.Tensor] | None = None,
    lr: list[float] | None = None,
    weight_decay: list[float] | None = None,
    momentum: list[float] | None = None,
    beta2: list[float] | None = None,
    eps: list[float] | None = None,
) -> None:
    """Dispatch the foreach optimizer template for a batch of params.

    This is the low-level dispatch.  The caller is responsible for
    batching params into groups ≤ `batch_size` and providing the
    rendered source / cache_key.

    B1: All buffers (param, grad, momentum, v) are ALWAYS provided.
    Momentum/V buffers are padded with dummies when the algorithm
    doesn't use them.  Push constants are always the full 7-float layout.
    ``num_outputs`` is always ``n_params * 3`` (param + momentum + v).
    """
    n_params = len(params)
    assert n_params <= batch_size, (
        f"_slang_foreach_optimizer: {n_params} params exceeds "
        f"rendered batch_size {batch_size}"
    )

    algorithm_type = _resolve_algorithm_type(algorithm)

    if src is None or cache_key is None:
        use_pa = _foreach_use_parameter_array()
        src = _render_foreach_optimizer_slang(
            algorithm_type, batch_size, output_dtype, parameter_array=use_pa
        )
        cache_key = _foreach_cache_key(algorithm_type, batch_size, output_dtype, use_pa)

    # Validate all params have the same dtype (but numels may differ).
    dtype = params[0].dtype
    for p in params:
        if p.dtype != dtype:
            raise ValueError(
                f"foreach optimizer: all params must have same dtype, "
                f"got {dtype} vs {p.dtype}"
            )

    # ── B1: Always pack full push constants ───────────────────────────
    default_lr = [0.01] * n_params
    default_wd = [0.0] * n_params

    numels = [p.numel() for p in params]
    # Grid X covers max numel; per-param early-return handles shorter params.
    max_numel = max(numels) if numels else 1
    lr_list = list(lr) if lr is not None else default_lr
    wd_list = list(weight_decay) if weight_decay is not None else default_wd
    mom_list = list(momentum) if momentum is not None else [0.0] * n_params
    b2_list = list(beta2) if beta2 is not None else [0.0] * n_params
    eps_list = list(eps) if eps is not None else [0.0] * n_params

    pc_bytes = _pack_push_constants(
        n_params, batch_size, numels, lr_list, wd_list,
        momentum=mom_list, beta2=b2_list, eps=eps_list,
    )

    # ── B1: Always build all 4 buffer types per param ─────────────────
    # Dummy tensor for padding slots.
    dummy = pool_acquire_scratch((1,), dtype, params[0].device)
    if dummy is None:
        dummy = torch.empty(1, device=params[0].device, dtype=dtype)

    def _pad(lst: list[torch.Tensor] | None) -> list[torch.Tensor]:
        if lst is None:
            return [dummy] * batch_size
        return list(lst) + [dummy] * (batch_size - len(lst))

    buffers: list[torch.Tensor] = []
    # Interleave: param0, grad0, momentum0, v0, param1, grad1, ...
    for i in range(n_params):
        buffers.append(params[i])
        buffers.append(grads[i])
        buffers.append(
            momentum_bufs[i] if momentum_bufs is not None else dummy
        )
        buffers.append(
            v_bufs[i] if v_bufs is not None else dummy
        )
    # Padding slots for batch_size - n_params.
    for _ in range(n_params, batch_size):
        buffers.append(dummy)  # param
        buffers.append(dummy)  # grad
        buffers.append(dummy)  # momentum
        buffers.append(dummy)  # v

    # Grid: X = max numel across params (per-param early-return handles shorter ones).
    threadgroup_size = 256
    grid_x = (max_numel + threadgroup_size - 1) // threadgroup_size
    grid_y = n_params
    grid_z = 1

    # B1: Always 3 outputs per param (param + momentum + v).
    # The C++ runtime uses SPIR-V reflection to identify which bindings
    # are storage buffers with write access; the count is for validation.
    num_outputs = n_params * 3

    compile_and_dispatch(
        src,
        buffers,
        grid_x,
        grid_y,
        grid_z,
        push_constants=pc_bytes,
        num_outputs=num_outputs,
        entry=f"computeMain<{algorithm_type}>",
        cache_key=cache_key,
    )
    # T6.7: dispatch is done; the padding ``dummy`` is dead.
    buffers.clear()
    pool_release_scratch(dummy)
    dummy = None  # type: ignore[assignment]


class _SlangForeachOptimizer:
    """Picklable callable for a rendered foreach optimizer template.

    Each instance is bound to a specific (algorithm_type, batch_size,
    output_dtype) combination.  The `__call__` signature takes tensor
    lists and scalar lists and dispatches the template, automatically
    batching across multiple dispatches if there are more params than
    `batch_size`.

    B1: ``algorithm_type`` is the Slang concrete type name (e.g. "AdamWImpl").
    ``algorithm`` (legacy string) is accepted and mapped internally.

    Source is cached at module level per (algorithm_type, batch_size,
    output_dtype) so multiple instances with the same key share the
    rendered source.
    """

    __slots__ = (
        "algorithm",
        "algorithm_type",
        "batch_size",
        "output_dtype",
        "__name__",
        "_src",
        "_cache_key",
    )

    _src_cache: dict[tuple, tuple[str, str]] = {}

    def __init__(self, algorithm: str, batch_size: int, output_dtype: str = "float"):
        self.algorithm = algorithm
        self.algorithm_type = _resolve_algorithm_type(algorithm)
        self.batch_size = batch_size
        self.output_dtype = output_dtype
        self.__name__ = f"slang_foreach_{self.algorithm_type}_{batch_size}_{output_dtype}"
        self._src: str | None = None
        self._cache_key: str | None = None
        self._ensure_source()

    def _ensure_source(self) -> None:
        use_pa = _foreach_use_parameter_array()
        key = (self.algorithm_type, self.batch_size, self.output_dtype, use_pa)
        cached = _SlangForeachOptimizer._src_cache.get(key)
        if cached is not None:
            self._src, self._cache_key = cached
            return
        self._src = _render_foreach_optimizer_slang(
            self.algorithm_type,
            self.batch_size,
            self.output_dtype,
            parameter_array=use_pa,
        )
        self._cache_key = _foreach_cache_key(
            self.algorithm_type, self.batch_size, self.output_dtype, use_pa
        )
        _SlangForeachOptimizer._src_cache[key] = (self._src, self._cache_key)

    def __call__(
        self,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        *,
        lr: list[float] | None = None,
        weight_decay: list[float] | None = None,
        momentum: list[float] | None = None,
        beta2: list[float] | None = None,
        eps: list[float] | None = None,
        momentum_bufs: list[torch.Tensor] | None = None,
        v_bufs: list[torch.Tensor] | None = None,
    ) -> None:
        """Dispatch the foreach optimizer template.

        If there are more params than `batch_size`, dispatches in
        multiple batches automatically.

        B1: ``momentum_bufs`` and ``v_bufs`` are optional; the low-level
        dispatch creates dummy buffers when they are None (for algorithms
        like SGD that don't use state).
        """
        n = len(params)
        bs = self.batch_size
        alg = self.algorithm
        dt = self.output_dtype

        for start in range(0, n, bs):
            end = min(start + bs, n)
            chunk_params = params[start:end]
            chunk_grads = grads[start:end]
            chunk_lr = lr[start:end] if lr is not None else None
            chunk_wd = weight_decay[start:end] if weight_decay is not None else None
            chunk_mom = momentum[start:end] if momentum is not None else None
            chunk_b2 = beta2[start:end] if beta2 is not None else None
            chunk_eps = eps[start:end] if eps is not None else None
            chunk_mom_bufs = (
                momentum_bufs[start:end] if momentum_bufs is not None else None
            )
            chunk_v_bufs = v_bufs[start:end] if v_bufs is not None else None

            _slang_foreach_optimizer(
                algorithm=alg,
                batch_size=bs,
                output_dtype=dt,
                params=list(chunk_params),
                grads=list(chunk_grads),
                src=self._src,
                cache_key=self._cache_key,
                momentum_bufs=chunk_mom_bufs,
                v_bufs=chunk_v_bufs,
                lr=chunk_lr,
                weight_decay=chunk_wd,
                momentum=chunk_mom,
                beta2=chunk_b2,
                eps=chunk_eps,
            )

    def __reduce__(self):
        return (
            _SlangForeachOptimizer,
            (self.algorithm, self.batch_size, self.output_dtype),
        )


# ── Batch size selection ──────────────────────────────────────────────

_foreach_default_callers: dict[str, _SlangForeachOptimizer] = {}


def _pick_foreach_optimizer_caller(
    algorithm: str,
    n_params: int,
    output_dtype: str = "float",
) -> _SlangForeachOptimizer:
    """Pick (or create) a _SlangForeachOptimizer instance with a
    batch_size large enough to cover `n_params`.

    Returns the smallest pre-defined batch_size ≥ n_params.

    B1: ``algorithm`` accepts both legacy names ("sgd", "adamw", ...)
    and Slang type names ("SGDImpl", "AdamWImpl", ...).
    """
    alg_type = _resolve_algorithm_type(algorithm)
    for bs in _OPTIMIZER_BATCH_SIZES:
        if bs >= n_params:
            key = f"{alg_type}_{bs}_{output_dtype}"
            caller = _foreach_default_callers.get(key)
            if caller is None:
                caller = _SlangForeachOptimizer(algorithm, bs, output_dtype)
                _foreach_default_callers[key] = caller
            return caller
    # Fallback: use the largest predefined size.
    bs = _OPTIMIZER_BATCH_SIZES[-1]
    key = f"{alg_type}_{bs}_{output_dtype}"
    caller = _foreach_default_callers.get(key)
    if caller is None:
        caller = _SlangForeachOptimizer(algorithm, bs, output_dtype)
        _foreach_default_callers[key] = caller
    return caller


# ── install_external_optimizer ─────────────────────────────────────────


def install_external_optimizer() -> None:
    """Pre-render the foreach optimizer template for the common
    (algorithm, batch_size, output_dtype) combinations, and register
    the custom ops for SGD, SGD+momentum, AdamW, Lion.

    The custom ops are registered via eager_patches; the FX pass
    (``_fuse_optimizer_step_to_foreach``) fuses optimizer-step
    patterns into these custom ops, which dispatch through the
    ``_SlangForeachOptimizer`` template.

    Safe to call multiple times — only installs once.
    """
    global _optimizer_installed
    if _optimizer_installed:
        return
    _optimizer_installed = True

    from ...fx_passes.eager_patches import (
        _ensure_foreach_adamw_step_op_registered,
        _ensure_foreach_lion_step_op_registered,
        _ensure_foreach_sgd_momentum_step_op_registered,
        _ensure_foreach_sgd_step_op_registered,
    )

    _ensure_foreach_sgd_step_op_registered()
    _ensure_foreach_sgd_momentum_step_op_registered()
    _ensure_foreach_adamw_step_op_registered()
    _ensure_foreach_lion_step_op_registered()


def _collect_optimizer_prewarm_specs() -> list[tuple[str, str]]:
    """Render (cache_key, slang_src) pairs for the common optimizer
    template variants.

    B1: Uses Slang type names (SGDImpl, etc.) for cache keys and rendering.
    """
    specs: list[tuple[str, str]] = []
    dtypes = ("float", "half")
    algorithms = ("SGDImpl", "SGDMomentumImpl", "AdamWImpl", "LionImpl")
    use_pa = _foreach_use_parameter_array()
    for alg in algorithms:
        for bs in _OPTIMIZER_BATCH_SIZES:
            for dt in dtypes:
                key = _foreach_cache_key(alg, bs, dt, use_pa)
                src = _render_foreach_optimizer_slang(
                    alg, bs, dt, parameter_array=use_pa
                )
                specs.append((key, src))
    return specs


def prewarm_optimizer_templates(*, sync: bool = False) -> int:
    """Submit optimizer template variants to the slangc thread pool.

    No-op when slangc is not available.  With ``sync=False`` the call
    returns immediately and the cache is populated in the background.
    """
    if os.environ.get("TORCH_VULKAN_NO_PREWARM") == "1":
        return 0
    from ...runtime import _slangc_available, prewarm_compile

    if not _slangc_available():
        return 0
    return prewarm_compile(_collect_optimizer_prewarm_specs(), sync=sync)


_optimizer_installed = False
