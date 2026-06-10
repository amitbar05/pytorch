"""Foreach optimizer template callers.

Provides rendering, dispatch, and installation for the foreach optimizer Slang
template (SGD, SGD+momentum, AdamW, Lion).
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
_OPTIMIZER_BATCH_SIZES = (1, 7, 15, 21, 32)


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
    algorithm: str,
    batch_size: int,
    output_dtype: str = "float",
    parameter_array: bool | None = None,
) -> str:
    """Render the foreach_optimizer.slang template for a given
    (algorithm, batch_size, output_dtype) combination.
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
        algorithm=algorithm,
        batch_size=batch_size,
        output_dtype=output_dtype,
        parameter_array=parameter_array,
    )


def _foreach_cache_key(
    algorithm: str,
    batch_size: int,
    output_dtype: str,
    parameter_array: bool,
) -> str:
    """Build the SPIR-V / pipeline cache key for a foreach variant."""
    suffix = "_pa" if parameter_array else ""
    return f"slang_foreach_{algorithm}_{batch_size}_{output_dtype}{suffix}"


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

    All per-param scalar lists must have length equal to `len(params)`.
    """
    n_params = len(params)
    assert n_params <= batch_size, (
        f"_slang_foreach_optimizer: {n_params} params exceeds "
        f"rendered batch_size {batch_size}"
    )

    if src is None or cache_key is None:
        use_pa = _foreach_use_parameter_array()
        src = _render_foreach_optimizer_slang(
            algorithm, batch_size, output_dtype, parameter_array=use_pa
        )
        cache_key = _foreach_cache_key(algorithm, batch_size, output_dtype, use_pa)

    # Validate all params have the same shape and dtype.
    numel = params[0].numel()
    dtype = params[0].dtype
    for p in params:
        if p.numel() != numel:
            raise ValueError(
                f"foreach optimizer: all params must have same numel, "
                f"got {numel} vs {p.numel()}"
            )
        if p.dtype != dtype:
            raise ValueError(
                f"foreach optimizer: all params must have same dtype, "
                f"got {dtype} vs {p.dtype}"
            )

    # Build push constants.
    # Layout: uint n_params, uint _pad[3], then ParamConfig[n].
    pc_parts: list[bytes] = []
    pc_parts.append(struct.pack("4I", n_params, 0, 0, 0))

    default_lr = [0.01]
    default_wd = [0.0]
    default_momentum = [0.9]
    default_beta2 = [0.999]
    default_eps = [1e-8]

    lr_list = lr if lr is not None else default_lr * n_params
    wd_list = weight_decay if weight_decay is not None else default_wd * n_params
    mom_list = momentum if momentum is not None else default_momentum * n_params
    b2_list = beta2 if beta2 is not None else default_beta2 * n_params
    eps_list = eps if eps is not None else default_eps * n_params

    for i in range(n_params):
        pc_parts.append(struct.pack("If", params[i].numel(), lr_list[i]))
        pc_parts.append(struct.pack("f", wd_list[i]))
        if algorithm in ("sgd_momentum", "adamw", "lion"):
            pc_parts.append(struct.pack("f", mom_list[i]))
        if algorithm == "adamw":
            pc_parts.append(struct.pack("f", b2_list[i]))
            pc_parts.append(struct.pack("f", eps_list[i]))
        if algorithm == "lion":
            pc_parts.append(struct.pack("f", b2_list[i]))

    # Pad remaining ParamConfig slots for unused entries.
    for _ in range(n_params, batch_size):
        pc_parts.append(struct.pack("Iff", 0, 0.0, 0.0))
        if algorithm in ("sgd_momentum", "adamw", "lion"):
            pc_parts.append(struct.pack("f", 0.0))
        if algorithm == "adamw":
            pc_parts.append(struct.pack("f", 0.0))
            pc_parts.append(struct.pack("f", 0.0))
        if algorithm == "lion":
            pc_parts.append(struct.pack("f", 0.0))

    pc_bytes = b"".join(pc_parts)

    # Build buffer list: [param0, grad0, param1, grad1, ...].
    # Padding slots use a real-storage dummy (size 1, not 0) because the
    # runtime rejects null-storage tensors at dispatch (PF.51).
    dummy = pool_acquire_scratch((1,), dtype, params[0].device)
    if dummy is None:
        dummy = torch.empty(1, device=params[0].device, dtype=dtype)
    buffers: list[torch.Tensor] = []
    for i in range(n_params):
        buffers.append(params[i])
        buffers.append(grads[i])
    for _ in range(n_params, batch_size):
        buffers.append(dummy)  # param slot
        buffers.append(dummy)  # grad slot

    def _pad(lst: list[torch.Tensor] | None) -> list[torch.Tensor]:
        if lst is None:
            return [dummy] * batch_size
        return list(lst) + [dummy] * (batch_size - len(lst))

    # Add momentum / v buffers — always batch_size entries.
    if algorithm == "sgd_momentum":
        buffers.extend(_pad(momentum_bufs))
    elif algorithm == "adamw":
        buffers.extend(_pad(momentum_bufs))
        buffers.extend(_pad(v_bufs))
    elif algorithm == "lion":
        buffers.extend(_pad(momentum_bufs))

    # Grid: X = elements per param, Y = n_params.
    threadgroup_size = 256
    grid_x = (numel + threadgroup_size - 1) // threadgroup_size
    grid_y = n_params
    grid_z = 1

    # num_outputs: all param buffers + momentum/v buffers are mutated.
    num_outputs = n_params
    if algorithm == "sgd_momentum":
        num_outputs += n_params
    elif algorithm == "adamw":
        num_outputs += 2 * n_params
    elif algorithm == "lion":
        num_outputs += n_params

    compile_and_dispatch(
        src,
        buffers,
        grid_x,
        grid_y,
        grid_z,
        push_constants=pc_bytes,
        num_outputs=num_outputs,
        cache_key=cache_key,
    )
    # T6.7: dispatch is done; the padding ``dummy`` is dead.
    buffers.clear()
    pool_release_scratch(dummy)
    dummy = None  # type: ignore[assignment]


class _SlangForeachOptimizer:
    """Picklable callable for a rendered foreach optimizer template.

    Each instance is bound to a specific (algorithm, batch_size,
    output_dtype) combination.  The `__call__` signature takes tensor
    lists and scalar lists and dispatches the template, automatically
    batching across multiple dispatches if there are more params than
    `batch_size`.

    Source is cached at module level per (algorithm, batch_size, output_dtype)
    so multiple instances with the same key share the rendered source.
    """

    __slots__ = (
        "algorithm",
        "batch_size",
        "output_dtype",
        "__name__",
        "_src",
        "_cache_key",
    )

    _src_cache: dict[tuple, tuple[str, str]] = {}

    def __init__(self, algorithm: str, batch_size: int, output_dtype: str = "float"):
        self.algorithm = algorithm
        self.batch_size = batch_size
        self.output_dtype = output_dtype
        self.__name__ = f"slang_foreach_{algorithm}_{batch_size}_{output_dtype}"
        self._src: str | None = None
        self._cache_key: str | None = None
        self._ensure_source()

    def _ensure_source(self) -> None:
        use_pa = _foreach_use_parameter_array()
        key = (self.algorithm, self.batch_size, self.output_dtype, use_pa)
        cached = _SlangForeachOptimizer._src_cache.get(key)
        if cached is not None:
            self._src, self._cache_key = cached
            return
        self._src = _render_foreach_optimizer_slang(
            self.algorithm,
            self.batch_size,
            self.output_dtype,
            parameter_array=use_pa,
        )
        self._cache_key = _foreach_cache_key(
            self.algorithm, self.batch_size, self.output_dtype, use_pa
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
    """
    for bs in _OPTIMIZER_BATCH_SIZES:
        if bs >= n_params:
            key = f"{algorithm}_{bs}_{output_dtype}"
            caller = _foreach_default_callers.get(key)
            if caller is None:
                caller = _SlangForeachOptimizer(algorithm, bs, output_dtype)
                _foreach_default_callers[key] = caller
            return caller
    # Fallback: use the largest predefined size.
    bs = _OPTIMIZER_BATCH_SIZES[-1]
    key = f"{algorithm}_{bs}_{output_dtype}"
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
    template variants."""
    specs: list[tuple[str, str]] = []
    dtypes = ("float", "half")
    algorithms = ("sgd", "sgd_momentum", "adamw", "lion")
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
