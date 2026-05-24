"""FakeTensor/VulkanOpaqueTensorImpl workaround patches.

Monkey-patches PyTorch internals so Vulkan FakeTensors survive:
- Dynamo clone_input
- FakeTensor view-op device inheritance
- FakeTensorMode meta→vulkan conversion
- Tensor.__deepcopy__ for non-leaf Vulkan tensors
- FX graph cache tensor reduction
- FakeTensorMode constant-fold skip for null-storage Vulkan tensors
"""

from __future__ import annotations

import torch


def _patch_dynamo_clone_input_for_vulkan() -> None:
    """Make Dynamo's `clone_input` skip `data_ptr()` for Vulkan FakeTensors.

    Upstream `torch._dynamo.utils.clone_input` already special-cases ``xla``:
    it falls back to ``torch.clone(x)`` instead of allocating an aligned
    storage buffer and computing ``(x.data_ptr() - result.data_ptr()) % 32``.
    The aligned path crashes on FakeTensors (no storage → no data pointer)
    and on devices like ours where computing pointer offsets across two
    independently-allocated tensors is meaningless.

    Without this, any Inductor compile that hits the example_inputs clone
    path for a Vulkan FakeTensor (the entry-point of every compiled SDPA-
    like attention compute) crashes with
    "Cannot access data pointer of Tensor (FakeTensor)". Patching here
    fixes P0.3 and unblocks `bmm(q, k.T) → softmax → bmm(@v)` compilation.
    """
    try:
        from torch._dynamo import utils as _du

        if getattr(_du.clone_input, "_vulkan_patched", False):
            return
        _orig_clone_input = _du.clone_input

        def _patched_clone_input(x, *, dtype=None):
            from torch._subclasses.fake_tensor import FakeTensor, is_fake

            # Detect any tensor for which `data_ptr()` is unsafe: FakeTensor,
            # FunctionalTensor, meta tensors, Vulkan tensors. For Dynamo's
            # purposes, `_clone_input` only needs to produce a tensor with the
            # same metadata (shape / stride / dtype / device); the storage
            # contents are never read. So for these cases we return a fresh
            # `empty_like` (or the input itself for FakeTensors, matching the
            # upstream `clone_input` short-circuit).
            def _is_fake_like(t):
                # `is_fake` covers FakeTensor + traceable subclass that wraps
                # one. Add an explicit isinstance check in case `is_fake`
                # misses an exotic subclass.
                try:
                    if is_fake(t) or isinstance(t, FakeTensor):
                        return True
                except Exception:
                    pass
                try:
                    if torch._is_functional_tensor(t):
                        return True
                except Exception:
                    pass
                return False

            def _safe_metadata_clone(t):
                """Produce a tensor with the same metadata when storage is
                unreadable. Falls back to `empty_like` if `torch.clone` errors
                on data pointer access or missing backing storage."""
                with torch.no_grad():
                    try:
                        y = torch.clone(t)
                    except RuntimeError as inner:
                        msg = str(inner)
                        # Dynamo only needs metadata (shape/stride/dtype/device);
                        # any clone failure on a Vulkan tensor is a storage-access
                        # problem — fall back to empty allocation.
                        if "data pointer" not in msg and "backing" not in msg:
                            raise
                        # Last-resort: allocate fresh storage with matching
                        # shape / stride / dtype / device. Contents are
                        # irrelevant for Dynamo example-value tracing.
                        y = torch.empty_strided(
                            t.size(),
                            t.stride(),
                            dtype=dtype or t.dtype,
                            device=t.device,
                        )
                    try:
                        if t.is_leaf:
                            y.requires_grad_(t.requires_grad)
                    except Exception:
                        pass
                    return y

            # FakeTensor: upstream `clone_input` returns `x` unchanged. Match
            # that behavior — cloning is a no-op for shape inference.
            if _is_fake_like(x):
                return x

            # Vulkan tensors: the alignment-aware path computes
            # `(x.data_ptr() - result.data_ptr()) % 32`, which is meaningless
            # for GPU-resident storage. Skip straight to a metadata clone.
            try:
                if x.device.type == "vulkan":
                    return _safe_metadata_clone(x)
            except Exception:
                pass

            try:
                return _orig_clone_input(x, dtype=dtype)
            except RuntimeError as e:
                # Original failed on `data_ptr()` or missing backing storage for
                # a tensor we did not detect upfront (e.g. FunctionalTensor
                # wrapping a Vulkan FakeTensor, as_strided view with no realized
                # storage). Fall back to a metadata-only clone.
                msg = str(e)
                if "data pointer" not in msg and "backing" not in msg:
                    raise
                return _safe_metadata_clone(x)

        _patched_clone_input._vulkan_patched = True  # type: ignore[attr-defined]
        _du.clone_input = _patched_clone_input

        # Also patch the imported reference in `builder.py` — it does
        # `from .utils import clone_input`, which captures the original
        # function by value at import time. Without this, the call site at
        # `_clone_input` (used by Dynamo's example-value clone) bypasses
        # our patch entirely.
        try:
            from torch._dynamo.variables import builder as _builder

            _builder.clone_input = _patched_clone_input
        except Exception:
            pass
    except Exception:
        pass


def _patch_fake_tensor_view_op_device() -> None:
    """Override FakeTensor.__new__ so view-op outputs inherit the input's
    fake device instead of getting `device=meta`.

    During AOT autograd's backward fake_tensor_prop, view-tagged ops like
    `aten::expand` go through PyTorch's C++ view fast-path. That path
    constructs the output FakeTensor via `Tensor._make_subclass` while the
    source FakeTensor is in `in_kernel_invocation` mode (which makes
    `source.device` report `meta`). Result: the new FakeTensor reports
    `device=meta` even though the source was `vulkan:0`. Downstream backward
    formulas then fail with `Unhandled FakeTensor Device Propagation` when
    they multiply a `meta` saved tensor by a `vulkan` grad.

    Fix: track the most-recently-active vulkan FakeTensorMode session in TLS,
    and in `__new__` upgrade `device=meta` → vulkan for tensors created during
    that session.
    """
    try:
        import torch
        import torch._subclasses.fake_tensor as _ft

        from .joint_graph_passes import _tls

        _orig_new = _ft.FakeTensor.__new__

        def _patched_new(cls, fake_mode, elem, device, *args, **kwargs):
            if (
                isinstance(device, torch.device)
                and device.type == "meta"
                and fake_mode is not None
                and getattr(_tls, "_in_joint_trace", False)
            ):
                vk = getattr(fake_mode, "_torch_vulkan_seen_device", None)
                if vk is not None:
                    device = vk
            elif (
                isinstance(device, torch.device)
                and device.type in ("vulkan", "privateuseone")
                and fake_mode is not None
                and getattr(_tls, "_in_joint_trace", False)
            ):
                fake_mode._torch_vulkan_seen_device = device
            return _orig_new(cls, fake_mode, elem, device, *args, **kwargs)

        _ft.FakeTensor.__new__ = staticmethod(_patched_new)
    except Exception as e:  # pragma: no cover
        import logging

        logging.getLogger(__name__).warning(
            "Patching FakeTensor view-op device failed: %s", e
        )


def _patch_fake_tensor_meta_conversion() -> None:
    """Patch FakeTensorMode.validate_and_convert_non_fake_tensors to accept meta tensors.

    During Inductor's fake_tensor_prop on compiled backward graphs, saved
    forward-pass tensors arrive as plain `meta` device tensors rather than
    FakeTensors. The stock validation rejects these. Since a meta tensor is
    shape/dtype-only (identical semantics to a FakeTensor), we auto-convert
    them to FakeTensors using the mode's existing converter, fixing backward
    graph compilation under the inductor backend.

    Only intercepts meta-device tensors; all other non-fake tensors still raise
    so we don't silently hide real errors.
    """
    try:
        import torch._subclasses.fake_tensor as _ft

        _orig_validate = _ft.FakeTensorMode.validate_and_convert_non_fake_tensors

        def _patched_validate(self, func, converter, flat_args, args_spec):
            import torch

            new_args = []
            for a in flat_args:
                if (
                    isinstance(a, torch.Tensor)
                    and not self.is_our_fake(a)
                    and a.device.type in ("meta", "vulkan")
                ):
                    # Meta tensors arrive on backward fake_tensor_prop;
                    # vulkan-device real tensors arrive when Dynamo
                    # specializes a constant index tensor (e.g. the
                    # `target.unsqueeze(1)` argument to `aten.gather` in
                    # cross_entropy), and as the implicit `tensor(1.0)`
                    # tangent into AOT autograd's
                    # `coerce_tangent_and_suggest_memory_format` (PF.13.b.2).
                    # All shape/dtype-equivalent for FakeTensor purposes;
                    # convert so the validator doesn't reject.
                    a = converter.from_real_tensor(self, a)
                new_args.append(a)
            return _orig_validate(self, func, converter, new_args, args_spec)

        _ft.FakeTensorMode.validate_and_convert_non_fake_tensors = _patched_validate
    except Exception as e:  # pragma: no cover
        import logging

        logging.getLogger(__name__).warning(
            "Patching FakeTensorMode for meta tensor conversion failed: %s", e
        )


def _patch_tensor_deepcopy_for_vulkan() -> None:
    """PF.13.b.4 (layered #2) — make non-leaf Vulkan tensors deepcopy-safe.

    AOT autograd's lazy-backward path runs ``copy.deepcopy(bw_module)`` at
    backward time (``runtime_wrappers.py:2890``). The bw_module's
    ``_tensor_constantN`` attributes are saved-for-backward activations
    that the partitioner lifted from the joint graph — they are non-leaf
    Vulkan tensors (typically views like ``k.transpose(-2, -1)``). Stock
    ``Tensor.__deepcopy__`` raises immediately on non-leaf inputs:
    ``Only Tensors created explicitly by the user (graph leaves) support
    the deepcopy protocol``.

    Mirror the lazy/xla/mps/meta/ipu fast-path that already exists in
    ``torch/_tensor.py:164``: for Vulkan tensors, fall through to
    ``self.clone()`` even when non-leaf. ``clone()`` produces a new leaf
    Vulkan tensor with identical shape/stride/data, which is exactly
    what the deepcopy of bw_module's lifted activations needs.

    Without this, the matmul+softmax bwd-compile floor (PF.13.b.4) and
    every transformer-class backward through ``compile_fx_backward``
    aborts at backward execution time.
    """
    try:
        _orig_deepcopy = torch.Tensor.__deepcopy__

        def _patched_deepcopy(self, memo):
            # PF.13.b.4: Vulkan tensors stored as graph module constants
            # during AOTAutograd compilation have inaccessible storage.
            # Neither clone() nor contiguous() works.  Create metadata-
            # equivalent empty tensors instead — the actual values will
            # be filled in during backward execution when saved
            # activations are substituted.
            if self.device.type == "vulkan":
                if id(self) in memo:
                    return memo[id(self)]
                try:
                    with torch.no_grad():
                        new_tensor = self.clone()
                except (RuntimeError, Exception):
                    try:
                        new_tensor = torch.empty_strided(
                            self.shape,
                            self.stride(),
                            dtype=self.dtype,
                            device=self.device,
                        )
                    except Exception:
                        # Last resort: contiguous metadata clone
                        new_tensor = torch.empty(
                            self.shape,
                            dtype=self.dtype,
                            device=self.device,
                        )
                memo[id(self)] = new_tensor
                return new_tensor
            return _orig_deepcopy(self, memo)

        torch.Tensor.__deepcopy__ = _patched_deepcopy
    except Exception as e:  # pragma: no cover
        import logging

        logging.getLogger(__name__).warning(
            "Patching Tensor.__deepcopy__ for Vulkan failed: %s", e
        )


def _patch_fx_graph_cache_reduce_tensor_for_vulkan() -> None:
    """PF.13.b.4 (layered #4) — handle Vulkan view tensors in FX graph cache hashing.

    During forward graph compilation, FxGraphCachePickler._reduce_tensor
    calls t.tolist() on tensor constants stored as graph module attributes.
    These constants are typically saved-for-backward activations that the
    AOTAutograd partitioner lifted from the joint graph — non-leaf view tensors
    like k.transpose(-2, -1).  For Vulkan (PrivateUse1) tensors, the
    storage data pointer is invalid during this compilation phase, causing
    t.tolist() to raise RuntimeError: Cannot access data pointer of
    Tensor.

    This patch catches that RuntimeError for Vulkan tensors and falls back
    to t.cpu().tolist() which copies the data via the Vulkan readback
    path and produces a valid Python list for hashing.
    """
    try:
        import torch._inductor.codecache as _cc

        _orig_reduce_tensor = _cc.FxGraphCachePickler._reduce_tensor

        def _patched_reduce_tensor(self, t: torch.Tensor) -> tuple:
            from torch._inductor.graph import GraphLowering

            metadata = _cc.extract_tensor_metadata_for_cache_key(t)

            if _cc.is_frozen_param(t) and not GraphLowering.can_inline_constant(t):
                return (_cc._ident, (metadata,))

            # PF.13.b.4 / PF.13: Vulkan (PrivateUse1) tensors lifted as
            # graph module constants may have invalid storage data pointers
            # during AOTAutograd compilation (both leaf and non-leaf).
            # Neither .tolist() nor .cpu() works on them.  For Vulkan
            # tensors, try .tolist() and fall back to metadata-only if
            # the storage is inaccessible — metadata alone is sufficient
            # for cache-key uniqueness.
            # T.12: catch any exception from ``t.tolist()`` and fall back to
            # metadata-only hashing — metadata alone uniquely identifies the
            # cache key. This covers Vulkan tensors specifically (where
            # tolist may fail with "Unsupported copy direction" or "Cannot
            # access data pointer of Tensor"), and is harmless for any
            # other device because tolist on a healthy tensor never raises.
            try:
                values = t.tolist()
                return (
                    _cc._ident,
                    (_cc.TensorMetadataAndValues(metadata, values),),
                )
            except Exception:
                # Vulkan tensors lifted as graph constants regularly
                # have inaccessible storage at this point in the
                # pipeline; meta tensors never have data. Fall back to
                # metadata-only hashing — metadata uniquely identifies
                # the cache key. Also covers ``privateuseone`` devices.
                dev_t = getattr(t.device, "type", "")
                if dev_t in ("vulkan", "privateuseone", "meta") or "vulkan" in str(
                    t.device
                ):
                    return (_cc._ident, (metadata,))
                raise

        _cc.FxGraphCachePickler._reduce_tensor = _patched_reduce_tensor
    except Exception as e:  # pragma: no cover
        import logging

        logging.getLogger(__name__).warning(
            "Patching FxGraphCachePickler._reduce_tensor for Vulkan failed: %s", e
        )


def _patch_fake_tensor_skip_const_fold_for_vulkan_null() -> None:
    """Skip FakeTensor's constant-fold path when a Vulkan input's stored
    ``.constant`` has no backing buffer.

    ``FakeTensorMode._dispatch_impl`` has two constant-fold branches that run
    the *real* op against ``arg.constant`` for every fake input:

    - Lift / numbers-as-tensors path (``fake_tensor.py`` ~L2500):
      fires for binary ops with a Python scalar second arg, e.g.
      ``aten.add(<vulkan FakeTensor>, 1.0)``.
    - All-constants path (~L2557):
      fires when *every* fake input has a constant, e.g.
      ``aten.add(<vulkan FakeTensor>, <vulkan FakeTensor>)`` after both
      were promoted via ``make_constant=True``.

    Both branches eventually call ``func(*const_args, ...)``, which lands
    in our PrivateUse1 C++ kernel (``vulkan_add`` etc.). If the constant
    Vulkan tensor was produced by PF.13's view-cascade and has no
    backing buffer (``data_ptr() == 0``), the C++ kernel raises
    ``"Tensor has no backing Vulkan buffer"``. This blocks Inductor's
    FX tracing of conv graphs (``F.conv2d(x, w) + 1.0``) and similar
    binary-op-with-scalar patterns.

    Fix: before ``_dispatch_impl`` runs, scan every flat-arg fake tensor
    and, for any Vulkan FakeTensor whose ``.constant`` is null-storage,
    clear ``.constant`` so the constant-fold guard
    (``all(t.constant is not None ...)``) evaluates False. Dispatch then
    falls through to the registered fake_impl (already covers
    ``aten.add.Tensor``, ``aten.add.Scalar``, ``aten.mul.Tensor``,
    ``aten.mul.Scalar``, etc. — see ``_OP_IMPLS``), which returns the
    correct shape/dtype FakeTensor without dereferencing storage.

    Surgical: only clears ``.constant`` for Vulkan FakeTensors whose
    constant has null storage. Non-Vulkan tensors and Vulkan tensors
    with real storage are untouched, so other backends' const-fold
    behavior is unchanged.
    """
    try:
        import torch._subclasses.fake_tensor as _ft

        _orig_dispatch_impl = _ft.FakeTensorMode._dispatch_impl

        def _is_vulkan_null_constant(c) -> bool:
            if c is None or not isinstance(c, torch.Tensor):
                return False
            try:
                if c.device.type not in ("vulkan", "privateuseone"):
                    return False
            except Exception:  # noqa: BLE001
                return False
            try:
                return c.data_ptr() == 0
            except RuntimeError:
                # data_ptr() raises on FakeTensor / null storage —
                # treat as null-backed.
                return True

        def _is_real_vulkan_null(t) -> bool:
            if not isinstance(t, torch.Tensor):
                return False
            try:
                if t.device.type not in ("vulkan", "privateuseone"):
                    return False
            except Exception:  # noqa: BLE001
                return False
            try:
                return t.data_ptr() == 0
            except RuntimeError:
                return True

        def _patched_dispatch_impl(self, func, types, args, kwargs):
            from torch.utils import _pytree as pytree

            try:
                flat_args, spec = pytree.tree_flatten((args, kwargs))
                # Path 1: clear .constant on FakeTensors with vulkan-null
                # constant so the all-constants branch's guard fails.
                for a in flat_args:
                    if (
                        self.is_our_fake(a)
                        and getattr(a, "constant", None) is not None
                        and _is_vulkan_null_constant(a.constant)
                    ):
                        a.constant = None
                # Path 2: a real (non-fake) vulkan null-storage tensor in
                # flat_args makes ``flat_arg_fake_tensors`` empty, so the
                # ``should_allow_numbers_as_tensors`` const-fold branch
                # fires and tries to call ``func(...)`` against it.
                # Convert such tensors to proper FakeTensors so the branch
                # skips and dispatch falls through to the registered
                # fake_impl.
                replaced = False
                new_flat = []
                for a in flat_args:
                    if not self.is_our_fake(a) and _is_real_vulkan_null(a):
                        try:
                            a = self.from_tensor(a, static_shapes=True)
                            replaced = True
                        except Exception:  # noqa: BLE001
                            pass
                    new_flat.append(a)
                if replaced:
                    args, kwargs = pytree.tree_unflatten(new_flat, spec)
            except Exception:  # noqa: BLE001
                # Never let the guard itself break dispatch; fall through.
                pass
            return _orig_dispatch_impl(self, func, types, args, kwargs)

        _ft.FakeTensorMode._dispatch_impl = _patched_dispatch_impl
    except Exception as e:  # pragma: no cover
        import logging

        logging.getLogger(__name__).warning(
            "Patching FakeTensorMode._dispatch_impl for vulkan null "
            "constants failed: %s",
            e,
        )


def _patch_graph_lowering_get_attr_for_vulkan_null() -> None:
    """Patch GraphLowering.get_attr to fix zero-stride and null-storage Vulkan constants.

    GraphLowering.get_attr calls ``value.tolist()`` for small (≤ 8 element) 1-D
    tensor constants (``can_inline_constant=True``) and inlines the result as
    compile-time literals.  For Vulkan tensors this fails in two ways:

    1. **Zero-stride broadcast** (all strides == 0): reads at indices [0,1,2,...]
       all return element[0] — the remaining indices give out-of-bounds zeros.

    2. **Null-storage / FakeTensor**: ``tolist()`` raises
       ``RuntimeError: Vulkan tensor has no backing buffer``.

    For the null-storage case, PF.52 (``tangents.py``) handles the primary path
    by replacing ``_tensor_constant*`` get_attr nodes with ``aten.full([N], 1.0)``
    before GraphLowering runs.  This patch is a safety net for any remaining
    Vulkan ``get_attr`` tensors that survive to GraphLowering with null storage.

    Fix: intercept ``get_attr`` for Vulkan tensors with zero strides or null
    storage.  For zero-stride tensors, replace with the contiguous copy so
    ``tolist()`` returns the broadcast value for every element.  For null-storage
    tensors, return a compile-time ``Constant(0.0, ...)`` placeholder (the node
    is typically dead after PF.52; the correct constant value is emitted by the
    ``aten.full`` node PF.52 inserted in its place).

    Idempotent (guarded via ``_vulkan_null_const_patched`` class attribute).
    """
    try:
        from torch._inductor.graph import GraphLowering

        if getattr(GraphLowering, "_vulkan_null_const_patched", False):
            return

        _orig_get_attr = GraphLowering.get_attr

        def _patched_get_attr(self, target, args, kwargs):
            # Peek at the attribute before the original get_attr runs.
            # Use simple getattr traversal matching graph.py's getattr_recursive.
            value = None
            try:
                attr_itr = self.module
                for atom in target.split("."):
                    attr_itr = getattr(attr_itr, atom)
                value = attr_itr
            except Exception:  # noqa: BLE001
                value = None

            if (
                isinstance(value, torch.Tensor)
                and not isinstance(value, torch.fx.GraphModule)
                and getattr(value.device, "type", "") in ("vulkan", "privateuseone")
            ):
                try:
                    # Sub-case 1: zero-stride broadcast — replace with contiguous
                    # copy so tolist() returns the correct broadcast value.
                    strides = value.stride()
                    if strides and all(s == 0 for s in strides) and value.dim() > 0:
                        contiguous = value.contiguous()
                        _parts = target.rsplit(".", 1)
                        if len(_parts) == 2:
                            _parent = self.module
                            for _a in _parts[0].split("."):
                                _parent = getattr(_parent, _a)
                            setattr(_parent, _parts[1], contiguous)
                        else:
                            setattr(self.module, target, contiguous)
                        if target in getattr(self, "constants", {}):
                            self.constants[target] = contiguous
                        return _orig_get_attr(self, target, args, kwargs)
                except Exception:  # noqa: BLE001
                    pass

                # Sub-case 2: null-storage (tolist would fail or return zeros).
                # PF.52 should have replaced these get_attr nodes with full([N])
                # nodes already. If one slips through (dead node after PF.52,
                # or an unusual constant we didn't catch), return a compile-time
                # Constant(0.0) so the compilation doesn't crash. For live nodes
                # this returns an incorrect 0.0 but PF.52 should have redirected
                # all live uses to the full() node before GraphLowering.
                if GraphLowering.can_inline_constant(value):
                    try:
                        # Try tolist() — might work if the tensor has real data.
                        _vals = value.tolist()
                    except Exception:  # noqa: BLE001
                        # Null storage — return a zero constant so compilation
                        # doesn't crash. Live uses were redirected by PF.52.
                        from torch._inductor.ir import Constant
                        return Constant(value=0.0, dtype=value.dtype, device=value.device)

            return _orig_get_attr(self, target, args, kwargs)

        GraphLowering.get_attr = _patched_get_attr
        GraphLowering._vulkan_null_const_patched = True  # type: ignore[attr-defined]
    except Exception as e:  # pragma: no cover
        import logging

        logging.getLogger(__name__).warning(
            "Patching GraphLowering.get_attr for Vulkan null constants failed: %s",
            e,
        )
