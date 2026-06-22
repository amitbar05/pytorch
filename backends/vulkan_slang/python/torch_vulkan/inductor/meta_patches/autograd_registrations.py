"""AutogradPrivateUse1 Python impls for view/permute/activation ops.

Registers Python kernels on AutogradPrivateUse1 that run BEFORE the C++
adapter, computing view/permute/transpose/t in pure Python so that
FakeTensorMode produces correct view aliases and grad_fn attachments.
"""

from __future__ import annotations

import torch

_view_symint_autograd_lib = None


def _register_view_symint_autograd_pyimpl() -> None:
    """PF.55 — Python ``AutogradPrivateUse1`` impl for view/reshape ops so
    SymInt-bearing sizes don't crash the C++ adapter's ``expect_int()``.

    Why: under ``torch.compile(..., dynamic=True)``, Dynamo's FakeTensor
    propagation calls ``Tensor.flatten`` / ``Tensor.view`` / ``Tensor.reshape``
    on a Vulkan FakeTensor whose ``size()`` carries symbolic SymInts.
    The dispatcher's first hit on the FakeTensor's keyset is
    ``AutogradPrivateUse1`` (always present in the autograd-aware
    keyset). Our C++ ``vulkan_view_autograd_adapter`` (and
    ``vulkan_reshape_autograd_adapter``, ``vulkan_view_adapter``,
    ``vulkan_reshape_adapter``) eagerly converts ``SymIntArrayRef`` to
    ``IntArrayRef`` via ``symint_to_int`` → ``SymInt::expect_int()``,
    which throws ``"when unpacking SymInt, expected int but got s33"``.
    Python ``__torch_dispatch__`` cannot intercept it because
    ``PythonFallbackKernel`` is registered as a *backend fallback* —
    only consulted when no explicit kernel exists at the active
    backend's keyset, which our C++ adapter pre-empts.

    Fix: register a Python kernel on ``AutogradPrivateUse1`` that runs
    *before* the C++ adapter. When the size carries any symbolic SymInt
    (or the input itself has symbolic shape), do shape inference in
    pure Python and return a fake-friendly tensor with the correct
    symbolic shape via ``self.new_empty``. When all sizes are concrete,
    redispatch under ``_ExcludeDispatchKeyGuard(AutogradPrivateUse1)``
    — the dispatcher then walks down to ``PrivateUse1`` where the
    untouched ``vulkan_view_adapter`` / ``vulkan_reshape_adapter`` runs
    eagerly with concrete ints (preserving zero-copy semantics).

    Note: we deliberately do NOT register at ``PrivateUse1`` because
    that would shadow the C++ adapter and we'd lose the ability to
    delegate concrete-size views to the eager path. The
    ``AutogradPrivateUse1`` key is always in a Vulkan FakeTensor's
    keyset (autograd is always live during Dynamo trace), so this one
    registration suffices to intercept all SymInt-bearing view calls.

    Covers ``aten::view``, ``aten::reshape``, and ``aten::_unsafe_view``
    — every shape op that takes a ``SymInt[]`` argument and dispatches
    to our adapters under FakeTensorMode.

    PF.13.b.4 fix: the kernel must wrap the result in a
    ``torch.autograd.Function`` so the output carries a backward
    grad_fn. Registering directly on ``AutogradPrivateUse1`` *replaces*
    the autograd kernel, so without the explicit Function wrap the
    dispatcher never attaches grad_fn — which causes the
    ``aten.matmul`` decomposition (``reshape → bmm → _unsafe_view``)
    to detach during fake-tensor metadata collection. AOTAutograd then
    flips ``needs_autograd = False`` (because ``output_info[i].requires_grad``
    is False everywhere), compiles as inference, and ``loss.backward()``
    on the compiled output raises ``element 0 of tensors does not
    require grad and does not have a grad_fn``. Mirrors the C++
    ``VulkanViewFunction`` / ``VulkanReshapeFunction`` shape-only
    backward in ``csrc/ops/autograd_ops.cpp:1520``.
    """
    global _view_symint_autograd_lib
    if _view_symint_autograd_lib is not None:
        return
    try:

        def _has_symint(size) -> bool:
            return any(isinstance(s, torch.SymInt) for s in size)

        def _resolve_view_size(self, size):
            """Compute output shape, inferring -1 against symbolic numel."""
            sizes = list(size)
            inferred = -1
            for i, s in enumerate(sizes):
                if isinstance(s, int) and s == -1:
                    if inferred != -1:
                        raise RuntimeError("only one -1 allowed in view")
                    inferred = i
            if inferred == -1:
                return sizes
            input_numel = 1
            for d in self.size():
                input_numel = input_numel * d
            other = 1
            for j, s in enumerate(sizes):
                if j != inferred:
                    other = other * s
            sizes[inferred] = input_numel // other
            return sizes

        _OP_DEFAULTS = {
            "view": torch.ops.aten.view.default,
            "reshape": torch.ops.aten.reshape.default,
            "_unsafe_view": torch.ops.aten._unsafe_view.default,
        }

        class _VulkanViewSymIntAutogradFn(torch.autograd.Function):
            """Re-attach grad_fn to view/reshape/_unsafe_view results.

            ``forward`` runs the underlying op (or a symbolic-shape
            new_empty for SymInt sizes); ``backward`` reshapes the grad
            back to the input's original shape — the exact contract of
            ``VulkanViewFunction::backward`` in C++ autograd_ops.cpp.
            """

            @staticmethod
            # pyrefly: ignore [bad-override]
            def forward(ctx, self, size, op_name):
                ctx.input_sizes = list(self.size())
                if _has_symint(size) or _has_symint(self.size()):
                    resolved = _resolve_view_size(self, size)
                    return self.new_empty(resolved)
                with torch._C._ExcludeDispatchKeyGuard(
                    torch._C.DispatchKeySet(torch._C.DispatchKey.AutogradPrivateUse1)
                ):
                    return _OP_DEFAULTS[op_name](self, size)

            @staticmethod
            def backward(ctx, grad):
                # ``reshape`` (not ``view``) handles non-contiguous grads
                # from upstream transpose / permute chains.
                return grad.reshape(ctx.input_sizes), None, None

        def _make_pyimpl(op_name):
            def _impl(self, size):
                return _VulkanViewSymIntAutogradFn.apply(self, size, op_name)

            return _impl

        _view_symint_autograd_lib = torch.library.Library(
            "aten", "IMPL", "AutogradPrivateUse1"
        )
        # Silences the one-shot "Overriding a previously registered kernel"
        # diagnostic — the override is intentional (replaces the default
        # LegacyBatchingRegistrations kernel for AutogradPrivateUse1).
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.filterwarnings(
                "ignore",
                message="Warning only once for all operators",
                category=UserWarning,
            )
            _view_symint_autograd_lib.impl("view", _make_pyimpl("view"))
            _view_symint_autograd_lib.impl("reshape", _make_pyimpl("reshape"))
            _view_symint_autograd_lib.impl("_unsafe_view", _make_pyimpl("_unsafe_view"))
    except Exception as e:  # pragma: no cover
        import logging

        logging.getLogger(__name__).warning(
            "Registering view symint pyimpl failed: %s", e
        )


_permute_family_autograd_lib: torch.library.Library | None = None


def _register_permute_family_autograd_pyimpl() -> None:
    """Register Python ``AutogradPrivateUse1`` impls for ``permute`` /
    ``transpose.int`` / ``t`` so they produce **proper view aliases** under
    FakeTensorMode.

    The C++ ``vulkan_permute`` (and its autograd wrapper) constructs a fresh
    output via ``at::empty`` (or ``make_vulkan_null`` on null storage). The
    result does not alias ``self``'s storage. Under FakeTensorMode the
    autograd graph captured by AOTAutograd then sees ``permute(arg)`` as
    input-independent — AOTAutograd lifts the result as a frozen tensor
    constant, and Inductor's ``constant_fold_uniform_value`` pass folds it
    to ``aten.full(uniform_value)``. The uniform value is whatever
    uninitialized memory happens to hold (commonly ``1.0f`` from a freed
    upstream allocation), producing kernels that write a constant 1.0 into
    every output slot.

    The fix mirrors the existing ``_register_view_symint_autograd_pyimpl``
    pattern: register a Python ``AutogradPrivateUse1`` kernel that runs
    *before* the C++ adapter, computes the permuted size+stride in pure
    Python, and dispatches to ``aten.as_strided`` (which has a built-in
    ``Meta`` kernel that produces a proper view aliasing ``self``'s
    storage). The autograd backward applies the inverse permutation.

    Without this fix, any compiled graph containing ``permute`` /
    ``transpose`` / ``t`` followed by a materialization (e.g. ``+ 0.0``,
    ``reshape``, ``contiguous``) silently returns wrong values
    (T.12.A).
    """
    global _permute_family_autograd_lib
    if _permute_family_autograd_lib is not None:
        return
    try:

        def _as_strided_view(self, sizes, strides):
            """Dispatch to ``aten.as_strided`` with AutogradPrivateUse1
            excluded so the standard autograd machinery (AsStridedBackward)
            handles the gradient instead of recursing into our wrapper.
            """
            with torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(torch._C.DispatchKey.AutogradPrivateUse1)
            ):
                return torch.ops.aten.as_strided.default(self, sizes, strides)

        class _VulkanPermutePyFn(torch.autograd.Function):
            @staticmethod
            # pyrefly: ignore [bad-override]
            def forward(ctx, self, dims):
                ndim = self.dim()
                ctx.dims = list(dims)
                mapped = [d if d >= 0 else ndim + d for d in dims]
                sizes = [int(self.size(i)) for i in mapped]
                strides = [int(self.stride(i)) for i in mapped]
                return _as_strided_view(self, sizes, strides)

            @staticmethod
            def backward(ctx, grad):
                ndim = len(ctx.dims)
                mapped = [d if d >= 0 else ndim + d for d in ctx.dims]
                inv = [0] * ndim
                for i, d in enumerate(mapped):
                    inv[d] = i
                return torch.ops.aten.permute.default(grad, inv), None

        class _VulkanTransposePyFn(torch.autograd.Function):
            @staticmethod
            # pyrefly: ignore [bad-override]
            def forward(ctx, self, dim0, dim1):
                ndim = self.dim()
                d0 = dim0 if dim0 >= 0 else ndim + dim0
                d1 = dim1 if dim1 >= 0 else ndim + dim1
                ctx.dim0 = dim0
                ctx.dim1 = dim1
                if d0 == d1:
                    return _as_strided_view(
                        self, list(self.size()), list(self.stride())
                    )
                sizes = list(self.size())
                strides = list(self.stride())
                sizes[d0], sizes[d1] = sizes[d1], sizes[d0]
                strides[d0], strides[d1] = strides[d1], strides[d0]
                return _as_strided_view(self, sizes, strides)

            @staticmethod
            def backward(ctx, grad):
                return (
                    torch.ops.aten.transpose.int(grad, ctx.dim0, ctx.dim1),
                    None,
                    None,
                )

        class _VulkanTPyFn(torch.autograd.Function):
            @staticmethod
            # pyrefly: ignore [bad-override]
            def forward(ctx, self):
                ctx.ndim = self.dim()
                if self.dim() < 2:
                    return _as_strided_view(
                        self, list(self.size()), list(self.stride())
                    )
                return _VulkanTransposePyFn.apply(self, 0, 1)

            @staticmethod
            def backward(ctx, grad):
                if ctx.ndim < 2:
                    return grad
                return torch.ops.aten.t.default(grad)

        def _permute_pyimpl(self, dims):
            return _VulkanPermutePyFn.apply(self, dims)

        def _transpose_pyimpl(self, dim0, dim1):
            return _VulkanTransposePyFn.apply(self, dim0, dim1)

        def _t_pyimpl(self):
            return _VulkanTPyFn.apply(self)

        _permute_family_autograd_lib = torch.library.Library(
            "aten", "IMPL", "AutogradPrivateUse1"
        )
        _permute_family_autograd_lib.impl("permute", _permute_pyimpl)
        _permute_family_autograd_lib.impl("transpose.int", _transpose_pyimpl)
        _permute_family_autograd_lib.impl("t", _t_pyimpl)
    except Exception as e:  # pragma: no cover
        import logging

        logging.getLogger(__name__).warning(
            "Registering permute family autograd pyimpl failed: %s", e
        )


