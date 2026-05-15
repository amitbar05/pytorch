"""Batch dispatch submission — DispatchBatcher context manager.

GPU.1 — When the wrapper codegen enters this context manager, calls to
``add(kernel_fn, *args)`` are collected instead of dispatched immediately.
On ``__exit__``, all pending dispatches are submitted back-to-back with
minimal Python overhead between them.
"""


class DispatchBatcher:
    """Batch multiple kernel dispatches into a single Vulkan submission.

    GPU.1 — When the wrapper codegen enters this context manager, calls to
    ``add(kernel_fn, *args)`` are collected instead of dispatched immediately.
    On ``__exit__``, all pending dispatches are submitted back-to-back with
    minimal Python overhead between them.

    Usage (emitted by wrapper codegen)::

        _batcher = DispatchBatcher()
        with _batcher:
            _batcher.add(kernel_0, arg0_0, arg0_1, ...)
            _batcher.add(kernel_1, arg1_0, arg1_1, ...)
        # All dispatches submitted on context exit.

    When the C++ ``_jit_dispatch_batch`` FFI is available, all dispatches are
    recorded into a single Vulkan command buffer and submitted with one
    ``vkQueueSubmit``.  Otherwise falls back to sequential individual dispatches
    (still beneficial: eliminates Python bytecode overhead between dispatches).
    """

    # Cached lookup of the batch FFI entry point (lazy, once per process).
    _batch_ffi = None
    _batch_ffi_probed: bool = False

    def __init__(self):
        self._pending: list[tuple] = []  # (kernel_callable, args_tuple)
        self._active: bool = False

    def __enter__(self):
        self._active = True
        self._pending.clear()
        return self

    def __exit__(self, *args):
        self._active = False
        if self._pending:
            self._flush()
        return False  # propagate exceptions

    def add(self, kernel_handle, *dispatch_args):
        """Collect a kernel dispatch for batched submission.

        When the batcher is active (inside a ``with`` block), the call is
        queued.  When inactive, dispatches immediately — this ensures
        correctness for callers that do not nest inside the batcher.
        """
        if self._active:
            self._pending.append((kernel_handle, dispatch_args))
        else:
            # Dispatch immediately (non-batched path).
            kernel_handle(*dispatch_args)

    def _flush(self):
        """Submit all pending dispatches.

        Tries the C++ ``_jit_dispatch_batch`` fast path first (single
        ``vkQueueSubmit`` for all kernels).  Falls back to sequential
        individual dispatches when the C++ FFI is unavailable.
        """
        if not self._pending:
            return

        # Try the C++ batch FFI path.
        batch_fn = self._resolve_batch_ffi()
        if batch_fn is not None:
            try:
                handles = [h for h, _ in self._pending]
                arg_lists = [list(a) for _, a in self._pending]
                batch_fn(handles, arg_lists)
                self._pending.clear()
                return
            except Exception:
                # Fall through to sequential path on any C++ error.
                pass

        # Sequential fallback: call each kernel in a tight loop.
        # Still beneficial vs. full Python wrapper overhead per dispatch.
        for kernel_handle, dispatch_args in self._pending:
            kernel_handle(*dispatch_args)
        self._pending.clear()

    @classmethod
    def _resolve_batch_ffi(cls):
        """Lazily resolve the ``_jit_dispatch_batch`` C++ FFI entry."""
        if cls._batch_ffi_probed:
            return cls._batch_ffi
        cls._batch_ffi_probed = True
        try:
            from torch_vulkan import _C as _c

            cls._batch_ffi = getattr(_c, "_jit_dispatch_batch", None)
        except Exception:
            cls._batch_ffi = None
        return cls._batch_ffi
