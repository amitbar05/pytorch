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

    # Cached lookup of C++ batch-mode functions (lazy, once per process).
    _begin_batch = None
    _end_batch = None
    _batch_probed: bool = False

    def __init__(self):
        self._pending: list[tuple] = []  # (kernel_callable, args_tuple)
        self._active: bool = False
        self._batch_active: bool = False  # C++ batch mode engaged

    def __enter__(self):
        self._active = True
        self._pending.clear()
        # M17.5: Engage C++ batch mode to suppress per-8-dispatch auto-flush.
        # When batch mode is active, all dispatches accumulate in a single
        # command buffer and are submitted together on __exit__.
        self._ensure_batch_resolved()
        if self._begin_batch is not None:
            try:
                self._begin_batch()
                self._batch_active = True
            except Exception:
                self._batch_active = False
        return self

    def __exit__(self, *args):
        self._active = False
        if self._pending:
            self._flush()
        # M17.5: Disengage C++ batch mode — flush remaining dispatches.
        if self._batch_active and self._end_batch is not None:
            try:
                self._end_batch()
            except Exception:
                pass
            self._batch_active = False
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

        When C++ batch mode is active (M17.5), the ``begin_batch_dispatch``
        call has suppressed per-8-dispatch auto-flush, so all dispatches
        accumulate in a single command buffer.  We replay them sequentially
        here, then ``end_batch_dispatch`` (called in ``__exit__``) flushes
        with a single ``vkQueueSubmit``.

        Falls back to sequential individual dispatches with per-8-dispatch
        auto-flush when the C++ batch functions are unavailable.
        """
        if not self._pending:
            return

        # Sequential replay: each kernel_handle calls through to
        # dispatch_shader() which, with batch_mode=true, skips auto-flush.
        # All dispatches accumulate in one command buffer.
        for kernel_handle, dispatch_args in self._pending:
            kernel_handle(*dispatch_args)
        self._pending.clear()

    @classmethod
    def _ensure_batch_resolved(cls):
        """Lazily resolve the C++ batch-mode entry points."""
        if cls._batch_probed:
            return
        cls._batch_probed = True
        try:
            from torch_vulkan import _C as _c

            cls._begin_batch = getattr(_c, "begin_batch_dispatch", None)
            cls._end_batch = getattr(_c, "end_batch_dispatch", None)
        except Exception:
            cls._begin_batch = None
            cls._end_batch = None
