"""Batch dispatch submission — DispatchBatcher context manager.

GPU.1 — When the wrapper codegen enters this context manager, calls to
``add(kernel_fn, *args)`` are collected instead of dispatched immediately.
On ``__exit__``, all pending dispatches are submitted back-to-back with
minimal Python overhead between them and, when C++ batch mode is available,
in a single ``vkQueueSubmit``.
"""


import torch


class DispatchBatcher:
    """Batch multiple kernel dispatches into a single Vulkan submission.

    GPU.1 — When the wrapper codegen enters this context manager, calls to
    ``add(kernel_fn, *args)`` are collected instead of dispatched immediately.
    On ``_flush()`` / ``__exit__``, all pending dispatches are sent while C++
    batch mode (M17.5) is active, so they accumulate in a single command buffer
    and are submitted together with one ``vkQueueSubmit``.

    Usage (emitted by wrapper codegen)::

        _batcher = DispatchBatcher()
        with _batcher:
            _batcher.add(kernel_0, arg0_0, arg0_1, ...)
            _batcher.add(kernel_1, arg1_0, arg1_1, ...)
        # All dispatches submitted on context exit.
    """

    # Cached lookup of C++ batch-mode functions (lazy, once per process).
    _begin_batch = None
    _end_batch = None
    _batch_probed: bool = False

    # M-NEW.12: class-level "current active" batcher reference so direct-call
    # template callers (``_slang_tile_conv2d`` / ``_slang_tile_mm`` — which are
    # emitted by the wrapper as immediate Python function calls, NOT routed
    # through ``_batcher.add``) can flush queued dispatches before they read
    # from buffers populated by a still-queued kernel. Without this, the
    # second conv2d in a ``maxpool → conv2`` chain reads a zero-initialised
    # buf2 (the maxpool kernel hasn't run yet) and emits bias-only output.
    _current: "DispatchBatcher | None" = None

    def __init__(self):
        self._pending: list[tuple] = []  # (kernel_callable, args_tuple)
        self._active: bool = False
        self._batch_active: bool = False  # C++ batch mode engaged

    def __enter__(self):
        self._active = True
        self._pending.clear()
        # M-NEW.12: register as the current active batcher so direct-call
        # template invocations can locate us and flush before dispatching.
        DispatchBatcher._current = self
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
        # M-NEW.12: clear current-batcher reference iff it still points to us
        # (defensive against nested batchers that may share the slot).
        if DispatchBatcher._current is self:
            DispatchBatcher._current = None
        return False  # propagate exceptions

    @classmethod
    def flush_current_if_active(cls) -> None:
        """Flush any queued dispatches on the currently active batcher.

        Direct-call template helpers (``_slang_tile_conv2d`` etc.) emitted by
        custom lowerings as immediate Python function calls must invoke this
        before dispatching, because the wrapper's ``_batcher.add(...)`` queue
        defers writes that the direct call expects to read from. Without
        this flush, a queued ``MaxPool2d`` kernel filling ``buf2`` will not
        have run when the immediate ``_slang_tile_conv2d(buf2, ..., out)``
        reads ``buf2`` — yielding a stale (zero-initialised) read.
        """
        cur = cls._current
        if cur is None or not cur._active:
            return
        if cur._pending:
            cur._flush()

    def add(self, kernel_handle, *dispatch_args):
        """Collect a kernel dispatch for batched submission.

        When the batcher is active (inside a ``with`` block), the call is
        queued.  When inactive, dispatches immediately.

        Inductor guarantees topological order in generated wrapper code, so
        no dependency-tracking is needed here — kernels are appended and
        dispatched in the order they arrive.
        """
        if not self._active:
            kernel_handle(*dispatch_args)
            return
        self._pending.append((kernel_handle, dispatch_args))

    def _flush(self):
        """Submit all pending dispatches, then re-enter batch mode.

        When C++ batch mode is active (M17.5), dispatches pending kernels
        while batch mode is still ON (so they accumulate in the command
        buffer), then calls ``_end_batch()`` to submit everything in a single
        ``vkQueueSubmit``.  Previously the order was reversed: ``_end_batch()``
        was called first (submitting an empty or partial buffer), then pending
        kernels were replayed in auto-flush mode — defeating the purpose of
        batching.

        Falls back to sequential individual dispatches when the C++ batch
        functions are unavailable.
        """
        if not self._pending:
            return

        if self._batch_active and self._end_batch is not None:
            # Dispatch while batch mode is active — these go into the C++
            # command buffer alongside any direct-dispatch calls already there.
            for kernel_handle, dispatch_args in self._pending:
                kernel_handle(*dispatch_args)
            self._pending.clear()
            # Submit the accumulated command buffer as one vkQueueSubmit.
            self._end_batch()
            self._batch_active = False
            # Re-enter batch mode so dispatches after the sync point
            # (the extern kernel and subsequent _jit_dispatch calls) accumulate
            # in a fresh command buffer instead of doing per-dispatch submits.
            if self._begin_batch is not None:
                try:
                    self._begin_batch()
                    self._batch_active = True
                except Exception:
                    self._batch_active = False
        else:
            # Sequential replay: no batch mode or no C++ batch functions.
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
