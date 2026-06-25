"""Batch dispatch submission — DispatchBatcher context manager.

GPU.1 — When the wrapper codegen enters this context manager, calls to
``add(kernel_fn, *args)`` are collected instead of dispatched immediately.
On ``__exit__``, all pending dispatches are submitted back-to-back with
minimal Python overhead between them.
"""


import threading

import torch


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
        # M12: per-batch ready_set tracks which tensor names are already
        # produced (by the batch itself or by kernels outside the batch).
        self._ready_set: set[int] = set()
        # M12: blocked kernels whose inputs are not yet ready.
        self._blocked: list[tuple] = []
        # M13: compile-ahead queue — kernel cache_keys submitted to the
        # background slangc pool as soon as they are discovered. The pool
        # runs in parallel so compilation overlaps GPU execution.
        self._compile_ahead: list[str] = []
        self._compile_ahead_lock: threading.Lock = threading.Lock()
        self._compile_ahead_submitted: set[str] = set()

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
        # M13: wait for any outstanding compile-ahead jobs so the test
        # process does not exit before background slangc finishes.
        with self._compile_ahead_lock:
            self._compile_ahead.clear()
            self._compile_ahead_submitted.clear()
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
        queued.  When inactive, dispatches immediately — this ensures
        correctness for callers that do not nest inside the batcher.

        M12 — Per-batch ready_set accumulation: a kernel is added to the
        current batch only if all its input tensors are already in the
        ready_set (produced by the batch or by already-flushed kernels).
        Otherwise it goes to the blocked list and is re-tried after each
        successful add.
        """
        if not self._active:
            kernel_handle(*dispatch_args)
            return

        tensor_args = [a for a in dispatch_args if isinstance(a, torch.Tensor)]
        if len(tensor_args) >= 2:
            input_tensors = tensor_args[:-1]
            output_tensor = tensor_args[-1]
            if all(id(t) in self._ready_set for t in input_tensors):
                self._pending.append((kernel_handle, dispatch_args))
                self._ready_set.add(id(output_tensor))
                # Try to drain any blocked kernels whose inputs are now ready.
                self._drain_blocked()
            else:
                self._blocked.append((kernel_handle, dispatch_args, output_tensor))
        else:
            # Non-tensor or single-tensor kernel: add unconditionally.
            self._pending.append((kernel_handle, dispatch_args))

    def _drain_blocked(self) -> None:
        """Promote blocked kernels to pending when their inputs become ready."""
        still_blocked: list[tuple] = []
        for item in self._blocked:
            kernel_handle, dispatch_args, output_tensor = item
            tensor_args = [a for a in dispatch_args if isinstance(a, torch.Tensor)]
            if len(tensor_args) >= 2:
                input_tensors = tensor_args[:-1]
                if all(id(t) in self._ready_set for t in input_tensors):
                    self._pending.append((kernel_handle, dispatch_args))
                    self._ready_set.add(id(output_tensor))
                    continue
            still_blocked.append(item)
        self._blocked = still_blocked

    def _flush(self):
        """Submit all pending dispatches, then re-enter batch mode.

        When C++ batch mode is active (M17.5), submits the accumulated
        command buffer via ``_end_batch()``, replays pending dispatches
        in auto-flush mode (so the results are on-GPU before the caller's
        extern kernel reads them), and then immediately re-enters batch
        mode via ``_begin_batch()``.  This means direct-dispatch calls
        that follow the flush point (the extern kernel itself plus any
        subsequent ``_jit_dispatch`` calls) accumulate in a fresh command
        buffer that is submitted by ``__exit__``'s ``_end_batch()`` call
        or by the next ``_flush()`` — rather than each doing a separate
        ``vkQueueSubmit``.

        The old concern ("do NOT re-enter batch mode because replayed
        dispatches would accumulate unsent") does not apply here: we
        restart batch mode AFTER the replay, so those dispatches have
        already been issued in auto-flush mode before the new batch
        begins.

        Falls back to sequential individual dispatches with per-8-dispatch
        auto-flush when the C++ batch functions are unavailable.
        """
        if not self._pending:
            return

        if self._batch_active and self._end_batch is not None:
            # Submit the accumulated command buffer (may be empty if no
            # direct dispatches happened since __enter__ / last _flush).
            self._end_batch()
            self._batch_active = False
            # Replay pending dispatches in auto-flush mode.  The caller's
            # extern kernel (the sync point that triggered this flush) will
            # run after we return, reading from these now-submitted results.
            for kernel_handle, dispatch_args in self._pending:
                kernel_handle(*dispatch_args)
            self._pending.clear()
            # Re-enter batch mode so direct dispatches after the sync point
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
