"""PF.60 — Fix RecursionError in torch._tensor_str._str for Vulkan tensors.

During ``torch._inductor.aot_compile``, upstream's tensor-str formatting
hits infinite recursion on Vulkan PrivateUse1 tensors:

  _str → _str_intern → _tensor_str → _Formatter → f"{value}" →
  __format__ → __repr__ → _str

Root cause: Vulkan tensors carry synthetic data_ptrs (1, 2, 3, ...) and
``.tolist()`` / ``.item()`` may fail during tracing (no backing buffer).
The failure path triggers another __repr__ → recursion.

Fix: monkey-patch ``_str_intern`` to detect Vulkan tensors when tensor
contents are not explicitly provided, and return a safe placeholder.
This only applies during diagnostic/formatting paths — actual tensor
operations are unaffected.
"""

from __future__ import annotations

import functools
import logging

_log = logging.getLogger(__name__)


def _install_pf60_tensor_str_fix() -> None:
    """Monkey-patch torch._tensor_str._str_intern for PF.60."""
    try:
        import torch
        from torch._tensor_str import _str_intern as _orig_str_intern
    except ImportError:
        return

    # Guard against double-patching
    if getattr(_orig_str_intern, "_vulkan_pf60_patched", False):
        return

    _recursion_guard = set()

    @functools.wraps(_orig_str_intern)
    def _safe_str_intern(self, *, tensor_contents=None):
        """Wrapper that prevents recursion on Vulkan tensors."""
        tid = id(self)
        if tid in _recursion_guard:
            # We're already trying to format this tensor — recursion detected.
            return f"<Vulkan tensor {tuple(self.shape)} {self.dtype}>"

        if self.device.type == "vulkan" and tensor_contents is None:
            # Vulkan tensor with no pre-provided contents — likely
            # during AOTI diagnostic printing.  Return a safe placeholder
            # to avoid the .tolist() → __repr__ → _str recursion chain.
            return f"<Vulkan tensor {tuple(self.shape)} {self.dtype}>"

        _recursion_guard.add(tid)
        try:
            return _orig_str_intern(self, tensor_contents=tensor_contents)
        finally:
            _recursion_guard.discard(tid)

    _safe_str_intern._vulkan_pf60_patched = True  # type: ignore[attr-defined]

    # Apply the patch
    import torch._tensor_str
    torch._tensor_str._str_intern = _safe_str_intern
    _log.info("PF.60: patched torch._tensor_str._str_intern for Vulkan tensors")
