"""T.10 / CP.3 — re-export shim for the RNN lowerings package.

This file is retained so that all existing imports (e.g.
``from .rnn import _register_rnn_fallbacks``) continue to work
unchanged.  The implementation lives in the ``rnn/`` package.
"""

from torch_vulkan.inductor.lowerings.rnn.common import (
    _register_rnn_fallbacks,
)

__all__ = ["_register_rnn_fallbacks"]
