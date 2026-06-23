"""T.10 / CP.3 — RNN lowerings package.

Re-exports the shared utilities and per-cell-type dispatch registrations.
"""

from torch_vulkan.inductor.lowerings.rnn.common import (
    _register_rnn_fallbacks,
)

__all__ = ["_register_rnn_fallbacks"]
