"""CP.9 / TRAIN.12 — PhiloxState wrapper for deterministic compile↔eager RNG.

Provides a session-scoped Philox RNG state that advances its offset
deterministically after each RNG call.  Both eager and compiled
(``FallbackKernel``) paths share the same state, so dropout masks
are reproducible across compile boundaries when starting from the
same initial seed.

Usage::

    from torch_vulkan.inductor.philox_state import get_philox_state, reset_philox_state

    torch.manual_seed(42)
    reset_philox_state()  # reset offset to 0
    state = get_philox_state()  # derive seed from torch.default_generator

    # In eager dispatch:
    offset = state.advance(num_elements)
    _dispatch_philox_rng(..., offset=offset)
"""

from __future__ import annotations

import hashlib
from typing import Optional

import torch


class PhiloxState:
    """Deterministic Philox RNG state shared across eager and compiled paths.

    The seed is derived from ``torch.default_generator`` when the state is
    initialised (or reset).  The offset advances monotonically with each
    RNG call, ensuring that the same sequence of calls produces the same
    sequence of random values.

    Attributes:
        seed_lo: Low 32 bits of the 64-bit Philox seed.
        seed_hi: High 32 bits of the 64-bit Philox seed.
        _offset: Current counter offset (monotonically increasing).
    """

    __slots__ = ("seed_lo", "seed_hi", "_offset")

    def __init__(self, seed_lo: int = 0, seed_hi: int = 0, offset: int = 0):
        self.seed_lo = seed_lo
        self.seed_hi = seed_hi
        self._offset = offset

    # ── Offset management ────────────────────────────────────────────

    @property
    def offset(self) -> int:
        """Current offset (does NOT advance)."""
        return self._offset

    def advance(self, n: int) -> int:
        """Advance the offset by *n* and return the **pre-advance** value.

        Callers use the returned offset for the dispatch and then drop it;
        the next call to ``advance()`` will return a later offset.

        Args:
            n: Number of Philox counter values to consume.  For
               ``uniform`` / ``fused_dropout`` mode this equals
               ``num_elements``; for ``normal`` mode the shader
               consumes 2 counters per element, so pass
               ``num_elements`` (the template bumps the key internally
               for the second output).
        """
        old = self._offset
        self._offset += n
        return old

    def reset(self) -> None:
        """Reset the offset to zero (seed is unchanged)."""
        self._offset = 0

    # ── Pickle support ───────────────────────────────────────────────

    def __getstate__(self):
        return (self.seed_lo, self.seed_hi, self._offset)

    def __setstate__(self, state):
        self.seed_lo, self.seed_hi, self._offset = state

    def __repr__(self) -> str:
        return (
            f"PhiloxState(seed=0x{self.seed_lo:08X}{self.seed_hi:08X}, "
            f"offset={self._offset})"
        )


# ── Global session state ──────────────────────────────────────────────────

_global_philox_state: Optional[PhiloxState] = None
_last_initial_seed: Optional[int] = None  # torch.default_generator.initial_seed()


def _derive_seed_from_torch() -> tuple[int, int]:
    """Derive a deterministic 64-bit Philox seed from the global PyTorch RNG.

    Uses the CPU generator's current state so ``torch.manual_seed(...)``
    controls reproducibility.  Returns ``(seed_lo, seed_hi)`` as two
    unsigned 32-bit integers.
    """
    gen = torch.default_generator
    raw_state = gen.get_state()
    state_bytes = (
        raw_state.tobytes()
        if hasattr(raw_state, "tobytes")
        else str(raw_state).encode()
    )
    h = hashlib.sha256(state_bytes).digest()[:8]
    seed64 = int.from_bytes(h, "little")
    seed_lo = seed64 & 0xFFFFFFFF
    seed_hi = (seed64 >> 32) & 0xFFFFFFFF
    return seed_lo, seed_hi


def get_philox_state() -> PhiloxState:
    """Return the session-scoped PhiloxState, creating/re-deriving as needed.

    The seed is derived from ``torch.default_generator`` on first access and
    re-derived whenever ``torch.manual_seed()`` (or equivalent) changes the
    generator's initial seed.  This ensures PF.27.b correctness: different
    seeds produce different Philox output.

    PF.27.c correctness: the offset advances monotonically across calls, so
    consecutive calls with the same seed produce different output.
    """
    global _global_philox_state, _last_initial_seed

    # Detect torch.manual_seed() re-call by checking initial_seed().
    # This is cheap (just reading an int field on the generator).
    try:
        cur_seed = torch.default_generator.initial_seed()
    except (AttributeError, RuntimeError):
        cur_seed = None

    if cur_seed is not None and cur_seed != _last_initial_seed:
        # Generator was re-seeded — force re-creation with fresh seed + offset=0.
        _global_philox_state = None
        _last_initial_seed = cur_seed

    if _global_philox_state is None:
        seed_lo, seed_hi = _derive_seed_from_torch()
        _global_philox_state = PhiloxState(seed_lo=seed_lo, seed_hi=seed_hi, offset=0)
        if cur_seed is not None:
            _last_initial_seed = cur_seed
    return _global_philox_state


def reset_philox_state() -> PhiloxState:
    """Re-derive the seed from ``torch.default_generator`` and reset offset.

    Returns the fresh state (also stored as the global session state).
    """
    global _global_philox_state
    seed_lo, seed_hi = _derive_seed_from_torch()
    _global_philox_state = PhiloxState(seed_lo=seed_lo, seed_hi=seed_hi, offset=0)
    return _global_philox_state
