"""M19.6 — Vulkan foreach pointwise lowering coverage for torch.compile mode.

The 16 target foreach element-wise ops are already covered by upstream
Inductor's ``make_foreach_pointwise`` / ``register_foreach_pointwise`` path
(``torch/_inductor/lowering.py``), which decomposes each foreach op into N
independent pointwise lowerings and, because Vulkan advertises
``BackendFeature.FOREACH``, registers the resulting buffers into a single
``ForeachKernelSchedulerNode`` that ``VulkanComboKernel`` merges into one
multi-entry Slang shader.

This module provides:

1. ``_check_foreach_lowerings_registered`` — asserts (at registration time)
   that all 16 ops landed in ``lowerings`` as expected.  Fires early so a
   missing entry is caught at backend-init time, not at first compile.

2. ``_suppress_foreach_aot_decomps`` — removes any ``_foreach_*`` entries
   from the AOT autograd decomp table that would consume the ops before
   Inductor's lowering phase (empirically none exist in PyTorch 2.11, but
   the guard future-proofs against upstream additions).

3. ``register_foreach_pointwise_lowerings`` — public entry called from
   ``lowerings/__init__.py::register()``.

Ops covered (all 16 from M19.6):
  Functional binary (list / scalar):
    _foreach_add.{List,Scalar}  _foreach_sub.{List,Scalar}
    _foreach_mul.{List,Scalar}  _foreach_div.{List,Scalar}
  Functional unary:
    _foreach_neg.default  _foreach_abs.default
    _foreach_sqrt.default  _foreach_reciprocal.default
  In-place:
    _foreach_add_.{List,Scalar}  _foreach_mul_.Scalar

All are lowered by the upstream path; this module validates + guards them.
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)


# ── Op list ──────────────────────────────────────────────────────────────────

# (aten_op_name, overload) pairs matching the 16 M19.6 target ops.
# ``overload`` may be None for packet-level registrations (e.g. _foreach_sqrt)
# where upstream uses ``get_overloads`` to expand all overloads automatically.
_TARGET_OPS: list[tuple[str, str | None]] = [
    # Binary list ops
    ("_foreach_add", "List"),
    ("_foreach_add", "Scalar"),
    ("_foreach_sub", "List"),
    ("_foreach_sub", "Scalar"),
    ("_foreach_mul", "List"),
    ("_foreach_mul", "Scalar"),
    ("_foreach_div", "List"),
    ("_foreach_div", "Scalar"),
    # Unary ops
    ("_foreach_neg", "default"),
    ("_foreach_abs", "default"),
    # sqrt and reciprocal are registered at the packet level in upstream, so
    # both .default and .out overloads are covered; check .default here.
    ("_foreach_sqrt", "default"),
    ("_foreach_reciprocal", "default"),
    # In-place ops
    ("_foreach_add_", "List"),
    ("_foreach_add_", "Scalar"),
    ("_foreach_mul_", "Scalar"),
    # _foreach_mul_.List is also registered by upstream (inplaceable_foreach_ops);
    # include it so the guard is complete.
    ("_foreach_mul_", "List"),
]


def _check_foreach_lowerings_registered() -> None:
    """Assert all M19.6 target ops are in the Inductor lowering table.

    Called once at ``register_foreach_pointwise_lowerings()`` time.  Logs a
    warning (rather than raising) for any missing entry so a single broken
    overload does not abort the entire registration sequence.
    """
    import torch
    from torch._inductor.lowering import lowerings

    aten = torch.ops.aten
    missing: list[str] = []

    for op_name, overload in _TARGET_OPS:
        op_packet = getattr(aten, op_name, None)
        if op_packet is None:
            missing.append(f"aten.{op_name} (not found in aten)")
            continue
        if overload is not None:
            op = getattr(op_packet, overload, None)
            if op is None:
                missing.append(f"aten.{op_name}.{overload} (overload not found)")
                continue
            if op not in lowerings:
                missing.append(f"aten.{op_name}.{overload}")
        else:
            # Check at least one overload is in lowerings.
            found = any(
                getattr(op_packet, ov, None) in lowerings
                for ov in op_packet.overloads()
            )
            if not found:
                missing.append(f"aten.{op_name} (no overload in lowerings)")

    if missing:
        _log.warning(
            "M19.6 foreach pointwise: %d ops not in Inductor lowerings — "
            "they will fall back to CPU or raise NotImplemented: %s",
            len(missing),
            ", ".join(missing),
        )
    else:
        _log.debug(
            "M19.6 foreach pointwise: all %d target ops registered.", len(_TARGET_OPS)
        )


def _suppress_foreach_aot_decomps() -> None:
    """Remove _foreach_* entries from the AOT decomp table if any exist.

    As of PyTorch 2.11 none of the 16 target ops appear in the AOT
    decomposition table (``torch._decomp.decomposition_table``).  This guard
    runs anyway so that any upstream addition in a later PyTorch version does
    not silently eat the op before Inductor's lowering phase.
    """
    import torch
    from torch._decomp import decomposition_table as _aot_decomps

    aten = torch.ops.aten
    removed: list[str] = []

    for op_name, overload in _TARGET_OPS:
        op_packet = getattr(aten, op_name, None)
        if op_packet is None:
            continue
        if overload is not None:
            op = getattr(op_packet, overload, None)
            if op is not None and op in _aot_decomps:
                _aot_decomps.pop(op)
                removed.append(f"aten.{op_name}.{overload}")
        else:
            for ov in getattr(op_packet, "overloads", lambda: [])():
                op = getattr(op_packet, ov, None)
                if op is not None and op in _aot_decomps:
                    _aot_decomps.pop(op)
                    removed.append(f"aten.{op_name}.{ov}")

    if removed:
        _log.info(
            "M19.6 foreach pointwise: suppressed %d AOT decomps: %s",
            len(removed),
            ", ".join(removed),
        )


def register_foreach_pointwise_lowerings() -> None:
    """Install M19.6 foreach pointwise guards and validate upstream lowerings.

    Called once from ``lowerings/__init__.py::register()``.
    """
    _suppress_foreach_aot_decomps()
    _check_foreach_lowerings_registered()
