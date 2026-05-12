"""DR.6 — Vec4/packed16 vectorization audit pass.

Walks through pointwise kernel bodies after vec4/packed16 rewriting
and counts:

  * Total load/store operations
  * Vec4-eligible loads/stores (contiguous, aligned)
  * Successfully vectorized loads/stores (actually rewritten to vec4/packed16)

Reports hit rate as a percentage.  Gated by ``TORCH_VULKAN_VEC4_AUDIT=1``.

Usage::

    from ..heuristics.vectorization_audit import VectorizationAudit

    audit = VectorizationAudit(
        body_str,
        in_decls,
        out_decls,
        inout_decls,
        vec4_bufs,
        packed16_bufs,
        vec4_active,
        packed16_active,
    )
    summary = audit.run()
    # summary → {"total_loads": 12, "vec_loads": 9, "hit_rate": 75.0, ...}
"""

from __future__ import annotations

import logging
import re
from typing import Optional

_log = logging.getLogger(__name__)


class VectorizationAudit:
    """Count loads/stores in a pointwise kernel body and compute vectorization hit rates.

    The audit is called AFTER vec4/packed16 rewriting in
    ``HeaderMixin.codegen_kernel``, so the body code already reflects the
    final binding types (``StructuredBuffer<float4>``, ``StructuredBuffer<uint>``, etc.).
    """

    # Patterns for load-like operations (reads from input/inout buffers).
    _LOAD_RE = re.compile(
        r"""
        (?:                               # CSE assignment:  float cse_N = ...
            (?:float|int|int64_t|uint|bool|half)\s+\w+\s*=\s*
        )?
        (?:                               # different load forms:
            _vk_unpack_\w+\s*\(\s*\w+\s*\[        # packed16 unpack
            | _vk_vec4_h\w+\s*\(\s*\w+\s*,       # vec4 horizontal reduce
            | \w+\s*\[                            # plain buffer[idx]
        )
        """,
        re.VERBOSE,
    )

    # Patterns for store-like operations (writes to output/inout buffers).
    _STORE_RE = re.compile(
        r"""
        (?:
            _vk_pack_\w+\s*\(\s*                    # packed16 pack
            | \w+\s*\[\s*[^\]]+\s*\]\s*=            # plain buffer[idx] =
        )
        """,
        re.VERBOSE,
    )

    # Simple counters.
    _BUF_REF_RE = re.compile(r"\b(\w+)\s*\[")

    def __init__(
        self,
        body_str: str,
        in_decls: list[tuple[str, str]],
        out_decls: list[tuple[str, str]],
        inout_decls: list[tuple[str, str]],
        vec4_bufs: set[str],
        packed16_bufs: set[str],
        vec4_active: bool,
        packed16_active: bool,
        kernel_name: str = "",
    ) -> None:
        self._body = body_str
        self._in_inners = [inner for _, inner in in_decls]
        self._out_inners = [inner for _, inner in out_decls]
        self._inout_inners = [inner for _, inner in inout_decls]
        self._all_io_inners = self._in_inners + self._out_inners + self._inout_inners
        self._vec4_bufs = vec4_bufs
        self._packed16_bufs = packed16_bufs
        self._vec4_active = vec4_active
        self._packed16_active = packed16_active
        self._kernel_name = kernel_name

    def run(self) -> dict:
        """Execute the audit and return a summary dict.

        Returns a dict with keys:
          * ``total_loads`` — number of buffer-read operations found
          * ``total_stores`` — number of buffer-write operations found
          * ``vec_loads`` — how many loads involve vec4/packed16 buffers
          * ``vec_stores`` — how many stores involve vec4/packed16 buffers
          * ``eligible_loads`` — loads that could theoretically be vectorized
          * ``eligible_stores`` — stores that could theoretically be vectorized
          * ``load_hit_rate`` — vec_loads / max(eligible_loads, 1) * 100
          * ``store_hit_rate`` — vec_stores / max(eligible_stores, 1) * 100
          * ``overall_hit_rate`` — blended (vec_loads + vec_stores) / total * 100
          * ``vec4_active`` — whether vec4 rewrite was applied
          * ``packed16_active`` — whether packed16 rewrite was applied
          * ``vec4_buf_count`` — number of vec4-typed buffers
          * ``packed16_buf_count`` — number of packed16 buffers
        """
        total_loads = 0
        total_stores = 0
        vec_loads = 0
        vec_stores = 0
        eligible_loads = 0
        eligible_stores = 0

        lines = self._body.splitlines()
        vec_bufs_union = self._vec4_bufs | self._packed16_bufs

        for ln in lines:
            stripped = ln.strip()
            if not stripped or stripped.startswith("//"):
                continue

            # Count buffer references on this line.
            buf_refs = self._BUF_REF_RE.findall(stripped)
            for buf_name in buf_refs:
                if buf_name not in self._all_io_inners:
                    continue
                is_read = buf_name in self._in_inners or buf_name in self._inout_inners
                is_write = (
                    buf_name in self._out_inners or buf_name in self._inout_inners
                )
                is_vec = buf_name in vec_bufs_union

                # Classify as load / store based on surrounding context.
                # A line like "float cse = buf[idx]" is a load.
                # A line like "buf[idx] = cse" is a store.
                # A line like "vk_atomic_add(buf, idx, val)" is a store.

                # Heuristic: if the line contains "= buf_name[" → load
                # if the line contains "buf_name[idx] =" → store
                load_pat = re.compile(r"=\s*" + re.escape(buf_name) + r"\s*\[")
                store_pat = re.compile(
                    re.escape(buf_name)
                    + r"\s*\[[^\]]*\]\s*=|[^a-zA-Z]"
                    + re.escape(buf_name)
                    + r"\s*,"
                )
                # More precise: check for assignment to buffer (store).
                explicit_store = re.search(
                    re.escape(buf_name) + r"\s*\[[^\]]*\]\s*=", stripped
                )
                atomic_store = re.search(
                    r"vk_atomic_add\s*\(\s*" + re.escape(buf_name), stripped
                )
                explicit_load = re.search(
                    r"=\s*" + re.escape(buf_name) + r"\s*\[", stripped
                )
                # Unpack/pack patterns:
                unpack_load = re.search(
                    r"_vk_unpack_\w+\s*\(\s*" + re.escape(buf_name) + r"\s*\[",
                    stripped,
                )
                pack_store = re.search(
                    r"_vk_pack_\w+\s*\(\s*.*?,\s*" + re.escape(buf_name) + r"\s*\[",
                    stripped,
                )

                is_explicit_load = bool(explicit_load or unpack_load)
                is_explicit_store = bool(explicit_store or atomic_store or pack_store)

                if is_explicit_store or (is_write and not is_explicit_load):
                    total_stores += 1
                    if is_vec:
                        vec_stores += 1
                    # Eligible stores are those in vec4/packed16-eligible buffers.
                    if buf_name in vec_bufs_union:
                        eligible_stores += 1

                if is_explicit_load or (is_read and not is_explicit_store):
                    total_loads += 1
                    if is_vec:
                        vec_loads += 1
                    # Eligible loads are those in vec4/packed16-eligible buffers.
                    if buf_name in vec_bufs_union:
                        eligible_loads += 1

        total = total_loads + total_stores
        vec_total = vec_loads + vec_stores
        eligible_total = eligible_loads + eligible_stores

        summary = {
            "total_loads": total_loads,
            "total_stores": total_stores,
            "vec_loads": vec_loads,
            "vec_stores": vec_stores,
            "eligible_loads": eligible_loads,
            "eligible_stores": eligible_stores,
            "load_hit_rate": (
                (100.0 * vec_loads / max(eligible_loads, 1))
                if eligible_loads > 0
                else 0.0
            ),
            "store_hit_rate": (
                (100.0 * vec_stores / max(eligible_stores, 1))
                if eligible_stores > 0
                else 0.0
            ),
            "overall_hit_rate": (
                (100.0 * vec_total / max(eligible_total, 1))
                if eligible_total > 0
                else 0.0
            ),
            "vec4_active": self._vec4_active,
            "packed16_active": self._packed16_active,
            "vec4_buf_count": len(self._vec4_bufs),
            "packed16_buf_count": len(self._packed16_bufs),
        }
        return summary

    def log_summary(self) -> None:
        """Log the audit summary at DEBUG level."""
        s = self.run()
        tag = f"[{self._kernel_name}] " if self._kernel_name else ""
        _log.debug(
            "%svec4 audit: %d/%d loads vectorized (%.0f%%), "
            "%d/%d stores vectorized (%.0f%%), "
            "vec4_active=%s packed16_active=%s",
            tag,
            s["vec_loads"],
            s["eligible_loads"],
            s["load_hit_rate"],
            s["vec_stores"],
            s["eligible_stores"],
            s["store_hit_rate"],
            s["vec4_active"],
            s["packed16_active"],
        )

    def store_stats(self) -> None:
        """Store audit metrics in the inductor stats dict (DR.6).

        These metrics persist across compilations and are queryable via
        ``inductor_stats.get_stats()`` when ``TORCH_VULKAN_INDUCTOR_STATS=1``.
        """
        try:
            from torch_vulkan.inductor.inductor_stats import _get_stats_dict

            s = self.run()
            stats = _get_stats_dict()
            key = "vec4_audit"
            if key not in stats:
                stats[key] = {
                    "call_count": 0,
                    "total_loads": 0,
                    "total_stores": 0,
                    "vec_loads": 0,
                    "vec_stores": 0,
                    "eligible_loads": 0,
                    "eligible_stores": 0,
                }
            entry = stats[key]
            entry["call_count"] += 1
            entry["total_loads"] += s["total_loads"]
            entry["total_stores"] += s["total_stores"]
            entry["vec_loads"] += s["vec_loads"]
            entry["vec_stores"] += s["vec_stores"]
            entry["eligible_loads"] += s["eligible_loads"]
            entry["eligible_stores"] += s["eligible_stores"]
            entry["load_hit_rate"] = (
                (100.0 * entry["vec_loads"] / max(entry["eligible_loads"], 1))
                if entry["eligible_loads"] > 0
                else 0.0
            )
            entry["store_hit_rate"] = (
                (100.0 * entry["vec_stores"] / max(entry["eligible_stores"], 1))
                if entry["eligible_stores"] > 0
                else 0.0
            )
        except Exception:
            # Stats collection is best-effort; never fail compilation.
            pass


def audit_kernel(
    body_str: str,
    in_decls: list[tuple[str, str]],
    out_decls: list[tuple[str, str]],
    inout_decls: list[tuple[str, str]],
    vec4_bufs: set[str],
    packed16_bufs: set[str],
    vec4_active: bool,
    packed16_active: bool,
    kernel_name: str = "",
) -> Optional[dict]:
    """Convenience function: run audit if enabled, else return None.

    Checks ``TORCH_VULKAN_VEC4_AUDIT=1`` via ``config.vec4_audit_enabled()``.
    """
    from .. import config

    if not config.vec4_audit_enabled():
        return None
    audit = VectorizationAudit(
        body_str,
        in_decls,
        out_decls,
        inout_decls,
        vec4_bufs,
        packed16_bufs,
        vec4_active,
        packed16_active,
        kernel_name,
    )
    audit.log_summary()
    audit.store_stats()
    return audit.run()
