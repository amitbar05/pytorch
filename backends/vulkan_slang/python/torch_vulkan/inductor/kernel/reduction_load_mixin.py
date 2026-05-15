"""Reduction load mixin — dtype sizing and groupshared variable allocation.

Extracted from ``ReductionMixin`` (M15.1.g — Track 1 anti-goal #7 split).
Handles loading input data for reduction kernels: buffer path resolution,
packed16 decisions, load masking, etc.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import sympy
import torch
from torch._inductor.virtualized import V
from torch.utils._sympy.value_ranges import ValueRanges

from .reduction_tile_picker import (
    _BANK_CONFLICT_PAD,
    _analyze_smem_bank_conflict_risk,
)

if TYPE_CHECKING:
    from torch._inductor.codegen.common import CSEVariable


class ReductionLoadMixin:
    """Mixin for reduction load/allocation: dtype sizing and groupshared
    variable creation with bank-conflict-aware padding.
    """

    @staticmethod
    def _slang_dtype_bytes(dtype_str: str) -> int:
        return {
            "double": 8,
            "int64_t": 8,
            "uint64_t": 8,
            "float": 4,
            "int": 4,
            "uint": 4,
            "half": 2,
            "int16_t": 2,
            "uint16_t": 2,
            "int8_t": 1,
            "uint8_t": 1,
            "bool": 1,
        }.get(dtype_str, 4)

    def _new_idxvar(
        self,
        dtype,
        elem_count: Optional[int] = None,
        default_value: Optional[Any] = None,
        is_threadgroup: bool = True,
        bounds: ValueRanges = ValueRanges.unknown(),
        access_pattern: str = "generic",
    ) -> "CSEVariable":
        """Create a CSE variable, optionally backed by groupshared memory.

        Args:
            access_pattern: One of ``"reduction"``, ``"welford"``,
                ``"scan"``, ``"sort"``, or ``"generic"``. Used by M20
                bank-conflict analysis to compute appropriate padding.
        """
        from torch._inductor.codegen.common import CSEVariable

        if isinstance(dtype, torch.dtype):
            dtype = self.dtype_to_str(dtype)
        var_name = f"tmp_acc_{next(self.acc_var_ids)}"
        var = V.kernel.create_cse_var(var_name, bounds, dtype)
        if is_threadgroup:
            self._pw_uses_groupshared = True
            count = 1
            # DR.6: resolve bank-padding gate once (used for both budget
            # tracking and array declaration).
            from .. import config as _cfg

            _use_pad = _cfg.bank_conflict_pad()
            if elem_count is not None and isinstance(elem_count, (int, sympy.Integer)):
                count = max(1, int(elem_count))
                if _use_pad:
                    # M20: use access-pattern-aware padding instead of
                    # blanket +32.  Fall back to 32 if analysis is
                    # unavailable or returns an unexpected value.
                    _elem_bytes = self._slang_dtype_bytes(dtype)
                    _analyzed_pad = _analyze_smem_bank_conflict_risk(
                        num_threads=int(elem_count),
                        element_size=_elem_bytes,
                        access_pattern=access_pattern,
                    )
                    _pad = _analyzed_pad
                    count = count + _pad
            raw_bytes = self._slang_dtype_bytes(dtype) * count
            # M11.6: Power-of-2 padding for groupshared arrays > 1 KB.
            # RDNA1 LDS stride-1 access patterns can alias on banks when
            # the stride between adjacent work-items lands on the same
            # bank.  Padding to a power of 2 shifts the bank mapping for
            # the tail elements and eliminates most stride-1 bank
            # conflicts.  Only applied to arrays large enough to matter
            # (> 1 KB) and only when bank-conflict padding is enabled.
            if _use_pad and raw_bytes > 1024:
                next_pow2 = 1
                while next_pow2 < raw_bytes:
                    next_pow2 <<= 1
                # How many extra elements we need to reach the next pow2
                extra_bytes = next_pow2 - raw_bytes
                extra_elements = (
                    extra_bytes + self._slang_dtype_bytes(dtype) - 1
                ) // self._slang_dtype_bytes(dtype)
                count = count + extra_elements
                raw_bytes = self._slang_dtype_bytes(dtype) * count
            new_bytes = (raw_bytes + 15) & ~15
            if (
                self._groupshared_bytes_used + new_bytes
                > self._groupshared_budget_bytes
            ):
                raise NotImplementedError(
                    f"Vulkan Inductor: groupshared LDS budget exceeded "
                    f"({self._groupshared_bytes_used + new_bytes} bytes > "
                    f"{self._groupshared_budget_bytes} budget). The "
                    f"driver would spill to scratch — disable persistent "
                    f"reduction for this kernel or shrink rnumel."
                )
            self._groupshared_bytes_used += new_bytes
            decl = f"groupshared {dtype} {var_name}"
            # M20 / DR.6: bank-conflict padding (analysis-driven)
            if elem_count:
                if _use_pad:
                    decl += f"[{self.sexpr(elem_count)} + {_pad}]"
                    # Debug-level log of the analysis decision.
                    _log = logging.getLogger(__name__)
                    _log.debug(
                        "M20 bank-conflict: pattern=%s, dtype=%s, "
                        "threads=%s → pad=%s (was blanket %s)",
                        access_pattern,
                        dtype,
                        int(elem_count),
                        _pad,
                        _BANK_CONFLICT_PAD,
                    )
                else:
                    decl += f"[{self.sexpr(elem_count)}]"
            self.module_scope_decls.writeline(decl + self.suffix)
        else:
            decl = f"{dtype} {var_name}"
            if elem_count:
                decl += f"[{self.sexpr(elem_count)}]"
            if default_value is not None:
                decl += f" = {default_value}"
            self.indexing_code.writeline(decl + self.suffix)
        return var
