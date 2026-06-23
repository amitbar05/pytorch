"""Shared helpers used across multiple FX pattern modules.

These helpers were previously inlined in ``builtin_patterns.py``. They are
hoisted here so per-pattern modules can share them without duplication.

* ``_TemplateKeyStub`` — placeholder for the future ``TemplateKey`` returned by
  ``template_key_fn``. Lives here until ``..codegen`` / ``..template_registry``
  expose a real type.
* ``_template_key_mm`` / ``_template_key_bmm`` — used by patterns whose match
  context yields a 2-D matmul / 3-D batch-matmul template dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class _TemplateKeyStub:
    op_class: str = ""
    dtype: Optional[torch.dtype] = None
    shape_class: str = ""


def _template_key_mm(dtype: torch.dtype) -> Optional[_TemplateKeyStub]:
    """Return a TemplateKey for 2-D matmul templates (mm / addmm)."""
    return _TemplateKeyStub(op_class="mm", dtype=dtype, shape_class="2d")


def _template_key_bmm(dtype: torch.dtype) -> Optional[_TemplateKeyStub]:
    """Return a TemplateKey for 3-D batch-matmul templates (bmm)."""
    return _TemplateKeyStub(op_class="bmm", dtype=dtype, shape_class="3d")
