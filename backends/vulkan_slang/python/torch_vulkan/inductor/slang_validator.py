"""Re-export shim — delegates to ``slang_validate/`` for the implementation.

``from torch_vulkan.inductor.slang_validator import validate_slang_source``
continues to work exactly as before; the per-pass logic now lives in the
``slang_validate`` sub-package (M15.1.i split).
"""

from __future__ import annotations

from torch_vulkan.inductor.slang_validate import validate_slang_source  # noqa: F401
