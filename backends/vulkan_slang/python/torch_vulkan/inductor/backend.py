"""SlangVulkanBackend façade.

P8.21 (first cut) — a no-op delegator that consolidates the scattered
``register_*`` calls in ``__init__.py`` behind a single class. The façade
exists so subsequent migrations (Frontend / SlangIR / Runtime / Heuristics
subsystems per the reorganization plan) become local edits to typed
attributes on this class rather than additions to the module-level
``_legacy_register`` body.

Subsequent PRs migrate ``_register_with_inductor`` piece by piece —
the legacy entry point stays alive until P8.21's last sub-PR removes it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class BackendConfig:
    """Static configuration. Mutable runtime knobs live on subsystems."""

    async_compile: bool = True
    cold_compile_budget_us: int = 1_000_000
    spirv_cache_dir: Optional[str] = None


class SlangVulkanBackend:
    """Single entry point for the Vulkan/Slang Inductor backend.

    Idempotent: ``register()`` returns the same instance across calls.
    Instance is retrievable later via ``instance()``.
    """

    _instance: Optional["SlangVulkanBackend"] = None

    def __init__(self, config: Optional[BackendConfig] = None) -> None:
        self.config = config or BackendConfig()

    @classmethod
    def register(
        cls, config: Optional[BackendConfig] = None
    ) -> "SlangVulkanBackend":
        if cls._instance is not None:
            return cls._instance
        cls._instance = cls(config)
        cls._instance._register_with_inductor()
        return cls._instance

    @classmethod
    def instance(cls) -> Optional["SlangVulkanBackend"]:
        return cls._instance

    def _register_with_inductor(self) -> None:
        # Deferred import avoids the circular dependency with
        # ``__init__.py``, which exposes this façade as the public seam
        # while the legacy registration body still lives there.
        from . import _legacy_register
        _legacy_register()

    def stats(self) -> dict[str, Any]:
        from .inductor_stats import get_stats
        return {"compile": get_stats()}
