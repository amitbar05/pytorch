"""Vulkan Inductor codegen — re-export shim + OpClass->CodegenStrategy dispatch.

The implementation is split across sibling modules:
  kernel/               — VulkanKernel (monolith; split in progress, T1.1)
  lowerings/            — op lowerings (split in T1.2)
  fx_passes/            — FX graph passes (split in T1.3)
  expr_printer.py       — VulkanExprPrinter
  overrides.py          — DTYPE_TO_SLANG, value_to_slang, VulkanOverrides
  slang_helpers.py      — emit_helpers
  scheduling.py         — VulkanScheduling
  vulkan_template.py    — VulkanTemplateKernel, SlangTemplate
  vulkan_template_caller.py — install_external_mm
  template_registry.py  — TemplateRegistry (T1.5, populated by Track 4)
  config.py             — environment kill-switches

Import from here for backwards compatibility.
"""

# Re-exports (backward-compat surface).
from .expr_printer import VulkanExprPrinter
from .kernel import VulkanKernel
from .overrides import DTYPE_TO_SLANG, VulkanOverrides, value_to_slang
from .scheduling import VulkanScheduling
from .slang_helpers import emit_helpers
from .vulkan_template import SlangTemplate, VulkanTemplateKernel
from .vulkan_template_caller import (
    install_external_mm,
    install_external_rng,
    install_external_scatter,
)

__all__ = [
    "VulkanExprPrinter",
    "DTYPE_TO_SLANG",
    "value_to_slang",
    "VulkanOverrides",
    "emit_helpers",
    "VulkanKernel",
    "VulkanScheduling",
    "VulkanTemplateKernel",
    "SlangTemplate",
    "install_external_mm",
    "install_external_rng",
    "install_external_scatter",
    "OpClass",
    "CODEGEN_STRATEGIES",
    "register_codegen_strategy",
    "get_codegen_strategy",
]

# ── Track 1.4 / Track 4: OpClass → CodegenStrategy dispatch ─────────────

from enum import Enum, auto


class OpClass(Enum):
    """Heavy op classes served by pattern-matched template dispatch (Track 4).

    Each value maps to a group of aten ops that share a codegen strategy.
    Track 4's ``FxPatternRegistry`` picks a strategy by matching an FX
    subgraph to ``(op_class, dtype, shape_class)``.
    """

    POINTWISE = auto()
    MATMUL = auto()
    BMM = auto()
    CONV = auto()
    NORM = auto()
    SOFTMAX = auto()
    ATTENTION = auto()
    RNG = auto()
    OPTIMIZER = auto()
    SCATTER = auto()
    REDUCTION = auto()


class CodegenStrategy:
    """Metadata for how an op-class is lowered through the pipeline.

    Attributes:
        template_key:  Jinja template name under ``templates/`` (or ``None``
                       for IR-codegen'd primitives like pointwise).
        lowering_fn:   Callable that registers a lowering for the op class
                       (or ``None`` if the op is reached via FX pattern only).
        autotune:      Whether the autotuner should profile tile choices for
                       the generated kernels.
    """

    def __init__(self, template_key=None, lowering_fn=None, autotune=False):
        self.template_key = template_key
        self.lowering_fn = lowering_fn
        self.autotune = autotune


CODEGEN_STRATEGIES: dict[OpClass, CodegenStrategy] = {}


def register_codegen_strategy(
    op_class: OpClass,
    strategy: CodegenStrategy,
) -> None:
    """Register a codegen strategy for an op class.  Idempotent."""
    CODEGEN_STRATEGIES[op_class] = strategy


def get_codegen_strategy(op_class: OpClass) -> CodegenStrategy | None:
    """Look up the registered strategy for *op_class*."""
    return CODEGEN_STRATEGIES.get(op_class)


# ── Pre-register known strategies ──────────────────────────────────────
# Matmul: slang_mm.py.jinja template, autotuned.
register_codegen_strategy(
    OpClass.MATMUL,
    CodegenStrategy(
        template_key="slang_mm",
        autotune=True,
    ),
)
register_codegen_strategy(
    OpClass.BMM,
    CodegenStrategy(
        template_key="slang_mm",
        autotune=True,
    ),
)
# Conv: Path 2 — dedicated slang_conv2d template for groups==1.
# The FX pattern (conv_im2col) is gated to only fire for groups>1.
# For groups==1, the lowering in lowerings/conv.py routes through
# the tiled direct conv2d template.
register_codegen_strategy(
    OpClass.CONV,
    CodegenStrategy(
        template_key="slang_mm",
    ),
)
# RNG: philox_rng.py.jinja template (T4.6).
register_codegen_strategy(
    OpClass.RNG,
    CodegenStrategy(
        template_key="philox_rng",
        autotune=False,
    ),
)
# Attention: flash_attention.py.jinja template (T4.7) — un-gated.
# The SDPA FX pattern rewrites aten.scaled_dot_product_attention to
# torch_vulkan::flash_attention_fused, which is lowered through the
# template via install_external_flash_attention() (vulkan_template_caller.py).
register_codegen_strategy(
    OpClass.ATTENTION,
    CodegenStrategy(
        template_key="flash_attention",
        autotune=False,
    ),
)
# Optimizer: foreach_optimizer.py.jinja template (T4.8).
register_codegen_strategy(
    OpClass.OPTIMIZER,
    CodegenStrategy(
        template_key="foreach_optimizer",
        autotune=False,
    ),
)
# Scatter: TODO(T4.5) — scatter/gather/index_put template not yet
# implemented.  No scatter_atomic.py.jinja template exists.  This entry
# is a placeholder for when T4.5 delivers the template.
register_codegen_strategy(
    OpClass.SCATTER,
    CodegenStrategy(
        template_key="scatter_atomic",
    ),
)
