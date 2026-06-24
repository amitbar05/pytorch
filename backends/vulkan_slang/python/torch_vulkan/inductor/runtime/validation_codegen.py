"""M21.2 — validation-as-codegen-check.

For each kernel candidate that survives autotune benchmarking, we can run
a single dispatch with the Khronos validation layer + best-practices +
sync-validation features enabled, and check for VUIDs. Any VUID =
candidate rejection (in ``error`` mode) or compile-time warning (in
``warn`` mode). The contract is:

* Catches descriptor-set lifecycle UB (M17.5 / M17.8.d class) at
  compile time, not runtime.
* Catches binding-set mismatches (M17.1-gap2 / M22.12 class) at compile
  time.
* Catches buffer-pool sync hazards before they corrupt training runs.

Gate via ``TORCH_VULKAN_VALIDATE_CODEGEN``:

* ``off`` (default) — no validation runs. Zero overhead.
* ``warn``          — validation runs; VUIDs logged via ``trace_structured``
                      and a stdlib logger but compilation succeeds.
* ``error``         — validation runs; VUIDs raise ``RuntimeError`` from
                      the autotune commit path so the choice is rejected.

The validation subprocess is slow (~1-3 s per candidate cold + slangc).
To keep autotune time bounded, only the **winner** is validated, not
every candidate. ``validate_codegen_dispatch`` returns a
:class:`ValidationResult`; ``handle_validation_result`` applies the mode
contract (log vs raise).

Module is import-safe even when validation layers aren't installed —
``layer_installed()`` returns ``False`` and ``run_with_validation``
returns a sentinel ``ValidationResult(returncode=0, vuids=[])`` so
callers can always treat the result the same way.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)


# ── Public regex + paths ───────────────────────────────────────────────


VUID_RE = re.compile(r"VUID-[A-Za-z0-9-]+")

# Standard Khronos validation layer manifest locations (Linux).
LAYER_JSONS: tuple[str, ...] = (
    "/usr/share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json",
    "/usr/share/vulkan/implicit_layer.d/VkLayer_khronos_validation.json",
    "/etc/vulkan/explicit_layer.d/VkLayer_khronos_validation.json",
)

# Default ICD pin candidates — the first one that exists wins. Without
# a pin, the loader's ICD-sort-with-layers heuristic puts Lavapipe ahead
# of the real GPU on this dev box (M21.4 finding).
ICD_FILENAMES: tuple[str, ...] = (
    "/usr/share/vulkan/icd.d/radeon_icd.json",
    "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json",
    "/usr/share/vulkan/icd.d/nvidia_icd.json",
    "/usr/share/vulkan/icd.d/intel_icd.json",
)

# Ratchet file (created by M21.4). VUIDs listed there are accepted with
# justification; everything else is a regression.
_REPO_ROOT = Path(__file__).resolve().parents[3]
KNOWN_VUIDS_PATH = _REPO_ROOT / "tests" / "data" / "m21_4_known_vuids.txt"


# ── Result dataclass ───────────────────────────────────────────────────


@dataclass
class ValidationResult:
    returncode: int
    stdout: str
    stderr: str
    vuids: list[str] = field(default_factory=list)

    def unexpected_vuids(self, known: set[str]) -> list[str]:
        """Filter ``vuids`` against an accepted-VUIDs allowlist."""
        return [v for v in self.vuids if v not in known]


# ── Layer / ICD detection ──────────────────────────────────────────────


def layer_installed() -> bool:
    """Does the Khronos validation layer manifest exist anywhere?"""
    return any(os.path.exists(p) for p in LAYER_JSONS)


def find_icd_filename() -> Optional[str]:
    """First existing ICD-pin candidate; ``None`` if none found."""
    for cand in ICD_FILENAMES:
        if os.path.exists(cand):
            return cand
    return None


def load_known_vuids() -> set[str]:
    """Parse ``tests/data/m21_4_known_vuids.txt``; empty on missing file."""
    if not KNOWN_VUIDS_PATH.exists():
        return set()
    out: set[str] = set()
    for raw in KNOWN_VUIDS_PATH.read_text().splitlines():
        ln = raw.strip()
        if not ln or ln.startswith("#"):
            continue
        head = ln.split("#", 1)[0].strip()
        if head:
            out.add(head)
    return out


# ── Subprocess runner ──────────────────────────────────────────────────


def _build_validation_env(env_extra: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Compose the subprocess env needed to surface VUIDs to stderr.

    Mirrors M21.4's harness: validation layer + best-practices + sync,
    backend's debug-utils messenger at INFO severity, ICD-pin to the
    real GPU, no autotune (would recurse), no prewarm.
    """
    env = os.environ.copy()
    # VK_INSTANCE_LAYERS is only needed for subprocess validation runs.
    # In-process validation uses TORCH_VULKAN_VALIDATION (set before import).
    env["VK_INSTANCE_LAYERS"] = "VK_LAYER_KHRONOS_validation"
    env["VK_LAYER_ENABLES"] = (
        "VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT,"
        "VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_BIT_KHR"
    )
    env["TORCH_VULKAN_DEBUG_UTILS"] = "1"
    if not env.get("VK_ICD_FILENAMES"):
        icd = find_icd_filename()
        if icd is not None:
            env["VK_ICD_FILENAMES"] = icd
    env.pop("TORCH_DEVICE_BACKEND_AUTOLOAD", None)
    env["PYTHONUNBUFFERED"] = "1"
    env["TORCH_VULKAN_MAX_AUTOTUNE"] = "0"
    env["TORCH_VULKAN_NO_PREWARM"] = "1"
    # Critical: do NOT recurse — child must not run validation again.
    env["TORCH_VULKAN_VALIDATE_CODEGEN"] = "off"
    if env_extra:
        env.update(env_extra)
    return env


def run_with_validation(
    script_body: str,
    env_extra: Optional[dict[str, str]] = None,
    timeout_s: int = 180,
) -> ValidationResult:
    """Run ``script_body`` in a child Python with validation layers on.

    Stderr is parsed for ``VUID-...`` tokens via :data:`VUID_RE`.

    The body is wrapped via :func:`textwrap.dedent` so callers can use
    triple-quoted indented strings. The body MUST import its own
    ``torch_vulkan`` etc. — validation env vars need to be set BEFORE
    Vulkan instance creation, which happens at ``import torch_vulkan``.
    """
    env = _build_validation_env(env_extra)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script_body)],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        partial_err = e.stderr or b""
        if isinstance(partial_err, bytes):
            partial_err = partial_err.decode("utf-8", errors="replace")
        partial_out = e.stdout or b""
        if isinstance(partial_out, bytes):
            partial_out = partial_out.decode("utf-8", errors="replace")
        return ValidationResult(
            returncode=124,
            stdout=partial_out,
            stderr=partial_err + f"\n[TIMEOUT after {timeout_s}s]\n",
            vuids=VUID_RE.findall(partial_err),
        )

    combined = (proc.stderr or "") + (proc.stdout or "")
    return ValidationResult(
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        vuids=VUID_RE.findall(combined),
    )


def assert_clean(result: ValidationResult) -> None:
    """Test helper: assert no unexpected VUIDs + clean exit.

    Used by the M21.4 stress harness. Library callers should use
    :func:`handle_validation_result` instead so the env-var contract
    is respected.
    """
    known = load_known_vuids()
    unexpected = result.unexpected_vuids(known)
    assert result.returncode == 0, (
        f"subprocess crashed (rc={result.returncode}); stderr tail:\n"
        f"{result.stderr[-1500:]}"
    )
    assert not unexpected, (
        f"unexpected VUIDs surfaced ({len(unexpected)}):\n  - "
        + "\n  - ".join(unexpected[:10])
        + f"\n\nstderr tail:\n{result.stderr[-1500:]}"
    )


# ── Mode contract ──────────────────────────────────────────────────────


_MODE_OFF = "off"
_MODE_WARN = "warn"
_MODE_ERROR = "error"


def get_codegen_validation_mode() -> str:
    """Resolve ``TORCH_VULKAN_VALIDATE_CODEGEN`` to a canonical mode.

    Defaults to ``off`` when:
    * the env var is unset / empty,
    * the env var is ``0`` / ``off`` / ``false`` / ``no``, or
    * the Khronos validation layer manifest isn't installed (no point
      spawning subprocesses that can't surface VUIDs).

    Returns one of ``{"off", "warn", "error"}``.

    M-VAL.2 (v7): when ``TORCH_VULKAN_VUID_AS_ERROR`` is not "0"
    (default-ON after M-VAL.1/M-VAL.3), the default mode is ``error``
    instead of ``warn`` — autotune candidates that emit VUIDs are
    rejected.  This makes test-mode autotune VUID-clean by default.
    """
    raw = os.environ.get("TORCH_VULKAN_VALIDATE_CODEGEN", "").strip().lower()
    if raw in ("", "0", "off", "false", "no"):
        return _MODE_OFF
    if not layer_installed():
        # Can't surface VUIDs without the layer — silently downgrade so
        # CI without layers doesn't pay subprocess cost for nothing.
        return _MODE_OFF
    if raw == "error":
        return _MODE_ERROR
    if raw == "warn":
        return _MODE_WARN
    # M-VAL.2: default to error when VUID-as-error is active (test mode),
    # warn otherwise.
    if os.environ.get("TORCH_VULKAN_VUID_AS_ERROR", "") == "0":
        return _MODE_WARN
    return _MODE_ERROR


def is_codegen_validation_enabled() -> bool:
    return get_codegen_validation_mode() != _MODE_OFF


# Instrumentation hook so tests can verify whether the gate fired.
_CALL_COUNT = 0


def _call_count_reset() -> None:
    global _CALL_COUNT
    _CALL_COUNT = 0


def _call_count_get() -> int:
    return _CALL_COUNT


def validate_codegen_dispatch(
    script_body: str,
    *,
    kernel_name: str = "<unknown>",
    env_extra: Optional[dict[str, str]] = None,
    timeout_s: int = 60,
) -> ValidationResult:
    """Run one validation-layer dispatch for the named kernel.

    ``script_body`` is the Python code that imports ``torch_vulkan``,
    dispatches the kernel of interest, and (optionally) synchronizes.
    The body is run in a subprocess so ``VK_INSTANCE_LAYERS`` takes
    effect before ``import torch_vulkan`` creates the Vulkan instance.

    Returns the :class:`ValidationResult`. The caller decides what to
    do with it — see :func:`handle_validation_result`.

    No-op fast-path: when :func:`get_codegen_validation_mode` returns
    ``off``, this function returns a clean sentinel result without
    spawning a subprocess. The fast-path is the common case
    (autotune always calls this; mode is off by default).
    """
    global _CALL_COUNT
    mode = get_codegen_validation_mode()
    if mode == _MODE_OFF:
        # No-op fast-path — never spawn.
        return ValidationResult(returncode=0, stdout="", stderr="", vuids=[])

    _CALL_COUNT += 1
    _log.info("validate_codegen_dispatch: running on %s (mode=%s)", kernel_name, mode)
    return run_with_validation(script_body, env_extra=env_extra, timeout_s=timeout_s)


# ── Mode-driven outcome handling ───────────────────────────────────────


def _trace_structured_safe(name: str, payload: dict) -> None:
    """Try to log via ``trace_structured`` (tlparse-visible); fall back to logger."""
    try:
        from torch._logging import trace_structured

        trace_structured(
            name,
            metadata_fn=lambda: {"kind": "vulkan-codegen-validation"},
            payload_fn=lambda: payload,
        )
    except Exception:
        # ``trace_structured`` requires Inductor's logging context, which
        # may not exist in eager benchmarks. Best-effort.
        _log.info("[%s] %s", name, payload)


def handle_validation_result(
    result: ValidationResult,
    *,
    kernel_name: str,
    raise_on_vuid: Optional[bool] = None,
) -> None:
    """Apply the ``TORCH_VULKAN_VALIDATE_CODEGEN`` mode contract.

    ``raise_on_vuid``:
    * ``True``  — raise ``RuntimeError`` if any unexpected VUID surfaced.
    * ``False`` — log only, never raise.
    * ``None``  — read the env var (``error`` ⇒ raise, ``warn`` ⇒ log).
    """
    if raise_on_vuid is None:
        mode = get_codegen_validation_mode()
        raise_on_vuid = mode == _MODE_ERROR

    known = load_known_vuids()
    unexpected = result.unexpected_vuids(known)

    if result.returncode != 0:
        # A crash is treated as a hard failure under ``error``; under
        # ``warn`` we still surface the diagnostic but don't raise.
        msg = (
            f"validate_codegen_dispatch[{kernel_name}] subprocess "
            f"failed (rc={result.returncode}); stderr tail:\n"
            f"{result.stderr[-1500:]}"
        )
        _trace_structured_safe(
            "vulkan-codegen-validation",
            {
                "kernel": kernel_name,
                "returncode": result.returncode,
                "vuids": unexpected,
            },
        )
        if raise_on_vuid:
            raise RuntimeError(
                f"validation-as-codegen-check failed for {kernel_name}: "
                f"subprocess rc={result.returncode}"
            )
        _log.warning(msg)
        return

    if not unexpected:
        return  # clean — nothing to do.

    payload = {
        "kernel": kernel_name,
        "returncode": result.returncode,
        "vuids": unexpected,
    }
    _trace_structured_safe("vulkan-codegen-validation", payload)
    if raise_on_vuid:
        raise RuntimeError(
            f"validation-as-codegen-check failed for {kernel_name}: "
            f"unexpected VUIDs: {unexpected[:5]}"
        )
    _log.warning(
        "validate_codegen_dispatch[%s] surfaced %d unexpected VUIDs: %s",
        kernel_name,
        len(unexpected),
        unexpected[:5],
    )


# ── Convenience: validate by replaying a kernel from its source ────────


_DEFAULT_VALIDATION_BODY_TMPL = """
import torch, torch_vulkan
from torch_vulkan.inductor.runtime import compile_and_dispatch

# Single dispatch of the candidate. ``cache_key`` keeps the SPIR-V
# warm across the autotune phase and the post-commit consumer dispatch.
src = {src!r}
cache_key = {cache_key!r}
entry = {entry!r}
{tensor_setup}
push_constants = {pc!r}
wg = ({wg_x}, {wg_y}, {wg_z})
compile_and_dispatch(
    src=src,
    tensors=tensors,
    wg_x=wg[0], wg_y=wg[1], wg_z=wg[2],
    push_constants=push_constants,
    num_outputs={num_outputs},
    entry=entry,
    cache_key=cache_key,
)
torch_vulkan.synchronize()
print("VALIDATE_OK")
"""


def validate_kernel_source(
    *,
    src: str,
    cache_key: str,
    tensor_setup: str,
    wg: tuple[int, int, int],
    push_constants: bytes = b"",
    num_outputs: int = 1,
    entry: str = "computeMain",
    kernel_name: Optional[str] = None,
    timeout_s: int = 60,
) -> ValidationResult:
    """High-level helper: dispatch a kernel under validation layers.

    ``tensor_setup`` is a python snippet that must produce a list
    named ``tensors``. Example::

        tensor_setup = (
            "x = torch.zeros(1024, device='vulkan')\\n"
            "y = torch.ones(1024, device='vulkan')\\n"
            "tensors = [x, y, torch.empty_like(x)]"
        )

    No-op fast-path applies when validation is disabled (mode=off).
    """
    body = _DEFAULT_VALIDATION_BODY_TMPL.format(
        src=src,
        cache_key=cache_key,
        entry=entry,
        tensor_setup=tensor_setup,
        pc=push_constants,
        wg_x=wg[0],
        wg_y=wg[1],
        wg_z=wg[2],
        num_outputs=num_outputs,
    )
    return validate_codegen_dispatch(
        body,
        kernel_name=kernel_name or cache_key,
        timeout_s=timeout_s,
    )
