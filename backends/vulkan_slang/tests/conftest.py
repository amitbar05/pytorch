import os

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Run tests on real GPU hardware (RDNA1) instead of software Vulkan",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "vulkan: test requires Vulkan device")
    config.addinivalue_line(
        "markers",
        "slow_compile(seconds=N): override the per-test slangc cold-compile budget",
    )
    config.addinivalue_line(
        "markers",
        "gpu: test must run on hardware GPU (RDNA1)",
    )
    config.addinivalue_line(
        "markers",
        "sw_vulkan: test runs on software Vulkan (Lavapipe) only",
    )
    config.addinivalue_line(
        "markers",
        "both: test runs on both software and hardware Vulkan, compares results",
    )


@pytest.fixture
def vulkan_device():
    """Provides a Vulkan device, skipping if unavailable."""
    try:
        import torch_vulkan

        if not torch_vulkan.is_available():
            pytest.skip("No Vulkan device (install SwiftShader for CPU testing)")
    except ImportError:
        pytest.skip("torch_vulkan not installed")
    return torch.device("vulkan:0")


_DEFAULT_COLD_BUDGET_S: float = float(
    os.environ.get("TORCH_VULKAN_COLD_BUDGET_S", "30.0")
)


@pytest.fixture(autouse=True)
def _slangc_cold_budget(request):
    """PF.15: fail a test that cold-compiles more slangc SPIR-V than the budget.

    The budget is ``_DEFAULT_COLD_BUDGET_S`` seconds (default 30 s, override via
    ``TORCH_VULKAN_COLD_BUDGET_S``). A test can raise its own budget with::

        @pytest.mark.slow_compile(seconds=120)
        def test_big_kernel(self): ...
    """
    try:
        from torch_vulkan.inductor.runtime import _COMPILE_STATS
    except Exception:
        yield
        return

    marker = request.node.get_closest_marker("slow_compile")
    budget_s = (
        float(marker.kwargs.get("seconds", _DEFAULT_COLD_BUDGET_S))
        if marker
        else _DEFAULT_COLD_BUDGET_S
    )

    before = float(_COMPILE_STATS.get("cold_compile_us", 0.0))
    yield
    after = float(_COMPILE_STATS.get("cold_compile_us", 0.0))
    delta_s = (after - before) / 1e6
    if delta_s > budget_s:
        raise AssertionError(
            f"PF.15: cold-compile budget exceeded {delta_s:.2f}s > {budget_s:.2f}s"
        )


@pytest.fixture(autouse=True)
def _reset_inductor_caches():
    """GAP 7.3 — reset per-test mutable caches for deterministic dispatch counts.

    Without this, dispatch-count tests pass standalone but fail in the full
    suite under ``pytest -n 4`` because in-memory SPIR-V caches
    (``_cache_by_key``, ``_cache_by_hash``) and compile stats leak across
    tests within the same xdist worker.
    """
    try:
        from torch_vulkan.inductor.runtime import reset_per_test_caches

        reset_per_test_caches()
    except Exception:
        pass
    try:
        import torch._dynamo

        torch._dynamo.reset()
    except Exception:
        pass
    yield
    try:
        import torch._dynamo

        torch._dynamo.reset()
    except Exception:
        pass


# ── GPU / software-Vulkan device fixtures ───────────────────────────

_RADEON_ICD = "/usr/share/vulkan/icd.d/radeon_icd.json"


@pytest.fixture
def gpu_device():
    """Returns ``'vulkan:0'`` backed by the Radeon ICD, or skips if the
    GPU is unavailable.

    Sets ``VK_ICD_FILENAMES`` to point at the RDNA1 (RX 5600 XT)
    ``/dev/dri/renderD128`` ICD.  Callers receive ``"vulkan:0"`` as a
    convenience string that they can pass to ``torch.device(…)``.
    """
    if not os.path.exists(_RADEON_ICD):
        pytest.skip("Radeon GPU ICD not found at " + _RADEON_ICD)
    os.environ["VK_ICD_FILENAMES"] = _RADEON_ICD
    try:
        t = torch.empty(1, device="vulkan:0")
        return "vulkan:0"
    except Exception:
        pytest.skip("GPU not available even with Radeon ICD")


@pytest.fixture
def sw_vulkan():
    """Returns ``'vulkan:0'`` guaranteed to use software Vulkan (Lavapipe).

    Explicitly clears ``VK_ICD_FILENAMES`` so the system-default ICD
    (Lavapipe / SwiftShader) is used, regardless of whether ``--gpu``
    was passed.
    """
    os.environ.pop("VK_ICD_FILENAMES", None)
    try:
        import torch_vulkan

        if not torch_vulkan.is_available():
            pytest.skip("No software Vulkan device available")
    except ImportError:
        pytest.skip("torch_vulkan not installed")
    return "vulkan:0"


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests when ``--gpu`` is not passed, and skip sw_vulkan
    tests when ``--gpu`` is passed (avoids mixing hardware/software
    results).
    """
    gpu_requested = config.getoption("--gpu", default=False)
    for item in items:
        # Tests marked 'gpu' require --gpu
        if item.get_closest_marker("gpu") and not gpu_requested:
            item.add_marker(pytest.mark.skip(reason="--gpu not passed"))
        # Tests marked 'sw_vulkan' are skipped when --gpu is active
        if item.get_closest_marker("sw_vulkan") and gpu_requested:
            item.add_marker(pytest.mark.skip(reason="--gpu active, skipping sw_vulkan"))
