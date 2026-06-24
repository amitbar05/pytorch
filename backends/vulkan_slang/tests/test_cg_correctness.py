"""CG.3 regression: packed16 vec4 rewrite must not fire when has_welford.

With fp16 GroupNorm (Welford reduction), the packed16 path would vectorize
the body into float4, destroying Welford's sequential ordering and producing
garbage mean/m2.

This file is created as part of CG.3 fix.
"""

import torch
import torch.nn.functional as F
import pytest

RTOL = 1e-4
ATOL = 1e-4


@pytest.fixture(autouse=True)
def setup():
    try:
        import torch_vulkan
        if not torch_vulkan.is_available():
            pytest.skip("No Vulkan device")
    except ImportError:
        pytest.skip("torch_vulkan not installed")


def _to_vulkan(t):
    """Move tensor to vulkan device."""
    return t.to("vulkan")


class TestPacked16Vec4WelfordGuard:
    """CG.3: packed16 path in _vec4_pw_eligible must return False when
    has_welford is True, so that Welford reductions keep their sequential
    ordering and produce correct mean/m2."""

    def test_group_norm_fp16_forward_matches_cpu(self):
        """F.group_norm with fp16 input must match CPU forward pass.

        The input shape (1, 16, 8, 8) with num_groups=4 gives reduction
        extent 8*8=16 per group — large enough to exercise the Welford
        reduction path rather than a trivial fallback.
        """
        torch.manual_seed(0)
        x_cpu = torch.randn(1, 16, 8, 8, dtype=torch.float16)
        x_vk = _to_vulkan(x_cpu)

        out_cpu = F.group_norm(x_cpu, num_groups=4)
        out_vk = F.group_norm(x_vk, num_groups=4)

        torch.testing.assert_close(out_vk.cpu(), out_cpu, rtol=RTOL, atol=ATOL)

    def test_group_norm_fp16_no_packed16_vec4_in_emitted_source(self):
        """The emitted Slang source must NOT contain packed16 vec4
        identifiers (_pvw_in_ / _pvw_out_), confirming the guard
        prevented the packed16 path from rewriting the Welford body."""
        from torch_vulkan.inductor import runtime as rt

        captured: list[str] = []
        orig = rt.compile_slang_to_spirv

        def spy(src, entry="computeMain", cache_key=None, **kwargs):
            captured.append(src)
            return orig(src, entry=entry, cache_key=cache_key, **kwargs)

        rt.compile_slang_to_spirv = spy
        try:
            torch.manual_seed(0)
            x = torch.randn(1, 16, 8, 8, dtype=torch.float16)
            x_vk = _to_vulkan(x)

            # Warm-up compile
            F.group_norm(x_vk, num_groups=4)
        finally:
            rt.compile_slang_to_spirv = orig

        # Find any kernel that is a group_norm pointwise (non-reduction
        # kernel that has the packed16 vec4 prefix patterns).
        pvw_sources = [
            s for s in captured
            if "_pvw_in_" in s or "_pvw_out_" in s
        ]
        assert not pvw_sources, (
            "packed16 vec4 identifiers (_pvw_in_/_pvw_out_) must not appear "
            "in any emitted kernel when has_welford is True "
            f"(found {len(pvw_sources)} matching kernels)"
        )
