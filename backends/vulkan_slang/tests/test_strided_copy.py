"""Regression test for dispatch_strided_copy with storage_offset > 0."""
import sys
import pytest
import torch


def _vulkan_available():
    try:
        import torch_vulkan
        return torch_vulkan.is_available()
    except ImportError:
        return False


@pytest.fixture(autouse=True)
def setup():
    try:
        import torch_vulkan
        if not torch_vulkan.is_available():
            pytest.skip("No Vulkan device")
        if "torch.vulkan" not in sys.modules:
            torch_vulkan._register()
    except ImportError:
        pytest.skip("torch_vulkan not installed")


def test_clone_offset_view_1d():
    """1D tensor with storage_offset > 0 clones correctly."""
    x = torch.arange(8, dtype=torch.float32).to("vulkan:0")
    # Slice: elements [4:8] with offset=4
    x_slice = x[4:]
    assert x_slice.storage_offset() == 4
    assert x_slice.is_contiguous()
    # .clone() must produce correct values (not zeros or wrong offset)
    result = x_slice.clone().cpu()
    expected = torch.arange(4, 8, dtype=torch.float32)
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=0)


def test_clone_offset_view_4d_contiguous():
    """4D contiguous tensor with storage_offset > 0 clones correctly."""
    # Simulates weight group slice: weight_4d[4:8, :, :, :]
    w = torch.arange(96, dtype=torch.float32).reshape(8, 4, 3, 1).to("vulkan:0")
    w_slice = w[4:8, :, :, :]  # offset=48, shape=[4,4,3,1], strides=[12,3,1,1]
    assert w_slice.storage_offset() == 48
    assert w_slice.is_contiguous()
    result = w_slice.clone().cpu()
    expected = torch.arange(48, 96, dtype=torch.float32).reshape(4, 4, 3, 1)
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=0)


def test_clone_noncontiguous_offset():
    """4D non-contiguous tensor with storage_offset > 0 clones correctly."""
    # Simulates input group 1 slice: input_4d[:, 4:8, :, :]
    # After unsqueeze(-1): [2, 8, 32, 1], strides=[256, 32, 1, 1]
    # Group 1 slice: [2, 4, 32, 1], strides=[256, 32, 1, 1], offset=128
    x = torch.arange(512, dtype=torch.float32).reshape(2, 8, 32, 1).to("vulkan:0")
    x_g1 = x[:, 4:8, :, :]  # non-contiguous (stride[0]=256 != expected 128), offset=128
    assert x_g1.storage_offset() == 128
    assert not x_g1.is_contiguous()
    result = x_g1.clone().cpu()
    # Expected: elements at positions [n, c+4, l, 0] of original tensor
    expected = torch.arange(512, dtype=torch.float32).reshape(2, 8, 32, 1)[:, 4:8, :, :]
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=0)


def test_clone_offset_view_propagation():
    """Cloning an offset view produces the correct CONTENT, not zeros."""
    # The VulkanAllocator zero-initializes buffers. If clone() returns a
    # zero-initialized buffer without actually copying, this test fails.
    w = torch.ones(8, 4, 3, 1, dtype=torch.float32).to("vulkan:0") * 99.0
    w[4:, :, :, :] = torch.ones(4, 4, 3, 1) * 42.0
    w_g1 = w[4:, :, :, :]
    result = w_g1.clone().cpu()
    assert (result == 42.0).all(), f"Expected all 42, got {result.unique()}"


class TestDescriptorByteOffsetPerDtype:
    """S3.5: off_bytes = storage_offset * element_size computed correctly per dtype."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_storage_offset_per_dtype(self, dtype):
        # int8 excluded — Vulkan backend rejects cast_from_float32 → Char on slice.
        x_cpu = torch.arange(64, dtype=dtype)
        x = x_cpu.to("vulkan:0")
        view = x[8:]
        # The backend may materialize slices (storage_offset=0) or keep views;
        # either way the cloned values must match elements [8:64].
        y = view.clone().cpu()
        expected = torch.arange(8, 64, dtype=dtype)
        assert torch.equal(y, expected), (
            f"dtype={dtype}: got {y[:4]} expected {expected[:4]}"
        )


def test_strided_copy_float16_offset_view():
    """S3.5: float16 slice (nominal storage_offset=4) clones to correct values."""
    src = torch.arange(32, dtype=torch.float16).to("vulkan:0")[4:]
    dst = src.clone().cpu()
    assert torch.equal(dst, torch.arange(4, 32, dtype=torch.float16))


def test_strided_copy_non_contiguous_float16():
    """S3.5: non-contiguous float16 slice ([:, 2:]) clones correctly."""
    z = torch.arange(32, dtype=torch.float16).reshape(4, 8).to("vulkan:0")[:, 2:]
    w = z.clone().cpu()
    expected = torch.arange(32, dtype=torch.float16).reshape(4, 8)[:, 2:]
    assert torch.equal(w, expected)
