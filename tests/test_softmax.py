# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

import pytest
import torch
import torch.nn.functional as F

from quack.softmax import softmax


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
# @pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "N",
    [192, 256, 512, 760, 1024, 1128, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
)
@pytest.mark.parametrize("M", [1, 37, 199])
# @pytest.mark.parametrize("M", [1])
def test_softmax_forward(M, N, input_dtype):
    """Test Softmax forward pass against reference implementation."""
    device = "cuda"
    # Set tolerance based on dtype
    if input_dtype == torch.bfloat16:
        atol = 1e-2
        rtol = 1e-2
    elif input_dtype == torch.float16:
        atol = 1e-3
        rtol = 1e-3
    else:
        atol = 1e-4
        rtol = 1e-4
    torch.random.manual_seed(0)
    # Create input tensors (scale down to avoid overflow in softmax)
    x = 0.1 * torch.randn(M, N, device=device, dtype=input_dtype, requires_grad=False)
    x_ref = x.detach().clone()

    # Forward pass
    out = softmax(x)
    out_ref = F.softmax(x_ref, dim=-1)

    # Check output shape and dtype
    assert out.shape == x.shape
    assert out.dtype == input_dtype
    # Check accuracy
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)
    # Check softmax properties
    # Sum along last dimension should be 1
    sums = torch.sum(out, dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4)
    # All values should be positive
    assert (out >= 0).all()
    # All values should be <= 1
    assert (out <= 1).all()


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
def test_softmax_extreme_values(input_dtype):
    """Test Softmax with extreme input values."""
    device = "cuda"
    M, N = 16, 1024

    # Test with large positive values
    x_large = torch.full((M, N), 10.0, device=device, dtype=input_dtype)
    out_large = softmax(x_large)

    # Should be uniform since all values are the same
    expected = torch.full_like(out_large, 1.0 / N)
    torch.testing.assert_close(out_large, expected, atol=1e-3, rtol=1e-3)

    # Test with large negative values
    x_small = torch.full((M, N), -10.0, device=device, dtype=input_dtype)
    out_small = softmax(x_small)

    # Should also be uniform
    torch.testing.assert_close(out_small, expected, atol=1e-3, rtol=1e-3)

    # Test with mixed extreme values
    x_mixed = torch.zeros((M, N), device=device, dtype=input_dtype)
    x_mixed[:, 0] = 10.0  # One large value per row
    x_mixed[:, 1:] = -10.0  # Rest are small

    out_mixed = softmax(x_mixed)

    # First column should be close to 1, rest close to 0
    assert (out_mixed[:, 0] > 0.99).all()
    assert (out_mixed[:, 1:] < 0.01).all()


def test_softmax_numerical_stability():
    """Test that softmax is numerically stable."""
    device = "cuda"
    M, N = 8, 512

    # Create input with a wide range of values
    x = torch.randn(M, N, device=device, dtype=torch.float32)

    # Add large constant to test numerical stability
    x_shifted = x + 100.0

    out = softmax(x)
    out_shifted = softmax(x_shifted)

    # Results should be identical (softmax is translation invariant)
    torch.testing.assert_close(out, out_shifted, atol=1e-6, rtol=1e-6)
