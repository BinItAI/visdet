"""Test PatchMerging compatibility with MMCV implementation."""

import torch
import torch.nn as nn
from visdet.models.layers import PatchMerging


def test_patch_merging_matches_unfold():
    """Test that PatchMerging produces same output as nn.Unfold."""
    # Test parameters
    batch_size = 2
    in_channels = 96
    out_channels = 192
    height, width = 32, 32
    stride = 2

    # Create input
    seq_len = height * width
    x = torch.randn(batch_size, seq_len, in_channels)

    # Create PatchMerging module
    patch_merge = PatchMerging(in_channels=in_channels, out_channels=out_channels, stride=stride)

    # Get output from PatchMerging
    with torch.no_grad():
        # Set weights to identity-like for testing
        patch_merge.norm.weight.fill_(1.0)
        patch_merge.norm.bias.fill_(0.0)
        patch_merge.reduction.weight.zero_()
        for i in range(min(out_channels, stride * stride * in_channels)):
            patch_merge.reduction.weight[i, i] = 1.0

    output, (h_out, w_out) = patch_merge(x, (height, width))

    # Manually compute expected output using nn.Unfold
    x_reshaped = x.view(batch_size, height, width, in_channels).permute(0, 3, 1, 2)
    unfold = nn.Unfold(kernel_size=stride, stride=stride)
    x_unfolded = unfold(x_reshaped).transpose(1, 2)

    # Apply same norm and reduction
    ln = nn.LayerNorm(stride * stride * in_channels)
    ln.weight.data.fill_(1.0)
    ln.bias.data.fill_(0.0)
    x_normed = ln(x_unfolded)

    # Apply reduction
    reduction = nn.Linear(stride * stride * in_channels, out_channels, bias=False)
    reduction.weight.data.zero_()
    for i in range(min(out_channels, stride * stride * in_channels)):
        reduction.weight.data[i, i] = 1.0
    expected = reduction(x_normed)

    # Compare outputs
    assert torch.allclose(output, expected, atol=1e-5), (
        f"Outputs don't match! Max diff: {(output - expected).abs().max()}"
    )

    # Check output shape
    assert h_out == height // stride
    assert w_out == width // stride
    assert output.shape == (batch_size, h_out * w_out, out_channels)

    print("✓ PatchMerging matches nn.Unfold implementation!")


def test_patch_merging_with_padding():
    """Test PatchMerging with odd dimensions that require padding."""
    batch_size = 1
    in_channels = 48
    out_channels = 96
    height, width = 31, 33  # Odd dimensions
    stride = 2

    # Create input
    seq_len = height * width
    x = torch.randn(batch_size, seq_len, in_channels)

    # Create PatchMerging module
    patch_merge = PatchMerging(in_channels=in_channels, out_channels=out_channels, stride=stride)

    # Get output
    output, (h_out, w_out) = patch_merge(x, (height, width))

    # Check padded dimensions
    h_padded = height + (stride - height % stride) % stride
    w_padded = width + (stride - width % stride) % stride

    assert h_out == h_padded // stride == 16
    assert w_out == w_padded // stride == 17
    assert output.shape == (batch_size, h_out * w_out, out_channels)

    print("✓ PatchMerging handles padding correctly!")


if __name__ == "__main__":
    test_patch_merging_matches_unfold()
    test_patch_merging_with_padding()
    print("\nAll tests passed! PatchMerging is compatible with MMCV.")
