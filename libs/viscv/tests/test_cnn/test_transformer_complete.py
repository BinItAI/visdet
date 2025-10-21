# Copyright (c) OpenMMLab. All rights reserved.

import pytest
import torch
from viscv.cnn.bricks.transformer import (
    FFN,
    AdaptivePadding,
    BaseTransformerLayer,
    MultiheadAttention,
    PatchEmbed,
    PatchMerging,
    TransformerLayerSequence,
    build_attention,
    build_feedforward_network,
    build_transformer_layer,
    build_transformer_layer_sequence,
)


def test_build_functions():
    """Test build functions for transformer components."""
    # Test build_attention
    cfg = dict(type="MultiheadAttention", embed_dims=256, num_heads=8)
    attn = build_attention(cfg)
    assert isinstance(attn, MultiheadAttention)
    assert attn.embed_dims == 256
    assert attn.num_heads == 8

    # Test build_feedforward_network
    cfg = dict(type="FFN", embed_dims=256, feedforward_channels=1024)
    ffn = build_feedforward_network(cfg)
    assert isinstance(ffn, FFN)
    assert ffn.embed_dims == 256
    assert ffn.feedforward_channels == 1024

    # Test build_transformer_layer
    cfg = dict(
        type="BaseTransformerLayer",
        attn_cfgs=dict(type="MultiheadAttention", embed_dims=256, num_heads=8),
        ffn_cfgs=dict(type="FFN", embed_dims=256, feedforward_channels=1024),
        operation_order=("self_attn", "norm", "ffn", "norm"),
    )
    layer = build_transformer_layer(cfg)
    assert isinstance(layer, BaseTransformerLayer)
    assert layer.embed_dims == 256

    # Test build_transformer_layer_sequence
    cfg = dict(
        type="TransformerLayerSequence",
        transformerlayers=dict(
            type="BaseTransformerLayer",
            attn_cfgs=dict(type="MultiheadAttention", embed_dims=256, num_heads=8),
            ffn_cfgs=dict(type="FFN", embed_dims=256, feedforward_channels=1024),
            operation_order=("self_attn", "norm", "ffn", "norm"),
        ),
        num_layers=2,
    )
    sequence = build_transformer_layer_sequence(cfg)
    assert isinstance(sequence, TransformerLayerSequence)
    assert sequence.num_layers == 2


def test_adaptive_padding():
    """Test AdaptivePadding module."""
    for padding in ("same", "corner"):
        kernel_size = 16
        stride = 16
        dilation = 1
        input = torch.rand(1, 1, 15, 17)
        adap_pad = AdaptivePadding(kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        out = adap_pad(input)
        # padding to divisible by 16
        assert (out.shape[2], out.shape[3]) == (16, 32)

        input = torch.rand(1, 1, 16, 17)
        out = adap_pad(input)
        # padding to divisible by 16
        assert (out.shape[2], out.shape[3]) == (16, 32)

    # assert only support "same" "corner"
    with pytest.raises(AssertionError):
        AdaptivePadding(kernel_size=kernel_size, stride=stride, dilation=dilation, padding=1)


def test_patch_embed():
    """Test PatchEmbed module."""
    B = 2
    H = 3
    W = 4
    C = 3
    embed_dims = 10
    kernel_size = 3
    stride = 1
    dummy_input = torch.rand(B, C, H, W)
    patch_merge_1 = PatchEmbed(
        in_channels=C,
        embed_dims=embed_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        dilation=1,
        norm_cfg=None,
    )

    x1, shape = patch_merge_1(dummy_input)
    # test out shape
    assert x1.shape == (2, 2, 10)
    # test outsize is correct
    assert shape == (1, 2)
    # test L = out_h * out_w
    assert shape[0] * shape[1] == x1.shape[1]


def test_patch_merging():
    """Test PatchMerging module."""
    # Test the model with int padding
    in_c = 3
    out_c = 4
    kernel_size = 3
    stride = 3
    padding = 1
    dilation = 1
    bias = False
    # test the case with int padding
    patch_merge = PatchMerging(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
    )
    B, L, C = 1, 100, 3
    input_size = (10, 10)
    x = torch.rand(B, L, C)
    x_out, out_size = patch_merge(x, input_size)
    assert x_out.size() == (1, 16, 4)
    assert out_size == (4, 4)
    # assert out size is consistent with real output
    assert x_out.size(1) == out_size[0] * out_size[1]


def test_multiheadattention():
    """Test MultiheadAttention module."""
    batch_dim = 2
    embed_dim = 5
    num_query = 100
    attn_batch_first = MultiheadAttention(
        embed_dims=5,
        num_heads=5,
        attn_drop=0,
        proj_drop=0,
        dropout_layer=dict(type="DropPath", drop_prob=0.0),
        batch_first=True,
    )

    attn_query_first = MultiheadAttention(
        embed_dims=5,
        num_heads=5,
        attn_drop=0,
        proj_drop=0,
        dropout_layer=dict(type="DropPath", drop_prob=0.0),
        batch_first=False,
    )

    param_dict = dict(attn_query_first.named_parameters())
    for n, v in attn_batch_first.named_parameters():
        param_dict[n].data = v.data

    input_batch_first = torch.rand(batch_dim, num_query, embed_dim)
    input_query_first = input_batch_first.transpose(0, 1)

    assert torch.allclose(
        attn_query_first(input_query_first).sum(),
        attn_batch_first(input_batch_first).sum(),
    )

    key_batch_first = torch.rand(batch_dim, num_query, embed_dim)
    key_query_first = key_batch_first.transpose(0, 1)

    assert torch.allclose(
        attn_query_first(input_query_first, key_query_first).sum(),
        attn_batch_first(input_batch_first, key_batch_first).sum(),
    )

    identity = torch.ones_like(input_query_first)

    # check deprecated arguments can be used normally
    assert torch.allclose(
        attn_query_first(input_query_first, key_query_first, identity=identity).sum(),
        attn_batch_first(input_batch_first, key_batch_first).sum() + identity.sum() - input_batch_first.sum(),
    )


def test_basetransformerlayer():
    """Test BaseTransformerLayer module."""
    # Test basic functionality
    operation_order = ("self_attn", "norm", "ffn", "norm")
    baselayer = BaseTransformerLayer(
        operation_order=operation_order,
        batch_first=True,
        attn_cfgs=dict(
            type="MultiheadAttention",
            embed_dims=256,
            num_heads=8,
        ),
        ffn_cfgs=dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=1024,
        ),
    )

    x = torch.rand(2, 10, 256)
    output = baselayer(x)
    assert output.shape == torch.Size([2, 10, 256])

    # Test with cross attention
    operation_order = ("self_attn", "norm", "cross_attn", "norm", "ffn", "norm")
    baselayer = BaseTransformerLayer(
        operation_order=operation_order,
        batch_first=True,
        attn_cfgs=[
            dict(type="MultiheadAttention", embed_dims=256, num_heads=8),
            dict(type="MultiheadAttention", embed_dims=256, num_heads=8),
        ],
        ffn_cfgs=dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=1024,
        ),
    )

    query = torch.rand(2, 10, 256)
    key = value = torch.rand(2, 20, 256)
    output = baselayer(query, key, value)
    assert output.shape == torch.Size([2, 10, 256])

    # Test pre-norm
    operation_order = ("norm", "self_attn", "norm", "ffn")
    baselayer = BaseTransformerLayer(
        operation_order=operation_order,
        batch_first=True,
        attn_cfgs=dict(
            type="MultiheadAttention",
            embed_dims=256,
            num_heads=8,
        ),
        ffn_cfgs=dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=1024,
        ),
    )
    assert baselayer.pre_norm is True

    x = torch.rand(2, 10, 256)
    output = baselayer(x)
    assert output.shape == torch.Size([2, 10, 256])


def test_transformerlayersequence():
    """Test TransformerLayerSequence module."""
    # Test with dict config
    transformerlayers = dict(
        type="BaseTransformerLayer",
        attn_cfgs=dict(type="MultiheadAttention", embed_dims=256, num_heads=8),
        ffn_cfgs=dict(type="FFN", embed_dims=256, feedforward_channels=1024),
        operation_order=("self_attn", "norm", "ffn", "norm"),
    )
    num_layers = 3

    sequence = TransformerLayerSequence(transformerlayers=transformerlayers, num_layers=num_layers)

    assert sequence.num_layers == 3
    assert len(sequence.layers) == 3
    assert sequence.embed_dims == 256

    # Test forward
    query = key = value = torch.rand(2, 10, 256)
    output = sequence(query, key, value)
    assert output.shape == torch.Size([2, 10, 256])

    # Test with list config
    transformerlayers = [
        dict(
            type="BaseTransformerLayer",
            attn_cfgs=dict(type="MultiheadAttention", embed_dims=256, num_heads=8),
            ffn_cfgs=dict(type="FFN", embed_dims=256, feedforward_channels=1024),
            operation_order=("self_attn", "norm", "ffn", "norm"),
        )
        for _ in range(2)
    ]

    sequence = TransformerLayerSequence(transformerlayers=transformerlayers, num_layers=2)

    assert sequence.num_layers == 2
    assert len(sequence.layers) == 2

    # Test forward with positional encoding and masks
    query = key = value = torch.rand(2, 10, 256)
    query_pos = key_pos = torch.rand(2, 10, 256)
    # Each layer has 1 self-attention, so we need 1 mask per layer
    attn_masks = None  # Let it be handled automatically

    output = sequence(query, key, value, query_pos=query_pos, key_pos=key_pos, attn_masks=attn_masks)
    assert output.shape == torch.Size([2, 10, 256])


if __name__ == "__main__":
    test_build_functions()
    test_adaptive_padding()
    test_patch_embed()
    test_patch_merging()
    test_multiheadattention()
    test_basetransformerlayer()
    test_transformerlayersequence()
    print("All tests passed!")
