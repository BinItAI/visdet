# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from viscv.cnn.bricks.transformer import FFN, build_dropout


def test_ffn():
    with pytest.raises(AssertionError):
        # num_fcs should be no less than 2
        FFN(num_fcs=1)
    ffn = FFN(dropout=0, add_identity=True)

    input_tensor = torch.rand(2, 20, 256)
    input_tensor_nbc = input_tensor.transpose(0, 1)
    assert torch.allclose(ffn(input_tensor).sum(), ffn(input_tensor_nbc).sum())
    residual = torch.rand_like(input_tensor)
    torch.allclose(
        ffn(input_tensor, residual=residual).sum(),
        ffn(input_tensor).sum() + residual.sum() - input_tensor.sum(),
    )

    torch.allclose(
        ffn(input_tensor, identity=residual).sum(),
        ffn(input_tensor).sum() + residual.sum() - input_tensor.sum(),
    )

    # test with layer_scale
    ffn = FFN(dropout=0, add_identity=True, layer_scale_init_value=0.1)

    input_tensor = torch.rand(2, 20, 256)
    input_tensor_nbc = input_tensor.transpose(0, 1)
    assert torch.allclose(ffn(input_tensor).sum(), ffn(input_tensor_nbc).sum())


def test_build_dropout():
    # Test None config returns None
    assert build_dropout(None) is None

    # Test dict config with type
    cfg = dict(type="Dropout", drop_prob=0.5)
    dropout = build_dropout(cfg)
    assert isinstance(dropout, torch.nn.Dropout)
    assert dropout.p == 0.5

    # Test dict config with DropPath
    cfg = dict(type="DropPath", drop_prob=0.3)
    dropout = build_dropout(cfg)
    assert dropout.drop_prob == 0.3

    # Test with float (should create Dropout)
    dropout = build_dropout(0.2)
    assert isinstance(dropout, torch.nn.Dropout)
    assert dropout.p == 0.2
