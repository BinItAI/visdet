# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch.nn as nn
from viscv.cnn.bricks import (
    ConvTranspose2d,
    build_conv_layer,
    build_norm_layer,
    build_upsample_layer,
)
from viscv.cnn.bricks.norm import infer_abbr as infer_norm_abbr
from visengine.registry import MODELS


def test_build_conv_layer():
    with pytest.raises(TypeError):
        # cfg must be a dict
        cfg = "Conv2d"
        build_conv_layer(cfg)

    with pytest.raises(KeyError):
        # `type` must be in cfg
        cfg = dict(kernel_size=3)
        build_conv_layer(cfg)

    with pytest.raises(KeyError):
        # unsupported conv type
        cfg = dict(type="FancyConv")
        build_conv_layer(cfg)

    kwargs = dict(in_channels=4, out_channels=8, kernel_size=3, groups=2, dilation=2)
    cfg = None
    layer = build_conv_layer(cfg, **kwargs)
    assert isinstance(layer, nn.Conv2d)
    assert layer.in_channels == kwargs["in_channels"]
    assert layer.out_channels == kwargs["out_channels"]
    assert layer.kernel_size == (kwargs["kernel_size"], kwargs["kernel_size"])
    assert layer.groups == kwargs["groups"]
    assert layer.dilation == (kwargs["dilation"], kwargs["dilation"])

    cfg = dict(type="Conv")
    layer = build_conv_layer(cfg, **kwargs)
    assert isinstance(layer, nn.Conv2d)
    assert layer.in_channels == kwargs["in_channels"]
    assert layer.out_channels == kwargs["out_channels"]
    assert layer.kernel_size == (kwargs["kernel_size"], kwargs["kernel_size"])
    assert layer.groups == kwargs["groups"]
    assert layer.dilation == (kwargs["dilation"], kwargs["dilation"])

    cfg = dict(type="deconv")
    layer = build_conv_layer(cfg, **kwargs)
    assert isinstance(layer, nn.ConvTranspose2d)
    assert layer.in_channels == kwargs["in_channels"]
    assert layer.out_channels == kwargs["out_channels"]
    assert layer.kernel_size == (kwargs["kernel_size"], kwargs["kernel_size"])
    assert layer.groups == kwargs["groups"]
    assert layer.dilation == (kwargs["dilation"], kwargs["dilation"])


def test_infer_norm_abbr():
    with pytest.raises(TypeError):
        # class_type must be a class
        infer_norm_abbr(0)

    class MyNorm:
        _abbr_ = "mn"

    assert infer_norm_abbr(MyNorm) == "mn"

    class FancyBatchNorm:
        pass

    assert infer_norm_abbr(FancyBatchNorm) == "bn"

    class FancyInstanceNorm:
        pass

    assert infer_norm_abbr(FancyInstanceNorm) == "in"

    class FancyLayerNorm:
        pass

    assert infer_norm_abbr(FancyLayerNorm) == "ln"

    class FancyGroupNorm:
        pass

    assert infer_norm_abbr(FancyGroupNorm) == "gn"

    class FancyNorm:
        pass

    assert infer_norm_abbr(FancyNorm) == "norm_layer"


def test_build_norm_layer():
    with pytest.raises(TypeError):
        # cfg must be a dict
        cfg = "BN"
        build_norm_layer(cfg, 3)

    with pytest.raises(KeyError):
        # `type` must be in cfg
        cfg = dict()
        build_norm_layer(cfg, 3)

    with pytest.raises(KeyError):
        # unsupported norm type
        cfg = dict(type="FancyNorm")
        build_norm_layer(cfg, 3)

    with pytest.raises(AssertionError):
        # postfix must be int or str
        cfg = dict(type="BN")
        build_norm_layer(cfg, 3, postfix=[1, 2])

    with pytest.raises(AssertionError):
        # `num_groups` must be in cfg when using 'GN'
        cfg = dict(type="GN")
        build_norm_layer(cfg, 3)

    # test each type of norm layer in norm_cfg
    abbr_mapping = {
        "BN": "bn",
        "BN1d": "bn",
        "BN2d": "bn",
        "BN3d": "bn",
        "SyncBN": "bn",
        "GN": "gn",
        "LN": "ln",
        "IN": "in",
        "IN1d": "in",
        "IN2d": "in",
        "IN3d": "in",
    }
    for type_name, module in MODELS.module_dict.items():
        if type_name not in abbr_mapping:
            continue
        if type_name == "MMSyncBN":  # skip MMSyncBN
            continue
        for postfix in ["_test", 1]:
            for type_name_ in (type_name, module):
                cfg = dict(type=type_name_)
                if type_name == "GN":
                    cfg["num_groups"] = 3
                name, layer = build_norm_layer(cfg, 3, postfix=postfix)
                assert name == abbr_mapping[type_name] + str(postfix)
                assert isinstance(layer, module)
                if type_name == "GN":
                    assert layer.num_channels == 3
                    assert layer.num_groups == cfg["num_groups"]
                elif type_name != "LN":
                    assert layer.num_features == 3


def test_upsample_layer():
    with pytest.raises(TypeError):
        # cfg must be a dict
        cfg = "bilinear"
        build_upsample_layer(cfg)

    with pytest.raises(KeyError):
        # `type` must be in cfg
        cfg = dict()
        build_upsample_layer(cfg)

    with pytest.raises(KeyError):
        # unsupported activation type
        cfg = dict(type="FancyUpsample")
        build_upsample_layer(cfg)

    for type_name in ["nearest", "bilinear"]:
        cfg = dict(type=type_name)
        layer = build_upsample_layer(cfg)
        assert isinstance(layer, nn.Upsample)
        assert layer.mode == type_name

    cfg = dict(type=nn.Upsample)
    layer_from_cls = build_upsample_layer(cfg)
    assert isinstance(layer_from_cls, nn.Upsample)
    assert layer_from_cls.mode == "nearest"

    cfg = dict(type="deconv", in_channels=3, out_channels=3, kernel_size=3, stride=2)
    layer = build_upsample_layer(cfg)
    assert isinstance(layer, nn.ConvTranspose2d)

    for type_name in ("deconv", ConvTranspose2d):
        cfg = dict(type=ConvTranspose2d)
        kwargs = dict(in_channels=3, out_channels=3, kernel_size=3, stride=2)
        layer = build_upsample_layer(cfg, **kwargs)
        assert isinstance(layer, nn.ConvTranspose2d)
        assert layer.in_channels == kwargs["in_channels"]
        assert layer.out_channels == kwargs["out_channels"]
        assert layer.kernel_size == (kwargs["kernel_size"], kwargs["kernel_size"])
        assert layer.stride == (kwargs["stride"], kwargs["stride"])

        layer = build_upsample_layer(cfg, 3, 3, 3, 2)
        assert isinstance(layer, nn.ConvTranspose2d)
        assert layer.in_channels == kwargs["in_channels"]
        assert layer.out_channels == kwargs["out_channels"]
        assert layer.kernel_size == (kwargs["kernel_size"], kwargs["kernel_size"])
        assert layer.stride == (kwargs["stride"], kwargs["stride"])
