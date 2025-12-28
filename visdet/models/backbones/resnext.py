# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Any

from visdet.cv.cnn import build_conv_layer, build_norm_layer
from visdet.models.backbones.resnet import Bottleneck as _Bottleneck
from visdet.models.backbones.resnet import ResNet
from visdet.models.utils import ResLayer
from visdet.registry import MODELS


class Bottleneck(_Bottleneck):
    """Bottleneck block for ResNeXt.

    If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
    it is "caffe", the stride-two layer is the first 1x1 conv layer.
    """

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        groups: int = 1,
        base_width: int = 4,
        base_channels: int = 64,
        **kwargs: Any,
    ) -> None:
        # Extract groups and base_width before calling parent
        self.groups = groups
        self.base_width = base_width
        self.base_channels = base_channels

        super(Bottleneck, self).__init__(inplanes, planes, **kwargs)

        if groups == 1:
            width = self.planes
        else:
            width = math.floor(self.planes * (base_width / base_channels)) * groups

        # Rebuild norm and conv layers with grouped convolutions
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, width, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, width, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(self.norm_cfg, self.planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.inplanes,
            width,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False,
        )
        self.add_module(self.norm1_name, norm1)

        # Use grouped convolution with pure PyTorch fallback for DCN
        self.conv2 = build_conv_layer(
            self.conv_cfg,
            width,
            width,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=self.dilation,
            dilation=self.dilation,
            groups=groups,
            bias=False,
        )

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            self.conv_cfg,
            width,
            self.planes * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.add_module(self.norm3_name, norm3)


@MODELS.register_module()
class ResNeXt(ResNet):
    """ResNeXt backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        groups (int): Group of resnext.
        base_width (int): Base width of resnext.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(self, groups=1, base_width=4, **kwargs):
        self.groups = groups
        self.base_width = base_width
        super(ResNeXt, self).__init__(**kwargs)

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``"""
        return ResLayer(
            groups=self.groups,
            base_width=self.base_width,
            base_channels=self.base_channels,
            **kwargs,
        )
