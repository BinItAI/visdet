# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from visdet.cv.cnn import build_conv_layer, build_norm_layer
from visdet.engine.model import BaseModule, Sequential
from visdet.models.backbones.resnet import Bottleneck as _Bottleneck
from visdet.models.backbones.resnet import ResNet
from visdet.registry import MODELS


class Bottle2neck(_Bottleneck):
    """Bottle2neck block for Res2Net.

    Multi-scale bottleneck block that splits intermediate features into
    multiple scales and processes them separately before concatenation.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of intermediate channels (per scale).
        scales (int): Number of scales. Default: 4
        base_width (int): Basic width of each scale. Default: 26
        stage_type (str): Type of stage ('normal' or 'stage'). Default: 'normal'
        kwargs (dict): Additional arguments for base Bottleneck.
    """

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        scales=4,
        base_width=26,
        stage_type="normal",
        **kwargs,
    ):
        super(Bottle2neck, self).__init__(inplanes, planes, **kwargs)
        assert scales > 1, "Res2Net degenerates to ResNet when scales = 1."
        width = int(math.floor(self.planes * (base_width / 64)))

        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, width * scales, postfix=1)  # type: ignore[unresolved-attribute]
        self.norm3_name, norm3 = build_norm_layer(self.norm_cfg, self.planes * self.expansion, postfix=3)  # type: ignore[unresolved-attribute]

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.inplanes,
            width * scales,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False,
        )
        self.add_module(self.norm1_name, norm1)

        if stage_type == "stage" and self.conv2_stride != 1:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=self.conv2_stride, padding=1)

        convs = []
        bns = []
        for i in range(scales - 1):
            convs.append(
                build_conv_layer(
                    self.conv_cfg,
                    width,
                    width,
                    kernel_size=3,
                    stride=self.conv2_stride,
                    padding=self.dilation,
                    dilation=self.dilation,
                    bias=False,
                )
            )
            bns.append(build_norm_layer(self.norm_cfg, width, postfix=i + 1)[1])

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = build_conv_layer(
            self.conv_cfg,
            width * scales,
            self.planes * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.add_module(self.norm3_name, norm3)

        self.stage_type = stage_type  # type: ignore[unresolved-attribute]
        self.scales = scales  # type: ignore[unresolved-attribute]
        self.width = width  # type: ignore[unresolved-attribute]

        # Remove conv2 since we replaced it with multi-scale convs
        delattr(self, "conv2")
        delattr(self, self.norm2_name)

    @property
    def norm1(self):
        """nn.Module: normalization layer after conv1"""
        return getattr(self, self.norm1_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after conv3"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward pass with multi-scale processing."""

        def _inner_forward(x):
            identity = x

            # Conv1: 1x1 convolution to increase width
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            # Split into scales along channel dimension
            spx = torch.split(out, self.width, 1)
            sp = self.convs[0](spx[0].contiguous())
            sp = self.relu(self.bns[0](sp))
            out = sp

            # Process intermediate scales
            for i in range(1, self.scales - 1):
                if self.stage_type == "stage":
                    sp = spx[i]
                else:
                    sp = sp + spx[i]
                sp = self.convs[i](sp.contiguous())
                sp = self.relu(self.bns[i](sp))
                out = torch.cat((out, sp), 1)

            # Handle last scale
            if self.stage_type == "normal" or self.conv2_stride == 1:
                out = torch.cat((out, spx[self.scales - 1]), 1)
            elif self.stage_type == "stage":
                out = torch.cat((out, self.pool(spx[self.scales - 1])), 1)

            # Conv3: 1x1 convolution to final dimension
            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            return out

        if self.with_cp and x.requires_grad:
            import torch.utils.checkpoint as cp

            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)
        return out


class Res2Layer(Sequential):
    """Res2Layer to build Res2Net style backbone.

    Args:
        block (nn.Module): Block class used to build layer.
        inplanes (int): Number of input channels.
        planes (int): Number of output channels per scale.
        num_blocks (int): Number of blocks in this layer.
        stride (int): Stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when downsampling.
            Default: True
        conv_cfg (dict): Config dict for convolution layer. Default: None
        norm_cfg (dict): Config dict for normalization layer. Default: dict(type='BN')
        scales (int): Number of scales. Default: 4
        base_width (int): Basic width of each scale. Default: 26
        kwargs (dict): Additional arguments passed to blocks.
    """

    def __init__(
        self,
        block,
        inplanes,
        planes,
        num_blocks,
        stride=1,
        avg_down=True,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        scales=4,
        base_width=26,
        **kwargs,
    ):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(
                    kernel_size=stride,
                    stride=stride,
                    ceil_mode=True,
                    count_include_pad=False,
                ),
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                build_norm_layer(norm_cfg, planes * block.expansion)[1],
            )

        layers = []
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                scales=scales,
                base_width=base_width,
                stage_type="stage",
                **kwargs,
            )
        )
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    scales=scales,
                    base_width=base_width,
                    stage_type="normal",
                    **kwargs,
                )
            )
        super(Res2Layer, self).__init__(*layers)


@MODELS.register_module()
class Res2Net(ResNet):
    """Res2Net backbone.

    Res2Net improves ResNet by replacing the standard 3x3 convolution
    in bottleneck blocks with multi-scale branches, allowing the network
    to process features at different scales within each block.

    Args:
        depth (int): Depth of backbone from {50, 101, 152}. Default: 50
        scales (int): Number of scales in each bottleneck block. Default: 4
        base_width (int): Basic width of each scale. Default: 26
        in_channels (int): Number of input image channels. Default: 3
        num_stages (int): Number of stages. Default: 4
        strides (Sequence[int]): Strides of first block in each stage.
            Default: (1, 2, 2, 2)
        dilations (Sequence[int]): Dilations of each stage. Default: (1, 1, 1, 1)
        out_indices (Sequence[int]): Output from which stages. Default: (0, 1, 2, 3)
        frozen_stages (int): Stages to freeze (stop gradient). Default: -1 (none)
        norm_cfg (dict): Config for normalization layer. Default: dict(type='BN')
        norm_eval (bool): Whether to set norm layers to eval mode. Default: False
        with_cp (bool): Use checkpoint to save memory. Default: False
        zero_init_residual (bool): Initialize last BN in residual blocks to zero.
            Default: True
        deep_stem (bool): Replace 7x7 conv with 3x3 conv stem. Default: True
        avg_down (bool): Use AvgPool for downsampling. Default: True
        pretrained (str, optional): Path to pretrained weights. Default: None
        init_cfg (dict or list[dict], optional): Initialization config. Default: None
    """

    arch_settings = {
        50: (Bottle2neck, (3, 4, 6, 3)),
        101: (Bottle2neck, (3, 4, 23, 3)),
        152: (Bottle2neck, (3, 8, 36, 3)),
    }

    def __init__(
        self,
        depth=50,
        scales=4,
        base_width=26,
        deep_stem=True,
        avg_down=True,
        **kwargs,
    ):
        self.scales = scales  # type: ignore[unresolved-attribute]
        self.base_width = base_width  # type: ignore[unresolved-attribute]
        super(Res2Net, self).__init__(
            depth=depth,
            deep_stem=deep_stem,
            avg_down=avg_down,
            **kwargs,
        )

    def make_res_layer(self, **kwargs):
        """Build ResLayer for Res2Net."""
        return Res2Layer(
            scales=self.scales,
            base_width=self.base_width,
            base_channels=self.base_channels,
            **kwargs,
        )
