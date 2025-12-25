# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from visdet.cv.cnn import ConvModule
from visdet.registry import MODELS


class Bottleneck(nn.Module):
    """Bottleneck block for DilatedEncoder used in `YOLOF.

    <https://arxiv.org/abs/2103.09460>`.

    The Bottleneck contains three ConvLayers and one residual connection.

    Args:
        in_channels (int): The number of input channels.
        mid_channels (int): The number of middle output channels.
        dilation (int): Dilation rate.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """

    def __init__(
        self,
        in_channels,
        mid_channels,
        dilation,
        norm_cfg=dict(type="BN", requires_grad=True),
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvModule(in_channels, mid_channels, 1, norm_cfg=norm_cfg)
        self.conv2 = ConvModule(
            mid_channels,
            mid_channels,
            3,
            padding=dilation,
            dilation=dilation,
            norm_cfg=norm_cfg,
        )
        self.conv3 = ConvModule(mid_channels, in_channels, 1, norm_cfg=norm_cfg)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out


@MODELS.register_module()
class DilatedEncoder(nn.Module):
    """Dilated Encoder for YOLOF <https://arxiv.org/abs/2103.09460>`.

    This module contains two types of components:
        - the original FPN lateral convolution layer and fpn convolution layer,
              which are 1x1 conv + 3x3 conv
        - the dilated residual block

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        block_mid_channels (int): The number of middle block output channels
        num_residual_blocks (int): The number of residual blocks.
        block_dilations (list): The list of residual blocks dilation.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        block_mid_channels,
        num_residual_blocks,
        block_dilations,
        norm_cfg=dict(type="BN", requires_grad=True),
    ):
        super(DilatedEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_mid_channels = block_mid_channels
        self.num_residual_blocks = num_residual_blocks
        self.block_dilations = block_dilations
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self):
        self.lateral_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.lateral_norm = nn.BatchNorm2d(self.out_channels)
        self.fpn_conv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.fpn_norm = nn.BatchNorm2d(self.out_channels)
        encoder_blocks = []
        for i in range(self.num_residual_blocks):
            dilation = self.block_dilations[i]
            encoder_blocks.append(
                Bottleneck(
                    self.out_channels,
                    self.block_mid_channels,
                    dilation=dilation,
                    norm_cfg=self.norm_cfg,
                )
            )
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)

    def init_weights(self):
        # We use default initialization for now
        pass

    def forward(self, feature):
        # YOLOF only uses the last feature map (C5)
        out = self.lateral_norm(self.lateral_conv(feature[-1]))
        out = self.fpn_norm(self.fpn_conv(out))
        return (self.dilated_encoder_blocks(out),)
