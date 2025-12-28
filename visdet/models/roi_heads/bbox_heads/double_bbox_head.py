# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from visdet.cv.cnn import ConvModule
from visdet.models.backbones.resnet import Bottleneck
from visdet.registry import MODELS

from .bbox_head import BBoxHead


class BasicResBlock(nn.Module):
    """Basic residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_cfg: dict | None = None,
        norm_cfg: dict = dict(type="BN"),
    ) -> None:
        super(BasicResBlock, self).__init__()

        # main path
        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
        )
        self.conv2 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

        # identity path
        self.conv_identity = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        identity = self.conv_identity(identity)
        out = x + identity

        out = self.relu(out)
        return out


@MODELS.register_module()
class DoubleConvFCBBoxHead(BBoxHead):
    r"""Bbox head used in Double-Head R-CNN"""

    def __init__(
        self,
        num_convs: int = 0,
        num_fcs: int = 0,
        conv_out_channels: int = 1024,
        fc_out_channels: int = 1024,
        conv_cfg: dict | None = None,
        norm_cfg: dict = dict(type="BN"),
        **kwargs,
    ) -> None:
        kwargs.setdefault("with_avg_pool", True)
        super(DoubleConvFCBBoxHead, self).__init__(**kwargs)
        assert self.with_avg_pool
        assert num_convs > 0
        assert num_fcs > 0
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # increase the channel of input features
        self.res_block = BasicResBlock(self.in_channels, self.conv_out_channels)

        # add conv heads
        self.conv_branch = nn.ModuleList()
        for i in range(self.num_convs):
            self.conv_branch.append(
                Bottleneck(
                    inplanes=self.conv_out_channels,
                    planes=self.conv_out_channels // 4,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                )
            )
        # add fc heads
        self.fc_branch = nn.ModuleList()
        for i in range(self.num_fcs):
            fc_in_channels = self.in_channels * self.roi_feat_area if i == 0 else self.fc_out_channels
            self.fc_branch.append(nn.Linear(fc_in_channels, self.fc_out_channels))

        out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
        self.fc_reg = nn.Linear(self.conv_out_channels, out_dim_reg)

        self.fc_cls = nn.Linear(self.fc_out_channels, self.num_classes + 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_cls, x_reg):
        # conv head
        x_conv = self.res_block(x_reg)

        for conv in self.conv_branch:
            x_conv = conv(x_conv)

        if self.with_avg_pool:
            x_conv = self.avg_pool(x_conv)

        x_conv = x_conv.view(x_conv.size(0), -1)
        bbox_pred = self.fc_reg(x_conv)

        # fc head
        x_fc = x_cls.view(x_cls.size(0), -1)
        for fc in self.fc_branch:
            x_fc = self.relu(fc(x_fc))

        cls_score = self.fc_cls(x_fc)

        return cls_score, bbox_pred
