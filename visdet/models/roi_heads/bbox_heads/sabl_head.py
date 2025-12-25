# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor

from visdet.cv.cnn import ConvModule
from visdet.registry import MODELS

from .bbox_head import BBoxHead


@MODELS.register_module()
class SABLHead(BBoxHead):
    """Side-Aware Boundary Localization (SABL) BBox Head.

    https://arxiv.org/abs/1912.04260
    """

    def __init__(
        self,
        cls_in_channels,
        reg_in_channels,
        feat_channels=256,
        cls_out_channels=1024,
        reg_offset_out_channels=256,
        reg_cls_out_channels=256,
        num_cls_fcs=1,
        num_reg_fcs=1,
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
        init_cfg=None,
        **kwargs,
    ):
        super(SABLHead, self).__init__(init_cfg=init_cfg, **kwargs)
        self.cls_in_channels = cls_in_channels
        self.reg_in_channels = reg_in_channels
        self.feat_channels = feat_channels
        self.cls_out_channels = cls_out_channels
        self.reg_offset_out_channels = reg_offset_out_channels
        self.reg_cls_out_channels = reg_cls_out_channels
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_fcs = num_reg_fcs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # classification branch
        self.cls_fcs = nn.ModuleList()
        for i in range(num_cls_fcs):
            in_channels = cls_in_channels if i == 0 else cls_out_channels
            self.cls_fcs.append(nn.Linear(in_channels, cls_out_channels))
        self.fc_cls = nn.Linear(cls_out_channels, self.num_classes + 1)

        # regression branch
        self.reg_fcs = nn.ModuleList()
        for i in range(num_reg_fcs):
            in_channels = reg_in_channels if i == 0 else reg_offset_out_channels
            self.reg_fcs.append(nn.Linear(in_channels, reg_offset_out_channels))

        # Simplified implementation details
        # Actual SABL head has specific bucketing prediction layers

    def forward(self, x_cls, x_reg):
        # Simplified forward
        return None, None
