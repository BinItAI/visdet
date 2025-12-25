# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from visdet.cv.cnn import ConvModule
from visdet.models.utils import multi_apply
from visdet.registry import MODELS, TASK_UTILS

from .base_dense_head import BaseDenseHead


@MODELS.register_module()
class AnchorFreeHead(BaseDenseHead):
    """Anchor-free head (FCOS, RepPoints, etc.).

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        stacked_convs (int): Number of stacking convs of the head.
        strides (tuple): Downsample factor of each feature map.
        conv_bias (bool | str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        bbox_coder (dict): Config of bbox coder.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        strides: tuple = (4, 8, 16, 32, 64),
        conv_bias: str | bool = "auto",
        loss_cls: dict = dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox: dict = dict(type="IoULoss", loss_weight=1.0),
        bbox_coder: dict = dict(type="DistancePointBBoxCoder"),
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        train_cfg: dict | None = None,
        test_cfg: dict | None = None,
        init_cfg: dict | list[dict] = dict(
            type="Normal",
            layer="Conv2d",
            std=0.01,
            override=dict(type="Normal", name="conv_cls", std=0.01, bias_prob=0.01),
        ),
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        assert conv_bias == "auto" or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)

        self.prior_generator = TASK_UTILS.build(dict(type="MlvlPointGenerator", strides=strides))

        self.num_base_priors = self.prior_generator.num_base_priors[0]

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()

    def _init_cls_convs(self) -> None:
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias,
                )
            )

    def _init_reg_convs(self) -> None:
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias,
                )
            )

    def _init_predictor(self) -> None:
        """Initialize predictor layers of the head."""
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

    def forward(self, feats: tuple[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually contain classification scores and bbox predictions.
                cls_scores (list[Tensor]): Box scores for each scale level,
                    each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * 4.
        """
        return multi_apply(self.forward_single, feats)[:2]

    def forward_single(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.

        Returns:
            tuple: Scores for each class, bbox predictions, features
                after classification and regression conv layers, some
                models needs these features like FCOS.
        """
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        return cls_score, bbox_pred, cls_feat, reg_feat

    @abstractmethod
    def loss_by_feat(self, **kwargs) -> dict:
        """Calculate the loss based on the features extracted by the detection head."""
        pass

    @abstractmethod
    def get_targets(self, **kwargs) -> tuple:
        """Compute regression and classification targets."""
        pass
