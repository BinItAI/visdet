# Copyright (c) OpenMMLab. All rights reserved.
"""Anchor-free head base class for visdet."""

from abc import abstractmethod

import torch.nn as nn
from torch import Tensor

from visdet.engine.structures import InstanceData
from visdet.models.dense_heads.base_dense_head import BaseDenseHead
from visdet.models.task_modules.prior_generators import MlvlPointGenerator
from visdet.models.task_modules.samplers import PseudoSampler
from visdet.models.utils import multi_apply
from visdet.registry import MODELS, TASK_UTILS
from visdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig


@MODELS.register_module()
class AnchorFreeHead(BaseDenseHead):
    """Anchor-free head (FCOS, ATSS, etc.).

    Anchor-free heads directly predict bounding boxes from dense feature maps
    without predefined anchor boxes.

    Args:
        num_classes (int): Number of categories excluding background.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Defaults to 256.
        stacked_convs (int): Number of stacking convs. Defaults to 4.
        strides (list[int]): Strides of feature maps. Defaults to
            (4, 8, 16, 32, 64).
        dcn_on_last_conv (bool): If True, use DCN on the last conv layer.
            This requires CUDA extensions. Defaults to False.
        conv_bias (bool | str): If True, add bias to conv layers. If 'auto',
            add bias only for separable conv. Defaults to 'auto'.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        conv_cfg (dict, optional): Config dict for conv layer.
        norm_cfg (dict): Config dict for normalization layer.
        train_cfg (dict, optional): Training config.
        test_cfg (dict, optional): Testing config.
        init_cfg (dict, optional): Initialization config dict.
    """

    _version = 1

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        strides: tuple[int, ...] = (4, 8, 16, 32, 64),
        dcn_on_last_conv: bool = False,
        conv_bias: bool | str = "auto",
        loss_cls: ConfigType = dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_bbox: ConfigType = dict(type="IoULoss", loss_weight=1.0),
        bbox_coder: ConfigType = dict(type="DistancePointBBoxCoder"),
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="GN", num_groups=32, requires_grad=True),
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        init_cfg: OptMultiConfig = dict(
            type="Normal",
            layer="Conv2d",
            std=0.01,
            override=dict(type="Normal", name="conv_cls", std=0.01, bias_prob=0.01),
        ),
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == "auto" or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self.prior_generator = MlvlPointGenerator(strides)
        self.num_base_priors = self.prior_generator.num_base_priors[0]

        # Use sigmoid for classification
        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg["assigner"])
            if self.train_cfg.get("sampler", None) is not None:
                self.sampler = TASK_UTILS.build(self.train_cfg["sampler"], default_args=dict(context=self))
            else:
                self.sampler = PseudoSampler(context=self)

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
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                # Deformable conv not supported in pure PyTorch visdet
                raise NotImplementedError("DCN is not supported in visdet. Set dcn_on_last_conv=False.")
            else:
                conv = nn.Conv2d(chn, self.feat_channels, 3, stride=1, padding=1)

            self.cls_convs.append(
                nn.Sequential(
                    conv,
                    self._get_norm_layer(self.feat_channels),
                    nn.ReLU(inplace=True),
                )
            )

    def _init_reg_convs(self) -> None:
        """Initialize regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                raise NotImplementedError("DCN is not supported in visdet. Set dcn_on_last_conv=False.")
            else:
                conv = nn.Conv2d(chn, self.feat_channels, 3, stride=1, padding=1)

            self.reg_convs.append(
                nn.Sequential(
                    conv,
                    self._get_norm_layer(self.feat_channels),
                    nn.ReLU(inplace=True),
                )
            )

    def _init_predictor(self) -> None:
        """Initialize predictor layers of the head."""
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.conv_reg = nn.Conv2d(
            self.feat_channels,
            self.bbox_coder.encode_size,
            3,
            padding=1,
        )

    def _get_norm_layer(self, num_channels: int) -> nn.Module:
        """Get normalization layer based on norm_cfg."""
        if self.norm_cfg is None:
            return nn.Identity()
        norm_type = self.norm_cfg.get("type", "GN")
        if norm_type == "GN":
            num_groups = self.norm_cfg.get("num_groups", 32)
            return nn.GroupNorm(num_groups, num_channels)
        elif norm_type == "BN":
            return nn.BatchNorm2d(num_channels)
        else:
            return nn.Identity()

    def forward(self, x: tuple[Tensor, ...]) -> tuple[list[Tensor], list[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction.
                - cls_scores (list[Tensor]): Classification scores for all
                    scale levels, each is a 4D-tensor.
                - bbox_preds (list[Tensor]): Box predictions for all scale
                    levels, each is a 4D-tensor.
        """
        outs = multi_apply(self.forward_single, x)
        return outs[:2]

    def forward_single(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                - cls_score (Tensor): Cls scores for a single scale level.
                - bbox_pred (Tensor): Box predictions for a single scale level.
        Returns:
            tuple: Classification scores, bbox predictions, cls features and
            reg features.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.conv_cls(cls_feat)
        bbox_pred = self.conv_reg(reg_feat)
        return cls_score, bbox_pred, cls_feat, reg_feat

    @abstractmethod
    def loss_by_feat(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: list[InstanceData] | None = None,
    ) -> dict:
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels.
            bbox_preds (list[Tensor]): Box predictions for all scale levels.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance containing ground truth.
            batch_img_metas (list[dict]): Meta information of each image.
            batch_gt_instances_ignore: Ignored instances.

        Returns:
            dict: A dictionary of loss components.
        """
        raise NotImplementedError

    def get_targets(
        self,
        points: list[Tensor],
        batch_gt_instances: InstanceList,
    ) -> tuple:
        """Compute regression and classification targets for points.

        Args:
            points (list[Tensor]): Points of each fpn level.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance containing ground truth.

        Returns:
            tuple: Regression and classification targets.
        """
        raise NotImplementedError
