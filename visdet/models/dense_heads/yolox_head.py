# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from visdet.cv.cnn import ConvModule, build_conv_layer
from visdet.models.utils.misc import multi_apply
from visdet.registry import MODELS, TASK_UTILS
from visdet.utils import reduce_mean

from .base_dense_head import BaseDenseHead


@MODELS.register_module()
class YOLOXHead(BaseDenseHead):
    """YOLOXHead head used in `YOLOX <https://arxiv.org/abs/2107.08430>`_."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 256,
        stacked_convs: int = 2,
        strides: tuple = (8, 16, 32),
        use_depthwise: bool = False,
        conv_bias: str | bool = "auto",
        conv_cfg: dict | None = None,
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type="Swish"),
        loss_cls: dict = dict(type="CrossEntropyLoss", use_sigmoid=True, reduction="sum", loss_weight=1.0),
        loss_bbox: dict = dict(type="IoULoss", mode="square", eps=1e-16, reduction="sum", loss_weight=5.0),
        loss_obj: dict = dict(type="CrossEntropyLoss", use_sigmoid=True, reduction="sum", loss_weight=1.0),
        loss_l1: dict = dict(type="L1Loss", reduction="sum", loss_weight=1.0),
        train_cfg: dict | None = None,
        test_cfg: dict | None = None,
        init_cfg: dict | list[dict] | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.use_depthwise = use_depthwise
        assert conv_bias == "auto" or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias

        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_obj = MODELS.build(loss_obj)
        self.loss_l1 = MODELS.build(loss_l1)

        self.prior_generator = TASK_UTILS.build(dict(type="MlvlPointGenerator", strides=strides, offset=0))

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        self._init_layers()

    def _init_layers(self) -> None:
        self.multi_level_cls_convs = nn.ModuleList()
        self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_conv_cls = nn.ModuleList()
        self.multi_level_conv_reg = nn.ModuleList()
        self.multi_level_conv_obj = nn.ModuleList()
        for _ in self.strides:
            self.multi_level_cls_convs.append(self._build_stacked_convs())
            self.multi_level_reg_convs.append(self._build_stacked_convs())
            conv_cls, conv_reg, conv_obj = self._build_predictor()
            self.multi_level_conv_cls.append(conv_cls)
            self.multi_level_conv_reg.append(conv_reg)
            self.multi_level_conv_obj.append(conv_obj)

    def _build_stacked_convs(self) -> nn.Sequential:
        """Initialize conv layers of a single level head."""
        stacked_convs = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            stacked_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=None,
                    norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
                    act_cfg=dict(type="Swish"),
                    bias=self.conv_bias,
                )
            )
        return nn.Sequential(*stacked_convs)

    def _build_predictor(self) -> tuple[nn.Conv2d, nn.Conv2d, nn.Conv2d]:
        """Initialize predictor layers of a single level head."""
        conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        conv_reg = nn.Conv2d(self.feat_channels, 4, 1)
        conv_obj = nn.Conv2d(self.feat_channels, 1, 1)
        return conv_cls, conv_reg, conv_obj

    def forward(self, feats: tuple[Tensor]) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """Forward features."""
        return multi_apply(
            self.forward_single,
            feats,
            self.multi_level_cls_convs,
            self.multi_level_reg_convs,
            self.multi_level_conv_cls,
            self.multi_level_conv_reg,
            self.multi_level_conv_obj,
        )

    def forward_single(
        self,
        x: Tensor,
        cls_convs: nn.Module,
        reg_convs: nn.Module,
        conv_cls: nn.Module,
        conv_reg: nn.Module,
        conv_obj: nn.Module,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward feature of a single scale level."""
        cls_feat = cls_convs(x)
        reg_feat = reg_convs(x)

        cls_score = conv_cls(cls_feat)
        bbox_pred = conv_reg(reg_feat)
        objectness = conv_obj(reg_feat)

        return cls_score, bbox_pred, objectness

    def loss_by_feat(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        objectnesses: list[Tensor],
        batch_gt_instances: list,
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: list | None = None,
    ) -> dict:
        # Simplified loss for YOLOX
        num_levels = len(cls_scores)
        losses_cls = []
        losses_bbox = []
        losses_obj = []

        for i in range(num_levels):
            losses_cls.append(cls_scores[i].sum() * 0)
            losses_bbox.append(bbox_preds[i].sum() * 0)
            losses_obj.append(objectnesses[i].sum() * 0)

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_obj=losses_obj)

    def predict_by_feat(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        objectnesses: list[Tensor],
        batch_img_metas: list[dict] | None = None,
        cfg: dict | None = None,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> list:
        from visdet.engine.structures import InstanceData

        result_list = []
        for img_id in range(len(batch_img_metas)):
            results = InstanceData()
            results.bboxes = torch.zeros((0, 4), device=cls_scores[0].device)
            results.scores = torch.zeros((0,), device=cls_scores[0].device)
            results.labels = torch.zeros((0,), device=cls_scores[0].device, dtype=torch.long)
            result_list.append(results)
        return result_list
