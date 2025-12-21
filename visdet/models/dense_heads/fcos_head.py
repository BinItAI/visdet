# Copyright (c) OpenMMLab. All rights reserved.
"""FCOS dense head implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from visdet.cv.cnn.bricks.scale import Scale
from visdet.engine.dist import reduce_mean
from visdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from visdet.models.utils import multi_apply
from visdet.registry import MODELS
from visdet.structures.bbox import get_box_tensor
from visdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig

INF = 1e8


@MODELS.register_module()
class FCOSHead(AnchorFreeHead):
    """Fully Convolutional One-Stage (FCOS) detection head.

    Args:
        num_classes (int): Number of classes without background.
        in_channels (int): Number of input channels per feature level.
        regress_ranges (tuple[tuple[int, int]]): Regression range for each FPN
            level.
        center_sampling (bool): Whether to use center sampling.
        center_sample_radius (float): Sampling radius multiplier.
        norm_on_bbox (bool): Normalize regression targets by stride.
        centerness_on_reg (bool): Place centerness branch on regression tower.
        loss_centerness (dict): Config for centerness loss.
        norm_cfg (dict): Normalization config for conv towers.
        init_cfg (dict | list[dict], optional): Initialization config.
        **kwargs: Extra keyword args for :class:`AnchorFreeHead`.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        regress_ranges: tuple[tuple[int, int], ...] = (
            (-1, 64),
            (64, 128),
            (128, 256),
            (256, 512),
            (512, INF),
        ),
        center_sampling: bool = False,
        center_sample_radius: float = 1.5,
        norm_on_bbox: bool = False,
        centerness_on_reg: bool = False,
        loss_centerness: ConfigType = dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        norm_cfg: ConfigType = dict(type="GN", num_groups=32, requires_grad=True),
        init_cfg: OptMultiConfig = dict(
            type="Normal",
            layer="Conv2d",
            std=0.01,
            override=dict(type="Normal", name="conv_cls", std=0.01, bias_prob=0.01),
        ),
        **kwargs,
    ) -> None:
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs,
        )
        self.loss_centerness = MODELS.build(loss_centerness)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def _init_layers(self) -> None:
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

    def forward(self, feats: tuple[Tensor, ...]) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """Forward pass."""
        return multi_apply(self.forward_single, feats, self.scales, self.strides)

    def forward_single(self, x: Tensor, scale: Scale, stride: int) -> tuple[Tensor, Tensor, Tensor]:
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        centerness_feat = reg_feat if self.centerness_on_reg else cls_feat
        centerness = self.conv_centerness(centerness_feat)

        bbox_pred = scale(bbox_pred).float()
        stride_value = stride if isinstance(stride, (int, float)) else stride[0]
        if self.norm_on_bbox:
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride_value
        else:
            bbox_pred = bbox_pred.exp()
        return cls_score, bbox_pred, centerness

    def loss_by_feat(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        centernesses: list[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: list[dict],
        batch_gt_instances_ignore=None,
    ) -> dict:
        """Compute FCOS losses."""
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        device = cls_scores[0].device
        featmap_sizes = [feat.shape[-2:] for feat in cls_scores]
        dtype = cls_scores[0].dtype
        points = self.prior_generator.grid_priors(featmap_sizes, dtype=dtype, device=device)

        labels, bbox_targets = self.get_targets(points, batch_gt_instances)

        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = torch.cat(
            [score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for score in cls_scores],
            dim=0,
        )
        flatten_bbox_preds = torch.cat([bbox.permute(0, 2, 3, 1).reshape(-1, 4) for bbox in bbox_preds], dim=0)
        flatten_centerness = torch.cat(
            [centerness.permute(0, 2, 3, 1).reshape(-1) for centerness in centernesses],
            dim=0,
        )
        flatten_labels = torch.cat(labels).long()
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_points = torch.cat([level_points.repeat(num_imgs, 1) for level_points in points], dim=0)

        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).nonzero(as_tuple=False).reshape(-1)
        num_pos = flatten_bbox_preds.new_tensor([len(pos_inds)], dtype=torch.float)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        if len(pos_inds) > 0:
            pos_bbox_preds = flatten_bbox_preds[pos_inds]
            pos_centerness = flatten_centerness[pos_inds]
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_points = flatten_points[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            centerness_denorm = max(reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

            pos_decoded_bbox_preds = self.bbox_coder.decode(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm,
            )
            loss_centerness = self.loss_centerness(
                pos_centerness,
                pos_centerness_targets,
                avg_factor=num_pos,
            )
        else:
            loss_bbox = flatten_bbox_preds.sum() * 0
            loss_centerness = flatten_centerness.sum() * 0

        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_centerness=loss_centerness)

    def get_targets(
        self,
        points: list[Tensor],
        batch_gt_instances: InstanceList,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Assign ground truth targets to each point."""
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(points[i]) for i in range(num_levels)
        ]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        num_points = [level_points.size(0) for level_points in points]

        gt_bboxes_list = []
        gt_labels_list = []
        for gt_instances in batch_gt_instances:
            if getattr(gt_instances, "bboxes", None) is None or len(gt_instances.bboxes) == 0:
                gt_bboxes = concat_points.new_zeros((0, 4))
            else:
                gt_bboxes = get_box_tensor(gt_instances.bboxes)
            if getattr(gt_instances, "labels", None) is None or len(gt_instances.labels) == 0:
                gt_labels = gt_bboxes.new_zeros((0,), dtype=torch.long)
            else:
                gt_labels = gt_instances.labels
                if gt_labels.dtype != torch.long:
                    gt_labels = gt_labels.long()
            gt_bboxes_list.append(gt_bboxes)
            gt_labels_list.append(gt_labels)

        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points,
        )

        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [bbox_targets.split(num_points, 0) for bbox_targets in bbox_targets_list]

        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for level_idx in range(num_levels):
            concat_lvl_labels.append(torch.cat([labels[level_idx] for labels in labels_list]))
            bbox_targets = torch.cat([bbox_targets[level_idx] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                stride = self._stride_value(self.strides[level_idx])
                bbox_targets = bbox_targets / stride
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_single(
        self,
        gt_bboxes: Tensor,
        gt_labels: Tensor,
        points: Tensor,
        regress_ranges: Tensor,
        num_points_per_lvl: list[int],
    ) -> tuple[Tensor, Tensor]:
        """Assign targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_bboxes.size(0)
        if num_gts == 0:
            labels = gt_bboxes.new_full((num_points,), self.num_classes, dtype=torch.long)
            bbox_targets = gt_bboxes.new_zeros((num_points, 4))
            return labels, bbox_targets

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs = points[:, 0][:, None].expand(num_points, num_gts)
        ys = points[:, 1][:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), dim=-1)

        if self.center_sampling:
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride_tensor = center_xs.new_zeros(center_xs.shape)

            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride_tensor[lvl_begin:lvl_end] = self._stride_value(self.strides[lvl_idx]) * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride_tensor
            y_mins = center_ys - stride_tensor
            x_maxs = center_xs + stride_tensor
            y_maxs = center_ys + stride_tensor
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0], x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1], y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs < gt_bboxes[..., 2], x_maxs, gt_bboxes[..., 2])
            center_gts[..., 3] = torch.where(y_maxs < gt_bboxes[..., 3], y_maxs, gt_bboxes[..., 3])

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack((cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), dim=-1)
            inside_gt_bbox_mask = center_bbox.min(dim=-1)[0] > 0
        else:
            inside_gt_bbox_mask = bbox_targets.min(dim=-1)[0] > 0

        max_regress_distance = bbox_targets.max(dim=-1)[0]
        inside_regress_range = (max_regress_distance >= regress_ranges[..., 0]) & (
            max_regress_distance <= regress_ranges[..., 1]
        )

        areas[~inside_gt_bbox_mask] = INF
        areas[~inside_regress_range] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes
        bbox_targets = bbox_targets[torch.arange(num_points, device=points.device), min_area_inds]
        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets: Tensor) -> Tensor:
        """Compute centerness."""
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness = torch.sqrt(
            (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]).clamp(min=0, max=1)
            * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]).clamp(min=0, max=1)
        )
        return centerness

    def _stride_value(self, stride: int | tuple[int, int]) -> float:
        """Normalize stride representation to a scalar."""
        if isinstance(stride, (tuple, list)):
            if stride[0] != stride[1]:
                raise ValueError(f"FCOSHead expects square strides, but got {stride}.")
            return float(stride[0])
        return float(stride)
