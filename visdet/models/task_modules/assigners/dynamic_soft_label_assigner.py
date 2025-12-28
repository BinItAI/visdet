# Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

from typing import Any, Optional, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from visdet.engine.structures import InstanceData
from visdet.models.task_modules.assigners.assign_result import AssignResult
from visdet.models.task_modules.assigners.base_assigner import BaseAssigner
from visdet.registry import TASK_UTILS
from visdet.structures.bbox import BaseBoxes

INF = 100000000
EPS = 1.0e-7


def center_of_mass(masks: Tensor, eps: float = 1e-7) -> Tensor:
    """Compute center of mass for binary masks."""

    n, h, w = masks.shape
    grid_h = torch.arange(h, device=masks.device)[:, None]
    grid_w = torch.arange(w, device=masks.device)
    normalizer = masks.sum(dim=(1, 2)).float().clamp(min=eps)
    center_y = (masks * grid_h).sum(dim=(1, 2)) / normalizer
    center_x = (masks * grid_w).sum(dim=(1, 2)) / normalizer
    return torch.cat([center_x[:, None], center_y[:, None]], dim=1)


@TASK_UTILS.register_module()
class DynamicSoftLabelAssigner(BaseAssigner):
    """Dynamic soft label assigner used by RTMDet.

    Produces 1-based ``gt_inds``.
    """

    def __init__(
        self,
        soft_center_radius: float = 3.0,
        topk: int = 13,
        iou_weight: float = 3.0,
        iou_calculator: dict = dict(type="BboxOverlaps2D"),
    ) -> None:
        self.soft_center_radius = soft_center_radius
        self.topk = topk
        self.iou_weight = iou_weight
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def assign(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        gt_instances_ignore: Optional[InstanceData] = None,
        **kwargs,
    ) -> AssignResult:
        gt_bboxes = cast(Tensor | BaseBoxes, getattr(gt_instances, "bboxes"))
        gt_labels = cast(Tensor, getattr(gt_instances, "labels"))
        if isinstance(gt_bboxes, BaseBoxes):
            num_gt = len(gt_bboxes)
        else:
            num_gt = gt_bboxes.size(0)
        num_gt = int(num_gt)

        decoded_bboxes = cast(Tensor, getattr(pred_instances, "bboxes"))
        pred_scores = cast(Tensor, getattr(pred_instances, "scores"))
        priors = cast(Tensor, getattr(pred_instances, "priors"))
        num_bboxes = int(decoded_bboxes.size(0))

        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        prior_center = priors[:, :2]
        if isinstance(gt_bboxes, BaseBoxes):
            is_in_gts = gt_bboxes.find_inside_points(prior_center)
        else:
            lt_ = prior_center[:, None] - gt_bboxes[:, :2]
            rb_ = gt_bboxes[:, 2:] - prior_center[:, None]
            deltas = torch.cat([lt_, rb_], dim=-1)
            is_in_gts = deltas.min(dim=-1).values > 0

        valid_mask = is_in_gts.sum(dim=1) > 0

        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = int(valid_decoded_bbox.size(0))

        if num_valid == 0:
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        if hasattr(gt_instances, "masks"):
            gt_center = center_of_mass(gt_instances.masks, eps=EPS)
        elif isinstance(gt_bboxes, BaseBoxes):
            gt_center = gt_bboxes.centers
        else:
            gt_center = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2.0

        valid_prior = priors[valid_mask]
        strides = valid_prior[:, 2]
        distance = (valid_prior[:, None, :2] - gt_center[None, :, :]).pow(2).sum(-1).sqrt() / strides[:, None]
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)

        pairwise_ious = self.iou_calculator(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + EPS) * self.iou_weight

        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1]).float().unsqueeze(0).repeat(num_valid, 1, 1)
        )
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)

        soft_label = gt_onehot_label * pairwise_ious[..., None]
        scale_factor = soft_label - valid_pred_scores.sigmoid()
        soft_cls_cost = F.binary_cross_entropy_with_logits(valid_pred_scores, soft_label, reduction="none")
        soft_cls_cost = soft_cls_cost * scale_factor.abs().pow(2.0)
        soft_cls_cost = soft_cls_cost.sum(dim=-1)

        cost_matrix = soft_cls_cost + iou_cost + soft_center_prior

        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(cost_matrix, pairwise_ious, num_gt, valid_mask)

        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()

        max_overlaps = assigned_gt_inds.new_full((num_bboxes,), -INF, dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious

        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def dynamic_k_matching(
        self,
        cost: Tensor,
        pairwise_ious: Tensor,
        num_gt: int,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        candidate_topk = min(self.topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=int(dynamic_ks[gt_idx].item()), largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            _, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1

        fg_mask_inboxes = matching_matrix.sum(1) > 0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds
