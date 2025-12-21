# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from torch import Tensor

from visdet.models.task_modules.assigners.assign_result import AssignResult
from visdet.models.task_modules.assigners.base_assigner import BaseAssigner
from visdet.registry import TASK_UTILS


@TASK_UTILS.register_module()
class SimOTAAssigner(BaseAssigner):
    """SimOTA (Simplified Optimal Transport Assignment) Assigner.

    SimOTA is used in YOLOX for dynamic label assignment. It uses a
    simplified version of Optimal Transport to assign labels.

    Paper: https://arxiv.org/abs/2107.08430

    Args:
        center_radius (float): Radius of center region for selecting
            candidate priors. Defaults to 2.5.
        candidate_topk (int): Top-k candidates to select. Defaults to 10.
        iou_weight (float): Weight of IoU cost. Defaults to 3.0.
        cls_weight (float): Weight of classification cost. Defaults to 1.0.
    """

    def __init__(
        self,
        center_radius: float = 2.5,
        candidate_topk: int = 10,
        iou_weight: float = 3.0,
        cls_weight: float = 1.0,
    ) -> None:
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight

    def assign(
        self,
        pred_instances,
        gt_instances,
        gt_instances_ignore=None,
        **kwargs,
    ) -> AssignResult:
        """Assign gt to priors using SimOTA.

        Args:
            pred_instances: Predictions containing 'priors', 'scores', 'bboxes'.
            gt_instances: Ground truth containing 'bboxes' and 'labels'.
            gt_instances_ignore: Ignored ground truth (unused).

        Returns:
            AssignResult: Assignment result.
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_gt = gt_bboxes.size(0)

        priors = pred_instances.priors
        num_priors = priors.size(0)

        # Initialize assignment
        assigned_gt_inds = priors.new_full((num_priors,), 0, dtype=torch.long)
        assigned_labels = priors.new_full((num_priors,), -1, dtype=torch.long)

        if num_gt == 0 or num_priors == 0:
            return AssignResult(
                num_gts=num_gt,
                gt_inds=assigned_gt_inds,
                max_overlaps=priors.new_zeros(num_priors),
                labels=assigned_labels,
            )

        decoded_bboxes = pred_instances.bboxes
        pred_scores = pred_instances.scores

        # Get center points of priors
        prior_cx = priors[:, 0]
        prior_cy = priors[:, 1]
        if priors.size(-1) >= 3:
            strides = priors[:, 2]
        else:
            strides = prior_cx.new_ones(prior_cx.size())

        # Get gt centers
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0

        # Find valid priors (inside gt boxes or within center region)
        # Check if prior center is inside gt box
        l_ = prior_cx[None, :] - gt_bboxes[:, 0:1]
        t_ = prior_cy[None, :] - gt_bboxes[:, 1:2]
        r_ = gt_bboxes[:, 2:3] - prior_cx[None, :]
        b_ = gt_bboxes[:, 3:4] - prior_cy[None, :]
        is_in_boxes = (l_ > 0) & (t_ > 0) & (r_ > 0) & (b_ > 0)
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        # Check if prior center is within center region
        center_dist_x = torch.abs(prior_cx[None, :] - gt_cx[:, None])
        center_dist_y = torch.abs(prior_cy[None, :] - gt_cy[:, None])
        center_radius = self.center_radius * strides[None, :]
        is_in_centers = (center_dist_x < center_radius) & (center_dist_y < center_radius)
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # Combine: prior is valid if inside any gt box or center region
        is_pos = is_in_boxes_all | is_in_centers_all

        valid_mask = is_in_boxes | is_in_centers

        if not is_pos.any():
            return AssignResult(
                num_gts=num_gt,
                gt_inds=assigned_gt_inds,
                max_overlaps=priors.new_zeros(num_priors),
                labels=assigned_labels,
            )

        # Compute costs for valid priors
        valid_prior_idxs = torch.nonzero(is_pos, as_tuple=False).squeeze(-1)
        valid_decoded_bbox = decoded_bboxes[valid_prior_idxs]
        valid_pred_scores = pred_scores[valid_prior_idxs]
        num_valid = valid_prior_idxs.size(0)

        # IoU cost
        pairwise_ious = self._compute_iou(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + 1e-8)

        # Classification cost
        gt_onehot_label = F.one_hot(gt_labels.long(), valid_pred_scores.size(-1)).float()
        valid_pred_scores = valid_pred_scores.unsqueeze(0).repeat(num_gt, 1, 1)
        soft_label = gt_onehot_label.unsqueeze(1).repeat(1, num_valid, 1)

        # Modulate by IoU
        soft_label = soft_label * pairwise_ious.unsqueeze(-1)
        scale_factor = soft_label - valid_pred_scores.sigmoid()
        cls_cost = F.binary_cross_entropy_with_logits(valid_pred_scores, soft_label, reduction="none").sum(
            dim=-1
        ) * scale_factor.abs().pow(2).sum(dim=-1)

        # Total cost: (num_gt, num_valid)
        cost = cls_cost * self.cls_weight + iou_cost * self.iou_weight

        # Filter by valid mask
        cost = cost + (~valid_mask[:, valid_prior_idxs]).float() * 1e8

        # Dynamic k selection based on IoU
        matched_gt_inds, matched_pred_inds = self._dynamic_k_matching(cost, pairwise_ious, num_gt)

        # Assign
        assigned_gt_inds[valid_prior_idxs[matched_pred_inds]] = matched_gt_inds + 1
        assigned_labels[valid_prior_idxs[matched_pred_inds]] = gt_labels[matched_gt_inds]

        # Get max overlaps
        max_overlaps = priors.new_zeros(num_priors)
        max_overlaps[valid_prior_idxs] = pairwise_ious.max(dim=0)[0]

        return AssignResult(
            num_gts=num_gt,
            gt_inds=assigned_gt_inds,
            max_overlaps=max_overlaps,
            labels=assigned_labels,
        )

    def _compute_iou(self, bboxes1: Tensor, bboxes2: Tensor) -> Tensor:
        """Compute IoU between two sets of boxes.

        Args:
            bboxes1 (Tensor): Shape (N, 4).
            bboxes2 (Tensor): Shape (M, 4).

        Returns:
            Tensor: IoU matrix with shape (M, N).
        """
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

        lt = torch.max(bboxes1[:, None, :2], bboxes2[None, :, :2])
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        union = area1[:, None] + area2[None, :] - inter
        iou = inter / (union + 1e-8)
        return iou.T  # (M, N)

    def _dynamic_k_matching(
        self,
        cost: Tensor,
        ious: Tensor,
        num_gt: int,
    ) -> tuple[Tensor, Tensor]:
        """Dynamic k-matching based on IoU.

        Args:
            cost (Tensor): Cost matrix with shape (num_gt, num_valid).
            ious (Tensor): IoU matrix with shape (num_gt, num_valid).
            num_gt (int): Number of ground truth boxes.

        Returns:
            tuple[Tensor, Tensor]: Matched gt indices and pred indices.
        """
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        # Calculate dynamic k per gt
        n_candidate_k = min(self.candidate_topk, ious.size(1))
        topk_ious, _ = torch.topk(ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(dim=1).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[gt_idx, pos_idx] = 1

        # Remove duplicates (assign each pred to only one gt)
        anchor_matching_gt = matching_matrix.sum(dim=0)
        if (anchor_matching_gt > 1).any():
            _, cost_min_idx = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] = 0
            matching_matrix[cost_min_idx, anchor_matching_gt > 1] = 1

        # Get matched indices
        matched_gt_inds, matched_pred_inds = torch.nonzero(matching_matrix, as_tuple=True)

        return matched_gt_inds, matched_pred_inds
