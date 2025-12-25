# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

from visdet.models.task_modules.assigners.assign_result import AssignResult
from visdet.models.task_modules.assigners.base_assigner import BaseAssigner
from visdet.models.task_modules.assigners.match_costs import bbox_xyxy_to_cxcywh
from visdet.registry import TASK_UTILS


@TASK_UTILS.register_module()
class UniformAssigner(BaseAssigner):
    """Uniform Matching between the anchors and gt boxes.

    This assignment strategy achieves balance in positive anchors.
    See `YOLOF <https://arxiv.org/abs/2103.09460>`_ for details.

    Args:
        pos_ignore_thr (float): the threshold to ignore positive anchors
        neg_ignore_thr (float): the threshold to ignore negative anchors
        match_times (int): Number of positive anchors for each gt box.
           Default 4.
        iou_calculator (dict): iou_calculator config
    """

    def __init__(
        self,
        pos_ignore_thr: float,
        neg_ignore_thr: float,
        match_times: int = 4,
        iou_calculator: dict = dict(type="BboxOverlaps2D"),
    ) -> None:
        self.match_times = match_times
        self.pos_ignore_thr = pos_ignore_thr
        self.neg_ignore_thr = neg_ignore_thr
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def assign(
        self,
        bbox_pred: Tensor,
        anchor: Tensor,
        gt_bboxes: Tensor,
        gt_bboxes_ignore: Tensor | None = None,
        gt_labels: Tensor | None = None,
    ) -> AssignResult:
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes,), 0, dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            assign_result = AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
            return assign_result

        # 2. Compute the L1 cost between boxes
        # Note that we use anchors and predict boxes both
        cost_bbox = torch.cdist(bbox_xyxy_to_cxcywh(bbox_pred), bbox_xyxy_to_cxcywh(gt_bboxes), p=1)
        cost_bbox_anchors = torch.cdist(bbox_xyxy_to_cxcywh(anchor), bbox_xyxy_to_cxcywh(gt_bboxes), p=1)

        # self.match_times x n
        index = torch.topk(cost_bbox, k=self.match_times, dim=0, largest=False)[1]
        index1 = torch.topk(cost_bbox_anchors, k=self.match_times, dim=0, largest=False)[1]

        # (self.match_times*2) x n
        indexes = torch.cat((index, index1), dim=1).reshape(-1)

        pred_overlaps = self.iou_calculator(bbox_pred, gt_bboxes)
        anchor_overlaps = self.iou_calculator(anchor, gt_bboxes)
        pred_max_overlaps, _ = pred_overlaps.max(dim=1)
        anchor_max_overlaps, _ = anchor_overlaps.max(dim=0)

        # 3. Compute the ignore indexes use gt_bboxes and predict boxes
        ignore_idx = pred_max_overlaps > self.neg_ignore_thr
        assigned_gt_inds[ignore_idx] = -1

        # 4. Compute the ignore indexes of positive sample use anchors
        # and predict boxes
        pos_gt_index = torch.arange(0, cost_bbox.size(1), device=bbox_pred.device).repeat(self.match_times * 2)
        pos_ious = anchor_overlaps[indexes, pos_gt_index]
        pos_ignore_idx = pos_ious < self.pos_ignore_thr

        pos_gt_index_with_ignore = pos_gt_index + 1
        pos_gt_index_with_ignore[pos_ignore_idx] = -1
        assigned_gt_inds[indexes] = pos_gt_index_with_ignore

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        assign_result = AssignResult(num_gts, assigned_gt_inds, anchor_max_overlaps, labels=assigned_labels)
        return assign_result
