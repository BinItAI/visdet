# Copyright (c) OpenMMLab. All rights reserved.
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from visdet.registry import TASK_UTILS
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from .match_costs import bbox_cxcywh_to_xyxy


@TASK_UTILS.register_module()
class HungarianAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_cost (dict): The classification cost config.
        reg_cost (dict): The regression L1 cost config.
        iou_cost (dict): The regression iou cost config.
    """

    def __init__(
        self,
        cls_cost: dict = dict(type="ClassificationCost", weight=1.0),
        reg_cost: dict = dict(type="BBoxL1Cost", weight=1.0),
        iou_cost: dict = dict(type="IoUCost", iou_mode="giou", weight=1.0),
    ) -> None:
        self.cls_cost = TASK_UTILS.build(cls_cost)
        self.reg_cost = TASK_UTILS.build(reg_cost)
        self.iou_cost = TASK_UTILS.build(iou_cost)

    def assign(
        self,
        bbox_pred: Tensor,
        cls_pred: Tensor,
        gt_bboxes: Tensor,
        gt_labels: Tensor,
        img_meta: dict,
        gt_bboxes_ignore: Tensor | None = None,
        eps: float = 1e-7,
    ) -> AssignResult:
        """Computes one-to-one matching based on the weighted costs."""
        assert gt_bboxes_ignore is None, "Only case when gt_bboxes_ignore is None is supported."
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

        img_h, img_w = img_meta["img_shape"][:2]
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        normalize_gt_bboxes = gt_bboxes / factor
        reg_cost = self.reg_cost(bbox_pred, normalize_gt_bboxes)
        # regression iou cost, defaultly giou is used in official DETR.
        bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
        iou_cost = self.iou_cost(bboxes, gt_bboxes)
        # weighted sum of above three costs
        cost = cls_cost + reg_cost + iou_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
