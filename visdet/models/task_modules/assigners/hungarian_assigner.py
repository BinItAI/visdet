# Copyright (c) OpenMMLab. All rights reserved.
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from visdet.models.task_modules.assigners.assign_result import AssignResult
from visdet.models.task_modules.assigners.base_assigner import BaseAssigner
from visdet.registry import TASK_UTILS


@TASK_UTILS.register_module()
class HungarianAssigner(BaseAssigner):
    """Hungarian Assigner for DETR-style detectors.

    Uses the Hungarian algorithm (also known as the Kuhn-Munkres algorithm)
    to find an optimal bipartite matching between predictions and ground truth.

    Paper: https://arxiv.org/abs/2005.12872

    Args:
        cls_cost (dict): Config for classification cost. Should be a dict
            with type and weight keys. Defaults to FocalLossCost with weight=2.
        reg_cost (dict): Config for regression cost. Defaults to BBoxL1Cost
            with weight=5.
        iou_cost (dict): Config for IoU cost. Defaults to IoUCost with
            iou_mode='giou' and weight=2.
    """

    def __init__(
        self,
        cls_cost: dict | None = None,
        reg_cost: dict | None = None,
        iou_cost: dict | None = None,
    ) -> None:
        # Default costs if not provided
        self.cls_cost_cfg = cls_cost or dict(type="FocalLossCost", weight=2.0)
        self.reg_cost_cfg = reg_cost or dict(type="BBoxL1Cost", weight=5.0)
        self.iou_cost_cfg = iou_cost or dict(type="IoUCost", iou_mode="giou", weight=2.0)

        # Build cost functions
        self.cls_cost = TASK_UTILS.build(self.cls_cost_cfg)
        self.reg_cost = TASK_UTILS.build(self.reg_cost_cfg)
        self.iou_cost = TASK_UTILS.build(self.iou_cost_cfg)

    def assign(
        self,
        pred_instances,
        gt_instances,
        img_meta: dict | None = None,
        gt_instances_ignore=None,
        **kwargs,
    ) -> AssignResult:
        """Assign gt to predictions using Hungarian matching.

        Args:
            pred_instances: Predictions containing 'scores' and 'bboxes'.
            gt_instances: Ground truth containing 'bboxes' and 'labels'.
            img_meta (dict, optional): Image meta info.
            gt_instances_ignore: Ignored ground truth (unused).

        Returns:
            AssignResult: Assignment result.
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_gt = gt_bboxes.size(0)

        pred_scores = pred_instances.scores
        pred_bboxes = pred_instances.bboxes
        num_preds = pred_bboxes.size(0)

        # Initialize assignment
        assigned_gt_inds = pred_bboxes.new_full((num_preds,), 0, dtype=torch.long)
        assigned_labels = pred_bboxes.new_full((num_preds,), -1, dtype=torch.long)

        if num_gt == 0 or num_preds == 0:
            if num_gt == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gt,
                gt_inds=assigned_gt_inds,
                max_overlaps=pred_bboxes.new_zeros(num_preds),
                labels=assigned_labels,
            )

        # Compute costs
        # Classification cost: (num_preds, num_gt)
        cls_cost = self.cls_cost(pred_scores, gt_labels)

        # Normalize bboxes for L1 cost
        img_h, img_w = 1, 1
        if img_meta is not None:
            img_h = img_meta.get("img_shape", (1, 1))[0]
            img_w = img_meta.get("img_shape", (1, 1))[1]

        factor = pred_bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        normalized_gt_bboxes = gt_bboxes / factor
        normalized_pred_bboxes = pred_bboxes / factor

        # Regression cost: (num_preds, num_gt)
        reg_cost = self.reg_cost(normalized_pred_bboxes, normalized_gt_bboxes)

        # IoU cost: (num_preds, num_gt)
        iou_cost = self.iou_cost(pred_bboxes, gt_bboxes)

        # Total cost: (num_preds, num_gt)
        cost = cls_cost + reg_cost + iou_cost

        # Hungarian matching
        cost_np = cost.detach().cpu().numpy()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost_np)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(pred_bboxes.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(pred_bboxes.device)

        # Assign matched predictions
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        return AssignResult(
            num_gts=num_gt,
            gt_inds=assigned_gt_inds,
            max_overlaps=pred_bboxes.new_zeros(num_preds),
            labels=assigned_labels,
        )


# Cost functions for HungarianAssigner


@TASK_UTILS.register_module()
class FocalLossCost:
    """Focal loss cost for classification.

    Args:
        weight (float): Weight of the cost. Defaults to 1.0.
        alpha (float): Focal loss alpha. Defaults to 0.25.
        gamma (float): Focal loss gamma. Defaults to 2.0.
        eps (float): Epsilon for numerical stability. Defaults to 1e-12.
    """

    def __init__(
        self,
        weight: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        eps: float = 1e-12,
    ) -> None:
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def __call__(self, cls_pred: Tensor, gt_labels: Tensor) -> Tensor:
        """Compute focal loss cost.

        Args:
            cls_pred (Tensor): Classification scores with shape (num_preds, num_classes).
            gt_labels (Tensor): GT labels with shape (num_gt,).

        Returns:
            Tensor: Cost matrix with shape (num_preds, num_gt).
        """
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (1 - cls_pred).pow(self.gamma)

        # Gather costs for gt labels
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight


@TASK_UTILS.register_module()
class ClassificationCost:
    """Classification cost (cross entropy).

    Args:
        weight (float): Weight of the cost. Defaults to 1.0.
    """

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    def __call__(self, cls_pred: Tensor, gt_labels: Tensor) -> Tensor:
        """Compute classification cost.

        Args:
            cls_pred (Tensor): Classification scores with shape (num_preds, num_classes).
            gt_labels (Tensor): GT labels with shape (num_gt,).

        Returns:
            Tensor: Cost matrix with shape (num_preds, num_gt).
        """
        cls_pred = cls_pred.softmax(-1)
        cls_cost = -cls_pred[:, gt_labels]
        return cls_cost * self.weight


@TASK_UTILS.register_module()
class BBoxL1Cost:
    """L1 cost for bounding box regression.

    Args:
        weight (float): Weight of the cost. Defaults to 1.0.
        box_format (str): Box format. 'xyxy' or 'xywh'. Defaults to 'xyxy'.
    """

    def __init__(
        self,
        weight: float = 1.0,
        box_format: str = "xyxy",
    ) -> None:
        self.weight = weight
        self.box_format = box_format

    def __call__(self, bbox_pred: Tensor, gt_bboxes: Tensor) -> Tensor:
        """Compute L1 cost.

        Args:
            bbox_pred (Tensor): Predicted bboxes with shape (num_preds, 4).
            gt_bboxes (Tensor): GT bboxes with shape (num_gt, 4).

        Returns:
            Tensor: Cost matrix with shape (num_preds, num_gt).
        """
        reg_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return reg_cost * self.weight


@TASK_UTILS.register_module()
class IoUCost:
    """IoU cost for bounding boxes.

    Args:
        iou_mode (str): IoU type. 'iou', 'giou', 'diou', 'ciou'. Defaults to 'giou'.
        weight (float): Weight of the cost. Defaults to 1.0.
    """

    def __init__(
        self,
        iou_mode: str = "giou",
        weight: float = 1.0,
    ) -> None:
        self.iou_mode = iou_mode
        self.weight = weight

    def __call__(self, bboxes: Tensor, gt_bboxes: Tensor) -> Tensor:
        """Compute IoU cost.

        Args:
            bboxes (Tensor): Predicted bboxes with shape (num_preds, 4).
            gt_bboxes (Tensor): GT bboxes with shape (num_gt, 4).

        Returns:
            Tensor: Cost matrix with shape (num_preds, num_gt).
        """
        # Compute pairwise IoU
        overlaps = self._bbox_overlaps(bboxes, gt_bboxes, mode=self.iou_mode)
        # Cost is negative IoU (lower is better)
        iou_cost = -overlaps
        return iou_cost * self.weight

    def _bbox_overlaps(
        self,
        bboxes1: Tensor,
        bboxes2: Tensor,
        mode: str = "iou",
        eps: float = 1e-6,
    ) -> Tensor:
        """Compute bbox overlaps.

        Args:
            bboxes1 (Tensor): Shape (N, 4).
            bboxes2 (Tensor): Shape (M, 4).
            mode (str): 'iou' or 'giou'.
            eps (float): Epsilon for numerical stability.

        Returns:
            Tensor: Overlap matrix with shape (N, M).
        """
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

        lt = torch.max(bboxes1[:, None, :2], bboxes2[None, :, :2])
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        union = area1[:, None] + area2[None, :] - inter
        iou = inter / (union + eps)

        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[:, None, :2], bboxes2[None, :, :2])
            enclosed_rb = torch.max(bboxes1[:, None, 2:], bboxes2[None, :, 2:])
            enclosed_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
            enclosed_area = enclosed_wh[:, :, 0] * enclosed_wh[:, :, 1]
            giou = iou - (enclosed_area - union) / (enclosed_area + eps)
            return giou

        return iou
