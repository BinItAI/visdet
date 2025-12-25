# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from visdet.registry import TASK_UTILS
from visdet.structures.bbox import bbox_overlaps


class BaseMatchCost:
    """Base match cost class.

    Args:
        weight (float | int): The scale factor of match cost. Defaults to 1.
    """

    def __init__(self, weight: Union[float, int] = 1.0) -> None:
        self.weight = weight

    @abstractmethod
    def __call__(self, preds: Any, targets: Any) -> Any:
        pass


@TASK_UTILS.register_module()
class BBoxL1Cost(BaseMatchCost):
    """BBoxL1Cost.

    Args:
        weight (int | float, optional): loss_weight.
        box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse R-CNN.
            Defaults to 'xyxy'.
    """

    def __init__(self, weight: Union[float, int] = 1.0, box_format: str = "xyxy") -> None:
        super().__init__(weight=weight)
        self.box_format = box_format

    def __call__(self, bbox_pred: Tensor, gt_bboxes: Tensor) -> Tensor:
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                (num_query, 4).
            gt_bboxes (Tensor): Ground truth boxes with normalized coordinates
                (x1, y1, x2, y2). Shape (num_gt, 4).

        Returns:
            torch.Tensor: bbox_cost with shape (num_query, num_gt)
        """
        if self.box_format == "xywh":
            gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
        elif self.box_format == "xyxy":
            bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)

        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@TASK_UTILS.register_module()
class FocalLossCost(BaseMatchCost):
    """FocalLossCost.

    Args:
        weight (int | float, optional): loss_weight.
        alpha (int | float, optional): focal_loss alpha.
        gamma (int | float, optional): focal_loss gamma.
        eps (float, optional): default 1e-12.
        binary_input (bool, optional): Whether the input is binary.
            Defaults to False.
    """

    def __init__(
        self,
        weight: Union[float, int] = 1.0,
        alpha: float = 0.25,
        gamma: int = 2,
        eps: float = 1e-12,
        binary_input: bool = False,
    ) -> None:
        super().__init__(weight=weight)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.binary_input = binary_input

    def __call__(self, cls_pred: Tensor, gt_labels: Tensor) -> Tensor:
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_query, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost with shape (num_query, num_gt)
        """
        if self.binary_input:
            cls_pred = cls_pred.sigmoid()
        else:
            # cls_pred is logits
            cls_pred = cls_pred.sigmoid()

        neg_cost = -(1 - cls_pred + self.eps).log() * (1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (1 - cls_pred).pow(self.gamma)

        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight


@TASK_UTILS.register_module()
class ClassificationCost(BaseMatchCost):
    """ClassificationCost.

    Args:
        weight (int | float, optional): loss_weight.
    """

    def __init__(self, weight: Union[float, int] = 1.0) -> None:
        super().__init__(weight=weight)

    def __call__(self, cls_pred: Tensor, gt_labels: Tensor) -> Tensor:
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_query, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost with shape (num_query, num_gt)
        """
        # TODO: removing this after checking if it is safe
        # cls_pred = cls_pred.softmax(-1)
        cls_cost = -cls_pred[:, gt_labels]
        return cls_cost * self.weight


@TASK_UTILS.register_module()
class IoUCost(BaseMatchCost):
    """IoUCost.

    Args:
        iou_mode (str, optional): iou mode such as 'iou' | 'giou'
        weight (int | float, optional): loss_weight.
    """

    def __init__(self, iou_mode: str = "giou", weight: Union[float, int] = 1.0) -> None:
        super().__init__(weight=weight)
        self.iou_mode = iou_mode

    def __call__(self, bboxes: Tensor, gt_bboxes: Tensor) -> Tensor:
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape (num_query, 4).
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape (num_gt, 4).

        Returns:
            torch.Tensor: iou_cost with shape (num_query, num_gt)
        """
        # overlaps: [num_bboxes, num_gt]
        overlaps = bbox_overlaps(bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)
        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * self.weight


def bbox_cxcywh_to_xyxy(bbox: Tensor) -> Tensor:
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    new_bbox = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(new_bbox, dim=-1)


def bbox_xyxy_to_cxcywh(bbox: Tensor) -> Tensor:
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    return torch.cat(bbox_new, dim=-1)
