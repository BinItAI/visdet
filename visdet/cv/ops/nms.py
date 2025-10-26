# ruff: noqa
"""
Non-Maximum Suppression (NMS) operations.

This module provides NMS and related suppression algorithms.
"""

from typing import Dict, Tuple, Union
import torch
from torch import Tensor


def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    nms_cfg: Union[Dict, float],
) -> Tuple[Tensor, Tensor]:
    """Apply NMS per class to boxes.

    Args:
        boxes: Tensor of shape (N, 4) with box coordinates [x1, y1, x2, y2]
        scores: Tensor of shape (N,) with detection scores
        idxs: Tensor of shape (N,) with class indices
        nms_cfg: NMS configuration dict with 'iou_threshold' or a float iou_threshold

    Returns:
        Tuple of:
            - dets: Tensor of shape (K, 5) with [x1, y1, x2, y2, score]
            - keep: Tensor of shape (K,) with indices of kept boxes
    """
    # Extract iou_threshold from nms_cfg
    if isinstance(nms_cfg, dict):
        iou_threshold = nms_cfg.get("iou_threshold", 0.5)
    else:
        iou_threshold = nms_cfg

    # Get unique class indices
    unique_classes = torch.unique(idxs, sorted=False)
    keep_list = []

    for class_id in unique_classes:
        # Get mask for this class
        class_mask = idxs == class_id
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        class_inds = torch.where(class_mask)[0]

        # Apply NMS for this class
        keep_class = nms(class_boxes, class_scores, iou_threshold)
        keep_list.append(class_inds[keep_class])

    # Combine kept indices from all classes
    if len(keep_list) > 0:
        keep = torch.cat(keep_list)
        # Sort by original order
        keep = keep.sort()[0]
    else:
        keep = torch.empty(0, dtype=torch.long, device=boxes.device)

    # Create detections with scores
    if keep.numel() > 0:
        kept_boxes = boxes[keep]
        kept_scores = scores[keep]
        dets = torch.cat([kept_boxes, kept_scores.unsqueeze(1)], dim=1)
    else:
        dets = torch.empty((0, 5), dtype=boxes.dtype, device=boxes.device)

    return dets, keep


def nms(
    boxes: Tensor,
    scores: Tensor,
    iou_threshold: float = 0.5,
) -> Tensor:
    """Apply NMS to boxes.

    Args:
        boxes: Tensor of shape (N, 4) with box coordinates [x1, y1, x2, y2]
        scores: Tensor of shape (N,) with detection scores
        iou_threshold: IoU threshold for suppression

    Returns:
        Tensor of shape (K,) with indices of kept boxes
    """
    if boxes.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    # Sort by scores in descending order
    sorted_scores, sorted_inds = scores.sort(descending=True)
    sorted_boxes = boxes[sorted_inds]

    # Compute pairwise IoU
    ious = box_iou(sorted_boxes, sorted_boxes)

    # Suppress boxes
    keep = torch.ones(len(sorted_boxes), dtype=torch.bool, device=boxes.device)
    for i in range(len(sorted_boxes)):
        if not keep[i]:
            continue
        # Suppress all boxes with IoU > threshold
        keep[i + 1 :] &= ious[i, i + 1 :] <= iou_threshold

    return sorted_inds[keep]


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute pairwise IoU between two sets of boxes.

    Args:
        boxes1: Tensor of shape (N, 4) with box coordinates [x1, y1, x2, y2]
        boxes2: Tensor of shape (M, 4) with box coordinates [x1, y1, x2, y2]

    Returns:
        Tensor of shape (N, M) with pairwise IoU values
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, :2].unsqueeze(1), boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, 2:].unsqueeze(1), boxes2[:, 2:])  # [N, M, 2]
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter
    iou = inter / union

    return iou


__all__ = ["batched_nms", "nms", "box_iou"]
