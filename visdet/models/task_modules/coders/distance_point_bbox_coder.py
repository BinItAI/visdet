# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

from visdet.registry import TASK_UTILS


@TASK_UTILS.register_module()
class DistancePointBBoxCoder:
    """Distance Point BBox Coder.

    This coder encodes bounding boxes as distances from a center point.
    Used by anchor-free detectors like FCOS.

    The format is (left, top, right, bottom) distances from the center point
    to the four edges of the bounding box.

    Args:
        clip_border (bool): Whether to clip the coordinates inside the
            border of the image. Defaults to True.
    """

    def __init__(self, clip_border: bool = True) -> None:
        self.clip_border = clip_border

    @property
    def encode_size(self) -> int:
        """Return the encoded size (4 for ltrb format)."""
        return 4

    def encode(
        self,
        points: Tensor,
        gt_bboxes: Tensor,
        max_dis: float | None = None,
        eps: float = 0.1,
    ) -> Tensor:
        """Encode bounding boxes to distances.

        Args:
            points (Tensor): Points with shape (N, 2), representing (x, y).
            gt_bboxes (Tensor): Ground truth boxes with shape (N, 4),
                in (x1, y1, x2, y2) format.
            max_dis (float, optional): Maximum distance for normalization.
                If provided, distances are normalized by this value.
            eps (float): Small value to avoid division by zero.

        Returns:
            Tensor: Encoded distances (left, top, right, bottom) with shape (N, 4).
        """
        assert points.size(0) == gt_bboxes.size(0)
        assert points.size(-1) == 2
        assert gt_bboxes.size(-1) == 4

        left = points[..., 0] - gt_bboxes[..., 0]
        top = points[..., 1] - gt_bboxes[..., 1]
        right = gt_bboxes[..., 2] - points[..., 0]
        bottom = gt_bboxes[..., 3] - points[..., 1]

        if max_dis is not None:
            left = left.clamp(min=0, max=max_dis - eps)
            top = top.clamp(min=0, max=max_dis - eps)
            right = right.clamp(min=0, max=max_dis - eps)
            bottom = bottom.clamp(min=0, max=max_dis - eps)

        return torch.stack([left, top, right, bottom], dim=-1)

    def decode(
        self,
        points: Tensor,
        pred_bboxes: Tensor,
        max_shape: tuple[int, int] | None = None,
    ) -> Tensor:
        """Decode distances to bounding boxes.

        Args:
            points (Tensor): Points with shape (N, 2), representing (x, y).
            pred_bboxes (Tensor): Predicted distances with shape (N, 4),
                in (left, top, right, bottom) format.
            max_shape (tuple, optional): (height, width) of the image for
                clipping. Defaults to None.

        Returns:
            Tensor: Decoded bboxes (x1, y1, x2, y2) with shape (N, 4).
        """
        assert points.size(-1) == 2
        assert pred_bboxes.size(-1) == 4

        x1 = points[..., 0] - pred_bboxes[..., 0]
        y1 = points[..., 1] - pred_bboxes[..., 1]
        x2 = points[..., 0] + pred_bboxes[..., 2]
        y2 = points[..., 1] + pred_bboxes[..., 3]

        bboxes = torch.stack([x1, y1, x2, y2], dim=-1)

        if max_shape is not None and self.clip_border:
            bboxes[..., 0] = bboxes[..., 0].clamp(min=0, max=max_shape[1])
            bboxes[..., 1] = bboxes[..., 1].clamp(min=0, max=max_shape[0])
            bboxes[..., 2] = bboxes[..., 2].clamp(min=0, max=max_shape[1])
            bboxes[..., 3] = bboxes[..., 3].clamp(min=0, max=max_shape[0])

        return bboxes
