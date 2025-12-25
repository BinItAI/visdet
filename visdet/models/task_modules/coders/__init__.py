# ruff: noqa
import torch
from visdet.registry import TASK_UTILS
from visdet.structures.bbox import distance2bbox, bbox2distance


@TASK_UTILS.register_module()
class DeltaXYWHBBoxCoder:
    """Delta XYWH BBox coder."""

    def __init__(
        self,
        target_means=(0.0, 0.0, 0.0, 0.0),
        target_stds=(1.0, 1.0, 1.0, 1.0),
        clip_border=True,
    ):
        self.means = target_means
        self.stds = target_stds
        self.clip_border = clip_border

    @property
    def encode_size(self):
        """int: the encoded size"""
        return 4

    def encode(self, bboxes, gt_bboxes):
        """Encode deltas between bboxes and ground truth boxes."""
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == 4
        assert gt_bboxes.size(-1) == 4

        px = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
        py = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
        pw = bboxes[..., 2] - bboxes[..., 0]
        ph = bboxes[..., 3] - bboxes[..., 1]

        gx = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) * 0.5
        gy = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) * 0.5
        gw = gt_bboxes[..., 2] - gt_bboxes[..., 0]
        gh = gt_bboxes[..., 3] - gt_bboxes[..., 1]

        dx = (gx - px) / pw
        dy = (gy - py) / ph
        dw = torch.log(gw / pw)
        dh = torch.log(gh / ph)

        deltas = torch.stack([dx, dy, dw, dh], dim=-1)

        # Normalize by mean and std
        means = deltas.new_tensor(self.means).view(1, -1)
        stds = deltas.new_tensor(self.stds).view(1, -1)
        deltas = (deltas - means) / stds

        return deltas

    def decode(self, bboxes, pred_bboxes, max_shape=None):
        """Decode deltas to get predicted boxes."""
        assert pred_bboxes.size(-1) == 4

        # Denormalize
        means = pred_bboxes.new_tensor(self.means).view(1, -1)
        stds = pred_bboxes.new_tensor(self.stds).view(1, -1)
        pred_bboxes = pred_bboxes * stds + means

        px = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
        py = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
        pw = bboxes[..., 2] - bboxes[..., 0]
        ph = bboxes[..., 3] - bboxes[..., 1]

        gx = px + pred_bboxes[..., 0] * pw
        gy = py + pred_bboxes[..., 1] * ph
        gw = pw * pred_bboxes[..., 2].exp()
        gh = ph * pred_bboxes[..., 3].exp()

        x1 = gx - gw * 0.5
        y1 = gy - gh * 0.5
        x2 = gx + gw * 0.5
        y2 = gy + gh * 0.5

        decoded_bboxes = torch.stack([x1, y1, x2, y2], dim=-1)

        # Clip bboxes to image boundaries if max_shape is provided
        if max_shape is not None:
            decoded_bboxes[..., 0] = decoded_bboxes[..., 0].clamp(min=0, max=max_shape[1])
            decoded_bboxes[..., 1] = decoded_bboxes[..., 1].clamp(min=0, max=max_shape[0])
            decoded_bboxes[..., 2] = decoded_bboxes[..., 2].clamp(min=0, max=max_shape[1])
            decoded_bboxes[..., 3] = decoded_bboxes[..., 3].clamp(min=0, max=max_shape[0])

        return decoded_bboxes


@TASK_UTILS.register_module()
class DistancePointBBoxCoder:
    """Distance Point BBox coder.

    This coder encodes gt bboxes (x1, y1, x2, y2) into (top, bottom, left,
    right) and decode it back to the original.

    Args:
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
    """

    def __init__(self, clip_border=True):
        self.clip_border = clip_border

    @property
    def encode_size(self):
        """int: the encoded size"""
        return 4

    def encode(self, points, gt_bboxes, max_dis=None, eps=0.1):
        """Encode bounding box to distances."""
        assert points.size(0) == gt_bboxes.size(0)
        assert points.size(-1) == 2
        assert gt_bboxes.size(-1) == 4
        return bbox2distance(points, gt_bboxes, max_dis, eps)

    def decode(self, points, pred_bboxes, max_shape=None):
        """Decode distance prediction to bounding box."""
        assert points.size(0) == pred_bboxes.size(0)
        assert points.size(-1) == 2
        assert pred_bboxes.size(-1) == 4
        if self.clip_border is False:
            max_shape = None
        return distance2bbox(points, pred_bboxes, max_shape)


from visdet.models.task_modules.coders.yolo_coder import YOLOBBoxCoder
from visdet.models.task_modules.coders.bucketing_bbox_coder import BucketingBBoxCoder

__all__ = [
    "DeltaXYWHBBoxCoder",
    "DistancePointBBoxCoder",
    "YOLOBBoxCoder",
    "BucketingBBoxCoder",
]
