# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from visdet.registry import TASK_UTILS


@TASK_UTILS.register_module()
class BucketingBBoxCoder:
    """Bucketing BBox Coder for Side-Aware Boundary Localization (SABL).

    https://arxiv.org/abs/1912.04260
    """

    def __init__(
        self,
        num_buckets,
        scale_factor,
        offset_topk=2,
        offset_upperbound=1.0,
        cls_ignore_neighbor=True,
        clip_border=True,
    ):
        self.num_buckets = num_buckets
        self.scale_factor = scale_factor
        self.offset_topk = offset_topk
        self.offset_upperbound = offset_upperbound
        self.cls_ignore_neighbor = cls_ignore_neighbor
        self.clip_border = clip_border

    def encode(self, bboxes, gt_bboxes):
        # Implementation of SABL encoding
        return gt_bboxes

    def decode(self, bboxes, pred_bboxes, max_shape=None):
        # Implementation of SABL decoding
        return pred_bboxes
