# Copyright (c) OpenMMLab. All rights reserved.
import torch

from visdet.registry import TASK_UTILS


@TASK_UTILS.register_module()
class YOLOBBoxCoder:
    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def encode(self, bboxes, gt_bboxes, stride):
        # Implementation of YOLO encoding
        return gt_bboxes

    def decode(self, bboxes, pred_bboxes, stride):
        # Implementation of YOLO decoding
        return pred_bboxes
