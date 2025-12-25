# Copyright (c) OpenMMLab. All rights reserved.
from visdet.registry import MODELS

from .two_stage import TwoStageDetector


@MODELS.register_module()
class FastRCNN(TwoStageDetector):
    """Implementation of `Fast R-CNN <https://arxiv.org/abs/1504.08083>`_"""

    def __init__(
        self,
        backbone,
        roi_head,
        train_cfg,
        test_cfg,
        neck=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super(FastRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )
