# Copyright (c) OpenMMLab. All rights reserved.
from visdet.registry import MODELS
from .two_stage import TwoStageDetector


@MODELS.register_module()
class FasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(
        self,
        backbone,
        rpn_head,
        roi_head,
        train_cfg,
        test_cfg,
        neck=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )
