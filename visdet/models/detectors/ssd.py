# Copyright (c) OpenMMLab. All rights reserved.
from visdet.registry import MODELS

from .single_stage import SingleStageDetector


@MODELS.register_module()
class SSD(SingleStageDetector):
    """Implementation of `SSD <https://arxiv.org/abs/1512.02325>`_"""

    def __init__(
        self,
        backbone,
        neck=None,
        bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )
