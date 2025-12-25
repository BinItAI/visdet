# Copyright (c) OpenMMLab. All rights reserved.
from visdet.registry import MODELS

from .single_stage import SingleStageDetector


@MODELS.register_module()
class GFL(SingleStageDetector):
    """Implementation of `GFL <https://arxiv.org/abs/2006.04388>`_"""

    def __init__(
        self,
        backbone,
        neck,
        bbox_head,
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
