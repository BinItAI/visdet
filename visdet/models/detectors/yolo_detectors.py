# Copyright (c) OpenMMLab. All rights reserved.
from visdet.registry import MODELS

from .single_stage import SingleStageDetector


@MODELS.register_module()
class YOLOV3(SingleStageDetector):
    """Implementation of `YOLOv3 <https://arxiv.org/abs/1804.02767>`_"""

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


@MODELS.register_module()
class YOLOX(SingleStageDetector):
    """Implementation of `YOLOX <https://arxiv.org/abs/2107.08430>`_"""

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
