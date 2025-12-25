# Copyright (c) OpenMMLab. All rights reserved.
from visdet.registry import MODELS
from .single_stage import SingleStageDetector


@MODELS.register_module()
class FoveaBox(SingleStageDetector):
    """Implementation of `FoveaBox <https://arxiv.org/abs/1904.03797>`_"""

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
class FSAF(SingleStageDetector):
    """Implementation of `FSAF <https://arxiv.org/abs/1903.00621>`_"""

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
class CenterNet(SingleStageDetector):
    """Implementation of `CenterNet (Object as Points) <https://arxiv.org/abs/1904.07850>`_"""

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
