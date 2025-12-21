# Copyright (c) OpenMMLab. All rights reserved.
"""FCOS detector for visdet."""

from __future__ import annotations

from visdet.models.detectors.single_stage import SingleStageDetector
from visdet.registry import MODELS
from visdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class FCOS(SingleStageDetector):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_."""

    def __init__(
        self,
        backbone: ConfigType,
        neck: OptConfigType = None,
        bbox_head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )
