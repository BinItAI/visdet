# Copyright (c) OpenMMLab. All rights reserved.
"""RetinaNet detector for visdet."""

from __future__ import annotations

from visdet.models.detectors.single_stage import SingleStageDetector
from visdet.registry import MODELS
from visdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class RetinaNet(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_.

    Args:
        backbone (ConfigType): Backbone config or instance.
        neck (ConfigType, optional): Neck config or instance. Defaults to None.
        bbox_head (ConfigType): Bounding box head config or instance.
        train_cfg (ConfigType, optional): Training config. Defaults to None.
        test_cfg (ConfigType, optional): Testing config. Defaults to None.
        data_preprocessor (ConfigType, optional): Data preprocessor config.
            Defaults to None.
        init_cfg (ConfigType, optional): Detector init config. Defaults to None.
    """

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
