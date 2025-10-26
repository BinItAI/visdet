# ruff: noqa
# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.
"""Cascade R-CNN detector implementation for visdet."""

from visdet.registry import MODELS
from visdet.utils.typing_utils import ConfigType, OptConfigType, OptMultiConfig

from .two_stage import TwoStageDetector


@MODELS.register_module()
class CascadeRCNN(TwoStageDetector):
    r"""Implementation of `Cascade R-CNN <https://arxiv.org/abs/1906.09756>`_."""

    def __init__(
        self,
        backbone: ConfigType,
        neck: OptConfigType = None,
        rpn_head: OptConfigType = None,
        roi_head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
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
