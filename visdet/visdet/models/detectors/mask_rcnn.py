# ruff: noqa
# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.
"""Mask R-CNN implementation for visdet."""

from visengine.config import ConfigDict

from visdet.registry import MODELS
from visdet.utils.typing_utils import OptConfigType, OptMultiConfig

from .two_stage import TwoStageDetector


@MODELS.register_module()
class MaskRCNN(TwoStageDetector):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_

    Mask R-CNN extends Faster R-CNN by adding a branch for predicting
    segmentation masks on each Region of Interest (RoI), in parallel
    with the existing branch for classification and bounding box regression.

    Args:
        backbone (ConfigDict): Configuration for the backbone network.
        rpn_head (ConfigDict): Configuration for the RPN head.
        roi_head (ConfigDict): Configuration for the RoI head.
        train_cfg (ConfigDict): Training configuration.
        test_cfg (ConfigDict): Testing configuration.
        neck (OptConfigType, optional): Configuration for the neck network.
            Default: None.
        data_preprocessor (OptConfigType, optional): Configuration for the
            data preprocessor. Default: None.
        init_cfg (OptMultiConfig, optional): Initialization configuration.
            Default: None.
    """

    def __init__(
        self,
        backbone,  # Can be ConfigDict or direct object
        rpn_head,  # Can be ConfigDict or direct object
        roi_head,  # Can be ConfigDict or direct object
        train_cfg,  # Can be ConfigDict or direct object
        test_cfg,  # Can be ConfigDict or direct object
        neck=None,  # Can be ConfigDict or direct object
        data_preprocessor=None,  # Can be ConfigDict or direct object
        init_cfg=None,
    ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor,
        )
