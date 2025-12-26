# Copyright (c) OpenMMLab. All rights reserved.

import torch

from visdet.engine.dist import get_world_size
from visdet.engine.logging import print_log
from visdet.registry import MODELS
from visdet.utils.typing_utils import ConfigType, OptConfigType, OptMultiConfig
from visdet.models.detectors.single_stage import SingleStageDetector


@MODELS.register_module()
class RTMDet(SingleStageDetector):
    """Implementation of RTMDet."""

    def __init__(
        self,
        backbone: ConfigType,
        neck: ConfigType,
        bbox_head: ConfigType,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
        use_syncbn: bool = True,
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

        if use_syncbn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
            print_log("Using SyncBatchNorm()", logger="current")
