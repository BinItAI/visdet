# Copyright (c) OpenMMLab. All rights reserved.
"""Single-stage detector for visdet."""

import torch
from torch import Tensor

from visdet.models.detectors.base import BaseDetector, SampleList
from visdet.registry import MODELS
from visdet.utils.typing_utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly predict dense bounding boxes without
    the need for region proposal networks (RPN). Examples include RetinaNet,
    FCOS, SSD, YOLO, etc.

    Args:
        backbone (dict): Config for the backbone network.
        neck (dict, optional): Config for the neck (feature aggregation).
        bbox_head (dict, optional): Config for the detection head.
        train_cfg (dict, optional): Training config.
        test_cfg (dict, optional): Testing config.
        data_preprocessor (dict, optional): Data preprocessor config.
        init_cfg (dict, optional): Initialization config.
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
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # Build backbone
        if isinstance(backbone, dict):
            self.backbone = MODELS.build(backbone)
        else:
            self.backbone = backbone

        # Build neck
        if neck is not None:
            if isinstance(neck, dict):
                self.neck = MODELS.build(neck)
            else:
                self.neck = neck

        # Build bbox head
        if bbox_head is not None:
            if isinstance(bbox_head, dict):
                bbox_head.update(train_cfg=train_cfg, test_cfg=test_cfg)
                self.bbox_head = MODELS.build(bbox_head)
            else:
                self.bbox_head = bbox_head

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, batch_inputs: Tensor) -> tuple[Tensor, ...]:
        """Extract features from images.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H, W).

        Returns:
            tuple[Tensor]: Multi-level features from backbone/neck.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def _forward(
        self,
        batch_inputs: Tensor,
        batch_data_samples: SampleList | None = None,
    ) -> tuple:
        """Network forward process.

        Usually includes backbone, neck and head forward without any
        post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`], optional): Each
                item contains the meta information of each image.

        Returns:
            tuple: A tuple of features from the bbox head.
        """
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return results

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples containing ground truth.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)
        return losses

    def predict(
        self,
        batch_inputs: Tensor,
        batch_data_samples: SampleList,
        rescale: bool = True,
    ) -> SampleList:
        """Predict results from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples containing meta information.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results.
        """
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)
        return batch_data_samples
