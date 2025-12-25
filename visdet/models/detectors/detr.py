# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from visdet.registry import MODELS
from visdet.utils.typing_utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector


@MODELS.register_module()
class DETR(SingleStageDetector):
    """Implementation of `DETR <https://arxiv.org/abs/2005.12872>`_"""

    def __init__(
        self,
        backbone: ConfigType,
        bbox_head: OptConfigType,
        neck: OptConfigType = None,
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

    def forward(self, inputs: Tensor, data_samples: list | None = None, mode: str = "tensor"):
        """The unified entry for a forward process in both training and test."""
        if mode == "loss":
            return self.loss(inputs, data_samples)
        elif mode == "predict":
            return self.predict(inputs, data_samples)
        elif mode == "tensor":
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". Only supports loss, predict and tensor mode')

    def loss(self, batch_inputs: Tensor, batch_data_samples: list) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        x = self.extract_feat(batch_inputs)

        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]
        outputs_class, outputs_coord = self.bbox_head(x, batch_img_metas)

        batch_gt_instances = [data_samples.gt_instances for data_samples in batch_data_samples]

        losses = self.bbox_head.loss(
            outputs_class,
            outputs_coord,
            batch_gt_instances,
            batch_img_metas,
        )
        return losses

    def predict(self, batch_inputs: Tensor, batch_data_samples: list, rescale: bool = True) -> list:
        """Predict results from a batch of inputs and data samples."""
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.predict(x, batch_data_samples, rescale=rescale)
        return results
