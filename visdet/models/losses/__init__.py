# ruff: noqa
# Copyright (c) OpenMMLab. All rights reserved.

from visdet.models.losses.accuracy import accuracy
from visdet.models.losses.balanced_l1_loss import BalancedL1Loss, balanced_l1_loss
from visdet.models.losses.cross_entropy_loss import CrossEntropyLoss, CrossEntropyCustomLoss
from visdet.models.losses.dice_loss import DiceLoss, dice_loss
from visdet.models.losses.focal_loss import FocalLoss, sigmoid_focal_loss
from visdet.models.losses.gaussian_focal_loss import GaussianFocalLoss, gaussian_focal_loss
from visdet.models.losses.gfocal_loss import (
    DistributionFocalLoss,
    QualityFocalLoss,
    distribution_focal_loss,
    quality_focal_loss,
)
from visdet.models.losses.ghm_loss import GHMC, GHMR
from visdet.models.losses.iou_loss import (
    BoundedIoULoss,
    CIoULoss,
    DIoULoss,
    EIoULoss,
    GIoULoss,
    IoULoss,
    bbox_overlaps,
    bounded_iou_loss,
    ciou_loss,
    diou_loss,
    eiou_loss,
    giou_loss,
    iou_loss,
)
from visdet.models.losses.mse_loss import MSELoss, mse_loss
from visdet.models.losses.smooth_l1_loss import L1Loss, SmoothL1Loss, l1_loss, smooth_l1_loss
from visdet.models.losses.utils import reduce_loss, weight_reduce_loss, weighted_loss
from visdet.models.losses.varifocal_loss import VarifocalLoss, varifocal_loss

__all__ = [
    # Accuracy
    "accuracy",
    # Utils
    "reduce_loss",
    "weight_reduce_loss",
    "weighted_loss",
    # Classification losses
    "CrossEntropyLoss",
    "CrossEntropyCustomLoss",
    "FocalLoss",
    "sigmoid_focal_loss",
    "QualityFocalLoss",
    "quality_focal_loss",
    "DistributionFocalLoss",
    "distribution_focal_loss",
    "VarifocalLoss",
    "varifocal_loss",
    "GaussianFocalLoss",
    "gaussian_focal_loss",
    "GHMC",
    "GHMR",
    # Regression losses
    "L1Loss",
    "l1_loss",
    "SmoothL1Loss",
    "smooth_l1_loss",
    "MSELoss",
    "mse_loss",
    "BalancedL1Loss",
    "balanced_l1_loss",
    # IoU losses
    "IoULoss",
    "iou_loss",
    "BoundedIoULoss",
    "bounded_iou_loss",
    "GIoULoss",
    "giou_loss",
    "DIoULoss",
    "diou_loss",
    "CIoULoss",
    "ciou_loss",
    "EIoULoss",
    "eiou_loss",
    "bbox_overlaps",
    # Segmentation losses
    "DiceLoss",
    "dice_loss",
]
