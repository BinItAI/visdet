# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from visdet.models.losses.utils import weight_reduce_loss
from visdet.registry import MODELS


def varifocal_loss(
    pred: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    alpha: float = 0.75,
    gamma: float = 2.0,
    iou_weighted: bool = True,
    reduction: str = "mean",
    avg_factor: float | None = None,
) -> Tensor:
    """Varifocal Loss.

    Varifocal loss is proposed in VarifocalNet: An IoU-aware Dense Object
    Detector. It focuses on training high-quality examples more than
    hard negatives.

    Args:
        pred (Tensor): Predicted classification logits with shape (N, C).
        target (Tensor): Target IoU scores with shape (N, C). The positive
            locations should have IoU values, while negative locations
            should be 0.
        weight (Tensor, optional): Sample-wise loss weight.
        alpha (float): A balance factor for the negative examples.
            Defaults to 0.75.
        gamma (float): The gamma for weighting the loss. Defaults to 2.0.
        iou_weighted (bool): Whether to weight the loss by the target IoU.
            Defaults to True.
        reduction (str): The reduction method.
        avg_factor (float, optional): Average factor for the loss.

    Returns:
        Tensor: Varifocal loss.
    """
    assert pred.size() == target.size()
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)

    if iou_weighted:
        focal_weight = (
            target * (target > 0.0).float() + alpha * (pred_sigmoid - target).abs().pow(gamma) * (target <= 0.0).float()
        )
    else:
        focal_weight = (target > 0.0).float() + alpha * (pred_sigmoid - target).abs().pow(gamma) * (
            target <= 0.0
        ).float()

    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * focal_weight

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@MODELS.register_module()
class VarifocalLoss(nn.Module):
    """Varifocal Loss.

    Varifocal loss is proposed in VarifocalNet paper. Unlike focal loss which
    treats all negative samples equally, varifocal loss uses asymmetric
    weighting: high alpha for negatives and IoU-weighted for positives.

    Args:
        use_sigmoid (bool): Whether to use sigmoid for classification.
            Only True is supported. Defaults to True.
        alpha (float): A balance factor for negative examples.
            Defaults to 0.75.
        gamma (float): The gamma for weighting the loss. Defaults to 2.0.
        iou_weighted (bool): Whether to weight positive samples by IoU.
            Defaults to True.
        reduction (str): The reduction method. Options are "none", "mean"
            and "sum". Defaults to "mean".
        loss_weight (float): Weight of the loss. Defaults to 1.0.
    """

    def __init__(
        self,
        use_sigmoid: bool = True,
        alpha: float = 0.75,
        gamma: float = 2.0,
        iou_weighted: bool = True,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        assert use_sigmoid is True, "Only sigmoid varifocal loss is supported"
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Tensor | None = None,
        avg_factor: float | None = None,
        reduction_override: str | None = None,
    ) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, C).
            target (Tensor): Target IoU scores with shape (N, C).
            weight (Tensor, optional): Sample-wise loss weight.
            avg_factor (float, optional): Average factor for the loss.
            reduction_override (str, optional): Override the reduction method.

        Returns:
            Tensor: The calculated varifocal loss.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * varifocal_loss(
            pred,
            target,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            iou_weighted=self.iou_weighted,
            reduction=reduction,
            avg_factor=avg_factor,
        )
        return loss_cls
