# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor

from visdet.models.losses.utils import weight_reduce_loss
from visdet.registry import MODELS


def gaussian_focal_loss(
    pred: Tensor,
    gaussian_target: Tensor,
    alpha: float = 2.0,
    gamma: float = 4.0,
    reduction: str = "mean",
    avg_factor: float | None = None,
) -> Tensor:
    """Gaussian Focal Loss for heatmap prediction in CenterNet.

    This loss is designed for training heatmap-based object detectors like
    CenterNet. It uses a penalty-reduced pixel-wise logistic regression with
    a focal loss for the negative class.

    Args:
        pred (Tensor): Predicted heatmap with shape (N, C, H, W).
        gaussian_target (Tensor): Target Gaussian heatmap with shape
            (N, C, H, W). Values range from 0 to 1.
        alpha (float): Focal loss alpha. Defaults to 2.0.
        gamma (float): Focal loss gamma. Defaults to 4.0.
        reduction (str): The reduction method.
        avg_factor (float, optional): Average factor for the loss.

    Returns:
        Tensor: Gaussian focal loss.
    """
    eps = 1e-12
    pred = pred.sigmoid().clamp(min=eps, max=1 - eps)

    pos_weights = gaussian_target.eq(1).float()
    neg_weights = (1 - gaussian_target).pow(gamma)

    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights

    loss = pos_loss + neg_loss
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)
    return loss


@MODELS.register_module()
class GaussianFocalLoss(nn.Module):
    """Gaussian Focal Loss for CenterNet.

    This loss is used for training heatmap-based detectors. The positive
    locations get standard focal loss, while negative locations get a
    penalty-reduced focal loss weighted by a Gaussian kernel.

    Args:
        alpha (float): Focal loss alpha parameter. Defaults to 2.0.
        gamma (float): Penalty for negative samples. Defaults to 4.0.
        reduction (str): The reduction method. Options are "none", "mean"
            and "sum". Defaults to "mean".
        loss_weight (float): Weight of the loss. Defaults to 1.0.
    """

    def __init__(
        self,
        alpha: float = 2.0,
        gamma: float = 4.0,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
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
            pred (Tensor): Predicted heatmap logits.
            target (Tensor): Target Gaussian heatmap (0-1 values).
            weight (Tensor, optional): Sample-wise loss weight.
            avg_factor (float, optional): Average factor for the loss.
            reduction_override (str, optional): Override the reduction method.

        Returns:
            Tensor: The calculated Gaussian focal loss.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * gaussian_focal_loss(
            pred,
            target,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=reduction,
            avg_factor=avg_factor,
        )
        return loss
