# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor

from visdet.models.losses.utils import weighted_loss
from visdet.registry import MODELS


@weighted_loss
def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean Squared Error loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The target.

    Returns:
        Tensor: The calculated MSE loss.
    """
    return (pred - target) ** 2


@MODELS.register_module()
class MSELoss(nn.Module):
    """Mean Squared Error Loss.

    Args:
        reduction (str): The reduction method. Options are "none", "mean"
            and "sum". Defaults to "mean".
        loss_weight (float): Weight of the loss. Defaults to 1.0.
    """

    def __init__(
        self,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
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
            pred (Tensor): The prediction.
            target (Tensor): The target.
            weight (Tensor, optional): Sample-wise loss weight.
            avg_factor (float, optional): Average factor for the loss.
            reduction_override (str, optional): Override the reduction method.

        Returns:
            Tensor: The calculated MSE loss.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * mse_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss
