# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from visdet.registry import MODELS
from .utils import weighted_loss


@weighted_loss
def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Wrapper of mse loss."""
    return F.mse_loss(pred, target, reduction="none")


@MODELS.register_module()
class MSELoss(nn.Module):
    """MSELoss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, reduction: str = "mean", loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Tensor | None = None,
        avg_factor: int | None = None,
        reduction_override: str | None = None,
    ) -> Tensor:
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * mse_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss
