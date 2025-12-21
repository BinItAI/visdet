# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from visdet.models.losses.utils import weighted_loss
from visdet.registry import MODELS


@weighted_loss
def balanced_l1_loss(
    pred: Tensor,
    target: Tensor,
    beta: float = 1.0,
    alpha: float = 0.5,
    gamma: float = 1.5,
) -> Tensor:
    """Balanced L1 Loss.

    Balanced L1 loss is from Libra R-CNN paper. It aims to balance the
    loss contributions from different levels of regression error.

    Args:
        pred (Tensor): The prediction with shape (n, 4).
        target (Tensor): The target with shape (n, 4).
        beta (float): The threshold for switching between L1 and L2 loss.
        alpha (float): The denominator alpha in the balanced L1 loss.
        gamma (float): The gamma for the balanced L1 loss.

    Returns:
        Tensor: The calculated balanced L1 loss.
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()

    diff = torch.abs(pred - target)
    b = np.e ** (gamma / alpha) - 1
    loss = torch.where(
        diff < beta,
        alpha / b * (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
        gamma * diff + gamma / b - alpha * beta,
    )
    return loss


@MODELS.register_module()
class BalancedL1Loss(nn.Module):
    """Balanced L1 Loss.

    Balanced L1 loss from Libra R-CNN paper balances the loss contributions
    from different levels of regression error.

    Args:
        alpha (float): The denominator alpha in the balanced L1 loss.
            Defaults to 0.5.
        gamma (float): The gamma for the balanced L1 loss. Defaults to 1.5.
        beta (float): The threshold for switching between L1 and L2.
            Defaults to 1.0.
        reduction (str): The reduction method. Options are "none", "mean"
            and "sum". Defaults to "mean".
        loss_weight (float): Weight of the loss. Defaults to 1.0.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 1.5,
        beta: float = 1.0,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Tensor | None = None,
        avg_factor: float | None = None,
        reduction_override: str | None = None,
        **kwargs,
    ) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The target.
            weight (Tensor, optional): Sample-wise loss weight.
            avg_factor (float, optional): Average factor for the loss.
            reduction_override (str, optional): Override the reduction method.

        Returns:
            Tensor: The calculated balanced L1 loss.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * balanced_l1_loss(
            pred,
            target,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs,
        )
        return loss_bbox
