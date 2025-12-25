# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from visdet.registry import MODELS

from .utils import weighted_loss


@weighted_loss
def balanced_l1_loss(
    pred: Tensor,
    target: Tensor,
    beta: float = 1.0,
    alpha: float = 0.5,
    gamma: float = 1.5,
) -> Tensor:
    """Calculate balanced L1 loss.

    Please see the `Libra R-CNN <https://arxiv.org/pdf/1904.02701.pdf>`_
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

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 1.5,
        beta: float = 1.0,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super(BalancedL1Loss, self).__init__()
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
        avg_factor: int | None = None,
        reduction_override: str | None = None,
        **kwargs,
    ) -> Tensor:
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
