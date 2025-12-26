# Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

import torch
import torch.nn as nn

from visdet.models.losses.utils import weight_reduce_loss
from visdet.registry import MODELS


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-3,
    reduction: str = "mean",
    naive_dice: bool = False,
    avg_factor: int | None = None,
) -> torch.Tensor:
    input = pred.flatten(1)
    target = target.flatten(1).float()

    a = torch.sum(input * target, 1)
    if naive_dice:
        b = torch.sum(input, 1)
        c = torch.sum(target, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(input * input, 1) + eps
        c = torch.sum(target * target, 1) + eps
        d = (2 * a) / (b + c)

    loss = 1 - d
    if weight is not None:
        if weight.ndim != loss.ndim:
            raise ValueError("weight must have the same dims as loss")
        if len(weight) != len(pred):
            raise ValueError("weight must have same length as pred")
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@MODELS.register_module()
class DiceLoss(nn.Module):
    def __init__(
        self,
        use_sigmoid: bool = True,
        activate: bool = True,
        reduction: str = "mean",
        naive_dice: bool = False,
        loss_weight: float = 1.0,
        eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.activate = activate
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        reduction_override: str | None = None,
        avg_factor: int | None = None,
    ) -> torch.Tensor:
        if reduction_override not in (None, "none", "mean", "sum"):
            raise ValueError("invalid reduction_override")
        reduction = reduction_override if reduction_override else self.reduction

        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            else:
                raise NotImplementedError

        return self.loss_weight * dice_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            naive_dice=self.naive_dice,
            avg_factor=avg_factor,
        )
