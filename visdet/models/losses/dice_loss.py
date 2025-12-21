# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor

from visdet.registry import MODELS


def dice_loss(
    pred: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    eps: float = 1e-3,
    reduction: str = "mean",
    naive_dice: bool = False,
    avg_factor: float | None = None,
) -> Tensor:
    """Dice Loss.

    Compute the dice loss between prediction and target. Commonly used in
    segmentation tasks.

    Args:
        pred (Tensor): Predicted probabilities with shape (N, C, *) or (N, *).
        target (Tensor): Target labels with shape (N, C, *) or (N, *).
        weight (Tensor, optional): Sample-wise loss weight.
        eps (float): A value added for numerical stability. Defaults to 1e-3.
        reduction (str): The reduction method.
        naive_dice (bool): If True, use the naive dice formula
            (without laplace smoothing). Defaults to False.
        avg_factor (float, optional): Average factor for the loss.

    Returns:
        Tensor: Dice loss.
    """
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
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
        loss = loss * weight

    if avg_factor is None:
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss
    else:
        if reduction == "mean":
            return loss.sum() / avg_factor
        else:
            return loss


@MODELS.register_module()
class DiceLoss(nn.Module):
    """Dice Loss.

    Computes the dice loss between prediction and target, commonly used
    in segmentation tasks.

    Args:
        use_sigmoid (bool): Whether to use sigmoid for the prediction.
            Defaults to True.
        activate (bool): Whether to activate the prediction. Defaults to True.
        reduction (str): The reduction method. Options are "none", "mean"
            and "sum". Defaults to "mean".
        naive_dice (bool): If True, use naive dice formula. Defaults to False.
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        eps (float): A value for numerical stability. Defaults to 1e-3.
    """

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
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Tensor | None = None,
        reduction_override: str | None = None,
        avg_factor: float | None = None,
    ) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted probabilities or logits.
            target (Tensor): Target labels.
            weight (Tensor, optional): Sample-wise loss weight.
            reduction_override (str, optional): Override the reduction method.
            avg_factor (float, optional): Average factor for the loss.

        Returns:
            Tensor: The calculated dice loss.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction

        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            else:
                raise NotImplementedError

        loss = self.loss_weight * dice_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            naive_dice=self.naive_dice,
            avg_factor=avg_factor,
        )
        return loss
