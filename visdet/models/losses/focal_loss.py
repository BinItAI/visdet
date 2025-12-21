# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from visdet.models.losses.utils import weight_reduce_loss
from visdet.registry import MODELS


def sigmoid_focal_loss(
    pred: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "mean",
    avg_factor: float | None = None,
) -> Tensor:
    """Sigmoid focal loss.

    Pure PyTorch implementation of focal loss, a dynamically scaled cross
    entropy loss, where the scaling factor decays to zero as confidence
    in the correct class increases.

    Args:
        pred (Tensor): Predicted classification logits with shape (N, C)
            where C is the number of classes.
        target (Tensor): Target labels with shape (N, C) as one-hot encoding
            or (N,) as class indices.
        weight (Tensor, optional): Sample-wise loss weight.
        gamma (float): The gamma for calculating the modulating factor.
            Defaults to 2.0.
        alpha (float): A balanced form for focal loss. Defaults to 0.25.
        reduction (str): The reduction method. Options are "none", "mean"
            and "sum". Defaults to "mean".
        avg_factor (float, optional): Average factor for the loss.

    Returns:
        Tensor: The calculated focal loss.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)

    # Binary cross entropy
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)

    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * focal_weight

    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def py_focal_loss_with_prob(
    pred: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "mean",
    avg_factor: float | None = None,
) -> Tensor:
    """Focal loss with probabilities as input.

    Args:
        pred (Tensor): Predicted probabilities with shape (N, C).
        target (Tensor): Target labels with shape (N, C) as one-hot.
        weight (Tensor, optional): Sample-wise loss weight.
        gamma (float): The gamma for calculating the modulating factor.
        alpha (float): A balanced form for focal loss.
        reduction (str): The reduction method.
        avg_factor (float, optional): Average factor for the loss.

    Returns:
        Tensor: The calculated focal loss.
    """
    target = target.type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)

    # Use binary cross entropy
    loss = F.binary_cross_entropy(pred, target, reduction="none") * focal_weight

    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@MODELS.register_module()
class FocalLoss(nn.Module):
    """Focal Loss for dense object detection.

    This is the standard focal loss used in RetinaNet. It down-weights
    well-classified examples and focuses on hard negatives.

    Loss = -alpha * (1 - pt)^gamma * log(pt)

    Args:
        use_sigmoid (bool): Whether to use sigmoid activation for
            classification. Only True is supported. Defaults to True.
        gamma (float): The gamma for calculating the modulating factor.
            Defaults to 2.0.
        alpha (float): A balanced form for focal loss. Defaults to 0.25.
        reduction (str): The reduction method. Options are "none", "mean"
            and "sum". Defaults to "mean".
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        activated (bool): Whether the input is activated. If True, the input
            is probabilities, otherwise it is logits. Defaults to False.
    """

    def __init__(
        self,
        use_sigmoid: bool = True,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        activated: bool = False,
    ) -> None:
        super().__init__()
        assert use_sigmoid is True, "Only sigmoid focal loss is supported"
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

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
            target (Tensor): Target labels with shape (N, C) or (N,).
            weight (Tensor, optional): Sample-wise loss weight.
            avg_factor (float, optional): Average factor for the loss.
            reduction_override (str, optional): Override the reduction method.

        Returns:
            Tensor: The calculated focal loss.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction

        if self.activated:
            processed_target = target
            if processed_target.shape != pred.shape:
                raise ValueError("Activated focal loss expects probability targets matching prediction shape.")
            loss_cls = self.loss_weight * py_focal_loss_with_prob(
                pred,
                processed_target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor,
            )
        else:
            processed_target = target
            if processed_target.shape != pred.shape:
                if processed_target.ndim > 1 and processed_target.shape[-1] == 1:
                    processed_target = processed_target.view(-1)
                if processed_target.ndim == 1:
                    processed_target = processed_target.long()
                num_classes = pred.size(1)
                processed_target = F.one_hot(processed_target, num_classes=num_classes + 1)
                processed_target = processed_target[:, :num_classes]
            processed_target = processed_target.type_as(pred)
            loss_cls = self.loss_weight * sigmoid_focal_loss(
                pred,
                processed_target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor,
            )
        return loss_cls
