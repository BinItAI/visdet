# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from visdet.models.losses.utils import weighted_loss
from visdet.registry import MODELS


@weighted_loss
def quality_focal_loss(
    pred: Tensor,
    target: tuple[Tensor, Tensor],
    beta: float = 2.0,
) -> Tensor:
    """Quality Focal Loss (QFL).

    QFL is proposed in Generalized Focal Loss (GFL) paper. Instead of using
    hard labels (0 or 1), QFL uses the IoU between predicted bbox and GT bbox
    as soft labels for classification.

    Args:
        pred (Tensor): Predicted joint representation of classification and
            quality (IoU) with shape (N, C).
        target (tuple[Tensor, Tensor]): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.

    Returns:
        Tensor: Quality focal loss.
    """
    assert len(target) == 2
    label, score = target

    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)

    loss = F.binary_cross_entropy_with_logits(pred, zerolabel, reduction="none") * scale_factor.pow(beta)

    # FG cat_id: [0, num_classes - 1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero(as_tuple=False).squeeze(1)
    pos_label = label[pos].long()

    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
        pred[pos, pos_label], score[pos], reduction="none"
    ) * scale_factor.abs().pow(beta)

    loss = loss.sum(dim=1, keepdim=False)
    return loss


@weighted_loss
def quality_focal_loss_with_prob(
    pred: Tensor,
    target: tuple[Tensor, Tensor],
    beta: float = 2.0,
) -> Tensor:
    """Quality Focal Loss (QFL) with probability input.

    Args:
        pred (Tensor): Predicted probabilities with shape (N, C).
        target (tuple[Tensor, Tensor]): Target label and quality.
        beta (float): The beta parameter.

    Returns:
        Tensor: Quality focal loss.
    """
    assert len(target) == 2
    label, score = target

    scale_factor = pred
    zerolabel = scale_factor.new_zeros(pred.shape)

    loss = F.binary_cross_entropy(pred, zerolabel, reduction="none") * scale_factor.pow(beta)

    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero(as_tuple=False).squeeze(1)
    pos_label = label[pos].long()

    scale_factor = score[pos] - pred[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy(
        pred[pos, pos_label], score[pos], reduction="none"
    ) * scale_factor.abs().pow(beta)

    loss = loss.sum(dim=1, keepdim=False)
    return loss


@weighted_loss
def distribution_focal_loss(pred: Tensor, label: Tensor) -> Tensor:
    """Distribution Focal Loss (DFL).

    DFL is used in GFL for bounding box regression. It uses a discrete
    probability distribution to represent bounding box regression targets,
    achieving better localization accuracy.

    Args:
        pred (Tensor): Predicted general distribution of bounding boxes
            (before softmax) with shape (N, n+1), where n is max regression
            target.
        label (Tensor): Target regression value with shape (N,).

    Returns:
        Tensor: Distribution focal loss.
    """
    dis_left = label.long()
    dis_right = dis_left + 1
    weight_left = dis_right.float() - label
    weight_right = label - dis_left.float()

    loss = (
        F.cross_entropy(pred, dis_left, reduction="none") * weight_left
        + F.cross_entropy(pred, dis_right, reduction="none") * weight_right
    )
    return loss


@MODELS.register_module()
class QualityFocalLoss(nn.Module):
    """Quality Focal Loss (QFL).

    QFL is proposed in Generalized Focal Loss paper. It uses soft labels based
    on IoU between predictions and ground truth for classification.

    Args:
        use_sigmoid (bool): Whether to use sigmoid for classification.
            Only True is supported. Defaults to True.
        beta (float): The beta parameter for calculating modulating factor.
            Defaults to 2.0.
        reduction (str): The reduction method. Options are "none", "mean"
            and "sum". Defaults to "mean".
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        activated (bool): Whether the input is activated (probabilities).
            Defaults to False.
    """

    def __init__(
        self,
        use_sigmoid: bool = True,
        beta: float = 2.0,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        activated: bool = False,
    ) -> None:
        super().__init__()
        assert use_sigmoid is True, "Only sigmoid QFL is supported"
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(
        self,
        pred: Tensor,
        target: tuple[Tensor, Tensor],
        weight: Tensor | None = None,
        avg_factor: float | None = None,
        reduction_override: str | None = None,
    ) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, C).
            target (tuple[Tensor, Tensor]): Target label and quality score.
            weight (Tensor, optional): Sample-wise loss weight.
            avg_factor (float, optional): Average factor for the loss.
            reduction_override (str, optional): Override the reduction method.

        Returns:
            Tensor: The calculated QFL loss.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction

        if self.activated:
            loss_cls = self.loss_weight * quality_focal_loss_with_prob(
                pred, target, weight, beta=self.beta, reduction=reduction, avg_factor=avg_factor
            )
        else:
            loss_cls = self.loss_weight * quality_focal_loss(
                pred, target, weight, beta=self.beta, reduction=reduction, avg_factor=avg_factor
            )
        return loss_cls


@MODELS.register_module()
class DistributionFocalLoss(nn.Module):
    """Distribution Focal Loss (DFL).

    DFL is proposed in GFL for better bounding box regression. It represents
    the regression target as a discrete probability distribution.

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
            pred (Tensor): Predicted distribution with shape (N, n+1).
            target (Tensor): Target regression value with shape (N,).
            weight (Tensor, optional): Sample-wise loss weight.
            avg_factor (float, optional): Average factor for the loss.
            reduction_override (str, optional): Override the reduction method.

        Returns:
            Tensor: The calculated DFL loss.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * distribution_focal_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss
