# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from visdet.registry import MODELS
from .utils import weighted_loss


@weighted_loss
def quality_focal_loss(pred: Tensor, target: tuple[Tensor, Tensor], beta: float = 2.0) -> Tensor:
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.
    """
    assert len(target) == 2, """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    label, score = target

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(pred, zerolabel, reduction="none") * scale_factor.pow(beta)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
        pred[pos, pos_label], score[pos], reduction="none"
    ) * scale_factor.abs().pow(beta)

    loss = loss.sum(dim=1, keepdim=False)
    return loss


@weighted_loss
def distribution_focal_loss(pred: Tensor, label: Tensor) -> Tensor:
    r"""Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.
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
    def __init__(
        self,
        use_sigmoid: bool = True,
        beta: float = 2.0,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        activated: bool = False,
    ) -> None:
        super(QualityFocalLoss, self).__init__()
        assert use_sigmoid is True, "Only sigmoid in QFL supported now."
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
        avg_factor: int | None = None,
        reduction_override: str | None = None,
    ) -> Tensor:
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if self.use_sigmoid:
            if self.activated:
                raise NotImplementedError("activated=True is not supported in QFL yet")
            else:
                loss_cls = self.loss_weight * quality_focal_loss(
                    pred,
                    target,
                    weight,
                    beta=self.beta,
                    reduction=reduction,
                    avg_factor=avg_factor,
                )
        else:
            raise NotImplementedError
        return loss_cls


@MODELS.register_module()
class DistributionFocalLoss(nn.Module):
    def __init__(self, reduction: str = "mean", loss_weight: float = 1.0) -> None:
        super(DistributionFocalLoss, self).__init__()
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
        loss_cls = self.loss_weight * distribution_focal_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss_cls
