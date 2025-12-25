# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from visdet.registry import MODELS
from .utils import weight_reduce_loss, weighted_loss


@weighted_loss
def py_sigmoid_focal_loss(
    pred: Tensor,
    target: Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> Tensor:
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (Tensor): The learning label of the prediction.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.

    Returns:
        Tensor: Loss tensor.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * focal_weight
    return loss


def focal_loss_weight_reshape(loss: Tensor, weight: Tensor | None) -> Tensor | None:
    """Reshape weight to match the shape of loss."""
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    return weight


@MODELS.register_module()
class FocalLoss(nn.Module):
    def __init__(
        self,
        use_sigmoid: bool = True,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        activated: bool = False,
    ) -> None:
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            activated (bool, optional): Whether the input is activated.
                If True, it means the input has been activated and can be
                treated as probabilities. Else, it should be treated as logits.
                Defaults to False.
        """
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, "Only sigmoid focal loss supported now."
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
        avg_factor: int | None = None,
        reduction_override: str | None = None,
    ) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning label of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            Tensor: The calculated loss
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if self.use_sigmoid:
            if self.activated:
                # TODO: implement py_focal_loss_with_prob if needed
                raise NotImplementedError("activated=True is not supported yet")
            else:
                num_classes = pred.size(1)
                target = F.one_hot(target, num_classes=num_classes + 1)
                target = target[:, :num_classes].type_as(pred)

                # Reshape weight if needed
                if weight is not None:
                    if weight.shape != pred.shape:
                        if weight.size(0) == pred.size(0):
                            weight = weight.view(-1, 1)
                        else:
                            assert weight.numel() == pred.numel()
                            weight = weight.view(pred.size(0), -1)

                loss_cls = self.loss_weight * py_sigmoid_focal_loss(
                    pred,
                    target,
                    weight,
                    gamma=self.gamma,
                    alpha=self.alpha,
                    reduction=reduction,
                    avg_factor=avg_factor,
                )

        else:
            raise NotImplementedError
        return loss_cls
