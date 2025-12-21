# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from torch import Tensor

from visdet.models.losses.utils import weighted_loss
from visdet.registry import MODELS


def bbox_overlaps(
    bboxes1: Tensor,
    bboxes2: Tensor,
    mode: str = "iou",
    is_aligned: bool = False,
    eps: float = 1e-6,
) -> Tensor:
    """Calculate overlap between two sets of bboxes.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) or (m, 4).
        bboxes2 (Tensor): shape (B, n, 4) or (n, 4).
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
        is_aligned (bool): If True, then m and n must be equal and the result
            is (m,) or (B, m). Otherwise, the result is (m, n) or (B, m, n).
        eps (float): A value added to the denominator for numerical stability.

    Returns:
        Tensor: IoU values with shape (m, n), (B, m, n), (m,) or (B, m).
    """
    assert mode in ["iou", "iof", "giou"], f"Unsupported mode {mode}"
    assert bboxes1.size(-1) == 4 and bboxes2.size(-1) == 4

    if bboxes1.dim() == 2:
        bboxes1 = bboxes1.unsqueeze(0)
        bboxes2 = bboxes2.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    m = bboxes1.size(1)
    n = bboxes2.size(1)

    if is_aligned:
        assert m == n
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]

        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

        if mode == "iou":
            union = area1 + area2 - overlap
            ious = overlap / (union + eps)
        elif mode == "iof":
            ious = overlap / (area1 + eps)
        else:  # giou
            union = area1 + area2 - overlap
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
            enclosed_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
            enclosed_area = enclosed_wh[..., 0] * enclosed_wh[..., 1]
            ious = overlap / (union + eps) - (enclosed_area - union) / (enclosed_area + eps)
    else:
        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]

        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

        if mode == "iou":
            union = area1[..., :, None] + area2[..., None, :] - overlap
            ious = overlap / (union + eps)
        elif mode == "iof":
            ious = overlap / (area1[..., :, None] + eps)
        else:  # giou
            union = area1[..., :, None] + area2[..., None, :] - overlap
            enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
            enclosed_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
            enclosed_area = enclosed_wh[..., 0] * enclosed_wh[..., 1]
            ious = overlap / (union + eps) - (enclosed_area - union) / (enclosed_area + eps)

    if squeeze:
        ious = ious.squeeze(0)

    return ious


@weighted_loss
def iou_loss(pred: Tensor, target: Tensor, linear: bool = False, eps: float = 1e-6) -> Tensor:
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU or linear version.

    Args:
        pred (Tensor): Predicted bboxes of shape (n, 4).
        target (Tensor): Target bboxes of shape (n, 4).
        linear (bool): If True, use linear IoU loss, else use log IoU loss.
        eps (float): A value for numerical stability.

    Returns:
        Tensor: IoU loss.
    """
    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
    if linear:
        loss = 1 - ious
    else:
        loss = -ious.log()
    return loss


@weighted_loss
def bounded_iou_loss(pred: Tensor, target: Tensor, beta: float = 0.2, eps: float = 1e-3) -> Tensor:
    """BIoU Loss.

    This is an implementation of paper: Improving Object Localization with
    Fitness NMS and Bounded IoU Loss.

    Args:
        pred (Tensor): Predicted bboxes of shape (n, 4).
        target (Tensor): Target bboxes of shape (n, 4).
        beta (float): Beta parameter for smooth L1 loss.
        eps (float): A value for numerical stability.

    Returns:
        Tensor: BIoU loss.
    """
    pred_ctrx = (pred[:, 0] + pred[:, 2]) * 0.5
    pred_ctry = (pred[:, 1] + pred[:, 3]) * 0.5
    pred_w = pred[:, 2] - pred[:, 0]
    pred_h = pred[:, 3] - pred[:, 1]

    target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
    target_ctry = (target[:, 1] + target[:, 3]) * 0.5
    target_w = target[:, 2] - target[:, 0]
    target_h = target[:, 3] - target[:, 1]

    dx = target_ctrx - pred_ctrx
    dy = target_ctry - pred_ctry

    loss_dx = 1 - torch.max((target_w - 2 * dx.abs()) / (target_w + 2 * dx.abs() + eps), torch.zeros_like(dx))
    loss_dy = 1 - torch.max((target_h - 2 * dy.abs()) / (target_h + 2 * dy.abs() + eps), torch.zeros_like(dy))
    loss_dw = 1 - torch.min(target_w / (pred_w + eps), pred_w / (target_w + eps))
    loss_dh = 1 - torch.min(target_h / (pred_h + eps), pred_h / (target_h + eps))

    loss_comb = torch.stack([loss_dx, loss_dy, loss_dw, loss_dh], dim=-1).view(loss_dx.size(0), -1)

    loss = torch.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta, loss_comb - 0.5 * beta)
    return loss


@weighted_loss
def giou_loss(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """GIoU loss.

    Computing the GIoU loss between a set of predicted bboxes and target bboxes.
    GIoU = IoU - |C - A ∪ B| / |C| where C is the smallest enclosing box.

    Args:
        pred (Tensor): Predicted bboxes of shape (n, 4).
        target (Tensor): Target bboxes of shape (n, 4).
        eps (float): A value for numerical stability.

    Returns:
        Tensor: GIoU loss.
    """
    gious = bbox_overlaps(pred, target, mode="giou", is_aligned=True, eps=eps)
    loss = 1 - gious
    return loss


@weighted_loss
def diou_loss(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """DIoU loss.

    Computing the DIoU loss between a set of predicted bboxes and target bboxes.
    DIoU = IoU - d²/c² where d is the distance between box centers and c is
    the diagonal length of the smallest enclosing box.

    Args:
        pred (Tensor): Predicted bboxes of shape (n, 4).
        target (Tensor): Target bboxes of shape (n, 4).
        eps (float): A value for numerical stability.

    Returns:
        Tensor: DIoU loss.
    """
    # Overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # Areas
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # Enclosing box
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    # Center distance squared
    pred_center = (pred[:, :2] + pred[:, 2:]) / 2
    target_center = (target[:, :2] + target[:, 2:]) / 2
    center_dist_sq = ((pred_center - target_center) ** 2).sum(dim=-1)

    # Diagonal length squared of enclosing box
    enclose_diag_sq = (enclose_wh**2).sum(dim=-1) + eps

    # DIoU
    dious = ious - center_dist_sq / enclose_diag_sq
    loss = 1 - dious
    return loss


@weighted_loss
def ciou_loss(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """CIoU loss.

    Computing the CIoU loss between a set of predicted bboxes and target bboxes.
    CIoU = IoU - d²/c² - αv, where v measures aspect ratio consistency and
    α is a trade-off parameter.

    Args:
        pred (Tensor): Predicted bboxes of shape (n, 4).
        target (Tensor): Target bboxes of shape (n, 4).
        eps (float): A value for numerical stability.

    Returns:
        Tensor: CIoU loss.
    """
    # Overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # Areas
    pred_w = pred[:, 2] - pred[:, 0]
    pred_h = pred[:, 3] - pred[:, 1]
    target_w = target[:, 2] - target[:, 0]
    target_h = target[:, 3] - target[:, 1]

    ap = pred_w * pred_h
    ag = target_w * target_h
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # Enclosing box
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    # Center distance squared
    pred_center = (pred[:, :2] + pred[:, 2:]) / 2
    target_center = (target[:, :2] + target[:, 2:]) / 2
    center_dist_sq = ((pred_center - target_center) ** 2).sum(dim=-1)

    # Diagonal length squared of enclosing box
    enclose_diag_sq = (enclose_wh**2).sum(dim=-1) + eps

    # Aspect ratio consistency
    v = (4 / (math.pi**2)) * torch.pow(torch.atan(target_w / (target_h + eps)) - torch.atan(pred_w / (pred_h + eps)), 2)

    with torch.no_grad():
        alpha = v / (1 - ious + v + eps)

    # CIoU
    cious = ious - center_dist_sq / enclose_diag_sq - alpha * v
    loss = 1 - cious
    return loss


@weighted_loss
def eiou_loss(pred: Tensor, target: Tensor, eps: float = 1e-7, smooth: bool = False) -> Tensor:
    """EIoU loss.

    Computing the EIoU loss between predicted and target bboxes. EIoU loss
    decouples the width and height regression to improve the regression
    performance.

    Args:
        pred (Tensor): Predicted bboxes of shape (n, 4).
        target (Tensor): Target bboxes of shape (n, 4).
        eps (float): A value for numerical stability.
        smooth (bool): Whether to use smooth L1 style loss for width/height.

    Returns:
        Tensor: EIoU loss.
    """
    # Overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # Areas and dimensions
    pred_w = pred[:, 2] - pred[:, 0]
    pred_h = pred[:, 3] - pred[:, 1]
    target_w = target[:, 2] - target[:, 0]
    target_h = target[:, 3] - target[:, 1]

    ap = pred_w * pred_h
    ag = target_w * target_h
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # Enclosing box
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    # Center distance squared
    pred_center = (pred[:, :2] + pred[:, 2:]) / 2
    target_center = (target[:, :2] + target[:, 2:]) / 2
    center_dist_sq = ((pred_center - target_center) ** 2).sum(dim=-1)

    # Enclosing dimensions squared
    cw_sq = enclose_wh[:, 0] ** 2 + eps
    ch_sq = enclose_wh[:, 1] ** 2 + eps

    # Width and height differences squared
    rho_w_sq = (target_w - pred_w) ** 2
    rho_h_sq = (target_h - pred_h) ** 2

    # Diagonal length squared
    c_sq = cw_sq + ch_sq

    # EIoU
    eious = ious - center_dist_sq / c_sq - rho_w_sq / cw_sq - rho_h_sq / ch_sq
    loss = 1 - eious
    return loss


@MODELS.register_module()
class IoULoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between predicted and target bboxes.

    Args:
        linear (bool): If True, use linear IoU loss, else use log IoU loss.
        eps (float): A value for numerical stability.
        reduction (str): The reduction method. Options are "none", "mean"
            and "sum".
        loss_weight (float): Weight of the loss.
    """

    def __init__(
        self,
        linear: bool = False,
        eps: float = 1e-6,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.eps = eps
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
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        loss = self.loss_weight * iou_loss(
            pred,
            target,
            weight,
            linear=self.linear,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs,
        )
        return loss


@MODELS.register_module()
class BoundedIoULoss(nn.Module):
    """Bounded IoU Loss.

    Args:
        beta (float): Beta parameter for smooth L1 loss.
        eps (float): A value for numerical stability.
        reduction (str): The reduction method.
        loss_weight (float): Weight of the loss.
    """

    def __init__(
        self,
        beta: float = 0.2,
        eps: float = 1e-3,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.eps = eps
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
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * bounded_iou_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs,
        )
        return loss


@MODELS.register_module()
class GIoULoss(nn.Module):
    """Generalized IoU Loss.

    Args:
        eps (float): A value for numerical stability.
        reduction (str): The reduction method.
        loss_weight (float): Weight of the loss.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.eps = eps
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
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        loss = self.loss_weight * giou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs,
        )
        return loss


@MODELS.register_module()
class DIoULoss(nn.Module):
    """Distance-IoU Loss.

    Args:
        eps (float): A value for numerical stability.
        reduction (str): The reduction method.
        loss_weight (float): Weight of the loss.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.eps = eps
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
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        loss = self.loss_weight * diou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs,
        )
        return loss


@MODELS.register_module()
class CIoULoss(nn.Module):
    """Complete-IoU Loss.

    Args:
        eps (float): A value for numerical stability.
        reduction (str): The reduction method.
        loss_weight (float): Weight of the loss.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.eps = eps
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
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        loss = self.loss_weight * ciou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs,
        )
        return loss


@MODELS.register_module()
class EIoULoss(nn.Module):
    """Efficient-IoU Loss.

    Args:
        eps (float): A value for numerical stability.
        reduction (str): The reduction method.
        loss_weight (float): Weight of the loss.
        smooth (bool): Whether to use smooth L1 style loss.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        smooth: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.smooth = smooth

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Tensor | None = None,
        avg_factor: float | None = None,
        reduction_override: str | None = None,
        **kwargs,
    ) -> Tensor:
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        loss = self.loss_weight * eiou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            smooth=self.smooth,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs,
        )
        return loss
