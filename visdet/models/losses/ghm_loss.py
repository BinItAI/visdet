# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from visdet.registry import MODELS


@MODELS.register_module()
class GHMC(nn.Module):
    """GHM Classification Loss.

    Gradient Harmonizing Mechanism (GHM) for classification loss. GHM reduces
    the contribution of easy examples that dominate the gradient and lets the
    model focus more on the hard examples.

    Paper: Gradient Harmonized Single-stage Detector
    https://arxiv.org/abs/1811.05181

    Args:
        bins (int): Number of gradient density bins. Defaults to 10.
        momentum (float): Momentum for updating the gradient density.
            0 means no momentum update. Defaults to 0.
        use_sigmoid (bool): Whether to use sigmoid for classification.
            Only True is supported. Defaults to True.
        loss_weight (float): Weight of the loss. Defaults to 1.0.
    """

    def __init__(
        self,
        bins: int = 10,
        momentum: float = 0,
        use_sigmoid: bool = True,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.bins = bins
        self.momentum = momentum
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight

        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer("edges", edges)
        self.edges[-1] += 1e-6

        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer("acc_sum", acc_sum)

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        label_weight: Tensor | None = None,
        reduction_override: str | None = None,
        **kwargs,
    ) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted classification logits with shape (N, C).
            target (Tensor): Target labels with shape (N, C).
            label_weight (Tensor, optional): Weight of each sample.
            reduction_override (str, optional): Override the reduction method.

        Returns:
            Tensor: The calculated GHM classification loss.
        """
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # Gradient
        g = torch.abs(pred.detach().sigmoid() - target)

        valid = label_weight > 0 if label_weight is not None else target >= 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # Total number of non-empty bins

        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1

        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(pred, target, weights, reduction="sum") / tot * self.loss_weight
        return loss


@MODELS.register_module()
class GHMR(nn.Module):
    """GHM Regression Loss.

    Gradient Harmonizing Mechanism (GHM) for regression loss.

    Args:
        mu (float): The minimum value for the loss. Defaults to 0.02.
        bins (int): Number of gradient density bins. Defaults to 10.
        momentum (float): Momentum for updating the gradient density.
            Defaults to 0.
        loss_weight (float): Weight of the loss. Defaults to 1.0.
    """

    def __init__(
        self,
        mu: float = 0.02,
        bins: int = 10,
        momentum: float = 0,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.mu = mu
        self.bins = bins
        self.momentum = momentum
        self.loss_weight = loss_weight

        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer("edges", edges)
        self.edges[-1] = 1e3

        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer("acc_sum", acc_sum)

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        label_weight: Tensor | None = None,
        avg_factor: float | None = None,
        reduction_override: str | None = None,
    ) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted regression values with shape (N, 4).
            target (Tensor): Target regression values with shape (N, 4).
            label_weight (Tensor, optional): Weight of each sample.
            avg_factor (float, optional): Average factor for the loss.
            reduction_override (str, optional): Override the reduction method.

        Returns:
            Tensor: The calculated GHM regression loss.
        """
        mu = self.mu
        edges = self.edges
        mmt = self.momentum

        # ASL1 loss
        diff = pred - target
        loss = torch.sqrt(diff * diff + mu * mu) - mu

        # Gradient
        g = torch.abs(diff / torch.sqrt(mu * mu + diff * diff)).detach()

        weights = torch.zeros_like(g)

        valid = label_weight > 0 if label_weight is not None else torch.ones_like(g, dtype=torch.bool)
        tot = max(valid.float().sum().item(), 1.0)
        n = 0

        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                n += 1
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin

        if n > 0:
            weights /= n

        loss = loss * weights
        loss = loss.sum() / tot
        return loss * self.loss_weight
