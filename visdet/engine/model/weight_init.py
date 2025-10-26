# ruff: noqa
"""
Weight initialization utilities for visdet.

This module provides functions for initializing network weights.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    """Initialize module parameters with constant values.

    Args:
        module: Module to initialize
        val: Constant value for weight initialization
        bias: Constant value for bias initialization
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def trunc_normal_(tensor: torch.Tensor, mean: float = 0, std: float = 1, a: float = -2, b: float = 2) -> None:
    """Truncated normal initialization.

    Args:
        tensor: Tensor to initialize
        mean: Mean of distribution
        std: Standard deviation of distribution
        a: Lower truncation bound
        b: Upper truncation bound
    """
    with torch.no_grad():
        # Calculate uniform bounds corresponding to truncated normal
        normal = torch.normal(mean, std, size=tensor.shape)
        # Clip to bounds
        normal = torch.clamp(normal, a, b)
        # Renormalize to ensure desired distribution
        scale = (b - a) / 4  # Approximate scale factor
        tensor.copy_(normal * scale)


def trunc_normal_init(module: nn.Module, mean: float = 0, std: float = 1) -> None:
    """Apply truncated normal initialization to a module.

    Args:
        module: Module to initialize
        mean: Mean of distribution
        std: Standard deviation of distribution
    """
    if hasattr(module, "weight") and module.weight is not None:
        trunc_normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, 0)


__all__ = ["constant_init", "trunc_normal_", "trunc_normal_init"]
