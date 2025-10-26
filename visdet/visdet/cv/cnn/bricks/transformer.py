# ruff: noqa
"""
Transformer and attention modules.

This module provides transformer layers and attention mechanisms.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class FFN(nn.Module):
    """Feed-Forward Network module.

    Simple MLP for transformer blocks.
    """

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: int,
        num_fcs: int = 2,
        act_cfg: Optional[Dict] = None,
        dropout_cfg: Optional[Dict] = None,
        add_identity: bool = False,
    ) -> None:
        """Initialize FFN module."""
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels

        layers = []
        in_channels = embed_dims
        for i in range(num_fcs - 1):
            layers.append(nn.Linear(in_channels, feedforward_channels))
            layers.append(nn.GELU())
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))

        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.layers(x)
        if self.add_identity:
            out = out + x
        return out


def build_dropout(dropout_cfg: Optional[Dict] = None) -> Optional[nn.Module]:
    """Build dropout layer.

    Args:
        dropout_cfg: Config dict with 'type' and other parameters

    Returns:
        Dropout module or None
    """
    if dropout_cfg is None or dropout_cfg.get("type") == "None":
        return None

    if isinstance(dropout_cfg, dict):
        dropout_cfg = dropout_cfg.copy()
        dropout_type = dropout_cfg.pop("type", "Dropout")
        p = dropout_cfg.pop("p", dropout_cfg.pop("drop_prob", 0.0))

        if dropout_type == "Dropout":
            return nn.Dropout(p)
        elif dropout_type == "DropPath":
            return nn.Dropout(p)  # Fallback to Dropout
        else:
            return nn.Dropout(p)
    elif isinstance(dropout_cfg, float):
        return nn.Dropout(dropout_cfg)

    return None


__all__ = ["FFN", "build_dropout"]
