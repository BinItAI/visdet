# ruff: noqa
"""Normalization layers for visdet."""

from typing import Any, Optional, Tuple

import torch.nn as nn


def build_norm_layer(cfg: dict, num_features: int) -> Tuple[Optional[str], Optional[nn.Module]]:
    """Build normalization layer.

    Args:
        cfg: Config dict with keys:
            type: Type of norm layer (e.g., 'BN', 'LN', 'GN', 'SyncBN')
            **kwargs: Additional arguments for the layer
        num_features: Number of features (channels)

    Returns:
        Tuple of (layer_name, layer_module)
    """
    if cfg is None:
        return None, None

    cfg = cfg.copy()
    layer_type = cfg.pop("type", "BN")

    if layer_type == "BN":
        return "bn", nn.BatchNorm2d(num_features, **cfg)
    elif layer_type == "SyncBN":
        # For now, use regular BatchNorm
        return "bn", nn.BatchNorm2d(num_features, **cfg)
    elif layer_type == "GN":
        num_groups = cfg.pop("num_groups", 32)
        return "gn", nn.GroupNorm(num_groups, num_features, **cfg)
    elif layer_type == "LN":
        return "ln", nn.LayerNorm(num_features, **cfg)
    else:
        raise ValueError(f"Unsupported norm layer type: {layer_type}")


__all__ = ["build_norm_layer"]
