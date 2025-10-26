# ruff: noqa
"""
Builder functions for CNN components.

This module provides functions to build standard CNN layers.
"""

from typing import Dict, Optional, Tuple, Union
import torch.nn as nn

from .conv import ConvModule


def build_conv_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    """Build a conv layer.

    Args:
        cfg: Config dict for the conv layer
        *args: Positional arguments passed to the conv layer
        **kwargs: Keyword arguments passed to the conv layer

    Returns:
        Built convolutional layer
    """
    if cfg is None:
        return nn.Conv2d(*args, **kwargs)

    cfg = cfg.copy()
    layer_type = cfg.pop("type", "Conv2d")

    if layer_type == "Conv2d":
        return nn.Conv2d(*args, **kwargs)
    elif layer_type == "ConvModule":
        return ConvModule(*args, **kwargs, **cfg)
    else:
        return nn.Conv2d(*args, **kwargs)


def build_upsample_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    """Build an upsample layer.

    Args:
        cfg: Config dict for the upsample layer
        *args: Positional arguments passed to the upsample layer
        **kwargs: Keyword arguments passed to the upsample layer

    Returns:
        Built upsampling layer
    """
    if cfg is None:
        return nn.Upsample(*args, **kwargs)

    cfg = cfg.copy()
    layer_type = cfg.pop("type", "nearest")

    if isinstance(layer_type, str):
        return nn.Upsample(mode=layer_type, *args, **kwargs)
    else:
        return nn.Upsample(*args, **kwargs)


__all__ = ["build_conv_layer", "build_upsample_layer"]
