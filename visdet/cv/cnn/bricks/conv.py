# ruff: noqa
"""
Convolutional modules and utilities.

This module provides convolutional building blocks.
"""

from typing import Dict, Optional, Tuple, Union
import torch.nn as nn

from .norm import build_norm_layer


class ConvModule(nn.Module):
    """Convolutional module with optional normalization and activation.

    A standard convolutional layer with optional batch normalization and activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        norm_cfg: Optional[Dict] = None,
        act_cfg: Optional[Dict] = None,
        inplace: bool = False,
        **kwargs,
    ) -> None:
        """Initialize ConvModule.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolutional kernel
            stride: Stride of convolution
            padding: Padding for convolution
            dilation: Dilation rate
            groups: Number of groups for grouped convolution
            bias: Whether to use bias
            norm_cfg: Config dict for normalization layer
            act_cfg: Config dict for activation layer
            inplace: Whether to use inplace operations
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.with_bias = bias
        self.with_norm = norm_cfg is not None
        self.with_act = act_cfg is not None
        self.inplace = inplace

        # Build conv layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )

        # Build norm layer
        if self.with_norm:
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_channels)
            self.add_module(norm_name if norm_name else "bn", norm_layer)

        # Build activation layer
        if self.with_act:
            if isinstance(act_cfg, dict):
                act_cfg = act_cfg.copy()
                act_type = act_cfg.pop("type", "ReLU")
                if act_type == "ReLU":
                    self.activate = nn.ReLU(inplace=inplace)
                elif act_type == "GELU":
                    self.activate = nn.GELU()
                elif act_type == "SiLU":
                    self.activate = nn.SiLU(inplace=inplace)
                else:
                    self.activate = nn.ReLU(inplace=inplace)
            else:
                self.activate = nn.ReLU(inplace=inplace)
        else:
            self.activate = None

    def forward(self, x):
        """Forward pass."""
        x = self.conv(x)
        if self.with_norm:
            # Apply norm layer
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    x = m(x)
                    break
        if self.with_act:
            x = self.activate(x)
        return x


__all__ = ["ConvModule"]
