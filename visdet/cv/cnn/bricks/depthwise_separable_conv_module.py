# Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

import torch.nn as nn

from visdet.cv.cnn.bricks.conv_module import ConvModule


class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise separable convolution module.

    This is a lightweight alternative to a standard ConvModule, commonly used
    in mobile/efficient backbones and heads (e.g. YOLOX/RTMDet with
    ``use_depthwise=True``).

    The module is implemented as:
    1) depthwise ConvModule (groups=in_channels)
    2) pointwise ConvModule (1x1)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool | str = "auto",
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = dict(type="ReLU"),
        dw_norm_cfg: dict | None = None,
        pw_norm_cfg: dict | None = None,
        dw_act_cfg: dict | None = None,
        pw_act_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if dw_norm_cfg is None:
            dw_norm_cfg = norm_cfg
        if pw_norm_cfg is None:
            pw_norm_cfg = norm_cfg
        if dw_act_cfg is None:
            dw_act_cfg = act_cfg
        if pw_act_cfg is None:
            pw_act_cfg = act_cfg

        self.depthwise = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            conv_cfg=conv_cfg,
            norm_cfg=dw_norm_cfg,
            act_cfg=dw_act_cfg,
            **kwargs,
        )
        self.pointwise = ConvModule(
            in_channels,
            out_channels,
            1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias,
            conv_cfg=conv_cfg,
            norm_cfg=pw_norm_cfg,
            act_cfg=pw_act_cfg,
            **kwargs,
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
