# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
from torch import Tensor

from visdet.engine.model import BaseModule
from visdet.engine.utils import digit_version


class ChannelAttention(BaseModule):
    """Channel attention Module.

    Args:
        channels (int): The input (and output) channels of the attention layer.
        init_cfg (dict | list[dict] | None): Initialization config dict.
            Defaults to None.
    """

    def __init__(self, channels: int, init_cfg: dict | list[dict] | None = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        if digit_version(torch.__version__) < digit_version("1.7.0"):
            self.act = nn.Hardsigmoid()
        else:
            self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out
