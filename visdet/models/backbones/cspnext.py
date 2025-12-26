# Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

import math
from collections.abc import Sequence

import torch.nn as nn
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from visdet.cv.cnn import ConvModule
from visdet.cv.cnn.bricks.depthwise_separable_conv_module import DepthwiseSeparableConvModule
from visdet.engine.model import BaseModule
from visdet.models.layers import CSPLayer, SPPBottleneck
from visdet.registry import MODELS
from visdet.utils.typing_utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class CSPNeXt(BaseModule):
    """CSPNeXt backbone used in RTMDet."""

    arch_settings = {
        "P5": [
            [64, 128, 3, True, False],
            [128, 256, 6, True, False],
            [256, 512, 6, True, False],
            [512, 1024, 3, False, True],
        ],
        "P6": [
            [64, 128, 3, True, False],
            [128, 256, 6, True, False],
            [256, 512, 6, True, False],
            [512, 768, 3, True, False],
            [768, 1024, 3, False, True],
        ],
    }

    def __init__(
        self,
        arch: str = "P5",
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        out_indices: Sequence[int] = (2, 3, 4),
        frozen_stages: int = -1,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        arch_ovewrite: list | None = None,
        spp_kernel_sizes: Sequence[int] = (5, 9, 13),
        channel_attention: bool = True,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type="SiLU"),
        norm_eval: bool = False,
        init_cfg: OptMultiConfig = dict(
            type="Kaiming",
            layer="Conv2d",
            a=math.sqrt(5),
            distribution="uniform",
            mode="fan_in",
            nonlinearity="leaky_relu",
        ),
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        if not set(out_indices).issubset(i for i in range(len(arch_setting) + 1)):
            raise ValueError("out_indices must be within valid stage indices")
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError(f"frozen_stages must be in range(-1, len(arch_setting) + 1). But received {frozen_stages}")

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        stem_channels = int(arch_setting[0][0] * widen_factor // 2)
        self.stem = nn.Sequential(
            ConvModule(
                3,
                stem_channels,
                3,
                padding=1,
                stride=2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            ConvModule(
                stem_channels,
                stem_channels,
                3,
                padding=1,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            ConvModule(
                stem_channels,
                int(arch_setting[0][0] * widen_factor),
                3,
                padding=1,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
        )
        self.layers = ["stem"]

        for i, (in_channels, out_channels, num_blocks, add_identity, use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)

            stage: list[nn.Module] = []
            stage.append(
                conv(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )
            if use_spp:
                stage.append(
                    SPPBottleneck(
                        out_channels,
                        out_channels,
                        kernel_sizes=tuple(spp_kernel_sizes),
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                    )
                )
            stage.append(
                CSPLayer(
                    out_channels,
                    out_channels,
                    num_blocks=num_blocks,
                    add_identity=add_identity,
                    use_depthwise=use_depthwise,
                    use_cspnext_block=True,
                    expand_ratio=expand_ratio,
                    channel_attention=channel_attention,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )
            self.add_module(f"stage{i + 1}", nn.Sequential(*stage))
            self.layers.append(f"stage{i + 1}")

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                module = getattr(self, self.layers[i])
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for module in self.modules():
                if isinstance(module, _BatchNorm):
                    module.eval()

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        outs: list[Tensor] = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
