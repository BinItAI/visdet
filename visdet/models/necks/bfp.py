# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from visdet.cv.cnn import ConvModule
from visdet.registry import MODELS


@MODELS.register_module()
class BFP(nn.Module):
    """BFP (Balanced Feature Pyramids)

    Args:
        in_channels (int): Number of input channels.
        num_levels (int): Number of input feature levels.
        refine_level (int): Index of integration and refine level.
        refine_type (str): Type of the refine op, [None, 'conv'].
            'non_local' is not implemented yet.
    """

    def __init__(
        self,
        in_channels: int,
        num_levels: int,
        refine_level: int = 2,
        refine_type: str | None = None,
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
    ) -> None:
        super(BFP, self).__init__()
        assert refine_type in [None, "conv"]

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.refine_level = refine_level
        self.refine_type = refine_type
        assert 0 <= self.refine_level < self.num_levels

        if self.refine_type == "conv":
            self.refine = ConvModule(
                self.in_channels,
                self.in_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
            )

    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """Forward function."""
        assert len(inputs) == self.num_levels

        # step 1: gather multi-level features by resize and average
        feats: list[Tensor] = []
        gather_size = inputs[self.refine_level].size()[2:]
        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(inputs[i], size=gather_size, mode="nearest")
            feats.append(gathered)

        bsf = torch.stack(feats, dim=0).mean(dim=0)

        # step 2: refine gathered features
        if self.refine_type is not None:
            bsf = self.refine(bsf)

        # step 3: scatter refined features to multi-levels by a residual path
        outs: list[Tensor] = []
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode="nearest")
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
            outs.append(residual + inputs[i])

        return tuple(outs)
