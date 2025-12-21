# Copyright (c) OpenMMLab. All rights reserved.
"""RetinaNet dense head implementation."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from visdet.models.dense_heads.anchor_head import AnchorHead
from visdet.registry import MODELS
from visdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class RetinaHead(AnchorHead):
    """Anchor-based dense head used in RetinaNet.

    The head is composed of two subnetworks: one for classification and one
    for bounding box regression. Each subnet is built from several stacked
    convolution blocks followed by the task-specific prediction conv.

    Args:
        num_classes (int): Number of categories without the background class.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv blocks per subnet. Defaults to 4.
        conv_cfg (dict, optional): Conv config. Only ``type='Conv2d'`` is
            supported in visdet. Defaults to None.
        norm_cfg (dict, optional): Normalization config. Supports ``BN`` and
            ``GN``. Defaults to ``None``.
        anchor_generator (dict): Anchor generator config.
        init_cfg (dict | list[dict], optional): Initialization config.
        **kwargs: Extra keyword args passed to :class:`AnchorHead`.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        stacked_convs: int = 4,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        anchor_generator: ConfigType = dict(
            type="AnchorGenerator",
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128],
        ),
        init_cfg: OptMultiConfig = dict(
            type="Normal",
            layer="Conv2d",
            std=0.01,
            override=dict(type="Normal", name="retina_cls", std=0.01, bias_prob=0.01),
        ),
        **kwargs,
    ) -> None:
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs,
        )

    def _init_layers(self) -> None:
        """Initialize subnetworks and predictors."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            in_ch = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(self._build_conv_block(in_ch))
            self.reg_convs.append(self._build_conv_block(in_ch))

        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            kernel_size=3,
            padding=1,
        )
        self.retina_reg = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.bbox_coder.encode_size,
            kernel_size=3,
            padding=1,
        )

    def _build_conv_block(self, in_channels: int) -> nn.Sequential:
        """Build a single conv -> norm -> ReLU block."""
        norm_layer = self._build_norm(self.feat_channels)
        bias = norm_layer is None
        if self.conv_cfg is not None and self.conv_cfg.get("type", "Conv2d") != "Conv2d":
            raise NotImplementedError("RetinaHead only supports standard Conv2d layers in visdet.")
        conv = nn.Conv2d(
            in_channels,
            self.feat_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        layers = [conv]
        if norm_layer is not None:
            layers.append(norm_layer)
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _build_norm(self, num_channels: int) -> nn.Module | None:
        """Create the configured normalization layer."""
        if self.norm_cfg is None:
            return None
        norm_type = self.norm_cfg.get("type", "BN")
        requires_grad = self.norm_cfg.get("requires_grad", True)
        if norm_type in {"BN", "SyncBN"}:
            layer = nn.BatchNorm2d(num_channels)
        elif norm_type == "GN":
            num_groups = self.norm_cfg.get("num_groups", 32)
            layer = nn.GroupNorm(num_groups, num_channels)
        else:
            raise ValueError(f"Unsupported norm type '{norm_type}' in RetinaHead.")
        for param in layer.parameters():
            param.requires_grad = requires_grad
        return layer

    def forward_single(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward feature map of a single level.

        Args:
            x (Tensor): Feature map of shape (N, C, H, W).

        Returns:
            tuple[Tensor, Tensor]: Classification scores and bbox predictions.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred
