# Copyright (c) OpenMMLab. All rights reserved.
import logging
import warnings
from typing import Any

import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm

from visdet.cv.cnn import build_conv_layer, build_norm_layer
from visdet.engine.model import BaseModule
from visdet.models.utils import ResLayer
from visdet.registry import MODELS

logger = logging.getLogger(__name__)


class BasicBlock(BaseModule):
    """Basic block for ResNet.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int): Stride of the first block. Default: 1
        dilation (int): Dilation of the first block. Default: 1
        downsample (nn.Module): Downsample layer. Default: None
        style (str): Style of ResNet. Default: 'pytorch'
        with_cp (bool): Use checkpoint or not. Default: False
        conv_cfg (dict): Dictionary to construct conv layer. Default: None
        norm_cfg (dict): Dictionary to construct norm layer. Default: dict(type='BN')
        dcn (dict): Dictionary to construct dcn layer. Not supported. Default: None
        plugins (list[dict]): List of plugins. Not supported. Default: None
        init_cfg (dict or list[dict]): Initialization config. Default: None
    """

    expansion = 1

    norm1_name: str
    norm2_name: str
    conv1: nn.Module
    conv2: nn.Module
    relu: nn.ReLU
    downsample: nn.Module | None
    stride: int
    dilation: int
    with_cp: bool

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        style="pytorch",
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        dcn=None,
        plugins=None,
        init_cfg=None,
    ):
        super(BasicBlock, self).__init__(init_cfg=init_cfg)
        assert dcn is None, "DCN is not supported in BasicBlock"
        assert plugins is None, "Plugins are not supported yet"

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)  # type: ignore[unresolved-attribute]
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)  # type: ignore[unresolved-attribute]

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # type: ignore[unresolved-attribute]
        self.stride = stride  # type: ignore[unresolved-attribute]
        self.dilation = dilation  # type: ignore[unresolved-attribute]
        self.with_cp = with_cp  # type: ignore[unresolved-attribute]

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    """Bottleneck block for ResNet.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int): Stride of the first block. Default: 1
        dilation (int): Dilation of the second block. Default: 1
        downsample (nn.Module): Downsample layer. Default: None
        style (str): Style of ResNet ('pytorch' or 'caffe'). Default: 'pytorch'
        with_cp (bool): Use checkpoint or not. Default: False
        conv_cfg (dict): Dictionary to construct conv layer. Default: None
        norm_cfg (dict): Dictionary to construct norm layer. Default: dict(type='BN')
        dcn (dict): Dictionary to construct dcn layer. Default: None
        plugins (list[dict]): List of plugins. Not supported. Default: None
        init_cfg (dict or list[dict]): Initialization config. Default: None
    """

    expansion = 4

    inplanes: int
    planes: int
    stride: int
    dilation: int
    style: str
    with_cp: bool
    conv_cfg: dict[str, Any] | None
    norm_cfg: dict[str, Any]
    dcn: dict[str, Any] | None
    with_dcn: bool
    conv1_stride: int
    conv2_stride: int
    norm1_name: str
    norm2_name: str
    norm3_name: str
    conv1: nn.Module
    conv2: nn.Module
    conv3: nn.Module
    relu: nn.ReLU
    downsample: nn.Module | None

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        style="pytorch",
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        dcn=None,
        plugins=None,
        init_cfg=None,
    ):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(init_cfg=init_cfg)
        assert style in ["pytorch", "caffe"]
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None, "Plugins are not supported yet"

        self.inplanes = inplanes  # type: ignore[unresolved-attribute]
        self.planes = planes  # type: ignore[unresolved-attribute]
        self.stride = stride  # type: ignore[unresolved-attribute]
        self.dilation = dilation  # type: ignore[unresolved-attribute]
        self.style = style  # type: ignore[unresolved-attribute]
        self.with_cp = with_cp  # type: ignore[unresolved-attribute]
        self.conv_cfg = conv_cfg  # type: ignore[unresolved-attribute]
        self.norm_cfg = norm_cfg  # type: ignore[unresolved-attribute]
        self.dcn = dcn  # type: ignore[unresolved-attribute]
        self.with_dcn = dcn is not None  # type: ignore[unresolved-attribute]

        if self.style == "pytorch":
            self.conv1_stride = 1  # type: ignore[unresolved-attribute]
            self.conv2_stride = stride  # type: ignore[unresolved-attribute]
        else:
            self.conv1_stride = stride  # type: ignore[unresolved-attribute]
            self.conv2_stride = 1  # type: ignore[unresolved-attribute]

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)  # type: ignore[unresolved-attribute]
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)  # type: ignore[unresolved-attribute]
        self.norm3_name, norm3 = build_norm_layer(norm_cfg, planes * self.expansion, postfix=3)  # type: ignore[unresolved-attribute]

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False,
        )
        self.add_module(self.norm1_name, norm1)

        # Handle DCN: Use pure PyTorch fallback with warning
        if self.with_dcn:
            logger.warning(
                "Deformable convolution (DCN) requested but using standard convolution "
                "as pure PyTorch fallback. For full DCN support, consider using mmcv."
            )

        self.conv2 = build_conv_layer(
            conv_cfg,
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(conv_cfg, planes, planes * self.expansion, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # type: ignore[unresolved-attribute]

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@MODELS.register_module()
class ResNet(BaseModule):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Number of stem channels. If not specified,
            it will be the same as `base_channels`. Default: None.
        base_channels (int): Number of base channels of res layer. Default: 64.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: (1, 2, 2, 2)
        dilations (Sequence[int]): Dilation of each stage. Default: (1, 1, 1, 1)
        out_indices (Sequence[int]): Output from which stages. Default: (0, 1, 2, 3)
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default: 'pytorch'
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1
        conv_cfg (dict): Dictionary to construct conv layer. Default: None
        norm_cfg (dict): Dictionary to construct norm layer.
            Default: dict(type='BN', requires_grad=True)
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats. Only applicable to Batch Norm. Default: True
        with_cp (bool): Use checkpoint or not. Default: False
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks. Default: True
        pretrained (str, optional): model pretrained path. Deprecated. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Example:
        >>> from visdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=50)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 256, 8, 8)
        (1, 512, 4, 4)
        (1, 1024, 2, 2)
        (1, 2048, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    zero_init_residual: bool
    depth: int
    stem_channels: int
    base_channels: int
    num_stages: int
    strides: tuple[int, ...]
    dilations: tuple[int, ...]
    out_indices: tuple[int, ...]
    style: str
    deep_stem: bool
    avg_down: bool
    frozen_stages: int
    conv_cfg: dict[str, Any] | None
    norm_cfg: dict[str, Any]
    with_cp: bool
    norm_eval: bool
    dcn: dict[str, Any] | None
    stage_with_dcn: tuple[bool, ...]
    block: type[BasicBlock] | type[Bottleneck]
    stage_blocks: tuple[int, ...]
    inplanes: int
    res_layers: list[str]
    feat_dim: int
    stem: nn.Sequential
    conv1: nn.Module
    norm1_name: str
    relu: nn.ReLU
    maxpool: nn.MaxPool2d

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_channels=None,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(0, 1, 2, 3),
        style="pytorch",
        deep_stem=False,
        avg_down=False,
        frozen_stages=-1,
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        dcn=None,
        stage_with_dcn=(False, False, False, False),
        with_cp=False,
        zero_init_residual=True,
        pretrained=None,
        init_cfg=None,
    ):
        super(ResNet, self).__init__(init_cfg=init_cfg)
        self.zero_init_residual = zero_init_residual  # type: ignore[unresolved-attribute]
        if depth not in self.arch_settings:
            raise KeyError(f"invalid depth {depth} for resnet")

        block_init_cfg = None
        assert not (init_cfg and pretrained), "init_cfg and pretrained cannot be specified at the same time"
        if isinstance(pretrained, str):
            warnings.warn("DeprecationWarning: pretrained is deprecated, please use 'init_cfg' instead")
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)  # type: ignore[unresolved-attribute]
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [  # type: ignore[unresolved-attribute]
                    dict(type="Kaiming", layer="Conv2d"),
                    dict(type="Constant", val=1, layer=["_BatchNorm", "GroupNorm"]),
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(type="Constant", val=0, override=dict(name="norm2"))
                    elif block is Bottleneck:
                        block_init_cfg = dict(type="Constant", val=0, override=dict(name="norm3"))
        else:
            raise TypeError("pretrained must be a str or None")

        self.depth = depth  # type: ignore[unresolved-attribute]
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels  # type: ignore[unresolved-attribute]
        self.base_channels = base_channels  # type: ignore[unresolved-attribute]
        self.num_stages = num_stages  # type: ignore[unresolved-attribute]
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides  # type: ignore[unresolved-attribute]
        self.dilations = dilations  # type: ignore[unresolved-attribute]
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices  # type: ignore[unresolved-attribute]
        assert max(out_indices) < num_stages
        self.style = style  # type: ignore[unresolved-attribute]
        self.deep_stem = deep_stem  # type: ignore[unresolved-attribute]
        self.avg_down = avg_down  # type: ignore[unresolved-attribute]
        self.frozen_stages = frozen_stages  # type: ignore[unresolved-attribute]
        self.conv_cfg = conv_cfg  # type: ignore[unresolved-attribute]
        self.norm_cfg = norm_cfg  # type: ignore[unresolved-attribute]
        self.with_cp = with_cp  # type: ignore[unresolved-attribute]
        self.norm_eval = norm_eval  # type: ignore[unresolved-attribute]
        self.dcn = dcn  # type: ignore[unresolved-attribute]
        self.stage_with_dcn = stage_with_dcn  # type: ignore[unresolved-attribute]
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.block, stage_blocks = self.arch_settings[depth]  # type: ignore[unresolved-attribute]
        self.stage_blocks = stage_blocks[:num_stages]  # type: ignore[unresolved-attribute]
        self.inplanes = stem_channels  # type: ignore[unresolved-attribute]

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []  # type: ignore[unresolved-attribute]
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                init_cfg=block_init_cfg,
            )
            self.inplanes = planes * self.block.expansion  # type: ignore[unresolved-attribute]
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2 ** (len(self.stage_blocks) - 1)  # type: ignore[unresolved-attribute]

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, stem_channels, postfix=1)  # type: ignore[unresolved-attribute]
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"layer{i}")
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
