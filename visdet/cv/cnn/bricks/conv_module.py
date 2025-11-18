from __future__ import annotations

# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.instancenorm import _InstanceNorm

from visdet.cv.cnn.bricks.activation import build_activation_layer
from visdet.cv.cnn.bricks.conv import build_conv_layer
from visdet.cv.cnn.bricks.norm import build_norm_layer
from visdet.cv.cnn.bricks.padding import build_padding_layer
from visdet.engine.model import constant_init, kaiming_init
from visdet.engine.registry import MODELS

EfficientConvBnEvalForward = Callable[[_BatchNorm, _ConvNd, Tensor], Tensor]


def efficient_conv_bn_eval_forward(bn: _BatchNorm, conv: _ConvNd, x: Tensor) -> Tensor:
    """
    Implementation based on https://arxiv.org/abs/2305.11624
    "Tune-Mode ConvBN Blocks For Efficient Transfer Learning"
    It leverages the associative law between convolution and affine transform,
    i.e., normalize (weight conv feature) = (normalize weight) conv feature.
    It works for Eval mode of ConvBN blocks during validation, and can be used
    for training as well. It reduces memory and computation cost.

    Args:
        bn (_BatchNorm): a BatchNorm module.
        conv (nn._ConvNd): a conv module
        x (torch.Tensor): Input feature map.
    """
    # These lines of code are designed to deal with various cases
    # like bn without affine transform, and conv without bias
    running_var = bn.running_var
    running_mean = bn.running_mean
    if running_var is None or running_mean is None:
        msg = "BatchNorm running stats must exist when efficient_conv_bn_eval_forward is enabled"
        raise RuntimeError(msg)

    weight_on_the_fly = conv.weight
    if conv.bias is not None:
        bias_on_the_fly = conv.bias
    else:
        bias_on_the_fly = torch.zeros_like(running_var)

    if bn.weight is not None:
        bn_weight = bn.weight
    else:
        bn_weight = torch.ones_like(running_var)

    if bn.bias is not None:
        bn_bias = bn.bias
    else:
        bn_bias = torch.zeros_like(running_var)

    # shape of [C_out, 1, 1, 1] in Conv2d
    weight_coeff = torch.rsqrt(running_var + bn.eps).reshape([-1] + [1] * (len(conv.weight.shape) - 1))
    # shape of [C_out, 1, 1, 1] in Conv2d
    coefff_on_the_fly = bn_weight.view_as(weight_coeff) * weight_coeff

    # shape of [C_out, C_in, k, k] in Conv2d
    weight_on_the_fly = weight_on_the_fly * coefff_on_the_fly
    # shape of [C_out] in Conv2d
    bias_on_the_fly = bn_bias + coefff_on_the_fly.flatten() * (bias_on_the_fly - running_mean)

    return conv._conv_forward(x, weight_on_the_fly, bias_on_the_fly)


@MODELS.register_module()
class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
        efficient_conv_bn_eval (bool): Whether use efficient conv when the
            consecutive bn is in eval mode (either training or testing), as
            proposed in https://arxiv.org/abs/2305.11624 . Default: `False`.
    """

    _abbr_ = "conv_block"
    conv_cfg: dict[str, Any] | None
    norm_cfg: dict[str, Any] | None
    act_cfg: dict[str, Any] | None
    order: tuple[str, str, str]
    padding_layer: nn.Module | None
    activate: nn.Module | None
    efficient_conv_bn_eval_forward: EfficientConvBnEvalForward | None
    norm_name: str | None
    conv: _ConvNd

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool | str = "auto",
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = dict(type="ReLU"),
        inplace: bool = True,
        with_spectral_norm: bool = False,
        padding_mode: str = "zeros",
        order: tuple = ("conv", "norm", "act"),
        efficient_conv_bn_eval: bool = False,
    ):
        super().__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ["zeros", "circular"]
        # Store config dicts as attributes - these are simple data, not tensors
        object.__setattr__(self, "conv_cfg", conv_cfg)
        object.__setattr__(self, "norm_cfg", norm_cfg)
        object.__setattr__(self, "act_cfg", act_cfg)
        object.__setattr__(self, "inplace", inplace)
        object.__setattr__(self, "with_spectral_norm", with_spectral_norm)
        object.__setattr__(self, "with_explicit_padding", padding_mode not in official_padding_mode)
        object.__setattr__(self, "order", order)
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == {"conv", "norm", "act"}

        object.__setattr__(self, "with_norm", norm_cfg is not None)
        object.__setattr__(self, "with_activation", act_cfg is not None)
        self.padding_layer: nn.Module | None = None
        self.activate: nn.Module | None = None
        object.__setattr__(self, "efficient_conv_bn_eval_forward", None)
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = not self.with_norm
        object.__setattr__(self, "with_bias", bias)

        if self.with_explicit_padding:
            pad_cfg = dict(type=padding_mode)
            self.padding_layer = build_padding_layer(pad_cfg, padding)

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        conv_layer = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.conv = cast(_ConvNd, conv_layer)
        # export the attributes of self.conv to a higher level for convenience
        object.__setattr__(self, "in_channels", self.conv.in_channels)
        object.__setattr__(self, "out_channels", self.conv.out_channels)
        object.__setattr__(self, "kernel_size", self.conv.kernel_size)
        object.__setattr__(self, "stride", self.conv.stride)
        object.__setattr__(self, "padding", padding)
        object.__setattr__(self, "dilation", self.conv.dilation)
        object.__setattr__(self, "transposed", self.conv.transposed)
        object.__setattr__(self, "output_padding", self.conv.output_padding)
        object.__setattr__(self, "groups", self.conv.groups)

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index("norm") > order.index("conv"):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            assert norm_cfg is not None
            norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            object.__setattr__(self, "norm_name", norm_name)
            self.add_module(norm_name, norm)
            if self.with_bias:
                if isinstance(norm, (_BatchNorm, _InstanceNorm)):
                    warnings.warn("Unnecessary conv bias before batch/instance norm")
        else:
            object.__setattr__(self, "norm_name", None)

        self.turn_on_efficient_conv_bn_eval(efficient_conv_bn_eval)

        # build activation layer
        if self.with_activation:
            assert act_cfg is not None
            act_cfg_ = cast(dict[str, Any], act_cfg.copy())
            # nn.Tanh has no 'inplace' argument
            if act_cfg_["type"] not in [
                "Tanh",
                "PReLU",
                "Sigmoid",
                "HSigmoid",
                "Swish",
                "GELU",
            ]:
                act_cfg_.setdefault("inplace", inplace)
            self.activate = build_activation_layer(act_cfg_)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self) -> nn.Module | None:
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, "init_weights"):
            if self.with_activation and self.act_cfg is not None and self.act_cfg["type"] == "LeakyReLU":
                nonlinearity = "leaky_relu"
                a = self.act_cfg.get("negative_slope", 0.01)
            else:
                nonlinearity = "relu"
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            norm_layer = self.norm
            if norm_layer is not None:
                constant_init(norm_layer, 1, bias=0)

    def forward(self, x: torch.Tensor, activate: bool = True, norm: bool = True) -> torch.Tensor:
        layer_index = 0
        while layer_index < len(self.order):
            layer = self.order[layer_index]
            if layer == "conv":
                if self.with_explicit_padding:
                    if self.padding_layer is None:
                        raise RuntimeError("Padding layer is not initialized")
                    x = self.padding_layer(x)
                # if the next operation is norm and we have a norm layer in
                # eval mode and we have enabled `efficient_conv_bn_eval` for
                # the conv operator, then activate the optimized forward and
                # skip the next norm operator since it has been fused
                norm_layer = self.norm
                if (
                    layer_index + 1 < len(self.order)
                    and self.order[layer_index + 1] == "norm"
                    and norm
                    and self.with_norm
                    and norm_layer is not None
                    and not norm_layer.training
                    and self.efficient_conv_bn_eval_forward is not None
                ):
                    bn_module = cast(_BatchNorm, norm_layer)
                    self.conv.forward = partial(self.efficient_conv_bn_eval_forward, bn_module, self.conv)  # type: ignore[method-assign]
                    layer_index += 1
                    x = self.conv(x)
                    del self.conv.forward  # type: ignore[attr-defined]
                else:
                    x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                norm_layer = self.norm
                if norm_layer is None:
                    raise RuntimeError("Norm layer not initialized")
                x = norm_layer(x)
            elif layer == "act" and activate and self.with_activation:
                if self.activate is None:
                    raise RuntimeError("Activation layer not initialized")
                x = self.activate(x)
            layer_index += 1
        return x

    def turn_on_efficient_conv_bn_eval(self, efficient_conv_bn_eval: bool = True) -> None:
        # efficient_conv_bn_eval works for conv + bn
        # with `track_running_stats` option
        norm_layer = self.norm
        if (
            efficient_conv_bn_eval
            and norm_layer is not None
            and isinstance(norm_layer, _BatchNorm)
            and norm_layer.track_running_stats
        ):
            object.__setattr__(self, "efficient_conv_bn_eval_forward", efficient_conv_bn_eval_forward)
        else:
            object.__setattr__(self, "efficient_conv_bn_eval_forward", None)

    @staticmethod
    def create_from_conv_bn(
        conv: _ConvNd,
        bn: _BatchNorm,
        efficient_conv_bn_eval: bool = True,
    ) -> "ConvModule":
        """Create a ConvModule from a conv and a bn module."""
        self = ConvModule.__new__(ConvModule)
        super(ConvModule, self).__init__()

        object.__setattr__(self, "conv_cfg", None)
        object.__setattr__(self, "norm_cfg", None)
        object.__setattr__(self, "act_cfg", None)
        object.__setattr__(self, "inplace", False)
        object.__setattr__(self, "with_spectral_norm", False)
        object.__setattr__(self, "with_explicit_padding", False)
        object.__setattr__(self, "order", ("conv", "norm", "act"))

        object.__setattr__(self, "with_norm", True)
        object.__setattr__(self, "with_activation", False)
        object.__setattr__(self, "with_bias", conv.bias is not None)

        # build convolution layer
        self.conv = conv
        # export the attributes of self.conv to a higher level for convenience
        object.__setattr__(self, "in_channels", self.conv.in_channels)
        object.__setattr__(self, "out_channels", self.conv.out_channels)
        object.__setattr__(self, "kernel_size", self.conv.kernel_size)
        object.__setattr__(self, "stride", self.conv.stride)
        object.__setattr__(self, "padding", self.conv.padding)
        object.__setattr__(self, "dilation", self.conv.dilation)
        object.__setattr__(self, "transposed", self.conv.transposed)
        object.__setattr__(self, "output_padding", self.conv.output_padding)
        object.__setattr__(self, "groups", self.conv.groups)

        # build normalization layers
        norm_name: str = "bn"
        object.__setattr__(self, "norm_name", norm_name)
        self.add_module(norm_name, bn)

        self.turn_on_efficient_conv_bn_eval(efficient_conv_bn_eval)

        return self
