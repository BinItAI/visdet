# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from visengine.registry import MODELS

# Import activation classes from their modules
from .hsigmoid import HSigmoid
from .hswish import HSwish
from .swish import Swish

for module in [
    nn.ReLU,
    nn.LeakyReLU,
    nn.PReLU,
    nn.RReLU,
    nn.ReLU6,
    nn.ELU,
    nn.Sigmoid,
    nn.Tanh,
]:
    MODELS.register_module(module=module)

# Register custom activation modules
MODELS.register_module(module=HSigmoid)
MODELS.register_module(module=HSwish)
MODELS.register_module(module=Swish)
MODELS.register_module(module=nn.SiLU, name="SiLU")
MODELS.register_module(module=nn.GELU)


@MODELS.register_module(name="Clip")
@MODELS.register_module()
class Clamp(nn.Module):
    """Clamp activation layer.

    This activation function is to clamp the feature map value within
    :math:`[min, max]`. More details can be found in ``torch.clamp()``.

    Args:
        min (Number | optional): Lower-bound of the range to be clamped to.
            Default to -1.
        max (Number | optional): Upper-bound of the range to be clamped to.
            Default to 1.
    """

    def __init__(self, min: float = -1.0, max: float = 1.0):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: Clamped tensor.
        """
        return torch.clamp(x, min=self.min, max=self.max)


class GELU(nn.Module):
    r"""Applies the Gaussian Error Linear Units function:

    .. math::
        \text{GELU}(x) = x * \Phi(x)
    where :math:`\Phi(x)` is the Cumulative Distribution Function for
    Gaussian Distribution.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/GELU.png

    Examples::

        >>> m = nn.GELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.gelu(input)


def build_activation_layer(cfg: dict) -> nn.Module:
    """Build activation layer.

    Args:
        cfg (dict): The activation layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    return MODELS.build(cfg)
