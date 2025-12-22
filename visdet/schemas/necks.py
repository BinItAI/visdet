"""Neck configuration schemas.

Necks connect backbone feature maps to detection heads.
FPN (Feature Pyramid Network) is the most commonly used neck.

Example:
    >>> from visdet.schemas.necks import FPNConfig
    >>> neck = FPNConfig(in_channels=[256, 512, 1024, 2048], out_channels=256)
"""

from typing import Annotated, Literal, Union

from pydantic import Field

from visdet.schemas.base import NeckConfig, OptionalConfig


class FPNConfig(NeckConfig):
    """Feature Pyramid Network (FPN) configuration.

    FPN creates a multi-scale feature pyramid from backbone outputs,
    enabling detection at multiple scales.

    Reference: "Feature Pyramid Networks for Object Detection"
    (https://arxiv.org/abs/1612.03144)

    Attributes:
        in_channels: Number of input channels per backbone level.
        out_channels: Number of output channels (same for all levels).
        num_outs: Number of output feature maps.

    Example:
        >>> # FPN for ResNet-50 backbone
        >>> cfg = FPNConfig(
        ...     in_channels=[256, 512, 1024, 2048],
        ...     out_channels=256,
        ...     num_outs=5
        ... )
        >>> # FPN for Swin-Tiny backbone
        >>> cfg = FPNConfig(
        ...     in_channels=[96, 192, 384, 768],
        ...     out_channels=256,
        ...     num_outs=5
        ... )
    """

    type: Literal["FPN"] = "FPN"

    # Channel configuration
    in_channels: list[int] = Field(
        ...,
        min_length=1,
        description="Number of input channels per backbone level",
    )
    out_channels: int = Field(
        ...,
        gt=0,
        description="Number of output channels (used at each scale)",
    )
    num_outs: int = Field(
        ...,
        ge=1,
        description="Number of output feature maps (typically 5 for detection)",
    )

    # Level selection
    start_level: int = Field(
        default=0,
        ge=0,
        description="Index of the start input backbone level",
    )
    end_level: int = Field(
        default=-1,
        description="End input backbone level index (-1 = last level)",
    )

    # Extra convolutions
    add_extra_convs: bool | Literal["on_input", "on_lateral", "on_output"] = Field(
        default=False,
        description="Add conv layers on top of original feature maps",
    )
    relu_before_extra_convs: bool = Field(
        default=False,
        description="Apply ReLU before extra convolutions",
    )

    # Normalization
    no_norm_on_lateral: bool = Field(
        default=False,
        description="Skip normalization on lateral connections",
    )

    # Layer configurations
    conv_cfg: OptionalConfig = Field(
        default=None,
        description="Convolution layer config",
    )
    norm_cfg: OptionalConfig = Field(
        default=None,
        description="Normalization layer config",
    )
    act_cfg: OptionalConfig = Field(
        default=None,
        description="Activation layer config",
    )
    upsample_cfg: OptionalConfig = Field(
        default_factory=lambda: {"mode": "nearest"},
        description="Upsampling layer config",
    )

    # Initialization
    init_cfg: OptionalConfig = Field(
        default_factory=lambda: {
            "type": "Xavier",
            "layer": "Conv2d",
            "distribution": "uniform",
        },
        description="Weight initialization config",
    )


# Type alias for neck configurations
NeckType = Annotated[
    Union[FPNConfig],
    Field(discriminator="type"),
]
"""Type alias for any neck configuration."""
