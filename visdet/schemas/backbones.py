"""Backbone network configuration schemas.

This module provides Pydantic schemas for backbone networks with full
IDE autocomplete and validation support.

Example:
    >>> from visdet.schemas.backbones import SwinTransformerConfig
    >>> backbone = SwinTransformerConfig(embed_dims=128, depths=(2, 2, 18, 2))
    >>> backbone.to_dict()
"""

from typing import Annotated, Literal, Optional, Union

from pydantic import Field

from visdet.schemas.base import BackboneConfig, OptionalConfig


class SwinTransformerConfig(BackboneConfig):
    """Swin Transformer backbone configuration.

    A PyTorch implementation of "Swin Transformer: Hierarchical Vision
    Transformer using Shifted Windows" (https://arxiv.org/abs/2103.14030).

    Common presets:
    - Swin-Tiny: embed_dims=96, depths=(2,2,6,2), num_heads=(3,6,12,24)
    - Swin-Small: embed_dims=96, depths=(2,2,18,2), num_heads=(3,6,12,24)
    - Swin-Base: embed_dims=128, depths=(2,2,18,2), num_heads=(4,8,16,32)
    - Swin-Large: embed_dims=192, depths=(2,2,18,2), num_heads=(6,12,24,48)

    Attributes:
        embed_dims: Feature embedding dimension.
        depths: Number of blocks at each stage.
        num_heads: Number of attention heads at each stage.
        window_size: Window size for local attention.
        drop_path_rate: Stochastic depth rate.

    Example:
        >>> # Swin-Tiny configuration
        >>> cfg = SwinTransformerConfig(
        ...     embed_dims=96,
        ...     depths=(2, 2, 6, 2),
        ...     num_heads=(3, 6, 12, 24),
        ... )
    """

    type: Literal["SwinTransformer"] = "SwinTransformer"

    # Image and patch settings
    pretrain_img_size: int | tuple[int, int] = Field(
        default=224, description="Input image size for pretraining"
    )
    in_channels: int = Field(default=3, ge=1, description="Number of input channels")
    patch_size: int | tuple[int, int] = Field(default=4, ge=1, description="Patch size")

    # Architecture settings
    embed_dims: int = Field(
        default=96, gt=0, description="Feature embedding dimension (96=Tiny, 128=Base, 192=Large)"
    )
    depths: tuple[int, ...] = Field(
        default=(2, 2, 6, 2),
        min_length=1,
        description="Number of transformer blocks at each stage",
    )
    num_heads: tuple[int, ...] = Field(
        default=(3, 6, 12, 24),
        min_length=1,
        description="Number of attention heads at each stage",
    )
    window_size: int = Field(default=7, gt=0, description="Window size for local attention")
    mlp_ratio: int = Field(default=4, gt=0, description="Ratio of MLP hidden dim to embedding dim")

    # Stride and output settings
    strides: tuple[int, ...] = Field(
        default=(4, 2, 2, 2), description="Patch merging stride at each stage"
    )
    out_indices: tuple[int, ...] = Field(
        default=(0, 1, 2, 3), description="Output feature map indices"
    )

    # Attention settings
    qkv_bias: bool = Field(default=True, description="Add learnable bias to Q, K, V")
    qk_scale: Optional[float] = Field(default=None, description="Override default qk scale")
    patch_norm: bool = Field(default=True, description="Apply normalization to patch embedding")

    # Regularization
    drop_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Dropout rate")
    attn_drop_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Attention dropout rate")
    drop_path_rate: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Stochastic depth rate"
    )

    # Position embedding
    use_abs_pos_embed: bool = Field(
        default=False, description="Use absolute position embedding"
    )

    # Layer configurations
    act_cfg: OptionalConfig = Field(
        default_factory=lambda: {"type": "GELU"}, description="Activation layer config"
    )
    norm_cfg: OptionalConfig = Field(
        default_factory=lambda: {"type": "LN"}, description="Normalization layer config"
    )

    # Training settings
    with_cp: bool = Field(
        default=False, description="Use gradient checkpointing to save memory"
    )
    attn_backend: Literal["torch", "flash"] = Field(
        default="torch", description="Attention backend"
    )

    # Pretrained weights
    pretrained: Optional[str] = Field(default=None, description="Pretrained checkpoint path")
    convert_weights: bool = Field(
        default=False, description="Convert weights from official repo format"
    )

    # Freezing
    frozen_stages: int = Field(
        default=-1, ge=-1, description="Freeze stages up to this index (-1 = none)"
    )

    # Initialization
    init_cfg: OptionalConfig = Field(default=None, description="Weight initialization config")


class ResNetConfig(BackboneConfig):
    """ResNet backbone configuration.

    Standard ResNet architecture with support for ResNet-18/34/50/101/152.

    Common presets:
    - ResNet-50: depth=50 (most common for detection)
    - ResNet-101: depth=101 (better accuracy, more compute)

    Attributes:
        depth: ResNet depth (18, 34, 50, 101, or 152).
        out_indices: Which stages to output features from.
        frozen_stages: Freeze parameters up to this stage.
        norm_eval: Freeze BatchNorm running stats.

    Example:
        >>> cfg = ResNetConfig(depth=50, frozen_stages=1)
    """

    type: Literal["ResNet"] = "ResNet"

    # Architecture
    depth: Literal[18, 34, 50, 101, 152] = Field(
        ..., description="ResNet depth (18, 34, 50, 101, or 152)"
    )
    in_channels: int = Field(default=3, ge=1, description="Number of input channels")
    stem_channels: Optional[int] = Field(
        default=None, description="Stem channels (defaults to base_channels)"
    )
    base_channels: int = Field(default=64, gt=0, description="Base channel count")
    num_stages: int = Field(default=4, ge=1, le=4, description="Number of ResNet stages")

    # Stride and dilation
    strides: tuple[int, ...] = Field(
        default=(1, 2, 2, 2), description="Stride of first block in each stage"
    )
    dilations: tuple[int, ...] = Field(
        default=(1, 1, 1, 1), description="Dilation rate for each stage"
    )
    out_indices: tuple[int, ...] = Field(
        default=(0, 1, 2, 3), description="Output feature map indices"
    )

    # Style
    style: Literal["pytorch", "caffe"] = Field(
        default="pytorch", description="ResNet style (affects stride placement)"
    )
    deep_stem: bool = Field(
        default=False, description="Replace 7x7 conv with three 3x3 convs"
    )
    avg_down: bool = Field(
        default=False, description="Use AvgPool for downsampling in bottleneck"
    )

    # Freezing and normalization
    frozen_stages: int = Field(
        default=-1, ge=-1, description="Freeze stages up to this index (-1 = none)"
    )
    norm_eval: bool = Field(
        default=True, description="Freeze BatchNorm running stats during training"
    )

    # Layer configurations
    conv_cfg: OptionalConfig = Field(default=None, description="Convolution layer config")
    norm_cfg: OptionalConfig = Field(
        default_factory=lambda: {"type": "BN", "requires_grad": True},
        description="Normalization layer config",
    )

    # DCN (Deformable Convolution) - rarely used
    dcn: OptionalConfig = Field(default=None, description="DCN config (if using deformable convs)")
    stage_with_dcn: tuple[bool, ...] = Field(
        default=(False, False, False, False), description="Which stages use DCN"
    )

    # Training settings
    with_cp: bool = Field(
        default=False, description="Use gradient checkpointing to save memory"
    )
    zero_init_residual: bool = Field(
        default=True, description="Zero-initialize last norm layer in residual blocks"
    )

    # Pretrained weights
    pretrained: Optional[str] = Field(default=None, description="Pretrained checkpoint path")
    init_cfg: OptionalConfig = Field(default=None, description="Weight initialization config")


class ResNeXtConfig(ResNetConfig):
    """ResNeXt backbone configuration.

    ResNeXt extends ResNet with grouped convolutions.
    Inherits all ResNet parameters plus group settings.

    Attributes:
        groups: Number of groups for grouped convolutions.
        base_width: Base width for each group.
    """

    type: Literal["ResNeXt"] = "ResNeXt"

    groups: int = Field(default=32, gt=0, description="Number of groups in grouped convolutions")
    base_width: int = Field(default=4, gt=0, description="Base width for each group")


# Discriminated union for any backbone type
BackboneType = Annotated[
    Union[SwinTransformerConfig, ResNetConfig, ResNeXtConfig],
    Field(discriminator="type"),
]
"""Type alias for any backbone configuration (with discriminator on 'type' field)."""
