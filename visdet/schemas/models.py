"""Complete model configuration schemas.

This module provides schemas for complete detector configurations,
combining backbones, necks, heads, and training settings.

Example:
    >>> from visdet.schemas.models import MaskRCNNConfig
    >>> model = MaskRCNNConfig(
    ...     backbone=SwinTransformerConfig(embed_dims=96),
    ...     num_classes=80
    ... )
"""

from typing import Annotated, Literal, Optional, Union

from pydantic import Field, model_validator

from visdet.schemas.backbones import BackboneType, ResNetConfig, SwinTransformerConfig
from visdet.schemas.base import ComponentConfig, OptionalConfig, VisdetBaseConfig
from visdet.schemas.data import DataLoaderConfig, DetDataPreprocessorConfig
from visdet.schemas.heads import (
    FCNMaskHeadConfig,
    RCNNTestConfig,
    RCNNTrainConfig,
    RPNHeadConfig,
    RPNProposalConfig,
    RPNTestConfig,
    RPNTrainConfig,
    Shared2FCBBoxHeadConfig,
    SingleRoIExtractorConfig,
    StandardRoIHeadConfig,
)
from visdet.schemas.necks import FPNConfig, NeckType
from visdet.schemas.training import (
    EpochBasedTrainLoopConfig,
    OptimWrapperConfig,
    SchedulerType,
    ValLoopConfig,
)


# =============================================================================
# Training and Test Configuration
# =============================================================================


class TwoStageTrainConfig(VisdetBaseConfig):
    """Training configuration for two-stage detectors."""

    rpn: RPNTrainConfig = Field(
        default_factory=RPNTrainConfig, description="RPN training config"
    )
    rpn_proposal: RPNProposalConfig = Field(
        default_factory=RPNProposalConfig, description="RPN proposal config"
    )
    rcnn: RCNNTrainConfig = Field(
        default_factory=RCNNTrainConfig, description="RCNN training config"
    )


class TwoStageTestConfig(VisdetBaseConfig):
    """Test configuration for two-stage detectors."""

    rpn: RPNTestConfig = Field(
        default_factory=RPNTestConfig, description="RPN test config"
    )
    rcnn: RCNNTestConfig = Field(
        default_factory=RCNNTestConfig, description="RCNN test config"
    )


# =============================================================================
# Detector Configurations
# =============================================================================


class MaskRCNNConfig(ComponentConfig):
    """Mask R-CNN detector configuration.

    Complete configuration for the Mask R-CNN two-stage detector
    with instance segmentation support.

    Attributes:
        backbone: Feature extraction backbone.
        neck: Feature pyramid network.
        rpn_head: Region proposal network head.
        roi_head: RoI prediction head with bbox and mask branches.
        train_cfg: Training configuration.
        test_cfg: Testing configuration.

    Example:
        >>> from visdet.schemas.backbones import SwinTransformerConfig
        >>> cfg = MaskRCNNConfig(
        ...     backbone=SwinTransformerConfig(embed_dims=96),
        ...     num_classes=80
        ... )
    """

    type: Literal["MaskRCNN"] = "MaskRCNN"

    # Data preprocessor
    data_preprocessor: DetDataPreprocessorConfig = Field(
        default_factory=DetDataPreprocessorConfig,
        description="Data preprocessor config",
    )

    # Architecture
    backbone: SwinTransformerConfig | ResNetConfig = Field(
        ..., description="Backbone network config"
    )
    neck: FPNConfig = Field(..., description="Neck network config")
    rpn_head: RPNHeadConfig = Field(
        default_factory=RPNHeadConfig, description="RPN head config"
    )
    roi_head: StandardRoIHeadConfig = Field(..., description="RoI head config")

    # Training and testing
    train_cfg: TwoStageTrainConfig = Field(
        default_factory=TwoStageTrainConfig, description="Training config"
    )
    test_cfg: TwoStageTestConfig = Field(
        default_factory=TwoStageTestConfig, description="Testing config"
    )

    @model_validator(mode="before")
    @classmethod
    def auto_configure_neck(cls, data: dict) -> dict:
        """Auto-configure FPN in_channels based on backbone."""
        if "backbone" in data and "neck" in data:
            backbone = data["backbone"]
            neck = data["neck"]

            # If neck doesn't have in_channels, compute from backbone
            if isinstance(neck, dict) and "in_channels" not in neck:
                if isinstance(backbone, dict):
                    backbone_type = backbone.get("type", "")
                    if backbone_type == "SwinTransformer":
                        embed_dims = backbone.get("embed_dims", 96)
                        neck["in_channels"] = [
                            embed_dims,
                            embed_dims * 2,
                            embed_dims * 4,
                            embed_dims * 8,
                        ]
                    elif backbone_type == "ResNet":
                        depth = backbone.get("depth", 50)
                        if depth in [18, 34]:
                            neck["in_channels"] = [64, 128, 256, 512]
                        else:  # 50, 101, 152
                            neck["in_channels"] = [256, 512, 1024, 2048]

        return data


class FasterRCNNConfig(ComponentConfig):
    """Faster R-CNN detector configuration.

    Two-stage detector without instance segmentation.
    """

    type: Literal["FasterRCNN"] = "FasterRCNN"

    data_preprocessor: DetDataPreprocessorConfig = Field(
        default_factory=DetDataPreprocessorConfig,
        description="Data preprocessor config",
    )
    backbone: SwinTransformerConfig | ResNetConfig = Field(
        ..., description="Backbone network config"
    )
    neck: FPNConfig = Field(..., description="Neck network config")
    rpn_head: RPNHeadConfig = Field(
        default_factory=RPNHeadConfig, description="RPN head config"
    )
    roi_head: StandardRoIHeadConfig = Field(..., description="RoI head config")
    train_cfg: TwoStageTrainConfig = Field(
        default_factory=TwoStageTrainConfig, description="Training config"
    )
    test_cfg: TwoStageTestConfig = Field(
        default_factory=TwoStageTestConfig, description="Testing config"
    )


# =============================================================================
# Complete Experiment Configuration
# =============================================================================


class ExperimentConfig(VisdetBaseConfig):
    """Complete experiment configuration.

    Top-level config that combines model, data, training settings.
    This is the main config type passed to SimpleRunner.

    Attributes:
        model: Complete detector configuration.
        train_dataloader: Training data loading config.
        val_dataloader: Validation data loading config (optional).
        optim_wrapper: Optimizer wrapper config.
        param_scheduler: Learning rate scheduler config (optional).
        train_cfg: Training loop config.
        work_dir: Output directory for logs and checkpoints.

    Example:
        >>> from visdet.schemas.models import ExperimentConfig
        >>> cfg = ExperimentConfig(
        ...     model=MaskRCNNConfig(...),
        ...     train_dataloader=DataLoaderConfig(...),
        ...     optim_wrapper=OptimWrapperConfig(...),
        ...     work_dir='./work_dirs/my_experiment'
        ... )
    """

    # Scope
    default_scope: str = Field(default="visdet", description="Default registry scope")

    # Model
    model: MaskRCNNConfig | FasterRCNNConfig = Field(..., description="Model config")

    # Data
    train_dataloader: DataLoaderConfig = Field(..., description="Training dataloader")
    val_dataloader: Optional[DataLoaderConfig] = Field(
        default=None, description="Validation dataloader"
    )
    test_dataloader: Optional[DataLoaderConfig] = Field(
        default=None, description="Test dataloader"
    )

    # Optimization
    optim_wrapper: OptimWrapperConfig = Field(..., description="Optimizer wrapper")
    param_scheduler: Optional[SchedulerType] = Field(
        default=None, description="LR scheduler"
    )

    # Training loop
    train_cfg: EpochBasedTrainLoopConfig = Field(
        default_factory=EpochBasedTrainLoopConfig, description="Training loop config"
    )
    val_cfg: Optional[ValLoopConfig] = Field(
        default=None, description="Validation loop config"
    )

    # Evaluation
    val_evaluator: OptionalConfig = Field(
        default=None, description="Validation evaluator config"
    )

    # Output
    work_dir: str = Field(default="./work_dirs", description="Output directory")

    # Hooks
    default_hooks: OptionalConfig = Field(default=None, description="Default hooks")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_processor: OptionalConfig = Field(
        default_factory=lambda: {"window_size": 50}, description="Log processor config"
    )


# Type aliases
ModelType = Annotated[
    Union[MaskRCNNConfig, FasterRCNNConfig], Field(discriminator="type")
]
