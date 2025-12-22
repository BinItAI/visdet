"""Pydantic configuration schemas for visdet.

This package provides type-safe configuration with full IDE support.

Key Benefits:
1. Validation/Type Safety - Catch config errors before runtime
2. Reduce Repetition - Factory functions, inheritance, composition
3. IDE Experience - Autocomplete, jump-to-definition, inline docs
4. Performance - Pydantic v2 uses Rust core, fast validation

Example:
    >>> from visdet.schemas import (
    ...     SwinTransformerConfig,
    ...     FPNConfig,
    ...     MaskRCNNConfig,
    ...     ExperimentConfig,
    ... )
    >>> # Create a model config with full IDE autocomplete
    >>> backbone = SwinTransformerConfig(embed_dims=96, depths=(2, 2, 6, 2))
    >>> neck = FPNConfig(in_channels=[96, 192, 384, 768], out_channels=256, num_outs=5)
"""

# Base classes
from visdet.schemas.base import (
    BackboneConfig,
    ComponentConfig,
    DatasetConfig,
    HeadConfig,
    LossConfig,
    NeckConfig,
    OptionalConfig,
    TransformConfig,
    VisdetBaseConfig,
)

# Backbones
from visdet.schemas.backbones import (
    BackboneType,
    ResNetConfig,
    ResNeXtConfig,
    SwinTransformerConfig,
)

# Necks
from visdet.schemas.necks import FPNConfig, NeckType

# Heads and components
from visdet.schemas.heads import (
    AnchorGeneratorConfig,
    CrossEntropyLossConfig,
    DeltaXYWHBBoxCoderConfig,
    FCNMaskHeadConfig,
    L1LossConfig,
    MaxIoUAssignerConfig,
    NMSConfig,
    RandomSamplerConfig,
    RCNNTestConfig,
    RCNNTrainConfig,
    RoIAlignConfig,
    RoIHeadType,
    RPNHeadConfig,
    RPNHeadType,
    RPNProposalConfig,
    RPNTestConfig,
    RPNTrainConfig,
    Shared2FCBBoxHeadConfig,
    SingleRoIExtractorConfig,
    SmoothL1LossConfig,
    StandardRoIHeadConfig,
)

# Data and transforms
from visdet.schemas.data import (
    CocoDatasetConfig,
    DataLoaderConfig,
    DatasetType,
    DefaultSamplerConfig,
    DetDataPreprocessorConfig,
    LoadAnnotationsConfig,
    LoadImageFromFileConfig,
    PackDetInputsConfig,
    RandomFlipConfig,
    ResizeConfig,
    TransformType,
)

# Training
from visdet.schemas.training import (
    AdamConfig,
    AdamW8bitConfig,
    AdamWConfig,
    CheckpointHookConfig,
    CosineAnnealingLRConfig,
    EpochBasedTrainLoopConfig,
    IterBasedTrainLoopConfig,
    LinearLRConfig,
    LoggerHookConfig,
    MultiStepLRConfig,
    OneCycleLRConfig,
    OptimizerType,
    OptimWrapperConfig,
    SchedulerType,
    SGDConfig,
    TestLoopConfig,
    TrainLoopType,
    ValLoopConfig,
)

# Complete models
from visdet.schemas.models import (
    ExperimentConfig,
    FasterRCNNConfig,
    MaskRCNNConfig,
    ModelType,
    TwoStageTestConfig,
    TwoStageTrainConfig,
)

__all__ = [
    # Base
    "VisdetBaseConfig",
    "ComponentConfig",
    "BackboneConfig",
    "NeckConfig",
    "HeadConfig",
    "LossConfig",
    "DatasetConfig",
    "TransformConfig",
    "OptionalConfig",
    # Backbones
    "SwinTransformerConfig",
    "ResNetConfig",
    "ResNeXtConfig",
    "BackboneType",
    # Necks
    "FPNConfig",
    "NeckType",
    # Heads
    "RPNHeadConfig",
    "StandardRoIHeadConfig",
    "Shared2FCBBoxHeadConfig",
    "FCNMaskHeadConfig",
    "RPNHeadType",
    "RoIHeadType",
    # Head components
    "AnchorGeneratorConfig",
    "DeltaXYWHBBoxCoderConfig",
    "MaxIoUAssignerConfig",
    "RandomSamplerConfig",
    "RoIAlignConfig",
    "SingleRoIExtractorConfig",
    "NMSConfig",
    # Losses
    "CrossEntropyLossConfig",
    "L1LossConfig",
    "SmoothL1LossConfig",
    # Train/test configs
    "RPNTrainConfig",
    "RPNProposalConfig",
    "RCNNTrainConfig",
    "RPNTestConfig",
    "RCNNTestConfig",
    "TwoStageTrainConfig",
    "TwoStageTestConfig",
    # Data
    "CocoDatasetConfig",
    "DataLoaderConfig",
    "DetDataPreprocessorConfig",
    "DefaultSamplerConfig",
    "DatasetType",
    # Transforms
    "LoadImageFromFileConfig",
    "LoadAnnotationsConfig",
    "ResizeConfig",
    "RandomFlipConfig",
    "PackDetInputsConfig",
    "TransformType",
    # Training
    "AdamWConfig",
    "SGDConfig",
    "AdamConfig",
    "AdamW8bitConfig",
    "OptimWrapperConfig",
    "OptimizerType",
    # Schedulers
    "OneCycleLRConfig",
    "MultiStepLRConfig",
    "CosineAnnealingLRConfig",
    "LinearLRConfig",
    "SchedulerType",
    # Loops
    "EpochBasedTrainLoopConfig",
    "IterBasedTrainLoopConfig",
    "ValLoopConfig",
    "TestLoopConfig",
    "TrainLoopType",
    # Hooks
    "CheckpointHookConfig",
    "LoggerHookConfig",
    # Complete models
    "MaskRCNNConfig",
    "FasterRCNNConfig",
    "ExperimentConfig",
    "ModelType",
]
