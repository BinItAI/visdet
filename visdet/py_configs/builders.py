"""Factory functions for building configuration objects.

This module provides builder functions that create properly configured
detection models with sensible defaults and full IDE autocomplete support.

Example:
    >>> from visdet.py_configs.builders import mask_rcnn, swin_tiny
    >>> # Create Mask R-CNN with Swin-Tiny backbone
    >>> model = mask_rcnn(backbone=swin_tiny(), num_classes=80)
    >>> model.to_dict()  # Convert to dict for MMEngine
"""

from visdet.schemas import (
    AdamWConfig,
    CocoDatasetConfig,
    DataLoaderConfig,
    DefaultSamplerConfig,
    DetDataPreprocessorConfig,
    EpochBasedTrainLoopConfig,
    ExperimentConfig,
    FCNMaskHeadConfig,
    FPNConfig,
    LoadAnnotationsConfig,
    LoadImageFromFileConfig,
    MaskRCNNConfig,
    OneCycleLRConfig,
    OptimWrapperConfig,
    PackDetInputsConfig,
    RandomFlipConfig,
    RCNNTestConfig,
    RCNNTrainConfig,
    ResizeConfig,
    ResNetConfig,
    RPNHeadConfig,
    RPNProposalConfig,
    RPNTestConfig,
    RPNTrainConfig,
    Shared2FCBBoxHeadConfig,
    SingleRoIExtractorConfig,
    StandardRoIHeadConfig,
    SwinTransformerConfig,
    TwoStageTestConfig,
    TwoStageTrainConfig,
    ValLoopConfig,
)


# =============================================================================
# Backbone Builders
# =============================================================================


def swin_tiny(
    drop_path_rate: float = 0.2,
    frozen_stages: int = -1,
    **kwargs,
) -> SwinTransformerConfig:
    """Create Swin Transformer Tiny backbone config.

    Args:
        drop_path_rate: Stochastic depth rate.
        frozen_stages: Freeze stages up to this index (-1 = none).
        **kwargs: Additional overrides.

    Returns:
        Configured SwinTransformerConfig.

    Example:
        >>> backbone = swin_tiny(drop_path_rate=0.3)
    """
    return SwinTransformerConfig(
        embed_dims=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        drop_path_rate=drop_path_rate,
        frozen_stages=frozen_stages,
        convert_weights=True,
        **kwargs,
    )


def swin_small(
    drop_path_rate: float = 0.3,
    frozen_stages: int = -1,
    **kwargs,
) -> SwinTransformerConfig:
    """Create Swin Transformer Small backbone config.

    Deeper than Tiny (18 blocks vs 6 in stage 3).

    Args:
        drop_path_rate: Stochastic depth rate.
        frozen_stages: Freeze stages up to this index.
        **kwargs: Additional overrides.

    Returns:
        Configured SwinTransformerConfig.
    """
    return SwinTransformerConfig(
        embed_dims=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        drop_path_rate=drop_path_rate,
        frozen_stages=frozen_stages,
        convert_weights=True,
        **kwargs,
    )


def swin_base(
    drop_path_rate: float = 0.5,
    frozen_stages: int = -1,
    **kwargs,
) -> SwinTransformerConfig:
    """Create Swin Transformer Base backbone config.

    Wider channels (128 vs 96) than Small.

    Args:
        drop_path_rate: Stochastic depth rate.
        frozen_stages: Freeze stages up to this index.
        **kwargs: Additional overrides.

    Returns:
        Configured SwinTransformerConfig.
    """
    return SwinTransformerConfig(
        embed_dims=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        window_size=7,
        drop_path_rate=drop_path_rate,
        frozen_stages=frozen_stages,
        convert_weights=True,
        **kwargs,
    )


def resnet50(
    frozen_stages: int = 1,
    norm_eval: bool = True,
    **kwargs,
) -> ResNetConfig:
    """Create ResNet-50 backbone config.

    Standard ResNet-50 with frozen stem (stage 0).

    Args:
        frozen_stages: Freeze stages up to this index.
        norm_eval: Freeze BatchNorm stats during training.
        **kwargs: Additional overrides.

    Returns:
        Configured ResNetConfig.
    """
    return ResNetConfig(
        depth=50,
        frozen_stages=frozen_stages,
        norm_eval=norm_eval,
        **kwargs,
    )


def resnet101(
    frozen_stages: int = 1,
    norm_eval: bool = True,
    **kwargs,
) -> ResNetConfig:
    """Create ResNet-101 backbone config."""
    return ResNetConfig(
        depth=101,
        frozen_stages=frozen_stages,
        norm_eval=norm_eval,
        **kwargs,
    )


# =============================================================================
# Neck Builders
# =============================================================================


def fpn_for_swin(embed_dims: int = 96, out_channels: int = 256, **kwargs) -> FPNConfig:
    """Create FPN neck configured for Swin Transformer backbone.

    Auto-computes in_channels based on Swin embed_dims.

    Args:
        embed_dims: Swin backbone embed_dims (96=Tiny/Small, 128=Base).
        out_channels: FPN output channels.
        **kwargs: Additional overrides.

    Returns:
        Configured FPNConfig.
    """
    return FPNConfig(
        in_channels=[embed_dims, embed_dims * 2, embed_dims * 4, embed_dims * 8],
        out_channels=out_channels,
        num_outs=5,
        **kwargs,
    )


def fpn_for_resnet(depth: int = 50, out_channels: int = 256, **kwargs) -> FPNConfig:
    """Create FPN neck configured for ResNet backbone.

    Auto-computes in_channels based on ResNet depth.

    Args:
        depth: ResNet depth (18, 34, 50, 101, 152).
        out_channels: FPN output channels.
        **kwargs: Additional overrides.

    Returns:
        Configured FPNConfig.
    """
    if depth in [18, 34]:
        in_channels = [64, 128, 256, 512]
    else:  # 50, 101, 152
        in_channels = [256, 512, 1024, 2048]

    return FPNConfig(
        in_channels=in_channels,
        out_channels=out_channels,
        num_outs=5,
        **kwargs,
    )


# =============================================================================
# Head Builders
# =============================================================================


def standard_rpn_head(in_channels: int = 256, **kwargs) -> RPNHeadConfig:
    """Create standard RPN head config.

    Args:
        in_channels: Input feature channels from neck.
        **kwargs: Additional overrides.

    Returns:
        Configured RPNHeadConfig.
    """
    return RPNHeadConfig(
        in_channels=in_channels,
        feat_channels=256,
        **kwargs,
    )


def standard_roi_head(num_classes: int, with_mask: bool = True, **kwargs) -> StandardRoIHeadConfig:
    """Create standard RoI head config for Mask R-CNN.

    Args:
        num_classes: Number of object classes.
        with_mask: Include mask head for instance segmentation.
        **kwargs: Additional overrides.

    Returns:
        Configured StandardRoIHeadConfig.
    """
    mask_roi_extractor = None
    mask_head = None

    if with_mask:
        mask_roi_extractor = SingleRoIExtractorConfig(
            roi_layer={"type": "RoIAlign", "output_size": 14, "sampling_ratio": 0},
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        )
        mask_head = FCNMaskHeadConfig(
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=num_classes,
        )

    return StandardRoIHeadConfig(
        bbox_roi_extractor=SingleRoIExtractorConfig(
            roi_layer={"type": "RoIAlign", "output_size": 7, "sampling_ratio": 0},
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=Shared2FCBBoxHeadConfig(
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
        ),
        mask_roi_extractor=mask_roi_extractor,
        mask_head=mask_head,
        **kwargs,
    )


# =============================================================================
# Model Builders
# =============================================================================


def mask_rcnn(
    backbone: SwinTransformerConfig | ResNetConfig,
    num_classes: int = 80,
    neck_out_channels: int = 256,
    **kwargs,
) -> MaskRCNNConfig:
    """Create Mask R-CNN config with auto-configured components.

    This builder automatically configures the FPN neck based on the
    backbone type, and sets up RPN and RoI heads with standard settings.

    Args:
        backbone: Backbone configuration (Swin or ResNet).
        num_classes: Number of object classes (default: 80 for COCO).
        neck_out_channels: FPN output channels.
        **kwargs: Additional overrides.

    Returns:
        Complete MaskRCNNConfig.

    Example:
        >>> model = mask_rcnn(backbone=swin_tiny(), num_classes=20)
    """
    # Auto-configure neck based on backbone
    if isinstance(backbone, SwinTransformerConfig):
        neck = fpn_for_swin(backbone.embed_dims, neck_out_channels)
    else:
        neck = fpn_for_resnet(backbone.depth, neck_out_channels)

    return MaskRCNNConfig(
        backbone=backbone,
        neck=neck,
        rpn_head=standard_rpn_head(neck_out_channels),
        roi_head=standard_roi_head(num_classes, with_mask=True),
        **kwargs,
    )


# =============================================================================
# Optimizer and Scheduler Builders
# =============================================================================


def adamw_default(lr: float = 1e-4, weight_decay: float = 0.05, **kwargs) -> OptimWrapperConfig:
    """Create AdamW optimizer wrapper with default settings.

    Args:
        lr: Learning rate.
        weight_decay: Weight decay coefficient.
        **kwargs: Additional optimizer overrides.

    Returns:
        Configured OptimWrapperConfig.
    """
    return OptimWrapperConfig(
        optimizer=AdamWConfig(lr=lr, weight_decay=weight_decay, **kwargs),
    )


def one_cycle_scheduler(max_lr: float = 1e-3, **kwargs) -> OneCycleLRConfig:
    """Create 1cycle LR scheduler config.

    Args:
        max_lr: Maximum learning rate at peak.
        **kwargs: Additional scheduler overrides.

    Returns:
        Configured OneCycleLRConfig.
    """
    return OneCycleLRConfig(max_lr=max_lr, **kwargs)


# =============================================================================
# Data Builders
# =============================================================================


def coco_train_pipeline() -> list[dict]:
    """Create standard COCO training data pipeline."""
    return [
        LoadImageFromFileConfig().to_dict(),
        LoadAnnotationsConfig(with_bbox=True, with_mask=True).to_dict(),
        ResizeConfig(scale=(1333, 800), keep_ratio=True).to_dict(),
        RandomFlipConfig(prob=0.5).to_dict(),
        PackDetInputsConfig().to_dict(),
    ]


def coco_test_pipeline() -> list[dict]:
    """Create standard COCO test/val data pipeline."""
    return [
        LoadImageFromFileConfig().to_dict(),
        ResizeConfig(scale=(1333, 800), keep_ratio=True).to_dict(),
        LoadAnnotationsConfig(with_bbox=True, with_mask=True).to_dict(),
        PackDetInputsConfig(
            meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")
        ).to_dict(),
    ]


def coco_dataset(
    data_root: str,
    ann_file: str,
    img_prefix: str = "",
    pipeline: list[dict] | None = None,
    test_mode: bool = False,
    **kwargs,
) -> CocoDatasetConfig:
    """Create COCO dataset config.

    Args:
        data_root: Root data directory.
        ann_file: Annotation file path (relative to data_root).
        img_prefix: Image directory prefix.
        pipeline: Data pipeline (auto-set if None).
        test_mode: Test mode (no GT loading).
        **kwargs: Additional dataset overrides.

    Returns:
        Configured CocoDatasetConfig.
    """
    if pipeline is None:
        pipeline = coco_test_pipeline() if test_mode else coco_train_pipeline()

    return CocoDatasetConfig(
        data_root=data_root,
        ann_file=ann_file,
        data_prefix={"img": img_prefix},
        pipeline=pipeline,
        test_mode=test_mode,
        **kwargs,
    )


def train_dataloader(
    dataset: CocoDatasetConfig,
    batch_size: int = 2,
    num_workers: int = 2,
    **kwargs,
) -> DataLoaderConfig:
    """Create training dataloader config.

    Args:
        dataset: Dataset configuration.
        batch_size: Samples per batch per GPU.
        num_workers: Data loading workers.
        **kwargs: Additional dataloader overrides.

    Returns:
        Configured DataLoaderConfig.
    """
    return DataLoaderConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        sampler=DefaultSamplerConfig(shuffle=True),
        dataset=dataset,
        **kwargs,
    )


def val_dataloader(
    dataset: CocoDatasetConfig,
    batch_size: int = 1,
    num_workers: int = 2,
    **kwargs,
) -> DataLoaderConfig:
    """Create validation dataloader config.

    Args:
        dataset: Dataset configuration.
        batch_size: Samples per batch per GPU.
        num_workers: Data loading workers.
        **kwargs: Additional dataloader overrides.

    Returns:
        Configured DataLoaderConfig.
    """
    return DataLoaderConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        sampler=DefaultSamplerConfig(shuffle=False),
        dataset=dataset,
        **kwargs,
    )
