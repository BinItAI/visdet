"""Dataset and data loading configuration schemas.

This module provides schemas for datasets, data pipelines, and dataloaders.

Example:
    >>> from visdet.schemas.data import CocoDatasetConfig, DataLoaderConfig
    >>> dataset = CocoDatasetConfig(
    ...     data_root='/data/coco',
    ...     ann_file='annotations/instances_train2017.json'
    ... )
"""

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import Field

from visdet.schemas.base import (
    ComponentConfig,
    ConfigList,
    DatasetConfig,
    OptionalConfig,
    TransformConfig,
    VisdetBaseConfig,
)


# =============================================================================
# Transform Configurations
# =============================================================================


class LoadImageFromFileConfig(TransformConfig):
    """Load image from file transform."""

    type: Literal["LoadImageFromFile"] = "LoadImageFromFile"
    to_float32: bool = Field(default=False, description="Convert to float32")
    color_type: str = Field(default="color", description="Color type")
    backend_args: OptionalConfig = Field(default=None, description="Backend args")


class LoadAnnotationsConfig(TransformConfig):
    """Load annotations transform."""

    type: Literal["LoadAnnotations"] = "LoadAnnotations"
    with_bbox: bool = Field(default=True, description="Load bounding boxes")
    with_mask: bool = Field(default=False, description="Load instance masks")
    with_seg: bool = Field(default=False, description="Load semantic segmentation")
    poly2mask: bool = Field(default=True, description="Convert polygons to masks")


class ResizeConfig(TransformConfig):
    """Resize image transform."""

    type: Literal["Resize"] = "Resize"
    scale: tuple[int, int] = Field(
        default=(1333, 800), description="Target scale (w, h)"
    )
    keep_ratio: bool = Field(default=True, description="Keep aspect ratio")
    backend: str = Field(default="cv2", description="Resize backend")


class RandomFlipConfig(TransformConfig):
    """Random horizontal flip transform."""

    type: Literal["RandomFlip"] = "RandomFlip"
    prob: float = Field(default=0.5, ge=0.0, le=1.0, description="Flip probability")
    direction: str = Field(default="horizontal", description="Flip direction")


class PackDetInputsConfig(TransformConfig):
    """Pack detection inputs transform."""

    type: Literal["PackDetInputs"] = "PackDetInputs"
    meta_keys: tuple[str, ...] = Field(
        default=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
        description="Keys to include in meta info",
    )


# =============================================================================
# Dataset Configurations
# =============================================================================


class CocoDatasetConfig(DatasetConfig):
    """COCO format dataset configuration.

    Attributes:
        data_root: Root directory containing images and annotations.
        ann_file: Path to annotation file (relative to data_root).
        data_prefix: Prefix dict for data paths.
        pipeline: Data loading and augmentation pipeline.
        metainfo: Dataset metadata (classes, palette).

    Example:
        >>> cfg = CocoDatasetConfig(
        ...     data_root='/data/coco',
        ...     ann_file='annotations/instances_train2017.json',
        ...     data_prefix={'img': 'train2017'}
        ... )
    """

    type: Literal["CocoDataset"] = "CocoDataset"
    data_root: str = Field(..., description="Root directory for data")
    ann_file: str = Field(..., description="Annotation file path (relative to data_root)")
    data_prefix: dict[str, str] = Field(
        default_factory=lambda: {"img": ""},
        description="Prefix paths for data (e.g., {'img': 'train2017'})",
    )
    pipeline: ConfigList = Field(
        default_factory=list, description="Data loading/augmentation pipeline"
    )
    metainfo: OptionalConfig = Field(
        default=None, description="Dataset metadata (classes, palette)"
    )
    filter_cfg: OptionalConfig = Field(
        default_factory=lambda: {"filter_empty_gt": True, "min_size": 32},
        description="Filter config for invalid samples",
    )
    backend_args: OptionalConfig = Field(default=None, description="Backend arguments")
    test_mode: bool = Field(default=False, description="Test mode (no GT loading)")


# =============================================================================
# Data Preprocessor Configuration
# =============================================================================


class DetDataPreprocessorConfig(ComponentConfig):
    """Detection data preprocessor configuration.

    Handles normalization, padding, and format conversion.
    """

    type: Literal["DetDataPreprocessor"] = "DetDataPreprocessor"
    mean: tuple[float, float, float] = Field(
        default=(123.675, 116.28, 103.53), description="Normalization mean (RGB)"
    )
    std: tuple[float, float, float] = Field(
        default=(58.395, 57.12, 57.375), description="Normalization std (RGB)"
    )
    bgr_to_rgb: bool = Field(default=True, description="Convert BGR to RGB")
    pad_mask: bool = Field(default=True, description="Pad instance masks")
    pad_size_divisor: int = Field(
        default=32, gt=0, description="Pad to multiple of this value"
    )


# =============================================================================
# Sampler Configurations
# =============================================================================


class DefaultSamplerConfig(ComponentConfig):
    """Default data sampler configuration."""

    type: Literal["DefaultSampler"] = "DefaultSampler"
    shuffle: bool = Field(default=True, description="Shuffle data")


class InfiniteSamplerConfig(ComponentConfig):
    """Infinite data sampler configuration."""

    type: Literal["InfiniteSampler"] = "InfiniteSampler"
    shuffle: bool = Field(default=True, description="Shuffle data")


# =============================================================================
# DataLoader Configuration
# =============================================================================


class DataLoaderConfig(VisdetBaseConfig):
    """DataLoader configuration.

    Configures batch loading, workers, and sampling.

    Attributes:
        batch_size: Samples per batch per GPU.
        num_workers: Number of data loading workers.
        persistent_workers: Keep workers alive between epochs.
        sampler: Sampling strategy configuration.
        dataset: Dataset configuration.

    Example:
        >>> cfg = DataLoaderConfig(
        ...     batch_size=2,
        ...     num_workers=4,
        ...     dataset=CocoDatasetConfig(
        ...         data_root='/data/coco',
        ...         ann_file='annotations/instances_train2017.json'
        ...     )
        ... )
    """

    batch_size: int = Field(default=2, ge=1, description="Batch size per GPU")
    num_workers: int = Field(default=2, ge=0, description="Number of data loading workers")
    persistent_workers: bool = Field(
        default=True, description="Keep workers alive between epochs"
    )
    sampler: DefaultSamplerConfig | InfiniteSamplerConfig = Field(
        default_factory=DefaultSamplerConfig, description="Sampler config"
    )
    dataset: CocoDatasetConfig = Field(..., description="Dataset config")
    pin_memory: bool = Field(default=True, description="Pin memory for faster transfer")
    drop_last: bool = Field(default=False, description="Drop incomplete final batch")


# Type aliases
DatasetType = Annotated[Union[CocoDatasetConfig], Field(discriminator="type")]
TransformType = Annotated[
    Union[
        LoadImageFromFileConfig,
        LoadAnnotationsConfig,
        ResizeConfig,
        RandomFlipConfig,
        PackDetInputsConfig,
    ],
    Field(discriminator="type"),
]
