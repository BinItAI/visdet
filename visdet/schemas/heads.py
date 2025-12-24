"""Detection head configuration schemas.

This module provides schemas for RPN heads, RoI heads, and their components.
These are the most complex configs in detection as they involve nested
components for anchors, coders, losses, extractors, and samplers.

Example:
    >>> from visdet.schemas.heads import RPNHeadConfig, StandardRoIHeadConfig
    >>> rpn = RPNHeadConfig(in_channels=256, feat_channels=256)
    >>> roi = StandardRoIHeadConfig(num_classes=80)
"""

from typing import Annotated, Literal, Optional, Union

from pydantic import Field

from visdet.schemas.base import ComponentConfig, HeadConfig, LossConfig, OptionalConfig


# =============================================================================
# Loss Configurations
# =============================================================================


class CrossEntropyLossConfig(LossConfig):
    """Cross entropy loss configuration."""

    type: Literal["CrossEntropyLoss"] = "CrossEntropyLoss"
    use_sigmoid: bool = Field(default=False, description="Use sigmoid instead of softmax")
    use_mask: bool = Field(default=False, description="Use mask-based loss")
    loss_weight: float = Field(default=1.0, gt=0, description="Loss weight")


class L1LossConfig(LossConfig):
    """L1 (smooth) loss configuration."""

    type: Literal["L1Loss"] = "L1Loss"
    loss_weight: float = Field(default=1.0, gt=0, description="Loss weight")


class SmoothL1LossConfig(LossConfig):
    """Smooth L1 loss configuration."""

    type: Literal["SmoothL1Loss"] = "SmoothL1Loss"
    beta: float = Field(default=1.0, gt=0, description="Smooth L1 beta parameter")
    loss_weight: float = Field(default=1.0, gt=0, description="Loss weight")


# =============================================================================
# Anchor and BBox Coder Configurations
# =============================================================================


class AnchorGeneratorConfig(ComponentConfig):
    """Anchor generator configuration.

    Generates anchor boxes at each feature map location.

    Attributes:
        scales: Anchor scales (multiplied by stride).
        ratios: Anchor aspect ratios.
        strides: Feature map strides (one per FPN level).
    """

    type: Literal["AnchorGenerator"] = "AnchorGenerator"
    scales: list[float] = Field(
        default=[8], description="Anchor scales (multiplied by stride)"
    )
    ratios: list[float] = Field(
        default=[0.5, 1.0, 2.0], description="Anchor aspect ratios"
    )
    strides: list[int] = Field(
        default=[4, 8, 16, 32, 64],
        description="Feature map strides (typically one per FPN level)",
    )


class DeltaXYWHBBoxCoderConfig(ComponentConfig):
    """Delta XYWH bounding box coder configuration.

    Encodes/decodes bounding boxes as deltas from anchors.
    """

    type: Literal["DeltaXYWHBBoxCoder"] = "DeltaXYWHBBoxCoder"
    target_means: tuple[float, float, float, float] = Field(
        default=(0.0, 0.0, 0.0, 0.0), description="Target mean for normalization"
    )
    target_stds: tuple[float, float, float, float] = Field(
        default=(1.0, 1.0, 1.0, 1.0), description="Target std for normalization"
    )


# =============================================================================
# Assigner and Sampler Configurations
# =============================================================================


class MaxIoUAssignerConfig(ComponentConfig):
    """MaxIoU assigner configuration.

    Assigns ground truth to anchors based on IoU thresholds.
    """

    type: Literal["MaxIoUAssigner"] = "MaxIoUAssigner"
    pos_iou_thr: float = Field(
        default=0.7, ge=0.0, le=1.0, description="IoU threshold for positive assignment"
    )
    neg_iou_thr: float = Field(
        default=0.3, ge=0.0, le=1.0, description="IoU threshold for negative assignment"
    )
    min_pos_iou: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Minimum IoU for positive assignment"
    )
    match_low_quality: bool = Field(
        default=True, description="Match low quality proposals to GT"
    )
    ignore_iof_thr: float = Field(
        default=-1, description="IoF threshold for ignoring (-1 = disabled)"
    )


class RandomSamplerConfig(ComponentConfig):
    """Random sampler configuration.

    Randomly samples positive and negative proposals.
    """

    type: Literal["RandomSampler"] = "RandomSampler"
    num: int = Field(default=256, gt=0, description="Total number of samples")
    pos_fraction: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Fraction of positive samples"
    )
    neg_pos_ub: int = Field(
        default=-1, description="Upper bound on negatives per positive (-1 = unlimited)"
    )
    add_gt_as_proposals: bool = Field(
        default=False, description="Add GT boxes as proposals"
    )


# =============================================================================
# RoI Extractor Configurations
# =============================================================================


class RoIAlignConfig(ComponentConfig):
    """RoI Align layer configuration."""

    type: Literal["RoIAlign"] = "RoIAlign"
    output_size: int = Field(default=7, gt=0, description="Output feature size")
    sampling_ratio: int = Field(
        default=0, ge=0, description="Sampling ratio (0 = adaptive)"
    )


class SingleRoIExtractorConfig(ComponentConfig):
    """Single-level RoI feature extractor configuration."""

    type: Literal["SingleRoIExtractor"] = "SingleRoIExtractor"
    roi_layer: RoIAlignConfig = Field(
        default_factory=lambda: RoIAlignConfig(),
        description="RoI pooling layer config",
    )
    out_channels: int = Field(default=256, gt=0, description="Output channels")
    featmap_strides: list[int] = Field(
        default=[4, 8, 16, 32], description="Feature map strides"
    )


# =============================================================================
# BBox and Mask Head Configurations
# =============================================================================


class Shared2FCBBoxHeadConfig(HeadConfig):
    """Shared 2-FC bbox head configuration.

    Standard bbox head with 2 shared fully-connected layers.
    """

    type: Literal["Shared2FCBBoxHead"] = "Shared2FCBBoxHead"
    in_channels: int = Field(default=256, gt=0, description="Input channels")
    fc_out_channels: int = Field(default=1024, gt=0, description="FC layer output channels")
    roi_feat_size: int = Field(default=7, gt=0, description="RoI feature size")
    num_classes: int = Field(..., gt=0, description="Number of object classes")
    bbox_coder: DeltaXYWHBBoxCoderConfig = Field(
        default_factory=lambda: DeltaXYWHBBoxCoderConfig(
            target_stds=(0.1, 0.1, 0.2, 0.2)
        ),
        description="BBox coder config",
    )
    reg_class_agnostic: bool = Field(
        default=False, description="Use class-agnostic regression"
    )
    loss_cls: CrossEntropyLossConfig = Field(
        default_factory=CrossEntropyLossConfig, description="Classification loss"
    )
    loss_bbox: L1LossConfig = Field(
        default_factory=L1LossConfig, description="Regression loss"
    )


class FCNMaskHeadConfig(HeadConfig):
    """FCN mask head configuration.

    Fully convolutional mask prediction head.
    """

    type: Literal["FCNMaskHead"] = "FCNMaskHead"
    num_convs: int = Field(default=4, ge=1, description="Number of conv layers")
    in_channels: int = Field(default=256, gt=0, description="Input channels")
    conv_out_channels: int = Field(default=256, gt=0, description="Conv output channels")
    num_classes: int = Field(..., gt=0, description="Number of object classes")
    loss_mask: CrossEntropyLossConfig = Field(
        default_factory=lambda: CrossEntropyLossConfig(use_mask=True),
        description="Mask loss config",
    )


# =============================================================================
# RPN Head Configuration
# =============================================================================


class RPNHeadConfig(HeadConfig):
    """Region Proposal Network (RPN) head configuration.

    Generates object proposals from feature maps.

    Attributes:
        in_channels: Number of input feature channels (from neck).
        feat_channels: Number of feature channels in conv layers.
        anchor_generator: Anchor generation config.
        bbox_coder: BBox encoding/decoding config.
        loss_cls: Classification loss for objectness.
        loss_bbox: Regression loss for box refinement.

    Example:
        >>> cfg = RPNHeadConfig(
        ...     in_channels=256,
        ...     feat_channels=256,
        ...     anchor_generator=AnchorGeneratorConfig(
        ...         strides=[4, 8, 16, 32, 64]
        ...     )
        ... )
    """

    type: Literal["RPNHead"] = "RPNHead"
    in_channels: int = Field(default=256, gt=0, description="Input feature channels")
    feat_channels: int = Field(default=256, gt=0, description="Conv feature channels")
    num_convs: int = Field(default=1, ge=1, description="Number of conv layers")
    anchor_generator: AnchorGeneratorConfig = Field(
        default_factory=AnchorGeneratorConfig, description="Anchor generator config"
    )
    bbox_coder: DeltaXYWHBBoxCoderConfig = Field(
        default_factory=DeltaXYWHBBoxCoderConfig, description="BBox coder config"
    )
    loss_cls: CrossEntropyLossConfig = Field(
        default_factory=lambda: CrossEntropyLossConfig(use_sigmoid=True),
        description="Classification loss",
    )
    loss_bbox: L1LossConfig = Field(
        default_factory=L1LossConfig, description="Regression loss"
    )
    init_cfg: OptionalConfig = Field(
        default_factory=lambda: {"type": "Normal", "layer": "Conv2d", "std": 0.01},
        description="Weight initialization config",
    )


# =============================================================================
# RoI Head Configuration
# =============================================================================


class StandardRoIHeadConfig(HeadConfig):
    """Standard RoI head configuration.

    Two-stage detection head with bbox and optional mask branches.

    Attributes:
        bbox_roi_extractor: Extracts RoI features for bbox prediction.
        bbox_head: Bbox classification and regression head.
        mask_roi_extractor: Extracts RoI features for mask prediction (optional).
        mask_head: Mask prediction head (optional).

    Example:
        >>> cfg = StandardRoIHeadConfig(num_classes=80)
    """

    type: Literal["StandardRoIHead"] = "StandardRoIHead"
    bbox_roi_extractor: SingleRoIExtractorConfig = Field(
        default_factory=SingleRoIExtractorConfig,
        description="BBox RoI feature extractor",
    )
    bbox_head: Shared2FCBBoxHeadConfig = Field(
        ..., description="BBox prediction head"
    )
    mask_roi_extractor: Optional[SingleRoIExtractorConfig] = Field(
        default=None, description="Mask RoI feature extractor (None = share with bbox)"
    )
    mask_head: Optional[FCNMaskHeadConfig] = Field(
        default=None, description="Mask prediction head (optional)"
    )


# =============================================================================
# NMS Configuration
# =============================================================================


class NMSConfig(ComponentConfig):
    """Non-Maximum Suppression configuration."""

    type: Literal["nms"] = "nms"
    iou_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="IoU threshold for NMS"
    )


# =============================================================================
# Train and Test Configurations
# =============================================================================


class RPNTrainConfig(ComponentConfig):
    """RPN training configuration."""

    type: Literal["RPNTrainConfig"] = Field(default="RPNTrainConfig", exclude=True)
    assigner: MaxIoUAssignerConfig = Field(
        default_factory=lambda: MaxIoUAssignerConfig(
            pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3
        ),
        description="RPN assigner config",
    )
    sampler: RandomSamplerConfig = Field(
        default_factory=lambda: RandomSamplerConfig(
            num=256, pos_fraction=0.5, add_gt_as_proposals=False
        ),
        description="RPN sampler config",
    )
    allowed_border: int = Field(default=-1, description="Allowed border for anchors")
    pos_weight: float = Field(default=-1, description="Positive sample weight")
    debug: bool = Field(default=False, description="Enable debug mode")


class RPNProposalConfig(ComponentConfig):
    """RPN proposal generation configuration."""

    type: Literal["RPNProposalConfig"] = Field(default="RPNProposalConfig", exclude=True)
    nms_pre: int = Field(default=2000, gt=0, description="NMS candidates before filtering")
    max_per_img: int = Field(default=1000, gt=0, description="Max proposals per image")
    nms: NMSConfig = Field(
        default_factory=lambda: NMSConfig(iou_threshold=0.7), description="NMS config"
    )
    min_bbox_size: int = Field(default=0, ge=0, description="Min bbox size")


class RCNNTrainConfig(ComponentConfig):
    """RCNN (RoI head) training configuration."""

    type: Literal["RCNNTrainConfig"] = Field(default="RCNNTrainConfig", exclude=True)
    assigner: MaxIoUAssignerConfig = Field(
        default_factory=lambda: MaxIoUAssignerConfig(
            pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5
        ),
        description="RCNN assigner config",
    )
    sampler: RandomSamplerConfig = Field(
        default_factory=lambda: RandomSamplerConfig(
            num=512, pos_fraction=0.25, add_gt_as_proposals=True
        ),
        description="RCNN sampler config",
    )
    mask_size: int = Field(default=28, gt=0, description="Mask output size")
    pos_weight: float = Field(default=-1, description="Positive sample weight")
    debug: bool = Field(default=False, description="Enable debug mode")


class RPNTestConfig(ComponentConfig):
    """RPN test configuration."""

    type: Literal["RPNTestConfig"] = Field(default="RPNTestConfig", exclude=True)
    nms_pre: int = Field(default=1000, gt=0, description="NMS candidates before filtering")
    max_per_img: int = Field(default=1000, gt=0, description="Max proposals per image")
    nms: NMSConfig = Field(
        default_factory=lambda: NMSConfig(iou_threshold=0.7), description="NMS config"
    )
    min_bbox_size: int = Field(default=0, ge=0, description="Min bbox size")


class RCNNTestConfig(ComponentConfig):
    """RCNN test configuration."""

    type: Literal["RCNNTestConfig"] = Field(default="RCNNTestConfig", exclude=True)
    score_thr: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Score threshold"
    )
    nms: NMSConfig = Field(
        default_factory=lambda: NMSConfig(iou_threshold=0.5), description="NMS config"
    )
    max_per_img: int = Field(default=100, gt=0, description="Max detections per image")
    mask_thr_binary: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Binary mask threshold"
    )


# Type aliases
RPNHeadType = Annotated[Union[RPNHeadConfig], Field(discriminator="type")]
RoIHeadType = Annotated[Union[StandardRoIHeadConfig], Field(discriminator="type")]
