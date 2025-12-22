# Copyright (c) OpenMMLab. All rights reserved.
"""Backward compatibility for visdet.datasets.pipelines module.

This module re-exports transforms from visdet.datasets.transforms for
backward compatibility with the old pipelines namespace.
"""

# Import available transforms
from visdet.datasets.transforms.formatting import PackDetInputs
from visdet.datasets.transforms.load_image import (
    LoadImageFromFile,
    LoadImageFromWebcam,
    LoadMultiChannelImageFromFiles,
)
from visdet.datasets.transforms.loading import (
    FilterAnnotations,
    LoadAnnotations,
)
from visdet.datasets.transforms.transforms import (
    Pad,
    RandomCrop,
    RandomFlip,
)

# These transforms don't exist in visdet yet - set to None for compatibility
RandomResize = None  # type: ignore[assignment]
DefaultFormatBundle = None  # type: ignore[assignment]
RandomApply = None  # type: ignore[assignment]
RandomChoice = None  # type: ignore[assignment]

__all__ = [
    "FilterAnnotations",
    "LoadAnnotations",
    "LoadImageFromFile",
    "LoadImageFromWebcam",
    "LoadMultiChannelImageFromFiles",
    "Pad",
    "RandomCrop",
    "RandomFlip",
]

if RandomResize is not None:
    __all__.append("RandomResize")
if PackDetInputs is not None:
    __all__.extend(["PackDetInputs", "DefaultFormatBundle"])
if RandomApply is not None:
    __all__.extend(["RandomApply", "RandomChoice"])
