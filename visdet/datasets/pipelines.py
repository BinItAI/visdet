# Copyright (c) OpenMMLab. All rights reserved.
"""Backward compatibility for visdet.datasets.pipelines module.

This module re-exports transforms from visdet.datasets.transforms for
backward compatibility with the old pipelines namespace.
"""

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

# Try to import optional transforms that may not exist in all versions
try:
    from visdet.datasets.transforms.transforms import RandomResize
except ImportError:
    RandomResize = None

try:
    from visdet.datasets.transforms.formatting import DefaultFormatBundle, PackDetInputs
except ImportError:
    PackDetInputs = None
    DefaultFormatBundle = None

try:
    from visdet.datasets.transforms.wrappers import RandomApply, RandomChoice
except ImportError:
    RandomApply = None
    RandomChoice = None

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
