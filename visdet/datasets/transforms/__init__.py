# Copyright (c) OpenMMLab. All rights reserved.
# I really don't think this file should have to exist at all.
# It sort of makes sense for the viscv transforms to be here, but the visdet transforms should be in the visdet package.
# I'm not sure why they are here.

from visdet.cv.transforms import LoadImageFromFile as _LoadImageFromFile
from visdet.cv.transforms import Normalize as _Normalize
from visdet.cv.transforms import RandomApply as _RandomApply
from visdet.cv.transforms import RandomChoice as _RandomChoice
from visdet.cv.transforms import RandomFlip as _RandomFlip
from visdet.cv.transforms import RandomResize as _RandomResize
from visdet.cv.transforms import Resize as _Resize

# RandomChoiceResize doesn't exist in visdet.cv.transforms.processing
# from visdet.cv.transforms.processing import RandomChoiceResize as _RandomChoiceResize
# Can't be imported from this location!! Where is it then??
from visdet.datasets.transforms.formatting import PackDetInputs as _PackDetInputs
from visdet.datasets.transforms.loading import LoadAnnotations as _LoadAnnotations
from visdet.datasets.transforms.transforms import Pad as _Pad
from visdet.datasets.transforms.transforms import RandomCrop as _RandomCrop
from visdet.registry import TRANSFORMS


def _register_pipelines():
    """Lazy registration to avoid circular imports."""
    from visdet.datasets.builder import PIPELINES

    # Register to both TRANSFORMS and PIPELINES for compatibility
    # PIPELINES.register_module()(_Resize)
    # PIPELINES.register_module()(_Pad)
    # PIPELINES.register_module()(_Normalize)
    # PIPELINES.register_module()(_LoadImageFromFile)
    # PIPELINES.register_module()(_RandomFlip)
    # PIPELINES.register_module()(_RandomResize)
    # PIPELINES.register_module()(_LoadAnnotations)
    # PIPELINES.register_module()(_PackDetInputs)
    # PIPELINES.register_module()(_RandomApply)
    # PIPELINES.register_module()(_RandomCrop)
    # PIPELINES.register_module()(_RandomChoice)
    # PIPELINES.register_module()(_RandomChoiceResize)


# Register immediately (but builder may not be imported yet)
try:
    _register_pipelines()
except ImportError:
    pass

# Also register to TRANSFORMS - THESE ARE REDUNDANT AND CAUSE WARNINGS
# Resize = TRANSFORMS.register_module()(_Resize)
# Pad = TRANSFORMS.register_module()(_Pad)
# Normalize = TRANSFORMS.register_module()(_Normalize)
# LoadImageFromFile = TRANSFORMS.register_module()(_LoadImageFromFile)
# RandomFlip = TRANSFORMS.register_module()(_RandomFlip)
# RandomResize = TRANSFORMS.register_module()(_RandomResize)
# LoadAnnotations = TRANSFORMS.register_module()(_LoadAnnotations)
# PackDetInputs = TRANSFORMS.register_module()(_PackDetInputs)
# RandomApply = TRANSFORMS.register_module()(_RandomApply)
# RandomCrop = TRANSFORMS.register_module()(_RandomCrop)
# RandomChoice = TRANSFORMS.register_module()(_RandomChoice)
# RandomChoiceResize = TRANSFORMS.register_module()(_RandomChoiceResize)

# Expose them for import
Resize = _Resize
Pad = _Pad
Normalize = _Normalize
LoadImageFromFile = _LoadImageFromFile
RandomFlip = _RandomFlip
RandomResize = _RandomResize
LoadAnnotations = _LoadAnnotations
PackDetInputs = _PackDetInputs
RandomApply = _RandomApply
RandomCrop = _RandomCrop
RandomChoice = _RandomChoice

__all__ = [
    "LoadAnnotations",
    "LoadImageFromFile",
    "Normalize",
    "PackDetInputs",
    "Pad",
    "RandomApply",
    "RandomChoice",
    # "RandomChoiceResize",  # Commented out - doesn't exist
    "RandomCrop",
    "RandomFlip",
    "RandomResize",
    "Resize",
]
