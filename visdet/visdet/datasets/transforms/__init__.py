# Copyright (c) OpenMMLab. All rights reserved.
# I really don't think this file should have to exist at all.
# It sort of makes sense for the viscv transforms to be here, but the visdet transforms should be in the visdet package.
# I'm not sure why they are here.

from viscv.transforms import LoadImageFromFile as _LoadImageFromFile
from viscv.transforms import Normalize as _Normalize
from viscv.transforms import RandomApply as _RandomApply
from viscv.transforms import RandomChoice as _RandomChoice
from viscv.transforms import RandomFlip as _RandomFlip
from viscv.transforms import RandomResize as _RandomResize
from viscv.transforms import Resize as _Resize
from viscv.transforms.processing import RandomChoiceResize as _RandomChoiceResize

# Can't be imported from this location!! Where is it then??
from visdet.datasets.transforms.formatting import PackDetInputs as _PackDetInputs
from visdet.datasets.transforms.loading import LoadAnnotations as _LoadAnnotations
from visdet.datasets.transforms.transforms import RandomCrop as _RandomCrop
from visdet.datasets.transforms.transforms import Pad as _Pad
from visdet.registry import TRANSFORMS

Resize = TRANSFORMS.register_module()(_Resize)
Pad = TRANSFORMS.register_module()(_Pad)
Normalize = TRANSFORMS.register_module()(_Normalize)
LoadImageFromFile = TRANSFORMS.register_module()(_LoadImageFromFile)
RandomFlip = TRANSFORMS.register_module()(_RandomFlip)
RandomResize = TRANSFORMS.register_module()(_RandomResize)
LoadAnnotations = TRANSFORMS.register_module()(_LoadAnnotations)
PackDetInputs = TRANSFORMS.register_module()(_PackDetInputs)
RandomApply = TRANSFORMS.register_module()(_RandomApply)
RandomCrop = TRANSFORMS.register_module()(_RandomCrop)
RandomChoice = TRANSFORMS.register_module()(_RandomChoice)
RandomChoiceResize = TRANSFORMS.register_module()(_RandomChoiceResize)

__all__ = ["LoadAnnotations", "LoadImageFromFile", "Normalize", "PackDetInputs", "Pad", "RandomApply", "RandomChoice", "RandomChoiceResize", "RandomCrop", "RandomFlip", "RandomResize", "Resize"]
