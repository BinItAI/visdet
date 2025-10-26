# Copyright (c) OpenMMLab. All rights reserved.
"""Backward compatibility for visdet.core.mask module."""

from visdet.structures.mask import BitmapMasks, PolygonMasks

__all__ = [
    "BitmapMasks",
    "PolygonMasks",
]
