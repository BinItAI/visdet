# Copyright (c) OpenMMLab. All rights reserved.
"""Backward compatibility for visdet.core.mask submodule."""

from visdet.structures.mask import BitmapMasks, PolygonMasks

__all__ = ["BitmapMasks", "PolygonMasks"]
