# ruff: noqa
"""
Computer Vision utilities.

This module provides access to computer vision functionality through visdet.cv
for better namespace organization and discoverability.

Usage:
    from visdet import cv
    from visdet.cv import image, cnn, transforms, ops, fileio
"""

# Import submodules to make them accessible under the `visdet.cv` namespace
# (e.g., `visdet.cv.image`)
from . import cnn, fileio, image, ops, transforms  # noqa: F401
from .image import imfrombytes, imwrite  # noqa: F401

__all__ = ["cnn", "fileio", "image", "ops", "transforms", "imfrombytes", "imwrite"]
