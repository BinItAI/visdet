# ruff: noqa
"""
Computer Vision utilities (re-export of viscv).

This module provides access to viscv functionality through visdet.cv
for better namespace organization and discoverability.

Usage:
    # New preferred way
    from visdet import cv
    from visdet.cv import image

    # Legacy way (still works but discouraged inside visdet)
    import viscv
    from viscv import image

All viscv functionality is re-exported here for backwards compatibility
and namespace consistency within the visdet package.
"""

# 1. Re-export all top-level symbols from the original `viscv` library.
from viscv import *  # noqa: F401, F403

# 2. Explicitly import submodules using relative paths for clarity and to make
#    them accessible under the `visdet.cv` namespace (e.g., `visdet.cv.image`).
from . import cnn, fileio, image, ops, transforms

# 3. Construct `__all__` to control `from visdet.cv import *` behavior.
#    This combines top-level symbols from viscv with the submodule names.
try:
    # Dynamically get __all__ from the original library if it exists.
    from viscv import __all__ as viscv_all
except ImportError:
    # Fallback if viscv.__all__ is not defined.
    viscv_all = []

# Expose both the re-exported symbols and the submodules.
__all__ = list(viscv_all) + ["cnn", "fileio", "image", "ops", "transforms"]
