# ruff: noqa
"""
Computer Vision utilities (re-export of viscv).

This module provides access to viscv functionality through visdet.cv
for better namespace organization and discoverability.

Usage:
    # New preferred way
    from visdet import cv
    from visdet.cv import image

    # Legacy way (still works)
    import viscv
    from viscv import image

All viscv functionality is re-exported here for backwards compatibility
and namespace consistency within the visdet package.
"""

# Re-export all viscv functionality
from viscv import *  # noqa: F401, F403

# Explicitly re-export key modules for better IDE support
from viscv import image, transforms  # noqa: F401

__all__ = ["image", "transforms"]
